"""Launch all 16 Wan MemDual experiments from two models and three datasets."""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODES = "baseline,kvf,anchor,combined"


@dataclass
class WanModel:
    kind: str
    pipeline: Path
    transformer: Path
    patch_size: tuple[int, int, int]
    channels: int
    text_dim: int


def parse_args():
    parser = argparse.ArgumentParser(description="输入两种模型的路径和三个数据集的路径，运行 16 组 Wan MemDual 实验。")
    parser.add_argument("--t2v-model", required=True, type=Path, help="Wan2.2-T2V Diffusers pipeline or transformer.")
    parser.add_argument(
        "--ti2v-model", required=True, type=Path, help="Wan2.2-TI2V Diffusers pipeline or transformer."
    )
    parser.add_argument(
        "--datasets",
        required=True,
        nargs=3,
        type=Path,
        metavar=("DATASET_1", "DATASET_2", "DATASET_3"),
        help="Exactly three dataset roots, each containing t2v/ and ti2v/ offline Parquet shards.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Measured optimizer steps per experiment (default: 20).")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Untimed optimizer steps (default: 5).")
    parser.add_argument("--nodes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", 0)))
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR"))
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)))
    parser.add_argument("--peer", help="Start node 1 via SSH from node 0 with shared model/data/output paths.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/memdual/matrix"))
    parser.add_argument("--max-pack-tokens", type=int, help="Override the SP-padded token budget for each run.")
    parser.add_argument("--long-boundary", type=int, help="Override the token threshold for singleton anchors.")
    parser.add_argument("--fixed-target-runtime", type=float, help="Override the fixed packing target in L² units.")
    parser.add_argument(
        "--synthetic-frame-counts",
        nargs="+",
        default=["4:288", "8:144", "16:48"],
        metavar="FRAMES:COUNT",
        help="Exact latent-frame histogram for synthetic data.",
    )
    parser.add_argument("--synthetic-latent-height", type=int, default=32)
    parser.add_argument("--synthetic-latent-width", type=int, default=32)
    parser.add_argument("--synthetic-context-tokens", type=int, default=512)
    parser.add_argument(
        "--synthetic-random-init", action="store_true", help="Use random weights only for synthetic data."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs and write configs without starting GPUs."
    )
    args = parser.parse_args()
    if args.steps < 0 or args.warmup_steps < 0:
        parser.error("Step counts must be nonnegative. Use --steps 0 to measure a full epoch.")
    if not 0 <= args.node_rank < args.nodes or not 1 <= args.master_port <= 65520:
        parser.error("Invalid node rank or base port for 16 consecutive rendezvous ports.")
    if args.gpus_per_node % 2 or args.gpus_per_node < (4 if args.nodes == 1 else 2):
        parser.error("Use an even GPU count with at least two GPUs in each Ulysses group.")
    if args.peer and (args.nodes != 2 or args.node_rank != 0):
        parser.error("--peer requires the two-node launch on node 0.")
    if args.nodes == 2 and not args.master_addr and not args.dry_run:
        parser.error("Set --master-addr to the reachable address of node 0.")
    return args


def load_model(kind, path, ulysses_size):
    path = path.resolve()
    if (path / "model_index.json").is_file():
        pipeline, transformer = path, path / "transformer"
    else:
        pipeline, transformer = path.parent, path
    index = json.loads((pipeline / "model_index.json").read_text())
    config = json.loads((transformer / "config.json").read_text())
    if config.get("_class_name", config.get("model_type")) != "WanTransformer3DModel":
        raise ValueError(f"{transformer} must be a Diffusers Wan transformer.")
    if bool(index.get("expand_timesteps", False)) != (kind == "ti2v"):
        raise ValueError(f"{path} does not match the {kind.upper()} path. Check model_index.json expand_timesteps.")
    if config["in_channels"] != config["out_channels"]:
        raise ValueError(f"{path} uses channel-concatenated I2V conditioning.")
    if config["num_attention_heads"] % ulysses_size:
        raise ValueError(f"{kind.upper()} attention heads must be divisible by Ulysses size {ulysses_size}.")
    return WanModel(
        kind, pipeline, transformer, tuple(config["patch_size"]), config["out_channels"], config["text_dim"]
    )


def _row_length(value, model, shard):
    shape = tuple(pickle.loads(value).shape)
    patch = model.patch_size
    if (
        len(shape) != 5
        or shape[:2] != (1, 2 * model.channels)
        or any(size < p or size % p for size, p in zip(shape[-3:], patch))
        or (model.kind == "ti2v" and shape[2] < 2)
    ):
        raise ValueError(f"{shard} has invalid {model.kind.upper()} latents shape {shape}.")
    return math.prod(size // p for size, p in zip(shape[-3:], patch))


def read_dataset(root, model):
    import pyarrow.parquet as pq

    directory = root / model.kind
    shards = sorted(directory.rglob("*.parquet"))
    if not shards:
        raise ValueError(f"No {model.kind.upper()} offline Parquet shards in {directory}.")
    cached_lengths = directory / "lengths.json"
    lengths = json.loads(cached_lengths.read_text()) if cached_lengths.is_file() else None
    if lengths is not None and not isinstance(lengths, list):
        raise ValueError(f"{cached_lengths} must contain a JSON list of token lengths.")
    row_count = 0
    scanned_lengths = []
    for shard in shards:
        if "," in str(shard):
            raise ValueError(f"VeOmni dataset paths cannot contain commas: {shard}")
        parquet = pq.ParquetFile(shard)
        if not {"latents", "context"} <= set(parquet.schema_arrow.names):
            raise ValueError(f"{shard} must contain Wan offline latents and context columns.")
        rows = parquet.metadata.num_rows
        if rows == 0:
            continue
        sample = next(parquet.iter_batches(batch_size=1, columns=["latents", "context"]))
        context = pickle.loads(sample.column(1)[0].as_py())
        if context.ndim != 3 or context.shape[0] != 1 or context.shape[-1] != model.text_dim:
            raise ValueError(f"{shard} does not match the {model.kind.upper()} text context shape.")
        if lengths is None:
            for batch in parquet.iter_batches(batch_size=4, columns=["latents"]):
                scanned_lengths.extend(_row_length(value.as_py(), model, shard) for value in batch.column(0))
        else:
            actual = _row_length(sample.column(0)[0].as_py(), model, shard)
            if row_count >= len(lengths) or lengths[row_count] != actual:
                raise ValueError(f"{shard} does not match its lengths.json.")
        row_count += rows
    if lengths is None:
        lengths = scanned_lengths
    if not lengths or len(lengths) != row_count or any(type(x) is not int or x < 1 for x in lengths):
        raise ValueError(
            f"{directory}/lengths.json must have one positive token length per row in sorted shard order."
        )
    return shards, lengths


def write_config(args, model, source, ulysses_size, datasets, lengths):
    directory = args.out_dir / model.kind / source
    directory.mkdir(parents=True, exist_ok=True)
    is_real = source == "real"
    synthetic = {
        "frame_counts": args.synthetic_frame_counts,
        "latent_height": args.synthetic_latent_height,
        "latent_width": args.synthetic_latent_width,
        "context_tokens": args.synthetic_context_tokens,
    }
    if is_real:
        lengths_path = directory / f"lengths.node{args.node_rank}.json"
        lengths_path.write_text(json.dumps(lengths) + "\n")
        train_path = ",".join(str(path) for path in datasets)
        longest = max(lengths)
    else:
        lengths_path = ""
        train_path = "synthetic"
        pt, ph, pw = model.patch_size
        try:
            counts = [tuple(map(int, entry.split(":"))) for entry in synthetic["frame_counts"]]
            frames = [frames for frames, count in counts if count > 0]
        except ValueError as exc:
            raise ValueError("--synthetic-frame-counts entries must use FRAMES:COUNT.") from exc
        height, width = synthetic["latent_height"], synthetic["latent_width"]
        min_frames = 2 if model.kind == "ti2v" else 1
        if (
            len(frames) != len(counts)
            or not frames
            or any(n < min_frames or n % pt for n in frames)
            or min(height, width, synthetic["context_tokens"]) < 1
            or height % ph
            or width % pw
        ):
            raise ValueError(f"Synthetic geometry and counts must be positive and match the {model.kind} patch grid.")
        longest = (max(frames) // pt) * (height // ph) * (width // pw)
    padded_longest = math.ceil(longest / ulysses_size) * ulysses_size
    budget = args.max_pack_tokens if args.max_pack_tokens is not None else padded_longest
    boundary = args.long_boundary if args.long_boundary is not None else math.ceil(longest / 2)
    target = args.fixed_target_runtime if args.fixed_target_runtime is not None else longest**2
    if budget < padded_longest or boundary < 1 or not math.isfinite(target) or target <= 0:
        raise ValueError(
            f"{model.kind}/{source} needs a pack budget ≥ {padded_longest} and positive packing settings."
        )
    config = {
        "model": {
            "config_path": str(model.transformer / "config.json"),
            "model_path": str(model.transformer) if is_real or not args.synthetic_random_init else None,
            "condition_model_path": str(model.pipeline),
            "condition_model_cfg": {"expand_timesteps": model.kind == "ti2v"},
            "ops_implementation": {
                "attn_implementation": "veomni_sdpa",
                "rotary_pos_emb_implementation": "eager",
                "rms_norm_implementation": "eager",
            },
            "accelerator": {
                "init_device": "meta",
                "ulysses_size": ulysses_size,
                "dp_replicate_size": 2,
                "dp_shard_size": 1,
                "fsdp_config": {
                    "fsdp_mode": "fsdp2",
                    "mixed_precision": {"enable": True, "param_dtype": "bfloat16", "reduce_dtype": "bfloat16"},
                },
            },
        },
        "data": {
            "train_path": train_path,
            "datasets_type": "mapping",
            "dataloader": {"num_workers": 0, "drop_last": True, "pin_memory": False},
            "anchor_batching": {
                "lengths_path": str(lengths_path),
                "long_boundary": boundary,
                "max_pack_tokens": budget,
            },
        },
        "train": {
            "training_task": "offline_training",
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "dyn_bsz": False,
            "checkpoint": {"output_dir": str(directory / "runargs"), "save_hf_weights": False},
        },
        "benchmark": {
            "data_source": source,
            "shuffle_real_samples": is_real,
            "synthetic": synthetic,
            "warmup_steps": args.warmup_steps,
            "steps": args.steps,
            "fixed_target_runtime": target,
        },
    }
    # JSON is a YAML subset accepted by the runner. Per-node files avoid concurrent writes on shared storage.
    config_path = directory / f"matrix.node{args.node_rank}.yaml"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def main():
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    roots = [path.resolve() for path in args.datasets]
    if len(set(roots)) != 3 or not all(path.is_dir() for path in roots):
        raise ValueError("--datasets requires three distinct existing dataset directories.")
    ulysses_size = args.gpus_per_node * args.nodes // 2
    models = [
        load_model("t2v", args.t2v_model, ulysses_size),
        load_model("ti2v", args.ti2v_model, ulysses_size),
    ]
    configs = []
    for model in models:
        files, lengths = [], []
        for root in roots:
            shards, shard_lengths = read_dataset(root, model)
            files.extend(shards)
            lengths.extend(shard_lengths)
        for source in ("real", "synthetic"):
            configs.append(write_config(args, model, source, ulysses_size, files, lengths))

    for index, config in enumerate(configs):
        port = args.master_port + 4 * index
        print(f"{config.parent.relative_to(args.out_dir)}: {MODES} (ports {port}–{port + 3})", flush=True)
        if args.dry_run:
            continue
        command = [
            "bash",
            str(ROOT / "scripts/profile/run_memdual.sh"),
            str(config),
            "--nodes",
            str(args.nodes),
            "--gpus-per-node",
            str(args.gpus_per_node),
            "--node-rank",
            str(args.node_rank),
            "--master-port",
            str(port),
            "--python",
            sys.executable,
            "--out-dir",
            str(config.parent),
            "--modes",
            MODES,
        ]
        if args.master_addr:
            command.extend(("--master-addr", args.master_addr))
        if args.peer:
            command.extend(("--peer", args.peer))
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
