"""Launch Wan MemDual ablations on fixed, trace-derived, and workloads from dataset metadata."""

import argparse
import ast
import csv
import gzip
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METADATA_MODES = ("baseline", "kvf", "anchor", "combined")
FIXED_MODES = ("baseline", "kvf")
TRACE_MODES = ("baseline", "anchor", "combined")
KVF_SECONDS = (14, 21, 28, 35)
EXTRA_KVF_CP = {"t2v": 10, "ti2v": 12}
TRACE_RATIOS = ((1, 0, 49), (1, 3, 6), (1, 49, 50), (1, 6, 3), (3, 1, 6), (3, 6, 1), (6, 1, 3), (6, 3, 1))
VIDEO_FPS = 25
DATA_SEED = 42
DURATION_COLUMNS = ("seconds", "duration", "timestamp")
ISO_DURATION = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?")


@dataclass
class WanModel:
    kind: str
    pipeline: Path
    transformer: Path
    patch_size: tuple[int, int, int]
    vae_temporal_factor: int
    vae_spatial_factor: int
    rope_max_seq_len: int


def parse_ratio(value):
    try:
        weights = tuple(int(item) for item in value.split(":"))
    except ValueError:
        weights = ()
    if len(weights) != 3 or min(weights) < 0 or not any(weights):
        raise argparse.ArgumentTypeError("Use three nonnegative integer weights such as 1:3:6.")
    divisor = math.gcd(*weights)
    return tuple(weight // divisor for weight in weights)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wan MemDual ablations for two models and three sets of video metadata."
    )
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
        help="Three metadata files or directories of CSV, CSV.GZ, or Parquet shards.",
    )
    parser.add_argument("--steps", type=int, default=20, help="Measured optimizer steps per experiment (default: 20).")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Untimed optimizer steps (default: 5).")
    parser.add_argument("--nodes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", 0)))
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR"))
    parser.add_argument("--master-port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)))
    parser.add_argument("--peer", help="Start node 1 via SSH from node 0 with shared model/data/output paths.")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "result", help="Output directory (default: <repo>/result)."
    )
    parser.add_argument("--max-pack-tokens", type=int, help="Override the SP-padded token budget for each run.")
    parser.add_argument("--long-boundary", type=int, help="Override the token threshold for singleton anchors.")
    parser.add_argument("--fixed-target-runtime", type=float, help="Override the fixed packing target in L² units.")
    parser.add_argument(
        "--dataset-ratios",
        type=parse_ratio,
        default="1:1:1",
        metavar="A:B:C",
        help="Positive row-count weights in --datasets order (default: 1:1:1).",
    )
    parser.add_argument(
        "--metadata-samples",
        type=int,
        default=5000,
        help="Maximum mixed metadata clips for each model (default: 5000).",
    )
    parser.add_argument(
        "--metadata-max-seconds",
        type=float,
        help="Exclude metadata clips longer than this duration (default: only apply model limits).",
    )
    parser.add_argument(
        "--kvf-backend",
        choices=("nvshmem", "nccl"),
        default="nvshmem",
        help="Use hybrid NVSHMEM row exchanges or the all-NCCL KVF comparison.",
    )
    parser.add_argument(
        "--kvf-seconds",
        type=int,
        nargs="+",
        choices=KVF_SECONDS,
        default=KVF_SECONDS,
        help="Paper Table 1 durations, each benchmarked separately (default: all four).",
    )
    parser.add_argument(
        "--skip-extra-kvf-cp",
        action="store_true",
        help="Skip the additional two-node KVF pairs at CP 10 (T2V) and CP 12 (TI2V).",
    )
    parser.add_argument(
        "--trace-ratios",
        type=parse_ratio,
        nargs="+",
        default=TRACE_RATIOS,
        metavar="L:M:S",
        help="Long:medium:short experiment weights (default: all eight ratios in Figure 12).",
    )
    parser.add_argument(
        "--trace-max-seconds",
        type=int,
        default=35,
        help="Upper duration bound for the open-ended trace long bin (default: 35).",
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=500,
        help="Number of rows per synthetic workload (default: 500).",
    )
    parser.add_argument("--synthetic-latent-height", type=int, help="Override the 1080p-derived latent height.")
    parser.add_argument("--synthetic-latent-width", type=int, help="Override the 1080p-derived latent width.")
    parser.add_argument("--synthetic-context-tokens", type=int, default=512)
    parser.add_argument(
        "--synthetic-random-init",
        action="store_true",
        help="Use random weights for fixed and trace synthetic workloads.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs and write configs without starting GPUs."
    )
    args = parser.parse_args()
    if args.steps < 0 or args.warmup_steps < 0:
        parser.error("Step counts must be nonnegative. Use --steps 0 to measure a full epoch.")
    args.kvf_seconds = sorted(set(args.kvf_seconds))
    if not all(args.dataset_ratios):
        parser.error("--dataset-ratios requires a positive weight for each dataset.")
    if args.metadata_samples < sum(args.dataset_ratios):
        parser.error("--metadata-samples must allow at least one complete block of --dataset-ratios.")
    if args.metadata_max_seconds is not None and (
        not math.isfinite(args.metadata_max_seconds) or args.metadata_max_seconds <= 0
    ):
        parser.error("--metadata-max-seconds must be finite and positive.")
    if args.trace_max_seconds <= 20:
        parser.error("--trace-max-seconds must be greater than 20.")
    args.trace_ratios = list(dict.fromkeys(args.trace_ratios))
    if args.synthetic_samples < 1 or any(args.synthetic_samples % sum(ratio) for ratio in args.trace_ratios):
        parser.error("--synthetic-samples must be a positive multiple of each reduced trace ratio total.")
    args.extra_kvf_cp = {
        kind: cp_size
        for kind, cp_size in EXTRA_KVF_CP.items()
        if args.nodes == 2 and not args.skip_extra_kvf_cp and cp_size // 2 <= args.gpus_per_node
    }
    runs = 2 * (
        len(METADATA_MODES) + len(TRACE_MODES) * len(args.trace_ratios) + len(FIXED_MODES) * len(args.kvf_seconds)
    ) + len(args.extra_kvf_cp) * len(FIXED_MODES) * len(args.kvf_seconds)
    if not 0 <= args.node_rank < args.nodes or not 1 <= args.master_port <= 65536 - runs:
        parser.error(f"Invalid node rank or base port for {runs} consecutive rendezvous ports.")
    if args.gpus_per_node % 2 or args.gpus_per_node < (4 if args.nodes == 1 else 2):
        parser.error("Use an even GPU count with at least two GPUs in each Ulysses group.")
    if args.peer and (args.nodes != 2 or args.node_rank != 0):
        parser.error("--peer requires the two-node launch on node 0.")
    if args.nodes == 2 and not args.master_addr and not args.dry_run:
        parser.error("Set --master-addr to the reachable address of node 0.")
    return args


def load_model(kind, path, ulysses_size, extra_ulysses_size=None):
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
    for size in (ulysses_size, extra_ulysses_size):
        if size is not None and config["num_attention_heads"] % size:
            raise ValueError(f"{kind.upper()} attention heads must be divisible by Ulysses size {size}.")
    vae_config_path = pipeline / "vae" / "config.json"
    vae = json.loads(vae_config_path.read_text()) if vae_config_path.is_file() else {}
    return WanModel(
        kind,
        pipeline,
        transformer,
        tuple(config["patch_size"]),
        vae.get("scale_factor_temporal", 4),
        vae.get("scale_factor_spatial", 8 if kind == "t2v" else 16),
        config.get("rope_max_seq_len", 1024),
    )


def _duration_column(columns, path):
    for column in DURATION_COLUMNS:
        if column in columns:
            return column
    raise ValueError(f"{path} needs a metadata column named seconds, duration, or timestamp.")


def _metadata_values(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        column = _duration_column(parquet.schema_arrow.names, path)
        for batch in parquet.iter_batches(columns=[column]):
            for value in batch.column(0).to_pylist():
                yield column, value
    else:
        # Panda rows can contain hundreds of captions even though only timestamps are needed.
        csv.field_size_limit(sys.maxsize)
        open_csv = gzip.open if path.suffix == ".gz" else open
        with open_csv(path, "rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            column = _duration_column(reader.fieldnames or (), path)
            for row in reader:
                yield column, row[column]


def _clock_seconds(value):
    if isinstance(value, str) and ":" in value:
        seconds = 0.0
        for part in value.split(":"):
            seconds = 60 * seconds + float(part)
        return seconds
    return float(value)


def _clip_durations(column, value):
    if column != "timestamp":
        try:
            match = ISO_DURATION.fullmatch(value.strip()) if isinstance(value, str) else None
            if match:
                hours, minutes, seconds = (float(part or 0) for part in match.groups())
                yield hours * 3600 + minutes * 60 + seconds
            else:
                yield _clock_seconds(value)
        except (TypeError, ValueError):
            yield None
        return

    try:
        clips = ast.literal_eval(value) if isinstance(value, str) else value
    except (SyntaxError, ValueError):
        clips = None
    if not isinstance(clips, (list, tuple)) or not clips:
        yield None
        return
    if len(clips) == 2 and not isinstance(clips[0], (list, tuple)) and not isinstance(clips[1], (list, tuple)):
        clips = [clips]
    for clip in clips:
        try:
            if not isinstance(clip, (list, tuple)):
                raise ValueError("Expected a start and an end timestamp.")
            start, end = clip
            yield _clock_seconds(end) - _clock_seconds(start)
        except (TypeError, ValueError):
            yield None


def read_metadata(path, *, min_frames, max_frames, max_seconds):
    """Count clip lengths at the target FPS, including each interval in a Panda CSV row."""
    extensions = (".csv", ".csv.gz", ".parquet")
    if path.is_dir():
        files = sorted(file for file in path.rglob("*") if file.is_file() and file.name.endswith(extensions))
    else:
        files = [path] if path.name.endswith(extensions) else []
    if not files:
        raise ValueError(f"No CSV, CSV.GZ, or Parquet metadata found at {path}.")

    histogram = Counter()
    stats = {"total_clips": 0, "invalid": 0, "too_short": 0, "above_max_seconds": 0, "above_model_limit": 0}
    for file in files:
        for column, value in _metadata_values(file):
            for duration in _clip_durations(column, value):
                stats["total_clips"] += 1
                if duration is None or not math.isfinite(duration) or duration <= 0:
                    stats["invalid"] += 1
                elif max_seconds is not None and duration > max_seconds:
                    stats["above_max_seconds"] += 1
                else:
                    frames = round(duration * VIDEO_FPS)
                    if frames < min_frames:
                        stats["too_short"] += 1
                    elif frames > max_frames:
                        stats["above_model_limit"] += 1
                    else:
                        histogram[frames] += 1
    return histogram, stats


def mix_metadata(histograms, weights, sample_limit):
    """Sample clips without replacement, then shuffle within each block of the requested mixture."""
    available = [histogram.total() for histogram in histograms]
    blocks = min(sample_limit // sum(weights), *(count // weight for count, weight in zip(available, weights)))
    if not blocks:
        raise ValueError(f"Cannot mix usable metadata clip counts {available} at ratio {weights}.")
    rng = random.Random(DATA_SEED)
    counts = [blocks * weight for weight in weights]
    selected = []
    for histogram, count in zip(histograms, counts):
        population = sorted(histogram)
        selected.append(rng.sample(population, counts=[histogram[value] for value in population], k=count))
    order = []
    for block in range(blocks):
        clips = []
        for source, weight in zip(selected, weights):
            clips.extend(source[block * weight : (block + 1) * weight])
        rng.shuffle(clips)
        order.extend(clips)
    return order, {"weights": weights, "available_counts": available, "selected_counts": counts}


def latent_frames(model, raw_frames):
    pt = model.patch_size[0]
    frames = (raw_frames - 1) // model.vae_temporal_factor + 1
    return frames // pt * pt


def min_video_frames(model):
    min_latent = 2 if model.kind == "ti2v" else 1
    pt = model.patch_size[0]
    return (math.ceil(min_latent / pt) * pt - 1) * model.vae_temporal_factor + 1


def synthetic_workload(args, model, *, seconds, trace_ratio, metadata_frames):
    """Convert the paper's 25 FPS workloads to Wan latent geometry."""
    pt, ph, pw = model.patch_size
    spatial = model.vae_spatial_factor
    height = args.synthetic_latent_height
    width = args.synthetic_latent_width
    if height is None:
        height = 1080 // (spatial * ph) * ph
    if width is None:
        width = 1920 // (spatial * pw) * pw

    if metadata_frames is not None:
        counts = Counter(metadata_frames)
    elif seconds is not None:
        counts = {latent_frames(model, seconds * VIDEO_FPS): args.synthetic_samples}
    else:
        # Section 5.3.2 uses <5 s, 5–20 s, and >20 s. Table 2 starts at 2 s.
        bins = (
            (20 * VIDEO_FPS + 1, args.trace_max_seconds * VIDEO_FPS),
            (5 * VIDEO_FPS, 20 * VIDEO_FPS),
            (2 * VIDEO_FPS, 5 * VIDEO_FPS - 1),
        )
        blocks = args.synthetic_samples // sum(trace_ratio)
        counts = Counter()
        for index, ((low, high), weight) in enumerate(zip(bins, trace_ratio)):
            rng = random.Random(DATA_SEED + index)
            counts.update(latent_frames(model, rng.randint(low, high)) for _ in range(blocks * weight))
    if (
        min(counts) < (2 if model.kind == "ti2v" else 1)
        or min(height, width, args.synthetic_context_tokens) < 1
        or height % ph
        or width % pw
    ):
        raise ValueError(f"Synthetic geometry must be positive and match the {model.kind} patch grid.")
    return {
        "frame_counts": [f"{frames}:{count}" for frames, count in sorted(counts.items())],
        "latent_height": height,
        "latent_width": width,
        "context_tokens": args.synthetic_context_tokens,
    }, (max(counts) // pt) * (height // ph) * (width // pw)


def write_config(args, model, workload, ulysses_size, *, dp_replicas=2, metadata=None, seconds=None, trace_ratio=None):
    directory = args.out_dir / model.kind / workload
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = ""
    frames = None
    if metadata is not None:
        raw_frames, mix = metadata
        frames = [latent_frames(model, count) for count in raw_frames]
        manifest = directory / f"metadata.node{args.node_rank}.json"
        manifest.write_text(json.dumps({"latent_frames": frames, **mix}) + "\n")
        metadata_path = str(manifest)
    synthetic, longest = synthetic_workload(
        args, model, seconds=seconds, trace_ratio=trace_ratio, metadata_frames=frames
    )
    padded_longest = math.ceil(longest / ulysses_size) * ulysses_size
    budget = args.max_pack_tokens if args.max_pack_tokens is not None else padded_longest
    boundary = args.long_boundary if args.long_boundary is not None else math.ceil(longest / 2)
    target = None
    if seconds is None:
        target = args.fixed_target_runtime if args.fixed_target_runtime is not None else longest**2
    if budget < padded_longest or boundary < 1 or (target is not None and (not math.isfinite(target) or target <= 0)):
        raise ValueError(
            f"{model.kind}/{workload} needs a pack budget ≥ {padded_longest} and positive packing settings."
        )
    benchmark = {
        "data_source": "metadata" if metadata is not None else "synthetic",
        "metadata_path": metadata_path,
        "synthetic": synthetic,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "fixed_target_runtime": target,
    }
    config = {
        "model": {
            "config_path": str(model.transformer / "config.json"),
            "model_path": str(model.transformer) if metadata is not None or not args.synthetic_random_init else None,
            "condition_model_path": str(model.pipeline),
            "condition_model_cfg": {"expand_timesteps": model.kind == "ti2v"},
            "ops_implementation": {
                "attn_implementation": "veomni_sdpa",
                "wan_kvf_implementation": args.kvf_backend,
                "rotary_pos_emb_implementation": "eager",
                "rms_norm_implementation": "eager",
            },
            "accelerator": {
                "init_device": "meta",
                "ulysses_size": ulysses_size,
                "dp_replicate_size": dp_replicas,
                "dp_shard_size": 1,
                "fsdp_config": {
                    "fsdp_mode": "fsdp2",
                    "mixed_precision": {"enable": True, "param_dtype": "bfloat16", "reduce_dtype": "bfloat16"},
                },
            },
        },
        "data": {
            "train_path": "synthetic",
            "datasets_type": "mapping",
            "dataloader": {"num_workers": 0, "drop_last": True, "pin_memory": False, "shuffle": metadata is None},
            "anchor_batching": {
                "lengths_path": "",
                "long_boundary": boundary,
                "max_pack_tokens": budget,
            },
        },
        "train": {
            "training_task": "offline_training",
            "global_batch_size": dp_replicas,
            "micro_batch_size": 1,
            "dyn_bsz": False,
            "checkpoint": {"output_dir": str(directory / "runargs"), "save_hf_weights": False},
        },
        "benchmark": benchmark,
    }
    # JSON is a YAML subset accepted by the runner. Per-node files avoid concurrent writes on shared storage.
    config_path = directory / f"matrix.node{args.node_rank}.yaml"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def main():
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    paths = [path.resolve() for path in args.datasets]
    if len(set(paths)) != 3 or not all(path.exists() for path in paths):
        raise ValueError("--datasets requires three distinct existing metadata paths.")
    ulysses_size = args.gpus_per_node * args.nodes // 2
    models = [
        load_model("t2v", args.t2v_model, ulysses_size, args.extra_kvf_cp.get("t2v")),
        load_model("ti2v", args.ti2v_model, ulysses_size, args.extra_kvf_cp.get("ti2v")),
    ]
    min_frames = max(min_video_frames(model) for model in models)
    # The temporal RoPE table caps patch tokens per axis, before packing.
    max_frames = min(
        ((model.rope_max_seq_len + 1) * model.patch_size[0] - 1) * model.vae_temporal_factor for model in models
    )
    sources = [
        read_metadata(path, min_frames=min_frames, max_frames=max_frames, max_seconds=args.metadata_max_seconds)
        for path in paths
    ]
    histograms, source_stats = zip(*sources)
    raw_frames, mix = mix_metadata(histograms, args.dataset_ratios, args.metadata_samples)
    mix.update(
        datasets=[str(path) for path in paths],
        source_stats=source_stats,
        sample_limit=args.metadata_samples,
        max_seconds=args.metadata_max_seconds,
        min_frames=min_frames,
        max_model_frames=max_frames,
        target_fps=VIDEO_FPS,
        seed=DATA_SEED,
    )
    configs = []
    for model in models:
        measured = write_config(args, model, "metadata", ulysses_size, metadata=(raw_frames, mix))
        configs.append((measured, METADATA_MODES, ulysses_size, 2))
        for seconds in args.kvf_seconds:
            fixed = write_config(args, model, f"synthetic_fixed/{seconds}s", ulysses_size, seconds=seconds)
            configs.append((fixed, FIXED_MODES, ulysses_size, 2))
        for ratio in args.trace_ratios:
            long, medium, short = ratio
            workload = f"synthetic_trace/L{long}_M{medium}_S{short}"
            trace = write_config(args, model, workload, ulysses_size, trace_ratio=ratio)
            configs.append((trace, TRACE_MODES, ulysses_size, 2))

    for model in models:
        if model.kind not in args.extra_kvf_cp:
            continue
        cp_size = args.extra_kvf_cp[model.kind]
        for seconds in args.kvf_seconds:
            workload = f"synthetic_fixed_cp{cp_size}/{seconds}s"
            fixed = write_config(args, model, workload, cp_size, dp_replicas=1, seconds=seconds)
            configs.append((fixed, FIXED_MODES, cp_size, 1))

    port = args.master_port
    for config, modes, cp_size, dp_replicas in configs:
        gpus_per_node = cp_size * dp_replicas // args.nodes
        last_port = port + len(modes) - 1
        modes_arg = ",".join(modes)
        print(
            f"{config.parent.relative_to(args.out_dir)}: {modes_arg} "
            f"(CP {cp_size}, DP {dp_replicas}, {gpus_per_node} GPUs/node, ports {port}–{last_port})",
            flush=True,
        )
        first_port, port = port, last_port + 1
        if args.dry_run:
            continue
        command = [
            "bash",
            str(ROOT / "scripts/profile/run_memdual.sh"),
            str(config),
            "--nodes",
            str(args.nodes),
            "--gpus-per-node",
            str(gpus_per_node),
            "--node-rank",
            str(args.node_rank),
            "--master-port",
            str(first_port),
            "--python",
            sys.executable,
            "--out-dir",
            str(config.parent),
            "--modes",
            modes_arg,
        ]
        if dp_replicas == 1:
            command.append("--single-cross-node-group")
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
