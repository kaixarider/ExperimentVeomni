"""Benchmark Wan offline training with KVF and Anchor Batching."""

import json
import math
import os
import random
import socket
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Literal, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Subset

from veomni.arguments import parse_args
from veomni.data.diffusion.anchor_batching import AnchorStepSampler
from veomni.distributed.parallel_state import use_parallel_state
from veomni.distributed.sequence_parallel.ulysses import observe_ulysses_waits
from veomni.ops.kernels.attention.kvf_nvshmem import (
    close_nvshmem_kvf,
    kvf_block_bytes,
    nvshmem_kvf_symmetric_bytes,
    prepare_nvshmem_kvf,
)
from veomni.trainer.base import VeOmniIter
from veomni.trainer.dit_trainer import DiTModelRuntime, DiTTrainer, VeOmniDiTArguments


@dataclass
class SyntheticDataConfig:
    """Geometry and exact histogram of generated offline Wan samples."""

    frame_counts: list[str] = field(
        default_factory=lambda: ["4:48", "8:12", "16:4"],
        metadata={"help": "Exact latent-frame histogram as FRAMES:COUNT entries."},
    )
    latent_height: int = 8
    latent_width: int = 8
    context_tokens: int = 32


@dataclass
class BenchmarkConfig:
    """Select the MemDual ablation and measurement window."""

    mode: Literal["baseline", "kvf", "anchor", "combined"] = "combined"
    data_source: Literal["real", "synthetic", "metadata"] = "real"
    shuffle_real_samples: bool = False
    real_dataset_mix_path: str = ""
    metadata_path: str = ""
    synthetic: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)
    reference_attention: Literal["ulysses", "async_ulysses"] = field(
        default="ulysses", metadata={"help": "Attention backend for baseline and anchor modes."}
    )
    warmup_steps: int = 3
    steps: int = field(default=20, metadata={"help": "Measured optimizer steps. Zero measures the full epoch."})
    kvf_query_blocks: Optional[int] = field(default=None, metadata={"help": "Override the KVF block count."})
    fixed_target_runtime: Optional[float] = field(
        default=None, metadata={"help": "Enable fixed-runtime packing for baseline and KVF modes."}
    )
    fixed_target_runtime_ms: Optional[int] = field(
        default=None, metadata={"help": "H800 peak-compute runtime budget used to derive the fixed packing target."}
    )
    measure_exposed_wait: Optional[bool] = field(
        default=None, metadata={"help": "Run a separate wait measurement. Enabled by default for kvf mode."}
    )
    output: Optional[str] = field(default=None, metadata={"help": "JSON result path on rank zero."})


@dataclass
class BenchmarkArguments(VeOmniDiTArguments):
    """Use a regular Wan training config plus benchmark options."""

    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


def configure_benchmark(args):
    """Apply one ablation before constructing the training runtime."""
    bench = args.benchmark
    if bench.mode not in ("baseline", "kvf", "anchor", "combined"):
        raise ValueError(f"Unknown benchmark mode {bench.mode!r}.")
    if bench.data_source not in ("real", "synthetic", "metadata"):
        raise ValueError(f"Unknown benchmark data source {bench.data_source!r}.")
    if (bench.data_source == "metadata") != bool(bench.metadata_path):
        raise ValueError("Set benchmark.metadata_path for the metadata data source.")
    if bench.data_source != "real":
        args.train.training_task = "offline_training"
    if args.train.training_task != "offline_training" or args.data.dataset_name != "mapping":
        raise ValueError("The benchmark requires Wan offline_training with a mapping dataset.")
    if args.train.checkpoint.load_path:
        raise ValueError("Use model weights through model.model_path for this benchmark, with no DCP resume.")
    if min(bench.warmup_steps, bench.steps) < 0:
        raise ValueError("Benchmark step counts must be nonnegative.")

    with_kvf = bench.mode in ("kvf", "combined")
    with_anchor = bench.mode in ("anchor", "combined")
    if bench.reference_attention not in ("ulysses", "async_ulysses"):
        raise ValueError(f"Unknown reference attention {bench.reference_attention!r}.")
    args.model.accelerator.enable_async = not with_kvf and bench.reference_attention == "async_ulysses"
    if bench.measure_exposed_wait is None:
        bench.measure_exposed_wait = bench.mode == "kvf"
    if bench.measure_exposed_wait and not with_kvf:
        raise ValueError("Exposed-wait timing requires KVF.")
    if args.model.accelerator.ulysses_size < 2 and (with_kvf or args.model.accelerator.enable_async):
        raise ValueError("KVF and async Ulysses require model.accelerator.ulysses_size > 1.")
    if bench.measure_exposed_wait:
        if os.getenv("TORCH_NCCL_BLOCKING_WAIT", "0") == "1" or os.getenv("NCCL_BLOCKING_WAIT", "0") == "1":
            raise ValueError("Exposed-wait timing requires NCCL stream waits. Unset NCCL blocking-wait variables.")

    args.model.model_config = dict(args.model.model_config or {})
    if with_kvf:
        blocks = bench.kvf_query_blocks
        if blocks is None:
            blocks = args.model.model_config.get("kvf_query_blocks") or 4
        if blocks < 1:
            raise ValueError("KVF requires a positive query block count.")
        args.model.model_config["kvf_query_blocks"] = blocks
        ops = args.model.ops_implementation
        if ops.wan_kvf_implementation is None:
            ops.wan_kvf_implementation = "nvshmem"
    else:
        args.model.model_config["kvf_query_blocks"] = 0

    fixed_target = bench.fixed_target_runtime
    if fixed_target is not None and (not math.isfinite(fixed_target) or fixed_target <= 0):
        raise ValueError("Fixed runtime target must be finite and positive.")
    if bench.fixed_target_runtime_ms is not None and (fixed_target is None or bench.fixed_target_runtime_ms <= 0):
        raise ValueError("A millisecond runtime budget requires a positive fixed packing target.")
    if with_anchor or fixed_target is not None:
        if args.data.anchor_batching is None:
            raise ValueError("Anchor or fixed-target packing needs data.anchor_batching calibration.")
    else:
        args.data.anchor_batching = None

    args.train.max_steps = None
    args.data.log_sample = False


def _synthetic_frames(config, seed):
    try:
        pairs = [str(entry).split(":") for entry in config.frame_counts]
        counts = [(int(frames), int(count)) for frames, count in pairs]
    except ValueError as exc:
        raise ValueError("Synthetic frame_counts entries must use FRAMES:COUNT.") from exc
    if not counts or any(frames < 1 or count < 1 for frames, count in counts):
        raise ValueError("Synthetic frame_counts must specify positive frames and sample counts.")
    result = [frames for frames, count in counts for _ in range(count)]
    random.Random(seed).shuffle(result)
    return result


class SyntheticWanDataset(Dataset):
    """Generate a deterministic latent posterior and text context per row."""

    def __init__(self, config, model_config, seed, *, expand_timesteps, frames=None):
        self.frames = _synthetic_frames(config, seed) if frames is None else frames
        if not self.frames or any(type(count) is not int or count < 1 for count in self.frames):
            raise ValueError("Generated Wan samples require positive integer latent frame counts.")
        if min(config.latent_height, config.latent_width, config.context_tokens) < 1:
            raise ValueError("Synthetic latent height, width, and context token count must be positive.")
        self.height = config.latent_height
        self.width = config.latent_width
        self.context_shape = (1, config.context_tokens, model_config.text_dim)
        self.channels = model_config.out_channels
        self.seed = seed
        pt, ph, pw = model_config.patch_size
        if self.height % ph or self.width % pw or any(frames % pt for frames in self.frames):
            raise ValueError("Synthetic latent dimensions must be divisible by the Wan patch size.")
        if expand_timesteps and min(self.frames) < 2:
            raise ValueError("Synthetic expanded timesteps require at least two latent frames per clip.")
        spatial_tokens = (self.height // ph) * (self.width // pw)
        self.lengths = [frames // pt * spatial_tokens for frames in self.frames]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frames = self.frames[index]
        generator = torch.Generator().manual_seed(self.seed + index % len(self))
        return [
            {
                "latents": torch.randn(1, 2 * self.channels, frames, self.height, self.width, generator=generator),
                "context": torch.randn(self.context_shape, generator=generator, dtype=torch.bfloat16),
            }
        ]


class SyntheticWanRuntime(DiTModelRuntime):
    """Run Wan offline conditioning without tokenizer, T5, or VAE weights."""

    def _build_condition_model(self, condition_model_type):
        from diffusers import FlowMatchEulerDiscreteScheduler

        from veomni.models.diffusers.wan_t2v.wan_condition.configuration_wan_condition import (
            WanTransformer3DConditionModelConfig,
        )
        from veomni.models.diffusers.wan_t2v.wan_condition.modeling_wan_condition import WanTransformer3DConditionModel

        class SyntheticWanCondition(WanTransformer3DConditionModel):
            def _load_components(self):
                self.scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.config.num_train_timesteps, shift=self.config.shift
                )

            def _normalize_latents(self, latents):
                return latents

        model_config = self.model_config
        overrides = dict(self.args.condition_model_cfg or {})
        if (
            condition_model_type != "WanTransformer3DConditionModel"
            or model_config.in_channels != model_config.out_channels
            or overrides.get("image_conditioning")
        ):
            raise ValueError(
                "Synthetic input supports Wan T2V and TI2V without channel-concatenated image conditioning."
            )
        overrides.update(
            seed=self.train_args.seed,
            patch_size=model_config.patch_size,
            cfg_negative_prob=0.0,
        )
        config = WanTransformer3DConditionModelConfig(**overrides)
        self.condition_model = SyntheticWanCondition(config)


def _fixed_target_rounds(lengths, config, target, dp_size, sp_size):
    """Greedily fill DP packs under one fixed runtime target and a memory limit."""
    costs = [(length + sp_size - 1) // sp_size * sp_size for length in lengths]
    a, b, c = config.runtime_coefficients
    runtimes = [(a * length + b) * length + c for length in lengths]
    rounds = []
    remaining = []
    for start in range(0, len(lengths), config.window_size):
        remaining.extend(range(start, min(len(lengths), start + config.window_size)))
        remaining.sort(key=lengths.__getitem__, reverse=True)
        while len(remaining) >= dp_size:
            packs = []
            for rank in range(dp_size):
                first, *remaining = remaining
                pack = [first]
                tokens, runtime = costs[first], runtimes[first]
                # An over-target individual sample runs as a singleton.
                reserve = dp_size - rank - 1
                skipped = []
                for pos, index in enumerate(remaining):
                    if (
                        len(remaining) - pos + len(skipped) > reserve
                        and tokens + costs[index] <= config.max_pack_tokens
                        and runtime + runtimes[index] <= target
                    ):
                        pack.append(index)
                        tokens += costs[index]
                        runtime += runtimes[index]
                    else:
                        skipped.append(index)
                packs.append(pack)
                remaining = skipped
            rounds.append(packs)
    return rounds


class MemDualBenchmarkTrainer(DiTTrainer):
    """Run actual DiT optimizer steps without logging or checkpoint callbacks."""

    samples = 0

    def _build_model_runtime(self):
        args = self.base.args
        if args.benchmark.data_source != "real":
            return SyntheticWanRuntime(args.model, "base", train=args.train)
        return super()._build_model_runtime()

    def _load_training_dataset(self):
        args = self.base.args
        if args.benchmark.data_source == "real":
            super()._load_training_dataset()
            dataset = self.base.train_dataset
            if args.benchmark.real_dataset_mix_path:
                mix = json.loads(Path(args.benchmark.real_dataset_mix_path).read_text())
                order = mix.pop("indices")
                if len(dataset) != sum(mix["available_counts"]):
                    raise ValueError("Real dataset row count does not match the mix manifest.")
                self.real_dataset_mix = mix
                self.base.train_dataset = Subset(dataset, order)
                args.compute_train_steps(len(order) / args.model.accelerator.dp_size)
                self.base.train_steps = args.train_steps
            elif args.benchmark.shuffle_real_samples:
                order = list(range(len(dataset)))
                random.Random(args.train.seed).shuffle(order)
                self.base.train_dataset = Subset(dataset, order)
            return
        frames = None
        if args.benchmark.data_source == "metadata":
            mix = json.loads(Path(args.benchmark.metadata_path).read_text())
            frames = mix.pop("latent_frames")
            self.metadata_mix = mix
        model = self.base.model
        self.base.train_dataset = SyntheticWanDataset(
            args.benchmark.synthetic,
            model.model_config,
            args.train.seed,
            expand_timesteps=model.condition_model.config.expand_timesteps,
            frames=frames,
        )
        args.compute_train_steps(len(self.base.train_dataset) / args.model.accelerator.dp_size)
        self.base.train_steps = args.train_steps

    def _load_anchor_lengths(self, config):
        if self.base.args.benchmark.data_source != "real":
            return self.base.train_dataset.lengths
        lengths = super()._load_anchor_lengths(config)
        if (
            self.base.args.benchmark.shuffle_real_samples
            and not self.base.args.benchmark.real_dataset_mix_path
            and isinstance(lengths, list)
        ):
            dataset = self.base.train_dataset
            if len(lengths) == len(dataset):
                return [lengths[index] for index in dataset.indices]
        return lengths

    def _setup_anchor_plan(self):
        super()._setup_anchor_plan()
        args = self.base.args
        target = args.benchmark.fixed_target_runtime
        if self.anchor_sampler is None or target is None or args.benchmark.mode in ("anchor", "combined"):
            return
        parallel = self.base.model.parallel_state
        rounds = _fixed_target_rounds(
            self.anchor_lengths, args.data.anchor_batching, target, parallel.dp_size, parallel.sp_size
        )
        self.anchor_sampler = AnchorStepSampler(
            rounds,
            parallel.dp_rank,
            parallel.dp_size,
            args.train.gradient_accumulation_steps,
            args.data.dataloader.shuffle,
            args.train.seed,
        )
        args._train_steps = len(self.anchor_sampler)
        self.base.train_steps = args.train_steps

    def on_step_begin(self, micro_batches=None):
        """Count each offline sample once across its Ulysses group."""
        if not self.base.model.parallel_state.sp_enabled or self.base.model.parallel_state.sp_rank == 0:
            self.samples += sum(len(batch["latents"]) for batch in micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None, aux_metrics=None):
        """Keep checkpoint, evaluation and metric logging outside the timed experiment."""


class _ExposedWaits:
    def __init__(self):
        self.events = []

    def __call__(self, wait):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wait()
        end.record()
        self.events.append((start, end))

    def drain(self):
        torch.cuda.synchronize()
        elapsed = sum(start.elapsed_time(end) for start, end in self.events)
        count = len(self.events)
        self.events.clear()
        return elapsed, count


@contextmanager
def _benchmark_iterator(trainer):
    parallel = trainer.base.model.parallel_state
    with use_parallel_state(parallel):
        trainer._build_dataloader()
    loader = trainer.base.train_dataloader
    iterator = None
    if loader is not None:
        loader.set_epoch(0)
        iterator = VeOmniIter(
            loader, use_background_prefetcher=trainer.base.args.data.dataloader.use_background_prefetcher
        )
    try:
        yield iterator
    finally:
        if iterator is not None:
            iterator.stop()


def _kvf_backend(model_args):
    if model_args.model_config["kvf_query_blocks"]:
        return getattr(getattr(model_args, "ops_implementation", None), "wan_kvf_implementation", None) or "nccl"
    return None


def _reserve_benchmark_kvf(trainer, bench):
    """Keep heap allocation outside the timed synthetic and metadata workloads."""
    args = trainer.base.args.model
    if _kvf_backend(args) != "nvshmem" or bench.data_source == "real":
        return
    if bench.data_source == "metadata":
        frames = max(json.loads(Path(bench.metadata_path).read_text())["latent_frames"])
    else:
        frames = max(int(entry.split(":")[0]) for entry in bench.synthetic.frame_counts)
    runtime = trainer.base.model
    config = runtime.model_config
    parallel = runtime.parallel_state
    pt, ph, pw = config.patch_size
    tokens = (frames // pt) * (bench.synthetic.latent_height // ph) * (bench.synthetic.latent_width // pw)
    block_bytes = kvf_block_bytes(
        1,
        math.ceil(tokens / parallel.ulysses_size),
        config.num_attention_heads,
        config.attention_head_dim,
        args.model_config["kvf_query_blocks"],
    )
    prepare_nvshmem_kvf(parallel.ulysses_group, torch.device("cuda", torch.cuda.current_device()), block_bytes)


def run_benchmark(trainer, bench):
    """Measure an uninstrumented training window and optionally collect KVF waits."""
    total_steps = trainer.base.train_steps
    steps = bench.steps or total_steps
    if steps > total_steps or bench.warmup_steps > total_steps or steps < 1:
        raise ValueError(f"The loader provides {total_steps} steps; adjust benchmark.steps or warmup_steps.")

    if bench.warmup_steps:
        with _benchmark_iterator(trainer) as iterator:
            for _ in range(bench.warmup_steps):
                trainer.train_step(iterator)
    trainer.samples = 0
    with _benchmark_iterator(trainer) as iterator:
        torch.cuda.synchronize()
        dist.barrier()
        start = perf_counter()
        for _ in range(steps):
            trainer.train_step(iterator)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start

    parallel = trainer.base.model.parallel_state
    rank = dist.get_rank()
    group_root = dist.get_global_rank(parallel.ulysses_group, 0) if parallel.ulysses_enabled else rank
    rank_stats = [None] * dist.get_world_size()
    dist.all_gather_object(
        rank_stats,
        (
            elapsed,
            trainer.samples,
            socket.gethostname(),
            parallel.local_rank,
            group_root,
            nvshmem_kvf_symmetric_bytes(),
        ),
    )
    slowest_rank = max(range(len(rank_stats)), key=lambda rank: rank_stats[rank][0])
    seconds = rank_stats[slowest_rank][0]
    samples = sum(row[1] for row in rank_stats)
    kvf_backend = _kvf_backend(trainer.base.args.model)
    uses_anchor = bench.mode in ("anchor", "combined")
    result = {
        "mode": bench.mode,
        "data_source": bench.data_source,
        "model_config_path": getattr(trainer.base.args.model, "config_path", None),
        "model_weights_path": getattr(trainer.base.args.model, "model_path", None),
        "synthetic": asdict(bench.synthetic) if bench.data_source != "real" else None,
        "metadata_mix": getattr(trainer, "metadata_mix", None),
        "real_dataset_mix": getattr(trainer, "real_dataset_mix", None),
        "packing": "anchor"
        if uses_anchor
        else "fixed_runtime"
        if bench.fixed_target_runtime is not None
        else "fixed_samples",
        "fixed_target_runtime": bench.fixed_target_runtime if not uses_anchor else None,
        "fixed_target_runtime_ms": bench.fixed_target_runtime_ms if not uses_anchor else None,
        "attention": "kvf" if bench.mode in ("kvf", "combined") else bench.reference_attention,
        "kvf_query_blocks": trainer.base.args.model.model_config["kvf_query_blocks"],
        "kvf_backend": kvf_backend,
        "nvshmem_symmetric_size": os.environ.get("NVSHMEM_SYMMETRIC_SIZE") if kvf_backend == "nvshmem" else None,
        "kvf_symmetric_bytes_per_rank": [stats[5] for stats in rank_stats],
        "dp_size": parallel.dp_size,
        "ulysses_size": parallel.ulysses_size,
        "world_size": len(rank_stats),
        "warmup_steps": bench.warmup_steps,
        "steps": steps,
        "samples": samples,
        "e2e_seconds": seconds,
        "e2e_slowest_rank": slowest_rank,
        "rank_placement": [
            {"rank": rank, "host": host, "local_rank": local_rank, "ulysses_group_root": group}
            for rank, (_, _, host, local_rank, group, _) in enumerate(rank_stats)
        ],
        "e2e_ms_per_step": 1000 * seconds / steps,
        "samples_per_second": samples / seconds,
    }

    if bench.measure_exposed_wait:
        tracker = _ExposedWaits()
        elapsed_ms = count = 0
        with _benchmark_iterator(trainer) as iterator, observe_ulysses_waits(tracker):
            for _ in range(steps):
                trainer.train_step(iterator)
                step_ms, step_count = tracker.drain()
                elapsed_ms += step_ms
                count += step_count
        waited = [None] * dist.get_world_size()
        dist.all_gather_object(waited, (elapsed_ms / steps, count))
        waits, counts = zip(*waited, strict=True)
        if not sum(counts):
            raise RuntimeError("No KVF Ulysses waits were observed.")
        result["kvf_exposed_wait_ms_per_step"] = {
            "mean_across_ranks": mean(waits),
            "max_across_ranks": max(waits),
        }
        result["kvf_wait_count_per_rank"] = list(counts)
    return result


def main():
    """Build Wan and run one benchmark mode through the usual distributed launcher."""
    args = parse_args(BenchmarkArguments)
    configure_benchmark(args)
    trainer = MemDualBenchmarkTrainer(args)
    try:
        if trainer.base.model.model_config.model_type != "WanTransformer3DModel":
            raise ValueError("The MemDual benchmark requires WanTransformer3DModel.")
        _reserve_benchmark_kvf(trainer, args.benchmark)
        result = run_benchmark(trainer, args.benchmark)
        close_nvshmem_kvf()
        if dist.get_rank() == 0:
            payload = json.dumps(result, indent=2)
            print(payload, flush=True)
            if args.benchmark.output:
                path = Path(args.benchmark.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(payload + "\n")
    finally:
        trainer.base.destroy_distributed()


if __name__ == "__main__":
    main()
