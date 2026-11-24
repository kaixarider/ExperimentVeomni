"""Step-aligned, length-aware Anchor Batching for Wan DiT."""

import math
import random
from dataclasses import dataclass, field
from itertools import cycle

from torch.utils.data import Dataset, Sampler

from ...utils.device import get_device_type
from ..data_loader import DistributedDataloader, _build_worker_init_fn


@dataclass
class AnchorBatchingConfig:
    """Memory and runtime calibration for the anchor planner."""

    lengths_path: str
    long_boundary: int
    max_pack_tokens: int
    window_size: int = 256
    runtime_coefficients: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])

    def __post_init__(self):
        if min(self.long_boundary, self.max_pack_tokens, self.window_size) < 1:
            raise ValueError("Anchor boundaries, memory budget, and window size must be positive.")
        if (
            len(self.runtime_coefficients) != 3
            or any(not math.isfinite(c) or c < 0 for c in self.runtime_coefficients)
            or not any(self.runtime_coefficients)
        ):
            raise ValueError("Set three finite, nonnegative runtime coefficients [quadratic, linear, constant].")


def _long_round(remaining, dp_size, costs, runtimes, budget):
    anchor, *remaining = remaining
    packs = [[anchor]]
    for rank in range(1, dp_size):
        pack = []
        tokens = 0
        runtime = 0.0
        skipped = []
        reserve = dp_size - rank - 1
        for position, index in enumerate(remaining):
            if (
                len(remaining) - position + len(skipped) > reserve
                and tokens + costs[index] <= budget
                and runtime + runtimes[index] <= runtimes[anchor]
            ):
                pack.append(index)
                tokens += costs[index]
                runtime += runtimes[index]
            else:
                skipped.append(index)
        packs.append(pack)
        remaining = skipped
    return packs, remaining


def _short_round(remaining, dp_size, costs, budget):
    packs = [[index] for index in remaining[:dp_size]]
    tokens = [costs[pack[0]] for pack in packs]
    destinations = cycle([*range(dp_size - 1, -1, -1), *range(dp_size)])
    skipped = []
    for index in remaining[dp_size:]:
        for _ in range(2 * dp_size):
            rank = next(destinations)
            if tokens[rank] + costs[index] <= budget:
                packs[rank].append(index)
                tokens[rank] += costs[index]
                break
        else:
            skipped.append(index)
    return packs, skipped


def plan_anchor_rounds(lengths, dp_size, sp_size, config):
    """Return synchronized DP packs and the unscheduled final indices."""
    if dp_size < 1 or sp_size < 1 or config.window_size < dp_size:
        raise ValueError("Anchor Batching requires a planning window at least as large as the DP group.")
    if not all(type(length) is int and length > 0 for length in lengths):
        raise ValueError("Anchor Batching requires one positive integer token length per dataset row.")
    costs = [(length + sp_size - 1) // sp_size * sp_size for length in lengths]
    if any(cost > config.max_pack_tokens for cost in costs):
        raise ValueError("An individual SP-padded sample exceeds anchor_batching.max_pack_tokens.")
    a, b, c = config.runtime_coefficients
    runtimes = [(a * length + b) * length + c for length in lengths]
    rounds = []
    remaining = []
    for start in range(0, len(lengths), config.window_size):
        remaining.extend(range(start, min(len(lengths), start + config.window_size)))
        remaining.sort(key=lengths.__getitem__, reverse=True)
        while len(remaining) >= dp_size:
            if lengths[remaining[0]] >= config.long_boundary:
                packs, remaining = _long_round(remaining, dp_size, costs, runtimes, config.max_pack_tokens)
            else:
                packs, remaining = _short_round(remaining, dp_size, costs, config.max_pack_tokens)
            rounds.append(packs)
    return rounds, remaining


class _AnchorDataset(Dataset):
    def __init__(self, dataset, lengths, patch_size):
        self.dataset = dataset
        self.lengths = lengths
        self.patch_size = patch_size

    def __getitem__(self, key):
        index, pack, global_count = key
        (sample,) = self.dataset[index]
        grid = sample["latents"].shape[-3:]
        if any(size % patch for size, patch in zip(grid, self.patch_size)):
            raise ValueError(f"Wan latent dimensions {grid} at dataset row {index} must divide by patch_size.")
        length = math.prod(size // patch for size, patch in zip(grid, self.patch_size))
        if length != self.lengths[index]:
            raise ValueError(f"Anchor length metadata for dataset row {index} is {self.lengths[index]}, got {length}.")
        return sample, pack, global_count

    def __len__(self):
        return len(self.dataset)


class AnchorStepSampler(Sampler):
    """Yield per-replica indices for synchronized optimizer steps."""

    def __init__(self, rounds, dp_rank, dp_size, accumulation, shuffle, seed):
        self.rounds = rounds
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.accumulation = accumulation
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.rounds) / self.accumulation)

    def __iter__(self):
        order = list(range(len(self.rounds)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        for start in range(0, len(order), self.accumulation):
            round_ids = order[start : start + self.accumulation]
            global_count = sum(len(pack) for index in round_ids for pack in self.rounds[index])
            yield [
                (sample, micro_step, global_count)
                for micro_step, index in enumerate(round_ids)
                for sample in self.rounds[index][(self.dp_rank + index + self.epoch) % self.dp_size]
            ]


class _AnchorCollator:
    def __init__(self, collate):
        self.collate = collate

    def __call__(self, rows):
        packs = {}
        for sample, pack, _ in rows:
            packs.setdefault(pack, []).append(sample)
        return [self.collate(pack) for pack in packs.values()], rows[0][2]


class _AnchorDataLoader(DistributedDataloader):
    def set_epoch(self, epoch):
        """Keep the ordering and rank rotation synchronized across replicas."""
        self.batch_sampler.epoch = epoch


def build_anchor_dataloader(dataset, lengths, patch_size, sampler, collate, config):
    """Load the packs assigned to this replica with VeOmni's stateful dataloader."""
    worker_init_fn = (
        _build_worker_init_fn(config.worker_num_threads) if config.worker_num_threads is not None else None
    )
    return _AnchorDataLoader(
        _AnchorDataset(dataset, lengths, patch_size),
        batch_sampler=sampler,
        collate_fn=_AnchorCollator(collate),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        pin_memory_device=get_device_type(),
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        in_order=config.in_order,
        worker_init_fn=worker_init_fn,
    )
