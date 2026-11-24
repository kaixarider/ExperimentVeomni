"""Exercise Anchor Batching planning, replica alignment, and loader resume."""

from types import SimpleNamespace

import pytest
import torch

from veomni.data.diffusion.anchor_batching import (
    AnchorBatchingConfig,
    AnchorStepSampler,
    build_anchor_dataloader,
    plan_anchor_rounds,
)


def test_anchor_rounds_keep_the_longest_sample_alone_and_balance_shorts():
    config = AnchorBatchingConfig("unused", long_boundary=8, max_pack_tokens=14, window_size=6)
    lengths = [12, 9, 8, 5, 4, 3, 5, 5, 4, 2, 2, 1, 1]
    rounds, dropped = plan_anchor_rounds(lengths, 3, 2, config)
    assert rounds[0][0] == [0]
    all_packs = [pack for row in rounds for pack in row]
    assert sorted(dropped + [idx for pack in all_packs for idx in pack]) == list(range(len(lengths)))
    assert all(pack and sum(lengths[i] + lengths[i] % 2 for i in pack) <= 14 for pack in all_packs)
    short_rounds, dropped = plan_anchor_rounds([7, 6, 5, 4, 3, 2, 1, 1, 1], 3, 1, config)
    assert short_rounds == [[[0, 5], [1, 4], [2, 3]], [[6], [7], [8]]]
    assert not dropped


def test_anchor_replicas_agree_on_microsteps_and_global_sample_count():
    rounds = [[[0], [1, 2]], [[3, 4], [5]], [[6], [7, 8, 9]]]
    samplers = [AnchorStepSampler(rounds, rank, 2, 2, True, 42) for rank in range(2)]
    for sampler in samplers:
        sampler.epoch = 1

    sampled = []
    for step in zip(*samplers):
        assert len({max(pack for _, pack, _ in batch) for batch in step}) == 1
        global_count = sum(map(len, step))
        for batch in step:
            assert {count for _, _, count in batch} == {global_count}
            sampled.extend(index for index, _, _ in batch)
    assert sorted(sampled) == list(range(10))


@pytest.mark.parametrize("num_workers", (0, 2))
def test_anchor_loader_resumes_at_the_same_pack(num_workers):
    rows = [[{"latents": torch.zeros(1, 2, length, 1, 1), "id": i}] for i, length in enumerate((7, 4, 5, 2, 1, 1))]
    lengths = [row[0]["latents"].size(2) for row in rows]
    rounds = [[[0], [1, 3]], [[2], [4, 5]]]
    config = SimpleNamespace(
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2 if num_workers else None,
        persistent_workers=False,
        in_order=True,
        worker_num_threads=None,
    )

    def build():
        sampler = AnchorStepSampler(rounds, 0, 2, 1, True, 4)
        loader = build_anchor_dataloader(rows, lengths, (1, 1, 1), sampler, lambda x: [s["id"] for s in x], config)
        loader.set_epoch(3)
        return loader

    loader = build()
    iterator = iter(loader)
    next(iterator)
    state = loader.state_dict()
    expected = list(iterator)
    resumed = build()
    resumed.load_state_dict(state)
    assert list(resumed) == expected
