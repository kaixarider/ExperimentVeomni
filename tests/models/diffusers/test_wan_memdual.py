"""Combined packed Wan, KVF, checkpoint recomputation, and FSDP2 parity."""

from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.checkpoint import checkpoint

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from scripts.profile.benchmark_memdual import BenchmarkConfig, MemDualBenchmarkTrainer, run_benchmark
from tests.models.tiny_configs import tiny_wan_t2v_config
from veomni.distributed.parallel_state import (
    ParallelState,
    _init_parallel_state,
    clear_parallel_state,
    use_parallel_state,
)
from veomni.models import get_model_class
from veomni.ops.kernels.attention.kvf_nvshmem import close_nvshmem_kvf
from veomni.trainer.dit_trainer import (
    AnchorBatchingConfig,
    DiTDataArguments,
    DiTModelRuntime,
)
from veomni.utils.import_utils import is_package_available


def _samples():
    generator = torch.Generator().manual_seed(93)
    samples = []
    for frames, width in ((3, 6), (2, 4), (1, 6)):
        hidden = torch.randn(1, 4, frames, 2, width, generator=generator).to(torch.bfloat16)
        context = torch.randn(1, 5, 32, generator=generator).to(torch.bfloat16)
        target = torch.randn(hidden.shape, generator=generator).to(torch.bfloat16)
        steps = torch.arange(frames * width // 2).reshape(1, -1)
        samples.append((hidden, steps, context, target))
    return samples


def _model(device, *, kvf, dtype=torch.bfloat16, backend=None):
    torch.manual_seed(9)
    config = tiny_wan_t2v_config(num_layers=2, attention_head_dim=32, kvf_query_blocks=kvf)
    config._set_ops_implementation({"attn_implementation": "veomni_sdpa", "wan_kvf_implementation": backend})
    return get_model_class(config)(config).to(device, dtype)


def _run_model(model, samples, *, packed):
    hidden, steps, context, targets = map(list, zip(*samples))
    return model(
        latents=None,
        hidden_states=hidden,
        timestep=steps,
        encoder_hidden_states=context,
        training_target=targets,
        pack_samples=packed,
    )


def _offline_condition(*, latents, timestep, context, target):
    return {
        "latents": latents,
        "hidden_states": latents,
        "timestep": timestep,
        "encoder_hidden_states": context,
        "training_target": target,
    }


def _run_train_step(model, parallel, samples, sidecar, device, expected, backend):
    data = DiTDataArguments(
        train_path="unused",
        anchor_batching=AnchorBatchingConfig(sidecar, long_boundary=8, max_pack_tokens=10),
    )
    data.dataloader.shuffle = False
    data.dataloader.num_workers = 0
    data.dataloader.prefetch_factor = None
    data.dataloader.pin_memory = False
    trainer = MemDualBenchmarkTrainer.__new__(MemDualBenchmarkTrainer)
    with_kvf = backend != "async_ulysses"
    benchmark = BenchmarkConfig(
        mode="combined" if with_kvf else "anchor",
        reference_attention="ulysses" if with_kvf else "async_ulysses",
        warmup_steps=1,
        steps=1,
        measure_exposed_wait=with_kvf,
    )
    trainer.training_task = "offline_training"
    runtime = DiTModelRuntime.__new__(DiTModelRuntime)
    runtime.model_name = "base"
    runtime.model = model
    runtime.model_config = model.config
    runtime.condition_model = SimpleNamespace(process_condition=_offline_condition)
    runtime.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    runtime.lr_scheduler = torch.optim.lr_scheduler.StepLR(runtime.optimizer, step_size=1)
    before = {name: parameter.to_local().detach().clone() for name, parameter in model.named_parameters()}
    gradients = {}

    def capture_gradients():
        for name, parameter in model.named_parameters():
            gradients[name] = parameter.grad.full_tensor().detach()
        return 0.0

    runtime.clip_grad_norm = capture_gradients
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(
            data=data,
            train=SimpleNamespace(gradient_accumulation_steps=1, seed=9),
            train_steps=1,
            benchmark=benchmark,
            model=SimpleNamespace(
                model_config={"kvf_query_blocks": model.config.kvf_query_blocks},
                ops_implementation=SimpleNamespace(wan_kvf_implementation=backend),
            ),
        ),
        device=device,
        model=runtime,
        state=SimpleNamespace(global_step=0),
        train_steps=1,
        train_dataset=[
            [{"latents": hidden, "timestep": step, "context": context, "target": target}]
            for hidden, step, context, target in samples
        ],
        model_fwd_context=nullcontext(),
        model_bwd_context=nullcontext(),
        LOG_SAMPLE=False,
    )
    for name in (
        "on_step_begin",
        "on_step_end",
        "sync_before_train_step",
        "_reset_async_activation_offload_if_enabled",
        "model_reshard",
        "_configure_hsdp_allreduce",
    ):
        setattr(trainer.base, name, Mock())

    with use_parallel_state(parallel):
        trainer._setup_anchor_plan()
        trainer._build_dataloader()
    loader = trainer.base.train_dataloader
    predictions = []
    handle = model.register_forward_hook(lambda _model, _args, output: predictions.extend(output.predictions))
    trainer.train_step(iter(loader) if loader is not None else None)
    handle.remove()

    local_indices = [0] if parallel.dp_rank == 0 else [1, 2]
    for prediction, i in zip(predictions, local_indices, strict=True):
        torch.testing.assert_close(prediction, expected.predictions[i], atol=0.025, rtol=0.04)
    assert trainer.base.state.global_step == 1
    assert runtime.lr_scheduler.last_epoch == 1
    assert any(not torch.equal(before[name], parameter.to_local()) for name, parameter in model.named_parameters())
    runtime.clip_grad_norm = lambda: 0.0
    result = run_benchmark(trainer, benchmark)
    assert result["steps"] == 1
    assert result["samples"] == 3
    assert result["e2e_seconds"] > 0
    assert result["attention"] == ("kvf" if with_kvf else "async_ulysses")
    assert result["kvf_backend"] == (backend if with_kvf else None)
    assert all((size > 0) == (backend == "nvshmem") for size in result["kvf_symmetric_bytes_per_rank"])
    if with_kvf:
        assert result["kvf_exposed_wait_ms_per_step"]["max_across_ranks"] > 0
        assert all(count > 0 for count in result["kvf_wait_count_per_rank"])
    return gradients


def _check_memdual(rank, store, sidecar, backend):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=store, rank=rank, world_size=4)
    try:
        parallel = _init_parallel_state(
            dp_size=2, dp_shard_size=2, ulysses_size=2, device_type="cuda", async_enabled=backend == "async_ulysses"
        )
        samples = _samples()
        reference = _model(device, kvf=0)
        with use_parallel_state(ParallelState(dp_size=4, dp_shard_size=4, device_type="cuda")):
            expected = _run_model(reference, [tuple(x.to(device) for x in sample) for sample in samples], packed=False)
            expected.loss["mse_loss"].backward()

        with use_parallel_state(parallel):
            model = _model(
                device,
                kvf=0 if backend == "async_ulysses" else 3,
                dtype=torch.float32,
                backend=None if backend == "async_ulysses" else backend,
            )
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            model.gradient_checkpointing = True
            model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
            for block in model.blocks:
                fully_shard(block, mesh=parallel.fsdp_mesh, mp_policy=policy)
            fully_shard(model, mesh=parallel.fsdp_mesh, mp_policy=policy)
        gradients = _run_train_step(model, parallel, samples, sidecar, device, expected, backend)
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(gradients[name], parameter.grad.float(), atol=0.012, rtol=0.09, msg=name)
        close_nvshmem_kvf()
    finally:
        dist.destroy_process_group()
        clear_parallel_state()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=4))
@pytest.mark.skipif(
    torch.version.hip is not None or runner_device_count() < 4 or not dist.is_nccl_available(),
    reason="combined Wan MemDual parity requires four NCCL-enabled NVIDIA GPUs",
)
@pytest.mark.parametrize("backend", ("nccl", "nvshmem", "async_ulysses"))
def test_wan_memdual_pack_on_uneven_replicas(tmp_path, backend):
    if backend == "nvshmem" and not all(is_package_available(name) for name in ("nvshmem", "cuda.core")):
        pytest.skip("Hybrid KVF requires NVSHMEM4Py and CUDA Python.")
    if torch.cuda.device_count() < 4 or any(torch.cuda.get_device_capability(rank)[0] < 8 for rank in range(4)):
        pytest.skip("Requires 4 visible NVIDIA SM80+ GPUs.")
    sidecar = tmp_path / "lengths.json"
    sidecar.write_text("[9, 4, 3]")
    mp.spawn(_check_memdual, args=((tmp_path / "nccl").as_uri(), str(sidecar), backend), nprocs=4)
