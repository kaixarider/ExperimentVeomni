"""Compare the KVF row pipeline against dense attention on two NVIDIA GPUs."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from veomni.distributed.sequence_parallel.ulysses import start_ulysses_exchange
from veomni.ops import VeomniOp
from veomni.ops.kernels.attention.kvf_nvshmem import close_nvshmem_kvf
from veomni.utils.import_utils import is_package_available


def _check_kvf(rank, store, backend):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=store, rank=rank, world_size=2)
    try:
        group = dist.group.WORLD
        op = VeomniOp("attention", "ulysses_kvf", backend)
        for batch, length, heads, head_dim, blocks, dtype in (
            (2, 32, 4, 64, 1, torch.bfloat16),
            (2, 29, 4, 64, 3, torch.bfloat16),
            (1, 1, 4, 64, 4, torch.bfloat16),
            (3, 17, 6, 7, 2, torch.float16),
            (2, 31, 4, 97, 3, torch.bfloat16),
            (1, 13, 2, 255, 3, torch.float16),
        ):
            shard_size = (length + 1) // 2
            start = rank * shard_size
            torch.manual_seed(72)
            reference = [
                torch.randn(batch, length, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
                for _ in range(3)
            ]
            inputs = [
                F.pad(x.detach(), (0, 0, 0, 0, 0, 2 * shard_size - length))[:, start : start + shard_size]
                .contiguous()
                .requires_grad_()
                for x in reference
            ]
            doutput = torch.randn_like(reference[0])
            local_grad = F.pad(doutput, (0, 0, 0, 0, 0, 2 * shard_size - length))[:, start : start + shard_size]
            query, key, value = inputs
            output = op(
                query,
                key,
                value,
                key_wait=start_ulysses_exchange(key, group),
                value_wait=start_ulysses_exchange(value, group),
                length=length,
                blocks=blocks,
                group=group,
            )
            expected = F.scaled_dot_product_attention(*(x.transpose(1, 2).float() for x in reference)).transpose(1, 2)
            expected.backward(doutput.float())
            output.backward(local_grad)
            stop = min(start + shard_size, length)
            torch.testing.assert_close(
                output[:, : stop - start].float(), expected[:, start:stop], atol=0.012, rtol=0.025
            )
            for actual_input, ref_input in zip(inputs, reference):
                padded = F.pad(ref_input.grad, (0, 0, 0, 0, 0, 2 * shard_size - length))
                torch.testing.assert_close(
                    actual_input.grad, padded[:, start : start + shard_size], atol=0.025, rtol=0.04
                )
        close_nvshmem_kvf()
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2))
@pytest.mark.skipif(
    torch.version.hip is not None or runner_device_count() < 2 or not dist.is_nccl_available(),
    reason="KVF requires two NCCL-enabled NVIDIA GPUs",
)
@pytest.mark.parametrize("backend", ("nccl", "nvshmem"))
def test_kvf_output_and_qkv_gradient_parity(tmp_path, backend):
    if backend == "nvshmem" and not all(is_package_available(name) for name in ("nvshmem", "cuda.core")):
        pytest.skip("Hybrid KVF requires NVSHMEM4Py and CUDA Python.")
    if torch.cuda.device_count() < 2 or any(torch.cuda.get_device_capability(rank)[0] < 8 for rank in range(2)):
        pytest.skip("Requires 2 visible NVIDIA SM80+ GPUs.")
    mp.spawn(_check_kvf, args=((tmp_path / backend).as_uri(), backend), nprocs=2)
