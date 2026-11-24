"""KV-first query/output pipelining for bidirectional Ulysses attention."""

from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ....distributed.sequence_parallel.ulysses import start_ulysses_exchange
from ...platform import GpuKernelRequirement, NvidiaGpuPlatform
from ...registry import register_op


def _process_blocks(blocks, exchange, compute, reverse):
    """Prefetch the next input and drain the previous output across compute."""
    pending_input = exchange(blocks[0])
    pending_output = None
    outputs = []
    for index in range(len(blocks)):
        current = pending_input
        if index + 1 < len(blocks):
            pending_input = exchange(blocks[index + 1])
        result = compute(index, current)
        next_output = reverse(result)
        if pending_output is not None:
            outputs.append(pending_output())
        pending_output = next_output
    outputs.append(pending_output())
    return torch.cat(outputs, dim=1)


class _FlashResult(NamedTuple):
    """Attention output and the statistics needed by the fused backward."""

    output: torch.Tensor
    lse: torch.Tensor
    cu_q: torch.Tensor | None
    cu_k: torch.Tensor | None
    max_q: int
    max_k: int
    seed: torch.Tensor
    offset: torch.Tensor


def _flash(query, key, value, scale):
    return _FlashResult(*torch.ops.aten._scaled_dot_product_flash_attention(query, key, value, scale=scale)[:-1])


def _subtract_shard(tensor, start, end):
    return torch.cat((tensor[:, :, :start], tensor[:, :, end:]), dim=2)


def _merge_attention(local, remote):
    lse = torch.logaddexp(local.lse, remote.lse)
    local_weight = (local.lse - lse).exp().unsqueeze(-1)
    remote_weight = (remote.lse - lse).exp().unsqueeze(-1)
    output = local.output.float() * local_weight + remote.output.float() * remote_weight
    return output.to(local.output.dtype), lse


class _KVF(torch.autograd.Function):
    """Run a row pipeline and save FlashAttention statistics for backward."""

    @staticmethod
    def forward(ctx, query, key, value, key_wait, value_wait, length, blocks, group, scale):
        """Exchange query blocks, run attention, and return sequence shards."""
        ctx.group = group
        ctx.scale = scale
        ctx.local_length = query.size(1)
        ctx.blocks = min(blocks, ctx.local_length)
        padded_length = ctx.local_length * dist.get_world_size(group)
        ctx.padding = padded_length - length
        rank = dist.get_rank(group)
        heads_per_rank = query.size(2) // dist.get_world_size(group)
        heads = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
        kv_start = rank * ctx.local_length
        kv_end = min(kv_start + ctx.local_length, length)
        query_blocks = query.tensor_split(ctx.blocks, dim=1)
        full_key = full_value = None
        states = []

        def compute(index, wait):
            nonlocal full_key, full_value
            partial = None
            if index == 0:
                # Start with the local query rows and local KV while the early
                # K/V exchange and the next query block are in flight.
                if 0 < kv_end - kv_start < length:
                    local_q = query_blocks[0][:, :, heads].transpose(1, 2)
                    local_k = key[:, : kv_end - kv_start, heads].transpose(1, 2)
                    local_v = value[:, : kv_end - kv_start, heads].transpose(1, 2)
                    partial = _flash(local_q, local_k, local_v, scale)
                full_key = key_wait()[:, :length].transpose(1, 2)
                full_value = value_wait()[:, :length].transpose(1, 2)

            q = wait().transpose(1, 2)
            if partial is None:
                state = _flash(q, full_key, full_value, scale)
            else:
                rows = query_blocks[0].size(1)
                start, end = rank * rows, (rank + 1) * rows
                remote_q = _flash(_subtract_shard(q, start, end), full_key, full_value, scale)
                remote_kv = _flash(
                    q[:, :, start:end],
                    _subtract_shard(full_key, kv_start, kv_end),
                    _subtract_shard(full_value, kv_start, kv_end),
                    scale,
                )
                local_output, local_lse = _merge_attention(partial, remote_kv)
                out = torch.cat((remote_q.output[:, :, :start], local_output, remote_q.output[:, :, start:]), dim=2)
                lse = torch.cat((remote_q.lse[..., :start], local_lse, remote_q.lse[..., start:]), dim=-1)
                state = remote_q._replace(output=out, lse=lse, max_q=q.size(2))
            states.append((q, state))
            return state.output.transpose(1, 2)

        output = _process_blocks(
            query_blocks,
            lambda x: start_ulysses_exchange(x, group),
            compute,
            lambda x: start_ulysses_exchange(x, group, reverse=True),
        )
        ctx.flash_sizes = [(state.max_q, state.max_k) for _, state in states]
        ctx.save_for_backward(
            full_key,
            full_value,
            *(
                tensor
                for q, state in states
                for tensor in (q, state.output, state.lse, state.cu_q, state.cu_k, state.seed, state.offset)
            ),
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Pipeline dO/dQ and reduce the K/V gradients across query blocks."""
        key, value, *states = ctx.saved_tensors
        dk = torch.zeros_like(key, dtype=torch.float32)
        dv = torch.zeros_like(value, dtype=torch.float32)
        group = ctx.group

        def compute(index, wait):
            q, out, lse, cu_q, cu_k, seed, offset = states[7 * index : 7 * (index + 1)]
            max_q, max_k = ctx.flash_sizes[index]
            dq, delta_k, delta_v = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                wait().transpose(1, 2),
                q,
                key,
                value,
                out,
                lse,
                cu_q,
                cu_k,
                max_q,
                max_k,
                0.0,
                False,
                seed,
                offset,
                scale=ctx.scale,
            )
            dk.add_(delta_k)
            dv.add_(delta_v)
            return dq.transpose(1, 2)

        dq = _process_blocks(
            grad_output.tensor_split(ctx.blocks, dim=1),
            lambda x: start_ulysses_exchange(x, group),
            compute,
            lambda x: start_ulysses_exchange(x, group, reverse=True),
        )
        dk_wait = start_ulysses_exchange(
            F.pad(dk.to(key.dtype), (0, 0, 0, ctx.padding)).transpose(1, 2), group, reverse=True
        )
        dv_wait = start_ulysses_exchange(
            F.pad(dv.to(value.dtype), (0, 0, 0, ctx.padding)).transpose(1, 2), group, reverse=True
        )
        return dq, dk_wait(), dv_wait(), None, None, None, None, None, None


def kvf_attention(query, key, value, *, key_wait, value_wait, length, blocks, group):
    """Apply bidirectional KVF to matching sequence-sharded [B, S, H, D] inputs."""
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Ulysses KVF requires matching Q/K/V shapes [batch, local_sequence, heads, head_dim].")
    world_size = dist.get_world_size(group)
    if not 0 < length <= query.size(1) * world_size or blocks < 1 or query.size(2) % world_size:
        raise ValueError(
            "Ulysses KVF requires a valid sequence length, positive block count, and heads divisible by SP."
        )

    head_dim = query.size(-1)
    if not 0 < head_dim <= 256:
        raise ValueError("The FlashAttention backend for Ulysses KVF supports head dimensions from 1 to 256.")
    scale = head_dim**-0.5
    padding = (-head_dim) % 8
    if not padding:
        return _KVF.apply(query, key, value, key_wait, value_wait, length, blocks, group, scale)

    def pad(tensor):
        return F.pad(tensor, (0, padding))

    # K/V exchanges were launched before Q was ready. Pad their results
    # after the wait, and let autograd crop the gradients through input padding.
    output = _KVF.apply(
        pad(query),
        pad(key),
        pad(value),
        lambda: pad(key_wait()),
        lambda: pad(value_wait()),
        length,
        blocks,
        group,
        scale,
    )
    return output[..., :head_dim]


register_op(
    "attention",
    "ulysses_kvf",
    "nccl",
    wrapper=kvf_attention,
    default=True,
    description="KV-first bidirectional Ulysses with pipelined NCCL exchanges and FlashAttention",
    requirement=GpuKernelRequirement(platforms=(NvidiaGpuPlatform(min_cc=80),)),
)
