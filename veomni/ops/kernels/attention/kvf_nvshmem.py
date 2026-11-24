"""Bounded NVSHMEM row exchanges for the Wan KVF attention pipeline."""

import ctypes
from functools import partial
from math import ceil

import torch
import torch.distributed as dist

from ....distributed.sequence_parallel.ulysses import wait_ulysses


# Two alternating slots for input and two for output. Forward and backward
# reuse the same slots. Activations saved for backward live in ordinary memory.
_SLOTS = 4
_exchange = None


def kvf_block_bytes(batch, local_rows, heads, head_dim, blocks, element_size=2):
    """Return bytes per rank of the largest row block after head padding."""
    rows = ceil(local_rows / min(blocks, local_rows))
    return batch * rows * heads * (ceil(head_dim / 8) * 8) * element_size


def _nvshmem_runtime():
    try:
        from cuda import core as cuda_core
        from nvshmem import bindings, core
        from nvshmem.core.interop import torch as interop

        device_type = getattr(cuda_core, "Device", None)
        if device_type is None:
            # CUDA Core 0.4, paired with the NVSHMEM 3.4 bindings.
            from cuda.core.experimental import Device

            device_type = Device

        # The tensor interop API resolves only allocation bases. Use the host
        # API that accepts byte counts and interior symmetric addresses.
        put_signal_nbi = getattr(bindings, "putmem_signal_nbi_on_stream", None)
        if put_signal_nbi is None:
            from cuda.pathfinder import load_nvidia_dynamic_lib

            library = load_nvidia_dynamic_lib("nvshmem_host")
            host = ctypes.CDLL(library.abs_path or "libnvshmem_host.so.3")
            put_signal_nbi = host.nvshmemx_putmem_signal_nbi_on_stream
            put_signal_nbi.argtypes = (
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            )
            put_signal_nbi.restype = None
        for name in ("quiet_on_stream", "signal_op_on_stream", "signal_wait_until_on_stream"):
            if not callable(getattr(bindings, name, None)):
                raise AttributeError(f"Missing NVSHMEM4Py host binding {name}.")
        return device_type, core, bindings, interop, put_signal_nbi
    except (ImportError, AttributeError, OSError) as exc:
        raise RuntimeError(
            "Hybrid KVF requires NVSHMEM4Py with host NBI support and a matching NVSHMEM/CUDA runtime. "
            "For the NCCL comparison use model.ops_implementation.wan_kvf_implementation=nccl."
        ) from exc


def _pack_view(tensor, peers, reverse):
    batch, rows, heads, head_dim = tensor.shape
    if reverse:
        tokens = rows // peers
        return tensor.unflatten(1, (peers, tokens)).permute(1, 0, 2, 3, 4), (batch, tokens, heads * peers, head_dim)
    heads_per_peer = heads // peers
    return tensor.unflatten(2, (peers, heads_per_peer)).permute(2, 0, 1, 3, 4), (
        batch,
        peers * rows,
        heads_per_peer,
        head_dim,
    )


def _unpack_view(output, peers, reverse):
    if reverse:
        return output.unflatten(2, (peers, output.size(2) // peers)).permute(2, 0, 1, 3, 4)
    return output.unflatten(1, (peers, output.size(1) // peers)).permute(1, 0, 2, 3, 4)


class _RowExchange:
    """Own the NVSHMEM PE, streams, and reusable symmetric exchange slots."""

    def __init__(self, group, device):
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.peers = [i for i in range(self.size) if i != self.rank]
        self.capacity = 0

        device_type, self.core, self.api, self.interop, self.put_signal_nbi = _nvshmem_runtime()
        torch.cuda.set_device(device)
        nvshmem_device = device_type(device.index)
        nvshmem_device.set_current()

        # Run the NVSHMEM4Py UID bootstrap within this Ulysses group.
        payload = [self.core.get_unique_id() if self.rank == 0 else None]
        dist.broadcast_object_list(payload, src=dist.get_global_rank(group, 0), group=group)
        dist.barrier(group=group)
        self.core.init(
            device=nvshmem_device, uid=payload[0], rank=self.rank, nranks=self.size, initializer_method="uid"
        )
        self.signal_add = int(self.core.SignalOp.SIGNAL_ADD)
        self.cmp_ge = int(self.core.ComparisonType.CMP_GE)
        self.send_stream = torch.cuda.Stream(device=device, priority=-1)
        self.recv_stream = torch.cuda.Stream(device=device, priority=-1)

    @property
    def symmetric_bytes(self):
        """Bytes in this rank's symmetric tensors, excluding runtime overhead."""
        return _SLOTS * (2 * self.capacity + 16) if self.capacity else 0

    def reserve(self, block_bytes):
        """Grow the slots collectively to fit a rank-local block."""
        if self.capacity >= block_bytes:
            return
        self.free()
        # Each slot contains one outbound and one inbound full rank-local
        # block, including all PE partitions. Grow only between pipelines.
        self.send = self.interop.tensor((_SLOTS, block_bytes), dtype=torch.uint8)
        self.recv = self.interop.tensor((_SLOTS, block_bytes), dtype=torch.uint8)
        self.signals = self.interop.tensor((_SLOTS, 2), dtype=torch.int64)
        self.signals.zero_()
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=self.group)
        self.capacity = block_bytes
        self.uses = [0] * _SLOTS
        self.copied = [None] * _SLOTS
        self.next_slot = [0, 0]

    def free(self):
        """Complete pending communication and collectively free the slots."""
        if not self.capacity:
            return
        # Complete all NBI puts and all release signals before collective free.
        for stream in (self.send_stream, self.recv_stream):
            self.api.quiet_on_stream(stream.cuda_stream)
        self.send_stream.synchronize()
        self.recv_stream.synchronize()
        for tensor in (self.signals, self.recv, self.send):
            self.interop.free_tensor(tensor)
        self.capacity = 0

    def start(self, tensor, *, reverse=False):
        """Exchange a CUDA row block and return a callable that waits for its output."""
        lane = int(reverse)
        slot = 2 * lane + self.next_slot[lane]
        self.next_slot[lane] ^= 1
        use = self.uses[slot]
        self.uses[slot] += 1
        source, output_shape = _pack_view(tensor, self.size, reverse)
        nbytes = tensor.numel() * tensor.element_size()
        chunk_bytes = nbytes // self.size
        sending = self.send[slot, :nbytes].view(tensor.dtype).view(source.shape)
        receiving = self.recv[slot, :nbytes].view(tensor.dtype).view(source.shape)
        arrival, release = self.signals[slot, 0].data_ptr(), self.signals[slot, 1].data_ptr()
        send_handle = self.send_stream.cuda_stream
        recv_handle = self.recv_stream.cuda_stream
        others = len(self.peers)

        # A sender needs both local source completion and every reader's
        # release before reusing the slot. Local self-copy also waits for its
        # previous reader. No device-wide barrier is needed in the pipeline.
        self.send_stream.wait_stream(torch.cuda.current_stream(self.device))
        if use:
            self.api.quiet_on_stream(send_handle)
            self.api.signal_wait_until_on_stream(release, self.cmp_ge, use * others, send_handle)
            self.send_stream.wait_event(self.copied[slot])
        with torch.cuda.stream(self.send_stream):
            sending.copy_(source, non_blocking=True)
            receiving[self.rank].copy_(sending[self.rank], non_blocking=True)
        self_copy = self.send_stream.record_event()
        tensor.record_stream(self.send_stream)

        for peer in self.peers:
            self.put_signal_nbi(
                receiving.data_ptr() + self.rank * chunk_bytes,
                sending.data_ptr() + peer * chunk_bytes,
                chunk_bytes,
                arrival,
                1,
                self.signal_add,
                peer,
                send_handle,
            )

        self.recv_stream.wait_event(self_copy)
        self.api.signal_wait_until_on_stream(arrival, self.cmp_ge, (use + 1) * others, recv_handle)
        with torch.cuda.stream(self.recv_stream):
            output = torch.empty(output_shape, device=self.device, dtype=tensor.dtype)
            _unpack_view(output, self.size, reverse).copy_(receiving, non_blocking=True)
        ready = self.recv_stream.record_event()
        self.copied[slot] = ready
        for peer in self.peers:
            self.api.signal_op_on_stream(release, 1, self.signal_add, peer, recv_handle)

        def wait():
            current = torch.cuda.current_stream(self.device)
            wait_ulysses(partial(current.wait_event, ready))
            output.record_stream(current)
            return output

        return wait


def prepare_nvshmem_kvf(group, device, block_bytes):
    """Reserve a reusable, group-local buffer for each outstanding row exchange."""
    global _exchange
    if _exchange is None:
        _exchange = _RowExchange(group, device)
    elif _exchange.group is not group or _exchange.device != device:
        raise RuntimeError("An NVSHMEM KVF process can own only one Ulysses group and GPU at a time.")
    _exchange.reserve(block_bytes)
    return _exchange


def nvshmem_kvf_symmetric_bytes():
    """Return current symmetric tensor bytes on this rank, or zero."""
    return _exchange.symmetric_bytes if _exchange is not None else 0


def close_nvshmem_kvf():
    """Release the process-local transport collectively after the last KVF step."""
    global _exchange
    if _exchange is not None:
        _exchange.free()
        _exchange.core.finalize()
        _exchange = None
