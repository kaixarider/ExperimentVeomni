# Long-Sequence Training Using Ulysses

## 📚 Overview
In this tutorial, we introduce the implementation of DeepSpeed-Ulysses for efficient long-sequence training in VeOmni. The Ulysses method optimizes memory usage by splitting both the input tensor and intermediate activations along the sequence dimension. This innovative approach significantly enhances memory efficiency, enabling the training of models with longer sequence lengths.

Reference Paper: [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)

## 🚀 Quick Start
To enable Ulysses, users can specify the `accelerator.ulysses_size` parameter in the configuration file or the launch command:

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen25_vl/qwen25_vl.yaml \
    --model.model_path YOUR_MODEL_PATH \
    --data.train_path YOUR_DATA_PATH \
    --model.accelerator.ulysses_size 4
```

VeOmni currently supports Ulysses on the following models:

Language models:
- LLaMA
- Qwen2.5
- Qwen3.5 (hybrid softmax + linear attention, requires transformers v5)

Multimodal models:
- Qwen2-VL
- Qwen2.5-VL

## 🔍 Dive into Ulysses Sequence Parallelism
Sequence Parallel (SP) is a common strategy for handling sequences that exceed the memory limit of a single GPU. Ulysses uses all-to-all collective communication operations to implement SP for attention.

### What is all_to_all?

Suppose we have P GPUs and a sequence whose shape is [S, H], where S denotes the sequence length and H represents the hidden size (number of heads \* head dimension). Each GPU initially holds an [S/P, H] partition. After an all_to_all operation, each GPU holds a head-partitioned sequence whose shape is [S, H/P]. The following figure illustrates the case where P = 4:

![all_to_all communication](../assets/all_2_all.jpg)

### DeepSpeed-Ulysses
We use the all-to-all-based sequence parallelism proposed by DeepSpeed, named DeepSpeed-Ulysses.

![DeepSpeed-Ulysses](../assets/ulysses.png)

(Image source: [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509))

The figure above shows the overall architecture of DeepSpeed-Ulysses. It introduces only two additional all_to_all communications in the attention module and does not modify other components such as normalization and MLP. The input sequence is first divided evenly across the GPUs. The first all_to_all communication gathers the query, key, and value ([S/P, H]) along the sequence dimension and scatters them along the head dimension ([S, H/P]). After attention, another all_to_all transfers the output ([S, H/P]) from head-partitioned form back to sequence-partitioned form ([S/P, H]). DeepSpeed-Ulysses is attention-agnostic because it gathers the entire sequence during attention computation, so it can be used with FlashAttention. However, the number of query heads must be divisible by the Ulysses parallel size; grouped-query attention has additional key/value head constraints. DeepSpeed-Ulysses does not affect memory consumed by model states. For long-sequence training with a large language model, it can be combined with ZeRO or FSDP.

### Communication Analysis
The communication volume transmitted per link for an all-to-all operation with an aggregate message of size M over P GPUs can be estimated as $M(P-1)/P^2$. For a transformer model with hidden size H, sequence length S, and parallelism degree P, let $\mu = (P-1)/P$. Before attention, DeepSpeed-Ulysses performs an all-to-all for the QKV projections with an aggregate message size of $3SH$, introducing a communication volume of $3SH\mu/P$. Each transformer layer performs another all-to-all for the output context projection with an aggregate message size of $SH$, introducing $SH\mu/P$. Therefore, DeepSpeed sequence parallelism incurs an aggregate communication volume per link of $4SH\mu/P = 4M(P-1)/P^2$.

## ⚙️ Core API

1. gather_seq_scatter_heads

A method to do all-to-all before attention, make sure the sequence is full for attention score computation.

```Python
def gather_seq_scatter_heads(
    x: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    async_op: bool = False,
) -> Tensor
```

Args:
- x: tensor to be synced
- seq_dim: sequence dim that will be gathered
- head_dim: head dim that will be scattered. "head" is a concept from Multi-head Attention or Group Query Attention
- unpadded_dim_size: the full sequence size before padding and sharding
- async_op: if True, will return a torch._C._distributed_c10d.Work object, users can use this arg to do comm-compute overlap

2. gather_heads_scatter_seq

A method to do all-to-all after attention, transforming the activation from head-split to sequence-split.

```Python
def gather_heads_scatter_seq(
    x: torch.Tensor,
    head_dim: int,
    seq_dim: int
) -> Tensor
```

Args:
- x: Tensor to be synced
- head_dim: head dim that will be gathered
- seq_dim: sequence dim that will be scattered


3. reduce_sequence_parallel_loss

A method to reduce loss within the sequence parallel group, re-scale the loss according to the number of valid tokens.

```Python
def reduce_sequence_parallel_loss(
    loss: torch.Tensor,
    num_valid_tokens: torch.Tensor
) -> torch.Tensor
```

Args:
- loss: loss tensor to be reduced
- num_valid_tokens: the number of valid tokens in current rank

## 🛠️ Support Ulysses for a New Model

Typically, enabling Ulysses for a new model involves three key steps:

1. Create a sequence parallel group based on the specified Ulysses parallel size.
2. Shard input sequences across the sequence parallel groups.
3. Modify the model’s attention and loss computation to support Ulysses.

In VeOmni, the first two steps are automated. Users only need to specify the `accelerator.ulysses_size` parameter, and VeOmni will handle the creation of sequence parallel groups and the sharding of input data. This allows users to focus solely on step 3—modifying the model’s architecture to implement Ulysses sequence parallelism.

To make this process easier, we provide an abstract pseudo-code example as a reference for implementing Ulysses in a new model:

```Python
from veomni.distributed.sequence_parallel import (
    gather_seq_scatter_heads_qkv,
    gather_heads_scatter_seq,
    reduce_sequence_parallel_loss,
)
# Step1: Create a sequence parallel group
# VeOmni will construct a sequence parallel group based on the parallel size

# Step2: Shard input sequences
# Suppose we get an input x of shape [batch_size, seq_len, dim]
# we first shard x among sequence parallel groups
# now x is of shape [batch_size, seq_len/n, dim] on each sp rank

# Step3 (part1): modify attention computation
x = self.qkv(x) # [batch_size, seq_pad/n, dim]
x = gather_seq_scatter_heads(x, seq_dim=1, head_dim=2) # [batch_size, seq_len, dim/n]
...
output = F.scaled_dot_product_attention(q, k, v, ...).reshape(...)
...
output = gather_heads_scatter_seq(output, head_dim=2, seq_dim=1) # [batch_size, seq_pad/n, dim]

# Step3 (part2): reduce loss after model forward
loss = loss_fct(logits, labels)
loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
return loss
```

## 🧩 Implementation Details: Data Pipeline and Model Interaction

Understanding how sequence parallelism data flows through the VeOmni pipeline is critical for
adding SP support to new models and for debugging shape mismatches. This section documents the
full lifecycle of tensors from the data collator to the model forward pass.

### Data Collator: `SequenceParallelCollator`

When SP is enabled, `MainCollator` appends a `SequenceParallelCollator` to its pipeline
(see `veomni/data/data_collator.py`). This collator handles three operations in order:

1. **Label shifting** — shifts `labels` left by 1 token (for next-token prediction).
2. **SP padding and slicing** — for each key in the batch:
   - **Pad** the sequence to be evenly divisible by `sp_size` (using `sp_pad_value` from `DataCollateInfo`).
   - **Slice** the sequence for the current SP rank (`tensor.narrow(dim, rank * chunk, chunk)`).
3. **Flash attention kwargs** — computed from `position_ids` *before* slicing it.

The default `DataCollateInfo` for each key:

| Key | `sp_slice` | `sp_pad_value` | Notes |
|-----|-----------|----------------|-------|
| `input_ids` | True | 0 | Sliced to local length |
| `labels` | True | -100 | Sliced to local length |
| `attention_mask` | False | 1 | Padded but NOT sliced (always all-ones for FA) |
| `position_ids` | False | 0 | Sliced **after** FA kwargs are computed from it |

**Key ordering detail:** `position_ids` is intentionally excluded from the general slicing loop.
It is first used at full length to compute `cu_seq_lens_q/k` via `add_flash_attention_kwargs_from_position_ids`,
then sliced afterward. This ensures the FA kwargs describe the full packed sequence boundaries, which
is needed by the flash attention kernel after the Ulysses all-to-all gathers the full sequence.

After the collator, the model receives:

| Tensor | Sequence length | Description |
|--------|----------------|-------------|
| `input_ids` | `S / sp_size` | Local token IDs |
| `labels` | `S / sp_size` | Local shifted labels |
| `position_ids` | `S / sp_size` | Local positions (correct absolute values) |
| `attention_mask` | `S` | Full-length all-ones mask |
| `cu_seq_lens_q` | varies | Computed from full `position_ids` before slicing |

### Softmax Attention (Flash Attention) SP Flow

For standard softmax attention layers (e.g., `Qwen3_5Attention`), Ulysses SP is handled
**internally** by `flash_attention_forward` in
`veomni/ops/kernels/attention/standard/flash.py`.

The flow through a softmax attention layer:

```
hidden_states [B, S_local, D]           # already local from collator
  -> QKV projection                      # [B, S_local, num_heads, head_dim]
  -> apply_rotary_pos_emb(q, k, cos, sin)  # RoPE on local-length q/k
  -> flash_attention_forward:
       gather_seq_scatter_heads(q,k,v)   # [B, S_full, local_heads, head_dim]
       flash_attention_op(...)       # attention on full sequence, local heads
       gather_heads_scatter_seq(output)  # [B, S_local, num_heads, head_dim]
  -> output projection                   # [B, S_local, D]
```

**Important:** Position embeddings (`cos`, `sin`) must match `hidden_states` length (`S_local`).
Since `position_ids` is already sliced by the collator, `rotary_emb(hidden_states, position_ids)`
produces local-length position embeddings — no additional slicing is needed.

> **Note on Qwen3/Qwen3-MoE:** These models use LigerKernel's `apply_rotary_pos_emb` which tolerates
> mismatched sequence lengths between q/k and cos/sin. Qwen3.5 uses the standard HuggingFace
> implementation, which requires exact size match. When adding SP to a new model, always check
> which RoPE implementation is active.

### Loss Reduction

When SP is enabled, each rank computes loss on its local sequence shard. The loss must be
reduced across the SP group, re-scaled by the number of valid (non-padding) tokens per rank.
This is handled by `reduce_sequence_parallel_loss` or by the fused loss in the model's forward
method. See the Core API section above for details.

---

## 🔧 Linear Attention Ulysses (GatedDeltaNet)

Hybrid models like Qwen3.5 alternate between softmax attention layers and linear attention
layers (GatedDeltaNet). While softmax attention SP is handled transparently by
`flash_attention_forward`, linear attention layers require **explicit Ulysses SP logic** in
their forward method because they use a different attention mechanism (recurrence-based, not
score-based) with additional components like causal conv1d.

### Why Linear Attention Needs Special Handling

1. **No shared flash attention path** — GatedDeltaNet uses `chunk_gated_delta_rule` from FLA, not flash attention.
2. **Causal conv1d** — operates along the sequence dimension. Each rank only has a local sequence shard, so the conv1d must run on the full sequence with appropriately sharded weights.
3. **Per-head parameters** — `A_log` and `dt_bias` are indexed by head, so they need slicing for local heads.

### GatedDeltaNet Ulysses SP Flow

```
hidden_states [B, S_local, D]
  -> QKV + z + b + a projections

  IF ulysses_enabled:
    -> split mixed_qkv into Q, K, V and reshape to [B, S_local, num_heads, head_dim]
    -> gather_seq_scatter_heads(Q, K, V, b, a)   # [B, S_full, local_heads, ...]
    -> flatten heads + concat Q,K,V               # [B, S_full, local_conv_dim]
    -> _get_local_conv1d_weight(rank)             # shard conv1d weights for local heads
    -> causal_conv1d(mixed_qkv, local_weight, cu_seqlens=cu_seq_lens_q)
    -> split and reshape with local head counts
    -> slice A_log, dt_bias for local V-heads
    -> chunk_gated_delta_rule(q, k, v, g, beta, cu_seqlens=cu_seq_lens_q)
    -> gather_heads_scatter_seq(attn_out)         # [B, S_local, num_v_heads, head_v_dim]
    -> gated_norm(attn_out, z)                    # z was NOT all-to-all'd
    -> output projection
  ELSE:
    -> standard (non-SP) forward path
```

### Conv1d Weight Sharding

The conv1d weight has shape `[Q_channels | K_channels | V_channels, kernel_size]`.
Under Ulysses SP, each rank only processes local heads, so the conv1d weight must be
sharded to extract only the channels corresponding to local Q, K, V heads:

```python
def _get_local_conv1d_weight(self, ulysses_rank, local_key_dim, local_value_dim):
    w_full = self.conv1d.weight.squeeze(1)  # [conv_dim, kernel_size]
    k_off = ulysses_rank * local_key_dim
    v_off = ulysses_rank * local_value_dim
    w_q = w_full[k_off : k_off + local_key_dim]
    w_k = w_full[key_dim + k_off : key_dim + k_off + local_key_dim]
    w_v = w_full[2 * key_dim + v_off : 2 * key_dim + v_off + local_value_dim]
    return torch.cat([w_q, w_k, w_v], dim=0)
```

### Key Difference: `z` Is NOT All-to-All'd

The gating signal `z` is used *after* the attention output is gathered back to local
sequence layout (`gather_heads_scatter_seq`). Since `z` starts at `[B, S_local, num_v_heads, head_v_dim]`
and the gathered attention output is also `[B, S_local, num_v_heads, head_v_dim]`, they
match without any all-to-all on `z`. This saves one all-to-all communication.

### `cu_seq_lens_q` Usage

`cu_seq_lens_q` describes the packed sequence boundaries (computed from full `position_ids` before
slicing). After the all-to-all gathers the full sequence, `cu_seq_lens_q` correctly describes the
sequence boundaries for the causal conv1d and chunk attention kernels.

### Head Divisibility Requirement

Both `num_k_heads` and `num_v_heads` must be divisible by `ulysses_size`. For GQA (grouped query
attention) where `num_v_heads > num_k_heads`, the GQA repeat ratio `num_v_heads // num_k_heads`
is preserved after dividing both by `ulysses_size`.

### Testing

Tests for GatedDeltaNet Ulysses SP are in `tests/parallel/ulysses/test_qwen3_5_gated_deltanet_ulysses.py`:

1. **Conv1d weight slicing** — single-GPU, validates that sliced conv1d output matches the
   corresponding slice of full conv1d output.
2. **SP forward/backward equivalence** — multi-GPU, compares SP partitioned outputs and gradients
   against a non-SP baseline.
3. **SP forward determinism** — multi-GPU, verifies repeated forward passes produce identical outputs.
4. **Non-SP forward determinism** — single-GPU, baseline determinism check.

---

## ⚡ Async Ulysses CP

We also support **Async Ulysses** which further improves performance by overlapping communication and computation, reducing communication latency and improving hardware utilization.

### Asynchronous Ulysses

![async_ulysses](../assets/async_ulysses.jpg)

VeOmni extends the original Ulysses implementation with asynchronous communication capabilities, further improving performance by overlapping communication and computation.

#### Performance Benefits

By overlapping communication and computation, Async Ulysses:
- Reduces idle time during communication operations
- Lowers end-to-end training time
- Maintains nearly the same memory efficiency as original Ulysses

#### Enabling Async Ulysses

To enable Async Ulysses, simply set the `accelerator.enable_async` parameter to `True`:

Notice: Async Ulysses works when `accelerator.ulysses_size > 1`.

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.accelerator.ulysses_size 4 \
    --model.accelerator.enable_async true
```


### API

Async Ulysses is two registered ops. Modeling constructs a local handle and
calls it; there is no public `async_ulysses_qkv_projection` /
`async_ulysses_output_projection` function.

| Op | Variants | Implementation |
|---|---|---|
| `async_ulysses_qkv` | `standard`, `dit` | `eager` |
| `async_ulysses_o` | `standard`, `dit` | `eager` |

`standard` is the transformer path (Qwen3-VL / Qwen3-VL-MoE). `dit` is the
diffusion path (Wan). Nested RMSNorm uses a raw inner pair; LayerNorm still
goes through the fused CUDA helper in `veomni/ops/kernels/async_ulysses/shared/`.

```python
from veomni.ops import VeomniOp

q, k, v = VeomniOp("async_ulysses_qkv", "standard")(
    hidden_states=hidden_states,
    seq_dimension=1,
    head_dimension=2,
    q_weight=self.q_proj.weight,
    q_bias=self.q_proj.bias,
    k_weight=self.k_proj.weight,
    k_bias=self.k_proj.bias,
    v_weight=self.v_proj.weight,
    v_bias=self.v_proj.bias,
    norm_type="rmsnorm",
    norm_q_weight=self.q_norm.weight,
    norm_k_weight=self.k_norm.weight,
    normalized_shape=self.head_dim,
    eps=self.config.rms_norm_eps,
    unpadded_dim_size=unpadded_seq_len * get_ulysses_sequence_parallel_world_size(),
    head_dim=self.head_dim,
)

attn_output = VeomniOp("async_ulysses_o", "standard")(
    hidden_states=attn_output,
    seq_dimension=1,
    head_dimension=2,
    proj_weight=self.o_proj.weight,
    proj_bias=self.o_proj.bias,
    unpadded_dim_size=attn_output.shape[1],
)
```

See `veomni/models/transformers/qwen3_vl/modeling_qwen3_vl.py` for
the live transformer consume, and `veomni/models/transformers/wan/modeling_wan.py`
for the `dit` consume. Registry rows live in
`veomni/ops/kernels/async_ulysses/`.

### Enabling Async Ulysses

To enable Async Ulysses for an existing model, you need to:

1. Check that the model consumes `async_ulysses_qkv` / `async_ulysses_o`
2. Set `accelerator.enable_async=True` in your training configuration
3. Keep Ulysses sequence parallelism enabled (`accelerator.ulysses_size > 1`)
4. For the transformer path, use a flash-attention implementation; the async
   consume requires the packed-varlen contract

Async Ulysses is currently consumed by:

- Qwen3-VL
- Qwen3-VL-MoE
- Wan

Support for more models will be added in future releases.

## MemDual for Wan Video Diffusion

Wan's DiT trainer supports two independent opt-in settings from MemDual.
**KV-first Q-pipeline (KVF)** starts the V exchange before K and Q projection,
starts K exchange before Q projection, and computes an initial local attention
partial while communication is in flight. It then pipelines incoming Q blocks
and outgoing O blocks in forward, and dO and dQ in backward. K/V and dK/dV
use NCCL all-to-all. The MemDual benchmark uses host-issued
NVSHMEM nonblocking puts with completion signals for Q/O and dO/dQ. It
reuses four symmetric send/receive slots per Ulysses group across layers
and both passes. The all-NCCL implementation remains selectable.

**Anchor Batching** groups long and short videos across the data-parallel
replicas. In a round with a long anchor, the longest sample is assigned
a pack of its own. The other packs are filled under the memory budget and
the anchor's estimated runtime. A short-only round distributes clips in
zigzag length order under the memory budget. Wan runs each transformer
block once per pack, with separate self- and cross-attention sequences for
each video. Loss is averaged over all samples in the optimizer step.

Merge the following settings into a Wan configuration and launch through
`bash train.sh tasks/train_dit.py CONFIG.yaml`. Calibrate the example
thresholds against the model, accelerator, and Ulysses degree in use.

```yaml
model:
  model_config:
    kvf_query_blocks: 4
  accelerator:
    ulysses_size: 2
    cp_size: 1
    fsdp_config:
      fsdp_mode: fsdp2

data:
  train_path: /path/to/offline-wan-embeddings
  datasets_type: mapping
  anchor_batching:
    lengths_path: /path/to/wan-token-lengths.json
    long_boundary: 4096
    max_pack_tokens: 8192
    window_size: 256
    runtime_coefficients: [1.0, 0.0, 0.0]

train:
  training_task: offline_training
```

- Set `kvf_query_blocks` to a positive number of blocks per local sequence
  shard to enable KVF. The default is zero. KVF requires NVIDIA SM80+,
  NCCL, FP16 or BF16, and an Ulysses degree greater than one that divides
  the number of attention heads. The backend pads head dimensions from 1 to 256
  to FlashAttention alignment while preserving the original softmax scale. It applies to Wan bidirectional video
  self-attention. Set `model.ops_implementation.wan_kvf_implementation` to `nvshmem` for the
  hybrid backend or `nccl` for all-NCCL. The registry default is `nccl`;
  the MemDual benchmark selects `nvshmem` by default. The configured
  cross-attention backend continues to handle text and image context.
- Anchor Batching requires Wan offline embeddings, a mapping dataset,
  the native in-order loader with `drop_last: true`, FSDP2, and `cp_size: 1`.
  The JSON sidecar contains one positive integer per dataset row, in mapping
  dataset order. For an offline latent of shape `[1, C, F, H, W]` with patch
  size `[pt, ph, pw]`, use `(F // pt) * (H // ph) * (W // pw)`.
  The loader checks each scheduled row against its metadata. Changing
  the dataset or patch geometry requires regenerating the sidecar.
- `max_pack_tokens` is the total token budget per DP replica before
  sharding, with each video's length rounded up to an SP multiple.
  Calibrate it against peak memory for your training configuration.
  `long_boundary` is the sequence length at which the step for two
  equal-length clips approaches twice the runtime of one clip.
  `runtime_coefficients` estimates each sample's runtime as
  `a * length**2 + b * length + c`, and a pack's estimate is the sum.
- The accumulation count derived from `train.global_batch_size` and
  `train.micro_batch_size` is the maximum number of DP rounds per optimizer
  step. The number of videos per step varies with the plan. With shuffling
  enabled, whole rounds are shuffled each epoch. The loader drops a final
  tail of fewer than one sample per DP replica.

For offline embeddings produced by Wan's `offline_embedding` task, the
sidecar can be generated with VeOmni's mapping loader to preserve row order:

```python
import json
import math
from pathlib import Path

from veomni.data import build_data_transform, build_dataset
from veomni.data.multimodal.dit import data_transform  # Registers dit_offline.

dataset = build_dataset("mapping", train_path="/path/to/offline-wan-embeddings",
                        transform=build_data_transform("dit_offline"))
patch_size = (1, 2, 2)  # Read the actual patch size from the Wan config.
lengths = [
    math.prod(n // p for n, p in zip(dataset[i][0]["latents"].shape[-3:], patch_size))
    for i in range(len(dataset))
]
Path("wan-token-lengths.json").write_text(json.dumps(lengths))
```

### Benchmarking the MemDual ablation

Run `scripts/profile/benchmark_memdual.py` with a Wan configuration. The
default `benchmark.data_source: real` reads a map-style dataset of Wan
offline embeddings from `data.train_path` and requires
`train.training_task: offline_training`. Use `data.anchor_batching` for the
length sidecar, memory budget, and runtime calibration. Choose
`--benchmark.mode baseline`, `kvf`, `anchor`, or `combined`. Each invocation
builds a fresh model and runs one ablation. To compare against fixed-runtime
packing, provide the same `--benchmark.fixed_target_runtime` to all four
invocations. The baseline and KVF-only runs then use greedy packing under
that target and the Anchor memory budget, with an over-target sample in
its own pack. If the flag is omitted, those two modes use VeOmni's
fixed-sample loader.

Use `--benchmark.reference_attention async_ulysses` for the Async Ulysses
reference. It applies to the `baseline` and `anchor` modes, which use the
built-in DiT async QKV and output-projection ops with the same Wan model.
The default `ulysses` selects synchronous Ulysses. The `kvf` and
`combined` modes use the hybrid backend by default. Install
[NVSHMEM4Py for the CUDA runtime on the workers](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem4py-install-proc.html)
with host NBI support. Each worker owns one PE and each Ulysses group's
participating nodes must contribute the same number of ranks. For an
all-NCCL comparison, pass `--model.ops_implementation.wan_kvf_implementation nccl`
after the single-run launcher's `--` separator, or use `--kvf-backend nccl`
with the matrix launcher. For the CUDA 13.0 / NVSHMEM 3.4.5 stack, NVIDIA
pairs the runtime with `nvshmem4py-cu13==0.1.3` and CUDA Core 0.4.
Result JSON records `kvf_backend`, the requested `nvshmem_symmetric_size`,
and `kvf_symmetric_bytes_per_rank`.

Each Ulysses group follows the
[NVSHMEM4Py UID bootstrap example](https://docs.nvidia.com/nvshmem/api/latest/examples/language_bindings/python/index.html#torch-distributed-processgroup-initialization-example).
Its rank zero creates the UID and broadcasts it to the group before initialization.
Both launchers set `NVSHMEM_SYMMETRIC_SIZE` to `6G` per GPU by default.
Override it with the environment variable or `--nvshmem-symmetric-size 8G`.
The flag takes precedence. The SSH launcher sends the same value to the
peer. For manual launches, pass the same size on both nodes.
Set node-specific NVSHMEM network variables on each host.

For synthetic and metadata runs, the benchmark reserves the symmetric
payload before warmup using the longest single video in the input, including
Ulysses and head-dimension padding. Per rank the buffer size is eight times
the largest rank-local Q block in bytes, plus 64 bytes for signals. Each
independent video inside a pack uses the same slots. `--benchmark.kvf_query_blocks`
can reduce the required block size. For very long metadata inputs, set a
`--metadata-max-seconds` that fits the device budget. Leave headroom for
NVSHMEM runtime overhead and ordinary training allocations.

The launcher runs the four ablations in sequence with Async Ulysses as
the reference. With two eight-GPU nodes it sets Ulysses 8 and DP 2,
placing four GPUs from each node in each Ulysses group. Both nodes need
the same checkout, training YAML, data paths and Python environment.
From node 0, with passwordless SSH to node 1, run:

```bash
bash scripts/profile/run_memdual.sh "$CONFIG" \
    --master-addr "$MASTER_ADDR" --peer user@node1 \
    -- --benchmark.steps 20
```

Without SSH, omit `--peer` and run the command on both nodes with
`--node-rank 0` and `--node-rank 1`, respectively.
Use `--modes baseline-sync` for a separate synchronous baseline.
The launcher writes a log per node and a metrics JSON per mode under
`result/single/` by default.
Training and benchmark overrides go after `--`.
Use `--benchmark.fixed_target_runtime` there for a fixed-runtime packing
reference. Run `bash scripts/profile/run_memdual.sh --help` for the full
launcher options.

To smoke-test all four modes on four local NVIDIA GPUs with generated
latents and text embeddings and randomly initialized Wan weights, run

```bash
bash scripts/profile/run_memdual.sh configs/dit/wan_memdual_synthetic.yaml \
    --nodes 1 --gpus-per-node 4 --out-dir result/synthetic
```

The synthetic config starts with a two-layer model. To measure at the full
model size, use the corresponding Wan transformer config and remove the
size overrides under `model.model_config`. Keep `model.model_path` unset
for random initialization. For synthetic TI2V set
`model.condition_model_cfg.expand_timesteps: true`. With `data_source:
synthetic`, the runner generates rows in memory from the seeded histogram
`benchmark.synthetic.frame_counts` (entries like `"4:49"` and `"16:1"`,
in latent frames), `latent_height`, `latent_width`, and `context_tokens`.
Pack lengths come from the latent grid and Wan patch size. Set
`data.anchor_batching.long_boundary`, `max_pack_tokens`, and the fixed
target for that geometry. Keep the synthetic dataset large enough for
the requested warmup and measurement steps.

For a real-data run, use a separate offline-training YAML pointing to
cached Wan embeddings and the matching token-length JSON. Set the output
directory to a separate path such as `result/real`. Each result
records its `data_source` and, for synthetic runs, the input geometry and
frame histogram.

#### Run both models and both data sources

The matrix launcher takes two Wan Diffusers model paths and metadata from
three video datasets. It runs the following workloads for each model.

| Workload | Modes | Output directory under `result/<model>/` |
| --- | --- | --- |
| Fixed-length synthetic, per duration | `baseline`, `kvf` | `synthetic_fixed/<seconds>s/` |
| Fixed-length synthetic, larger cross-node CP | `baseline`, `kvf` | `synthetic_fixed_cp<cp>/<seconds>s/` |
| Trace-derived synthetic, per ratio | `baseline`, `anchor`, `combined` | `synthetic_trace/L<l>_M<m>_S<s>/` |
| Dataset metadata with generated latents | `baseline`, `kvf`, `anchor`, `combined` | `metadata/` |

The fixed-length pair uses one video per microbatch with async Ulysses
for the baseline. The trace-derived and metadata baselines use fixed-target
packing. Each experiment writes one `<mode>.json` and one
`<mode>.node<N>.log` per node.

```bash
python scripts/profile/run_memdual_matrix.py \
    --t2v-model /shared/models/Wan2.2-T2V-A14B-Diffusers \
    --ti2v-model /shared/models/Wan2.2-TI2V-5B-Diffusers \
    --datasets /shared/meta/panda.csv /shared/meta/webvid.csv /shared/meta/mira.parquet \
    --master-addr "$MASTER_ADDR" --peer user@node1
```

The main matrix defaults to two nodes with eight GPUs per node, DP 2 and
Ulysses 8. Each eight-rank Ulysses group contains four ranks from each node.
This layout exposes cross-node all-to-all at CP 8. The additional fixed-length
KVF pairs test larger CP groups that exceed the eight ranks available on one
node. T2V uses CP 10 (five ranks per node), and TI2V uses CP 12 (six ranks per
node). Each additional run uses DP 1, one video per optimizer step, and FSDP2
within the Ulysses group. Its baseline and KVF run share the same topology and
inputs. The full default matrix contains 88 runs, including the 16 additional
fixed-length runs. Use `--skip-extra-kvf-cp` to run the main matrix only.
Additional pairs are enabled on two-node launches when `--gpus-per-node`
covers their rank count.

Each run uses five warmup steps and 20 measured optimizer steps. The KVF-only mode also measures exposed waits in a separate
20-step pass. Adjust the window with `--warmup-steps` and `--steps`.
For a short run, use `--warmup-steps 1 --steps 3`. Use `--dry-run`
to validate inputs and write the workload configs without launching workers.
If launching manually, omit `--peer` and run the same command on both
nodes with their respective `--node-rank`. With `--peer`, the checkout
must be at the same path and the output directory shared with node 1.

Supply each dataset as a local CSV, CSV.GZ, or Parquet metadata file, or
a directory of metadata shards. Panda-70M's `timestamp` can contain several
start/end pairs per row, and each pair contributes one clip. WebVid's
`duration` accepts numeric seconds and ISO 8601 time durations such as
`PT1M2.5S`. MiraData uses `seconds`. The launcher reads these fields and
creates Wan latent shapes from each sampled clip length at 25 FPS.
The benchmark generates random latent posteriors and text embeddings.

The mixture uses `--dataset-ratios A:B:C` in `--datasets` order,
with `1:1:1` as the default. It samples without replacement up to
`--metadata-samples` (5,000 clips by default), preserving the ratio.
Both models use the same sampled video lengths, and all four modes use
the same ordered latent-frame manifest for their model. The packers
can change the order and number of samples processed in the measured
window. Result JSON records the mix, excluded clip counts, and latent-frame
histogram. The default keeps the duration range supported by both models'
temporal patch and RoPE limits. Set `--metadata-max-seconds 35`, for example,
to exclude clips above a duration that fits the test setup. End-to-end
step time includes latent generation by the data loader.

Each model path may be the whole pipeline or the transformer subdirectory.
A pipeline root selects its `transformer/` component. Pass
`transformer_2/` explicitly to benchmark that component on a checkpoint
with two transformers. By default all workloads use the same transformer
weights. Use `--synthetic-random-init` to skip checkpoint loading for the
fixed-length and trace-derived runs.

- Fixed-length workloads use the four durations in MemDual Table 1
  (14, 21, 28, and 35 seconds), each at 25 FPS. Use `--kvf-seconds 14 21`
  to run a subset.
- Trace-derived workloads use the duration boundaries from the paper's
  three-dataset analysis. Short is below 5 seconds, medium is 5 through
  20 seconds, and long is above 20 seconds. The launcher samples uniformly
  over integer video frame counts in each bin at 25 FPS with fixed seeds.
  It uses the observed 2-second minimum and a configurable 35-second
  upper cap (`--trace-max-seconds`). The proportions are experimental
  settings. By default the matrix runs all eight long:medium:short ratios
  from Figure 12. For example, use `--trace-ratios 1:3:6 1:0:49` to run
  the two short-heavy mixes.
- Synthetic workloads use 500 samples by default. If you change
  `--synthetic-samples`, choose a count divisible by each selected reduced
  ratio's sum. For all three workloads, the launcher converts video lengths
  using the model VAE's temporal factor and rounds 1080×1920 down to the VAE and transformer
  patch grid. Use `--synthetic-latent-height`, `--synthetic-latent-width`,
  and `--synthetic-context-tokens` to change the generated geometry.
  Result JSON saves each workload's realized latent-frame histogram.

By default the pack budget is the longest sequence rounded up for SP,
the long boundary is half the longest sequence, and the fixed target
for trace-derived and metadata workloads is the longest sequence length squared.
The launcher accepts `--max-pack-tokens`, `--long-boundary`, and
`--fixed-target-runtime` overrides. Compare samples per second alongside
step time when the packing strategies process different numbers of videos
per step.

- E2E time is the maximum rank wall time over a complete measurement window
  of `DiTTrainer.train_step` calls. It includes data loading, the SP broadcast,
  forward, backward, optimizer, and scheduler. Warmup, checkpointing,
  evaluation, and logging callbacks are outside that window.
- KVF mode also runs a separate instrumented window. CUDA events on the
  consuming stream bracket KVF NCCL work waits and NVSHMEM receive-event waits.
  The script reports
  the mean and maximum across ranks of the summed waits per step. This
  estimates exposed stream wait and excludes time already overlapped before
  the wait, KVF tensor repacking, and FSDP/DP collectives. The rank sums
  are a diagnostic for individual streams. Switch this pass off with
  `--benchmark.measure_exposed_wait false`, or enable it in combined mode.
- Rank zero saves JSON with topology, measured steps, sample count, wall time,
  and samples per second. Compare samples per second when the modes pack
  different numbers of videos per step. Set `--benchmark.steps 0` to time a
  full epoch, and check the sample counts for any tail dropped by either
  packing policy. Benchmark flags select the window independently of
  `train.max_steps`.
