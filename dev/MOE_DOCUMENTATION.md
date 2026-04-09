# MoE (Mixture of Experts) Documentation

## Overview

This branch adds Mixture of Experts (MoE) to nanochat, replacing the dense MLP in each
transformer block with a routed expert layer. The design follows DeepSeek-V3 conventions:
sigmoid router, top-K selection, auxiliary-loss-free load balancing, and shared experts.

## Architecture

### MoE Layer (`nanochat/moe.py`)

Each transformer block's MLP is replaced by `MoE`, which contains:

- **Router** (`TopKRouter`): sigmoid-gated, selects top-K experts per token
- **Routed experts** (`ExpertGroup`): N independent MLPs stored as 3D weight tensors
  `(num_experts, hidden_dim, model_dim)`. Uses `torch._grouped_mm` on CUDA for efficient
  batched dispatch, with a for-loop fallback on CPU/MPS.
- **Shared expert** (`SharedExpert`): a single dense MLP that processes ALL tokens (no routing).
  Always active, provides a stable gradient signal.

Architecture per token:
```
input -> Router selects top-K experts
      -> Routed experts process token (weighted by routing scores)
      -> Shared expert processes token (always)
      -> output = routed_output + shared_output
```

### Iso-FLOP Design

Expert hidden dim is sized so that per-token compute matches a dense MLP:
```
Dense MLP:  2 * dim * (4*dim) = 8*dim² FLOPs
MoE:        (top_k + num_shared) * 2 * dim * expert_hidden ≈ 8*dim² FLOPs
```
`expert_hidden = round(4 * dim / (top_k + num_shared) / 128) * 128`

This means MoE uses the same FLOPs per token as dense, but has more total parameters
(only a fraction are active per token).

### Load Balancing (DeepSeek-V3 style)

Auxiliary-loss-free: a learned bias is added to router scores for expert SELECTION
(but not for gating weights). After each step, the bias is nudged to equalize token
counts across experts: underloaded experts get bias increased, overloaded decreased.

Called via `model.update_moe_balancing()` before `optimizer.step()`.

### Key Differences from Dense nanochat

| Component | Dense (upstream) | MoE (this branch) |
|---|---|---|
| Block FFN | `MLP` (nn.Linear) | `MoE` (router + experts) |
| Expert weights | 2D `nn.Linear` | 3D `nn.Parameter` tensors |
| Matmul dispatch | Standard F.linear | `torch._grouped_mm` (CUDA) |
| Optimizer | Muon on 2D matrices | Muon on 3D tensors (expert dim = batch dim) |
| Param counting | All active | Reports both total and active params |
| FLOPs counting | All params contribute | Excludes inactive expert params |

## Features Integrated from Upstream

The `moe-new` branch merges all of karpathy's upstream/master on top of the MoE branch:

- **Smear**: mixes previous token's embedding into current token (cheap bigram info).
  Operates BEFORE transformer blocks. Orthogonal to MoE.
- **Backout**: subtracts mid-layer residual stream before final norm. Operates AFTER
  transformer blocks. Orthogonal to MoE.
- **COMPUTE_DTYPE system**: replaces autocast. `Linear` class casts fp32 weights to
  match input dtype in forward. Shared between `gpt.py` and `moe.py` via `common.py`.
- **GradScaler**: wraps optimizer.step() for fp16 training (no-op for bf16/fp32).
  MoE balancing call happens before the scaler-wrapped step.
- **ClimbMix dataset**: replaces FineWeb-EDU.
- **FlexAttention**: block-sparse sliding window for Blackwell/Ada GPUs.
- **Autoresearch hyperparameters**: tuned init scales, LR schedules, etc.

## Dtype Pipeline

| What | Storage | Compute | Notes |
|---|---|---|---|
| Master weights (all) | FP32 | — | Optimizer precision |
| Attention (c_q/c_k/c_v/c_proj) | FP32 | BF16 | `Linear` casts in forward |
| MoE router gate | FP32 | BF16 | `Linear` casts in forward |
| MoE shared expert | FP32 | BF16 | `Linear` casts in forward |
| MoE routed experts | FP32 | BF16 | Explicit `.bfloat16()` in `_run_experts_grouped_mm` |
| Embeddings (wte, value_embeds) | BF16 | BF16 | Cast at init to save memory |
| Muon momentum | FP32 | — | Accumulation stability |
| Muon polar express | FP32→BF16 | BF16 | Cast for speed |
| Activations | — | BF16 | Via COMPUTE_DTYPE |
| Gradients | — | BF16 | Matches activation dtype |

### FP8 Training (`--fp8`)

FP8 converts `nn.Linear` layers to `Float8Linear` which does matmuls in FP8
(torch._scaled_mm) while keeping fp32 master weights. The path is:
```
FP32 (storage) → FP8 (matmul, ~2x faster on H100/B300) → BF16 (output)
```

**What gets FP8 in MoE mode:**
- Attention projections (c_q, c_k, c_v, c_proj) ✓
- Shared expert (w_up, w_down) ✓
- lm_head ✓

**What does NOT get FP8:**
- Routed expert 3D weights — they use `grouped_mm` not `nn.Linear`, so the FP8
  conversion pass doesn't see them. They remain BF16 compute.
- Router gate, ve_gate, smear_gate — too small (skipped by size filter).

This means FP8 currently provides limited speedup for MoE since the routed experts
(the bulk of MoE compute) stay in BF16. Future work: FP8 grouped_mm.

## CLI Arguments

```
--num-experts N        Number of routed experts (default: 8)
--top-k K              Active routed experts per token (default: 2)
--num-shared-experts S Shared (always-active) experts (default: 1)
```

Use `--num-experts=1 --top-k=1 --num-shared-experts=0` for a dense baseline.

## Benchmarks (d12, 100 steps, seq=1024, bs=8, total_batch=8192)

### Loss Comparison

| Step | MoE BF16 | MoE FP8 | Dense BF16 | Dense FP8 |
|------|----------|---------|------------|-----------|
| 0    | 10.3977  | 10.3977 | 10.3977    | 10.3977   |
| 20   | 9.788    | 9.793   | 9.866      | 9.876     |
| 40   | 7.549    | 7.556   | 7.839      | 7.832     |
| 60   | 6.839    | 6.843   | 6.954      | 6.959     |
| 80   | 6.630    | 6.632   | 6.690      | 6.695     |
| 99   | **6.535**  | **6.537** | 6.588    | 6.595     |

Initial loss of 10.3977 = ln(32768) confirms correct uniform initialization.

### Key Observations

1. **MoE consistently beats dense** at same active FLOPs (6.535 vs 6.588 final loss)
2. **FP8 matches BF16 within noise** — loss difference is <0.01 at convergence,
   confirming FP8 precision is sufficient
3. **FP8 is SLOWER for MoE** (172K vs 230K tok/sec) because the FP8 conversion adds
   overhead to attention/shared expert, but the routed experts (bulk of MoE compute)
   don't benefit. FP8 overhead > FP8 benefit when most compute is non-FP8.
4. **FP8 is also slower for dense** on B300 (225K vs 286K tok/sec) — suggests the custom
   FP8 implementation's opaque torch.compile boundary hurts more than the FP8 matmul helps
   at this model size. May improve at larger scale where matmul dominates.

### Performance Summary (d12)

| Config | Params (total/active) | tok/sec | Peak Memory |
|---|---|---|---|
| MoE BF16 | 399.6M / 286.3M | 230K | 7.1 GB |
| MoE FP8 | 399.6M / 286.3M | 172K | 8.3 GB |
| Dense BF16 | 286.3M / 286.3M | 286K | 5.7 GB |
| Dense FP8 | 286.3M / 286.3M | 225K | 7.1 GB |

## Scaling

Tested on NVIDIA B300 SXM6 (275 GB HBM, SM 10.3).

### MoE 64x8+1 scaling (single GPU)

| Depth | Total Params | Active Params | Active/Total | FLOPs/tok | Step Time | tok/sec | Memory |
|---|---|---|---|---|---|---|---|
| d12 | 399.6M | 286.3M | 71.7% | 7.1e8 | 35ms | 230K | 7.1 GB |
| d24 | 4.00B | 1.36B | 33.9% | 4.6e9 | 233ms | 35K | 77 GB |
| d32 | 9.38B | 2.81B | 29.9% | 1.07e10 | 546ms | 15K | 187 GB |

Scaling is near-linear with FLOPs: d32/d24 = 2.32x FLOPs → 2.34x step time.
Memory scales ~2.4x per step (slightly superlinear due to activation growth).

d32 (9.4B total, 2.8B active) is the largest that fits on a single 275GB B300.
d48 (31.4B total, 8.2B active) would require multi-GPU.

### Optimizer Note

At d24 64x8, the Muon optimizer's polar express step on 64 expert weight matrices
takes ~23% of step time. This is because each expert's 3D weight tensor gets
independently orthogonalized. The optimizer cost scales linearly with num_experts.

## Nsight Systems Profile (d24 64x8+1, bs=4, seq=2048)

Profile saved at: `/mnt/data/moe_64x8_d24_profile.nsys-rep`

Per-step breakdown:
- Forward: 25ms (10.7%)
- Backward: 43ms (18.5%)
- Optimizer: 54ms (23.2%)
- Sync/overhead: 112ms (47.5%)

Top GPU kernels: Muon norm (5.9%), attention nvJet kernels (4-5% each),
FlexAttention backward (4.1%), MoE token dispatch CatBatchedCopy (3.6%),
MoE cutlass grouped GEMMs (3.4-3.5% each).

## Known Limitations

1. **FP8 doesn't cover routed experts** — the 3D `nn.Parameter` + `grouped_mm` path
   bypasses the `Float8Linear` conversion. Routed experts stay BF16.
2. **No expert parallelism** — all experts live on one GPU. For >d32, need to shard
   experts across GPUs (expert parallelism or tensor parallelism).
3. **torch.compile overhead** — first step takes 30-60s for compilation. The 252GB
   memory spike during d32 compilation is transient but approaches GPU limits.
4. **Muon on 3D tensors** — the second momentum buffer uses `*shape[:-1]` to preserve
   the expert dimension. This was a merge conflict with upstream's 2D simplification
   and must be maintained.

## Files Modified

- `nanochat/moe.py` — NEW: MoE layer (router, experts, shared expert, grouped_mm)
- `nanochat/gpt.py` — Block uses MoE instead of MLP; smear/backout/MoE param counting
- `nanochat/common.py` — `Linear` class moved here (shared with moe.py); `nn` and `F` imports
- `nanochat/optim.py` — 3D-aware second momentum buffer for MoE expert weights
- `scripts/base_train.py` — MoE CLI args, MoE balancing + GradScaler coexistence
- `scripts/chat_sft.py` — MoE balancing + GradScaler coexistence
- `nanochat/flash_attention.py` — FlexAttention backend (from flex branch)
