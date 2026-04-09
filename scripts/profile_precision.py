"""Profile a single training step for BF16/FP8/FP4 at d32.

Usage: CUDA_VISIBLE_DEVICES=5 python -m scripts.profile_precision

Produces:
  - PyTorch profiler traces: profile_d32_{bf16,fp8,fp4}.json
  - Nsight Systems is run externally wrapping this script with --mode flag
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import time
import torch
import torch.nn as nn

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.common import COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, choices=["bf16", "fp8", "fp4"])
args = parser.parse_args()

device = torch.device("cuda")
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

# Build d32 model
depth = 32
aspect_ratio = 64
head_dim = 128
base_dim = depth * aspect_ratio
model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
num_heads = model_dim // head_dim
config = GPTConfig(
    sequence_len=2048, vocab_size=vocab_size,
    n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
print(f"Model: d{depth}, dim={model_dim}, params={sum(p.numel() for p in model.parameters()):,}")

# Apply precision mode
if args.mode == "fp8":
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
    def fp8_filter(mod, fqn):
        if not isinstance(mod, nn.Linear): return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0: return False
        if min(mod.in_features, mod.out_features) < 128: return False
        return True
    convert_to_float8_training(model, config=Float8LinearConfig(), module_filter_fn=fp8_filter)
    num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
    print(f"FP8: converted {num_fp8} layers")

elif args.mode == "fp4":
    from nanochat.fp4 import Float4Linear, convert_to_float4_training, register_optimizer_hook
    def fp4_filter(mod, fqn):
        if not isinstance(mod, nn.Linear): return False
        if mod.in_features % 128 != 0 or mod.out_features % 128 != 0: return False
        return True
    convert_to_float4_training(model, module_filter_fn=fp4_filter)
    num_fp4 = sum(1 for m in model.modules() if isinstance(m, Float4Linear))
    print(f"FP4: converted {num_fp4} layers")

# No torch.compile for profiling — we want to see raw kernel behavior
# model = torch.compile(model, dynamic=False)

# Setup optimizer (simple AdamW for profiling)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

if args.mode == "fp4":
    from nanochat.fp4 import register_optimizer_hook
    register_optimizer_hook(model, optimizer)

# Warmup: 2 steps to warm up CUDA caches
print("Warming up...")
for i in range(2):
    x = torch.randint(0, vocab_size, (8, 2048), device=device)
    y = torch.randint(0, vocab_size, (8, 2048), device=device)
    loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(f"  warmup step {i}: loss={loss.item():.4f}")

torch.cuda.synchronize()
print("Warmup done. Profiling 1 step...")

# Profile with PyTorch profiler
x = torch.randint(0, vocab_size, (8, 2048), device=device)
y = torch.randint(0, vocab_size, (8, 2048), device=device)

trace_path = f"profile_d32_{args.mode}.json"

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    # Signal nsys to start capturing
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"train_step_{args.mode}")

    torch.cuda.nvtx.range_push("forward")
    loss = model(x, y)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("optimizer")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()  # train_step
    torch.cuda.cudart().cudaProfilerStop()

prof.export_chrome_trace(trace_path)
print(f"PyTorch trace saved to {trace_path}")

# Print key averages
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Also time it cleanly
torch.cuda.synchronize()
t0 = time.perf_counter()
torch.cuda.nvtx.range_push(f"timed_step_{args.mode}")
loss = model(x, y)
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
t1 = time.perf_counter()
print(f"\nTimed step: {(t1-t0)*1000:.2f}ms")
print(f"Throughput: {8*2048/(t1-t0):,.0f} tok/sec")
