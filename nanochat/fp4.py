"""FP4 training for nanochat via quartet2 (CloverLM).

Drop-in replacement for nn.Linear that does FP4 compute using quartet2's
Quartet_II_linear. All three matmuls (forward + two backward) run in FP4
using NVIDIA's nvfp4 format via flashinfer kernels.

How FP4 training works (Quartet II)
====================================
Similar to FP8, but with 4-bit quantization + Hadamard rotation:
  forward:   quantize input & weight to FP4, matmul via flashinfer mm_fp4
  backward:  re-quantize with Randomized Hadamard Transform (RHT) + EDEN
             rounding for unbiased gradients, then FP4 matmul

Key differences from FP8:
  - Uses 4-bit (e2m1) format instead of 8-bit, with per-16-element micro-scales
  - Backward pass uses Hadamard rotation to decorrelate quantization errors
  - Requires dims divisible by 128 (not just 16 like FP8)
  - Uses flashinfer + custom CUDA kernels (not torch._scaled_mm)

References:
  - quartet2: https://github.com/IST-DASLab/CloverLM
  - NVFP4 format: NVIDIA Blackwell architecture
"""

import torch
import torch.nn as nn

from quartet2.linear import Quartet_II_linear, register_optimizer_hook


class Float4Linear(Quartet_II_linear):
    """Wrapper around Quartet_II_linear matching nanochat's conversion API.

    Adds from_float() classmethod for zero-copy conversion from nn.Linear,
    and removes bias handling (nanochat uses bias=False everywhere).
    """

    def forward(self, input):
        from quartet2.linear import Quartet_II_fn
        # Cast input and weight to bfloat16 (nanochat keeps master weights in fp32)
        input = input.to(torch.bfloat16)
        weight = self.weight.to(torch.bfloat16)
        # Handle 2D input (no batch dim) by adding a dummy batch dim
        squeeze = input.ndim == 2
        if squeeze:
            input = input.unsqueeze(0)
        # quartet2 requires batch*seq divisible by 128. Pad if needed.
        B, T, D = input.shape
        tokens = B * T
        pad = (128 - tokens % 128) % 128
        if pad > 0:
            input = input.reshape(1, tokens, D)
            input = torch.nn.functional.pad(input, (0, 0, 0, pad))
        output = Quartet_II_fn.apply(
            input, weight, self.had, self.mode, False,
            self.weight_abs_max, None, self.scratch_amax
        )
        if pad > 0:
            output = output.reshape(tokens + pad, -1)[:tokens].reshape(B, T, -1)
        if squeeze:
            output = output.squeeze(0)
        return output

    @classmethod
    def from_float(cls, mod):
        """Create Float4Linear from nn.Linear, sharing the same weight and bias."""
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=mod.bias is not None)
        new_mod.weight = mod.weight
        if mod.bias is not None:
            new_mod.bias = mod.bias
        # _apply will handle hadamard matrix init when moving from meta to real device
        # But if weight is already on a real device, init now
        if mod.weight.device.type != 'meta':
            new_mod.had = new_mod._make_hadamard(mod.weight.device)
            new_mod.scratch_amax = torch.empty((), dtype=torch.uint32, device=mod.weight.device)
        return new_mod

    def _make_hadamard(self, device):
        from quartet2.linear import get_hadamard_matrix
        return get_hadamard_matrix(128, torch.bfloat16, device)


def convert_to_float4_training(module, *, module_filter_fn=None):
    """Replace nn.Linear layers with Float4Linear throughout a module.

    Same API as convert_to_float8_training for consistency.
    quartet2 requires both dimensions divisible by 128.
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float4Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float4Linear.from_float(child))

    _convert(module)
    return module
