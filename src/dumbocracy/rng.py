"""Generator-driven samplers missing from torch (gamma, Dirichlet, Beta) — spec §9.

``torch._standard_gamma`` has no ``generator`` argument, so it cannot honour the named seed streams.
These implementations use only ``torch.rand``/``torch.randn`` with an explicit generator, so every
draw is reproducible per (seed stream, device) and runs on CPU or CUDA.
"""
from __future__ import annotations

import torch


def gamma(concentration: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Gamma(concentration, 1) samples (Marsaglia & Tsang 2000), elementwise, any shape."""
    if torch.any(concentration <= 0):
        raise ValueError("gamma concentration must be > 0")
    dtype, device = concentration.dtype, concentration.device
    flat = concentration.reshape(-1)
    boost = flat < 1
    shape = torch.where(boost, flat + 1, flat)
    d = shape - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    out = torch.empty_like(flat)
    pending = torch.arange(flat.numel(), device=device)
    while pending.numel():
        x = torch.randn(pending.numel(), generator=generator, device=device, dtype=dtype)
        u = torch.rand(pending.numel(), generator=generator, device=device, dtype=dtype)
        dp, cp = d[pending], c[pending]
        v = (1 + cp * x) ** 3
        safe_v = torch.where(v > 0, v, torch.ones_like(v))
        accept = (v > 0) & (torch.log(u) < 0.5 * x * x + dp - dp * safe_v + dp * torch.log(safe_v))
        out[pending[accept]] = (dp * v)[accept]
        pending = pending[~accept]
    if torch.any(boost):
        u = torch.rand(int(boost.sum()), generator=generator, device=device, dtype=dtype)
        out[boost] = out[boost] * u ** (1.0 / flat[boost])
    return out.reshape(concentration.shape)


def dirichlet(concentration: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Dirichlet samples along the last dimension."""
    g = gamma(concentration, generator)
    return g / g.sum(dim=-1, keepdim=True)


def beta(a: torch.Tensor, b: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Beta(a, b) samples via the gamma ratio."""
    ga = gamma(a, generator)
    gb = gamma(b, generator)
    return ga / (ga + gb)
