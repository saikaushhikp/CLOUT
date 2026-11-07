"""Region sizes (spec §2.1, R7): every region gets at least ceil(f·N) voters (f = 0.15 by default)."""
from __future__ import annotations

import math

import torch

from ..rng import dirichlet
from ..sectors import NUM_REGIONS


def minimum_region_size(num_voters: int, min_fraction: float = 0.15) -> int:
    return math.ceil(min_fraction * num_voters)


def sample_region_sizes(num_voters: int, dirichlet_a: float, generator: torch.Generator,
                        min_fraction: float = 0.15) -> torch.Tensor:
    """Return int64 sizes ``[4]`` (on the generator's device) summing to ``num_voters``."""
    if num_voters < NUM_REGIONS:
        raise ValueError("need at least one voter per region")
    device = generator.device
    floor = minimum_region_size(num_voters, min_fraction)
    if NUM_REGIONS * floor > num_voters:
        raise ValueError("num_voters too small for the minimum region fraction")
    remaining = num_voters - NUM_REGIONS * floor
    shares = dirichlet(torch.full((NUM_REGIONS,), float(dirichlet_a), dtype=torch.float64, device=device), generator)
    sizes = torch.full((NUM_REGIONS,), floor, dtype=torch.int64, device=device)
    # Multinomial(remaining, shares) via sequential conditional binomials (supports a generator).
    left, mass_left = remaining, 1.0
    for region in range(NUM_REGIONS - 1):
        share = float(shares[region])
        prob = min(1.0, max(0.0, share / mass_left)) if mass_left > 0 else 0.0
        draw = int(torch.binomial(torch.tensor([float(left)], dtype=torch.float64, device=device),
                                  torch.tensor([prob], dtype=torch.float64, device=device), generator=generator).item())
        sizes[region] += draw
        left -= draw
        mass_left -= share
    sizes[NUM_REGIONS - 1] += left
    return sizes


def region_of(sizes: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
    """Region index per voter (voters are contiguous by region)."""
    target = sizes.device if device is None else torch.device(device)
    return torch.repeat_interleave(torch.arange(len(sizes), device=target), sizes.to(target))


def region_offsets(sizes: torch.Tensor) -> list[int]:
    """Start index of each region in the contiguous voter ordering."""
    offsets = [0]
    for size in sizes.tolist()[:-1]:
        offsets.append(offsets[-1] + int(size))
    return offsets
