"""Public libraries: sectors, regions, voter archetypes and regional personalities (spec §1, §2.3, §2.4).

Everything in this module is common knowledge to every player. Which archetype and which
personality each region actually has is hidden state, sampled in ``electorate``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SECTORS: tuple[str, ...] = ("Edu", "Welfare", "Agri", "Industry", "Env")
SECTOR_LABELS: tuple[str, ...] = (
    "Education",
    "Freebies / Social Welfare",
    "Agriculture & Rural Development",
    "Industry & Trade",
    "Environment / Climate Action",
)
REGIONS: tuple[str, ...] = ("N", "S", "E", "W")
REGION_LABELS: tuple[str, ...] = ("North", "South", "East", "West")
NUM_SECTORS = len(SECTORS)
NUM_REGIONS = len(REGIONS)


@dataclass(frozen=True, slots=True)
class Archetype:
    id: str
    name: str
    mean: tuple[float, float, float, float, float]  # order = SECTORS


@dataclass(frozen=True, slots=True)
class Personality:
    id: str
    name: str
    alpha: tuple[float, float]  # Beta(a, b) for social susceptibility
    beta: tuple[float, float]   # Beta(a, b) for media susceptibility
    gamma: tuple[float, float]  # Beta(a, b) for campaign (worker) susceptibility


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype("A0", "Agrarian heartland", (0.15, 0.20, 0.40, 0.10, 0.15)),
    Archetype("A1", "Industrial belt", (0.15, 0.15, 0.10, 0.45, 0.15)),
    Archetype("A2", "Green urban", (0.25, 0.10, 0.10, 0.15, 0.40)),
    Archetype("A3", "Welfare-reliant", (0.15, 0.45, 0.15, 0.10, 0.15)),
    Archetype("A4", "Knowledge hub", (0.40, 0.10, 0.10, 0.25, 0.15)),
    Archetype("A5", "Balanced middle", (0.20, 0.20, 0.20, 0.20, 0.20)),
)

PERSONALITIES: tuple[Personality, ...] = (
    Personality("P0", "Socially driven", (6.0, 3.0), (2.0, 5.0), (3.0, 4.0)),
    Personality("P1", "Media-savvy", (3.0, 4.0), (6.0, 3.0), (2.0, 5.0)),
    Personality("P2", "Independent", (2.0, 6.0), (2.0, 6.0), (2.0, 6.0)),
    Personality("P3", "Receptive to canvassing", (3.0, 4.0), (3.0, 4.0), (6.0, 3.0)),
)


def sector_index(name: str) -> int:
    """Index of a sector by short name (``"Agri"``) or full label."""
    if name in SECTORS:
        return SECTORS.index(name)
    if name in SECTOR_LABELS:
        return SECTOR_LABELS.index(name)
    raise KeyError(f"unknown sector {name!r}")


def region_index(name: str) -> int:
    """Index of a region by short name (``"N"``) or full label (``"North"``)."""
    if name in REGIONS:
        return REGIONS.index(name)
    if name in REGION_LABELS:
        return REGION_LABELS.index(name)
    raise KeyError(f"unknown region {name!r}")


def archetype_means(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Tensor ``[len(ARCHETYPES), NUM_SECTORS]`` of archetype mean vectors."""
    return torch.tensor([a.mean for a in ARCHETYPES], device=device, dtype=dtype)


def personality_beta_params(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Tensor ``[len(PERSONALITIES), 3, 2]``: (alpha, beta, gamma) x (a, b) Beta parameters."""
    return torch.tensor([[p.alpha, p.beta, p.gamma] for p in PERSONALITIES], device=device, dtype=dtype)
