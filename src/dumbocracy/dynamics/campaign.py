"""Campaign mechanisms: party workers, word-of-mouth SIR cascade, media, costs (spec §3).

Every mechanism maps one player's allocation for one round to per-voter *exposure intensities*
``e ∈ [0, 1]`` (tensors of shape ``[N]``, single world). Intensities combine with the product rule
``c_i = 1 − Π_e (1 − e)`` and zealots always get ``c_i = 0`` (spec §3, §2.4).

All randomness comes from an explicit ``torch.Generator`` whose device must equal the tensors'
device (spec §9: named streams ``workers``, ``cascade``, ``media``). Every function is vectorised
over voters/edges; the only Python loops run over the ≤ 4 regions and the ≤ L cascade hops. Each
random draw has a size that depends only on the inputs' shapes and the allocation — never on the
outcome of earlier draws — so a fixed (generator seed, device) gives an identical result.

Money is ``decimal.Decimal`` rounded half-up to 0.01 (spec §1).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext

import torch

from ..config import MediaConfig, WorkerConfig
from ..electorate.electorate import Electorate
from ..electorate.sbm import Graph
from ..sectors import NUM_REGIONS

CENT = Decimal("0.01")

__all__ = [
    "Allocation", "CampaignOutcome", "CascadeResult", "campaign_round", "combine_intensities",
    "media_cost", "media_exposure", "media_intensity", "sir_cascade", "word_of_mouth_intensity",
    "worker_contacts", "worker_cost", "worker_intensity",
]


# --------------------------------------------------------------------------------------------- helpers

def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_negative_ints(values: Sequence[int], name: str, length: int = NUM_REGIONS) -> list[int]:
    values = list(values)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}")
    if not all(_is_int(v) and v >= 0 for v in values):
        raise ValueError(f"{name} must contain non-negative integers")
    return values


def _flags(values: Sequence[bool], name: str, length: int = NUM_REGIONS) -> list[bool]:
    values = list(values)
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}")
    if not all(isinstance(v, bool) for v in values):
        raise ValueError(f"{name} must contain booleans")
    return values


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _check_generator(generator: torch.Generator, device: torch.device) -> None:
    gen = generator.device
    if gen.type != device.type or (gen.index is not None and device.index is not None and gen.index != device.index):
        raise ValueError(f"generator device {gen} does not match tensor device {device}")


# --------------------------------------------------------------------------------- party workers §3.1

def worker_contacts(region_of: torch.Tensor, workers_by_region: Sequence[int], contacts_per_worker: int,
                    generator: torch.Generator) -> torch.Tensor:
    """Number of direct worker contacts per voter, int64 ``[N]`` (spec §3.1).

    Each of the ``workers_by_region[r]`` workers in region r contacts ``contacts_per_worker`` voters
    drawn uniformly *with replacement* from region r, so region r receives exactly
    ``workers_by_region[r] · contacts_per_worker`` contacts. Region membership is read from
    ``region_of`` (voters need not be contiguous, although they are in a generated electorate).
    """
    workers = _non_negative_ints(workers_by_region, "workers_by_region")
    if not _is_int(contacts_per_worker) or contacts_per_worker < 0:
        raise ValueError("contacts_per_worker must be a non-negative integer")
    device = region_of.device
    _check_generator(generator, device)
    num_voters = region_of.numel()
    counts = torch.zeros(num_voters, dtype=torch.int64, device=device)
    if num_voters == 0 or contacts_per_worker == 0 or sum(workers) == 0:
        return counts
    sizes = torch.bincount(region_of, minlength=NUM_REGIONS).tolist()
    members = torch.argsort(region_of, stable=True)  # voters grouped by region
    offset = 0
    picks = []
    for r in range(NUM_REGIONS):
        draws = workers[r] * contacts_per_worker
        if draws and sizes[r] == 0:
            raise ValueError(f"workers assigned to empty region {r}")
        if draws:
            local = torch.randint(0, sizes[r], (draws,), generator=generator, device=device, dtype=torch.int64)
            picks.append(members[offset + local])
        offset += sizes[r]
    return torch.bincount(torch.cat(picks), minlength=num_voters)


def worker_intensity(contact_counts: torch.Tensor, campaign_susceptibility: torch.Tensor,
                     strength: float) -> torch.Tensor:
    """``1 − (1 − η_w·γ_i)^{count_i}``: each contact gives ``η_w·γ_i``, compounding (spec §3.1)."""
    strength = _probability(strength, "strength")
    per_contact = (strength * campaign_susceptibility).clamp(0.0, 1.0)
    exponent = contact_counts.to(per_contact.dtype)
    return 1.0 - torch.pow(1.0 - per_contact, exponent)


@dataclass(frozen=True, eq=False)
class CascadeResult:
    """Outcome of one word-of-mouth cascade."""

    reached: torch.Tensor   # bool [N]: newly reached secondary voters (seeds excluded)
    per_hop_new: list[int]  # number of voters newly reached at hop h = 1..L (length L)


def sir_cascade(graph: Graph, seeds: torch.Tensor, can_spread: torch.Tensor, transmission: float,
                recovery: float, hops: int, generator: torch.Generator) -> CascadeResult:
    """Hop-limited SIR word-of-mouth cascade on the unweighted social graph (spec §3.1).

    * Infectious set ``I_0 = seeds & can_spread`` (zealots never spread).
    * For hop h = 1..L: every node infectious at the start of the hop transmits to each neighbour
      independently w.p. ``transmission`` (w_ij = 1). A hit neighbour j is *reached* iff it is not a
      seed, was not reached earlier in this call and ``can_spread[j]``; non-spreading voters (zealots)
      are never marked reached and never relay.
    * Recovery (resolution of the spec's "after each hop each infectious node recovers w.p. ρ"):
      nodes infectious at the start of hop h recover w.p. ``recovery`` *after* transmitting; nodes
      reached at hop h are infectious for hop h+1 and face their first recovery draw after it. So a
      node transmits at least once and on average ``1/ρ`` times; with ρ = 0 the cascade is a
      transmission-limited BFS of depth L. Recovered nodes and seeds are never re-reached.

    RNG consumption per hop is fixed (one uniform per directed CSR entry, one per voter), independent
    of the cascade's progress.
    """
    if not _is_int(hops) or hops < 0:
        raise ValueError("hops must be a non-negative integer")
    transmission = _probability(transmission, "transmission")
    recovery = _probability(recovery, "recovery")
    device = graph.device
    _check_generator(generator, device)
    n = graph.num_nodes
    if seeds.shape != (n,) or can_spread.shape != (n,):
        raise ValueError("seeds and can_spread must have shape [num_nodes]")
    seeds = seeds.to(device=device, dtype=torch.bool)
    can_spread = can_spread.to(device=device, dtype=torch.bool)

    eligible = can_spread & ~seeds          # may still become reached
    infectious = seeds & can_spread
    reached = torch.zeros(n, dtype=torch.bool, device=device)
    rows = graph.rows()
    per_hop_new: list[int] = []
    for _ in range(hops):
        fire = infectious[rows] & (torch.rand(rows.numel(), generator=generator, device=device) < transmission)
        hit = torch.zeros(n, dtype=torch.bool, device=device)
        hit[graph.indices[fire]] = True
        new = hit & eligible
        eligible &= ~new
        reached |= new
        survive = torch.rand(n, generator=generator, device=device) >= recovery
        infectious = (infectious & survive) | new
        per_hop_new.append(int(new.sum()))
    return CascadeResult(reached, per_hop_new)


def word_of_mouth_intensity(reached: torch.Tensor, campaign_susceptibility: torch.Tensor,
                            strength: float) -> torch.Tensor:
    """``η_wom·γ_j`` on cascade-reached voters, 0 elsewhere (spec §3.1)."""
    strength = _probability(strength, "strength")
    value = (strength * campaign_susceptibility).clamp(0.0, 1.0)
    return torch.where(reached.to(torch.bool), value, torch.zeros_like(value))


# -------------------------------------------------------------------------------------- media §3.2

def media_exposure(region_of: torch.Tensor, aired: Sequence[bool], exposure_prob: float,
                   generator: torch.Generator) -> torch.Tensor:
    """bool ``[N]``: voter in an aired region and exposed w.p. ``p_m`` (spec §3.2).

    Draws one uniform per voter (fixed consumption). Zealots can be *exposed*; they simply get
    zero intensity because β = 0 and they are frozen in :func:`combine_intensities`.
    """
    flags = _flags(aired, "aired")
    exposure_prob = _probability(exposure_prob, "exposure_prob")
    device = region_of.device
    _check_generator(generator, device)
    aired_mask = torch.tensor(flags, dtype=torch.bool, device=device)[region_of]
    return aired_mask & (torch.rand(region_of.numel(), generator=generator, device=device) < exposure_prob)


def _media_strength(region_of: torch.Tensor, prior_airings: Sequence[int], strength: float, decay: float,
                    dtype: torch.dtype) -> torch.Tensor:
    """Per-voter ``η_m(n_r) = η_m0 / (1 + λ n_r)`` with n_r = prior airings in the voter's region."""
    airings = _non_negative_ints(prior_airings, "prior_airings")
    strength = _probability(strength, "strength")
    if not float(decay) >= 0.0:
        raise ValueError("decay must be >= 0")
    eta = [strength / (1.0 + float(decay) * n) for n in airings]
    return torch.tensor(eta, dtype=dtype, device=region_of.device)[region_of]


def media_intensity(region_of: torch.Tensor, aired: Sequence[bool], prior_airings: Sequence[int],
                    media_susceptibility: torch.Tensor, exposure_prob: float, strength: float, decay: float,
                    generator: torch.Generator) -> torch.Tensor:
    """``η_m(n)·β_i`` for exposed voters in aired regions, 0 elsewhere (spec §3.2)."""
    eta = _media_strength(region_of, prior_airings, strength, decay, media_susceptibility.dtype)
    exposed = media_exposure(region_of, aired, exposure_prob, generator)
    return _exposed_intensity(exposed, eta, media_susceptibility)


def _exposed_intensity(exposed: torch.Tensor, eta: torch.Tensor, media_susceptibility: torch.Tensor) -> torch.Tensor:
    value = (eta * media_susceptibility).clamp(0.0, 1.0)
    return torch.where(exposed, value, torch.zeros_like(value))


# --------------------------------------------------------------------------------- combination §3

def combine_intensities(*intensities: torch.Tensor, frozen: torch.Tensor | None = None) -> torch.Tensor:
    """``c_i = 1 − Π_e (1 − e_i)`` clamped to [0, 1]; ``frozen`` (zealot) voters get 0 (spec §3)."""
    if not intensities:
        raise ValueError("combine_intensities needs at least one intensity tensor")
    remaining = torch.ones_like(intensities[0])
    for e in intensities:
        remaining = remaining * (1.0 - e.clamp(0.0, 1.0))
    combined = (1.0 - remaining).clamp(0.0, 1.0)
    if frozen is not None:
        combined = torch.where(frozen.to(torch.bool), torch.zeros_like(combined), combined)
    return combined


# --------------------------------------------------------------------------------------- costs

def media_cost(sizes: Sequence[int] | torch.Tensor, aired: Sequence[bool], base_cost: Decimal) -> Decimal:
    """Σ over aired regions of ``m0 · n_r / (N/4)``, each rounded half-up to 0.01 (spec §3.2)."""
    counts = [int(x) for x in (sizes.tolist() if isinstance(sizes, torch.Tensor) else sizes)]
    counts = _non_negative_ints(counts, "sizes")
    flags = _flags(aired, "aired")
    if not isinstance(base_cost, Decimal) or not base_cost.is_finite() or base_cost < 0:
        raise ValueError("base_cost must be a finite non-negative Decimal")
    total = sum(counts)
    if total <= 0:
        raise ValueError("sizes must sum to a positive number of voters")
    cost = Decimal("0.00")
    with localcontext() as ctx:
        ctx.prec = 60
        for n_r, on in zip(counts, flags):
            if on:
                cost += (base_cost * NUM_REGIONS * n_r / total).quantize(CENT, rounding=ROUND_HALF_UP)
    return cost


def worker_cost(extra_workers: int, unit_cost: Decimal) -> Decimal:
    """``extra_workers · cost_w`` rounded half-up to 0.01 (spec §3.1)."""
    if not _is_int(extra_workers) or extra_workers < 0:
        raise ValueError("extra_workers must be a non-negative integer")
    if not isinstance(unit_cost, Decimal) or not unit_cost.is_finite() or unit_cost < 0:
        raise ValueError("unit_cost must be a finite non-negative Decimal")
    return (unit_cost * extra_workers).quantize(CENT, rounding=ROUND_HALF_UP)


# --------------------------------------------------------------------------- one player's round

@dataclass(frozen=True)
class Allocation:
    """One player's campaign allocation for one round (spec §3, §7).

    ``workers_by_region`` counts free + extra workers assigned per region; unassigned workers are
    allowed and simply wasted.
    """

    workers_by_region: tuple[int, int, int, int]
    extra_workers: int = 0
    media_regions: tuple[bool, bool, bool, bool] = (False, False, False, False)

    def validate(self, free_workers: int) -> None:
        """Raise ``ValueError`` for wrong lengths, negative counts or over-assignment."""
        if not _is_int(free_workers) or free_workers < 0:
            raise ValueError("free_workers must be a non-negative integer")
        workers = _non_negative_ints(self.workers_by_region, "workers_by_region")
        if not _is_int(self.extra_workers) or self.extra_workers < 0:
            raise ValueError("extra_workers must be a non-negative integer")
        _flags(self.media_regions, "media_regions")
        if sum(workers) > free_workers + self.extra_workers:
            raise ValueError(f"assigned {sum(workers)} workers but only {free_workers} free + "
                             f"{self.extra_workers} extra are available")


@dataclass(frozen=True, eq=False)
class CampaignOutcome:
    """Result of :func:`campaign_round` for one player.

    ``direct_contacts`` counts distinct voters with ≥ 1 worker contact; ``media_exposed`` counts
    exposed voters (zealots included, though their intensity is 0).
    """

    intensity: torch.Tensor  # c_i^p [N] in [0, 1], zealots 0
    direct_contacts: int
    word_of_mouth_reached: int
    media_exposed: int
    cost: Decimal


def campaign_round(electorate: Electorate, allocation: Allocation, prior_airings: Sequence[int],
                   worker_cfg: WorkerConfig, media_cfg: MediaConfig,
                   generators: dict[str, torch.Generator]) -> CampaignOutcome:
    """Compose workers → SIR cascade → media into one player's intensity ``c^p`` (spec §3).

    ``generators`` must provide ``"workers"``, ``"cascade"`` and ``"media"`` on the electorate's
    device. Cascade seeds are the directly contacted voters; zealots cannot spread and are frozen.
    """
    allocation.validate(worker_cfg.free_workers)
    missing = {"workers", "cascade", "media"} - set(generators)
    if missing:
        raise ValueError(f"missing generators: {sorted(missing)}")
    sus = electorate.susceptibility
    region_of = electorate.region_of
    zealot = sus.zealot.to(torch.bool)

    counts = worker_contacts(region_of, allocation.workers_by_region,
                             worker_cfg.contacts_per_worker(electorate.num_voters), generators["workers"])
    contacted = counts > 0
    cascade = sir_cascade(electorate.graph, contacted, ~zealot, worker_cfg.transmission, worker_cfg.recovery,
                          worker_cfg.hops, generators["cascade"])
    eta_media = _media_strength(region_of, prior_airings, media_cfg.strength, media_cfg.decay, sus.media.dtype)
    exposed = media_exposure(region_of, allocation.media_regions, media_cfg.exposure_prob, generators["media"])

    intensity = combine_intensities(
        worker_intensity(counts, sus.campaign, worker_cfg.contact_strength),
        word_of_mouth_intensity(cascade.reached, sus.campaign, worker_cfg.word_of_mouth_strength),
        _exposed_intensity(exposed, eta_media, sus.media),
        frozen=zealot,
    )
    cost = worker_cost(allocation.extra_workers, worker_cfg.extra_worker_cost) + \
        media_cost(electorate.sizes, allocation.media_regions, media_cfg.base_cost)
    return CampaignOutcome(intensity, int(contacted.sum()), int(cascade.reached.sum()), int(exposed.sum()), cost)
