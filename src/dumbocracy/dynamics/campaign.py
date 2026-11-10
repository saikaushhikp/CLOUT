"""Campaign mechanisms: permanent party workers + round-by-round Independent Cascade, media, costs
(spec §3; P8 rules R21–R28 in ``docs/v2/plans/P8_party_workers_ic.md``).

Party workers (P8) are real voter nodes bought in a region, pinned at the manifesto for the whole game.
Influence spreads as an Independent Cascade, ONE hop per campaign round: the spreaders of round t are the
workers bought in round t plus the voters converted in round t−1; each tries every friend once
(chance ``γ_j P_j``, at most ``attempt_cap`` attempts per voter per game) and a converted voter takes
``v = b``. Media maps an allocation to per-voter *exposure intensities* ``e ∈ [0, 1]`` that enter the
§4 preference update (zealots always get 0).

All randomness comes from an explicit ``torch.Generator`` whose device must equal the tensors'
device (spec §9: named streams ``workers``, ``cascade``, ``media``). The cascade draws one uniform per
voter per hop and media one per voter, independent of outcomes; worker picks draw one ``randperm`` of the
eligible voters per region with hires. A fixed (generator seed, device) gives an identical result.

Money is ``decimal.Decimal`` rounded half-up to 0.01 (spec §1).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, localcontext

import torch

from ..config import MediaConfig, VotingConfig, WorkerConfig
from ..electorate.electorate import Electorate
from ..electorate.sbm import Graph
from ..sectors import NUM_REGIONS
from ..voting import similarity, vote_probability

CENT = Decimal("0.01")

__all__ = [
    "Allocation", "CampaignOutcome", "WorkerState", "apply_assignments", "campaign_round", "combine_intensities",
    "conversion_chance", "eligible_counts", "eligible_workers", "ic_from_counts", "ic_hop", "media_cost", "media_exposure",
    "media_intensity", "pick_workers", "spreader_edge_counts", "worker_cost",
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


# ------------------------------------------------------------------- party workers (P8, R21–R27)

def eligible_workers(degree: torch.Tensor, zealot: torch.Tensor, blocked: torch.Tensor, min_degree: int) -> torch.Tensor:
    """bool ``[N]``: voters that may be bought as workers (R22): degree ≥ ``min_degree``, not a zealot and
    not ``blocked`` (already a worker)."""
    if not _is_int(min_degree) or min_degree < 0:
        raise ValueError("min_degree must be a non-negative integer")
    return (degree >= min_degree) & ~zealot.to(torch.bool) & ~blocked.to(torch.bool)


def eligible_counts(region_of: torch.Tensor, eligible: torch.Tensor) -> list[int]:
    """Number of eligible voters per region ``[N, S, E, W]``."""
    return [int(x) for x in torch.bincount(region_of[eligible.to(torch.bool)], minlength=NUM_REGIONS).tolist()]


def pick_workers(region_of: torch.Tensor, eligible: torch.Tensor, hires_by_region: Sequence[int],
                 generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    """Pick ``hires_by_region[r]`` distinct eligible voters of region r uniformly at random (R22).

    For every region with hires > 0 (in order N, S, E, W) one ``randperm`` of that region's eligible voters
    (ascending ids) is drawn from ``generator`` (the ``workers`` stream) and its first k entries are taken.
    Returns one int64 id tensor per region. Raises ``ValueError`` if a region has fewer eligible voters
    than requested (the engine turns this into ``INVALID_ALLOCATION``).
    """
    hires = _non_negative_ints(hires_by_region, "hires_by_region")
    device = region_of.device
    _check_generator(generator, device)
    eligible = eligible.to(torch.bool)
    out = []
    for r in range(NUM_REGIONS):
        if hires[r] == 0:
            out.append(torch.zeros(0, dtype=torch.int64, device=device))
            continue
        candidates = torch.nonzero(eligible & (region_of == r)).flatten()
        if candidates.numel() < hires[r]:
            raise ValueError(f"region {r} has only {candidates.numel()} eligible voters for {hires[r]} new workers")
        perm = torch.randperm(candidates.numel(), generator=generator, device=device)[: hires[r]]
        out.append(candidates[perm])
    return tuple(out)


def spreader_edge_counts(graph: Graph, spreaders: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
    """``count_j`` = number of directed edges spreader → j, int64 with the shape of ``spreaders`` ``[..., N]``."""
    n = graph.num_nodes
    if spreaders.shape[-1] != n:
        raise ValueError("spreaders must have shape [..., num_nodes]")
    rows = graph.rows() if rows is None else rows
    spreaders = spreaders.to(torch.bool)
    if spreaders.ndim == 1:
        return torch.bincount(graph.indices[spreaders[rows]], minlength=n)
    flat = spreaders.reshape(-1, n)
    which, entry = torch.nonzero(flat[:, rows], as_tuple=True)
    counts = torch.bincount(which * n + graph.indices[entry], minlength=flat.shape[0] * n)
    return counts.view(spreaders.shape)


def conversion_chance(campaign_susceptibility: torch.Tensor, v: torch.Tensor, manifesto: torch.Tensor,
                      sensitivity: float, threshold: float) -> torch.Tensor:
    """R25: per-attempt conversion chance ``γ_j · P_j`` with ``P_j = σ(β_v (S_j − z))`` on ``v`` ``[..., N, 5]``."""
    prob = vote_probability(similarity(v, manifesto), sensitivity, threshold)
    return (campaign_susceptibility * prob).clamp(0.0, 1.0)


def ic_hop(graph: Graph, spreaders: torch.Tensor, chance: torch.Tensor, attempts: torch.Tensor,
           blocked: torch.Tensor, attempt_cap: int, generator: torch.Generator,
           rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One Independent-Cascade hop (R24, R25, R27). Supports leading batch dims ``[..., N]``.

    * ``count_j`` = directed edges spreader → j; ``eff_j = min(count_j, cap − attempts_j)`` (0 for ``blocked``
      voters: workers and zealots are never targets and their attempts are not counted);
    * ``attempts_j += eff_j`` (failures included);
    * j converts iff ``u_j < 1 − (1 − chance_j)^{eff_j}``, with ONE uniform per voter (``[N]``, shared over
      any batch dims = common random numbers) — RNG consumption is independent of the outcome.

    Returns ``(converted bool [..., N], attempts int64 [..., N])``.
    """
    _check_generator(generator, graph.device)
    counts = spreader_edge_counts(graph, spreaders, rows)
    return ic_from_counts(counts, chance, attempts, blocked, attempt_cap, generator)


def ic_from_counts(counts: torch.Tensor, chance: torch.Tensor, attempts: torch.Tensor, blocked: torch.Tensor,
                   attempt_cap: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """The conversion part of :func:`ic_hop` given ``count_j`` (spreader → j edge counts) ``[..., N]``."""
    if not _is_int(attempt_cap) or attempt_cap < 0:
        raise ValueError("attempt_cap must be a non-negative integer")
    device = counts.device
    _check_generator(generator, device)
    n = counts.shape[-1]
    counts = counts.to(torch.int64)
    remaining = (attempt_cap - attempts).clamp_min(0)
    eff = torch.where(blocked.to(torch.bool), torch.zeros_like(counts), torch.minimum(counts, remaining))
    u = torch.rand(n, generator=generator, device=device, dtype=chance.dtype)
    p = 1.0 - torch.pow(1.0 - chance.clamp(0.0, 1.0), eff.to(chance.dtype))
    converted = (u < p) & (eff > 0)
    return converted, attempts + eff


@dataclass(frozen=True, eq=False)
class WorkerState:
    """One player's hidden party-worker state (P8). Never serialised to a player.

    ``pinned`` marks the player's permanent workers (held at b, R23); ``worker_ids`` lists them per region
    (purchase order); ``spreaders`` are the voters converted last round, who spread in the next hop (R24);
    ``attempts`` counts conversion attempts received per voter (R27).
    """

    pinned: torch.Tensor                      # bool [N]
    worker_ids: tuple[torch.Tensor, ...]      # 4 x int64 [k_r]
    spreaders: torch.Tensor                   # bool [N]
    attempts: torch.Tensor                    # int64 [N]

    @classmethod
    def empty(cls, num_voters: int, device: torch.device | str = "cpu") -> "WorkerState":
        z = torch.zeros(num_voters, dtype=torch.bool, device=device)
        ids = tuple(torch.zeros(0, dtype=torch.int64, device=device) for _ in range(NUM_REGIONS))
        return cls(z, ids, z.clone(), torch.zeros(num_voters, dtype=torch.int64, device=device))

    @property
    def team_by_region(self) -> tuple[int, int, int, int]:
        return tuple(int(x.numel()) for x in self.worker_ids)  # type: ignore[return-value]

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


def worker_cost(count: int, unit_cost: Decimal) -> Decimal:
    """``count · cost_w`` (one-time fee per bought worker, R21) rounded half-up to 0.01."""
    if not _is_int(count) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if not isinstance(unit_cost, Decimal) or not unit_cost.is_finite() or unit_cost < 0:
        raise ValueError("unit_cost must be a finite non-negative Decimal")
    return (unit_cost * count).quantize(CENT, rounding=ROUND_HALF_UP)


# --------------------------------------------------------------------------- one player's round

@dataclass(frozen=True)
class Allocation:
    """One player's campaign allocation for one round (P8: R21–R22, R28).

    ``hires_by_region`` = NEW permanent workers bought this round per region [N, S, E, W] (paid once);
    ``media_regions`` = regions aired this round.
    """

    hires_by_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    media_regions: tuple[bool, bool, bool, bool] = (False, False, False, False)

    @property
    def total_hires(self) -> int:
        return int(sum(self.hires_by_region))

    def validate(self) -> None:
        """Raise ``ValueError`` for wrong lengths, negative/non-integer hires or non-boolean media flags."""
        _non_negative_ints(self.hires_by_region, "hires_by_region")
        _flags(self.media_regions, "media_regions")


@dataclass(frozen=True, eq=False)
class CampaignOutcome:
    """Result of :func:`campaign_round` for one player (steps 1–3 of the P8 round order).

    ``assigned`` = this round's new workers and converts (both set to ``v = b`` before the social step);
    ``intensity`` = media intensity ``c^p`` [N] (zealots 0); ``state`` = the player's worker state after
    the round (``state.spreaders`` = this round's converts, who spread next round).
    """

    assigned: torch.Tensor   # bool [N]
    intensity: torch.Tensor  # media c_i^p [N] in [0, 1], zealots 0
    hires: int
    spreaders: int
    converted: int
    media_exposed: int
    cost: Decimal
    state: WorkerState


def campaign_round(electorate: Electorate, allocation: Allocation, state: WorkerState, preferences: torch.Tensor,
                   manifesto: torch.Tensor, prior_airings: Sequence[int], worker_cfg: WorkerConfig,
                   media_cfg: MediaConfig, voting_cfg: VotingConfig, generators: dict[str, torch.Generator], *,
                   exclude: torch.Tensor | None = None) -> CampaignOutcome:
    """Steps 1–3 of the P8 round order for one player: hires → one cascade hop → media intensity.

    1. Hires (R22): pick the new workers from the ``workers`` stream among region voters with degree ≥
       ``min_degree``, excluding existing workers (this player's and ``exclude``) and zealots; they join
       ``pinned`` and this round's spreaders (with last round's converts, R24).
    2. Cascade hop (R25, R27): chance ``γ_j P_j`` with ``P_j`` computed on ``preferences`` = v at the START of
       the round (before hires); attempt cap; workers and zealots are never targets (``cascade`` stream).
    3. Media intensity from the ``media`` stream (R28).

    Raises ``ValueError`` on a malformed allocation or too few eligible voters. The caller sets
    ``v[assigned] = b``, runs the social step with ``pinned`` workers held at b and pays ``cost``.
    """
    allocation.validate()
    missing = {"workers", "cascade", "media"} - set(generators)
    if missing:
        raise ValueError(f"missing generators: {sorted(missing)}")
    sus = electorate.susceptibility
    region_of = electorate.region_of
    graph = electorate.graph
    zealot = sus.zealot.to(torch.bool)
    others = state.pinned if exclude is None else state.pinned | exclude.to(torch.bool)

    eligible = eligible_workers(graph.degree, zealot, others, worker_cfg.min_degree)
    new_ids = pick_workers(region_of, eligible, allocation.hires_by_region, generators["workers"])
    new = torch.zeros_like(state.pinned)
    new[torch.cat(new_ids)] = True
    pinned = state.pinned | new
    spreaders = state.spreaders | new

    chance = conversion_chance(sus.campaign, preferences, manifesto, voting_cfg.sensitivity, voting_cfg.threshold)
    converted, attempts = ic_hop(graph, spreaders, chance, state.attempts, others | new | zealot,
                                 worker_cfg.attempt_cap, generators["cascade"])

    eta_media = _media_strength(region_of, prior_airings, media_cfg.strength, media_cfg.decay, sus.media.dtype)
    exposed = media_exposure(region_of, allocation.media_regions, media_cfg.exposure_prob, generators["media"])
    intensity = combine_intensities(_exposed_intensity(exposed, eta_media, sus.media), frozen=zealot)

    cost = worker_cost(allocation.total_hires, worker_cfg.worker_cost) + \
        media_cost(electorate.sizes, allocation.media_regions, media_cfg.base_cost)
    ids = tuple(torch.cat([old, add]) for old, add in zip(state.worker_ids, new_ids))
    new_state = WorkerState(pinned, ids, converted, attempts)
    return CampaignOutcome(new | converted, intensity, int(new.sum()), int(spreaders.sum()), int(converted.sum()),
                           int(exposed.sum()), cost, new_state)


def apply_assignments(v: torch.Tensor, outcomes: Sequence[CampaignOutcome], manifestos: torch.Tensor) -> torch.Tensor:
    """``v_j ← b_p`` for every voter assigned (new worker or convert) by player p, in player order."""
    for outcome, b in zip(outcomes, manifestos):
        v = torch.where(outcome.assigned.unsqueeze(-1), b.to(v.dtype), v)
    return v
