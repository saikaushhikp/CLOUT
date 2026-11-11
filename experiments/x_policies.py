"""Scripted ``RoundContext`` policies for the v3 experiments, in one place (plan §2).

Every policy is a pure function of the ``RoundContext`` (no hidden state between calls), either a plain
function or a frozen dataclass whose fields are its explicit parameters. They are used by ``simulate_world``
and by the batched runner ``x_batch.run_batch`` (which hands each world its own TRUE context).

From ``balance.py`` (the same function objects, so X0 reproduces the v2 gate code exactly):
  ``none``, ``pay_workers`` (even), ``front_workers``, ``pay_media``, ``pay_split``, ``planner`` (ORACLE).
New (§2; X4, X6, X11, X17):
  :class:`WorkerTiming`    fixed total of workers bought front / even / late, placed largest / proportional /
                           best-matched / one region (X4);
  :class:`RegionFocus`     the round's pace spent in one region only, on workers, media or both;
  :class:`Sequenced`       per-round channel schedule, e.g. media in round 1 then workers in round 2 (X11);
  :class:`Probe`           round 1 spends on one region and one channel, then hands over to another policy
                           (a RoundContext stub of the X17 probe; the real probe player is observation-only);
  :class:`PeoplePleaser`   RoundContext version of player I: even pace, media then workers on the
                           1–2 largest regions that form a majority;
  :class:`StrongArm`       RoundContext version of player III: the split pattern at 1.5× pace.
Manifesto helpers for the RoundContext versions: :func:`people_pleaser_manifesto`, :func:`strong_arm_manifesto`.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal

import torch

import balance
from balance import (  # noqa: F401  (re-exported: the gate policies and their helpers)
    NO_MEDIA, media_within, policy_front_workers, policy_none, policy_pay_media, policy_pay_split,
    policy_pay_workers, policy_planner, round_cap, sizes_of, split_by_weight,
)
from dumbocracy.dynamics.campaign import Allocation, media_cost
from dumbocracy.simulate import RoundContext
from dumbocracy.voting import similarity, vote_probability

CENT = Decimal("0.01")
Policy = Callable[[RoundContext], Allocation]

# The X0 gate set (names as in balance.POLICIES; "free_even" is an alias of "none" under P8).
GATE_POLICIES: dict[str, Policy] = dict(balance.POLICIES)


# ------------------------------------------------------------------------------------------ helpers
def _floor(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_FLOOR)


def pace_cap(ctx: RoundContext, pace: float = 1.0) -> Decimal:
    """``pace`` × the even per-round cap, never above the remaining treasury."""
    return min(ctx.budget, _floor(round_cap(ctx) * Decimal(str(pace))))


def affordable_workers(ctx: RoundContext, cap: Decimal) -> int:
    unit = ctx.config.workers.worker_cost
    return int(cap // unit) if cap > 0 and unit > 0 else 0


def regional_support(ctx: RoundContext) -> list[float]:
    """Mean vote probability per region on the TRUE current v (oracle information)."""
    cfg = ctx.config
    prob = vote_probability(similarity(ctx.preferences, ctx.manifesto), cfg.voting.sensitivity, cfg.voting.threshold)
    sums = torch.zeros(4, dtype=prob.dtype, device=prob.device).index_add_(0, ctx.electorate.region_of, prob)
    return [float(s) / max(1, n) for s, n in zip(sums.tolist(), sizes_of(ctx))]


def majority_regions(sizes: Sequence[int]) -> list[int]:
    """The 1–2 largest regions (largest first) whose voters exceed half of N; 3 if two are not enough."""
    order = sorted(range(4), key=lambda r: -sizes[r])
    total, chosen = sum(sizes), []
    for r in order:
        chosen.append(r)
        if 2 * sum(sizes[i] for i in chosen) > total:
            break
    return chosen


def _placement_weights(ctx: RoundContext, placement: str) -> list[float]:
    sizes = sizes_of(ctx)
    if placement == "proportional":
        return [float(x) for x in sizes]
    if placement == "largest":
        best = max(range(4), key=lambda r: sizes[r])
        return [1.0 if r == best else 0.0 for r in range(4)]
    if placement == "best_matched":   # ORACLE: highest true mean support for b
        support = regional_support(ctx)
        best = max(range(4), key=lambda r: support[r])
        return [1.0 if r == best else 0.0 for r in range(4)]
    if placement.startswith("region:"):
        target = int(placement.split(":", 1)[1])
        return [1.0 if r == target else 0.0 for r in range(4)]
    raise ValueError(f"unknown placement {placement!r}")


def _one_region(r: int) -> tuple[bool, bool, bool, bool]:
    return tuple(i == r for i in range(4))  # type: ignore[return-value]


# ------------------------------------------------------------------------------------------ X4 timing
@dataclass(frozen=True)
class WorkerTiming:
    """Buy ``total`` workers over the game: ``timing`` "front" (all in round 1), "even" (largest-remainder split
    over the rounds) or "late" (all in the last round); placed by ``placement`` ("proportional", "largest",
    "best_matched" = ORACLE, or "region:<r>"). Hires are clipped to what the treasury affords. No media."""

    total: int
    timing: str = "even"
    placement: str = "proportional"

    def per_round(self, rounds: int) -> list[int]:
        if self.timing == "front":
            return [self.total] + [0] * (rounds - 1)
        if self.timing == "late":
            return [0] * (rounds - 1) + [self.total]
        if self.timing == "even":
            base, extra = divmod(self.total, rounds)
            return [base + (1 if t < extra else 0) for t in range(rounds)]
        raise ValueError(f"unknown timing {self.timing!r}")

    def __call__(self, ctx: RoundContext) -> Allocation:
        want = self.per_round(ctx.rounds)[ctx.round]
        count = min(want, affordable_workers(ctx, ctx.budget))
        return Allocation(split_by_weight(count, _placement_weights(ctx, self.placement)), NO_MEDIA)


# ------------------------------------------------------------------------------------------ region focus
@dataclass(frozen=True)
class RegionFocus:
    """Spend ``pace`` × the even cap in ``region`` only: "workers", "media" or "split" (media first if it fits in
    half the cap, the rest on workers)."""

    region: int
    channel: str = "split"
    pace: float = 1.0

    def __call__(self, ctx: RoundContext) -> Allocation:
        cap = pace_cap(ctx, self.pace)
        r = self.region
        hires = [0, 0, 0, 0]
        media = NO_MEDIA
        if self.channel in ("media", "split"):
            limit = cap if self.channel == "media" else cap / 2
            price = media_cost(sizes_of(ctx), _one_region(r), ctx.config.media.base_cost)
            if price <= limit:
                media = _one_region(r)
                cap -= price
        if self.channel in ("workers", "split"):
            hires[r] = affordable_workers(ctx, cap)
        elif self.channel != "media":
            raise ValueError(f"unknown channel {self.channel!r}")
        return Allocation(tuple(hires), media)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------ X11 sequencing
@dataclass(frozen=True)
class Sequenced:
    """Per-round channel schedule. ``schedule[t]`` ∈ {"none", "workers", "media", "both"}; rounds past the end
    of the schedule use "none". ``workers`` = number bought in that round (placed by ``placement``);
    ``media_regions`` = regions aired in a media round (default: the largest region). Clipped to the treasury
    (media first)."""

    schedule: tuple[str, ...]
    workers: int = 0
    media_regions: tuple[bool, bool, bool, bool] | None = None
    placement: str = "proportional"

    def __call__(self, ctx: RoundContext) -> Allocation:
        kind = self.schedule[ctx.round] if ctx.round < len(self.schedule) else "none"
        if kind not in ("none", "workers", "media", "both"):
            raise ValueError(f"unknown schedule entry {kind!r}")
        budget = ctx.budget
        media = NO_MEDIA
        if kind in ("media", "both"):
            sizes = sizes_of(ctx)
            wanted = self.media_regions or _one_region(max(range(4), key=lambda r: sizes[r]))
            price = media_cost(sizes, wanted, ctx.config.media.base_cost)
            if price <= budget:
                media, budget = tuple(wanted), budget - price
        hires = (0, 0, 0, 0)
        if kind in ("workers", "both"):
            count = min(self.workers, affordable_workers(ctx, budget))
            hires = split_by_weight(count, _placement_weights(ctx, self.placement))
        return Allocation(hires, media)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------ X17 probe stub
@dataclass(frozen=True)
class Probe:
    """Round 1: spend ``pace`` × the even cap on ONE region and ONE channel ("media" or "workers"); later rounds:
    delegate to ``then`` (default: the ORACLE planner). The region choice for a real probe (maximal split of
    the τ hypotheses) belongs to X17's observation-only player; here it is a parameter."""

    region: int
    channel: str = "media"
    pace: float = 1.0
    then: Policy = policy_planner

    def __call__(self, ctx: RoundContext) -> Allocation:
        if ctx.round == 0:
            return RegionFocus(self.region, self.channel, self.pace)(ctx)
        return self.then(ctx)


# ------------------------------------------------------------------------------------------ X6 personas
@dataclass(frozen=True)
class PeoplePleaser:
    """RoundContext version of player I: even pace; air media in the majority coalition (largest first) while
    it fits in ``media_share`` of the cap, spend the rest on workers in the coalition (by size)."""

    pace: float = 1.0
    media_share: float = 0.5

    def __call__(self, ctx: RoundContext) -> Allocation:
        sizes = sizes_of(ctx)
        coalition = majority_regions(sizes)
        cap = pace_cap(ctx, self.pace)
        media = media_within(ctx, _floor(cap * Decimal(str(self.media_share))), coalition)
        cap -= media_cost(sizes, media, ctx.config.media.base_cost)
        weights = [float(sizes[r]) if r in coalition else 0.0 for r in range(4)]
        return Allocation(split_by_weight(affordable_workers(ctx, cap), weights), media)


@dataclass(frozen=True)
class StrongArm:
    """RoundContext version of player III: the "split" pattern (half media on the largest regions, the rest on
    workers by region size) at ``pace`` × the even cap (default 1.5×, so the money runs out early)."""

    pace: float = 1.5

    def __call__(self, ctx: RoundContext) -> Allocation:
        cap = pace_cap(ctx, self.pace)
        order = sorted(range(4), key=lambda r: -sizes_of(ctx)[r])
        media = media_within(ctx, cap / 2, order)
        cap -= media_cost(sizes_of(ctx), media, ctx.config.media.base_cost)
        return Allocation(split_by_weight(affordable_workers(ctx, cap), [float(x) for x in sizes_of(ctx)]), media)


def people_pleaser_manifesto(electorate, config) -> torch.Tensor:
    """ORACLE-side stand-in for player I's manifesto: the size-weighted blend of the TRUE anticipated regional
    means of the majority coalition (the observation-only player uses the posterior instead, X6)."""
    from dumbocracy.strategies.oracle import anticipate
    v = anticipate(electorate, config)
    sizes = [int(x) for x in electorate.sizes.tolist()]
    coalition = majority_regions(sizes)
    mask = torch.zeros_like(electorate.region_of, dtype=torch.bool)
    for r in coalition:
        mask |= electorate.region_of == r
    b = v[mask].to(torch.float64).mean(0)
    return (b / b.sum()).to(v.dtype)


def strong_arm_manifesto(kind: str = "fixed", *, seed: int = 0, dtype: torch.dtype = torch.float32,
                         device: torch.device | str = "cpu", sector: int = 0) -> torch.Tensor:
    """Player III's manifesto: "fixed" = all weight 0.6 on one ``sector`` and 0.1 elsewhere; "random" = a flat
    Dirichlet(1) draw from a CPU generator keyed by ``seed``."""
    if kind == "fixed":
        b = torch.full((5,), 0.1, dtype=torch.float64)
        b[sector] = 0.6
    elif kind == "random":
        g = torch.Generator().manual_seed(int(seed))
        x = -torch.log(torch.rand(5, generator=g, dtype=torch.float64))
        b = x / x.sum()
    else:
        raise ValueError("kind must be 'fixed' or 'random'")
    return b.to(device=device, dtype=dtype)


# ------------------------------------------------------------------------------------------ registry
def make_policy(name: str, **params) -> Policy:
    """Build a policy by name: the gate names of ``balance.POLICIES`` (no parameters), or one of
    ``timing``, ``region_focus``, ``sequenced``, ``probe``, ``people_pleaser``, ``strong_arm`` with keyword
    parameters (e.g. ``make_policy("timing", total=10, timing="front")``)."""
    if name in GATE_POLICIES:
        if params:
            raise ValueError(f"policy {name!r} takes no parameters")
        return GATE_POLICIES[name]
    classes = {"timing": WorkerTiming, "region_focus": RegionFocus, "sequenced": Sequenced, "probe": Probe,
               "people_pleaser": PeoplePleaser, "strong_arm": StrongArm}
    if name not in classes:
        raise ValueError(f"unknown policy {name!r}")
    return classes[name](**params)


def workers_for_spend(spend: Decimal, unit: Decimal) -> int:
    """Number of workers a spend buys (for matched-spend arms, X4/X11)."""
    return int(math.floor(spend / unit)) if unit > 0 else 0
