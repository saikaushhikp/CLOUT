"""Observation-only rollout planner for campaign allocations (spec §3, §10.2; P4, P8).

P8: an allocation buys NEW permanent workers per region (``hires_by_region``, one-time ``costs.worker`` each)
plus media regions. Workers are front-loaded value: a worker bought in round t seeds an Independent Cascade
that advances one hop per remaining round, so early hires reach further.

``candidate_allocations`` builds a compact, diverse, always-valid set of allocations from the public
observation (treasury, prices, region sizes, own airing counts). ``plan_round`` scores every candidate by
playing it this round in each sampled world and then a fixed default continuation ("even" pacing: half the
per-round budget on media in the least-aired regions, the rest on workers split by region size) for the
remaining rounds; the objective is the mean expected share at the election (spec §5) over the worlds. All
C candidates × B worlds are simulated in one batched call (:class:`~.worlds.WorldBatch`, carrying each plan's
worker state) with common random numbers, so the choice is deterministic per seed.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from decimal import ROUND_FLOOR, Decimal
from typing import Any

import torch

from ..config import GameConfig, SeedStreams
from ..dynamics.campaign import Allocation, media_cost, worker_cost
from ..sectors import NUM_REGIONS, REGIONS
from .worlds import BatchWorkers, WorldBatch, public_sizes

__all__ = ["allocation_cost", "candidate_allocations", "default_allocation", "front_allocation", "plan_round",
           "round_cap",
           "split_by_weight", "workers_for"]

CENT = Decimal("0.01")
NO_MEDIA = (False,) * NUM_REGIONS


def split_by_weight(total: int, weights: Sequence[float]) -> tuple[int, int, int, int]:
    """Largest-remainder integer split of ``total`` proportional to ``weights`` (ties → lower index)."""
    s = float(sum(weights))
    if total <= 0 or s <= 0:
        return (0, 0, 0, 0)
    raw = [total * w / s for w in weights]
    base = [int(math.floor(x)) for x in raw]
    order = sorted(range(NUM_REGIONS), key=lambda i: (-(raw[i] - base[i]), i))
    for i in order[: total - sum(base)]:
        base[i] += 1
    return tuple(base)  # type: ignore[return-value]


def _money(text: str | Decimal) -> Decimal:
    return Decimal(str(text))


def rounds_left(observation: dict[str, Any]) -> int:
    """Rounds still to be allocated, including the current one (≥ 1 in ROUND)."""
    if observation["phase"] == "SURVEY":
        return int(observation["rounds"])
    return max(1, int(observation["rounds"]) - int(observation["round"]) + 1)


def round_cap(budget: Decimal, left: int) -> Decimal:
    """Even pacing: ``budget / rounds_left`` rounded down to the cent."""
    return (budget / max(1, left)).quantize(CENT, rounding=ROUND_FLOOR)


def _media_order(sizes: Sequence[int], airings: Sequence[int]) -> list[int]:
    """Prefer regions aired least (diminishing returns), then larger regions."""
    return sorted(range(NUM_REGIONS), key=lambda r: (airings[r], -sizes[r], r))


def _media_set(sizes: Sequence[int], order: Sequence[int], count: int, cap: Decimal, base: Decimal) -> tuple[bool, ...]:
    """Greedily air up to ``count`` regions in ``order`` while the total stays within ``cap``."""
    chosen = [False] * NUM_REGIONS
    for r in order:
        if sum(chosen) >= count:
            break
        trial = list(chosen)
        trial[r] = True
        if media_cost(sizes, trial, base) <= cap:
            chosen = trial
    return tuple(chosen)


def _with_workers(sizes: Sequence[int], media: tuple[bool, ...], money: Decimal, weights: Sequence[float],
                  config: GameConfig) -> Allocation:
    """Spend what is left of ``money`` after ``media`` on new workers split by ``weights``."""
    left = money - media_cost(sizes, media, config.media.base_cost)
    return Allocation(workers_for(left, weights, config), tuple(media))  # type: ignore[arg-type]


def workers_for(money: Decimal, weights: Sequence[float], config: GameConfig) -> tuple[int, int, int, int]:
    """As many workers as ``money`` buys, split by ``weights`` (largest remainder)."""
    unit = config.workers.worker_cost
    count = int(money // unit) if money > 0 and unit > 0 else 0
    return split_by_weight(count, weights)


def allocation_cost(a: Allocation, sizes: Sequence[int], config: GameConfig) -> Decimal:
    """``Σ hires × worker_cost + media price`` (exact Decimal, same as the engine)."""
    return worker_cost(a.total_hires, config.workers.worker_cost) + media_cost(sizes, a.media_regions, config.media.base_cost)


def default_allocation(sizes: Sequence[int], budget: Decimal, left: int, airings: Sequence[int],
                       config: GameConfig) -> Allocation:
    """The "even" continuation: per-round cap, half on media (least aired first), the rest on new workers
    split by region size."""
    cap = round_cap(budget, left)
    media = _media_set(sizes, _media_order(sizes, airings), NUM_REGIONS, cap / 2, config.media.base_cost)
    return _with_workers(sizes, media, cap, [float(s) for s in sizes], config)


def front_allocation(sizes: Sequence[int], budget: Decimal, left: int, airings: Sequence[int], config: GameConfig,
                     *, first: bool) -> Allocation:
    """The "front" continuation (optional, X14; the default rollout stays "even").

    In the first continuation round (``first``) media follows the even pace (half the per-round cap, least aired
    first) and ALL the remaining worker money is spent at once: everything except a media reserve of
    ``(left − 1) × cap/2`` for the later rounds buys new workers split by region size (workers bought earlier
    cascade for more hops). Later rounds buy no workers and spend their even cap on media only."""
    cap = round_cap(budget, left)
    order = _media_order(sizes, airings)
    if not first:
        return Allocation((0, 0, 0, 0), _media_set(sizes, order, NUM_REGIONS, cap, config.media.base_cost))  # type: ignore[arg-type]
    media = _media_set(sizes, order, NUM_REGIONS, cap / 2, config.media.base_cost)
    reserve = ((cap / 2) * (left - 1)).quantize(CENT, rounding=ROUND_FLOOR)
    money = max(Decimal("0.00"), budget - reserve)
    return _with_workers(sizes, media, money, [float(s) for s in sizes], config)


def candidate_allocations(observation: dict[str, Any], config: GameConfig) -> list[Allocation]:
    """A compact, diverse candidate set (≤ 20), every member valid and affordable.

    * no spending; workers by region size for 25 / 50 / 100 % of the per-round cap (cap = treasury / rounds
      left), a front-loaded 2× cap and half the treasury (hires cascade for every remaining round);
    * media on 1 / 2 / all affordable regions (least-aired first), remaining cap on workers by size, and the
      same media sets alone (media-only plans);
    * focus plans: the cap (at least one fee, if the treasury allows) on workers in one region; one airing +
      workers in one region; the cheapest single airing alone.
    """
    sizes = public_sizes(observation)
    treasury = _money(observation["treasury"])
    airings = [int(observation.get("media_airings", {}).get(r, 0)) for r in REGIONS]
    left = rounds_left(observation)
    cap = round_cap(treasury, left)
    base = config.media.base_cost
    by_size = [float(s) for s in sizes]
    order = _media_order(sizes, airings)
    out: list[Allocation] = [Allocation()]
    for frac in ("0.25", "0.5", "1"):
        out.append(_with_workers(sizes, NO_MEDIA, (cap * Decimal(frac)).quantize(CENT, rounding=ROUND_FLOOR), by_size, config))
    out.append(_with_workers(sizes, NO_MEDIA, min(treasury, cap * 2), by_size, config))
    if left > 2:
        out.append(_with_workers(sizes, NO_MEDIA, (treasury / 2).quantize(CENT, rounding=ROUND_FLOOR), by_size, config))
    for count in (1, 2, NUM_REGIONS):
        media = _media_set(sizes, order, count, cap, base)
        out.append(_with_workers(sizes, media, cap, by_size, config))
        out.append(Allocation((0, 0, 0, 0), media))  # type: ignore[arg-type]
    cheapest = _media_set(sizes, sorted(range(NUM_REGIONS), key=lambda r: (sizes[r], r)), 1, treasury, base)
    out.append(Allocation((0, 0, 0, 0), cheapest))  # type: ignore[arg-type]   # media-only even below the cap
    # focus plans: at least one worker (and one airing) when the treasury allows it even if the even cap
    # does not — workers are front-loaded value, so a low cap must not remove them from the set
    fee = config.workers.worker_cost
    for r in range(NUM_REGIONS):
        focus = [1.0 if k == r else 0.0 for k in range(NUM_REGIONS)]
        out.append(_with_workers(sizes, NO_MEDIA, min(treasury, max(cap, fee)), focus, config))
        one = tuple(k == r for k in range(NUM_REGIONS))
        price = media_cost(sizes, one, base)
        if price <= treasury:
            out.append(_with_workers(sizes, one, min(treasury, max(cap, price + fee)), focus, config))
    unique: list[Allocation] = []
    for a in out:
        a.validate()
        if allocation_cost(a, sizes, config) <= treasury and a not in unique:
            unique.append(a)
    return unique[:20]


def value_guided_candidates(observation: dict[str, Any], batch: WorldBatch, current: torch.Tensor,
                            manifesto: Sequence[float] | torch.Tensor, config: GameConfig,
                            multipliers: Sequence[str] = ("0.5", "1", "1.5")) -> list[Allocation]:
    """Candidates from first-order marginal values estimated on the player's *own* sampled worlds.

    Computed on synthetic worlds replayed to the current round (beliefs, not the true state). Media: an
    airing in r is worth ``p_m η_m(n_r) Σ_r β dP/dc`` with ``dP_i/dc_i = β_v P_i (1 − P_i)(1 − S_i)``. Workers
    (P8): a worker in r seeds a cascade with per-attempt chance ``a = E_r[γP]`` and mean degree k̄_r, so it
    reaches ≈ ``k̄_r a Σ_{h<L} (k̄_r a)^h`` voters over the L remaining hops (capped at half the region), each
    gaining ``E_r[γP (P_b − P)] / a`` in vote probability (P_b = P at similarity 1). Media is bought where its
    value per unit cost beats the best worker's; the rest of the money buys workers concentrated (∝ value²)
    on the best regions. Values are per TRUE-sized world (media scaled by N/n).
    """
    from ..voting import similarity, vote_probability
    vote = config.voting
    b = batch.manifesto_tensor(manifesto)
    sim = similarity(current, b)
    prob = vote_probability(sim, vote.sensitivity, vote.threshold)
    gain = vote.sensitivity * prob * (1 - prob) * (1 - sim)                                    # [Bn]
    p_full = float(vote_probability(torch.tensor(1.0), vote.sensitivity, vote.threshold))
    chance = (batch.campaign * prob).clamp(0, 1)
    region = batch.local_region
    degree = batch.graph.degree.to(current.dtype)
    hops = rounds_left(observation)
    airings = [int(observation.get("media_airings", {}).get(r, 0)) for r in REGIONS]
    sizes_true = public_sizes(observation)
    up = sum(sizes_true) / batch.n
    worker_value, media_value = [], []
    for r in range(NUM_REGIONS):
        mask = region == r
        a = float(chance[mask].mean())
        kbar = float(degree[mask].mean())
        r0 = kbar * a
        reach = min(kbar * a * sum(r0 ** h for h in range(hops)), 0.5 * sizes_true[r])
        per_convert = float((chance[mask] * (p_full - prob[mask]).clamp_min(0)).mean()) / a if a > 0 else 0.0
        worker_value.append(reach * per_convert)
        eta = config.media.strength / (1 + config.media.decay * airings[r])
        media_value.append(up * float(config.media.exposure_prob * eta * (batch.media[mask] * gain[mask]).sum()) / batch.num_worlds)
    treasury = _money(observation["treasury"])
    cap0 = round_cap(treasury, hops)
    cost_w = config.workers.worker_cost
    best_worker = max(worker_value)
    ratios = []
    for r in range(NUM_REGIONS):
        one = [k == r for k in range(NUM_REGIONS)]
        price = media_cost(sizes_true, one, config.media.base_cost)
        ratios.append((media_value[r] / float(price) if price > 0 else 0.0, r, price))
    weights = [max(v, 0.0) ** 2 for v in worker_value]
    if sum(weights) <= 0:
        weights = [float(s) for s in sizes_true]
    out = []
    for mult in multipliers:
        cap = min(treasury, (cap0 * Decimal(mult)).quantize(CENT, rounding=ROUND_FLOOR))
        media = [False] * NUM_REGIONS
        remaining = cap
        for ratio, r, price in sorted(ratios, reverse=True):
            if ratio > best_worker / max(float(cost_w), 1e-9) and price <= remaining:
                media[r] = True
                remaining -= price
        out.append(_with_workers(sizes_true, tuple(media), cap, weights, config))
    return out


def plan_round(observation: dict[str, Any], worlds: WorldBatch | Sequence, manifesto: Sequence[float] | torch.Tensor,
               config: GameConfig, *, rollout: str = "even", seed: int = 0, current: torch.Tensor | None = None,
               current_workers: BatchWorkers | None = None, candidates: list[Allocation] | None = None,
               value_guided: bool = True) -> tuple[Allocation, dict[str, Any]]:
    """Pick the candidate allocation with the highest mean expected election share over ``worlds``.

    ``worlds`` is a :class:`WorldBatch` or a list of electorates; ``current`` (``[B·n, 5]``) and
    ``current_workers`` are the worlds' state at the current round (from :meth:`WorldBatch.replay_state`); if
    ``current`` is omitted both are replayed from the observation. ``value_guided`` adds
    :func:`value_guided_candidates` (computed on the same worlds) to the candidate set. ``rollout="even"``
    continues every candidate with :func:`default_allocation`; ``"none"`` continues with no spending (the
    workers already bought keep their cascades going); ``"front"`` (optional, X14) spends the remaining worker
    money in the next round and media only afterwards (:func:`front_allocation`).
    """
    if rollout not in ("even", "none", "front"):
        raise ValueError("rollout must be 'even', 'none' or 'front'")
    start = time.perf_counter()
    batch = worlds if isinstance(worlds, WorldBatch) else WorldBatch(list(worlds), config)
    if current is None:
        current, current_workers = batch.replay_state(observation, seed)
    sizes_true = public_sizes(observation)
    treasury = _money(observation["treasury"])
    cands = list(candidates) if candidates is not None else candidate_allocations(observation, config)
    if value_guided:
        for a in value_guided_candidates(observation, batch, current, manifesto, config):
            if a not in cands and allocation_cost(a, sizes_true, config) <= treasury:
                cands.append(a)
    b = batch.manifesto_tensor(manifesto)
    C = len(cands)
    budgets = [treasury - allocation_cost(a, sizes_true, config) for a in cands]
    airings0 = [int(observation.get("media_airings", {}).get(r, 0)) for r in REGIONS]
    airings = torch.tensor([airings0] * C, dtype=torch.int64, device=batch.device)
    left = rounds_left(observation)
    t0 = int(observation["round"])
    streams = SeedStreams(int(seed))
    v = current.unsqueeze(0).expand(C, -1, -1).contiguous()
    workers = current_workers
    plans = list(cands)
    for step in range(left):
        gen = streams.generator("strategy", "plan", t0 + step, device=batch.device)
        v, workers = batch.step(v, plans, airings, b, gen, workers)
        airings = airings + torch.tensor([p.media_regions for p in plans], dtype=torch.int64, device=batch.device)
        if step + 1 < left:
            remaining = left - step - 1
            nxt = []
            for c in range(C):
                if rollout == "even":
                    a = default_allocation(sizes_true, budgets[c], remaining, airings[c].tolist(), config)
                elif rollout == "front":
                    a = front_allocation(sizes_true, budgets[c], remaining, airings[c].tolist(), config,
                                         first=(step == 0))
                else:
                    a = Allocation()
                budgets[c] -= allocation_cost(a, sizes_true, config)
                nxt.append(a)
            plans = nxt
    shares = batch.shares(v, b)                                  # [C, B]
    score = shares.mean(1)
    wins = (shares > 0.5).to(score.dtype).mean(1)
    best = int(torch.argmax(score))
    diagnostics = {
        "scores": [float(x) for x in score.tolist()],
        "win_fraction": [float(x) for x in wins.tolist()],
        "best_index": best,
        "best_score": float(score[best]),
        "num_candidates": C,
        "num_worlds": batch.num_worlds,
        "candidates": cands,
        "seconds": time.perf_counter() - start,
    }
    return cands[best], diagnostics
