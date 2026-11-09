"""Observation-only rollout planner for campaign allocations (spec §3, §10.2; P4).

``candidate_allocations`` builds a compact, diverse, always-valid set of allocations from the public
observation (treasury, prices, region sizes, own airing counts, permanent hires so far — R16: available
workers = free + hired, and every extra worker bought is a permanent hire that keeps working later). ``plan_round`` scores every candidate by
playing it this round in each sampled world and then a fixed default continuation ("even" pacing:
half the per-round budget on media in the least-aired regions, the rest on extra workers split by
region size) for the remaining rounds; the objective is the mean expected share at the election
(spec §5) over the worlds. All C candidates × B worlds are simulated in one batched call
(:class:`~.worlds.WorldBatch`) with common random numbers, so the choice is deterministic per seed.
"""
from __future__ import annotations

import math
import time
from collections.abc import Sequence
from decimal import ROUND_FLOOR, Decimal
from typing import Any

import torch

from ..config import GameConfig, SeedStreams
from ..dynamics.campaign import Allocation, media_cost
from ..sectors import NUM_REGIONS, REGIONS
from .worlds import WorldBatch, hired_workers, public_sizes

__all__ = ["candidate_allocations", "default_allocation", "plan_round", "round_cap", "split_by_weight"]

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
                  config: GameConfig, hired: int = 0) -> Allocation:
    """Spend what is left of ``money`` after ``media`` on new hires; all workers (free + ``hired`` earlier
    hires + new hires, R16) split by ``weights``."""
    left = money - media_cost(sizes, media, config.media.base_cost)
    extra = max(0, int(left // config.workers.extra_worker_cost)) if left > 0 else 0
    workers = split_by_weight(config.workers.free_workers + hired + extra, weights)
    return Allocation(workers, extra, tuple(media))  # type: ignore[arg-type]


def default_allocation(sizes: Sequence[int], budget: Decimal, left: int, airings: Sequence[int],
                       config: GameConfig, hired: int = 0) -> Allocation:
    """The "even" continuation: per-round cap, half on media (least aired first), rest on new hires; all
    available workers (free + ``hired`` earlier hires + new hires) split by region size."""
    cap = round_cap(budget, left)
    media = _media_set(sizes, _media_order(sizes, airings), NUM_REGIONS, cap / 2, config.media.base_cost)
    return _with_workers(sizes, media, cap, [float(s) for s in sizes], config, hired)


def candidate_allocations(observation: dict[str, Any], config: GameConfig) -> list[Allocation]:
    """A compact, diverse candidate set (≈ 10–20), every member valid and affordable.

    Every plan assigns all available workers (free + earlier permanent hires, R16) plus its new hires.

    * all workers by region size with new-hire spend at 0 / 25 / 50 / 100 % of the per-round cap
      (cap = treasury / rounds left) and a front-loaded 2× cap;
    * media on 1 / 2 / all affordable regions (least-aired first), remaining cap on workers by size;
    * focus plans: all workers (free + cap) in one region; media + all workers in one region.
    """
    sizes = public_sizes(observation)
    treasury = _money(observation["treasury"])
    airings = [int(observation.get("media_airings", {}).get(r, 0)) for r in REGIONS]
    cap = round_cap(treasury, rounds_left(observation))
    base = config.media.base_cost
    hired = hired_workers(observation)
    by_size = [float(s) for s in sizes]
    order = _media_order(sizes, airings)
    out: list[Allocation] = []
    for frac in ("0", "0.25", "0.5", "1"):
        out.append(_with_workers(sizes, NO_MEDIA, (cap * Decimal(frac)).quantize(CENT, rounding=ROUND_FLOOR), by_size,
                                 config, hired))
    out.append(_with_workers(sizes, NO_MEDIA, min(treasury, cap * 2), by_size, config, hired))
    for count in (1, 2, NUM_REGIONS):
        out.append(_with_workers(sizes, _media_set(sizes, order, count, cap, base), cap, by_size, config, hired))
    for r in range(NUM_REGIONS):
        focus = [1.0 if k == r else 0.0 for k in range(NUM_REGIONS)]
        out.append(_with_workers(sizes, NO_MEDIA, cap, focus, config, hired))
        media = _media_set(sizes, [r], 1, cap, base)
        if any(media):
            out.append(_with_workers(sizes, media, cap, focus, config, hired))
    unique: list[Allocation] = []
    for a in out:
        a.validate(config.workers.free_workers, hired)
        cost = a.extra_workers * config.workers.extra_worker_cost + media_cost(sizes, a.media_regions, base)
        if cost <= treasury and a not in unique:
            unique.append(a)
    return unique


def value_guided_candidates(observation: dict[str, Any], batch: WorldBatch, current: torch.Tensor,
                            manifesto: Sequence[float] | torch.Tensor, config: GameConfig,
                            multipliers: Sequence[str] = ("0.5", "1", "1.5")) -> list[Allocation]:
    """Candidates from first-order marginal values estimated on the player's *own* sampled worlds.

    Same heuristic as the ORACLE marginal-value planner (``experiments/balance.policy_planner``) but
    computed on synthetic worlds replayed to the current round, i.e. on beliefs rather than the true state:
    ``dP_i/dc_i = β_v P_i (1 − P_i)(1 − S_i)``; a worker in region r is worth ``q η_w E_r[γ dP/dc]`` and an
    airing ``p_m η_m(n_r) Σ_r β dP/dc``; media is bought where its value per unit cost beats the best
    worker's, the rest of the money goes to extra workers concentrated (∝ value²) on the best regions.
    """
    from ..voting import similarity, vote_probability
    vote = config.voting
    b = batch.manifesto_tensor(manifesto)
    sim = similarity(current, b)
    prob = vote_probability(sim, vote.sensitivity, vote.threshold)
    gain = vote.sensitivity * prob * (1 - prob) * (1 - sim)                                    # [Bn]
    region = batch.local_region
    q = batch.contacts_per_worker
    airings = [int(observation.get("media_airings", {}).get(r, 0)) for r in REGIONS]
    sizes_true = public_sizes(observation)
    worker_value, media_value = [], []
    for r in range(NUM_REGIONS):
        mask = region == r
        worker_value.append(float(q * config.workers.contact_strength * (batch.campaign[mask] * gain[mask]).mean()))
        eta = config.media.strength / (1 + config.media.decay * airings[r])
        media_value.append(float(config.media.exposure_prob * eta * (batch.media[mask] * gain[mask]).sum()) / batch.num_worlds)
    treasury = _money(observation["treasury"])
    cap0 = round_cap(treasury, rounds_left(observation))
    cost_w = config.workers.extra_worker_cost
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
            if ratio > best_worker / float(cost_w) and price <= remaining:
                media[r] = True
                remaining -= price
        out.append(_with_workers(sizes_true, tuple(media), cap, weights, config, hired_workers(observation)))
    return out


def _cost(a: Allocation, sizes: Sequence[int], config: GameConfig) -> Decimal:
    return a.extra_workers * config.workers.extra_worker_cost + media_cost(sizes, a.media_regions, config.media.base_cost)


def plan_round(observation: dict[str, Any], worlds: WorldBatch | Sequence, manifesto: Sequence[float] | torch.Tensor,
               config: GameConfig, *, rollout: str = "even", seed: int = 0, current: torch.Tensor | None = None,
               candidates: list[Allocation] | None = None, value_guided: bool = True) -> tuple[Allocation, dict[str, Any]]:
    """Pick the candidate allocation with the highest mean expected election share over ``worlds``.

    ``worlds`` is a :class:`WorldBatch` or a list of electorates; ``current`` (``[B·n, 5]``) is the
    worlds' state at the current round (e.g. from :meth:`WorldBatch.replay`); if omitted the worlds are
    replayed from the observation. ``value_guided`` adds :func:`value_guided_candidates` (computed on the
    same worlds) to the candidate set. ``rollout="even"`` continues every candidate with
    :func:`default_allocation`; ``"none"`` continues with the available workers
    (free + permanent hires so far, R16) and no spending.
    """
    if rollout not in ("even", "none"):
        raise ValueError("rollout must be 'even' or 'none'")
    start = time.perf_counter()
    batch = worlds if isinstance(worlds, WorldBatch) else WorldBatch(list(worlds), config)
    if current is None:
        current = batch.replay(observation, seed)
    cands = list(candidates) if candidates is not None else candidate_allocations(observation, config)
    if value_guided:
        for a in value_guided_candidates(observation, batch, current, manifesto, config):
            if a not in cands and _cost(a, public_sizes(observation), config) <= _money(observation["treasury"]):
                cands.append(a)
    sizes_true = public_sizes(observation)
    b = batch.manifesto_tensor(manifesto)
    C = len(cands)
    treasury = _money(observation["treasury"])
    budgets = [treasury - _cost(a, sizes_true, config) for a in cands]
    airings0 = [int(observation.get("media_airings", {}).get(r, 0)) for r in REGIONS]
    airings = torch.tensor([airings0] * C, dtype=torch.int64, device=batch.device)
    left = rounds_left(observation)
    t0 = int(observation["round"])
    streams = SeedStreams(int(seed))
    v = current.unsqueeze(0).expand(C, -1, -1).contiguous()
    plans = list(cands)
    hires = [hired_workers(observation) + a.extra_workers for a in cands]   # R16: permanent, per candidate
    for step in range(left):
        gen = streams.generator("strategy", "plan", t0 + step, device=batch.device)
        v = batch.step(v, plans, airings, b, gen)
        airings = airings + torch.tensor([p.media_regions for p in plans], dtype=torch.int64, device=batch.device)
        if step + 1 < left:
            remaining = left - step - 1
            nxt = []
            for c in range(C):
                if rollout == "even":
                    a = default_allocation(sizes_true, budgets[c], remaining, airings[c].tolist(), config, hires[c])
                else:
                    a = Allocation(split_by_weight(config.workers.free_workers + hires[c], [float(s) for s in sizes_true]))
                budgets[c] -= _cost(a, sizes_true, config)
                hires[c] += a.extra_workers
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
