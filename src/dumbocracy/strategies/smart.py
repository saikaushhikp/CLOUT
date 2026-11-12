"""The "smart player": :class:`BayesPlanner` with the Method 4 touch-ups (reports/v3/experimentation/strategies.md).

:class:`SmartPlayer` is a :class:`~.players.BayesPlanner` (same surveys, posterior and observation-only worlds) with
four opt-in options. Each one is off when set to the value in brackets, and with all four off the player makes
exactly the same decisions as ``BayesPlanner`` (tests/unit/test_smart_player.py):

* ``front_candidates`` [False] — add the X13 plan-search shape to the planner's candidates
  (:func:`front_media_candidates`): a few workers in one of the best regions plus media in 2–4 regions,
  spending beyond the even per-round cap;
* ``rollout`` ["even"] — the continuation used to score candidates; ``"taper"`` (``planner.taper_allocation``)
  continues like the X13 best plans: no more workers, front-loaded media;
* ``plan_world_fraction`` [None] — imagined planning electorates get at least this fraction of the true N voters
  (capped at N and at ``plan_world_cap``), so a handful of workers is not scaled below one voter
  (``WorldBatch.hire_scale = n / N``);
* ``manifesto_method`` ["pg"] — ``"admm"`` uses the consensus-ADMM manifesto (``manifesto_opt.admm_manifesto``, the
  X6 ``bayes_admm`` arm) instead of projected gradient on the same anticipated posterior worlds.

The recommended configuration (S1, ``experiments/s1_smart_player.py``) is :data:`RECOMMENDED`; build it with
:func:`smart_player`. Everything here reads only the observation dict (plus the public default rules).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Sequence

import torch

from ..config import GameConfig
from ..dynamics.campaign import Allocation, media_cost
from ..sectors import NUM_REGIONS, REGIONS
from .manifesto_opt import admm_manifesto
from .oracle import anticipate
from .planner import _media_set, allocation_cost, candidate_allocations, plan_round, region_values
from .players import BayesPlanner, simplex_list
from .worlds import WorldBatch, public_sizes

__all__ = ["RECOMMENDED", "SmartPlayer", "front_media_candidates", "smart_player"]


def front_media_candidates(observation: dict[str, Any], config: GameConfig, worker_value: Sequence[float],
                           media_value: Sequence[float], *, worker_counts: Sequence[int] = (1, 2, 3, 4, 5),
                           media_counts: Sequence[int] = (2, 3, 4), worker_regions: int = 2) -> list[Allocation]:
    """The X13 best-plan shape as one-round candidates: ``k`` new workers in ONE region (each of the
    ``worker_regions`` regions with the highest first-order worker value) plus media in ``m`` regions, chosen by
    media value per ₫ (``planner.region_values``), paid from the whole treasury (not the even cap); also the
    media-only sets (k = 0). Every candidate is valid and affordable; duplicates removed."""
    sizes = public_sizes(observation)
    treasury = Decimal(str(observation["treasury"]))
    fee = config.workers.worker_cost
    base = config.media.base_cost
    airings = [int(observation.get("media_airings", {}).get(r, 0)) for r in REGIONS]
    prices = [media_cost(sizes, [k == r for k in range(NUM_REGIONS)], base) for r in range(NUM_REGIONS)]
    ratio = [media_value[r] / float(prices[r]) if prices[r] > 0 else 0.0 for r in range(NUM_REGIONS)]
    order = sorted(range(NUM_REGIONS), key=lambda r: (-ratio[r], airings[r], -sizes[r], r))
    regions = sorted(range(NUM_REGIONS), key=lambda r: (-worker_value[r], -sizes[r], r))[:max(1, worker_regions)]
    out: list[Allocation] = []
    for k in (0, *worker_counts):
        for r in (regions if k > 0 else regions[:1]):
            money = treasury - fee * k
            if money < 0:
                continue
            hires = tuple(k if i == r else 0 for i in range(NUM_REGIONS))
            for m in media_counts:
                out.append(Allocation(hires, _media_set(sizes, order, m, money, base)))  # type: ignore[arg-type]
    unique: list[Allocation] = []
    for a in out:
        a.validate()
        if allocation_cost(a, sizes, config) <= treasury and a not in unique:
            unique.append(a)
    return unique


@dataclass
class SmartPlayer(BayesPlanner):
    """BayesPlanner + opt-in Method 4 touch-ups (module docstring). Defaults = plain BayesPlanner behaviour."""

    front_candidates: bool = False
    front_worker_counts: tuple[int, ...] = (1, 2, 3, 4, 5)
    front_media_counts: tuple[int, ...] = (2, 3, 4)
    front_worker_regions: int = 2
    taper_ratio: float = 0.6
    plan_world_fraction: float | None = None
    plan_world_cap: int = 20_000
    manifesto_method: str = "pg"
    admm_rho: float = 5.0
    admm_iterations: int = 300
    admm_records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.manifesto_method not in ("pg", "admm"):
            raise ValueError("manifesto_method must be 'pg' or 'admm'")
        if self.plan_world_fraction is not None and not 0 < self.plan_world_fraction <= 1:
            raise ValueError("plan_world_fraction must lie in (0, 1]")

    # -- imagined electorates
    def plan_world_voters(self, num_voters: int) -> int:
        """Voters per imagined planning electorate for a game of ``num_voters`` (before the min with N)."""
        if self.plan_world_fraction is None:
            return self.world_voters
        return min(max(self.world_voters, math.ceil(self.plan_world_fraction * num_voters)), self.plan_world_cap)

    def _worlds(self, observation: dict[str, Any], posterior, count: int, tag: str):
        if tag != "plan" or self.plan_world_fraction is None:
            return super()._worlds(observation, posterior, count, tag)
        saved = self.world_voters
        self.world_voters = self.plan_world_voters(int(observation["num_voters"]))
        try:
            return super()._worlds(observation, posterior, count, tag)
        finally:
            self.world_voters = saved

    # -- decisions
    def manifesto(self, observation: dict[str, Any], posterior=None) -> list[float]:
        if self.manifesto_method == "pg":
            return super().manifesto(observation, posterior)
        posterior = posterior or self._posterior(observation)
        cfg, worlds = self._worlds(observation, posterior, self.manifesto_worlds, "manifesto")
        v_t = torch.cat([anticipate(w, cfg) for w in worlds])
        region_of = torch.cat([w.region_of for w in worlds]).to(v_t.device)
        start = time.perf_counter()
        sol = admm_manifesto(v_t, region_of, cfg.voting.sensitivity, cfg.voting.threshold, rho=self.admm_rho,
                             iterations=self.admm_iterations)
        self.admm_records.append({"iterations": sol.iterations, "share_on_worlds": sol.share,
                                  "seconds": time.perf_counter() - start})
        return simplex_list(sol.manifesto)

    def allocation(self, observation: dict[str, Any], posterior=None) -> tuple[Allocation, dict[str, Any]]:
        if not self.planning:
            return super().allocation(observation, posterior)
        posterior = posterior or self._posterior(observation)
        cfg, worlds = self._worlds(observation, posterior, self.num_worlds, "plan")
        batch = WorldBatch(worlds, cfg)
        seed = self._streams(observation).derive("strategy", "rollout", observation["round"])
        current, workers = batch.replay_state(observation, seed)
        candidates = None
        if self.front_candidates:
            wv, mv = region_values(observation, batch, current, observation["manifesto"], cfg)
            candidates = candidate_allocations(observation, cfg)
            for a in front_media_candidates(observation, cfg, wv, mv, worker_counts=self.front_worker_counts,
                                            media_counts=self.front_media_counts,
                                            worker_regions=self.front_worker_regions):
                if a not in candidates:
                    candidates.append(a)
        return plan_round(observation, batch, observation["manifesto"], cfg, rollout=self.rollout, seed=seed,
                          current=current, current_workers=workers, candidates=candidates,
                          taper_ratio=self.taper_ratio)


#: The recommended smart player (S1, X6 worlds, 100 seeds, N = 10⁴): E 0.669 vs BayesPlanner 0.639, paired
#: +0.030 [0.027, 0.033]; each option helps in combination (reports/v3/experimentation/s1.json).
RECOMMENDED: dict[str, Any] = {"front_candidates": True, "rollout": "taper", "plan_world_fraction": 0.4,
                               "manifesto_method": "admm"}


def smart_player(**overrides: Any) -> SmartPlayer:
    """``SmartPlayer`` with :data:`RECOMMENDED` options; keyword ``overrides`` (e.g. ``seed``, ``device``) win."""
    return SmartPlayer(**{**RECOMMENDED, **overrides})
