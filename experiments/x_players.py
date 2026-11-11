"""Observation-only players for the v3 experiments (plan §5 X6; not in the library).

Every player here implements the library ``Player`` protocol: ``act(observation) -> action dict``, using the
allowlisted observation only (never the engine state), exactly like ``strategies.players``.

* :class:`PeoplePleaser`  (player I)  — BayesPlanner's survey and posterior; manifesto = "blend my hunches" (R20)
  with the hunch = the posterior MAP archetype per region, over the 1–2 largest regions that hold a majority
  (3 if two are not enough); allocation = even pace, media in the coalition (largest first) within half the
  cap, the rest on workers in the coalition by size (the RoundContext rule ``x_policies.PeoplePleaser`` on
  public facts only).
* :class:`StrongArm`      (player III) — no queries; a fixed or random manifesto
  (``x_policies.strong_arm_manifesto``); the "split" pattern at 1.5× the even pace (half media on the largest
  regions, the rest on workers by size), so the money runs out early.
* :class:`AdmmBayesPlanner` — BayesPlanner with the consensus-ADMM manifesto (``manifesto_opt.admm_manifesto``) on
  the same anticipated posterior worlds; it records the primal and dual residuals of every solve.
* :class:`BlendBayesPlanner` — BayesPlanner with the "blend my hunches" manifesto of player I (manifesto-optimiser
  arm: PG vs ADMM vs blend, all with the rollout planner).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal
from typing import Any

import torch

from dumbocracy.config import GameConfig
from dumbocracy.dynamics.campaign import Allocation, media_cost
from dumbocracy.sectors import NUM_SECTORS, archetype_means
from dumbocracy.strategies.manifesto_opt import admm_manifesto, project_simplex
from dumbocracy.strategies.oracle import anticipate
from dumbocracy.strategies.planner import _media_set, round_cap, rounds_left, workers_for
from dumbocracy.strategies.players import BayesPlanner, _Timed, _treasury, allocate_dict, public_config, simplex_list
from dumbocracy.strategies.worlds import public_sizes

CENT = Decimal("0.01")


def majority_coalition(sizes: list[int]) -> list[int]:
    """The 1–2 largest regions (largest first) whose voters exceed half of N; 3 if two are not enough
    (same rule as ``x_policies.majority_regions``)."""
    order = sorted(range(4), key=lambda r: (-sizes[r], r))
    total, chosen = sum(sizes), []
    for r in order:
        chosen.append(r)
        if 2 * sum(sizes[i] for i in chosen) > total:
            break
    return chosen


def blend_hunches(sigma: list[int] | torch.Tensor, sizes: list[int], regions: list[int]) -> list[float]:
    """R20 "blend my hunches": b = Σ_r n_r μ_σ(r) / Σ_r n_r over ``regions`` (unrounded)."""
    mu = archetype_means(dtype=torch.float64)
    sigma = [int(x) for x in (sigma.tolist() if isinstance(sigma, torch.Tensor) else sigma)]
    b = sum(sizes[r] * mu[sigma[r]] for r in regions) / sum(sizes[r] for r in regions)
    return simplex_list(b)


def coalition_allocation(observation: dict[str, Any], config: GameConfig, regions: list[int], *, pace: float = 1.0,
                         media_share: float = 0.5) -> Allocation:
    """``pace`` × even cap; media in ``regions`` (in that order) within ``media_share`` of the cap; the rest buys
    workers in ``regions`` by size."""
    sizes = public_sizes(observation)
    treasury = _treasury(observation)
    cap = min(treasury, (round_cap(treasury, rounds_left(observation)) * Decimal(str(pace))).quantize(CENT, ROUND_FLOOR))
    media = _media_set(sizes, regions, len(regions), (cap * Decimal(str(media_share))).quantize(CENT, ROUND_FLOOR),
                       config.media.base_cost)
    left = cap - media_cost(sizes, media, config.media.base_cost)
    weights = [float(sizes[r]) if r in regions else 0.0 for r in range(4)]
    return Allocation(workers_for(left, weights, config), media)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------ player I
@dataclass
class PeoplePleaser(BayesPlanner):
    """Player I (see module docstring). Survey behaviour is BayesPlanner's (inherited)."""

    pace: float = 1.0
    media_share: float = 0.5

    def manifesto(self, observation: dict[str, Any], posterior=None) -> list[float]:
        posterior = posterior or self._posterior(observation)
        sizes = public_sizes(observation)
        return blend_hunches(posterior.map_assignment(), sizes, majority_coalition(sizes))

    def allocation(self, observation: dict[str, Any], posterior=None):
        cfg = public_config(observation, self.config)
        sizes = public_sizes(observation)
        a = coalition_allocation(observation, cfg, majority_coalition(sizes), pace=self.pace, media_share=self.media_share)
        return a, {"best_score": float("nan"), "num_candidates": 1, "seconds": 0.0}


@dataclass
class BlendBayesPlanner(BayesPlanner):
    """BayesPlanner (survey + rollout planner) with player I's blend manifesto."""

    def manifesto(self, observation: dict[str, Any], posterior=None) -> list[float]:
        posterior = posterior or self._posterior(observation)
        sizes = public_sizes(observation)
        return blend_hunches(posterior.map_assignment(), sizes, majority_coalition(sizes))


@dataclass
class AdmmBayesPlanner(BayesPlanner):
    """BayesPlanner with the consensus-ADMM manifesto on the anticipated posterior worlds (regions = blocks).
    ``admm_records`` keeps the primal/dual residual traces of every solve."""

    admm_rho: float = 5.0
    admm_iterations: int = 300
    admm_records: list[dict[str, Any]] = field(default_factory=list)

    def manifesto(self, observation: dict[str, Any], posterior=None) -> list[float]:
        posterior = posterior or self._posterior(observation)
        cfg, worlds = self._worlds(observation, posterior, self.manifesto_worlds, "manifesto")
        v_t = torch.cat([anticipate(w, cfg) for w in worlds])
        region_of = torch.cat([w.region_of for w in worlds]).to(v_t.device)
        start = time.perf_counter()
        sol = admm_manifesto(v_t, region_of, cfg.voting.sensitivity, cfg.voting.threshold, rho=self.admm_rho,
                             iterations=self.admm_iterations)
        self.admm_records.append({"iterations": sol.iterations, "share_on_worlds": sol.share,
                                  "primal_final": sol.primal_residuals[-1], "dual_final": sol.dual_residuals[-1],
                                  "primal": [round(x, 8) for x in sol.primal_residuals],
                                  "dual": [round(x, 8) for x in sol.dual_residuals],
                                  "seconds": time.perf_counter() - start})
        return simplex_list(sol.manifesto)


# ------------------------------------------------------------------------------------------ player III
@dataclass
class StrongArm(_Timed):
    """Player III: no queries, a fixed (0.6 on ``sector``, 0.1 elsewhere) or random (flat Dirichlet keyed by
    ``seed``) manifesto, the split pattern at ``pace`` × the even cap (largest regions first for media)."""

    player_id: str = "p0"
    seed: int = 0
    kind: str = "fixed"
    sector: int = 0
    pace: float = 1.5
    config: GameConfig | None = None
    latency: dict[str, list[float]] = field(default_factory=dict)

    def _manifesto(self) -> list[float]:
        if self.kind == "fixed":
            b = torch.full((NUM_SECTORS,), 0.1, dtype=torch.float64)
            b[self.sector] = 0.6
        elif self.kind == "random":
            g = torch.Generator().manual_seed(int(self.seed))
            x = -torch.log(torch.rand(NUM_SECTORS, generator=g, dtype=torch.float64))
            b = x / x.sum()
        else:
            raise ValueError("kind must be 'fixed' or 'random'")
        return simplex_list(project_simplex(b))

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        if observation["phase"] == "SURVEY":
            action = {"type": "commit", "manifesto": self._manifesto()}
        else:
            cfg = public_config(observation, self.config)
            sizes = public_sizes(observation)
            order = sorted(range(4), key=lambda r: (-sizes[r], r))
            action = allocate_dict(coalition_allocation(observation, cfg, order, pace=self.pace, media_share=0.5))
        self._record(action["type"], time.perf_counter() - start)
        return action
