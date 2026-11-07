"""Single-world campaign simulation used by calibration, planners and experiments (spec §3–§5, §9).

This is the campaign → social dynamics → election loop *without* the player-facing engine
(phases, surveys, observations — P3). Policies here receive an oracle ``RoundContext`` that includes
the true preference state; only calibration tools and labelled ORACLE baselines may use it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable

import torch

from .config import GameConfig, SeedStreams
from .dynamics.campaign import Allocation, campaign_round
from .dynamics.social import aggregate
from .dynamics.update import step_preferences
from .electorate import Electorate
from .voting import ElectionResult, run_election, similarity, vote_probability


@dataclass(frozen=True, eq=False)
class RoundContext:
    round: int                     # 0-based campaign round
    rounds: int
    budget: Decimal                # remaining treasury
    preferences: torch.Tensor      # TRUE v(t) — oracle information
    electorate: Electorate
    manifesto: torch.Tensor
    prior_airings: tuple[int, int, int, int]
    config: GameConfig


Policy = Callable[[RoundContext], Allocation]


@dataclass(eq=False)
class WorldOutcome:
    result: ElectionResult
    share_trajectory: list[float]           # expected share before round 1 and after every round
    spent: Decimal
    allocations: list[Allocation] = field(default_factory=list)
    final_preferences: torch.Tensor | None = None


def current_share(v: torch.Tensor, manifesto: torch.Tensor, config: GameConfig) -> float:
    prob = vote_probability(similarity(v, manifesto), config.voting.sensitivity, config.voting.threshold)
    return float(prob.mean())


def simulate_world(electorate: Electorate, config: GameConfig, manifesto: torch.Tensor, policy: Policy, *,
                   seeds: SeedStreams | None = None, player: str = "p0") -> WorldOutcome:
    seeds = seeds or config.seeds()
    device = electorate.device
    manifesto = manifesto.to(device=device, dtype=electorate.preferences.dtype)
    v = electorate.preferences.clone()
    sus = electorate.susceptibility
    social_weight = config.dynamics.social_step * sus.social
    budget = config.budget
    airings = [0, 0, 0, 0]
    trajectory = [current_share(v, manifesto, config)]
    allocations: list[Allocation] = []
    for t in range(config.rounds):
        ctx = RoundContext(t, config.rounds, budget, v, electorate, manifesto, tuple(airings), config)
        allocation = policy(ctx)
        allocation.validate(config.workers.free_workers)
        generators = {name: seeds.generator(name, t, player, device=device) for name in ("workers", "cascade", "media")}
        outcome = campaign_round(electorate, allocation, tuple(airings), config.workers, config.media, generators)
        if outcome.cost > budget:
            raise ValueError(f"policy overspent in round {t}: cost {outcome.cost} > budget {budget}")
        budget -= outcome.cost
        airings = [a + int(aired) for a, aired in zip(airings, allocation.media_regions)]
        aggregated = aggregate(v, electorate.graph, config.dynamics.social_rule, config.dynamics.wmsr_trim)
        v = step_preferences(v, aggregated, social_weight, outcome.intensity.unsqueeze(0), manifesto.unsqueeze(0), sus.zealot)
        allocations.append(allocation)
        trajectory.append(current_share(v, manifesto, config))
    result = run_election(v, manifesto, config.voting.sensitivity, config.voting.threshold,
                          seeds.generator("election", player, device=device))
    return WorldOutcome(result, trajectory, config.budget - budget, allocations, v)
