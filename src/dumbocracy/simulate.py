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
from .dynamics.campaign import Allocation, CampaignOutcome, WorkerState, apply_assignments, campaign_round
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
    workers: WorkerState | None = None   # TRUE worker state (P8) — oracle information

    @property
    def team_by_region(self) -> tuple[int, int, int, int]:
        """Permanent workers bought so far per region (P8)."""
        return self.workers.team_by_region if self.workers is not None else (0, 0, 0, 0)


Policy = Callable[[RoundContext], Allocation]


@dataclass(eq=False)
class WorldOutcome:
    result: ElectionResult
    share_trajectory: list[float]           # expected share before round 1 and after every round
    spent: Decimal
    allocations: list[Allocation] = field(default_factory=list)
    final_preferences: torch.Tensor | None = None
    converted: list[int] = field(default_factory=list)   # voters converted per round (P8)
    spreaders: list[int] = field(default_factory=list)   # spreaders per round (P8)
    workers: WorkerState | None = None


def current_share(v: torch.Tensor, manifesto: torch.Tensor, config: GameConfig) -> float:
    prob = vote_probability(similarity(v, manifesto), config.voting.sensitivity, config.voting.threshold)
    return float(prob.mean())


def resolve_campaign_round(electorate: Electorate, config: GameConfig, v: torch.Tensor, manifestos: torch.Tensor,
                           outcomes: list[CampaignOutcome]) -> torch.Tensor:
    """Steps 1–4 tail of the P8 round order, shared with the engine: new workers and converts take
    ``v = b`` → social step on that v with media intensities, zealots frozen and workers pinned at b."""
    sus = electorate.susceptibility
    v = apply_assignments(v, outcomes, manifestos)
    aggregated = aggregate(v, electorate.graph, config.dynamics.social_rule, config.dynamics.wmsr_trim)
    intensity = torch.stack([o.intensity for o in outcomes])
    pinned = torch.stack([o.state.pinned for o in outcomes])
    return step_preferences(v, aggregated, config.dynamics.social_step * sus.social, intensity, manifestos,
                            sus.zealot, pinned=pinned)


def simulate_world(electorate: Electorate, config: GameConfig, manifesto: torch.Tensor, policy: Policy, *,
                   seeds: SeedStreams | None = None, player: str = "p0") -> WorldOutcome:
    """P8 round order: hires → cascade hop → media → social step (workers pinned) → next round."""
    seeds = seeds or config.seeds()
    device = electorate.device
    manifesto = manifesto.to(device=device, dtype=electorate.preferences.dtype)
    v = electorate.preferences.clone()
    budget = config.budget
    airings = [0, 0, 0, 0]
    trajectory = [current_share(v, manifesto, config)]
    allocations: list[Allocation] = []
    workers = WorkerState.empty(electorate.num_voters, device)
    converted, spreaders = [], []
    for t in range(config.rounds):
        ctx = RoundContext(t, config.rounds, budget, v, electorate, manifesto, tuple(airings), config, workers)
        allocation = policy(ctx)
        allocation.validate()
        generators = {name: seeds.generator(name, t, player, device=device) for name in ("workers", "cascade", "media")}
        outcome = campaign_round(electorate, allocation, workers, v, manifesto, tuple(airings), config.workers,
                                 config.media, config.voting, generators)
        if outcome.cost > budget:
            raise ValueError(f"policy overspent in round {t}: cost {outcome.cost} > budget {budget}")
        budget -= outcome.cost
        workers = outcome.state
        airings = [a + int(aired) for a, aired in zip(airings, allocation.media_regions)]
        v = resolve_campaign_round(electorate, config, v, manifesto.unsqueeze(0), [outcome])
        allocations.append(allocation)
        converted.append(outcome.converted)
        spreaders.append(outcome.spreaders)
        trajectory.append(current_share(v, manifesto, config))
    result = run_election(v, manifesto, config.voting.sensitivity, config.voting.threshold,
                          seeds.generator("election", player, device=device))
    return WorldOutcome(result, trajectory, config.budget - budget, allocations, v, converted, spreaders, workers)
