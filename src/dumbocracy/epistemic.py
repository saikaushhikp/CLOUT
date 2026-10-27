"""Bounded exact branching checks over explicitly supplied finite engine worlds."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .campaign import CampaignAction
from .engine import GameEngine, GameState
from .information import Observation


class CheckStatus(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class EpistemicResult:
    possible_victory: CheckStatus
    guaranteed_victory: CheckStatus
    inevitable_defeat: CheckStatus
    knows_guaranteed_victory: CheckStatus
    knows_inevitable_defeat: CheckStatus
    world_count: int
    action_abstraction: str
    horizon: int
    explored_nodes: int
    node_limit: int
    horizon_limit: int
    world_limit: int
    scope: str = "exact only over supplied finite worlds and enumerated actions"

    def to_public_dict(self) -> dict[str, object]:
        return {"possible_victory": self.possible_victory.value,
                "guaranteed_victory": self.guaranteed_victory.value,
                "inevitable_defeat": self.inevitable_defeat.value,
                "knows_guaranteed_victory": self.knows_guaranteed_victory.value,
                "knows_inevitable_defeat": self.knows_inevitable_defeat.value,
                "world_count": self.world_count, "action_abstraction": self.action_abstraction,
                "horizon": self.horizon, "explored_nodes": self.explored_nodes,
                "node_limit": self.node_limit, "horizon_limit": self.horizon_limit,
                "world_limit": self.world_limit, "scope": self.scope, "evaluation_mode": "finite_model"}


def observation_key(obs: Observation) -> tuple[object, ...]:
    """The observation allowlist defines indistinguishability; no hidden fields."""
    return (obs.schema_version, obs.round_index, obs.remaining_budget, obs.aggregate_support,
            obs.graph_summary, obs.terminal_status,
            tuple((a.action_id, a.cost, a.target_requirement, a.parameters, a.eligible_target_ids) for a in obs.legal_actions),
            tuple((e.event_type, e.round_index, e.action_id, e.error_code) for e in obs.public_events))


def check_finite_worlds(engine: GameEngine, worlds: tuple[GameState, ...], *, node_limit: int = 10000,
                        target_cap: int = 8, horizon_limit: int = 128,
                        world_limit: int = 256) -> EpistemicResult:
    if not worlds or any(not isinstance(w, GameState) for w in worlds):
        raise ValueError("worlds must be a nonempty tuple of GameState values")
    if isinstance(node_limit, bool) or not isinstance(node_limit, int) or node_limit < 1:
        raise ValueError("node_limit must be a positive integer")
    if len({w.round_index for w in worlds}) != 1 or len({w.config.horizon_rounds for w in worlds}) != 1:
        raise ValueError("worlds must share round and horizon")
    if isinstance(target_cap, bool) or not isinstance(target_cap, int) or target_cap < 0:
        raise ValueError("target_cap must be a nonnegative integer")
    if isinstance(horizon_limit, bool) or not isinstance(horizon_limit, int) or horizon_limit < 1:
        raise ValueError("horizon_limit must be a positive integer")
    if isinstance(world_limit, bool) or not isinstance(world_limit, int) or world_limit < 1:
        raise ValueError("world_limit must be a positive integer")
    abstraction = f"broadcast/no_op; first {target_cap} lexicographic singleton targets"
    horizon = worlds[0].config.horizon_rounds
    if horizon > horizon_limit or len(worlds) > world_limit:
        return EpistemicResult(CheckStatus.INCOMPLETE, CheckStatus.INCOMPLETE,
                               CheckStatus.INCOMPLETE, CheckStatus.INCOMPLETE,
                               CheckStatus.INCOMPLETE, len(worlds), abstraction,
                               horizon, 0, node_limit, horizon_limit, world_limit)
    limit = node_limit
    explored = 0
    incomplete = False

    def actions(state: GameState) -> dict[str, CampaignAction]:
        result = {}
        for descriptor in engine.legal_actions(state):
            if descriptor.action_id == "targeted_contact":
                for target in sorted(descriptor.eligible_target_ids)[:target_cap]:
                    result[f"targeted_contact:{target}"] = CampaignAction("targeted_contact", (target,))
            else:
                result[descriptor.action_id] = CampaignAction(descriptor.action_id)
        return result

    def terminal_win(states: tuple[GameState, ...]) -> bool:
        return all(s.result is not None and s.result.victory for s in states)

    def solve(belief: tuple[GameState, ...]) -> bool:
        nonlocal explored, incomplete
        if explored >= limit:
            incomplete = True
            return False
        explored += 1
        if all(s.terminal for s in belief):
            return terminal_win(belief)
        options = [actions(s) for s in belief]
        common = set(options[0]).intersection(*(set(o) for o in options[1:]))
        for action_id in sorted(common):
            successors = tuple(engine.step(s, options[i][action_id]).state for i, s in enumerate(belief))
            groups: dict[tuple[object, ...], list[GameState]] = {}
            for state in successors:
                groups.setdefault(observation_key(engine.observe(state)), []).append(state)
            if all(solve(tuple(group)) for group in groups.values()):
                return True
        return False

    possible_values = []
    for state in worlds:
        remaining = limit - explored
        if remaining <= 0:
            possible_values.append((False, False))
            incomplete = True
            continue
        value, complete, work = _omniscient_can_win(engine, state, actions, remaining)
        explored += work
        possible_values.append((value, complete))
        if not complete:
            incomplete = True
    possible = any(value for value, complete in possible_values)
    if any(not complete for value, complete in possible_values):
        incomplete = True
    initial_groups: dict[tuple[object, ...], list[GameState]] = {}
    for world in worlds:
        initial_groups.setdefault(observation_key(engine.observe(world)), []).append(world)
    guaranteed = all(solve(tuple(group)) for group in initial_groups.values())
    inevitable = all(not value for value, complete in possible_values) and not incomplete
    status = lambda value: CheckStatus.INCOMPLETE if incomplete else CheckStatus.TRUE if value else CheckStatus.FALSE
    return EpistemicResult(status(possible), status(guaranteed), status(inevitable),
                           status(guaranteed), status(inevitable), len(worlds),
                           abstraction, horizon, explored, limit, horizon_limit, world_limit)


def _omniscient_can_win(engine: GameEngine, initial: GameState, action_fn, node_limit: int) -> tuple[bool, bool, int]:
    seen = 0
    complete = True
    def visit(state: GameState) -> bool:
        nonlocal seen, complete
        if seen >= node_limit:
            complete = False
            return False
        seen += 1
        if state.terminal:
            return bool(state.result and state.result.victory)
        return any(visit(engine.step(state, action).state) for action in action_fn(state).values())
    result = visit(initial)
    return result, complete, seen
