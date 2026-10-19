"""Transactional episode lifecycle integrating graph, MAS and campaign layers."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
import importlib.metadata
import hashlib
import json
import math
import platform
from typing import Iterable

import numpy as np

from .campaign import ActionContext, CampaignAction, ResourceLedger, commit_action, validate_action
from .config import GameConfig
from .errors import DumbocracyError, ErrorCode
from .graph import ElectorateGraph, GraphSummary, erdos_renyi_graph
from .information import ActionDescriptor, Observation, PublicEvent, public_json
from .mas import DeviceGraphView, VoterState, advance_support, prepare_device_graph, validate_voters


@dataclass(frozen=True, slots=True)
class InternalEvent:
    sequence: int
    event_type: str
    round_index: int
    attributes: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True, slots=True)
class TraceRecord:
    round_index: int
    action_id: str | None
    accepted: bool | None
    budget_nonnegative: bool
    state_valid: bool
    terminal: bool
    vote_share: float | None = None
    action_atomic: bool | None = None
    state_fingerprint_before: str | None = None
    state_fingerprint_after: str | None = None

    def predicates(self) -> dict[str, bool | None]:
        """Return named M0 predicates for later bounded-trace evaluators."""
        return {
            "Terminal": self.terminal,
            "Victory": (self.vote_share > 0.5) if self.terminal and self.vote_share is not None else (False if self.terminal else None),
            "BudgetNonnegative": self.budget_nonnegative,
            "StateValid": self.state_valid,
            "ActionAtomic": self.action_atomic,
        }


@dataclass(frozen=True, slots=True)
class GameResult:
    terminal_reason: str
    player_votes: int
    electorate_size: int
    vote_share: float
    victory: bool
    score: float
    budget_used: Decimal
    metadata: tuple[tuple[str, str], ...]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "terminal_reason": self.terminal_reason,
            "player_votes": self.player_votes,
            "electorate_size": self.electorate_size,
            "vote_share": self.vote_share,
            "victory": self.victory,
            "score": self.score,
            "budget_used": str(self.budget_used),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GameState:
    config: GameConfig
    graph: ElectorateGraph
    graph_summary: GraphSummary
    voters: tuple[VoterState, ...]
    initial_voters: tuple[VoterState, ...]
    round_index: int
    ledger: ResourceLedger
    current_contributions: tuple[float, ...]
    broadcast_used: bool
    terminal: bool
    result: GameResult | None
    internal_events: tuple[InternalEvent, ...]
    public_events: tuple[PublicEvent, ...]
    trace: tuple[TraceRecord, ...]
    backend: str = "cpu"
    device_graph: DeviceGraphView | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.round_index <= self.config.horizon_rounds:
            raise ValueError("round index is outside the configured horizon")
        validate_voters(self.graph, self.voters)
        validate_voters(self.graph, self.initial_voters)
        if len(self.current_contributions) != len(self.graph.node_ids) or any(not math.isfinite(value) for value in self.current_contributions):
            raise ValueError("current campaign contribution vector is malformed")
        if self.graph_summary.node_count != len(self.graph.node_ids):
            raise ValueError("cached graph summary does not match graph")
        if self.ledger.initial_budget != self.config.initial_budget:
            raise ValueError("ledger initial budget does not match game config")
        if self.backend.startswith("cuda") != (self.device_graph is not None):
            raise ValueError("device graph presence does not match selected backend")
        if self.terminal != (self.round_index == self.config.horizon_rounds):
            raise ValueError("terminal flag and round index disagree")
        if self.terminal != (self.result is not None):
            raise ValueError("terminal state must have exactly one result")


@dataclass(frozen=True, slots=True)
class StepResponse:
    accepted: bool
    error_code: str | None
    state: GameState
    observation: Observation
    terminal: bool
    action_id: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "error_code": self.error_code,
            "observation": self.observation.to_public_dict(),
            "terminal": self.terminal,
            "action_id": self.action_id,
            "result": self.state.result.to_public_dict() if self.accepted and self.state.result else None,
        }


class GameEngine:
    """Stateless orchestrator; each accepted operation returns a new snapshot."""

    def __init__(self, backend: str = "cpu", revision: str = "uncommitted") -> None:
        if backend != "cpu" and not backend.startswith("cuda"):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "backend must be 'cpu' or a CUDA device")
        if backend.startswith("cuda"):
            try:
                import torch
            except ImportError as exc:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "PyTorch is required for CUDA") from exc
            if not torch.cuda.is_available():
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "requested CUDA backend is unavailable")
            try:
                requested_device = torch.device(backend)
                if requested_device.type != "cuda" or (requested_device.index is not None and requested_device.index >= torch.cuda.device_count()):
                    raise ValueError("device index is not present")
            except (RuntimeError, ValueError) as exc:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "requested CUDA device is invalid") from exc
        self.backend = backend
        self.revision = revision

    def initialize(
        self,
        config: GameConfig,
        graph: ElectorateGraph | None = None,
        voters: Iterable[VoterState] | None = None,
    ) -> GameState:
        if not isinstance(config, GameConfig):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "config must be a validated GameConfig")
        if graph is None:
            if config.graph.kind != "erdos_renyi":
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "edge_list games require an explicit graph")
            graph = erdos_renyi_graph(config.graph.node_count, config.graph.edge_probability or 0.0, config.seeds.graph)
        if not isinstance(graph, ElectorateGraph):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "graph must be an ElectorateGraph")
        if len(graph.node_ids) != config.graph.node_count:
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "graph node count does not match config")
        graph_summary = graph.summary()
        if voters is None:
            rng = np.random.default_rng(config.seeds.initial_state)
            supports = rng.random(config.graph.node_count)
            susceptibilities = rng.random(config.graph.node_count)
            voters = tuple(VoterState(node_id, float(supports[i]), float(susceptibilities[i])) for i, node_id in enumerate(graph.node_ids))
        else:
            voters = validate_voters(graph, voters)
        device_graph = prepare_device_graph(graph, self.backend) if self.backend.startswith("cuda") else None
        ledger = ResourceLedger.create(config.initial_budget)
        init_event = InternalEvent(0, "game_initialized", 0, (("node_count", len(graph.node_ids)), ("edge_count", len(graph.edges))))
        voter_tuple = tuple(voters)
        state = GameState(config, graph, graph_summary, voter_tuple, voter_tuple, 0, ledger, tuple(0.0 for _ in graph.node_ids), False, False, None, (init_event,), (), (), self.backend, device_graph)
        obs_event = InternalEvent(1, "observation_generated", 0)
        state = replace(state, internal_events=state.internal_events + (obs_event,))
        return replace(state, public_events=self._public_events(state.internal_events))

    def legal_actions(self, state: GameState) -> tuple[ActionDescriptor, ...]:
        if state.terminal:
            return ()
        actions: list[ActionDescriptor] = []
        if state.ledger.remaining_budget >= state.config.actions.targeted_cost:
            actions.append(ActionDescriptor("targeted_contact", f"{state.config.actions.targeted_cost} per target", "one or more unique public voter IDs", (("delta_target", state.config.actions.delta_target),), state.graph.node_ids))
        if not state.broadcast_used and state.ledger.remaining_budget >= state.config.actions.broadcast_cost:
            actions.append(ActionDescriptor("broadcast", str(state.config.actions.broadcast_cost), "no target; at most once per round", (("delta_broadcast", state.config.actions.delta_broadcast),)))
        actions.append(ActionDescriptor("no_op", "0.00", "no target", ()))
        return tuple(actions)

    def observe(self, state: GameState) -> Observation:
        aggregate = math.fsum(voter.support for voter in state.voters) / len(state.voters)
        return Observation(
            round_index=state.round_index,
            remaining_budget=str(state.ledger.remaining_budget),
            aggregate_support=aggregate,
            graph_summary=state.graph_summary,
            legal_actions=self.legal_actions(state),
            terminal_status=state.terminal,
            public_events=state.public_events,
        )

    def result(self, state: GameState) -> GameResult | None:
        return state.result

    def is_terminal(self, state: GameState) -> bool:
        return state.terminal

    def step(self, state: GameState, action: CampaignAction) -> StepResponse:
        if state.terminal:
            return self._reject(state, action.action_id if isinstance(action, CampaignAction) else "unknown", ErrorCode.TERMINAL_STATE)
        if not isinstance(action, CampaignAction):
            return self._reject(state, "unknown", ErrorCode.INVALID_ACTION)
        proposal = InternalEvent(self._next_sequence(state), "action_proposed", state.round_index, (("action_id", action.action_id), ("targets", action.target_ids)))
        ctx = ActionContext(state.graph.node_ids, state.ledger, state.broadcast_used)
        validation = validate_action(ctx, action, state.config.actions)
        if not validation.accepted:
            rejected = proposal, InternalEvent(proposal.sequence + 1, "action_rejected", state.round_index, (("action_id", action.action_id), ("error_code", validation.error_code.value if validation.error_code else "INVALID_ACTION")))
            vote_share = state.result.vote_share if state.result else None
            fingerprint = self._state_fingerprint(state)
            rejected_trace = TraceRecord(state.round_index, action.action_id, False, state.ledger.remaining_budget >= 0, all(math.isfinite(v.support) and 0 <= v.support <= 1 for v in state.voters), state.terminal, vote_share, True, fingerprint, fingerprint)
            updated = replace(state, internal_events=state.internal_events + rejected, trace=state.trace + (rejected_trace,))
            updated = replace(updated, public_events=self._public_events(updated.internal_events))
            obs_event = InternalEvent(self._next_sequence(updated), "observation_generated", state.round_index)
            updated = replace(updated, internal_events=updated.internal_events + (obs_event,))
            updated = replace(updated, public_events=self._public_events(updated.internal_events))
            return StepResponse(False, validation.error_code.value if validation.error_code else ErrorCode.INVALID_ACTION.value, updated, self.observe(updated), False, self._public_action_id(action.action_id))

        try:
            committed = commit_action(ctx, validation)
            provisional = replace(state, ledger=committed.context.ledger, current_contributions=committed.contributions, broadcast_used=committed.context.broadcast_used)
            next_voters = advance_support(provisional.graph, provisional.voters, provisional.config.dynamics.alpha, provisional.current_contributions, self.backend, provisional.device_graph)
            next_round = state.round_index + 1
            terminal = next_round == state.config.horizon_rounds
            next_ledger = committed.context.ledger
            new_events = list(state.internal_events)
            seq = proposal.sequence
            new_events.append(proposal)
            seq += 1
            new_events.append(InternalEvent(seq, "action_accepted", state.round_index, (("action_id", action.action_id), ("cost", str(committed.cost)))))
            seq += 1
            if committed.cost > 0:
                new_events.append(InternalEvent(seq, "resource_spent", state.round_index, (("cost", str(committed.cost)),)))
                seq += 1
            new_events.append(InternalEvent(seq, "voter_update_summary", state.round_index, (("aggregate_support", math.fsum(v.support for v in next_voters) / len(next_voters)),)))
            seq += 1
            new_events.append(InternalEvent(seq, "round_completed", next_round))
            seq += 1
            result = self._make_result(state, next_voters, next_ledger) if terminal else None
            if terminal:
                new_events.append(InternalEvent(seq, "game_terminated", next_round, (("terminal_reason", "fixed_horizon"),)))
            trace = state.trace + (TraceRecord(next_round, action.action_id, True, next_ledger.remaining_budget >= 0, all(math.isfinite(v.support) and 0 <= v.support <= 1 for v in next_voters), terminal, result.vote_share if result else None),)
            candidate = GameState(state.config, state.graph, state.graph_summary, next_voters, state.initial_voters, next_round, next_ledger, tuple(0.0 for _ in state.graph.node_ids), False, terminal, result, tuple(new_events), (), trace, self.backend, state.device_graph)
            obs_event = InternalEvent(self._next_sequence(candidate), "observation_generated", next_round)
            candidate = replace(candidate, internal_events=candidate.internal_events + (obs_event,))
            candidate = replace(candidate, public_events=self._public_events(candidate.internal_events))
            return StepResponse(True, None, candidate, self.observe(candidate), terminal, action.action_id)
        except DumbocracyError as exc:
            if exc.code == ErrorCode.NUMERICAL_INSTABILITY:
                return self._reject(state, action.action_id, exc.code)
            raise

    def reset(self, state: GameState) -> GameState:
        graph = state.graph if state.config.graph.kind == "edge_list" else None
        return self.initialize(state.config, graph=graph, voters=state.initial_voters)

    def replay(self, config: GameConfig, actions: Iterable[CampaignAction], graph: ElectorateGraph | None = None) -> tuple[GameState, ...]:
        state = self.initialize(config, graph=graph)
        snapshots = [state]
        for action in actions:
            response = self.step(state, action)
            state = response.state
            snapshots.append(state)
        return tuple(snapshots)

    @staticmethod
    def _next_sequence(state: GameState) -> int:
        return state.internal_events[-1].sequence + 1 if state.internal_events else 0

    def _reject(self, state: GameState, action_id: str, code: ErrorCode) -> StepResponse:
        sequence = self._next_sequence(state)
        events = state.internal_events + (
            InternalEvent(sequence, "action_proposed", state.round_index, (("action_id", action_id),)),
            InternalEvent(sequence + 1, "action_rejected", state.round_index, (("action_id", action_id), ("error_code", code.value))),
        )
        rejected_state = replace(state, internal_events=events)
        fingerprint = self._state_fingerprint(state)
        rejected_state = replace(rejected_state, trace=state.trace + (TraceRecord(state.round_index, action_id, False, state.ledger.remaining_budget >= 0, all(math.isfinite(v.support) and 0 <= v.support <= 1 for v in state.voters), state.terminal, state.result.vote_share if state.result else None, True, fingerprint, fingerprint),))
        obs_event = InternalEvent(sequence + 2, "observation_generated", state.round_index)
        rejected_state = replace(rejected_state, internal_events=rejected_state.internal_events + (obs_event,))
        rejected_state = replace(rejected_state, public_events=self._public_events(rejected_state.internal_events))
        return StepResponse(False, code.value, rejected_state, self.observe(rejected_state), state.terminal, self._public_action_id(action_id))

    @staticmethod
    def _public_action_id(action_id: str) -> str:
        return action_id if action_id in ("targeted_contact", "broadcast", "no_op") else "unknown"

    @staticmethod
    def _state_fingerprint(state: GameState) -> str:
        canonical = {
            "round": state.round_index,
            "ledger": [str(state.ledger.initial_budget), str(state.ledger.remaining_budget), str(state.ledger.spent)],
            "broadcast_used": state.broadcast_used,
            "current_contributions": state.current_contributions,
            "terminal": state.terminal,
            "accepted_actions": [
                [event.round_index, dict(event.attributes).get("action_id"), dict(event.attributes).get("cost")]
                for event in state.internal_events if event.event_type == "action_accepted"
            ],
            "voters": [[v.voter_id, v.support, v.susceptibility, v.campaign_exposure] for v in state.voters],
            "result": state.result.to_public_dict() if state.result else None,
        }
        return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    @staticmethod
    def _public_events(events: tuple[InternalEvent, ...] | list[InternalEvent]) -> tuple[PublicEvent, ...]:
        safe: list[PublicEvent] = []
        summaries = {
            "game_initialized": "Game initialized.",
            "action_proposed": "Action submitted.",
            "action_accepted": "Action accepted.",
            "action_rejected": "Action rejected.",
            "resource_spent": "Campaign resources spent.",
            "voter_update_summary": "Voter update completed.",
            "round_completed": "Round completed.",
            "game_terminated": "Game completed at the fixed horizon.",
            "observation_generated": "Player observation generated.",
        }
        for event in events:
            if event.event_type == "action_rejected":
                attrs = dict(event.attributes)
                action_id = GameEngine._public_action_id(str(attrs.get("action_id", "")))
                safe.append(PublicEvent(event.sequence, event.event_type, event.round_index, summaries[event.event_type], action_id, str(attrs.get("error_code", "INVALID_ACTION"))))
            elif event.event_type in summaries:
                attrs = dict(event.attributes)
                action_id = str(attrs["action_id"]) if event.event_type == "action_accepted" and "action_id" in attrs else None
                safe.append(PublicEvent(event.sequence, event.event_type, event.round_index, summaries[event.event_type], action_id))
        return tuple(safe)

    def _make_result(self, state: GameState, voters: tuple[VoterState, ...], ledger: ResourceLedger) -> GameResult:
        votes = sum(voter.support > state.config.support_threshold for voter in voters)
        share = votes / len(voters)
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = "not-installed"
        device_name = "CPU"
        cuda_version = "not-applicable"
        if self.backend.startswith("cuda"):
            import torch
            device = torch.device(self.backend)
            device_name = torch.cuda.get_device_name(device)
            cuda_version = str(torch.version.cuda)
        config_json = json.dumps({
            "graph_kind": state.config.graph.kind,
            "node_count": state.config.graph.node_count,
            "edge_probability": state.config.graph.edge_probability,
            "horizon_rounds": state.config.horizon_rounds,
            "initial_budget": str(state.config.initial_budget),
            "support_threshold": state.config.support_threshold,
            "alpha": state.config.dynamics.alpha,
            "epsilon_mode": state.config.dynamics.epsilon_mode,
            "targeted_cost": str(state.config.actions.targeted_cost),
            "broadcast_cost": str(state.config.actions.broadcast_cost),
            "delta_target": state.config.actions.delta_target,
            "delta_broadcast": state.config.actions.delta_broadcast,
        }, sort_keys=True, separators=(",", ":"))
        graph_summary = state.graph_summary
        metadata = (
            ("root_seed", str(state.config.seeds.root_seed)),
            ("graph_seed", str(state.config.seeds.graph)),
            ("initial_state_seed", str(state.config.seeds.initial_state)),
            ("transition_seed", str(state.config.seeds.transition)),
            ("strategy_seed", str(state.config.seeds.strategy)),
            ("python", platform.python_version()),
            ("numpy", importlib.metadata.version("numpy")),
            ("torch", torch_version),
            ("cuda_runtime", cuda_version),
            ("backend", self.backend),
            ("device_name", device_name),
            ("revision", self.revision),
            ("config", config_json),
            ("graph_nodes", str(graph_summary.node_count)),
            ("graph_edges", str(graph_summary.edge_count)),
            ("graph_density", str(graph_summary.density)),
        )
        return GameResult("fixed_horizon", votes, len(voters), share, share > 0.5, share, ledger.spent, metadata)
