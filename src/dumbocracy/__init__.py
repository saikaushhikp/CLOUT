"""Dumbocracy engine, public-information, logic, and strategy APIs."""

from .config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from .graph import ElectorateGraph, GraphSummary, erdos_renyi_graph
from .mas import VoterState, advance_support, player_vote_share
from .engine import GameEngine, GameResult, GameState, StepResponse
from .information import ActionDescriptor, Observation
from .logic import (
    And, Atom, CanonicalTrace, Eventually, Globally, Implies, LogicInputError,
    LogicResult, LogicStatus, Not, Or, TraceAuditEvent, TraceState,
    evaluate_diagnostics, evaluate_trace,
)
from .strategies import (
    FirstLegalStrategy, HistoryEntry, NoOpStrategy, PublicActionSummary,
    SeededRandomLegalStrategy, Strategy, StrategyError,
)

__all__ = [
    "ActionConfig", "DynamicsConfig", "GameConfig", "GraphSpec", "SeedConfig",
    "ElectorateGraph", "GraphSummary", "VoterState", "advance_support",
    "erdos_renyi_graph", "player_vote_share",
    "GameEngine", "GameResult", "GameState", "StepResponse",
    "ActionDescriptor", "Observation",
    "CanonicalTrace", "LogicResult", "LogicStatus", "evaluate_trace", "evaluate_diagnostics",
    "TraceState", "TraceAuditEvent", "LogicInputError", "Atom", "And", "Or", "Not", "Implies", "Globally", "Eventually",
    "Strategy", "StrategyError", "HistoryEntry", "PublicActionSummary", "NoOpStrategy", "FirstLegalStrategy", "SeededRandomLegalStrategy",
]
