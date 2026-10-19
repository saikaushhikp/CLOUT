"""Detached player observations with an explicit JSON-safe allowlist."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from .graph import GraphSummary


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    action_id: str
    cost: str
    target_requirement: str
    parameters: tuple[tuple[str, float], ...] = ()
    eligible_target_ids: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "cost": self.cost,
            "target_requirement": self.target_requirement,
            "parameters": {key: value for key, value in self.parameters},
            "eligible_target_ids": list(self.eligible_target_ids),
        }


@dataclass(frozen=True, slots=True)
class PublicEvent:
    sequence: int
    event_type: str
    round_index: int
    summary: str
    action_id: str | None = None
    error_code: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "round_index": self.round_index,
            "summary": self.summary,
            "action_id": self.action_id,
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    round_index: int
    remaining_budget: str
    aggregate_support: float
    graph_summary: GraphSummary
    legal_actions: tuple[ActionDescriptor, ...]
    terminal_status: bool
    public_events: tuple[PublicEvent, ...]
    schema_version: str = "1.0"

    def to_public_dict(self) -> dict[str, Any]:
        """Build a fresh, allowlisted tree of JSON primitives on every call."""
        summary = self.graph_summary
        return {
            "schema_version": self.schema_version,
            "round_index": self.round_index,
            "remaining_budget": self.remaining_budget,
            "aggregate_support": self.aggregate_support,
            "graph_summary": {
                "node_count": summary.node_count,
                "edge_count": summary.edge_count,
                "density": summary.density,
                "min_degree": summary.min_degree,
                "max_degree": summary.max_degree,
                "mean_degree": summary.mean_degree,
                "component_count": summary.component_count,
            },
            "legal_actions": [action.to_public_dict() for action in self.legal_actions],
            "terminal_status": self.terminal_status,
            "public_events": [event.to_public_dict() for event in self.public_events],
        }


def public_json(observation: Observation) -> str:
    """Stable JSON encoding for public observations only."""
    return json.dumps(observation.to_public_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
