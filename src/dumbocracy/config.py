"""Immutable validated configuration and stable subsystem seed derivation."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import math
from typing import Literal

from .errors import DumbocracyError, ErrorCode


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, f"{name} must be finite")
    return float(value)


def _decimal(value: Decimal, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < Decimal("0"):
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, f"{name} must be a finite nonnegative Decimal")
    return value


@dataclass(frozen=True, slots=True)
class SeedConfig:
    root_seed: int

    def __post_init__(self) -> None:
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "root_seed must be an integer")

    def derive(self, subsystem: Literal["graph", "initial_state", "transition", "strategy"]) -> int:
        """Return a stable 64-bit seed without Python's randomized hash()."""
        digest = hashlib.blake2b(
            f"dumbocracy-v1:{self.root_seed}:{subsystem}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big", signed=False)

    @property
    def graph(self) -> int:
        return self.derive("graph")

    @property
    def initial_state(self) -> int:
        return self.derive("initial_state")

    @property
    def transition(self) -> int:
        return self.derive("transition")

    @property
    def strategy(self) -> int:
        return self.derive("strategy")


@dataclass(frozen=True, slots=True)
class GraphSpec:
    kind: Literal["edge_list", "erdos_renyi"]
    node_count: int
    edge_probability: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("edge_list", "erdos_renyi"):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "unsupported graph kind")
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int) or self.node_count < 1:
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "node_count must be an integer >= 1")
        if self.kind == "erdos_renyi":
            if self.edge_probability is None or not 0 <= _finite_number(self.edge_probability, "edge_probability") <= 1:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "edge_probability must be in [0, 1]")
        elif self.edge_probability is not None:
            _finite_number(self.edge_probability, "edge_probability")


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    alpha: float
    epsilon_mode: Literal["none"] = "none"

    def __post_init__(self) -> None:
        if not 0 <= _finite_number(self.alpha, "alpha") <= 1:
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "alpha must be in [0, 1]")
        if self.epsilon_mode != "none":
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "v1 transition noise mode must be 'none'")


@dataclass(frozen=True, slots=True)
class ActionConfig:
    delta_target: float
    delta_broadcast: float
    targeted_cost: Decimal = Decimal("1.00")
    broadcast_cost: Decimal = Decimal("5.00")

    def __post_init__(self) -> None:
        _finite_number(self.delta_target, "delta_target")
        _finite_number(self.delta_broadcast, "delta_broadcast")
        _decimal(self.targeted_cost, "targeted_cost")
        _decimal(self.broadcast_cost, "broadcast_cost")


@dataclass(frozen=True, slots=True)
class GameConfig:
    seeds: SeedConfig
    graph: GraphSpec
    horizon_rounds: int
    initial_budget: Decimal
    dynamics: DynamicsConfig
    actions: ActionConfig
    support_threshold: float = 0.5

    def __post_init__(self) -> None:
        if isinstance(self.horizon_rounds, bool) or not isinstance(self.horizon_rounds, int) or self.horizon_rounds < 1:
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "horizon_rounds must be an integer >= 1")
        _decimal(self.initial_budget, "initial_budget")
        if _finite_number(self.support_threshold, "support_threshold") != 0.5:
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "v1 support_threshold must equal 0.5")
