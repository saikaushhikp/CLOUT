"""Player actions and PURE validators with stable error codes (spec §7; web_api_v2.md "Errors").

Validators never mutate anything: they return ``None`` (valid) or a :class:`Rejection`. The engine
applies an action only after its validator passed, so rejected actions change nothing (atomic).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Union

from ..dynamics.campaign import media_cost, worker_cost
from ..sectors import NUM_REGIONS, NUM_SECTORS, REGIONS, SECTORS

if TYPE_CHECKING:  # pragma: no cover
    from .state import GameState, PlayerState


class ErrorCode:
    """Stable rejection codes (contract + spec §7)."""

    INVALID_ACTION = "INVALID_ACTION"
    INVALID_PHASE = "INVALID_PHASE"
    INVALID_MANIFESTO = "INVALID_MANIFESTO"
    INVALID_ALLOCATION = "INVALID_ALLOCATION"
    INVALID_QUERY = "INVALID_QUERY"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    DUPLICATE_QUERY = "DUPLICATE_QUERY"
    UNKNOWN_PLAYER = "UNKNOWN_PLAYER"
    TERMINAL_STATE = "TERMINAL_STATE"


MANIFESTO_TOLERANCE = 1e-6  # |Σb − 1| (spec §7)


@dataclass(frozen=True, slots=True)
class Rejection:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class Query:
    """Paid exact query (spec §6.3): % of ``region`` voters with v[sector_b] > v[sector_a]."""

    sector_a: str
    sector_b: str
    region: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.sector_a, self.sector_b, self.region)


@dataclass(frozen=True, slots=True)
class Commit:
    """Fix the manifesto b ∈ Δ (spec §7)."""

    manifesto: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Allocate:
    """One round's campaign allocation (spec §3, §7)."""

    workers_by_region: tuple[int, ...]
    extra_workers: int = 0
    media_regions: tuple[bool, ...] = (False,) * NUM_REGIONS


@dataclass(frozen=True, slots=True)
class Forfeit:
    """Timer expired: equivalent to allocating nothing (spec §7)."""


Action = Union[Query, Commit, Allocate, Forfeit]
ACTION_TYPES: dict[str, type] = {"query": Query, "commit": Commit, "allocate": Allocate, "forfeit": Forfeit}


def action_type(action: Action) -> str:
    return {Query: "query", Commit: "commit", Allocate: "allocate", Forfeit: "forfeit"}[type(action)]


def action_from_dict(data: Mapping[str, Any]) -> Action:
    """Parse an action dict (``{"type": "query", ...}``) as produced by a :class:`Player`.

    Raises ``ValueError`` on an unknown type or missing/extra fields; value validation is left to
    the validators so that it yields the stable error codes.
    """
    if not isinstance(data, Mapping) or data.get("type") not in ACTION_TYPES:
        raise ValueError(f"unknown action type {data.get('type') if isinstance(data, Mapping) else data!r}")
    kind = data["type"]
    fields = {k: v for k, v in data.items() if k != "type"}
    if kind == "commit":
        raw = fields["manifesto"]
        return Commit(tuple(raw) if isinstance(raw, (list, tuple)) else raw)
    if kind == "allocate":
        return Allocate(tuple(fields["workers_by_region"]), fields.get("extra_workers", 0),
                        tuple(fields.get("media_regions", (False,) * NUM_REGIONS)))
    return ACTION_TYPES[kind](**fields)


def action_to_dict(action: Action) -> dict[str, Any]:
    """JSON-safe inverse of :func:`action_from_dict`."""
    if isinstance(action, Query):
        return {"type": "query", "sector_a": action.sector_a, "sector_b": action.sector_b, "region": action.region}
    if isinstance(action, Commit):
        return {"type": "commit", "manifesto": [float(x) for x in action.manifesto]}
    if isinstance(action, Allocate):
        return {"type": "allocate", "workers_by_region": list(action.workers_by_region),
                "extra_workers": action.extra_workers, "media_regions": list(action.media_regions)}
    return {"type": "forfeit"}


# ----------------------------------------------------------------------------------- helpers

def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def allocation_cost(action: Allocate, sizes: Sequence[int], extra_worker_cost: Decimal, media_base_cost: Decimal) -> Decimal:
    """``extra × cost_w + Σ media price over aired regions`` (spec §3.1–§3.2), exact Decimal."""
    return worker_cost(action.extra_workers, extra_worker_cost) + media_cost(list(sizes), list(action.media_regions), media_base_cost)


def _phase_guard(state: "GameState", expected: tuple[str, ...]) -> Rejection | None:
    if state.phase == "DONE":
        return Rejection(ErrorCode.TERMINAL_STATE, "the game is over")
    if state.phase not in expected:
        return Rejection(ErrorCode.INVALID_PHASE, f"not allowed in phase {state.phase}")
    return None


# -------------------------------------------------------------------------------- validators

def validate_query(state: "GameState", player: "PlayerState", action: Query) -> Rejection | None:
    """Spec §6.3: SURVEY or ROUND; distinct known sectors, known region; not repeated this
    phase/round; affordable."""
    rejection = _phase_guard(state, ("SURVEY", "ROUND"))
    if rejection:
        return rejection
    if action.sector_a not in SECTORS or action.sector_b not in SECTORS:
        return Rejection(ErrorCode.INVALID_QUERY, f"sectors must be in {list(SECTORS)}")
    if action.region not in REGIONS:
        return Rejection(ErrorCode.INVALID_QUERY, f"region must be in {list(REGIONS)}")
    if action.sector_a == action.sector_b:
        return Rejection(ErrorCode.INVALID_QUERY, "sector_a and sector_b must differ")
    if action.key in player.query_keys:
        return Rejection(ErrorCode.DUPLICATE_QUERY, "identical query already asked this phase/round")
    if not player.ledger.can_afford(state.config.survey.query_cost):
        return Rejection(ErrorCode.INSUFFICIENT_FUNDS, "not enough money for a query")
    return None


def validate_commit(state: "GameState", player: "PlayerState", action: Commit) -> Rejection | None:
    """Spec §7: SURVEY only, once; b has 5 finite entries ≥ 0 summing to 1 ± 1e-6."""
    rejection = _phase_guard(state, ("SURVEY",))
    if rejection:
        return rejection
    if player.manifesto is not None:
        return Rejection(ErrorCode.INVALID_PHASE, "manifesto already committed")
    b = action.manifesto
    if not isinstance(b, tuple) or len(b) != NUM_SECTORS or not all(_is_number(x) for x in b):
        return Rejection(ErrorCode.INVALID_MANIFESTO, f"manifesto must be {NUM_SECTORS} finite numbers")
    if any(x < 0 for x in b):
        return Rejection(ErrorCode.INVALID_MANIFESTO, "manifesto entries must be >= 0")
    if abs(math.fsum(b) - 1.0) > MANIFESTO_TOLERANCE:
        return Rejection(ErrorCode.INVALID_MANIFESTO, "manifesto must sum to 1 (± 1e-6)")
    return None


def _round_guard(state: "GameState", player: "PlayerState") -> Rejection | None:
    rejection = _phase_guard(state, ("ROUND",))
    if rejection:
        return rejection
    if player.ready:
        return Rejection(ErrorCode.INVALID_PHASE, "already allocated this round; waiting for other players")
    return None


def validate_allocate(state: "GameState", player: "PlayerState", action: Allocate) -> Rejection | None:
    """Spec §3, §7 + contract: 4 non-negative int worker counts summing ≤ free + extra, extra ≥ 0,
    4 boolean media flags; cost ≤ treasury."""
    rejection = _round_guard(state, player)
    if rejection:
        return rejection
    workers, media = action.workers_by_region, action.media_regions
    if not isinstance(workers, tuple) or len(workers) != NUM_REGIONS or not all(_is_int(w) and w >= 0 for w in workers):
        return Rejection(ErrorCode.INVALID_ALLOCATION, "workers_by_region must be 4 non-negative integers")
    if not _is_int(action.extra_workers) or action.extra_workers < 0:
        return Rejection(ErrorCode.INVALID_ALLOCATION, "extra_workers must be a non-negative integer")
    if not isinstance(media, tuple) or len(media) != NUM_REGIONS or not all(isinstance(m, bool) for m in media):
        return Rejection(ErrorCode.INVALID_ALLOCATION, "media_regions must be 4 booleans")
    available = state.config.workers.free_workers + action.extra_workers
    if sum(workers) > available:
        return Rejection(ErrorCode.INVALID_ALLOCATION, f"assigned {sum(workers)} workers but only {available} available")
    cost = allocation_cost(action, state.sizes, state.config.workers.extra_worker_cost, state.config.media.base_cost)
    if not player.ledger.can_afford(cost):
        return Rejection(ErrorCode.INSUFFICIENT_FUNDS, f"allocation costs {cost:.2f} but treasury is {player.ledger.balance:.2f}")
    return None


def validate_forfeit(state: "GameState", player: "PlayerState", action: Forfeit) -> Rejection | None:
    """Spec §7: ROUND only, player not yet ready."""
    return _round_guard(state, player)


VALIDATORS = {Query: validate_query, Commit: validate_commit, Allocate: validate_allocate, Forfeit: validate_forfeit}


def validate(state: "GameState", player: "PlayerState", action: object) -> Rejection | None:
    """Dispatch to the pure validator for ``action``'s type."""
    validator = VALIDATORS.get(type(action))
    if validator is None:
        return Rejection(ErrorCode.INVALID_ACTION, f"unknown action {type(action).__name__}")
    return validator(state, player, action)
