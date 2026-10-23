"""Small, deterministic bounded-LTL diagnostics over completed episode traces.

This module evaluates only the supplied finite trace. It does not enumerate
states, strategies, or future action branches.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, TypeAlias

from .engine import GameState
from .errors import ErrorCode


ATOMS = frozenset({"Terminal", "Victory", "BudgetNonnegative", "StateValid", "ActionAtomic"})
SAFE_ERROR_CODES = frozenset(code.value for code in ErrorCode)
Truth: TypeAlias = bool | None


class LogicInputError(ValueError):
    """Raised when a trace or formula is malformed, not when a property fails."""


class LogicStatus(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TraceState:
    time_index: int
    predicates: tuple[tuple[str, Truth], ...]

    def __post_init__(self) -> None:
        if isinstance(self.time_index, bool) or not isinstance(self.time_index, int):
            raise LogicInputError("time_index must be an integer")
        try:
            predicates = tuple(tuple(item) for item in self.predicates)
        except TypeError as exc:
            raise LogicInputError("predicates must be string/value pairs") from exc
        if any(len(item) != 2 or not isinstance(item[0], str) for item in predicates):
            raise LogicInputError("predicates must be string/value pairs")
        if any(type(value) is not bool and value is not None for _, value in predicates):
            raise LogicInputError("predicate values must be true, false, or null")
        object.__setattr__(self, "predicates", tuple(sorted(predicates)))

    @classmethod
    def from_mapping(cls, time_index: int, predicates: Mapping[str, Truth]) -> "TraceState":
        if any(not isinstance(name, str) for name in predicates):
            raise LogicInputError("predicate names must be strings")
        return cls(time_index, tuple(sorted(predicates.items())))

    def predicate(self, name: str) -> Truth:
        return dict(self.predicates)[name]

    def to_public_dict(self) -> dict[str, object]:
        return {"time_index": self.time_index, "predicates": dict(self.predicates)}


@dataclass(frozen=True, slots=True)
class TraceAuditEvent:
    sequence: int
    round_index: int
    event_type: str
    action_id: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.sequence, self.round_index)):
            raise LogicInputError("audit sequence and round index must be nonnegative integers")
        if not isinstance(self.event_type, str):
            raise LogicInputError("audit event type must be a string")
        if self.action_id is not None and not isinstance(self.action_id, str):
            raise LogicInputError("audit action ID must be a string or null")
        if self.error_code is not None and not isinstance(self.error_code, str):
            raise LogicInputError("audit error code must be a string or null")
        if self.error_code is not None and self.error_code not in SAFE_ERROR_CODES:
            raise LogicInputError("audit error code must use a stable public code")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "round_index": self.round_index,
            "event_type": self.event_type,
            "action_id": self.action_id,
            "error_code": self.error_code,
        }


@dataclass(frozen=True, slots=True)
class CanonicalTrace:
    """H+1 predicate snapshots; audit sequence numbers are separate metadata."""

    horizon: int
    states: tuple[TraceState, ...]
    audit_events: tuple[TraceAuditEvent, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        try:
            states = tuple(self.states)
            audit_events = tuple(self.audit_events)
        except TypeError as exc:
            raise LogicInputError("states and audit_events must be iterable sequences") from exc
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "audit_events", audit_events)
        try:
            raw_provenance = tuple(self.provenance)
        except TypeError as exc:
            raise LogicInputError("provenance must contain string key/value pairs") from exc
        if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in raw_provenance):
            raise LogicInputError("provenance must contain string key/value pairs")
        provenance = tuple((pair[0], pair[1]) for pair in raw_provenance)
        object.__setattr__(self, "provenance", tuple(sorted(provenance)))
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 0:
            raise LogicInputError("horizon must be a nonnegative integer")
        if any(not isinstance(state, TraceState) for state in self.states):
            raise LogicInputError("states must be TraceState values")
        if any(not isinstance(event, TraceAuditEvent) for event in self.audit_events):
            raise LogicInputError("audit_events must be TraceAuditEvent values")
        if any(len(pair) != 2 or not all(isinstance(value, str) for value in pair) for pair in self.provenance):
            raise LogicInputError("provenance must contain string key/value pairs")
        if len(self.states) != self.horizon + 1:
            raise LogicInputError("trace must contain exactly horizon + 1 state snapshots")
        for expected, state in enumerate(self.states):
            if state.time_index != expected:
                raise LogicInputError("state time indices must be contiguous from zero")
            names = [name for name, _ in state.predicates]
            if len(names) != len(set(names)):
                raise LogicInputError("predicate names must be unique at each time index")
            if set(names) != ATOMS:
                raise LogicInputError("each state must contain exactly the required predicate atoms")
            for name, value in state.predicates:
                if type(value) is not bool and value is not None:
                    raise LogicInputError(f"predicate {name} must be true, false, or null")
        terminal_indices = [s.time_index for s in self.states if s.predicate("Terminal") is True]
        if terminal_indices and terminal_indices != [self.horizon]:
            raise LogicInputError("a terminal state may occur only at the final trace index")
        for state in self.states[:-1]:
            if state.predicate("Terminal") is not False:
                raise LogicInputError("Terminal must be false before the final trace index")
            if state.predicate("Victory") is not None:
                raise LogicInputError("Victory must be null before terminal state")
        final = self.states[-1]
        if final.predicate("Terminal") is True and final.predicate("Victory") is None:
            raise LogicInputError("Victory must be defined at a terminal state")
        if final.predicate("Terminal") is False and final.predicate("Victory") is not None:
            raise LogicInputError("Victory must be null when the final state is nonterminal")
        if any(event.round_index < 0 or event.round_index > self.horizon for event in self.audit_events):
            raise LogicInputError("audit event round index is outside the trace")
        if any(event.event_type not in ("action_proposed", "action_accepted", "action_rejected") for event in self.audit_events):
            raise LogicInputError("unsupported audit event type")
        if any(event.action_id is not None and event.action_id not in ("targeted_contact", "broadcast", "no_op", "unknown") for event in self.audit_events):
            raise LogicInputError("audit action ID is not a safe public action identifier")
        rejected_rounds = {event.round_index for event in self.audit_events if event.event_type == "action_rejected"}
        for state in self.states:
            atomic = state.predicate("ActionAtomic")
            if state.time_index in rejected_rounds and type(atomic) is not bool:
                raise LogicInputError("a rejection audit requires a boolean ActionAtomic value")
            if state.time_index not in rejected_rounds and atomic is not None:
                raise LogicInputError("ActionAtomic must be null when no rejection audit applies")
        sequences = [event.sequence for event in self.audit_events]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise LogicInputError("audit event sequence numbers must be strictly increasing")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CanonicalTrace":
        if not isinstance(data, Mapping):
            raise LogicInputError("trace payload must be an object")
        if data.get("schema_version", "1.0") != "1.0":
            raise LogicInputError("unsupported canonical trace schema version")
        horizon = data.get("horizon")
        raw_states = data.get("states")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not isinstance(raw_states, list):
            raise LogicInputError("trace requires integer horizon and state list")
        states: list[TraceState] = []
        for index, raw_state in enumerate(raw_states):
            if not isinstance(raw_state, Mapping):
                raise LogicInputError("each trace state must be an object")
            predicates = raw_state.get("predicates", raw_state)
            if not isinstance(predicates, Mapping):
                raise LogicInputError("each trace state requires a predicate object")
            time_index = raw_state.get("time_index", index) if "predicates" in raw_state else index
            if isinstance(time_index, bool) or not isinstance(time_index, int):
                raise LogicInputError("time_index must be an integer")
            states.append(TraceState.from_mapping(time_index, predicates))
        raw_audit = data.get("audit_events", [])
        if not isinstance(raw_audit, list):
            raise LogicInputError("audit_events must be a list")
        audits: list[TraceAuditEvent] = []
        for event in raw_audit:
            if not isinstance(event, Mapping):
                raise LogicInputError("audit event must be an object")
            sequence, round_index = event.get("sequence"), event.get("round_index")
            event_type, action_id, error_code = event.get("event_type"), event.get("action_id"), event.get("error_code")
            if (isinstance(sequence, bool) or not isinstance(sequence, int)
                    or isinstance(round_index, bool) or not isinstance(round_index, int)
                    or not isinstance(event_type, str)
                    or (action_id is not None and not isinstance(action_id, str))
                    or (error_code is not None and not isinstance(error_code, str))):
                raise LogicInputError("audit event has invalid field types")
            audits.append(TraceAuditEvent(sequence, round_index, event_type, action_id, error_code))
        raw_provenance = data.get("provenance", {})
        if not isinstance(raw_provenance, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in raw_provenance.items()):
            raise LogicInputError("provenance must be a string-to-string object")
        return cls(horizon, tuple(states), tuple(audits), tuple(sorted(raw_provenance.items())))

    @classmethod
    def from_game_state(cls, state: GameState) -> "CanonicalTrace":
        """Adapt a completed engine state, folding same-round audit records.

        Only predicate values and safe event summaries leave this adapter; voter
        state is inspected solely to compute the M0 StateValid atom.
        """
        if not isinstance(state, GameState):
            raise LogicInputError("state adapter requires a GameState")
        horizon = state.config.horizon_rounds
        if not state.terminal or state.round_index != horizon or state.result is None:
            raise LogicInputError("a completed fixed-horizon GameState is required")
        rows: list[dict[str, Truth]] = [
            {
                "Terminal": False,
                "Victory": None,
                "BudgetNonnegative": state.config.initial_budget >= 0,
                "StateValid": all(
                    math.isfinite(v.support) and 0 <= v.support <= 1
                    and math.isfinite(v.susceptibility) and 0 <= v.susceptibility <= 1
                    for v in state.initial_voters
                ),
                "ActionAtomic": None,
            }
        ]
        rows.extend({
            "Terminal": False,
            "Victory": None,
            "BudgetNonnegative": None,
            "StateValid": None,
            "ActionAtomic": None,
        } for _ in range(horizon))

        accepted_rounds: set[int] = set()
        atomic_checks: dict[int, list[bool | None]] = {}
        for record in state.trace:
            round_index = record.round_index
            if not 0 <= round_index <= horizon:
                raise LogicInputError("engine trace contains an out-of-range round index")
            if record.accepted:
                if round_index == 0 or round_index in accepted_rounds:
                    raise LogicInputError("engine trace has duplicate or invalid accepted state time")
                accepted_rounds.add(round_index)
                rows[round_index].update(record.predicates())
            elif record.accepted is False:
                atomic_checks.setdefault(round_index, []).append(record.action_atomic)

        if accepted_rounds != set(range(1, horizon + 1)):
            raise LogicInputError("engine trace is missing an accepted transition state")
        rows[horizon]["Terminal"] = True
        rows[horizon]["Victory"] = state.result.victory
        for round_index, checks in atomic_checks.items():
            rows[round_index]["ActionAtomic"] = False if False in checks else (True if all(v is True for v in checks) else None)

        audits: list[TraceAuditEvent] = []
        for event in state.internal_events:
            if event.event_type not in ("action_proposed", "action_accepted", "action_rejected"):
                continue
            attributes = dict(event.attributes)
            action_id = attributes.get("action_id")
            error_code = attributes.get("error_code")
            audits.append(TraceAuditEvent(
                event.sequence,
                event.round_index,
                event.event_type,
                str(action_id) if action_id in ("targeted_contact", "broadcast", "no_op") else ("unknown" if action_id is not None else None),
                str(error_code) if error_code is not None else None,
            ))
        provenance = tuple(sorted((str(key), str(value)) for key, value in state.result.metadata))
        return cls(horizon, tuple(TraceState.from_mapping(i, row) for i, row in enumerate(rows)), tuple(audits), provenance)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "horizon": self.horizon,
            "time_convention": "state index 0 is initial; index t is t accepted transitions",
            "states": [state.to_public_dict() for state in self.states],
            "audit_events": [event.to_public_dict() for event in self.audit_events],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class Atom:
    name: str


@dataclass(frozen=True, slots=True)
class And:
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True, slots=True)
class Or:
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True, slots=True)
class Not:
    child: "Formula"


@dataclass(frozen=True, slots=True)
class Implies:
    left: "Formula"
    right: "Formula"


@dataclass(frozen=True, slots=True)
class Globally:
    child: "Formula"
    bound: int | None = None


@dataclass(frozen=True, slots=True)
class Eventually:
    child: "Formula"
    bound: int


Formula: TypeAlias = Atom | And | Or | Not | Implies | Globally | Eventually


@dataclass(frozen=True, slots=True)
class LogicResult:
    name: str
    formula: str
    status: LogicStatus
    checked_interval: tuple[int, int]
    witness_index: int | None = None
    counterexample_index: int | None = None
    message: str = ""

    def to_public_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "formula": self.formula,
            "status": self.status.value,
            "checked_interval": list(self.checked_interval),
            "witness_index": self.witness_index,
            "counterexample_index": self.counterexample_index,
            "message": self.message,
        }


def _validate_formula(formula: Formula) -> None:
    if isinstance(formula, Atom):
        if not isinstance(formula.name, str) or formula.name not in ATOMS:
            raise LogicInputError(f"unknown predicate atom: {formula.name}")
    elif isinstance(formula, (And, Or, Implies)):
        _validate_formula(formula.left)
        _validate_formula(formula.right)
    elif isinstance(formula, (Not, Globally, Eventually)):
        if isinstance(formula, (Globally, Eventually)) and formula.bound is not None:
            if isinstance(formula.bound, bool) or not isinstance(formula.bound, int) or formula.bound < 0:
                raise LogicInputError("temporal bound must be a nonnegative integer")
        if isinstance(formula, Eventually) and formula.bound is None:
            raise LogicInputError("Eventually requires an explicit finite bound")
        _validate_formula(formula.child)
    else:
        raise LogicInputError("formula must use the closed logic AST")


def _format(formula: Formula) -> str:
    if isinstance(formula, Atom):
        return formula.name
    if isinstance(formula, And):
        return f"({_format(formula.left)} and {_format(formula.right)})"
    if isinstance(formula, Or):
        return f"({_format(formula.left)} or {_format(formula.right)})"
    if isinstance(formula, Not):
        return f"not {_format(formula.child)}"
    if isinstance(formula, Implies):
        return f"({_format(formula.left)} -> {_format(formula.right)})"
    if isinstance(formula, Globally):
        suffix = "" if formula.bound is None else f"_<= {formula.bound}"
        return f"G{suffix}({_format(formula.child)})"
    return f"F_<= {formula.bound}({_format(formula.child)})"


def _not(value: Truth) -> Truth:
    return None if value is None else not value


def _and(left: Truth, right: Truth) -> Truth:
    if left is False or right is False:
        return False
    return True if left is True and right is True else None


def _or(left: Truth, right: Truth) -> Truth:
    if left is True or right is True:
        return True
    return False if left is False and right is False else None


def _value(trace: CanonicalTrace, formula: Formula, time_index: int) -> Truth:
    if isinstance(formula, Atom):
        return trace.states[time_index].predicate(formula.name)
    if isinstance(formula, And):
        return _and(_value(trace, formula.left, time_index), _value(trace, formula.right, time_index))
    if isinstance(formula, Or):
        return _or(_value(trace, formula.left, time_index), _value(trace, formula.right, time_index))
    if isinstance(formula, Not):
        return _not(_value(trace, formula.child, time_index))
    if isinstance(formula, Implies):
        return _or(_not(_value(trace, formula.left, time_index)), _value(trace, formula.right, time_index))
    end = trace.horizon if formula.bound is None else min(trace.horizon, time_index + formula.bound)
    values = [_value(trace, formula.child, index) for index in range(time_index, end + 1)]
    if isinstance(formula, Globally):
        return False if False in values else (None if None in values else True)
    return True if True in values else (None if None in values else False)


def _checked_end(formula: Formula, start: int, horizon: int) -> int:
    if isinstance(formula, Atom):
        return start
    if isinstance(formula, (And, Or, Implies)):
        return max(_checked_end(formula.left, start, horizon), _checked_end(formula.right, start, horizon))
    if isinstance(formula, Not):
        return _checked_end(formula.child, start, horizon)
    end = horizon if formula.bound is None else min(horizon, start + formula.bound)
    return _checked_end(formula.child, end, horizon)


def evaluate_trace(trace: CanonicalTrace, formula: Formula, *, name: str | None = None) -> LogicResult:
    if not isinstance(trace, CanonicalTrace):
        raise LogicInputError("trace must be a validated CanonicalTrace")
    if name is not None and not isinstance(name, str):
        raise LogicInputError("formula name must be a string")
    _validate_formula(formula)
    value = _value(trace, formula, 0)
    if isinstance(formula, (Globally, Eventually)):
        root_end = trace.horizon if formula.bound is None else min(trace.horizon, formula.bound)
    else:
        root_end = _checked_end(formula, 0, trace.horizon)
    end = _checked_end(formula, 0, trace.horizon)
    status = LogicStatus.UNKNOWN if value is None else (LogicStatus.SATISFIED if value else LogicStatus.VIOLATED)
    witness = next((i for i in range(root_end + 1) if _value(trace, formula.child, i) is True), None) if isinstance(formula, Eventually) and value is True else None
    counterexample = None
    if isinstance(formula, Globally) and value is False:
        counterexample = next(i for i in range(root_end + 1) if _value(trace, formula.child, i) is False)
    message = {
        LogicStatus.SATISFIED: "The formula holds on the supplied finite trace.",
        LogicStatus.VIOLATED: "The formula is falsified on the supplied finite trace.",
        LogicStatus.UNKNOWN: "Unknown atom values prevent a definite result on this trace.",
    }[status]
    return LogicResult(name or _format(formula), _format(formula), status, (0, end), witness, counterexample, message)


SAFETY = Globally(And(Atom("BudgetNonnegative"), Atom("StateValid")))
TERMINAL_OUTCOME_DEFINED = Globally(Implies(Atom("Terminal"), Or(Atom("Victory"), Not(Atom("Victory")))))


def evaluate_diagnostics(trace: CanonicalTrace) -> dict[str, LogicResult]:
    formulas = (
        ("G(BudgetNonnegative and StateValid)", SAFETY),
        (f"F_<=H Terminal (H={trace.horizon})", Eventually(Atom("Terminal"), trace.horizon)),
        ("Terminal -> (Victory or not Victory)", TERMINAL_OUTCOME_DEFINED),
    )
    return {name: evaluate_trace(trace, formula, name=name) for name, formula in formulas}
