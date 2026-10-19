"""Pure campaign validation plus immutable, atomic ledger/effect commit."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Literal, Mapping

from .config import ActionConfig
from .errors import ErrorCode


ActionId = Literal["targeted_contact", "broadcast", "no_op"]


@dataclass(frozen=True, slots=True)
class CampaignAction:
    action_id: str
    target_ids: tuple[str, ...] = ()
    parameters: Mapping[str, object] = field(default_factory=dict)
    client_action_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_ids", tuple(self.target_ids))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class ResourceLedger:
    initial_budget: Decimal
    remaining_budget: Decimal
    spent: Decimal = Decimal("0.00")

    def __post_init__(self) -> None:
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in (self.initial_budget, self.remaining_budget, self.spent)):
            raise ValueError("ledger values must be finite Decimal values")
        if self.initial_budget < 0 or self.remaining_budget < 0 or self.spent < 0 or self.remaining_budget + self.spent != self.initial_budget:
            raise ValueError("ledger invariant violated")

    @classmethod
    def create(cls, budget: Decimal) -> "ResourceLedger":
        return cls(budget, budget, Decimal("0.00"))

    def debit(self, cost: Decimal) -> "ResourceLedger":
        return ResourceLedger(self.initial_budget, self.remaining_budget - cost, self.spent + cost)


@dataclass(frozen=True, slots=True)
class ActionContext:
    node_ids: tuple[str, ...]
    ledger: ResourceLedger
    broadcast_used: bool = False


@dataclass(frozen=True, slots=True)
class ActionValidation:
    accepted: bool
    error_code: ErrorCode | None
    normalized_action: CampaignAction
    cost: Decimal
    contributions: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CampaignCommit:
    context: ActionContext
    contributions: tuple[float, ...]
    action: CampaignAction
    cost: Decimal


def validate_action(context: ActionContext, action: CampaignAction, config: ActionConfig) -> ActionValidation:
    """Validate without changing the supplied context, ledger, or action."""
    zero = tuple(0.0 for _ in context.node_ids)
    if not isinstance(action, CampaignAction) or not isinstance(action.action_id, str):
        return ActionValidation(False, ErrorCode.INVALID_ACTION, CampaignAction("invalid"), Decimal("0.00"), zero)
    if not isinstance(action.target_ids, tuple) or any(not isinstance(target, str) for target in action.target_ids):
        return ActionValidation(False, ErrorCode.INVALID_ACTION, action, Decimal("0.00"), zero)
    if action.parameters:
        return ActionValidation(False, ErrorCode.INVALID_ACTION, action, Decimal("0.00"), zero)
    if action.action_id not in ("targeted_contact", "broadcast", "no_op"):
        return ActionValidation(False, ErrorCode.INVALID_ACTION, action, Decimal("0.00"), zero)
    if action.action_id == "no_op":
        if action.target_ids:
            return ActionValidation(False, ErrorCode.INVALID_ACTION, action, Decimal("0.00"), zero)
        return ActionValidation(True, None, action, Decimal("0.00"), zero)
    if action.action_id == "broadcast":
        if action.target_ids or context.broadcast_used:
            return ActionValidation(False, ErrorCode.INVALID_ACTION, action, Decimal("0.00"), zero)
        cost, amount = config.broadcast_cost, float(config.delta_broadcast)
        effects = tuple(amount for _ in context.node_ids)
    else:
        if not action.target_ids or len(set(action.target_ids)) != len(action.target_ids):
            return ActionValidation(False, ErrorCode.INVALID_TARGET, action, Decimal("0.00"), zero)
        indices = {node: index for index, node in enumerate(context.node_ids)}
        if any(target not in indices for target in action.target_ids):
            return ActionValidation(False, ErrorCode.INVALID_TARGET, action, Decimal("0.00"), zero)
        cost, amount = config.targeted_cost * len(action.target_ids), float(config.delta_target)
        mutable = [0.0] * len(context.node_ids)
        for target in action.target_ids:
            mutable[indices[target]] = amount
        effects = tuple(mutable)
    if cost > context.ledger.remaining_budget:
        return ActionValidation(False, ErrorCode.INSUFFICIENT_BUDGET, action, cost, zero)
    return ActionValidation(True, None, action, cost, effects)


def commit_action(context: ActionContext, validation: ActionValidation) -> CampaignCommit:
    """Commit an already accepted action by returning a wholly new context."""
    if not validation.accepted:
        raise ValueError("cannot commit a rejected action")
    updated = ActionContext(
        node_ids=context.node_ids,
        ledger=context.ledger.debit(validation.cost),
        broadcast_used=context.broadcast_used or validation.normalized_action.action_id == "broadcast",
    )
    return CampaignCommit(updated, validation.contributions, validation.normalized_action, validation.cost)
