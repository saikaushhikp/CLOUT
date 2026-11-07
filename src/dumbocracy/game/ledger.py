"""Per-player Decimal treasury with an atomic debit (spec §1 money, §7 treasury U14)."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def money(value: Decimal | int | str) -> Decimal:
    """Quantise to 2 dp, half-up (spec §1)."""
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def money_str(value: Decimal) -> str:
    """Contract format: decimal string with exactly 2 dp, e.g. ``"500.00"``."""
    return f"{money(value):.2f}"


class InsufficientFunds(ValueError):
    """Raised by :meth:`Ledger.debit` when the amount exceeds the balance (nothing is changed)."""


@dataclass(frozen=True, slots=True)
class Ledger:
    """Immutable treasury: ``debit`` returns a new ledger or raises, never partially applies."""

    balance: Decimal
    spent: Decimal = Decimal("0.00")

    def __post_init__(self) -> None:
        for name in ("balance", "spent"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be a finite non-negative Decimal")
            object.__setattr__(self, name, money(value))

    def can_afford(self, amount: Decimal) -> bool:
        return money(amount) <= self.balance

    def debit(self, amount: Decimal) -> "Ledger":
        amount = money(amount)
        if amount < 0:
            raise ValueError("debit amount must be non-negative")
        if amount > self.balance:
            raise InsufficientFunds(f"cost {money_str(amount)} exceeds treasury {money_str(self.balance)}")
        return Ledger(self.balance - amount, self.spent + amount)
