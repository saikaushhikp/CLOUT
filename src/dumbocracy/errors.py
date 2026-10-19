"""Stable errors shared by the Week 2 foundations."""
from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    INVALID_ACTION = "INVALID_ACTION"
    INVALID_TARGET = "INVALID_TARGET"
    INSUFFICIENT_BUDGET = "INSUFFICIENT_BUDGET"
    MALFORMED_OBSERVATION = "MALFORMED_OBSERVATION"
    NUMERICAL_INSTABILITY = "NUMERICAL_INSTABILITY"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


class DumbocracyError(ValueError):
    """A predictable domain failure that carries a stable, non-private code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)
