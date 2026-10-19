"""Voter validation and the approved synchronous support transition."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .errors import DumbocracyError, ErrorCode
from .graph import ElectorateGraph


@dataclass(frozen=True, slots=True)
class VoterState:
    voter_id: str
    support: float
    susceptibility: float
    campaign_exposure: float = 0.0

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("support", self.support, 0.0, 1.0),
            ("susceptibility", self.susceptibility, 0.0, 1.0),
            ("campaign_exposure", self.campaign_exposure, 0.0, math.inf),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not lower <= value <= upper:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, f"{name} is outside its valid range")


def validate_voters(graph: ElectorateGraph, voters: Iterable[VoterState]) -> tuple[VoterState, ...]:
    result = tuple(voters)
    if tuple(voter.voter_id for voter in result) != graph.node_ids:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "voters must match graph node IDs in graph order")
    return result


def advance_support(
    graph: ElectorateGraph,
    voters: Iterable[VoterState],
    alpha: float,
    contributions: Iterable[float] | None = None,
) -> tuple[VoterState, ...]:
    """Return new voters using only the complete pre-transition support vector."""
    old = validate_voters(graph, voters)
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "alpha must be finite and in [0, 1]")
    n = len(old)
    effects = np.zeros(n, dtype=float) if contributions is None else np.asarray(tuple(contributions), dtype=float)
    if effects.shape != (n,) or not np.all(np.isfinite(effects)):
        raise DumbocracyError(ErrorCode.NUMERICAL_INSTABILITY, "campaign contributions must be a finite vector matching voters")
    support = np.asarray([voter.support for voter in old], dtype=float)
    susceptibility = np.asarray([voter.susceptibility for voter in old], dtype=float)
    neighbor_sum = np.zeros(n, dtype=float)
    degree = np.zeros(n, dtype=int)
    for left, right in graph.edges:
        neighbor_sum[left] += support[right]
        neighbor_sum[right] += support[left]
        degree[left] += 1
        degree[right] += 1
    means = np.divide(neighbor_sum, degree, out=np.zeros(n, dtype=float), where=degree != 0)
    updated = support + susceptibility * (float(alpha) * (means - support) + effects)
    if not np.all(np.isfinite(updated)):
        raise DumbocracyError(ErrorCode.NUMERICAL_INSTABILITY, "support update produced a non-finite value")
    clipped = np.clip(updated, 0.0, 1.0)
    return tuple(
        VoterState(voter.voter_id, float(clipped[index]), voter.susceptibility, voter.campaign_exposure)
        for index, voter in enumerate(old)
    )


def player_vote_share(voters: Iterable[VoterState]) -> float:
    states = tuple(voters)
    if not states:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "cannot score an empty electorate")
    return sum(voter.support > 0.5 for voter in states) / len(states)
