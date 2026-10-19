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


@dataclass(frozen=True, slots=True)
class DeviceGraphView:
    """Episode-local sparse edge indices and degrees resident on one device."""

    device: str
    edge_index: object
    degree: object


def prepare_device_graph(graph: ElectorateGraph, device: str) -> DeviceGraphView:
    if not device.startswith("cuda"):
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "device graph requires a CUDA device")
    try:
        import torch
        target = torch.device(device)
        edge_index = torch.as_tensor(graph.edges, dtype=torch.long, device=target).reshape((-1, 2))
        degree = torch.zeros(len(graph.node_ids), dtype=torch.float64, device=target)
        if graph.edges:
            ones = torch.ones(len(graph.edges), dtype=torch.float64, device=target)
            degree.index_add_(0, edge_index[:, 0], ones)
            degree.index_add_(0, edge_index[:, 1], ones)
        return DeviceGraphView(device, edge_index, degree)
    except Exception as exc:
        raise DumbocracyError(ErrorCode.INTERNAL_FAILURE, "could not prepare sparse graph on CUDA") from exc


def validate_voters(graph: ElectorateGraph, voters: Iterable[VoterState]) -> tuple[VoterState, ...]:
    result = tuple(voters)
    if any(not isinstance(voter, VoterState) for voter in result):
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "voter entries must be VoterState values")
    if tuple(voter.voter_id for voter in result) != graph.node_ids:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "voters must match graph node IDs in graph order")
    return result


def advance_support(
    graph: ElectorateGraph,
    voters: Iterable[VoterState],
    alpha: float,
    contributions: Iterable[float] | None = None,
    device: str = "cpu",
    device_graph: DeviceGraphView | None = None,
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
    if device == "cpu":
        neighbor_sum = np.zeros(n, dtype=float)
        degree = np.zeros(n, dtype=int)
        for left, right in graph.edges:
            neighbor_sum[left] += support[right]
            neighbor_sum[right] += support[left]
            degree[left] += 1
            degree[right] += 1
        means = np.divide(neighbor_sum, degree, out=np.zeros(n, dtype=float), where=degree != 0)
        updated = support + susceptibility * (float(alpha) * (means - support) + effects)
    elif device.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "PyTorch is required for a CUDA backend") from exc
        if not torch.cuda.is_available():
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "requested CUDA backend is unavailable")
        try:
            target_device = torch.device(device)
            x = torch.as_tensor(support, dtype=torch.float64, device=target_device)
            susceptibility_t = torch.as_tensor(susceptibility, dtype=torch.float64, device=target_device)
            effects_t = torch.as_tensor(effects, dtype=torch.float64, device=target_device)
            if device_graph is not None and device_graph.device != device:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "device graph belongs to a different backend")
            if device_graph is not None:
                edge = device_graph.edge_index
                degree_t = device_graph.degree
                neighbor_sum_t = torch.zeros(n, dtype=torch.float64, device=target_device)
                if graph.edges:
                    left, right = edge[:, 0], edge[:, 1]
                    neighbor_sum_t.index_add_(0, left, x[right])
                    neighbor_sum_t.index_add_(0, right, x[left])
                means_t = torch.where(degree_t > 0, neighbor_sum_t / degree_t.clamp_min(1), 0)
            elif graph.edges:
                edge = torch.as_tensor(graph.edges, dtype=torch.long, device=target_device)
                left, right = edge[:, 0], edge[:, 1]
                neighbor_sum_t = torch.zeros(n, dtype=torch.float64, device=target_device)
                degree_t = torch.zeros(n, dtype=torch.float64, device=target_device)
                neighbor_sum_t.index_add_(0, left, x[right])
                neighbor_sum_t.index_add_(0, right, x[left])
                ones = torch.ones(len(graph.edges), dtype=torch.float64, device=target_device)
                degree_t.index_add_(0, left, ones)
                degree_t.index_add_(0, right, ones)
                means_t = torch.where(degree_t > 0, neighbor_sum_t / degree_t.clamp_min(1), 0)
            else:
                means_t = torch.zeros(n, dtype=torch.float64, device=target_device)
            updated_t = x + susceptibility_t * (float(alpha) * (means_t - x) + effects_t)
            if not torch.isfinite(updated_t).all().item():
                raise DumbocracyError(ErrorCode.NUMERICAL_INSTABILITY, "support update produced a non-finite value")
            updated = updated_t.clamp(0.0, 1.0).cpu().numpy()
        except DumbocracyError:
            raise
        except Exception as exc:
            raise DumbocracyError(ErrorCode.INTERNAL_FAILURE, "CUDA transition failed") from exc
    else:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "device must be 'cpu' or a CUDA device")
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
