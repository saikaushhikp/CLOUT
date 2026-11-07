"""Voting (spec §5): similarity, vote probability, expected share and the official Bernoulli election.

``S_i = 1 − ½‖v_i − b‖₁ ∈ [0, 1]``, ``P_i = σ(β_v (S_i − z))``. Each voter votes for the player
w.p. P_i (election seed stream); victory iff votes > N/2 (ties lose). The expected share
``E = mean_i P_i`` and its sd are recorded for analysis and logic.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


def similarity(v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``S_i = 1 − ½‖v_i − b‖₁``. ``v``: ``[..., N, K]``; ``b``: ``[K]`` or ``[..., K]`` (broadcast over N)."""
    if v.ndim < 2 or b.ndim < 1 or b.shape[-1] != v.shape[-1]:
        raise ValueError(f"incompatible shapes v={list(v.shape)}, b={list(b.shape)}")
    return 1 - 0.5 * (v - b.to(v.dtype).unsqueeze(-2)).abs().sum(-1)


def vote_probability(similarity: torch.Tensor, sensitivity: float, threshold: float) -> torch.Tensor:
    """``P_i = σ(β_v (S_i − z))`` (spec §5)."""
    return torch.sigmoid(sensitivity * (similarity - threshold))


def expected_share(prob: torch.Tensor) -> torch.Tensor:
    """Expected vote share ``E = mean_i P_i`` over the last (voter) dimension."""
    return prob.mean(-1)


def sample_ballots(prob: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Independent Bernoulli(P_i) ballots, ``bool`` with the shape of ``prob``.

    ``generator`` must live on ``prob``'s device (election seed stream, spec §9).
    """
    if generator.device != prob.device:
        raise ValueError(f"generator device {generator.device} != prob device {prob.device}")
    draws = torch.rand(prob.shape, generator=generator, dtype=prob.dtype, device=prob.device)
    return draws < prob


@dataclass(frozen=True)
class ElectionResult:
    """Official outcome of one election (spec §5). Contains only public primitives."""

    votes: int
    num_voters: int
    vote_share: float
    victory: bool          # votes > N/2 strictly; ties lose
    expected_share: float  # E = mean_i P_i
    expected_sd: float     # sd of the vote share: sqrt(Σ P_i (1 − P_i)) / N

    def to_public_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def run_election(v: torch.Tensor, manifesto: torch.Tensor, sensitivity: float, threshold: float,
                 generator: torch.Generator) -> ElectionResult:
    """Run the official election of a single world (``v``: ``[N, K]``, ``manifesto``: ``[K]``)."""
    if v.ndim != 2 or manifesto.ndim != 1:
        raise ValueError(f"run_election expects v [N, K] and manifesto [K], got {list(v.shape)}, {list(manifesto.shape)}")
    num_voters = int(v.shape[0])
    if num_voters == 0:
        raise ValueError("an election needs at least one voter")
    prob = vote_probability(similarity(v, manifesto), sensitivity, threshold)
    votes = int(sample_ballots(prob, generator).sum())
    variance = float((prob * (1 - prob)).sum())
    return ElectionResult(
        votes=votes,
        num_voters=num_voters,
        vote_share=votes / num_voters,
        victory=2 * votes > num_voters,
        expected_share=float(expected_share(prob)),
        expected_sd=math.sqrt(variance) / num_voters,
    )
