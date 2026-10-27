"""Sparse deterministic graph generators used only by experiment scenarios."""
from __future__ import annotations

import random

from .errors import DumbocracyError, ErrorCode
from .graph import ElectorateGraph


def ring_lattice_graph(node_count: int, degree: int, seed: int = 0) -> ElectorateGraph:
    """Return an undirected fixed-degree ring lattice, with seeded node relabeling."""
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 3:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "ring lattice node_count must be >= 3")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 2 or degree % 2 or degree >= node_count:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "degree must be even, >= 2, and < node_count")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "seed must be an integer")
    half = degree // 2
    edges = {(min(i, (i + offset) % node_count), max(i, (i + offset) % node_count))
             for i in range(node_count) for offset in range(1, half + 1)}
    # A deterministic permutation changes labels but preserves the topology.
    ids = [f"v{i}" for i in range(node_count)]
    random.Random(seed).shuffle(ids)
    return ElectorateGraph.from_edge_ids(tuple(ids), ((ids[a], ids[b]) for a, b in sorted(edges)))
