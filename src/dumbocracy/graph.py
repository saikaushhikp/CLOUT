"""Sparse immutable electorate graph construction and summaries."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .errors import DumbocracyError, ErrorCode


@dataclass(frozen=True, slots=True)
class GraphSummary:
    node_count: int
    edge_count: int
    density: float
    min_degree: int
    max_degree: int
    mean_degree: float
    component_count: int


@dataclass(frozen=True, slots=True)
class ElectorateGraph:
    """Loop-free undirected graph stored as canonical sparse index pairs."""

    node_ids: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.node_ids or any(not isinstance(n, str) or not n for n in self.node_ids):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "node IDs must be unique nonempty strings")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "node IDs must be unique nonempty strings")
        n = len(self.node_ids)
        canonical: list[tuple[int, int]] = []
        for edge in self.edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "each edge must be an index pair")
            left, right = edge
            if isinstance(left, bool) or isinstance(right, bool) or not isinstance(left, int) or not isinstance(right, int):
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "edge endpoints must be integers")
            if not (0 <= left < n and 0 <= right < n):
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "edge endpoint is out of range")
            if left >= right:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "edges must be canonical non-loop pairs")
            canonical.append((left, right))
        if len(set(canonical)) != len(canonical) or tuple(sorted(canonical)) != self.edges:
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "edges must be sorted and unique")

    @property
    def node_index(self) -> dict[str, int]:
        return {node_id: index for index, node_id in enumerate(self.node_ids)}

    @classmethod
    def from_edge_ids(cls, node_ids: Iterable[str], edges: Iterable[Iterable[str]]) -> "ElectorateGraph":
        ids = tuple(node_ids)
        if not ids or any(not isinstance(node, str) or not node for node in ids):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "node IDs must be unique nonempty strings")
        if len(set(ids)) != len(ids):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "node IDs must be unique")
        mapping = {node: index for index, node in enumerate(ids)}
        pairs: list[tuple[int, int]] = []
        for edge in edges:
            pair = tuple(edge)
            if len(pair) != 2:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "each edge needs two endpoints")
            if any(not isinstance(endpoint, str) for endpoint in pair):
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "edge endpoints must be node ID strings")
            if pair[0] not in mapping or pair[1] not in mapping:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "edge refers to unknown node")
            left, right = sorted((mapping[pair[0]], mapping[pair[1]]))
            if left == right:
                raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "self-loops are forbidden")
            pairs.append((left, right))
        if len(set(pairs)) != len(pairs):
            raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "duplicate undirected edge")
        return cls(ids, tuple(sorted(pairs)))

    def neighbors(self) -> tuple[tuple[int, ...], ...]:
        buckets: list[list[int]] = [[] for _ in self.node_ids]
        for left, right in self.edges:
            buckets[left].append(right)
            buckets[right].append(left)
        return tuple(tuple(sorted(bucket)) for bucket in buckets)

    def summary(self) -> GraphSummary:
        n, edge_count = len(self.node_ids), len(self.edges)
        adjacent = self.neighbors()
        degrees = [len(nodes) for nodes in adjacent]
        unseen = set(range(n))
        components = 0
        while unseen:
            components += 1
            stack = [unseen.pop()]
            while stack:
                current = stack.pop()
                for neighbor in adjacent[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
        return GraphSummary(n, edge_count, 0.0 if n < 2 else 2 * edge_count / (n * (n - 1)), min(degrees), max(degrees), sum(degrees) / n, components)


def erdos_renyi_graph(node_count: int, probability: float, seed: int, node_ids: tuple[str, ...] | None = None) -> ElectorateGraph:
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 1:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "node_count must be an integer >= 1")
    if node_ids is None:
        node_ids = tuple(f"v{index}" for index in range(node_count))
    if len(node_ids) != node_count:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "node_ids length does not match node_count")
    if not isinstance(probability, (int, float)) or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "edge probability must be in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "graph seed must be an integer")
    pair_count = node_count * (node_count - 1) // 2
    if probability == 0 or pair_count == 0:
        return ElectorateGraph(tuple(node_ids), ())
    if probability == 1:
        edges = tuple((left, right) for left in range(node_count) for right in range(left + 1, node_count))
        return ElectorateGraph(tuple(node_ids), edges)
    # Geometric skip sampling visits expected edges rather than all N(N-1)/2
    # pairs, which keeps sparse large-graph initialization practical.
    rng = np.random.default_rng(seed)
    positions: list[int] = []
    position = -1
    while True:
        position += int(rng.geometric(probability))
        if position >= pair_count:
            break
        positions.append(position)
    edges: list[tuple[int, int]] = []
    for flat in positions:
        left = int(((2 * node_count - 1) - math.sqrt((2 * node_count - 1) ** 2 - 8 * flat)) / 2)
        before = left * (2 * node_count - left - 1) // 2
        while flat < before:
            left -= 1
            before = left * (2 * node_count - left - 1) // 2
        while left + 1 < node_count and flat >= (left + 1) * (2 * node_count - left - 2) // 2:
            left += 1
            before = left * (2 * node_count - left - 1) // 2
        right = left + 1 + (flat - before)
        edges.append((left, right))
    return ElectorateGraph(tuple(node_ids), tuple(edges))


def load_edge_list_fixture(path: str | Path) -> ElectorateGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("directed") is not False or payload.get("allow_self_loops") is not False:
        raise DumbocracyError(ErrorCode.INVALID_CONFIGURATION, "v1 fixtures must be undirected and loop-free")
    return ElectorateGraph.from_edge_ids(payload["nodes"], payload["edges"])
