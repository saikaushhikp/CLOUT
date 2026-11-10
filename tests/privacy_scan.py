"""Recursive leak scanner for player-facing payloads (spec §6.6, web_api_v2.md privacy rules)."""
from __future__ import annotations

from typing import Any

FORBIDDEN_KEYS = {
    "preferences", "susceptibility", "zealot", "zealots", "archetype_of_region", "personality_of_region",
    "p_intra", "p_inter", "sparsity", "sparsity_capped", "edge_probability", "src", "dst", "indptr", "indices",
    "region_of", "alpha", "gamma", "share_trajectory", "alignment_snapshots", "electorate", "knowledge_state",
    "attempts", "worker_ids", "pinned", "node_id", "node_ids", "workers_state",
}
FORBIDDEN_SUBSTRINGS = ("zealot", "archetype_of", "personality_of", "p_intra", "p_inter", "sparsity")


def find_leaks(payload: Any, num_voters: int, path: str = "$") -> list[str]:
    """Return a list of leak descriptions (empty = clean)."""
    leaks: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_KEYS or any(s in lowered for s in FORBIDDEN_SUBSTRINGS):
                leaks.append(f"{path}.{key}: forbidden key")
            leaks += find_leaks(value, num_voters, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        if len(payload) == num_voters and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in payload):
            leaks.append(f"{path}: numeric array of length num_voters")
        if len(payload) == num_voters and all(isinstance(x, (list, tuple)) for x in payload):
            leaks.append(f"{path}: per-voter nested array")
        for i, item in enumerate(payload):
            leaks += find_leaks(item, num_voters, f"{path}[{i}]")
    elif isinstance(payload, str):
        if any(s in payload.lower() for s in FORBIDDEN_SUBSTRINGS):
            leaks.append(f"{path}: forbidden text")
    return leaks


def assert_no_leaks(payload: Any, num_voters: int) -> None:
    leaks = find_leaks(payload, num_voters)
    assert not leaks, leaks[:10]
