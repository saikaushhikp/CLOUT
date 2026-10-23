#!/usr/bin/env python3
"""Transport-neutral client: reads only public Observation JSON from stdin.

Smoke test with the committed public-only fixture:
    python examples/week3_mock_client.py < tests/fixtures/public_observation_v1.json
"""
from __future__ import annotations

import json
import sys
from typing import Any


def choose_action(observation: dict[str, Any]) -> dict[str, Any]:
    """Return one legal action payload without access to simulator state."""
    if not isinstance(observation, dict):
        raise ValueError("observation must be a JSON object")
    legal = observation.get("legal_actions")
    if not isinstance(legal, list):
        raise ValueError("observation must contain a legal_actions list")
    if any(not isinstance(item, dict) or not isinstance(item.get("action_id"), str) for item in legal):
        raise ValueError("each legal action must be an object with an action_id")
    for descriptor in legal:
        if descriptor.get("action_id") == "targeted_contact":
            targets = descriptor.get("eligible_target_ids")
            if not isinstance(targets, list) or any(not isinstance(target, str) for target in targets):
                raise ValueError("targeted action must contain a public eligible_target_ids list")
            if targets:
                return {"action_id": "targeted_contact", "target_ids": [targets[0]]}
    for descriptor in legal:
        if descriptor.get("action_id") == "no_op":
            return {"action_id": "no_op", "target_ids": []}
    raise ValueError("observation has no supported legal action")


def main() -> int:
    observation = json.load(sys.stdin)
    print(json.dumps(choose_action(observation), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
