#!/usr/bin/env python3
"""Transport-neutral example client. It only reads a public Observation JSON.
example usage:

    PYTHONPATH=src python -c 'from decimal import Decimal; from dumbocracy.config import *; from dumbocracy.engine import GameEngine; from dumbocracy.information import public_json; c=GameConfig(SeedConfig(20261019), GraphSpec("erdos_renyi", 3, 0.5), 5, Decimal("10.00"), DynamicsConfig(0.35), ActionConfig(0.2, 0.1)); engine=GameEngine(); print(public_json(engine.observe(engine.initialize(c))))' | PYTHONPATH=src python examples/week3_mock_client.py
"""
from __future__ import annotations

import json
import sys
from typing import Any


def choose_action(observation: dict[str, Any]) -> dict[str, Any]:
    """Return one legal action payload without access to simulator state."""
    legal = observation.get("legal_actions")
    if not isinstance(legal, list):
        raise ValueError("observation must contain a legal_actions list")
    for descriptor in legal:
        if descriptor.get("action_id") == "targeted_contact" and descriptor.get("eligible_target_ids"):
            return {"action_id": "targeted_contact", "target_ids": [descriptor["eligible_target_ids"][0]]}
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
