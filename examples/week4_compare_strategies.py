#!/usr/bin/env python3
"""Small paired-seed baseline comparison; human-run after static review."""
from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.experiments import ExperimentScenario, run_paired_scenarios
from dumbocracy.strategies import FirstLegalStrategy, NoOpStrategy, SeededRandomLegalStrategy


def code_revision() -> str:
    try:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
            return "uncommitted"
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=32, help="small illustrative ER scenario")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--backend", default="cpu", help="cpu or explicitly selected cuda:N")
    args = parser.parse_args()
    if args.nodes < 1 or args.rounds < 1:
        parser.error("nodes and rounds must be positive")
    config_small = GameConfig(
        SeedConfig(args.seeds[0]),
        GraphSpec("erdos_renyi", args.nodes, min(1.0, 8.0 / max(1, args.nodes - 1))),
        args.rounds,
        Decimal("100.00"),
        DynamicsConfig(.35),
        ActionConfig(.02, .005),
    )
    medium_nodes = args.nodes * 2
    config_medium = replace(config_small, graph=GraphSpec("erdos_renyi", medium_nodes, min(1.0, 8.0 / max(1, medium_nodes - 1))))
    report = run_paired_scenarios(
        (
            ExperimentScenario(f"er-{args.nodes}", config_small),
            ExperimentScenario(f"er-{medium_nodes}", config_medium),
        ),
        tuple(args.seeds),
        {
            "first_legal": FirstLegalStrategy,
            "no_op": NoOpStrategy,
            "seeded_random_legal": SeededRandomLegalStrategy,
        },
        backend=args.backend,
        revision=code_revision(),
    )
    sys.stdout.write(report.to_jsonl())
    print(json.dumps({"record_type": "aggregate", "schema_version": "1.0", "aggregates": report.aggregates}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
