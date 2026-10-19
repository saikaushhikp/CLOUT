#!/usr/bin/env python3
"""Developer-only Week 2 foundation smoke demo; not a public Observation payload."""
from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dumbocracy.campaign import ActionContext, CampaignAction, ResourceLedger, commit_action, validate_action
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.graph import load_edge_list_fixture
from dumbocracy.mas import VoterState, advance_support


def revision() -> str:
    try:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
            return "uncommitted"
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "uncommitted"


def main() -> int:
    graph = load_edge_list_fixture(ROOT / "tests/fixtures/tiny_path_graph.json")
    payload = json.loads((ROOT / "tests/fixtures/tiny_initial_state.json").read_text())
    voters = tuple(VoterState(**item) for item in payload["voters"])
    config = GameConfig(SeedConfig(20261019), GraphSpec("edge_list", 3), 3, Decimal("10.00"), DynamicsConfig(.5), ActionConfig(.2, .1))
    context = ActionContext(graph.node_ids, ResourceLedger.create(config.initial_budget))
    accepted = validate_action(context, CampaignAction("targeted_contact", ("v0",), client_action_id="demo-1"), config.actions)
    if not accepted.accepted:
        raise RuntimeError(f"unexpected rejection: {accepted.error_code}")
    committed = commit_action(context, accepted)
    after = advance_support(graph, voters, config.dynamics.alpha, committed.contributions)
    rejected = validate_action(committed.context, CampaignAction("targeted_contact", ("unknown",), client_action_id="demo-2"), config.actions)
    if rejected.accepted or committed.context.ledger.remaining_budget != Decimal("9.00"):
        raise RuntimeError("rejected action atomicity check failed")
    try:
        import torch
        backend = "cuda" if torch.cuda.is_available() else "cpu (CUDA unavailable)"
        torch_version = torch.__version__
    except ImportError:
        backend, torch_version = "cpu (PyTorch unavailable)", "not installed"
    print("Week 2 developer demo — true state output, not a public observation")
    print("graph:", graph.summary())
    print("before support:", [v.support for v in voters])
    print("after support:", [v.support for v in after])
    print("action:", committed.action.action_id, "cost:", committed.cost)
    print("ledger:", committed.context.ledger)
    print("rejected action:", rejected.error_code, "ledger unchanged:", committed.context.ledger.remaining_budget)
    print("metadata:", {"revision": revision(), "python": platform.python_version(), "torch": torch_version, "backend": backend, "seeds": {"root": config.seeds.root_seed, "graph": config.seeds.graph, "initial_state": config.seeds.initial_state, "transition": config.seeds.transition, "strategy": config.seeds.strategy}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
