"""Standard-library M0 artifact and fixture validation."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "docs/spec/technical_spec_v1.md": [
        "## 1. Requirements", "## 2. Product loop", "## 4. Baseline dynamics",
        "## 6. Partial information", "## 8. Temporal and epistemic semantics",
    ],
    "docs/spec/data_contracts.md": ["# Dumbocracy Data Contracts", "## Conceptual APIs"],
    "docs/spec/test_plan.md": ["# M0 Test Plan", "## M0 validation command"],
    "docs/decisions/decision_log.md": ["# Dumbocracy Decision Register", "## Frozen contract rule"],
    "docs/spec/week1_handoff.md": ["# Week 1 Hand-off"],
    "reports/week1_self_review.md": ["# Week 1 Self-Review"],
}


def main() -> None:
    for relative, headings in REQUIRED.items():
        content = (ROOT / relative).read_text(encoding="utf-8")
        for heading in headings:
            assert heading in content, f"{relative} missing {heading!r}"

    decision_log = (ROOT / "docs/decisions/decision_log.md").read_text(encoding="utf-8")
    assert "## Frozen contract rule" in decision_log
    for decision_id in range(2, 10):
        assert f"| DEC-{decision_id:03d} |" in decision_log

    specification = (ROOT / "docs/spec/technical_spec_v1.md").read_text(encoding="utf-8")
    assert "M0 Technical Specification v1, approved" in specification
    assert "support values `> 0.5`" in specification
    assert "| `no_op` |" in specification

    graph = json.loads((ROOT / "tests/fixtures/tiny_path_graph.json").read_text())
    nodes, edges = graph["nodes"], graph["edges"]
    assert len(nodes) == 3 and len(set(nodes)) == 3
    assert len(edges) == 2
    assert all(len(edge) == 2 and edge[0] != edge[1] for edge in edges)
    assert all(endpoint in nodes for edge in edges for endpoint in edge)
    assert len({tuple(sorted(edge)) for edge in edges}) == len(edges)

    state = json.loads((ROOT / "tests/fixtures/tiny_initial_state.json").read_text())
    assert state["round_index"] == 0 < state["horizon_rounds"]
    assert state["ledger"]["remaining_budget"] == state["ledger"]["initial_budget"]
    assert state["ledger"]["spent"] == "0.00"
    assert [v["voter_id"] for v in state["voters"]] == nodes
    for voter in state["voters"]:
        assert 0 <= voter["support"] <= 1
        assert 0 <= voter["susceptibility"] <= 1
        assert voter["campaign_exposure"] >= 0
    print("M0 artifact validation: documentation and fixture checks passed")


if __name__ == "__main__":
    main()
