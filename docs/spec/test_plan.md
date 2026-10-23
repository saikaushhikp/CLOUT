# M0 Test Plan

This plan distinguishes M0 documentation/fixture checks from tests that await
the approved Week 2 implementation. CPU is mandatory for correctness tests;
GPU tests are optional and skip cleanly when unavailable.

| ID | Requirement | Test / fixture | Phase | Expected result |
| --- | --- | --- | --- | --- |
| DOC-001 | Required M0 sections exist | Markdown heading/link audit | M0 | Spec, contracts, decisions, handoff, report exist |
| DOC-002 | Frozen decisions are traceable | Decision-label review | M0 | DEC-002–009 marked approved/frozen |
| FIX-001 | Tiny graph has valid IDs/edges | Parse `tiny_path_graph.json` | M0 | 3 nodes, 2 edges, no loops/duplicates |
| FIX-002 | Tiny state has valid domains | Parse `tiny_initial_state.json` | M0 | Support/susceptibility ranges, nonnegative ledger |
| CFG-001 | Invalid config rejected | Config unit tests | Week 2 | Bad horizons/ranges/seeds fail clearly |
| GRAPH-001 | Graph invariants | Generator/property tests | Week 2 | Valid mapping; no forbidden loops/duplicates |
| MAS-001 | Dynamics are bounded/finite | Hand-computable path/isolate tests | Week 2 | Synchronous update and isolate rule match spec |
| CAMP-001 | Ledger is atomic | Invalid and insufficient-budget tests | Week 2 | No debit/state mutation on rejection |
| INFO-001 | No private-data leak | Recursive serialization and mutation-alias tests | Week 3 | No hidden fields or state reference |
| ENG-001 | Round order/results | Fixed action-sequence integration test | Week 3 | Effects, expiry, horizon, result are ordered |
| REPRO-001 | Repeatability | Two identical seeded episodes | Week 3 | Equal graph/state/events/results |
| LOGIC-001 | Predicates are meaningful | Valid/invalid trace fixtures | Week 4 | Bounded-LTL checks agree with documented finite-trace semantics |
| GPU-001 | CPU/GPU agreement | Marked CUDA test | Later | Within approved tolerance or skipped |
| SCALE-001 | Resource use | Representative-size benchmark | Later | Record wall time/memory; no unmeasured claim |

## M0 validation command

Run `python tests/validate_week1_artifacts.py`. It uses only the standard
library, validates fixture values and required documentation headings, and does
not claim to validate unimplemented dynamics. Future packages should use the
repository test runner once a stack is approved.

## Future edge cases

Test isolates, empty edge sets, disconnected components, minimum/maximum
support, zero susceptibility, zero budget, invalid targets, action order,
terminal boundary, exact 0.5 support, numerical non-finiteness, and data
leakage. Record seeds/configuration/backend/revision for every experiment.

## Week 4 checks (coding-agent run 2026-10-23; reviewer rerun requested)

| Area | Artifacts | Human-run focused command | Status |
| --- | --- | --- | --- |
| Canonical traces / bounded diagnostics | `tests/unit/test_logic.py`, `tests/fixtures/logic/*.json` | `PYTHONPATH=src python -m unittest tests.unit.test_logic -v` | Passed (15 tests) |
| Observation-only baselines | `tests/unit/test_strategies.py` | `PYTHONPATH=src python -m unittest tests.unit.test_strategies -v` | Passed (8 tests) |
| Paired experiments and aggregates | `tests/unit/test_experiments.py` | `PYTHONPATH=src python -m unittest tests.unit.test_experiments -v` | Passed (11 tests) |
| Public mock-client schema | `tests/unit/test_mock_client.py`, `tests/fixtures/public_observation_v1.json` | `PYTHONPATH=src python -m unittest tests.unit.test_mock_client -v` | Passed (7 tests), including CLI smoke, JSON round-trip/additive-field compatibility |
| Full suite and M0 fixture validation | `tests/` | `python tests/validate_week1_artifacts.py` then `PYTHONPATH=src python -m unittest discover -s tests -v` | Validator passed; 69 passed, CUDA tests passed |
| Matched backend sample | `examples/week4_backend_comparison.py` | See [week4_handoff.md](week4_handoff.md) | N=1k CPU/CUDA comparison passed in upgraded `mas`; optional reviewer rerun |

Week 3 reviewer-reported results remain historical evidence in
`docs/spec/reviewer_runbook_outputs.txt`; they are not evidence that any Week 4
check passed. The Week 4 coding agent executed the checks listed above in a
GPU-visible `mas` context. A restricted command sandbox hid NVIDIA devices and
caused CPU fallback during an initial attempt; this was resolved by rerunning
with GPU-visible access. Reviewers should use a GPU-enabled shell/context for
CUDA stages and record the actual selected backend.
