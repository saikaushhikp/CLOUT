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
| INFO-001 | No private-data leak | Serialize observation and mutation-alias test | Week 2 | No hidden fields or state reference |
| ENG-001 | Round order/results | Fixed action-sequence integration test | Week 2 | Effects, expiry, horizon, result are ordered |
| REPRO-001 | Repeatability | Two identical seeded episodes | Week 2 | Equal graph/state/events/results |
| LOGIC-001 | Predicates are meaningful | Valid/invalid trace fixtures | Week 2 | Declared bounded-LTL checks agree |
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
