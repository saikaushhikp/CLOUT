# Week 4 Baseline Audit

**Status:** Week 3 is accepted, as directed by `Implim_Week4.md`. The work
below carries accepted APIs forward; it does not reopen Week 3 approval.
Week 4 coding-agent execution was completed in the `mas` environment on
2026-10-23. The full suite passed (69 tests, including both CUDA tests), the
M0 validator passed, focused Week 4 suites passed (41 tests), strategy and mock-client
examples ran, Week 3 GPU scale checks ran on CUDA, and matched Week 4 CPU/CUDA
comparisons passed at N=1k and N=10k. See the detailed command/result ledger in
[`week4_handoff.md`](../docs/spec/week4_handoff.md). The five-member manual
review was completed and approved.

## Starting point and API seam

The working tree at task start was clean apart from the newly created Week 4
plan. The available repository history identifies `master` and Week 3 commit
`6b6ed87`; the Week 2 baseline is `83f749d`. Existing modules provide the
authoritative engine (`src/dumbocracy/engine.py`), immutable `TraceRecord`s,
safe `Observation` projection (`information.py`), pure campaign validation,
seed derivation, and CPU/CUDA MAS paths.

`GameState.trace` is an audit-oriented tuple: an accepted action is recorded at
the resulting round index while a rejected action is recorded at the unchanged
round index. It therefore cannot be interpreted by tuple position as game
time. The logic adapter added in Week 4 folds records by round, emits exactly
`horizon + 1` predicate snapshots, and keeps safe audit sequence metadata
separate. Its exported trace contains no voter-level data.

The strategy seam takes `Observation`, immutable prior public history, the
engine's legal descriptors, and a caller-owned `random.Random`; a policy never
receives engine/state objects. The experiment runner constructs a fresh engine,
policy, history, and RNG per paired seed. Deterministic config/seed derivation
gives each policy the same initial scenario; policies still transition their
own independent state. A rejected proposal is recorded and ends that episode
as a policy failure, avoiding an unbounded retry loop.

## Inventory

| Work item | Status | Concrete evidence / Week 4 action |
| --- | --- | --- |
| Engine lifecycle, atomic actions, outcome, replay | Done — accepted | [`engine.py`](../src/dumbocracy/engine.py), [`test_engine.py`](../tests/unit/test_engine.py); reused unchanged |
| Detached observations and safe public schema | Done — accepted | [`information.py`](../src/dumbocracy/information.py), [`test_information.py`](../tests/unit/test_information.py); reused |
| CPU reference and CUDA transition; per-episode sparse device graph | Done — accepted | [`mas.py`](../src/dumbocracy/mas.py), [`graph.py`](../src/dumbocracy/graph.py); performance limitations audited in [`week4_performance_audit.md`](week4_performance_audit.md) |
| Historical Week 3 full suite / CUDA reviewer run | Preserved as reviewer-reported evidence | [`reviewer_runbook_outputs.txt`](../docs/spec/reviewer_runbook_outputs.txt); Week 4 full-suite and CUDA checks were rerun separately in the upgraded environment |
| Bounded-LTL evaluator | Not implemented in Week 3 | [`logic.py`](../src/dumbocracy/logic.py), [`logic_semantics_v1.md`](../docs/spec/logic_semantics_v1.md), [`test_logic.py`](../tests/unit/test_logic.py) |
| Observation-only strategy interface and baselines | Not implemented in Week 3 | [`strategies.py`](../src/dumbocracy/strategies.py), [`test_strategies.py`](../tests/unit/test_strategies.py), [`strategy_api_v1.md`](../docs/spec/strategy_api_v1.md) |
| Paired-seed runner and JSONL/aggregate schema | Not implemented in Week 3 | [`experiments.py`](../src/dumbocracy/experiments.py), [`week4_compare_strategies.py`](../examples/week4_compare_strategies.py), [`test_experiments.py`](../tests/unit/test_experiments.py) |
| Standalone mock-client CLI smoke | Week 3 historical evidence remains conflicting; Week 4 execution recorded | The safe fixture CLI ran on 2026-10-23 and returned the expected targeted-contact action. The historical Week 3 transcript is unchanged. |
| GPU state transfer / performance evidence | Matched CPU/CUDA comparisons now measured; no general speedup claim | Static audit and executed [`week4_backend_comparison.py`](../examples/week4_backend_comparison.py) at N=1k/10k; persistent GPU state deferred |
| M0 CPU-reference wording vs accepted Week 3 CUDA path | Documentation emphasis differs | `docs/spec/week4_compute_policy.md`; no frozen semantic change or new team approval inferred |
| Full epistemic model checker, optimization, production UI | Not done; deferred | Explicit Week 5 backlog in handoff |

## Carry-forward register

| ID | Resolution |
| --- | --- |
| W3-B01 | Implemented as finite-trace bounded diagnostics; no future-branch search |
| W3-B02 | Implemented with three observation-only baseline strategies and explicit RNG |
| W3-B03 | Implemented paired-seed runner, safe per-episode JSONL records and defined aggregates |
| W3-B04 | CLI schema, fixture, runbook, and coding-agent CLI invocation completed; historical Week 3 transcript gap retained |
| W3-B05 | Static bottleneck audit, GPU-visible Week 3 scale runs, and matched Week 4 CPU/GPU runs at N=1k and N=10k; no general speedup claim |
| W3-B06 | Compatibility note clarifies reference/fallback vs CUDA preference without modifying M0 |

## Compatibility and review notes

M0 formulas and the public observation schema are unchanged. `Victory` remains
undefined (`null`) before terminal; `Terminal` is false before the horizon.
Missing terminal at the configured final index is a valid falsifying trace for
the eventuality diagnostic, while malformed indexing or a terminal followed
by later states is rejected as invalid input. Unknown atom values use explicit
three-valued propagation.

The GPU note is an implementation-policy reconciliation, not a new approved
decision: CPU remains the semantic reference; the already accepted CUDA path
may be selected for appropriate workloads; graph construction/orchestration
may remain on CPU; speedup claims require matched measurements; each A5000 has
separate VRAM; and no multi-GPU execution is implied.

## Human gates

All Week 4 tests and applicable runtime checks have been executed by the coding
agent and are recorded in the handoff. The historical Week 3 mock-client CLI
transcript gap remains historical, but the same safe fixture CLI was executed
and recorded during Week 4. The five-member manual review was completed and
approved. Optional independent reruns do not affect the approval status.
