# `Implim_Week4.md` — Dumbocracy Implementation Plan (Week 4)

> **Milestone:** M4 — bounded temporal diagnostics, observation-only strategies, reproducible experiments, client contract integration, and GPU/CPU performance audit  
> **Repository:** https://github.com/saikaushhikp/CLOUT  
> **Baseline branch:** `master` (the coding agent should inspect the current checkout before editing)  
> **Status at Week 4 start:** Week 3 implementation and review are accepted. Any discrepancy between Week 3 plan, implementation, tests, handoff, and self-review is treated as approved Week 3 work plus a forward-looking repair/reconciliation item; it is not grounds to reopen Week 3 approval.  
> **Source of truth:** `docs/spec/technical_spec_v1.md`, `docs/spec/data_contracts.md`, `docs/decisions/decision_log.md`, project assumptions, Week 3 API notes/handoff, and this plan.  
> **Execution rule:** Author tests and give the user/reviewer an exact runbook as well.  
> **Compute rule:** GPU-capable and GPU-preferred for suitable tensor-heavy workloads, with CPU retained as the semantic reference/fallback. The two NVIDIA A5000s have separate VRAM; do not assume pooled memory or introduce multi-GPU execution without measured need.

---

## 0. Mission and week boundary

Week 3 is accepted and complete. The repository contains an authoritative game engine and lifecycle, detached public observation projection, safe public serialization, terminal result computation, reset/replay, trace predicate hooks, CPU/CUDA transition paths, episode and benchmark examples, tests, API documentation, handoff notes, and a reviewer-reported execution record.

Week 4 should turn those accepted foundations into a **logic- and strategy-ready research platform**. Priorities are to carry forward Week 3 items that were explicitly deferred or not evidenced, implement bounded-LTL diagnostics over completed episode traces, add a strict observation-only strategy interface and simple baseline policies, enable fair reproducible paired-seed experiments, finish the mock-client/public schema integration documentation, and make the existing GPU path’s performance limits measurable and honestly documented.

This is not the week to build a production web frontend, change the game model, train an RL agent, implement broad strategy optimization, or create a full exhaustive epistemic model checker.

### Expected capability at the end of Week 4

A developer should be able to:

1. construct a canonical trace from a completed episode and evaluate the M0 bounded-temporal diagnostics with documented boundary semantics;
2. run baseline strategies through a stable strategy contract that exposes only observations, public history, legal actions, and an explicitly seeded RNG;
3. compare those strategies on identical declared seeds/scenarios and produce reproducible per-episode JSONL plus aggregate metrics;
4. consume a sample public observation with the mock client without importing or serializing `GameState`;
5. inspect a human-run CPU/GPU comparison recipe that reports both correctness and resource/time measurements without unsupported speedup claims;
6. understand exactly what carries into Week 5: epistemic model checking, strategy optimization, deeper performance work, integration/beta concerns, and any optional independent confirmation.

---

## 1. Week 3 completion map and carry-forward backlog

The following inventory is grounded in the committed repository structure, `docs/spec/week3_api_notes.md`, `docs/spec/week3_handoff.md`, and `reports/week3_self_review.md`. Recorded test/benchmark results are **reviewer-reported historical evidence**. The Week 4 coding agent may rerun them when useful, but must label any new execution evidence separately with its environment and revision.

| Work item | Week 3 status | Week 4 treatment |
|---|---|---|
| Immutable `GameState`, `GameResult`, `TraceRecord`, `StepResponse`; initialization and round lifecycle | **Done — accepted** | Reuse the engine as the authoritative lifecycle. Add regression artifacts only where needed by the new logic/strategy seams. |
| Accepted/rejected action handling, `Decimal` accounting, effect expiry, broadcast guard, `no_op` dynamics, horizon and strict vote-share semantics | **Done — accepted** | Preserve M0 behavior. Do not rewrite the engine to suit an evaluator or policy. |
| Detached `Observation`, action descriptors, explicit JSON allowlist serializer, safe public events and error codes | **Done — accepted** | Treat this as a hard boundary. Add tests proving strategies and experiments depend only on public information. |
| Reset/replay, terminal result metadata, predicate hooks | **Done — accepted** | Reuse to build canonical trace input and deterministic evaluation. |
| CPU reference and CUDA transition path; sparse device graph prepared per episode | **Done — accepted** | Do not call it optimized solely because it uses a GPU. Document transfers and create matched CPU/GPU run instructions. |
| GPU scale harness for N=1,000 / 10,000 / 100,000 | **Done — accepted** | Preserve it; rerun when useful and label new measurements separately. Historical numbers remain scoped to the reported environment/revision. |
| Week 3 unit/integration tests, full runbook and reviewer-reported 30-test run | **Done as reported** | Preserve historical evidence separately from the Week 4 coding-agent executions recorded in the handoff. |
| Bounded-LTL formula evaluator | **Not done; explicitly deferred** | Implement in MM4.2. `TraceRecord.predicates()` is a source of atoms, not an LTL evaluator. |
| Strategy interface in `src/dumbocracy/strategies.py` and baseline policies | **Not done** | Implement in MM4.3. Do not treat `examples/week3_mock_client.py` as a reusable strategy module. |
| Reproducible paired-seed strategy experiment runner and aggregate report | **Not done** | Implement in MM4.4. |
| Standalone mock-client CLI smoke | **Partially done** — client and unit tests exist, but Week 3 report says its standalone CLI smoke is not recorded | Carry the CLI invocation into the Week 4 runbook and execute it when validating the integration; report whether the current agent or a reviewer performed the run. Improve interface/fixture artifacts only if static inspection finds a concrete mismatch. |
| GPU transfer/performance improvement | **Partially done / limitation documented** — voter support, susceptibility, and effects cross the CPU/GPU boundary; immutable CPU snapshots remain canonical | MM4.6 performs static bottleneck audit and proposes the next safe measurement/decision. Persistent GPU state is not automatic Week 4 scope. |
| Reconciliation between frozen M0 wording (“CPU reference; CUDA optional after profiling”) and the accepted Week 3 GPU-preferred plan | **Partially done** — implementation is accepted, but the docs express different emphasis | Add an auditable decision/compatibility note. Do not silently edit frozen semantics. |
| Full epistemic information-set model checker | **Not done; intentionally deferred** | Do not implement in Week 4. Clarify the interface boundary and put full enumeration/model checking in Week 5+ backlog. |
| Strategy search/optimization, RL, parameter sweeps, production UI/API | **Not done; deferred by roadmap** | Keep deferred. The baseline policies are for integration and reproducibility, not advanced optimization. |

### Carry-forward IDs

- **W3-B01:** bounded-LTL evaluator — target MM4.2.
- **W3-B02:** observation-only strategy interface and baseline policies — target MM4.3.
- **W3-B03:** paired-seed experiment runner — target MM4.4.
- **W3-B04:** standalone mock-client CLI smoke absent from historical Week 3 evidence — retain its command in MM4.5 and record Week 4 agent/reviewer invocations separately.
- **W3-B05:** GPU state-transfer cost and absence of a measured CPU-vs-GPU speedup — target MM4.6 audit/runbook.
- **W3-B06:** M0/CUDA-policy documentation reconciliation — target MM4.0 and MM4.6.

The user has expressly accepted Week 3. Do not use these backlog labels to imply Week 3 is awaiting approval.

---

## 2. Frozen M0 semantics — hard constraints

Read and retain the contracts in `docs/spec/technical_spec_v1.md`, `docs/spec/data_contracts.md`, and `docs/decisions/decision_log.md`.

- The game has one controllable campaign; competing campaigns are deferred.
- Voter support `x_i` is finite and in `[0,1]`. A voter votes for the player iff `support > 0.5`; exactly `0.5` is not a player vote.
- The graph is fixed, undirected, unweighted, sparse, and loop-free. Disconnected components and isolates are valid.
- Synchronous transition semantics remain:

  `m_i = mean(x_j for j in N(i))`, or `0` for an isolate.

  `x_i' = clip_[0,1](x_i + s_i * (alpha * (m_i - x_i) + c_i))`.

  Every voter’s next value uses the same pre-transition support vector; `alpha ∈ [0,1]`; `epsilon_i = 0` in v1.
- `targeted_contact` costs `Decimal("1.00")` per unique valid voter ID; `broadcast` costs `Decimal("5.00")`, affects all voters for this round, and is limited to once per round; `no_op` costs `Decimal("0.00")` but still advances a round and runs the dynamics.
- Rejected actions do not debit resources, modify voter/effect state, or advance the round. Safe audit/rejection summaries can still be recorded as the accepted Week 3 API specifies.
- Effects expire at the end of the current round. A round advances exactly once per accepted action. Budget exhaustion does not end the game early.
- The game terminates iff `round_index == horizon_rounds`; terminal score is vote share; victory iff vote share is strictly greater than `0.5`; equality is defeat.
- Strategies/UI must receive detached `Observation` data and public history only. Do not provide `GameState`, individual support/susceptibility, seeds, private events, effect vectors, internal diagnostics, or hidden graph edges.
- Reproducibility requires the root/derived seeds, config, code revision, software versions, backend/device, and relevant graph summary to be recorded.
- CPU remains the semantic reference/fallback. CUDA can be preferred for workloads that benefit from it, without changing game semantics. Each GPU has its own VRAM; no pooled 48 GB assumption and no implicit distributed execution.
- Temporal properties are evaluated over finite completed fixed-horizon traces. The Week 4 evaluator must not claim to quantify every possible future action sequence.
- Epistemic relation is conceptually induced by equal public observations and compatible public context/history. Full hidden-state enumeration is not a Week 4 requirement.

Any genuine semantic alteration requires a new explicit decision-log entry and compatibility note. Otherwise, implement the frozen contract exactly.

---

## 3. Mini-milestones and review checkpoints

| Mini-milestone | Main result | Required checkpoint |
|---|---|---|
| **MM4.0 — Baseline audit** | `reports/week4_baseline_audit.md`; status/backlog and policy reconciliation | Every Week 3 artifact is mapped; all Week 3 work remains accepted; no frozen semantics changed. |
| **MM4.1 — Canonical trace schema** | Stable time-indexed trace representation and hand-calculable fixtures | Rejections do not become game-time steps; unknown/undefined predicate values are explicit. |
| **MM4.2 — Bounded-LTL diagnostics** | Logic module, tests, `docs/spec/logic_semantics_v1.md` | Every required formula has hand-calculated positive/negative cases and specified boundary behavior. |
| **MM4.3 — Strategy contract** | `strategies.py`, baseline policies, contract tests and API doc | Policy code uses only public observations/history/legal actions and explicit RNG. |
| **MM4.4 — Experiment harness** | Paired-seed episode evaluator, JSONL records, aggregates, tests | Identical declared seeds/scenarios are used across policies; output has complete provenance. |
| **MM4.5 — Client/API seam** | Public fixture/transcript, mock-client documentation and human-run CLI instructions | Client consumes public payload only; no `GameState` dependency or unsupported smoke-test claim. |
| **MM4.6 — Performance audit** | Static bottleneck report and matched CPU/GPU runbook | Claims are separated into measured historical results, static hypotheses, and proposed measurements. |
| **MM4.7 — Final review/handoff** | Week 4 self-review, handoff, updated test plan and Week 5 backlog | Relevant checks are executed, results and environments are recorded, files and diffs are reviewed, and manual review is complete and approved. |

At each checkpoint, the coding agent must report: (1) changed files, (2) completed/partial/not-done items, (3) decisions made, (4) tests added and their execution results, (5) static findings and risks, and (6) the next dependency. A checkpoint must distinguish runtime results from static review and must not claim verification that was not performed.

---

## MM4.0 — Baseline audit and reconciliation

### Tasks

1. Inspect the current repository state from the supplied repository/IDE context before editing. The coding agent may run shell commands and Git queries to collect metadata, inspect dependencies, and check the working tree. Record the command and its result when it affects the report.
2. Read project assumptions, M0 spec/contracts/decision log, Week 3 API notes/handoff, Week 3 self-review, engine, information, MAS, campaign, graph, test fixtures, tests, examples, and `pyproject.toml`.
3. Create `reports/week4_baseline_audit.md` with a Done / Partially Done / Not Done inventory and links to concrete files. Include the table in Section 1 or a more precise version.
4. State explicitly that Week 3 is accepted. Discrepancies are implementation follow-ups, compatibility/documentation cleanup, or Week 5 backlog—not reopened approval items.
5. Reconcile GPU policy documentation without silently changing M0: CPU is the correctness/reference path; the CUDA path is implemented and approved; prefer CUDA for appropriate large tensor workloads; small workloads/orchestration/graph construction may remain CPU-side; performance claims require same-config measurement; GPU memory is per device; multi-GPU is not implied.
6. Record the Week 3 report’s statement that the standalone mock-client CLI smoke was not recorded. Do not reclassify its unit test as CLI evidence.
7. Look for API seams required by Week 4: how `TraceRecord` and `GameResult` become a canonical logic trace, how history is represented to a strategy, and how a policy’s action is replayed through the authoritative `GameEngine`.
8. Do not expand the Week 4 task scope to correct unrelated style issues. If a clear defect blocks a new API contract, record it and make the smallest compatible fix plus a regression test; preserve frozen semantics.

### Deliverables

- `reports/week4_baseline_audit.md`
- One-page list of Week 3 carry-forward IDs W3-B01 through W3-B06
- Documentation change/decision note for the GPU execution policy

### Acceptance gate

- Week 3 completion and approval are taken as given.
- All declared work items have a concrete target and acceptance evidence.
- No semantics or public schema fields are changed without an explicit reason/version note.

---

## MM4.1 — Canonical trace schema and predicate normalization

### Goal

Make the engine trace consumable by a deterministic temporal evaluator without conflating audit-event order with game time and without leaking private state.

### Tasks

1. Inspect `TraceRecord` and `GameEngine` event ordering. Define the evaluator’s input representation explicitly. Prefer adapting the existing trace records rather than creating duplicate state storage.
2. The minimum atomic predicates are:
   - `Terminal`
   - `Victory`
   - `BudgetNonnegative`
   - `StateValid`
   - `ActionAtomic`
3. Define trace time: index `0` denotes the initial state; each accepted action creates the next game-state index; the terminal state is at index `H` for horizon `H`. A rejected action may add an audit event but **must not** create another temporal state index or advance the round.
4. Distinguish event `sequence` identifiers from state-time indices. Never treat a rejection log record as an additional round.
5. Define atom values when they are not meaningful. The current hook exposes `Victory=None` before termination and may expose `ActionAtomic=None` when there is no rejection/atomicity assertion at a step. Preserve or normalize this to a documented explicit `UNKNOWN` representation; do not silently reinterpret unknown as false. `Terminal=False` before termination is not unknown.
6. Specify whether the canonical trace is a sequence of state snapshots, a sequence of predicate maps, or a wrapper containing both. Prefer predicate maps plus public metadata needed to interpret time; never require per-voter values.
7. Include run provenance separately from player-visible observation: root seed, serialized game config, code revision, backend, and relevant package versions. Keep private state out of public records.
8. If the engine’s current trace lacks a required atom or horizon context, add a compatibility adapter or a minimally additive trace field. Avoid changing `Observation` to include hidden values.
9. Define malformed trace behavior: wrong-length arrays, absent required atom, invalid truth type, non-monotonic time, contradictory terminal/horizon indices, or a state after terminal must yield a clear validation error or explicit invalid result—not a misleading pass.

### Fixtures to author (validate via execution)

Add `tests/fixtures/logic/` with small hand-computable JSON fixtures for:

- valid trace with terminal exactly at configured horizon;
- trace in which a safety predicate is false at a known index;
- a trace missing terminal at the configured bound;
- a rejected-action audit event that does not increase temporal state count;
- unknown/malformed predicate values;
- exact vote-share/victory boundary consistent with strict `> 0.5` semantics (the trace records outcomes; it does not recompute hidden voter votes from an observation).

### Deliverables

- Canonical trace schema or adapter, if needed
- `tests/fixtures/logic/*.json`
- `docs/spec/logic_semantics_v1.md` initial section describing indexing, atoms, unknowns, and malformed inputs

### Acceptance gate

A reviewer can hand-calculate each trace. Audit history and temporal state indices are unambiguous. No hidden voter state is required by the evaluator.

---

## MM4.2 — Implement the bounded-LTL diagnostics

### Scope

Implement the small set of formulas approved by M0. This is a deterministic finite-trace diagnostic utility, not a generic theorem prover and not a search over possible futures.

A suggested API, subject to repository style, is:

```python
evaluate_trace(trace, formula, horizon=None) -> LogicResult
evaluate_diagnostics(trace, configured_horizon) -> dict[str, LogicResult]
```

Use a closed AST, enum/operator structure, or strictly validated formula representation. Do not use `eval`, `exec`, arbitrary Python expressions, or permissive string-to-code parsing.

### Required atomic predicates and formulas

Atoms:

- `Terminal`
- `Victory`
- `BudgetNonnegative`
- `StateValid`
- `ActionAtomic`

M0 diagnostics:

1. `G(BudgetNonnegative and StateValid)`
2. `F_<=H Terminal`
3. `Terminal -> (Victory or not Victory)`

You may support a small finite set of boolean connectives needed by these formulas, but do not expand to a full LTL grammar unless there is an immediate, documented requirement.

### Semantics requirements

1. Write exact semantics for the finite trace domain and bounded future interval. State clearly whether bounds count transitions or state indices; use **transition distance** as the documented convention, with the correct endpoint included.
2. `G` checks every applicable state in its declared bounded interval. `F` succeeds if the atom holds at any applicable state inside the bound. A missing required witness at the end of a complete valid trace is a falsification, not an inconclusive result.
3. Keep trace validation distinct from formula evaluation. A malformed trace should not be converted into a property failure that suggests a model violation; return a clear invalid-input result/error.
4. Define propagation for unknown atom values. Prefer three statuses—`SATISFIED`, `VIOLATED`, `UNKNOWN`—if an unknown can affect the result. Do not conflate “not yet defined before terminal” with a demonstrated loss.
5. For `Terminal -> (Victory or not Victory)`, ensure the terminal state’s victory atom is defined and the formula reflects the tautological exhaustiveness diagnostic described by M0. Do not turn this into a victory guarantee.
6. `F_<=H Terminal` checks that a terminal state occurs within the declared horizon. Do not interpret `F_<=H Victory` as a guarantee over policies or every future branch; it is only a trace-occurrence property and is not a required M0 diagnostic in this week.
7. Each `LogicResult` should contain formula/name, status, checked interval, witness or first counterexample index when useful, and an explanatory message. Output must be JSON-safe and deterministic.
8. Keep the evaluator CPU-only: it is small over a finite trace and need not run on GPU.
9. Document whether terminal index `H` makes the final index included when evaluating `F_<=H`; cover `H=0` in unit tests even though game config currently requires at least one round, because the evaluator may receive minimal fixtures.
10. Ensure rejected actions are reflected in `ActionAtomic`/audit diagnostics as appropriate without being counted as an additional time step.

### Tests to author (validate via execution)

- true and false hand-calculated cases for each required formula;
- terminal exactly at `H`, terminal earlier than `H`, missing terminal, and invalid multiple terminal states;
- zero bound and one-step bound behavior;
- rejection audit entries that do not increment temporal time;
- each primitive atom’s truth/unknown handling;
- unknown atom propagation for `G`, `F`, conjunction, disjunction, negation/implication if present;
- invalid predicate name, malformed trace, invalid bound, and wrong value type;
- a clear first counterexample index for a false `StateValid` or `BudgetNonnegative`;
- repeat evaluation over identical trace produces byte-for-byte equivalent JSON-safe result values;
- exact `0.5` vote semantics are consistent with `GameResult.victory` (victory only if share `> 0.5`).

### Deliverables

- `src/dumbocracy/logic.py` (or equivalently scoped module)
- `tests/unit/test_logic.py`, optionally `tests/integration/test_logic_episode.py`
- `docs/spec/logic_semantics_v1.md`
- Intended public exports only, if relevant
- Human-run focused and full-suite test instructions

### Acceptance gate

All three approved M0 formulas are implemented with finite-trace boundaries documented. Logic results describe the provided trace only. No exhaustive future/action branching or “inevitable victory” capability is claimed.

---

## MM4.3 — Strategy contract and baseline policies

### Goal

Provide a minimal reusable strategy interface that respects partial observability. Baselines are integration/evaluation fixtures, not competitive optimized policies.

### Contract

Implement `src/dumbocracy/strategies.py` with an interface compatible with the approved conceptual signature:

```python
Strategy.choose(observation, history, legal_actions, rng) -> CampaignAction
```

The concrete Python protocol/base class may follow existing patterns, but document its types and error behavior.

### Required restrictions

1. `choose` receives only `Observation` (or its public immutable representation), public history, `ActionDescriptor` values, and an explicit RNG passed by the episode runner.
2. Never give a strategy `GameState`, voter tuple/support/susceptibility, hidden seeds, internal events, device tensors, `GameResult` fields unavailable in history, or direct engine references.
3. History should be an immutable sequence of prior public observations and the policy’s public action/result summaries. Explain whether the current observation is already included; do not accidentally duplicate it.
4. Use a deterministic RNG derived from the configured `strategy` seed, preferably with documented per-episode strategy sub-seeds. No global unseeded random functions.
5. Terminal observations should not be passed to strategy selection unless the contract expressly defines a behavior; the episode runner should stop asking for actions once terminal.
6. A policy’s action may be rejected. The runner must pass it to `GameEngine.step`, which remains the authoritative validator. Capture rejects as part of evaluation rather than silently retrying forever.
7. Convert descriptors into action proposals carefully: `targeted_contact` needs one or more unique eligible IDs; `broadcast` has no target; `no_op` has no target. Do not pass descriptor-only metadata as unsupported action parameters.

### Required baseline policies

1. **`NoOpStrategy`** — chooses `no_op` for every active round. It is the simplest control baseline.
2. **`FirstLegalStrategy`** — deterministically selects the first supported action in a documented stable ordering. When targeted contact is selected, choose one eligible public voter ID. Do not rely on dictionary order or unspecified descriptor input order without documenting it.
3. **`SeededRandomLegalStrategy`** — samples from currently supplied public legal descriptors and samples target IDs only from the descriptor’s public eligible list, using the passed RNG. Same inputs and RNG seed must yield the same choices within a recorded software version.

These baselines must not be described as optimal or omniscient. Do not add oracle policies this week unless kept strictly separate as future scope.

### Tests to author (validate via execution)

- all baselines return valid `CampaignAction` types;
- policies choose only a supplied legal action ID and use the descriptor’s eligible IDs;
- targeted actions contain no duplicate or unknown target IDs;
- same seed, same observation/history/descriptors yields same pseudo-random sequence;
- behavior when broadcast/targeted action is unavailable because of budget/guard;
- empty/no-action descriptor list produces a clear exception/result rather than an invalid silent action;
- strategy inputs contain no hidden-state object or banned fields;
- mutating a test copy of history/observation cannot mutate the engine state;
- malicious/invalid policy output is rejected by the engine and does not spend budget or advance a round;
- no strategy selects again after terminal.

### Deliverables

- `src/dumbocracy/strategies.py`
- `tests/unit/test_strategies.py`, optionally `tests/integration/test_strategy_episode.py`
- `docs/spec/strategy_api_v1.md`
- Example showing how the strategy receives only an observation and returns an action payload

### Acceptance gate

The boundary is demonstrably observation-only. Baselines are deterministic as specified and every action still passes through the engine’s existing validation/transition path.

---

## MM4.4 — Reproducible paired-seed experiment runner

### Goal

Evaluate the three baseline strategies fairly on declared finite scenario/seed sets and emit enough metadata for later optimization work. Do not perform broad parameter search or infer statistical significance this week.

### Tasks

1. Implement `src/dumbocracy/experiments.py` or a small dedicated runner module. Add `examples/week4_compare_strategies.py` for a human to run later.
2. Each episode must create a fresh config/state/strategy instance. Make reset behavior explicit and do not leak mutable history, RNG state, events or ledger values from one episode into another.
3. Use paired-seed evaluation: every strategy is run on the same declared root seeds and scenario configurations. Describe this as a controlled comparison, not automatically as statistical significance.
4. Capture per-episode metadata:
   - strategy name/version;
   - root seed and derived subsystem seeds;
   - graph type, `N`, `E`, density, degree summary, component count, edge probability/seed;
   - horizon, initial budget, `alpha`, action parameters and fixed support threshold;
   - code revision or `uncommitted`, Python/NumPy/PyTorch/CUDA versions when available;
   - backend/device;
   - action sequence including rejected proposals where relevant, accepted/rejected counts, and safe trace summary;
   - terminal reason, player votes, electorate size, vote share, victory, score and budget used;
   - timing fields if available, with definitions that distinguish initialization, strategy selection, step/transition and total episode wall time;
   - logic diagnostic summary, when enabled.
5. Emit stable JSONL (one episode record per line) plus a compact aggregate. Aggregate should include sample count, mean terminal score, defined spread (for example population standard deviation), victory rate, mean budget used, reject count/rate, and failed/skipped episode counts. Define all denominators and failure treatment.
6. Never silently omit an episode that raises. Emit an explicit failure record with strategy/scenario/seed and a safe error description; do not include environment secrets or hidden voter state. The aggregate must count such failures separately from completed episodes.
7. Use identical seed sets and graph scenarios across strategies. If each strategy’s decisions change its downstream state (expected), the initial graph and initial voter state must still match for that seed/scenario. Provide a design for sharing immutable initial config/graph/seed while each strategy receives its own independent game episode.
8. Make small workloads default. The normal example should run on small networks and a modest seed list. Large-network performance runs remain opt-in and separated from correctness/strategy evaluation.
9. Reuse the engine’s reproducibility metadata. Do not duplicate metadata computation in inconsistent formats if a shared serializer can serve both episodes and experiments.
10. Define how policy selection handles rejected actions. Minimum acceptable behavior: record the rejection; subsequent round remains the same and runner follows an explicit documented policy for another proposal (e.g., abort episode as policy error or request a bounded number of further actions). Never create an infinite retry loop.
11. Do not add optimization, hyperparameter sweeps, RL, or large combinatorial search.

### Tests to author (validate via execution)

- same strategy/config/root seed gives same action sequence, per-episode record and result;
- paired strategies start from the same initial graph and voter state for a matched seed;
- JSONL output round-trips to valid JSON and has a stable schema;
- aggregation computes known hand-built metrics correctly;
- failed/invalid episode is counted and surfaced instead of silently dropped;
- no cross-episode RNG/history/state leakage;
- accepted/rejected actions and completed episode denominator are correct;
- aggregate results do not expose per-voter hidden data;
- missing PyTorch/CUDA still supports CPU-only strategy experiments if all dependencies allow it;
- a small illustrative record in documentation is labelled **illustrative, not measured** and must never look like actual benchmark evidence.

### Deliverables

- `src/dumbocracy/experiments.py` or a clearly scoped equivalent
- `examples/week4_compare_strategies.py`
- `tests/unit/test_experiments.py`, optionally `tests/integration/test_strategy_episode.py`
- `reports/week4_experiment_template.md` or a versioned JSON schema and example
- Human-run commands, expected fields and diagnostics in `docs/spec/week4_handoff.md`

### Acceptance gate

Experiments are attributable, reproducible, paired by initial seed/scenario, and output per-episode records plus correctly defined aggregates. No scores or performance numbers are invented.

---

## MM4.5 — Finish the public client/API contract

### Tasks

1. Statically inspect `examples/week3_mock_client.py`, `tests/unit/test_mock_client.py`, `Observation.to_public_dict()`, `StepResponse.to_public_dict()`, and `docs/spec/week3_api_notes.md`.
2. Confirm public field names and action payloads agree among code, fixtures, and docs. If there is a mismatch, update the smallest surface and include a compatibility note. Do not change the public schema’s meaning casually.
3. The standalone Week 3 mock-client CLI smoke is not recorded in historical Week 3 evidence. Keep its exact command and expected output shape in the Week 4 handoff; record any Week 4 invocation with actor/date/result, without rewriting the historical transcript or implying a human ran it.
4. Add or update a committed public observation fixture if useful. It must contain only safe public fields: never real hidden voter arrays, susceptibility, seeds, device tensors, private events, or authoritative state.
5. Document the transport-neutral request/response flow:
   - server/engine projects an observation;
   - client serializes only the public observation;
   - strategy/client returns an action proposal with supported keys;
   - engine validates and returns a `StepResponse` public projection;
   - terminal result is serialized only when the accepted terminal step produces one.
6. Make explicit that the mock client consumes a public observation and emits a proposal; it does not initialize or inspect the authoritative hidden game state.
7. Document schema version compatibility: additive non-breaking fields get a minor version update; changed meaning or removed fields require a major version/compatibility note. Hidden state must never become a public field in any version.
8. Include a short serialized request/response transcript in the docs. Example values must be clearly labelled schematic/illustrative if they are not copied from a fixture or human-run record.
9. Keep HTTP/React and production frontend choices out of core engine/strategy modules.

### Tests to author (validate via execution)

- committed observation fixture loads and matches `schema_version` and required fields;
- mock client emits a valid action shape with allowed keys only;
- no import/dependency on `GameState` in mock client path;
- serialization round-trip and compatibility checks;
- unsupported or malformed payloads are rejected safely;
- example transcript matches the documented schema.

### Deliverables

- `docs/spec/strategy_api_v1.md` and/or clarified Week 3 API notes
- safe observation fixture, if justified
- mock-client contract test(s)
- mock-client CLI command, expected behavior, and dated execution record identifying the actor

### Acceptance gate

Public request/response contract is unambiguous and traceable to code. No unrecorded CLI execution is presented as evidence.

---

## MM4.6 — GPU/CPU integration and performance audit

### Goal

Use static evidence to identify the next useful performance work. Runtime evidence may come from the coding agent or a human reviewer, but must identify the actor, selected backend, environment, and result. Do not claim a speedup from CUDA use alone.

### Current accepted architecture

Week 3 uses PyTorch CUDA operations for neighbor aggregation/update and keeps the sparse edge-index/degree representation resident for a CUDA game. The canonical immutable game state uses CPU-side voter tuples, so support/susceptibility/contribution vectors are moved at each transition boundary. This is a documented design limitation, not an automatic correctness defect. Week 3 records a GPU functional smoke/scale run, not a same-workload speedup comparison against CPU.

### Tasks

1. Inspect `src/dumbocracy/mas.py`, `engine.py`, `graph.py`, `examples/week3_gpu_episode.py`, and `examples/benchmark_week3.py` statically, and run the relevant examples or benchmarks when validating behavior and performance. Label all measurements with their environment and revision.
2. Create `reports/week4_performance_audit.md` and classify each statement as:
   - **Historical measured evidence** — existing reviewer transcript only, with its stated environment and revision limitations;
   - **Static observation/hypothesis** — code-derived possibility, not measured;
   - **Proposed experiment** — work to run next, either by the coding agent or a human reviewer, with status recorded after execution.
3. At minimum, consider these hypotheses:
   - CPU-to-GPU transfer of support, susceptibility and effects every round;
   - reconstruction of immutable voter tuples per transition;
   - CPU graph generation/summary/component traversal cost;
   - GPU `index_add_` aggregation and synchronization cost;
   - immutable state/event/trace history copying over long horizons;
   - host synchronization for checking finite values / reading outputs;
   - small graph cases where CUDA launch/transfer overhead may exceed compute cost.
4. For each hypothesis, cite the relevant module/function statically and suggest a measure that can confirm/disconfirm it. Do not write a benchmark result from code inspection.
5. Recommend whether persistent GPU-resident voter state should be deferred to Week 5. Default to **defer** unless a clear use case and safe ownership/transaction design justify a limited experiment. If proposed, it must preserve atomic rollback, reproducible reset/replay, the immutable public/state API, CPU reference parity tolerance, and per-device VRAM limits.
6. Document a matched CPU/GPU comparison recipe for identical graph/config/root seed/action sequence at a small and medium size. Compare final voter state/outcomes and ledger as well as wall time. Use warm-up and device synchronization where required; report initialization separately from transitions; report median and spread rather than a single best time.
7. Report device model, framework/CUDA runtime, driver as available, free/total VRAM at run start, per-device allocated/reserved memory and actual selected backend. A run that `auto` falls back to CPU must be labelled CPU and must not be counted as GPU evidence.
8. Keep 100k CPU/GPU parity optional; a small fixed fixture is the correctness comparison, and separate large-scale measurements are performance evidence only.
9. Make clear that every discovered A5000 has separate memory. Record the baseline/live device-count discrepancy; do not implement distributed execution or assume pooled VRAM in Week 4.
10. Add no optimizations merely for aesthetics. Any performance-driven code change must include a targeted test artifact and be justified by a measurable hypothesis; run the measurement when practical and label it with its actual status and environment.

### Human-run experiment template

| Run | Device/backend | N/E | Seed/config/action sequence | Init time | Median step/spread | Total episode time | Peak allocated/reserved memory | Correctness result | Status |
|---|---|---|---|---|---|---|---|---|---|
| Historical Week 3 records | Reviewer-reported RTX A5000/CUDA 12.1 | 1k/10k/100k; exact config in Week 3 handoff | 20261019; scripted episode and benchmark actions differ | See Week 3 record | See Week 3 record | See Week 3 record | See Week 3 record | Separate small CPU/GPU parity was reported passing | Historical reviewer evidence, not rerun |
| Week 4 matched CPU/GPU | Coding-agent RTX A5000/CUDA 11.8, driver 535.183.01 | N=1k completed; N=10k optional | Same seed/config/actions for each backend | 13.9808 ms CUDA / 17.6509 ms CPU at N=1k | 4.4778 / 8.2564 ms median; spread in JSON record | 0.0379 s CUDA / 0.0609 s CPU | 0.132/2 MiB CUDA; 24,245.5 MiB total | Passed; max support delta 2.22e-16, ledger/result equal | **agent run passed 2026-10-23; optional reviewer rerun** |
| Week 4 matched CPU/GPU follow-up | Coding-agent RTX A5000/CUDA 11.8, driver 535.183.01 | N=1k and 10k | Seed 20261019; same config/actions within each size | N=1k: 12.555/16.878 ms CUDA/CPU; N=10k: 184.820/193.575 ms | N=1k: 4.050/7.653 ms; N=10k: 32.980/71.349 ms CUDA/CPU | N=1k: 0.0338/0.0566 s; N=10k: 0.4034/0.5543 s CUDA/CPU | N=1k: 0.132/2 MiB; N=10k: 1.294/2 MiB allocated/reserved | Both passed; deltas 1.11e-16/2.22e-16; ledger/result equal | Agent run passed 2026-10-23; optional reviewer rerun |

Do not paste historical figures into rows that imply Week 4 was benchmarked. Reference Week 3 report/handoff instead.

### Deliverables

- `reports/week4_performance_audit.md`
- updated human-run CPU/GPU comparison instructions in `docs/spec/week4_handoff.md`
- any code optimization only if clearly justified and paired with authored regression artifacts

### Acceptance gate

No unsupported speedup claim. Historical reviewer-reported measurements, executed Week 4 agent results, proposed follow-up experiments, and static hypotheses are clearly separated. Multi-GPU remains out of scope.

---

## MM4.7 — Documentation, self-review, and Week 5 handoff

### Required review

Read all changed source/tests/docs back statically. Inspect import graph, public exports, dataclass/API types, RNG ownership, trace boundary conventions, JSON schemas, error paths, failed-episode accounting, memory assumptions, and whether examples can work with the documented inputs. You may execute commands to validate them.

### Required deliverables

- `reports/week4_baseline_audit.md`
- `docs/spec/logic_semantics_v1.md`
- `src/dumbocracy/logic.py` (or named equivalent)
- `tests/fixtures/logic/` fixtures
- `tests/unit/test_logic.py` and optional logic integration test
- `src/dumbocracy/strategies.py`
- `tests/unit/test_strategies.py` and optional strategy integration test
- `docs/spec/strategy_api_v1.md`
- `src/dumbocracy/experiments.py` or scoped equivalent
- `examples/week4_compare_strategies.py`
- experiment schema/template/report (illustrative records labelled clearly)
- `reports/week4_performance_audit.md`
- updated `docs/spec/test_plan.md`
- `docs/spec/week4_handoff.md`
- `reports/week4_self_review.md`

Do not create redundant modules merely to match this list; if a file is not justified, explain which existing file contains that responsibility.

### Required runbook policy

The handoff must have a staged, exact **user/reviewer-only** verification sequence. Static inspection should confirm all script paths and flags before documenting them. Include:

1. Week 1 artifact validator;
2. full unit/integration test discovery;
3. focused logic and strategy tests;
4. mock-client CLI using the safe committed public observation fixture;
5. reproducible small/medium paired-seed comparison;
6. Week 3 GPU episode/benchmark as separate opt-in scale commands;
7. CPU/GPU parity test;
8. human diff/status review.

For each command include prerequisites, working directory, expected output fields or pass/fail summary, common likely failure causes and the relevant code/documentation path. Do not invent exact printed outputs or timings.

Record each Week 4 command execution status accurately, including the command, environment, result, and whether user/reviewer confirmation is still useful. The agent may execute anything during testing/validation/benchmarking, including `git diff --check`, formatters, linters, task runners, IDE auto-tests, pre-commit hooks, or CI triggers.

### Week 5 backlog to make explicit

- Full epistemic information-set model checking or state-equivalence analysis.
- Bounded-logic expansion only if new properties are specified and accepted.
- Strategy optimization/search (evolutionary search, RL, gradients, or large parameter sweeps).
- Persistent GPU-resident authoritative state if matched measurements show sufficient benefit and an atomic ownership design is approved.
- Broader performance tuning and GPU/CPU scale envelopes, based on human-run evidence.
- Multi-GPU/distributed partitioning only if one-device limits justify it and project owner approves a design for separate VRAM, communication, and deterministic replay.
- Full app/frontend/service integration, persistence, security hardening and beta workflow.
- Larger statistical evaluation, uncertainty estimates, and claims beyond a small paired-seed baseline.
- Any remaining Week 4 human-run checks or implementation items explicitly not complete.

### Acceptance gate

A human reviewer can tell what is implemented, what is inferred, and what remains unverified without executing the agent’s session. No new claim of correctness/performance is based solely on authored tests or expected outputs.

---

## 4. Week 4 backlog register

| ID | Item | Week 4 start status | Target | Acceptance evidence |
|---|---|---|---|---|
| W3-B01 | Bounded-LTL evaluator | Not implemented in Week 3 | MM4.2 | Hand-calculable fixtures and logic test artifacts; exact finite-trace semantics |
| W3-B02 | Observation-only strategy API / baseline policies | Not implemented in Week 3 | MM4.3 | Contract tests, explicit RNG, no hidden-state input |
| W3-B03 | Paired-seed evaluation runner | Not implemented in Week 3 | MM4.4 | JSONL schema, correct aggregates, same seed/scenario set across policies |
| W3-B04 | Standalone mock-client CLI smoke | Not recorded in Week 3; executed by coding agent in Week 4 | MM4.5 | Safe fixture/runbook plus dated coding-agent CLI output; historical Week 3 transcript preserved |
| W3-B05 | GPU transfer/bottleneck review | Limitation documented; no Week 3 matched CPU/GPU result | MM4.6 | Static audit, Week 3 CUDA scale rerun, Week 4 matched CPU/CUDA N=1k and N=10k; no general speedup claim |
| W3-B06 | M0/CUDA policy reconciliation | Documentation emphasis differs | MM4.0/MM4.6 | Compatibility/decision note; no silent change to frozen M0 semantics |

---

## 5. Explicitly out of scope for Week 4

- Full epistemic state-space enumeration/model checking at scale.
- Known-inevitable-victory reasoning or quantification over all future action branches.
- Strategy optimization, RL, evolutionary algorithms, gradient methods, hyperparameter search or broad parameter sweeps.
- Production UI framework, HTTP service, authentication, database/persistence, deployment.
- Multi-GPU distributed state, automatic VRAM pooling, or GPU partitioning.
- Transition noise or new stochastic model behavior; changing topology; turnout/abstention; multiple parties/campaigns.
- Large-scale statistical claims beyond what the actual recorded experiment design supports.
- Refactoring the accepted Week 3 engine solely to satisfy a preference for different class/module names.

If any item appears necessary to unblock a Week 4 mini-milestone, add a clear proposal and ask the owner for a decision rather than silently expanding scope.

---

## 6. Coding-agent workflow and command execution

### Phase A — static audit

- Read current checkout, M0 files, Week 3 handoff, modules, tests, and report.
- Create the baseline audit and preserve Week 3 approval.
- **You may execute commands to check status, tests, dependencies, or GPU state.** Use repository/UI information and committed records as additional evidence, and record relevant command output.

### Phase B — trace and logic

- Agree the trace shape, atom values, indexing and boundary semantics in documentation.
- Author hand-calculable fixtures and evaluator tests.
- Statically trace formula evaluation against fixtures and run the focused evaluator tests.

### Phase C — strategies and experiments

- Add observation-only policy contract and three baselines.
- Add paired-seed experiment runner and structured records.
- Inspect code paths for RNG isolation, failed episodes, invalid policy actions and output privacy, then run the focused contract and integration tests.

### Phase D — client/performance integration

- Complete mock-client contract docs/fixtures and run the CLI smoke when its prerequisites are available.
- Write GPU/CPU performance audit, clearly distinguishing static hypotheses from coding-agent and human-run evidence.
- Do not optimize without a specific hypothesis and regression artifact.

### Phase E — self-review and stop

- Update handoff, test plan, self-review, and Week 5 backlog.
- Record tests as authored, reviewed, executed, skipped, or failed, including the command and environment where relevant.
- Stop for human review. Do not proceed into Week 5 work in this plan.

### Commands permitted for the coding agent

The coding agent may run shell/terminal, IDE, notebook, CI, build, package, test-runner, lint, formatter, profiling, benchmark and validator commands that operate on the project, as appropriate. This includes:

- tests and `unittest`/pytest discovery;
- `tests/validate_week1_artifacts.py`;
- `examples/week2_gpu_smoke.py`, `examples/week3_gpu_episode.py`, `examples/benchmark_week3.py`, mock-client and Week 4 strategy examples;
- Python imports/snippets that exercise project runtime behavior;
- `git diff --check`, formatters, linters, build/package commands;
- IDE-run tasks, pre-commit hooks and CI workflows.

The coding agent may also read files, review diffs in the Git UI, perform static searches and code reasoning, write new source/tests/docs, and inspect existing committed logs/results. Commands should remain relevant to the implementation and validation work described in this plan.

---

## 7. Required final report

The coding agent’s final report must use this structure:

1. **Status:** complete / partial / blocked.
2. **Starting revision/branch:** report repository metadata obtained from the workspace or by running appropriate Git commands.
3. **Mini-milestones MM4.0–MM4.7:** complete / partial / not done, with artifact paths.
4. **Week 3 backlog IDs W3-B01–W3-B06:** resolution/status and evidence.
5. **Files added/changed:** short, accurate purpose of each.
6. **Logic semantics:** supported formulas, bounds/unknowns, trace-index convention and limitations.
7. **Strategy interface:** baseline definitions and observation-only guarantee.
8. **Experiment outputs:** record schema, paired seed treatment, aggregate metric definitions; no invented scores.
9. **GPU/CPU audit:** historical Week 3 measurements versus the 2026-10-23 coding-agent measurement versus any optional reviewer reruns.
10. **Test status:** test artifacts authored, statically reviewed, and executed where applicable; report pass/fail/skip status and separate historical Week 3 reviewer evidence.
11. **Privacy/reproducibility:** what is kept public/private and what metadata is captured.
12. **Known risks / Week 5 backlog.**
13. **Human review gate:** report that the five-member manual review was completed and approved; do not invent reviewer identities or additional evidence.
14. **Week 5 readiness:** ready / partial / blocked and why.

## 8. Week 4 exit criteria

Week 4 is complete and approved when:

- [x] Week 3 is explicitly treated as approved and the backlog/reconciliation audit exists.
- [x] Trace time indexing is distinct from event sequence numbering.
- [x] The three M0 bounded-temporal diagnostics have implemented, documented semantics and hand-calculable tests.
- [x] Unknown/undefined atoms and malformed traces are handled transparently.
- [x] Three baseline strategies implement an observation-only contract and explicit RNG.
- [x] The experiment runner uses paired initial seed/scenario sets and emits JSONL plus defined aggregates.
- [x] Errors/failed episodes are not silently omitted from aggregate evaluation.
- [x] The public mock-client schema and sample fixture/command are documented correctly.
- [x] The historical Week 3 CLI evidence gap is retained as historical; the Week 4 CLI invocation is separately recorded as a coding-agent run.
- [x] GPU cost audit distinguishes measured facts from static hypotheses and reports matched results without making a general speedup claim.
- [x] The M0/CUDA policy wording is reconciled without a silent semantic change.
- [x] Week 4 docs, test plan, handoff, self-review and Week 5 backlog are complete.
- [x] Relevant checks were executed where configured/applicable: tests, validator, examples, GPU benchmarks, matched comparisons, compilation and diff check. No project formatter/linter is configured; Ruff is absent from `mas`.
- [x] Week 4 execution evidence identifies coding-agent results; the five-member manual review is complete and approved. Optional reviewer reruns remain separate confirmation only.

**Stop condition:** complete the code/test/documentation artifacts and static review for MM4.0–MM4.7; provide the human-run runbook. Do not begin Week 5 tasks in this plan.
