# `Implim_Week2.md` — Dumbocracy Implementation Plan (Week 2)

> **Milestone:** M1/M2 foundations — executable graph/MAS baseline and campaign/resource foundations  
> **Repository reviewed:** [`saikaushhikp/CLOUT`](https://github.com/saikaushhikp/CLOUT), default branch `master`  
> **Week 1 baseline commit reviewed:** `8adddede97e315b84ffc52b66a62f2c32677551b`  
> **Week 1 status:** M0 documentation and contract package reported approved by the project owner and recorded in the repository.  
> **Primary Week 2 goal:** turn the frozen M0 contracts into a small, executable, GPU-correct foundation ready for Week 3 game-engine integration.

This is an execution plan for the coding agent. The agent must inspect the live checkout, implement the approved contracts, test the behavior, fix reproducible defects, rerun tests, and produce evidence for manual review. It is not permission to change approved semantics silently.

---

## 1. Mission and definition of success

Week 1 produced and approved a substantial specification package, but the reviewed tree is still documentation/fixture-oriented rather than an executable simulator. The current repository contains the technical specification, data contracts, decision register, M0 test plan, hand-off, assumptions, tiny JSON fixtures, and `tests/validate_week1_artifacts.py`. The Week 1 self-review states that no source package, dependency manifest, package configuration, CI, or pre-existing implementation test suite existed at the start of Week 1, and that PyTorch was not installed in the inspected `mas` environment. **Recheck the current live checkout and environment before relying on that snapshot.**

Week 2 must implement the small, testable foundations described below. By the end, a developer should be able to run a GPU scenario that builds and validates a graph, initializes voters, validates and applies a campaign action with exact budget accounting, computes the approved synchronous transition, and verifies the result with automated tests.

### Frozen M0 decisions to implement

Treat `docs/spec/technical_spec_v1.md`, `docs/spec/data_contracts.md`, and `docs/decisions/decision_log.md` as authoritative. In particular:

- **DEC-002:** one player-controlled campaign; competing campaigns are deferred.
- **DEC-003:** a voter's mutable political state is scalar `support` in `[0, 1]`; it is not latent belief or turnout. At the terminal horizon, support strictly greater than `0.5` counts as the player's vote; exactly `0.5` does not.
- **DEC-004:** fixed, undirected, unweighted, loop-free, sparse graph. Required graph inputs are explicit edge lists and Erdős–Rényi `G(N,p)`.
- **DEC-005:** synchronous bounded neighbor-plus-campaign support update, with `alpha` in `[0,1]` and no random transition noise in v1.
- **DEC-006:** deterministic `targeted_contact`, `broadcast`, and `no_op` semantics; costs use `Decimal`; action validation precedes mutation.
- **DEC-007:** private true state is separate from detached player observations. Week 2 must not leak hidden voter arrays, susceptibility, or seed to strategies/UI payloads.
- **DEC-008:** fixed horizon; vote share is score; strictly more than `0.5` vote share is victory; equality is defeat. No early “inevitable defeat” adjudication in v1.
- **DEC-009:** bounded-LTL diagnostics over completed fixed-horizon traces and knowledge based on the approved information relation; a full evaluator is not the Week 2 objective.
- **DEC-001/010:** Python, GPU/CPU correctness reference, deterministic subsystem seeds, record configuration/revision/environment/backend, and use CUDA only after profiling demonstrates value.

### Out of scope for Week 2

Do **not** build the full game-loop/orchestration engine, web UI, strategy optimizer, complete epistemic model checker, or full experiments framework. The dependency graph places graph/MAS/resource foundations before the Week 3 Game Engine Team integration, and puts optimization, UI integration, and formal verification later. Week 2 may prepare clean interfaces and test hooks for those components, but must not quietly expand into them.

---

## 2. Week 1 work map: done, partial, and pending

This status reflects what is present in the reviewed `master` tree, not a claim that every historical command was independently rerun in this conversation. Revalidate the current tree at the start of the agent run.

| Week 1 deliverable/requirement | Status at reviewed commit | Evidence and Week 2 treatment |
|---|---|---|
| Technical Specification v1 | **Done as specification** | `docs/spec/technical_spec_v1.md` freezes loop ordering, graph, dynamics, action, information, outcome, logic and reproducibility semantics. Implement those contracts; do not treat them as executable behavior yet. |
| Data/API contracts | **Done as documentation; code pending** | `docs/spec/data_contracts.md` defines conceptual object fields and signatures. Implement a minimal compatible subset in Week 2. |
| Decision register | **Done** | `docs/decisions/decision_log.md` records DEC-002–DEC-009. Any semantic change requires a new reviewed decision. |
| Project assumptions | **Done** | `docs/Assumptions/PROJECT_ASSUMPTIONS.md` records server/GPU and coding-agent/review assumptions. |
| M0 test plan | **Done as plan; future tests pending** | `docs/spec/test_plan.md` maps requirements to future test phases. Week 2 implements the graph/config/MAS/action tests it marks for Week 2. |
| Tiny path graph fixture | **Done** | `tests/fixtures/tiny_path_graph.json` contains a 3-node, 2-edge path. Week 2 tests must use it in real graph code. |
| Tiny initial-state fixture | **Done** | `tests/fixtures/tiny_initial_state.json` contains three support/susceptibility values and a budget ledger; it is data only and does not implement dynamics. |
| M0 artifact validator | **Done for documentation and fixture sanity only** | `tests/validate_week1_artifacts.py` checks headings, some decisions, and fixture fields. Keep it and rerun it. It is not a simulator test suite. |
| Week 1 self-review | **Done as reported evidence** | `reports/week1_self_review.md` records the environment snapshot and M0 validation. Re-run relevant checks on the current server; record differences. |
| Week 1 hand-off | **Done as documented hand-off** | `docs/spec/week1_handoff.md` declares contracts frozen and Week 2 eligible to implement them. |
| Executable typed configuration and seed derivation | **Not present in reviewed tree; Week 2 backlog** | Implement validation and deterministic subsystem seed derivation. |
| Graph object, graph validators, edge-list/ER generators | **Not present in reviewed tree; Week 2 backlog** | Implement sparse graph representation, canonicalization, graph metrics, and deterministic generators. |
| Executable voter state and synchronous dynamics | **Not present in reviewed tree; Week 2 backlog** | Implement DEC-005 exactly and test against hand-computable cases. |
| Campaign action/resource subsystem and atomic ledger | **Not present in reviewed tree; Week 2 backlog** | Implement pure validation, exact `Decimal` costs, effect construction, and atomic rejection behavior. |
| Unit/integration tests of actual dynamics/actions | **Not present in reviewed tree; Week 2 backlog** | Add test runner/config only as needed and build a GPU-first automated suite. |
| Complete game engine, observation projection, result lifecycle | **Specified, not implemented; Week 3 workstream** | Week 2 keeps model/action interfaces compatible and prepares boundary tests. The complete engine and end-to-end information flow remain Week 3. |
| Temporal/epistemic evaluator | **Specified at semantic-contract level; later implementation** | Week 2 must not claim a model checker is delivered. Preserve the defined state/result fields and leave explicit hooks only where useful. |
| PyTorch/CUDA implementation and scale benchmarking | **Not done; intentionally conditional/deferred** | Reported PyTorch absence is not itself a defect. First make GPU correctness complete, then benchmark the actual workload before adding GPU dependencies. |
| Package metadata, lockfile, CI | **Not present in reviewed tree at snapshot; recheck** | Add minimal reproducible project/test setup if needed. CI is optional if it would distract from the model foundations, but commands must be reproducible. |
| Five-member M0 review | **Reported approved by project owner** | Repository records approval, not individual names. A new human review of Week 2 code and test evidence is mandatory before Week 3 depends on it. |

### Explicit carry-forward backlog from Week 1

These are implementation tasks unlocked by M0 approval, not necessarily Week 1 failures:

1. Translate conceptual contracts into executable config, graph, voter-state, action, ledger and error interfaces.
2. Implement sparse graph representation, edge-list construction, and deterministic Erdős–Rényi generation.
3. Implement the approved synchronous voter update and numerical-safety behavior.
4. Implement targeted contact, broadcast, no-op-compatible action semantics, exact budget accounting, effect construction, and rejection atomicity.
5. Turn the checked-in fixtures into real unit/integration tests.
6. Add tests for isolates, disconnected graphs, p-boundaries, exact threshold behavior, invalid inputs, non-finite values, seed reproducibility, and immutability after rejected actions.
7. Provide a runnable GPU smoke demo and exact setup/test instructions.
8. Produce Week 2 self-review and stable interface hand-off for the Week 3 game engine.

---

## 3. Non-negotiable execution rules

1. Read `docs/Assumptions/PROJECT_ASSUMPTIONS.md`, `docs/spec/technical_spec_v1.md`, `docs/spec/data_contracts.md`, `docs/spec/test_plan.md`, and `docs/decisions/decision_log.md` before editing.
2. At the start, inspect latest `git status`, current branch/head, and repository tree. Do not assume the reviewed SHA is still HEAD.
3. Preserve DEC-002–DEC-009 exactly. If implementation exposes a genuine ambiguity, stop, explain options, and request/record a reviewed decision before changing behavior.
4. Do not delete existing fixtures or weaken the M0 validator just to get a green test run.
5. Do not mark tests as passed if they were not run; do not claim a full game engine or logic evaluator is complete in Week 2.
6. CPU/GPU correctness tests must work without CUDA. GPU availability must not be a prerequisite for normal development or CI.
7. Do not pass `GameState`, raw private voter arrays, hidden seed, or susceptibility into strategy/UI-facing observations.
8. Do not introduce stochastic transition noise: the approved v1 equation uses `epsilon_i = 0`.
9. A rejected action must leave every affected state field unchanged: budget, spent amount, campaign effects, voters, round state and accepted-action history.
10. Do not use a dense `N x N` adjacency matrix as the core representation for large sparse graphs.
11. Two A5000 GPUs have distinct device memory; 48 GB combined is not a single automatically pooled allocation. Do not implement multi-GPU execution in Week 2.
12. Coding-agent completion is not acceptance. The five members must manually review code, tests, demo outputs and the self-review before Week 3 engine work relies on these interfaces.

---

## 4. Repository and dependency setup

### 4.1 Reconcile before writing code

Run and record at least:

```bash
git status --short --branch
git rev-parse --show-toplevel
git rev-parse HEAD
git log -5 --oneline
find . -maxdepth 4 -type f -not -path './.git/*' | sort
python --version
python -m pip --version
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
```

If a command is unavailable, record that fact rather than masking it. Inspect the existing test/configuration tools and run the baseline checks before installing anything:

```bash
python tests/validate_week1_artifacts.py
git diff --check
```

Do not expose SSH keys, credentials, tokens, or secret environment values in logs/reports. Do not update system packages or CUDA drivers as a side effect of project setup.

### 4.2 Minimal Python setup

Use existing Conda environment especially for this project named as `mas` and can be used via `conda activte mas`. and python at (`/data/ai23btech11017/miniconda3/envs/mas/bin/python`)

If the repository still lacks package metadata/test tooling, add a minimal reproducible setup using one established tool and one test runner. Prefer a small, local environment and pinned direct dependencies. Do not introduce multiple package managers. Keep the model runnable on GPU.

Avoid dependency bloat:

- The standard library is sufficient for some validation and exact-Decimal accounting.
- NumPy is a reasonable numerical backend if available and accepted by the project; record its version.
- NetworkX may be used for graph analysis/generation if justified, but the model’s canonical representation must remain compact and sparse. Do not depend on NetworkX object traversal in the per-round hot path for large graphs.

---

## 5. Proposed module layout and boundaries

Preserve any useful structure that has been added since Week 1. If no source package exists, a compact layout is appropriate:

```text
src/dumbocracy/
  __init__.py
  config.py       # validated config and subsystem seed derivation
  graph.py        # graph object, builders, validation and metrics
  mas.py          # voter schema and synchronous transition
  campaign.py     # action schema, validation, ledger and effects
  errors.py       # stable project errors/error codes
  types.py        # shared types only if they avoid circular imports

tests/
  unit/
    test_config.py
    test_graph.py
    test_mas.py
    test_campaign.py
  integration/
    test_week2_baseline.py
  fixtures/
    README.md
    tiny_path_graph.json
    tiny_initial_state.json
  validate_week1_artifacts.py

examples/
  week2_gpu_smoke.py

docs/
  spec/week2_api_notes.md
  ...
reports/
  week2_self_review.md
```

This is a proposed organization, not a requirement to create empty modules. Prefer small typed dataclasses/immutable config records and pure functions where appropriate. Keep graph, MAS and campaign modules independent of the future UI. Avoid circular imports; the Week 3 engine should orchestrate these interfaces rather than having model components import the engine.

---

## 6. Task A — Configuration and reproducible seeds

### Required behavior

Implement the approved `GameConfig`, `DynamicsConfig`, `ActionConfig`, `GraphSpec`, and seed configuration contract, either with these exact names or documented equivalents. Validate at construction/boundaries:

- `node_count >= 1` and `horizon_rounds >= 1`;
- initial budget and monetary costs are valid, finite, nonnegative `Decimal` values;
- `alpha` is finite and belongs to `[0,1]`;
- ER `edge_probability` is finite and belongs to `[0,1]`;
- campaign deltas are finite numbers with documented support-units-per-round semantics;
- graph kind is supported;
- transition noise mode is exactly `"none"` in v1;
- required seed inputs are valid integers under a documented policy.

Make validated configuration immutable or treat it as immutable after creation. Fail before allocating/partially creating model state if configuration is invalid.

### Seed derivation

Derive distinct deterministic subsystem seeds from the root seed for `graph`, `initial_state`, `transition`, and `strategy`. The derivation method must be explicit, stable, and independent of global RNG state and Python’s randomized `hash()`. Separate generators/streams should be used where random choices are made. Transition seed exists for reproducibility/future compatibility but must not introduce random noise in this v1 model.

### Tests

- A known valid configuration is accepted.
- Invalid node count, horizon, graph kind, noise mode, alpha, p and budget are rejected clearly.
- Reject NaN/Inf in all relevant floating parameters.
- Currency validation uses `Decimal`, not binary float arithmetic.
- Same root seed produces the same named derived seeds and same graph/initialization; different roots should not all collapse to the same set.
- Invalid config fails before graph/voter state creation.

---

## 7. Task B — Sparse electorate graph

### Required graph functionality

Implement a stable `ElectorateGraph` compatible with `docs/spec/data_contracts.md`, including stable external node IDs mapped to contiguous internal indices. Implement:

1. Explicit edge-list construction, including parsing the checked-in `tiny_path_graph.json` fixture.
2. Erdős–Rényi `G(N,p)` generation for undirected, unweighted graphs: sample each possible unordered pair once with probability `p`; never generate self-loops.
3. Input validation, canonical edge ordering, duplicate detection, and read-only/immutable graph data after creation.
4. Graph summary: node count, edge count, density, degree summary and connected-component count.

### Invariants

- Node IDs are unique; mapping is stable and round-trips.
- Endpoints are valid and refer to known nodes/indices.
- Each undirected pair is canonicalized as `(min_index, max_index)`.
- No loops or duplicate undirected edges.
- The graph is fixed for an episode.
- Isolates and disconnected components are legal and must be reported, not silently repaired.
- Empty edge sets are legal; `p=0` creates no edges. For small N, `p=1` creates the complete graph without loops.
- Reject malformed edges and unknown endpoints with documented errors.

### Representation/performance

Use an edge list, CSR, or another appropriate sparse representation. Avoid an `N x N` dense adjacency matrix in the core storage path. For repeated neighbor averaging, choose a clear sparse/vectorized GPU technique; maintain correctness and simple interfaces over speculative optimization. Expensive spectral/Laplacian statistics are optional diagnostics, not required on every initialization.

NetworkX can be used for non-hot-path components if dependency cost is justified. Do not make Python graph-object traversal the only route to a large repeated-state-update implementation.

### Tests

- Fixture has exactly 3 nodes and 2 edges and maps `v0—v1—v2` correctly.
- ID/index mapping stable and correct.
- Reject malformed, duplicate, self-loop and unknown-endpoint edges.
- One-node graph, no-edge graph, isolates and disconnected components are valid.
- `p=0` produces no edges and `p=1` produces the complete small graph.
- Equal graph seed/config yields equal edge sets; tests must not assume an arbitrary ER graph is connected.
- Density, degrees and component count match small hand-calculated examples.
- Include at least one moderate sparse-graph test that exercises sparse storage and does not require dense adjacency allocation.

---

## 8. Task C — Voter-state initialization and synchronous MAS dynamics

### Exact approved equation

For voter `i` with support `x_i`, susceptibility `s_i`, neighbors `N(i)`, and the current round’s campaign contribution `c_i`, calculate:

\[
m_i=\begin{cases}
\frac{1}{|N(i)|}\sum_{j\in N(i)}x_j,& |N(i)|>0,\\
0,& |N(i)|=0.
\end{cases}
\]

Then:

\[
x_i'=\operatorname{clip}_{[0,1]}\left[x_i+s_i\left(\alpha(m_i-x_i)+c_i\right)\right],
\]

where `alpha` is in `[0,1]`, and `epsilon_i = 0` (no stochastic noise in v1). This is a frozen decision from `docs/spec/technical_spec_v1.md` and DEC-005. Do not substitute a standard consensus update, sigmoid, threshold model, asynchronous scheme, or stochastic transition.

### Implementation requirements

- `support` and `susceptibility` are finite in `[0,1]`.
- Voter IDs are unique and correspond exactly to graph node IDs.
- All `m_i` values and all output supports are computed from the **same pre-transition support vector**. Avoid mutating voter supports one-by-one while traversing nodes; that would silently change synchronous semantics.
- Isolates use `m_i = 0`, as approved, even if another rule might seem intuitive.
- `c_i` represents support units added for this round only; effects expire after the transition.
- Apply clipping after computing the full update.
- Invalid/non-finite inputs fail clearly. If a computed output becomes non-finite, return/raise `NUMERICAL_INSTABILITY`; original state remains unchanged.
- Prefer returning a new state/vector or another clearly documented transactional boundary.
- Distinguish continuous support from a vote: only at terminal scoring does `support > 0.5` count for the player. `support == 0.5` is not a player vote.

### Hand-computable test cases required

Use tiny vectors where expected outputs are computed and recorded in test comments or test documentation:

- the three-node path fixture;
- an isolate and an edgeless graph;
- `alpha=0` and `alpha=1`;
- zero susceptibility;
- support at lower/upper bounds;
- campaign effects causing lower/upper clipping;
- exact support values `0`, `0.5`, and `1`;
- invalid non-finite inputs and numerical-failure atomicity;
- a test that would fail under in-place asynchronous updates, proving all outputs use the old support vector.

For the core transition, keep a straightforward CPU reference implementation. If a vectorized path is added, test it against the reference using identical inputs and a documented numeric tolerance.

---

## 9. Task D — Campaign actions, resource ledger and atomicity

### Frozen action semantics

| Action | Cost | Effect | Constraints |
|---|---:|---|---|
| `targeted_contact` | `1.00` per unique voter ID | Adds `delta_target` to each selected target’s campaign contribution for the current round | Target IDs must exist; affordable; unique targets cannot be double-charged/applied |
| `broadcast` | `5.00` | Adds `delta_broadcast` to every voter’s current-round campaign contribution | At most once per round; affordable |
| `no_op` | `0.00` | No campaign effect; engine-level action that advances the round | Legal before terminal state; round advancement belongs to the future engine |

`no_op` is an engine action in the semantic model, not a campaign resource. It may be represented by the same action union for interface convenience, but document the distinction. Week 2 need not implement the entire round engine; it must provide enough action validation/effect output for Week 3 to enforce the approved order.

### Required implementation

- Typed `CampaignAction` schema and normalized validation result.
- Pure action validation. It must not debit funds, mutate support, apply effects, or append an accepted-action event as a side effect.
- `Decimal` for initial/remaining/spent budget and all monetary costs. Never use binary floats for currency calculations.
- Targeted cost is calculated from unique valid target IDs. Either reject duplicates or normalize them to a unique set, but the choice must be deterministic and cannot double-charge/double-apply. Document it in `docs/spec/week2_api_notes.md` or an equivalent implementation note.
- Accepted action follows validate-then-commit: debit and effect registration are atomic. Failure midway must roll back/avoid all partial effects.
- Rejected action returns a stable error code and leaves budget, spent amount, voter support, active effects, round index and accepted-action history unchanged.
- Effect values are deterministic and expire after that round’s voter transition.
- Broadcast-at-most-once-per-round must be enforceable. Expose the needed context/flag so Week 3 engine controls the round boundary; do not hide it in global mutable state.
- Unknown/malformed action, invalid target, repeated broadcast and insufficient budget reject cleanly.
- Internal event attribution must be available without exposing hidden voter state through public logs.

### Atomic rejection test pattern

For each rejection case:

1. Create a known-valid state and action context.
2. Snapshot/fingerprint all fields the subsystem could mutate (ledger, active effects, support state, round context and event history).
3. Submit the invalid action.
4. Check the exact expected rejection code.
5. Assert the state fingerprint is unchanged.

Test pure validation separately from commit. Specifically prove an insufficient-budget targeted contact does not debit a partial amount and repeated broadcast does not consume money or change effects on rejection.

---

## 10. Task E — Interface alignment for Week 3 engine/information integration

The Week 3 Game Engine Team needs stable contracts from the MAS API, campaign API, and observation API. Week 2 owns config, graph, voter dynamics, and campaign/resource primitives; the full game lifecycle and information-model integration remain Week 3 scope.

Ensure that:

- `GameState` will be owned by the future engine; model/campaign code does not introduce a UI dependency.
- `Observation` is a detached projection. It must not contain voter-support arrays, susceptibility, hidden seeds, diagnostics, or a reference to `GameState`.
- Strategy-facing types receive only observation/history/legal-action details, never authoritative state.
- Dynamics accepts explicit graph, voter state, campaign contribution, and parameters; it does not read global mutable state.
- Campaign validation/commit returns explicit cost/effect/result data so the future engine can execute the defined round order.
- Broadcast usage tracking and effect expiry have clear ownership and a hand-off contract.
- Stable errors and return types are documented; modules do not import the future engine.

Add `docs/spec/week2_api_notes.md` to record the actual implemented signatures, ownership, mutation guarantees, error codes, dependency choices, and any integration caveat not already obvious from `data_contracts.md`. Update `data_contracts.md` only for implementation-level clarifications or approved compatibility changes; avoid duplicating the canonical spec.

If the Week 2 scope does not yet include an actual Observation constructor, mark the hidden-state leakage test as a Week 3 integration gate. Do not claim it passed. If you do implement a small observation projection for boundary testing, ensure it matches DEC-007 and keep it separate from full lifecycle/orchestration.

---

## 11. Task F — Structured errors, runnable GPU demo and reproducibility

### Stable errors

Implement documented error categories only where relevant:

- `INVALID_CONFIGURATION`
- `INVALID_ACTION`
- `INVALID_TARGET`
- `INSUFFICIENT_BUDGET`
- `MALFORMED_OBSERVATION`
- `NUMERICAL_INSTABILITY`
- `INTERNAL_FAILURE`

Keep exception behavior predictable and messages useful without revealing private state. Preserve useful exception chaining. Do not convert every failure into a generic success-like value.

### Minimal smoke demo

Add `examples/week2_gpu_smoke.py` or an equivalent runnable example that:

1. Loads the checked-in tiny graph/state fixture (or reconstructs it explicitly through the same validated interfaces).
2. Validates config and graph.
3. Submits one approved campaign action.
4. Applies its current-round contribution to the approved synchronous transition.
5. Shows the graph summary, before/after support, budget/spent amount, action result and reproducibility metadata.
6. Tries at least one rejected action and demonstrates that its state/budget/effects remain unchanged.
7. Exits with a non-zero status on unexpected failure.

This is a developer smoke demo, not a public UI or complete game loop. Clearly label any printed true state as local developer-demo output; never reuse that output as the public Observation contract.

### Reproducibility metadata

Record root and derived seeds, validated config, git revision (or `uncommitted` for a dirty tree), Python and dependency versions, backend/device, graph metrics and chosen action sequence. No unseeded global randomness. Explicit fixtures remain explicit: their `seed_provenance` is metadata, not a requirement to regenerate the literal fixture via an RNG.

---

## 12. GPU, vectorization and scalability policy

The project assumption is a remote server with 2 × NVIDIA A5000 and 48 GB total VRAM, but memory is per device. The Week 1 self-review says PyTorch was absent in the inspected environment. PyTorch has been sucessfully installed in the `mas` environment. Recheck current state instead of blindly installing a large stack.

### Required approach

1. CPU/GPU reference implementation first.
2. Choose the simplest representation that satisfies correctness and an initial useful scale. Use sparse graph storage; avoid dense adjacency for large sparse graphs.
3. Profile graph generation, one dynamics step, and a repeat/batch of steps separately on representative synthetic graph sizes.
4. Record N, E, density, update implementation/backend, runtime, memory method, repeat count, warm-up policy and software versions.
5. If vectorization materially improves CPU/GPU performance, implement it and compare against the reference. Keep readable correctness tests.
6. Add a CUDA/PyTorch implementation only if a benchmark shows a meaningful benefit and a supported setup is available. It must remain optional for core tests and be checked against the CPU/GPU reference within a documented tolerance.
7. Do not introduce DDP, multi-GPU sharding, or "48 GB single tensor" assumptions in Week 2. Separate independent runs across GPUs only in a later experiment design with explicit scheduling and per-device memory accounting.

GPU availability is not a Week 2 acceptance requirement. Correctness, atomicity, reproducibility, sparse storage and interface stability are.

---

## 13. Testing and validation requirements

Use an existing repository test runner if one has emerged. Otherwise select a single minimal test runner and document exact commands.

### 13.1 Preserve M0 checks

Run and preserve:

```bash
python tests/validate_week1_artifacts.py
git diff --check
```

The first must continue validating M0 docs/fixtures. Update its expected strings only if a genuine approved documentation change makes that necessary; never remove a check solely to pass validation.

### 13.2 Unit tests

**Configuration/seeds**
- valid and invalid cases;
- finite numeric validation and `Decimal` budget/cost handling;
- same-root subsystem-seed reproducibility and distinct subsystem streams.

**Graph**
- parse fixture and preserve node mapping;
- malformed/duplicate/self-loop/unknown-endpoint rejection;
- one-node, edgeless, isolate and disconnected graph behavior;
- ER p=0 and p=1 boundaries;
- same seed gives same ER edge set;
- hand-checked graph metrics.

**MAS dynamics**
- hand-computed path example;
- isolate uses zero neighbor mean;
- synchronous update ordering;
- alpha endpoints and zero susceptibility;
- clipping at 0 and 1;
- non-finite invalid input/output error handling;
- original state unchanged on failure;
- threshold vote semantics remain strict `> 0.5` when scoring helper is included.

**Campaign/resources**
- action schema and validation purity;
- targeted-contact cost per unique ID;
- broadcast cost/effect and once-per-round guard;
- no-op is zero-cost/no-effect at the campaign layer;
- exact Decimal ledger invariant;
- unaffordable, malformed, unsupported and invalid-target rejection;
- every rejection preserves state fingerprint;
- every accepted action consumes exactly the expected amount and returns the expected effects.

### 13.3 Integration test

Create a fixture-driven integration test from config/graph/voter state through action validation, accepted current-round effect and one synchronous update. Assert exact/hand-derived results, the resulting ledger, and unchanged state after a separately attempted invalid action. Do not merely assert “no exception was raised.”

Run the same seeded/configured scenario twice and compare graph, initial state, resulting support, ledger, action result, and deterministic metadata (excluding inherently variable fields such as wall-clock timestamp, which should not be part of equality assertions).

### 13.4 Property-style invariants

Use Hypothesis only if it is already present or can be added simply; deterministic parameterized loops are acceptable. Verify:

- every successful support update is finite and within `[0,1]`;
- graph endpoints are valid and there are no duplicate/loop edges;
- `0 <= remaining_budget <= initial_budget`, `spent >= 0`, and `remaining_budget + spent == initial_budget` exactly;
- targeted targets cannot double-apply;
- rejected actions preserve the complete relevant state fingerprint;
- identical seed/config gives identical initialized graphs/results.

### 13.5 Future/later tests that must not be falsely marked Week 2 complete

- full round-order/terminal-result integration belongs to Week 3;
- detached observation serialization/leakage requires an actual observation projection (Week 3 if not implemented now);
- bounded-LTL/epistemic evaluation is later work;
- CPU/GPU parity is required only if a GPU path is actually introduced;
- full large-scale performance claims require documented benchmarks.

---

## 14. Coding-agent execution sequence

### Phase 0 — Revalidate checkout, assumptions and baseline

- Inspect the current branch/head/tree and dirty working tree.
- Re-read approved spec, data contracts, test plan, decision log and assumptions.
- Run the M0 validator and `git diff --check` before edits.
- Reconfirm Python/dependency/GPU availability.
- Record baseline test outcomes.

**Gate:** if current docs conflict or an unreviewed semantic change is already present, stop and report it instead of coding around it.

### Phase 1 — Implementation design and setup

- Pick the smallest viable package/testing setup consistent with the repository.
- Document intended signatures and error ownership in `docs/spec/week2_api_notes.md`.
- Flag actual spec ambiguity to the human owner before changing behavior.

### Phase 2 — Config and graph

- Implement validated config and deterministic subsystem seeds.
- Implement graph representation, edge-list builder, ER builder, validation and summaries.
- Add/execute unit tests before proceeding.

### Phase 3 — Voter state and dynamics

- Implement validation/initialization and the exact approved synchronous equation.
- Include a simple reference calculation and hand-computed expected values.
- Add numerical stability and failure-atomicity tests.
- If optimized/vectorized code is added, compare it to the reference.

### Phase 4 — Campaign and resource layer

- Implement typed actions, pure validation, exact `Decimal` ledger, effect construction and atomic commit boundary.
- Add explicit context for repeated-broadcast protection and document the Week 3 round-reset ownership.
- Write rejection-state-fingerprint tests.

### Phase 5 — Fixture-driven integration and smoke demo

- Use both checked-in fixture files in real code/tests.
- Execute action/effect/dynamics path on GPU and assert numeric results.
- Demonstrate rejected-action atomicity.
- Check interfaces and imports are ready for the engine; do not implement the complete engine.

### Phase 6 — Self-review and human hand-off

- Run the entire suite from a clean/reproducible command sequence, more than once.
- Inspect the diff for unrelated changes, unseeded RNG, dense graph allocations, float currency arithmetic, hidden-state leaks, duplicated dependencies, and tests that only check “does not crash.”
- Fix reproducible defects that do not require a new design decision; document any that do.
- Write `reports/week2_self_review.md` and update API/handoff documentation.
- Stop for review by all five team members; do not begin Week 3 implementation.

---

## 15. Required Week 2 deliverables

- [ ] A minimal, documented Python project/test setup, if the current repository still lacks one.
- [ ] Executable validated config and subsystem seed derivation.
- [ ] Sparse graph type with edge-list and Erdős–Rényi `G(N,p)` generation.
- [ ] Graph invariant validation and summary metrics.
- [ ] Voter-state validation compatible with `tiny_initial_state.json`.
- [ ] The exact approved synchronous update equation.
- [ ] Campaign action schema and pure validation.
- [ ] Exact `Decimal` resource ledger.
- [ ] Targeted-contact and broadcast effect handling with broadcast guard; no-op-compatible contract for Week 3 engine.
- [ ] Atomic accepted-action commit and proof that rejected actions do not mutate state.
- [ ] Stable documented error codes.
- [ ] Unit tests for config, graph, dynamics, actions and ledger.
- [ ] Fixture-driven GPU integration test with hand-derived expected values.
- [ ] Determinism/reproducibility tests.
- [ ] Runnable GPU smoke demo.
- [ ] Original M0 artifact validator remains passing.
- [ ] `docs/spec/week2_api_notes.md` or equivalent stable interface notes.
- [ ] `reports/week2_self_review.md` with commands, real test results, environment, limitations and open risks.
- [ ] Week 3 hand-off that calls out any observation/engine integration dependency.

---

## 16. Week 2 exit criteria

Week 2 may be marked **ready for human review** only when all applicable checks are satisfied:

- [ ] The approved M0 specification and decisions remain unchanged unless a reviewed decision record explicitly allows a change.
- [ ] M0 artifact validation and `git diff --check` pass.
- [ ] Configuration and graph construction run from a clean checkout.
- [ ] ER generation is repeatable for a fixed seed and respects all graph invariants.
- [ ] Dynamics match hand-computed expected values, use synchronous updates, obey isolate semantics, clip correctly and fail atomically on numerical instability.
- [ ] Resource validation is pure; currency uses `Decimal`; budget invariants hold exactly.
- [ ] Accepted actions have correct cost/effect; rejected actions preserve ledger, effects, voters and round context.
- [ ] A fixture-driven GPU integration test passes and checks actual numeric results.
- [ ] Repeated deterministic runs produce equal model outputs.
- [ ] Core correctness suite requires no GPU.
- [ ] Any performance claim includes benchmark evidence; GPU use is justified or explicitly deferred.
- [ ] The smoke demo runs with documented instructions.
- [ ] Hidden state is not wired into public observation/strategy interfaces.
- [ ] Self-review records exact commands, real pass/fail/skip counts, environment, limitations and non-tested future requirements.
- [ ] All five team members manually inspect code and tests and record their own review outcomes.
- [ ] Week 3 can start the engine around stable graph/MAS/campaign interfaces without guessing.

---

## 17. Required final response from the coding agent

Use this high-level structure:

1. **Status:** complete / partial / blocked.
2. **Commit and branch:** SHA, branch name and working-tree status.
3. **Implemented components:** config/seeds, graph, MAS, campaign/resource.
4. **Carry-forward backlog closed:** map each item in Section 2 to files/tests.
5. **Files created/changed:** path and purpose.
6. **Commands executed:** exact commands.
7. **Tests:** actual pass/fail/skip counts and failure detail.
8. **Reproducibility:** Python/dependency versions, seeds, revision, backend.
9. **Graph/dynamics benchmark:** N, E, density, runtime, memory method, backend—or why it was not run.
10. **GPU status:** per-device visibility, PyTorch/CUDA availability, whether GPU was used and why.
11. **Known defects/limitations:** explicit list.
12. **Spec changes/new decisions:** none, or decision IDs and review status.
13. **Human review:** pending until each member records their own outcome; do not invent individual identities or sign-off.
14. **Week 3 readiness:** ready / blocked with precise API/dependency reasons.

**Stop condition:** after the implementation, tests, smoke demo and self-review are ready for human review, stop. Do not begin the full Week 3 game engine before the five-member review gate is complete.
