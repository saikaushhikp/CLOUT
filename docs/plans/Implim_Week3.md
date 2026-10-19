# `Implim_Week3.md` — Dumbocracy Implementation Plan (Week 3)

> **Milestone:** M3 foundations — executable Game Engine + Information/Observation Model, integrating the approved Week 2 MAS and campaign/resource APIs  
> **Repository:** `https://github.com/saikaushhikp/CLOUT`  
> **Reviewed Week 2 commit:** `83f749deff45edfdc5dc353555893fde01de8dbe`  
> **Source of truth:** approved M0 technical specification, data contracts, decision log, Week 2 API notes/handoff, and this plan  
> **Primary goal:** deliver a deterministic end-to-end GPU-capable game episode at a realistic, materially larger electorate/network scale, with a correct round lifecycle, private true state, detached public observations, outcome calculation, safe event history, reset/replay, and integration tests. GPU execution is the preferred path when a compatible device is available; CPU remains a correctness/reference fallback.

Week 2 implementation is treated as approved at the user's direction. Any item with a discrepancy in review is carried forward as an implementation follow-up, not used to block Week 3. The coding agent must still report the observed state honestly, fix defects where they are in scope, and not claim tests it did not run.

---

## 1. Mission and Week 3 scope

Week 1 froze model semantics. Week 2 added executable foundations for configuration/seeds, sparse graph creation, voter state and synchronous dynamics, campaign action validation, `Decimal` ledger/atomic commits, tests, and a developer smoke example.

Week 3 now integrates these foundations into the full round-based **Game Engine** and completes the **Information Model** contract. The dependency graph shows the Game Engine Team ending at W3 after the MAS API, campaign API, and observation API are ready. The Week 2 hand-off explicitly states no engine or observation projection has yet been implemented.

Week 3 must turn the components into one testable program. It is not a week for formal strategy optimization, production frontend polish, exhaustive model checking, or large-scale experimental campaigns.

### End-of-week user-visible developer capability

A developer must be able to:

1. create a validated game configuration and deterministic electorate;
2. initialize the authoritative hidden game state;
3. obtain an initial player observation without access to the hidden state;
4. query legal action descriptors;
5. submit a campaign action or `no_op`;
6. validate an action without mutation, then atomically commit accepted actions;
7. apply the approved synchronous voter dynamics exactly once per accepted round;
8. expire round-local campaign effects and reset round-local guards in the correct order;
9. advance the horizon, compute terminal vote share/victory/score, and reject actions after termination;
10. inspect only sanitized public event summaries;
11. replay an action sequence from the same config/seeds and obtain the same public trajectory and result;
12. have a documented, reproducible test-and-benchmark runbook for a realistic-scale GPU episode, including exact commands, prerequisites, expected output structure, and acceptance thresholds. **The coding agent must not execute any test, validation, smoke-demo, or benchmark commands during Week 3.**

---

## 2. Week 2 review: done, partial, not done

The repository was inspected at commit `83f749deff45edfdc5dc353555893fde01de8dbe`. It contains `src/dumbocracy/config.py`, `graph.py`, `mas.py`, `campaign.py`, and `errors.py`; unit tests for these modules; a fixture-driven baseline integration test; a minimal `pyproject.toml`; `examples/week2_gpu_smoke.py`; API notes; a Week 2 hand-off; and `reports/week2_self_review.md`.

The self-review reports 13 passing tests, a passed M0 artifact validator, `git diff --check`, a passing CPU demo, and a CPU benchmark for N=1000/E=1975. These are repository-reported results: re-run the checks in the actual current environment before using them as current evidence.

| Week 2 item | Status | Week 3 treatment |
|---|---|---|
| Immutable validated config and deterministic subsystem seeds | **Done** | Reuse; add engine-level end-to-end reproducibility tests. |
| Sparse graph, edge-list fixture loading, ER generator and graph summaries | **Done** | Reuse; fix any defects found in integration. |
| Synchronous voter support dynamics and vote-share helper | **Done** | Reuse exactly; integrate with round semantics. |
| Pure action validation, Decimal ledger, immutable commit, broadcast guard | **Done** | Integrate; test action validation and commit together with engine state. |
| Week 2 unit/integration tests and developer smoke demo | **Done as reported** | Preserve them and add engine/observation integration tests; document commands for later user execution, but do not run them. |
| API notes and Week 2 hand-off | **Done** | Update only for clarified implementation contracts or new findings. |
| Full GameState / GameEngine lifecycle | **Not done** | Core Week 3 implementation. |
| Initial and per-round observation projection | **Not done** | Core Week 3 implementation. |
| Legal-action descriptors exposed safely to the player | **Not done** | Implement through observation/engine contract. |
| Engine-managed round ordering and state transition transaction | **Not done** | Core Week 3 implementation. |
| Terminal outcome lifecycle, result record, post-terminal guard | **Not done** | Core Week 3 implementation. |
| Structured safe event history and public event sanitization | **Partially specified, not implemented** | Implement internal vs public events and tests. |
| Reset, deterministic replay, action history handling | **Specified, not implemented** | Implement and test as engine-level behavior. |
| Observation leakage / detached-data tests | **Explicitly deferred** | Mandatory Week 3 gate. |
| Full LTL/epistemic evaluator and strategy optimizer | **Deferred by roadmap** | Define hooks/trace semantics only; full tools remain later work. |
| GPU execution and realistic network scale | **Required for Week 3** | Make GPU execution a first-class supported path, profile a representative network scale on the available GPU hardware, and document the resource/memory envelope. Multi-GPU distribution remains optional unless the single-device memory/performance study shows it is necessary. |
| Frontend integration | **UI skeleton is a parallel W2–W3 workstream in the dependency graph** | Provide a stable transport-neutral API and, if available, test a mock UI client. Do not build a production UI here. |

Any additional Week 2 discrepancy the agent discovers is approved for repair within this plan, provided the repair preserves M0 semantics. If the repair requires changing an approved semantic choice, add a new explicit decision record and compatibility note before implementation; do not silently alter behavior.

---

## 3. Frozen M0 semantics — do not reinterpret

Read these before editing:

- `docs/Assumptions/PROJECT_ASSUMPTIONS.md`
- `docs/spec/technical_spec_v1.md`
- `docs/spec/data_contracts.md`
- `docs/decisions/decision_log.md`
- `docs/spec/week2_api_notes.md`
- `docs/spec/week2_handoff.md`
- `docs/spec/test_plan.md`

### Core rules

- One controllable campaign; competing campaigns are deferred.
- Voter support `x_i` is a finite scalar in `[0,1]`, not latent belief or turnout.
- A vote for the player is `support > 0.5`; exactly `0.5` is not a player vote.
- Graph is fixed, undirected, unweighted, sparse and loop-free; disconnected components and isolates are valid.
- Dynamics are synchronous:

\[
m_i =
\begin{cases}
\operatorname{mean}(x_j:j\in N(i)), & \deg(i)>0,\\
0, & \deg(i)=0,
\end{cases}
\]

\[
x_i'=\operatorname{clip}_{[0,1]}\left[
x_i+s_i\left(\alpha(m_i-x_i)+c_i\right)
\right],
\]

where `alpha` is in `[0,1]`, `c_i` is the current-round campaign contribution, and `epsilon_i=0`. Use only pre-transition support values for every voter.
- `targeted_contact` costs `1.00` per unique valid target and contributes `delta_target` for this round.
- `broadcast` costs `5.00`, adds `delta_broadcast` to all voters for this round, and is allowed at most once per round.
- `no_op` costs `0.00`, has no campaign contribution, and advances a round. It is an engine-level action even if it shares the same tagged-union type.
- Validate without mutation. Rejected actions cannot debit the ledger, change voters/effects, append accepted-action side effects, or advance the round.
- Accepted action effect is applied in the current round's dynamics and then expires. Reset the broadcast-use flag and effects at the explicit round boundary.
- Terminal condition: `round_index == horizon_rounds`. Budget exhaustion does not terminate the game.
- At terminal time, score is player vote share; victory iff vote share is **strictly greater than** `0.5`; equality is defeat.
- The player and strategy receive detached observations only. Never expose support arrays, susceptibility, true seeds, full `GameState`, internal diagnostics, or private event data.
- Reuse the exact `Decimal` resource accounting from Week 2. Keep a CPU reference/fallback for correctness comparisons, but the primary performance path must use available GPU hardware where appropriate.

---

## 4. Repository reconnaissance and safety

Before edits:

1. Record current branch, `git status --short --branch`, HEAD SHA, and latest commit summary.
2. Read the sources listed in Section 3 and inspect the current `src/`, `tests/`, `examples/`, `pyproject.toml`, and report files.
3. Run the entire existing suite and the M0 validator. Save exact command and output summary.
4. Confirm Python and dependency availability; re-check GPU/PyTorch status rather than relying on old environment notes.
5. Confirm there are no local user changes to overwrite.
6. Do not print secrets or sensitive environment variables.
7. Do not install system packages or change drivers. Do not force-push or rewrite history.
8. **Do not execute tests, validators, smoke examples, demos, or benchmarks.** Static source inspection and reading existing recorded outputs are allowed. The agent must create the tests and a precise execution runbook, but leave all commands unexecuted for the user/reviewer to run later.

Treat the current Week 2 commit as the reference baseline, but always work from the latest checkout. Preserve the existing module architecture unless a small, explained change improves integration.

---

## 5. Required architecture and module ownership

Add new modules that preserve the current boundaries. Suggested additions:

```text
src/dumbocracy/
  engine.py        # GameState, GameEngine, step/round sequencing
  information.py   # Observation projection, safe serialization
  events.py        # internal events and public safe summaries, if useful
  strategies.py     # interface stub only if needed for smoke integration
  results.py        # optional immutable GameResult
tests/
  unit/
    test_engine.py
    test_information.py
  integration/
    test_game_episode.py
    test_replay.py
    test_observation_privacy.py
examples/
  week3_gpu_episode.py
docs/spec/
  week3_api_notes.md
  week3_handoff.md
reports/
  week3_self_review.md
```

Do not split modules solely to mirror this layout. A clear implementation with fewer files is preferable to redundant abstraction. Keep engine-agnostic data types in lower-level modules and avoid import cycles.

### Ownership

- **Config/seed layer:** validates configuration; derives stable subsystem seeds.
- **Graph layer:** graph topology/metrics, read-only to dynamics and engine.
- **MAS layer:** validates voters and returns new voter state after one synchronous transition.
- **Campaign/resource layer:** pure action validation and immutable commit data.
- **Information layer:** maps true state to a detached observation; it must not mutate true state.
- **Engine:** sole authority for round order, committing the state transition, round count, event history, terminal state, and outcome.
- **UI and strategies:** clients of observation/legal-action/result data only, never the `GameState`.

---

## 6. Task A — Implement the authoritative `GameState`

Create an explicit, validated state model consistent with `docs/spec/data_contracts.md`.

It must own at least:

- validated configuration / game identity;
- immutable graph reference;
- ordered voter-state tuple aligned to graph node IDs;
- current round index in `[0, horizon_rounds]`;
- current immutable resource ledger;
- current-round campaign contributions/effects;
- broadcast-used flag or equivalent round-scoped campaign context;
- internal event history or reference to an engine-owned event log;
- terminal flag/status and result once completed.

### State invariants

- Number and order of voters exactly match graph node IDs.
- No negative budget; `remaining_budget + spent == initial_budget` exactly.
- Round index never decreases or exceeds horizon.
- A terminal state cannot accept another round.
- All support/susceptibility values satisfy the M0 domains.
- Round-local effects do not survive into the next round.
- Public observation construction does not share mutable references with the state.
- Any state transition that fails leaves the previously authoritative state usable and unchanged.

Prefer immutable snapshots or a validate/compute/commit transaction. Avoid partially mutating a live `GameState` and trying to roll back scattered fields after an exception.

---

## 7. Task B — Implement `GameEngine` and exact round lifecycle

Expose a clear transport-neutral API. The exact class/function signatures may follow existing style, but document them and add type annotations. A useful conceptual API is:

```python
GameEngine.initialize(config, initial_state_or_factory=None) -> GameState
GameEngine.observe(state) -> Observation
GameEngine.legal_actions(state_or_observation) -> tuple[ActionDescriptor, ...]
GameEngine.step(state, action) -> StepResult
GameEngine.is_terminal(state) -> bool
GameEngine.result(state) -> GameResult | None
GameEngine.run_episode(config, actions) -> EpisodeTrace  # optional convenience
```

`step` must return enough information to know whether the action was accepted, the error code if rejected, the new state, the new observation, and whether the episode is terminal. Rejection must not advance the game.

### Canonical action/round order

Implement and test this exact ordering, respecting the M0 specification:

1. If state is already terminal, reject a new action with a documented stable terminal-state error/result. Do not mutate anything.
2. Validate action against the current state/context without mutation.
3. If invalid, produce a rejected-action event and safe response; keep round, state, budget, effects, and result unchanged. A rejected action does not consume a round.
4. If valid, atomically commit resource debit/action effects.
5. Apply the synchronous MAS transition exactly once, using the pre-transition voter support vector and the current-round campaign contributions.
6. Expire campaign contributions/effects after the dynamics.
7. Advance `round_index` exactly once.
8. Determine whether the fixed horizon has been reached. If not terminal, the state has no final result. If terminal, compute terminal vote counts, vote share, score, victory, budget used, terminal reason, and reproducibility metadata.
9. Append internal audit events in the defined order and expose only safe summaries in the observation.
10. Construct a fresh detached observation for the returned state.

For `no_op`, there is no campaign contribution but the dynamics still occur once and the round advances once. It must not accidentally skip voter dynamics or call the transition twice.

### Failure atomicity

The engine should calculate the proposed next values in local variables and only produce/commit a new authoritative state after all validations and calculations succeed. Numerical instability must not leave a debited budget with unmodified voters, or changed voters with an old round counter.

### Legal actions

Define `ActionDescriptor` so the client can understand the action ID, costs, target requirements, and parameters it must supply. The engine must be authoritative: a client-provided legal-action list is advisory; submitted actions must be validated again against the true current context.

Do not leak hidden state through the list of legal actions. Costs and rules are public under M0. Keep action descriptors serializable without exposing internal mutable objects.

---

## 8. Task C — Complete the Information/Observation Model

This is one of the most important Week 3 tasks. Build `Observation` as a distinct data type rather than returning a `GameState` with a few fields hidden in the UI.

### Required visible fields

Per M0 spec, the observation contains only:

- `round_index`;
- `remaining_budget`;
- `aggregate_support` (mean support; explicitly allowed by M0);
- public graph summary;
- legal-action descriptors;
- terminal status;
- public-safe event summaries.

Add only additional fields approved by M0 or an explicit new decision. Do not expose all-voter supports, susceptibilities, root/derived seeds, private diagnostics, raw internal state, hidden random streams, or campaign effect vectors.

### Observation properties

- **Detached:** mutation of a returned observation or any nested dict/list must not mutate hidden state.
- **No shared references:** if frozen dataclasses are used, ensure nested containers are immutable or copied; a frozen outer dataclass does not freeze nested mappings.
- **Safe serialization:** implement an explicit `to_public_dict()`/equivalent allowlist. Never use `dataclasses.asdict(game_state)` or generic serialization of internal state as the public payload.
- **Deterministic:** equal observable public context yields equal observation payloads; event ordering is deterministic.
- **JSON-safe:** public serialization uses stable primitives (strings, numbers, booleans, null, lists/maps) and no Python object instances, Decimal objects without an explicit encoding rule, NumPy arrays/scalars, or secret fields.
- **Consistent:** `legal_actions` and `terminal_status` match the state at the same round boundary.
- **Errors safe:** public errors contain stable code and non-sensitive explanation only.

### Information-set semantics

Preserve the M0 conceptual information set:

\[
\mathcal I(O_t)=\{S:O(S)=O_t\}
\]

with matching public context/history. Week 3 need not enumerate every hidden state or construct a Bayesian posterior. It must preserve the boundary and provide enough observation and trace data for the later epistemic/logic layer. Do not claim that an information-set model checker exists merely because this equation is documented.

### Mandatory security/privacy tests

- Assert no banned keys (`voters`, `support`, `susceptibility`, `seed`, derived seed names, `GameState`, diagnostics, campaign contribution/effect arrays) occur in public serialized observation paths, except permitted aggregate support.
- Search recursively through nested mappings/lists and serialization output; do not just check top-level fields.
- Mutate returned nested public data and verify hidden state is unchanged.
- Verify the observation does not hold a direct reference to the true state, voter tuple, internal event list, or resource context.
- Verify private data cannot leak through event summaries or error messages.
- Verify two states with different private per-voter detail but identical allowed public context generate identical public payloads, subject to the specified event/history equivalence.
- Verify legal action descriptors contain no hidden values.
- Ensure test names and assertions clearly explain what constitutes a leakage violation.

A security boundary test that only checks one example object is not enough; include recursive key inspection and a serialization-level test.

---

## 9. Task D — Outcome, terminal state, and budget correctness

Implement the M0 result schema and terminal lifecycle.

At `round_index == horizon_rounds`:

- `player_votes = count(voter.support > 0.5)`;
- `electorate_size = node_count`;
- `vote_share = player_votes / electorate_size`;
- `score = vote_share`;
- `victory = (vote_share > 0.5)`;
- `budget_used = initial_budget - remaining_budget`;
- `terminal_reason` is the fixed-horizon completion reason;
- include config/seed/revision/environment metadata to the extent practical, with secrets excluded.

Rules:

- Exact 0.5 support is not a player vote.
- Vote-share equality at exactly 0.5 is defeat.
- Budget exhaustion is not an early terminal condition.
- Reject stepping after terminal state.
- Result is immutable and cannot be changed by later client mutation.
- Terminal result is generated exactly once and corresponds to the final voter state, final ledger, and final round.
- Empty electorate is rejected by config and cannot divide by zero.
- All numeric outputs are finite.
- The current M0 spec says early “inevitable defeat” is not evaluated; do not add it.

---

## 10. Task E — Safe events, errors, reset, and replay

### Event stream

The internal audit trail should include the events required by M0, as applicable:

- game initialized;
- action proposed;
- action accepted/rejected;
- resource spent;
- voter-update summary;
- observation generated;
- round completed;
- game terminated.

Internal events may include details needed for debugging, but public event summaries are an allowlist. Do not expose a full raw event containing hidden fields to the UI or strategy. Do not store private voter state in public logs.

Events should have an event type, round index, stable ordering/sequence, and safe attributes. Use structured values rather than parsing human-readable log messages. Do not log every voter’s support vector per round by default; aggregate summaries are enough for the public path.

### Stable failures

Reuse `ErrorCode` for M0 errors where applicable: `INVALID_CONFIGURATION`, `INVALID_ACTION`, `INVALID_TARGET`, `INSUFFICIENT_BUDGET`, `MALFORMED_OBSERVATION`, `NUMERICAL_INSTABILITY`, `INTERNAL_FAILURE`. Add a terminal-action-specific code only if needed, documented, tested, and backward-compatible.

Errors must not return tracebacks, seeds, voter arrays, susceptibility, campaign contributions, or raw internal event objects in public payloads.

### Reset and replay

Implement deterministic reset with the same validated config and subsystem seed derivation. Replay means applying the same ordered accepted-action sequence from a fresh identical initialization. Rejected attempts can be included in a diagnostic request log, but they must not advance state. Specify whether the canonical replay sequence stores only accepted actions or stores all submissions plus explicit acceptance outcomes; implement one clear format.

Test:

- two fresh games with same config/seed have identical initial observations and hidden-state fixtures;
- replaying accepted actions produces identical public observations, event summaries, result, and budget;
- changing the seed changes generated stochastic graph/state where that subsystem has randomness (avoid brittle claims for edge cases such as `p=0`);
- rejected actions are replay-stable and do not advance the round;
- reset returns a fresh initial state and does not retain previous episode events/effects/guards.

---

## 11. Task F — Trace and logic hooks, without overbuilding a model checker

M0 defines bounded-LTL diagnostics over completed fixed-horizon traces and named predicates including `Terminal`, `Victory`, `BudgetNonnegative`, `StateValid`, and `ActionAtomic`.

Week 3 must capture enough transition/event data so later logic tooling does not need to reconstruct the game from console logs:

- canonical per-round trace records;
- round index and terminal flag;
- approved result predicates and relevant public/internal values separated;
- action outcome and atomic-rejection evidence;
- reproducibility/config metadata.

Add a small pure helper for computing these predicates if it fits naturally. Do not implement an expansive LTL/CTL parser or enumerate all possible histories in Week 3 unless the project has explicitly reassigned that work. Ensure trace fields have documented semantics and don't leak into public observation by default.

For epistemic work, retain the distinction between truth, possibility, and knowledge. Week 3 supplies the observation boundary and contextual history; full knowledge reasoning may remain Week 4.

---

## 12. Task G — API and UI hand-off

The dependency graph shows a UI Prototype stream beginning in Week 2–4 and UI Integration continuing through Week 5. Week 3 should provide the stable engine-facing API and a mock client contract.

Deliver `docs/spec/week3_api_notes.md` with:

- initialization parameters;
- JSON-safe observation schema;
- legal-action descriptor schema;
- action submission schema;
- step response (accepted/rejected, error code, observation, terminal status);
- result schema;
- event/public summary schema;
- reset/replay semantics;
- versioning and backward compatibility notes;
- one full serialized example episode or a short JSON transcript.

If a frontend skeleton already exists in the current checkout, inspect it and provide a thin adapter/example. If no frontend exists, use a tiny mock/client script to show that the engine can be consumed without importing internal `GameState`. Do not add a production UI framework just to show integration.

Keep engine code independent of HTTP/React/other UI details. Transport details can be added by the UI workstream.

---

## 12A. GPU-first implementation and representative network scale

GPU support is a Week 3 requirement, not an optional stretch goal. Use the available GPU hardware after statically inspecting the installed project dependencies and existing device abstraction. Do not assume a particular GPU model or VRAM amount; the later user-run environment report must identify the actual hardware.

### GPU implementation requirements

- Keep graph storage sparse. Avoid dense `N x N` adjacency matrices for large networks.
- Provide a clear device/backend selection, with GPU as the preferred path when a supported device is available and an explicit CPU reference/fallback path.
- Move the computationally heavy voter transition and graph-neighbor aggregation to GPU-compatible tensor/sparse operations where the chosen framework supports them. Avoid per-voter Python loops and repeated CPU↔GPU transfers in the hot path.
- Preserve synchronous semantics: every next-round support value must be computed from the same pre-transition support vector.
- Keep action validation, stable error handling, and exact `Decimal` ledger accounting correct. Convert campaign effects to the numeric device representation only at a well-defined boundary; do not use floating-point arithmetic for money.
- Seed graph generation, initialization, and stochastic subsystems deterministically. Document any limits to bitwise reproducibility across GPU architectures/framework versions.
- Make CPU and GPU implementations share one semantic contract. Include a small deterministic reference fixture whose expected values can be checked by the later user-run test suite.
- Avoid silently moving the entire graph/state to CPU during a supposedly GPU run. Report the selected backend and verify the compute path is actually using the intended device.
- Keep multi-GPU/distributed execution optional. First demonstrate a well-engineered single-GPU path; document whether network/state size or memory measurements justify multi-GPU work later.

### Required scale study (commands must not be executed by the agent)

Implement a benchmark harness and a result template for at least these graph sizes, subject to a documented memory-aware adjustment if the available GPU cannot hold the largest case:

- **Small correctness case:** 1,000 nodes, sparse ER graph, fixed seed.
- **Medium case:** 10,000 nodes, sparse ER graph, fixed seed.
- **Large case:** 100,000 nodes, sparse ER graph, fixed seed, or the largest clearly justified scale that fits available VRAM.

Use a declared ER edge probability and record actual node/edge counts, average degree, density, initialization time, per-round transition time, end-to-end episode time, peak GPU memory, and selected backend. The purpose is to make the network materially larger than the Week 2 N=1,000 CPU smoke benchmark and to quantify the scaling envelope. If the large case is infeasible, document the expected memory calculation and choose the largest defensible size; do not pretend it ran.

The harness should warm up the GPU before timing, synchronize the device around measured regions when the framework requires it, separate graph-generation time from transition time, and repeat timed transitions enough to report a robust summary (for example median and spread). Include a CPU/GPU correctness comparison on a smaller identical graph using an explicitly declared tolerance. Do not require the CPU to run the 100k case if that would make the later validation impractical.

### Expected benchmark report format

Create a table with columns:

`backend | GPU/device | N | E | p | seed | rounds | init_seconds | median_step_ms | episode_seconds | peak_GPU_memory_MiB | final_vote_share | victory | correctness_check`

Every row must be labelled one of:
- `not run — awaiting user/reviewer`
- `passed`
- `failed`
- `skipped — reason`

Until the user/reviewer runs the commands, all new benchmark rows must be marked `not run — awaiting user/reviewer`. Do not fabricate timing, memory, speedup, or scale results.

---

## 13. Test artifacts and execution runbook — DO NOT RUN COMMANDS

**Hard instruction to the coding agent: do not execute or run any commands during the testing/validation/benchmarking phase.** In particular, do not invoke unit tests, integration tests, artifact validators, smoke scripts, demos, profiling, performance benchmarks, `git diff --check`, or similar verification commands. Do not run a test indirectly through an IDE, task runner, pre-commit hook, or CI trigger. Do not spend tokens reporting command output that was never obtained.

The agent must still implement comprehensive tests, inspect them statically, and document exactly what the user/reviewer should execute later. This restriction is intended to reduce LLM/agentic execution cost. Existing outputs may be read as historical evidence, but must be labeled as such and not represented as freshly verified.

Preserve all Week 1/2 tests and add the Week 3 suite. Create/update the runbook in `docs/spec/week3_handoff.md` or `docs/spec/week3_api_notes.md` with the exact commands, prerequisites, execution order, expected output, and pass/fail interpretation. The commands below are **for the user/reviewer to execute later, not for the coding agent to run**.

### 13.0 User/reviewer execution sequence (NOT FOR AGENT EXECUTION)

The agent must write these commands into the hand-off documentation and explain any environment-specific adjustment. Do not execute them in this task.

**Prerequisites to document:** supported Python version; project install/dependency command based on `pyproject.toml`; GPU framework version; CUDA/runtime compatibility; how to activate the intended environment; and how to select/confirm the GPU backend. Do not invent a package installation command that is unsupported by the repository.

**Stage 1 — Static/deterministic project checks (user/reviewer only):**

```bash
python tests/validate_week1_artifacts.py
PYTHONPATH=src python -m unittest discover -s tests -v
git diff --check
```

Expected result: the artifact validator prints its success message; the unit suite ends with a summary reporting zero failures/errors; `git diff --check` exits successfully with no output. These are expected outcomes, not claims that the commands have been run.

**Stage 2 — Existing Week 2 smoke example (user/reviewer only):**

```bash
python examples/week2_gpu_smoke.py
```

Expected result: a clean process exit and the documented smoke/benchmark summary. The hand-off must state what output the existing script actually emits after static inspection; do not invent exact text or timing.

**Stage 3 — Week 3 GPU episode (user/reviewer only):**

```bash
python examples/week3_gpu_episode.py --device auto --nodes 1000 --seed 20261019
python examples/week3_gpu_episode.py --device auto --nodes 10000 --seed 20261019
python examples/week3_gpu_episode.py --device auto --nodes 100000 --seed 20261019
```

If implementation chooses different CLI flags, update this runbook to match the actual script. `auto` must select a supported GPU when available and clearly announce any CPU fallback. The run should print a structured summary including selected backend/device, graph N/E/density/average degree, seed, rounds completed, final vote share, victory, total episode time, per-round timing, and peak GPU memory when available. The 100,000-node run may be marked skipped with a documented VRAM/memory justification, but the agent must not run it to find out.

**Stage 4 — GPU/CPU/GPU correctness comparison (user/reviewer only):**

```bash
python -m unittest tests.integration.test_gpu_cpu_equivalence -v
```

Expected result: the same fixed small graph, seed, and action sequence produce CPU/GPU results within the documented numeric tolerance; discrete outcomes and action acceptance decisions must agree. Update the command if the repository's test discovery convention requires a different invocation.

**Stage 5 — Diff and evidence review (user/reviewer only):**

```bash
git diff --check
git status --short
```

Expected result: no whitespace errors; only intended files are modified/added. The reviewer should inspect the diff and compare actual output with the expected-output descriptions.

**Expected test report:** list each command, whether it was executed by the user/reviewer, exact exit status, pass/fail/skip count, relevant output excerpt, GPU/device details, and deviations. If the agent has not executed commands (as required), mark all Week 3 checks `not run — awaiting user/reviewer`; never substitute an expected result for actual evidence.


### 13.1 Unit tests: engine

- initialization with a minimal valid config;
- voter count/order and graph-ID alignment;
- initial round is zero and terminal result is absent;
- valid target action commits cost/effect exactly once;
- broadcast can be used once in a round and correctly becomes available next round;
- `no_op` advances exactly one round and applies exactly one dynamics update;
- invalid action leaves state and round unchanged;
- insufficient budget leaves state unchanged;
- terminal transition produces correct vote totals/share/score;
- step after terminal does not mutate state;
- budget exhaustion alone does not terminate;
- numerical failure is atomic;
- event order and event type are stable.

### 13.2 Unit tests: information

- allowed fields all present and correctly typed;
- hidden voter array/support/susceptibility/seeds/effect state absent;
- recursive serialization leak scan;
- JSON encode/decode round-trip for public payload;
- no alias from nested observation values back into hidden state;
- public events/errors are sanitized;
- terminal observation and result are consistent.

### 13.3 End-to-end integration tests

At least these scenarios:

1. **Three-node path, targeted action:** load committed fixture, initialize game, observe, target `v0`, verify exact cost/effect/dynamics, get next observation, and reach expected state.
2. **No-op horizon:** submit `no_op` until terminal; ensure each accepted action advances once, voter dynamics run each round, and result is created at exactly the horizon.
3. **Atomic rejection:** submit unknown target and insufficient-budget action; verify no round advance, budget debit, state change, effect change, or accepted-event side effect.
4. **Broadcast lifecycle:** broadcast, finish round, then verify broadcast becomes available only after correct reset; ensure repeated broadcast in the same round is rejected.
5. **Terminal result:** use hand-constructed support values to prove exact 0.5 voter and exactly 0.5 vote share are both handled correctly.
6. **Observation privacy:** internal states differ but allowed public context is the same; compare public payloads and recursive key checks.
7. **Reset/replay:** same seed and accepted action sequence yield same observations/result/events.
8. **Configuration/graph failure:** invalid config is rejected before creating a partially initialized state.

### 13.4 Property-style invariants

Use deterministic parameterized tests if a property-based dependency is not available:

- round count monotonic; never exceeds horizon;
- no accepted action occurs after terminal;
- budget accounting exact in `Decimal`;
- all support values stay finite and in `[0,1]`;
- final score equals vote share;
- victory iff vote share `> 0.5`;
- every rejection preserves a state fingerprint;
- public observation does not include banned private data;
- all event sequence numbers/order are stable;
- replay is deterministic for same config/seeds/actions.

### 13.5 Test discipline

Tests must assert correctness, not only non-crashing. Include expected numeric values and explicit invariants. Do not mark unimplemented logic/GPU properties as passed. Since the coding agent must not run tests, report each test as **implemented, not executed** unless there is pre-existing recorded evidence; clearly separate historical results from future expected results.

GPU is the preferred execution path when available. Keep a small CPU reference test for semantic comparison and fallback, but do not let CPU-only success substitute for GPU validation. The later user/reviewer run should report GPU model/name, total and free VRAM at start, framework/CUDA versions, device selected, graph node/edge counts, runtime, peak allocated/reserved GPU memory where available, and any CPU/GPU numeric tolerance.

---

## 14. Required Week 3 deliverables

- [ ] `GameState` and validated initialization.
- [ ] `GameEngine` with canonical lifecycle and atomic action/round transition.
- [ ] Initial and per-round detached `Observation`.
- [ ] Explicit allowlist serializer for observations and public events.
- [ ] Legal-action descriptors and action submission/step response schema.
- [ ] Terminal result computation and post-terminal guard.
- [ ] Safe structured event history.
- [ ] Reset and deterministic replay support.
- [ ] Trace records/predicate hooks suitable for later temporal/epistemic work.
- [ ] Unit tests for engine lifecycle, info privacy, outcomes, and errors.
- [ ] End-to-end integration tests using Week 2 fixtures and APIs.
- [ ] GPU-capable full-episode example at a documented representative network scale, with CPU fallback/reference mode.
- [ ] `docs/spec/week3_api_notes.md` and `docs/spec/week3_handoff.md`.
- [ ] `reports/week3_self_review.md` with exact command/test/environment evidence.
- [ ] Prior Week 1/2 checks are documented for later execution; do not claim they pass in Week 3 unless existing evidence supports that claim.
- [ ] Five-member manual review evidence is recorded honestly before Week 4 begins.

---

## 15. Week 3 execution workflow for the coding agent

### Phase A — Baseline audit

- Inspect current branch/commit/status and source files.
- **Do not run existing tests or the M0 validator.** Record them in the later-execution runbook instead.
- Confirm the current Week 2 APIs and docs by static inspection.
- List any observed Week 2 discrepancy. Treat this backlog as approved for repair if required, but preserve M0 semantics and document all changes.

**Gate:** do not proceed with hidden semantic changes.

### Phase B — State/response design

- Define exact `GameState`, `Observation`, `ActionDescriptor`, `StepResult`, `GameResult`, and event/trace record schemas.
- Draw or document the lifecycle state machine.
- Document atomic boundaries and public/private fields before implementation.
- Check API seams with the existing Week 2 modules.

### Phase C — Observation first

- Implement detached observation projection and explicit safe serialization.
- Add privacy/aliasing tests before wiring the UI/client.
- Ensure no generic serializer can traverse into `GameState`.

### Phase D — Engine lifecycle

- Implement initialize, legal actions, validate/commit, voter update, expiration, round increment, terminal result, event append, and observation.
- Make each step transactional.
- Add focused unit tests as the lifecycle is implemented.

### Phase E — Replay and integration

- Implement reset/replay.
- Add fixture-driven scenarios and numeric expectations.
- Produce a GPU-capable episode demo and transport-neutral client example; include a small CPU reference mode.
- Confirm the Week 2 resource guard and effects are reset correctly at the round boundary.

### Phase F — Self-review

- Inspect all new/changed tests and implementation statically for hidden-state leaks, mutation paths, nondeterminism, duplicated state, and incorrect terminal handling.
- Do not execute tests or rerun failures. Document the exact commands and diagnostic steps the user/reviewer should use later.
- Produce Week 3 self-review, API notes, and hand-off.
- Stop for the five-member review gate.

---

## 16. Week 3 exit criteria

Week 3 is complete only when:

- [ ] Existing Week 1/2 tests are preserved and included in the runbook; their current status is labelled accurately because Week 3 agent execution is prohibited.
- [ ] A deterministic scripted episode is implemented for GPU execution at representative scale, with a CPU reference/fallback path; execution remains pending user/reviewer validation.
- [ ] A rejected action is demonstrably a no-op on all authoritative fields.
- [ ] Accepted actions debit exactly once and influence exactly the current round.
- [ ] `no_op` still advances a round and runs one dynamics transition.
- [ ] Broadcast guard/effects reset at the agreed boundary.
- [ ] Terminal round, vote share, victory, score, and budget-used values are correct.
- [ ] Post-terminal actions are rejected without mutation.
- [ ] Observation serialization is allowlist-based, detached, stable, JSON-safe, and passes recursive privacy checks.
- [ ] Public error/event payloads reveal no hidden state.
- [ ] Reset/replay preserves deterministic outputs for a fixed seed/action sequence.
- [ ] Trace records have sufficient information for later logic checks without logging secrets publicly.
- [ ] A mock or example client consumes the observation/action API without importing hidden state.
- [ ] Documentation identifies stable signatures and all unresolved integration points.
- [ ] GPU/network-scale benchmark harness and reporting template are implemented. No benchmark is executed by the agent; report expected output format and mark performance results pending user/reviewer execution.
- [ ] Self-review and exact test evidence are complete.
- [ ] All five members have manually reviewed actual code and test evidence.

---

## 17. Required final report from the coding agent

Report using this structure:

1. **Status:** complete / partial / blocked.
2. **Commit, branch, working-tree status** (static repository metadata only; do not run commands to collect it if doing so would invoke the testing phase).
3. **Week 2 backlog closed:** list each carry-forward item and proof.
4. **Game lifecycle:** summarize ordering, terminal behavior, and atomicity.
5. **Observation privacy:** tests executed and results.
6. **Files created/changed.**
7. **Execution runbook:** exact commands, prerequisites, expected outputs, and diagnostic guidance for the user/reviewer.
8. **Test status:** implemented/not executed; include only historical pass/fail/skip evidence that already exists, and label all new checks pending user/reviewer execution.
9. **End-to-end demo:** config, seed, action sequence, resulting score/victory, and output evidence.
10. **Replay/reproducibility:** describe test coverage and expected evidence; mark actual results pending user/reviewer execution.
11. **Known limitations and unresolved work.**
12. **Spec/decision changes:** none, or exact new decision IDs and rationale.
13. **Five-member review:** pending until all actual reviews are recorded; do not invent identities.
14. **Week 4 readiness:** ready / blocked and why.

**Stop condition:** finish the Week 3 engine and information-model implementation, test artifacts, GPU-capable demo, execution runbook, documentation, and static self-review. Do not execute any testing, validation, smoke-demo, or benchmark commands. Clearly label all unexecuted checks as pending. Wait for manual review and user-run validation before proceeding into Week 4 formal verification/optimization integration.
