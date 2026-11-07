# `Implim_Week5.md` — Dumbocracy Implementation Plan (Week 5)

> **Milestone:** M5/M6/M7 beta — strategy optimization v1, finite-world epistemic verification, MAS/resource/information experiments, and end-to-end playable web integration  
> **Repository:** https://github.com/saikaushhikp/CLOUT  
> **Baseline branch:** `master`  
> **Baseline commit inspected:** `775633d5bbe6360b423dedf8d0647230fa1e2cfb` (Week 4 approval/handoff update; inspect the latest commit again before work)  
> **Status at Week 5 start:** Week 3 and Week 4 are accepted. Treat any discrepancy in earlier-week work as accepted work plus a forward-looking repair/documentation item; do not reopen those approvals.  
> **Execution policy:** The coding agent **IS AUTHORIZED AND EXPECTED to execute commands during implementation, testing, validation, smoke testing, profiling, benchmarking, and integration testing**. Run checks, diagnose failures, fix them, and rerun relevant checks. Preserve exact command output and execution metadata.  
> **Compute policy:** GPU-first for suitable large numerical workloads; keep CPU as the semantic reference and fallback. Use CUDA for the representative large-network workload. Each GPU has separate VRAM; discover device count at runtime. No pooled-VRAM assumption or multi-GPU implementation without an explicitly reviewed need.  
> **Contract policy:** Preserve the frozen M0 v1 semantics and public information boundary by default. Any richer observation profile required by information-level experiments must be opt-in, separately versioned, and gated by a documented decision; do not silently widen the v1 observation.

---

## 0. Mission: return to the original product and research problem

Dumbocracy is not merely a graph-dynamics library or a collection of standalone tests. It is an **interactive election-campaign resource-allocation game** in which a player must choose campaign actions with a limited budget while the electorate evolves as a graph-based multi-agent system. The game intentionally combines:

1. **A graph-based electorate:** voters have hidden individual state/attributes, peer influence, and heterogeneous susceptibility; network topology controls how local influence propagates.
2. **Campaign resources:** actions have costs, targeting rules, and different influence scopes. In the current frozen v1, the concrete actions are `targeted_contact`, `broadcast`, and `no_op`; do not invent extra core actions in this week.
3. **Partial information:** the player receives a controlled public observation, not the true voter state. Strategy code must decide only from information actually provided to it.
4. **Strategy and optimization:** compare simple baseline policies with a first greedy/model-based policy, evaluating victory probability, terminal vote share, resource efficiency, and robustness across seeds.
5. **Temporal and epistemic reasoning:** connect formulas to actual simulator traces and reason—within a clearly declared finite model—about possible victory, guaranteed victory under partial information, viable states, and inevitable defeat. A sampled finite model must never be described as a proof over the full continuous state space.
6. **Decision-support dashboard/gameplay:** a user should be able to start a game, see the permitted information, allocate campaign resources, observe round progression, and review results/diagnostics without accessing hidden state.
7. **Experimental evaluation:** run reproducible comparisons across strategies, graph structures, budgets, and (only if explicitly approved) information levels; produce plots and reports explaining the results.

The uploaded `Proposal.pdf` frames these objectives as a web-based game, a graph/MAS electorate, resource sinks, a partial-information dashboard, Temporal/Epistemic Logic, and a search for effective strategies. `logical_next_steps.md` decomposes this into the MAS, campaign, partial-information, game, strategy, logic, dashboard, and experiment phases. `planning_tentative.pdf` names Week 5 as **Full Integration and Verification**: strategy results, MAS freeze, information experiments, formal verification, and a full pipeline `UI ↔ Game ↔ MAS ↔ Strategy ↔ Logic`. Its six-week plan reserves Week 6 for final experiments, refinement, and presentation.

**Source inventory note:** no standalone Dumbocracy file literally named `Problem Statement` was present in the attached-file/repository inventory inspected for this plan. The problem framing here is grounded in `Proposal.pdf`, `logical_next_steps.md`, and `planning_tentative.pdf`; do not substitute an unrelated problem statement from elsewhere.

### Week 5 north star

By the end of Week 5, a reviewer should be able to launch a local browser-based game connected to the real engine, play an entire deterministic episode, compare baseline and greedy strategy runs, inspect bounded-temporal and finite-world epistemic diagnostics, reproduce a defined GPU/CPU test set, and view preliminary experiment plots. The result is a **functional research beta**, not a polished production deployment or a mathematically universal proof over all continuous states.

---

## 1. Week 4 audit: done, partially done, and not done

This inventory is based on the current repository tree, `reports/week4_self_review.md`, `docs/spec/week4_handoff.md`, `reports/week4_baseline_audit.md`, `reports/week4_performance_audit.md`, the Week 4 strategy/logic/experiment modules, tests, and examples. The self-review/handoff reports that the focused Week 4 tests passed (41), the full suite passed (69), the M0 validator passed, 18/18 paired example episodes completed, and CPU/CUDA comparisons passed at N=1,000 and N=10,000. These are **Week 4 agent-reported execution results**; Week 5 must preserve the provenance and must not claim those are new Week 5 runs.

| Area | Week 4 status | Week 5 treatment |
|---|---|---|
| Frozen M0 specification and Week 3 engine | **Done — accepted** | Reuse; protect with regression tests. MAS semantics change only for critical bugs and with a decision record if semantics truly change. |
| Detached observation/public serializer and hidden-state privacy tests | **Done — accepted** | Keep as a hard boundary through the API, dashboard, strategies, experiments, exports, and logs. |
| Bounded-LTL trace diagnostics for the three M0 formulas | **Done — accepted** | Reuse the finite-trace evaluator. Extend only where the new formula semantics are written down and tests provide known answers. |
| Strategy protocol plus `NoOpStrategy`, `FirstLegalStrategy`, `SeededRandomLegalStrategy` | **Done — baseline only** | Add uniform/resource-scheduling and a real greedy/model-based strategy. Keep policies observation-only. |
| Paired-seed JSONL runner and aggregates | **Done — basic framework** | Extend/reuse for graph-topology, budget, strategy, and approved information-profile experiments; add plots and uncertainty summaries. |
| Public mock client and sample observation | **Done — accepted** | Keep as a contract test/reference, but do not confuse this CLI mock client with a graphical web game. |
| CUDA path, 1k/10k/100k harness, matched CPU/CUDA checks | **Done — accepted, bounded performance evidence** | Use GPU for real Week 5 workloads; run a repeated 100k-node benchmark and record actual backend/device/memory. Do not redesign device ownership without evidence. |
| Standalone LTL over recorded trace | **Done for required M0 diagnostics** | Add `Eventually Victory` and state properties only with definitions compatible with fixed-horizon outcome semantics. |
| Epistemic information set / possible-world reasoning | **Not done** | Add a finite-world, bounded model-checking layer in MM5.2–MM5.3. Exactness is only relative to the explicitly enumerated finite world set/action abstraction. |
| `Viable`, `InevitableDefeat`, `PossibleVictory`, `GuaranteedVictory`, `K_player(...)` | **Not done as engine/model-checking semantics** | Define and evaluate them within finite scenarios. Do not change the engine's v1 terminal rule or present early defeat as an authoritative game termination. |
| Greedy/optimized strategy | **Not done** | Implement a first, bounded, observation-respecting greedy/model-based policy and evaluate it. Full RL/global optimization remains out of scope. |
| Uniform resource-allocation baseline | **Not done in the current strategy module** | Implement an explicitly defined budget/horizon-aware baseline.
| Network-aware player strategy under current observation | **Not yet supported by available public features** | Do not read hidden graph edges/degree/support. Gate any degree/community signals behind an opt-in, versioned observation-profile decision. If not approved, label network-aware player policy as blocked by aggregate-only information; an oracle upper bound must be clearly segregated from player policies. |
| Playable graphical dashboard/web/API | **Not present in repository tree** | Build a minimal integrated browser beta in MM5.5–MM5.6, using the existing engine and observation API. |
| MAS/model validation report and plots | **Not present as a Week 5 artifact** | Run controlled sanity/topology/influence checks, freeze the model except critical defects, produce figures and report. |
| Network-topology experiments beyond ER/explicit edges | **Not present in current graph factory** | Add experiment-only static topology generators (at least ER vs ring-lattice) without changing the frozen v1 graph invariants. |
| Resource-budget sensitivity experiments | **Not done** | Run paired-seed budget levels and report outcome/efficiency metrics. |
| Controlled information-availability experiments | **Not done** | Start with a formal decision gate; preserve aggregate-only v1 if richer disclosure is not approved. |
| End-to-end `Start → Play → End → Analysis` proof | **Not done** | Mandatory Week 5 beta acceptance criterion.
| Week 5/6 result artifacts and final-presentation preparation | **Not done** | Save logs, machine-readable outcomes, plots, and reproducibility notes; defer final polish to Week 6. |

### Week 4 backlog IDs carried into Week 5

- **W4-B01 — Epistemic model checker:** finite-world information-set reasoning over states compatible with the same observation/public history.
- **W4-B02 — Possible/guaranteed/inevitable reasoning:** define and implement bounded `PossibleVictory`, `GuaranteedVictory`, `Viable`, `InevitableDefeat`, `K_player(GuaranteedVictory)`, and `K_player(InevitableDefeat)` within an explicit finite model.
- **W4-B03 — Strategy optimization:** go beyond NoOp/FirstLegal/SeededRandom with uniform and greedy/model-based policies; keep the player-policy information boundary strict.
- **W4-B04 — Playable UI and service integration:** current tree includes a CLI mock client but no frontend/server; build the minimal local browser flow.
- **W4-B05 — MAS / graph experiments:** introduce an experimental topology alternative and controlled influence/dynamics tests; protect the v1 model.
- **W4-B06 — Resource budget experiments:** quantify budget sensitivity using the real engine.
- **W4-B07 — Information availability experiment:** proposal/tentative plan asks for information-level comparison, but the current frozen v1 observation contract is aggregate-only. Resolve via an explicit extension decision gate; no silent disclosure changes.
- **W4-B08 — Result visualization/reviewer evidence:** generate auditable plots, machine-readable outputs, logs, and individual review checklists.

All Week 4 work remains accepted. These are forward deliverables, not grounds to reopen Week 4.

---

## 2. Frozen contracts and scope boundaries

Before code changes, read and preserve:

- `docs/spec/technical_spec_v1.md`
- `docs/spec/data_contracts.md`
- `docs/decisions/decision_log.md`
- `docs/spec/week3_api_notes.md` and `docs/spec/week3_handoff.md`
- `docs/spec/logic_semantics_v1.md`
- `docs/spec/strategy_api_v1.md`
- `docs/spec/week4_compute_policy.md`
- `docs/spec/week4_handoff.md`
- `docs/Assumptions/PROJECT_ASSUMPTIONS.md`

### Hard constraints

1. **One campaign in core v1.** Do not add rival campaigns, candidates, abstention, turnout, or stochastic transition noise this week.
2. **Support and outcome:** support is finite in `[0,1]`; a player vote iff support `> 0.5`; exactly `0.5` is not a player vote. Terminal score equals vote share; victory iff vote share `> 0.5`.
3. **Graph:** static, undirected, unweighted, sparse, loop-free. Isolates and disconnected components are valid. ER and explicit-edge graphs remain supported. An experimental topology generator may produce another graph satisfying the same graph invariants but must not redefine the graph contract.
4. **Transition:** synchronous update using the full pre-transition support vector and the frozen M0 equation. No per-voter in-place update.
5. **Actions:** `targeted_contact`, `broadcast`, and `no_op` only in core v1; exact Decimal costs and current-round effect expiry remain unchanged.
6. **Atomicity:** rejected action must not debit budget, change voter state/effects, or advance game round. Safe rejection/audit events may be appended as existing Week 3 semantics specify.
7. **Information:** existing `Observation` remains detached and allowlist-serialized; do not expose private state, voter support/susceptibility arrays, hidden seed, private event attributes, device tensors, or hidden graph edges.
8. **Strategy:** a normal player policy receives only `Observation`, public prior history, `ActionDescriptor`s, and an explicit per-episode RNG. Never pass `GameState` or hidden arrays to a player policy. If an oracle is implemented for analysis, segregate it behind a different interface and label every output `oracle`, not a playable strategy.
9. **Temporal logic:** formulas are diagnostics over finite traces unless explicitly connected to the finite model checker. A property that evaluates on one trace does not establish a universal future guarantee.
10. **Epistemic scope:** the core state contains continuous support/susceptibility values, so exhaustively quantifying the full uncountable hidden-state space is not feasible. Week 5 exact epistemic claims are **relative to a named finite world set and declared action abstraction**. Report `world_count`, `action_abstraction`, `horizon`, and search limits for every result. Never label sampling as a universal proof.
11. **Early defeat:** `InevitableDefeat` is an analysis proposition only. The live v1 engine still terminates at the fixed horizon and must not stop early because a diagnostic predicts loss.
12. **GPU:** use a GPU where it benefits numeric network/state simulation, especially at 10k/100k nodes. CPU remains required for correctness and small finite-world search. Discover GPU count and per-device VRAM at runtime; do not assume two devices from old notes or pool memory.
13. **Reproducibility:** seed, config, revision, backend/device, environment, command, stdout/stderr, exit status, and generated artifact paths must be saved. Timing fields are allowed to vary; compare correctness fields deterministically.

### Extension decision gate: observation profiles

The proposal/tentative plan asks for Full/High/Medium/Low/Aggregate information and example levels 100%, 75%, 50%, 25%, and aggregate-only. The frozen M0 v1 contract currently discloses only aggregate support, graph summary, legal actions, budget, terminal status, and public events; it does not reveal voter-level support, susceptibility, or graph edges.

- In **MM5.1**, draft a decision record defining whether to preserve aggregate-only v1 or authorize a versioned experimental projection.
- DEC-012 is approved to preserve aggregate-only Observation v1. This does not approve richer profile fields; implementation requires a subsequent decision defining them and their privacy semantics.
- The default v1 `Observation` must remain unchanged. Implement profiles as an opt-in, versioned experimental projection separate from the default allowlist, or keep the richer work blocked if approval is not available.
- The approved DEC-012 gate leaves richer profiles unauthorized. Continue with aggregate-only experiments and report the limitation; do not fabricate 25/50/75/100% information results.

---

## 3. Mini-milestones and checkpointing

| Mini-milestone | Focus | Core output | Gate |
|---|---|---|---|
| **MM5.0** | Baseline/definition freeze | Week 5 audit, requirements-to-files matrix, review logs | Week 4 accepted and carry-forward backlog recorded; contracts unchanged. |
| **MM5.1** | Epistemic semantics + observation-profile decision | `docs/spec/epistemic_semantics_v1.md`, finite-world schema, decision gate | Definitions are precise; no guarantee outside declared finite model; default public schema preserved. |
| **MM5.2** | Finite possible-world representation and bounded model checker | `epistemic.py`/`model_checking.py`, fixtures, tests | Same-observation worlds are grouped correctly; action choices are observation-consistent; search caps work. |
| **MM5.3** | Temporal/epistemic integration | `verification.py`, formula adapters, scenarios and tests | LTL and epistemic outputs distinguish trace result, existential possibility, finite-model guarantee and known propositions. |
| **MM5.4** | Strategy engine v1 optimization | Uniform and greedy/model-based policies, benchmark tests | Strategies use permitted information only and improve or characterize performance under repeated paired seeds. |
| **MM5.5** | MAS/network/resource experiments | generators, experiment runner extensions, tables/plots/reports | ER vs alternative topology and budget sensitivity are reproducible; MAS semantics remain frozen. |
| **MM5.6** | Playable browser beta/API | minimal server, web UI, API integration tests | Start → Observe → Act → Update → End → Analysis works end to end without state leakage. |
| **MM5.7** | GPU scale and runtime integration | 1k/10k/100k measurements, CPU/CUDA parity logs, memory notes | At least one real CUDA run at the large target when hardware is available; CPU fallback never counts as GPU evidence. |
| **MM5.8** | Integrated experiments/results | strategy/topology/budget (and approved information-level) plots and reports | Outputs include actual data, uncertainty, command provenance, and clear limitations. |
| **MM5.9** | Regression, documentation, reviewer reproduction | self-review, handoff, end-to-end runbook and five-person checklist | Agent tests pass after fixes; each reviewer can reproduce assigned results from the same commit. |

At each gate, the coding agent should summarize changed files, newly implemented requirements, tests executed and exact counts, outputs generated, defects fixed, performance results, remaining partial items, and the next milestone. Do not start later milestones by silently skipping a failed gate.

---

## MM5.0 — Repository audit, scope, and reproducibility baseline

### Tasks

1. Inspect current branch, HEAD, working tree, and latest Week 4 files before editing. Preserve unrelated local/user changes; do not force-push or rewrite history.
2. Read the source documents listed above and map their requirements to implementation/test/report artifacts. In `reports/week5_baseline_audit.md`, explicitly map the proposal/tentative-plan objectives to Week 5 deliverables.
3. Record the exact baseline commit SHA from the current checkout and the Week 4 starting test report separately from Week 5 execution.
4. Confirm installed Python, NumPy, PyTorch/CUDA, device visibility/count, and any web/plot packages. The agent is authorized to run these read-only probes and subsequent project commands.
5. Update `docs/Assumptions/PROJECT_ASSUMPTIONS.md` only if Week 5 discovers a verified environment change; do not hard-code that there are exactly two GPUs. Treat each GPU as a separate memory pool.
6. Create a traceability table with each proposal-level question and artifact:
   - What happened? — episode/action/outcome summaries and plots.
   - Why did it happen? — aggregate support/resource trajectories and MAS analyses.
   - Could another strategy have worked? — paired strategy evaluation and bounded search.
   - Could the player have known it would work? — finite-world epistemic model under a declared world set.
   - Was victory possible or guaranteed? — model-checking result with explicit scope.
   - At what state did victory/defeat become viable/inevitable? — diagnostic only within the finite abstraction; not live early termination.
7. Create `reports/week5_baseline_audit.md` with accepted items, new work, scope constraints, conditional information-profile gate, and Week 6 backlog.

### Deliverables

- `reports/week5_baseline_audit.md`
- `reports/week5_requirements_traceability.md`
- recorded baseline revision and environment inventory in the Week 5 self-review or run log

### Acceptance gate

The agent can explain how each planned feature advances the election-campaign game, not just the codebase. No frozen M0 semantics or default public fields change during the audit.

---

## MM5.1 — Define epistemic semantics and gate experimental observation profiles

### Part A: finite-world semantics

Create `docs/spec/epistemic_semantics_v1.md`. State the terminology in both plain language and equations. Use a finite input set `W` of explicitly supplied worlds, each with a compatible config/horizon and a deterministic hidden initial state. For the initial information set at observation `O`, define:

`I_W(O, h) = { w in W : public_history_and_observation(w, h) == (h, O) }`.

The implementation must state whether equality is exact public JSON equality or a declared normalized comparison. Do not quantize away meaningful public fields silently. Public history, not just a single aggregate, is part of compatibility when required by the strategy contract.

Define the following carefully:

- **Possible victory:** at least one world/policy continuation that is permitted by the observation-based action rule reaches a terminal victory. It is existential and relative to `W` and the action abstraction.
- **Guaranteed victory:** there exists one observation-based contingent policy that selects the same action for worlds that are indistinguishable at the current public history and wins in every world in the current information set. It is not enough that a different omniscient policy can win each world separately.
- **Viable state:** in a single fully specified world state, at least one continuation from the declared action abstraction can still reach terminal victory. This is an analysis proposition, not a public observation field.
- **Inevitable defeat:** for a specified world state and declared action abstraction, no admissible continuation can reach victory before/at the fixed horizon. It does not change the live game's terminal rule.
- **`K_player(phi)`:** `phi` holds for every world compatible with the current public observation/history according to the declared finite world set. For `K_player(GuaranteedVictory)`, evaluate the **same policy tree across the full information set**, not separate per-world omniscient winners.
- **`K_player(InevitableDefeat)`:** every world in the current information set satisfies the declared inevitable-defeat predicate. Report that this is finite-set knowledge unless the state space is proven exhaustive for the model being claimed.
- **Unknown / not evaluable:** insufficient world coverage, incompatible public histories, action abstraction too restrictive, search cap reached, or unknown input must yield `UNKNOWN` / `INCOMPLETE`, never a false “guaranteed” or “inevitable” claim.

### Part B: bounded search contract

- Start with tiny instances (recommended N ≤ 8, horizon H ≤ 4); all fixture values must be hand-verifiable.
- The model checker consumes explicit finite world fixtures or a deterministic world-set builder. It must not discover private state by requesting the engine’s current `GameState` during a player strategy call.
- Each world is transitioned by the real authoritative `GameEngine`, not a duplicated alternative dynamics equation.
- Define a bounded action abstraction. Include `no_op`, legal `broadcast`, and `targeted_contact` with a finite, named set of singleton target candidates. If multi-target subsets are excluded, say so in every result. Conclusions are exact only relative to the declared abstraction.
- At each belief/world-set node, a candidate action must be legal in all indistinguishable worlds; simulate the same action in each world. Partition successor worlds by resulting public observation/history, then require the recursive policy to succeed in each partition for a guaranteed win.
- Define caps for worlds, horizon, number of search nodes, elapsed time, and memory. Return `INCOMPLETE` with the cap reached, explored nodes and reason. Never return `GUARANTEED_VICTORY` if the search ended early or the world/action space was truncated.
- Memoize belief/action subproblems using stable public-world-set identifiers where safe. Record counts and algorithm versions.
- The checker can run on CPU; the expensive large-network MAS workloads remain GPU-first. Do not force GPU into a combinatorial tiny search if it does not help.

### Part C: observation profiles decision gate

1. Write a small proposal for a versioned experimental observation policy aligned with the source roadmap's `Aggregate`, `Low`, `Medium`, `High`, and `Full` levels (and, if selected, 25/50/75/100% sampled-disclosure levels).
2. Define exactly which attributes each level exposes: aggregate mean; coarse histogram/range buckets; limited sample support values; graph degree/community features; and/or full values. Do not mix graph edges and voter support into one vague “information percentage”.
3. Define how voter IDs are chosen, how selection is seeded, whether selections persist across rounds, how strategies receive it, and how output/schema versions change.
4. No susceptibility disclosure or hidden-edge disclosure unless explicitly listed and approved. Never expose the root/derived seed in player-facing payloads.
5. Record the proposal as a decision-gate document / proposed decision. **Do not mark it approved unless actual owner/team approval exists.**
6. Keep `Observation` v1 default unchanged. If approval is recorded during the task, implement an explicit optional profile object/API and separate tests; if not, mark information-profile experiments `BLOCKED_PENDING_DECISION` and continue the other experiments.

### Tests to author and run

- same public observation/history across different hidden states places them in the same finite information set;
- worlds with different public graph summaries/history are not incorrectly merged;
- a winning omniscient action that differs across worlds is not misreported as a guaranteed policy;
- a single policy that succeeds in all worlds is recognized as guaranteed;
- incomplete/truncated search returns `INCOMPLETE`, never a guarantee;
- default v1 observation remains byte/schema-compatible absent an approved extension;
- experimental profile serializers disclose exactly their declared fields and nothing else.

### Deliverables and gate

- `docs/spec/epistemic_semantics_v1.md`
- `docs/spec/observation_profiles_experiment.md`
- DEC-012 decision record: approved aggregate-only v1; richer profiles remain blocked pending a later exact-field decision
- hand-constructed finite-world fixtures in `tests/unit/test_epistemic.py` (typed world descriptors; no separate JSON fixtures)
- `docs/spec/epistemic_semantics_v1.md` as the epistemic design artifact

**Gate:** semantics and disclosure contracts are reviewable before implementing the model checker. DEC-012 approves aggregate-only v1; richer profiles remain blocked pending a new exact-field decision.

---

## MM5.2 — Implement finite-world epistemic/model-checking engine

### Required module boundary

Implement a separate module, for example `src/dumbocracy/epistemic.py` or `model_checking.py`; do not bury future-state search inside `GameEngine.step`. `GameEngine` remains a deterministic one-action transition service.

Suggested API shape (final names may follow the codebase, but document them):

```python
FiniteWorldSet.from_fixtures(...)
EpistemicChecker.check_possible_victory(world_set, observation_history, action_abstraction, limits) -> CheckResult
EpistemicChecker.check_guaranteed_victory(world_set, observation_history, action_abstraction, limits) -> CheckResult
EpistemicChecker.check_inevitable_defeat(world_set, action_abstraction, limits) -> CheckResult
EpistemicChecker.evaluate_knowledge(world_set, observation_history, propositions, limits) -> KnowledgeReport
```

### Implementation tasks

1. Define immutable world descriptors with a stable `world_id`, game config, explicit graph/voters or committed initial fixture, and the public-observation lineage needed to test equivalence. Hidden state must not be serialized into player-facing response objects.
2. Create a scenario adapter that initializes a fresh `GameEngine` per world and executes actions only through `step`; never duplicate the dynamics.
3. Build the finite action abstraction deterministically from public descriptors and a declared target candidate set. Include an explicit maximum branching factor. Never enumerate all subsets of a 100k-node target list.
4. Implement terminal evaluation using the frozen terminal `GameResult` (`victory`, `score`) and not intermediate support threshold crossing.
5. Implement possible-win exploration and guaranteed-win belief-state search as defined in MM5.1. Enforce consistency: worlds with equal observation/history must receive the same next action.
6. Implement per-world viability/inevitable-defeat search under the same action abstraction; keep these as analysis diagnostics, not live terminal transitions.
7. Implement knowledge queries by quantifying over the declared world set/information set. Results include true/false/unknown/incomplete, witness/counterexample worlds, explored states, action abstraction, search limits, and provenance.
8. Cache safe model-search results keyed by configuration/world-set hashes and action abstraction. Do not cache based solely on observation aggregates if the public history differs.
9. Add graceful time/node/memory caps and deterministic tie-breaking. Record partial exploration and never quietly promote a partial result to a proof.
10. Write an example that contrasts (a) victory possible, (b) winning policy exists for every world in the information set, (c) each world has a different omniscient winning action but no single policy succeeds, and (d) inevitable defeat in all worlds. Include exact expected outcomes from small hand-constructed fixtures.

### Tests to author and run

- world-set validation and exact observation equivalence;
- deterministic repeated model-checking result;
- action consistency across indistinguishable worlds;
- success/failure fixtures for possible and guaranteed victory;
- distinction between `∀ world ∃ omniscient policy` and `∃ observation-based policy ∀ world`;
- inevitable-defeat case and a case where defeat is not inevitable even though guaranteed victory fails;
- horizon terminal conditions and strict `score > 0.5` rule;
- budget rejection and action legality in all worlds;
- timeout/search-cap semantics;
- reset/replay and event privacy;
- no hidden world details in player-facing explanations or dashboard/API.

### Deliverables

- `src/dumbocracy/epistemic.py` and/or `model_checking.py`
- `tests/unit/test_epistemic.py`
- `tests/integration/test_epistemic_episode.py`
- finite-world fixtures and `reports/week5_model_check_results.md`

**Gate:** model-checking claims are exactly scoped and the checker's output cannot be confused with the engine's official terminal outcome.

---

## MM5.3 — Connect temporal and epistemic reasoning to actual episodes

### Tasks

1. Reuse the Week 4 canonical trace and bounded-LTL evaluator. Do not build a second unrelated trace system.
2. Add approved trace diagnostics from the source plan where their definitions can be evaluated correctly:
   - `Eventually Victory` over terminal-state `Victory` (not early support threshold crossings);
   - `Globally StateValid` and `Globally BudgetNonnegative` safety checks;
   - `Eventually InevitableDefeat` only if every `InevitableDefeat` atom is generated by the documented finite model checker and has a scope/version attached;
   - `Globally Viable` only where viability was evaluated at every state under a declared action abstraction. Do not silently infer viability from support mean.
3. Keep ordinary trace diagnostics (true/false/unknown on one trace) separate from branching/future-state checks. Each report must state `trace_only`, `finite_model`, or `incomplete` evaluation mode.
4. Implement combined reports for `PossibleVictory`, `GuaranteedVictory`, `InevitableDefeat`, `K_player(GuaranteedVictory)`, and `K_player(InevitableDefeat)` as far as the finite-world engine supports. If two names are logically equivalent under the chosen definition, document why; do not invent semantics just to make each field distinct.
5. Produce a compact “scenario table” with the five rows requested in the tentative plan: possible victory, guaranteed victory, inevitable defeat, player knows victory is guaranteed, player knows defeat is inevitable. Add columns for world set, horizon, action abstraction, result, witness/counterexample, explored nodes, and confidence/scope (exact in finite model / incomplete).
6. Do not use `InevitableDefeat` as an early-stop optimization in gameplay. A player may see an analysis warning only if product wording explicitly makes it a scoped diagnostic.
7. Preserve existing `logic.py` API behavior or version it explicitly; Week 4 tests remain regression requirements.

### Tests to author and run

- existing M0 diagnostic truth tables remain unchanged;
- trace-only eventuality is never mislabeled as a universal guarantee;
- model-check result mode/scope appears in output;
- missing worlds/unknown atoms/capped search propagate `UNKNOWN` or `INCOMPLETE`;
- UI/report summaries do not expose hidden voter data;
- repeat analysis on identical config/seeds/world-set produces identical logic results except documented timing metadata.

### Deliverables

- `src/dumbocracy/verification.py` or a documented extension of `logic.py`
- `tests/unit/test_verification.py`
- `tests/fixtures/epistemic/`
- `reports/week5_model_check_results.md`
- updated `docs/spec/logic_semantics_v1.md` only if semantics truly expands, with a version/history note

**Gate:** M0 formulas still pass the original contract; all new claims state their finite-model assumptions and scope.

---

## MM5.4 — Strategy Engine v1: uniform allocation and greedy/model-based policy

### Current baseline policies to preserve

- `NoOpStrategy`
- `FirstLegalStrategy`
- `SeededRandomLegalStrategy`

These are control baselines, not optimized policies. Keep their determinism, explicit RNG, and contract tests.

### New policies

1. **UniformBudgetStrategy** — an explicitly defined, horizon-aware baseline that distributes spending opportunities across remaining rounds, rather than always choosing the first listed action. Document its allocation rule (when it spends on broadcast vs a targeted contact vs no-op), ensure it never exceeds public budget, and specify how it handles budget too small to allocate this round.
2. **GreedyPublicModelStrategy** — a first bounded approximate policy that scores the available public action choices by predicted terminal score/score improvement under a deterministic finite scenario prior. The prior must be declared and reproducibly seeded. It may use the public observation, public history, legal-action descriptors, known v1 parameters, and its own scenario samples; it must not use the actual hidden state, true graph edges, voter supports, susceptibilities, transition seed, or engine state.
3. **Target choice rule under aggregate-only information** — when the player has no public features that distinguish eligible voters, targeted voter IDs are exchangeable. Choose a target uniformly from public eligible IDs using the strategy RNG or use an explicitly described public-only rule; do not pretend the strategy knows which voter is susceptible/high-support/high-degree.
4. **Optional network-aware strategy** — only implement as a playable policy if a reviewed observation profile exposes relevant structural features. Otherwise mark the player-facing network-aware policy blocked by the approved aggregate-only v1 observation. An `oracle` upper-bound run is allowed only through a visibly separate privileged analysis harness; never put it in the regular player-policy registry or label it deployable under partial information.
5. **Greedy action evaluation** — evaluate a declared small candidate set (no-op, broadcast when legal, and a bounded set of singleton targeted-contact candidates). Score estimated expected terminal vote share across the prior world ensemble; apply deterministic tie-breaking; document runtime/sample count and known bias. This is an approximation, not a global optimum proof.
6. **No accidental engine bypass** — every selected `CampaignAction` is still validated by the authoritative `GameEngine`. A rejection should be recorded as a failed policy episode (consistent with Week 4 runner) instead of infinite retry.
7. **Objective definitions** — report at least (a) mean terminal score/vote share, (b) victory rate, (c) mean and distribution of budget used, (d) per-unit-spend score delta with safe handling for zero spend, and (e) robustness over the declared seed/scenario set. Do not claim “optimal” when the policy is greedy/approximate.

### Strategy input contract

```python
choose(observation, prior_public_history, legal_action_descriptors, rng) -> CampaignAction
```

If a model prior is configured at policy construction, it may include fixed public assumptions and an independent deterministic model seed. It must not be initialized by passing `GameState`, a real voter tuple, or the actual graph edge list to the strategy. The experiment harness may generate synthetic worlds for policy estimation independently from the real episode state; record the generator, prior parameters, seed, and conditioning method.

### Tests to author and run

- old policies remain deterministic and compatible;
- uniform spending obeys horizon/budget constraints and does not access hidden fields;
- greedy strategy deterministically scores public candidate actions for fixed model seed and observation;
- target choice uses only public eligible IDs;
- no `GameState` or private fields are supplied to any normal strategy;
- strategy cannot mutate observation/history/engine state;
- one illegal proposal terminates that policy episode as failed with safe error information;
- strategy records never contain hidden voter arrays or edge lists;
- the greedy policy is compared against baselines on paired seeds with sufficient repeated samples; no quality claim is made from a single scenario.

### Deliverables

- update `src/dumbocracy/strategies.py`
- `tests/unit/test_strategies.py` and `tests/unit/test_experiments.py` (episode behavior is covered through the paired experiment runner; no separate strategy integration module exists)
- `docs/spec/strategy_api_v1.md` update, including prior/greedy behavior and limitations
- `docs/spec/strategy_api_v1.md` for strategy design and `reports/week5_strategy_results.md` for evaluation

**Gate:** at least one genuine policy beyond fixed/random baselines is implemented and evaluated honestly under the partial-information contract.

---

## MM5.5 — MAS/topology/resource experiments and model freeze

### Part A: add one controlled topology alternative

Current v1 supports ER and explicit edge lists. For controlled experiments, add at least one static graph generator alongside ER—recommended **ring-lattice with fixed even degree**—as an experiment-only graph construction helper that returns the existing `ElectorateGraph` contract. Keep graph topology undirected, unweighted, loop-free, and fixed within an episode. A two-community sparse generator may be added if it is implemented and validated within the milestone; it is not a blocker for the first comparison.

Requirements:
- deterministic named seed; stable node IDs;
- no loops, duplicate edges, or out-of-range endpoints;
- sparse representation; no dense N×N adjacency for 10k/100k nodes;
- actual node/edge counts, density, mean/min/max degree and component count in records;
- graph generation time separate from engine initialization and transition time;
- same N/horizon/dynamics/action parameters/initial voter seed policy across topology conditions where the comparison design supports it;
- document structural differences (e.g. ring-lattice's regular degree vs ER variability); never pretend topology treatments have identical edge sets.

### Part B: sanity/stability/dynamics experiments

Run fixed-seed scenarios for:
1. no campaign / `no_op` throughout;
2. one broadcast then no-ops;
3. repeated legal broadcasts across rounds;
4. targeted contacts on a declared public-target selection rule;
5. peer influence only (campaign contribution zero); and
6. combined peer-plus-campaign effects.

Report aggregate support trajectory, final vote share, number of player votes, victory, costs, and graph properties. Be clear that “convergence” is empirical over a fixed horizon, not a requirement to reach consensus. Include boundary/isolate/disconnected graph fixtures. Freeze the approved dynamics in `reports/week5_mas_validation.md`; only critical correctness defects may change core code, with regression tests and an explicit change note.

### Part C: resource budget sensitivity

Evaluate several budget multipliers, recommended `0.5×`, `1×`, `2×` of an explicit base budget, using exact Decimal inputs. For each factor:
- same strategy set and declared root seeds;
- same horizon/network/dynamics/action effectiveness;
- independent fresh episode per (strategy, budget, seed);
- record remaining budget, budget spent, score, victory and accepted action sequence;
- plot victory rate and mean score against budget, with sample counts and uncertainty intervals;
- do not infer causality from non-paired runs or alter action costs across budget conditions.

### Part D: results and plots

Use `matplotlib` only as an optional analysis dependency; keep experiment generation in structured JSONL/CSV and plot generation in a separate script. Required figures include:
- sample aggregate support trajectory by round for representative scenarios;
- strategy comparison (mean terminal score/victory rate and uncertainty);
- victory rate vs initial budget;
- ER vs ring-lattice topology comparison on matched declared conditions;
- campaign effect/action summary and budget use;
- optional logic/model-check scenario table visualization.

Every plot needs a title, axis labels, units/meaning, N, strategy/scenario, sample count, seed policy, and a short caption in the report. Never put real hidden voter states into public game plots. These are aggregate analysis artifacts, not a dump of internal state.

### Tests to author and run

- ring-lattice invariants for small N, even degree and wraparound; invalid parameter handling;
- same seed creates the same edge list and summary;
- graph remains sparse at 10k; no dense adjacency materialization;
- paired topology experiment produces complete provenance;
- budget multipliers preserve all non-budget parameters;
- repeat experiment aggregation is deterministic aside from timing;
- plotting consumes saved result records only and does not re-run episodes implicitly.

### Deliverables

- a deterministic experiment graph generator (e.g. `src/dumbocracy/graph_experiments.py` or similarly scoped helper)
- `examples/week5_run_experiments.py` and a separate `examples/week5_plot_results.py`
- `reports/week5_mas_validation.md`
- `reports/week5_strategy_results.md`
- `reports/week5_budget_results.md`
- `reports/figures/week5/` with PNG/SVG figures
- machine-readable outputs under `reports/results/week5/`

**Gate:** the core dynamics are frozen and the first topology/resource experiments are reproducible. Larger parameter sweeps and final statistical analysis remain Week 6 work.

---

## MM5.6 — Playable local web dashboard and transport API

### Goal

The proposal explicitly calls for an interactive web-based game/dashboard. The current repository has a transport-neutral engine and a CLI mock client but no frontend/server. Week 5 must deliver a minimal integrated browser beta; do not defer the first playable UI to Week 6.

### Implementation approach

Use a small local-only web service and plain browser assets. A reasonable implementation is FastAPI + Uvicorn with `web/` static HTML/CSS/JavaScript; the agent may choose an equally small stack if it documents the reason and provides the same contract. Add optional project extras (for example `web = ["fastapi...", "uvicorn..."]` and `analysis = ["matplotlib..."]`) instead of making the core MAS depend on UI/plot libraries. If TestClient integration tests are used, declare the needed testing dependency in the appropriate optional extra.

**Implementation update:** the production frontend is React 19 + Vite 8, rooted
in `web/`, with responsive layout, Recharts visualization, interaction tests,
and a generated `web/dist/` bundle served by FastAPI. This selection uses the
available Node/React/npm workflow while preserving the API contract and keeping
all frontend dependencies outside the core Python runtime. Node >=22.12 and
npm >=10 are documented in `pyproject.toml` frontend metadata; the npm lockfile
is authoritative for JS dependencies.

Suggested layout:

```text
src/dumbocracy/
  webapp.py            # optional FastAPI app/route adapter; no engine internals in responses
  web_sessions.py      # local in-memory session registry; stores true state server-side
web/
  package.json
  package-lock.json
  vite.config.js
  index.html
  src/
    App.jsx
    SupportChart.jsx
    styles.css
    App.test.jsx
  dist/                 # generated production assets; ignored by git
tests/integration/
  test_web_api.py
examples/
  week5_run_demo.py
```

Adapt names to repository conventions, but keep UI/API dependencies outside core engine modules.

### API behavior

Provide documented endpoints equivalent to:

- `GET /api/health` — health/version information only;
- `POST /api/games` — create a game using declared N, horizon, budget, dynamics and backend options;
- `GET /api/games/{id}/observation` — detached public observation and safe timeline/history;
- `POST /api/games/{id}/actions` — submit `action_id` and optional public `target_ids`; call authoritative `GameEngine.step`;
- `POST /api/games/{id}/reset` — reproducibly reset the session using server-side game configuration;
- `GET /api/games/{id}/result` — return `null` until terminal, then the public `GameResult` allowlist;
- `GET /api/games/{id}/analysis` — completed-trace diagnostics and any available finite-world results, with scope labels.

Final endpoint names may vary, but document every request/response schema in `docs/spec/web_api_v1.md` and test it. The API must never serialize `GameState`, voter support/susceptibility arrays, hidden edge lists, internal events, campaign effect vectors, seeds, or device tensors. Only existing public observation/result/event allowlists and explicitly approved versioned extension payloads can leave the process.

### Dashboard minimum features

1. **Start/Reset:** create/restart a seeded game with sensible default N=1,000. Allow a reviewer to select a backend explicitly (`auto`, `cpu`, `cuda:N`) and see which one was actually selected. Default settings should be safe for local use.
2. **Round/status panel:** current round, horizon, remaining budget, aggregate support, terminal state, graph metrics, selected backend/device, and a small safe event feed.
3. **Action panel:** targeted contact (select one public eligible voter ID), broadcast, and no-op. Show cost and expected action scope before confirmation. Disable actions after terminal; display clear rejection errors without fabricating a round advance.
4. **Trajectory charts:** aggregate support by round, budget remaining/spent by round, action/cost timeline, terminal vote share and victory/defeat.
5. **Analysis panel:** bounded-temporal results; when finite-world data is supplied, possible/guaranteed/inevitable diagnostics with world count, action abstraction and exact/incomplete scope. Avoid showing hidden state or presenting a finite-world result as universally true.
6. **Network panel:** show the *permitted* graph summary (N/E/density/degree/component metrics). Do not render or transmit the actual hidden edge list. A schematic illustration must be labelled illustrative, not the actual hidden graph.
7. **No internet dependency required at runtime:** bundle local JS/CSS and use native SVG/canvas for simple plots or local assets. Do not rely on an external CDN that may fail during the demo.
8. **Session/resource guards:** local development beta only; cap N/horizon and number of live sessions; reject malformed actions/config; do not expose admin/private debug endpoints to the browser. Document that in-memory sessions are not durable production storage.

### Acceptance tests to author and run

- route tests for create → observe → accepted action → observe → terminal result → reset;
- rejected and unaffordable actions return a safe error and keep authoritative gameplay fields/round unchanged;
- unknown game ID/config/action handling;
- recursive response scan proving no private fields, support arrays, edge lists, seed fields, or hidden state objects leak;
- UI/client never imports `GameState` and only uses API responses;
- public observation `schema_version` is tested and compatible;
- action accepted in the dashboard matches the engine contract and displayed budget update;
- terminal UI displays official `GameResult`, not client-computed victory;
- browser/network failure and malformed JSON have readable error handling;
- game/backend metadata displays CPU fallback as CPU, not as successful GPU execution.

### Reviewer walkthrough (mandatory)

The handoff must tell each reviewer how to start the server, open the page (including SSH port forwarding if the server is remote), start a game, execute at least one action of each applicable type, trigger terminal state, inspect result/analysis, reset/replay, and confirm the public response does not contain hidden data. Capture screenshots or a short local screen recording if permitted; include no private arrays in artifacts.

### Deliverables

- optional web dependency extras in `pyproject.toml`
- `src/dumbocracy/webapp.py` and session/adapter module
- local `web/` assets
- `tests/integration/test_web_api.py`
- `docs/spec/web_api_v1.md`
- `reports/week5_ui_walkthrough.md` with actual executed results and screenshots where useful

**Gate:** a reviewer can play a complete episode in the browser using the actual engine, with no mock electorate state and no hidden-state disclosure.

---

## MM5.7 — GPU-first execution and representative network scale

### Requirements

1. Existing Week 3/4 CUDA implementation remains accepted. Do not rewrite it just to make Week 5 look new.
2. Use GPU for numerical voter transitions in at least the 10k and 100k-node scenarios when a compatible GPU-visible environment is available. CPU is required for small correctness/reference and finite model checking; the finite epistemic search need not use GPU.
3. A benchmark command that reports `backend=cpu` because `auto` fell back is **not** evidence of GPU execution. For GPU evidence use an explicit device such as `cuda:0`, confirm the device is visible, and record the actual device reported by PyTorch.
4. The server baseline has been documented as two RTX A5000 GPUs, but a Week 4 GPU-visible run reported three. Discover `torch.cuda.device_count()` and `nvidia-smi -L` at runtime; record every device's model and VRAM separately. Do not sum VRAM for capacity planning.
5. Preserve sparse graph data; never allocate a dense N×N adjacency matrix for the 100k workload.
6. The main scale test must target **N = 1,000; 10,000; 100,000** nodes with ER `p = min(1, 8/(N-1))`, fixed recorded seed, configured rounds and declared action schedule. If graph class/topology changes, record the topology and expected degree accordingly.
7. Run at least three measured repetitions after warmup for performance summaries if resources permit. Synchronize CUDA around timed steps. Report initialization breakdown when feasible, median and spread of per-round latency, episode wall time, allocated/reserved VRAM, free/total VRAM per device, N/E/density/degrees/components, actions, final score/victory, versions/driver, commit, and correctness status.
8. Perform matched CPU/CUDA comparisons at least at N=1,000 and N=10,000 on identical config, seed, graph, initial voters, action sequence, and horizon. Use the accepted `1e-10` support tolerance (or retain stricter measured differences); discrete outcomes, action acceptance, Decimal ledger and result must match. A 100k CPU comparison is optional if expensive; 100k CUDA execution is the key scale target.
9. Do not claim general speedup based on one run. Report individual measured observations and the distribution/repeats. If host load/VRAM limits affect results, record the cause and rerun under a controlled condition when practical.
10. If `nvidia-smi` fails only inside an execution sandbox while the remote host is expected to have GPUs, follow the existing Week 4 handoff: check a GPU-visible context, compare PyTorch CUDA visibility and host `nvidia-smi`, do not change drivers, and do not classify a CPU fallback as CUDA.
11. Profile only after correctness passes. Capture bottleneck evidence for transfer, Python tuple reconstruction, graph construction/summary, `index_add_`, synchronization, and history copying. Persistent device-resident state or multi-GPU execution remain out of scope unless evidence shows a material need and a reviewed design addresses immutable snapshots/rollback/replay/per-device VRAM.

### Suggested new artifacts

- `examples/week5_gpu_scale_benchmark.py` (multi-size, repeated synchronized runs)
- update `examples/week4_backend_comparison.py` only if its outputs need compatibility/CLI changes; preserve its earlier recorded evidence
- `reports/week5_performance_results.md`
- `reports/results/week5/gpu_scale.jsonl`
- `reports/results/week5/backend_comparison.jsonl`

**Gate:** at least one actual GPU-backed run reaches 100k nodes where available VRAM permits, plus matched CPU/GPU correctness at 1k and 10k. If hardware access genuinely prevents it, provide the diagnostic outputs and mark the GPU-specific gate blocked; never substitute CPU timing.

---

## MM5.8 — Reproducible experiment suite and analysis products

### Experiment design

All experiment settings must be explicit and reproducible. Use deterministic seed lists for the first analysis; recommended seed sets are at least 20–30 paired seeds for low-cost N=1k comparisons, a smaller declared set (e.g. 5–10) for N=10k if runtime requires it, and 3–5 for the 100k scale/scenario sanity run. This is a budget-conscious preliminary Week 5 study, not the final statistical campaign.

**A. Strategy comparison**

Compare available policies such as NoOp, SeededRandomLegal, UniformBudget, FirstLegal, and GreedyPublicModel. Use the same config/scenario/root seeds per strategy; initialize fresh state, policy, history and RNG for each episode. Record failures rather than silently dropping them. Report sample count, completed/failed/skipped counts, mean score, population standard deviation or a defined confidence interval, victory rate, budget used, and rejected proposals. State that outcomes are paired by declared seed but post-action trajectories differ by policy.

**B. Network topology**

Compare ER and the experimental ring-lattice at matched N/horizon/alpha/action configuration, with fixed declared graph-seed treatment and the same initial voter seed policy where meaningful. Report actual graph metrics and effect on terminal support/victory. Do not imply the graph instances are identical across topology conditions.

**C. Budget sensitivity**

Compare `0.5×`, `1×`, `2×` of a declared base budget. Keep other game parameters and seed list fixed, and use exact Decimal budgets. Plot victory rate and mean score vs initial budget, and budget used. Include episode denominators and error bars/uncertainty summaries.

**D. Campaign/peer-influence sanity cases**

Run no-op/peer-only, targeted, broadcast, and combined influence scenarios with exact small expected fixtures plus larger simulation summaries. Describe bounded trajectory behavior and influence spread without requiring unanimous consensus.

**E. Information availability**

Only execute the 25/50/75/100% or Low/Medium/High profile experiment if MM5.1's approval gate has been passed and the new opt-in profiles are implemented and tested. If not, report aggregate-only baseline behavior and document the exact blocked question/decision. Never fabricate information-level metrics or leak voter-level fields under the default v1 schema.

**F. Epistemic verification scenarios**

Use hand-constructed finite world sets that show the difference between possible, guaranteed, and inevitable outcomes. Save machine-readable results and a human-readable table including world count/action abstraction/search cap/result scope. The result is an exact check on that specified finite model, not a proof about all continuous hidden states.

### Output storage and reporting

- Save per-episode JSONL in `reports/results/week5/`.
- Save aggregate CSV/JSON files and a config/seed manifest next to each result set.
- Save plots in `reports/figures/week5/` as PNG and, if useful, SVG.
- Generate plots from saved JSONL/CSV—not by implicitly re-running costly experiments when plotting.
- Every result set must include its exact command, exit status, start/end timestamp, commit SHA or `uncommitted`, Python/NumPy/PyTorch/CUDA/driver information, backend/device, graph size, horizon, seed list, scenario/policy names, and artifact paths.
- Never write voter support/susceptibility arrays, hidden edges, private internal events, or device tensors to experiment result files. Use aggregate trajectories and safe summaries only.
- The agent must save actual observed result rows; the report template's placeholders and expected outputs are not evidence.

### Deliverables

- `examples/week5_run_experiments.py`
- `examples/week5_plot_results.py`
- `reports/week5_strategy_results.md`
- `reports/week5_mas_validation.md`
- `reports/week5_budget_results.md`
- `reports/week5_information_results.md` (or an explicit `BLOCKED_PENDING_DECISION` report)
- `reports/week5_model_check_results.md`
- `reports/results/week5/` machine-readable outputs/manifest
- `reports/figures/week5/` plots

**Gate:** the main report can answer what happened, why it happened within the model's diagnostic limits, how strategies differ, and what the declared experimental design does and does not establish.

---

## MM5.9 — Integrated regression, reviewer reproduction, and handoff

### Required code/report files

At minimum deliver:

- `reports/week5_baseline_audit.md`
- `reports/week5_requirements_traceability.md`
- `docs/spec/epistemic_semantics_v1.md`
- `docs/spec/observation_profiles_experiment.md` and a decision record/proposal with approval status
- `src/dumbocracy/epistemic.py` / `model_checking.py` and tests
- `src/dumbocracy/verification.py` if needed
- updated `src/dumbocracy/strategies.py` and tests
- experiment topology helper and updates to experiment runner
- `src/dumbocracy/webapp.py` and server-side session adapter
- React/Vite frontend rooted at `web/`; generated bundle served from `web/dist/`
- `tests/unit/test_epistemic.py`, `test_verification.py`, updated `test_strategies.py`, plus any graph/experiment tests
- `tests/integration/test_epistemic_episode.py`, `test_web_api.py`; policy episode coverage is in `tests.unit.test_experiments.py`
- `examples/week5_gpu_scale_benchmark.py`, `week5_run_demo.py`, `week5_run_experiments.py`, `week5_plot_results.py`
- `reports/week5_performance_results.md`, `week5_strategy_results.md`, `week5_mas_validation.md`, `week5_budget_results.md`, `week5_model_check_results.md`, `week5_ui_walkthrough.md`
- `reports/week5_self_review.md`
- `docs/spec/week5_handoff.md`
- updated `docs/spec/test_plan.md`

Use equivalent names where repository conventions justify it, but keep every requirement mapped to a real artifact.

### Integration exit criteria

- a local user can start an actual engine-backed browser game, perform actions, progress to the horizon and view the outcome;
- actual public observation and result allowlists are used by the API/client;
- model checking is separate, finite, bounded, deterministic and scope-labeled;
- the baseline and greedy policy comparison runs with explicit seeds;
- topology and budget experiments have actual saved outputs and readable plots;
- information-level experiments are either implemented under an approved opt-in schema or explicitly marked blocked with no fabricated outputs;
- GPU scale and CPU/CUDA parity have actual run evidence with device/driver/VRAM information;
- no broad optimizer, episteme, performance, or victory-guarantee claim exceeds the evidence;
- Week 6 tasks are clearly identified: larger experimental matrix, uncertainty analysis, final MAS/strategy/information/logic reports, UI refinement, final demo/presentation.

---

## 4. Command execution and testing policy — agent MUST RUN AND REPORT

**Unlike Weeks 3/4's earlier restrictions, Week 5 explicitly authorizes the coding agent to run commands. Do not interpret this plan as a static-only task.** During implementation and validation, the agent must run relevant tests, validators, demos, server/API checks, GPU measurements, static checks, and profiling/benchmark commands when needed. If a command fails, inspect the failure, fix the underlying issue, and rerun the relevant command. Do not suppress failures or substitute expected output for actual output.

### General rules

1. Before editing, inspect current worktree and protect user changes. During the task, use normal Git/read-only commands and permitted commits according to the project's workflow; do not force-push or erase unrelated changes.
2. Run focused tests at each mini-milestone and the full suite at MM5.9. Use test-discovery paths supported by the actual repo.
3. Execute the M0 artifact validator after documentation/fixture changes.
4. Run GPU benchmarks in a GPU-visible shell/SSH environment. `auto` fallback to CPU must be visibly reported; a CPU run cannot satisfy a GPU requirement.
5. Record exact command, environment, backend/device, start/end time, exit status, summary counts, and output artifact path. Save stdout/stderr in reproducibility logs; redact only truly sensitive machine/user information, not correctness evidence.
6. Record failures as well as passes and document any reruns. The final report should show the initial failure and the fix if meaningful, not only the last green result.
7. Do not report machine-specific timings as general speedup claims. Use repeats, medians, spreads, same config/seeds/actions, CUDA synchronization, and separate per-device VRAM.
8. At the end run a full suite, M0 validator, required demo/API tests, selected GPU scale test, matched CPU/CUDA parity, plot/report generation, and Git diff/status review. `git diff --check`, compile checks and linters are allowed and expected where available.
9. If the environment hides GPU devices, follow the Week 4 handoff's GPU-visible-context troubleshooting; do not change drivers or label CPU fallback as GPU success.
10. If optional GPU or UI dependencies are missing, install only from the declared project extras. Update `pyproject.toml` so a reviewer can reproduce the environment with one documented install command.

---

## 5. Environment setup and initial checks (agent + reviewers)

Use the same repository commit when comparing results. On the documented SSH server, activate the intended environment:

```bash
conda activate mas
python --version
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import torch; print('torch', torch.__version__, 'torch_cuda_build', torch.version.cuda, 'cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i), torch.cuda.mem_get_info(i)) for i in range(torch.cuda.device_count())]"
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader
```

If the Week 5 plan adds optional extras, after `pyproject.toml` is updated install the project using the committed extra names, for example:

```bash
python -m pip install -e ".[gpu,web,analysis]"
```

The agent must confirm that these exact extras exist and that the resulting environment is documented. Core tests should remain runnable on CPU without FastAPI/matplotlib if the implementation keeps those extras optional. Do not assume CUDA runtime equals the host driver version; report PyTorch's build/runtime fields and the driver separately.

---

## 6. Required executable test matrix and expected outputs

The exact new test module names below are part of this plan. If the implementation uses different names, the coding agent must update this section and `docs/spec/week5_handoff.md` to the actual committed paths; do not leave broken commands in the runbook.

### Stage A — M0 validator and full unit/integration suite

```bash
python tests/validate_week1_artifacts.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

**Expected output/acceptance:** validator reports M0 documentation/fixture success; the test suite ends with `OK` and zero failures/errors. Optional tests may skip only when their declared capability is genuinely absent; on the GPU-visible target environment, the CUDA-specific tests should execute rather than skip. Save exact test count and skip count.

**Failure diagnostics:** inspect the named test, check imports/dependency versions, and use the relevant file (`config.py`, `graph.py`, `mas.py`, `campaign.py`, `engine.py`, `information.py`, `logic.py`, strategy/epistemic/API modules). Do not weaken an assertion to make it green unless its expected behavior conflicts with the frozen contract and the decision is documented.

### Stage B — focused Week 5 model checker and verification tests

```bash
PYTHONPATH=src python -m unittest tests.unit.test_epistemic -v
PYTHONPATH=src python -m unittest tests.unit.test_verification -v
PYTHONPATH=src python -m unittest tests.integration.test_epistemic_episode -v
```

**Expected:** hand-calculated finite-world fixtures report the declared result; guaranteed-win logic enforces a shared observation-based policy; model caps yield `INCOMPLETE`; exact terminal and public privacy rules remain valid. Tests must include at least one mixed-outcome information set.

### Stage C — strategy and experiment tests

```bash
PYTHONPATH=src python -m unittest tests.unit.test_strategies -v
PYTHONPATH=src python -m unittest tests.unit.test_experiments -v
```

**Expected:** NoOp/FirstLegal/SeededRandom continue to pass; uniform and greedy policies produce legal proposals from public inputs only; repeated fixed-seed runs reproduce action/result records excluding timings; aggregates include failed episodes with documented denominators.

### Stage D — graph and MAS experiment tests

```bash
PYTHONPATH=src python -m unittest tests.unit.test_graph_experiments -v
PYTHONPATH=src python -m unittest tests.unit.test_experiments -v
```

**Expected:** ER and ring-lattice both satisfy the same base graph invariants; generated graphs reproduce for fixed seed; experiment outputs record N/E/density/degree/components; graph treatment differences are documented.

### Stage E — web/API integration tests

```bash
PYTHONPATH=src python -m unittest tests.integration.test_web_api -v
```

**Expected:** Start → observe → valid action → round update → terminal result → reset passes through the real engine. Rejected action does not advance the authoritative round or debit budget. Recursive response scans find no private voter arrays, susceptibility, hidden edges, seed fields, device tensors, or `GameState` serialization. Route tests run without needing a browser; a separate manual walkthrough is required.

### Stage F — public mock-client regression

```bash
PYTHONPATH=src python -m unittest tests.unit.test_mock_client -v
PYTHONPATH=src python examples/week3_mock_client.py < tests/fixtures/public_observation_v1.json
```

**Expected:** test suite passes and the command emits one JSON object with the action schema, typically `{"action_id":"targeted_contact","target_ids":["v0"]}` for the committed fixture. This is a client-contract test, not a full web UI test.

### Stage G — strategy comparison and generated plots

Example policy comparison command (agent must implement this CLI and keep it consistent with the docs):

```bash
PYTHONPATH=src python examples/week5_run_experiments.py --experiment strategies --backend cuda:0 --nodes 1000 10000 --seeds 11 23 37 41 53 --rounds 5 --strategies no_op seeded_random_legal uniform_budget greedy_public_model --output-dir reports/results/week5/strategies
```

Expected artifacts include one JSONL record per strategy/scenario/seed, an aggregate JSON/CSV file, a manifest, and a terminal summary showing completed/failed/skipped/rejected counts. There must be no hidden per-voter arrays in the records.

Generate plots only from saved results:

```bash
PYTHONPATH=src python examples/week5_plot_results.py --input-dir reports/results/week5 --output-dir reports/figures/week5
```

**Expected:** PNG/SVG files for strategy outcomes and the budget/topology experiment sets that have completed. The script should report missing/blocked experiment categories rather than filling them with made-up values.

### Stage H — GPU scale run (must attempt on actual GPU)

```bash
PYTHONPATH=src python examples/week5_gpu_scale_benchmark.py --device cuda:0 --nodes 1000 10000 100000 --seed 20261019 --rounds 5 --repeats 3 --output reports/results/week5/gpu_scale.jsonl
```

**Expected:** three JSONL rows for 1k/10k/100k, or an explicit failed/skipped row for any size not completed with a measured VRAM/resource reason. Each successful row names (eg:`cuda:0`), the concrete GPU model, N/E/density/degree, p/seed, rounds, init and median/spread step timings, episode time, peak allocated/reserved VRAM, free/total VRAM at start, PyTorch/CUDA/driver versions, and revision. 100k is the target scale; do not quietly lower it to 10k.

If the selected CUDA device is not visible, run the documented GPU-visibility diagnostics; `--device cuda:0` should fail loudly rather than silently switch to CPU. A separate CPU smoke can still be run for troubleshooting but does not satisfy this stage.

### Stage I — matched CPU/CUDA correctness and timing

Preserve and execute the accepted Week 4 comparison tool at both supported sizes:

```bash
PYTHONPATH=src python examples/week4_backend_comparison.py --nodes 1000 --seed 20261019 --rounds 5 --cuda-device cuda:0
PYTHONPATH=src python examples/week4_backend_comparison.py --nodes 10000 --seed 20261019 --rounds 5 --cuda-device cuda:0
```

Expected JSON reports identical initialization, final support max absolute difference `<= 1e-10`, equal ledger and terminal result, plus backend timing/memory. It must exit nonzero on parity failure. Repeat at least once for performance stability; report medians/ranges, not just the fastest run. Optionally repeat with `cuda:1` if present and useful; treat it as a separate device.

### Stage J — full GPU-backed episode and browser/API run

```bash
PYTHONPATH=src python examples/week5_run_demo.py --backend cuda:0 --nodes 1000 --seed 20261019 --rounds 5 --strategy greedy_public_model --output reports/results/week5/demo_episode.json
```

Expected: full episode summary and saved artifact including selected backend, seed/config/revision, actions, aggregate support trajectory, budget history, final vote share/victory, bounded logic diagnostics and any scoped model-check output. No hidden per-voter array is written. A distinct 10k or 100k engine smoke should be recorded in the GPU scale stage.

Launch the web beta:

```bash
PYTHONPATH=src python -m uvicorn dumbocracy.webapp:app --host 127.0.0.1 --port 61007
```

In another terminal, verify service health and create a game (the agent must ensure these exact paths/fields match the implemented API and record the final canonical command in the handoff):

```bash
curl -fsS http://127.0.0.1:61007/api/health
curl -fsS -X POST http://127.0.0.1:61007/api/games \
  -H 'content-type: application/json' \
  -d '{"node_count":1000,"horizon_rounds":5,"initial_budget":"20.00","seed":20261019,"backend":"cuda:0"}'
```

Use the returned `game_id` to get the public observation, post each legal action, and retrieve the terminal result/analysis. The coding agent must include complete copy-paste curl commands and example *shape* for all actions/responses in `docs/spec/web_api_v1.md`. Never leave reviewers to invent an API payload.

**Browser manual flow:** open `http://127.0.0.1:61007` in a browser when running locally. If the server is remote, document the SSH local-port-forwarding form for the actual server/port; do not hard-code the user's hostname. Start a game, verify round/budget/aggregate display, run targeted contact, broadcast when legal, no-op, reach terminal, inspect victory/defeat and charts, read formal analysis scope, reset/replay, and inspect the API response to confirm no hidden fields.

### Stage K — formatting/compile and final Git review

```bash
python -m compileall -q src/dumbocracy tests examples
git diff --check
git status --short
```

If a configured linter/type checker exists, run it and record version/output; do not invent a linter not present in the project. Review the full diff and generated artifact sizes.

**Expected:** compile check succeeds; `git diff --check` reports no whitespace errors; status/diff show only intended source/tests/docs and generated result/report files. Preserve raw logs but avoid committing huge duplicate arrays or temporary caches.

---

## 7. Five-member reviewer reproduction plan

The tentative plan assigns primary ownership to Mad (strategy/optimization), Deek (MAS/model), Hem (resource/information), Psk (temporal/epistemic logic), and Vis (UI/visualization). Use this as reviewer focus, without isolating people from integration review.

| Reviewer role from the plan | Focused reproduction and evidence |
|---|---|
| **Mad — Strategy/Optimization** | Run strategy unit/integration tests; execute paired strategy comparison on declared seeds; review greedy prior assumptions, uniform spending, score/victory/budget metrics, and the saved strategy plots. Confirm no policy receives hidden state. |
| **Deek — MAS/Graph** | Run MAS/graph tests, ER/ring-lattice invariants, CPU/CUDA parity, and GPU scale command; review graph metrics, support trajectories, influence effects, and `reports/week5_mas_validation.md`. |
| **Hem — Resources/Information** | Run campaign/observation/API tests; verify Decimal budgets, action costs, round-local effect expiry, rejected-action atomicity, and the observation-profile decision status. Confirm no unapproved disclosure. |
| **Psk — Temporal/Epistemic Logic** | Run the full logic and epistemic suites; hand-check the scenario table; inspect world-set/action abstraction/scope, unknown/incomplete behavior, and knowledge-vs-omniscient distinctions. |
| **Vis — UI/Visualization** | Run web API tests; follow the browser walkthrough; test start/action/rejection/terminal/reset; inspect charts, labels, errors, responsive layout, and all public payloads. |

### Reproduction protocol

1. All reviewers check out the same Week 5 commit SHA, activate the documented environment.
2. Every reviewer runs Stage A (M0 validator + full suite) and the focused stages for their area. At least one GPU-visible reviewer runs Stages H and I end-to-end; ideally all who have compatible hardware repeat them.
3. Compare deterministic fields exactly where expected. Exclude/allow timing fields to vary; do not demand identical wall-clock durations or device availability across machines.
4. The agent must include a one-command reproduction bundle or a complete command list in the handoff. If a command is intentionally expensive (100k repeats), state why, and distinguish core validation from optional reruns—but do not omit the large GPU result from the agent's own record when hardware is available.

---

## 8. Required saved evidence and output schemas

Every execution result should be reproducible from saved configuration and command data. At minimum, every episode record should include:

- schema version, run ID, scenario ID, strategy name/version;
- root seed and named derived seeds, game config, graph generator/parameters;
- actual node/edge counts, density, degree and component summaries;
- horizon, initial/remaining/spent budget, dynamics parameters and action cost/effect parameters;
- exact accepted/rejected action sequence with public IDs and safe stable error codes;
- terminal `player_votes`, electorate size, vote share, victory, score and budget used;
- Python, NumPy, optional PyTorch/CUDA runtime, driver version, backend and device model;
- repository revision, command, exit status, timings with definitions, and output file path;
- temporal diagnostic results and finite model-checking results with world set, action abstraction, explored nodes, limits, witness/counterexample and result scope;
- failures/skips/incomplete states counted explicitly with a safe failure type/stage.

Never include hidden per-voter arrays, susceptibility, true seed streams in the public game response, internal private event attributes, graph edge lists in public payloads, or device tensors in logs/results. For reviewer-only internal fixture files, do not store real session state just to improve plots.

### Required report bundle

- `reports/week5_self_review.md` — done/partial/not done, commands run, pass/fail/skip counts, environment, defects/fixes, output paths, limitations, Week 6 backlog.
- `docs/spec/week5_handoff.md` — exact install/run commands, CLI help/args, API schemas, expected output shapes, diagnostics, browser walkthrough, GPU troubleshooting, reviewer roles, and actual observed results.
- `reports/week5_strategy_results.md` — strategy definitions/constraints, paired seed design, metrics, plots, limitations.
- `reports/week5_mas_validation.md` — graph/dynamics sanity checks, trajectories, model freeze statement.
- `reports/week5_budget_results.md` — budget factors and results.
- `reports/week5_information_results.md` — actual approved-profile results or clearly marked blocked decision.
- `reports/week5_model_check_results.md` — scenario definitions, result tables, finite-world/action-abstraction scope.
- `reports/week5_performance_results.md` — GPU/CPU runs, device/driver/versions, repeated timings and memory; historical Week 3/4 numbers separated from Week 5 measurements.
- `reports/week5_ui_walkthrough.md` — manual browser/API steps, actual responses, screenshots if useful, privacy checks.
- `reports/week5_baseline_audit.md` and `reports/week5_requirements_traceability.md`.
- Updated `docs/spec/test_plan.md` and `pyproject.toml` extras/dependency notes.

---

## 9. Week 5 definition of done and execution status (2026-10-27)

Implementation and automated verification are substantially complete. The
following are based on recorded test/build/benchmark evidence; the human browser
walkthrough and five-person Week 5 acceptance remain open.

### Product/gameplay
- [x] Engine-backed React/FastAPI UI, public observation/action flow, official result and reset are implemented and exercised by UI/API tests.
- [x] UI/API uses the engine's result and public allowlists; recursive privacy checks pass.
- [x] Responsive aggregate trajectory/action/budget/outcome views and permitted graph summary are implemented.
- [ ] Reviewer performs TCP/browser playthrough at desktop and phone widths; current executor denied TCP binds (see `reports/week5_ui_walkthrough.md`).

### Strategies and experiments
- [x] Baseline policies retained; uniform-budget and greedy public-model policies are implemented.
- [x] Strategy comparison has paired-seed machine records and plots.
- [x] Ring-lattice topology and exact-Decimal budget sensitivity comparisons are recorded.
- [x] MAS sanity/influence report exists; approved engine dynamics were not changed.
- [x] DEC-012 approves aggregate-only v1. Richer information profiles remain unauthorized; no profile results are claimed.

### Logic/verification
- [x] Bounded temporal regressions pass.
- [x] Finite-world checker is bounded/deterministic and transitions via the authoritative engine.
- [x] Shared-policy consistency, knowledge scope, search limits and incomplete semantics are tested/reported.
- [x] Inevitable defeat is analysis-only; gameplay remains fixed-horizon.

### GPU/performance/reproducibility
- [x] Explicit CUDA benchmark reached N=100,000 with three repeats.
- [x] CPU/CUDA parity at 1,000/10,000 passed within `1e-10` with equal ledger/result.
- [x] Device/driver/runtime, timing and per-device memory are recorded; no pooled VRAM/general speedup claim is made.
- [x] Reproduction commands and result paths are in the reviewer handoff.

### Quality/review
- [x] M0 validator passed; full unittest suite passed 82/82, focused requested suite passed 18/18, with no skips.
- [x] Frontend suite passed 3/3; production build, FastAPI integration, compile check and privacy checks passed.
- [x] Required report bundle, result manifests, figures and handoff are present.
- [ ] `git diff --check` is not fully clean because two pre-existing trailing-space lines remain in modified Week 3/4 handoffs; they were not changed as unrelated. Reviewer should inspect the complete working diff.
- [ ] Five-member Week 5 review and browser walkthrough on a common reviewed revision remain open; DEC-012 approval alone is not Week 5 acceptance.

---

## 10. Week 6 backlog (do not let this week's scope grow without limit)

The six-week source plan puts final refinement and demonstration in Week 6. Leave the following explicit unless a small necessary portion is required to unblock the Week 5 beta:

- larger strategy seed sets and final statistical robustness analysis;
- extended topology matrix beyond ER and the first approved alternative;
- full information-level comparison if it needs more world/observation-profile work than fits Week 5;
- additional resources/actions or rival campaigns, only with a new model decision;
- deeper optimization/search, multi-step planning beyond the bounded greedy/model checker, evolutionary methods or RL;
- larger or more complete epistemic-world construction and any claim intended to generalize beyond the finite world set;
- persistent GPU-resident authoritative state or multi-GPU/distributed graph/state partitioning;
- production hosting, authentication, persistence, multi-user concurrency, and deployment security hardening;
- remaining UI polish after the responsive, labelled browser flow lands (reviewer browser pass, visual/accessibility QA, and fixes from that pass);
- final figures/reports, 5–10 minute live demo scenario, presentation story and summary of limitations.

---

## 11. Required final report format from the coding agent

1. **Status:** complete / partial / blocked.
2. **Commit/branch/revision and worktree status.**
3. **Week 4 backlog W4-B01–W4-B08:** done/partial/not done, with exact proof/file paths.
4. **Mini-milestones MM5.0–MM5.9:** status and checkpoint outcomes.
5. **Original problem alignment:** how the game/dashboard/strategy/formal layer/experiments now answer the proposal goals.
6. **Epistemic semantics:** definitions, finite world count, action abstraction, bound/caps, exact vs incomplete results, and known limitations.
7. **Strategy definitions/results:** public information used, baselines, greedy assumptions, paired seeds, metrics and plots.
8. **MAS/topology/budget/info experiments:** configs, results, plots, sample counts, decision gate status, model-freeze statement.
9. **Web beta:** startup command, API paths, browser walkthrough, privacy checks and screenshots.
10. **GPU results:** device count/model/per-device VRAM, PyTorch/CUDA/driver, actual selected backend, 1k/10k/100k measurements, parity, repeats and limitations.
11. **Tests/commands:** exact executed commands, exit codes, pass/fail/skip counts, relevant output excerpts, logs and artifact paths. Include failing attempts and fixes when meaningful.
12. **Generated outputs:** JSONL/CSV/PNG/SVG paths and manifests.
13. **Files changed / dependency extras / installation command.**
14. **Remaining defects, risks and Week 6 backlog.**
15. **Five-member review state:** do not claim review is complete until the actual review of Week 5 code and evidence has occurred.
16. **Week 6 readiness:** ready / blocked, with reasons.

**Stop condition:** deliver a playable research beta, bounded finite-world epistemic verification, strategy/model experiment results, a GPU-backed 100k scale record where the hardware allows it, the report/plot bundle, and a complete reproduction handoff. Do not quietly defer the UI or model checker to Week 6, and do not claim universality beyond the declared finite model. Once MM5.9 is satisfied, stop for the five-member review and Week 6 finalization.
