# `Implim_Week1.md` — Dumbocracy Implementation Plan (Week 1)

> **Milestone:** M0 — Technical Specification v1  
> **Status at start:** Week 1 has not started. This document is the coding-agent execution plan, not evidence that the milestone is already complete.  
> **Primary objective:** turn the project proposal and roadmap into a reviewable, internally consistent specification that the five team members can approve before parallel Week 2–3 implementation begins.

---

## 1. Mission and definition of success

Dumbocracy is an interactive, graph-based multi-agent election-campaign resource-allocation game. The player spends a finite campaign budget on campaign resources/actions; voters have individual attributes and interact through a social graph; the player sees an observation of the underlying state rather than necessarily seeing everything. The project also needs explicit temporal/epistemic properties and a way to compare strategies.

**Week 1 is a specification-and-contract milestone. It is not the week to build the full simulator, optimization system, formal verifier, or finished web UI.** The dependency graph makes M0 the prerequisite for the parallel Week 2–3 workstreams: Baseline Strategies, MAS Model, Resource Engine, Information Model, UI Prototype, and Temporal Model.

By the end of Week 1, a new developer should be able to answer, from the approved documents alone:

1. What is the game state, and what changes at each round?
2. What is a voter/agent, and what data does it own?
3. How is the electorate graph represented and generated?
4. What actions can a campaign take, what do they cost, and what constraints apply?
5. How do campaign actions and voter-to-voter influence affect voter state?
6. What is hidden, what is observable, and how are observations produced?
7. What counts as victory, defeat, a score, and an invalid action?
8. What APIs must the MAS, campaign/resource engine, observation model, game engine, strategy layer, and UI expose?
9. Which temporal and epistemic properties will be checked against the model?
10. What is fixed for Week 2, what remains an explicitly recorded decision, and who approves it?

**Completion gate:** do not declare M0 complete merely because the Markdown exists. All mandatory decisions below must be either (a) approved and frozen in Technical Specification v1, or (b) explicitly listed as an unresolved blocker with an owner and deadline. The team must not silently turn assumptions into requirements.

---

## 2. Project constraints and engineering assumptions

These assumptions were supplied by the project owner and must be preserved in later implementation plans.

- Development, execution, and testing happen on an SSH-accessible server.
- The server has **2 × NVIDIA A5000 GPUs, 48 GB VRAM combined**. Treat this as two separate GPUs (typically 24 GB each); do **not** assume their VRAM automatically behaves like one 48 GB allocation.
- Coding agents are part of the development workflow. Each weekly `Implim_WeekN.md` is an actionable plan for an agent to implement, test, diagnose failures, fix defects, rerun tests, and produce review evidence.
- All **five group members** manually validate the agent's code, tests, outputs, and documentation before the team moves to the next week's work.
- Agent completion is not acceptance. Human review and sign-off are required.

### Engineering implications

- Use Python as the initial simulation/modeling language unless the repository already establishes a different approved stack. Keep the model API independent of any UI framework.
- Prefer vectorized NumPy/PyTorch operations over per-agent Python loops for large state updates. Keep graph storage sparse when the graph is sparse.
- Use GPU acceleration where it materially improves throughput, especially for batched experiments, large matrix/tensor updates, or many independent simulation runs.
- Do **not** move all work to the GPU by default. Graph generation, orchestration, validation, small simulations, and many NetworkX operations may be faster or simpler on CPU. Benchmark representative workloads before choosing a backend.
- Do not introduce multi-GPU distributed training or DDP just because two GPUs exist. Use one GPU or CPU as a baseline; use multi-GPU execution only for a measured workload that benefits from it.
- Every experiment must record seed, configuration, code version/commit, device/backend, and relevant environment versions.
- Avoid requiring a GPU for unit tests. Core correctness tests must run on CPU; optional GPU tests should be marked and skipped when CUDA is unavailable.
- No Week 1 deliverable may require large-scale GPU use. The purpose of the hardware is to enable later scale, not to replace sound interfaces and tests.

---

## 3. Source-of-truth hierarchy

The coding agent must inspect the project files before editing and use this order of authority:

1. Project proposal and explicitly approved decisions from the project owner/team.
2. The dependency graph and project roadmap for sequencing and hand-off dependencies.
3. This implementation plan for Week 1 tasks and acceptance criteria.
4. Supporting theory notes (MAS/consensus, graph and linear algebra, temporal/epistemic logic, infection model, proximal methods, ADMM, distributed learning) as explanatory references—not as proof that a particular project design has already been selected.
5. Coding-agent suggestions, which are proposals until reviewed.

Do not silently resolve conflicts among sources. Record the conflict in `docs/decisions/` (or the repository's existing decision-record location) and ask the team to decide.

**Important modeling distinction:** consensus is a useful reference for graph dynamics, but an election model is not required to converge to consensus. Political disagreement and polarization can be legitimate states. Resilient consensus/W-MSR, Erdős–Rényi graphs, ADMM, and proximal methods are candidate techniques or study topics, not automatically mandatory components of the first implementation.

---

## 4. Week 1 scope

### In scope

- Inspect and summarize the current repository structure, conventions, dependencies, and existing tests.
- Produce Technical Specification v1 (`docs/spec/technical_spec_v1.md`, or the repository's existing canonical path).
- Produce a decision register (`docs/decisions/decision_log.md`) listing decisions, options, rationale, status, owner, and date.
- Define state/data schemas, update order, component boundaries, and API contracts.
- Define baseline configuration and reproducibility requirements.
- Define the minimum test strategy and synthetic fixtures for later weeks.
- Add only lightweight scaffolding or schema/contract validation that helps detect specification inconsistencies early.
- Produce a Week 1 implementation/review report with evidence and blockers.

### Out of scope

Do **not** implement the complete voter dynamics, campaign strategy optimizer, full game engine, epistemic/temporal model checker, production UI, or large experiment suite in Week 1. Those depend on the contracts being approved first.

Do not choose a complicated distributed optimization algorithm solely because ADMM/proximal methods are in the reading set. Do not add neural networks or GPU dependencies to the core model without a specified need and baseline comparison.

---

## 5. Required deliverables

The coding agent must create or update the following. If the repository already has equivalent files, extend them rather than duplicating them.

| Deliverable | Required content | Acceptance condition |
|---|---|---|
| `docs/spec/technical_spec_v1.md` | Full model/game/API specification, decisions and open questions | Five-person review can trace each component and hand-off |
| `docs/decisions/decision_log.md` | Decision records and unresolved questions | No material design choice is hidden in code |
| `docs/spec/data_contracts.md` | State, voter, graph, resource, action, observation, result schemas | Fields, types, invariants, and ownership are explicit |
| `docs/spec/test_plan.md` | Unit, property, integration, reproducibility, scale, and GPU tests | Each requirement maps to one or more planned tests |
| `docs/spec/week1_handoff.md` | Approved contracts, blockers, API owners, next-week hand-off | Week 2–3 workstreams know what they may implement |
| `tests/fixtures/` | Tiny deterministic graphs/states and example configurations, if useful | Fixtures are documented and reproducible |
| `reports/week1_self_review.md` | Agent self-review, commands, results, limitations, and unresolved risks | Reviewer can independently reproduce checks |
| `Implim_Week1.md` | This plan, committed to the project repository | Available to the next coding agent/team members |

Suggested decision-record naming, if the repository uses ADRs: `docs/decisions/ADR-0001-...md`. Do not create a second competing convention if one already exists.

---

## 6. Technical Specification v1: required sections

The canonical technical specification must include all sections below. For every choice, distinguish **approved requirement**, **proposed default**, and **open decision**.

### 6.1 Product/game loop

Specify the round lifecycle in a strict order, for example:

1. Initialize game and electorate.
2. Generate the campaign player's observation.
3. Present legal actions and their known costs/effects.
4. Validate the chosen action against budget, action limits, and other constraints.
5. Apply campaign action effects.
6. Apply voter interaction/state dynamics according to the approved update schedule.
7. Apply any media/resource decay, cooldown, or duration rules (only if adopted).
8. Update round index and remaining budget.
9. Compute outcome/score and terminal conditions.
10. Generate the next observation and event log.

This is an ordering template, not a preselected final rule. The team must choose the exact order and document whether updates are synchronous or asynchronous. The order must be deterministic for a fixed configuration and seed unless randomness is explicitly part of the model.

Define initial state, legal-action generation, invalid-action behavior, termination, reset/replay, and whether the player controls one campaign or competes against another campaign.

### 6.2 Voter/agent state

Define each voter's stable identity and the mutable state required by the game. Consider (do not automatically adopt without rationale) fields such as:

- stable `voter_id`;
- preference/leaning representation (e.g. a bounded scalar, vector over parties/policies, or categorical choice);
- current support/choice state and how it differs from underlying preference;
- susceptibility/influence parameters;
- turnout/participation propensity if turnout is in scope;
- media exposure or campaign-contact state if it affects transitions;
- immutable attributes versus per-round mutable state.

For every field, specify type, valid range, units/meaning, initialization, whether it is hidden or observed, and which component may mutate it.

Do not conflate preference, belief, stated support, and final vote. If the project uses only one of these concepts, say so and justify the simplification. Define tie handling, abstention/undecided state if applicable, and whether voter state is discrete, continuous, or hybrid.

### 6.3 Electorate graph and topology

Specify:

- graph directionality (directed/undirected), weighted/unweighted, self-loop policy, and edge meaning;
- canonical representation and node-ID mapping;
- generation interface and configuration;
- validation invariants and graph statistics to log;
- how disconnected components, isolates, and extreme degrees are handled;
- whether topology is fixed for one game or can change over time.

The proposal must distinguish required topology support from optional experimental generators. Erdős–Rényi \(G(N,p)\) may be included as a configurable baseline, but do not assume it is the only or final topology. If another topology is chosen, record why.

For large sparse graphs, use a sparse edge-list/CSR-style representation in the computational path. A dense \(N \times N\) adjacency matrix has \(O(N^2)\) memory cost and should not be the default. NetworkX may be useful for generation/analysis on small graphs, but avoid making a Python-object-heavy graph representation the hot path for large GPU experiments.

Document graph metrics useful for validation and experiments: node/edge count, density, degree distribution, components/connectivity, and (when relevant) Laplacian-related metrics. Do not require expensive spectral calculations in every game initialization.

### 6.4 Voter influence/state-update model

This is a blocking design decision. Technical Specification v1 must choose and precisely define a baseline transition equation or mark it as an explicit Week 1 blocker.

The specification must define:

- variables and dimensions;
- influence from neighbors and from campaign/media actions;
- parameter ranges and units;
- whether influence is additive, normalized, bounded, probabilistic, threshold-based, or some combination;
- clipping/normalization and tie-breaking;
- update schedule (synchronous/asynchronous) and event order;
- stochastic components and seed handling;
- invariants (e.g. finite values, allowed range, valid probability simplex);
- behavior on isolates and zero total neighbor weight;
- whether campaign action changes preferences, beliefs, exposure, turnout, or another variable;
- what constitutes a vote and how votes are aggregated.

Consensus dynamics such as \(x_{t+1}=Wx_t\) or \(\dot{x}=-Lx\) may inform mathematical checks, but they must not be copied as the election model without an explicit modeling decision. If a bounded continuous state is used, define the mapping from state to party support/vote. If a categorical model is used, define transition probabilities.

For Week 1, a **small, explicit baseline** is preferable to a complicated model with unclear semantics. Record alternative dynamics as future extensions rather than leaving the baseline ambiguous.

### 6.5 Campaign resources and actions

For every resource/action, specify:

- unique action/resource identifier and display name;
- cost and currency/unit;
- target scope (voter, node subset, community, region, or global);
- prerequisites, cooldown, duration, or maximum-use constraints, if any;
- effect semantics and when it is applied;
- whether effect is deterministic or stochastic;
- what the player knows about its cost/effect;
- how effects are logged and attributed;
- behavior when budget is insufficient or a target is invalid.

Define the budget as a typed value with clear initialization, decrement, and lower bound. State whether unused budget has value and whether actions must be integer-valued. The resource engine must expose legal actions and a validation result without silently mutating game state on invalid requests.

Do not invent a long list of realistic campaign tactics in Week 1. Define the action interface and the smallest representative set that the team approves for the first playable model.

### 6.6 Partial information / observation model

Formalize the hidden state \(S_t\), observation \(O_t=\mathcal O(S_t)\), and the set of states consistent with an observation, for example:

\[
\mathcal I(O_t)=\{S : \mathcal O(S)=O_t\}.
\]

Define:

- what is hidden from the player and what is public;
- whether information is deterministic, noisy, delayed, sampled, aggregated, or action-dependent;
- what the player sees at initialization and after each action;
- whether resources reveal information as well as influence voters;
- whether observation rules are common knowledge;
- how observations are represented in the API;
- how information sets or belief states are approximated/computed, if needed;
- how hidden-state data is prevented from leaking through UI payloads, logs, error messages, or strategy APIs.

Separate **the true simulator state** from **the player observation** in both code and data contracts. Strategy code that is supposed to operate under partial information must receive an observation object, not a reference to the full hidden state. Add a test for this boundary.

### 6.7 Victory, defeat, score, and game termination

Specify the exact outcome function. It must answer:

- What is the target: most votes, a vote share threshold, seats, regional wins, or another criterion?
- Is turnout modeled? How are abstentions and ties handled?
- When is the result computed: after a fixed number of rounds, on a terminal condition, or both?
- What does “inevitable defeat” mean in a finite game, and what information is required to establish it?
- What is the score, and does it differ from the binary victory result?
- How are invalid actions, budget exhaustion, and stalled states handled?
- Are results deterministic given the same seed and action sequence?

Do not define “optimal strategy” as an unexplained synonym for “highest score.” Specify the objective, constraints, horizon, uncertainty assumptions, and comparison baseline. If this cannot be settled in Week 1, record it as a named blocker and prevent strategy implementation from guessing.

### 6.8 Temporal and epistemic properties

Define the state predicates first; only then define formulas. Candidate examples from the project roadmap include eventual victory, continued viability, and inevitable defeat. They must be translated into precise predicates over the game state before being used.

For each proposed property, record:

- informal meaning;
- exact formula and logic (e.g. LTL, CTL, or another selected formalism);
- whether it is evaluated over one trace or branching futures;
- the state predicates it uses;
- whether timing bounds are needed;
- whether it is a hard safety/liveness requirement, a diagnostic, or a research question;
- what tool/test will evaluate it.

For epistemic claims, define the possible-worlds/information relation and knowledge semantics. Do not write \(K_P(\Diamond Victory)\), or similar, as a decorative formula: state exactly what states the player considers possible and what “knows” means under that relation. Distinguish “true,” “possible,” “guaranteed,” and “known.”

Week 1 defines the semantic contract and initial properties; a full model checker is not required this week.

### 6.9 Strategy and optimization interface

Define a strategy as a policy from permitted information to a legal action, e.g. observation/history to action. It must not read hidden state unless the experiment explicitly labels the strategy as an omniscient oracle baseline.

Define:

- strategy input/output types;
- deterministic versus stochastic policies;
- legal-action and budget enforcement;
- objective/fitness metric;
- evaluation seeds and scenario sets;
- baseline strategy names and minimum comparison protocol;
- logging of chosen action, observation, cost, resulting score, and termination.

Baseline strategies and optimization are Week 2–5 workstreams in the roadmap. Week 1 defines interfaces and evaluation contracts only. GPU batching for many independent episodes can be explored later; do not prematurely design an optimizer around unapproved dynamics.

### 6.10 Reproducibility, performance, and GPU policy

The specification must require:

- a central configuration schema with validation;
- one explicit seed or seed hierarchy for graph generation, initial voter states, transition randomness, and strategy randomness;
- a record of config, seed, git commit, Python/package versions, device, and run parameters;
- no unseeded global randomness in core simulation code;
- a CPU reference path for correctness;
- an optional PyTorch/CUDA path only where tensor workloads justify it;
- equivalence/tolerance criteria between CPU and GPU implementations;
- profiling of graph construction, one simulation step, and a batch of episodes on representative sizes;
- memory-aware batching rather than loading all experiment states onto a GPU at once.

**Multi-GPU caution:** two A5000s do not imply transparent 48 GB per-process VRAM. Separate devices have separate memory. If later experiments use both, define explicit sharding or independent-run scheduling and measure communication overhead. Do not use multi-GPU for a single small simulation without evidence that it helps.

### 6.11 Logging, errors, and auditability

Define structured events for at least: game initialization, action proposed, action accepted/rejected, resource expenditure, voter-state update summary, observation generated, round completed, and game termination. Avoid dumping all private voter state into user-facing logs.

Define stable error categories for invalid configuration, invalid action, insufficient budget, malformed observation, numerical instability, and internal failure. Invalid requests must fail clearly and must not leave partially mutated state.

---

## 7. API and module contracts

Use the repository's existing architecture if present. Otherwise propose a small package boundary such as:

```text
src/dumbocracy/
  config/          # typed configuration and validation
  graph/           # graph construction, representation, validation
  mas/             # voter state and transition interface
  campaign/        # resources, costs, legal actions, action effects
  information/     # hidden state -> player observation
  engine/          # round lifecycle, state transitions, outcomes
  strategies/      # strategy interface and baseline placeholders
  logic/           # property definitions/semantics contracts
  experiments/     # reproducible batch runs (later weeks)
tests/
  unit/
  integration/
  properties/
  fixtures/
docs/
  spec/
  decisions/
reports/
```

Do not create empty modules merely to match this tree if the project already has a useful structure. The goal is clear ownership and contracts, not directory proliferation.

At minimum, define typed conceptual interfaces for:

- `GameConfig` — validated game parameters and seed configuration.
- `GraphSpec` / graph object — topology parameters, node mapping, sparse edges, validation.
- `VoterState` / state container — explicit field schema and invariants.
- `CampaignAction` — action ID, target, parameters.
- `ResourceLedger` — initial budget, remaining budget, legal-cost checks, atomic spend.
- `GameState` — authoritative hidden state and round index.
- `Observation` — player-visible state only; no accidental pointer/reference to hidden state.
- `GameEngine` — initialization, legal actions, action application, one round/step, terminal status, result.
- `Strategy` — observation/history and legal-action information to proposed action.
- `GameResult` — terminal reason, vote totals/score, budget use, seed/config metadata.

These are conceptual contracts, not mandatory exact class names. The specification must document function signatures, input/output types, mutation rules, exceptions, and deterministic behavior. Prefer explicit return values over hidden side effects.

### Atomic state transitions

A rejected action must not partially spend budget or partially change voters. Specify whether the engine uses validate-then-commit or immutable/new-state transitions. This is a high-value invariant and should be tested once the engine is implemented.

---

## 8. Repository reconnaissance and setup tasks

The coding agent starts by inspecting, not replacing, the current environment.

1. Record `pwd`, repository root, `git status`, current branch, recent commit, Python version, available CUDA devices, and package manager.
2. Inspect existing `README`, dependency files, package configuration, tests, CI, source layout, and coding conventions.
3. Run existing tests before edits. Record baseline pass/fail status and do not “fix” unrelated failures without documenting them.
4. Inspect the server using safe, read-only commands such as `nvidia-smi`, `python --version`, `python -c ...`, and `git status`.
5. Do not print secrets, tokens, SSH credentials, environment variables containing credentials, or private keys in reports.
6. Do not upgrade system packages, alter CUDA drivers, install system-wide packages, delete files, rewrite history, or push changes unless the human owner explicitly authorizes it.
7. Use a project-local environment and lock/pin dependencies according to the existing repository convention.
8. Record hardware and environment information without assuming that CUDA/PyTorch compatibility is correct until verified.

If no repository exists yet, create the minimal agreed structure and document the setup. If the repository already has structure, preserve it and explain any proposed change.

---

## 9. Week 1 implementation workflow for the coding agent

The agent must follow these phases in order and stop at decision gates.

### Phase A — Reconnaissance

- Inventory source documents and repository files.
- Summarize existing implementation, if any.
- Run existing tests and capture baseline results.
- Identify conflicts, missing requirements, and duplicated sources of truth.
- Create no model-specific implementation before understanding existing code.

**Gate A:** report findings and blockers. Do not silently overwrite existing code or make product decisions.

### Phase B — Requirements extraction

- Convert the proposal and roadmap into a requirement list with stable IDs, e.g. `REQ-GAME-001`.
- Label each requirement as `must`, `should`, or `future/optional`.
- Trace each requirement to its source (proposal, roadmap/dependency graph, or approved team decision).
- Mark inferred design choices as `PROPOSAL`, not as source-derived facts.
- Add a verification method and owner for each requirement.

### Phase C — Specification and decision log

- Draft Technical Specification v1 using Section 6.
- Add a decision record for every choice that changes state semantics, action semantics, information boundaries, or API compatibility.
- Highlight decisions needed from the team with a short option comparison and recommendation.
- Do not claim all five members have approved a decision on their behalf.
- Identify which decisions block each Week 2–3 workstream.

**Gate C:** all five members review the decision table. Any unresolved blocking decision must have an owner and deadline before dependent implementation begins.

### Phase D — Contracts and lightweight validation

- Define typed schemas/contracts and examples.
- Add schema/config validation only if it is small, useful, and consistent with the existing stack.
- Add tiny deterministic fixtures (e.g. a 3–5 node graph and hand-written voter states) to make the intended semantics concrete.
- Add contract tests for ranges, dimensions, invalid configurations, seed behavior, and observation/hidden-state separation where possible.
- Do not build a full simulator as a side effect of creating contracts.

### Phase E — Consistency and dependency review

- Check the spec against every Week 2–3 hand-off shown in `Dependency_graph.png`:
  - Baseline Strategies: objective, legal action interface, scoring, reproducible evaluation.
  - MAS Model: voter schema, graph representation, update contract.
  - Resource Engine: action schema, cost/budget semantics, effect timing.
  - Information Model: hidden state, observation schema, visibility rules.
  - UI Prototype: game loop, observation payload, legal actions, result/status schema.
  - Temporal Model: state predicates, transition ordering, terminal semantics.
- Check that the Game Engine can integrate the MAS API, campaign API, and observation API without depending on a UI framework.
- Confirm UI work can use a mock/skeleton API before the engine is complete, as allowed by the dependency graph.
- Identify any circular dependencies or fields owned by multiple components.

### Phase F — Self-review and hand-off

- Execute all relevant checks.
- Read the produced documents as a new developer would.
- Verify every requirement is traceable and all unresolved decisions are explicit.
- Produce `reports/week1_self_review.md` and `docs/spec/week1_handoff.md`.
- Summarize exactly what changed, commands run, test results, known limitations, and the human decisions still needed.
- Stop before beginning Week 2 implementation.

---

## 10. Testing and validation plan

Week 1 primarily validates the specification and its contracts. It does not prove that future dynamics are correct.

### 10.1 Documentation/specification checks

- All mandatory sections in Section 6 exist.
- Every defined variable has a type, meaning, units/range where applicable, and owner.
- Every state transition has an explicit order.
- All random choices have a seed source.
- Every player-visible field is explicitly listed.
- Every action has cost, preconditions, effect timing, and invalid-action behavior.
- Victory/defeat/score are defined without contradictory conditions.
- Temporal/epistemic formulas reference defined predicates and semantics.
- Each Week 2–3 component has an API contract and hand-off.
- All open questions have an owner and deadline.
- No unapproved proposal is written as an accepted requirement.

If the repository supports Markdown linting or documentation checks, run them. Otherwise use a small script/checklist rather than adding a large toolchain.

### 10.2 Contract/unit tests (only for scaffolding actually added)

- Valid and invalid configurations are accepted/rejected as specified.
- Graph fixtures have correct node/edge counts, no forbidden self-loops, and valid node mappings.
- Voter fields obey the documented domains and dimensions.
- Resource spending cannot make the budget negative.
- An invalid action does not mutate the budget or game state.
- Observation construction excludes fields marked hidden.
- Repeating initialization with the same seed yields the same initial fixture/state where determinism is promised.
- Tests do not depend on a GPU unless explicitly marked as GPU-only.

Do not write tests against invented model behavior that the team has not approved. If an interface is not implemented yet, record the test as planned rather than fabricating a passing test.

### 10.3 Numerical and graph-model sanity checks to plan for later weeks

The specification/test plan should define future tests for:

- finite values and valid ranges after every update;
- normalized probability vectors where applicable;
- isolated nodes and zero-neighbor cases;
- disconnected graphs and degenerate topologies;
- action order and round-boundary behavior;
- reproducibility across repeated runs;
- CPU/GPU agreement within documented tolerances;
- graph size/density and memory stress cases;
- no hidden-state leakage into strategy/UI inputs;
- hand-computable toy examples with expected outputs;
- terminal-state and score consistency.

Consensus reference tests may be useful to validate a generic graph/dynamics utility, if such a utility is selected. They must not be used to force the electorate to consensus.

### 10.4 GPU and scale validation

Week 1 should only verify that the environment can see the hardware if the server is available:

- `nvidia-smi`
- Python/PyTorch version and CUDA availability
- per-device name and available memory
- a minimal tensor allocation/operation on each GPU, if PyTorch/CUDA is already installed and this is safe

Do not install a new CUDA stack just to make this check pass. Report the actual result. Full performance benchmarks belong to the implementation weeks, after workload sizes and update equations are defined.

### 10.5 Required evidence

The self-review report must include:

- environment summary (no secrets);
- repository commit/branch and changed-file list;
- exact commands executed;
- pass/fail/skip counts;
- test output or concise log references;
- any failures that predate Week 1;
- a list of untested requirements and why;
- known limitations and open decisions;
- confirmation that the working tree contains no unrelated or accidental changes.

---

## 11. Five-member human review protocol

All five members validate the coding-agent output. Reviewers should review independently before merging, not merely accept the agent's summary.

Suggested review lenses, adaptable to the team's actual roles:

1. **MAS/graph reviewer:** voter schema, graph semantics, update order, edge cases, scalability.
2. **Campaign/resource reviewer:** actions, costs, budget accounting, constraints, outcome/scoring.
3. **Information/game-engine reviewer:** hidden-state boundary, observations, round lifecycle, API integration.
4. **Logic/verification reviewer:** predicates, temporal/epistemic semantics, truth/possibility/knowledge distinctions.
5. **Strategy/experiment/UI reviewer:** policy interface, evaluation metrics, reproducibility, observation/result payloads.

Each reviewer must record:
- `approve`, `approve with non-blocking comments`, or `request changes`;
- the sections/files reviewed;
- concrete issues and severity;
- whether their workstream can begin without guessing.

Do not assign a reviewer a role they have not accepted. The role mapping above is a review checklist, not a claim about current ownership. The project proposal's tentative roles can be used as a starting point, subject to team confirmation.

**Merge gate:** no unresolved blocking issue may remain in a contract needed by Week 2–3. Non-blocking issues must be tracked with an owner and due date.

---

## 12. Week 1 exit criteria

Week 1 is complete only when all applicable items are checked:

- [ ] Repository/environment reconnaissance recorded.
- [ ] Existing tests run and baseline status captured.
- [ ] Technical Specification v1 exists and covers every Section 6 topic.
- [ ] Approved decisions and open questions are clearly distinguished.
- [ ] Decision log has owners and deadlines for unresolved blockers.
- [ ] State, graph, action, resource, observation, engine, strategy, and result contracts are documented.
- [ ] Round ordering, invalid-action behavior, terminal conditions, and determinism policy are explicit.
- [ ] Partial-information boundary is defined and leakage risks are addressed.
- [ ] Temporal/epistemic properties have defined predicates and semantics or are marked as blockers.
- [ ] Week 2–3 APIs and dependencies are mapped to the dependency graph.
- [ ] Test plan maps requirements to validation methods.
- [ ] Any Week 1 scaffolding passes its tests; no unimplemented behavior is falsely reported as tested.
- [ ] Self-review report and Week 2 hand-off are complete.
- [ ] All five team members have recorded their review outcomes.
- [ ] No blocking design question is left for the coding agent to guess.
- [ ] Changes are committed on a review branch or otherwise preserved according to the repository's workflow; no unapproved push/merge has occurred.

---

## 13. Explicit questions that must not be guessed

The agent must surface these for team decision if no approved answer exists:

1. What does a voter's state represent: preference, belief, support, vote intention, turnout, or a defined subset?
2. Is there one campaign player, multiple competing campaigns, or a chosen initial simplification?
3. What is the initial electorate topology, and which topology generators are required for the first version?
4. What is the baseline influence/update equation and its parameterization?
5. Which campaign actions/resources exist in the first playable version?
6. Do campaign actions affect preferences, beliefs, exposure, turnout, or a combination?
7. What is hidden from the player, and what observation mechanism reveals information?
8. What are the budget rules, action constraints, round horizon, and invalid-action semantics?
9. What is the exact victory/defeat/score definition?
10. What is the strategy optimization objective and evaluation protocol?
11. Which temporal logic and epistemic semantics are required for the first verification milestone?
12. Which Python/package/tooling versions are already established on the SSH server and in the repository?
13. What performance target and network sizes are required for the first demonstrable version?

The agent may recommend defaults with trade-offs, but it must label them as proposals and wait for human approval before treating them as frozen requirements.

---

## 14. Final agent response format

At the end of its run, the coding agent must report using this structure:

1. **Status:** complete / blocked / partial.
2. **Milestone:** M0 Technical Specification v1.
3. **Files created/changed:** paths and purpose.
4. **Requirements covered:** requirement IDs and traceability.
5. **Commands run:** exact commands.
6. **Tests:** pass/fail/skip counts and any failure details.
7. **Environment:** Python, CUDA/PyTorch availability, GPUs visible (if checked).
8. **Decisions needing humans:** concise list with options and recommendation.
9. **Risks/limitations:** explicit, not hidden.
10. **Self-review checklist:** completed items and unchecked items.
11. **Five-person review:** pending; do not claim sign-off on anyone's behalf.
12. **Week 2 readiness:** ready / blocked, with reasons.

**Stop condition:** once M0 artifacts and evidence are ready for human review, stop. Do not begin implementing Week 2 components until the team has reviewed and approved the contracts.
