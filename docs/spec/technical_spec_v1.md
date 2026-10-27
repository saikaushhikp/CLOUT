# Dumbocracy Technical Specification v1

**Milestone:** M0 Technical Specification v1, approved 2026-10-19.

**Authority:** Project assumptions, Week 1 plan, and approved decisions in [decision_log.md](../decisions/decision_log.md). The baseline choices below are frozen requirements unless a later decision record supersedes them.

## 1. Requirements

| ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| REQ-GAME-001 | Loop, actions, outcomes, errors are explicit. | Week 1 §§6.1, 6.7 | Engine integration test |
| REQ-MAS-001 | Voter state, graph, baseline transition are defined. | Week 1 §§6.2–6.4 | Toy state test |
| REQ-CAMP-001 | Budget/action processing is atomic. | Week 1 §§6.5, 7 | Ledger tests |
| REQ-INFO-001 | Observation is separate from hidden state. | Week 1 §6.6 | Leakage test |
| REQ-LOGIC-001 | Logic predicates and semantics are defined. | Week 1 §6.8 | Trace checks |
| REQ-STRAT-001 | Policy uses permitted data, reproducibly. | Week 1 §6.9 | Contract test |
| REQ-REPRO-001 | Seed, config, revision, environment recorded. | Assumptions; §6.10 | Metadata test |
| REQ-PERF-001 | CPU reference; GPU only after profiling. | Assumptions; §6.10 | Benchmark |

Implementation/review owners remain TBD and must be assigned by the team.

## 2. Product loop (Approved: DEC-002, DEC-006, DEC-008)

The initial model has one player campaign and a fixed horizon. Order: initialize; generate detached observation; enumerate legal actions; propose an action; validate without mutation; atomically debit and apply accepted effect; apply synchronous voter dynamics; expire effects; increment round; determine terminal result; append safe event summary and observe again. Reset uses config/seed and replay uses accepted actions. Rejection is logged, does not advance time, and changes nothing. Terminal is `round_index == horizon_rounds`; budget exhaustion does not terminate. Public `no_op` costs `0.00`, affects no voters, and advances the round; it is an engine action, not a campaign resource.

## 3. Voter state and topology (Approved: DEC-003, DEC-004)

`support` is a finite continuous propensity for the player in [0,1], not latent belief or turnout. At terminal time, support `> 0.5` is a player vote; exact 0.5 is a non-player vote. Turnout, abstention, multiparty choice, regions, and changing topology are extensions. Full types, ownership, and visibility are in [data_contracts.md](data_contracts.md).

Edges mean potential peer influence. The graph is fixed, undirected, unweighted, loop-free, sparse, and represented as contiguous indexed edge-list or CSR data. Required generators: explicit edge list and `erdos_renyi(N,p)`. Isolates have zero peer influence; disconnected components are valid. Log nodes, edges, density, degree summary, and components.

## 4. Baseline dynamics (Approved: DEC-005)

For voter `i`, support `x_i`, susceptibility `s_i`, neighbor set `N(i)`, and campaign effect `c_i`, use this synchronous update:

`m_i = mean(x_j for j in N(i))`, or `0` for an isolate.

`x_i' = clip_[0,1](x_i + s_i * (alpha * (m_i - x_i) + c_i + epsilon_i))`.

`alpha` is in [0,1], `epsilon_i = 0`, and effects are support units per round. Any adopted noise requires distribution and a transition sub-seed. Outputs use the pre-transition vector. Non-finite outputs fail atomically with `NUMERICAL_INSTABILITY`. Disagreement is valid; this does not require consensus.

## 5. Resources and action semantics (Approved: DEC-006)

| Action | Cost and target | Effect | Constraint |
| --- | --- | --- | --- |
| `targeted_contact` | `1.00` per unique voter ID | Add `delta_target` to target `c_i` this round | Existing IDs; affordable |
| `broadcast` | `5.00`, all voters | Add `delta_broadcast` to each `c_i` this round | Once/round; affordable |
| `no_op` | `0.00`, no target | No campaign effect; advances the round | Always legal before terminal state |

Effects are deterministic/public and expire after dynamics. Validation is pure; only accepted actions commit atomically and log attribution. Unknown, malformed, limit-violating, or unaffordable actions reject without debit. Currency uses `Decimal`; unused budget has no score value.

## 6. Partial information (Approved: DEC-007)

Hidden `S_t` holds state, seeds, voter support/susceptibility, and active effects. `O_t = O(S_t)` has round, remaining budget, aggregate mean support, graph summary, legal actions, terminal status, and public event summaries. It has no voter arrays, susceptibility, seed, diagnostics, or state reference. `I(O_t) = {S | O(S)=O_t}` uses matching public context. Observation rules are common knowledge. UI and strategies receive detached observations only; errors and public logs cannot disclose private data. No revealing resource exists in v1.

## 7. Outcome and strategy (Approved: DEC-008)

At horizon, votes count support values `> 0.5`; share is votes/node count. Victory is share `> 0.5`; equality is defeat. Score is vote share. A strategy maximizes mean terminal score over a declared finite scenario/seed set, subject to legal actions/budget. Omniscient policies are labeled oracle baselines only. `Strategy.choose(observation, history, legal_actions, rng) -> CampaignAction`. Logs contain safe observation summary, action, cost, outcome, config, and seed. There is no early “inevitable defeat” result in v1; only the horizon result is authoritative.

## 8. Temporal and epistemic semantics (Approved: DEC-009)

Use bounded LTL diagnostics over completed fixed-horizon traces. Predicates are `Terminal`, `Victory`, `BudgetNonnegative`, `StateValid` (finite/ranged support), and `ActionAtomic` (rejection preserves ledger/state fingerprint). Diagnostics: `G(BudgetNonnegative and StateValid)`, `F_<=H Terminal`, and `Terminal -> (Victory or not Victory)`. `F_<=H Victory` is trace occurrence, not a cross-future guarantee.

`K_player phi` holds when phi holds in every state in `I(O_t)` compatible with public config/history; `Possible phi` holds in one. Known inevitable victory is not evaluated in v1 because it needs action/noise quantification. Future tooling must state whether it evaluates traces, samples, or exhaustive branches.

## 9. Reproducibility, performance, and audit (Approved constraints)

Derive graph, initialization, transition, and strategy RNGs from a root seed; do not use unseeded global randomness. Record config, root/derived seeds, git revision (or `uncommitted`), Python/package versions, device/backend, graph metrics, and parameters. CPU is correctness reference. CUDA is optional after profiling and must declare CPU/GPU tolerance. The A5000s have separate VRAM; no distributed execution is implied.

Events: game initialized; action proposed/accepted/rejected; resource spent; voter-update summary; observation generated; round completed; game terminated. Public events are summaries. Stable errors are in the data contract.

## 10. Module boundaries and approval gate

`config` validates; `graph` builds/metrics; `mas` transitions; `campaign` validates actions/ledger/effects; `information` projects; `engine` orders and results; `strategies` sees observations; `logic` evaluates predicates; `experiments` records runs. Engine has no UI dependency. These approved interfaces are ready for Week 2 implementation.
