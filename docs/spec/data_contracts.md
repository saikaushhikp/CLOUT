# Dumbocracy Data Contracts (M0)

**Status:** approved by DEC-002–DEC-009. These names describe the Week 2
interfaces; they are not implemented classes.

## Ownership and schemas

| Object | Fields (type; invariant) | Owner / mutation |
| --- | --- | --- |
| `GameConfig` | `seed: int`; `node_count: int >= 1`; `horizon_rounds: int >= 1`; `initial_budget: Decimal >= 0`; `support_threshold: float = 0.5`; `dynamics: DynamicsConfig`; `actions: ActionConfig`; `graph: GraphSpec` | Config layer; immutable after validation. |
| `DynamicsConfig` | `alpha: float in [0,1]`; `epsilon_mode: Literal["none"] = "none"` | Config layer; immutable. No stochastic transition noise in v1. |
| `ActionConfig` | `targeted_cost: Decimal = 1.00`; `broadcast_cost: Decimal = 5.00`; `delta_target: float`; `delta_broadcast: float`; each delta finite | Config layer; immutable. Deltas are documented support-units/round. |
| `SeedConfig` | `root_seed: int`; named derived seeds for `graph`, `initial_state`, `transition`, `strategy` | Config layer; derivation is deterministic. |
| `GraphSpec` | `kind: "edge_list" | "erdos_renyi"`; `node_count`; `edge_probability: float in [0,1]` when ER; `directed: bool = false`; `allow_self_loops: bool = false` | Graph layer; immutable input. |
| `ElectorateGraph` | `node_ids: tuple[str,...]` unique; `src: int[]`; `dst: int[]`; optional `weight: float[] > 0` aligned to edges | Graph layer builds/validates; read-only thereafter. IDs map once to contiguous indices. |
| `VoterState` | `voter_id: str`; `support: float in [0,1]`; `susceptibility: float in [0,1]`; `campaign_exposure: float >= 0` | MAS layer mutates `support` and exposure only during committed transitions; ID/susceptibility immutable. |
| `GameState` | `round_index: int in [0,horizon]`; graph; voters; `ledger`; active effects; `event_log`; `terminal: bool` | Engine is authoritative owner. Never pass this object to partial-information strategy/UI code. |
| `CampaignAction` | `action_id: "targeted_contact" | "broadcast" | "no_op"`; `target_ids: tuple[str,...]`; `parameters: mapping`; `client_action_id: str` | Campaign layer validates; no state mutation in action object. `no_op` has an empty target tuple. |
| `ActionValidation` | `accepted: bool`; `error_code: str?`; `cost: Decimal?`; normalized action | Campaign layer returns prior to commit. |
| `ResourceLedger` | `initial_budget: Decimal >= 0`; `remaining_budget: Decimal in [0,initial_budget]`; `spent: Decimal >= 0` | Campaign layer atomically debits only after validation succeeds. |
| `Observation` | `round_index`; `remaining_budget`; `aggregate_support: float`; `graph_summary`; `legal_actions`; `terminal_status`; `public_events` | Information layer produces detached values only. No voter support array, susceptibility, hidden seed, or `GameState` reference. |
| `GameResult` | `terminal_reason`; `vote_totals`; `vote_share: float in [0,1]`; `victory: bool`; `score: float`; `budget_used`; reproducibility metadata | Engine produces at termination; immutable. |

## Graph invariants

The baseline graph is a fixed, sparse, undirected graph. Edges use
canonical `(min_index, max_index)` ordering, contain no self-loops or duplicate
pairs, and reference valid node indices. Isolates and disconnected components
are legal and logged. Compute node/edge count, density, degree summary, and
component count at initialization; spectral metrics are opt-in diagnostics.

## Action and error contract

`validate_action(observation_or_state, action) -> ActionValidation` is pure.
`commit_action(state, validation) -> state` may run only for an accepted
validation. Rejections return one stable code—`INVALID_CONFIGURATION`,
`INVALID_ACTION`, `INVALID_TARGET`, `INSUFFICIENT_BUDGET`, `MALFORMED_OBSERVATION`,
`NUMERICAL_INSTABILITY`, or `INTERNAL_FAILURE`—and leave state and ledger
unchanged. Currency uses `Decimal`, never binary float.

## Conceptual APIs

```text
GameEngine.initialize(config) -> GameState
GameEngine.observe(state) -> Observation
GameEngine.legal_actions(observation) -> tuple[ActionDescriptor, ...]
GameEngine.step(state, action) -> StepResult  # validate then commit or reject
GameEngine.is_terminal(state) -> bool
GameEngine.result(state) -> GameResult

Strategy.choose(observation, history, legal_actions, rng) -> CampaignAction
GraphFactory.build(spec, seed) -> ElectorateGraph
Dynamics.advance(voters, graph, active_effects, rng) -> VoterState[]
```

`step` is deterministic for equal validated config, seed hierarchy, initial
state, and action sequence. It returns a new state or uses an explicitly
documented validate-then-commit boundary; it never partially mutates on error.

## Week 3 implementation clarifications

- `GameState` snapshots also retain immutable initial voters, a cached graph
  summary, the boundary contribution vector (zero outside a step transaction),
  trace records, and safe/public plus internal event tuples. The engine uses a
  provisional immutable snapshot for resource debit/effect application; it
  publishes only after the transition succeeds.
- `Observation.legal_actions` includes eligible public node IDs for the
  targeted action so clients can form valid targets. IDs have no associated
  support, susceptibility, or graph-edge data.
- The JSON-safe `StepResponse` includes `result` only for an accepted terminal
  step. Rejected responses omit result metadata; the observation itself never
  contains seeds or authoritative state.
