# Week 2 Hand-off for the Game Engine

**Status:** original Week 2 hand-off; the engine has since been implemented in
Week 3. See [week3_handoff.md](week3_handoff.md).

## Stable inputs available to Week 3

- `ElectorateGraph` has ordered external node IDs and immutable canonical sparse
  edge pairs. `graph.summary()` reports metrics without dense adjacency storage.
- `VoterState` is ordered to match graph node IDs. `advance_support(graph,
  voters, alpha, contributions)` returns a new voter tuple and implements the
  approved synchronous equation.
- `ActionContext` contains an exact-Decimal ledger and the per-round broadcast
  guard. `validate_action` is pure. `commit_action` returns a new context plus
  one current-round contribution vector in graph order.
- `no_op` is valid/zero cost/no effect. The Game Engine advances rounds, resets
  `broadcast_used`, and clears contribution vectors after dynamics.

## Integration constraints

The engine owns `GameState`, event history, terminal outcome, and Observation
projection. It must not send `VoterState`, supports, susceptibilities, seed
config, or `ActionContext` to public/strategy interfaces. The Week 2 package
has no engine/UI/strategy imports, so it is safe to orchestrate from above.

## Deferred items

Observation leakage tests, complete round/terminal lifecycle tests, result
aggregation, bounded-LTL/epistemic evaluation, optimization, and GPU parity
remain later work.
