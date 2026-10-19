# Week 1 deterministic fixtures

These are data-only, hand-auditable fixtures for future contract tests. They
do not implement voter dynamics.

- `tiny_path_graph.json`: undirected path `v0—v1—v2`; useful for isolate and
  neighbor-normalization tests when combined with a single-node graph.
- `tiny_initial_state.json`: matching IDs, support values, and budget under the
  proposed M0 contract.

Both files carry seed `20261019` as provenance only; their values are explicit,
so consumers must not rely on a random-number generator to recreate them.
