# Week 4 Strategy Experiment Record Template

This file defines the output contract; all example values below are **illustrative, not measured**. The coding-agent paired example run completed 18/18 episodes with no failures, skips, or rejected proposals on 2026-10-23; its measured output was emitted to the terminal and is summarized in `docs/spec/week4_handoff.md`. The focused experiment suite passed (11 tests); reviewer rerun is recommended.

## Per-episode JSONL

`examples/week4_compare_strategies.py` emits one JSON object per episode. The
versioned record has: `schema_version`; `scenario_id`; `paired_run_id`; strategy name/version; root and graph/initial-state/transition/strategy seeds; graph kind, N/E/density/degrees/components and edge probability; horizon, budget, alpha, action costs/effects and support threshold; runtime Python, NumPy, optional PyTorch/CUDA, backend/device and code revision; action proposals with round, public action ID, publicly eligible target IDs (only when all proposed targets are in the current public eligible list), accepted flag and safe error code; accepted/rejected/failed/skipped counts; timing definitions; terminal result; safe trace summary; and M0 logic diagnostic summaries.

The record does not serialize voter-level support/susceptibility, hidden graph edges, internal event attributes, or device tensors. On strategy exceptions, invalid proposals, or setup failures, one explicit `status: failed` record is emitted with a safe stage/type or stable error code. Failed episodes are not silently omitted. The policy runner aborts on the first rejected proposal; there is no retry loop. `skipped_count` is currently zero; future skip behavior must be explicit and separately tested.

## Aggregate definitions

Metrics are grouped by strategy name and version:

- `sample_count`: all records, including failures/skips.
- `completed_count`, `failed_episode_count`, `skipped_episode_count`: disjoint status counts.
- `mean_terminal_score`: arithmetic mean of completed episodes only.
- `terminal_score_population_stddev`: population standard deviation (denominator `n`, not `n-1`) over completed scores; null if none complete.
- `victory_rate_completed`: wins / completed episodes.
- `mean_budget_used_completed`: arithmetic mean over completed episodes.
- `rejected_proposal_rate`: rejected proposals / all recorded proposals, including proposals from episodes that later fail; null if no proposals.

No significance or broad generalization claim follows from the small paired seed set. Each strategy gets the same declared root seed/scenario pairs and an independent fresh engine/policy/history/RNG. The engine seed hierarchy makes the initial graph and voter values identical for a matched pair; after policy actions diverge, trajectories are expected to diverge.

Timing values are wall-clock attribution fields with CUDA synchronization at initialization and step boundaries when applicable. They include normal runtime startup effects and are not a benchmark claim; exact record equality across repeated runs excludes timing fields.

## Illustrative line (not measured)

```json
{"schema_version":"1.0","scenario_id":"tiny","paired_run_id":"tiny:17","strategy":"no_op","strategy_version":"1","root_seed":17,"status":"completed","accepted_count":2,"rejected_count":0,"failed_count":0,"skipped_count":0,"terminal_result":{"score":0.5,"victory":false},"logic_diagnostics":{}}
```

This shortened line illustrates shape only; actual records include full provenance, graph/game metadata, action sequence, timing, and diagnostics.
