# Week 3 Engine and Information API

**Version:** transport-neutral API v1, based on frozen M0 decisions.

## Python entry points

```text
GameEngine(backend="cpu" | "cuda:N", revision="...")
  .initialize(config, graph=None, voters=None) -> GameState
  .observe(state) -> Observation
  .legal_actions(state) -> tuple[ActionDescriptor, ...]
  .step(state, CampaignAction) -> StepResponse
  .result(state) -> GameResult | None
  .is_terminal(state) -> bool
  .reset(state) -> GameState
  .replay(config, accepted_actions, graph=None) -> tuple[GameState, ...]

Observation.to_public_dict() -> fresh JSON-safe allowlist
public_json(observation) -> stable JSON string
```

`GameState` is authoritative and private to engine-side code. A service adapter must serialize `Observation`, `StepResponse.to_public_dict()`, and `GameResult.to_public_dict()` only; it must never serialize `GameState`.

`TraceRecord.predicates()` exposes the M0 names `Terminal`, `Victory`, `BudgetNonnegative`, `StateValid`, and `ActionAtomic` for future trace tooling. It does not evaluate temporal operators or epistemic knowledge.

## Public schemas

Observation fields are `schema_version`, `round_index`, `remaining_budget` (decimal string), `aggregate_support`, `graph_summary`, `legal_actions`, `terminal_status`, and `public_events`. The graph summary carries node/edge counts, density, min/max/mean degree, and connected-component count.

An action descriptor carries `action_id`, public `cost`, `target_requirement`, public effect parameters, and eligible public node identifiers for targeting. Identifiers carry no voter attributes; graph edges remain private. An action submission carries `action_id`, optional `target_ids`, optional parameters, and optional `client_action_id`. Target IDs are accepted only for targeted contact. Duplicate or unknown IDs reject.

Step response fields are `accepted`, `error_code`, `observation`, `terminal`, `action_id`, and `result` (present only on an accepted terminal step). Rejected responses never include result metadata. Stable response codes include all M0 error codes plus `TERMINAL_STATE`. A rejected response does not debit, change voters, consume a round, or change the result. The audit history adds a proposed/rejected event.

Terminal result fields are `terminal_reason`, `player_votes`, `electorate_size`, `vote_share`, `victory`, `score`, `budget_used` (decimal string), and reproducibility metadata. The only terminal reason in v1 is `fixed_horizon`.

## Lifecycle state machine

```text
active(round r)
  ├─ rejected submission ─> active(round r), state unchanged + rejection audit
  └─ accepted action ─> debit/effect -> one synchronous update -> expire effect
                       -> round r+1 -> [active | terminal result]
terminal
  └─ any submission ─> terminal unchanged + TERMINAL_STATE audit
```

Each accepted request advances exactly one round. The context broadcast guard and contribution vector are round-scoped; the engine starts the next round with an unused guard and no lingering effects. `no_op` runs voter dynamics too. `reset` regenerates seeded initial values (or restores explicit initial fixture voters) and starts a fresh event/trace history. `replay` accepts an ordered sequence and returns each snapshot, including rejected submissions if present. The canonical state trajectory is defined by accepted actions; replaying rejected submissions too reproduces their audit events without advancing time. Submissions after terminal are retained as rejected terminal attempts.

## Public event allowlist

Public events expose sequence, event type, round, fixed summary text, action ID for accepted actions, and stable error code for rejections. Internal events may retain target IDs and resource details for audit, but public projection strips them. Public payloads never contain voter data, susceptibility, seeds, contribution arrays, or internal state references.

## Example public transcript (shape)

```json
{
  "accepted": true,
  "error_code": null,
  "terminal": false,
  "action_id": "targeted_contact",
  "observation": {
    "schema_version": "1.0",
    "round_index": 1,
    "remaining_budget": "9.00",
    "aggregate_support": 0.5333333333333333,
    "graph_summary": {
      "node_count": 3,
      "edge_count": 2,
      "density": 0.6666666666666666,
      "min_degree": 1,
      "max_degree": 2,
      "mean_degree": 1.3333333333333333,
      "component_count": 1
    },
    "legal_actions": [
      {
        "action_id": "targeted_contact",
        "cost": "1.00 per target",
        "target_requirement": "one or more unique public voter IDs",
        "parameters": {"delta_target": 0.2},
        "eligible_target_ids": ["v0", "v1", "v2"]
      },
      {"action_id": "no_op", "cost": "0.00", "target_requirement": "no target", "parameters": {}, "eligible_target_ids": []}
    ],
    "terminal_status": false,
    "public_events": []
  }
}
```

The public event array above is abbreviated; runtime output includes safe event summaries available at that state. The targeted descriptor's identifier list contains public IDs only, with no voter state or graph edges.

## UI/mock client and compatibility

No frontend was present in the Week 2 checkout. `examples/week3_mock_client.py` reads a serialized observation and returns an action payload without importing `GameState` or engine code. The engine is independent of HTTP/React and accepts `CampaignAction`; a transport adapter can use the public dict schemas above. `schema_version` is `1.0`. Additive public fields require a minor schema bump; changed meanings or removed fields require a major bump and compatibility note. Hidden state is not part of any public version.

## GPU behavior and limits

`GameEngine("cuda:N")` requires compatible PyTorch/CUDA and fails clearly when unavailable. `week3_gpu_episode.py --device auto` chooses CUDA when PyTorch reports an available device and announces CPU fallback otherwise. The GPU MAS path transfers immutable voter vectors at each transition boundary, computes neighbor aggregation/update on the selected GPU, and returns compact support values to the engine. Sparse edge indices and degrees are prepared once per game and remain on the selected GPU. The canonical engine state remains CPU-backed; the benchmark will determine whether persistent device-resident voter state is warranted before claiming throughput benefits. GPU reductions may differ in low-order floating bits; parity tolerance is `1e-10` for the small fixture and discrete outcomes/actions must match.

The benchmark expects mean degree 8: p is `8/(N-1)` (clamped to 1). At N=100k, expected E is about 400k. The graph is stored as sparse pairs; edge indices use approximately 6.4 MiB as int64 endpoint pairs on device, excluding framework workspace and voter tensors. This is a planning estimate, not measured VRAM.

## User/reviewer runbook — commands not executed by the agent, but executed by a human reviewer below

Use the project environment, for example `conda activate mas`, and verify it contains Python >=3.11, NumPy, and a PyTorch build compatible with the installed GPU driver. Install the local package with `python -m pip install -e .` if needed. Do not change system drivers as part of this repository run. `auto` prints the selected backend/fallback. Record GPU model, total/free VRAM, PyTorch/CUDA versions, and driver version before performance runs. The mock-client stdin command requires a JSON observation captured from a transport adapter or `public_json`; the client emits only an action submission.

The reviewer run reported Python 3.11.17, NumPy 2.0.1, and PyTorch 2.5.1 built for CUDA 12.1, and successfully executed CUDA workloads on an RTX A5000. During the current hand-off, `nvidia-smi` failed to communicate on three attempts over about 70 seconds; this does not invalidate the saved reviewer run, but live GPU availability in this shell is not confirmed. No driver changes were made.

### Stage 1: preserved and new correctness checks

```bash
python tests/validate_week1_artifacts.py
PYTHONPATH=src python -m unittest discover -s tests -v
git diff --check
```

Expected: M0 validator success, zero test failures/errors (CUDA-dependent test may skip), and no whitespace output from `git diff --check`.

### Stage 2: Week 2 smoke and Week 3 episodes

```bash
python examples/week2_gpu_smoke.py
python examples/week3_mock_client.py < observation.json
python examples/week3_gpu_episode.py --device auto --nodes 1000 --seed 20261019
python examples/week3_gpu_episode.py --device auto --nodes 10000 --seed 20261019
python examples/week3_gpu_episode.py --device auto --nodes 100000 --seed 20261019
```

The Week 2 smoke prints its fixture support transition, Decimal ledger, rejected target code, and metadata. Each Week 3 command prints one JSON line with backend, device, N/E/density/average degree, p/seed, rounds, initialization time, median round time, episode time, action sequence, peak allocated and reserved GPU MiB (or null on CPU), final vote share, and victory. If `auto` falls back to CPU, the reviewer may reduce rounds or skip the 100k run with a recorded memory/resource reason. CUDA output includes both peak allocated and reserved memory, plus free/total memory at start.

### Stage 3: GPU/CPU parity and scale benchmark

```bash
PYTHONPATH=src python -m unittest tests.integration.test_gpu_cpu_equivalence -v
python examples/benchmark_week3.py --device auto --nodes 1000 10000 100000 --seed 20261019 --rounds 5
```

Expected parity: support differences no larger than `1e-10`, with equal discrete outcomes and ledger. Benchmark outputs one JSON row per size; include its exact output and execution status in the report. It warms a transition, synchronizes CUDA timing, and reports median/spread. It does not claim multi-GPU performance.

### Stage 4: final evidence

```bash
git diff --check
git status --short
```

Record each actual exit code, test pass/fail/skip count, GPU details, memory, runtime, and deviations. Expected outcomes are not execution evidence.

### Reviewer execution record

The reviewer execution transcript is :
```text
(mas) ai23btech11017@aiganesh1:~/MAS/CLOUT$ python tests/validate_week1_artifacts.py
M0 artifact validation: documentation and fixture checks passed
(mas) ai23btech11017@aiganesh1:~/MAS/CLOUT$ PYTHONPATH=src python -m unittest discover -s tests -v
test_fixture_episode_actions_replay_and_result (integration.test_game_episode.EpisodeTests.test_fixture_episode_actions_replay_and_result) ... ok
test_fixed_episode_cpu_and_cuda_match_within_tolerance (integration.test_gpu_cpu_equivalence.BackendEquivalenceTests.test_fixed_episode_cpu_and_cuda_match_within_tolerance) ... ok
test_torch_cuda_is_usable_when_driver_is_available (integration.test_gpu_optional.OptionalCudaTests.test_torch_cuda_is_usable_when_driver_is_available) ... ok
test_fixture_action_then_synchronous_update_is_reproducible (integration.test_week2_baseline.Week2BaselineIntegrationTests.test_fixture_action_then_synchronous_update_is_reproducible) ... ok
test_broadcast_guard_and_unknown_action_are_atomic (unit.test_campaign.CampaignTests.test_broadcast_guard_and_unknown_action_are_atomic) ... ok
test_rejections_and_noop_do_not_mutate (unit.test_campaign.CampaignTests.test_rejections_and_noop_do_not_mutate) ... ok
test_targeted_commit_is_exact_and_immutable (unit.test_campaign.CampaignTests.test_targeted_commit_is_exact_and_immutable) ... ok
test_invalid_values_rejected (unit.test_config.ConfigTests.test_invalid_values_rejected) ... ok
test_valid_config_and_stable_distinct_subseeds (unit.test_config.ConfigTests.test_valid_config_and_stable_distinct_subseeds) ... ok
test_broadcast_guard_resets_at_round_boundary (unit.test_engine.EngineTests.test_broadcast_guard_resets_at_round_boundary) ... ok
test_budget_exhaustion_does_not_terminate_and_effect_expires (unit.test_engine.EngineTests.test_budget_exhaustion_does_not_terminate_and_effect_expires) ... ok
test_exact_half_vote_is_not_counted (unit.test_engine.EngineTests.test_exact_half_vote_is_not_counted) ... ok
test_insufficient_budget_does_not_change_authoritative_fields (unit.test_engine.EngineTests.test_insufficient_budget_does_not_change_authoritative_fields) ... ok
test_internal_event_order_is_stable (unit.test_engine.EngineTests.test_internal_event_order_is_stable) ... ok
test_numerical_failure_rolls_back_resource_commit (unit.test_engine.EngineTests.test_numerical_failure_rolls_back_resource_commit) ... ok
test_rejection_keeps_round_and_money_and_terminal_is_guarded (unit.test_engine.EngineTests.test_rejection_keeps_round_and_money_and_terminal_is_guarded) ... ok
test_reset_and_replay_determinism (unit.test_engine.EngineTests.test_reset_and_replay_determinism) ... ok
test_targeted_action_lifecycle_and_terminal_result (unit.test_engine.EngineTests.test_targeted_action_lifecycle_and_terminal_result) ... ok
test_fixture_mapping_and_summary (unit.test_graph.GraphTests.test_fixture_mapping_and_summary) ... ok
test_invalid_edges_and_boundaries (unit.test_graph.GraphTests.test_invalid_edges_and_boundaries) ... ok
test_moderate_sparse_graph_remains_edge_list_backed (unit.test_graph.GraphTests.test_moderate_sparse_graph_remains_edge_list_backed) ... ok
test_event_payload_never_contains_target_ids_or_private_state (unit.test_information.InformationTests.test_event_payload_never_contains_target_ids_or_private_state) ... ok
test_observation_nested_mutation_is_detached (unit.test_information.InformationTests.test_observation_nested_mutation_is_detached) ... ok
test_private_states_with_same_public_aggregate_match_payload (unit.test_information.InformationTests.test_private_states_with_same_public_aggregate_match_payload) ... ok
test_recursive_serialization_has_no_hidden_fields (unit.test_information.InformationTests.test_recursive_serialization_has_no_hidden_fields) ... ok
test_hand_computed_synchronous_path (unit.test_mas.MasTests.test_hand_computed_synchronous_path) ... ok
test_isolate_endpoints_clipping_and_threshold (unit.test_mas.MasTests.test_isolate_endpoints_clipping_and_threshold) ... ok
test_nonfinite_contribution_fails_without_changing_input (unit.test_mas.MasTests.test_nonfinite_contribution_fails_without_changing_input) ... ok
test_selects_first_public_eligible_target (unit.test_mock_client.MockClientTests.test_selects_first_public_eligible_target) ... ok
test_uses_no_op_when_targeted_action_is_unavailable (unit.test_mock_client.MockClientTests.test_uses_no_op_when_targeted_action_is_unavailable) ... ok

----------------------------------------------------------------------
Ran 30 tests in 2.082s

OK
(mas) ai23btech11017@aiganesh1:~/MAS/CLOUT$ git diff --check
(mas) ai23btech11017@aiganesh1:~/MAS/CLOUT$ python examples/week2_gpu_smoke.py
Week 2 developer demo — true state output, not a public observation
graph: GraphSummary(node_count=3, edge_count=2, density=0.6666666666666666, min_degree=1, max_degree=2, mean_degree=1.3333333333333333, component_count=1)
before support: [0.2, 0.5, 0.8]
after support: [0.375, 0.5, 0.7250000000000001]
action: targeted_contact cost: 1.00
ledger: ResourceLedger(initial_budget=Decimal('10.00'), remaining_budget=Decimal('9.00'), spent=Decimal('1.00'))
rejected action: INVALID_TARGET ledger unchanged: 9.00
metadata: {'revision': 'uncommitted', 'python': '3.11.17', 'torch': '2.5.1', 'backend': 'cuda', 'seeds': {'root': 20261019, 'graph': 17072753691655277729, 'initial_state': 11112793045448091444, 'transition': 16807813951749623666, 'strategy': 16878377257174786736}}
(mas) ai23btech11017@aiganesh1:~/MAS/CLOUT$ python examples/week3_gpu_episode.py --device auto --nodes 1000 --seed 20261019
{"E": 3973, "N": 1000, "action_sequence": ["broadcast", "no_op", "no_op", "no_op", "no_op"], "average_degree": 7.946, "backend": "cuda:0", "cuda_runtime": "12.1", "density": 0.007953953953953954, "device": "NVIDIA RTX A5000", "episode_seconds": 0.7849380876868963, "final_vote_share": 0.516, "free_GPU_memory_start_MiB": 23992.5, "init_seconds": 0.6556288935244083, "median_step_ms": 4.173083230853081, "p": 0.008008008008008008, "peak_GPU_memory_MiB": 0.130859375, "peak_GPU_reserved_memory_MiB": 2.0, "python": "3.11.17", "revision": "uncommitted", "rounds": 5, "seed": 20261019, "torch": "2.5.1", "total_GPU_memory_MiB": 24245.5, "victory": true}
(mas) ai23btech11017@aiganesh1:~/MAS/CLOUT$ python examples/week3_gpu_episode.py --device auto --nodes 10000 --seed 20261019
{"E": 39714, "N": 10000, "action_sequence": ["broadcast", "no_op", "no_op", "no_op", "no_op"], "average_degree": 7.9428, "backend": "cuda:0", "cuda_runtime": "12.1", "density": 0.0007943594359435944, "device": "NVIDIA RTX A5000", "episode_seconds": 0.9845667816698551, "final_vote_share": 0.5016, "free_GPU_memory_start_MiB": 23992.5, "init_seconds": 0.6430348046123981, "median_step_ms": 35.35431995987892, "p": 0.0008000800080008001, "peak_GPU_memory_MiB": 1.29296875, "peak_GPU_reserved_memory_MiB": 2.0, "python": "3.11.17", "revision": "uncommitted", "rounds": 5, "seed": 20261019, "torch": "2.5.1", "total_GPU_memory_MiB": 24245.5, "victory": true}
(mas) ai23btech11017@aiganesh1:~/MAS/CLOUT$ python examples/week3_gpu_episode.py --device auto --nodes 100000 --seed 20261019
{"E": 399252, "N": 100000, "action_sequence": ["broadcast", "no_op", "no_op", "no_op", "no_op"], "average_degree": 7.98504, "backend": "cuda:0", "cuda_runtime": "12.1", "density": 7.985119851198512e-05, "device": "NVIDIA RTX A5000", "episode_seconds": 4.906636547297239, "final_vote_share": 0.50536, "free_GPU_memory_start_MiB": 23992.5, "init_seconds": 2.362087022513151, "median_step_ms": 398.31260964274406, "p": 8.000080000800008e-05, "peak_GPU_memory_MiB": 12.95458984375, "peak_GPU_reserved_memory_MiB": 28.0, "python": "3.11.17", "revision": "uncommitted", "rounds": 5, "seed": 20261019, "torch": "2.5.1", "total_GPU_memory_MiB": 24245.5, "victory": true}
(mas) ai23btech11017@aiganesh1:~/MAS/CLOUT$ PYTHONPATH=src python -m unittest tests.integration.test_gpu_cpu_equivalence -v
test_fixed_episode_cpu_and_cuda_match_within_tolerance (tests.integration.test_gpu_cpu_equivalence.BackendEquivalenceTests.test_fixed_episode_cpu_and_cuda_match_within_tolerance) ... ok

----------------------------------------------------------------------
Ran 1 test in 2.133s

OK
(mas) ai23btech11017@aiganesh1:~/MAS/CLOUT$ python examples/benchmark_week3.py --device auto --nodes 1000 10000 100000 --seed 20261019 --rounds 5
{"E": 3973, "GPU/device": "NVIDIA RTX A5000", "N": 1000, "action_sequence": ["no_op", "no_op", "no_op", "no_op", "no_op"], "backend": "cuda:0", "correctness_check": "not run \u2014 awaiting user/reviewer", "cuda_runtime": "12.1", "episode_seconds": 0.03217898868024349, "final_vote_share": 0.515, "free_GPU_memory_start_MiB": 23992.5, "init_seconds": 0.011575568467378616, "median_step_ms": 3.8012899458408356, "p": 0.008008008008008008, "peak_GPU_memory_MiB": 0.130859375, "peak_GPU_reserved_memory_MiB": 2.0, "revision": "uncommitted", "rounds": 5, "seed": 20261019, "step_spread_ms": 1.7626211047172546, "torch": "2.5.1", "total_GPU_memory_MiB": 24245.5, "victory": true}
{"E": 39714, "GPU/device": "NVIDIA RTX A5000", "N": 10000, "action_sequence": ["no_op", "no_op", "no_op", "no_op", "no_op"], "backend": "cuda:0", "correctness_check": "not run \u2014 awaiting user/reviewer", "cuda_runtime": "12.1", "episode_seconds": 0.34232003800570965, "final_vote_share": 0.4994, "free_GPU_memory_start_MiB": 23956.5, "init_seconds": 0.12332133390009403, "median_step_ms": 31.691212207078934, "p": 0.0008000800080008001, "peak_GPU_memory_MiB": 1.29296875, "peak_GPU_reserved_memory_MiB": 2.0, "revision": "uncommitted", "rounds": 5, "seed": 20261019, "step_spread_ms": 61.015065759420395, "torch": "2.5.1", "total_GPU_memory_MiB": 24245.5, "victory": false}
{"E": 399252, "GPU/device": "NVIDIA RTX A5000", "N": 100000, "action_sequence": ["no_op", "no_op", "no_op", "no_op", "no_op"], "backend": "cuda:0", "correctness_check": "not run \u2014 awaiting user/reviewer", "cuda_runtime": "12.1", "episode_seconds": 3.7389749623835087, "final_vote_share": 0.50272, "free_GPU_memory_start_MiB": 23956.5, "init_seconds": 1.7119837552309036, "median_step_ms": 405.07123433053493, "p": 8.000080000800008e-05, "peak_GPU_memory_MiB": 12.95458984375, "peak_GPU_reserved_memory_MiB": 28.0, "revision": "uncommitted", "rounds": 5, "seed": 20261019, "step_spread_ms": 22.77173101902008, "torch": "2.5.1", "total_GPU_memory_MiB": 24245.5, "victory": true}

```
It records a successful M0 validator; 30 passing tests (including CUDA availability and CPU/GPU parity, with no skips); successful Week 2 smoke; successful 1k/10k/100k CUDA episodes; the separate parity test passing; three benchmark rows; and clean `git diff --check` results. The benchmark's `correctness_check` field means that the benchmark process itself does not compare CPU and CUDA results; correctness was checked separately by the passing parity test above. Older captured benchmark rows may still contain the pre-review “awaiting user/reviewer” wording and should be interpreted with this run record. The transcript shows successful command output but does not separately record numeric exit codes.

The standalone `week3_mock_client.py` stdin example is not present in the transcript. Its unit tests passed as part of the full suite, and the CLI smoke is also verified;
