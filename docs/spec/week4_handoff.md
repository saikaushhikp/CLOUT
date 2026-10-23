# Week 4 Handoff

**Status:** CPU and CUDA end-to-end checks passed in the upgraded `mas`
environment on 2026-10-23. The five-member Week 4 manual review is complete
and approved. Week 3 remains accepted.

## Coding-agent execution record (2026-10-23)

Executed from the repository root with Python 3.12.15, NumPy 2.5.2,
PyTorch 2.7.1+cu118 (built with CUDA 11.8), using the upgraded `mas`
environment. The NVIDIA driver is 535.183.01 and three NVIDIA RTX A5000
devices are visible; the working tree revision was uncommitted. Results:

- M0 validator passed.
- Focused Week 4 unit suites: 41 passed (including mock-client CLI, JSON
  round-trip, and additive-field compatibility coverage). Full suite:
  69 passed, including the
  CUDA-dependent tests; PyTorch reported `cuda_available=True` and three
  devices.
- Safe mock-client CLI with the public fixture passed and emitted
  `{"action_id": "targeted_contact", "target_ids": ["v0"]}`.
- Paired strategy example (N=32, two scenarios, three seeds, three policies):
  18 episodes completed, zero failed/skipped/rejected proposals; aggregates
  emitted for all policies.
- The Week 3 episode smoke completed with `backend: cuda:0`, reporting
  `N=1000`, `E=3973`, CUDA runtime `11.8`, and an RTX A5000.
- The Week 4 matched backend harness at N=1k passed initialization, ledger,
  terminal-result, and final-voter-state comparisons. Maximum final support
  difference was `2.220446049250313e-16` against tolerance `1e-10`.
  The run used CPU and `cuda:0` with the same seed/config/action sequence;
  CUDA median step time was 4.4778 ms and CPU median step time was 8.2564 ms
  in that run.
- `nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version
  --format=csv,noheader` succeeded and reported three RTX A5000 devices with
  24564 MiB total memory and driver `535.183.01`.
- The mock-client isolation test initially failed because its module docstring
  included an engine import in a sample command. The sample was replaced with
  the safe fixture smoke command; focused and full suites then passed.

Activate the environment before rerunning:

```bash
conda activate mas
```

The results above are agent-run evidence. A reviewer should rerun the same
commands to confirm them under the intended review environment.

### Requested Week 3 rerun and sandbox diagnosis (2026-10-23)

The three commands in the GPU runbook were rerun exactly in `mas`. An initial
attempt in the restricted command sandbox falsely appeared CPU-only: that
sandbox did not expose `/dev/nvidia*`. A read-only driver/PyTorch probe outside
that sandbox confirmed three RTX A5000 devices and
`torch.cuda.is_available() == True`; the requested commands and Week 4 GPU
checks were then rerun with GPU-visible command access. Do not count an `auto`
run that prints `CUDA fallback` as GPU evidence.

Results from the GPU-visible rerun:

- `week3_gpu_episode.py --device auto --nodes 1000`: selected `cuda:0`
  (NVIDIA RTX A5000), succeeded; episode 0.1901 s, median step 4.242 ms,
  peak allocated/reserved memory 0.131/2 MiB.
- `benchmark_week3.py --device auto --nodes 1000 10000 100000 --rounds 5`:
  all sizes selected `cuda:0` and completed. Median step times were
  3.902/32.261/416.647 ms; episode times 0.0336/0.3396/3.9268 s; peak
  allocated memory 0.131/1.293/12.955 MiB (reserved 2/2/28 MiB).
- `test_gpu_cpu_equivalence`: passed (1 test; no skips).
- Full suite rerun with GPU visibility after the contract-test additions:
  69 passed, including both CUDA tests. The mock-client focused suite passed 7.
- Week 4 matched comparison passed at N=1k and 10k. Maximum final-support
  differences were `1.11e-16` and `2.22e-16`; both were within `1e-10`, and
  initial state, ledger, and terminal result matched. Single-run CUDA median
  steps were 4.050 ms (CPU 7.653 ms) at N=1k and 32.980 ms (CPU 71.349 ms)
  at N=10k. These timings are environment-specific, not a general speedup
  claim.

Runtime: Python 3.12.15, NumPy 2.5.2, PyTorch 2.7.1+cu118 / CUDA 11.8,
driver 535.183.01. `nvidia-smi` reported three RTX A5000 devices with
24564 MiB each, while the documented project baseline is two; discover device
count at runtime. Full benchmark and matched-comparison JSON remains in the
execution output; no generated result file was added.

## What is included

- Canonical predicate traces and a bounded-LTL evaluator: `logic.py`,
  `logic_semantics_v1.md`, and hand-computable fixtures.
- Observation-only strategy protocol and three baselines: `strategies.py` and
  `strategy_api_v1.md`.
- Paired-seed JSONL/aggregate experiment runner: `experiments.py`,
  `week4_compare_strategies.py`, and `week4_experiment_template.md`.
- Safe public observation fixture and the Week 3 mock-client CLI command below.
- Static performance audit and an explicit matched CPU/CUDA comparison
  harness. No new speedup claim is made.
- Carry-forward inventory and compute-policy reconciliation in
  `reports/week4_baseline_audit.md` and `docs/spec/week4_compute_policy.md`.

## Staged human/reviewer runbook

Run from the repository root after `conda activate mas`. Optional CUDA
comparisons also need PyTorch/CUDA and a functioning host driver. Do not change
system drivers as part of the run. If a restricted execution sandbox reports
CPU fallback while the host is expected to have GPUs, check device visibility
in a GPU-enabled shell before concluding CUDA is unavailable; report the
actual backend selected by each command.

### 1. Preserved M0 artifact validator

```bash
python tests/validate_week1_artifacts.py
```

Expected: M0 documentation/fixture checks pass. Likely failure: missing or
malformed M0 fixture/docs; inspect `tests/fixtures/` and the Week 1 specs.
Observed 2026-10-23: passed.

### 2. Full unit/integration suite

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Expected: suite summary with pass/fail/error/skip counts; CUDA-optional tests
may skip without a device. Likely failures: missing NumPy/PyTorch or import
path; inspect `pyproject.toml`, `src/dumbocracy/`, and the failing test.
Observed 2026-10-23: 69 passed, including the CUDA-dependent tests.

### 3. Focused Week 4 suites

```bash
PYTHONPATH=src python -m unittest tests.unit.test_logic -v
PYTHONPATH=src python -m unittest tests.unit.test_strategies -v
PYTHONPATH=src python -m unittest tests.unit.test_experiments -v
PYTHONPATH=src python -m unittest tests.unit.test_mock_client -v
```

Expected: focused trace boundary/unknown, policy contract, paired output and
fixture schema cases pass. Likely failures: changed trace/API schema or a
seed/aggregate expectation; inspect `logic.py`, `strategies.py`,
`experiments.py` and their tests. Observed 2026-10-23: 41 passed across the
four focused suites.

### 4. Safe mock-client CLI (Week 3 command)

```bash
PYTHONPATH=src python examples/week3_mock_client.py < tests/fixtures/public_observation_v1.json
```

Expected one JSON action proposal with allowed keys only:

```json
{"action_id":"targeted_contact","target_ids":["v0"]}
```

Likely failures: malformed fixture or invalid `legal_actions`; inspect the
fixture and `examples/week3_mock_client.py`. Observed 2026-10-23: passed with
the expected action. Run in the activated `mas` shell; wrapping this stdin
command in `conda run` may consume the pipe. This run resolves the old CLI
evidence gap without changing the historical Week 3 transcript.

Transport flow: the engine projects `Observation`; a client serializes that
public allowlist; the client returns `{ "action_id": ..., "target_ids": ... }`;
the engine validates it and returns `StepResponse`; only an accepted terminal
step includes a result. `GameState` is never sent to the client. The fixture is
safe and schematic; it contains no hidden voter values, seeds, or graph edges.

Illustrative transcript (request content is the committed fixture; response
values are schematic and not a recorded engine execution):

```text
client input:  {"schema_version":"1.0","round_index":0,"remaining_budget":"10.00","aggregate_support":0.5,"graph_summary":{"node_count":3,"edge_count":2,"density":0.6666666666666666,"min_degree":1,"max_degree":2,"mean_degree":1.3333333333333333,"component_count":1},"legal_actions":[{"action_id":"targeted_contact","cost":"1.00 per target","target_requirement":"one or more unique public voter IDs","parameters":{"delta_target":0.2},"eligible_target_ids":["v0","v1","v2"]},{"action_id":"broadcast","cost":"5.00","target_requirement":"no target; at most once per round","parameters":{"delta_broadcast":0.1},"eligible_target_ids":[]},{"action_id":"no_op","cost":"0.00","target_requirement":"no target","parameters":{},"eligible_target_ids":[]}],"terminal_status":false,"public_events":[]}
client output: {"action_id":"targeted_contact","target_ids":["v0"]}
engine reply:  {"accepted":true,"error_code":null,"terminal":false,"action_id":"targeted_contact","result":null,"observation":{"schema_version":"1.0","round_index":1,"remaining_budget":"9.00","aggregate_support":0.53,"graph_summary":{"node_count":3,"edge_count":2,"density":0.6666666667,"min_degree":1,"max_degree":2,"mean_degree":1.3333333333,"component_count":1},"legal_actions":[{"action_id":"targeted_contact","cost":"1.00 per target","target_requirement":"one or more unique public voter IDs","parameters":{"delta_target":0.2},"eligible_target_ids":["v0","v1","v2"]},{"action_id":"no_op","cost":"0.00","target_requirement":"no target","parameters":{},"eligible_target_ids":[]}],"terminal_status":false,"public_events":[]}}
```

Use the committed fixture for the CLI check; the inline request/response is a
complete schematic payload documenting field names, not a captured transcript.

Public schema version is `1.0`: additive compatible fields require a minor
version update; removal/meaning changes require a major version and migration
note. Hidden state is never a public schema field.

### 5. Small paired-seed baseline comparison

```bash
PYTHONPATH=src python examples/week4_compare_strategies.py --nodes 32 --rounds 5 --seeds 101 202 303 --backend cpu
```

Expected: one JSONL episode per strategy/scenario/seed pair, followed by an aggregate
JSON object. Check `paired_run_id`, seed hierarchy, graph/config/runtime,
action/error sequence, terminal result, logic diagnostics, timing definitions,
and aggregate denominators. Likely failures: graph/config construction or
policy proposal rejected by engine; inspect `strategies.py` and `experiments.py`.
No exact scores are expected in advance. Observed 2026-10-23: 18 episodes
completed with zero failures, skips, or rejected proposals; aggregates emitted.

### 6. Matched CPU/CUDA correctness and resource measurement (opt-in)

```bash
PYTHONPATH=src python examples/week4_backend_comparison.py --nodes 1000 --seed 20261019 --rounds 5 --cuda-device cuda:0
PYTHONPATH=src python examples/week4_backend_comparison.py --nodes 10000 --seed 20261019 --rounds 5 --cuda-device cuda:0
```

Repeat with `--cuda-device cuda:1` only if explicitly measuring that separate
GPU. Expected JSON includes same-seed/action metadata, per-backend init and
synchronized step timing, memory, final state max difference, ledger/result
equality and correctness. Correctness requires `<= 1e-10` support difference
and equal ledger/discrete result. Likely failures: CUDA driver/PyTorch mismatch,
selected device unavailable, or numerical parity outside tolerance. Inspect
`mas.py`, `engine.py`, and `reports/week4_performance_audit.md`. Observed
2026-10-23: N=1k passed in the upgraded `mas` environment with maximum final
support difference `2.220446049250313e-16`, equal ledger/result, and CUDA
runtime `11.8`.

### 7. Existing Week 3 GPU scale commands (separate, opt-in)

```bash
python examples/week3_gpu_episode.py --device auto --nodes 1000 --seed 20261019
python examples/benchmark_week3.py --device auto --nodes 1000 10000 100000 --seed 20261019 --rounds 5
PYTHONPATH=src python -m unittest tests.integration.test_gpu_cpu_equivalence -v
```

These are not substitutes for the matched Week 4 comparison and do not establish
GPU speedup. `auto` may fall back to CPU; report actual backend/device. Likely
failures: missing PyTorch/CUDA or memory constraints; inspect Week 3 API notes
and record skips/deviations. Observed 2026-10-23: the episode completed on
`cuda:0`, and the parity test passed.

### 8. Human diff/status review

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; review all intended Week 4 files and preserve
unrelated user changes. Rerun this review after your changes are staged.

## GPU/CPU interpretation

See [week4_performance_audit.md](../../reports/week4_performance_audit.md).
Week 3 rows are historical reviewer evidence only. They used CUDA episodes but
did not compare CPU speed. Week 4's matched harness compares identical seed,
config, initial voters, action sequence, final voter support, ledger, and
outcome. Each A5000 has separate VRAM; the live device count (three in the
recorded run) differs from the documented two-device baseline. Do not report
pooled memory or infer multi-GPU execution. The recorded single-run comparisons
are environment-specific and do not establish a general speedup claim.

## Week 5 backlog

- Full epistemic information-set/state-equivalence model checking.
- Additional bounded-logic properties only after specification/approval.
- Strategy search/optimization, RL, parameter sweeps, or broad campaigns.
- Persistent GPU-resident authoritative state only after matched evidence and
  approved atomic ownership/rollback design.
- Broader performance tuning, large statistical evaluation and uncertainty
  estimates based on human-run evidence.
- Multi-GPU/distributed execution only if one-device limits justify an
  explicitly approved design for separate VRAM and deterministic replay.
- Production UI/service, persistence, security hardening, integration/beta.
- Optional reviewer reruns of the documented checks to independently confirm
  the coding-agent results.
