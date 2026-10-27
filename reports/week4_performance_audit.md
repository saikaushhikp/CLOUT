# Week 4 GPU/CPU Performance Audit

**Status:** the matched CPU/CUDA harness and CUDA episode smoke passed in the upgraded `mas` environment on 2026-10-23. Historical Week 3 CUDA outputs remain separate from the new matched result.

## Current execution evidence (CPU and CUDA; 2026-10-23)

Environment: conda environment named `mas`, Python 3.12.15, NumPy 2.5.2, PyTorch 2.7.1+cu118 built with
CUDA 11.8, driver 535.183.01, and three NVIDIA RTX A5000 devices. CUDA was
available. The exact `nvidia-smi` query succeeded with 24564 MiB total memory per device and 24199-24218 MiB free at query time.

The N=1k matched CPU/CUDA run used seed 20261019 and five actions (`broadcast`, then four `no_op` actions). CPU median step time was 8.2564 ms and CUDA median step time was 4.4778 ms; initialization was 17.6509 ms CPU and 13.9808 ms CUDA. Final support, ledger, and terminal result matched, with maximum support difference `2.220446049250313e-16` and tolerance `1e-10`. These are single-run, environment-specific measurements and do not establish a general speedup claim.

The requested Week 3 commands and matched Week 4 comparison were rerun with GPU-visible command access on 2026-10-23. The restricted command sandbox used for an earlier attempt concealed NVIDIA devices, which caused `--device auto` to fall back to CPU; outside that sandbox, `nvidia-smi` and PyTorch detected three RTX A5000s and CUDA was available. GPU-visible rerun results:

| Workload | Backend | Median step | Episode | Peak allocated VRAM |
| --- | --- | ---: | ---: | ---: |
| Week 3 episode, N=1k, broadcast + four no-ops | cuda:0 | 4.242 ms | 0.1901 s | 0.131 MiB |
| Week 3 benchmark, N=1k, no-ops | cuda:0 | 3.902 ms | 0.0336 s | 0.131 MiB |
| Week 3 benchmark, N=10k, no-ops | cuda:0 | 32.261 ms | 0.3396 s | 1.293 MiB |
| Week 3 benchmark, N=100k, no-ops | cuda:0 | 416.647 ms | 3.9268 s | 12.955 MiB |
| Matched Week 4, N=1k | CPU / cuda:0 | 7.653 / 4.050 ms | 0.0566 / 0.0338 s | CUDA 0.132 MiB |
| Matched Week 4, N=10k | CPU / cuda:0 | 71.349 / 32.980 ms | 0.5543 / 0.4034 s | CUDA 1.294 MiB |

The GPU CPU-equivalence test passed, and the full suite passed 69 tests with both CUDA tests active. Matched initial states, ledgers, and terminal results were equal at both sizes; maximum support differences were `1.11e-16` and `2.22e-16` against `1e-10`. These are individual runs and must not be generalized into a speedup claim. Full commands and setup context are recorded in [week4_handoff.md](../docs/spec/week4_handoff.md).

## Historical measured evidence (Week 3 reviewer-reported only)

The saved [Week 3 transcript](../docs/spec/reviewer_runbook_outputs.txt) reports RTX A5000, PyTorch 2.5.1 / CUDA runtime 12.1, Python 3.11.17. The benchmark reported N=1k/10k/100k median step times 3.801/31.691/405.071 ms and episode times 0.0322/0.3423/3.7390 seconds; peak allocated memory was 0.131/1.293/12.955 MiB (reserved 2/2/28 MiB). These rows used `no_op` for five rounds and did not compare CPU performance. A separate small CPU/CUDA parity test was reported passing. A different scripted episode used broadcast then no-ops. These measurements are historical, hardware/revision-specific, and establish neither speedup nor a general performance envelope. The transcript did not record a driver version. The historical environment details are intentionally preserved here and are not the current `mas` environment.

## Static observations / hypotheses (not measurements)

| Hypothesis | Static basis | Proposed measure |
| --- | --- | --- |
| Per-round CPU→GPU transfer may dominate smaller episodes | `mas.py:advance_support` builds support/susceptibility/effect tensors from CPU voter tuples; `GameState` is CPU canonical in `engine.py` | Matched CPU/CUDA by N and horizon; separately time transition and synchronization; profile transfer/copy events |
| Immutable voter tuple reconstruction has host cost | `advance_support` constructs returned `VoterState` tuple each transition | Profile tuple/object construction by N and compare against tensor kernel time; do not change ownership without rollback design |
| ER graph creation and component summary may dominate initialization | `graph.py:erdos_renyi_graph` creates host edge pairs; `ElectorateGraph.summary` performs adjacency/component traversal; `engine.initialize` summarizes before device preparation | Separate graph build, summary, device preparation, and engine init timings |
| `index_add_` and output synchronization may dominate CUDA transition | `mas.py` device transition uses sparse index-add aggregation; engine validates finite values and converts outputs back to host | CUDA profiler plus synchronized step timing; capture kernel, transfer, synchronization and validation costs |
| State/event/trace copying grows with horizon | `engine.step` constructs immutable snapshots and concatenates event/trace tuples | Hold N fixed, vary horizon, measure allocation/time and retained history size |
| Small N may favor CPU | CUDA has context, launch, transfer, and synchronization overhead while CPU path is direct NumPy/Python | Matched N sweep beginning with 1k; report crossover only if measured on same machine |

These are code-derived hypotheses, not measured diagnoses. Persistent GPU-resident voter state is **deferred**: it would complicate immutable snapshot ownership, transactional rollback, reset/replay, and parity. Revisit only if matched measurements show a meaningful cost and a clear design keeps CPU reference semantics and per-device VRAM limits.

## Proposed matched experiment

`examples/week4_backend_comparison.py` runs the same ER config/root seed and fixed `broadcast, no_op...` sequence on CPU and one explicitly selected CUDA device. It checks identical initialization, final support maximum absolute difference (tolerance `1e-10`), ledger equality, and terminal-result equality; it reports per-backend initialization, median/spread of synchronized step times, total time, device and allocated/reserved/free/total VRAM. Its JSON omits voter arrays even though the internal comparison uses them. Use N=1k and 10k runs separately; use `--cuda-device cuda:0` or `cuda:1` only when that device is available. VRAM is per device; do not combine capacities.

The measured run record should be attached to Week 4 handoff with GPU model, driver/runtime/framework versions, selected backend, seed/config/actions, correctness result and resource measurements. A result from CPU fallback is not CUDA evidence. A single run is not a robust performance claim; compare medians/spreads across repeated human-run invocations under controlled load.
