# Week 3 Hand-off

**Status:** implementation and reviewer runbook execution are complete.

## Week 4 integration surface

- `GameEngine.initialize` creates seeded graph/voters or accepts a fixture.
- `GameEngine.observe` returns detached `Observation`; clients serialize only
  `to_public_dict()` / `public_json()`.
- `GameEngine.step` is the sole round transaction and returns `StepResponse`.
- `GameResult`, `TraceRecord`, and safe public events are available for later
  temporal/epistemic consumers. No LTL evaluator is included.
- `GameEngine.replay` and `reset` provide seeded episode reproducibility.

## Review runbook

Follow the stages in [week3_api_notes.md](week3_api_notes.md). Begin with M0 and full unittest checks, then run the existing Week 2 smoke, 1k/10k/100k episode runs, CPU/GPU comparison, and scale harness. Results below summarize the recorded run; the transcript contains exact commands and full JSON rows.

## Benchmark report template

| backend | GPU/device | N | E | p | seed | rounds | init_seconds | median_step_ms | episode_seconds | peak_GPU_memory_MiB | final_vote_share | victory | correctness_check |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cuda:0 | NVIDIA RTX A5000 | 1,000 | 3,973 | 8/999 | 20261019 | 5 | 0.0116 | 3.801 | 0.0322 | 0.131 | 0.515 | yes | separate parity test passed |
| cuda:0 | NVIDIA RTX A5000 | 10,000 | 39,714 | 8/9999 | 20261019 | 5 | 0.1233 | 31.691 | 0.3423 | 1.293 | 0.4994 | no | separate parity test passed |
| cuda:0 | NVIDIA RTX A5000 | 100,000 | 399,252 | 8/99999 | 20261019 | 5 | 1.7120 | 405.071 | 3.7390 | 12.955 | 0.50272 | yes | separate parity test passed |

The report records the full unittest suite as 30/30 passing with no skips, M0 validation and both whitespace checks passing, and all three scripted CUDA episodes and the separate CPU/GPU parity test passing. Hardware was RTX A5000 (24,245.5 MiB total); episode runs reported 23,992.5 MiB free at start, with Python 3.11.17, PyTorch 2.5.1 and CUDA runtime 12.1. Benchmark timings are observations for this environment, not a GPU-vs-CPU speedup claim. The benchmark's own correctness field only describes checks done inside that process; parity was checked by the separate passing integration test.

The transcript does not show the standalone `week3_mock_client.py` stdin smoke (its unit tests did pass), see the self-review. The reviewer CUDA results remain valid recorded evidence.

## Known integration boundary 
The current CUDA implementation computes graph aggregation/update on GPU and keeps sparse edge indices/degrees resident for the game. It transfers voter vectors at each transition boundary to preserve the canonical immutable CPU state API. The recorded harness quantifies this path at the listed sizes. No measured GPU speedup is claimed.
