# v3 experiment suite: handoff to the owner

The owner makes every commit (plan, Working rules). Nothing below has been committed, deleted or moved.

## Phase 0: shared infrastructure (2026-11-10)

### New files

| File | Contents |
|---|---|
| `experiments/x_common.py` | Re-exports `p7_common` (wilson, rate, mean_ci, paired, mcnemar, spearman, provenance, save, run_tasks); adds `holm`, `power_paired_binary`, `seed_block` (§1.5 table), `write_row` / `read_rows` (JSONL), `override` (a copy of the one in `balance.py`, which still works; accepts `"a.b=v"` or `(path, raw)`, strings included), `apply_overrides`; GPU helpers `select_device` (cuda:2 by default, else the GPU with the most free memory that is not busy and not in `X_AVOID_GPUS`, default "1"; never silently CPU), `calibrate_batch` (memory-based batch sizer, 70% of free memory), `run_chunks` (OOM back-off: halve, `empty_cache`, retry, log), `GpuMonitor` (`nvidia-smi` utilisation thread), `provenance_v3` (device name, batch sizes, peak and free memory, wall time, mean GPU utilisation); `parity_check` (CPU vs GPU, statistical tolerance); `features_of` (hidden-world features of §2, cached to `reports/v3/raw/features_<seed>.json`; spectra by GPU `torch.lobpcg` or SciPy, chosen by measured speed). |
| `experiments/x_batch.py` | `run_batch`: B true electorates (same N) in one disjoint-union graph, each with its own manifesto, its own per-round `RoundContext` policy and its own seed streams, in the P8 round order of `simulate_world`. Hooks: `pick_fn` (X4/X18 pick rules), `streams` (X20 salted streams), `observer` (per-round tensors); cascade off and the conversion rule work through the config. `outcome_row` for JSONL rows. |
| `experiments/x_policies.py` | The gate policies of `balance.py` (same objects) plus `WorkerTiming` (front/even/late × proportional/largest/best_matched/region), `RegionFocus`, `Sequenced` (X11), `Probe` (X17 RoundContext stub), `PeoplePleaser`, `StrongArm`, manifesto helpers, `make_policy`. |
| `experiments/x_streams.py` | `SaltedStreams(root, salts)`: equals `SeedStreams` exactly with zero salts. |
| `experiments/plot_v3.py` | Scaffolding: one registered function per experiment, figures drawn only from `reports/v3/<id>.json`. No plots yet. |
| `experiments/x_gpu_benchmarks.py` | Phase 0 benchmarks → `reports/v3/gpu_benchmarks.json`. |
| `tests/unit/test_x_common.py` | 22 tests (2 with the `gpu` marker). |
| `reports/v3/report.md` | Skeleton per §14.3; every experiment "Not run yet", X24 dropped, X25 deferred. |
| `reports/v3/predictions.md` | Every "Predict" line copied verbatim from the plan, registered 2026-11-10 (X9 and X22 have none in the plan). |
| `reports/v3/gpu_benchmarks.json`, `reports/v3/logs/gpu_benchmarks*.log` | Benchmark results and logs. |
| `reports/v3/handoff_owner.md` | This file. |

### Changed files (approved library change, D2 / §9: X23 conversion rule)

| File | Change |
|---|---|
| `src/dumbocracy/config.py` | `WorkerConfig.conversion_rule: Literal["gamma", "gamma_p", "gamma_p2"] = "gamma_p"` (validated), `CONVERSION_RULES`. |
| `src/dumbocracy/dynamics/campaign.py` | `conversion_chance(..., rule="gamma_p")`; `campaign_round` passes `worker_cfg.conversion_rule`. The default branch is the old expression unchanged. |
| `src/dumbocracy/strategies/worlds.py` | `WorldBatch.step` passes `cfg_w.conversion_rule`. |

Default preserved bit for bit: `test_conversion_rule_default_is_bit_identical` and
`test_conversion_rule_default_reproduces_games_and_rules_differ`. `GameConfig.to_record()` now contains
`workers.conversion_rule` (new key in every result record). This change was made in Phase 0 on the lead's
instruction, although plan §2 says "no library code changes in Phase 0"; it is recorded as a deviation in `report.md`.

### Archive move: NOT performed

Archive move not performed, offered: `git mv reports/v2 reports/archive/v2.1` together with the final report if the
owner wants. Reason (lead's decision): D4 keeps `reports/v2/final_report.md` as the v2 record and its links point at
`reports/v2/*.json` and `figures/`; v3 output goes to the separate `reports/v3/`.

### Results

- **Tests:** full suite `OMP_NUM_THREADS=4 python -m pytest tests -q`: **285 passed**, 0 failed, 0 skipped (GPU tests ran on
  cuda:2). New: 22 in `tests/unit/test_x_common.py`.
- **x_batch parity** (`gpu_benchmarks.json` → `parity`, 20 seeds from 10000, N = 10⁴, ORACLE manifesto, ORACLE planner):
  - CPU: **20/20 worlds bit-identical** to separate `simulate_world` runs (trajectory, votes, converts, spreaders,
    allocations, spend, final v, worker ids, attempts). Unit tests prove it for B = 1 and B = 4 with mixed per-world
    policies, the conversion rule, salted streams and cascade-off.
  - GPU: **20/20 worlds with identical discrete outcomes** (votes, victory, converts, spreaders, allocations, spend);
    floats within 6·10⁻⁸. Bitwise identity is impossible there: `simulate_world` itself is not bitwise
    reproducible on CUDA (0/20 bit-identical run to run, 20/20 discrete-identical) because `index_add_` in the DeGroot
    aggregation uses float atomics. The batch therefore uses a deterministic CSR sparse mat-mul on CUDA (the
    `WorldBatch` approach); `aggregation="exact"` reproduces the library op instead.
  - CPU vs GPU (different random streams per device), statistical check: passed (ΔE = 0.0012, win rate 0.85 vs 0.85).
- **Benchmarks** (cuda:2, RTX A5000; per game, 5 rounds, ₫500; `pay_split` policy):

  | N | B | ms per game (GPU batch) | CPU `simulate_world` ms | speed-up | peak GiB above inputs |
  |---|---|---|---|---|---|
  | 10⁴ | 1 / 8 / 32 / 64 | 13.8 / 4.6 / 3.5 / 3.3 | 42 | 3.0 / 9.1 / 11.9 / 12.9 | 0.01 / 0.09 / 0.35 / 0.70 |
  | 10⁵ | 1 / 8 / 32 / 64 | 14.6 / 8.0 / 7.2 / 7.2 | 382 | 26 / 48 / 53 / 53 | 0.06 / 0.87 / 3.46 / 6.91 |

  With the ORACLE planner policy the per-world Python decision code dominates (≈ 14–17 ms per game): ×3.9 at 10⁴ and
  ×22 at 10⁵ for B = 64. BayesPlanner: rollout planning 5–29× faster on the GPU, whole decision only 1.2–1.4×. Logic
  timeline: GPU 1.2× (16 worlds) and 1.3× (64 worlds) faster. Spectra: SciPy faster at 10⁴ (0.5 s vs 2.5 s), GPU
  `lobpcg` faster at 10⁵ (5.4 s vs 7.2 s); `features_of` picks the GPU from 5·10⁴ voters on.
- **Recommended batch sizes for Phase 1** (throughput is flat beyond B ≈ 32 because the per-world policy loop is
  CPU-bound; memory per world measured by `calibrate_batch`: ≈ 17 MB at 10⁴ and ≈ 170 MB at 10⁵ including inputs):
  N = 10³–10⁴: one batch per cell (B = 50–100); N = 10⁵: B = 64 (≈ 11 GB with inputs); N = 10⁶: B ≈ 8 (estimated
  ≈ 1.7 GB per world, not measured; run `calibrate_batch` first and rely on `run_chunks` back-off).

### Findings worth the owner's attention

1. **`BayesPlanner(device="cuda")` crashes** in `manifesto()` (`strategies/players.py`, the `torch.tensor(simplex_list(...))`
   start is built on the CPU and stacked with GPU tensors). The benchmark works around it experiment-side (manifesto
   worlds on the CPU, rollouts on the GPU). Suggested one-line library fix, **needs approval**: add
   `device=v_t.device` to that `torch.tensor(...)` call. X5, X6, X14, X16, X17 and X21 would use it.
2. The P8 handoff states CPU is fastest for BayesPlanner; on this host the GPU is faster for the planning step at every
   tested size (8–32 worlds × 1,600–4,000 voters).
3. `provenance_v3` records `torch.cuda.max_memory_allocated()` since the last reset; the benchmark resets it per B, so
   use the per-B `peak_bytes_above_inputs` rows and `gpu_mem_used_mib_max` (12.96 GB on the full run) for the totals.

### Plan ambiguities resolved

- `override(config, "a.b=v")` (plan) vs `override(config, path, raw)` (balance.py): both signatures are accepted.
- `seed_block("X9", n)` starts at the canonical seed 20261107; `seed_block("X15", …)` raises (post-hoc, no block).
- "Random-walk gap 1 − μ₂(D⁻¹A)" (§2) vs "1 − |μ₂|" (X1) vs the library's 1 − max(|μ₂|, |μ_min|): `features_of`
  records `rw_gap` = 1 − μ₂, `rw_abs_gap` = the library definition, and μ₂, μ_min.
- Feature cache name: `features_<seed>.json` for the default config at N = 10⁴; other configs get a short config hash
  suffix so different N or rules never collide.
- People Pleaser / Strong-Arm as RoundContext policies: allocation rules only (coalition-first media then workers;
  split at 1.5× pace); their manifestos are helpers (`people_pleaser_manifesto` uses the TRUE anticipated means, so it
  is an ORACLE stand-in; the observation-only players belong to `x_players.py` in Phase 3).
- `Probe` takes the probed region as a parameter; choosing it from the τ hypotheses is X17's player.
- Parity "statistical tolerance": |mean_gpu − mean_cpu| ≤ max(0.01, 3·SE of the paired differences), per metric.

### Suggested commit message

```
v3 Phase 0: shared experiment infrastructure, batched true-world runner, X23 conversion-rule knob

- experiments/x_common.py: stats (holm, power_paired_binary), seed blocks, JSONL rows, override,
  GPU device selection, memory-based batch sizer, OOM back-off, nvidia-smi monitor, provenance,
  CPU/GPU parity check, hidden-world features (GPU lobpcg / SciPy)
- experiments/x_batch.py: B true electorates per call with per-world manifesto, policy and seed
  streams; bit-identical to simulate_world on CPU, discrete-identical on GPU
- experiments/x_policies.py, x_streams.py (SaltedStreams), plot_v3.py (scaffolding),
  x_gpu_benchmarks.py -> reports/v3/gpu_benchmarks.json
- WorkerConfig.conversion_rule (gamma | gamma_p | gamma_p2, default gamma_p, bit-identical; D2)
- tests/unit/test_x_common.py (22 tests); full suite 285 passed
- reports/v3: report.md skeleton, predictions.md (pre-registered 2026-11-10), handoff_owner.md
```

Files to stage: `experiments/x_common.py experiments/x_batch.py experiments/x_policies.py experiments/x_streams.py
experiments/plot_v3.py experiments/x_gpu_benchmarks.py tests/unit/test_x_common.py src/dumbocracy/config.py
src/dumbocracy/dynamics/campaign.py src/dumbocracy/strategies/worlds.py reports/v3/` (and, if wanted,
`docs/v2/plans/implementation_plan.md`, which is still untracked).

## Phase 1 checkpoint (D7)

Phase 1 is complete (X0, X10, X7, X23; results in `reports/v3/report.md` §4). The options with the numbers follow.
Separately, the owner is asked to approve a one-line library bug fix: `BayesPlanner(device="cuda")` crashes in
`strategies/players.py` `manifesto()`, because a CPU tensor is stacked with GPU tensors. The fix adds `device=v_t.device`.
Phase 1 used an experiment-side workaround. X5, X6, X14, X16, X17 and X21 need it.

New Phase 1 files: `experiments/x0_balance.py`, `x10_n_t_balance.py`, `x7_prices.py`, `x23_worker_rules.py`;
`reports/v3/{x0,x10,x7,x23}.json`, `raw/*.jsonl`, `logs/*.log`, `figures/x0_*|x10_*|x7_*|x23_*.png`, `sections/*.md`.
Suggested commit message: "v3 Phase 1: X0 balance gate, X10 N×T grid, X7 price sensitivity, X23 worker-rule
ablations (scripts, results, figures, report sections)".

## Phase 1 owner checkpoint (D7): how to respond to X10

**For the owner. Nothing below is implemented.** Prices stay at ₫20 / ₫10 (D5). Under D7, Phase 2 waits for this choice.

Every number cites `reports/v3/x10.json` (50 seeds per cell, treasury ₫500) or `reports/v3/x0.json` (100 seeds, N = 10⁴, T = 5).

### What X10 found (the facts the decision rests on)

1. **No (N, T) cell passes the gate** (`gate_map`), and neither does the current default (X0, all treasuries).
   - The default fails criteria 4a and 4b: split wins 88–95% and media-only is never the best pattern (`x0.json` → `gate_by_treasury`).
2. **T is the main driver of failure, not N:**
   - **T = 1 fails criterion 2 at every N.** The ORACLE planner wins only 0.12 (10⁶) to 0.36 (10³) (`cells.<N|1>.summary.oracle/planner`).
   - **T = 10 fails criterion 1 at every N.** Uniform/none wins 0.38–0.44, and every paid pattern wins 0.94–1.00 (`cells.<N|10>.summary`).
   - **T ∈ {3, 5} come closest.** These cells miss a single criterion:
     - (3·10³, 3) and (10⁴, 3) miss criterion 2 (oracle/planner 0.78 and 0.74);
     - (10⁵, 5) misses criterion 4a (the best paid pattern wins exactly 0.80);
     - (10⁶, 5) misses criterion 2 (oracle/planner 0.80).
3. **The value of a worker scales with N in a way that depends on T.** Log–log slope of Δshare per worker vs N (`slopes.dshare_per_worker`):
   - −0.990 [−0.999, −0.982] at T = 1;
   - −0.751 [−0.786, −0.719] at T = 3;
   - −0.237 [−0.275, −0.205] at T = 5;
   - +0.016 [0.008, 0.023] at T = 10.

   The value of media per ₫ is flat in N at every T: all slope CIs contain 0 (`slopes.dshare_per_rupee_media`).
4. **The dominant channel flips across the grid** (`channel_dominance`):
   - **Workers dominate** at N = 10³ for every T (98–100% of seeds), and at T = 10 for every N (100%).
   - **Media dominates** at T = 1 for N ≥ 3·10³, and at T = 3 for N ≥ 10⁴ (0–30% of seeds favour workers).

### Option A — Scale the worker price with N (e.g. ₫20·N/10⁴)

- **What it fixes:** the 1/N decline of the worker's value, which holds only when the cascade is hop-limited (T = 1, slope −0.99).
- **What the numbers say against it:**
  - At T = 10 a worker's value does not depend on N (slope +0.016). There, ₫20·N/10⁴ would cost ₫2,000 at N = 10⁶, more than the whole ₫500 treasury, although a ₫20 worker already gives the same Δshare per worker at 10⁶ (5.38e-3) as at 10³ (4.84e-3) (`cells.<N|10>.channels.dshare_per_worker`).
  - At T = 5 the fitted price exponent would be about 0.24, not 1.
  - A correct formula would need a T-dependent exponent, roughly price ∝ N^{−slope(T)}, fitted from the four slopes above.
  - It also leaves the T-driven failures untouched: criterion 2 at T = 1 and criterion 1 at T = 10.
- **Cost:** a rule change, with a default-preserving test, and X0 rerun before Phase 2 (§3 checkpoint).

### Option B — Restrict the allowed N and/or T in the UI

- **What it fixes:** the structural failures at the ends of the T range.
  - T = 1: criterion 2 fails in 5 of 5 cells.
  - T = 10: criterion 1 fails in 5 of 5 cells.
- **What the numbers support:**
  - Restrict T to the middle of the range. Only T ∈ {3, 5} were measured; T = 2, 4 and 6–9 were not.
  - Optionally restrict N to [3·10³, 10⁵]. The four single-criterion cells all lie in N ∈ [3·10³, 10⁶] with T ∈ {3, 5}.
- **What it does not fix:** even inside that window no cell passes all five criteria. At the default (10⁴, 5), 4a and 4b still fail at ₫20 (X0). A restriction narrows the failure but does not remove it.
- **Cost:** a UI/API rule change (`api/app.py` bounds), with a default-preserving test.

### Option C — Document the limits ("balanced only near N = 10⁴, T = 5")

- **What it fixes:** nothing in the game. The report states the measured regimes:
  - media wins short games on large electorates;
  - workers win long games and small electorates;
  - T = 1 and T = 10 fail the gate structurally.
- **What the numbers say:** the wording "balanced near N = 10⁴, T = 5" would be inaccurate. The default cell fails 4a and 4b, both in X0 and in X10's (10⁴, 5) cell. The accurate claim is: "closest to balanced for T ∈ {3, 5}; no tested (N, T) passes all five §12 criteria at ₫20."
- **Cost:** none. Phase 2 starts directly.

### Notes for the decision

- The options can be combined. For example, B (restrict T to about 3–5) plus C (document the N-dependence), without a price formula.
- X7 (price and budget grid) runs only at N = 10⁴. It cannot test an N-scaled price unless it is extended.
- X11 (worker × media complementarity) remains the planned explanation for split dominating at the default cell.

## Owner decisions at the Phase 1 checkpoint (2026-11-11) and resulting changes

- **D11 / R31: the web game allows T = 3–5 rounds** (options B+C). The engine and experiments still accept 1–10.
  - Changed: `src/dumbocracy/api/app.py` (`MIN_ROUNDS, MAX_ROUNDS = 3, 5`), `web/lib/constants.ts`, `web/lib/mockApi.ts`,
    `docs/v2/web_api_v2.md`, `docs/v2/decisions_v2.md` (R31), `docs/v2/plans/implementation_plan.md` (D11, D12 rows).
  - Tests moved from 1–2 rounds to 3, plus new 422 checks for T ∈ {1, 2, 6, 10}: `tests/integration/test_api.py`,
    `tests/integration/test_api_epistemic.py`, `web/tests/{screens,epistemic,mockApi,graph_fetch}.test.ts(x)`.
- **D12: BayesPlanner GPU fix.** In `src/dumbocracy/strategies/players.py`, `manifesto()` now creates its extra start
  tensors on `v_t.device`. Regression test: `tests/integration/test_players.py::test_bayes_planner_runs_on_gpu`
  (marked `gpu`).
- Suites after these changes: Python 286 passed; vitest 59 passed; `tsc` clean.
- Suggested commit message: "R31/D11: restrict web game to 3–5 rounds after X10; D12: BayesPlanner GPU device fix".

## Phases 2–5 and 6 (X9): run, merged, ready to commit (2026-11-11)

All 23 scheduled experiments (X0–X23) have run with the planned seed counts. Each one has a JSON, raw JSONL, log,
figures and a section merged into `report.md` (status row, section, scorecard row). Every experiment passed its
CPU↔GPU parity check. GPUs used: cuda:0 and cuda:2 only, never GPU 1.

- Full test suite with GPUs: **305 passed** (`OMP_NUM_THREADS=4 python -m pytest tests -q`).
- All 55 figures redraw from JSON with one command: `python experiments/plot_v3.py` (it now dispatches to the
  per-phase plot modules).
- Number check: `python experiments/x_report_check.py` → `reports/v3/report_check.txt` (Phase 8 brings it to zero
  unmatched; see the report's §14).

### Approved library changes made in these phases (default behaviour bit-identical, each with a test)
| Change | Files | Test |
|---|---|---|
| Planner `rollout="front"` (X14, D2) | `src/dumbocracy/strategies/planner.py` | `tests/unit/test_planner_front_rollout.py` |
| `policy_set(max_policies=None)` (X16, D2); a local counter in `report.py` renamed to `largest_plan_set` | `src/dumbocracy/logic/epistemic.py`, `src/dumbocracy/logic/report.py` | `tests/unit/test_policy_set_max_policies.py` |

No other `src/` or `web/` file was changed after the Phase 1 checkpoint.

### New files
- Experiment scripts: `experiments/x{1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,18,19,20,21,22}_*.py`.
- Helpers: `experiments/x_parity.py`, `x_players.py`, `x_phase5a.py`, `plot_v3_phase4a.py`, `plot_v3_x8_x15.py`,
  `x_report_check.py`; `experiments/plot_v3.py` gained the Phase 2/3 blocks and the Phase 4/5 adapters.
- Results: `reports/v3/x<k>.json`, `reports/v3/raw/x<k>.jsonl`, `reports/v3/raw/features_<seed>.json` (200 cached
  world features used by X15), `reports/v3/logs/x<k>.log`, `reports/v3/figures/x<k>_*.png`,
  `reports/v3/sections/x<k>.md`, `reports/v3/report_check.txt`.
- **Size:** `reports/v3/` is 79 MB, of which `raw/` is 69 MB (largest file `raw/x3.jsonl`, 25 MB). All files are under
  GitHub's 100 MB limit. If you prefer a lighter commit, the raw JSONL files can be left out (or put in Git LFS); the
  summaries, figures and report do not need them, but X8 and X15 re-runs read `raw/x6.jsonl`, `raw/x0.jsonl` and
  `raw/x10.jsonl`.

### Findings worth the owner's attention
- **The ORACLE planner is a heuristic, not an upper bound** (X13): an open-loop plan search beats it by +0.044 E on
  100/100 seeds with media-heavy, front-loaded plans. Every "ORACLE wins" number is a lower bound on what is
  achievable, and BayesPlanner beats the ORACLE (X6).
- **The vote threshold z dominates balance** (X21, X22); the worker fee at ±30% is at the noise floor (agrees with X7).
- **Most of the outcome is the world draw** (X20: 97–99% of variance), and losses concentrate where the two largest
  regions want different things (X15).
- **Epistemic verdicts cannot see region personalities** (X17): the poll-consistency check's 0.01 slack is wider
  than the poll differences personalities make. "Guaranteed" was never falsely asserted (X16), but gets rarer with
  more sampled worlds.
- Deviations were recorded per experiment (JSON `deviations` key and section) and are collected in the report's §14.
  The most visible: Phase 3, X8 and X16 ran on cuda:0 instead of the plan's cuda:2. X14 used a one-factor-at-a-time
  design. X21 and X22 re-parameterised the z perturbation on the margin 1 − z because z × 1.5 > 1.

### Suggested commit message
```
v3 experiments Phases 2–6: X1–X5, X6, X8, X9, X11–X22 run, merged into reports/v3/report.md

- mechanics (X1–X4, X11, X18), scale (X9), strategies (X6, X12–X14, X19),
  information/logic (X5, X8, X16, X17), robustness (X15, X20–X22)
- approved library changes, defaults bit-identical: planner rollout="front" (X14),
  policy_set(max_policies) (X16), each with a default-preservation test
- plot_v3.py redraws all figures; x_report_check.py checks every cited number
```

## Phase 8: report consolidated (2026-11-11) — needs the owner's review

- `reports/v3/report.md` §1, §2, §3, §11–§15 written. Number check: 5,301 numbers, 0 unmatched, 13 documented
  derivations (each marked "derived" on its line and listed in `report_check.txt` for recomputation).
- Text corrections made against the JSON (old → new): X1 +0.089 → +0.088; X6 ORACLE media ₫169 → ₫168; X12 branching
  5.50 → 5.49; X5 Holm p 0.085 → 0.084; two unsourced numbers removed (X12 "about 660 worlds", X24 "8–12 people").
  The section drafts in `reports/v3/sections/` were synced to the same values.
- **Please review (plan §14.6):** §2 Summary and player advice, §3.1 model summary (written from the v2 report and
  `x0.json` config), §13 Limitations.
- Known caveats:
  - The number check proves a value exists in the cited JSON, not that it comes from the right key. §2 and §12 were
    verified by key; older section numbers only had the weaker check.
  - The X21 command in the report has `--threads 4`, which `x21.json` `provenance.command` lacks.
  - Scorecard: of 22 experiments with a prediction, 4 confirmed, 1 refuted, 17 partly confirmed.

## Paper draft and consensus figures (2026-11-11)

- New: `reports/v3/paper/report.md`, a paper-style summary of the suite. The full record stays in `reports/v3/report.md`.
- New: `experiments/x_consensus.py` → `reports/v3/x_consensus.json`, `logs/x_consensus.log`,
  `figures/xc_consensus_rounds.png`, `figures/xc_consensus_example.png`. It is a descriptive run on the X0 seeds
  (10000–10049), cuda:2, 36 s, showing the electorate moving towards b round by round. It is registered in
  `plot_v3.py`, which now redraws 57 figures.
