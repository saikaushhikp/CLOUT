# Experiment suite v3 — implementation plan

Status: **FINAL for evaluation, 2026-11-10.** The owner's decisions are recorded in D1–D9 below. Nothing is
implemented yet. One decision is deliberately left until results exist: how to respond to X10 (D7).

This plan merges two sources into one plan that can be built:
- the ideation in [`P9_experiments_ideation.md`](P9_experiments_ideation.md), experiments X0–X9;
- the follow-up review of the code and reports, which added experiments X10–X25 and fixes to X0–X9.

P9 stays as the record of the original ideas and is not edited. Implementers read **this** file. Where the two
disagree, this file wins.

Rules under test: spec v2.1 plus the post-v2.1 changes R16–R30, summarised in P9's header:
- workers are bought for ₫20, permanent, and pinned at b;
- the IC cascade moves one hop per round, with chance γ·P and at most 3 attempts per voter;
- a survey query costs ₫10;
- the treasury is ₫500–₫1000.

## Working rules for implementers (owner, 2026-11-10; binding)

These apply to everyone who implements this plan, including AI agents and sub-agents.

1. **Git is allowed, but nothing that creates a commit.**
   - **Allowed:**
     - read-only commands, e.g. `git status`, `diff`, `log`, `show`, `ls-files`, `rev-parse`, `blame`;
     - `git mv` (staging only), for moves this plan asks for;
     - scripts that call git read-only (e.g. `p7_common.provenance`).
   - **Not allowed:** anything that creates, changes or publishes commits or history: `git commit` (including
     `--amend`), `merge`, `pull`, `rebase`, `cherry-pick`, `revert`, `am`, `reset` to another commit, `tag`, `push`.
   - Also not allowed: switching branches (`checkout`/`switch`) and discarding work (`restore`, `checkout -- <file>`,
     `reset --hard`, `clean`, `stash drop`), because these can lose the owner's uncommitted changes.
   - **The owner makes every commit.** When work is ready, the implementer writes a suggested commit message and the
     list of changed files in `reports/v3/handoff_owner.md`, then stops.
2. **No deletions.** No existing file or directory is deleted, emptied or overwritten wholesale. This includes
   `git rm`. New files are added; existing files change only through the edits listed in this plan (§9) or after
   the owner approves them.
3. **Moves.** The implementer may move files with `git mv` where this plan asks for it (e.g. the v2 reports to
   `reports/archive/v2.1/`). Each move is recorded in `reports/v3/handoff_owner.md`, and the owner commits it.
4. **The revision is looked up read-only.** Scripts record the current revision with `git rev-parse` plus a
   `+uncommitted` flag from `git status`, as the existing `p7_common.provenance` already does.

## Owner decisions (2026-11-10; binding)

| ID | Topic | Decision | Effect on this plan |
|---|---|---|---|
| D1 | Scope | **All experiments except X25** | X0–X23 are scheduled, including the optional X17 and X22. X24 is dropped (D8). X25 stays deferred. |
| D2 | Library changes | **Approved:** conversion-rule knob (X23), front-loaded rollout (X14), plan-set size knob (X16). **Not approved:** LLM player. Two-player engine: not scheduled (D1). | §9 lists status per change. The LLM arm is removed. |
| D3 | Seeds | **100 per core cell, 50 per sweep cell** | Every seed count below follows this. Detectable paired win-rate difference ≈ 8–10 points (checked in `power.json`). |
| D4 | Results report | **Separate:** `reports/v3/report.md` is the v3 results document; `reports/v2/final_report.md` stays untouched as the v2 record | §14 |
| D5 | X0 outcome | If the gate fails at ₫20 (e.g. "split" wins ≥ 80%), **keep ₫20 and document it.** The failed criterion is reported honestly, and X11 explains the dominance. | Prices are fixed for the whole suite: ₫20 workers, ₫10 queries. |
| D6 | Treasury | Tested at ₫500 / 750 / 1000 (R17 range) | X0, X7 |
| D7 | X10 outcome | **Decide after results.** Phase 1 stops at the owner checkpoint; the choice between scaling the worker price with N, restricting N/T, or documenting the limits is made with the numbers in hand. | Phase 2 waits for this decision. |
| D8 | Human playtest | **Skipped.** X24 is not run; humans are listed under future work in `report.md`. | X24 removed from the schedule. |
| D9 | Git and files | Git allowed, nothing committed; no deletions (Working rules above) | All phases |
| D11 | X10 response (D7, 2026-11-11) | **Restrict T to 3–5 in the API/UI and document the limits** (options B+C). N stays free; prices unchanged; default (10⁴, 5) unchanged, so X0 is not rerun | R31; `api/app.py`, `web/lib/constants.ts`; report §4 |
| D12 | BayesPlanner GPU bug (2026-11-11) | **Approved:** `strategies/players.py` `manifesto()` builds its extra start tensors on `v_t.device` (was CPU) | Phases 3–5 run BayesPlanner fully on the GPU |
| D10 | Compute | **GPU first, approved.** Experiments and tests run their heavy work on the GPU: batched tensors for parallel computation over worlds, plans and voters, and the available GPU memory used deliberately. CPU is used only where the GPU is measured slower or the work cannot run there. | §1.1 (binding rules), §11 item 9 |

---

## 0. Contents

| ID | Name | Source | Priority | Phase | Library change? |
|---|---|---|---|---|---|
| X0 | Balance gate at the new prices | P9 | Core | 1 | no |
| X10 | Balance across the player-chosen N and T | new (N1) | **Core** | 1 | no |
| X7 | Budget and price sensitivity | P9 | Recommended | 1 | no |
| X23 | Worker-rule ablations (cap, min degree, conversion rule) | new (N14) | Recommended | 1 | small, optional |
| X1 | Network structure and consensus speed | P9 (fixed) | Core | 2 | no |
| X2 | Stubborn agents, zealots, W-MSR (+ polarisation metric) | P9 + N15 | Core | 2 | no |
| X3 | Worker cascade threshold and reach | P9 (fixed) | Core | 2 | no |
| X4 | Worker timing and placement | P9 | Recommended | 2 | no |
| X11 | Worker × media interaction (complementarity) | new (N2) | **Core** | 2 | no |
| X18 | Submodularity of the worker value curve | new (N9) | Recommended | 2 | no |
| X6 | Strategy tournament | P9 (fixed) | Core | 3 | no |
| X12 | Broad vs narrow ("viral") manifesto | new (N3) | **Core** | 3 | no |
| X13 | ORACLE optimality: open-loop plan search | new (N4) | **Core** | 3 | no |
| X14 | Planner calibration (prediction vs reality) | new (N5) | **Core** | 3 | small (rollout option) |
| X19 | Manifesto geometry and a third optimiser | new (N10) | Recommended | 3 | additive only |
| X17 | Learning personalities by probing | new (N8) | Optional | 4 | no |
| X5 | Value of information at ₫10 | P9 (fixed) | Core | 4 | no |
| X16 | Reliability and calibration of epistemic verdicts | new (N7) | Recommended | 4 | small (Π size knob) |
| X8 | Temporal properties of campaigns | P9 | Recommended | 4 | no |
| X15 | Which elections are unwinnable (post-hoc) | new (N6) | **Core** | 5 | no |
| X20 | Luck vs skill: variance decomposition | new (N11) | Recommended | 5 | no |
| X21 | Robustness to model misspecification | new (N12) | Recommended | 5 | no |
| X22 | Global sensitivity analysis of the design | new (N13) | Optional | 5 | no |
| X9 | Scale and GPU performance | P9 | Core (short) | 6 | no |
| X24 | Human playtest (+ optional LLM player) | new (N16) | **Dropped (D8)** | — | — |
| X25 | Two-party meta-game and manifesto convergence | new (2-player) | **Deferred (D1)** | — | **large**, not scheduled |

Library-change column: the three "small" changes for X14, X16 and X23 are **approved (D2)**.

**Mapping from the review's labels:**
- N1→X10, N2→X11, N3→X12, N4→X13, N5→X14, N6→X15, N7→X16, N8→X17;
- N9→X18, N10→X19, N11→X20, N12→X21, N13→X22, N14→X23, N16→X24;
- N15 is folded into X2 as an extra metric, and the two-player idea is X25.

---

## 1. Common protocol (applies to every experiment)

It is inherited from P9 §0, with these additions.

### 1.1 Compute: GPU first (D10; binding for experiments and tests)

**Principle.** All heavy lifting runs on the GPU as batched torch tensor operations. This covers:
- electorate generation, social dynamics, cascade hops, media, voting and polls;
- planner rollouts and world sampling;
- manifesto optimisation.

The work is parallelised across **worlds × candidate plans × voters** on the GPU's threads. The CPU only
orchestrates, plus the few steps listed under "CPU exceptions" below.

**1. Device selection (shared machine).**
- The default device is `cuda:2`. Before launching, check free memory with `config.cuda_free_memory()`.
- If `cuda:2` lacks the memory the run needs, use the GPU with the most free memory (`resolve_device("auto")`) and
  record which one.
- Do not take over a GPU that others are using heavily. Falling back to CPU is allowed only when no GPU has room;
  it is recorded as a deviation in `report.md`.

**2. Parallelism by batching, not by Python loops.**
- **Worlds.** Simulate many worlds per call through the leading batch dimension the library already supports:
  - `[B, N, 5]` preferences;
  - `WorldBatch`, which joins B equal-sized worlds into one disjoint-union graph;
  - batched `spreader_edge_counts` / `ic_from_counts`.
- **Plans.** Score all C candidate plans in one batched call (`WorldBatch.step` with C plans and common random
  numbers). This covers the planner, the X13 plan search, the X16 policy sets and X18 CELF evaluations.
- **Sweeps.** Within a cell, stack seeds of equal N into one batch. Across cells, iterate in Python only over the
  things that change the graph size or the config.
- **Optimisers.** Run multi-starts batched as `[M, 5]`, as `optimise_manifesto` already does. The X19 mirror-descent
  and Frank–Wolfe implementations follow the same pattern.
- **No dense N × N tensors.** All graph work stays sparse: edge lists and CSR with `index_add_`, `bincount` and
  `gather`.

**3. Memory use.**
- **Size batches from free memory.** Estimate the bytes per world (≈ preferences + susceptibilities + 2E edge
  indices + per-plan state) with a short calibration step at start-up. Then choose the largest B (and C) that fits
  in about 70% of the free memory on the chosen GPU, leaving headroom for other users.
- **Adapt on out-of-memory.** On `torch.cuda.OutOfMemoryError`, halve the batch, call `torch.cuda.empty_cache()`,
  retry, and log the final batch size. A run never silently falls back to CPU.
- **Inference only.** Simulation runs under `torch.inference_mode()`. Autograd is enabled only inside the manifesto
  optimisers.
- **Precision.** float32 by default; float64 only for the parity checks.
- **Keep data on the GPU.** Tensors stay on the device between rounds. Avoid `.item()`, `.cpu()` and `.tolist()`
  inside inner loops (they force a sync). Move only the per-game summaries to the host, once per batch.
- **Generation.** For batch experiments, the electorate is drawn on the GPU (`generation="device"`), which is
  85× faster at 10⁶ (E7). The canonical CPU draw (R8) is used only where the experiment must reproduce a specific
  game exactly, and this is stated.

**4. Existing CPU defaults are switched to GPU where measured faster.**
- `BayesPlanner(device="cpu")` and the logic layer's world sampling (`consistent_worlds`, `epistemic_timeline`)
  default to CPU, because small 1,600-voter worlds were faster there. Both already take a `device` argument, so
  this needs no library change.
- In Phase 0 the implementer benchmarks them on the GPU with **batched** worlds. They then use the GPU for X5, X6,
  X14, X16, X17 and X21 unless it is measured slower. The benchmark is recorded in `reports/v3/gpu_benchmarks.json`.
- Spectral quantities use GPU power iteration (`analysis/spectral.largest_eigenpair`; the X3 weighted operator) and,
  for λ₂ / μ₂, a GPU sparse eigensolver (`torch.lobpcg` or deflated power iteration). SciPy on CPU stays as the
  reference and fallback, and its use is recorded.

**5. CPU exceptions (allowed, recorded).**
- Python-level orchestration.
- SciPy fallbacks.
- The X15 regression and tree fitting.
- Report and figure generation.
- The parity reference runs.

Process parallelism (`p7_common.run_tasks`) is used for CPU-bound parts only. At most 2 processes share one GPU, and
more only if a benchmark shows it is faster. `OMP_NUM_THREADS=4` on the shared host.

**6. Measured and reported.** Every run records in its provenance block:
- the device name;
- the batch sizes B and C actually used;
- `torch.cuda.max_memory_allocated()` (peak memory) and the free memory at start;
- the wall time;
- the mean GPU utilisation, sampled with `nvidia-smi` every few seconds during the run when available.

`report.md` shows these per experiment in its status table.

**7. Correctness.**
- Each experiment does one small CPU↔GPU parity check: 3 seeds, compared with statistical tolerances as in spec §9.
  The result is logged in the provenance block.
- New tests carry the `gpu` pytest marker where they need CUDA, and run on `cuda:2` when it is free. They skip
  cleanly without CUDA.

### 1.2 Seeds
- Paired seeds: every arm of an experiment uses the same seed list.
- Seed blocks are disjoint per experiment, so results are not tuned on the same worlds. The table is in §1.5.
- **D3:** core cells use **100** seeds at N = 10,000; sweep cells (grids, ablation arms, parameter
  perturbations) use **50**. Any further reduction is a deviation and must be stated in `report.md`.
- With 100 paired seeds a McNemar test detects win-rate differences of roughly 8–10 points at 80% power. Smaller
  effects are reported through the paired difference in expected share E, which is far more sensitive.

**Seeds per experiment (binding; this table wins over any count in the text):**

| Experiment | Seeds | Kind |
|---|---|---|
| X0 balance gate | 100 per treasury | core |
| X10 N × T grid | 50 per (N, T) cell | sweep |
| X7 price grid | 50 per cell | sweep |
| X23 rule ablations | 50 per arm | sweep |
| X1 topology grid | 50 per (k̄, p_g) cell | sweep |
| X2 zealots / workers × rule | 50 per cell | sweep |
| X3 cascade threshold | 50 worlds × 9 manifesto points × 4 values of k, 10 cascade trials each | sweep |
| X4 timing / placement / pick rule | 50 per arm | sweep |
| X11 worker × media factorial | 100 (all four cells and the sequencing arms) | core |
| X18 worker value curve | 50 (CELF reference: 10) | sweep |
| X6 strategy tournament | 100 per player | core |
| X12 manifesto interpolation | 50 per λ point; 100 for the joint-optimisation arm | sweep / core |
| X13 open-loop plan search | 100 | core |
| X14 planner calibration | 50 per arm | sweep |
| X19 manifesto geometry | 100 | core |
| X5 value of information | 50 per (k, selection rule) cell | sweep |
| X17 probing | 50 | sweep |
| X16 verdict reliability | 50 per configuration, per player | sweep |
| X8 temporal | reuses X6 (100) | — |
| X15 hard-world analysis | reuses X0 + X6 + the X10 default cell | — |
| X20 luck vs skill | 50 worlds × 20 salts per stream | sweep |
| X21 misspecification | 50 per arm | sweep |
| X22 sensitivity analysis | 50 per design point; r = 10 Morris trajectories (≈ 110 points) | sweep |
| X9 scale | 1 canonical seed × 3 repeats | — |

### 1.3 Statistics
Reuse `experiments/p7_common.py`: `wilson`, `mean_ci`, `paired` (Wilcoxon), `mcnemar`, `spearman`,
`provenance`, `save`. Add to it:
- `holm(pvalues)`: Holm–Bonferroni correction. Every family of tests in one experiment is corrected, and the report
  states the family.
- `power_paired_binary(p_disc, delta, alpha, power)`: the seeds needed for a McNemar test. Run it once before
  Phase 3 with the discordance rates observed in Phase 1, and record the result in `reports/v3/power.json`.
- Effect sizes are reported for every test: the paired difference with its CI, and the discordant counts.
- **Pre-registration:** each experiment's theory prediction (the "Predict" line below) is copied into
  `reports/v3/predictions.md` *before* the run. The results section then says confirmed, refuted or partial.

### 1.4 Data flow and provenance
- Raw per-game rows → `reports/v3/raw/<id>.jsonl`, one JSON object per (seed, arm).
- Summary → `reports/v3/<id>.json`. Figures → `reports/v3/figures/<id>_*.png`, drawn **only** from the saved JSON.
- One results report, `reports/v3/report.md`. It is filled in experiment by experiment as each one finishes and
  consolidated in Phase 8. Its structure and rules are in §14.
- One command per experiment:
  `python experiments/x<k>_<name>.py --seeds … --first-seed … --device cuda:2 --out reports/v3/x<k>.json`.
- Every run records: config (`GameConfig.to_record()`), prices, seed list, the git revision (read-only lookup, with an
  uncommitted-changes flag; see Working rules), device, torch/CUDA
  versions, and wall time.
- Old v2 reports move to `reports/archive/v2.1/` (owner said yes in P9). The implementer moves them with `git mv`
  and lists them in `reports/v3/handoff_owner.md`. **The owner** commits the move. Nothing is deleted.

### 1.5 Seed blocks

| Experiments | First seed |
|---|---|
| X0, X7, X10, X23 | 10000 |
| X1, X2, X3, X4, X11, X18 | 11000 |
| X6, X12, X13, X14, X19 | 12000 |
| X5, X8, X16, X17 | 13000 |
| X20, X21, X22 | 14000 |
| X9 | 20261107 (canonical) |

X15 is post-hoc and uses the seeds of the experiments it reads.

---

## 2. Shared infrastructure (Phase 0)

All of it lives in `experiments/`. **No library code changes in Phase 0.**

| File | Contents |
|---|---|
| `experiments/x_common.py` | Imports `p7_common`, including its `provenance` (read-only git lookup, allowed by the Working rules). Adds `holm`, `power_paired_binary` and `seed_block(exp_id, n)`. `write_row(path, row)` appends JSONL. `override(config, "a.b=v")` is moved from `balance.py` and re-exported. `parity_check(fn, seeds)` runs on CPU and GPU. `features_of(electorate)` returns the hidden-world features for X15 (below). |
| `experiments/x_policies.py` | All scripted `RoundContext` policies in one place: those from `balance.py` plus the new ones (People Pleaser, Strong-Arm, front/even/late timing, region-focus, probe). Each policy is a small pure function or a class with explicit parameters. |
| `experiments/x_streams.py` | `SaltedStreams(root, salts: dict[str, int])`: an object with the same `generator(stream, *keys, device=)` / `derive` interface as `SeedStreams`. It adds a per-stream salt so that **one** random stream can be varied while all others stay fixed (X20). It is passed to `simulate_world(..., seeds=...)`, so no library change is needed. |
| `experiments/plot_v3.py` | Draws every figure from `reports/v3/*.json`. |

**Hidden-world features** (`features_of`, used by X15 and as covariates elsewhere):
- region sizes and their Gini coefficient;
- σ and τ;
- the TV distance between the archetype means of the two largest regions, and the minimum and mean pairwise
  TV distance over the four regions;
- the personality of the largest region;
- the realised zealot count per region;
- λ₂(L), the random-walk gap 1 − μ₂(D⁻¹A), and λ_max(A_rr) per region (`analysis/spectral.py`);
- the census inter-region edge share.

Spectra are computed on the GPU (power iteration / `torch.lobpcg`, §1.1 point 4), with SciPy on CPU as the reference and
fallback. Each world's features are cached in `reports/v3/raw/features_<seed>.json`.

**Tests for the infrastructure** (`tests/unit/test_x_common.py`; the test suite is the only place outside `experiments/`
that gets a new file):
- `holm` matches a hand-computed example;
- with all salts at 0, `SaltedStreams` gives exactly the same generators as `SeedStreams`;
- `features_of` is deterministic per seed.

---

## 3. Phase 1 — Calibration (run first; gates everything else)

Prices are fixed at ₫20 workers and ₫10 queries (D5). The allowed (N, T) range is settled at the owner checkpoint
after Phase 1 (D7). **Phases 2–6 start only after that checkpoint.**

### X0 — Balance gate at the new prices *(Core; P9, unchanged in substance)*
- **How:** rerun the five §12 checks at worker ₫20, query ₫10, treasury ∈ {₫500, 750, 1000}.
  - Policies: none, workers spread evenly, workers front-loaded, media only, split, ORACLE planner.
  - Manifestos: uniform, top-3, informed, oracle.
- **Code:** `balance.py` already does this. It runs as
  `python experiments/balance.py --seeds 100 --first-seed 10000 --device cuda:2 --set budget=750.00 …`. A thin
  wrapper `x0_balance.py` loops over the three treasuries and writes `reports/v3/x0.json`.
- **Predict:** at ₫20, "split" wins ≥ 80% (gate 4a fails), as the P8 price check found at ₫10–₫50.
- **If the gate fails (D5):** ₫20 is kept. `report.md` states which criteria failed, and X11 tests whether worker–media
  complementarity explains the dominant pattern. No price retuning.
- **Output:** pass/fail per criterion per treasury; best-pattern counts per seed; win rates with Wilson CIs.
- **Cost:** low (≈ 5 min per treasury on GPU, from the 256 s gate run).

### X10 — Balance across the player-chosen N and T *(Core; new)*
- **Question:** the API lets the player choose N ∈ [10³, 10⁶] and T ∈ [1, 10]
  (`api/app.py:38–40`), but balance was only ever checked at N = 10⁴, T = 5. Is the game balanced over the whole
  choice range?
- **Why:**
  - Media price is N-independent: it depends on the region's share of voters (m0·n_r/(N/4)).
  - A ₫20 worker's cascade reaches a roughly fixed *number* of voters, since R0 depends on k̄·γP and not on N.
    Its effect on vote *share* should therefore fall like 1/N.
  - Survey precision also changes with N: the 10% sample is about 15–40 voters per region at N = 10³ (MoE ±15–25
    pp) and nearly exact at 10⁶.
  - The cascade depth equals T, and social drift grows with T.
- **How:**
  - Grid N ∈ {10³, 3·10³, 10⁴, 10⁵, 10⁶} × T ∈ {1, 3, 5, 10} at ₫500. Seeds: 50 per cell (D3). If wall time
    forces fewer at N = 10⁶ (minimum 25), this is a recorded deviation.
  - Run the X0 policy × manifesto set in every cell.
  - Also record converts per worker (absolute) and Δshare per ₫ spent on each channel.
- **Code:** `balance.py --n <N> --set rounds=<T>`, wrapped by `x10_n_t_balance.py`. N = 10⁶ uses
  `generation="device"` for speed; this is stated in the output.
- **Predict:**
  - Δshare per worker ∝ 1/N (log–log slope ≈ −1), while media's Δshare per ₫ is flat in N.
  - Workers dominate at N = 10³ (the "one worker converts most voters" problem) and are near-worthless at 10⁶.
  - T = 1 favours media, because a cascade gets only 1 hop.
- **Output:** the gate pass/fail map over (N, T); a channel-dominance heat map; slopes with CIs.
- **Decision it feeds (D7, made after the results):** scale the worker price with N (e.g. ₫20·N/10⁴), restrict
  the allowed N or T in the UI, or document "balanced only near N = 10⁴, T = 5".
  - The implementer presents the three options with the X10 numbers in `reports/v3/handoff_owner.md` and stops.
  - A price formula or a UI restriction would be a rule change. It is only implemented after the owner chooses it.
- **Cost:** medium (the 10⁶ cells dominate; ≈ 1–2 GPU-h total, an estimate).

### X7 — Budget and price sensitivity *(Recommended; P9)*
- **How:** grid treasury {500, 750, 1000} × worker fee {₫10, 20, 50, 100} × query fee {₫5, 10}. Players: the ORACLE
  planner and the Balanced player (BayesPlanner). Record the win rate, the spend mix, and the dominant pattern.
  Seeds: 50 per cell (D3). This is a sensitivity study only: prices stay at ₫20 / ₫10 (D5).
- **Code:** `balance.py --set` for the ORACLE arms; `strategies.py` with a config override for BayesPlanner. This
  needs `--set` support in `strategies.py`; add it in the X7 wrapper (experiment-side, by building the config
  before the games start).
- **Fix carried from the review:** BayesPlanner's `survey_budget` defaults to ₫50 and its `min_gain_per_coin`
  docstring assumes ₫5 queries. In every query-fee arm, set `survey_budget` explicitly and record it.
- **Predict:** money has diminishing returns; workers stop being worth it relative to media somewhere between
  ₫50 and ₫100 at N = 10⁴ (P8 table).
- **Cost:** medium.

### X23 — Worker-rule ablations *(Recommended; new)*
- **Question:** are the owner's rule choices R22 (min degree 3), R25 (chance γ·P) and R27 (cap 3) what make the game
  work, or would nearby rules do as well?
- **How:** run the X0 policies with the ORACLE manifesto (50 seeds per arm, D3), changing one rule at a time:
  - `workers.attempt_cap` ∈ {1, 2, 3, 5, 1000 (≈ ∞)};
  - `workers.min_degree` ∈ {1, 3, 10};
  - conversion rule ∈ {γ, γ·P (default), γ·P²}.

  Metrics: cascade reach per worker, the gate criteria, and how often the cap binds.
- **Code:**
  - The cap and min degree work through `--set` (no change).
  - The conversion rule needs an **optional** config field `WorkerConfig.conversion_rule: Literal["gamma",
    "gamma_p", "gamma_p2"] = "gamma_p"`, read in `dynamics/campaign.conversion_chance` and in the batched world
    step. Default behaviour is identical (unit test: the default reproduces the current results bit for bit).
    *Library change, **approved (D2)**.*
- **Predict:** cap ∞ and rule γ bring back cascade saturation (the manifesto stops mattering, as P8 saw with γ
  alone). A cap of 1 makes workers subcritical.
- **Cost:** low–medium.

**Phase 1 owner checkpoint (D7):**
- The implementer writes the X0, X7, X10 and X23 summaries into `report.md` and the X10 options into
  `reports/v3/handoff_owner.md`, then stops.
- The owner decides the response to X10. Prices stay at ₫20 / ₫10 regardless of X0 (D5).
- If the owner chooses a rule change (an N-scaled price or an N/T restriction), it is implemented with a
  default-preserving test, and X0 is rerun under it before Phase 2.
- Otherwise Phase 2 starts directly.

---

## 4. Phase 2 — Network and cascade mechanics (lecture theory)

### X1 — Network structure and consensus speed *(Core; P9, with a fix)*
- **How:** as in P9:
  - k̄ ∈ {10, 20, 30} × p_g range ∈ {[0.01, 0.05], [0.01, 0.2], [0.1, 0.4]};
  - λ₂(L), the random-walk gap 1 − |μ₂(D⁻¹A)| and the cross-region conductance;
  - consensus-only dynamics, then the standard strategies on the same worlds.
- **Fix:** "rounds until regional means move halfway to the global mean" is far beyond T = 5. E1 measured
  1–5% mixing per round. Measure it on **long social-only runs** (≥ 400 rounds, zealot-free copy, as E1 did) and
  report it separately from the 5-round game outcome.
- **Code:** port `e1_topology.py` to `x1_topology.py`; switch to the X policies and the v3 output paths.
- **Predict:** the decay follows ā(1 − μ₂(D⁻¹A)), not λ₂(L)/k̄ (as E1 found). Under the P8 rules, high p_g raises
  the wins of a single nationwide manifesto.
- **Cost:** medium (the spectra dominate; E1 took 50 min with 4 k̄ values).

### X2 — Stubborn agents, zealots, W-MSR *(Core; P9 + polarisation metric)*
- **How:** as in P9.
  - Arm A: zealots ζ ∈ {0, 5, 15, 30%}.
  - Arm B: exactly k workers, k ∈ {0, 5, 10, 20}, with the **cascade switched off** (pinned stubborn agents only).
  - Arm C: A and B under W-MSR with F ∈ {1, 2}.
  - Compare with the (r, s)-robustness bounds.
- **Cascade switch:** set `workers.attempt_cap=0`. Then `ic_from_counts` gives `eff = 0`, so nobody converts but
  workers stay pinned. No library change.
- **Friedkin–Johnsen prediction for Arm B:**
  - Solve the stationary linear system for the mean v with pinned nodes, by sparse iterative solve
    (`scipy.sparse.linalg.cg`/`spsolve` on the reduced Laplacian).
  - Compare the predicted long-run share with the simulated share at T = 5 and at T = 200.
  - Each normal voter's final opinion is predicted as the random-walk absorption-weighted mix of the stubborn
    opinions.
- **Added metric (from review N15) — persistent disagreement / polarisation**, computed at election day with and
  without workers/zealots:
  - between-region TV dispersion of the regional means;
  - within-region mean TV distance to the region mean;
  - the fraction of voters with P > 0.9 or P < 0.1 (bimodality of support).
- **Predict:**
  - The worker anchor value is a sizeable share of the worker value: Arm B Δshare / full-cascade Δshare is
    between 0.2 and 0.6, to be measured.
  - Pinned agents stop consensus (disagreement does not decay to 0).
  - W-MSR trims workers, because they sit far from their friends, so it **lowers** the worker value. That
    makes W-MSR the network's defence against *your* campaign.
- **Cost:** medium.

### X3 — Worker cascade threshold and reach *(Core; P9, with a fix)*
- **How:** as in P9:
  - k ∈ {1, 3, 5, 10} workers in one region;
  - manifesto swept from "far from the region" to "ideal" (interpolate b between a distant archetype mean and
    the region's own mean, 9 points);
  - converts per hop;
  - cross-region conversions, cap-binding rate, and the drift-back of converts after their turn.
- **Fix:** the spectral prediction must use the **weighted** operator. Let D_q = diag(q) with q_j = γ_j P_j(b).
  The threshold is λ_max(A D_q) = 1 (same spectrum as D_q^{1/2} A D_q^{1/2}, which is symmetric, so power
  iteration in `analysis/spectral.largest_eigenpair` can be adapted in the script). Also report:
  - the naive k̄·E[γP] and λ_max(A)·E[γP];
  - a cap-corrected branching estimate: the expected conversions per spreader given that each target accepts at
    most `cap − attempts_j` more attempts.
- **Code:** grow `cascade_threshold.py` into `x3_cascade.py` (GPU, batched over trials). The weighted power
  iteration is a local helper in the script.
- **Predict:** growth crosses 1 near λ_max(A D_q) ≈ 1, more sharply than the naive estimates; the cap binds
  mainly above threshold.
- **Cost:** low.

### X4 — Worker timing and placement *(Recommended; P9)*
- **How:**
  - At equal total worker spend, vary timing (all in round 1 / even / all in the last round) and placement
    (largest region / best-matched region / proportional).
  - Ablate the pick rule: random voter with ≥ 3 friends (the rule in force); highest degree in a 10% sample
    (the source document's rule); true highest degree (upper bound).
- **Code:** the timing and placement policies live in `x_policies.py`. The pick-rule ablation builds a
  `WorkerState` directly in the script, replacing `pick_workers` for that arm only (experiment-side, outside
  the game rules, labelled).
- **Predict:** buying early is worth more (more hops); centrality picking gives a modest gain under k̄ = 30 ER
  blocks (low degree variance).
- **Cost:** medium.

### X11 — Worker × media interaction *(Core; new, N2)*
- **Question:** do the two channels reinforce each other (supermodularity)? This would explain why "split"
  dominates at cheap worker prices.
- **Mechanism to test:**
  - media in round t raises S and therefore P, which raises the round-t+1 conversion chance γP;
  - pinned workers raise their friends' P, which helps media convert them to votes.
- **How:** a 2×2 factorial at **matched spend**:
  - {workers 0 / k} × {media 0 / m}, with the ORACLE manifesto, 100 seeds;
  - the interaction is Δ(both) − Δ(workers) − Δ(media), with a paired CI;
  - a sequencing arm: media in round 1 with workers in round 2, vs the reverse, vs both in round 1;
  - repeat at worker fees ₫20 and ₫100.
- **Code:** policies in `x_policies.py`; `simulate_world`; no library change.
- **Predict:** the interaction is positive, and larger in canvassing-receptive regions; "media first, then
  workers" beats the reverse.
- **Output:** the interaction term with CI per fee and per personality of the targeted region.
- **Cost:** low.

### X18 — Submodularity of the worker value curve *(Recommended; new, N9)*
- **Question:** influence-maximisation guarantees (greedy ≥ (1 − 1/e)·OPT) need diminishing returns. Pinned
  stubborn workers plus a conversion chance that depends on b may break that.
- **How:**
  - Buy k workers in round 1 in one region, k = 1…40 (step 1 to 10, then 5). Measure the marginal Δshare of the
    k-th worker per region (50 seeds, 4 regions).
  - Test for monotone decreasing increments: the share of adjacent pairs with gain(k+1) > gain(k) beyond
    noise, using a paired bootstrap.
  - Run a node-level greedy (CELF, lazy evaluation) for 10 seeds × 1 region × k ≤ 10, using the true graph, as an
    IM reference bound for X4.
- **Code:** the experiment builds `WorkerState` directly (outside the game rules, labelled). CELF evaluates
  candidates in GPU batches with `WorldBatch`-style stacking.
- **Predict:** roughly submodular in k, with possible early *super*-modularity in regions near threshold.
- **Cost:** medium (CELF is the expensive part; keep it small).

---

## 5. Phase 3 — Strategies and optimisation

### X6 — Strategy tournament *(Core; P9, with a fix)*
- **How:** paired games for:
  - I People Pleaser, II Balanced (BayesPlanner), III Strong-Arm;
  - baselines uniform / top-3 / random / ORACLE;
  - manifesto optimisers: projected gradient, consensus-ADMM, "blend my hunches".

  Report the primal and dual residuals of ADMM. IV (opponent-aware) is X25.
- **Fix:** the ORACLE manifesto is tuned to the electorate after social drift only (`strategies/oracle.anticipate`).
  It ignores the campaign, even though converts and workers are set to b. So "the gap to the ORACLE" is a gap
  to a heuristic. X6 reports the gap to **both** the ORACLE and the X13 upper bound, and states that the ORACLE
  is a heuristic.
- **Code:**
  - People Pleaser and Strong-Arm are new observation-only players in `experiments/x_players.py` (not in the
    library). People Pleaser uses BayesPlanner's posterior and "blend my hunches" on the 1–2 largest regions
    with a majority. Strong-Arm uses a fixed or random b and the "split" pattern at 1.5× pace.
  - `strategies.py` is reused for the existing players.
- **Predict:** II > I > III; the price of not knowing is ≤ 0.015 E.
- **Cost:** high (engine games with the BayesPlanner).

### X12 — Broad vs narrow ("viral") manifesto *(Core; new, N3)*
- **Question:** because the conversion chance is γ·P (R25), a manifesto close to one region's preferences makes that
  region's cascade grow on its own, while a broad manifesto pleases many voters a little. Which wins, and should
  b be chosen together with the campaign plan?
- **How:**
  - For each seed, build the ORACLE manifesto b₀ and the coalition blends b_C for each 2- and 3-region coalition
    holding > 50% of voters (size-weighted mean of the true anticipated regional means).
  - Interpolate b(λ) = (1 − λ) b₀ + λ b_C for λ ∈ {0, 0.25, 0.5, 0.75, 1}.
  - Play each b(λ) with (a) the ORACLE planner and (b) X13's plan search.
  - Record: final E, wins, cascade reach per region, and the per-region λ_max(A D_q).
- **Joint optimisation arm (campaign-aware b):** alternate (i) fix the plan and optimise b on the *simulated
  campaign* end-state, by running the planned campaign in expected-value form (converts replaced by their
  expected fraction, which is differentiable in b through γP), and (ii) re-plan with the new b. Three rounds of
  alternation. This is an experiment-side optimiser; no library change.
- **Predict:** the best λ > 0 on seeds where some coalition region is canvassing-receptive. Joint optimisation
  gains a measurable +0.005–0.02 E over "b first, then plan" (to be measured).
- **Cost:** medium.

### X13 — ORACLE optimality: open-loop plan search *(Core; new, N4)*
- **Question:** how far is the greedy ORACLE planner from the best achievable plan? This gives a true upper
  bound for X6.
- **How:**
  - With full information (true world, ORACLE manifesto), search open-loop 5-round plans. A plan is
    hires[t, r] ∈ ℕ and media[t, r] ∈ {0, 1}, under the treasury.
  - Use the cross-entropy method: a population of 256 plans per iteration, an elite of 10%, and 30 iterations.
    Then a local 1-move hill climb.
  - All candidate plans are simulated in one batch with common random numbers. `WorldBatch.step` already takes
    C plans; use the true electorate as the single world.
  - Use 100 seeds. Re-evaluate the final plan on 20 fresh cascade/media streams to remove selection bias
    (winner's curse).
- **Predict:** the search beats the ORACLE planner by ≤ 0.01 E on average, with a long tail of seeds where front-loading
  or concentration beats greedy by more.
- **Output:** the distribution of (search − ORACLE) E; the plan features that explain the gap (timing, focus).
- **Cost:** medium–high (≈ 256 × 30 rollouts per seed, batched).

### X14 — Planner calibration *(Core; new, N5)*
- **Question:** BayesPlanner predicts on 8 posterior worlds of 1,600 voters and scales hires by n/N with
  stochastic rounding. Small graphs saturate cascades differently, and at large N the scaled hire count is
  almost always rounded to 0. Is its prediction calibrated, and how much does the approximation cost?
- **How:**
  - Record the planner's predicted `best_score` (already in `BayesPlanner.diagnostics`) and compare with the
    realised E of the true game. Use the same seeds and the same manifesto.
  - Vary `world_voters` ∈ {800, 1600, 4000, N_true}, `num_worlds` ∈ {4, 8, 16, 32}, and `rollout` ∈
    {"even", "none", "front"}.
  - Repeat at N ∈ {10⁴, 10⁵}. The N = 10⁵ arm checks the rounding-to-zero issue.
- **Code:** `world_voters`, `num_worlds` and `rollout` are existing BayesPlanner fields.
  - "front" is **new**: a continuation that spends the remaining worker money in the next round. It is a small
    optional branch in `strategies/planner.plan_round` / `default_allocation`, default unchanged. *Library
    change, **approved (D2)**.*
- **Metrics:** bias (predicted − realised), RMSE, calibration slope, win rate, decision latency.
- **Predict:** positive bias (over-optimism) that shrinks with world size; "even" undervalues front-loaded worker
  buys.
- **Cost:** medium (engine games × arms; 50 seeds per arm).

### X19 — Manifesto geometry and a third optimiser *(Recommended; new, N10)*
- **Question:**
  - Where does the optimal manifesto sit relative to simple summaries of the electorate (a median-voter
    analogue on the simplex)?
  - How many regions does it carry (a minimum-winning-coalition analogue)?
  - How do the optimisers compare?
- **How:** on the anticipated true electorates of 100 seeds:
  - compare b* with: the mean; the per-sector median (projected to Δ); the L1 geometric median (Weiszfeld-type
    iteration under TV); every 2-, 3- and 4-region blend. Report the TV distance and E for each;
  - count the regions carried (regional mean P > 0.5) and the coalition size;
  - count distinct local optima reached from 64 random Dirichlet starts (cluster the end points at TV < 0.01);
  - optimisers:
    - projected gradient (prox of the simplex indicator);
    - consensus-ADMM;
    - **mirror descent / exponentiated gradient** (the entropic prox, natural on the simplex);
    - **Frank–Wolfe**.

    Measure iterations to 1e-4 of the best E, wall time, and final E.
- **Code:** mirror descent and Frank–Wolfe are added as **new functions** in `experiments/x19_manifesto.py`. They
  may be promoted to `strategies/manifesto_opt.py` later (additive; existing functions untouched).
- **Predict:** b* is closer to a 2–3-region coalition blend than to the mean or median; there are several local
  optima (multi-start matters); mirror descent converges in fewer iterations than PG on interior optima.
- **Cost:** low.

---

## 6. Phase 4 — Information and logic

### X5 — Value of information at ₫10 *(Core; P9, with a fix)*
- **How:** as in P9:
  - k ∈ {0, 1, 2, 4, 8, 12} questions, chosen by information gain vs at random;
  - posterior accuracy, manifesto, win rate, net value per ₫, and the per-round epistemic verdicts;
  - compare with ₫5 questions.
- **Fix:** BayesPlanner's default `survey_budget = ₫50` caps it at 5 questions at ₫10, which silently truncates
  the 8 and 12 arms. Set `survey_budget = k × price` (and `num_queries = k`) per arm, and record both.
- **Code:** `strategies.py --info-curve 0,1,2,4,8,12` (existing) and `info_sufficiency.py`, with the budget fix
  and the v3 paths.
- **Predict:** about 2 questions are still enough; beyond that every question is a net loss, larger than at ₫5.
- **Cost:** medium–high (logic world sampling and timelines on the GPU with batched worlds, §1.1 point 4).

### X17 — Learning personalities by probing *(Optional; new, N8)*
- **Question:** surveys reveal nothing about personalities τ (R13), and polls are national. Can a deliberate probe
  make τ learnable, and is exploration worth its price?
- **How:**
  - A probe player = BayesPlanner, except that in round 1 it spends on **one** region only (media in r, or
    workers in r), chosen to maximise the expected split of the τ hypotheses in the world sample. It then reads
    the round-2 poll and plays the planner.
  - Compare with plain BayesPlanner on 50 seeds.
  - Measure:
    - the entropy over τ of the accepted worlds (`logic/worlds.consistent_worlds` gives the accepted τ values) at
      rounds 1–5;
    - K_p(Guaranteed) per round;
    - the final win rate.
- **Code:** `experiments/x_players.py: ProbePlayer` (observation-only); world acceptance from `logic.worlds`.
- **Predict:** probing lowers τ entropy measurably, but the 400-voter national poll (SE ≈ 2.4 pp) limits it.
  The net win-rate effect is ≈ 0 or negative.
- **Cost:** medium–high (logic on the GPU with batched worlds, §1.1 point 4).

### X16 — Reliability and calibration of epistemic verdicts *(Recommended; new, N7)*
- **Question:** the verdicts ("guaranteed", "possible", "knows that", "inevitable defeat") are evaluated over a
  *sample* of 16 worlds of 1,600 voters and 14–16 plans. How often are they wrong, and how do they depend on
  the sample?
- **How:** on BayesPlanner and Top-3 games (50 seeds each):
  - Vary `worlds` ∈ {8, 16, 32, 64}, `world_voters` ∈ {800, 1600, 4000}, and the plan-set size |Π| ∈ {8, 16,
    all candidates}.
  - Compare each verdict with the truth row. Use a long-run truth too: ∃π in the X13 plan search wins in the
    true world.
  - Report false-assert and missed-truth rates per verdict (soundness and completeness, within the declared Π).
  - **Graded version:** for each decision point, p̂ = the fraction of (world, best plan) pairs that win. Compute
    its Brier score and a reliability diagram against the realised outcome. This shows knowledge as calibrated
    belief.
- **Code:** `logic/report.timeline_with_diagnostics(state, pid, worlds=…, world_voters=…)` (existing parameters).
  The |Π| knob needs an optional `max_policies` argument in `logic/epistemic.policy_set` (default: current
  behaviour). *Small library change, **approved (D2)**.*
- **Predict:** "guaranteed" becomes rarer and more reliable as the world count grows; p̂ is over-confident at 16
  worlds.
- **Cost:** high. A timeline took ≈ 4–5 s at 16 worlds on CPU in v2, and the time grows with the world count.
  Under §1.1 the worlds of all decision points are batched on the GPU, so the 32- and 64-world arms are where the
  GPU should pay off most. The measured time is reported.

### X8 — Temporal properties of campaigns *(Recommended; P9)*
- **How:** on the X6 traces: ◇Lead, □(margin > −δ), Comeback = ¬Lead U (Lead ∧ ◇Victory), and STL robustness;
  CTL-style EF/AF Victory over the batched futures; the point-of-no-return round per strategy.
- **Code:** `logic/temporal.py` (existing). Reads `reports/v3/raw/x6.jsonl`.
- **Predict:** most winners come from behind (as v2 found); the point of no return is round 2–3 for weak strategies.
- **Cost:** low.

---

## 7. Phase 5 — Robustness and explanation

### X15 — Which elections are unwinnable? *(Core; new, N6; post-hoc, nearly free)*
- **How:**
  - Join the X0, X6 and X10 (N = 10⁴, T = 5) rows with `features_of(seed)`.
  - Fit (a) a logistic regression with standardised coefficients and CIs (bootstrap) and (b) a depth-3
    decision tree, predicting ORACLE loss and Bayes loss. Use 5-fold CV AUC.
  - Report partial-dependence plots for the top 3 features.
- **Predict:** losses concentrate where the two largest regions have distant archetypes and a fragmented network
  (low p_g); the largest region's personality decides the best channel.
- **Output:** a short "why you lost" taxonomy that can also feed the UI debrief later (out of scope here).
- **Cost:** negligible (CPU, minutes).

### X20 — Luck vs skill: variance decomposition *(Recommended; new, N11)*
- **Question:** how much of the outcome is decided by the player's decisions, and how much by randomness?
- **How:**
  - For 50 worlds × 3 fixed policies (ORACLE planner, split, none), resample one stream at a time with
    `SaltedStreams`: `cascade`, `media`, `election`, `workers` (which voters are picked), plus the world itself
    (graph and preferences).
  - Use 20 salts per stream. Decompose Var(E) and Var(win) into within-stream and between-world components
    (nested ANOVA).
  - **Skill share** = Var(policy means) / total variance.
- **Code:** `x_streams.SaltedStreams` passed to `simulate_world(..., seeds=…)`. No library change.
- **Predict:** world draw > worker picks > cascade > media ≈ election at N = 10⁴. Election-day noise is tiny
  (sd ≈ 0.005).
- **Cost:** low–medium.

### X21 — Robustness to model misspecification *(Recommended; new, N12)*
- **Question:** BayesPlanner assumes the public default parameters (`players.public_config` uses `GameConfig()`
  defaults, except N, T and the treasury). How badly does it do when the true world differs?
- **How:**
  - Change one true parameter at a time by ×0.5 and ×1.5: ζ, κ, k̄, ω, β_v, z, p_m.
  - The engine uses the perturbed config; the player keeps the defaults (as it does today).
  - Compare with the ORACLE in the same perturbed world. 50 seeds per arm.
  - **Check first:** confirm that none of the perturbed values leaks into the observation. `costs` does; ζ, κ
    etc. must not. Write this check as an assertion in the script.
- **Predict:** the planner is most sensitive to z and β_v (they define who votes); the least sensitive is κ.
- **Cost:** medium (engine games).

### X22 — Global sensitivity analysis of the design *(Optional; new, N13)*
- **Question:** which parameters is the game's balance most sensitive to? This supports the "synthetic
  calibration" limitation in the report.
- **How:**
  - Morris elementary effects: r = 10 trajectories over 10 factors (β_v, z, κ, ω, ζ, k̄, p_m, η_m0, λ, worker fee)
    in ±30% boxes.
  - Outputs: ORACLE win rate, best-channel share, and the "split" win rate, each over 50 seeds per point.
  - Optional Sobol indices with Saltelli sampling if time allows (more costly).
- **Code:** numpy Morris implementation in the script; SALib only if already installed (do not add a dependency
  without asking).
- **Cost:** medium–high (≈ 110 design points × 50 seeds, ORACLE-only on GPU).

---

## 8. Phase 6 — Scale; and experiments not scheduled

### X9 — Scale and GPU performance *(Core, short; P9)*
- **How:** time per round and peak VRAM at N ∈ {10⁴, 10⁵, 10⁶}, single world and batched (B ∈ {1, 8, 32}), on
  cuda:2 vs CPU. Check that the cascade hop stays O(edges).
- **Code:** port `e7_scale.py`; add the batched-B arm using `WorldBatch`.
- **Cost:** low.

### X24 — Human playtest (+ LLM player) *(**Dropped, D8**; LLM player not approved, D2)*
- **Not run.** `report.md` lists it under future work. The design is kept here so it can be picked up later:
  - 8–12 people each play 2 games on fixed seeds through the web UI, reading the debrief between games;
  - compare them with BayesPlanner and Top-3 on the same seeds;
  - test whether game 2 improves on game 1 (whether the epistemic debrief teaches).

### X25 — Two-party meta-game and manifesto convergence *(Deferred; new)*
- **Question:**
  - With two scripted parties, which strategy profile is a Nash equilibrium of the meta-game, and is Strong-Arm
    dominated?
  - Do two optimising parties converge to the same manifesto?
- **Required build (large):**
  - multi-party softmax voting with an abstain option (spec §5, specified but not built);
  - a cascade rule for voters converted by several parties in the same round (P8 multi-player note);
  - engine changes for two active players.

  `step_preferences` already accepts P manifestos.
- **How (once built):**
  - Empirical game-theoretic analysis: a payoff matrix of {I, II, III, Top-3} × {I, II, III, Top-3}, 100 seeds
    per cell, and its Nash equilibrium via replicator dynamics or support enumeration.
  - Iterated best-response on manifestos.
- **Status:** **deferred (D1).** Not scheduled; listed under future work in `report.md`.

---

## 9. Library changes (status after D2)

Each approved change keeps the current default behaviour bit for bit and adds a unit test proving it.

| Change | File | For | Status |
|---|---|---|---|
| `WorkerConfig.conversion_rule` ∈ {gamma, gamma_p, gamma_p2}, default gamma_p | `config.py`, `dynamics/campaign.py`, `strategies/worlds.py` | X23 | **approved** |
| `rollout="front"` continuation | `strategies/planner.py` | X14 | **approved** |
| `max_policies` in `policy_set` | `logic/epistemic.py` | X16 | **approved** |
| `LLMPlayer` implementation | `players/llm_stub.py` | X24 (LLM arm) | not approved; X24 dropped |
| softmax voting + two-player engine | `voting.py`, `game/engine.py`, `dynamics/campaign.py` | X25 | not scheduled (deferred) |
| N-scaled worker price or N/T restriction | `config.py` / `api/app.py` / `web/` | X10 response | decided after X10 (D7) |

Everything else is experiment-side code under `experiments/`, plus new test files (`tests/unit/test_x_common.py`
and one default-preservation test per approved change). **No existing file is deleted.** The v2 reports are moved
to `reports/archive/v2.1/` as approved in P9: the implementer moves them with `git mv`, and the owner commits
(Working rules).

---

## 10. Schedule, dependencies and ownership

Owners are suggested from the v2 ownership table (final report, Appendix C); the owner may reassign.

| Phase | Experiments | Depends on | Suggested owner | Rough compute (estimate) |
|---|---|---|---|---|
| 0 Infra | `x_common` (including the memory-based batch sizer and OOM back-off of §1.1), `x_policies`, `x_streams`, `plot_v3`; GPU benchmarks of BayesPlanner and logic world sampling → `reports/v3/gpu_benchmarks.json`; archive the v2 reports with `git mv` and list the moves in `reports/v3/handoff_owner.md` (the owner commits) | — | lead | — |
| 1 Calibration | X0, X10, X7, X23 | 0 | Hem | 1.5–2.5 GPU-h |
| ↳ **owner checkpoint (D7)** | response to X10 (prices stay ₫20 / ₫10, D5) | 1 | owner | — |
| 2 Mechanics | X1, X2, X3, X4, X11, X18 | checkpoint | Deek (X1, X2), Hem (X3, X4, X11, X18) | 1.5–3 GPU-h |
| 3 Strategies | X6, X12, X13, X14, X19 | checkpoint (X13 before the X6 write-up) | Mad | 3–5 h (engine + GPU) |
| 4 Information/logic | X5, X17, X16, X8 | 3 (X8 reads X6; X16 uses X13) | Psk/Kaushik | 4–7 h (CPU estimate from v2; expected lower on GPU) |
| 5 Robustness | X15, X20, X21, X22 | 1, 3 | Deek (X20, X22), Mad (X21), lead (X15) | 2–4 h |
| 6 Scale | X9 | 0 | Deek | < 0.5 GPU-h |
| 1–6 Report (incremental) | the matching section of `reports/v3/report.md` is written as each experiment finishes | each experiment | that experiment's owner | — |
| 8 Report | consolidate `reports/v3/report.md` (§14): summary, prediction scorecard, v2 comparison, check of every cited number | all | lead | — |

The compute figures are extrapolated from the P7 wall times (E1 50 min, E2 10 min, E3 6 min, gate 256 s, BayesPlanner
≈ 3.3 s per game), scaled to the D3 seed counts. They are planning estimates only, on a shared machine.

Critical path: Phase 0 → X0/X10 → owner checkpoint → X13 → X6 → X8/X16 → report.

---

## 11. Definition of done (per experiment)

1. A single command reproduces `reports/v3/<id>.json` from seeds. The command is written at the top of the script.
2. Raw JSONL, summary JSON, figures (drawn from JSON only), and the provenance block are all present.
3. The CPU/GPU parity check passed and was logged.
4. The prediction in `predictions.md` is marked confirmed / refuted / partial, with numbers.
5. Statistics include CIs, effect sizes and Holm-corrected p-values for the family.
6. The full Python test suite still passes (`python -m pytest -q`), including any new default-preserving tests.
7. The experiment's section in `reports/v3/report.md` is written to the §14.3 template, and every number in it
   names the JSON file it comes from.
8. The Working rules were followed: nothing was committed, nothing was deleted, and the changed files, moves and a
   suggested commit message are written in `reports/v3/handoff_owner.md` for the owner to commit.
9. GPU first (D10, §1.1): the heavy work ran batched on the GPU; batch sizes, peak memory, utilisation and wall time
   are recorded; any CPU use is one of the §1.1 exceptions or a recorded deviation.

---

## 12. Lecture coverage after this plan

| Lecture | Experiments |
|---|---|
| Linear algebra / spectral | X1, X3 (weighted operator), X2 (Friedkin–Johnsen solve), X9 |
| MAS algorithms: consensus, W-MSR | X1, X2 |
| Infection models / IC | X3, X4, X11, X18, X23 |
| Influence maximisation | X4, X18 |
| Proximal operator | X6, X19 (PG, mirror descent, Frank–Wolfe) |
| Distributed learning / ADMM | X6, X19 |
| Epistemic logic | X5, X16, X17 |
| Temporal logic | X8 |
| Game theory | X25 (deferred; future work) |
| Design / calibration | X0, X7, X10, X15, X20, X21, X22 |

## 13. Owner decisions: status

All decisions are recorded in D1–D9 at the top. **One remains open by design:** the response to X10 (D7), taken at
the Phase 1 checkpoint with the results in hand.

Any later change of scope, seeds or approvals is added as a new D-row, with the date, rather than by editing the
existing rows.

---

## 14. The results report: `reports/v3/report.md`

### 14.1 Purpose and readers
- **What it is:** the single document that shows the results of the experiments in this plan.
- **Who reads it:** the course evaluators and the team.
- **What a reader should get from it:** for every experiment, what was asked, what theory predicted, what was
  measured, and what it means for the game and for a player. A reader should not need the JSON files to follow
  it, but every number points to one.

### 14.2 When and how it is written
- **Started in Phase 0** as a skeleton: every heading below, and every experiment marked **Not run yet**.
- **Filled in as each experiment finishes:** its section is written as soon as it meets the definition of done
  (§11), by the experiment's owner.
- **Consolidated in Phase 8** by the lead: the summary, the prediction scorecard, the comparison with v2, and the
  number check (§14.5).
- **Status block:** the report always shows which experiments are done, partial, skipped or not run, and why.

### 14.3 Structure

1. **Title and status block:**
   - git revision (with the uncommitted-changes flag), date, prices (worker, query, treasury), and the allowed (N, T) range after the Phase 1 decision;
   - a status table: ID | name | status (done / partial / skipped / not run) | seeds | device | batch size | peak GPU memory | mean GPU
     utilisation | wall time | data file.
2. **Summary:** at most 10 bullets of headline findings, each with its experiment ID. Then one paragraph of
   advice for a player.
3. **Setup:**
   - the model in one page (rules R16–R30, with links to the spec and the P8 plan, not repeated);
   - the common protocol (§1): seeds, statistics, and the definition of the ORACLE and of the X13 upper bound.
4. **Calibration and balance:** X0, X10, X7, X23, plus the owner decisions taken at the Phase 1 checkpoint and the
   evidence behind them.
5. **Network and cascade mechanics vs theory:** X1, X2 (with the polarisation metric), X3, X4, X11, X18.
6. **Strategies and optimisation:** X6, X12, X13, X14, X19.
7. **Information and logic:** X5, X17, X16, X8.
8. **Robustness and explanation:** X15, X20, X21, X22.
9. **Scale:** X9.
10. **Not run and future work:** X24 (human playtest and LLM player, dropped under D8/D2) and X25 (two-party
    game, deferred under D1). For each: a short description and why it was not run. No results.
11. **Prediction scorecard:** one table for all experiments: ID | prediction (copied unchanged from
    `predictions.md`) | key measured number with CI | verdict (confirmed / refuted / partial / not tested) |
    data file.
12. **What changed since v2:** each v2 result (E1–E7, balance gate) next to its v3 counterpart (E1→X1, E2→X2,
    E3→X3, E4→X5, E5→X7, E6→X6, E7→X9, gate→X0), with the rule change that explains any difference.
13. **Limitations and threats to validity:** sampled worlds and plans, the heuristic ORACLE, synthetic
    calibration, the shared hardware, CPU/GPU randomness, and every deviation from this plan.
14. **Deviations from the plan:** changed grids, reduced seeds, skipped arms, and library changes refused or
    approved, each with its reason.
15. **How to reproduce:** the single command per experiment, and `python experiments/plot_v3.py`.

**Template for each experiment section (sections 4–10):**

```markdown
### X<k> — <name>
**Question.** One or two sentences.
**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md.
**Setup.** Seeds (first seed, count), N, T, treasury, prices, arms, device, wall time.
  Deviations from the plan, if any.
**Results.** One table with the key numbers and their 95% CIs, the paired tests (Holm-corrected p, effect
  size, discordant counts), and the figure(s): ![…](figures/x<k>_….png).
**Theory vs observation.** Confirmed / refuted / partial, and why, with numbers.
**What it means.** For the game design (e.g. a price or rule change) and for a player (advice).
**Data.** reports/v3/x<k>.json, reports/v3/raw/x<k>.jsonl.
```

### 14.4 Writing rules
- **No number without a source.** Every number names the JSON file it comes from, as in the v2 final report.
- **Nothing estimated or made up.** An experiment that was not run says **Not run yet** (or **Skipped**) with
  the reason, and shows no numbers. A partial run says which arms are missing.
- **Report failures honestly.** A failed gate criterion, a refuted prediction or a negative result is reported
  as plainly as a success, together with the smallest fix suggested.
- **Uncertainty is always shown:** 95% CIs (Wilson for rates, t for means) and effect sizes, not only p-values.
- **Figures** come from `plot_v3.py` reading the saved JSON only, and are stored in `reports/v3/figures/`.
- **Plain language:** short sentences, and every symbol defined at first use or linked to the spec.

### 14.5 Number check (Phase 8)
- **Script:** `experiments/x_report_check.py` reads `report.md`. For each number next to a cited JSON file, it
  checks that the value (after rounding) appears in that file.
- **Output:** a list of unmatched numbers in `reports/v3/report_check.txt`. The target is zero unmatched,
  apart from documented rounding.
- **Precedent:** the v2 final report was checked by hand on a sample (96/98 numbers matched exactly, with
  2 rounding differences); this makes the check complete and repeatable.

### 14.6 Done when
- Every experiment in §0 has a section with a final status.
- The scorecard covers every prediction in `predictions.md`.
- `x_report_check.py` reports no unmatched numbers.
- The figures regenerate from JSON with one command.
- The owner has reviewed the summary and the limitations.
