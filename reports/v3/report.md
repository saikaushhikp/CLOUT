# Dumbocracy v3: experiment results

Results document of the v3 experiment suite (plan: [`docs/v2/plans/implementation_plan.md`](../../docs/v2/plans/implementation_plan.md), §14).
The v2 record stays in [`reports/v2/final_report.md`](../v2/final_report.md) (D4).

## 1. Status

- **Git revision:** `d8d9c4e+uncommitted` for every run (each JSON's `provenance` block). The owner commits; nothing in v3 is committed yet.
- **Date:** 2026-11-10 (Phase 0); 2026-11-11 (Phases 1–6: all 23 experiments; Phase 8: this consolidation).
- **Status:** 23 experiments done (X0–X23), with the planned seed counts and no seed reductions; X24 dropped (D8); X25 deferred (D1).
- **Prices:** worker ₫20, survey query ₫10 (D5); treasury ₫500 (default), ₫500 / 750 / 1000 tested in X0 and X7 (D6).
- **Allowed (N, T) range:** N ∈ [10³, 10⁶]; web game T ∈ [3, 5], engine and experiments T ∈ [1, 10] (R31 / D11, after X10).
- **Compute summary:** every experiment ran on one NVIDIA RTX A5000 (cuda:2 or cuda:0, never GPU 1), except X15, a post-hoc CPU fit. The
  wall times in the table add up to about 7.6 h (derived: sum of the Wall time column). The longest runs were X5
  (74.7 min), X16 (64.2 min) and X12 (60.3 min) (x5.json, x16.json, x12.json → `provenance.compute`). Every experiment passed
  its CPU↔GPU parity check (each JSON's `provenance.parity`).
- **Compute:** GPU first (D10, plan §1.1). Phase 0 benchmarks on an RTX A5000 (cuda:2), all from [`gpu_benchmarks.json`](gpu_benchmarks.json):
  - batched true-world runner `x_batch.run_batch`, policy `pay_split`: 3 ms per game at N = 10⁴ (B = 64) vs 42 ms for CPU `simulate_world` (×12.9); 7 ms per game at N = 10⁵ (B = 32–64) vs 382 ms (×53); peak 6.9 GiB above the inputs at N = 10⁵, B = 64 (`x_batch[*].policies.pay_split.batched`). With the ORACLE planner policy the Python decision code dominates: ×3.9 at 10⁴, ×22 at 10⁵ (`…planner.batched`);
  - parity: 20/20 worlds bit-identical to `simulate_world` on the CPU; 20/20 with identical discrete outcomes on the GPU, floats within 6·10⁻⁸ (`parity`);
  - BayesPlanner: the rollout planning step is 5–29× faster on the GPU, but the whole decision (world sampling, posterior, surveys) only 1.2–1.4× (`bayes_planner`); logic timeline: GPU 1.2–1.3× faster (`logic`); spectra: SciPy faster at 10⁴, GPU `lobpcg` faster at 10⁵ (`spectra`).

| ID | Name | Status | Seeds | Device | Batch size | Peak GPU memory | Mean GPU util. | Wall time | Data file |
|---|---|---|---|---|---|---|---|---|---|
| X0 | Balance gate at the new prices | done | 100 per treasury (10000–10099) | cuda:2 | 100 | 1.74 GB | 19% | 289 s | [x0.json](x0.json) |
| X10 | Balance across the player-chosen N and T | done | 50 per (N, T) cell (10000…) | cuda:2 | 50 (N ≤ 10⁵), 8 (10⁶) | 14.2 GB | 34% | 1,058 s | [x10.json](x10.json) |
| X7 | Budget and price sensitivity | done | 50 per cell (10000…) | cuda:0 (deviation) | ORACLE batched; BayesPlanner 2 procs | 80 MB per proc | 6% | 16.5 min | [x7.json](x7.json) |
| X23 | Worker-rule ablations | done | 50 per arm (10000…) | cuda:0 (deviation) | 50 | 0.95 GB | 12% | 104 s | [x23.json](x23.json) |
| X1 | Network structure and consensus speed | done | 50 per cell (11000…) | cuda:2 | 50 | 1.2 GB | 15% | 21.6 min | [x1.json](x1.json) |
| X2 | Stubborn agents, zealots, W-MSR | done | 50 per cell (11000…) | cuda:2 | 50 | 3.5 GB | 60% | 286 s | [x2.json](x2.json) |
| X3 | Worker cascade threshold and reach | done | 50 worlds × 9 × 4, 10 trials + 8 extension points (11000…) | cuda:2 | 500 | 6.3 GB | 60% | 156 s | [x3.json](x3.json) |
| X4 | Worker timing and placement | done | 50 per arm (11000…) | cuda:2 | 50 | 0.9 GB | 45% | 48 s | [x4.json](x4.json) |
| X11 | Worker × media interaction | done | 100 (11000…) | cuda:2 | 400 | 5.2 GB | 38% | 76 s | [x11.json](x11.json) |
| X18 | Submodularity of the worker value curve | done | 50; CELF 10 (11000…) | cuda:2 | 200 / 1024 (CELF) | 12.5 GB | 48% | 9.6 min | [x18.json](x18.json) |
| X6 | Strategy tournament | done | 100 per player (12000…) | cuda:0 | ORACLE 100; Bayes 8 worlds | 0.07 GB | 31% | 8.9 min | [x6.json](x6.json) |
| X12 | Broad vs narrow ("viral") manifesto | done | 50 per λ; 100 joint (12000…) | cuda:0 | manifestos × 20 streams per seed, chunked | 2.8 GB | 95% | 60.3 min | [x12.json](x12.json) |
| X13 | ORACLE optimality: open-loop plan search | done | 100 (12000…) | cuda:0 | C = 256 plans × K = 4 | 0.3 GB | 92% | 12.0 min | [x13.json](x13.json) |
| X14 | Planner calibration | done | 50 per arm, 18 arms (12000…) | cuda:0 | 4–32 worlds × 800–N voters | 0.07 GB | 9% | 30.6 min | [x14.json](x14.json) |
| X19 | Manifesto geometry and a third optimiser | done | 100 (12000…) | cuda:0 | 8 starts; 64 Dirichlet starts | 0.3 GB | 21% | 9.7 min | [x19.json](x19.json) |
| X5 | Value of information at ₫10 | done | 50 per cell, 22 arms (13000…) | cuda:2 | 1 game; 8 planner + 16 timeline worlds per decision | 0.13 GB | 17% | 74.7 min | [x5.json](x5.json) |
| X17 | Learning personalities by probing | done | 50 per arm, 3 arms (13000…) | cuda:2 | 512 candidate worlds per game | 1.6 GB | 1% | 42.2 min | [x17.json](x17.json) |
| X16 | Reliability and calibration of epistemic verdicts | done | 50 per player, 36 cells (13000…) | cuda:0 | worlds × up to 21 plans per decision; search 256 × 4 | 2.0 GB | 46% | 64.2 min | [x16.json](x16.json) |
| X8 | Temporal properties of campaigns | done | reuses X6 (1,000 games) + 64 futures per (game, round) | cuda:0 | 160 (calibrated 640, OOM back-off) | 3.9 GB | 37% | 23.9 min | [x8.json](x8.json) |
| X15 | Which elections are unwinnable? | done | reuses X0 + X6 + X10 (200 worlds) | CPU | — | — | — | 131 s | [x15.json](x15.json) |
| X20 | Luck vs skill: variance decomposition | done | 50 worlds × 3 policies × 8 streams × 21 draws (14000…) | cuda:2 | 250 (in-game) / 100 (world draws) | 3.4 GB | 27% | 793 s | [x20.json](x20.json) |
| X21 | Robustness to model misspecification | done | 50 per arm, 16 arms (14000…) | cuda:2 | ORACLE 50; Bayes 4 threads | 1.4 GB | 24% | 1,132 s | [x21.json](x21.json) |
| X22 | Global sensitivity analysis of the design | done | 50 per point, r = 10, 110 points (14000…) | cuda:2 | 250 | 4.1 GB | 32% | 743 s | [x22.json](x22.json) |
| X9 | Scale and GPU performance | done | 1 canonical × 3 repeats (20261107) | cuda:2 | 1 / 8 / 16 / 32 | 17.9 GB (B = 16 at 10⁶) | 7% | 2.1 min | [x9.json](x9.json) |
| XC | Consensus towards the manifesto (descriptive) | done | 50 (10000–10049) | cuda:2 | 1 world × 5 campaigns | — | — | 36 s | [x_consensus.json](x_consensus.json) |
| X24 | Human playtest (+ LLM player) | dropped (D8) | — | — | — | — | — | — | — |
| X25 | Two-party meta-game and manifesto convergence | deferred (D1) | — | — | — | — | — | — | — |

## 2. Summary

Symbols: E is the expected vote share on election day (the mean over voters of each voter's chance of voting for
the party); a win needs more than half the votes. N is the number of voters, T the number of rounds. "ORACLE" is the
full-information scripted baseline (§3.2).

1. **The balance gate fails at the new prices (X0).** With workers at ₫20 and queries at ₫10, the fixed "split"
   pattern (half workers, half media) wins 0.88 [0.80, 0.93] at ₫500 and ₫750 and 0.95 [0.89, 0.98] at ₫1000, so
   criteria 4a and 4b fail at every treasury (x0.json). The owner kept the prices and documented the failure (D5).
   Re-pricing does not fix it: neither ₫10 nor ₫100 workers make 4a pass (X7, x7.json).
2. **No (N, T) choice is balanced (X10).** No cell of the N × T grid passes all five criteria. T = 1 is too short
   (the ORACLE planner wins only 12–36%) and T = 10 too easy (no campaign at all wins 38–44%) at every N (x10.json).
   The web game is therefore limited to T = 3–5 rounds (R31, D11).
3. **The ORACLE planner is a heuristic, not an optimum (X13).** An open-loop plan search beats it by +0.044 E
   [0.043, 0.046] on 100/100 seeds (x13.json). Better plans spend about 89% on media (ORACLE 34%) and buy few
   workers (2.8 vs 16.2), 99% of them in round 1. Every "ORACLE win rate" in this report is a lower bound on what
   a skilled campaign can reach.
4. **Knowing beats guessing, and planning beats the ORACLE (X6).** The ordering II > I > III holds: the
   information-gathering BayesPlanner (II) reaches E 0.639 [0.625, 0.654] and wins 0.98; the People Pleaser (I) 0.560
   and 0.73; the fixed-agenda Strong-Arm (III) 0.088 and 0.00 (x6.json). The BayesPlanner beats the heuristic
   ORACLE by 0.011 E. Measured against the X13 bound, the price of not knowing is 0.033 [0.030, 0.035] E, above the
   predicted 0.015.
5. **Most of the outcome is the world you are dealt (X20).** The world draw explains 99.4% of the variance of the
   vote share for the ORACLE planner and 97.2% for "split"; all in-game randomness together is small (x20.json).
6. **The vote threshold z dominates the design (X21, X22).** z is the hidden parameter the BayesPlanner is most
   sensitive to (gap change 0.034, next k̄ at 0.014, x21.json), and the one that moves the ORACLE's win rate most
   (Morris μ* 0.88 [0.61, 1.16], x22.json). The worker fee (±30%) is at the noise floor.
7. **Cascades have a sharp threshold, worker value saturates, and timing beats placement (X3, X18, X4).** Cascade
   growth crosses 1 at λ_max(A D_q) ≈ 0.94, the spectral radius of the friendship matrix weighted by the
   conversion chances (x3.json). One worker already gives 62% of the plateau and five give 93% (x18.json). Buying
   in round 1 instead of round 5 is worth +0.12 E; where the workers go changes E by at most 0.010 (x4.json).
8. **About two questions are enough, and probing for personalities does not pay (X5, X17).** At ₫10, twelve
   questions score −0.0132 [−0.0176, −0.0087] E against two (x5.json). Spending round 1 on a test airing to learn a
   region's personality τ changes E by −0.0127 [−0.0195, −0.0059]; by round 5 it leaves no less uncertainty about τ than not probing (x17.json).
9. **"Guaranteed" verdicts are safe but get rarer with more worlds (X16).** The debrief never asserted a false
   "guaranteed" in any of the 36 cells. With 8 → 64 sampled worlds the share of decision points with a
   "guaranteed" verdict falls from 0.772 to 0.616 (x16.json). The 0.01 slack of the poll-consistency check is wider
   than the poll differences that personalities cause, so the verdicts are blind to τ (X17).
10. **Unwinnable worlds have a divided majority (X15).** Over 200 worlds at ₫500, the ORACLE loses 11/100
    [0.06, 0.19] where the two largest regions' archetypes are at total-variation (TV) distance ≥ 0.35, and 0/100
    below that (Fisher p 0.0004, x15.json).

**Advice for a player.** Optimise the manifesto for a coalition of two or three regions instead of blending every
region's wishes (X15, X19). Ask about two survey questions, then stop (X5); do not spend a round probing
personalities (X17). Spend the treasury early: air media in most regions in rounds 1–2 and buy a handful of
workers in round 1 (X13, X4); workers bought late are nearly worthless. Combine the two channels; if you must pick
one, pick workers over media (X0). Trust a "guaranteed" verdict, but treat "not guaranteed" as weak evidence:
most such positions were still winnable (X16). And a loss is weak evidence of bad play: the electorate decides
most of the outcome before the first move (X20).

## 3. Setup

### 3.1 The model

Rules under test: spec v2.1 plus the post-v2.1 changes R16–R31 ([`technical_spec_v2.md`](../../docs/v2/technical_spec_v2.md), [`decisions_v2.md`](../../docs/v2/decisions_v2.md), and the party-worker rules R21–R28 in [`P8_party_workers_ic.md`](../../docs/v2/plans/P8_party_workers_ic.md)). The v2 final report (§3) gives the full equations; this is the one-page version. Default values are from x0.json → `config`.

- **Electorate.** N voters (default 10,000) in 4 regions, each region at least 15% of the voters. Each voter holds a
  preference v_i on the simplex of 5 policy sectors. A hidden map σ gives each region one of 6 public archetypes;
  voters start at Dirichlet(κ·archetype) with κ = 60. A hidden map τ gives each region one of 4 personalities
  (socially driven, media-savvy, independent, receptive to canvassing), which sets each voter's susceptibility to
  friends (α), media (β) and canvassing (γ). A fraction ζ = 0.05 of voters are zealots who never move.
- **Network.** A block-structured random friendship graph with mean degree k̄ = 30: more friends inside a region
  than across regions.
- **Social dynamics.** Every round each voter moves towards the average of her friends (DeGroot, step ω = 0.15
  times α_i); W-MSR, which drops the most extreme neighbours, is an option studied in X2.
- **Voting.** Voter i's agreement with the manifesto b is S_i = 1 − ½‖v_i − b‖₁. She votes for the party with
  chance P_i = sigmoid(β_v (S_i − z)), with β_v = 20 and threshold z = 0.825. E is the mean of P_i; the official
  result is a seeded vote, won with more than half the votes.
- **The player's game.** Commit a manifesto b, then play T rounds (default 5; the web game allows 3–5, R31) with a
  treasury of ₫500 (₫500–₫1000 allowed, R17). Each round the player may buy:
  - **workers** at ₫20 (R30): a random voter of the chosen region with at least 3 friends becomes a permanent
    worker pinned at b (R21–R23);
  - **media airings** per region, priced by region size (₫50 for a quarter-sized region), with diminishing
    returns for repeated airings;
  - **survey queries** at ₫10 (R30): the share of a regional sample preferring one sector over another.
- **Cascade (Independent Cascade, R24–R27).** Workers bought in round t try to convert each friend once in that
  round; a friend j converts with chance γ_j·P_j and is set to v_j = b; converts spread one hop further in the next
  round. Each voter can be targeted at most 3 times per game; workers and zealots are never targets.
- **Information.** Public: N, T, the rules and prices, region sizes, the archetype and personality libraries, and
  a network census. Free each round: a top-3 sector statistic and a 400-voter national poll. Paid: the queries.
  σ, τ, the graph and the voters' preferences stay hidden.

### 3.2 Common protocol

- **Seeds:** paired across arms; disjoint blocks per experiment group (plan §1.5); 100 per core cell and 50 per sweep cell at N = 10,000 (D3; e.g. x0.json `seeds.count` 100, x4.json `seeds.count` 50).
- **Statistics:** Wilson 95% CIs for rates, t-based 95% CIs for means, Wilcoxon signed-rank for paired differences, exact McNemar for paired wins, Holm–Bonferroni within each experiment's family of tests (`experiments/x_common.py`).
- **Compute:** batched on the GPU (`experiments/x_batch.py`: B true worlds per call, each with its own manifesto, policy and seed streams; bit-identical to `simulate_world` on the CPU, discrete outcomes identical on the GPU — see `gpu_benchmarks.json` → `parity`).
- **ORACLE:** the labelled full-information baselines (`balance.policy_planner`, `strategies/oracle.oracle_manifesto`); the ORACLE manifesto is tuned to the social-only anticipated electorate and is a heuristic, not an optimum.
- **X13 upper bound:** the open-loop plan search of X13 (cross-entropy method + hill climb, re-evaluated on 20 fresh streams). It reaches a mean E of 0.672 on the X6 seeds, against 0.628 for the ORACLE (x13.json, x6.json → `x13_bound`). The search is itself a heuristic, so the bound is a lower bound on the true optimum.

## 4. Calibration and balance

Owner decisions taken on the Phase 1 evidence:
- **D5 (prices):** keep ₫20 workers and ₫10 queries although the gate fails criteria 4a and 4b (X0). X7 shows that
  no worker price from ₫10 to ₫100 makes 4a pass, and X22 that the fee barely matters (x0.json, x7.json, x22.json).
- **D11 / R31 (rounds, the D7 checkpoint):** restrict the web game to T = 3–5 and document the limits (X10: T = 1 and
  T = 10 fail at every N). N stays free; prices and the default (N = 10⁴, T = 5) are unchanged, so X0 was not rerun.
- **D12 (BayesPlanner on the GPU):** approve the one-line device fix found in Phase 0; X6 onwards ran the
  BayesPlanner fully on the GPU.

### X0 — Balance gate at the new prices

**Status:** done. The gate **fails at all three treasuries** on criteria 4a and 4b. Under D5 the ₫20 price stays.

**Question.** At the D5 prices (worker ₫20, query ₫10), do the five §12 balance-gate criteria hold at treasuries ₫500, ₫750 and ₫1000?

**Prediction (pre-registered).** Copied unchanged from `reports/v3/predictions.md`:
> - **Predict:** at ₫20, "split" wins ≥ 80% (gate 4a fails), as the P8 price check found at ₫10–₫50.

**Setup.**
- **Seeds:** 100 seeds (10000–10099, `seed_block("X0", 100)`). Every arm and every treasury uses the same seeds (paired).
- **Game:** N = 10⁴, T = 5, treasury ∈ {₫500, ₫750, ₫1000}, worker ₫20, query ₫10. Electorates use the canonical CPU draw (R8, `generation="cpu"`).
- **Arms:** exactly the runs of `balance.run`, with manifesto/policy names as in `balance.py`. There are 11 runs per seed:
  - uniform/none, uniform/planner, top3/planner, informed/planner;
  - oracle/none, oracle/planner;
  - informed/{free_even, pay_workers, front_workers, pay_media, pay_split}.
- **Gate logic:** taken from `balance.run` (`x0_balance.gate_of`).
- **Engine:** every game runs through `x_batch.run_batch` on the GPU.
- **Equivalence with `balance.py`:** a 4-seed smoke run gave identical mean expected shares, win rates, gate and best-pattern counts.
- **Informed manifesto:** the library's `synthetic_world(…, known_assignment=True, salt=1)` is drawn on the CPU, then `oracle_manifesto` runs on the GPU. This agrees with the CPU library path to about 1e-7 per coordinate in the pilot.
- **Compute:** device cuda:2 (NVIDIA RTX A5000), one batch of B = 100 worlds per run. Values from `reports/v3/x0.json` → `provenance.compute`:
  - peak memory 1.74 GB;
  - mean GPU utilisation 19% (max 100%);
  - wall time 289 s, of which 159 s were games and manifestos (`provenance.games_seconds`).
- **Parity (§1.1.7), passed, 3 seeds (`provenance.parity`):**
  - On the CPU, `run_batch` and `simulate_world` were bit-identical on 3/3 seeds.
  - CPU vs GPU, statistical check: mean difference in expected share +0.0038 for planner (tolerance 0.0101) and −0.0012 for split (tolerance 0.01); identical victories.
- **Zealot sweep (not a gate criterion):** oracle manifesto + ORACLE planner at ₫500 on the first 40 seeds. The seed count is reduced from 100; the sweep is informational only.
- **Deviations:** none from the plan's X0 design.
  - The run was launched from `experiments/`, so `provenance.command` shows `../reports/v3/x0.json`.
  - Canonical command: `python experiments/x0_balance.py --seeds 100 --first-seed 10000 --device cuda:2 --treasuries 500 750 1000 --out reports/v3/x0.json`.

**Results.** Win rates with 95% Wilson CIs, from `reports/v3/x0.json` → `per_treasury.<t>.summary`:

| run | ₫500 | ₫750 | ₫1000 |
|---|---|---|---|
| uniform/none | 0.07 [0.034, 0.137] | 0.07 [0.034, 0.137] | 0.07 [0.034, 0.137] |
| uniform/planner | 0.83 [0.745, 0.891] | 0.88 [0.802, 0.930] | 0.90 [0.826, 0.945] |
| top3/planner | 0.95 [0.888, 0.978] | 0.98 [0.930, 0.994] | 1.00 [0.963, 1.000] |
| informed/planner | 0.94 [0.875, 0.972] | 0.96 [0.902, 0.984] | 0.97 [0.915, 0.990] |
| oracle/none | 0.44 [0.347, 0.538] | 0.44 | 0.44 |
| oracle/planner | 0.94 [0.875, 0.972] | 0.98 [0.930, 0.994] | 0.99 [0.946, 0.998] |
| informed/free_even (≡ none) | 0.44 [0.347, 0.538] | 0.44 | 0.44 |
| informed/pay_workers | 0.86 [0.779, 0.915] | 0.86 [0.779, 0.915] | 0.87 [0.790, 0.922] |
| informed/front_workers | 0.86 [0.779, 0.915] | 0.87 [0.790, 0.922] | 0.86 [0.779, 0.915] |
| informed/pay_media | 0.70 [0.604, 0.781] | 0.75 [0.657, 0.825] | 0.81 [0.722, 0.875] |
| informed/pay_split | **0.88 [0.802, 0.930]** | **0.88 [0.802, 0.930]** | **0.95 [0.888, 0.978]** |

Gate per treasury, from `reports/v3/x0.json` → `gate_by_treasury`:

| criterion | ₫500 | ₫750 | ₫1000 |
|---|---|---|---|
| 1 uniform/none ≤ 30% | pass (0.07) | pass | pass |
| 2 oracle/planner ≥ 85% | pass (0.94) | pass (0.98) | pass (0.99) |
| 3 informed > top3, Wilcoxon p < 0.01 | pass | pass | pass |
| 4a no paid pattern ≥ 80% | **fail** (split 0.88) | **fail** (split 0.88) | **fail** (split 0.95) |
| 4b workers and media each best on some seed | **fail** (media best 0/100) | **fail** (media best 0/100) | **fail** (workers best 0/100) |

Criterion 3, informed vs top3 under the ORACLE planner. The paired difference is in expected share; p is the one-sided Wilcoxon, Holm-corrected over the 3 treasuries (`per_treasury.<t>.informed_vs_top3`):

| treasury | mean diff | 95% CI | p (raw) | p (Holm) | McNemar on wins (inf only / top3 only) |
|---|---|---|---|---|---|
| ₫500 | +0.0148 | [0.0104, 0.0193] | 7.5e-11 | 2.2e-10 | 2 / 3, p = 1.0 |
| ₫750 | +0.0110 | [0.0066, 0.0153] | 6.2e-08 | 1.2e-07 | 1 / 3, p = 0.625 |
| ₫1000 | +0.0102 | [0.0058, 0.0146] | 2.8e-07 | 2.8e-07 | 0 / 3, p = 0.25 |

The informed manifesto raises the expected share significantly at every treasury. The win rate does not change detectably: top3/planner already wins 95–100%.

Best paid pattern per seed, from `per_treasury.<t>.best_paid_pattern_counts`. Each count is the number of seeds, out of 100, on which that pattern had the highest expected share:

| pattern | ₫500 | ₫750 | ₫1000 |
|---|---|---|---|
| pay_workers | 4 | 2 | 0 |
| front_workers | 6 | 4 | 0 |
| pay_media | 0 | 0 | 1 |
| pay_split | 90 | 94 | 99 |

Split dominates the best-pattern counts for every personality of the largest region (`best_pattern_by_largest_region_personality`).

Zealot sweep, oracle/planner at ₫500 on 40 seeds (`zealot_sweep_oracle_planner`):

| zealot fraction | win rate | campaign gain in share (mean [CI]) |
|---|---|---|
| 0.00 | 0.90 | 0.282 [0.271, 0.293] |
| 0.05 | 0.90 | 0.266 [0.255, 0.277] |
| 0.15 | 0.825 | 0.233 [0.223, 0.244] |
| 0.30 | 0.75 | 0.183 [0.174, 0.193] |

Figures:
- ![X0 win rates](..figures/x0_win_rates.png)
- ![X0 gate](..figures/x0_gate.png)
- ![X0 best pattern](..figures/x0_best_pattern.png)

**Theory vs observation: confirmed.**
- **Split reaches 80%:** "split" wins 88% [80.2, 93.0] at ₫500 and ₫750 and 95% [88.8, 97.8] at ₫1000. Gate 4a fails at every treasury (`reports/v3/x0.json` → `per_treasury.<t>.max_paid_pattern`).
- **The workers-only patterns also clear 80%:** pay_workers wins 0.86–0.87 and front_workers 0.86–0.87. The lower Wilson bounds (0.78–0.79) are just under 0.80, so the 4a failure is not caused by split alone.
- **4b also fails:** media-only is never the best pattern at ₫500 or ₫750 (0/100). At ₫1000 media is best once, but workers-only is never best.
- The other three criteria pass comfortably.

**What it means.**
- **For the design:** the ₫20 economy rewards *combining* channels. Within a fixed treasury, half workers and half media beats either channel alone on 90–99% of seeds.
  - Media alone is the weakest paid pattern, with expected share 0.526 vs 0.623 for split at ₫500 (`summary.informed/pay_media`, `summary.informed/pay_split`).
  - Per D5 the price stays at ₫20 and the failure is documented. X11 (worker × media complementarity) is the planned test of why split dominates.
  - The smallest candidate fixes (not applied):
    - a gate 4b defined over *pure* channels only;
    - a media price cut that makes media-only competitive.
- **For a player:** spend on both channels every round. If you must pick one channel, pick workers; media alone is clearly worse at N = 10⁴, T = 5.

**Data.** `reports/v3/x0.json`, `reports/v3/raw/x0.jsonl` (3,300 rows = 100 seeds × 11 runs × 3 treasuries).

### X10 — Balance across the player-chosen N and T

**Status:** done, 50 seeds in every cell, including N = 10⁶. **No (N, T) cell passes all five gate criteria.**

**Question.** The API lets a player choose N ∈ [10³, 10⁶] and T ∈ [1, 10]. Does the §12 gate hold across that range? How do the two channels (workers and media) scale with N and T?

**Prediction (pre-registered).** Copied unchanged from `reports/v3/predictions.md`:
> - **Predict:**
>   - Δshare per worker ∝ 1/N (log–log slope ≈ −1), while media's Δshare per ₫ is flat in N.
>   - Workers dominate at N = 10³ (the "one worker converts most voters" problem) and are near-worthless at 10⁶.
>   - T = 1 favours media, because a cascade gets only 1 hop.

**Setup.**
- **Grid:** N ∈ {10³, 3·10³, 10⁴, 10⁵, 10⁶} × T ∈ {1, 3, 5, 10}, treasury ₫500, worker ₫20, query ₫10.
- **Seeds:** 50 per cell, 10000–10049 (`seed_block("X10", 50)`). The same seeds are used in every cell and every arm.
- **Arms:** the X0 set (11 manifesto/policy runs, as `balance.run`) in every cell, scored with the same gate logic.
- **Single-channel arms (same seeds, informed manifesto):**
  - workers-only = `informed/pay_workers`: 25 workers in every cell;
  - media-only = `informed/pay_media`;
  - none = `informed/free_even`.
- **Channel metrics:**
  - Δshare = the paired difference in expected vote share against "none";
  - converts per worker = cascade converts (absolute voters) ÷ workers hired;
  - Δshare per ₫ = Δshare ÷ the ₫ spent on that channel.
- **Electorates:** canonical CPU draws (R8) for N ≤ 10⁵. **N = 10⁶ uses `generation="device"`** (GPU draws), as recorded in `reports/v3/x10.json` → `generation_by_n`.
- **Slopes:** OLS of log₁₀(cell mean) on log₁₀ N over the 5 values of N, one fit per T. The 95% CI is a 2,000-sample seed bootstrap, with seeds resampled within each cell.
- **Compute:** cuda:2 (RTX A5000). Values from `reports/v3/x10.json` → `provenance.compute`:
  - wall time 1,058 s (17.6 min), of which N = 10⁶ took 543 s;
  - peak memory 14.2 GB;
  - mean GPU utilisation 34% (max 100%).
- **Batch sizes:** B = 50 worlds per `run_batch` call for N ≤ 10⁵ and B = 8 for N = 10⁶ (`provenance.batch_sizes.prepare_chunk_by_n`; sized by `calibrate_batch` at 70% of free memory).
  - Known issue: `provenance.batch_sizes.run_batch_B_used` shows only `[2]`, the last 10⁶ chunk, because of a record-overwrite bug. That bug is now fixed in `x0_balance.play_runs`.
- **Pilot:** 2 seeds per N, all T. It estimated 24.9 min for the full grid against a 3 h budget (`provenance.pilot`), so N = 10⁶ kept all 50 seeds.
- **Parity:** passed. 3 seeds at N = 10⁴: on the CPU, `run_batch` and `simulate_world` were bit-identical on 3/3, and the statistical CPU↔GPU check passed (`provenance.parity`).
- **Deviations:** none (`deviations: []`).
  - The run was launched from `experiments/`, so `provenance.command` shows `../reports/v3/x10.json`.
  - Canonical command: `python experiments/x10_n_t_balance.py --seeds 50 --device cuda:2 --treasury 500 --out reports/v3/x10.json`.

**Results.**

*Gate map.* The table lists the criteria that fail in each cell (`reports/v3/x10.json` → `gate_map`). No cell passes all five.

| T \ N | 10³ | 3·10³ | 10⁴ | 10⁵ | 10⁶ |
|---|---|---|---|---|---|
| 1 | 2, 4b | 2, 4b | 2, 4b | 2, 4b | 2, 4b |
| 3 | 2, 4b | **2** | **2** | 2, 4b | 2, 4b |
| 5 | 4a, 4b | 4a, 4b | 4a, 4b | **4a** | **2** |
| 10 | 1, 4a, 4b | 1, 4a, 4b | 1, 4a, 4b | 1, 4a, 4b | 1, 4a, 4b |

The failures follow T more than N (`cells.<N|T>.summary`):
- **T = 1 (criterion 2 fails):** the ORACLE planner wins only 12–36% (oracle/planner rate 0.36 at 10³, 0.12 at 10⁶). One round is too short to move the vote.
- **T = 10 (criterion 1 fails):** uniform/none wins 38–44%, and every paid pattern wins 94–100%. Social drift plus 10 cascade hops make winning easy.
- **T = 5 matches X0:** the (10⁴, 5) cell fails 4a and 4b, as X0 does at ₫500.
- **Cells that miss only one criterion:**
  - (3·10³, 3): oracle/planner 0.78;
  - (10⁴, 3): oracle/planner 0.74;
  - (10⁵, 5): split, workers and front all exactly 0.80, the 4a threshold;
  - (10⁶, 5): oracle/planner 0.80.
- **Criterion 3 passes in all 20 cells.** Informed beats top3 with one-sided Wilcoxon p ≤ 3.9e-4 after Holm correction over the 20 cells (`cells.<N|T>.informed_vs_top3.wilcoxon_p_holm`).

*Channel dominance.* The table gives the share of seeds on which workers-only gets a higher expected share than media-only (`channel_dominance.<N|T>.workers_better_rate`). The paired two-sided Wilcoxon, Holm-corrected over the 20 cells, gives p ≤ 7.5e-4 in every cell (`cells.<N|T>.channels.workers_vs_media_expected`).

| T \ N | 10³ | 3·10³ | 10⁴ | 10⁵ | 10⁶ |
|---|---|---|---|---|---|
| 1 | **0.98 W** | 0.00 M | 0.00 M | 0.00 M | 0.00 M |
| 3 | 1.00 W | 0.86 W | 0.30 M | 0.00 M | 0.00 M |
| 5 | 1.00 W | 1.00 W | 1.00 W | 0.70 W | 0.22 M |
| 10 | 1.00 W | 1.00 W | 1.00 W | 1.00 W | 1.00 W |

Mean Δshare of each single channel against none (`cells.<N|T>.channels.dshare_workers` / `dshare_media`):

| T | workers 10³ | workers 10⁴ | workers 10⁶ | media (all N) |
|---|---|---|---|---|
| 1 | 0.0492 | 0.0053 | 0.0001 | 0.036 |
| 3 | 0.1287 | 0.0500 | 0.0008 | 0.061–0.062 |
| 5 | 0.1328 | 0.1264 | 0.0248 | 0.051–0.052 |
| 10 | 0.1210 | 0.1238 | 0.1345 | 0.042–0.046 |

*Scaling.* Log–log slopes vs N (`slopes.<metric>.<T>`; bootstrap 95% CI):

| T | Δshare per worker | cascade converts per worker | media Δshare per ₫ |
|---|---|---|---|
| 1 | **−0.990 [−0.999, −0.982]** | 0.015 [0.000, 0.030] | −0.000 [−0.009, 0.008] |
| 3 | −0.751 [−0.786, −0.719] | 0.280 [0.234, 0.319] | −0.002 [−0.011, 0.008] |
| 5 | −0.237 [−0.275, −0.205] | 0.806 [0.760, 0.844] | −0.002 [−0.017, 0.012] |
| 10 | +0.016 [0.008, 0.023] | 1.021 [1.015, 1.028] | 0.013 [−0.009, 0.034] |

Converts per worker in absolute voters, mean [95% CI] (`cells.<N|T>.channels.converts_per_worker`):
- **T = 1:** about 4–5 at every N, from 4.1 [3.7, 4.4] at 10³ to 4.6 [4.2, 5.0] at 10⁶.
- **T = 10:** 17.8 at 10³, 190.5 at 10⁴ and 20,683 [19,956; 21,409] at 10⁶. Over the game, 25 workers convert about half of a 10⁶ electorate.

Media Δshare per ₫ is about 1.8e-4 at T = 1 and about 0.9e-4 at T = 10, at every N.

Figures:
- ![X10 gate map](..figures/x10_gate_map.png)
- ![X10 channel dominance](..figures/x10_channel_dominance.png)
- ![X10 scaling](..figures/x10_scaling.png)

**Theory vs observation: partial.**
1. **"Δshare per worker ∝ 1/N (slope ≈ −1), media Δshare per ₫ flat in N": partial.**
   - *The 1/N law holds only at T = 1:* slope −0.990 [−0.999, −0.982].
   - *The slope flattens as T grows:* −0.75 at T = 3, −0.24 at T = 5, and +0.016 [0.008, 0.023] at T = 10.
   - *Why:* the cascade is supercritical. Converts per worker grow with N with slope 0.28, 0.81 and 1.02 for T = 3, 5 and 10. Given enough hops, a worker's reach is a fixed *share* of the electorate, not a fixed *number* of voters. The premise that "R0 does not depend on N, so the reach is fixed" holds only while the cascade is hop-limited (T = 1).
   - *The media half is confirmed:* every media slope CI contains 0 (largest |slope| 0.013).
2. **"Workers dominate at 10³ and are near-worthless at 10⁶": partial.**
   - *At 10³ workers dominate:* workers-only beats media-only on 98–100% of seeds for every T.
   - *At 10⁶, T ≤ 3, workers are near-worthless:* Δshare 0.0001 and 0.0008 for 25 workers, against 0.036–0.061 for media.
   - *At 10⁶, T = 10, workers are the dominant channel instead:* they beat media on 100% of seeds, with Δshare 0.1345 against 0.046.
3. **"T = 1 favours media": partial.**
   - *Confirmed for N ≥ 3·10³:* media beats workers on 100% of seeds.
   - *Refuted at N = 10³:* workers beat media on 98% of seeds, because about 4 converts per worker is a sizeable share of 1,000 voters.

**What it means.**
- **For the design:**
  - Balance depends on *both* N and T, and the worker channel's value changes regime with T:
    - hop-limited at T = 1, where it falls like 1/N;
    - saturating at T = 10, where it is independent of N.
  - So no single N-scaled worker price can correct for N across all T.
  - Two failures do not involve N at all: T = 1 is too short for any strategy (criterion 2), and T = 10 is too easy without one (criterion 1).
  - The options for the owner checkpoint (D7) are in `reports/v3/sections/x10_checkpoint.md`.
- **For a player:**
  - With a short game on a large electorate (T ≤ 3, N ≥ 10⁵), put the money in media.
  - With a long game (T = 10), or any game on a small electorate (N = 10³), buy workers early.
  - Near T = 5, N ≤ 10⁴, combine both channels, as in X0.

**Data.** `reports/v3/x10.json`, `reports/v3/raw/x10.jsonl` (11,000 rows = 20 cells × 50 seeds × 11 runs).

### X7 — Budget and price sensitivity

**Question.** How do the win rate, expected share and spending mix change with the treasury and with the
prices of workers and survey queries? Does money have diminishing returns, and at what worker fee do workers stop
being worth it compared with media? This is a sensitivity study only: the game keeps ₫20 workers and ₫10 queries (D5).

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> money has diminishing returns; workers stop being worth it relative to media somewhere between
> ₫50 and ₫100 at N = 10⁴ (P8 table).

**Setup.**
- **Seeds:** first seed 10000, 50 paired seeds per cell (all cells use the same list).
- **Game:** N = 10⁴, T = 5. Electorates use the canonical CPU draw (R8), moved to the GPU, so all players face
  identical worlds.
- **Grid:** treasury {₫500, 750, 1000} × worker fee {₫10, 20, 50, 100} × query fee {₫5, 10}, giving 24 cells.
- **ORACLE:** the oracle manifesto plus `balance.policy_planner` on the true state, run batched through
  `x_batch.run_batch` (B = 50).
  - It is a reference, not a player.
  - It asks no queries, so it is query-fee invariant. It was run once per (treasury, worker fee), and its cells are
    keyed `Q any`.
- **Patterns:** the four scripted paid patterns of the X0 gate (pay_workers, front_workers, pay_media, pay_split), with
  the informed manifesto as in `balance.py` criterion 4. They define the per-cell dominant pattern.
- **Balanced player:** `BayesPlanner` playing full `GameEngine` games (1,200 games).
  - The cell's config is built experiment-side, the `--set` equivalent.
  - The config is passed to the engine and to `BayesPlanner.config`, because its planner prices workers from the
    config, not from the observation.
- **The plan's fix:** `survey_budget` is set explicitly to 5 questions' worth: ₫25 at ₫5 and ₫50 at ₫10
  (`survey_budget_rule` in reports/v3/x7.json). All other BayesPlanner parameters are the library defaults
  (`bayes_planner_parameters`).
- **Device:** cuda:0 (RTX A5000).
- **Wall time:** 987 s (16.5 min) in total. The BayesPlanner games ran in 2 spawned processes on the GPU, with a
  pilot of 1.73 s per game and an estimated 20 min.
- **GPU use:**
  - mean utilisation 6.2% (max 30%); the games are bound by Python orchestration;
  - peak memory 67 MB in the parent and 80 MB per child;
  - 2,293 MiB used on the device at most;
  - all from `provenance.compute` and `bayes_run` in reports/v3/x7.json.
- **Parity (CPU↔GPU, 3 seeds, cell ₫750/₫50/₫10):** passed for the ORACLE `run_batch` and for a BayesPlanner game.
  The CPU planner with the CPU engine was compared with the GPU-rollout planner and the GPU engine. Expected share
  0.622 vs 0.618 and 0.625 vs 0.620 (`provenance.parity`).
- **Deviations:**
  1. `BayesPlanner(device="cuda")` crashes in `manifesto()`. An experiment-side subclass
     (`x_gpu_benchmarks._gpu_planner_class`) keeps the manifesto worlds on the CPU, so the rollouts run on the GPU.
     The library fix awaits owner approval.
  2. Device cuda:0 instead of the plan default cuda:2, which is in use by X0/X10.
  3. No seed reduction was needed: the estimate was 20 min, against a 3 h limit.

**Results.** Win rate (95% Wilson CI) and mean expected share E (95% t-CI). Spend mix is the share of money spent
on workers / media / surveys, and "dominant" applies the 2/3 rule to the pooled spend. Source: `cells` in
reports/v3/x7.json.

| Treasury | Worker fee | ORACLE win | ORACLE E | ORACLE mix W/M | Bayes (₫10 query) win | Bayes E | Bayes mix W/M/S | Bayes dominant |
|---|---|---|---|---|---|---|---|---|
| ₫500 | ₫10 | 0.92 [0.81, 0.97] | 0.633 | 0.73 / 0.27 | 0.98 [0.90, 1.00] | 0.656 | 0.38 / 0.58 / 0.04 | mixed |
| ₫500 | ₫20 | 0.92 [0.81, 0.97] | 0.638 | 0.65 / 0.35 | 0.96 [0.87, 0.99] | 0.649 | 0.46 / 0.50 / 0.04 | mixed |
| ₫500 | ₫50 | 0.90 [0.79, 0.96] | 0.638 | 0.57 / 0.43 | 0.84 [0.72, 0.92] | 0.631 | 0.44 / 0.52 / 0.04 | mixed |
| ₫500 | ₫100 | 0.88 [0.76, 0.94] | 0.633 | 0.46 / 0.54 | 0.82 [0.69, 0.90] | 0.608 | 0.52 / 0.44 / 0.04 | mixed |
| ₫750 | ₫10 | 0.94 [0.84, 0.98] | 0.638 | 0.74 / 0.26 | 1.00 [0.93, 1.00] | 0.674 | 0.25 / 0.73 / 0.03 | media |
| ₫750 | ₫20 | 0.96 [0.87, 0.99] | 0.646 | 0.65 / 0.35 | 0.98 [0.90, 1.00] | 0.667 | 0.38 / 0.59 / 0.03 | mixed |
| ₫750 | ₫50 | 0.96 [0.87, 0.99] | 0.651 | 0.58 / 0.42 | 0.94 [0.84, 0.98] | 0.654 | 0.51 / 0.47 / 0.03 | mixed |
| ₫750 | ₫100 | 0.92 [0.81, 0.97] | 0.646 | 0.36 / 0.64 | 0.90 [0.79, 0.96] | 0.637 | 0.41 / 0.57 / 0.03 | mixed |
| ₫1000 | ₫10 | 0.98 [0.90, 1.00] | 0.642 | 0.74 / 0.26 | 1.00 [0.93, 1.00] | 0.687 | 0.23 / 0.75 / 0.02 | media |
| ₫1000 | ₫20 | 0.98 [0.90, 1.00] | 0.648 | 0.67 / 0.33 | 1.00 [0.93, 1.00] | 0.688 | 0.22 / 0.76 / 0.02 | media |
| ₫1000 | ₫50 | 0.96 [0.87, 0.99] | 0.656 | 0.58 / 0.42 | 1.00 [0.93, 1.00] | 0.672 | 0.41 / 0.57 / 0.02 | mixed |
| ₫1000 | ₫100 | 0.94 [0.84, 0.98] | 0.659 | 0.51 / 0.49 | 0.92 [0.81, 0.97] | 0.654 | 0.42 / 0.56 / 0.02 | mixed |

The ORACLE's dominant label is "workers" at ₫10 for every treasury and at ₫20 with ₫1000, and "mixed" otherwise. The
BayesPlanner with ₫5 queries has an expected share within 0.003 of its ₫10 numbers in every cell (family C below).
Its win rate differs by at most 0.02 (`cells.BayesPlanner`).

**Paired tests** (reports/v3/x7.json → `tests`; Holm within each family):
- **A. All-workers vs all-media (pay_workers − pay_media, E).**
  - All-workers is ahead in all 12 (treasury, fee) cells, including ₫100.
  - At ₫500 the gap is +0.078 [0.073, 0.084] at ₫10 and +0.048 [0.036, 0.059] at ₫100 (Holm p 2e-14 and 9e-9).
  - At ₫1000 it is +0.027 at ₫10 and +0.014 [0.006, 0.022] at ₫100 (Holm p 4e-4).
  - The gap shrinks with both fee and treasury but never changes sign.
  - On wins, pay_workers gains 0–7 seeds over pay_media and never loses one. No McNemar test reaches p < 0.05
    after Holm.
- **Best paid pattern per seed** (`patterns`):
  - **pay_split** is best in 47–50 of 50 seeds at ₫10–₫20 for every treasury, and in 34–49 at ₫50.
  - At ₫100, **front_workers** takes over at ₫500 (35/50) and ₫750 (38/50). At ₫1000, pay_split still leads (40/50).
  - pay_media is best in at most 5/50 seeds in any cell.
  - The best pattern's win rate is ≥ 0.80 in every cell (0.80–0.98), so the X0 criterion 4a ("no paid pattern
    ≥ 80%") would fail in all 12.
- **B. Diminishing returns** (gain 500→750 minus gain 750→1000, E):
  - **ORACLE:** significant at ₫20 (+0.005 [0.004, 0.007], Holm p 1e-7) and ₫50 (+0.007 [0.004, 0.010], p 2e-7).
    Not significant at ₫10 (+0.001, p 0.28) or ₫100 (+0.001, p 1).
  - **BayesPlanner:** not significant in any of its 8 fee columns (all Holm p ≥ 0.16). Its point estimates range
    from −0.002 to +0.013.
- **C. Query fee ₫5 vs ₫10 (BayesPlanner, same 5-question survey budget):**
  - The mean difference is −0.0012 to +0.0025 in E. Only one of 12 cells is significant (₫500/₫10: +0.0019,
    Holm p 0.018).
  - At most 1 discordant win per cell.
  - The player asks only 2.02 (₫10) or 2.06 (₫5) queries per game on average: its entropy and gain stopping rules
    stop it before the ₫25/₫50 cap binds.
- **D. Each worker fee vs ₫20, same treasury:**
  - **BayesPlanner** loses E as workers get dearer: −0.018 to −0.019 at ₫50 and −0.039 to −0.041 at ₫100 with ₫500
    (Holm p ≤ 8e-4). It gains +0.008 to +0.009 at ₫10.
  - **ORACLE** barely moves (−0.008 to +0.010). It is slightly *better* at ₫50–₫100 than at ₫20 with ₫750–₫1000
    (+0.005 to +0.010, Holm p ≤ 2e-5).
  - No win-rate difference survives Holm (all McNemar Holm p ≥ 0.42).

Figures:
- ![X7 win rates](..figures/x7_win_rates.png)
- ![X7 spend mix](..figures/x7_spend_mix.png)
- ![X7 workers vs media](..figures/x7_workers_vs_media.png)

**Theory vs observation. Partial.**
- **Diminishing returns: partly confirmed.** For the ORACLE, the second ₫250 buys less than the first at ₫20 and ₫50
  (Holm p ≤ 2e-7). The effect is not detectable at ₫10 or ₫100, nor for the BayesPlanner, whose gains per ₫250 step
  are about 0.017–0.030 with wide CIs.
- **"Workers stop being worth it between ₫50 and ₫100": refuted as stated.** Workers are still worth more than media
  at ₫100. All-workers beats all-media in all 12 cells (smallest gap +0.014, Holm p 4e-4).
- **What does happen:**
  - Worker value per ₫ falls.
  - The ORACLE's worker share of spend drops from 0.73–0.74 at ₫10 to 0.36–0.51 at ₫100.
  - The best scripted pattern moves from pay_split to front_workers at ₫100 (₫500, ₫750).

**What it means.**
- **For the design:**
  - Within the tested range, prices move the mix more than the outcome. ORACLE win rates stay within 0.88–0.98 and
    BayesPlanner within 0.82–1.00 across every price.
  - Neither ₫10 nor ₫100 workers would make the balance gate's 4a pass, so a price change alone does not fix the
    pay_split dominance found in X0. This is consistent with D5: keep ₫20.
  - The query fee hardly matters at the default stopping rules, because the BayesPlanner buys only about 2
    questions.
- **For a player:**
  - Combine channels, and front-load workers when they are expensive.
  - Extra treasury beyond ₫750 adds little expected share for a strong planner.
- **Caveat:** the ORACLE planner is a heuristic. It does better when workers cost more, which means it over-buys
  workers at ₫20; X13 looks at this.

**Data.** reports/v3/x7.json, reports/v3/raw/x7.jsonl (4,200 rows), log reports/v3/logs/x7.log. Command:
`OMP_NUM_THREADS=4 python experiments/x7_prices.py --seeds 50 --first-seed 10000 --device cuda:0 --out reports/v3/x7.json`.

### X23 — Worker-rule ablations

**Question.** Are the owner's worker rules what make the game work, or would nearby rules do as well? The rules are:
- R27: attempt cap 3;
- R22: min degree 3;
- R25: conversion chance γ·P.

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> cap ∞ and rule γ bring back cascade saturation (the manifesto stops mattering, as P8 saw with γ
> alone). A cap of 1 makes workers subcritical.

**Setup.**
- **Seeds:** first seed 10000, 50 paired seeds per arm.
- **Game:** N = 10⁴, T = 5, ₫500, ₫20 workers, ₫10 queries.
- **Arms:** one rule changed at a time from the default:
  - `workers.attempt_cap` ∈ {1, 2, 3, 5, 1000};
  - `workers.min_degree` ∈ {1, 3, 10};
  - `workers.conversion_rule` ∈ {γ, γ·P, γ·P²};
  - the three default values form one shared "default" arm.
- **Runs per arm, on the GPU** (`x_batch.run_batch`, B = 50):
  - the X0 policies with the ORACLE manifesto: none, pay_workers, front_workers, pay_media, pay_split, and the ORACLE
    planner;
  - the extra runs that `balance.py` needs for its five gate criteria (uniform/none, uniform/planner, top3/planner,
    informed/planner, informed/{4 paid patterns}).
- **Metrics:**
  - **Reach per worker:** distinct voters converted at least once, divided by the workers bought.
  - **Branching ratio:** Σ converts ÷ Σ spreaders over the rounds. A spreader spreads for one hop, so < 1 means
    subcritical.
  - **Cap binding:** the share of targeted voters whose final `attempts_j == cap`. It is read from the final
    `WorkerState` through an experiment-side observer.
- **Device and compute:** cuda:0 (RTX A5000); wall time 104 s; peak memory 0.95 GB; mean GPU utilisation 12% (max 41%).
- **Parity (CPU↔GPU, 3 seeds, γ arm, ORACLE planner):** passed. Expected share 0.826 vs 0.825; converts as a share of
  N 0.680 vs 0.678 (`provenance.parity`).
- **Deviations:**
  - Device cuda:0 instead of cuda:2, which is in use by X0/X10.
  - The parity metric for reach is converts / N rather than converts per worker. The check takes one absolute
    tolerance for every metric, and 0.01 voters per worker is far below run-to-run noise.
  - The ORACLE planner values workers with γ·P and no cap in every arm. It is rule-unaware and not re-tuned.

**Results.** ORACLE manifesto. E = mean expected share. Reach = distinct converts per worker (95% t-CI). BR = pooled
branching ratio. Cap = the share of targeted voters at the cap (pooled). Gate = criteria 1, 2, 3, 4a, 4b
(✓ pass / ✗ fail). Source: `arms.<arm>.runs`, `arms.<arm>.gate` and `arms.<arm>.manifesto_effect` in
reports/v3/x23.json.

| Arm | ORACLE planner win | Planner E | pay_split win | pay_workers win | Planner reach | Planner BR | Planner cap | Manifesto effect (oracle − uniform, planner) | Gate 1/2/3/4a/4b |
|---|---|---|---|---|---|---|---|---|---|
| cap 1 | 0.76 [0.63, 0.86] | 0.568 | 0.76 | 0.72 | 101 [94, 107] | 1.08 | 1.00 | 0.065 [0.049, 0.082] | ✓ ✗ ✓ ✓ ✗ |
| cap 2 | 0.84 [0.72, 0.92] | 0.607 | 0.86 | 0.78 | 175 [165, 185] | 1.13 | 0.95 | 0.063 [0.046, 0.080] | ✓ ✗ ✓ ✗ ✗ |
| **default** (cap 3, deg ≥ 3, γP) | 0.92 [0.81, 0.97] | 0.638 | 0.88 | 0.86 | 231 [218, 243] | 1.20 | 0.93 | 0.062 [0.044, 0.079] | ✓ ✓ ✓ ✗ ✗ |
| cap 5 | 0.98 [0.90, 1.00] | 0.680 | 0.92 | 0.90 | 303 [288, 319] | 1.37 | 0.87 | 0.060 [0.042, 0.078] | ✓ ✓ ✓ ✗ ✗ |
| cap ∞ (1000) | 1.00 [0.93, 1.00] | 0.752 | 0.94 | 0.94 | 416 [395, 437] | 2.13 | 0.00 | 0.066 [0.043, 0.090] | ✓ ✓ ✗ ✗ ✗ |
| deg ≥ 1 | 0.92 | 0.638 | 0.88 | 0.86 | 231 (identical) | 1.20 | 0.93 | 0.062 | ✓ ✓ ✓ ✗ ✗ |
| deg ≥ 10 | 0.92 | 0.638 | 0.88 | 0.86 | 231 [218, 243] | 1.21 | 0.93 | 0.061 | ✓ ✓ ✓ ✗ ✗ |
| rule γ | 1.00 [0.93, 1.00] | 0.826 | 1.00 | 1.00 | 433 [423, 444] | 1.00 | 1.00 | 0.019 [0.011, 0.026] | ✓ ✓ ✓ ✗ ✗ |
| rule γP² | 0.78 [0.65, 0.87] | 0.571 | 0.76 | 0.72 | 133 [120, 145] | 1.43 | 0.70 | 0.088 [0.069, 0.106] | ✓ ✗ ✓ ✓ ✓ |

Rule-invariant runs (no workers bought): oracle/none (win 0.52, E 0.481), oracle/pay_media (0.72, 0.531) and
uniform/none (0.10) are identical in every arm, as they must be.

**Paired tests vs the default arm** (`tests` in reports/v3/x23.json; Holm within each family):
- **Expected share (48 tests; Wilcoxon on arm − default):**
  - **cap 1:** −0.067 to −0.074 for the four worker-buying policies (planner −0.069 [−0.072, −0.066]).
  - **cap 2:** −0.028 to −0.032.
  - **cap 5:** +0.034 to +0.046.
  - **cap ∞:** +0.071 to +0.139 (front_workers +0.139).
  - **γ:** +0.188 to +0.206.
  - **γP²:** −0.067 to −0.074.
  - All of these have Holm p ≈ 9e-14.
  - **Min degree 1 and 10:** no change; every Holm p = 1, with min degree 1 bit-identical to the default.
- **Wins (48 exact McNemar tests):**
  - The discordant counts follow the same direction: cap 1 loses 6–8 seeds and gains 0; γ gains 4–7 and loses 0.
  - None survives Holm with 50 seeds (smallest Holm p 0.38).
- **Reach per worker (32 tests):**
  - **Planner:** cap 1 −130 [−137, −123]; cap ∞ +185 [173, 197]; γ +203 [189, 217]; γP² −98 [−108, −89].
    All Holm p ≤ 1e-8.
  - **Min degree:** |Δ| ≤ 0.9, Holm p = 1.
- **Does the manifesto matter less?** (8 tests; the oracle − uniform manifesto effect under the planner, arm minus
  default):
  - **γ:** −0.043 [−0.057, −0.029] (Holm p 2e-6).
  - **γP²:** +0.026 [0.019, 0.033] (Holm p 6e-10).
  - **cap ∞:** +0.005 [−0.009, 0.018] (Holm p 1).
  - **cap 1:** +0.004 (Holm p 0.33).
  - **Gate criterion 3 (informed − top3 under the planner)** falls from +0.016 [0.010, 0.022] at the default to
    +0.006 [0.002, 0.009] under γ (still passing). Under cap ∞ it is +0.012 [−0.001, 0.024]: the one-sided Wilcoxon
    p of 0.01 is not < 0.01, so criterion 3 fails there.
- **Cap binding** (`runs.<run>.cap_bind_share_pooled`):
  - Under the default, 93% of the voters the planner's cascade targets are at the cap of 3 by the end; 85–95%
    across the worker policies. Every world has a voter at the cap.
  - Cap 1 binds for 100%, cap 5 for 72–91%, and cap ∞ never.
  - Under γ the cap binds for 100%, with BR 0.997–1.018: every reachable voter is exhausted.
- **Saturation profile under γ** (`converted_by_round_mean`): the planner converts 68, 924, 4,847, 2,434 and 18 voters
  in rounds 1–5. The cascade burns through the electorate by round 3 and then runs out of targets.
- **Min degree is non-binding at N = 10⁴, k̄ = 30.** No non-zealot voter has degree < 3. Only 0.06% have degree < 10,
  and the smallest degree in 50 worlds is 4 (`provenance.degree_check`).

Figures:
- ![X23 reach](..figures/x23_reach.png)
- ![X23 win rates](..figures/x23_win_rates.png)
- ![X23 cap binding](..figures/x23_cap_binding.png)

**Theory vs observation. Partial.**
- **Rule γ brings back saturation: confirmed.**
  - Every voter the cascade can reach is converted: the cap binds for 100% of targets, the conversions stop by
    round 5, and reach per worker is tight at 269 [267, 272] for pay_workers.
  - Every paid pattern wins 50/50.
  - The manifesto matters much less (oracle − uniform falls from 0.062 to 0.019, Holm p 2e-6), but not zero.
- **Cap ∞: partly confirmed.**
  - The cascade becomes strongly supercritical (BR 2.13, reach +185 per worker), and the planner wins 50/50.
  - Gate criterion 3 fails (informed − top3 CI includes 0).
  - The oracle − uniform manifesto effect does not shrink (+0.005, Holm p 1). So the manifesto still matters for the
    planner; the cap ∞ world is "workers win everything" rather than "the manifesto stops mattering".
- **"A cap of 1 makes workers subcritical": refuted in the strict sense.**
  - The pooled branching ratio stays above 1 (1.08 for the planner, 1.03–1.23 across the worker policies).
  - Cap 1 does more than halve reach (231 → 101 per worker), and the cascade dies out by round 5 (front_workers:
    785 → 386 → 67 converts in rounds 3–5).
  - The ORACLE win rate falls to 0.76, failing gate criterion 2 (≥ 0.85).

**What it means.**
- **For the game design:**
  - The cap is the main dial of worker strength: from cap 1 to cap ∞, ORACLE E rises monotonically from 0.568 to 0.752.
  - Cap 3 is the smallest cap that keeps criterion 2 (0.92 ≥ 0.85).
  - γ·P (R25) is essential. The literal γ rule makes any worker-buying pattern a sure win and weakens the manifesto
    by about 70%.
  - γ·P² is the only arm where 4a and 4b both pass (pay_media best in 11/50 seeds), but it fails criterion 2 (0.78).
    A milder version, or γ·P² with cap 5, could be worth an owner look; it was not run.
  - The min-degree rule R22 has no effect at the default N and k̄. It is harmless, but it does nothing; it would
    only matter in sparse graphs (X1).
- **For a player:** under the real rules, a bought worker reaches about 230 voters (ORACLE planner) and the cap is
  almost always hit. Buying workers early, before targets are used up, matters more than buying many.

**Data.** reports/v3/x23.json, reports/v3/raw/x23.jsonl (6,300 rows), log reports/v3/logs/x23.log. Command:
`OMP_NUM_THREADS=4 python experiments/x23_worker_rules.py --seeds 50 --first-seed 10000 --device cuda:0 --out reports/v3/x23.json`.

## 5. Network and cascade mechanics vs theory

### XC — Consensus towards the manifesto (descriptive)

**Status:** done (added after the suite, at the owner's request; descriptive, no pre-registered prediction).

**Question.** The campaign's goal is to bring voters to the manifesto b. How far, and how fast, does each kind of
campaign pull the electorate towards b, and does it also pull the electorate together?

**Setup.** 50 seeds (10000–10049, the X0 block), N = 10⁴, T = 5, treasury ₫500, worker ₫20, ORACLE manifesto. Five
campaigns are played on the same worlds: no campaign, media only, workers only, split, and the ORACLE planner (the X0
policies). Every voter's preference is recorded after each round. A voter is *won over* when their agreement
S_i = 1 − TV(v_i, b) reaches the vote threshold z = 0.825, i.e. they vote for b with chance ≥ ½. *Disagreement* is
the mean TV distance of voters from the population mean (consensus drives it to 0). Device cuda:2, 36 s
(x_consensus.json → `provenance`).

**Results.** Mean and 95% CI over the 50 seeds (x_consensus.json → `summary`):

| Campaign | Mean agreement S̄, round 0 → 5 | Voters won over, round 0 → 5 | Disagreement, round 0 → 5 | Election win rate |
|---|---|---|---|---|
| no campaign | 0.783 → 0.815 [0.809, 0.821] | 0.343 → 0.506 [0.481, 0.532] | 0.217 → 0.184 | 0.52 |
| media only | 0.783 → 0.830 [0.824, 0.835] | 0.343 → 0.574 [0.548, 0.600] | 0.217 → 0.170 | 0.72 |
| workers only | 0.783 → 0.866 [0.858, 0.873] | 0.343 → 0.607 [0.581, 0.633] | 0.217 → 0.143 | 0.86 |
| split (half / half) | 0.783 → 0.873 [0.866, 0.881] | 0.343 → 0.641 [0.615, 0.668] | 0.217 → 0.136 | 0.88 |
| ORACLE planner | 0.783 → 0.875 [0.868, 0.882] | 0.343 → 0.651 [0.624, 0.678] | 0.217 → 0.133 | 0.92 |

![Consensus towards the manifesto over the rounds](..figures/xc_consensus_rounds.png)

*Mean agreement with b (left), share of voters won over with a majority line (middle), and disagreement (right), by
round; 50 seeds, mean and 95% CI (x_consensus.json).*

![One world: agreement before and after, and voters moving towards b](..figures/xc_consensus_example.png)

*Seed 10000 under the ORACLE planner (x_consensus.json → `example`).*
- **Left:** distribution of agreement with b at round 0, at round 5 with no campaign (outline) and at round 5 with
  the planner.
- **Middle and right:** 1,500 sampled voters projected on the first two principal components of their preferences,
  coloured by region, at round 0 and round 5; ☆ marks b. 2,263 voters, converted or hired, sit at b by round 5
  (`example.projection.count_at_b_tT`), against 1 at round 0.

**What it shows.**
- **Drift alone helps a well-chosen b.** Social averaging alone moves 34% → 51% of voters past the threshold.
- **Every paid campaign turns that drift into a clear majority,** in the order none < media < workers < split ≤
  planner, the same order as the win rates (and as in X0).
- **The campaign also builds consensus:** disagreement falls from 0.217 to 0.133 under the planner, against 0.184
  with no campaign.
- **Workers move agreement more than media,** because a converted voter jumps all the way to b (R25).
- **The pull is regional (seed 10000).** The two regions nearest b (regions 1 and 4) move towards it, while the two
  far regions barely move: the campaign builds a coalition, not a national consensus (cf. X15, X19).

**Data.** reports/v3/x_consensus.json, log reports/v3/logs/x_consensus.log. Command:
`OMP_NUM_THREADS=4 python experiments/x_consensus.py --seeds 50 --first-seed 10000 --device cuda:2`.

### X1 — Network structure and consensus speed

**Status:** done. Decay follows ā(1 − μ₂) (Spearman 0.997); high p_g raises nationwide-manifesto wins 0.71 → 0.92. Confirmed.

**Question.**
- Does the speed at which opinions mix follow the spectral gap, and which one?
- Do well-connected countries reward a single nationwide manifesto, while fragmented ones reward region-targeted
  play?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> the decay follows ā(1 − μ₂(D⁻¹A)), not λ₂(L)/k̄ (as E1 found). Under the P8 rules, high p_g raises
> the wins of a single nationwide manifesto.

**Setup.**
- **Seeds:** first seed 11000, 50 paired seeds per cell. The same seed gives the same regions, v(0) and
  susceptibilities in every cell; only the graph changes.
- **Grid:** k̄ ∈ {10, 20, 30} × p_g range ∈ {[.01, .05], [.01, .2] (default), [.1, .4]}.
- **Game:** N = 10⁴, T = 5, ₫500.
- **Spectra (SciPy on the CPU, giant component of the zealot-free copy):** λ₂(L); μ₂ and μ_min of D⁻¹A; ν₂ of the
  exact operator G^½ L G^½ with G = diag(a/d) and a_i = ω·α_i; cross-region conductance.
- **Long social-only runs (GPU, float64, all 50 worlds of a cell batched):**
  - zealot-free copy, x ← (1 − a)x + a·D⁻¹A x;
  - heterogeneous a for 2,000 rounds, homogeneous a = ā for 400 rounds.
- **Measures:**
  - observed contraction ρ_obs, from the slope of log‖x − 1c‖;
  - "rounds until the regional means move halfway to the global mean" (between-region TV dispersion ≤ ½);
  - the predictions ρ_rw = 1 − ā(1 − μ₂), ρ_L = 1 − ā·λ₂(L)/k̄ and ρ_exact = 1 − ν₂.
- **Game outcomes (T = 5, batched `run_batch`):** ORACLE manifesto with none / planner; the nationwide top-3 and
  uniform manifestos with the planner; the informed manifesto with pay_workers / pay_media / pay_split.
- **Device and compute:** cuda:2. Wall time 1,293 s (21.6 min), of which spectra took about 35 s per cell
  (`provenance.timing`); peak memory 1.2 GB; mean GPU utilisation 15% (the SciPy spectra and the planner's Python
  dominate).
- **Checks:**
  - CPU↔GPU parity (3 seeds, k̄ = 10, low p_g, ORACLE planner): passed (`provenance.parity`).
  - GPU `lobpcg` spectra vs SciPy on 3 worlds: max |Δ| = 5·10⁻⁹ (`provenance.spectra_gpu_vs_scipy`).
- **Deviations:**
  - The spectra use SciPy. At N = 10⁴ it is about 7× faster than the GPU path (0.3 s vs 2.3–2.5 s), which is the
    §1.1 "measured slower" exception.
  - The first full run was killed by a CUDA out-of-memory error after 3 cells, when it shared cuda:2 with X18's CELF
    batch. It was rerun alone from scratch; the JSON is from the complete rerun.
  - "Standard strategies" are the X0 policy set. The observation-only players are X6's job.
  - At k̄ = 10 the graphs have 2.3–6 components on average, so the spectra use the giant component.

**Results.** Source: reports/v3/x1.json (`cells`, `theory_vs_obs`, `nationwide`).

*Consensus speed.* Predictions vs observed decay rates over all 450 worlds (`theory_vs_obs`):

| Observed decay 1 − ρ | vs ā(1 − μ₂): Spearman / mean log(obs/pred) | vs ā·λ₂(L)/k̄ | vs exact 1 − ν₂ |
|---|---|---|---|
| homogeneous a, t 200–400 | **0.997 / +0.044** (RMSE 0.07) | 0.616 / +0.648 (RMSE 0.84) | −0.03 / +3.47 |
| homogeneous a, between-region dispersion | **0.926 / −0.018** | 0.698 / +0.587 | — |
| heterogeneous a, first 5 rounds | **0.939** / +0.66 | 0.483 / +1.27 | −0.05 / +4.08 |
| heterogeneous a, t 1500–2000 (asymptotic) | −0.29 / −2.86 | −0.15 / −2.26 | **0.896** / +0.56 |

- With a uniform step the random-walk gap predicts the observed decay within 4.5% on average. λ₂(L)/k̄
  under-predicts it by about 1.9× and fails to rank the worlds (ρ = 0.62).
- With the game's heterogeneous steps, the game-horizon decay ranks with the random-walk gap (ρ = 0.94). The slow
  asymptotic tail is set by the exact operator (ρ = 0.90), and it is still about 1.8× faster than 1 − ν₂ at
  t = 2,000: the tail has not converged.

*Regional mixing vs the 5-round game* (`cells.*.het_between_rel_mean`, `*_t_half_between`, `t_half_pred_rw`):

| p_g | μ₂ (k̄ = 30) | between-region dispersion left after 5 rounds | rounds to halve it (observed, heterogeneous a) | predicted ln 2/−ln ρ_rw |
|---|---|---|---|---|
| [.01, .05] | 0.892 | 0.95 | 93.5 (k̄ = 30); 97–108 (k̄ = 10–20) | 101 |
| [.01, .2] | 0.731 | 0.87 | 32.5 | 42 |
| [.1, .4] | 0.500 | 0.80 | 18.5 | 21 |

Within 5 rounds the regional means move only 5–20% of the way to the global mean (E1 found 1–5% per round).

*Game outcomes: nationwide manifestos and cross-region ties.* Win rates by p_g (`nationwide`; paired by seed,
high − low p_g, Holm over 12):

| Run | p_g low → high, k̄ = 10 | k̄ = 20 | k̄ = 30 | pooled | discordant (high-only / low-only) | ΔE pooled |
|---|---|---|---|---|---|---|
| top-3 manifesto + planner | 0.56 → 0.84 | 0.78 → 0.92 | 0.80 → 1.00 | 0.71 → 0.92 | 31 / 0 (Holm p 10⁻⁸) | +0.079 [0.075, 0.083] |
| ORACLE manifesto + planner | 0.70 → 0.90 | 0.80 → 0.96 | 0.84 → 1.00 | 0.78 → 0.95 | 26 / 0 (Holm p 3·10⁻⁷) | +0.083 [0.078, 0.087] |
| uniform manifesto + planner | 0.34 → 0.60 | 0.48 → 0.68 | 0.60 → 0.98 | 0.47 → 0.75 | 42 / 0 (Holm p 6·10⁻¹²) | +0.088 [0.085, 0.092] |

- All 12 Wilcoxon Holm p ≤ 2·10⁻¹⁴.
- With no campaign, the ORACLE manifesto wins 0.12 at low p_g and 0.70 at high p_g, for every k̄ (`cells.*.oracle/none`).
- **The best paid pattern depends on density** (`cells.*.best_paid_pattern`):
  - at k̄ = 10, pay_media is best in 26–38 of 50 seeds;
  - at k̄ = 30, pay_split is best in 39–47 of 50.
  - In sparse graphs the cascade is weaker: the ORACLE planner converts 740–1,640 voters at k̄ = 10 vs 4,175–4,928
    at k̄ = 30.

Figures:
- ![X1 mixing and wins](..figures/x1_mixing_and_wins.png)
- ![X1 regional mixing](..figures/x1_regional_mixing.png)

**Theory vs observation. Confirmed.**
- **"The decay follows ā(1 − μ₂(D⁻¹A)), not λ₂(L)/k̄": confirmed.** It holds near-exactly for a uniform step
  (Spearman 0.997, bias +4.5%), and as a ranking at the game horizon with the real heterogeneous steps (0.94).
  λ₂(L)/k̄ misses by about 2× and misranks. The asymptotic tail with heterogeneous steps belongs to the exact
  operator ν₂, not to either lecture form.
- **"High p_g raises the wins of a single nationwide manifesto": confirmed.** The nationwide top-3 manifesto goes
  from 0.71 to 0.92 (31 vs 0 discordant seeds), and the same holds for the ORACLE and uniform manifestos.

**What it means.**
- **For the game design:**
  - In 5 rounds the network barely mixes regions (5–20% of the way). Social dynamics alone cannot unify a
    fragmented country within the game, so region structure stays visible to the player.
  - The cross-region tie strength p_g is a strong hidden difficulty knob: +0.08 in expected share between the low
    and high ranges.
  - Sparse graphs (k̄ = 10) flip the best pattern from split to media. Workers are a dense-network weapon.
- **For a player:** in a well-connected country, one nationwide manifesto plus workers is enough. The network carries
  it, and even no campaign wins 70%. When polls show regions staying apart (low p_g), target the manifesto and the
  spending, and lean on media if the network seems sparse.

**Data.** reports/v3/x1.json, reports/v3/raw/x1.jsonl (450 world rows with spectra, long-run curves and the 7 game
runs), log reports/v3/logs/x1.log. Command:
`OMP_NUM_THREADS=4 python experiments/x1_topology.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x1.json`.

### X2 — Stubborn agents, zealots, W-MSR (+ polarisation metric)

**Status:** done. Anchor share 0.002–0.009 (refuted); W-MSR raises the worker value slightly (refuted); zealots stop consensus (confirmed, with a qualification).

**Question.**
- How much of a worker's value comes from being a permanent anchor (a Friedkin–Johnsen stubborn agent) rather
  than from the cascade?
- Do zealots cap the reachable share?
- Does W-MSR (voters ignore their F most extreme friends) protect voters from *your* workers?
- Do pinned agents stop consensus, and does the Friedkin–Johnsen (FJ) solve predict the long-run state?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> - The worker anchor value is a sizeable share of the worker value: Arm B Δshare / full-cascade Δshare is
>   between 0.2 and 0.6, to be measured.
> - Pinned agents stop consensus (disagreement does not decay to 0).
> - W-MSR trims workers, because they sit far from their friends, so it **lowers** the worker value. That
>   makes W-MSR the network's defence against *your* campaign.

**Setup.**
- **Seeds:** first seed 11000, 50 paired seeds.
- **Game:** N = 10⁴, T = 5, ₫500, ₫20 workers.
- **Manifesto:** one fixed manifesto per seed in every cell: the ORACLE manifesto of the default world (ζ = 0.05,
  DeGroot). Rule and zealot effects are therefore not confounded with a different b.
- **Social rule:** DeGroot, W-MSR F = 1, W-MSR F = 2. Arm C is Arms A and B under each W-MSR rule.
- **Arm A:** zealot fraction ζ ∈ {0, 5, 15, 30%}. The zealot draw has its own stream, so the graph and v(0) are the
  same at every ζ. Policies: none, pay_split, ORACLE planner.
- **Arm B:** exactly k ∈ {0, 5, 10, 20} workers bought in round 1 (proportional to region size), nothing else.
  - **Cascade off:** `workers.attempt_cap = 0`. Workers stay pinned and nobody converts.
  - **Cascade on:** the game rule.
- **Long runs (experiment-side; the game allows T ≤ 10):**
  - The cascade-off dynamics continue to T = 200 with the same pinned workers, for the default world
    (ζ = 0.05) and a zealot-free world (ζ = 0, k ∈ {0, 20}).
  - Their T = 5 prefix reproduces the game's share exactly: max |Δ| ≤ 1.2·10⁻⁷ (`long_runs.*.T5_matches_game`).
- **FJ prediction:** stubborn set = workers (at b) ∪ zealots (at v(0)). It is solved as L_ff v_f = A_fs v_s by
  Jacobi-preconditioned CG on the GPU (all 50 worlds in one block-diagonal system, ≤ 95 iterations, relative
  residual < 10⁻¹⁰). On 3 worlds it is checked against SciPy CG on the CPU: max |Δv| = 1.4·10⁻¹¹
  (`fj.degroot|z0.05|k10`).
- **Polarisation (N15),** over non-zealots:
  - **between:** mean over regions of TV(regional mean, global mean);
  - **within:** mean TV(v_i, own regional mean);
  - **bimodal:** share of voters with P < 0.1 or P > 0.9.
- **(r, s)-robustness:** from `analysis.robustness`.
- **Device and compute:** cuda:2; B = 50 worlds per call. Wall time 286 s; peak memory 3.5 GB; mean GPU utilisation
  60% (`provenance.compute`).
- **Parity (CPU↔GPU, 3 seeds, W-MSR F = 1, k = 10, cascade on):** passed. Expected share 0.655 vs 0.656; within-region
  0.0802 vs 0.0805 (`provenance.parity`).
- **Deviations:**
  - Arm A's "fixed campaign" is played with two policies (pay_split and the ORACLE planner).
  - The manifesto is fixed and not re-optimised per rule.
  - The FJ reference uses SciPy `cg` instead of `spsolve`. The `spsolve` LU fill-in on a k̄ = 30 graph took
    minutes per world.

**Results.** Source: reports/v3/x2.json. Values are mean [95% CI].

*Arm B — anchor vs cascade* (`anchor_share`; Δ = E(k) − E(0)). Bootstrap CI over seeds for the ratio.

| Rule | k | Δ cascade on | Δ cascade off (anchor only) | anchor share Δ_off/Δ_on |
|---|---|---|---|---|
| DeGroot | 5 | 0.121 [0.115, 0.128] | 0.00030 [0.00028, 0.00031] | 0.0024 [0.0023, 0.0027] |
| DeGroot | 10 | 0.124 [0.119, 0.130] | 0.00058 [0.00056, 0.00060] | 0.0047 [0.0044, 0.0050] |
| DeGroot | 20 | 0.125 [0.120, 0.130] | 0.00117 [0.00112, 0.00121] | 0.0094 [0.0089, 0.0099] |
| W-MSR F = 1 | 20 | 0.127 [0.122, 0.132] | 0.00119 | 0.0094 |
| W-MSR F = 2 | 20 | 0.129 [0.124, 0.134] | 0.00122 | 0.0095 |

- The anchor effect is positive in all 9 tests (Holm p ≤ 2·10⁻⁹; `tests.anchor_positive`), but it is about 1% of
  the worker value.
- With the cascade on, the worker value saturates by k = 5: 0.121 at k = 5 vs 0.125 at k = 20.

*Arm C — W-MSR vs DeGroot* (`wmsr_effect`, `tests.wmsr_minus_degroot_worker_value`, Holm over 12):
- **Worker value with the cascade:** W-MSR **raises** it, by +0.0014 [0.0010, 0.0018] (F = 1, k = 5) to
  +0.0043 [0.0038, 0.0048] (F = 2, k = 20). All Holm p ≤ 2·10⁻⁸.
- **Anchor-only value:** it also rises, by +0.00002 to +0.00005 (Holm p ≤ 5·10⁻⁹).
- **What W-MSR does lower is the baseline:** E(no campaign) falls from 0.479 (DeGroot) to 0.471 (F = 1) and 0.463
  (F = 2) (`arm_B.B|*|on|k0`). The campaign's absolute share falls with it (k = 20: 0.603 → 0.597 → 0.592).

*Arm A — zealots* (`arm_A`, `tests.zealot_persuasion_change`, Holm over 18):

| ζ | E none | E planner | planner win | persuasion (planner − none) | change vs ζ = 0 |
|---|---|---|---|---|---|
| 0 | 0.485 | 0.647 | 0.92 | 0.162 | — |
| 0.05 | 0.479 | 0.632 | 0.90 | 0.153 | −0.009 [−0.010, −0.008] |
| 0.15 | 0.467 | 0.602 | 0.86 | 0.135 | −0.027 [−0.029, −0.026] |
| 0.30 | 0.450 | 0.557 | 0.80 | 0.107 | −0.056 [−0.059, −0.052] |

- The pattern is the same under W-MSR (F = 2, ζ = 0.30: −0.058).
- pay_split loses slightly more than the planner (ζ = 0.30: −0.066). All Holm p ≤ 8·10⁻¹⁴.

*Long runs and the FJ prediction* (default world, cascade off; `fj`, `long_runs`):

| Cell | FJ long-run share | share T = 5 | share T = 200 | mean TV(v(200), v_FJ) | disagreement T = 200 | disagreement at the FJ point |
|---|---|---|---|---|---|---|
| ζ = 0.05, k = 0 | 0.885 [0.870, 0.899] | 0.479 | 0.830 [0.802, 0.859] | 0.037 | 0.0283 | **0.0180** [0.0162, 0.0198] |
| ζ = 0.05, k = 20 | 0.888 [0.874, 0.902] | 0.480 | 0.833 | 0.037 | 0.0282 | 0.0180 |
| ζ = 0, k = 20 (workers only) | **0.971** (all at b) | 0.486 | 0.833 [0.798, 0.868] | 0.082 | 0.0147 | **0** |

- **Decay of disagreement from T = 100 to T = 200** (ratio of disagreement at 200 to that at 100,
  `long_runs.*.disagreement_ratio_200_over_100`):
  - DeGroot: 0.59 with zealots, 0.36 without.
  - W-MSR F = 1: 0.47 with zealots, 0.39 without.
  - W-MSR F = 2: 0.42 with zealots, 0.44 without.
- **W-MSR mixes more slowly early on.** Disagreement at T = 50 is 0.078 (DeGroot), 0.086 (F = 1) and 0.096 (F = 2).

*Polarisation at election day* (`arm_A`, DeGroot, ζ = 0.05):
- **between:** 0.170 with no campaign vs 0.119 with the planner.
- **within:** 0.071 vs 0.076.
- **bimodal:** 0.17 vs 0.51.
- Workers with the cascade raise "within" (0.071 → 0.081, `arm_B.B|degroot|on|k20`). Pinned workers without the
  cascade barely change them (|Δ| ≤ 0.0004 for between and within, 0.002 for bimodal).

*Robustness* (`robustness`, 50 graphs):
- **r-robustness:** the upper bound on r equals the minimum degree in every graph, so the singleton certificate
  is always the binding one: mean 8.2, range 3–13.
- **(r, s) bounds:** the region-union certificates give only weak bounds (s ≤ 2,724 for r = 2 and ≤ 1,606 for r = 3).
- **F-local condition:**
  - Normal voters with more than F zealot neighbours: 43% for F = 1 and 19% for F = 2.
  - The same with more than F of 20 workers as neighbours: 0.16% and 0.002%.
- So W-MSR's guarantee fails against the zealots, but its threat model is satisfied by the workers.

Figures:
- ![X2 anchor vs cascade](..figures/x2_anchor.png)
- ![X2 long runs](..figures/x2_long_runs.png)

**Theory vs observation. Two predictions refuted, one confirmed with a qualification.**
- **"Anchor share between 0.2 and 0.6": refuted.**
  - It is 0.002–0.009, 20–200× smaller.
  - In 5 rounds a handful of pinned voters among 10⁴ moves almost nobody. FJ agrees: even in the long-run limit,
    20 workers next to 500 zealots move the predicted share by only +0.003 (0.885 → 0.888).
  - Workers are valuable only through the cascade.
- **"Pinned agents stop consensus": confirmed for zealots, not for workers.**
  - With zealots the FJ fixed point keeps a disagreement of 0.018 > 0 (between-region 0.016), and the T = 200
    runs head towards it (TV to v_FJ 0.037).
  - With workers alone (ζ = 0) all stubborn agents share one opinion, so FJ predicts full consensus at b
    (share 0.971). Workers capture the consensus rather than stopping it, but extremely slowly: share 0.833 at
    T = 200, with disagreement still falling (ratio 0.36 per 100 rounds).
  - FJ is far from the 5-round game (0.48 at T = 5 vs 0.89).
- **"W-MSR lowers the worker value": refuted.**
  - W-MSR lowers everyone's share (the baseline drops by 0.008–0.016), but the worker value rises slightly
    (+0.001 to +0.004).
  - Workers rarely make up more than F of a voter's friends (0.16% for F = 1), so trimming them is not what
    W-MSR does. It trims extreme honest neighbours and so slows mixing (disagreement at T = 50: 0.078 vs 0.096).

**What it means.**
- **For the game design:**
  - The "permanent, pinned" part of the worker rule (R23) is almost irrelevant to outcomes. The cascade (R24–R27)
    carries all of the value.
  - A cheaper worker that is not pinned would play almost the same.
  - Zealots are the real brake on persuasion: 30% zealots cut the planner's persuasion by a third (0.162 → 0.107).
- **For a player:** do not buy workers for their own vote or their pull on friends. Buy them early, for the
  conversions they trigger. W-MSR worlds are slightly harder overall, but they do not punish workers.

**Data.** reports/v3/x2.json, reports/v3/raw/x2.jsonl (3,150 rows), log reports/v3/logs/x2.log. Command:
`OMP_NUM_THREADS=4 python experiments/x2_stubborn.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x2.json`.

### X3 — Worker cascade threshold and reach

**Status:** done. Growth crosses 1 at λ_max(A D_q) ≈ 0.94; confirmed, though only marginally sharper than the naive estimates.

**Question.** When does a worker cascade take off? Is the take-off point set by the weighted operator λ_max(A D_q),
with q_j = γ_j·P_j(b), rather than by the naive estimates k̄·E[γP] or λ_max(A)·E[γP]? How far does a cascade reach,
how often does the 3-attempt cap bind, and do converts drift back?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> growth crosses 1 near λ_max(A D_q) ≈ 1, more sharply than the naive estimates; the cap binds
> mainly above threshold.

**Setup.**
- **Seeds:** first seed 11000, 50 worlds. Each (world, manifesto point, k) has 10 cascade trials. A trial salts the
  `workers` and `cascade` streams (`x_streams.SaltedStreams`); every other stream stays the world's own.
- **Game:** N = 10⁴, T = 5, ₫500, ₫20 workers. k ∈ {1, 3, 5, 10} workers are bought in round 1 in the target
  region, with nothing else bought. A k = 0 run gives the no-worker baseline. The target region is each world's
  largest region. Full game dynamics run through `x_batch.run_batch`: one IC hop per round, the social step,
  pinned workers and frozen zealots.
- **Manifesto sweep (plan):** b(θ) = (1 − θ)·b_far + θ·b_ideal for 9 values of θ in [0, 1]. b_ideal is the target
  region's mean v(0); b_far is the archetype mean farthest from it in TV distance.
- **Theory per world and point:**
  - q_j = γ_j·P_j(b) on v(0), with q = 0 for zealots.
  - The weighted operator λ_max(A D_q) = λ_max(D_q^½ A D_q^½), by batched GPU power iteration. All 50 worlds ×
    36 columns run in one sparse mat-mul per step. It is computed for the whole graph and restricted to the target
    region.
  - The naive estimates, regional (k̄_rr·E_r[q], k̄·E_r[q], λ_max(A_rr)·E_r[q]) and global (k̄·E[q], λ_max(A)·E[q]).
  - Two cap-corrected and uncapped expected branchings from the realised hops: Σ_j [1 − (1 − q_j)^{eff_j}] per
    spreader, with eff_j = min(count_j, cap − attempts_j), and the same with eff_j = count_j (no cap).
- **Observed growth:**
  - **R_obs:** converts produced per convert-spreader over hops 2–5, pooled over the 10 trials. Each voter spreads
    for one hop.
  - **R_early:** hop-2 converts ÷ hop-1 converts.
  - **c₁/k:** worker → convert in hop 1.
- **Device and compute:** cuda:2 (RTX A5000); 500 games per `run_batch` call; wall time 156 s; peak memory 6.3 GB;
  mean GPU utilisation 60% (`provenance.compute`).
- **Checks:**
  - GPU power iteration vs SciPy `eigsh` on 3 worlds, at θ = 0.5 and at the λ ≈ 1 point: max relative error
    1.1·10⁻¹⁵ (`provenance.power_vs_scipy`).
  - CPU↔GPU parity (3 seeds, θ = 1, k = 5) passed, with tolerance max(0.01, 3·SE) and counts scaled by N
    (`provenance.parity`).
- **Deviations:**
  - **Addition: 8 extension points per world.** They lie on b(φ) = (1 − φ)·(vertex of the least-liked sector) + φ·b_far,
    with φ chosen so that λ_max(A D_q) ≈ 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0. The pilot showed that the plan's
    sweep never goes below λ_max(A D_q) ≈ 3: b_far is another region's archetype and is liked there. Without these
    points the threshold is never sampled. Targets 1.6 and 2.0 are not reached in some worlds, so their realised
    means are 1.48 and 1.79.
  - The target region is the largest region; the plan says only "one region".

**Results.** Source: reports/v3/x3.json. In the tables, CIs are 95% t-intervals over worlds.

*The observed growth near threshold.* At k = 10, R_obs is close to λ_max(A D_q) (`extension_cells.target*_k10`):

| target λ_max(A D_q) | realised λ_max(A D_q) | R_obs | uncapped expected branching | distinct converts |
|---|---|---|---|---|
| 0.5 | 0.50 | 0.46 [0.40, 0.52] | 0.52 | 3.4 |
| 0.7 | 0.69 | 0.70 [0.65, 0.75] | 0.76 | 6.4 |
| 0.85 | 0.83 | 0.83 [0.78, 0.89] | 0.90 | 9.7 |
| 1.0 | 0.97 | 0.97 [0.92, 1.03] | 1.05 | 14.4 |
| 1.15 | 1.10 | 1.11 [1.05, 1.17] | 1.18 | 21.3 |
| 1.3 | 1.23 | 1.23 [1.16, 1.30] | 1.30 | 30.6 |
| 1.6 | 1.48 | 1.45 [1.35, 1.54] | 1.55 | 57.3 |
| 2.0 | 1.79 | 1.71 [1.59, 1.83] | 1.87 | 117.7 |

*Where R_obs crosses 1.* Points are pooled over all worlds, points and k (`binned_by_lambda_ADq`,
`threshold.R_obs`):
- **Binned by λ_max(A D_q):**
  - [0.85, 1.0): R_obs 0.78 [0.73, 0.82], with 26% of points above 1.
  - [1.0, 1.15): 0.99 [0.94, 1.04], with 62% above 1.
  - [1.15, 1.3): 1.12 [1.07, 1.17], with 84% above 1.
- **Logistic fit of P(R_obs > 1) on log x** (3,062 points). The crossing x* and slope are:

  | Estimator | Crossing x* | Slope |
  |---|---|---|
  | λ_max(A D_q) | **0.94** | 2.56 |
  | k̄·E[q] | 0.48 | 2.37 |
  | λ_max(A)·E[q] | 0.53 | 2.34 |
  | Regional naive and regional weighted | 0.025–0.035 | 0.49–0.52 |

  All Spearman tests have Holm p < 10⁻²³.
- **R_early** (`threshold.R_early`): λ_max(A D_q) has the highest Spearman ρ, 0.945. It crosses at 1.39, with
  slope 7.7. The global naive estimates cross at 0.75 and 0.84, with slopes 5.9 and 5.3. Its log-RMSE is the lowest:
  0.38 vs 0.42–0.49.

*The plan's sweep (k = 5)* (`cells.theta<i>_k5`). Every point is far above threshold: λ_max(A D_q) ≥ 4.6.

| θ | λ_max(A D_q) | λ_max(A_rr D_q,r) | k̄_rr·E_r[q] | R_obs | distinct converts | cross-region share | cap binding | Δ expected share vs k = 0 | win |
|---|---|---|---|---|---|---|---|---|---|
| 0 (far) | 4.58 | 0.52 | 0.45 | 2.03 [1.79, 2.26] | 936 | 0.85 | 0.28 | 0.026 | 0.00 |
| 0.25 | 5.06 | 1.45 | 1.32 | 2.20 | 1,641 | 0.74 | 0.46 | 0.063 | 0.08 |
| 0.5 | 5.50 | 3.67 | 3.45 | 1.79 | 2,630 | 0.56 | 0.74 | 0.119 [0.112, 0.126] | 0.38 |
| 0.625 | 6.26 | 5.18 | 4.90 | 1.49 | 2,965 | 0.49 | 0.83 | **0.120** [0.114, 0.126] | 0.59 |
| 0.75 | 7.08 | 6.56 | 6.23 | 1.31 | 3,078 | 0.43 | 0.86 | 0.105 | **0.64** |
| 1 (ideal) | 7.95 | 7.83 | 7.46 | 1.20 [1.13, 1.26] | 2,763 | 0.30 | 0.83 | 0.075 [0.066, 0.083] | 0.26 |

- **Cap binding vs threshold** (`cap_binding_vs_threshold`): below λ_max(A D_q) = 1, 0.6% [0.5%, 0.7%] of targeted
  voters reach the cap (768 points). Above it, 41.5% [40.1%, 42.8%] do (2,632 points). Mann–Whitney p = 5·10⁻³⁰¹.
  By bin, cap binding rises from 1–2% at λ ≈ 1–1.6 to 45% at 3–5 and 82% above 10.
- **The cap-corrected expected branching** tracks R_obs almost exactly. Its log-RMSE is 0.12 and its crossing is at
  0.985 (`threshold.R_obs.estimators.R_exp_capped`). The uncapped version overestimates R_obs by 40% on average
  (log ratio −0.34). Above threshold, the cap and depletion are what pull the realised growth below λ_max(A D_q):
  R_obs is 1.84 at λ = 5–10 and 1.18 above 10.
- **Reach saturates in k** (`cells.theta8_k*`, θ = 1). Distinct converts are 2,489, 2,718, 2,763 and 2,772 for
  k = 1, 3, 5 and 10. Δ expected share is 0.059, 0.072, 0.075 and 0.076.
- **Drift-back** (`cells.*.drift_final_P_by_steps`): converts start at P(b, b) = 0.971.
  - With the ideal manifesto (θ = 1), they keep P ≈ 0.956 after 5 social steps.
  - With the far manifesto (θ = 0), they fall to 0.88 after 5 steps. Their friends disagree with b, so a convert's
    vote is partly lost when the conversion happens early.
- **Cross-region spread:** 85% of the converts lie outside the target region at θ = 0, and 30% at θ = 1.

Figures:
- ![X3 threshold](..figures/x3_threshold.png)
- ![X3 sweep](..figures/x3_sweep.png)

**Theory vs observation. Confirmed (with one qualification).**
- **"Growth crosses 1 near λ_max(A D_q) ≈ 1": confirmed.**
  - The binned R_obs crosses 1 in [1.0, 1.15), and the logistic crossing is at 0.94.
  - Near threshold the weighted operator predicts the growth almost one for one: R_obs 0.97 at λ 0.97, and 1.23 at
    λ 1.23.
- **"More sharply than the naive estimates": confirmed for location, marginal for slope.**
  - The global naive estimates put the threshold at about half the right value (crossings 0.48–0.53).
  - The regional ones are useless (0.03): the cascade leaves the region.
  - The slope of the weighted estimator is only slightly steeper (2.56 vs 2.34–2.37).
  - For the first convert generation (R_early) the weighted crossing is at 1.39. The first hop starts in the target
    region, where q is lower than along the Perron vector, so early growth lags the asymptotic rate.
- **"The cap binds mainly above threshold": confirmed** (0.6% vs 41.5%).

**What it means.**
- **For the game design:**
  - At the default prices and k̄ = 30, any manifesto that a region's archetype does not reject is far above
    threshold: λ_max(A D_q) ≥ 4.6 on the whole sweep.
  - The 3-attempt cap, not the threshold, is what limits reach. This matches X23, where the cap is the main dial.
  - A threshold game ("will my cascade take off?") exists only for manifestos that nobody likes.
- **For a player:**
  - The worker value peaks for a manifesto between the region's ideal and a broader one (θ ≈ 0.5–0.75): Δ share
    0.12 vs 0.075 at the ideal.
  - An "ideal" regional manifesto saturates its own region and stops (cross-region share 30%, cap binding 83%).
  - One or three workers already reach most of what ten do (2,489 vs 2,772 converts at θ = 1), so buy few workers
    early.

**Data.** reports/v3/x3.json, reports/v3/raw/x3.jsonl (game rows plus one `kind: point` row per (world, point, k)),
log reports/v3/logs/x3.log. Command:
`OMP_NUM_THREADS=4 python experiments/x3_cascade.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x3.json`.

### X4 — Worker timing and placement

**Status:** done. Buying early dominates (front − late +0.12); placement ≤ 0.010; degree picking ≤ 0.006. Confirmed.

**Question.** At equal worker spend, how much is "buy early" worth? Does the choice of region matter more than the
number of workers? Would picking workers by degree centrality, as in the source document, change outcomes? If it
did, the random pick in force would hide an important skill.

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> buying early is worth more (more hops); centrality picking gives a modest gain under k̄ = 30 ER
> blocks (low degree variance).

**Setup.**
- **Seeds:** first seed 11000, 50 paired seeds.
- **Game:** N = 10⁴, T = 5, ₫500, ₫20 workers, ORACLE manifesto, workers only (no media).
- **Timing × placement:** a fixed total of K workers (K = 4 or 10, i.e. ₫80 or ₫200) is bought with one of three
  timings and one of three placements (`x_policies.WorkerTiming`):
  - **timing:** "front" (round 1), "even" (spread over rounds 1–5) or "late" (round 5);
  - **placement:** the "largest" region, the "best_matched" region (ORACLE: highest true mean P when buying), or
    "proportional" to region size.
- **Pick-rule ablation (experiment-side, outside the game rules):** the `pick_fn` hook of `x_batch.run_batch`
  replaces R22 for these arms only. The rules are:
  - **random:** a random eligible voter (the rule in force);
  - **sample10:** the highest degree in a 10% random sample;
  - **top_degree:** the true highest degree.

  Played at K = 10 front/proportional, K = 10 even/proportional, and K = 4 front/largest. The "random" `pick_fn`
  reproduces the library pick bit for bit (`random_pick_fn_equals_library: true`).
- **Device and compute:** cuda:2; B = 50 worlds per call. Wall time 48 s (`provenance.compute`).
- **Parity:** CPU↔GPU on 3 seeds passed (`provenance.parity`). The tolerance is max(0.01, 3·SE, 5% of the mean),
  because a fixed 0.01 is too tight for convert counts.
- **Deviation:** the plan fixes only "equal total spend"; two spend levels are run.

**Results.** Source: reports/v3/x4.json (`arms`, `tests`, `spread`). Values are mean Δ expected share vs no campaign
[95% CI]; no campaign has E = 0.479 and wins 0.44.

| K | timing | largest | best_matched | proportional |
|---|---|---|---|---|
| 4 | front | 0.116 [0.108, 0.124] | **0.121** [0.115, 0.127] | 0.119 |
| 4 | even | 0.091 | 0.101 | 0.091 |
| 4 | late | 0.0009 | 0.0008 | 0.0009 |
| 10 | front | 0.123 | 0.124 | **0.124** [0.119, 0.130] |
| 10 | even | 0.112 | 0.116 | 0.115 |
| 10 | late | 0.0022 | 0.0020 | 0.0023 |

- **Timing** (18 tests, Holm):
  - Front − even: +0.008 to +0.027, all Holm p ≤ 2·10⁻⁴.
  - Front − late: +0.115 to +0.122.
  - Even − late: +0.091 to +0.114. These have Holm p 3·10⁻¹⁴, and front-loading wins 19–20 games that late buying loses
    (McNemar).
  - Late workers get one hop (27–79 converts vs 4,268–4,528 when front-loaded).
- **Placement** (18 tests): every difference is ≤ 0.010. Only two negligible late-arm differences survive Holm
  (best_matched − proportional, −0.0001 and −0.0003, Holm p 0.03). The range across placements at the best timing is 0.0049 (K = 4) and
  0.0013 (K = 10).
- **Amount:** K = 10 − K = 4, front-loaded: +0.004 to +0.007. Even: +0.015 to +0.023. Late: +0.0013.
- **Pick rule** (6 tests, Holm), top_degree − random:
  - K = 10 front: +0.0004 [−0.0011, 0.0018] (Holm p 0.30).
  - K = 10 even: +0.0062 [0.0023, 0.0102] (Holm p 5·10⁻⁵).
  - K = 4 front/largest: +0.0062 [0.0021, 0.0102] (Holm p 0.09).
  - sample10 is the same as top_degree within 0.001.
  - The workers' mean degree is 30–34 for random, 44–50 for sample10 and 50–56 for top_degree. No pick rule
    changes a single win after Holm (McNemar Holm p = 1).

Figures:
- ![X4 timing × placement](..figures/x4_timing_placement.png)
- ![X4 pick rule](..figures/x4_pick_rule.png)

**Theory vs observation. Confirmed.**
- **"Buying early is worth more": confirmed, and the effect is large.** All workers in round 1 beat the even
  spread by 0.008–0.027 and the last round by 0.12. A round-5 worker is almost worthless (+0.001–0.002).
- **"Centrality picking gives a modest gain": confirmed.**
  - At most +0.006, even though the picked workers have 1.6× the degree.
  - Front-loaded at K = 10 there is no gain at all: the cascade already saturates (X3: reach saturates in k).
  - k̄ = 30 ER blocks have low degree variance (top-degree workers have 50–56 friends against about 30), so central nodes are only slightly
    better seeds.

**What it means.**
- **For the game design:** the random-pick simplification (R22) hides almost no skill. Replacing it with the source
  document's "highest degree in a 10% sample" would change E by ≤ 0.007 and no wins. Region choice is also almost
  irrelevant at the default k̄ = 30, because cascades cross regions (X3: 30–85% of converts are outside the target
  region).
- **For a player:** *when* you buy matters about 20–100× more than *where* or *how many*. Buy all your workers in
  round 1. Four front-loaded workers (₫80) already get 94–97% of what ten do (derived: Δ(K = 4, front) / Δ(K = 10, front) per placement, `arms.*.delta_vs_none`).

**Data.** reports/v3/x4.json, reports/v3/raw/x4.jsonl (1,400 rows), log reports/v3/logs/x4.log. Command:
`OMP_NUM_THREADS=4 python experiments/x4_timing_placement.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x4.json`.

### X11 — Worker × media interaction

**Status:** done. Interaction positive but tiny; "larger in P3" not supported; "media first" refuted.

**Question.** Do workers and media reinforce each other (supermodularity)? If they did, it would explain why
"split" dominates at cheap worker prices. Two mechanisms could cause it:
- media raises S, so P and then the next round's conversion chance γP go up;
- pinned workers raise their friends' P, which helps media turn them into votes.

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> the interaction is positive, and larger in canvassing-receptive regions; "media first, then
> workers" beats the reverse.

**Setup.**
- **Seeds:** first seed 11000, 100 paired seeds.
- **Game:** N = 10⁴, T = 5, ₫500, ORACLE manifesto, worker fees ₫20 (the game) and ₫100.
- **Units:** every seed is played once per target region r (all four regions), giving 400 units. All spending
  goes to r only.
- **Arms per unit:**
  - **2×2 factorial in round 1:** none; workers (k workers in r); media (one airing of r, cost m_r = ₫50·4·n_r/N);
    both.
  - **Matched spend:** k = max(1, round(m_r / fee)).
  - **Sequencing:** media in round 1 then the k workers in round 2 ("media→workers"), the reverse
    ("workers→media"), and both in round 1.
- **Interaction:** I = Δ(both) − Δ(workers) − Δ(media), where Δ = expected share minus "none" in the same world
  with the same streams. The overall CI uses seed-level means (the 4 regions of a seed averaged). The
  per-personality CIs are cluster bootstraps over seeds.
- **Device and compute:** cuda:2, one `run_batch` call of 400 games per arm. Wall time 76 s; peak memory 5.2 GB;
  mean GPU utilisation 38% (`provenance.compute`).
- **Parity (CPU↔GPU, 3 seeds, "both" at ₫20, largest region):** passed. Expected share 0.670 vs 0.673
  (`provenance.parity`).
- **Deviations:**
  - Matched spend is approximate. At ₫20, k = 2.5 workers on average (₫50.1) against a mean airing of ₫50. At ₫100,
    k = 1 (₫100) against ₫50 (`arms.*.spent_workers`).
  - All four regions are targeted in turn.

**Results.** Source: reports/v3/x11.json. Values are mean [95% CI].

| | ₫20 | ₫100 |
|---|---|---|
| Δ workers | 0.104 [0.098, 0.110] | 0.079 [0.073, 0.085] |
| Δ media | 0.0092 [0.0090, 0.0095] | 0.0092 (same arm) |
| Δ both | 0.115 [0.109, 0.120] | 0.090 [0.084, 0.096] |
| **Interaction I** | **+0.0011 [0.0007, 0.0015]** (Holm p 2·10⁻⁶) | **+0.0016 [0.0012, 0.0020]** (Holm p 5·10⁻¹⁰) |
| Units with I > 0 | 64% | 67% |
| Median I / (Δ workers + Δ media) | 0.7% | 0.8% |

Sources: `interaction.<fee>` and `tests.interaction`.

**By personality of the targeted region** (`by_personality.<fee>`; 100 units each; cluster-bootstrap CI; Holm over
10 tests):

| Personality | I at ₫20 | Holm p | I at ₫100 | Holm p | Δ media alone |
|---|---|---|---|---|---|
| P0 socially driven | 0.0006 [0.0001, 0.0012] | 0.14 | 0.0010 [0.0005, 0.0015] | 0.017 | 0.0064 |
| P1 media-savvy | **0.0016** [0.0010, 0.0022] | 1.5·10⁻⁵ | 0.0021 [0.0015, 0.0027] | 1.2·10⁻⁸ | 0.0152 |
| P2 independent | 0.0008 [0.0006, 0.0011] | 5·10⁻¹⁰ | 0.0009 [0.0007, 0.0012] | 5·10⁻¹² | 0.0058 |
| P3 receptive to canvassing | 0.0015 [0.0004, 0.0026] | 0.14 | **0.0024** [0.0011, 0.0038] | 0.031 | 0.0096 |

**Sequencing** (`sequencing.<fee>`; seed level; Holm over 6 tests, all Holm p ≤ 4·10⁻¹³):
- **Media→workers minus workers→media:**
  - ₫20: −0.041 [−0.044, −0.038]. Workers→media is better in 100/100 seeds. The discordant wins are 40 vs 1 (McNemar
    p 4·10⁻¹¹).
  - ₫100: −0.042 [−0.046, −0.038], with workers→media better in 99/100 seeds.
- **Both in round 1 minus workers→media:** +0.0007 [0.0005, 0.0009] at ₫20 and +0.0009 at ₫100. Discordant wins
  are 1 vs 0.
- **Mechanism 1 is real but small.** One airing raises the mean γP of region r at the start of round 2 by
  +0.017 [0.016, 0.018], from 0.158 to 0.175 (`interaction.20.00.q_gain_from_media_round1`, `arms.*`). But
  buying the workers one round later costs a whole cascade hop: converts fall from 3,240 to 2,123 (`arms.*.converts_total`).

Figures:
- ![X11 interaction](..figures/x11_interaction.png)
- ![X11 sequencing](..figures/x11_sequencing.png)

**Theory vs observation. Partial.**
- **"The interaction is positive": confirmed, but it is tiny.** I = +0.0011 at ₫20 and +0.0016 at ₫100. Both are
  significant, but they are under 1% of the joint effect.
- **"Larger in canvassing-receptive regions": not supported.**
  - P3 has the largest point estimate at ₫100 (0.0024), but its CI overlaps P1.
  - At ₫20, P1 (media-savvy) is the largest (0.0016), and P3's Holm p is 0.14.
  - The interaction tracks the media effect (largest for P1) at least as much as γ.
- **"Media first, then workers beats the reverse": refuted.** The reverse is better by 0.041 (100/100 seeds at ₫20). Timing
  of the workers (one more hop) outweighs the conversion boost that media gives them.

**What it means.**
- **For the game design:**
  - Workers and media are close to additive. The dominance of "split" and of worker-heavy play at ₫20 (X0) is
    not a complementarity effect. It comes from the raw worker effect: at matched spend, workers add 0.104 against
    0.009 for media, about 11×.
  - At ₫100, one worker (₫100) still adds 0.079 against 0.009 for a ₫50 airing: about 4.3× more per ₫. At ₫20
    the ratio is about 11× per ₫.
  - A balance fix must change the worker effect itself (fee, cap; see X23), not the interaction.
- **For a player:** buy workers first and as early as possible. Media in the same round is a small free bonus
  (+0.0007 over workers then media). Never delay workers to "prepare" a region with media.

**Data.** reports/v3/x11.json, reports/v3/raw/x11.jsonl (4,000 rows), log reports/v3/logs/x11.log. Command:
`OMP_NUM_THREADS=4 python experiments/x11_worker_media.py --seeds 100 --first-seed 11000 --device cuda:2 --out reports/v3/x11.json`.

### X18 — Submodularity of the worker value curve

**Status:** done. Strongly submodular (1 worker = 62% of the plateau); no early supermodularity near threshold.

**Question.** Influence-maximisation guarantees (greedy ≥ (1 − 1/e)·OPT) need diminishing returns. Pinned
stubborn workers and a conversion chance that depends on b could break that. Is the worker value curve
submodular in k? How far is the game's random pick from a node-level greedy (CELF) reference?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> roughly submodular in k, with possible early *super*-modularity in regions near threshold.

**Setup.**
- **Seeds:** first seed 11000, 50 paired seeds; CELF on the first 10.
- **Game:** N = 10⁴, T = 5, ORACLE manifesto, ₫20 workers.
- **Value curve:** k workers are bought in round 1 in one region, for each of the 4 regions (indexed by size rank),
  with k ∈ {0, 1, …, 10, 15, …, 40}.
  - The library pick takes the first k of one random permutation, so the worker sets are nested in k (checked:
    `nested_check: true`). All other randomness is common, so each curve is within-world.
  - The marginal gain per worker is g(k_i) = [E(k_i) − E(k_{i−1})]/(k_i − k_{i−1}).
  - An adjacent pair violates diminishing returns "beyond noise" when the 95% paired bootstrap CI of mean(g_{i+1} − g_i)
    lies above 0 (4,000 resamples; the pooled group resamples seeds).
  - Units are also split by the regional weighted threshold λ_max(A_rr D_q,r) (X3's operator, GPU power iteration)
    to look for early supermodularity near threshold.
- **CELF (experiment-side, outside the game rules):**
  - Lazy greedy on the true graph, node by node, over every eligible voter of the largest region, for k ≤ 10.
  - Objective: mean expected share over 4 salted cascade streams.
  - The 64 stale candidates with the largest bounds are re-evaluated per GPU batch: about 10,000 evaluations × 4
    streams per seed, batch 1,024 games.
  - The random pick (R22) and the top-degree pick are scored on the same objective.
- **Device and compute:** cuda:2. Wall time 576 s, of which CELF took about 8 min; peak memory 12.5 GB; mean GPU
  utilisation 48% (`provenance.compute`).
- **Parity (CPU↔GPU, 3 seeds, k = 10, largest region):** passed, with tolerance max(0.01, 3·SE, 5%)
  (`provenance.parity`).
- **Deviations:**
  - The treasury is ₫1000 so that 40 workers (₫800) are affordable. Nothing else is bought, so it has no other
    effect.
  - All 4 regions are used per seed.

**Results.** Source: reports/v3/x18.json.

*Value curve* (`curve.<group>`; Δ = E(k) − E(0), all 200 units):

| k | 1 | 2 | 3 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|---|
| Δ expected share | 0.077 | 0.099 | 0.107 | 0.116 | 0.123 | 0.1245 | 0.1245 |
| marginal gain per worker | 0.077 | 0.022 | 0.0088 | 0.0032 | 0.00047 | 0.00005 | 0.00001 |

- The first worker gives 62% of the plateau, and k = 5 gives 93%. The curve is flat from k ≈ 15.
- The smallest region (size rank 3) rises more slowly: 0.067 at k = 1 and 0.122 at k = 10.
- **Adjacent-pair test** (`violations`, `pairs`; 15 pairs per group):
  - Pooled: 1 violating pair (k 30 → 35, where the gains are −0.00003 vs +0.00002, i.e. noise around zero) and
    7 strictly decreasing pairs. The others have CIs that include 0.
  - By size rank: 1, 0, 0 and 0 violating pairs.
- **Near threshold** (`near_threshold`):

  | λ_max(A_rr D_q,r) | units | g(1) | g(2) | g(2) − g(1), 95% CI | units with g(2) > g(1) |
  |---|---|---|---|---|---|
  | < 1.5 | 69 | 0.055 | 0.025 | [−0.045, −0.016] | 26% |
  | 1.5–3 | 46 | 0.067 | 0.030 | [−0.050, −0.023] | 20% |
  | ≥ 3 | 85 | 0.100 | 0.016 | [−0.094, −0.073] | 7% |

  The regional λ has mean 3.2 and minimum 0.18. Even the regions where the cascade is subcritical *inside* the region
  show diminishing returns on average: the cascade leaves the region (X3).

*CELF reference* (`celf_summary`; Δ over no workers, 10 seeds; CELF − random tested with Wilcoxon):

| k | CELF | top degree | random (R22) | random / CELF | CELF − random, 95% CI |
|---|---|---|---|---|---|
| 1 | 0.118 | 0.111 | 0.088 | 0.75 | [0.006, 0.053] |
| 2 | 0.123 | 0.118 | 0.114 | 0.93 | [0.003, 0.015] |
| 3 | 0.124 | 0.121 | 0.118 | 0.95 | [0.003, 0.010] |
| 5 | 0.125 | 0.121 | 0.121 | 0.96 | [0.003, 0.006] |
| 10 | 0.126 | 0.121 | 0.121 | 0.96 | [0.004, 0.006] |

- CELF − random has Wilcoxon p = 0.002 at every k: CELF wins on all 10 seeds.
- CELF's own marginal gains fall from 0.118 to 0.00004 (`celf_marginal_gains_mean`). 15 of 90 adjacent CELF gains
  increase, all at gains < 0.002.
- CELF does not pick high-degree voters: mean degree 33.7 vs 35.2 for the random picks. It picks voters whose
  neighbourhoods convert well.

Figures:
- ![X18 curve](..figures/x18_curve.png)
- ![X18 CELF](..figures/x18_celf.png)

**Theory vs observation. Partial: submodular confirmed, early supermodularity refuted.**
- **"Roughly submodular in k": confirmed, strongly.** Marginal gains fall by more than three orders of magnitude
  from k = 1 to k = 20. Only 1 of 15 adjacent pairs rises beyond noise, and that is at gains of about 10⁻⁵.
- **"Possible early super-modularity in regions near threshold": refuted.** g(2) < g(1) in every λ group, including
  λ_region < 1.5 (CI [−0.045, −0.016]). Super-additive units are more common near threshold (26% vs 7%), but they
  are not the majority. A region that is subcritical by itself is not subcritical for the cascade, which spreads
  through the whole graph (X3: λ_max(A D_q) ≫ 1 for the ORACLE b).
- **The greedy guarantee is not needed in practice.** The game's random pick already reaches 93–96% of the CELF
  value from k = 2 on, against the (1 − 1/e) ≈ 63% worst case.

**What it means.**
- **For the game design:**
  - Worker value saturates after about 5 workers per region (93% of the plateau). Extra workers are wasted money,
    which caps the damage of cheap workers.
  - The random pick (R22) costs a player at most about 0.005 in E against a full-information greedy for k ≥ 2.
    The pick rule is not a hidden skill (agrees with X4).
- **For a player:** one worker gives about 60% of what any number can. Two or three per target region capture
  80–86%. Spend the rest on other regions, media or surveys.

**Data.** reports/v3/x18.json, reports/v3/raw/x18.jsonl (3,400 curve rows + 10 CELF records), log
reports/v3/logs/x18.log. Command:
`OMP_NUM_THREADS=4 python experiments/x18_submodularity.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x18.json`.

## 6. Strategies and optimisation

### X6 — Strategy tournament

**Status:** done. II > I > III confirmed; BayesPlanner beats the heuristic ORACLE; against the X13 bound the price of not knowing is 0.033. Partly confirmed.

**Question.** How do the scripted personas compare, and how much does the manifesto optimiser matter, when every
player sees only its observation? The personas are I People Pleaser, II Balanced (BayesPlanner) and III
Strong-Arm, against the baselines. How far is the best player from the ORACLE and from the X13 upper bound?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> II > I > III; the price of not knowing is ≤ 0.015 E.

**Setup.**
- **Seeds:** first seed 12000, 100 paired seeds. The same electorates as X13.
- **Game:** N = 10⁴, T = 5, ₫500, workers ₫20, queries ₫10.
- **Players:** every player plays full `GameEngine` games on observation dicts only. Engine, posterior worlds and
  rollouts run on cuda:0; BayesPlanner runs fully on the GPU since the D12 fix.
  - **uniform / top3 / random:** library baselines.
  - **I `people_pleaser`** (`experiments/x_players.py`):
    - BayesPlanner's survey and posterior;
    - manifesto: "blend my hunches" (R20) with the posterior MAP archetype per region, over the 1–2 largest regions
      that hold a majority;
    - allocation: even pace, media in the coalition within half the cap, the rest on workers in the coalition.
  - **II `bayes`:** BayesPlanner as in the library (manifesto by projected gradient = PG; rollout planner).
  - **Manifesto-optimiser arms, all with II's survey and rollout planner:**
    - `bayes_admm`: consensus-ADMM manifesto on the same posterior worlds, with residuals recorded;
    - `bayes_blend`: player I's blend manifesto.
  - **III `strong_arm`:** fixed b (0.6 on Education, 0.1 elsewhere), no queries, the split pattern at 1.5× pace.
    `strong_arm_random` is the same with a random b (flat Dirichlet keyed by seed).
- **ORACLE (reference, not a player):**
  - `oracle_manifesto` + `balance.policy_planner` on the true state, through `x_batch.run_batch` on the canonical
    streams (B = 100).
  - **The ORACLE is a heuristic.** Its manifesto ignores the campaign, and X13 shows its greedy planner is 0.044 E
    below a plan search.
- **X13 upper bound:** X13's search with the ORACLE manifesto, mean E over 20 fresh streams per seed
  (reports/v3/x13.json).
  - The ORACLE's single canonical-stream E agrees with its 20-stream mean (difference −0.00002 [−0.0005, 0.0004],
    `x13_bound.oracle_single_stream_vs_E20`).
  - So the bound and the single games are comparable.
- **Device and time:** cuda:0, 2 processes.
  - Wall time 533 s (8.9 min); the pilot estimated 9.0 min.
  - Mean GPU utilisation 31% (max 99%): the games are bound by Python orchestration.
  - Peak memory 75 MB per child; 2.7 GB used on the device at most (`provenance.compute`).
- **Parity (CPU engine + CPU players vs GPU, seeds 12000–12002):** passed for bayes (0.632 vs 0.635),
  people_pleaser (0.526 vs 0.529) and strong_arm (0.0984 vs 0.0984), with tolerance 0.02 (`provenance.parity`).
- **Deviations:** device cuda:0 instead of cuda:2, which is used by the Phase 2 agent. The engine games run fully on
  the GPU; the X7 CPU-manifesto workaround is no longer needed (D12).
- **Traces for X8:** raw/x6.jsonl has one row per (seed, player), 1,000 rows. Each `trace` holds:
  - the true E after each round, and the margin E − 0.5;
  - the public polls (estimate, MoE);
  - the treasury after each round;
  - every query, the manifesto and every allocation with its cost.

**Results.** Source: `players` in reports/v3/x6.json. The gaps are paired per seed: ORACLE − player, and X13 bound −
player.

| Player | Win rate [95% CI] | Mean E [95% CI] | Gap to ORACLE | Gap to X13 bound | Queries | Spend W / M (₫) |
|---|---|---|---|---|---|---|
| ORACLE (reference) | 0.95 [0.89, 0.98] | 0.628 [0.613, 0.643] | — | +0.044 | 0 | 322 / 168 |
| II + ADMM b | 0.99 [0.95, 1.00] | 0.649 [0.635, 0.662] | −0.021 [−0.024, −0.018] | +0.023 [0.020, 0.026] | 1.87 | 208 / 267 |
| **II BayesPlanner (PG)** | 0.98 [0.93, 0.99] | 0.639 [0.625, 0.654] | −0.011 [−0.014, −0.009] | +0.033 [0.030, 0.035] | 1.87 | 216 / 258 |
| top-3 | 0.85 [0.77, 0.91] | 0.602 [0.585, 0.619] | +0.026 | +0.070 | 0 | 269 / 222 |
| II + blend b | 0.84 [0.76, 0.90] | 0.597 [0.581, 0.612] | +0.031 | +0.075 | 1.87 | 216 / 257 |
| **I People Pleaser** | 0.73 [0.64, 0.81] | 0.560 [0.544, 0.576] | +0.068 [0.061, 0.074] | +0.112 | 1.87 | 408 / 69 |
| uniform (no spending) | 0.09 [0.05, 0.16] | 0.389 [0.373, 0.404] | +0.239 | +0.283 | 0 | 0 / 0 |
| III Strong-Arm, random b | 0.03 [0.01, 0.08] | 0.157 [0.128, 0.186] | +0.471 | +0.515 | 0 | 280 / 210 |
| random | 0.02 [0.01, 0.07] | 0.146 [0.121, 0.171] | +0.481 | +0.526 | 2.27 | 183 / 222 |
| **III Strong-Arm, fixed b** | 0.00 [0.00, 0.04] | 0.088 [0.077, 0.099] | +0.539 | +0.584 | 0 | 280 / 210 |

No player had an action rejected (`rejected_actions` = 0).

Mean decision latency (`decision_seconds`):
- BayesPlanner: 0.19 s;
- the ADMM variant: 0.70 s;
- the blend variant: 0.14 s.

**Paired tests** (reports/v3/x6.json → `tests`; Holm within each family):
- **P. Ordering (II − I, I − III, II − III):**
  - II − I = +0.079 [0.073, 0.085] E. II is better on 100/100 seeds (Holm p = 1e-17). On wins II gains 25 seeds and
    loses 0 (McNemar Holm p = 6e-8).
  - I − III = +0.472 [0.452, 0.491]. On wins, 73 gained and 0 lost.
  - II − III = +0.551 [0.533, 0.569]. On wins, 98 gained and 0 lost.
- **M. Manifesto optimisers** (all with the rollout planner), on game E and on the static share of the committed b
  on the anticipated true electorate (no campaign):
  - **ADMM − PG:** +0.0095 [0.0066, 0.0125] game E (ADMM better on 79/100 seeds, Holm p = 6e-11). The static share
    is the other way round: −0.0060 (Holm p = 1e-16). The wins are 1 vs 0 discordant (p = 1).
  - **PG − blend:** +0.043 [0.037, 0.049] game E and +0.041 static. On wins, 15 gained and 1 lost
    (Holm p = 5e-4).
  - **ADMM − blend:** +0.052 [0.046, 0.058] game E; on wins, 15 gained and 0 lost.
- **B. Against the baselines:**
  - BayesPlanner beats top-3 by +0.037 [0.031, 0.044] (13 wins gained, 0 lost; Holm p = 3e-16).
  - People Pleaser is **below** top-3: −0.042 [−0.050, −0.034]; 2 wins gained and 14 lost (Holm p = 9e-13).
  - The blend manifesto with the rollout planner ties with top-3 (−0.005, Holm p = 0.11).
- **ADMM residuals** (`admm_residuals`, 100 solves on 8 × 1,600 posterior voters, ρ = 5, δ = 0.02):
  - 94/100 solves stopped at tolerance 1e-6 within the 300-iteration limit, in 50 iterations on average
    [37, 63], taking 4.2 s.
  - Final primal residual: mean 0.0015, max 0.047. Final dual residual: mean 0.0032, max 0.113.
  - The six unconverged solves oscillate, as the library docstring warns for non-convex ADMM.
- **Price of not knowing:**
  - ORACLE − II = **−0.0115 [−0.0138, −0.0091]**. The observation-only BayesPlanner beats the full-information
    ORACLE heuristic (better on 86/100 seeds).
  - Against the X13 bound: bound − II = +0.0328 [0.0304, 0.0352].

Figures:
- ![X6 tournament](..figures/x6_tournament.png)
- ![X6 ADMM residuals](..figures/x6_admm_residuals.png)

**Theory vs observation. Partial.**
- **II > I > III: confirmed.** II − I = +0.079, I − III = +0.47 in E, both on 100/100 seeds (Holm p ≤ 1e-17).
  The win rates are 0.98 > 0.73 > 0.00.
- **"Price of not knowing ≤ 0.015 E": confirmed against the ORACLE, but for the wrong reason.**
  - The gap is negative (−0.011): the ORACLE heuristic is worse than the player that cannot see the world. It plays
    a greedy worker-heavy plan (₫322 on workers vs ₫216 for BayesPlanner).
  - Measured against the X13 search bound instead, the price is **+0.033 E** [0.030, 0.035]. That is above 0.015, so
    against a real upper bound the prediction is refuted.
- **Manifesto optimisers:**
  - Consensus-ADMM gives a *slightly worse* static manifesto but a *better* campaign (+0.0095 E). Its smoother
    surrogate (δ = 0.02) lands on a broader b, which converts better under the campaign. This is a first hint of
    X12's question.
  - The R20 blend is clearly worse than either optimiser (−0.04 to −0.05 E).

**What it means.**
- **For the design:**
  - Information-gathering plus rollout planning is worth about +0.25 E over doing nothing, and +0.04 over the free
    top-3 statistic. The epistemic layer and the planner both matter.
  - A fixed, voter-blind agenda (Strong-Arm) almost never wins (0/100 with b fixed).
  - "Gap to the ORACLE" should no longer be used as an optimality measure; use the X13 bound.
- **For a player:**
  - Survey a little (about 2 questions), optimise the manifesto rather than blending hunches, and plan the money.
  - People Pleaser's habit of buying mostly workers in the majority regions (₫408 of ₫496 on workers) costs about
    0.04 E against even the top-3 baseline.

**Data.** reports/v3/x6.json, reports/v3/raw/x6.jsonl (1,000 rows with per-round traces for X8), log
reports/v3/logs/x6.log. Command:
`OMP_NUM_THREADS=4 python experiments/x6_tournament.py --seeds 100 --first-seed 12000 --device cuda:0 --out reports/v3/x6.json`.

### X12 — Broad vs narrow ("viral") manifesto

**Status:** done. Joint manifesto and plan optimisation gains +0.008 to +0.011 E (inside the predicted range); receptivity does not predict the gain. Partly confirmed.

**Question.** The conversion chance is γ·P (R25). A manifesto close to one coalition's preferences could make its
cascades grow on their own, while a broad manifesto pleases many voters a little. Which wins? And should b be
chosen together with the campaign plan, rather than first?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> the best λ > 0 on seeds where some coalition region is canvassing-receptive. Joint optimisation
> gains a measurable +0.005–0.02 E over "b first, then plan" (to be measured).

**Setup.**
- **Seeds:** first seed 12000. 50 seeds for the λ interpolation (sweep) and 100 for the joint-optimisation arm
  (core). These are the same electorates as X6/X13.
- **Game:** N = 10⁴, T = 5, ₫500.
- **Manifestos:**
  - b₀ = the ORACLE manifesto.
  - For each 2- and 3-region coalition C with > 50% of voters, b_C = the size-weighted mean of the TRUE anticipated
    regional means. That gives 7 coalitions per seed in every seed (`coalitions_per_seed`).
  - b(λ) = (1 − λ) b₀ + λ b_C, with λ ∈ {0, 0.25, 0.5, 0.75, 1}.
- **Arms of the interpolation:**
  - (a) every (C, λ) is played by the ORACLE planner;
  - (b) for the best coalition C\* of each seed (highest mean ORACLE-planner E over λ > 0), every λ is also played by
    X13's plan search, with the same settings as X13.
  - C\* was a 3-region coalition on 49/50 seeds (`best_coalition_size_counts`).
- **Evaluation:** every game uses the same 20 fresh stream sets as X13 (`run_batch`). C\* is chosen on those same
  streams, so the ORACLE arm's λ > 0 numbers are slightly optimistic; the search arm is not affected by that choice.
- **Branching factor:** per region, λ_max(A_rr D_q) with q_j = γ_j P_j(v(0), b), zealots set to 0. It is the
  branching factor of the worker cascade inside region r, computed by power iteration on the GPU.
- **Joint arm:** start from b₀ and the ORACLE planner's executed plan (canonical streams). Then 3 alternations of:
  - (i) projected gradient on b (150 steps, lr 0.01) over an **expected-value campaign** with that plan fixed. The
    realised worker picks and the media schedule are kept. Conversions are replaced by 1 − (1 − γP)^E[count],
    media by its expected intensity, and workers stay pinned at b. It is differentiable in b.
  - (ii) re-plan with the ORACLE planner.
  - The final b_joint is played by the ORACLE planner and by the X13 search. It is compared with "b first, then plan":
    b₀ with the same planner. For the search, the b₀ baseline is the per-seed X13 result (same code, seeds and
    streams).
- **Device and time:** cuda:0, 2 processes.
  - Wall time 3,620 s (60.3 min); mean GPU utilisation 95%.
  - Peak memory 2.8 GB in the parent and 2.9 GB per child (`provenance.compute`, `provenance.child_peak_bytes_max`).
- **Parity (CPU↔GPU, seeds 12000–12002, 5 streams), passed:**
  - ORACLE E with b₀: 0.6190 vs 0.6191.
  - ORACLE E with a blend: 0.5713 vs 0.5744.
  - EV objective: 0.6904 vs 0.6906.
  - Mean λ_max: 2.5498 vs 2.5498.
  - Source: `provenance.parity`.
- **Deviations:**
  1. The search arm (b) ran for the best coalition per seed only: 5 λ points instead of 7 coalitions × 5 λ. The
     ORACLE-planner arm covers every coalition.
  2. The first launch was stopped after 21 minutes, with no output written. The ORACLE arm ran all of a seed's manifestos (each on 20 fresh
     streams) in one `run_batch` call, and the idle parent kept its pilot cache, so the GPU memory reached 24 GB and progress
     stalled. It was restarted with the ORACLE arm chunked (12 manifestos per call, OOM back-off) and the cache
     released between seeds. The chunked runner gives identical numbers (checked on CPU against unchunked
     `run_batch`).
  3. Device cuda:0 instead of cuda:2.

**Results — interpolation** (reports/v3/x12.json; E is the mean over 20 streams, then over the 50 seeds; 95%
t-CI):

| λ | ORACLE planner, C\* | ORACLE planner, all coalitions | X13 search, C\* | λ_max (max over C\* regions) | converts in C\* regions |
|---|---|---|---|---|---|
| 0 (b₀) | 0.6365 [0.6137, 0.6593] | — | 0.6806 [0.6583, 0.7029] | 5.74 | 3,613 |
| 0.25 | 0.6409 [0.6191, 0.6627] | 0.6375 | 0.6850 [0.6641, 0.7059] | 5.59 | 3,664 |
| 0.5 | **0.6423** [0.6213, 0.6633] | 0.6317 | **0.6866** [0.6666, 0.7067] | 5.39 | 3,697 |
| 0.75 | 0.6397 [0.6189, 0.6606] | 0.6198 | 0.6849 [0.6651, 0.7047] | 5.20 | 3,713 |
| 1 (b_C) | 0.6338 [0.6126, 0.6549] | 0.6020 | 0.6790 [0.6588, 0.6992] | 5.04 | 3,720 |

The sources are `oracle_arm_best_coalition_by_lambda`, `oracle_arm_all_coalitions_by_lambda`,
`search_arm_best_coalition_by_lambda` and `spectra_and_reach_by_lambda`.

- **Paired E(λ) − E(0)** (`tests_lambda_vs_0`, Holm within each arm):
  - ORACLE arm: +0.0044 [0.0025, 0.0063] at λ = 0.25 (Holm p = 4e-8) and +0.0058 [0.0029, 0.0086] at 0.5
    (p = 8e-5). At 0.75 it is +0.0033 (p = 0.25), and at 1 it is −0.0027 (p = 0.051).
  - Search arm: +0.0044 [0.0022, 0.0067] at 0.25 (p = 4e-4) and +0.0061 [0.0023, 0.0098] at 0.5 (p = 0.031). At
    0.75 and 1 it is not significant.
- **Best λ per seed:**
  - ORACLE arm: λ > 0 on 41/50 seeds (λ = 0: 9; 0.25: 15; 0.5: 19; 0.75: 5; 1: 2).
  - Search arm: λ > 0 on 36/50 (λ = 0: 14; derived: 50 − 14). Source: `*_best_lambda_counts`.
- **Averaged over all 7 coalitions,** narrowing only hurts, falling monotonically from 0.6375 at λ = 0.25 to 0.6020
  at λ = 1. Only the right coalition, and only half-way, helps.
- **Cascades:**
  - Every region is far above the cascade threshold. λ_max(A_rr D_q) is about 5–6 in every seed: 50/50 seeds have
    a supercritical region (`receptivity`).
  - Narrowing b *lowers* the largest branching factor (5.74 → 5.04 in the coalition, 6.23 → 5.49 over all
    regions). It moves converts slightly towards the coalition (3,613 → 3,720) and slightly away overall
    (4,507 → 4,417).
  - The R27 cap of 3 attempts, not λ_max, limits the reach.
- **Receptivity does not predict the gain:**
  - λ_max is 4.96 [4.35, 5.57] on seeds whose best λ > 0, against 5.42 [3.20, 7.63] on seeds whose best λ = 0.
  - Spearman of λ_max against the λ = 1 gain: ρ = 0.17 (p = 0.24).

**Results — joint optimisation** (100 seeds; `joint`):

| Planner | b first (b₀) | Joint b | Joint − b first [95% CI] | Seeds better | Holm p |
|---|---|---|---|---|---|
| ORACLE planner | 0.6277 | 0.6353 | **+0.0077** [0.0050, 0.0104] | 72/100 | 3e-7 |
| X13 plan search | 0.6719 | 0.6830 | **+0.0111** [0.0082, 0.0141] | 80/100 | 5e-13 |

- **Wins with the ORACLE planner:** win fraction 0.950 → 0.987.
- **Static E:** the static share of b on the anticipated electorate (no campaign) *falls* by −0.0183 [−0.0212,
  −0.0154]. Joint b is worse on every one of the 100 seeds if nobody campaigns.
- **Size of the change:** b moves by TV 0.058.
- **Alternations:** almost all of the gain comes from the first one.
  - The EV objective gains +0.0295, +0.0003 and +0.00002 in alternations 1–3.
  - The re-planned ORACLE plan rarely changes after the first.
- **The EV model:** it ranks seeds almost perfectly (Spearman 0.977 with the realised E for b₀) but is
  over-optimistic by +0.056 [0.053, 0.059]. Expected conversions spread the pull towards b more evenly than real,
  all-or-nothing conversions.

Figures:
- ![X12 interpolation](..figures/x12_lambda.png)
- ![X12 joint](..figures/x12_joint.png)

**Theory vs observation. Partial.**
- **"Best λ > 0 on receptive seeds": partial.**
  - The best λ is > 0 on most seeds: 41/50 with the ORACLE planner, 36/50 with the search (derived: 50 − 14). A half-way move towards
    the best 3-region coalition gains +0.006 E with either planner.
  - But "canvassing-receptive" does not separate seeds here: every region is far above threshold (λ_max ≈ 5), and
    λ_max does not predict the gain (ρ = 0.17, p = 0.24).
  - The fully narrow manifesto (λ = 1) is no better than b₀ (−0.003 to −0.002, not significant), so a "viral"
    narrow manifesto does not win.
- **Joint optimisation: confirmed.** The gain is +0.0077 [0.0050, 0.0104] with the ORACLE planner and +0.0111
  [0.0082, 0.0141] with the search planner. Both are inside the predicted +0.005–0.02.
- **The mechanism:** workers and converts are pinned at b. A campaign-aware b trades about 0.018 of static
  appeal for better conversion. This matches X6, where the ADMM manifesto had a lower static share but a higher game
  E.

**What it means.**
- **For the design:**
  - Manifesto and campaign interact as intended (R25): the best b depends on the campaign, by about +0.01 E.
  - Narrowing the manifesto is not a dominant strategy, because the attempt cap (R27) caps cascades well before
    the branching factor matters. The "viral manifesto" exploit the review worried about does not exist at the
    current rules.
- **For a player:**
  - Optimise b for the electorate you will face, then nudge it part-way (λ ≈ 0.25–0.5) towards your best 3-region
    coalition if you campaign heavily.
  - Do not go all the way to the coalition's taste.
- **Best result found:** the joint b with the X13 search, E = 0.683 (100 seeds). That is the best full-information
  benchmark in Phase 3 and a tighter upper reference than X13's 0.672.

**Data.** reports/v3/x12.json, reports/v3/raw/x12.jsonl (100 rows; 50 with the full (C, λ) grid), log
reports/v3/logs/x12.log. Command:
`OMP_NUM_THREADS=4 python experiments/x12_viral_manifesto.py --seeds 50 --joint-seeds 100 --first-seed 12000 --device cuda:0 --out reports/v3/x12.json`.

### X13 — ORACLE optimality: open-loop plan search

**Status:** done. The open-loop search beats the ORACLE planner by +0.044 E on 100/100 seeds; refuted (predicted ≤ 0.01 with a tail).

**Question.** How far is the greedy ORACLE planner from the best plan we can find with full information? The
answer gives a true upper bound for X6.

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> the search beats the ORACLE planner by ≤ 0.01 E on average, with a long tail of seeds where front-loading
> or concentration beats greedy by more.

**Setup.**
- **Seeds:** first seed 12000, 100 seeds (core).
- **Game:** N = 10⁴, T = 5, treasury ₫500, workers ₫20, media base ₫50. Electorates use the canonical CPU draw
  (R8), moved to the GPU.
- **Information:** full. The true electorate, with the ORACLE manifesto b₀ (`oracle_manifesto`) for both the search
  and the ORACLE.
- **Plan space:** `hires[t, r] ∈ ℕ` and `media[t, r] ∈ {0, 1}` over 5 rounds × 4 regions, with total cost ≤ treasury
  (integer cents, exact `media_cost` prices).
- **Search:**
  - **Cross-entropy method:** 256 plans per iteration, 10% elite (26 plans, derived: ⌈0.1 × 256⌉), 30 iterations, smoothing 0.7.
    - Distributions: a categorical over 0–25 hires per (t, r) and a Bernoulli airing per (t, r).
    - Over-budget samples are repaired by dropping random airings and workers.
    - Fresh streams every iteration.
    - Warm start: the executed plans of the ORACLE and the four X0 patterns.
    - Elitism: the top plan is kept between iterations.
  - **Then a steepest-ascent 1-move hill climb:** move, add or remove one worker, or toggle one airing, on 8 fixed
    streams.
  - **Simulator:** all candidates run in one batched `WorldBatch` call: C = 256 plans × K = 4 replicates of the
    true electorate, with common random numbers.
- **Re-evaluation (winner's curse):** the final plan, the ORACLE closed-loop policy and the ORACLE's executed plan
  replayed open-loop are all played with `x_batch.run_batch`, the engine round, on the **same 20 fresh stream
  sets** (`SaltedStreams`, salts 1–20, never seen by the search).
- **Device and time:** cuda:0 (RTX A5000), 2 processes.
  - Wall time 721 s (12.0 min); the pilot estimated 7.2 min.
  - Search time 11.7 s per seed.
  - Mean GPU utilisation 92% (max 100%).
  - Peak memory 0.27 GB in the parent and 2.0 GB per child.
  - All from `provenance.compute`, `search_seconds` and `provenance.child_peak_bytes_max` in reports/v3/x13.json.
- **Parity (CPU↔GPU, seeds 12000–12002):** passed, using a reduced search (64 plans × 6 iterations) and 5 fresh
  streams.
  - Search E: 0.6608 on CPU vs 0.6602 on GPU.
  - ORACLE E: 0.6190 vs 0.6191.
  - Gap: 0.0418 vs 0.0410, with tolerance 0.01 (`provenance.parity`).
- **Deviations:** device cuda:0 instead of cuda:2, which is used by the Phase 2 agent. Otherwise none: the seed
  count and the search settings are as planned.

**Results.** Source: reports/v3/x13.json. E is the mean over the 20 fresh streams per seed, then over seeds,
with 95% t-CIs.

| Arm | Mean E | Win rate (mean over seeds of the 20-stream win fraction) |
|---|---|---|
| Plan search (open loop) | 0.6719 [0.6575, 0.6863] | 0.991 |
| ORACLE planner (closed loop) | 0.6277 [0.6128, 0.6425] | 0.950 |
| ORACLE's executed plan, replayed open loop | 0.6277 [0.6128, 0.6426] | 0.950 |

- **Gap, search − ORACLE (`gap_search_minus_oracle`):**
  - +0.0442 [0.0425, 0.0459], sd 0.0086. Wilcoxon p = 4e-18; one-sided p = 2e-18.
  - The search is better on **100/100 seeds**.
  - Quantiles: minimum 0.0265, 5% 0.0310, median 0.0441, 95% 0.0577, maximum 0.0661.
  - The share of seeds with a gap > 0.01 is 1.00, and > 0.02 is also 1.00.
- **Wins:** +0.042 [0.004, 0.079] in the per-seed win fraction (Wilcoxon p = 0.018).
  - The search wins all 20 streams on 99 seeds; the ORACLE does on 93 (derived: rows of raw/x13.jsonl with `win20_* = 1`).
  - The ORACLE's losing seeds are 12003, 12070 (0/20 vs 20/20 for the search), 12002 (1/20 vs 20/20) and 12090
    (2/20 vs 20/20).
  - On seed 12012 both lose mostly (0/20 vs 2/20). These counts are from raw/x13.jsonl.
- **Closed loop does not help the ORACLE:** closed − open-loop replay of its own plan = −0.00001 [−0.00018, 0.00016]
  (`gap_oracle_closed_minus_open_loop`). Its decisions hardly depend on the realised cascades, so the whole gap is
  plan quality.
- **No winner's curse to speak of:** the score on the search streams minus the fresh-stream E is +0.0003
  [−0.0001, 0.0007] (`selection_bias`).
- **Search behaviour:**
  - Hill-climb passes: 1.47 on average.
  - The final pick never came from a warm-start plan (`cem_pick_from_warm` = 0).
  - The CEM best rose from 0.655 at iteration 1 to 0.672 at iteration 30 (raw/x13.jsonl `cem_trace`).

**What the better plans look like** (`features`: the search plan vs the mean of the ORACLE's 20 executed plans):

| Feature | Search | ORACLE |
|---|---|---|
| workers bought | 2.79 | 16.16 |
| media airings | 9.29 | 3.66 |
| media share of spend | 0.887 | 0.342 |
| share of workers bought in round 1 | 0.99 | 0.31 |
| spend-weighted mean round (0 = round 1) | 1.01 | 1.98 |
| regions with media | 3.49 | 2.53 |
| total spend (₫) | 493.1 | 490.9 |

Mean airings per round (raw/x13.jsonl):
- **Search:** 3.21, 2.79, 2.06, 0.91, 0.32 (derived: mean of `search_plan` per round, raw/x13.jsonl).
- **ORACLE:** 0, 0, 0.05, 1.62, 2.01. The ORACLE buys 5.0, 5.0, 4.85, 0.81 and 0.46 workers per round (derived: mean of
  `oracle_plan_canonical` per round, raw/x13.jsonl).

The search also uses fewer worker regions (1.48 vs 2.29). The two plans are opposite in timing and channel:
- the search puts media first, airing almost every region in rounds 1–2, and buys a few workers in round 1 only;
- the ORACLE buys workers in rounds 1–3 and airs media late.

**What explains the per-seed gap** (`feature_gap_correlations`, Holm over 10 features). The gap is large on every
seed, but its size varies little (sd 0.009). No single feature difference explains that variation after Holm:
- spend timing ρ = +0.25 (raw p = 0.013, Holm p = 0.13);
- worker regions used ρ = +0.22 (Holm p = 0.27);
- every other feature |ρ| ≤ 0.13.

A standardised OLS on five feature differences explains R² = 0.08 (`gap_ols`). The gap is systematic (the greedy
rule picks the wrong channel and timing everywhere), not a tail of special seeds.

Figures:
- ![X13 gap distribution](..figures/x13_gap.png)
- ![X13 plan features](..figures/x13_features.png)

**Theory vs observation. Refuted.**
- The search beats the ORACLE planner by 0.044 E on average [0.043, 0.046], more than four times the predicted
  ≤ 0.01.
- There is no long tail: the gap exceeds 0.02 on every seed and its 95th percentile is only 0.058.
- The part about front-loading is right in spirit, but the winning lever is **front-loaded media**, not concentrated
  workers.
- Why the greedy planner misses it:
  - Its first-order marginal value prices an airing by the immediate vote-probability gain only. It ignores that
    voters moved in round 1 keep pulling their friends towards b for four more social steps.
  - It also overrates workers, whose cascade reach it estimates with an optimistic branching formula.
  - It is myopic about timing: it saves media for the late rounds, when airing has the least time to spread.
- X7 already noted that the ORACLE does better when workers are dearer, which is consistent with this over-buying.

**What it means.**
- **For the design:** "the ORACLE" is a weak reference. Every "gap to the ORACLE" in X0, X6, X7 and X23 understates
  how much a skilled campaign can gain, by about 0.04 E.
  - The pay_split dominance found in X0 is a statement about scripted patterns. The best plans found here spend
    about 89% on media, front-loaded.
  - If the designers want workers to matter, the worker channel needs to be stronger relative to early media
    (e.g. a stronger media decay, or a higher worker attempt cap). This is a suggestion only; no rule change is made.
- **For a player:** with ₫500, air media in most regions in rounds 1–2 and buy a handful of workers in round 1.
  Do not save money for the last rounds.
- **For X6:** the X13 bound (0.672) is reported next to the ORACLE (0.628).
- **Caveats:**
  - The bound is for open-loop plans with the ORACLE manifesto. A campaign-aware manifesto could do better (X12).
  - The CEM is a heuristic search, so 0.672 is a lower bound on the true optimum.

**Data.** reports/v3/x13.json, reports/v3/raw/x13.jsonl (100 rows: plans, features, CEM traces, 20-stream
outcomes), log reports/v3/logs/x13.log. Command:
`OMP_NUM_THREADS=4 python experiments/x13_plan_search.py --seeds 100 --first-seed 12000 --device cuda:0 --out reports/v3/x13.json`.

### X14 — Planner calibration (prediction vs reality)

**Status:** done. The planner is pessimistic, not optimistic (round-1 bias −0.024 at 10⁴), and the bias shrinks with world size. Partly refuted.

**Question.** BayesPlanner predicts the election-day E of its chosen plan on a few small posterior worlds
(8 × 1,600 voters by default). It scales hires by n/N with stochastic rounding. Is that prediction calibrated, how
much do the world size, the world count and the rollout continuation matter, and does the scaled hire count round
to zero at large N?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> positive bias (over-optimism) that shrinks with world size; "even" undervalues front-loaded worker
> buys.

**Setup.**
- **Seeds:** first seed 12000, 50 paired seeds per arm (sweep). The engine games run on cuda:0 at N = 10⁴ and
  N = 10⁵, with T = 5 and ₫500.
- **Prediction vs outcome:** the prediction is the planner's `best_score` at each decision round (mean E at the
  election over its worlds, for its best candidate). The outcome is the realised E of the same game.
- **Arms (one factor at a time around the default world_voters 1600, num_worlds 8, rollout "even"):**
  - world_voters ∈ {800, 4000, N_true};
  - num_worlds ∈ {4, 16, 32};
  - rollout ∈ {"none", "front"}.
  - That gives 9 arms per N and 900 games.
- **The "front" rollout** is the approved library change (D2, §9): `planner.front_allocation`. The next round spends
  all the remaining worker money (treasury minus a media reserve), and media only follows afterwards. The default is
  bit-identical, checked by `tests/unit/test_planner_front_rollout.py` against scores recorded before the change.
- **Same manifesto:** every arm commits the default BayesPlanner's manifesto (`FixedManifestoPlanner`). Surveys are
  the library default, so the arms differ only in the allocation planner.
- **Device and time:** cuda:0, 2 processes.
  - Wall time 1,837 s (30.6 min); the pilot estimated 39 min.
  - Mean GPU utilisation 8.6%: the work is bound by Python and the CPU world sampler.
  - Peak memory 4.3 GB per child, in the N_true arms (`provenance`).
- **Parity (CPU↔GPU, default arm, N = 10⁴, seeds 12000–12002):** passed.
  - Realised E: 0.632 vs 0.635.
  - Round-1 prediction: 0.609 vs 0.599.
  - Round-1 bias: −0.023 vs −0.035, with tolerance 0.021 (statistical; the posterior worlds differ by device).
- **Deviations:**
  1. One-factor-at-a-time design (9 arms per N) instead of the full 4 × 4 × 3 factorial (48 arms per N). The
     factorial would be 4,800 games.
  2. Device cuda:0 instead of cuda:2.
  3. No seed reduction was needed: the N = 10⁵, N_true arm ran all 50 seeds (`seeds.heavy_arm_seeds`).

**Results** (reports/v3/x14.json → `N10000.arms`, `N100000.arms`).
- Bias = predicted − realised E at the round-1 decision (95% CI).
- Slope = OLS slope of realised on predicted over seeds (1 = calibrated).
- "Hires < 1/world" = the share of hiring rounds whose expected scaled hires per world are below 1, i.e. mostly
  rounded to zero.

| N | Arm | Round-1 bias | RMSE | Slope [95% CI] | Mean E | Win rate | Latency (s) | Hires < 1/world |
|---|---|---|---|---|---|---|---|---|
| 10⁴ | default (1600, 8, even) | −0.024 [−0.030, −0.019] | 0.031 | 0.94 [0.88, 1.01] | 0.648 | 0.98 | 0.21 | 0.80 |
| 10⁴ | world_voters 800 | −0.043 [−0.049, −0.038] | 0.048 | 0.98 | 0.639 | 0.98 | 0.13 | 1.00 |
| 10⁴ | world_voters 4000 | −0.017 [−0.022, −0.011] | 0.026 | 1.00 | 0.653 | 0.98 | 0.65 | 0.62 |
| 10⁴ | world_voters N | −0.016 [−0.022, −0.010] | 0.026 | 0.97 | 0.651 | 0.98 | 3.41 | 0.00 |
| 10⁴ | num_worlds 4 / 16 / 32 | −0.017 / −0.032 / −0.030 | 0.027–0.037 | 0.98–0.99 | 0.646–0.651 | 0.98 | 0.12–0.77 | 0.74–0.81 |
| 10⁴ | rollout none | −0.050 [−0.055, −0.044] | 0.053 | 0.93 | 0.643 | 0.98 | 0.21 | 0.72 |
| 10⁴ | rollout front | −0.010 [−0.016, −0.003] | 0.026 | 0.99 | 0.644 | 0.92 | 0.21 | 0.81 |
| 10⁵ | default | −0.047 [−0.057, −0.037] | 0.058 | 1.10 [0.96, 1.23] | 0.587 | 0.76 | 0.22 | 1.00 |
| 10⁵ | world_voters 800 / 4000 | −0.053 / −0.041 | 0.066 / 0.056 | 1.13 / 1.02 | 0.585 / 0.596 | 0.80 / 0.84 | 0.14 / 0.67 | 1.00 |
| 10⁵ | **world_voters N** | **−0.020** [−0.029, −0.012] | 0.037 | 0.99 [0.89, 1.09] | **0.621** | 0.84 | 3.50 | **0.00** |
| 10⁵ | num_worlds 4 / 16 / 32 | −0.045 / −0.060 / −0.057 | 0.061–0.071 | 1.09–1.18 | 0.589–0.595 | 0.76–0.82 | 0.13–0.80 | 1.00 |
| 10⁵ | rollout none | −0.089 [−0.098, −0.079] | 0.095 | 1.11 | 0.594 | 0.80 | 0.22 | 1.00 |
| 10⁵ | rollout front | −0.035 [−0.047, −0.024] | 0.054 | 0.99 | 0.588 | 0.78 | 0.22 | 1.00 |

- **The bias is the same in every round.** For the default arm at N = 10⁴ it is −0.024, −0.023, −0.026, −0.024
  and −0.022 in rounds 1–5, and −0.047 to −0.053 at 10⁵ (`per_round`).
- **Pearson r** between prediction and outcome: 0.96–0.97 at 10⁴ and 0.88–0.94 at 10⁵.
- **Bias vs world size:** the bias moves towards 0 as the worlds grow. Spearman ρ = +0.42 (p = 4e-10) at 10⁴ and
  +0.29 (p = 3e-5) at 10⁵ (`bias_vs_world_voters_spearman`).
- **Paired tests vs default** (`tests_vs_default`, Holm per metric within N):
  - At 10⁴, 800-voter worlds have a larger |bias| (+0.019, Holm p = 2e-9) and lower E (−0.009, p = 1e-5).
  - At 10⁴, 4,000-voter worlds raise E by +0.005 (p = 0.007).
  - At 10⁵, **N_true worlds raise E by +0.034 [0.025, 0.043]** (Holm p = 3e-11) and cut |bias| by 0.018
    (p = 0.001). They win 4 seeds the default loses and lose 0 (McNemar p = 0.125).
  - More worlds (16, 32) slightly *increase* |bias| at 10⁴ (+0.007, +0.006; Holm p ≤ 6e-4) without changing E
    materially.
  - Rollout "none" doubles |bias| (+0.025 at 10⁴, +0.041 at 10⁵; Holm p ≤ 8e-13).
  - Rollout "front" reduces |bias| at 10⁴ (−0.006, p = 8e-4) but does not change E (−0.0035, p = 0.79 at 10⁴;
    +0.0007, p = 1 at 10⁵).
- **Front-loading in practice:** round-1 workers are 5.9 with "even", 9.8 with "none" and 4.1 with "front" at 10⁴
  (`workers_round1`). The front continuation makes round-1 hires look *less* urgent, because the continuation
  already buys them in round 2.

Figures:
- ![X14 bias by arm](..figures/x14_bias.png)
- ![X14 bias by round](..figures/x14_bias_by_round.png)

**Theory vs observation. Refuted on the sign; partly confirmed otherwise.**
- **"Positive bias (over-optimism)": refuted.** The planner is **pessimistic** in every arm at both N: −0.010 to
  −0.089 at round 1.
  - The reason is structural. A rollout scores each candidate by continuing with a fixed base policy (even pacing),
    but the player re-plans every round and does better than that base policy. Rollout values are therefore a lower
    bound on what the re-planning player achieves (the policy-improvement property).
  - With "none" (spend nothing more) the bound is loosest (−0.050 / −0.089). With "front", whose continuation is
    closer to what the planner actually does, it is tightest (−0.010 at 10⁴).
- **"Shrinks with world size": confirmed for the magnitude.** It goes from −0.043 (800) to −0.016 (N_true) at 10⁴,
  and from −0.053 to −0.020 at 10⁵ (ρ = +0.42 and +0.29).
- **Rounding to zero: confirmed.**
  - At N = 10⁵ with 1,600-voter worlds, every hiring round scales to below one expected worker per world (share
    1.00). A planned worker usually vanishes from the planner's worlds.
  - Using full-size worlds fixes it: E rises from 0.587 to 0.621, at 3.5 s per decision instead of 0.22 s.
- **"Even undervalues front-loaded worker buys": not supported.**
  - The even continuation does give a more pessimistic round-1 value than the front continuation (−0.024 vs
    −0.010).
  - But planning with the front continuation buys *fewer* round-1 workers and does not raise E (−0.0035, n.s.). It
    lowers the win rate from 0.98 to 0.92 (3 discordant seeds, McNemar p = 0.25).
- **The ranking is well calibrated:** slopes 0.93–1.18 with CIs mostly covering 1, and r ≈ 0.96 at 10⁴. Only the
  level is shifted.

**What it means.**
- **For the design / UI:** the planner's "predicted share" is a conservative estimate: add about 0.02 at N = 10⁴ and
  about 0.05 at N = 10⁵ to read it as an expectation. It ranks plans correctly.
- **For the planner (a suggestion, no change made):** scale world_voters with N, or at least keep n/N × hires ≥ 1.
  At N = 10⁵ the default makes workers invisible to the planner, which costs 0.034 E. D11 restricts T, not N, so
  this matters for the large-N games the API allows.
- **For a player:** the planner's advice is good. Its number is cautious.

**Data.** reports/v3/x14.json, reports/v3/raw/x14.jsonl (900 rows: per-round predictions, realised E, allocations,
latencies), log reports/v3/logs/x14.log. Command:
`OMP_NUM_THREADS=4 python experiments/x14_planner_calibration.py --seeds 50 --first-seed 12000 --device cuda:0 --out reports/v3/x14.json`.

### X19 — Manifesto geometry and a third optimiser

**Status:** done. Optimisers tie (PG = MD, Frank–Wolfe most accurate); 1.27 local optima per seed; b* is closer to a 2–3-region blend than to the mean or median. Partly confirmed.

**Question.**
- Where does the optimal manifesto b\* sit relative to simple summaries of the electorate (a median-voter analogue
  on the simplex)?
- How many regions does it carry (a minimum-winning-coalition analogue)?
- How do projected gradient, consensus-ADMM, mirror descent and Frank–Wolfe compare?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> b* is closer to a 2–3-region coalition blend than to the mean or median; there are several local
> optima (multi-start matters); mirror descent converges in fewer iterations than PG on interior optima.

**Setup.**
- **Seeds:** first seed 12000, 100 seeds (core). The same electorates as X6/X12/X13.
- **Objective:** the exact expected share E(b) on the **anticipated** true electorate v_T (social-only dynamics for
  T = 5 rounds, N = 10⁴). This is the ORACLE manifesto's objective. The optimisers ascend the pseudo-Huber surrogate
  (δ = 1e-3; ADMM uses δ = 0.02, as the library does).
- **Optimisers,** batched over starts on cuda:0. Mirror descent and Frank–Wolfe are new, experiment-side functions
  in `experiments/x19_manifesto.py`; nothing was added to the library.
  - **PG:** Euclidean projection = prox of the simplex indicator. Tuned step 0.05; the library step 0.02 is also
    reported as `PG_lib_step`.
  - **ADMM:** consensus over the 4 regions, ρ = 5, 10 inner steps, vectorised. It reproduces `admm_manifesto`
    (TV 2e-7 between the solutions on a check seed, `library_check`).
  - **MD:** exponentiated gradient (entropic prox), tuned step 0.2.
  - **FW:** vertex direction plus a line search over γ ∈ {0} ∪ 64 log-spaced points in [1e-6, 1].
- **Step tuning:** the PG and MD steps were tuned on 5 held-out seeds (12100–12104), by the median iterations to
  1e-4 (`step_tuning`). Grids: PG {0.005, …, 0.1} → 0.05; MD {0.02, …, 1.0} → 0.2.
- **Starts:** each optimiser starts from the same 8 standard starts (the library's default starts + the top-3
  statistic).
- **"Iterations to 1e-4":** the first iteration at which the best start reaches E\* − 1e-4. E\* is the best exact E
  over every end point of every optimiser and of 64 flat-Dirichlet starts.
- **Local optima:** PG and MD from 64 Dirichlet starts, with end points clustered at TV < 0.01.
- **Geometry:** b\* is compared with:
  - the mean;
  - the per-sector median, projected to Δ;
  - the L1 geometric median (Weiszfeld-type iteration under TV);
  - every 2-, 3- and 4-region size-weighted blend of the anticipated regional means;
  - the ORACLE manifesto.
- **Regions carried:** a region counts as carried if its regional mean P > 0.5.
- **Device and time:** cuda:0, 1 process. Wall time 583 s (9.7 min); mean GPU utilisation 21%; peak memory 0.28 GB.
- **Parity (CPU↔GPU, seeds 12000–12002):** passed. E\* 0.450395 on both; PG distinct optima 1 vs 1; regions
  carried 1.33 vs 1.33; TV(mean, b\*) 0.05583 vs 0.05582 (`provenance.parity`).
- **Deviations:**
  1. A first run used a uniform 65-point line-search grid for Frank–Wolfe. Its smallest step (1/64) stalled FW near
     the optimum (5/100 seeds reached 1e-4). This was an implementation weakness, so the grid was made log-spaced
     and the whole experiment rerun. Only the rerun is reported.
  2. Device cuda:0 instead of cuda:2.

**Results — optimisers** (reports/v3/x19.json → `optimisers`, `optimiser_tests`; Holm within each metric):

| Optimiser | Reached E\* − 1e-4 | Median iterations [IQR] | Median time to 1e-4 | Final E gap to E\* (mean / max) | Starts within 1e-4 (of 8) |
|---|---|---|---|---|---|
| PG (step 0.05) | 100/100 | 5 [3, 9] | 0.008 s | 1.2e-4 / 9.5e-3 | 7.42 |
| PG (library step 0.02) | 100/100 | 11.5 [7, 20] | 0.020 s | 2.2e-5 / 1.9e-4 | 7.66 |
| Mirror descent (step 0.2) | 100/100 | 5 [3, 9] | 0.008 s | 2.2e-5 / 1.9e-4 | 7.66 |
| Frank–Wolfe (line search) | 100/100 | 10 [6, 15] | 0.026 s | 8e-7 / 4e-5 | 6.13 |
| Consensus-ADMM (δ = 0.02) | 4/100 | 9.5 (on those 4) | 0.31 s | 6.2e-3 / 2.9e-2 | 0 |

- **MD − PG:** −0.26 iterations [−0.67, 0.15] (Holm p = 0.37). MD is faster on 35 seeds; the rest are slower or
  tied.
- **FW − PG:** +4.5 iterations [3.3, 5.6] (Holm p = 4e-10), but a slightly better final E (+0.00012, Holm p = 2e-13).
- **Library step − tuned PG:** +9.4 iterations (Holm p = 2e-16).
- **ADMM's final E** is 0.0061 [0.0048, 0.0074] below PG's (Holm p = 2e-16). This is the smoothing bias of
  δ = 0.02 that the library documents.
- **ADMM residuals at the end:** median primal 3e-8 and dual 0. The worst start reaches primal 0.098 and dual 0.26,
  an oscillating start.
- **The tuned PG step overshoots:** with 0.05, PG reaches the tolerance fastest but then oscillates. On 7/100 seeds
  its *final* end point is more than 1e-4 below E\* (max 0.0095), against 4/100 for MD and 0/100 for FW
  (raw/x19.jsonl).
- **Interior vs boundary:** every b\* is interior; the smallest coordinate is 0.139 on average
  (`b_star_min_coordinate`). The "interior" comparison is therefore the whole sample: MD 5 vs PG 5 median iterations
  (`interior_vs_boundary`).

**Results — local optima** (`local_optima`, 64 Dirichlet starts):
- PG finds 1.27 [1.17, 1.37] distinct optima per seed (median 1, max 3). There is more than one on 23/100 seeds.
- 92% of starts reach b\* (MD: 95%).
- The secondary optima are poor. On average only 1.01 distinct optima lie within 0.01 E of E\*. The spread of
  end-point E is 0.099 on average, because the bad optima are much worse.

**Results — geometry** (`geometry`, `geometry_tests`, `nearest_reference_counts`):

| Reference | TV to b\* (mean [95% CI]) | E (mean) | E\* − E |
|---|---|---|---|
| ORACLE manifesto (library PG) | 0.004 [0.002, 0.006] | 0.4690 | 0.0000 |
| L1 geometric median (TV) | 0.044 [0.038, 0.049] | 0.4584 | 0.0106 |
| per-sector median (projected) | 0.055 [0.049, 0.061] | 0.4526 | 0.0163 |
| mean (= 4-region blend) | 0.056 [0.050, 0.062] | 0.4510 | 0.0180 |
| nearest 2- or 3-region blend (per seed; derived: mean over raw/x19.jsonl of `geometry[nearest_blend23]`) | 0.038 | 0.4582 | 0.011 |
| 3-region blends (each) | 0.076–0.083 | 0.429–0.434 | 0.035–0.040 |
| 2-region blends (each) | 0.110–0.124 | 0.403–0.411 | 0.058–0.066 |

E\* = 0.4690 [0.4560, 0.4820]. b\* alone, with no campaign, has E > 0.5 on 39/100 seeds.

- **Nearest 2–3-region blend vs each reference** (TV difference; Holm over 3 tests):
  - vs the mean: −0.018 [−0.024, −0.012] (blend closer on 67/100 seeds, Holm p = 6e-6);
  - vs the median: −0.018 [−0.024, −0.011] (65/100, p = 6e-6);
  - vs the TV geometric median: −0.006 [−0.011, −0.001] (57/100, Holm p = 0.13).
- **The single closest reference per seed:** geometric median 31, a 3-region blend 29, a 2-region blend 19, the
  median 15, the mean 3, the 4-region blend 3.
- **Regions carried by b\*** (`carried`):
  - Distribution: 0 regions on 1 seed, 1 region on 33, 2 on 51, 3 on 15, 4 on 0. Mean 1.80.
  - The carried regions hold 50.4% of voters on average. On 43 seeds they hold ≤ 50%.
  - The carried set is a minimal winning coalition on 44% [35%, 54%] of seeds.

Figure: ![X19 optimisers and geometry](..figures/x19_optimisers_geometry.png)

**Theory vs observation. Partial (one part partly confirmed, two refuted).**
- **"b\* is closer to a 2–3-region coalition blend than to the mean or median": partly confirmed.**
  - The *nearest* 2–3-region blend is closer than the mean or median (−0.018 TV, Holm p = 6e-6).
  - But this is the best of 10 blends per seed against a single summary, and any *fixed* blend is far from b\*
    (TV 0.08–0.12).
  - The L1 geometric median is as close as the nearest blend (difference −0.006, Holm p = 0.13). It is also the best
    single summary (E\* − E = 0.011).
  - b\* is a broad compromise, nearer the geometric median than any coalition's taste. It carries 1–2 regions with
    about half the voters, a minimum winning coalition in 44% of seeds.
- **"Several local optima, multi-start matters": refuted.** The objective has a single good basin on almost every
  seed: 1.01 near-optimal clusters, and 92–95% of random starts reach b\*. The library's 8 structured starts already
  find b\*. Multi-start mainly protects against a few poor basins (23% of seeds).
- **"Mirror descent converges in fewer iterations than PG on interior optima": refuted.**
  - Every optimum is interior, and with tuned steps MD and PG tie at a median of 5 iterations (p = 0.37).
  - MD is faster than PG at the library's step (5 vs 11.5) only because 0.02 is a conservative step for PG, and MD
    is more stable at the end (4 vs 7 seeds end > 1e-4 off).
  - Frank–Wolfe with a line search is the most accurate (max final gap 4e-5) but needs about twice the iterations.
  - ADMM is a distributed method: its δ = 0.02 surrogate costs 0.006 E. It is the slowest by wall time (2.3 s vs
    0.2 s), because each iteration runs 10 inner steps.

**What it means.**
- **For the design:**
  - The manifesto problem is benign: low-dimensional, interior and essentially uni-modal on the anticipated
    electorate. Any of PG, MD or FW solves it in about 10 ms on the GPU.
  - The real manifesto question is the campaign interaction (X12), not the optimiser.
  - "Blend my hunches" over a fixed 2–3-region coalition is a poor static manifesto (0.035–0.066 E below b\*). This
    agrees with X6, where the blend lost 0.04 E.
- **For a player:** a broad compromise beats a coalition platform. If you cannot optimise, the "geometric median" of
  the electorate is the best rule of thumb tested.

**Data.** reports/v3/x19.json, reports/v3/raw/x19.jsonl (100 rows: per-optimiser traces, local optima, geometry,
regions carried), log reports/v3/logs/x19.log. Command:
`OMP_NUM_THREADS=4 python experiments/x19_manifesto.py --seeds 100 --first-seed 12000 --device cuda:0 --out reports/v3/x19.json`.

## 7. Information and logic

### X5 — Value of information at ₫10

**Status:** done. Two questions are best at ₫10 and every further question lowers E; the extra loss vs ₫5 is in the predicted direction but not significant after Holm. Partly confirmed.

**Question.** How many survey queries are worth buying now that a query costs ₫10? Does it matter whether
they are chosen by expected information gain or at random? How does ₫10 compare with the old price of ₫5?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> about 2 questions are still enough; beyond that every question is a net loss, larger than at ₫5.

**Setup.**
- **Seeds:** first seed 13000, 50 paired seeds. Every arm plays the same 50 electorates.
- **Game:** N = 10⁴, T = 5, treasury ₫500, workers ₫20. Query price ₫10 (game) and ₫5 (comparison).
- **Player:** BayesPlanner forced to ask exactly k survey queries, with k ∈ {0, 1, 2, 4, 8, 12}.
  - Every stop rule is off, and there are no in-round queries.
  - **Budget fix (plan X5):** `survey_budget = k × price` and `num_queries = k` for each arm (`arms.*`).
    Every game asked exactly k queries; the script checks this.
  - The queries are chosen by expected information gain (`gain`, the library rule) or uniformly at random
    (`random`).
  - Arms: 2 prices × (k = 0 + 5 values of k × 2 rules) = 22 arms, 1,100 games.
- **Metrics:**
  - outcome: expected share E and win rate;
  - posterior accuracy at the start of round 1: posterior mass on the true σ (archetype per region);
  - manifesto quality: the static share of the committed manifesto on the anticipated true electorate;
  - per-round verdicts from `logic.report.timeline_with_diagnostics` (16 poll-consistent worlds of 1,600 voters);
  - net value: paired ΔE per step of k.
- **Device and compute:** cuda:2, pinned, with the process memory capped at 10 GB.
  - Games run one at a time. The GPU batches worlds × plans × voters inside each decision: 8 planner worlds and
    16 logic worlds per round.
  - Wall time 74.7 min. Peak memory 0.13 GB (`provenance.compute.peak_bytes` = 133,201,920).
  - Mean GPU utilisation 17% (`provenance.compute.gpu_util_mean`).
- **Parity (CPU↔GPU, 3 seeds, ₫10 k = 2 gain):** passed. E 0.6105 vs 0.6072, P(true σ) 0.9996 on both
  (`provenance.parity`).
- **Deviations:** none in the design. See `deviations` in the JSON for the recorded choices:
  - no in-round queries;
  - both selection rules are also run at ₫5;
  - games are not batched, only the worlds inside each decision.

**Results.** Source: reports/v3/x5.json. Values are mean [95% CI] over 50 seeds.

| k | E, ₫10 gain | E, ₫10 random | E, ₫5 gain | P(true σ), gain | P(true σ), random | query spend at ₫10 |
|---|---|---|---|---|---|---|
| 0 | 0.6542 [0.6368, 0.6716] | (same arm) | 0.6542 | 0.328 | 0.328 | ₫0 |
| 1 | 0.6573 | 0.6561 | 0.6579 | 0.739 | 0.508 | ₫10 |
| **2** | **0.6579** [0.6411, 0.6747] | 0.6557 | 0.6578 | 0.995 | 0.684 | ₫20 |
| 4 | 0.6547 | 0.6546 | **0.6586** | 1.000 | 0.849 | ₫40 |
| 8 | 0.6512 | 0.6502 | 0.6547 | 1.000 | 0.968 | ₫80 |
| 12 | 0.6447 [0.6270, 0.6624] | 0.6447 | 0.6518 | 1.000 | 1.000 | ₫120 |

Sources: `arms.<arm>.expected`, `arms.<arm>.posterior.p_true_sigma`, `arms.<arm>.query_spend`, with
arm = `p10_k2_gain` and so on. The best k by mean E is 2 at ₫10 (gain) and 4 at ₫5 (gain)
(`value_of_information.p10_gain.best_k_by_expected`, `…p5_gain…`).

- **The win rate is uninformative here.** It is 100% in every arm except k = 1 gain (98%, one game). The 95%
  Wilson CI is [0.929, 1.0] (`arms.*.win_rate`). The 50 seeds of block 13000 are easy single-player elections,
  so every comparison below uses the expected share E. All McNemar tests have 0–1 discordant pairs.
- **The value of the first questions is real but small.** Two gain queries make σ almost certain: P(true σ)
  0.995, MAP correct in 100% of games, entropy 0.02 nats against 1.71 at k = 0
  (`arms.p10_k2_gain.posterior.*`). Random queries need about 12 for the same accuracy.
  - Yet E(k = 2) − E(k = 0) is only +0.0036 [−0.0008, +0.0081] (Wilcoxon p 0.20; Holm p 1)
    (`value_of_information.p10_gain.curve[1].delta_vs_k0`).
  - The manifesto barely changes. The static share of the committed b is 0.4842 at k = 0 and 0.4850 at k = 2,
    against 0.4859 for the ORACLE manifesto (`arms.*.static_share_anticipated`; derived: mean of `seed_info[*].oracle_static_share`).
  - The free top-3 statistic and the poll already give a near-optimal manifesto.
- **Beyond k = 2 every extra question is a net loss at ₫10** (`value_of_information.p10_gain.curve[*].marginal_vs_prev_k`):

  | step | ₫10 gain: ΔE of the step | ₫5 gain: ΔE of the step |
  |---|---|---|
  | 0→1 | +0.0030 [−0.0012, +0.0073] | +0.0036 [−0.0008, +0.0081] |
  | 1→2 | +0.0006 [−0.0024, +0.0036] | −0.0000 [−0.0027, +0.0026] |
  | 2→4 | **−0.0032** [−0.0058, −0.0005] (p 0.014) | +0.0008 [−0.0014, +0.0030] |
  | 4→8 | **−0.0035** [−0.0058, −0.0011] (p 0.004) | **−0.0039** [−0.0064, −0.0014] |
  | 8→12 | **−0.0065** [−0.0104, −0.0026] (p 0.003) | **−0.0030** [−0.0053, −0.0006] |

  - Paired against k = 2 at ₫10 with gain queries (`tests.families.curve`; Holm over 32 E-tests):
    - k = 4: −0.0032, Holm p 0.34;
    - k = 8: −0.0067 [−0.0099, −0.0034], Holm p 0.002;
    - k = 12: −0.0132 [−0.0176, −0.0087], Holm p 6·10⁻⁷.
  - At ₫5 only k = 12 vs k = 2 survives Holm: −0.0061 [−0.0097, −0.0024], Holm p 0.024.
- **The loss is larger at ₫10 than at ₫5** (`tests.families.price_contrast`; Holm over 6). The paired
  difference of (E_k − E_2) between ₫10 and ₫5, with gain queries:
  - k = 4: −0.0039 [−0.0074, −0.0005];
  - k = 8: −0.0035 [−0.0070, −0.0000];
  - k = 12: −0.0071 [−0.0122, −0.0020] (Wilcoxon p 0.014, Holm p 0.084).

  All point estimates point the same way, but none survives Holm at 5%.
- **The mechanism is money.** Each query is campaign money lost. Campaign spend falls from ₫492 at k = 0 to
  ₫471 at k = 2 and ₫372 at k = 12 (₫10). Same-money arms (₫5 k = 4 vs ₫10 k = 2, both ₫20 of queries) reach the
  same E: 0.6586 vs 0.6579 (`arms.*.campaign_spend`, `arms.*.expected`).
- **Gain vs random:** no E difference at any k or price. The largest is +0.0022 [−0.0014, +0.0059] at ₫10 k = 2,
  Holm p 1 (`tests.families.gain_vs_random`). The posterior is much better with gain queries (0.995 vs 0.684 at
  k = 2), but the game does not reward the difference.
- **Verdicts.** "Possible victory" holds in 100% of rounds in every arm. "Inevitable defeat" never holds, and no
  rounds fall back to the closest worlds.
  - "Guaranteed" (∃π∀w) at round 1 falls as more queries are bought: 0.88 at k = 0, 0.88 at k = 2, 0.80 at k = 8
    and 0.68 at k = 12 (₫10 gain; `arms.*.verdicts_by_round.guaranteed`).
  - The extra knowledge about σ is outweighed by the smaller treasury.
  - "Guaranteed" was never wrong about the outcome: P(win | guaranteed at round 1) = 1.0 in every arm (n = 44 at
    k = 2; `arms.*.["P(win | guaranteed at round 1)"]`). With a 100% win rate this is not a strong test.

Figures:
- ![X5 curve](..figures/x5_curve.png)
- ![X5 marginal value and posterior](..figures/x5_marginal.png)
- ![X5 verdicts by round](..figures/x5_verdicts.png)

**Theory vs observation. Partial (direction confirmed everywhere; one clause not significant after Holm).**
- **"About 2 questions are still enough": confirmed.**
  - At ₫10 the mean E peaks at k = 2.
  - Two gain queries already identify σ (P(true σ) 0.995).
  - Neither step from 0 to 2 is significant: information is worth at most ~+0.004 E.
- **"Beyond that every question is a net loss": confirmed at ₫10.** Every step past k = 2 has a negative paired
  ΔE whose CI excludes 0. The cumulative losses at k = 8 and k = 12 survive Holm.
- **"Larger than at ₫5": confirmed in direction, not in significance.** ₫5 tolerates k = 4 (+0.0008 for 2→4).
  The ₫10 − ₫5 contrast is negative for every k, but its best Holm p is 0.084.

**What it means.**
- **For the game design:** at ₫10 the survey is a minor channel.
  - Information is nearly worthless beyond two questions, because the free statistics already give a near-oracle
    manifesto (static share 0.4842 vs ORACLE 0.4859 with no queries at all; the latter derived as the mean of `seed_info`).
  - Workers dominate the value of money (X0, X11), so every ₫10 spent on queries costs about half a worker.
  - If surveys should matter, the manifesto must depend more on σ (e.g. weaker free statistics) or queries must
    reveal τ. A price cut alone only moves the optimum from 2 to 4 questions.
- **For a player:** ask one or two questions, picked by information gain, then stop. Never buy more than 4.
  Random questions are almost as good for the outcome, but need about 6× as many to pin σ down.

**Data.** reports/v3/x5.json, reports/v3/raw/x5.jsonl (1,100 rows), log reports/v3/logs/x5.log.
Command: `OMP_NUM_THREADS=4 python experiments/x5_info_value.py --seeds 50 --first-seed 13000 --device cuda:2 --out reports/v3/x5.json`.
Figures: `python experiments/plot_v3_phase4a.py --only x5`.

### X17 — Learning personalities by probing

**Status:** done. Probing does not lower τ entropy on the plan's metric and costs −0.013 E; the national poll limits learning. Partly confirmed.

**Question.** Surveys reveal nothing about the personalities τ (R13), and polls are national. Can a deliberate
single-region probe in round 1 make τ learnable from the polls? Is the exploration worth its price?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> probing lowers τ entropy measurably, but the 400-voter national poll (SE ≈ 2.4 pp) limits it.
> The net win-rate effect is ≈ 0 or negative.

**Setup.**
- **Seeds:** first seed 13000, 50 paired seeds.
- **Game:** N = 10⁴, T = 5, ₫500, ₫20 workers, ₫10 queries.
- **Arms** (all observation-only, with BayesPlanner defaults for the survey and the planner):
  - `bayes`: the plain BayesPlanner. Its planning worlds draw τ uniformly and never look at the polls.
  - `poll_planner` (control, added): BayesPlanner whose planning worlds are poll-consistent
    (`logic.worlds.consistent_worlds`: 8 worlds; a world is kept if every poll matches within 2.5·(SE + 0.01)).
  - `probe`: the poll-consistent planner, except in round 1. There it plays the single-region probe (one airing in
    r, or the even cap ₫100 on workers in r) that best splits the τ hypotheses.
    - Score: between-τ_r variance of the predicted round-2 poll ÷ (SE² + within-group variance), over 96 posterior
      worlds.
    - 8 candidates are scored as one batched GPU step.
- **τ learning at each decision point t = 1..5:**
  - D = 512 candidate worlds: σ from the posterior, τ uniform over the 24 permutations, keys as in
    `consistent_worlds`. The same candidates are used for all arms and rounds.
  - The worlds are replayed once through the player's history on the GPU. The accepted set at t is the worlds
    that match all polls s ≤ t.
  - **Plan metric (hard filter):** the entropy of the accepted τ (H_perm). Prior ln 24 = 3.178; the plug-in value
    at D = 512 is 3.157.
  - **Secondary (soft):** the candidates weighted by the Gaussian poll likelihood without the 0.01 slack (H_soft).
    This shows what the polls could teach a strict Bayesian.
  - **Check against the library:** `consistent_worlds` returns the identical accepted set, the identical τ and
    poll shares within 6·10⁻⁸ (`library_check`).
- **K_p(Guaranteed) by round:** `logic.report` timeline with 16 worlds.
- **Device and compute:** cuda:2, pinned, with the process memory capped at 10 GB.
  - Wall time 42.2 min. Peak memory 1.62 GB (`provenance.compute.peak_bytes`).
  - Mean GPU utilisation 0.9%. The time goes into the library's world sampler (`sample_worlds`), which draws each
    micro-state with CPU generators: about 12 s per game for the 512 candidates. Replays, rollouts and probe
    scoring run on the GPU.
- **Parity (CPU↔GPU, 3 seeds, probe arm, D = 256):** passed. E 0.5941 vs 0.5908; same probe region in all 3 seeds;
  H_perm at round 5 3.121 vs 3.134 (`provenance.parity`).
- **Deviations** (`deviations`):
  - the control arm was added;
  - the one-replay τ measurement is equivalent to calling `consistent_worlds` at each t; σ for the τ worlds is
    drawn from the round-5 posterior, whose σ entropy is ≈ 0 after the survey;
  - the probe score is a proxy for the expected information gain;
  - the τ worlds are drawn on the CPU.

**Results.** Source: reports/v3/x17.json. Values are mean [95% CI] over 50 seeds.

| | bayes | poll_planner | probe |
|---|---|---|---|
| Expected share E | 0.6585 [0.6417, 0.6754] | 0.6576 [0.6414, 0.6739] | **0.6458** [0.6253, 0.6663] |
| Win rate | 1.00 [0.93, 1.00] | 1.00 [0.93, 1.00] | 0.94 [0.84, 0.98] |
| Round-1 spend | ₫114.0 | ₫118.7 | ₫58.7 |
| H_perm (hard), round 2 / 3 / 5 | 3.157 / 3.156 / 3.131 | 3.157 / 3.156 / 3.128 | 3.157 / 3.155 / 3.150 |
| Acceptance, round 5 | 0.837 | 0.830 | 0.955 |
| H_soft, round 2 / 3 / 5 | 3.151 / 3.114 / 2.929 | 3.151 / 3.114 / 2.947 | **3.121 / 3.073** / 2.929 |
| H_soft of the probed region, round 3 (prior 1.386) | 1.365 | 1.363 | **1.313** |
| P(true τ_r of the probed region), soft, round 3 (prior 0.25) | 0.268 | 0.267 | 0.292 |
| K(Guaranteed), rounds 1–5 | .88 .76 .80 .82 .84 | .88 .82 .78 .78 .88 | .88 .70 .78 .78 .78 |

Sources: `arms.<arm>.expected`, `win_rate`, `round1_spend`, `tau_by_round.<metric>[t]`, `guaranteed_by_round`.

- **The probe always aired media** (50/50 games; `probes.kind_counts`) in one region (`probes.region_counts`
  = 9/18/12/11). An airing reaches all exposed voters of a region, so it moves the national poll more cleanly than
  five workers, which become 0.8 workers in a 1,600-voter world.
- **Hard filter (the plan's metric): probing teaches nothing measurable.** At round 2, every candidate world is
  accepted in every arm, so the probe-vs-bayes difference is exactly 0.
  - At round 5 the probe arm has *higher* entropy than bayes: +0.0186 [+0.0041, +0.0331], Holm p 2·10⁻⁵
    (`tests.tau_learning`).
  - Reason: the probe spends half as much in round 1, so its campaign moves the polls less. Only 4.5% of worlds are
    rejected by round 5, against 16% for bayes.
  - The filter window is 2.5·(2.4 + 1.0) pp ≈ ±8.5 pp, wider than any τ-driven difference in the poll.
- **Soft likelihood: probing helps briefly.** Probe minus bayes:
  - H_soft at round 2: −0.030 [−0.041, −0.018], Holm p 3·10⁻⁹;
  - entropy of the probed region at round 3: −0.052 [−0.080, −0.025], Holm p 0.001.
  - By round 5 the ordinary campaign has taught the other arms as much: H_soft −0.0001 [−0.062, +0.062], Holm p 1.
  - Even at its best, the posterior mass on the true τ is 0.07 against the prior 0.042, and on the probed region's
    personality 0.33 against 0.25.
- **Using the polls without probing changes nothing.** `poll_planner` vs `bayes`: E −0.0009 [−0.0039, +0.0021],
  Holm p 0.57, and every τ difference has Holm p 1. The hard filter rejects almost no planning worlds, so the
  planner's worlds are essentially the same.
- **The probe costs about 1.2 points of E.**
  - Probe vs bayes: −0.0127 [−0.0195, −0.0059], Wilcoxon Holm p 0.007, probe better in 19/50 seeds.
  - Probe vs poll_planner: −0.0118, Holm p 0.007.
  - It lost 3 games that both other arms won: seeds 13005, 13011 and 13041, with E 0.475–0.486 (raw/x17.jsonl). McNemar 0 vs 3,
    Holm p 0.75 (`tests.outcome`).
  - K(Guaranteed) summed over rounds: −0.18 [−0.47, +0.11] rounds vs bayes (Holm p 0.56).
  - In the true world a win stayed possible in only 94% of probe games from round 2 on, against 100% for bayes
    (`arms.*.truth_possible_by_round`).

Figures:
- ![X17 τ entropy](..figures/x17_entropy.png)
- ![X17 outcome](..figures/x17_outcome.png)

**Theory vs observation. Partial.**
- **"Probing lowers τ entropy measurably": refuted for the plan's metric, confirmed only for the soft one.**
  - With the poll-consistency filter that the logic layer uses (the plan's metric), probing lowers nothing; the
    probe arm ends with *more* entropy.
  - With the strict Gaussian likelihood, the probe lowers τ entropy measurably at rounds 2–3. The size is ~1% of
    the prior entropy (≈ 4% for the probed region), and the gain is gone by round 5.
- **"The 400-voter poll limits it": confirmed, strongly.** Even in the best case, the true τ gets 7% of the
  posterior mass after five polls.
- **"Net win-rate effect ≈ 0 or negative": confirmed (negative).** E −0.013 (Holm p 0.007) and 3 lost games, none
  gained.

**What it means.**
- **For the game design:** τ is effectively unknowable within one game. Polls are national and noisy (SE 2.4 pp),
  and a single region's campaign response moves them by less than that.
  - The logic layer's 0.01 slack makes the hard filter blind to τ. "Knowledge" verdicts therefore never depend on
    personalities.
  - If τ should be learnable, the game needs regional polls or a paid "focus group" query on susceptibility.
- **For a player:** do not probe. Spending round 1 on a cheap test airing wastes the round in which workers
  cascade longest (X11), and the poll will not tell you the personality anyway. Campaign normally; the polls teach
  just as much by round 5.

**Data.** reports/v3/x17.json, reports/v3/raw/x17.jsonl (150 rows), log reports/v3/logs/x17.log.
Command: `OMP_NUM_THREADS=4 python experiments/x17_probe.py --seeds 50 --first-seed 13000 --device cuda:2 --out reports/v3/x17.json`.
Figures: `python experiments/plot_v3_phase4a.py --only x17`.

### X16 — Reliability and calibration of epistemic verdicts

**Status:** done. "Guaranteed" gets rarer as the world count grows and was never falsely asserted; p̂ is not over-confident at 16 worlds (slightly under-confident). Partly confirmed.

**Question.** The debrief verdicts ("guaranteed", "possible", "knows that a win exists", "inevitable defeat") are
evaluated over a *sample* of worlds (16 worlds of 1,600 voters by default) and a finite plan set Π of about 16
plans. How often are they wrong? How do they depend on the number of worlds W, the world size n and |Π|? And is the
graded belief p̂ calibrated?

**Prediction (pre-registered).** Copied unchanged from reports/v3/predictions.md:
> "guaranteed" becomes rarer and more reliable as the world count grows; p̂ is over-confident at 16
> worlds.

**Setup.**
- **Seeds:** first seed 13000, 50 seeds (sweep), paired. BayesPlanner (library defaults) and Top-3 each play the
  same 50 electorates.
- **Game:** N = 10⁴, T = 5, treasury ₫500, workers ₫20, queries ₫10, media base ₫50. Electorates use the canonical
  CPU draw (R8), moved to the GPU.
- **Grid (full, as planned):** W ∈ {8, 16, 32, 64} × n ∈ {800, 1600, 4000} × |Π| ∈ {8, 16, all}. That is 36 cells ×
  100 games × 5 decision points = 500 decision points per cell.
- **Verdicts:** recomputed per cell with exactly the seeds of `logic/report.timeline_with_diagnostics` (timeline
  seed 0).
  - The |Π| knob is the approved D2 change `policy_set(…, max_policies=k)`. It keeps the first k plans ("No spend"
    first).
  - The mean |Π_t| of "all" is 16.03, so |Π| = 16 and "all" almost coincide (`cells.*.mean_plan_set_size`).
- **Truth row:** Π_t is evaluated in the TRUE world at the decision point. This is the timeline's own
  `truth_victory_possible`.
- **Long-run truth L_t:** the X13 plan search (CEM 256 × 30, elite 10%, then a 1-move hill climb) is run from the
  TRUE state at each decision point.
  - It uses the player's manifesto, workers, airings, remaining treasury and remaining rounds.
  - It is warm-started with every plan of Π_t and with the player's own continuation.
  - L_t = 1 iff the best of these plans has mean expected share > 0.5 over 20 fresh streams.
  - X13's own rows could not be reused (other seeds; round 1 only; ORACLE manifesto), so the search code was re-run
    for X16's games. The seed choice is recorded in `seeds.choice`.
- **Metrics** (`metric_definitions`):
  - false-assert = P(target false | asserted), i.e. soundness;
  - missed-truth = P(not asserted | target true), i.e. completeness;
  - Brier score, over-confidence index OC = mean (p̂ − y)·sign(p̂ − ½), and the logistic calibration slope (< 1 means
    over-confident).
  - CIs come from a cluster bootstrap over games (2000 resamples).
- **Device and time:** cuda:0 (RTX A5000).
  - Wall time 3851 s (64.2 min), including the checks.
  - Peak GPU memory 2.01 GB. Mean GPU utilisation 46% (max 95%).
  - All from `provenance.compute` in reports/v3/x16.json.
  - The synthetic-world draw ran on the CPU in 8 spawn processes (see Deviations).
  - Long-run search: 20.6 s per game (`timing.long_run_seconds_mean_per_game`).
- **CPU vs GPU per library timeline** (`timing.library_timeline_cpu_vs_gpu`, BayesPlanner games, 1,600-voter
  worlds, 3 seeds):

  | Worlds | CPU | GPU | GPU speed-up |
  |---|---|---|---|
  | 16 | 3.35 s | 2.49 s | 1.35× |
  | 64 | 12.85 s | 9.43 s | 1.36× |

  The GPU gains little because ≈ 85% of a timeline is the library's CPU world generator.
- **Checks:**
  - **Library identity: passed** (`checks.library_identity`, 102 comparisons). The pooled sampler and the
    Π-prefix columns reproduce `timeline_with_diagnostics(…, max_policies=k)`: identical verdicts, p̂, best plan
    and truth row.
  - GPU float32 is not bit-reproducible: the library matched itself bit for bit in only 2 of 6 cells. Floats are
    therefore compared within 1e-5.
  - **CPU↔GPU parity: passed** (`provenance.parity`, seeds 13000–13002). E 0.611 vs 0.607; possible-rounds 5 vs 5;
    p̂ at 16 worlds 0.950 vs 0.992.
- **Deviations:** listed in `deviations` in reports/v3/x16.json.
  - cuda:0 instead of cuda:2.
  - World generation runs on the CPU (§1.1 point 5). The library's CPU generators define the worlds.
  - §1.1 point 4 is not done: decision points are not batched together, because `WorldBatch.step` applies one plan
    list to all worlds and Π_t differs per round. Each decision point's worlds × plans × voters are batched.
  - The long-run truth re-runs the X13 search per decision point.
  - The primary graded outcome is the official victory.

**Results.** Source: reports/v3/x16.json. "Pooled" means both players; n = 1600 and |Π| = all unless stated.

*How often each verdict is asserted, by W* (`world_axis_n1600_piall`, `cells.W*_n1600_piall.pooled.verdicts`):

| W | P(guaranteed) | missed-truth of guaranteed (truth row) | false-assert of guaranteed (row / long run) | false-assert of possible (row) |
|---|---|---|---|---|
| 8 | 0.772 [0.698, 0.840] | 0.211 [0.147, 0.280] | 0 / 0 (of 386) | 0.020 |
| 16 | 0.722 [0.644, 0.796] | 0.262 [0.191, 0.340] | 0 / 0 (of 361) | 0.022 |
| 32 | 0.668 | 0.317 | 0 / 0 | 0.022 |
| 64 | 0.616 [0.534, 0.696] | 0.370 [0.289, 0.453] | 0 / 0 (of 308) | 0.022 |

- **"Guaranteed" gets rarer as W grows** (`tests`, Holm over 12 tests):
  - pooled: W64 − W8 = −0.156 [−0.200, −0.110], exact McNemar p = 7e-24, Holm p = 8e-23;
  - BayesPlanner: −0.136 (0.856 → 0.720);
  - Top-3: −0.176 (0.688 → 0.512).
  - The same fall holds at every n and every |Π| (figure 1).
- **"Guaranteed" is never wrong.** It has zero false asserts in all 36 cells, against both the truth row (the
  witness plan wins in the true world) and the long-run truth.
  - At the default cell, 0 of 361 assertions failed, so the 95% upper bound on the error rate is ≈ 1%.
  - "knows_win_exists" also has zero false asserts in every cell.
- **The cost is completeness.** Missed-truth grows with W, from 0.21 to 0.37. Every extra sampled world is another
  chance for the best plan to lose in one world, which removes the ∀w claim.
- **World size matters more than W.** At W = 16, P(guaranteed) is 0.582 / 0.722 / 0.832 for n = 800 / 1600 / 4000.
  Small worlds are pessimistic.
- **Plan-set size matters little.** |Π| = 8 lowers P(guaranteed) by at most 0.05 against "all" (e.g. 0.706 vs
  0.722 at W16 n1600). |Π| = 16 equals "all" up to 0.002.
- **"Possible" is almost always asserted:** 1.000 at the default cell.
  - It is false against the truth row in 11 of 500 decision points (0.022). All 11 are close-run losses, with the
    best Π plan at 0.48–0.50 in the true world (raw/x16.jsonl).
  - Against the long-run truth it is false in only 4 of 500 decision points (Top-3, seeds 13005 and 13011).
- **"Inevitable defeat" is very rare:** asserted in at most 1.8% of decision points (W8 n4000 |Π| = 8). Every
  assertion is false against L_t (`…inevitable_defeat.long_run.false_assert` = 1.0, at most 9 cases per cell).
- **Ground truth is "winnable" almost everywhere** (`truth_agreement`):
  - the truth row holds in 489 of 500 decision points and L_t in 496;
  - L_t adds 7 decision points the declared Π misses;
  - the search's plan was the best on fresh streams in 497 of 500, by +0.020 [0.018, 0.021] E over the best Π plan.
  - Games: BayesPlanner won 50/50 (E 0.659), Top-3 won 45/50 (E 0.621) (`games`).

*Graded belief p̂ at the default cell* (`cells.W16_n1600_piall.pooled.graded`):

| Outcome y | Brier [95% CI] | OC [95% CI] | Calibration slope [95% CI] | Mean p̂ / base rate |
|---|---|---|---|---|
| game won | 0.032 [0.015, 0.055] | −0.024 [−0.050, 0.007] | 1.44 [0.90, 5.98] | 0.908 / 0.950 |
| best plan wins (truth row) | 0.034 [0.019, 0.054] | −0.022 [−0.044, 0.001] | 1.34 [1.01, 2.67] | 0.908 / 0.962 |
| long-run truth L_t | 0.047 [0.023, 0.077] | +0.002 [−0.031, 0.042] | 0.87 [0.71, 1.35] | 0.908 / 0.992 |

- **p̂ leans under-confident, not over-confident.**
  - In the reliability diagram, the decision points with p̂ ∈ [0.5, 0.9) (84 of 500) won at 0.75–1.00, above the
    diagonal.
  - p̂ ∈ [0.9, 1] (385 of 500) won at 1.00.
  - The over-confidence test at W16 is not significant for any outcome: Holm p = 0.87, 0.56 and 1.0.
- **By cell:**
  - Under-confidence is strongest with small worlds: OC −0.045, slope 2.27 at W8 n800.
  - It vanishes with large worlds: at W16 n4000, OC +0.002 and slope 1.27; at W64 n4000, OC +0.011 and slope 1.05.
  - The Brier score is ≈ 0.03 at n ≥ 1600 for every W, and 0.05–0.07 at n = 800.
- **The Brier score is small mostly because wins are common** (base rate 0.95). In the Murphy decomposition,
  reliability is 0.016 and resolution 0.031, against an uncertainty of 0.048.

Figures:
- ![X16 guaranteed vs worlds](..figures/x16_guaranteed_vs_worlds.png)
- ![X16 reliability of p̂](..figures/x16_reliability.png)
- ![X16 verdict errors](..figures/x16_verdict_errors.png)

**Theory vs observation. Partial.**
- **"Guaranteed becomes rarer as the world count grows": confirmed.** It falls from 0.772 to 0.616 between 8 and 64
  worlds, with Holm p = 8e-23.
- **"... and more reliable": not testable as stated.** "Guaranteed" was never falsely asserted at any W (0 errors in
  every cell, against both truths), so there is no error rate left to fall.
  - Larger W only buys more missed truths.
  - Reading "reliable" as precision, the claim is already at its ceiling at W = 8. Reading it as overall accuracy,
    it is refuted: missed-truth rises from 0.21 to 0.37.
- **"p̂ is over-confident at 16 worlds": refuted.**
  - OC is −0.024 [−0.050, 0.007] and the calibration slope is 1.44 against the game outcome.
  - p̂ is, if anything, under-confident. The sampled worlds include pessimistic micro-states (small, noisy
    1,600-voter graphs) that the real 10⁴-voter electorate rarely produces.
  - Against the long-run truth the slope is 0.87 [0.71, 1.35], consistent with calibration.

**What it means.**
- **For the design:** the debrief's "guaranteed" is safe to show. It never told a player a plan would win when it did
  not.
  - Its weakness is the opposite: with the default 16 × 1,600 it stays silent in 26% of decision points where a
    winning plan existed.
  - More voters per world (4000) help more than more worlds. At W = 16, P(guaranteed) rises from 0.72 to 0.83 and
    missed-truth falls from 0.26 to 0.15, at about 2× the world-generation cost.
  - "Inevitable defeat" should carry a stronger caveat. It is rare, but in this game it was wrong every time it was
    said: a better plan than Π offers (L_t) still won.
  - |Π| above 16 adds nothing at these settings.
- **For a player:** treat "guaranteed" as reliable. Treat "not guaranteed" as weak evidence: most such positions
  were still winnable. A p̂ of 0.6–0.9 meant about a 90% chance to win.
- **Caveats:**
  - At ₫500 almost every position was winnable (L_t 99%), so the false-assert rates of the positive verdicts are
    measured on few true negatives (11–14 per cell against the truth row).
  - The truth row uses one evaluation stream.
  - The long-run search is a heuristic lower bound on the true optimum.

**Data.** reports/v3/x16.json, reports/v3/raw/x16.jsonl (100 game rows with the truth row and the long-run search
per decision point, and 1,200 cell rows with per-round verdicts for each |Π|), and the log reports/v3/logs/x16.log.
Command:
`OMP_NUM_THREADS=4 python experiments/x16_verdicts.py --seeds 50 --first-seed 13000 --device cuda:0 --out reports/v3/x16.json`.
Library change (approved, D2): `max_policies` in `logic/epistemic.policy_set`, threaded through
`logic/report.timeline_with_diagnostics` / `epistemic_timeline`. The default is bit for bit unchanged
(`tests/unit/test_policy_set_max_policies.py`).

### X8 — Temporal properties of campaigns

**Status:** done. Every winner came from behind (548/548); the point of no return is round 1 (the manifesto commit) for 77% of weak losses, not round 2–3. Partly confirmed.

**Status:** done (all 1,000 X6 games; 64 sampled futures per game and decision round on cuda:0).

**Question.** How do campaigns unfold in time?
- Do winners lead early or come from behind?
- How often is a player never far behind?
- From which round is a lost game really lost? This is the point of no return, read off the branching futures
  (CTL-style EF / AF Victory).

**Prediction (pre-registered).** Copied unchanged from `reports/v3/predictions.md`:
> - **Predict:** most winners come from behind (as v2 found); the point of no return is round 2–3 for weak strategies.

**Setup.**
- **Data:** the X6 traces (`raw/x6.jsonl`): seeds 12000–12099, 9 players + ORACLE = 1,000 games. N = 10⁴, T = 5,
  ₫500, ₫20 workers, ₫10 queries.
- **Traces** (`logic.temporal`, unchanged). There are 7 states:
  - s₀: after the manifesto commit, before round 1;
  - s₁…s₅: after each round;
  - s₆: election day.
- **Two views of the margin:**
  - **poll** (the player's view, as `game_trace` and v2): the poll published at that point − 0.5;
  - **true:** E − 0.5.
  - `Lead` = margin > 0. On election day, `Lead` = `Victory` (the official result).
  - The ORACLE has no polls, so it has only the true view.
- **Formulas:**
  - ◇Lead = F[≤5] Lead from s₀ (led before election day);
  - □(margin > −0.05);
  - Comeback = ¬Lead U (Lead ∧ ◇Victory);
  - STL robustness of each (`robustness`).
- **Branching futures (GPU):**
  - **Rebuilding the state.** For each game and each decision round k = 1…5, the true state before round k is
    rebuilt with `x_batch.run_batch`: the same electorate, the game's manifesto, and its recorded allocations for
    rounds 1…k−1 on the game's own canonical seed streams.
  - **Replay check.** The replay is exact: 1,000/1,000 victories match, and the max |ΔE| along the trajectory is
    1.2e-7 (`replay`).
  - **Paths.** From that state the future branches over 8 continuation plans × 8 stream salts = 64 paths.
    - The plans: own remaining recorded plan, ORACLE planner, even workers, even media, even split, all-in workers
      now, all-in workers in the best-matched region, nothing.
    - The game's remaining treasury is used, net of query spending.
    - Salt 0 = the canonical streams. Salts 1–7 re-draw every stream from round k on, and the election.
  - **Verdicts:**
    - EF = some path wins (`exists_eventually`);
    - AF = every path wins (`forall_eventually`);
    - ⟨⟨p⟩⟩◇Victory = one plan wins on all 8 salts (`strategy_eventually`);
    - p_best = max over plans of the plan's win rate over the salts.
  - **Point of no return (PNR)** of a lost game: the first k from which EF is false at k and at every later
    round.
    - k = 1 means the game was lost at the manifesto commit.
    - k = 6 means it was still viable before the last round.
- **Compute:** cuda:0 only (hard-pinned, with a 4 GB per-process cap). The batch was calibrated at 640 worlds and
  backed off on OOM to **160 worlds** per `run_batch`. Values from `provenance.compute`:
  - peak 3.93 GB;
  - mean GPU utilisation 37% (max 99%);
  - wall time 1,434 s (23.9 min), including the CPU parity check.
- **Parity (§1.1.7):** CPU vs GPU, 3 seeds × 3 players, p_best and the path win rate at rounds 1 and 3. Passed
  (`provenance.parity`, tolerance 0.1).
- **Deviations:**
  - cuda:0 instead of cuda:2.
  - EF / AF are over the sampled 64-path tree, not over all plans and all randomness.
  - Because the realised game is one of the paths (own plan, salt 0), every won game has EF true at every round
    by construction. So P(win | EF false) = 0 is not informative.

**Results.** All numbers are from reports/v3/x8.json → `players.<player>` unless another key is named.

*Linear-time properties (poll view; ORACLE: true view).* Robustness values are means with t-CIs.

| player | wins | ◇Lead | □(margin > −0.05) | □Lead | Comeback (share of winners) | ρ(◇ margin > 0) | lead changes |
|---|---|---|---|---|---|---|---|
| ORACLE (true E) | 95 | 0.95 [0.89, 0.98] | 0.04 [0.02, 0.10] | 0 | 95/95 | +0.128 [0.113, 0.143] | 0.95 |
| II + ADMM | 99 | 0.99 [0.95, 1.00] | 0.05 [0.02, 0.11] | 0 | 99/99 | +0.149 [0.135, 0.162] | 1.01 |
| II Bayes (PG) | 98 | 0.95 [0.89, 0.98] | 0.05 [0.02, 0.11] | 0 | 98/98 | +0.140 [0.126, 0.155] | 1.02 |
| top-3 | 85 | 0.85 [0.77, 0.91] | 0.03 [0.01, 0.08] | 0 | 85/85 | +0.102 [0.085, 0.119] | 0.91 |
| II + blend | 84 | 0.86 [0.78, 0.91] | 0.02 [0.01, 0.07] | 0 | 84/84 | +0.097 [0.082, 0.112] | 0.94 |
| I People Pleaser | 73 | 0.74 [0.65, 0.82] | 0.02 [0.01, 0.07] | 0 | 73/73 | +0.063 [0.046, 0.079] | 0.81 |
| uniform | 9 | 0.09 [0.05, 0.16] | 0.01 [0.00, 0.05] | 0 | 9/9 | −0.104 [−0.119, −0.088] | 0.21 |
| III random b | 3 | 0.04 [0.02, 0.10] | 0 | 0 | 3/3 | −0.334 | 0.05 |
| random | 2 | 0.03 [0.01, 0.08] | 0 | 0 | 2/2 | −0.346 | 0.04 |
| III Strong-Arm | 0 | 0 | 0 | 0 | — | −0.402 [−0.413, −0.390] | 0 |

- **Every winner came from behind:**
  - 453/453 winners in the poll view and 548/548 in the true view [0.993, 1.000] (`pooled.winners_*`).
  - Every one of them trailed at s₀ (`behind_at_start`). The manifesto alone never gives a lead; the campaign
    makes it.
- **When winners first lead:**
  - BayesPlanner winners first lead in the poll after round 3 (31) or round 4 (37) (`poll.winners.first_lead_k_hist`).
  - 3/98 never led in a poll before winning.
  - Comeback robustness for winners is small: +0.024 [0.021, 0.027] for Bayes. Typical wins are by 2–3 points.
- **□(margin > −0.05) almost never holds** (≤ 5% for every player). A campaign starts more than 5 points behind.

*Branching futures.* EF = share of games with some winning sampled future before round k; `strategy_EF` = one plan
wins on all 8 salts.

| player | EF k=1 | EF k=3 | EF k=5 | ⟨⟨p⟩⟩◇Victory k=1 | AF k=1 → k=5 | p_best k=1 | losses: PNR histogram k=1/2/3/4/5/6 | median PNR |
|---|---|---|---|---|---|---|---|---|
| ORACLE | 0.95 | 0.97 | 0.95 | 0.95 | 0.34 → 0.92 | 0.950 | 2/0/1/2/0/0 | 3 |
| II Bayes (PG) | 0.99 | 0.99 | 0.99 | 0.96 | 0.33 → 0.96 | 0.980 | 1/0/0/0/0/1 | — (2 losses) |
| II + ADMM | 1.00 | 1.00 | 1.00 | 0.97 | 0.33 → 0.98 | 0.989 | 0/0/0/0/0/1 | — (1 loss) |
| top-3 | 0.96 | 0.93 | 0.87 | 0.91 | 0.20 → 0.81 | 0.940 | 4/1/2/1/5/2 | 4 |
| II + blend | 0.89 | 0.87 | 0.87 | 0.80 | 0.15 → 0.80 | 0.856 | 9/2/1/1/0/3 | 1 |
| I People Pleaser | 0.84 | 0.84 | 0.79 | 0.78 | 0.15 → 0.71 | 0.820 | 13/0/1/2/5/6 | 3 |
| uniform (no spending) | 0.75 | 0.53 | 0.27 | 0.66 | 0.07 → 0.07 | 0.708 | 25/11/11/13/13/18 | 3 |
| III random b | 0.03 | 0.05 | 0.04 | 0.03 | 0 → 0.02 | 0.030 | 94/0/0/0/2/1 | 1 |
| random | 0.03 | 0.03 | 0.02 | 0.02 | 0 → 0.02 | 0.021 | 97/0/0/0/1/0 | 1 |
| III Strong-Arm | 0 | 0 | 0 | 0 | 0 → 0 | 0 | 100/0/0/0/0/0 | 1 |

- **The point of no return for weak strategies is the commit, not round 2–3.**
  - Over the losses of the seven weak players (uniform, random, both Strong-Arms, People Pleaser, top-3, II +
    blend; 444 losses), PNR = 1 for 342 (0.77 [0.73, 0.81]).
  - PNR is 2 or 3 for only 29 (0.065 [0.046, 0.092]); the mean PNR is 1.79 [1.64, 1.93]
    (`pooled.weak_losses_pnr`).
  - The manifesto decides: with the Strong-Arm or a random b, no continuation in Π wins any sampled future even
    before round 1.
- **The passive baseline is the exception.**
  - uniform (b uniform, no spending) still had a winning future before round 1 in 75% of games, but each idle round
    removes about 12 points of viability (EF 0.75 → 0.64 → 0.53 → 0.40 → 0.27).
  - Its losses spread over every round (median PNR 3). This is the "round 2–3" pattern the prediction expected,
    and it belongs to the do-nothing player only.
  - The best continuation is mostly the ORACLE planner: 0.70 → 0.25 win rate from k = 1 to k = 5
    (`p_by_cont_by_round.planner`).
- **People Pleaser:**
  - About half of its losses are lost at the commit (13/27, 0.48 [0.31, 0.66]).
  - The rest stay viable until rounds 5–6 (11/27), so a better plan would still have won them.
  - Over all its games, the ORACLE-planner continuation wins 0.79–0.81 of the paths, against 0.73 for its own
    plan (`p_by_cont_by_round`).
- **Strong players:** BayesPlanner, ADMM and the ORACLE have ⟨⟨p⟩⟩◇Victory in 95–97% of games from round 1 on,
  and AF rises from 0.33 to 0.96 as the rounds pass.
  - AF is low at k = 1 only because "spend nothing" is one of the paths.
  - The two Bayes losses are 12012, lost at the commit (the "divided majority" world of X15, where the ORACLE also
    lost at k = 1), and 12086, viable until the last round (raw/x8.jsonl).
- **Knowledge vs outcome** (`tests.knowledge_vs_outcome`, all 1,000 games):
  - P(win | ⟨⟨p⟩⟩◇Victory at round 1) = 0.89 [0.86, 0.91] (539/608);
  - P(win | not) = 0.023 [0.012, 0.043] (9/392);
  - P(win | EF at round 5) = 0.94 [0.92, 0.96].
- **Paired p_best** (`tests.P_best_paired`; Wilcoxon, Holm over 6 tests):
  - Bayes − People Pleaser: +0.160 [0.091, 0.229] at round 1, Holm p = 2e-4;
  - +0.151 at round 3;
  - Bayes − top-3: +0.040 [0.006, 0.074] at round 1 (Holm p = 0.018) and +0.095 [0.041, 0.149] at round 3
    (Holm p = 0.003).
- EF is not exactly monotone in k: it is estimated on 8 sampled futures from the *realised* state, and a lucky
  realised round can reopen a game. For example, ORACLE 12002 had no winning sampled future before round 1 but one
  before round 2 (raw/x8.jsonl).

Figures:
- ![X8 viability](..figures/x8_viability.png)
- ![X8 point of no return](..figures/x8_point_of_no_return.png)
- ![X8 LTL properties](..figures/x8_ltl.png)

**Theory vs observation. Partial.**
- **"Most winners come from behind": confirmed, in the strongest form.** All 548 winners (453 in the poll view)
  trailed at s₀. As in v2, □Lead never holds (0/1,000). Every player's commit-time share is below 50%, and the
  lead is made by the campaign, typically after round 3–4.
- **"The point of no return is round 2–3 for weak strategies": refuted.**
  - For weak *manifestos* (Strong-Arm, random b, and most People Pleaser and blend losses) the game is already
    lost at the commit: PNR = 1 for 77% of weak-player losses, and 2–3 for only 6.5%.
  - Only the passive no-spending baseline shows a gradual point of no return (median 3, spread over all rounds).
  - The strong players almost never cross a point of no return: 8 losses over ORACLE, Bayes and ADMM.

**What it means.**
- **For the design:**
  - The game is decided mostly at the manifesto. The campaign turns a viable manifesto into a win (AF rises from
    about 0.3 to above 0.9), but it cannot rescue a bad one.
  - Trailing early is normal, so the UI should not call an early trailing poll "losing". A useful debrief signal
    is the strategic verdict at round 1: 0.89 vs 0.02 win rates.
- **For a player:**
  - Get the manifesto right first. A winning player is behind at the start almost surely and leads only from round
    3–4.
  - Do not wait: an idle player loses about 12 points of viability per round.
  - If you are still viable late (People Pleaser's late losses), switching to the planner's allocation would have
    won 6–8 points more often than its own plan.

**Data.** reports/v3/x8.json, reports/v3/raw/x8.jsonl (one row per game: LTL booleans and robustness in both views;
EF/AF/strategy, p_best and per-plan win rates per round; PNR; replay check), log reports/v3/logs/x8.log. Command:
`OMP_NUM_THREADS=4 python experiments/x8_temporal.py --device cuda:0 --out reports/v3/x8.json`; figures:
`python experiments/plot_v3_x8_x15.py --only x8`.

## 8. Robustness and explanation

### X15 — Which elections are unwinnable?

**Status:** done. Losses concentrate where the two largest regions are far apart (all 11 ORACLE losses at TV ≥ 0.35); network fragmentation is a weak effect; personality does not pick the best channel (split wins 92/100). Partly confirmed.

**Status:** done (post-hoc, CPU). Bayes loss could not be modelled (2 events); its margin was modelled instead.

**Question.** Which hidden features of a world make an election hard to win, even for the full-information ORACLE
and for the best observation-only player (BayesPlanner)? Can a simple model predict the loss, and what is a short
"why you lost" story?

**Prediction (pre-registered).** Copied unchanged from `reports/v3/predictions.md`:
> - **Predict:** losses concentrate where the two largest regions have distant archetypes and a fragmented network
>   (low p_g); the largest region's personality decides the best channel.

**Setup.**
- **Data (no new games):**
  - X0 `raw/x0.jsonl`: seeds 10000–10099, ORACLE = oracle manifesto + ORACLE planner, at ₫500 / 750 / 1000. The
    channel arms (informed manifesto + pay_workers / pay_media / pay_split) come from ₫500.
  - X6 `raw/x6.jsonl`: seeds 12000–12099, ₫500. ORACLE, BayesPlanner (II), People Pleaser (I).
  - X10 default cell `raw/x10.jsonl` (N = 10⁴, T = 5, ₫500, seeds 10000–10049): these are the X0 ₫500 worlds
    run again. 550/550 rows agree (max |ΔE| = 3e-7 from GPU float sums, 0 different results;
    reports/v3/x15.json → `x10_duplication`). So X10 adds no new world, and each world is counted once.
  - **200 worlds** in total (100 X0 + 100 X6), N = 10⁴, T = 5, ₫500, ₫20 workers, ₫10 queries.
- **Features** (`x_common.features_of`, canonical CPU electorates, SciPy spectra; cached in
  `raw/features_<seed>.json`):
  - archetype distances: `tv_two_largest` (TV distance between the archetype means of the two largest regions)
    and `tv_mean_pair` (the mean over all 4 regions);
  - fragmentation: `p_g_mean` (mean realised inter-region SBM probability; low = fragmented), `p_g_two_largest`,
    `inter_edge_share`, λ₂(L), the random-walk gap;
  - sizes: Gini, largest share, two-largest share;
  - zealot share, mean p_r, and the largest region's personality (dummies P1/P2/P3; P0 is the reference).
  - `start_share` (true E of the ORACLE manifesto before round 1) is used only in a second model, because it is
    partly an outcome.
- **Targets:**
  - ORACLE loss at ₫500 (11 events in 200 worlds);
  - Bayes loss (2 events in 100 X6 worlds; **not fitted**, see Deviations);
  - margins E − 0.5 for ORACLE (n = 200) and BayesPlanner (n = 100);
  - supplementary: People Pleaser loss (27 events in 100 worlds).
- **Models:**
  - (a) logistic regression on standardised features, with a weak ridge (λ = 1) against separation; ridge-linear
    for the margins. 95% CIs from 2,000 bootstrap resamples, stratified by outcome.
  - (b) a depth-3 CART tree (min leaf 5).
  - Scored by 5-fold stratified CV AUC (R² for the margins), repeated 20×, with a 200-shuffle label-permutation
    null.
- **Pre-registered tests (Holm over H1–H3):**
  - H1: the two largest regions' archetype distance is higher in ORACLE-loss worlds (one-sided Mann–Whitney);
  - H2: p_g is lower in ORACLE-loss worlds (one-sided Mann–Whitney);
  - H3: the largest region's personality is associated with the best paid channel (chi-square, X0 ₫500).
- **Compute:** CPU only (a §1.1 point 5 exception). Wall time 131 s, of which 87 s were spent rebuilding the
  electorates for the features (`provenance.compute.wall_seconds`, `provenance.feature_seconds`). No GPU, so there
  is no parity check.

**Results.** All numbers are from reports/v3/x15.json.

*Where the losses are* (`loss_by_archetype_distance`):

| two largest regions' archetype TV | worlds | ORACLE losses | People Pleaser losses (X6 worlds) | Bayes losses | mean ORACLE E |
|---|---|---|---|---|---|
| 0.20 | 12 | 0 | 0 / 4 | 0 | 0.690 |
| 0.25 | 88 | 0 | 3 / 44 | 0 | 0.672 |
| 0.35 | 40 | 6 | 8 / 21 | 1 | 0.584 |
| 0.40 | 60 | 5 | 16 / 31 | 1 | 0.587 |

- **Every ORACLE loss is in a world whose two largest regions are far apart.** With TV ≥ 0.35 the ORACLE loses
  11/100 [0.06, 0.19]. With TV < 0.35 it loses 0/100 [0, 0.037]. Fisher one-sided p = 0.0004
  (`fisher_tv_two_largest_ge_0.35`).
- Over all four regions, `tv_mean_pair` = 0.35 (the most divided worlds) holds 10 of the 11 ORACLE losses and both
  Bayes losses.
- The ORACLE loses less as the treasury grows: 6/100 at ₫500, 2/100 at ₫750, 1/100 at ₫1000 (X0 worlds,
  `oracle_loss_by_treasury_x0`).

*Models* (`models.<target>`):

| target | n (events) | CV AUC / R², logistic | CV, tree | permutation null q95, p (logistic) | top-3 features (\|coef\|) |
|---|---|---|---|---|---|
| ORACLE loss | 200 (11) | AUC 0.91 [0.88–0.94 over the repeats] | 0.76 | 0.68, p = 0.005 | tv_mean_pair, inter_edge_share, two_largest_share |
| ORACLE loss + start_share | 200 (11) | AUC 0.94 | 0.75 | 0.68, p = 0.005 | start_share, tv_mean_pair, inter_edge_share |
| Bayes loss | 100 (2) | not fitted | — | — | — |
| ORACLE margin | 200 | R² 0.83 | 0.80 | — | tv_mean_pair, tv_two_largest, inter_edge_share |
| Bayes margin | 100 | R² 0.83 | 0.71 | — | tv_mean_pair, tv_two_largest, two_largest_share |
| People Pleaser loss | 100 (27) | AUC 0.99 | 0.93 | 0.63, p = 0.005 | tv_mean_pair, tv_two_largest, largest_P3 |

- **ORACLE loss, standardised log-odds per SD [95% bootstrap CI]:**
  - tv_mean_pair +2.31 [1.84, 2.59];
  - tv_two_largest +0.49 [0.06, 1.08];
  - inter_edge_share −0.61 [−1.18, −0.02];
  - p_g_mean −0.31 [−1.05, 0.32], so the CI includes 0;
  - every other CI includes 0.
- **ORACLE loss tree:**
  - The first split is tv_mean_pair > 0.342 (29 worlds, loss rate 0.34).
  - Under it, p_g_mean ≤ 0.102 gives 10 worlds with loss rate 0.70. Inside those, a largest region ≤ 32% of N
    gives 5/5 losses.
  - Below the first split (171 worlds) the loss rate is 0.006.
- **Margins:** one SD of tv_mean_pair costs ≈ 0.056 E, for the ORACLE and for BayesPlanner alike. One SD of
  tv_two_largest costs ≈ 0.02 E. No network or personality coefficient has a CI that excludes 0.
- **Pre-registered tests** (`hypotheses`):
  - H1, archetype distance: median 0.35 in loss worlds vs 0.25 in won worlds; Holm p = 0.011.
  - H2, p_g: median 0.084 in loss worlds vs 0.105 in won worlds; Holm p = 0.032.
  - H3, personality vs channel: Cramér's V = 0.31; Holm p = 0.032.
- **Best channel** (`channel_by_personality`, X0 ₫500):
  - **split** is the best paid pattern in 92/100 worlds. Workers win the other 8, all in worlds whose largest
    region is P1 (4) or P2 (4).
  - Media never beats workers. E(media) − E(workers) is least negative when the largest region is P1
    media-savvy: −0.061 [−0.071, −0.051]. It is most negative for P0 social: −0.082 [−0.088, −0.076]
    (Kruskal–Wallis p = 0.010).

*"Why you lost" taxonomy* (`taxonomy`; tercile thresholds over the 200 worlds; the first class that matches wins):

| class | rule | ORACLE (11) | Bayes (2) | People Pleaser (27) | share of won ORACLE worlds |
|---|---|---|---|---|---|
| close call | \|E − 0.5\| ≤ 0.01: the election draw decided | 3 | 1 (12086) | 4 | — |
| divided majority | two largest regions' TV ≥ 0.35 **and** start share ≤ 0.349 | 8 | 1 (12012) | 22 | 45/189 |
| low start | start share ≤ 0.349 only | 0 | 0 | 1 | 11/189 |
| divided regions, fragmented network, independent largest, other | — | 0 | 0 | 0 | 133/189 |

- Every non-close-call loss of every player is a "divided majority" world, except one People Pleaser loss (low
  start).
- The class is necessary, not sufficient: the ORACLE still wins 45 of the 53 divided-majority worlds.
- The two Bayes losses (`losses.bayes`):
  - 12012: the two largest regions have TV 0.40, p_g is low (0.070, below the 0.095 tercile), and the largest
    region is independent (P2). The ORACLE lost there too (E 0.456).
  - 12086: E 0.494 is a close call, in a world the ORACLE won (E 0.533).

Figures:
- ![X15 coefficients](..figures/x15_coefficients.png)
- ![X15 PDP ORACLE loss](..figures/x15_pdp_oracle_loss.png)
- ![X15 PDP ORACLE margin](..figures/x15_pdp_oracle_margin.png)
- ![X15 channel by personality](..figures/x15_channel_by_personality.png)

**Theory vs observation. Partial.**
- **"Losses concentrate where the two largest regions have distant archetypes": confirmed, and strongly.** All 11
  ORACLE losses are in the half of the worlds with TV ≥ 0.35, against 0 in the other half (Fisher p = 0.0004).
  The feature model, dominated by the archetype distances, explains about 83% of the variance in the ORACLE and
  Bayes margins (CV R²). The
  cause is the manifesto: a single b cannot sit near two distant region means. The start share falls by ≈ 0.04 E
  per SD of tv_mean_pair (`models.start_share`), and the campaign cannot close that gap.
- **"…and a fragmented network (low p_g)": weakly supported.** p_g is lower in loss worlds (Holm p = 0.032). The
  tree uses it inside the divided worlds (p_g ≤ 0.102: 7 of 10 lost). But its logistic and margin coefficients
  have CIs that include 0. inter_edge_share, the realised version, does have a CI that excludes 0 for ORACLE loss
  (−0.61). Fragmentation matters only once the electorate is already divided.
- **"The largest region's personality decides the best channel": refuted as stated.** The association is real
  but small (V = 0.31, Holm p = 0.032). Split is best in 92% of worlds whatever the personality. Personality only
  moves the size of the media deficit (least bad for P1 media-savvy). Workers win in 8 worlds, but never in a P0
  or P3 world. Personality has no CI-excluding effect on losing.

**What it means.**
- **For the design:**
  - Unwinnable worlds are the ones where the two largest regions want different things. About 1 in 10 of those
    is lost even with full information at ₫500, and 0 of the others.
  - This is a clear, explainable source of difficulty, and more treasury reduces it (6 → 2 → 1 losses from ₫500 to
    ₫1000).
  - A UI debrief can name the class directly from features the game already has: "divided majority" or "close
    call".
- **For a player:**
  - If the census and the survey show the two big regions far apart, the manifesto has to choose a coalition.
    Blending both (People Pleaser) loses 24 of 52 such worlds (`loss_by_archetype_distance`).
  - Treat media as a supplement to workers, never a replacement, even when the biggest region is media-savvy.

**Deviations.**
- X10's default cell repeats X0 ₫500 (550/550 rows), so it contributes no new worlds.
- Bayes loss has 2 events. No regression, tree or CV AUC was fitted; the Bayes margin was modelled instead, and
  the two losses are described above.
- The supplementary targets (People Pleaser loss, start share, margins) are not in the plan.
- scikit-learn is not installed in the env. The models, AUC and partial dependence are numpy code in the script.
  Nothing was installed.
- With 11 events the ORACLE-loss model is fragile. It uses a ridge, its CV AUC comes with a permutation null, and
  the bootstrap CIs are wide.

**Data.** reports/v3/x15.json, reports/v3/raw/x15.jsonl (one row per world: features, outcomes), the cached
features reports/v3/raw/features_<seed>.json, log reports/v3/logs/x15.log. Command:
`OMP_NUM_THREADS=4 python experiments/x15_hard_worlds.py --out reports/v3/x15.json`; figures:
`python experiments/plot_v3_x8_x15.py --only x15`.

### X20 — Luck vs skill: variance decomposition

**Status:** done. The world draw explains 97–99% of the variance in vote share; election noise is small. Partly confirmed: the predicted ordering of in-game noise sources is refuted.

**Status:** done (50 worlds × 3 policies × 8 streams × 21 draws; 24,150 games).

**Question.** How much of the outcome is decided by the player's decisions, and how much by randomness? Which random
source matters most?

**Prediction (pre-registered).** Copied unchanged from `reports/v3/predictions.md`:
> - **Predict:** world draw > worker picks > cascade > media ≈ election at N = 10⁴. Election-day noise is tiny
>   (sd ≈ 0.005).

**Setup.**
- **Seeds:** 50 worlds, 14000–14049 (`seed_block("X20", 50)`), paired across policies and streams.
- **Game:** N = 10⁴, T = 5, ₫500, worker ₫20, query ₫10.
- **Policies** (all with the ORACLE manifesto, as X0's `oracle/*` runs): `planner` (ORACLE planner), `pay_split`
  ("split"), `none`.
- **Streams re-drawn one at a time** with `x_streams.SaltedStreams`, salts 1–20 plus the canonical draw (salt 0), so
  21 draws per (world, stream):
  - in-game: `cascade`, `media`, `workers` (which voters are picked), via `x_batch.run_batch(..., streams=...)`;
  - `election`: only the ballot draw. The final v of the salt-0 game is re-voted with `run_election` and the salted
    `election` generator (exactly what `run_batch` would do);
  - `world`: all six generation streams (regions, block_probs, graph, preferences, attributes, zealots) through
    `generate_electorate(cfg, seeds=SaltedStreams(...))`. The ORACLE manifesto is recomputed for every new world.
  - Extra: `world_graph` (block_probs + graph), `world_prefs` (σ and voter preferences), `world_attrs` (personalities,
    susceptibilities, zealots), with the region sizes kept.
- **Metrics:** E = expected share after the campaign (before the ballot); V = realised vote share; win = V > ½.
- **Decomposition** (per policy and metric; definitions in `metric_definitions`):
  - σ²_s = the mean over worlds of the variance over the 21 draws of stream s;
  - σ²_total = the variance over all 50 × 21 world-stream draws;
  - σ²_world = σ²_total − Σ σ²_s over the four luck streams (additive model);
  - **skill share** = Var(policy means) / [Var(policy means) + mean within-policy σ²_total].
  - 95% CIs: bootstrap over the 50 worlds (2,000 resamples), `bootstrap_ci95["metric|policy|component"]`.
- **Compute** (`provenance.compute`): cuda:2 (RTX A5000), pinned. Batches of B = 250 games for the in-game streams
  (`provenance.calibration.batch`), and chunks of 100 fresh worlds for the world streams. Peak memory 3.36 GB; mean
  GPU utilisation 27% (max 93%); wall time 793 s. The four world variants took 156–216 s each (`provenance.timings`),
  most of it the ORACLE manifesto of each new world.
- **Parity (CPU↔GPU, 3 seeds, canonical CPU-drawn worlds, ORACLE planner)** (`provenance.parity`): passed. E at salt 0
  was 0.6157 (CPU) vs 0.6164 (GPU); with cascade salt 1 it was 0.6157 vs 0.6177; victories were identical.
  `SaltedStreams` with all salts at 0 reproduces `SeedStreams` exactly (`salted_zero_equals_seedstreams`).
- **Deviations:**
  - Electorates were drawn on the GPU (`generation="device"`, allowed by §1.1 point 3), not with the canonical CPU
    draw.
  - Extra, beyond the plan: the world split into graph, preferences and attributes, and V reported alongside E and win.

**Results.** Within-world sd of each source (`components.<metric>.<policy>.<stream>.within_sd`; the world row is
√`world_var`), and shares of σ²_total (`components.<metric>.<policy>.shares`):

| source | planner sd(V) | share of Var(V), planner [95% CI] | split sd(V) | share of Var(V), split | share of Var(win), planner |
|---|---|---|---|---|---|
| world draw (sd derived: √`world_var`) | 0.0746 | 0.994 [0.993, 0.995] | 0.0783 | 0.972 | 0.38 [−0.35, 0.99] |
| — of which preferences (σ, voters) | 0.0719 | 0.92 [0.86, 0.99] | 0.0747 | 0.89 | 0.82 |
| — of which graph | 0.0190 | 0.064 [0.048, 0.087] | 0.0211 | 0.071 | 0.59 |
| — of which attributes / zealots | 0.0147 | 0.039 [0.033, 0.046] | 0.0225 | 0.080 | 0.13 |
| cascade | 0.0035 | 0.0022 [0.0015, 0.0031] | 0.0086 | 0.0117 | 0.20 [0, 0.45] |
| worker picks | 0.0028 | 0.0014 [0.0010, 0.0019] | 0.0095 | 0.0142 | 0.17 [0.01, 0.40] |
| election | 0.0034 | 0.0021 [0.0018, 0.0023] | 0.0035 | 0.0019 | 0.20 [0, 0.46] |
| media | 0.0011 | 0.0002 [0.0002, 0.0002] | 0.0011 | 0.0002 | 0.04 [0, 0.12] |

The world sub-components are each measured alone, so they do not add up to the world share. On E the election
stream is exactly 0 by construction, since E is computed before the ballot; the other E shares are within 0.001 of the
V shares (`components.E`).

- **Election-day noise** (`election_noise_sd_V`): sd(V) = 0.0034 (planner), 0.0035 (split) and 0.0041 (none).
- **Is the additive model right?** For E and V, yes. The between-world ANOVA variance with only the election varying
  (0.005564) matches σ²_world + the in-game components (0.005592) (`components.V.planner.additivity_check`). For win
  it does not (planner 0.0621 vs 0.0495). The win shares of the luck streams overlap, and the world share's CI
  includes 0, because a 0/1 outcome changes only in worlds near ½, and there several sources interact.
- **Pairwise order of the luck streams.** Wilcoxon tests on the per-world sd, 12 tests, Holm (`pair_tests`):
  - **Planner:** worker picks < cascade on V (−0.0007 [−0.0010, −0.0005], Holm p 9e-7) and on E (Holm p 1e-5).
    With split, worker picks ≈ cascade (+0.0005 [−0.0003, +0.0014], Holm p 1).
  - **Cascade > media:** in all four tests (Holm p 2e-14).
  - **Media < election on V:** −0.0023 [−0.0025, −0.0021] for the planner and −0.0024 for split (Holm p 2e-14).
- **Skill** (`skill.<metric>`):
  - **All three policies:** the skill share is 0.50 [0.48, 0.52] on E, 0.49 on V and 0.31 [0.28, 0.34] on win. The
    policy means of E are 0.632 (planner), 0.625 (split) and 0.475 (none).
  - **Planner vs split only:** the skill share is 0.002 [0.001, 0.003] on E and 0.002 on win. Mean wins over the
    world draws are 0.933 vs 0.911.
- **Canonical draw** (`canonical`): planner wins 0.92 [0.81, 0.97], split 0.88 and none 0.32. Mean E is 0.626, 0.615
  and 0.467.

Figures:
- ![X20 components](..figures/x20_components.png)
- ![X20 shares](..figures/x20_shares.png)

**Theory vs observation. Partial.**
- **"World draw first": confirmed, by a wide margin.** The world draw gives 99.4% of Var(V) for the planner and
  97.2% for split. Most of it comes from the preferences (σ and the voters, 92%), not the graph (6%). A world's sd of
  0.075 in V is 20–70 times any in-game source.
- **"Worker picks > cascade": refuted for the ORACLE planner and a tie for split.** The planner's cascade sd is
  larger (0.0035 vs 0.0028, Holm p 9e-7); with split the two are equal within noise.
- **"media ≈ election": refuted.** The election sd (0.0034) is three times the media sd (0.0011). Media noise only
  re-draws who of an aired region is exposed (p_m = 0.6), and that averages out over thousands of voters.
- **"Election-day noise is tiny (sd ≈ 0.005)": confirmed.** The measured sd is 0.0034–0.0041, below the Bernoulli
  bound √(0.25/N) = 0.005, because many voters have P close to 0 or 1.
- **The measured order** on V for the planner is world ≫ cascade ≈ election > worker picks ≫ media.

**What it means.**
- **For the game design:**
  - Given the world, the outcome is almost deterministic in expected share: all in-game randomness together has an sd of
    about 0.013 at most (split; the in-game `within_var` added up).
  - Luck matters only for the win/lose bit in close worlds. There, the cascade, the worker picks and the ballot
    each move the within-world win sd by about 0.10–0.14.
  - Which world you are dealt (mostly σ: what the regions want) explains about half the pooled variance of E; the
    choice of strategy explains the other half.
  - That "skill" is mostly "campaign at all versus not": the two sensible spending patterns (ORACLE planner vs
    split) differ by only 0.2% of the variance. This matches X0's finding that "split" is near-optimal at ₫20.
- **For a player:**
  - Losses are mostly decided before the first move, by the electorate, so a loss is weak evidence of bad play.
  - Spending at all is the big decision.
  - Fine-tuning between good allocations matters little next to the world, and the dice in a game (cascade,
    picks, ballot) only matter when the race is close to 50%.

**Data.** reports/v3/x20.json, reports/v3/raw/x20.jsonl (24,150 rows), log reports/v3/logs/x20.log. Command:
`OMP_NUM_THREADS=4 python experiments/x20_luck_skill.py --seeds 50 --first-seed 14000 --salts 20 --device cuda:2 --out reports/v3/x20.json`.

### X21 — Robustness to model misspecification

**Status:** done. Most sensitive to z (Δgap 0.034); least sensitive is ζ, not κ; no change in win rate survives the Holm correction. Partly confirmed.

**Runs:** 16 arms × 50 paired seeds; 1,550 BayesPlanner engine games and 800 ORACLE games; derived: 50 × (16 `bayes` + 15 `bayes_true` arms) and 50 × 16, together the 2,350 rows of raw/x21.jsonl.

**Question.** BayesPlanner models the world with the public default parameters (`players.public_config` =
`GameConfig()` defaults, except N, T and the treasury). How badly does it do when the true world differs?

**Prediction (pre-registered).** Copied unchanged from `reports/v3/predictions.md`:
> - **Predict:** the planner is most sensitive to z and β_v (they define who votes); the least sensitive is κ.

**Setup.**
- **Seeds:** 14000–14049 (`seed_block("X21", 50)`), the same in every arm.
- **Game:** N = 10⁴, T = 5, ₫500, worker ₫20, query ₫10.
- **Arms:** one true (engine) parameter changed at a time, by ×0.5 and ×1.5 (`arms.<arm>.set`):
  - ζ (zealot fraction) 0.025 / 0.075;
  - κ (Dirichlet concentration) 30 / 90;
  - k̄ (mean degree) 15 / 45;
  - ω (social step) 0.075 / 0.225;
  - β_v (vote sensitivity) 10 / 30;
  - z (vote threshold) 0.4125;
  - p_m (media exposure) 0.3 / 0.9;
  - z on the margin 1 − z: 0.9125 and 0.7375 (see Deviations).
- **Players per (arm, seed):**
  - `bayes`: `BayesPlanner(seed, device=cuda:2)`, which keeps the defaults (misspecified). It plays a full
    `GameEngine` game on observations only.
  - `oracle`: `oracle_manifesto` + the ORACLE planner on the true perturbed state (`x_batch.run_batch`; same streams as
    the engine).
  - `bayes_true` (extra control): the same BayesPlanner given the true perturbed config. Its survey answer model
    `bayes.build_answer_model()` takes no config and is unchanged.
- **Measures** (`metric_definitions`):
  - **Δgap (primary, as in the plan):** (E_bayes − E_oracle)_arm − (E_bayes − E_oracle)_default, paired by seed.
  - **Misspecification cost:** E_bayes − E_bayes_true, in the same world with the same streams.
  - **Sensitivity of a parameter:** the larger |mean| over its arms (`sensitivity`).
  - Holm within each family of tests (`test_families`).
- **Leak check, run first as an assertion** (`leak_check`): passed for 15 arms × 2 seeds.
  - `costs` is identical in every arm.
  - For ω, β_v, z and p_m the whole initial observation is identical to the default one.
  - For κ only `free_stats` differs, and for k̄ only the census fields `regions` and `inter_edges` (realised public
    statistics, allowed). For ζ nothing differs.
  - No observation key (initial or final, in every one of the 1,550 games, derived as above) is named like a perturbed config field,
    except the census `regions[*].mean_degree`.
  - `public_config(obs)` always equals the default public config, and still holds the default value of the
    perturbed parameter.
- **Compute** (`provenance.compute`): cuda:2, pinned. Bayes games ran 4 at a time in threads of one process; a pilot
  gave identical results with 1 and 4 threads at about half the wall time. ORACLE batches had B = 50. Peak memory
  1.37 GB; mean GPU utilisation 24% (max 95%); wall time 1,132 s (`provenance.arm_seconds`: 45–104 s per arm).
- **Parity (CPU↔GPU, 3 seeds, β_v × 0.5, canonical CPU draw)** (`provenance.parity`): passed. E_bayes was 0.6181 vs
  0.6228, E_oracle 0.6241 vs 0.6249, and victories were identical.
- **Deviations:**
  - z × 1.5 = 1.2375 is outside [0, 1] and would be rejected by the config (S ≤ 1, so nobody could vote). It is
    replaced by two arms on the margin 1 − z₀ = 0.175 (derived: 1 − 0.825): z = 0.9125 (margin × 0.5, the "×1.5 direction") and
    z = 0.7375 (margin × 1.5). The plan's z × 0.5 is kept.
  - Extra control arm `bayes_true`.
  - Electorates were drawn on the GPU (`generation="device"`).

**Results.** Win rates (`arms.<arm>.<player>.win`), the mean gap E_bayes − E_oracle (`arms.<arm>.gap_bayes_minus_oracle`),
Δgap (`tests.delta_gap`) and the misspecification cost (`tests.misspec_cost`). Each test shows the mean, its 95%
t-CI and the Holm p.

| arm | value | win bayes / bayes_true / oracle | gap bayes − oracle | Δgap vs default (Holm p) | misspec cost (Holm p) |
|---|---|---|---|---|---|
| default | — | 0.94 / 0.94 / 0.92 | +0.011 [0.008, 0.014] | — | 0 |
| ζ × 0.5 | 0.025 | 0.94 / 0.94 / 0.94 | +0.011 | −0.000 [−0.002, 0.002] (1) | −0.0005 [−0.0012, 0.0002] (0.001) |
| ζ × 1.5 | 0.075 | 0.92 / 0.92 / 0.88 | +0.011 | −0.000 [−0.002, 0.002] (1) | −0.0002 (0.04) |
| κ × 0.5 | 30 | 0.82 / 0.82 / 0.80 | +0.007 | −0.004 [−0.007, −0.000] (0.37) | −0.003 [−0.008, 0.001] (0.48) |
| κ × 1.5 | 90 | 0.94 / 0.96 / 0.94 | +0.011 | −0.000 (1) | −0.001 (1) |
| k̄ × 0.5 | 15 | 0.80 / 0.80 / 0.80 | −0.003 | **−0.014 [−0.019, −0.009]** (<1e-4) | 0.000 (identical) |
| k̄ × 1.5 | 45 | 0.94 / 0.94 / 0.92 | +0.013 | +0.002 (1) | 0.000 (identical) |
| ω × 0.5 | 0.075 | 0.82 / 0.78 / 0.78 | +0.017 | +0.006 [0.003, 0.009] (5e-4) | **+0.007 [0.003, 0.011]** (4e-4) |
| ω × 1.5 | 0.225 | 0.96 / 1.00 / 0.98 | +0.004 | −0.008 [−0.011, −0.005] (<1e-4) | **−0.009 [−0.013, −0.006]** (<1e-4) |
| β_v × 0.5 | 10 | 1.00 / 1.00 / 1.00 | −0.000 | −0.011 [−0.015, −0.008] (<1e-4) | −0.005 [−0.009, −0.002] (0.008) |
| β_v × 1.5 | 30 | 0.88 / 0.86 / 0.82 | +0.019 | +0.008 [0.004, 0.013] (0.009) | +0.007 [0.001, 0.013] (0.48) |
| z × 0.5 | 0.4125 | 1.00 / 1.00 / 1.00 | −0.000 | −0.011 [−0.014, −0.008] (<1e-4) | −0.0001 (<1e-4) |
| z margin × 0.5 | 0.9125 | 0.00 / 0.00 / 0.00 | −0.019 | **−0.030 [−0.037, −0.023]** (<1e-4) | **−0.019 [−0.028, −0.011]** (<1e-4) |
| z margin × 1.5 | 0.7375 | 1.00 / 1.00 / 1.00 | −0.023 | **−0.034 [−0.045, −0.023]** (<1e-4) | **−0.025 [−0.036, −0.014]** (<1e-4) |
| p_m × 0.5 | 0.3 | 0.90 / 0.90 / 0.88 | +0.006 | −0.005 [−0.006, −0.003] (<1e-4) | +0.000 (0.13) |
| p_m × 1.5 | 0.9 | 0.96 / 0.96 / 0.98 | +0.014 | +0.003 [0.000, 0.005] (0.005) | −0.003 [−0.005, −0.000] (0.08) |

- **Sensitivity ranking** (`ranking_delta_gap`, by max |Δgap|): z 0.034 > k̄ 0.014 > β_v 0.011 > ω 0.008 > p_m 0.005
  > κ 0.004 > ζ 0.0004 (`sensitivity.<param>.max_abs_delta_gap`).
- **Misspecification-cost ranking** (`ranking_misspec_cost`): z 0.025 > ω 0.009 > β_v 0.005 > κ 0.003 > p_m 0.003
  > ζ 0.0005 > k̄ 0.0.
- **Wins:** no win-rate difference survives Holm. Bayes vs ORACLE has at most 3 discordant seeds per arm, and bayes
  vs bayes_true at most 2 (`tests.wins_bayes_vs_oracle`, `tests.misspec_wins`; all Holm p = 1).
- **k̄ misspecification costs exactly nothing.** The two Bayes players play identically: BayesPlanner builds its
  posterior worlds from the census edge counts (`worlds._census_block_probabilities`), never from `mean_degree`. Its
  k̄ × 0.5 Δgap comes from the ORACLE gaining on Bayes in sparse graphs, not from a wrong belief.
- **In the default world BayesPlanner beats the ORACLE** (+0.011 [0.008, 0.014]), as in X6. The ORACLE is a
  first-order heuristic, so a Δgap partly measures how the heuristic ORACLE reacts to the parameter.

Figures:
- ![X21 sensitivity](..figures/x21_sensitivity.png)
- ![X21 win rates](..figures/x21_win_rates.png)

**Theory vs observation. Partial.**
- **"Most sensitive to z": confirmed** on both measures (Δgap −0.034, misspecification cost −0.025, Holm p < 1e-4).
  Both are effects on E. The wins do not change: at z = 0.9125 everyone loses, and at z = 0.7375 everyone wins.
- **"…and β_v": partly confirmed.** β_v is 3rd of 7 on both rankings (Δgap 0.011, misspecification cost 0.005).
  k̄ (Δgap) and ω (misspecification cost) are 2nd.
- **"Least sensitive is κ": refuted.** κ is 6th of 7 by Δgap (0.004, Holm p 0.37) and 4th by misspecification cost
  (0.003, Holm p 0.48). The least sensitive parameters are ζ (Δgap 0.0004; misspecification cost 0.0005) and, for the
  pure misspecification cost, k̄ (exactly 0, because the player reads it from the census).
- **Overall, BayesPlanner is robust.** Over a ×0.5–×1.5 change in any one parameter, the largest loss in E is 0.034,
  and no win rate changes significantly. Some misspecifications even help it: ω × 0.5 +0.007 and β_v × 1.5 +0.007
  over the correctly-specified player, because the rollout planner is itself approximate.

**What it means.**
- **For the game design:**
  - The hidden parameters that matter for a model-based player are the ones in the voting rule (z, then β_v) and
    the social step ω.
  - The electorate-generation parameters (ζ, κ, k̄) barely matter, because the player measures what it needs
    through the census and its survey.
  - Publishing z (or telling the player the voting rule) would remove the largest source of model error. Keeping
    it hidden costs at most about 0.03 of E.
- **For a player:** a good Bayesian strategy keeps working when the world differs from the rulebook. The decisive
  facts (who lives where and what the regions want) are learned from public data, not assumed.

**Data.** reports/v3/x21.json, reports/v3/raw/x21.jsonl (2,350 rows), log reports/v3/logs/x21.log. Command:
`OMP_NUM_THREADS=4 python experiments/x21_misspec.py --seeds 50 --first-seed 14000 --threads 4 --device cuda:2 --out reports/v3/x21.json`.

### X22 — Global sensitivity analysis of the design (Morris)

**Status:** done. The threshold z dominates every output; the worker fee (±30%) is at the noise floor.

**Status:** done (r = 10 trajectories, 110 design points × 50 paired seeds × 5 policies; 27,500 games, derived: 110 × 50 × 5). Sobol was not run.

**Question.** Which parameters is the game's balance most sensitive to? This backs the "synthetic calibration"
limitation in the report.

**Prediction (pre-registered).** None. `reports/v3/predictions.md`: "The plan gives no 'Predict' line for this
experiment; none is registered."

**Setup.**
- **Seeds:** 14000–14049 (`seed_block("X22", 50)`). The same 50 seeds are used at every design point (common random
  numbers).
- **Game:** N = 10⁴, T = 5, ₫500, query ₫10.
- **Factors:** 10, each in a ±30% box around the default (`factors.<f>.range`):
  - β_v 14–26;
  - z: 0.8775 down to 0.7725 (see Deviations);
  - κ 42–78;
  - ω 0.105–0.195;
  - ζ 0.035–0.065;
  - k̄ 21–39;
  - p_m 0.42–0.78;
  - η_m0 0.175–0.325;
  - λ 0.35–0.65;
  - worker fee ₫14–₫26.
- **Design** (`design`): Morris one-at-a-time trajectories, a numpy implementation in the script. SALib is not
  installed and was not added.
  - p = 4 levels, Δ = 2/3 of the box, r = 10 trajectories × 11 points = 110 points; rng seed 14000.
  - Unit level u maps to default × (0.7 + 0.6u).
- **Outputs per point** (ORACLE-only; `metric_definitions`). Every policy uses the ORACLE manifesto, and all five
  policies × 50 seeds run batched in `x_batch.run_batch`:
  - `oracle_win`: ORACLE planner win rate;
  - `split_win`: "pay_split" win rate;
  - `worker_best` (the "best-channel share"): the share of seeds whose best paid pattern by E is worker-only
    (pay_workers or front_workers);
  - also reported: `media_best`, and the mean E of the ORACLE and of split.
- **Statistics:**
  - EE = [y(x ± Δeᵢ) − y(x)]·(±1)/Δ: the change in the output over 2/3 of the box, per unit of the box.
  - μ* = the mean |EE| over the 10 trajectories, μ = the signed mean, σ = the sd.
  - The 95% CI of μ* comes from a bootstrap over trajectories (2,000 resamples).
  - The noise floor is the mean seed-sampling SE of one EE (`morris.<output>.noise_floor_ee_se`).
- **Compute** (`provenance.compute`): cuda:2, pinned, B = 250 games per call (`provenance.calibration.batch`; cap
  10 GiB). Peak memory 4.07 GB; mean GPU utilisation 32% (max 99%); wall time 743 s.
  - Electorates were re-drawn at 40 points and manifestos recomputed at 70; the other points reuse them
    (`provenance.electorate_draws`, `manifesto_draws`).
  - Seed counts were not reduced: the full design took 12 minutes.
- **Parity (CPU↔GPU, 3 seeds, default point, canonical CPU draw)** (`provenance.parity`): passed. E planner was
  0.6157 vs 0.6164, E split 0.6285 vs 0.6280, and victories were identical.
- **Deviations:**
  - **z box on the margin 1 − z** (0.175 × [0.7, 1.3]) because the literal box reaches z = 1.07 > 1, an invalid
    value. **For z, u ↑ means a lower threshold**, so a positive μ means "an easier vote helps".
  - "Best-channel share" is operationalised as the worker-best share.
  - Sobol was not run (optional in the plan).
  - Electorates were drawn on the GPU.

**Results.** Morris μ* [95% bootstrap CI] and signed μ (`morris.<output>.factors.<f>`), top factors per output. Each
output also lists the factors whose μ* CI lower bound is above 2 × the noise floor (`morris.<output>.above_noise`):

| output (noise floor) | 1st | 2nd | 3rd | 4th | above 2× noise |
|---|---|---|---|---|---|
| oracle_win (0.023) | z 0.88 [0.61, 1.16], μ +0.88 | ω 0.087 [0.039, 0.138], μ +0.087 | κ 0.051 [0.012, 0.096] | k̄ 0.042 [0.018, 0.066] | z |
| split_win (0.024) | z 0.94 [0.70, 1.19], μ +0.94 | ω 0.12 [0.05, 0.20], μ +0.12 | k̄ 0.099 [0.036, 0.165], μ +0.099 | κ 0.036 | z, ω |
| worker_best (0.045) | k̄ 0.27 [0.18, 0.39], μ −0.27 | z 0.22 [0.15, 0.29], μ −0.22 | β_v 0.17 [0.08, 0.28], μ −0.17 | p_m 0.13 [0.05, 0.21], μ −0.13 | k̄, z |
| oracle_E (0.0023) | z 0.42 [0.40, 0.44] | ω 0.059 [0.053, 0.066] | κ 0.046 [0.042, 0.050] | β_v 0.035 [0.020, 0.049] | 8 of 10 (all but fee, λ) |

- **Weakest factors.** On every win-rate output the worker fee (μ* ≤ 0.015) and the media decay λ (μ* ≤ 0.006) are
  at the bottom with ζ, inside the noise. On worker_best, ω (0.003) and κ (0.021) are the smallest.
- **Range over the 110 points** (`points`):
  - ORACLE win 0.00–1.00, mean 0.76;
  - split win 0.00–1.00, mean 0.74;
  - worker_best 0.00–0.80, mean 0.17;
  - media_best 0.00–0.38, mean 0.02.
- **Interactions.** σ is large next to μ* for z on the win rates (σ 0.48 vs μ* 0.88 for oracle_win). The effect of z
  depends on where the other factors put the race: a win rate saturates at 0 or 1. On worker_best, every top factor
  has σ ≈ μ*, which points to non-linear effects and interactions.

Figures:
- ![X22 Morris](..figures/x22_morris.png): μ* vs σ per output.
- ![X22 direction](..figures/x22_direction.png): signed μ.

**Theory vs observation.** No prediction was registered. The result agrees with X21:
- the vote threshold z is by far the most influential parameter;
- ω and the generation factors κ and k̄ come next;
- the prices tested here (worker fee ±30%) hardly matter for who wins.

**What it means.**
- **For the game design:**
  - The balance hinges on the voting rule. A step of 0.07 in z (2/3 of the box, i.e. 40% of the margin 1 − z) moves the
    ORACLE's win rate by about 0.6 on average (μ* 0.88 × Δ = 2/3).
  - z, ω and κ were "v2.1 calibrated" on synthetic targets. They are the parameters whose calibration most deserves
    the "synthetic calibration" caveat in the report.
  - Workers being the best channel is a matter of the network and the voting rule:
    - a denser network (k̄ ↑), an easier vote (z margin ↑), a steeper vote (β_v ↑) and wider media exposure (p_m ↑)
      all reduce the share of worlds where a worker-only pattern is best (μ < 0);
    - the worker fee moves it only a little (μ* 0.045, at the noise floor).
  - So re-pricing workers within ±30% will not fix the "split dominates" result of X0. The cascade rules (X23) are
    the stronger lever.
- **For a player:** nothing to act on directly. The parameters that decide balance (z, β_v, ω) are hidden, and X21
  shows that a Bayesian player loses at most about 0.03 of E by assuming the defaults.

**Data.** reports/v3/x22.json, reports/v3/raw/x22.jsonl (110 rows with per-seed outcomes), log reports/v3/logs/x22.log.
Command: `OMP_NUM_THREADS=4 python experiments/x22_sensitivity.py --seeds 50 --first-seed 14000 --trajectories 10 --device cuda:2 --out reports/v3/x22.json`.

## 9. Scale

### X9 — Scale and GPU performance

**Status:** done. 10⁶ voters at 11 ms per round on one A5000; GPU ×69 over CPU at 10⁶.

**Question.** Does the game, with the new cascade mechanic, meet the "GPU first, up to 10⁶ voters" requirement?
How do round time and memory scale with N and with batching over worlds, what is the CPU→GPU speed-up, and does
the cascade hop stay O(edges)?

**Prediction (pre-registered).** The plan gives no "Predict" line for this experiment; none is registered
(reports/v3/predictions.md).

**Setup.**
- **Seeds:** canonical seed 20261107; 3 timed repeats after one warm-up. The minimum is reported.
- **Isolation:** each (N, device, B) runs in a fresh subprocess, so peak memory is clean.
- **Configurations:**
  - N ∈ {10⁴, 10⁵, 10⁶}.
  - GPU (cuda:2, RTX A5000, 24 GB): B ∈ {1, 8, 16, 32} worlds per `x_batch.run_batch` call.
  - CPU (4 threads): B ∈ {1, 8} at N ≤ 10⁵, and B = 1 at 10⁶.
- **Game:** T = 5 with the pay_split policy, so workers, the cascade hop, media and the social step all run every
  round. The manifesto is the global top-3 of v(0), so no optimiser runs inside the timing.
- **Generation:** GPU uses `generation="device"`; CPU uses the canonical CPU draw.
- **Cascade hop:** `spreader_edge_counts` + `ic_from_counts` are timed alone (median of 20 synchronised repeats)
  for 0.1%, 1% and 10% spreaders.
- **Compute:** wall time 2.1 min (`provenance.compute`).
- **Parity (CPU↔GPU, seeds 20261107–09, N = 10⁴, pay_split):** passed. Expected share 0.554 vs 0.545, within the
  tolerance of 0.028 (max(0.01, 3·SE, 5%)) (`provenance.parity`).
- **Deviations:**
  - B = 16 was added to locate the largest batch that fits at 10⁶.
  - Batches join B copies of the canonical world. The timing does not depend on which worlds are joined.
  - **Incident:** an early N = 10⁴ pilot ran for about 20 s on cuda:0 (under 0.4 GB). `select_device` fell back
    there because cuda:2 was busy with X1 and X18. Every script now refuses to fall back from the requested GPU,
    and the reported run is entirely on cuda:2 (`provenance.compute.device_selection`).

**Results.** Source: reports/v3/x9.json (`table`, `speedup`, `cascade_hop_scaling`, `results`). ms per world-round
= one round of one world inside a batched call.

| N | GPU B = 1 | GPU B = 8 | GPU B = 16 | GPU B = 32 | CPU B = 1 | CPU B = 8 | peak GPU memory (B = 1 / largest B) |
|---|---|---|---|---|---|---|---|
| 10⁴ | 2.83 | 0.92 | 0.77 | **0.70** | 7.97 | 8.18 | 0.01 / 0.36 GB (B = 32) |
| 10⁵ | 2.95 | 1.55 | 1.45 | **1.42** | 82.1 | 68.8 | 0.12 / 3.5 GB (B = 32) |
| 10⁶ | 11.06 | 11.27 | 11.35 | OOM | 762.6 | — | 1.13 / 17.9 GB (B = 16) |

- **Speed-ups** (`speedup`):
  - One world on the GPU vs on the CPU: ×2.8 at 10⁴, ×28 at 10⁵ and ×69 at 10⁶.
  - Best batched GPU vs best CPU, per world-round: ×11, ×49 and ×69.
- **Largest batch that fits on the 24 GB GPU:** 32 at 10⁴ and 10⁵. At 10⁶ it is 16 (17.9 GB). B = 32 asks for a
  further 7.2 GB allocation and fails.
- **Interactive targets** (spec §9: under 1 s per round at 10⁴; 10⁵ required; 10⁶ single world):
  - Library `simulate_world`: 3.3 / 4.0 / 14.2 ms per round on the GPU, vs 9.1 / 128 / 1,184 ms on the CPU.
  - Device-side generation of a 10⁶-voter world takes 0.39 s, against 6.2 s for the canonical CPU draw.
- **Cascade hop vs edges** (`cascade_hop_scaling`; 2E = 0.3 M, 3 M and 30 M directed edges):
  - **GPU:** 0.39, 0.43 and 1.23 ms (1% spreaders). The log–log slope is 0.25: launch-latency bound up to 10⁵,
    and sub-linear (×2.8 for ×10 edges) up to 10⁶. The cost per edge falls from 1.3 to 0.04 ns.
  - **CPU:** 1.1, 6.9 and 65.5 ms. The slope is 0.88–0.94 (0.1–10% spreaders), with a flat cost per edge of about
    2.2 ns from 10⁵.
  - Neither device grows faster than the edge count.
- **Batching:** at 10⁴ the GPU is latency-bound, so batching 32 worlds cuts the cost per world-round by 4×.
  At 10⁶ a single world already fills the GPU (11.1 ms, flat in B).

Figure: ![X9 scaling](..figures/x9_scaling.png)

**Theory vs observation.** No prediction was registered. Descriptively:
- the cascade hop and the whole round are at most linear in the number of edges, as the sparse
  edge-list / CSR design intends;
- the GPU advantage grows with N (×3 → ×69);
- one 24 GB GPU holds 16 simultaneous 10⁶-voter worlds.

**What it means.**
- **For the game design:** the P8 cascade adds no scaling problem. The 10⁶-voter world plays at 11 ms per round
  on one GPU, far inside the 1 s interactive target. The CPU is fine for 10⁴ (8 ms) but not for 10⁶ (0.76 s per
  round, about 4 s per game).
- **For the experiments:** batch at small N (B ≥ 32 at 10⁴–10⁵ gives ×11–49 over the CPU). At 10⁶, budget about
  1.1 GB per world and at most 16 worlds per call. When two processes share one GPU, cap each process's batch: a
  calibrated batch that takes 70% of the free memory starves the other (the X1 incident).

**Data.** reports/v3/x9.json, reports/v3/raw/x9.jsonl (17 configuration rows), log reports/v3/logs/x9.log. Command:
`OMP_NUM_THREADS=4 python experiments/x9_scale.py --device cuda:2 --repeats 3 --out reports/v3/x9.json`.

## 10. Not run and future work

### X24 — Human playtest (+ optional LLM player)
**Status: Dropped (D8; the LLM player was not approved, D2).**

A small group of people (size as in the plan, §8) would each play 2 games on fixed seeds through the web UI, reading the epistemic debrief between games, compared with BayesPlanner and Top-3 on the same seeds, to test whether game 2 improves on game 1. Not run because the owner skipped the human playtest (D8). No results.

### X25 — Two-party meta-game and manifesto convergence
**Status: Deferred (D1).**

An empirical game-theoretic analysis of two scripted parties ({I, II, III, Top-3}², Nash equilibrium of the meta-game) and iterated best response on manifestos. It needs a large build (multi-party softmax voting with abstention, a cascade rule for voters converted by several parties, a two-player engine), which is not scheduled. No results.

## 11. Prediction scorecard

Every prediction in [`predictions.md`](predictions.md) has one row; the prediction cell is copied unchanged. Of the 22
experiments with a prediction, 4 are confirmed, 1 refuted and 17 partly confirmed (derived: count of the Verdict column).
X22 and X9 had no prediction in the plan; X24 and X25 were not run.

| ID | Prediction (from `predictions.md`) | Key measured number (95% CI) | Verdict | Data file |
|---|---|---|---|---|
| X0 | at ₫20, "split" wins ≥ 80% (gate 4a fails), as the P8 price check found at ₫10–₫50. | split wins 0.88 [0.80, 0.93] / 0.88 / 0.95 [0.89, 0.98] at ₫500/750/1000; 4a and 4b fail at all three | confirmed | [x0.json](x0.json) |
| X10 | Δshare per worker ∝ 1/N (log–log slope ≈ −1), while media's Δshare per ₫ is flat in N. Workers dominate at N = 10³ (the "one worker converts most voters" problem) and are near-worthless at 10⁶. T = 1 favours media, because a cascade gets only 1 hop. | slope Δshare/worker vs N: −0.990 [−0.999, −0.982] at T=1, −0.237 [−0.275, −0.205] at T=5, +0.016 [0.008, 0.023] at T=10; media slopes CI ∋ 0; workers dominate at N=10³ and at T=10 | partial | [x10.json](x10.json) |
| X7 | money has diminishing returns; workers stop being worth it relative to media somewhere between ₫50 and ₫100 at N = 10⁴ (P8 table). | ORACLE wins 0.88–0.98 in all 24 cells; all-workers beats all-media in 12/12 (treasury, fee) cells, +0.014 to +0.078 E | partial (worker-vs-media half refuted) | [x7.json](x7.json) |
| X23 | cap ∞ and rule γ bring back cascade saturation (the manifesto stops mattering, as P8 saw with γ alone). A cap of 1 makes workers subcritical. | rule γ: every paid pattern wins 50/50, manifesto effect 0.062 → 0.019; cap ∞: branching 2.1, manifesto still matters; cap 1: branching ≈ 1.08 | partial | [x23.json](x23.json) |
| X1 | the decay follows ā(1 − μ₂(D⁻¹A)), not λ₂(L)/k̄ (as E1 found). Under the P8 rules, high p_g raises the wins of a single nationwide manifesto. | ā(1 − μ₂): Spearman 0.997, bias +4.5%; λ₂(L)/k̄ off ≈ 1.9× (ρ 0.62); nationwide top-3 manifesto wins 0.71 → 0.92 at high p_g (31/0 discordant) | confirmed | [x1.json](x1.json) |
| X2 | The worker anchor value is a sizeable share of the worker value: Arm B Δshare / full-cascade Δshare is between 0.2 and 0.6, to be measured. Pinned agents stop consensus (disagreement does not decay to 0). W-MSR trims workers, because they sit far from their friends, so it **lowers** the worker value. That makes W-MSR the network's defence against *your* campaign. | anchor share 0.002–0.009 (predicted 0.2–0.6); W-MSR raises worker value by +0.001 to +0.004; FJ disagreement 0.018 with zealots; workers alone: FJ share 0.97, simulated 0.83 at T = 200 | partial (2 refuted, 1 confirmed) | [x2.json](x2.json) |
| X3 | growth crosses 1 near λ_max(A D_q) ≈ 1, more sharply than the naive estimates; the cap binds mainly above threshold. | growth = 1 at λ_max(A D_q) ≈ 0.94 (R_obs 0.97 at λ 0.97); naive global estimates cross at 0.48–0.53, regional at 0.03; cap binds for 0.6% of targets below vs 41.5% above threshold | confirmed (sharpness only marginal) | [x3.json](x3.json) |
| X4 | buying early is worth more (more hops); centrality picking gives a modest gain under k̄ = 30 ER blocks (low degree variance). | front − late = +0.12 E, late workers nearly worthless; placement effect ≤ 0.010; top-degree pick ≤ +0.006, no change in wins | confirmed | [x4.json](x4.json) |
| X11 | the interaction is positive, and larger in canvassing-receptive regions; "media first, then workers" beats the reverse. | interaction +0.0011 [0.0007, 0.0015] at ₫20 (+0.0016 at ₫100), < 1% of the joint effect; largest in P1, not P3; workers-then-media beats media-then-workers by 0.041 (100/100 seeds) | partial (positive confirmed; P3 and ordering refuted) | [x11.json](x11.json) |
| X18 | roughly submodular in k, with possible early *super*-modularity in regions near threshold. | k = 1 gives 62% and k = 5 gives 93% of the plateau; 1/15 adjacent pairs violate (gain ≈ 1e-5); random pick reaches 93–96% of CELF from k = 2 (75% at k = 1) | partial (submodular confirmed; early supermodularity refuted) | [x18.json](x18.json) |
| X6 | II > I > III; the price of not knowing is ≤ 0.015 E. | E / win: II+ADMM 0.649 / 0.99, II Bayes 0.639 / 0.98, ORACLE 0.628 / 0.95, top-3 0.602 / 0.85, I 0.560 / 0.73, III 0.088 / 0.00; II − I = +0.079, I − III = +0.47; price of not knowing −0.011 vs ORACLE, +0.033 [0.030, 0.035] vs X13 bound | partial (ordering confirmed; ≤ 0.015 holds only against the heuristic ORACLE) | [x6.json](x6.json) |
| X12 | the best λ > 0 on seeds where some coalition region is canvassing-receptive. Joint optimisation gains a measurable +0.005–0.02 E over "b first, then plan" (to be measured). | λ = 0.5 gains +0.006 E (Holm p 8e-5 ORACLE, 0.031 search), λ = 1 gains nothing; branching ≈ 5 in every region, ρ = 0.17 with the gain; joint +0.0077 [0.0050, 0.0104] (ORACLE), +0.0111 [0.0082, 0.0141] (search), best E 0.683 | partial (joint gain confirmed; receptivity condition not supported) | [x12.json](x12.json) |
| X13 | the search beats the ORACLE planner by ≤ 0.01 E on average, with a long tail of seeds where front-loading or concentration beats greedy by more. | search − ORACLE = +0.044 E [0.043, 0.046], 100/100 seeds (min +0.027); better plans are media-first, front-loaded (2.8 workers in round 1 vs 16.2 over rounds 1–3; media 89% vs 34% of spend); no feature explains the gap (Holm p ≥ 0.13) | refuted | [x13.json](x13.json) |
| X14 | positive bias (over-optimism) that shrinks with world size; "even" undervalues front-loaded worker buys. | round-1 bias −0.024 (N = 10⁴), −0.047 (10⁵), slope ≈ 1 (r ≈ 0.96); shrinks with world size (ρ = +0.42); rollout none −0.050, front −0.010; full-size worlds +0.034 E at 10⁵ (Holm p 3e-11); front rollout no gain in realised E | partial (sign refuted; shrinkage confirmed; "even undervalues front" not supported) | [x14.json](x14.json) |
| X19 | b* is closer to a 2–3-region coalition blend than to the mean or median; there are several local optima (multi-start matters); mirror descent converges in fewer iterations than PG on interior optima. | PG and MD median 5 iterations to 1e-4 (p = 0.37), FW 10, ADMM 0.006 E short; 1.27 local optima per seed, 92% of starts reach b*; nearest 2–3-region blend −0.018 TV vs mean/median (Holm p 6e-6), not vs TV geometric median; b* carries 1.8 regions ≈ 50% of voters | partial (blend confirmed; several local optima and MD > PG refuted) | [x19.json](x19.json) |
| X5 | about 2 questions are still enough; beyond that every question is a net loss, larger than at ₫5. | E by k (₫10, info gain): k0 0.6542, k2 0.6579, k12 0.6447; k12 − k2 = −0.0132 [−0.0176, −0.0087] (Holm p 6e-7); ₫10 vs ₫5 contrast −0.0071 [−0.0122, −0.0020] (Holm p 0.084); best k: 2 at ₫10, 4 at ₫5; win rates at ceiling | partial (≈ 2 questions and net loss beyond confirmed; ‘larger than at ₫5’ not significant) | [x5.json](x5.json) |
| X17 | probing lowers τ entropy measurably, but the 400-voter national poll (SE ≈ 2.4 pp) limits it. The net win-rate effect is ≈ 0 or negative. | hard-filter τ entropy at round 5: probe +0.019 higher (Holm p 2e-5); soft-likelihood entropy −0.030 at round 2 (Holm p 3e-9), gone by round 5; probe − planner E −0.0127 [−0.0195, −0.0059] (Holm p 0.007), 3 games lost, 0 gained | partial (entropy drop refuted on plan metric; poll limit and ≤ 0 net effect confirmed) | [x17.json](x17.json) |
| X16 | "guaranteed" becomes rarer and more reliable as the world count grows; p̂ is over-confident at 16 worlds. | P(guaranteed) 0.772 → 0.616 for W = 8 → 64 (W64 − W8 −0.156 [−0.200, −0.110], Holm p 8e-23); 0 false asserts in all 36 cells; missed truths 0.21 → 0.37; default cell Brier 0.032, OC −0.024 [−0.050, 0.007], slope 1.44 | partial (rarer confirmed; reliability untestable at 0 errors; over-confidence refuted) | [x16.json](x16.json) |
| X8 | most winners come from behind (as v2 found); the point of no return is round 2–3 for weak strategies. | winners trailed at s₀: 548/548 true, 453/453 poll [0.993, 1]; weak-loss PNR round 1 in 77% [0.73, 0.81], round 2–3 in 6.5% [0.046, 0.092]; P(win \| a plan wins all sampled futures at round 1) 0.89 vs 0.023 | partial (from behind confirmed, structural; PNR 2–3 refuted) | [x8.json](x8.json) |
| X15 | losses concentrate where the two largest regions have distant archetypes and a fragmented network (low p_g); the largest region's personality decides the best channel. | ORACLE losses 11/100 at archetype TV ≥ 0.35 vs 0/100 below (Fisher p 0.0004); CV AUC 0.91 logistic / 0.76 tree; Holm p: H1 0.011, H2 (p_g) 0.032 but coefficient CI ∋ 0, H3 Cramér's V 0.31 yet split best in 92/100 | partial (distant archetypes confirmed; fragmentation weak; personality → channel refuted) | [x15.json](x15.json) |
| X20 | world draw > worker picks > cascade > media ≈ election at N = 10⁴. Election-day noise is tiny (sd ≈ 0.005). | world draw = 99.4% of Var(V) for planner, 97.2% for split (mostly preferences, 0.92); within-world sd of V: cascade 0.0035 ≈ election 0.0034 > worker picks 0.0028 ≫ media 0.0011; skill share on E 0.50 [0.48, 0.52] (planner vs split only 0.002) | partial (world first and small election noise confirmed; picks > cascade and media ≈ election refuted) | [x20.json](x20.json) |
| X21 | the planner is most sensitive to z and β_v (they define who votes); the least sensitive is κ. | Δgap ranking z 0.034 > k̄ 0.014 > β_v 0.011 > ω 0.008 > p_m 0.005 > κ 0.004 > ζ 0.0004; misspecification cost z 0.025 > ω 0.009 > β_v 0.005; no win-rate change survives Holm; leak check passed | partial (z confirmed; least-sensitive κ refuted: ζ is least) | [x21.json](x21.json) |
| X22 | (no prediction in the plan) | ORACLE-win μ*: z 0.88 [0.61, 1.16] ≫ ω 0.087 > κ 0.051 > k̄ 0.042 (noise floor 0.023); worker-best share led by k̄ 0.27 > z 0.22 > β_v 0.17; worker fee at the noise floor | not tested (no prediction) | [x22.json](x22.json) |
| X9 | (no prediction in the plan) | GPU vs CPU ×2.8 / ×28 / ×69 (one world) and ×11 / ×49 / ×69 (batched) at 10⁴ / 10⁵ / 10⁶; 11 ms per round at 10⁶; B = 32 at 10⁶ is OOM; cascade hop ≤ linear in edges | not tested (no prediction) | [x9.json](x9.json) |
| X24 | (dropped under D8; no prediction registered) | not run | not tested (not run) | — |
| X25 | (deferred under D1; no prediction registered) | not run | not tested (not run) | — |

## 12. What changed since v2

v2 numbers are from [`reports/v2/final_report.md`](../v2/final_report.md) and its JSONs (`reports/v2/`); v3 numbers from
the v3 JSONs. The main rule changes between the two (§3.1): in v2 each round brought 10 free workers (extras ₫4) whose
contacts started an SIR word-of-mouth cascade with a fixed chance τ_c per edge, and a query cost ₫5. In v3 every worker
is bought for ₫20, is permanent and pinned at b (R21–R23, R30), the cascade is an Independent Cascade of one hop per
round with chance γ·P and at most 3 attempts per voter (R24–R27), and a query costs ₫10 (R30). v3 also uses fewer seeds
per cell (50 or 100, D3) than v2 (200 in most cells).

| v2 → v3 | v2 finding | v3 finding | Rule change that explains the difference |
|---|---|---|---|
| E1 → X1 | Contraction follows ā(1 − μ₂): Spearman 0.995; the λ₂(L)/k̄ form is worse (0.67) (v2/e1_topology.json) | Same: Spearman 0.997; λ₂(L)/k̄ ρ 0.62 (x1.json) | None for the result: the DeGroot dynamics did not change. The small differences are seed noise (50 vs 200 seeds per cell). |
| E2 → X2 | Zealots cut the ORACLE win rate from 0.955 to 0.720 as ζ goes 0 → 0.30; W-MSR does not slow persuasion (v2/e2_zealots.json) | Zealots cut the planner's win rate from 0.92 to 0.80 and its persuasion from 0.162 to 0.107 E; W-MSR slightly *raises* the worker value (+0.001 to +0.004); pinned workers add almost nothing on their own (anchor share 0.002–0.009) (x2.json) | The campaign channel changed (R21–R27): workers are pinned stubborn agents and converts jump to b. The direction of the zealot effect is the same; its size is not comparable across the two campaign rules. |
| E3 → X3 | SIR growth crosses 1 at x_rr ≈ 0.87, with x_rr = τ_c λ_max(A_rr)/ρ (v2/e3_cascades.json) | IC growth crosses 1 at λ_max(A D_q) ≈ 0.94, where D_q holds the per-voter conversion chances γ·P (x3.json) | R24–R25: the per-edge chance is γ_j·P_j instead of a constant τ_c, so the threshold operator is weighted by how much each voter likes b. Both thresholds sit just below 1, as theory predicts. |
| E4 → X5 | At ₫5: two queries +0.0027 E against none (p 0.05); twelve −0.0034 E (p 0.003) (v2/info_curve_r12.json) | At ₫10: two queries +0.0036 [−0.0008, +0.0081] E against none (Holm p 1); twelve −0.0095 [−0.0135, −0.0055] E (Holm p 6e-5) (x5.json) | R30 doubled the query price, and money now buys permanent workers, so each question costs more campaign. Two questions remain enough; extra questions now cost about 3 times as much. |
| E5 → X7 | ORACLE wins 0.835 / 0.960 / 1.000 at treasury ₫250 / ₫500 / ₫1000; the BayesPlanner shifts from media to workers as money grows (v2/e5_budget.json) | At the ₫20 fee, ORACLE wins 0.92 / 0.96 / 0.98 at ₫500 / ₫750 / ₫1000; the BayesPlanner's worker share of spend *falls* from 0.46 to 0.22 as money grows (x7.json) | R21–R27: workers are permanent and each voter can be targeted only 3 times, so extra workers soon hit used-up targets (X18: strongly submodular). Extra money goes to media instead. ₫250 is outside the R17 range and was not tested. |
| E6 → X6 | BayesPlanner wins 0.910 [0.862, 0.942], E 0.590; ORACLE 0.960, E 0.601; ORACLE − BayesPlanner = +0.011 E (v2/strategies_r12.json) | BayesPlanner wins 0.98 [0.93, 0.99], E 0.639; ORACLE 0.95 [0.89, 0.98], E 0.628; ORACLE − BayesPlanner = −0.011 [−0.014, −0.009] E (x6.json) | Under the new worker rules the ORACLE's greedy planner over-buys workers late (X13), while the BayesPlanner's rollouts value the rules directly. The ORACLE is no longer a ceiling; X13's search is the reference. |
| E7 → X9 | At 10⁶ voters a GPU round takes 28 ms vs 2.06 s on the CPU (about 74×) (v2/e7_scale.json) | At 10⁶ voters a round takes 11 ms on the GPU, ×69 over the CPU for one world (x9.json) | The cascade code changed (IC instead of SIR, R24–R27), and so did the per-round work; same GPU model (RTX A5000). The speed-up over the CPU is similar. |
| balance gate → X0 | All five criteria pass; the best fixed pattern wins 77.5% (< 80%); ORACLE wins 0.900 (v2/balance_gate_v2.1.json) | Criteria 4a and 4b fail at every treasury: "split" wins 0.88–0.95; ORACLE wins 0.94–0.99 (x0.json) | R21–R30: a bought, permanent ₫20 worker with an IC cascade is a much stronger channel than v2's workers. The P8 price check had found ₫100 passing and ₫50 failing (R29); the owner fixed ₫20 (R30) and kept it after X0 (D5). |

## 13. Limitations and threats to validity

- **The ORACLE is a heuristic.** X13's plan search beats the ORACLE planner on every seed (x13.json). Every "gap to
  the ORACLE" in X0, X6, X7 and X23 therefore understates how much a skilled campaign can gain, by about 0.04 E. The
  X13 bound is itself a heuristic search (cross-entropy method plus hill climb) over open-loop plans with the ORACLE
  manifesto, so the true optimum may be higher still (X12 finds +0.008 to +0.011 E more by also moving b).
- **Sampled worlds and plans.** The BayesPlanner plans on 8 posterior-sampled worlds of 1,600 voters; X14 shows its
  round-1 estimate is pessimistic and the bias depends on world size. The epistemic verdicts (X16) are over sampled
  worlds and a plan set of at most 21 plans; the temporal properties (X8) are over 64 sampled futures, not all plans
  and all randomness. A "guaranteed" or "inevitable" verdict is a statement about those samples.
- **Synthetic calibration.** z, ω and κ were calibrated in v2.1 on synthetic targets, not on real electorates. X21
  and X22 show the outcome is most sensitive to exactly z. No human played the v3 rules (X24 dropped, D8), so the
  player advice in §2 comes from scripted and planning players only.
- **Statistical power.** Sweep cells have 50 seeds (D3): a paired win-rate difference below about 8–10 points is not
  detectable. Many win rates sit near the ceiling (X5, X12, X21), so a non-significant win-rate difference is weak
  evidence of no effect; the expected-share E is the more sensitive outcome and is reported throughout.
- **Multiple testing.** Holm correction is applied within each experiment's family, not across the 23 experiments.
- **Default cell only.** Most experiments use N = 10⁴, T = 5 and a ₫500 treasury. X10 shows the balance depends on
  N and T, so results outside the default cell (especially at T = 1 or 10) should not be assumed to carry over.
- **Post-hoc analyses.** X8 and X15 reuse the games of X0, X6 and X10. X15 models 11 ORACLE loss events; the BayesPlanner
  had only 2 losses, too few to model, so its margin was modelled instead.
- **Scripted player archetypes.** People Pleaser (I) and Strong-Arm (III) are this project's implementations of those
  styles; the II > I > III result is about these implementations.
- **Different electorate draws.** X20, X21 and X22 drew their electorates on the GPU (`generation="device"`), so their
  worlds are not the canonical CPU draws used elsewhere. Within each experiment the arms are paired on the same worlds.
- **Shared hardware.** Two GPUs (cuda:0 and cuda:2) were shared with other users and agents. Wall times, utilisation
  and the X9 timings depend on that load. X1's first run was killed by an out-of-memory error while sharing cuda:2
  with X18, and was rerun alone.
- **CPU/GPU randomness.** The CPU and GPU use different random streams, and float atomics in the CUDA DeGroot step
  make GPU runs reproducible only up to the last float bits. Every experiment passed its CPU↔GPU parity check
  (discrete outcomes equal, means within tolerance); X15 found the shared X0/X10 worlds agree within 3.0e-07 in E
  (x15.json).
- **The number check is permissive.** `x_report_check.py` accepts a number when it appears, after rounding, anywhere
  in a cited JSON. A match shows the number exists in the source, not that it is the right key. 13 numbers are
  documented derivations (differences, ratios, products, a square root, means over raw rows); the checker lists them
  for a reviewer (`reports/v3/report_check.txt`).

## 14. Deviations from the plan

**Scope.** X24 was dropped (D8; the LLM player was not approved, D2). X25 is deferred (D1). Every other experiment ran
with the planned seed counts; no seed count was reduced.

**Approved library changes** (D2, D12; each keeps the default behaviour bit for bit and has a test):
- `WorkerConfig.conversion_rule` ∈ {gamma, gamma_p, gamma_p2}, default gamma_p, for X23 (`config.py`,
  `dynamics/campaign.py`, `strategies/worlds.py`; `tests/unit/test_x_common.py`). It was made in Phase 0, although
  plan §2 says no library change in Phase 0.
- Planner `rollout="front"` continuation, for X14 (`strategies/planner.py`; `tests/unit/test_planner_front_rollout.py`).
- `policy_set(max_policies=…)`, for X16 (`logic/epistemic.py`; a local counter in `logic/report.py` renamed;
  `tests/unit/test_policy_set_max_policies.py`).
- D11 / R31: the API and the web UI accept T = 3–5 rounds only (`api/app.py`, `web/lib/constants.ts`,
  `web/lib/mockApi.ts`, tests); the engine and the experiments still accept T = 1–10.
- D12: `BayesPlanner.manifesto()` builds its extra start tensors on the voters' device (`strategies/players.py`;
  `tests/integration/test_players.py::test_bayes_planner_runs_on_gpu`). Before D12, X7 used an experiment-side
  subclass that kept the manifesto worlds on the CPU.

**Not approved / not scheduled:** the LLM player (X24) and the softmax-voting two-player engine (X25).

**Phase 0.**
- The v2 reports were not moved to `reports/archive/v2.1/`: `reports/v2/final_report.md` (kept as the v2 record, D4)
  links to `reports/v2/*.json` and `..figures/`. The move is offered to the owner in `handoff_owner.md`.

**Device.** The plan's default device is cuda:2. X6, X7, X8, X12, X13, X14, X16, X19 and X23 ran on cuda:0 because
cuda:2 was in use by other runs. X15 ran on the CPU (a §1.1 exception for model fitting).

**Per experiment** (from each section and each JSON's `deviations` key; device-only deviations are listed above):
- **X0, X10:** none (`x0.json`, `x10.json`: `deviations` empty).
- **X7** (`x7.json` → `deviations`): the BayesPlanner CPU-manifesto workaround (resolved later by D12).
- **X23** (`x23.json` → `deviations`): the parity metric for reach is converts / N, not converts per worker; the ORACLE planner values workers
  with γ·P and no cap in every arm (it is rule-unaware).
- **X1** (`x1.json` → `deviations`): spectra by SciPy on the CPU (faster than the GPU at N = 10⁴, a §1.1 exception); the X0 policy set stands
  in for "standard strategies"; heterogeneous long runs are 2,000 rounds; at k̄ = 10 the spectra use the giant
  component; the first full run was killed by an out-of-memory error and rerun from scratch.
- **X2** (`x2.json` → `deviations`): arm A is played with two policies (pay_split and the ORACLE planner); one fixed manifesto per seed in every
  cell; the T = 200 long runs are experiment-side; the FJ reference uses SciPy `cg` instead of `spsolve`.
- **X3** (`x3.json` → `deviations`): 8 extension points per world added so that the threshold λ_max(A D_q) ≈ 1 is sampled (the plan's sweep
  never went below about 3); the target is the largest region; R_early is reported next to R_obs.
- **X4** (`x4.json` → `deviations`): two spend levels (K = 4 and 10 workers); the pick-rule arms replace R22 through an experiment-side hook.
- **X11** (`x11.json` → `deviations`): matched spend is approximate (k = max(1, round(m_r / fee)) workers per airing); all four regions are
  targeted in turn.
- **X18** (`x18.json` → `deviations`): treasury ₫1000 so that 40 workers are affordable; all 4 regions per seed; CELF and the top-degree pick are
  outside the game rules.
- **X6** (`x6.json` → `deviations`): none beyond the device.
- **X12** (`x12.json` → `deviations`): the search arm ran for the best coalition per seed only (5 λ points, not 7 coalitions × 5 λ); the first
  launch was stopped after 21 minutes (GPU memory) and rerun with the ORACLE arm chunked (identical numbers, checked).
- **X13** (`x13.json` → `deviations`): none beyond the device.
- **X14** (`x14.json` → `deviations`): a one-factor-at-a-time design around the default (9 arms per N) instead of the full 4 × 4 × 3 factorial.
- **X19** (`x19.json` → `deviations`): a first run with a uniform Frank–Wolfe line-search grid stalled near the optimum; the grid was made
  log-spaced and the whole experiment rerun; only the rerun is reported.
- **X5** (`x5.json` → `deviations`): queries are budgeted as k × price and there are no in-round queries (as in v2); the ₫5 comparison uses both
  selection rules; games run one at a time, with worlds × plans × voters batched inside each decision.
- **X17** (`x17.json` → `deviations`): an extra control arm poll_planner; τ learning measured with 512 candidate worlds per game replayed once;
  the probe score is a discriminability ratio; world micro-states drawn on the CPU and moved to the GPU.
- **X16** (`x16.json` → `deviations`): world generation on the CPU in 8 spawn processes (a §1.1 exception; about 85% of a timeline's cost);
  decision points are not batched across rounds (that would need an unapproved library change); |Π| = 8 and 16 are
  read as prefixes of the full plan set; the long-run truth reruns the X13 search from every decision point; the
  primary outcome is the official victory, with two further targets reported.
- **X8** (`x8.json` → `deviations`): EF / AF over a sampled tree of 64 paths per (game, round), not all plans and all randomness.
- **X15** (`x15.json` → `deviations`): the X10 default cell repeats the X0 ₫500 worlds and is counted once; the BayesPlanner loss (2 events) is
  not modelled, its margin is; supplementary targets added; numpy re-implementations of the models (scikit-learn is
  not installed); CPU only.
- **X20** (`x20.json` → `deviations`): electorates drawn on the GPU; the world component split into graph / preferences / attributes, and V
  reported alongside E and win.
- **X21** (`x21.json` → `deviations`): z × 1.5 = 1.2375 is invalid, so it is replaced by z = 0.9125 and 0.7375 (the margin 1 − z × 0.5 and × 1.5);
  an extra control arm bayes_true; electorates drawn on the GPU; BayesPlanner games run 4 at a time in threads.
- **X22** (`x22.json` → `deviations`): the z box is on the margin 1 − z (the literal ±30% box reaches z > 1); "best-channel share" is the
  worker-best share; Sobol indices not computed; electorates drawn on the GPU.
- **X9** (`x9.json` → `deviations`): B = 16 added; batches join copies of the canonical world; CPU batches only up to B = 8 and N ≤ 10⁵; an
  early pilot fell back to cuda:0 for about 20 s, after which every script refuses to fall back.

**Number check (Phase 8).** `x_report_check.py` reports 0 unmatched numbers. It now also reads the raw JSONL files
and the v2 JSONs when a line cites them, skips section references (§1.4), and lists numbers on lines marked "derived"
separately (docstring of `experiments/x_report_check.py`). Text corrected in Phase 8 to match the JSON
(old text in code spans):
- X1, uniform manifesto + planner gain: `+0.089` → +0.088 (x1.json `nationwide[11].expected_high_minus_low.mean` = 0.0885);
- X6, ORACLE media spend: `₫169` → ₫168 (x6.json `players.ORACLE.spend.media.mean` = 168.50);
- X12, largest branching factor over all regions at λ = 1: `5.50` → 5.49 (x12.json
  `spectra_and_reach_by_lambda.1.0.lambda_max_all_max.mean` = 5.4947);
- X5, best Holm p of the ₫10 − ₫5 contrast (section, scorecard): `0.085` → 0.084 (x5.json
  `tests.families.price_contrast.tests[2].E.holm_p` = 0.0845);
- X12, the unsourced `about 660 worlds` in the first-launch note was removed (no JSON or log records it). The section drafts in `reports/v3/sections/` still carry the old values; this report is the authoritative
version.

## 15. How to reproduce

Run from the repository root with the `mas` environment. Each command is also written at the top of its script.
Order matters for five of them: X13 before X6 and X12 (they read `x13.json`), X6 before X8 (it reads
`raw/x6.jsonl`), and X0, X6 and X10 before X15 (it reads their raw files).

| ID | Command |
|---|---|
| Phase 0 benchmarks | `OMP_NUM_THREADS=4 python experiments/x_gpu_benchmarks.py --device cuda:2 --out reports/v3/gpu_benchmarks.json` |
| X0 | `python experiments/x0_balance.py --seeds 100 --first-seed 10000 --device cuda:2 --treasuries 500 750 1000 --out reports/v3/x0.json` |
| X10 | `python experiments/x10_n_t_balance.py --seeds 50 --device cuda:2 --treasury 500 --out reports/v3/x10.json` |
| X7 | `OMP_NUM_THREADS=4 python experiments/x7_prices.py --seeds 50 --first-seed 10000 --device cuda:0 --out reports/v3/x7.json` |
| X23 | `OMP_NUM_THREADS=4 python experiments/x23_worker_rules.py --seeds 50 --first-seed 10000 --device cuda:0 --out reports/v3/x23.json` |
| X1 | `OMP_NUM_THREADS=4 python experiments/x1_topology.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x1.json` |
| X2 | `OMP_NUM_THREADS=4 python experiments/x2_stubborn.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x2.json` |
| X3 | `OMP_NUM_THREADS=4 python experiments/x3_cascade.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x3.json` |
| X4 | `OMP_NUM_THREADS=4 python experiments/x4_timing_placement.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x4.json` |
| X11 | `OMP_NUM_THREADS=4 python experiments/x11_worker_media.py --seeds 100 --first-seed 11000 --device cuda:2 --out reports/v3/x11.json` |
| X18 | `OMP_NUM_THREADS=4 python experiments/x18_submodularity.py --seeds 50 --first-seed 11000 --device cuda:2 --out reports/v3/x18.json` |
| X13 | `OMP_NUM_THREADS=4 python experiments/x13_plan_search.py --seeds 100 --first-seed 12000 --device cuda:0 --out reports/v3/x13.json` |
| X6 | `OMP_NUM_THREADS=4 python experiments/x6_tournament.py --seeds 100 --first-seed 12000 --device cuda:0 --out reports/v3/x6.json` |
| X12 | `OMP_NUM_THREADS=4 python experiments/x12_viral_manifesto.py --seeds 50 --joint-seeds 100 --first-seed 12000 --device cuda:0 --out reports/v3/x12.json` |
| X14 | `OMP_NUM_THREADS=4 python experiments/x14_planner_calibration.py --seeds 50 --first-seed 12000 --device cuda:0 --out reports/v3/x14.json` |
| X19 | `OMP_NUM_THREADS=4 python experiments/x19_manifesto.py --seeds 100 --first-seed 12000 --device cuda:0 --out reports/v3/x19.json` |
| X5 | `OMP_NUM_THREADS=4 python experiments/x5_info_value.py --seeds 50 --first-seed 13000 --device cuda:2 --out reports/v3/x5.json` |
| X17 | `OMP_NUM_THREADS=4 python experiments/x17_probe.py --seeds 50 --first-seed 13000 --device cuda:2 --out reports/v3/x17.json` |
| X16 | `OMP_NUM_THREADS=4 python experiments/x16_verdicts.py --seeds 50 --first-seed 13000 --device cuda:0 --out reports/v3/x16.json` |
| X8 | `OMP_NUM_THREADS=4 python experiments/x8_temporal.py --device cuda:0 --out reports/v3/x8.json` |
| X15 | `OMP_NUM_THREADS=4 python experiments/x15_hard_worlds.py --out reports/v3/x15.json` |
| X20 | `OMP_NUM_THREADS=4 python experiments/x20_luck_skill.py --seeds 50 --first-seed 14000 --salts 20 --device cuda:2 --out reports/v3/x20.json` |
| X21 | `OMP_NUM_THREADS=4 python experiments/x21_misspec.py --seeds 50 --first-seed 14000 --threads 4 --device cuda:2 --out reports/v3/x21.json` |
| X22 | `OMP_NUM_THREADS=4 python experiments/x22_sensitivity.py --seeds 50 --first-seed 14000 --trajectories 10 --device cuda:2 --out reports/v3/x22.json` |
| X9 | `OMP_NUM_THREADS=4 python experiments/x9_scale.py --device cuda:2 --repeats 3 --out reports/v3/x9.json` |
| XC | `OMP_NUM_THREADS=4 python experiments/x_consensus.py --seeds 50 --first-seed 10000 --device cuda:2` |
| figures (all 57) | `python experiments/plot_v3.py` |
| number check | `python experiments/x_report_check.py` (writes `reports/v3/report_check.txt`) |
| tests | `OMP_NUM_THREADS=4 python -m pytest tests -q` |

The device flags record where each run was made; any free GPU gives the same discrete results (parity checks), with
float differences in the last bits.
