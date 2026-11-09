# P4 — Strategies and optimisation (Days 6–7)

**Spec:** §10.2 (policy set Π), §11. **Owner area:** strategy (Mad). **Depends on:** P3.

## Deliverables
- `strategies/baselines.py`: uniform manifesto; "top-3 heuristic" manifesto (normalised free stats);
  random; fixed allocation patterns (workers-even, workers-largest-region, media-all,
  media-weakest-poll, mixed).
- `strategies/bayes.py`: posterior over θ = (σ, τ) (8640) from the knowledge log using precomputed
  per-archetype query-answer distributions (Monte Carlo cache); `expected_v_by_region(posterior)`;
  the query-selection policy that maximises expected information gain over σ.
- `strategies/manifesto_opt.py`:
  - projected gradient ascent on `E[votes](b)` over Δ, using a Huber-smoothed TV distance and
    `project_simplex` (sort-based, batched);
  - **consensus ADMM**: local `b_r` per region (local objective = region votes), `z = b`, scaled dual
    `u_r`; must match the centralised PG result (lecture: ADMM / prox);
  - inputs: either true v (oracle) or posterior samples (informed).
- `strategies/planner.py`: rollout planner that scores candidate allocations on B sampled worlds in
  one batched GPU call, either greedy per round or a short-horizon receding plan; worker placement
  uses region cascade potential (τ_c·λ_max) and posterior susceptibility.
- `strategies/oracle.py`: full-information upper bound (labelled ORACLE and never mixed with
  player results).
- `batch.py`: B-world simulator used by the planner, the logic layer and experiments.

## Tests
- `project_simplex` against known cases and the KKT conditions; batched equals looped.
- PG converges (KKT residual < 1e-4) on a toy concave instance; ADMM ≈ PG (TV distance < 1e-3).
- Posterior: the true θ keeps positive mass after its own exact answers; with enough queries the
  posterior concentrates; inconsistent θ get zero mass.
- Planner never proposes an invalid or unaffordable action; deterministic per seed.
- Paired comparison (≥ 200 seeds): planner + informed manifesto > baselines (p < 0.01).

## Exit criteria
The strategy comparison report `reports/v2/strategies.json` and plots exist; the §12 criterion 2
is re-checked with the real optimiser.

## Handoff (2026-11-08)
- Owner scope: experiments only (no in-game advisor); AI decisions must take < 1 s.
- **Built:**
  - `strategies/bayes.py` (agent E): σ posterior over 360 assignments from free stats + queries, information
    gain, `best_query`.
  - `strategies/{worlds,planner,players}.py` and `experiments/{strategies,plot_strategies}.py` (agent F):
    observation-consistent synthetic worlds with census-matched graphs, `WorldBatch` batched simulation,
    rollout planner, and observation-only AI players (Uniform, Top-3, Random, BayesPlanner).
  - Consensus ADMM `admm_manifesto` in `manifesto_opt.py` (lead): ρ = 5, Huber δ = 0.02; converges in ≈ 40
    iterations and matches centralised PG within 1e-3.
  - `strategies/oracle.py` (moved from the balance experiment).
- **Tests:** 194 passed (≈ 64 s).
- **Results** (`reports/v2/strategies.json`, figures in `reports/v2/figures/`; 200 seeds, N = 10k, CPU because
  cuda:2 was saturated by another user). Win rates: Uniform 19.5%, Top-3 83%, Random 2.5%, Bayes even-spend
  88%, **Bayes planner 91%** [86, 94], ORACLE 96%. Bayes beats Uniform and Top-3 with paired p ≈ 1e-34; the
  planner adds +0.012 share over even spend (p = 4e-22); the oracle gap is 0.011. Decision latency is
  0.28 s mean (allocate 0.35 s, commit 0.46 s).
- **Findings:**
  - σ is identified after about 2 queries (the free stat alone takes entropy from 5.89 to 1.66 nats).
  - The residual gap to the oracle is due to the hidden personalities τ, which surveys cannot reveal.
  - The planner shifts spending to media and back-loads it.
  - Raised with the owner: information is cheap under exact queries (affects P5 information sufficiency).

### R12 addendum (2026-11-08)
- Queries are now sampled (10% of the region, ± MoE, repeats pooled within a round).
- Engine, Bayes likelihood (per-answer sample sd), players and UI are updated; Python tests 202 passed,
  web tests 35 passed.
- Results (`reports/v2/strategies_r12.json`): identical to exact queries in 198/200 paired seeds (Wilcoxon
  p = 0.65). Bayes planner 91%, ORACLE 96%.
- Information curve (`reports/v2/info_curve_r12.json`, forced k queries, 60 seeds): flat, with an optimum
  near k = 2; extra queries only drain the budget. The owner kept this as is (R13); P5 targets the uncertainty
  that remains in τ, zealots and the network.
