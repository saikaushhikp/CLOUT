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
