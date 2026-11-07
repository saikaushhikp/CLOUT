# P1 — Electorate generation on GPU (Days 2–3)

**Spec:** §2, §9, §11. **Owner area:** MAS model (Deek). **Depends on:** P0.

## Deliverables
- `electorate/regions.py`: `sample_region_sizes(N, a, gen) -> LongTensor[4]` (§2.1).
- `electorate/sbm.py`: `sample_block_probabilities(gen)` (p_r, p_rs), `sparsity_factor(sizes, p_r, p_rs, k̄)`,
  `sample_sbm(sizes, P, gen, device)` → `Graph` (canonical edge list `src<dst`, CSR both directions, degree). Per-block
  Binomial count + uniform endpoints + `torch.unique` dedupe; never materialise N×N.
- `electorate/preferences.py`: `sample_assignment(gen) -> sigma[4]`, `sample_preferences(sizes,
  sigma, kappa, gen)` (Dirichlet via `torch.distributions.Dirichlet` or Gamma normalisation with the
  stream generator).
- `electorate/attributes.py`: personality permutation τ, Beta draws (Gamma-ratio with generator),
  zealot mask, zeroed susceptibilities for zealots.
- `electorate/electorate.py`: `Electorate` dataclass (sizes, region_of[N], graph, v0[N,5], alpha,
  beta, gamma, zealot, sigma, tau, p_r, p_rs) + `generate_electorate(config, seeds, device)`.
- `analysis/spectral.py`: λ2(L) and λ_max(A_rr) via `torch.lobpcg` or power/Lanczos on sparse;
  eigenvector centrality. `analysis/robustness.py`: min degree, region-union (r,s)-reachability
  bounds, greedy violating-pair search.
- `experiments/scale.py`: generation time and peak memory at 1e4/1e5/1e6.

## Tests (`tests/unit/test_electorate.py`, `test_sbm.py`, `test_analysis.py`, `tests/gpu/`)
- Region sizes: sum = N, each ≥ ceil(0.01N), deterministic per seed.
- SBM: canonical, no loops/duplicates; realised per-block edge counts within 4σ of Binomial(pairs, s·p)
  and mean degree within 3% of k̄ at N = 1e5; p_r, p_rs in their ranges; every P ≤ 1; symmetric CSR.
- Preferences on the simplex; empirical region means ≈ archetype μ (tolerance via κ and n_r).
- σ, τ are permutations; zealot fraction ≈ ζ; zealots have zero susceptibility.
- Seed determinism on one device; distributional agreement CPU vs CUDA (KS-style checks).
- Spectral: path/complete toy graphs with known λ2; block λ_max sanity.
- GPU (marked): 1e5 generation < 5 s; 1e6 fits on one A5000 (record peak memory).

## Exit criteria
All tests pass in `mas`. The scale report is saved to `reports/v2/scale_electorate.json`.

## Handoff (2026-11-07)
- Implemented: `rng.py` (generator-driven gamma/Dirichlet/Beta; torch's gamma has no `generator`),
  `electorate/{regions,sbm,preferences,attributes,electorate}.py`, `analysis/{spectral,robustness}.py`,
  `experiments/scale.py`. The graph follows spec §2.2 as revised (R4/R5): ER per block with
  `P = s·p`, edges unweighted.
- Design notes: small discrete draws (sizes, σ, τ, p_r, p_rs) use CPU streams, so the hidden world type
  is identical on every device; per-voter and per-edge draws use device streams (same seed → same result
  on the same device; CPU vs GPU agree statistically, tested).
- Tests: `pytest -q` → **49 passed** (including the GPU cross-device test).
- Scale (`reports/v2/scale_electorate.json`, cuda:2, warm): 1e4 0.012 s; 1e5 0.015 s / 0.18 GiB;
  **1e6 0.053 s / 1.77 GiB**. Mean degree 16.0 and ~79% intra edges at every N.
- Findings for P2/P7:
  - At mean degree 16 the graph has degree-1 voters and a few isolates at 1e5+. The robustness upper
    bound is r ≤ 1 (0 when isolated voters exist), so W-MSR with F ≥ 1 has **no** resilient-consensus
    guarantee on default electorates. This is a meaningful E2 result, and k̄ is the knob.
  - Region λ_max(A_rr) ≈ 6–20 (it scales with region size). With τ_c = 0.25 and ρ = 0.5, every region is
    far above the SIR threshold τ_c·λ_max > ρ, so cascades would saturate. τ_c must be calibrated
    downward in P2 (e.g. ≈ 0.02–0.05) to put workers near the threshold, where placement matters.
