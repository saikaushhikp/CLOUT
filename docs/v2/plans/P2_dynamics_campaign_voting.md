# P2 — Dynamics, campaign, voting + balance calibration (Days 3–4)

**Spec:** §3, §4, §5, §12. **Owner areas:** MAS (Deek), resources (Hem). **Depends on:** P1.

## Deliverables
- `dynamics/social.py`: `aggregate_degroot(v, graph)` (CSR SpMM / index_add, isolates keep own v);
  `aggregate_wmsr(v, graph, F)` (per-node trimming of the F farthest neighbours in TV distance;
  segment-sort on GPU).
- `dynamics/campaign.py`: `worker_exposure(electorate, workers_by_region, q, eta_w, gen)`,
  `sir_cascade(graph, seeds_mask, gamma, tau_c, rho, L, eta_wom, gen)`,
  `media_exposure(electorate, regions, airings, p_m, eta_m0, lam, gen)`, `combine(*intensities)`.
- `dynamics/update.py`: `step_preferences(v, a, c_by_player, b_by_player, agg)` (the scaling rule,
  simplex renormalisation, zealot freeze). Supports a leading batch dimension B.
- `voting.py`: `similarity(v, b)`, `vote_prob(S, beta_v, z)`, `expected_share`, `sample_election`.
- `experiments/balance.py`: the §12 gate over ≥ 200 seeds at N = 10k using simple fixed policies
  (no-campaign, workers-only by region, media-only, mixed) and a temporary oracle-b solver (simple
  projected gradient on the true v; the full optimiser comes in P4). Writes
  `reports/v2/balance_<date>.json` + plots.

## Tests
- Toy graphs (3–5 nodes) with hand-computed DeGroot and W-MSR updates.
- Simplex preserved after 50 random steps; zealots unchanged; a + c ≤ 1 respected.
- SIR: no spread when τ_c = 0; full reach of a connected component when τ_c·w = 1, ρ = 0 and L ≥ diameter;
  seeded determinism.
- Media: exposure fraction ≈ p_m; diminishing η_m(n).
- Voting: S = 1 when v = b; P at S = z equals 0.5; Bernoulli mean ≈ E.
- Batched (B > 1) equals a loop over single worlds.

## Calibration protocol
Grid over ⚙ (β_v, z, η_w, η_wom, τ_c, η_m0, m0, cost_w, W_free, κ, ω, ζ) using batched GPU worlds.
Choose the default that satisfies all §12 criteria with margin. Log every change in
`decisions_v2.md` (Parameter changes) and freeze the values in spec v2.1.

## Exit criteria
Tests pass; the balance report shows §12 criteria 1–5 satisfied; spec bumped to v2.1.

## Handoff (2026-11-07)
- Pre-P2 owner changes applied: k̄ = 30 (R6); every region ≥ 15% of N (R7); canonical CPU-stream
  electorate so a player's (N, seed) recreates the identical electorate on any device (R8/R9; about 0.4 s at
  1e5 and 3.8 s at 1e6); τ_c calibrated near the regional SIR threshold (R10).
- Built in parallel by two sub-agents plus the lead:
  - `dynamics/social.py` (DeGroot, W-MSR on the simplex), `dynamics/update.py`, `voting.py` (agent A);
  - `dynamics/campaign.py` (workers, SIR cascade, media, costs, `campaign_round`) and
    `experiments/cascade_threshold.py` (agent B);
  - `simulate.py` (single-world loop), `strategies/manifesto_opt.py` (simplex projection = prox of the
    indicator, multi-start projected gradient), `experiments/balance.py` (lead).
- Tests: **120 passed** (GPU parity tests on cuda:2 included).
- Cascade measurement (N = 10k): λ_max(A_rr) ≈ 18.5–27.9, so the threshold ρ/λ ≈ 0.018–0.027. τ_c = 0.03
  gives 1.6–3.3 secondary voters per contacted seed over L = 3 hops.
- **Balance gate passed** (200 seeds, all five §12 criteria): spec §14, decision C1,
  `reports/v2/balance_gate_v2.1.json`. The tuned defaults are frozen in `config.py`.
- Key insight for P4/P5: manifestos must be optimised against the *anticipated* electorate (after social-only
  dynamics), not v(0). `balance.anticipate` / `oracle_manifesto` / `synthetic_world` are prototypes to move
  into `strategies/` in P4.
- Open for P3+: `experiments/balance.py` policies are oracle or calibration tools, not player strategies.
