# P7 experiments — results (Dumbocracy v2.1)

Every experiment pairs a **measurement** with a **prediction from the course material** (MAS consensus
algorithms, Linear-Algebra review, Distributed Learning, Infection Models, LeBlanc et al. 2013 resilient
consensus). Default model = spec v2.1 §14 (N = 10 000, k̄ = 30, p_g ∈ [0.01, 0.2], ω = 0.15, ζ = 0.05,
τ_c = 0.03, ρ = 0.5, L = 3, B = 500, T = 5). All CIs are 95 %: Wilson for rates, t-intervals for means;
paired tests use the **same seeds in every cell** (Wilcoxon signed-rank on expected share E, exact McNemar on
victories). "ORACLE" = oracle manifesto (true electorate, anticipated under social-only dynamics) + the
oracle marginal-value planner of `experiments/balance.py`; it is a labelled benchmark, never a player.
"BayesPlanner" / "Top3Player" play through the player-facing `GameEngine` on observation dicts only.

| ID | Script | Seeds per cell | Compute | Wall time |
|---|---|---|---|---|
| E1 topology & consensus speed | `experiments/e1_topology.py` | 200 (12 cells, 2 400 worlds; 4 000-round tail on the first 60 seeds) | cuda:2 (RTX A5000) + CPU SciPy eigensolvers, 7 workers × 3 threads | 50.7 min |
| E2 zealots × W-MSR | `experiments/e2_zealots.py` | 200 (4 ζ × 3 rules) | cuda:2, 4 workers | 10.5 min |
| E3 cascades / SIR threshold | `experiments/e3_cascades.py` | 200 (8 τ_c) | cuda:2, 4 workers | 5.7 min |
| E4 information sufficiency | consolidated (`info_curve_r12.json`, `info_sufficiency.json`) | 60 / 40 | CPU (earlier run) | — |
| E5 budget | `experiments/e5_budget.py` | 200 (3 budgets) | CPU, 4 workers | 6.4 min |
| E6 strategy ranking | consolidated (`strategies_r12.json`) | 200 | CPU (earlier run) | — |
| E7 scale | `experiments/e7_scale.py` | 1 canonical seed × 3 repeats per N | cuda:2 and CPU (fresh subprocess per N) | 1.8 min |

Figures are produced only from the saved JSON by `python experiments/plot_p7.py` →
`reports/v2/figures/e{1..7}_*.png`. Logs: `reports/v2/logs/`.

---

## E1 — Topology and consensus speed

**Question.** Does community structure (k̄, inter-region probability p_g) set how fast social averaging mixes
opinions, as the spectral theory says, and does it change who wins and which campaign mechanism is best?

**Setup.** k̄ ∈ {8, 16, 30, 48} × p_g ∈ {[.01,.05], [.01,.2] (default), [.1,.4]}, 200 paired seeds (7000–7199).
Spectra on the giant component (SciPy Lanczos). Consensus is measured on a zealot-free copy of each world (same
graph and α) with **social-only** dynamics x ← (1 − a)x + a·D⁻¹A x.

**Theory (derivation).** With G = diag(a/d) the update is M = I − G L, similar to the symmetric I − G^½ L G^½; the
disagreement x − 1c (c weighted by w ∝ d/a, the conserved quantity) contracts per round by ρ_exact = 1 − ν₂(G^½LG^½).
For a **uniform step** a_i = ā (the lecture setting: lazy DeGroot / DGD mixing matrix W = (1 − ā)I + ā D⁻¹A),
ν₂ = ā(1 − μ₂(D⁻¹A)), so the predicted contraction is **ρ = 1 − ā(1 − μ₂)** — the Distributed-Learning
lecture's "larger spectral gap 1 − λ₂(W) ⇒ faster mixing". The regular-graph Laplacian form is
ρ_L = 1 − ā λ₂(L)/k̄ (Infection-Model lecture: "convergence speed ∝ λ₂(L)"). Round-1 within-region variance of
i.i.d. noise: W(1)/W(0) = mean_i[(1 − a_i)² + a_i²/d_i].

| cell (k̄ \| p_g) | λ₂(L) | μ₂(D⁻¹A) | observed decay 1−ρ (uniform ā) | predicted ā(1−μ₂) | pred/obs | ORACLE win | Bayes win | Top3 win | best paid pattern W/M/S |
|---|---|---|---|---|---|---|---|---|---|
| 8 \| low | 0.39 | 0.920 | 0.0060 | 0.0054 | 0.89 [0.88, 0.90] | 0.69 [0.62, 0.75] | 0.65 [0.58, 0.71] | 0.44 [0.37, 0.50] | 17/97/86 |
| 8 \| default | 0.48 | 0.813 | 0.0131 | 0.0125 | 0.95 [0.94, 0.96] | 0.84 [0.78, 0.88] | 0.79 [0.72, 0.84] | 0.66 [0.59, 0.72] | 18/101/81 |
| 8 \| high | 0.55 | 0.711 | 0.0218 | 0.0194 | 0.89 [0.87, 0.90] | 1.00 [0.98, 1.00] | 0.98 [0.95, 0.99] | 0.79 [0.72, 0.84] | 15/107/78 |
| 16 \| low | 1.25 | 0.904 | 0.0071 | 0.0064 | 0.91 [0.90, 0.92] | 0.72 [0.65, 0.78] | 0.67 [0.60, 0.73] | 0.52 [0.45, 0.58] | 45/65/90 |
| 16 \| default | 1.96 | 0.764 | 0.0161 | 0.0158 | 0.98 [0.98, 0.99] | 0.86 [0.80, 0.90] | 0.82 [0.76, 0.86] | 0.71 [0.64, 0.76] | 47/71/82 |
| 16 \| high | 2.37 | 0.563 | 0.0296 | 0.0293 | 0.99 [0.99, 0.99] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 0.86 [0.81, 0.90] | 41/79/80 |
| 30 \| low | 2.99 | 0.897 | 0.0075 | 0.0069 | 0.92 [0.91, 0.93] | 0.77 [0.70, 0.82] | 0.71 [0.64, 0.76] | 0.63 [0.56, 0.69] | 86/30/84 |
| **30 \| default** | 6.56 | 0.743 | 0.0174 | 0.0172 | 0.99 [0.99, 0.99] | 0.91 [0.86, 0.94] | 0.86 [0.80, 0.90] | 0.75 [0.68, 0.80] | 85/37/78 |
| 30 \| high | 9.05 | 0.508 | 0.0332 | 0.0329 | 0.99 [0.99, 0.99] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 0.94 [0.89, 0.96] | 85/41/74 |
| 48 \| low | 4.96 | 0.893 | 0.0077 | 0.0071 | 0.92 [0.92, 0.93] | 0.80 [0.73, 0.85] | 0.77 [0.70, 0.82] | 0.72 [0.65, 0.78] | 121/5/74 |
| 48 \| default | 11.68 | 0.734 | 0.0180 | 0.0178 | 0.99 [0.99, 0.99] | 0.98 [0.94, 0.99] | 0.94 [0.89, 0.96] | 0.88 [0.83, 0.92] | 125/11/64 |
| 48 \| high | 19.08 | 0.485 | 0.0347 | 0.0345 | 0.99 [0.99, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 1.00 [0.98, 1.00] | 128/14/58 |

(λ₂, μ₂, decay = cell means; their CIs are ≤ ±5 % of the mean and are in the JSON. ā = ω·E[α] ≈ 0.067.)

**Theory vs observation.**
* **Uniform step — confirmed.** Across all 2 400 worlds the observed contraction tracks 1 − ā(1 − μ₂(D⁻¹A)) with
  Spearman 0.995 and median predicted/observed decay 0.98 (`e1_consensus_theory.png` a). The prediction is
  slightly slow (pred/obs ≈ 0.9) only for the most modular graphs, where μ₂…μ₄ are nearly equal and the three
  community modes have not separated within the 150–400-round fit window. The **λ₂(L)/k̄ form is much worse**
  (Spearman 0.67, median ratio 0.68; 0.21–0.55 at k̄ = 8): λ₂(L) is held down by low-degree voters, so for
  irregular graphs the random-walk gap 1 − μ₂(D⁻¹A), not λ₂(L), is the right speed for DeGroot averaging.
* **Game dynamics over T = 5 rounds.** With heterogeneous a_i = ω·α_i the between-region variance decays at
  rate ā(1 − mean μ₂…μ₄) (Spearman 0.86, median ratio 0.87; panel b); the round-1 within-region ratio is
  0.8735 [0.8732, 0.8737] observed vs 0.8719 [0.8717, 0.8722] predicted by the i.i.d. bulk formula
  (`e1_variance_curves.png`). Within five rounds voters move only ~1–5 % of the way per round toward consensus:
  regional means barely mix when p_g is low and noticeably when it is high.
* **Long run — the spectral gap stops mattering.** The 4 000-round tail (60 seeds × 12 cells) is governed by the
  exact operator ν₂ (Spearman 0.98 with observed, median ν₂/obs 0.87) and **not by λ₂** (Spearman 0.06, p = 0.13):
  ν₂ ≈ 0.96 · min_i a_i (Spearman 0.994 with the least-susceptible voter's step). Heterogeneous step sizes create a
  mode localised on the most stubborn (lowest-α) voter — a reminder that the lecture's λ₂ result assumes a
  homogeneous mixing matrix.

**Outcomes.** Win rates rise with both k̄ and p_g for every player (paired vs default cell, Wilcoxon p < 1e−21
everywhere; e.g. ORACLE E: −0.064 at k̄ 8|low, +0.064 at k̄ 48|high). Faster between-region mixing pulls the
whole electorate toward a common mean that one manifesto can serve (no-campaign ORACLE wins 15 % at low p_g vs 66 %
at high p_g; Spearman(between-region decay, no-campaign E) = 0.50). The **best paid pattern flips with k̄**:
media is best on 49–54 % of seeds at k̄ = 8 but on ≤ 7 % at k̄ = 48, while workers go from 8 % to 61–64 % — exactly
the SIR prediction of E3: τ_c·λ_max(A_rr)/ρ ≈ 0.3–0.5 (subcritical) at k̄ = 8 vs ≈ 1.4–2.5 at k̄ = 48.

**Interpretation.** The lecture's spectral-gap law is quantitatively right for the uniform-step DeGroot map once
the gap is measured on the random-walk matrix D⁻¹A (the matrix the game actually uses). Over the five-round game,
community structure (p_g) controls how much regions converge, and k̄ controls whether word-of-mouth cascades are
super-critical; both raise the win rate of informed players, and k̄ decides workers-vs-media.

---

## E2 — Zealots × resilient consensus (W-MSR)

**Question.** Do zealots block persuasion, and does W-MSR (trimming the F farthest neighbours) neutralise them —
or does it also trim the campaign-shifted neighbours and slow persuasion?

**Setup.** ζ ∈ {0, 5, 15, 30 %} × {DeGroot, W-MSR F = 1, W-MSR F = 2}, 200 paired seeds (8000–8199). ORACLE
manifesto re-optimised under each rule; persuasion = E_T(planner) − E_T(no campaign) (same manifesto; also with
one fixed DeGroot manifesto). Drag probe (outside the game rules): zealots set to the simplex vertex of the
sector their region likes least; drag = TV shift of the non-zealot regional means vs honest zealots after T rounds.

**Theory.** LeBlanc et al. 2013: W-MSR reaches resilient consensus under the **F-total** model iff the graph is
**(F+1, F+1)-robust** (Thm 1); under the **F-local** model **(2F+1)-robustness suffices** and (F+1)-robustness is
necessary (Thm 2); normal values always stay in the hull of normal values (safety). First-order drag prediction:
a voter with z_i zealot neighbours feels zealot weight z_i/d_i under DeGroot and max(0, z_i − F)/(d_i − F) under
W-MSR, giving a predicted drag ratio W-MSR/DeGroot (table).

| ζ | rule | ORACLE win | persuasion (E points) | no-campaign E_T | drag T = 30 | drag ratio vs DeGroot obs (T=5 / T=30) | first-order prediction | normal voters with > F zealot nbrs |
|---|---|---|---|---|---|---|---|---|
| 0 | DeGroot | 0.955 [0.917, 0.976] | 0.1275 [0.1252, 0.1298] | 0.489 | — | — | — | 0 |
| 0 | W-MSR F=1 | 0.915 [0.868, 0.946] | 0.1270 [0.1247, 0.1293] | 0.480 | — | — | — | 0 |
| 0 | W-MSR F=2 | 0.890 [0.839, 0.926] | 0.1264 [0.1240, 0.1288] | 0.472 | — | — | — | 0 |
| 0.05 | DeGroot | 0.915 [0.868, 0.946] | 0.1187 [0.1166, 0.1208] | 0.483 | 0.045 [0.044, 0.046] | 1 | 1 | — |
| 0.05 | W-MSR F=1 | 0.880 [0.828, 0.918] | 0.1182 [0.1160, 0.1204] | 0.475 | 0.025 [0.024, 0.026] | 0.57 / 0.56 | 0.49 | 44 % |
| 0.05 | W-MSR F=2 | 0.850 [0.794, 0.893] | 0.1177 [0.1155, 0.1199] | 0.468 | 0.016 [0.016, 0.017] | 0.42 / 0.37 | 0.19 | 20 % |
| 0.15 | DeGroot | 0.815 [0.755, 0.863] | 0.1020 [0.1001, 0.1039] | 0.472 | 0.125 [0.122, 0.127] | 1 | 1 | — |
| 0.15 | W-MSR F=1 | 0.780 [0.718, 0.832] | 0.1015 [0.0996, 0.1034] | 0.465 | 0.107 [0.104, 0.109] | 0.81 / 0.86 | 0.79 | 93 % |
| 0.15 | W-MSR F=2 | 0.770 [0.707, 0.823] | 0.1012 [0.0992, 0.1031] | 0.458 | 0.083 [0.081, 0.085] | 0.62 / 0.67 | 0.59 | 81 % |
| 0.30 | DeGroot | 0.720 [0.654, 0.778] | 0.0793 [0.0778, 0.0808] | 0.455 | 0.222 [0.218, 0.227] | 1 | 1 | — |
| 0.30 | W-MSR F=1 | 0.710 [0.644, 0.768] | 0.0790 [0.0775, 0.0805] | 0.450 | 0.221 [0.217, 0.225] | 0.94 / 0.99 | 0.91 | 99.7 % |
| 0.30 | W-MSR F=2 | 0.705 [0.638, 0.764] | 0.0788 [0.0773, 0.0804] | 0.445 | 0.208 [0.204, 0.212] | 0.85 / 0.94 | 0.82 | 98.7 % |

**Robustness of the default electorate (200 graphs).** Best upper bound on r-robustness = min degree in every
graph (region-union and greedy-core certificates never beat it): mean 8.5 [8.2, 8.7], range 2–12. So the graphs
are *not refuted* as 3-robust (F = 1 local guarantee) in 99.5 % and as 5-robust (F = 2) in 95.5 % of seeds; one
graph has a degree-2 voter and therefore cannot be 3-robust. Region-union pairs give s-upper bounds ≥ 1 164 for
(3, s), far above F + 1, so nothing refutes (F+1, F+1)-robustness either. These are upper bounds only (the exact
check is coNP-hard). **But the threat models fail by a wide margin**: ζN ≈ 500–3 000 zealots ≫ F (F-total), and
44 % / 20 % of normal voters already have more than F = 1 / F = 2 zealot neighbours at ζ = 5 % (93 % / 81 % at 15 %),
so the F-local hypothesis of Theorem 2 is violated even though the topology is robust enough.

**Theory vs observation.**
* **Zealots block persuasion — yes, linearly.** Persuasion falls from 0.128 (ζ = 0) to 0.119 / 0.102 / 0.079
  (paired differences −0.009 / −0.026 / −0.048, p < 1e−33) and ORACLE wins drop from 95.5 % to 72 % (McNemar
  0 vs 47 discordant, p = 1e−14); zealots are frozen and keep pulling persuaded neighbours back.
* **W-MSR trims extreme zealots roughly as predicted where the F-local model nearly holds**: at ζ = 5 % drag falls
  to 0.56× (F = 1, predicted 0.49×) and 0.37× (F = 2, predicted 0.19×); the first-order formula underestimates the
  residual drag because zealot influence also travels two hops through already-shifted normal voters, which W-MSR
  does not trim. As ζ grows the protection vanishes (0.99× and 0.94× at ζ = 30 %), matching the theorem's message:
  once most voters see > F adversaries, trimming F values cannot help.
* **W-MSR does not measurably slow campaign persuasion.** Persuasion differences W-MSR − DeGroot are within
  ±0.001 (p ≥ 0.29 with the rule-specific manifesto; +0.0006 to +0.0026 with a fixed manifesto). Campaign pull acts
  directly on c_i, and campaign-shifted neighbours are rarely among the F farthest. What W-MSR does cost is
  **social convergence**: the no-campaign expected share drops by 0.009 (F = 1) and 0.018 (F = 2) at ζ = 0
  (p < 1e−33), because trimming the most different neighbours slows the drift toward regional means that a good
  manifesto exploits; that alone lowers ORACLE wins 95.5 % → 91.5 % → 89 % (McNemar p = 0.008 / 2e−4).

**Interpretation.** In this electorate the topology is (as far as can be certified) robust enough for W-MSR with
F = 1–2, but the zealot density violates both threat models, so W-MSR gives at best partial protection against
extreme stubborn agents (≈ 2× less drag at 5 % zealots, none at 30 %). Honest in-distribution zealots are not
outliers, so W-MSR mainly makes the social process more conservative; campaigns are neither helped nor hurt.

---

## E3 — Word-of-mouth cascades vs the SIR spectral threshold

**Question.** Does word-of-mouth reach collapse on the dimensionless x = τ_c·λ_max/ρ and change behaviour at
x ≈ 1, and when do paid workers beat paid media?

**Setup.** τ_c ∈ {0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.07, 0.12}, ρ = 0.5, L = 3, 200 paired seeds (9500–9699).
Reach probe: 250 random voters of one region (= the 10 free workers' contacts) seed a cascade on the full graph,
2 trials per region. λ_max(A_rr) = 21.3 [20.9, 21.7], λ_max(A) = 33.2 [32.9, 33.4].

**Theory.** Infection-Model lecture: linearised SIR i̇ ≈ (βA − γI)i grows iff β λ_max(A)/γ > 1. For the engine's
hop-synchronous cascade, I_{h+1} = (1 − ρ)I_h + τA I_h, so the per-hop growth is **g = 1 − ρ + τλ_max = 1 + ρ(x − 1)**
and the cascade grows hop-over-hop iff **x > 1**. Mean-field reach per seed = τk̄·Σ_{h<L} g^h. Regional threshold
τ* = ρ/λ_max(A_rr) = 0.025 [0.025, 0.026]; whole-graph τ* = 0.015.

| τ_c | x_rr | reach / seed | hop growth new₃/new₂ (pred. on x_rr) | P(pay_workers > pay_media) | best W/M/S | ORACLE win | ΔE vs τ = 0.03 (paired) |
|---|---|---|---|---|---|---|---|
| 0.005 | 0.21 | 0.26 [0.26, 0.27] | 0.65 (0.61) | 0.18 [0.13, 0.24] | 8/111/81 | 0.87 [0.82, 0.91] | −0.020 |
| 0.01 | 0.43 | 0.60 [0.59, 0.61] | 0.77 (0.71) | 0.26 [0.20, 0.32] | 16/103/81 | 0.88 [0.83, 0.92] | −0.017 |
| 0.015 | 0.64 | 1.02 [1.01, 1.04] | 0.88 (0.82) | 0.39 [0.33, 0.46] | 33/79/88 | 0.89 [0.84, 0.93] | −0.014 |
| 0.02 | 0.85 | 1.54 [1.51, 1.56] | 1.01 (0.93) | 0.55 [0.48, 0.61] | 46/60/94 | 0.91 [0.86, 0.94] | −0.010 |
| **0.03** | 1.28 | 2.86 [2.81, 2.90] | 1.23 (1.14) | 0.74 [0.68, 0.80] | 81/35/84 | 0.94 [0.90, 0.97] | 0 |
| 0.045 | 1.92 | 5.55 [5.46, 5.64] | 1.50 (1.46) | 0.92 [0.87, 0.95] | 116/14/70 | 0.98 [0.95, 0.99] | +0.017 |
| 0.07 | 2.98 | 11.5 [11.3, 11.6] | 1.80 (1.99) | 0.97 [0.93, 0.98] | 125/7/68 | 1.00 [0.97, 1.00] | +0.048 |
| 0.12 | 5.11 | 24.3 [24.1, 24.5] | 1.81 (3.06) | 0.79 [0.73, 0.84] | 54/24/122 | 1.00 [0.98, 1.00] | +0.091 |

(All paired ΔE: Wilcoxon p = 1.4e−34.)

**Theory vs observation.**
* **Collapse — yes, on the regional x.** Reach per seed vs x_rr = τ_c·λ_max(A_rr)/ρ falls on one smooth curve
  across τ_c, regions and seeds (`e3_cascade_threshold.png` a); at fixed τ_c, regional differences in reach are
  explained by λ_max(A_rr) (Spearman 0.79 at τ = 0.03; 0.68–0.80 for τ ≤ 0.07). The whole-graph x carries no
  within-τ information (all regions of a seed share it).
* **Behaviour changes at x ≈ 1 — yes.** Hop growth crosses 1 at x_rr ≈ 0.87 (x_full ≈ 1.27); the linearised
  g = 1 + ρ(x − 1) on x_rr is accurate to within 0.09 for x ≤ 2 (observed slightly higher: cross-region edges add
  spreading paths), i.e. the effective threshold lies between the regional
  and the whole-graph eigenvalue (≈ 52–64 % of a cascade stays in-region near threshold). Above x ≈ 3 growth
  saturates at 1.8 (depletion: neighbourhoods are already reached; L = 3 caps reach), so mean-field reach
  over-predicts by 2–4× there; below threshold it over-predicts by 20–30 % (overlapping neighbourhoods).
* **Workers vs media.** Paid workers beat paid media on 18 % of seeds well below threshold, 55 % at τ = 0.02
  (x_rr ≈ 0.85), 74 % at the default 0.03 and 92–97 % at τ ≥ 0.045; the crossover sits right at the SIR threshold.
  At τ = 0.12 both saturate (all three patterns win ≈ 100 %) and the split pattern becomes best. The ORACLE win
  rate rises monotonically from 87 % to 100 %.

**Interpretation.** The spectral epidemic threshold of the lecture predicts both the physics (reach collapses on
τλ_max/ρ, growth turns on at x ≈ 1) and the economics (workers overtake media at the threshold). The default
τ_c = 0.03 (x_rr ≈ 1.3) sits just above threshold, which is why the P2 balance gate found workers best on most
— but not all — seeds.

---

## E4 — Information sufficiency (consolidated; not rerun)

Sources: `reports/v2/info_curve_r12.json` (BayesPlanner forced to ask exactly k survey queries, 60 seeds) and
`reports/v2/info_sufficiency.json` (epistemic timeline, 40 seeds × {gain, random} query selection).

**Theory.** Epistemic-logic lecture: K_p φ holds iff φ is true in every world of the information set; Bayesian
experimental design picks the query with maximum expected information gain. Archetypes are cheap to learn (R13):
posterior entropy over σ falls from 1.60 nats (k = 0) to 0.52 (k = 1) and 0.045 (k = 2) under gain selection,
vs 0.94 / 0.48 under random selection.

| k | win rate, forced-k BayesPlanner (n = 60) | mean E | K_p(GuaranteedVictory) at round 1, gain (n = 40) | same, random |
|---|---|---|---|---|
| 0 | 0.87 [0.76, 0.93] | 0.5785 | 0.65 [0.50, 0.78] | 0.65 [0.50, 0.78] |
| 1 | 0.87 [0.76, 0.93] | 0.5807 | 0.70 [0.55, 0.82] | 0.68 [0.52, 0.80] |
| 2 | 0.87 [0.76, 0.93] | 0.5812 | 0.68 [0.52, 0.80] | 0.68 [0.52, 0.80] |
| 4 | 0.85 [0.74, 0.92] | 0.5799 | 0.68 [0.52, 0.80] | 0.68 [0.52, 0.80] |
| 8 | 0.85 [0.74, 0.92] | 0.5773 | 0.68 [0.52, 0.80] | 0.68 [0.52, 0.80] |
| 12 | 0.83 [0.72, 0.91] | 0.5750 | — | — |

Paired vs k = 0: +0.0027 E at k = 2 (Wilcoxon p = 0.05), −0.0034 at k = 12 (p = 0.003): two queries resolve σ and
every further query is a net cost (money that would otherwise buy campaign). When K_p(GuaranteedVictory) holds at
round 1 the player won in 26/26–28/28 games; when it does not, only 54–77 % won. **Interpretation:** information
about σ saturates after ~2 queries; what stays unknowable (personalities τ, zealots, the network) is what keeps
GuaranteedVictory below ~70 %.

---

## E5 — Budget: P(Victory | B)

**Question.** How does the win probability depend on the starting treasury, and how is money spent?

**Setup.** B ∈ {250, 500, 1000}, 200 paired seeds (5000–5199 — the seeds of E6; B = 500 reproduces
`strategies_r12.json` exactly: Bayes 0.910, Top3 0.830, ORACLE 0.960).

**Theory.** The oracle planner is a greedy Lagrangian/knapsack rule (marginal expected-vote gain per coin);
media has diminishing returns η_m0/(1 + λn) and campaign intensities compound multiplicatively, so the marginal
value of money should fall with B (concave P(Victory | B)).

| B | ORACLE win | BayesPlanner win | Top3 win | ORACLE E | Bayes E | Bayes spend workers / media / queries |
|---|---|---|---|---|---|---|
| 250 | 0.835 [0.777, 0.880] | 0.815 [0.755, 0.863] | 0.690 [0.623, 0.750] | 0.565 [0.557, 0.572] | 0.557 [0.549, 0.565] | 35 % / 61 % / 4 % |
| 500 | 0.960 [0.923, 0.980] | 0.910 [0.862, 0.942] | 0.830 [0.772, 0.876] | 0.601 [0.594, 0.609] | 0.590 [0.583, 0.598] | 42 % / 56 % / 2 % |
| 1000 | 1.000 [0.981, 1.000] | 0.990 [0.964, 0.997] | 0.995 [0.972, 0.999] | 0.651 [0.643, 0.658] | 0.640 [0.633, 0.648] | 60 % / 39 % / 1 % |

Paired: Bayes E −0.033 [−0.035, −0.032] for 500 → 250 and +0.050 [+0.049, +0.051] for 500 → 1000 (both
p = 1.4e−34; McNemar 0/19 and 16/0 discordant). ORACLE − Bayes gap: 0.008 / 0.011 / 0.010 E points.

**Theory vs observation.** *Per coin*, returns diminish as predicted: +0.037 E for the 250 extra coins from 250 to
500 (1.5e−4 per coin) vs +0.049 for the next 500 (1.0e−4 per coin). *Per doubling*, however, the expected-share gain
does not shrink, and the win rate saturates only because the electorate threshold is crossed (≥ 99 % at 1000).
The spend mix shifts from media to workers as B grows (ORACLE 36 % → 59 % workers; Bayes 35 % → 60 %): media airings
in a region decay as 1/(1 + 0.5n), so extra money is best spent on workers once the best media slots are used.
Top3's fixed half/half rule cannot afford any media at B = 250 (one airing costs more than its per-round cap), which
is why it loses most from a budget cut (−0.037 E, 28 wins lost).

**Interpretation.** Money matters roughly logarithmically in expected share but saturates in win probability:
halving the default budget costs 10–14 points of win rate for every player, doubling it makes victory near-certain.
The information-driven BayesPlanner stays within 0.01 E of the full-information ORACLE at every budget.

---

## E6 — Strategy ranking (consolidated; not rerun)

Source: `reports/v2/strategies_r12.json` (200 paired seeds, N = 10 000, R12 sampled surveys).

| player | win rate | mean E | queries | spend workers / media |
|---|---|---|---|---|
| Random | 0.025 [0.011, 0.057] | 0.147 | 1.9 | 309 / 157 |
| Uniform | 0.195 [0.146, 0.255] | 0.443 | 0 | 0 / 0 |
| Top-3 statistic | 0.830 [0.772, 0.876] | 0.559 | 0 | 286 / 212 |
| Bayes (even spend) | 0.880 [0.828, 0.918] | 0.578 | 2.0 | 279 / 209 |
| **BayesPlanner** | **0.910 [0.862, 0.942]** | **0.590** | 2.0 | 211 / 277 |
| ORACLE (benchmark) | 0.960 [0.923, 0.980] | 0.601 | — | 498 total |

Paired (Wilcoxon on E, all p < 1e−21): Bayes − Top3 +0.031; Bayes − Bayes-even +0.012 (the rollout planner's
value); ORACLE − Bayes +0.011 (the value of perfect information). **Interpretation:** the ranking follows the
lecture structure — information (Bayesian surveys → manifesto via projected-gradient/prox on the simplex) is
worth +0.019 E over the free statistic, planning by rollouts another +0.012, and perfect information only +0.011.

---

## E7 — Scale

**Question.** Does the implementation meet the spec §9 targets (interactive N = 10k < 1 s per round; 1e5 batched
experiments; 1e6 single world on one 24 GB GPU)?

**Setup.** Seed 20261107, 3 generation repeats, 5 engine rounds of a fixed allocation (20 workers incl. 10 extra,
media in the largest region in rounds 1–3), full games through `play_game`; each (N, device) in a fresh process.
GPU = cuda:2 (RTX A5000, idle at launch); CPU = 4 threads on a shared, loaded host (load average ≈ 12), so CPU
times are conservative. Total wall time 107 s.

**Theory.** Everything is edge-list/CSR based (no N × N tensor): O(E) = O(k̄N) memory and time per round, i.e.
linear in N at fixed k̄; batched GPU kernels should be latency-bound (flat) until the work saturates the device.

| N | edges | canonical generation (CPU draw → device) | GPU-side generation | engine round GPU / CPU | full game Top3 GPU / CPU | full game BayesPlanner GPU / CPU | peak GPU (generation / engine) | peak host RSS GPU / CPU run |
|---|---|---|---|---|---|---|---|---|
| 1e4 | 0.15 M | 0.42 s | 0.011 s | 4.8 ms / 16 ms | 0.03 s / 0.09 s | 3.4 s / 3.2 s | 0.05 / 0.01 GiB | 1.6 / 1.1 GiB |
| 1e5 | 1.50 M | 0.40 s | 0.016 s | 6.4 ms / 149 ms | 0.03 s / 0.74 s | 3.3 s / 3.8 s | 0.33 / 0.14 GiB | 1.6 / 1.1 GiB |
| 1e6 | 15.0 M | 6.3 s (CPU run 5.3 s) | 0.074 s | 28 ms / 2.06 s | 0.14 s / 10.5 s | 3.2 s / 13.5 s | 3.2 / 1.4 GiB | 3.2 / 3.5 GiB |

(Medians; generation = 3 repeats, round = 5 rounds; the first GPU-side generation includes ≈ 0.25 s of kernel warm-up.)

**Theory vs observation.** CPU rounds scale linearly as predicted (16 ms → 149 ms → 2.06 s: ×9.3 and ×13.8 per
decade); on the GPU a round stays latency-bound (4.8 → 6.4 ms) up to 1e5 and costs only 28 ms at 1e6 — 74× faster
than CPU. Peak GPU memory grows linearly (≈ 3.2 GiB per 1e6 voters at k̄ = 30, generation-dominated), i.e. 13 % of
the 24 GB card at 1e6. The canonical (bit-reproducible, R8) generation is flat at ≈ 0.4 s up to 1e5 (fixed costs)
and 5–6 s at 1e6; drawing on the GPU is 85× faster (0.074 s) but device-specific. BayesPlanner's game time is
≈ 3.3 s regardless of N on the GPU because it plans on 1 600-voter posterior worlds.

**Verdict.** All §9 targets met with large margins: N = 10k rounds in 5 ms (GPU) / 16 ms (CPU) vs the 1 s target;
1e6 runs a complete game in 0.14 s (scripted) or 3.2 s (BayesPlanner) after a 6 s canonical generation, using
3.2 GiB of GPU memory.

---

## Files

Scripts: `experiments/p7_common.py`, `experiments/e1_topology.py`, `experiments/e2_zealots.py`,
`experiments/e3_cascades.py`, `experiments/e5_budget.py`, `experiments/e7_scale.py`, `experiments/plot_p7.py`.
Data: `reports/v2/e1_topology.json`, `e2_zealots.json`, `e3_cascades.json`, `e5_budget.json`, `e7_scale.json`
(+ consolidated `info_curve_r12.json`, `info_sufficiency.json`, `strategies_r12.json`). Figures:
`reports/v2/figures/e1_consensus_theory.png`, `e1_variance_curves.png`, `e1_outcomes.png`, `e2_zealots.png`,
`e2_drag_ratio.png`, `e3_cascade_threshold.png`, `e3_workers_vs_media.png`, `e4_information.png`, `e5_budget.png`,
`e6_strategies.png`, `e7_scale.png`. Logs: `reports/v2/logs/`.

Reproduce (each a single command; results depend only on seeds, not on `--workers`):
```
python experiments/e1_topology.py --seeds 200 --n 10000 --device cuda:2 --workers 7 --out reports/v2/e1_topology.json
python experiments/e2_zealots.py  --seeds 200 --n 10000 --device cuda:2 --workers 4 --out reports/v2/e2_zealots.json
python experiments/e3_cascades.py --seeds 200 --n 10000 --device cuda:2 --workers 4 --out reports/v2/e3_cascades.json
python experiments/e5_budget.py   --seeds 200 --n 10000 --device cpu    --workers 4 --out reports/v2/e5_budget.json
python experiments/e7_scale.py --n 10000 100000 1000000 --device cuda:2 --cpu-too --out reports/v2/e7_scale.json
python experiments/plot_p7.py
```
No library code was changed (experiments only).
