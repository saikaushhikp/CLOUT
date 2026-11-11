# Experimentation: five questions about the game

> **Main report:** [experiments.md](experiments.md). It covers six questions, adding "Does the best manifesto stay the
> best as opinions change?" (study R4), and uses a new numbering:
> - Q1 manifesto at the start;
> - Q2 drift;
> - Q3 best manifesto over time (new);
> - Q4 cascades;
> - Q5 worker placement;
> - Q6 media vs workers.
>
> Strategy methods are in [strategies.md](strategies.md). The index below keeps the original five-question numbering.

This folder answers five questions about how the game works. It combines three new runs (R1–R3, 2026-11-11)
with results from the existing v3 suite (X0–X23, [`../report.md`](../report.md)).

- **New runs:** each has its own write-up (`r<k>.md`), data (`r<k>.json`, `raw/r<k>.jsonl`), figures
  (`figures/r<k>_*.png`) and log (`logs/r<k>.log`).
- **Numbers:** every number below comes from the named JSON file. Intervals are 95%.
- **Notation:** E is the expected vote share (the mean over voters of each voter's chance of voting for the
  manifesto). A win needs E > 0.5. "Won over" means a voter's agreement with the manifesto is at least 0.825,
  i.e. a vote chance of at least ½.

| Run | Question | Seeds | Device, wall time | Write-up |
|---|---|---|---|---|
| R1 | Q1: what does the ideal manifesto look like at round 0, before any campaign? | 100 (12000–12099, the same worlds as X19) | cuda:2, 140 s | [r1.md](r1.md) |
| R2 | Q2: how do opinions change with no media and no workers, with and without zealots? | 50 (10000–10049, the same worlds as XC) × 2 populations | cuda:2, 82 s | [r2.md](r2.md) |
| R3 | Q4: how should a fixed number of workers be assigned across regions? | 50 per arm (11000–11049, the same worlds as X4) × 3 population scenarios | cuda:0, 103 s | [r3.md](r3.md) |
| R4 | New (Q3 in experiments.md): does the best manifesto stay the best as opinions drift? | 100 × 2 populations (12000–12099, the same worlds as R1) | cuda:2, 634 s | [r4.md](r4.md) |

All three passed their CPU↔GPU parity checks (`provenance.parity` in each JSON) and their unit tests. Only new files
were created; no library or existing file was changed.

---

## Q1. What does the ideal manifesto look like? (round 0, no campaign)

**New: R1** ([r1.md](r1.md))

- **No manifesto wins at round 0 on its own.** The best possible manifesto b\*₀ comes from gradient ascent on the
  true starting preferences of every voter. It reaches E = 0.373 [0.363, 0.384] and wins alone on 0 of 100 seeds
  (`optimum.E_star`, `optimum.win_alone_rate`).
- **Every simple rule is worse than b\*₀ on every seed** (`overall.manifestos.*.gap_vs_bstar0`; Holm p ≤ 2·10⁻¹⁷):

  | Manifesto type | Gap to b\*₀ in E |
  |---|---|
  | best 2- or 3-region group on that seed | −0.008 |
  | TV geometric median (best single rule) | −0.021 |
  | per-sector median | −0.027 |
  | population mean | −0.031 |
  | largest region alone | −0.040 |
  | smaller regions alone | −0.085 to −0.144 |
  | uniform 20% each | −0.071 |

  - Which group is best changes from seed to seed. No fixed group comes within 0.036 of b\*₀ (`overall.derived`).
- **What b\*₀ wins over:** about 1.3 regions holding 39% of voters. Its supporters are concentrated in the largest
  region (`optimum`, `support_by_rank`).
- **Divided vs aligned worlds** (`scenario_split`): a world is "divided" when the two largest regions' archetypes
  are at TV distance ≥ 0.35.
  - E\*₀ is 0.340 in divided worlds and 0.409 in aligned ones.
  - Blending the two largest regions costs −0.058 in divided worlds but only −0.014 in aligned ones.
- **Aiming at election day costs little today.** The ORACLE's election-day manifesto, scored at round 0, is only
  0.008 below b\*₀ (`overall.manifestos.oracle`).
- **Gradient ascent:** a median of 10 iterations to within 10⁻⁴ of the optimum.

Figures: [share by manifesto](figures/r1_share_by_manifesto.png) ·
[region support heat map](figures/r1_region_support_heatmap.png) ·
[agreement histograms](figures/r1_agreement_histograms.png) ·
[gradient-ascent convergence](figures/r1_convergence.png) ·
[ternary landscape](figures/r1_ternary_landscape.png) ·
[divided vs aligned](figures/r1_divided_vs_aligned.png)

**Existing results used**
- **X19** ([../sections/x19.md](../sections/x19.md)): the same comparison on the *election-day* forecast electorate
  (E\* = 0.469), and the optimiser comparison (PG / mirror descent / Frank–Wolfe / ADMM).
- **X15:** the "divided majority" threshold of 0.35.
- **XC:** agreement with the ORACLE manifesto at round 0.

## Q2. How do opinions change with no media and no workers?

**New: R2** ([r2.md](r2.md)). Two paired populations: "normal" (no zealots) and "zealots" (5%). They share the
same graph, starting preferences and susceptibilities.

- **Within a 5-round game, zealots barely matter.** Disagreement (the mean TV distance to the population mean)
  goes from 0.217 to 0.183 (normal) and 0.184 (zealots) by round 5 (`per_round.<pop>.disagreement`).
- **In the long run they split.**
  - The normal population keeps converging: ×0.990 per round, a half-life of about 73 rounds (`floor.normal`).
    It reaches 0.015 by round 200.
  - The zealot population levels off at a floor. Disagreement is 0.038 at round 200, and the Friedkin–Johnsen
    limit is 0.028 (`floor.zealots`).
  - Zealots leave more disagreement on 50 of 50 seeds at every recorded round (`tests`).
- **Drift alone helps a fixed manifesto** (`per_round.*.E_oracle`). The ORACLE manifesto's share rises from 0.374
  at round 0 to:
  - 0.487 (normal) and 0.481 (zealots) by round 5;
  - 0.865 and 0.854 by round 200.

  Zealots cost it 0.006 E at round 5.

Figures: [regional sector means](figures/r2_region_sectors.png) ·
[disagreement / spread / dispersion](figures/r2_disagreement.png) ·
[voter snapshots](figures/r2_pca.png) ·
[manifesto share over rounds](figures/r2_manifesto_share.png)

**Existing results used**
- **XC** (in [../report.md](../report.md)): 5-round trajectories with a campaign, for comparison. Voters won over
  rise from 34% to 51% with no campaign.
- **X1:** how fast opinions mix depends on the network's spectral gap (Spearman 0.997).
- **X2:** stubborn-agent (Friedkin–Johnsen) theory behind the zealot floor.

## Q3. How do worker cascades spread and move preferences?

**Existing results answer this; no new run.**
- **X3** ([../sections/x3.md](../sections/x3.md)):
  - a cascade takes off when the weighted spectral radius λ_max(A D_q) passes about 0.94;
  - reach saturates in the number of workers;
  - 30–85% of converts lie outside the target region;
  - the 3-attempt cap binds for 41.5% of targeted voters above threshold;
  - converts drift back from a vote chance of 0.971 to 0.88 if their friends disagree with the manifesto.
- **X18:** one worker gives 62% of the plateau value.
- **X2:** a worker's pull just from being pinned at b is only about 1% of its value.
- **X23:** the effect of rule variants (attempt cap, minimum degree, conversion rule).
- **XC:** the workers-only campaign moves mean agreement from 0.783 to 0.866.

## Q4. How should a fixed number of workers be assigned across regions?

**New: R3** ([r3.md](r3.md)). K ∈ {4, 10} workers, all bought in round 1, no media. Three region-size scenarios:
balanced, default and skewed.

- **Every assignment is worth about +0.10 to +0.13 in E** over no campaign (`arms.*.delta_vs_none`). Every
  assignment wins 42–43 of 50 games.
- **Where the workers go is second-order.**
  - The best and worst assignments differ by 0.007–0.015 at K = 4 and 0.002–0.003 at K = 10 (`range_K4`,
    `range_K10`).
  - No pair of assignments differs in wins (McNemar, Holm p = 1).
- **The one mistake: a small team all in the hostile region.** At K = 4, all→antagonistic is 0.007–0.015 below
  all→aligned (Holm p ≤ 0.01, `tests`). At K = 10 no pair of arms differs.
- **Why:** converts spill out of the target region. The spill-over is 45–56% for all→aligned and 92–95% for
  all→antagonistic, and hostile voters rarely convert (`converts_outside_target_share`, `converts_by_role`).
- **Advice:** buy every worker in round 1; timing is worth far more than placement (X4). With about 4 workers,
  target the aligned region or split between aligned and swing. With 10 or more, spread them.

Figures: [ΔE by assignment](figures/r3_delta_by_arm.png) ·
[where the support lands](figures/r3_region_support.png) ·
[role × K heat map](figures/r3_heatmap.png) ·
[largest region aligned vs antagonistic](figures/r3_largest_split.png)

**Existing results used**
- **X4:** at equal spend, timing (round 1 vs round 5) changes E by 0.12, while placement changes it by ≤ 0.010.
- **X3:** a manifesto aligned with the target region converts far better than an antagonistic one.
- **X18:** the value of each additional worker.

## Q5. Media vs workers: where to invest?

**Existing results answer this; no new run.**
- **X0:** "split" (half workers, half media) is the best fixed pattern; media-only is almost never best.
- **X11:** at matched spend (about ₫50), workers add 0.104 E and one airing adds 0.009.
  - The two channels are close to additive.
  - Buying workers before media beats the reverse by 0.041.
- **X13:** the best plans found spend about 89% on media, front-loaded, with about 3 workers in round 1.
  - A few workers saturate the cascade (X18), so after that, extra money buys more votes as media.
- **X7, X10, X1:** how the answer shifts with prices, with N and T, and with network structure.

---

## Reproduce

```bash
OMP_NUM_THREADS=4 python experiments/r1_manifesto_round0.py --seeds 100 --first-seed 12000 --device cuda:2
OMP_NUM_THREADS=4 python experiments/r2_drift_no_campaign.py --seeds 50 --first-seed 10000 --rounds 200 --device cuda:2
OMP_NUM_THREADS=4 python experiments/r3_worker_assignment.py --seeds 50 --first-seed 11000 --device cuda:0
# add --plot-only to any command to redraw its figures from the saved JSON
python -m pytest -q tests/unit/test_r1_manifesto_round0.py tests/unit/test_r2_drift.py tests/unit/test_r3_worker_assignment.py
```

Scripts: `experiments/r1_manifesto_round0.py`, `experiments/r2_drift_no_campaign.py`,
`experiments/r3_worker_assignment.py`.
