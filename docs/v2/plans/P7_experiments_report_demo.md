# P7 — Experiments, report and demo (Days 9–10)

**Depends on:** P4, P5, P6. No model changes in this phase (bug fixes only, logged).

## Experiments (`experiments/`, batched GPU, ≥ 200 paired seeds per cell, 95% CIs)
| ID | Question | Factors | Metrics |
|---|---|---|---|
| E1 Topology | Does community structure matter? | k̄ ∈ {8,16,32}; p_g range ∈ {[0.01,0.05],[0.01,0.2],[0.1,0.4]} | λ2, win rate, best mechanism |
| E2 Resilience | Do zealots block persuasion? Does W-MSR change it? | ζ ∈ {0, .05, .15, .30}; social_rule ∈ {degroot, wmsr F=1,2} | win rate, final mean S per region, (r,s) bounds |
| E3 Cascades | When do workers beat media? | τ_c, ρ around the spectral threshold | reach, win rate vs τ_c·λ_max/ρ |
| E4 Information | How much surveying is enough? | queries 0..10 (optimal vs random selection) | K_p(GuaranteedVictory) rate, win rate, money trade-off |
| E5 Budget | P(Victory | B) | B ∈ {50, 100, 200} | win rate, spend mix |
| E6 Strategies | Ranking | uniform, top-3, random, informed+planner, oracle | win rate, margin, efficiency |
| E7 Scale | Performance | N ∈ {1e4, 1e5, 1e6} | s/round, peak VRAM |

## Deliverables
`reports/v2/*.json|.png` generated from saved JSONL only; `reports/v2/final_report.md` (model,
lecture links, results, limitations); `docs/v2/demo_script.md` (a 5–10 minute live game with talking
points: survey → manifesto → cascades → logic verdicts).

## Exit criteria
All experiments are reproducible from a single command each; the demo has been rehearsed end-to-end.

## Handoff (2026-11-08)
- Owner asked for: emphasised lecture links; final report (markdown), demo script, slide deck and shareable web report page.
- **Experiments (sub-agent L)**:
  - Scripts `experiments/e{1,2,3,5,7}_*.py` + `p7_common.py` + `plot_p7.py`. Data `reports/v2/e*.json`, figures
    `reports/v2/figures/e*.png`, consolidated `reports/v2/p7_results.md`.
  - 200 paired seeds per cell at N = 10k, Wilson/t CIs, Wilcoxon/McNemar tests.
  - Theory verdicts:
    - E1: consensus speed follows the random-walk gap (Spearman 0.995).
    - E2: zealots block persuasion; W-MSR helps only within its F-total/F-local threat models, and costs
      convergence.
    - E3: cascade reach collapses on τ·λ_max(A_rr)/ρ with a turn near 1; workers beat media above the threshold.
    - E4/E6 consolidated.
    - E5: budget keeps paying; spending shifts to workers.
    - E7: 1M voters at 28 ms per round on GPU.
- **Final report** `reports/v2/final_report.md` and **demo script** `docs/v2/demo_script.md` (sub-agent M):
  - 96/98 sampled numbers verified exactly against JSON; the other 2 differ only by rounding.
  - Demo: primary seed 11, a West-focused comeback from 31% to a 54.1% win, where "knew a win exists" holds
    from R1 and "knew a winning plan" only from R3. Backup seed 6006, a loss that shows truth ≠ knowledge.
- **Slide deck** (16 slides, lead): https://claude.ai/artifact/557DppcHrdX274fnH49rFr
- **Shareable results page** (lead): https://claude.ai/artifact/YDeg5sUR3aXxRjKN8D17Ci (both private until shared from
  the page's Share menu).
- **Lead fix:** the epistemic timeline's entropy was labelled bits but computed in nats; the UI now converts
  (bits = nats / ln 2). Report and demo script updated; production build rebuilt and live servers restarted.
- **Tests:** Python 233 passed; web 43 passed; e2e 5/5 (P6).
