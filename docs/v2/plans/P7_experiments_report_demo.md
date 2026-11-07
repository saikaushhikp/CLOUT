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
