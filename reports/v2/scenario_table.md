# P5 scenario table (60 seeds, N = 10000, 16 worlds/round)

Frequency over games of each property at decision point t (start of round t after the poll and the player's round-t queries). Knowledge columns use only the player's own information over sampled consistent worlds × a finite plan set; 'Truth' evaluates the same plans in the true world.

## bayes — win rate 78.3% (95% CI 66–87%), mean queries 2.0

| round | Possible (∃π∃w) | Guaranteed (∃π∀w) | Knows-that (∀w∃π) | Inevitable defeat (∀π∀w¬) | Truth: win possible | σ entropy (nats) | best-plan win frac |
|---|---|---|---|---|---|---|---|
| 1 | 97% | 70% | 72% | 3% | 82% | 0.00 | 0.80 |
| 2 | 95% | 70% | 72% | 5% | 82% | 0.00 | 0.80 |
| 3 | 93% | 72% | 72% | 7% | 85% | 0.00 | 0.82 |
| 4 | 95% | 73% | 75% | 5% | 85% | 0.00 | 0.84 |
| 5 | 93% | 72% | 72% | 7% | 82% | 0.00 | 0.82 |

Truth vs knowledge (all rounds pooled):

| | knows Possible | ¬Possible | knows Guaranteed | ¬Guaranteed |
|---|---|---|---|---|
| truth: win possible | 249 | 0 | 214 | 35 |
| truth: no win possible | 35 | 16 | 0 | 51 |

Knowledge vs outcome: P(win | guaranteed at round 1) = 100% (n=42); P(win | guaranteed at some round) = 100% (n=44); P(win | never guaranteed) = 19% (n=16); P(win | possible at round T) = 84% (n=56); P(win | not possible at round T) = 0% (n=4)

Temporal (per game trace): F Lead: 78%; G Lead: 0%; F Victory: 78%; G Viable: 78%; F Guaranteed: 78%; G(Viable) and not F(Victory): 0%; Lead U Victory: 0%; G(BudgetLeft -> Viable): 78%

STL mean robustness: rho F(margin>0) = +0.077; rho G(margin>0) = -0.131; rho G[<=2](margin>0) from round 3 = -0.049

Poll-consistency acceptance rate 100% (fallback rounds: 0); timeline 4.7s mean, 5.9s max.

## top3 — win rate 70.0% (95% CI 57–80%), mean queries 0.0

| round | Possible (∃π∃w) | Guaranteed (∃π∀w) | Knows-that (∀w∃π) | Inevitable defeat (∀π∀w¬) | Truth: win possible | σ entropy (nats) | best-plan win frac |
|---|---|---|---|---|---|---|---|
| 1 | 82% | 63% | 67% | 18% | 72% | 1.69 | 0.71 |
| 2 | 82% | 65% | 68% | 18% | 72% | 1.69 | 0.72 |
| 3 | 83% | 67% | 68% | 17% | 72% | 1.69 | 0.73 |
| 4 | 85% | 68% | 68% | 15% | 72% | 1.69 | 0.74 |
| 5 | 82% | 68% | 68% | 18% | 72% | 1.69 | 0.72 |

Truth vs knowledge (all rounds pooled):

| | knows Possible | ¬Possible | knows Guaranteed | ¬Guaranteed |
|---|---|---|---|---|
| truth: win possible | 215 | 0 | 199 | 16 |
| truth: no win possible | 33 | 52 | 0 | 85 |

Knowledge vs outcome: P(win | guaranteed at round 1) = 100% (n=38); P(win | guaranteed at some round) = 100% (n=41); P(win | never guaranteed) = 5% (n=19); P(win | possible at round T) = 86% (n=49); P(win | not possible at round T) = 0% (n=11)

Temporal (per game trace): F Lead: 70%; G Lead: 0%; F Victory: 70%; G Viable: 70%; F Guaranteed: 70%; G(Viable) and not F(Victory): 0%; Lead U Victory: 0%; G(BudgetLeft -> Viable): 70%

STL mean robustness: rho F(margin>0) = +0.044; rho G(margin>0) = -0.156; rho G[<=2](margin>0) from round 3 = -0.074

Poll-consistency acceptance rate 97% (fallback rounds: 1); timeline 5.3s mean, 8.8s max.
