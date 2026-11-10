# P8 — Party workers v3: permanent seeded workers + round-by-round Independent Cascade

Source: owner rewrite of `docs/v2/sources/graph_construction.md` §5.2, Q&A of 2026-11-09, approved 2026-11-10.
Decisions R21–R28 are in `decisions_v2.md`. The binding web contract is in `web_api_v2.md` (P8 section, below).

## Why
Before P8, workers were canvassing capacity: q random doorsteps per worker in each round, plus a 3-hop SIR within
the same round. The owner's intended mechanism makes a worker a real voter node, brainwashed to the manifesto,
permanent and fixed to its region. Influence then spreads as an Independent Cascade, **one hop per campaign
round**.

With the literal rule (chance = γ, spreaders try every one of their ~30 friends), one ₫4 worker converted
63–68% of 10k voters by round 5 across 3 seeds, so the manifesto stopped mattering. R25 therefore multiplies
the chance by P_j(vote for b): with 5 workers the cascade then reaches about 1.1k–2.9k voters, depending on how
well b suits the region.

## Rules (R21–R28)
| # | Rule |
|---|---|
| R21 | No free workers. Every worker is bought (one-time fee) and is permanent. |
| R22 | Buying k workers in region r picks k distinct voters uniformly at random among region-r voters with degree ≥ `min_degree` = 3, excluding existing workers and zealots. The draw uses the seeded `workers` stream. A worker never changes region, and there is no per-round reassignment. If there are fewer eligible voters than requested, the purchase is rejected (`INVALID_ALLOCATION`). |
| R23 | Workers are pinned at b for the whole game: friends and media never move them. They still contribute b to their friends' DeGroot/W-MSR aggregate every round. |
| R24 | One hop per round. The spreaders of round t are the workers bought in round t together with the voters converted in round t−1. Each spreader tries every friend once, then goes quiet. A bought worker spreads only in its purchase round. Voters converted in round T never spread. |
| R25 | Conversion chance for friend j is γ_j · P_j, where P_j = σ(β_v(S_j − z)) is computed on v at the start of the round (before this round's hires are applied). A converted voter gets v_j = b. |
| R26 | Converts are ordinary voters. In the round they are converted they take part in the social step like anyone else, so they drift right away. They spread only in the next round. |
| R27 | Each voter can receive at most `attempt_cap` = 3 conversion attempts per game, failures included. Workers and zealots are never targets: attempts on them are not counted and they are never converted. |
| R28 | Media is unchanged (the existing `(1−a−c)v + a·Agg + c·b` already matches §5.1). |

## Round order (player p, round t)
1. **Hires:**
   - cost = `worker_cost × Σ hires_by_region` + media cost, which must be ≤ treasury.
   - Pick the new worker nodes (R22). Set `v = b` and add them to `pinned` and to this round's spreaders.
2. **Cascade hop:**
   - `count_j` = number of directed (spreader → j) edges, for j not a worker and not a zealot.
   - `eff_j = min(count_j, cap − attempts_j)` and `attempts_j += eff_j`.
   - j converts if `u_j < 1 − (1 − γ_j P_j)^{eff_j}`, with one uniform per voter from the `cascade` stream (fixed RNG consumption).
   - Converts get `v_j = b` and become next round's spreaders.
3. **Media intensity:** as before, from the `media` stream.
4. **Social step:** `step_preferences`. Zealots are frozen at their own v; workers are frozen at b.
5. **Polls and free stats:** as before.

Multi-player note (not built): if several players convert the same voter in the same round, one of them is chosen
uniformly from the `cascade` stream.

## Config (`WorkerConfig`)
- Removed: `free_workers, contacts_fraction, contact_strength, transmission, recovery, hops, word_of_mouth_strength`.
- `worker_cost: Decimal` (provisional value until the price check), `min_degree: int = 3`, `attempt_cap: int = 3`.

## Price check (owner checkpoint)
Run a short sweep of the worker price on about 50 seeds with the existing scripted strategies, so that the game is
not trivial. The numbers and a recommended price go to the owner before the value is frozen. This is not the
P7 experiment re-run, which happens later at the owner's call. E3 (SIR threshold) needs a redesign for the IC
cascade (R0 ≈ mean over spreader edges of γP) at that re-run.

## Tests (exit criteria)
- **pick:** right region, degree ≥ 3, distinct, no zealots, no repeats across rounds, seeded, rejects when too few
  eligible voters.
- **One hop per round:** on a path graph, a worker bought in round 1 converts its neighbour in round 1 (γ=P=1),
  the neighbour's neighbour in round 2, and so on. A worker never re-spreads.
- **Cap:** attempts counted with failures; never more than 3; workers and zealots are never targets.
- **Chance:** γ=0 or P=0 never converts; a statistical check of γP on a star graph.
- **Pinned:** workers stay exactly at b after social and media steps; zealots unchanged.
- **Determinism:** same seed gives the same game; CPU/GPU parity on cuda:2 if one is free.
- **Observation privacy:** no node ids, γ, attempts or per-voter data.
- **Suites:** full Python suite, vitest, `npm run build`, Playwright e2e against the live servers.

## Handoff (2026-11-10)
- **Implemented:**
  - Engine: `dynamics/campaign.py` (`Allocation(hires_by_region, media_regions)`, `WorkerState`, `pick_workers`, `conversion_chance`, `ic_hop`, `campaign_round`), `step_preferences(pinned=)`, and `simulate.resolve_campaign_round` shared by the engine and `simulate_world`.
  - Ported: the game, the API (schema 2.2), strategies/planner/worlds, and logic.
  - Experiments ported to import and run but **not re-run**. E3 and `cascade_threshold` are minimal IC probes with a TODO.
  - Web: buy-workers stepper, permanent team strip, rewritten "How it works", look-back fields, mock API.
- **Interpretations:**
  - A convert is an ordinary target again, so it can be re-converted within the cap of 3 attempts (under 1% effect).
  - P is computed on v at round start.
  - Planner worlds scale hires by n/N with stochastic rounding.
- **Tests at ₫50:** Python 263 passed (GPU tests on cuda:2); vitest 59; `tsc` and `next build` clean; Playwright 5/5 against the live servers (round ≈ 134 ms mean, epistemic 3.6 s). BayesPlanner decision 0.19–0.30 s at N = 10k.
- **Real bug found by the fee change:** when the per-round spending cap fell below one fee, the planner's candidate generator returned only "do nothing". It is fixed, and a test now covers fees of ₫1–₫600.
- **Price check:** `experiments/balance.py --seeds 50 --set workers.worker_cost=X`, N = 10k, B = 500, CPU, zealot sweep off. Win rates per pattern:

| worker price | uniform/none | oracle/planner | front_workers | pay_workers | pay_media | pay_split | best pattern per seed (pay_workers/front/media/split) | gate 1–4b |
|---|---|---|---|---|---|---|---|---|
| ₫10 | 4% | 90% | 82% | 82% | 64% | **90%** | 0/0/0/50 | 4a, 4b fail |
| ₫25 | 4% | 90% | 82% | 80% | 64% | **86%** | 2/6/0/42 | 4a, 4b fail |
| ₫50 | 4% | 92% | 78% | 76% | 64% | **82%** | 1/16/0/33 | 4a, 4b fail |
| ₫75 | 4% | 86% | 74% | 70% | 64% | 62% | 8/41/0/1 | 4b fails (media never best) |
| **₫100** | 4% | 86% | 72% | 70% | 64% | 58% | 14/33/2/1 | **all pass** |
| ₫150 | 4% | 66% | 66% | 44% | 64% | 62% | 0/38/11/1 | 2 fails (oracle < 85%) |

  - Below ₫75 a few cheap workers saturate the cascade, so "split everything" always wins.
  - Above ₫100 the oracle can no longer reach 85%.
  - Recommendation was ₫100. **Owner chose ₫50 (R29)**, accepting that at ₫50 "split workers+media" wins 82% of
    games and media is never the single best pattern (gate 4a/4b). This is to be revisited at the full re-run.
