# Demo: the election you can't win (seed 250288)

In this demo the player can go through every phase and every round, and the campaign visibly lifts support, but the election is lost whatever the player does. The best campaign found by any full-information attack still ends about 2,900 votes short of a majority.

## 1. What to type on the setup screen

| Field | Value |
|---|---|
| Voters | **100000** |
| Seed | **250288** |
| Rounds | **5** |
| Treasury (budget) | **850.00** |

There is one player (`p0`). These four values rebuild exactly the same starting electorate on any machine, because the electorate is drawn on the CPU in a fixed way.

What the player sees at the start (public census, also in `initial_observation.json`):

| Region | Voters | Mean degree | Media price per round |
|---|---|---|---|
| North | 21,940 | 22.0 | ₫43.88 |
| **South** | **33,676** (largest) | 38.1 | ₫67.35 |
| East | 17,026 | 21.3 | ₫34.05 |
| West | 27,358 | 31.9 | ₫54.72 |

Workers cost ₫20 each and survey questions cost ₫10 each.

## 2. Suggested "best effort" play

This is the plan from `script.json`. It was found by the strongest planner we have (X13 plan search: cross-entropy method plus hill climb, run with full knowledge of the true electorate). `replay_lose.py` replays it exactly.

**Survey phase (2 questions, ₫20).** Each question is the one with the highest expected information gain about which archetype lives in which region:

1. "Agriculture over Welfare, region **West**". The answer is about 98% (the West is the Agrarian heartland).
2. "Environment over Welfare, region **East**". The answer is about 100% (the East is Green urban). After these two answers the player effectively knows which archetype lives in each region.

**Manifesto (whole percentages):**

| Education | Welfare | Agriculture | Industry | Environment |
|---|---|---|---|---|
| 18 | 28 | 21 | **14** | 19 |

The manifesto is a deliberate coalition of North (Welfare), West (Agriculture) and East (Environment). Industry, the issue the South cares about, gets very little weight.

**Campaign (treasury ₫830 after the questions):**

| Round | Workers (N, S, E, W) | Media in | Cost | Poll shown to player | True expected share after round |
|---|---|---|---|---|---|
| start | – | – | – | 24.0% ± 4.2 | 24.0% |
| 1 | 12, 0, 0, 7 (19 workers) | North, East, West | ₫512.65 | 28.3% | 29.0% |
| 2 | 0 | North, East, West | ₫132.65 | 31.5% | 32.8% |
| 3 | 0 | East, West | ₫88.77 | 35.3% | 35.9% |
| 4 | 0 | East, West | ₫88.77 | 42.0% | 39.9% |
| 5 | 0 | none | ₫0.00 | 46.3% (pre-election poll) | 46.3% |

Each poll row is the poll the player sees at the start of the next round, so the "round 5" row is the pre-election poll. Polls carry about ±4–5 points of sampling error. ₫7.16 is left unspent. The workers bought in round 1 start a cascade of converts: 69 voters, then 269, 1,253, 5,770 and 17,978 in round 5. That growth is the visible "the campaign works" moment of the demo.

**Final result (on GPU, cuda:2 and cuda:0 identical): 46,375 votes out of 100,000 (46.4%), DEFEAT, 3,626 votes short of a majority.**

On a CPU-only engine the dice differ slightly: 44,870 votes (44.9%), also a defeat, 5,131 votes short.

For comparison, with **no campaign** (same manifesto, nothing bought) the player gets 30,794 votes (30.8%). The campaign adds about 15 points, but the gap to 50% stays.

## 3. Why no strategy can win here

The hidden world, shown in the debrief:

| Region | Share of voters | What they want (archetype) | How they respond (personality) | Support at the end of best effort |
|---|---|---|---|---|
| North | 22% | Welfare-reliant | Socially driven | 80% |
| **South** | **34%** | **Industrial belt** | **Independent** | **8%** |
| East | 17% | Green urban | Receptive to canvassing | 37% |
| West | 27% | Agrarian heartland | Media-savvy | 72% |

In plain words:

1. **The biggest region is unreachable.** One voter in three lives in the South. Southerners want Industry above all, and their personality is *Independent*, so they are hardly moved by neighbours, media or canvassers. The South also has the most zealots (1,679 voters who never change their mind). Any manifesto that pleases the South loses the other three regions, and a manifesto that ignores the South gets only about 8% of it.
2. **The two largest regions want very different budgets.** The South (Industry) and the West (Agriculture) differ by 0.35 in total variation, which is the "divided world" pattern where our experiments found most losses.
3. **So the other three regions would have to deliver almost everything.** Without the South a majority needs 50,000 of the remaining 66,324 voters, about 75% support across North, East and West. North and West can be pushed to 72–80%. The East wants Environment and gets only about 37%, because its archetype pulls away from the Welfare/Agriculture centre of the coalition.
4. **Weak ties between regions.** Only 12% of all friendships cross a region border. The South is dense inside (mean degree 38) but has few links out: about 12,000 edges to the North, 18,000 to the East and 26,000 to the West, against 613,000 edges inside the South. Support built in one region barely spills over, and worker cascades stay inside their own region.

**What the debrief and epistemic timeline show.** The debrief reveals the table above and the ORACLE benchmark. The epistemic timeline marks every round 1–5 as **"inevitable defeat"**: no plan wins in any world the player considers possible, and no winning plan exists in the true world either (`truth_victory_possible = false` every round). Its headline reads: *"From round 1 no plan won in any world you considered possible. You lost with 46.4% of the vote."*

## 4. Evidence summary

Every attack below uses **full information** (the true electorate, true preferences, the network). That is an upper bound on what any human could do. Details are in `evidence.json`.

- **Manifestos tried:** 124 in total. They include the ORACLE (election-day) manifesto, the round-0 optimum, every single-region / 2-region / 3-region blend (on both start-of-game and election-day preferences), mean, median, TV geometric median, top-3, uniform, a multi-start optimum, 60 Dirichlet(1) and 20 Dirichlet(5) random manifestos, X12 joint manifesto + plan optimisation (from the ORACLE manifesto and from the best manifesto), a joint alternation with the X13 plan, and a manifesto hill climb run in the true simulator.
- **Plans per manifesto:** the ORACLE greedy planner and 20 fixed patterns: all workers in round 1 (proportional, best region, each single region), media in every affordable region every round, split, front-loaded, region focus, strong-arm and people-pleaser. The top 9 manifestos also got the X13 plan search (CEM + hill climb, ₫850).
- **Luck:** the best plans were re-run on 24 alternative random streams, and the **maximum** over streams was taken. On top of that, the plan was hill-climbed directly on the game's own dice (canonical streams) and checked on the CPU engine.

| Attack (best of its kind) | Best expected share |
|---|---|
| ORACLE manifesto + ORACLE planner (canonical dice) | 42.3% |
| Best fixed pattern, any manifesto (canonical dice) | 41.5% |
| Best of 124 manifestos × 21 planners (canonical dice) | 43.0% |
| X13 plan search on the ORACLE manifesto, luckiest of 24 streams | 44.5% |
| X12 joint manifesto+plan + X13, luckiest of 24 streams | 46.9% |
| Joint alternation (X13 plan → manifesto re-optimised → re-search), luckiest stream | **47.07%** |
| Plan tuned to the game's own dice (₫850, best manifesto) | 46.7% |
| Same plans on the CPU engine, luckiest of 25 streams | 46.8% |

**Best share found by any attack on any stream: 47.07%, which is about 2,930 votes short of a majority.** The best official vote count in any of those runs was 47.1%.

**Honesty notes.** "Can never win" is backed by strong upper bounds, not by a proof over every possible plan. The search space is huge and was searched, not enumerated. The owner's target was a best-attack share of at most 45% (at least 5,000 votes short). No world met it. Out of 5,893 distinct seeds screened at full size (N = 100,000) and 10 finalists stress-tested, every hard world rises to 47–51% under full-information joint manifesto and plan optimisation. Seed 250288 has the largest margin we found. Even the scripted best effort, which was itself found with hidden information, only reaches 45–46%. A human without that information should do no better.

## 5. Replaying the demo

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate mas
python Demo/lose/replay_lose.py --device cuda:2          # best effort + no campaign, asserts both lose
python Demo/lose/replay_lose.py --device cpu --epistemic  # also fetches the epistemic timeline
```
