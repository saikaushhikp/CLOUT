# Dumbocracy demo: one election you win, one you can't

Two fixed games for live demos. Both use the owner's settings: **100,000 voters, a treasury of ₫850, 5 rounds**.

Each game is fully defined by four values typed into the setup screen. The same values rebuild exactly the same
starting electorate on any machine.

| Demo | Seed | Voters | Rounds | Treasury | Outcome with the scripted play | Walkthrough |
|---|---|---|---|---|---|---|
| **Win** | **3685** | 100000 | 5 | 850.00 | **73,821 votes (73.8%): a win. Without campaigning: 45,439 votes (45.4%), a loss** | [win/walkthrough.md](win/walkthrough.md) |
| **Can't win** | **250288** | 100000 | 5 | 850.00 | **46,375 votes (46.4%): defeat, 3,626 short of a majority** | [lose/walkthrough.md](lose/walkthrough.md) |

Both outcomes were replayed through the real game API (the same code path the web game uses), and re-checked
independently with the replay scripts below.

---

## How to run a demo

1. Start the game: `scripts/run_game.sh` (engine on :61008, web game on :61007).
2. On the setup screen enter the seed, **100000** voters, **5** rounds and treasury **850.00**.
3. Follow the walkthrough for that demo. It lists the survey questions to ask, the manifesto to enter (whole
   percentages) and what to buy each round, with the expected poll after every round.

**Replaying without the browser** (plays the script through the API and checks the result):

```bash
conda activate mas
python Demo/win/replay_win.py   --device cuda:0    # asserts the no-campaign play loses AND the campaign wins
python Demo/lose/replay_lose.py --device cuda:2    # asserts every play loses (--play best|none|both, --epistemic)
```

**GPU vs CPU engine.**
- The starting electorate is identical everywhere.
- The random events during play (survey samples, worker picks, cascades, polls, the vote) are drawn on the engine's
  device. The walkthroughs' exact numbers are for a **CUDA (GPU) engine**; any two GPUs give identical results.
- On a CPU engine the numbers differ slightly, but the story is the same:

  | Demo | GPU engine | CPU engine |
  |---|---|---|
  | Win: campaign | 73.8%, win | 73.6%, win |
  | Win: no campaign | 45.4%, defeat | 45.3%, defeat |
  | Can't win | 46.4%, defeat | 44.9%, defeat |

---

## Demo 1: the election you win only by campaigning (seed 3685)

**Headline:** without campaigning you lose with **45.4%**; with this campaign you win with **73.8%**.

**The play** (full detail in [win/walkthrough.md](win/walkthrough.md); the action list is in
[win/script.json](win/script.json)):

| Step | What the player does | Expected vote share after, with the campaign | Expected vote share after, with no campaign |
|---|---|---|---|
| Survey | Ask 2 questions: North, Industry vs Environment; South, Agriculture vs Industry (₫20). The first alone reveals every region's archetype; the second confirms the South | — | — |
| Manifesto | Education 19 · Welfare 28 · Agriculture 14 · Industry 23 · Environment 16 | 0.34 at commit | 0.34 |
| Round 1 | 10 workers in the West; media in all four regions (₫400) | 0.41 | 0.36 |
| Round 2 | Media in all four regions (₫200) | 0.47 | 0.38 |
| Round 3 | Media in North, South, West (₫168) | 0.54 | 0.41 |
| Round 4 | Media in the South (₫48) | 0.66 | 0.43 |
| Round 5 | Nothing (₫14 left over) | **0.738, win** | **0.455, defeat** |

**Talking points:**
- **No manifesto can win this election on its own.** Opinions drift towards the party over the 5 rounds, but even the
  best possible manifesto, chosen with full knowledge of every voter, ends at 45.5% without a campaign. Every
  manifesto loses here unless the party campaigns.
- **The campaign turns a loss into a clear win.**
  - The 10 West workers start a cascade: the West holds about 40% of voters and is receptive to canvassing. The
    cascade converts about 50,000 voters by rounds 4–5.
  - Heavy early media does the rest.

  This is exactly what the experiments recommend
  ([../reports/v3/experimentation/experiments.md](../reports/v3/experimentation/experiments.md)).
- **How you campaign matters too** (same world):

  | Play | Vote share | Result |
  |---|---|---|
  | Even 20% manifesto, no spending | 37.2% | defeat |
  | Even 20% manifesto with this campaign | 62.9% | win |
  | Everything spent in round 5 | 50.8% | barely wins |
  | The planned campaign | **73.8%** | win |

- **Both outcomes are robust** across 10 different random draws:
  - the campaign wins every time (73.1–74.3%);
  - no campaign loses every time (45.2–45.6%).
- **Why this seed:** it was chosen from 4,060 screened seeds. Among worlds where no manifesto can win without
  campaigning, it gave the largest campaign win.
  - Runners-up: 2233 (72.9%), 3336 (72.7%), 457 (72.3%).
  - About 74% looks like the ceiling for this kind of world.
  - The search scripts are in [win/search/](win/search/).

## Demo 2: the election you can't win (seed 250288)

**The play.** The best-effort plan from [lose/walkthrough.md](lose/walkthrough.md) and
[lose/script.json](lose/script.json):
- 2 survey questions;
- manifesto Education 18 · Welfare 28 · Agriculture 21 · Industry 14 · Environment 19;
- the strongest campaign plan found.

The player can go through every phase and round. Support visibly rises from **24% to 46%**, but it stops short:
**46,375 votes, 3,626 short of a majority.** With no campaign the party gets 31%.

**Why it can't be won** (in plain words):
- **The largest region is out of reach.** The South holds 34% of voters. It wants an industrial budget, its voters
  are "independent" (barely moved by friends, media or canvassers), and it has the most zealots. Its support ends
  near 8% whatever the player does.
- **The two biggest regions want very different things.** South (Industry) and West (Agriculture) pull apart, so no
  single manifesto can please both.
- **The rest can't make up the gap.** A majority would need about 75% support across the other three regions combined.
  North reaches about 80% and West about 72%, but East only about 37%.
- **Regions barely talk to each other.** Only 12% of friendships cross a region border, so success in one region
  hardly spreads to the South.
- **The debrief agrees.** The game's epistemic timeline reports **"inevitable defeat"** in every round.

**How strong is "can't win"?** This is an upper bound from very strong searches, not a proof over every possible
plan. Every attack below had **full information**, far more than any human player:

| Attack | Best vote share found |
|---|---|
| Benchmark manifesto + greedy planner | 42.3% |
| 124 different manifestos × 21 spending plans | 43.0% |
| Computer plan search on the benchmark manifesto | 44.5% |
| Manifesto and plan optimised together, then plan search | 46.9% |
| Repeated manifesto ↔ plan optimisation (strongest) | **47.07%** (≈ 2,930 votes short) |
| Plans tuned on this game's own random draws | 46.7% |
| Best plans on a CPU engine, 25 random draws | 46.8% |

No attack reached 50% on any of the 24 random draws tried.

**Honest caveat.** The target was at least 5,000 votes short (best attack ≤ 45%). **No world met it.**
- 5,893 seeds were screened. In every hard world, optimising the manifesto and the plan together pulls the best
  share up to 47–51%.
- Seed 250288 has the largest margin found, about 2,900 votes.
- Runners-up: 349546 (47.7%), 140236 (48.2%), 319446 (48.6%).
- Two shortlisted seeds turned out to be winnable after all.

---

## Files

| Folder | Contents |
|---|---|
| `win/` | `game_settings.json` (setup values), `walkthrough.md` (presenter script), `script.json` (action list, campaign and no campaign), `initial_observation.json` (what the screen shows at the start), `verification.json` (API replays on two GPUs and CPU for the campaign, no campaign, and best-manifesto-no-campaign plays; robustness; comparison plays), `screening.jsonl` / `screening_summary.json` (how the seed was chosen), `search/` (the seed-search scripts), `replay_win.py`, `test_replay.py`, `share_over_rounds.png` (campaign vs no campaign) |
| `lose/` | Same layout, plus `evidence.json` (every attack's upper bound and the world's hidden features), `replay_lose.py`, `logs/` (stress-test logs) |

**Not in the repo:** the seed-search scripts for the can't-win demo are in a temporary working folder, not here
(its method is described in `lose/screening_summary.json`). The win demo's scripts are in `win/search/`.
