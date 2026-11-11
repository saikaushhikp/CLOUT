# Finding a good strategy: methods to consider

**Status: proposal.**
- Nothing in this file has been run yet, except the reference results quoted from earlier experiments.
- It lists the ways we discussed to turn the experimental insights ([experiments.md](experiments.md)) into a
  strategy that a real player could follow, and how each should be tested.

**Terms used here.**
- **Vote share** is the expected fraction of the 10,000 voters who vote for the party. A difference of 0.01 is about
  100 votes, and above 0.5 a win is expected.
- **Manifesto** is the party's promised budget split over five sectors. It is chosen once and never changed.
- **Full information** means seeing everything that is hidden from a real player: every voter's preference, the
  friendship network, personalities and zealots.

---

## 1. What a strategy has to decide

A complete strategy answers three questions:

1. **Surveys:** how many questions to ask before committing, and which ones. Each question costs ₫10 and reveals
   what a sample of one region prefers.
2. **Manifesto:** which budget split to promise. It is fixed for the whole game.
3. **Spending, round by round:** how many party workers to buy and in which regions, and which regions get a media
   airing. The total is limited to the ₫500 treasury.

A strategy may only use what a real player can see:
- the public rules and prices;
- region sizes and a census of how many friendships exist inside and between regions;
- the list of six possible region archetypes;
- a free national statistic (how often each sector is among voters' top three);
- its own survey answers;
- a 400-voter poll each round.

## 2. Where we stand: reference levels

All of these were played on the same 100 worlds in the strategy tournament (every player type on identical
electorates, 5 rounds, ₫500):

| Player | What it knows | Vote share | Win rate |
|---|---|---|---|
| **Best plan found by search** | Everything. A computer search over complete 5-round spending plans | **0.672** | 0.99 |
| Smart player, ADMM manifesto | What a player sees | 0.649 | 0.99 |
| **Smart player** (Bayesian planner) | What a player sees: about 2 survey questions, then planning by simulating imagined electorates | **0.639** | 0.98 |
| Full-information benchmark | Everything, but spends with a greedy one-step rule | 0.628 | 0.95 |
| Top-3 statistic, even spending | Only the free national statistic | 0.602 | 0.85 |
| "Blend my hunches" manifesto + smart spending | Guessed archetypes, blended by region size | 0.597 | 0.84 |
| People Pleaser | Serves the 1–2 largest regions that form a majority | 0.560 | 0.73 |
| No campaign (even 20% manifesto, no spending) | Nothing | 0.389 | 0.09 |
| Strong-Arm (fixed or random manifesto, heavy spending) | Nothing | 0.088–0.157 | 0.00–0.03 |

**What the gaps mean:**
- The smart player is **0.033 below the best plan found** (about 330 votes per 10,000). That is the room any new
  strategy can win back.
- It already beats the full-information benchmark, which shows that *how you plan* matters more than seeing
  everything.

## 3. What the experiments tell a strategy

From [experiments.md](experiments.md):
- **Manifesto:**
  - Design it for the electorate expected on election day, not for today.
  - Make it a compromise leaning towards the largest region.
  - When the two largest regions want different things, anchor on the largest and add the closest small region.
    Never blend the two large ones.
- **Surveys:** about 2 well-chosen questions identify every region's archetype. Asking more costs money that the
  campaign needs: 12 questions lose about 0.013 vote share against 2.
- **Workers:**
  - Buy them in round 1; a worker bought in round 5 is almost worthless.
  - About 3–5 capture most of their value.
  - Put a small team where the manifesto is already liked, never in a hostile region.
- **Media:** after the first few workers, early media buys the most votes per ₫. The best plans found spend about
  89% on media in rounds 1–2.
- **Conditions change the answer:**
  - In sparse friendship networks, media beats workers.
  - In very large electorates with short games, workers barely matter.
  - In small electorates or long games, workers dominate.
- **Luck:** the electorate a player is dealt explains over 97% of the variation in outcomes. Strategies must
  therefore be compared on the *same* worlds, and small differences only show up in vote share, not in win rate.

---

## 4. Manifesto methods to compare

A strategy is only as good as its manifesto, so the first job is to rank the realistic ways of choosing one. The
methods below are grouped by how much the player needs to know.

| Group | Method | How it is built | Information needed | Already tested? |
|---|---|---|---|---|
| **A. No surveys** | Top-3 statistic | Budget shares proportional to how often each sector is in voters' top three nationally | Free statistic | Yes: vote share 0.602 in the tournament |
| | Even split | 20% to every sector | None | Yes (floor only) |
| **B. Rules on guessed archetypes** (after about 2 questions) | All-region blend ("blend my hunches") | Average of the four guessed archetype budgets, weighted by region size. This is the game's own UI button | Region sizes + archetype guesses | Yes: 0.597 |
| | Majority-coalition blend | Average of the 2–3 regions that together hold over half the voters | Same | Only with full information |
| | Largest-region anchor | Largest region's archetype blended with the closest smaller region's | Same | Only with full information (best rule in divided worlds) |
| | Compromise ("geometric median") | The budget closest overall to the four guessed archetypes | Same | Only with full information (best single rule) |
| **C. Model-based** | Gradient ascent on imagined electorates | Build 8 electorates consistent with what is known, forecast 5 rounds of drift, then repeatedly nudge the budget towards more votes | Everything a player sees | Yes: the smart player's default, 0.639 |
| | ADMM variant | The same, solved region by region with a smoother objective. It gives broader manifestos | Same | Yes: 0.649 |
| | Campaign-aware, on imagined electorates | Alternately optimise the manifesto for a planned campaign and re-plan the campaign | Same | **No.** Done only with full information (+0.008) |
| **D. Reference only** | Full-information optimum (round 0, election day, campaign-aware) | The methods above, applied to the true hidden voters | Everything | Yes: these are the ceilings |

**How to rank them** (all on the same worlds):
1. **Before the campaign:** each method's vote share on the forecast election-day electorate, and its gap to the
   full-information optimum. This gap is the **price of information** for that method.
2. **With a fixed good campaign:** play every manifesto with the same spending plan, so only the manifesto
   differs. For example: a few workers in round 1, then media in rounds 1–2.
3. **With its own best campaign:** run the plan search for each manifesto. Some manifestos convert better under a
   campaign. The ADMM manifesto, for instance, scores slightly *lower* before the campaign but *higher* in the game.
4. **By world type:**
   - two largest regions agree vs disagree;
   - dense vs sparse friendship networks.

   The rule-based methods (group B) are expected to be close to optimal when regions agree and to fall short when
   they disagree.

The group B rules are the most useful for a human player. If any comes within about 0.01 of the model-based methods,
it is a strategy a person can apply without a computer.

---

## 5. Strategy methods

### Method 1: Playbook (rule-based player)

- **Idea:** write the insights down as fixed rules and test whether they work as a whole.
- **The rules:**
  1. Ask about 2 survey questions, chosen to tell archetypes apart.
  2. Build the manifesto with a group B rule, aimed at election day.
  3. Round 1: buy 3–5 workers in the region that likes the manifesto most, and air media in most regions.
  4. Round 2: air media again.
  5. Rounds 3–5: spend whatever is left; never hold money back.
- **Information:** only what a player sees.
- **Build:** a new scripted player, using the existing player interface and policy helpers.
- **Test:** the tournament harness, on the same 100 worlds as the reference levels.
- **Success measures:**
  - the gap to the best plan found (0.672);
  - the comparison with the smart player (0.639).
- **Cost:** low. Minutes of compute.

### Method 2: Tuned playbook (small policy search)

- **Idea:** treat the playbook's numbers as dials and let the computer set them.
- **Dials:**
  - the number of workers;
  - the round in which they are bought;
  - the media share in each round;
  - how far to lean the manifesto towards a coalition (a blend weight from 0 to 1).
- **Build:** grid search or Bayesian optimisation over the dials on one set of worlds (the training set).
- **Test:** report results only on a separate set of worlds (the test set), so the policy is not fitted to luck.
- **Output:** a small, readable policy ("buy 4 workers in round 1, put 60% of the money into media in rounds 1–2,
  ...") with a measured vote share.
- **Cost:** moderate. Hundreds of settings × worlds, all batched on the GPU.

### Method 3: Adaptive playbook (react to visible conditions)

- **Idea:** the best tool changes with conditions the player can observe.
- **What the player can observe:**
  - average friends per voter (from the census);
  - how connected regions are to each other (from the census);
  - the electorate size;
  - the number of rounds.
- **Example branches:**
  - sparse network → shift money from workers to media;
  - very large electorate with a short game → mostly media;
  - small electorate or long game → more workers.
- **Build:** add these branches to the tuned playbook. Set the thresholds from the earlier experiments on network
  density and on electorate size and game length.
- **Test:** across those varied settings, against a non-adaptive playbook on the same worlds.
- **Cost:** moderate.

### Method 4: Improve the smart player

- **Idea:** the smart player already plans well by simulation, but has two blind spots. Fix them and measure the
  gain.
- **The blind spots:**
  1. Its list of candidate plans does not include the "few workers, then early media" shape that the plan search
     found to be best.
  2. In very large electorates its imagined electorates are too small: a few workers scale down to below one and
     become invisible to it. An earlier calibration experiment found that full-size imagined electorates raise its
     vote share by 0.034 at 100,000 voters.
- **Build:**
  - add the search-found plan shape to its candidates;
  - scale the size of its imagined electorates with N;
  - optionally add campaign-aware manifesto selection (group C).
- **Test:** re-run the tournament and the calibration experiment.
- **Cost:** moderate. It changes library code, which needs owner approval.

### Method 5: Plan search on imagined electorates (the true price of not knowing)

- **Idea:** the best plan found (0.672) used full information. Run the same search on the smart player's imagined
  electorates, then play the resulting plan in the real world.
- **What it measures:**
  - The gap between this and the full-information search is exactly what hidden information costs.
  - The gap between this and the smart player is what better planning could still win.
- **Build:** reuse the existing plan-search code, fed with imagined electorates instead of the true one.
- **Cost:** moderate to high. The search is the expensive part, but it runs batched on the GPU.

### Method 6 (optional): Learning a policy

- **Idea:** reinforcement learning or evolutionary search over general policies.
- **Recommendation:** not recommended now. The game's structure is already well understood, so Methods 1–3 should
  reach most of the attainable gain with far less compute. They also produce rules a person can read.

---

## 6. How every method should be evaluated

- **Same worlds for every method** (paired), with the tournament's 100 worlds as the main set.
- **Main measures:**
  - vote share;
  - the gap to the best plan found (0.672).

  Win rate is a secondary measure: most sensible strategies already win 86–98% of games, so win rate hides real
  differences.
- **Hard worlds, reported separately:**
  - the two largest regions disagree (the benchmark loses 11 of 100 such worlds and 0 of 100 otherwise);
  - sparse networks;
  - very large electorates with short games.
- **Training and test worlds kept separate** for any method that is tuned (Methods 2–3).
- **Robustness:**
  - the vote threshold (the agreement level at which voters switch);
  - prices;
  - the treasury.

  An earlier robustness experiment found the vote threshold is the hidden setting strategies are most sensitive to.

## 7. Suggested order

1. **Rank the manifesto methods** (Section 4). This is cheap, and every later method needs the answer.
2. **Method 1, playbook.** Does the simple rule set work?
3. **Method 2, tuned playbook.** How close can a readable policy get to 0.672?
4. **Method 3, adaptive playbook.** Does reacting to visible conditions help on hard worlds?
5. **Method 5, search on imagined electorates.** The true price of not knowing.
6. **Method 4, improved smart player.** Only if Methods 1–3 leave a clear gap.

**The headline result to aim for.** A strategy that uses only player-visible information and comes within about 0.01
of the best plan found would show that the game rewards understanding, not hidden knowledge.
