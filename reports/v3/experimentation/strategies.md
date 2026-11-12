# Finding a good strategy: methods to consider

**Status: Sections 1–7 are the original proposal. Three methods have now been run; see
[Section 8 (results)](#8-results-2026-11-12) and [Section 9 (how to play it yourself)](#9-how-to-play-it-yourself).**
- **Run:**
  - the smart player, touched up (Method 4);
  - the playbook (Method 1, tuned as in Method 2);
  - the adaptive playbook (Method 3).
- **Not run:**
  - Method 5;
  - Method 6;
  - the full manifesto ranking of Section 4. Only the group B rules were ranked, as part of the playbook.
- The proposal lists the ways we discussed to turn the experimental insights ([experiments.md](experiments.md))
  into a strategy that a real player could follow, and how each should be tested.

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

---

## 8. Results (2026-11-12)

### 8.1 Setup

**Main test worlds**
- These are the strategy tournament's worlds: seeds 12000–12099, N = 10,000 voters, 5 rounds, ₫500. Every
  player plays every world.
- No run used more than 10,000 voters.
- Everything ran on cuda:0.

**Checking that the results line up with the tournament**
- The re-run baselines (uniform, top3, people_pleaser and the default smart player) reproduce the stored
  tournament results (`raw/x6.jsonl`) to within 3·10⁻⁷ per world.
- The new numbers are therefore directly comparable with the table in Section 2.
- Sources: `s2.json → pairing_check_max_abs_dE_vs_x6` and `s1.json → reproduction_vs_x6`.

**How differences are reported**
- Δ is a difference between two players on the same worlds, with a 95% t-interval.
- "Gap to 0.672" is the distance to the best plan found by search ([`../x13.json`](../x13.json)).
  - That bound averages 20 random streams per world, while the games here use one stream per world.
  - So treat the gaps as approximate to about ±0.003.

| Run | What | Files |
|---|---|---|
| S1 | Smart player: the default vs each touch-up vs all touch-ups | [`s1.json`](s1.json), [`raw/s1.jsonl`](raw/s1.jsonl), [`logs/s1.log`](logs/s1.log) |
| S1-holdout | The same comparison (default vs recommended only) on 100 new worlds, seeds 14000–14099 | [`s1_holdout.json`](s1_holdout.json), `raw/`, `logs/` |
| S2-tune / S2 | Playbook: tuned on seeds 13000–13099, tested on 12000–12099 | [`s2_tune.json`](s2_tune.json), [`s2.json`](s2.json), `raw/`, `logs/` |
| S3-tune / S3 | Adaptive playbook: thresholds tuned on seeds 13000–13039 × 12 conditions, tested on 12000–12099 × 12 conditions | [`s3_tune.json`](s3_tune.json), [`s3.json`](s3.json), `raw/`, `logs/` |

The exact commands, devices and wall times are in each JSON's `provenance` block.

### 8.2 Headline

| Player | Information used | Vote share E [95% CI] | Win rate | Δ vs smart player (default) | Gap to 0.672 |
|---|---|---|---|---|---|
| **Smart player, recommended (new)** | What a player sees | **0.669** [0.655, 0.682] | 0.99 | **+0.030** [0.027, 0.033] | **0.003** [0.001, 0.006] |
| **Playbook, tuned (new, human-executable)** | What a player sees | **0.653** [0.636, 0.670] | 0.95 | **+0.014** [0.007, 0.020] | 0.019 [0.013, 0.025] |
| Smart player, default (re-run) | What a player sees | 0.639 [0.625, 0.654] | 0.98 | 0 | 0.033 |
| Playbook exactly as written in Method 1 (untuned) | What a player sees | 0.616 [0.601, 0.632] | 0.91 | −0.023 [−0.029, −0.017] | 0.056 |
| Top-3 statistic, even spending (re-run) | Free statistic only | 0.602 | 0.85 | −0.037 | 0.070 |

Sources:
- smart player rows: `s1.json → arms.{smart,bayes}`;
- playbook and top-3 rows: `s2.json → players.{playbook,playbook_method1_untuned,top3}`.

**Claims:**
1. **The recommended smart player comes within about 0.003 of the full-information best plan** while using only
   what a player sees.
   - This meets the target in Section 7 ("within about 0.01 of the best plan found").
   - On these worlds, hidden information is worth almost nothing to a good planner. The game rewards
     understanding, not hidden knowledge.
2. **The gain holds on 100 held-out worlds** (seeds 14000–14099, never used to choose the options):
   - smart player 0.679 vs default 0.650;
   - Δ = +0.030 [0.026, 0.033];
   - the smart player is better on 98 of 100 worlds (`s1_holdout.json`).
3. **The tuned playbook beats the default smart player by 0.014 while using no computer at all.**
   - It is better on 84 of 100 worlds (Wilcoxon p = 4·10⁻⁷).
   - It is 0.016 below the recommended smart player.
   - On win rate the two are not distinguishable: McNemar p = 0.375; 1 world is won only by the playbook and 4
     only by the default smart player.
4. **Method 1 as written in Section 5 is not good enough.**
   - Its rules were: 2 questions about the 2 largest regions, a majority-coalition blend, and 3 workers in the
     friendly region.
   - It falls 0.023 below the default smart player.
   - Tuning added +0.036 [0.028, 0.044].
   - Section 8.4 says which rules changed.

### 8.3 Smart player (Method 4)

**Audit.** No bugs were found that change results.

**Information it uses**
- `BayesPlanner` and everything it calls read only the observation dict and the public rules.
- The observation does carry the true game `seed` (`game/observation.py:94`).
  - A malicious player could regenerate the hidden electorate from it.
  - The smart player only uses it as a random-number salt.
  - This is an API contract issue, not a bug in the player. It was left unchanged.

**Smaller loose ends**, all left unchanged and reported only:
- Prices are read from the config, not from `observation["costs"]`. This is harmless in every current game mode.
  It would mis-price a custom-price game if no config were passed.
- Polls are never used to update its beliefs.
- The candidate cap of 20 never binds: there are 13–19 unique candidates.

**Unrelated failing test seen during the audit.** `test_api.py::test_budget_out_of_range_or_malformed_is_422[1000.01]`
fails: the API accepts a budget of 1000.01. The API was not touched; this needs checking separately.

**Touch-ups**
- All are opt-in options on a new `SmartPlayer` class ([`src/dumbocracy/strategies/smart.py`](../../../src/dumbocracy/strategies/smart.py)).
- With every option off, it plays exactly like `BayesPlanner`. This is tested.
- `BayesPlanner`'s default behaviour is unchanged, so every earlier result still reproduces.

| Option | What it does | Δ alone vs default |
|---|---|---|
| `front_candidates=True` | Adds the shape the plan search found best (blind spot 1): 1–5 workers in one of the two highest-value regions, plus media in 2–4 regions, paid from the whole treasury | +0.012 [0.010, 0.014] |
| `plan_world_fraction=0.4` | Planning electorates scale with N: 4,000 imagined voters at N = 10⁴ instead of 1,600 (blind spot 2) | +0.006 [0.004, 0.009] |
| `manifesto_method="admm"` | The ADMM manifesto from the tournament | +0.010 [0.007, 0.013] |
| `rollout="taper"` | Its forecasts assume the rest of the money is spent early, not evenly | −0.001 [−0.003, 0.001] alone; +0.003 [0.001, 0.004] on top of the other three |
| **All four** (`RECOMMENDED`) | | **+0.030** [0.027, 0.033] |

Source: `s1.json → arms.<arm>.vs_bayes`.

**Notes on the recommended player**
- The gains add up roughly, and the gain is larger in divided worlds:
  - aligned worlds (48): +0.027;
  - divided worlds (52): +0.032.
- **How it spends now:** it puts 87% of its money into rounds 1–2, against 43% for the default.
  - About 4.4 workers in round 1.
  - Media in about 3 regions per round in rounds 1–2.
  - This is the "few workers, then early media" shape of the best plans found.
- **Speed:** about 7 s per game on the GPU, against 2 s for the default.
- **Untested:** campaign-aware manifesto selection was not tried, and N above 10,000 was not tested.

### 8.4 Playbook (Methods 1 and 2)

**Training.** The dials were tuned on worlds 13000–13099:
- 36 survey × manifesto-rule combinations;
- 684 spending arms;
- a final pass over the lean dial.

The tuned values are in `s2_tune.json → chosen` and `experiments/x_playbook.py → TUNED_PLAYBOOK`. Section 9 has the
rules as a human would play them.

**Manifesto rules** on the test worlds, each with the tuned spending (`s2.json → players["rule:…"]`):

| Rule (all built with the "Blend my hunches" button) | E |
|---|---|
| **All four regions** (4 × Agriculture–Industry questions) | **0.653** |
| All four regions (2 questions each about 4 regions) | 0.650 |
| Three largest regions | 0.634 |
| Two largest / majority coalition | 0.608 |
| Divided-aware (anchor if divided, else coalition) | 0.586 |
| Largest-region anchor + closest small region | 0.582 |

**Claims that change the guidance in Section 3**

**1. Manifesto: under a good campaign, blend all four regions.**
- Section 3 and the round-0 study R1 say to anchor on the largest region and never blend the two large ones.
- That advice holds for the manifesto before any campaign. Under a full campaign the order reverses: the anchor
  rule scored worst.
- This agrees with Section 4's warning that the round-0 score and the in-game score differ.

**2. Surveys: one sector pair, asked once in every region, beats 2 "well-chosen" questions for a human.**
- The pair is Agriculture vs Industry. The answer alone almost separates the six archetypes.
- With the dossier rule in Section 9:
  - 98.75% of region hunches were correct;
  - all four regions were correct in 95 of 100 worlds.
- Two questions are enough only for the computer's joint posterior. That variant, which is not human-executable,
  is only +0.004 [0.000, 0.008] better, mostly from the ₫20 it saves on questions.

**3. Workers: 3 is enough.**
- Best training E by number of workers (`s2_tune.json → B_by_workers`):

  | Workers | 0 | 1 | 2 | **3** | 4 | 8 |
  |---|---|---|---|---|---|---|
  | E | 0.542 | 0.648 | 0.667 | **0.673** | 0.670 | 0.664 |

- Placing them by region size beats placing them in the friendliest region, but only by ≤ 0.0002.

**4. Media: spend everything in rounds 1–2, in all four regions.**
- After round 2 there is nothing left. This matches the best plans found by search.

**5. "Aimed at election day" (forecasting drift) adds nothing measurable.**
- Δ = +0.002 [−0.001, +0.006], not significant (`players.nonhuman_election_day`).
- A human can therefore skip it.

**Hard worlds.** These come from `s2.json → players.playbook.{divided,united,fragmented}`.
- **Divided** (52 worlds): the playbook scores 0.614 and loses 3.
  - That is +0.013 [0.004, 0.022] against the default smart player, which loses 2.
  - Top-3 loses 14 of these worlds and People Pleaser 24.
- **United** (48): it scores 0.695 and loses 2.
- **Fragmented** (the third of worlds with the fewest friendships between regions): it scores 0.642.
  - That is +0.009 [−0.006, +0.024] against the default smart player, which is not significant.

### 8.5 Adaptive playbook (Method 3)

**Branches as tuned** (`s3_tune.json`). The first match applies:

| Branch | When it applies | Spending |
|---|---|---|
| Small or long | N ≤ 1,500 or T ≥ 8 | 2 workers |
| Sparse | average friends per voter < 22.5 | 6 workers in round 1 and paced media |
| Short | T = 3 | 12 workers in round 1 |
| Base | everything else | The base playbook |

The survey and the manifesto are the same in every branch.

**Results** on the test worlds, adaptive vs non-adaptive (`s3.json → conditions.<c>.players`):

| Condition | Branch | Playbook | Adaptive | Δ [95% CI] | Smart (default) | Top-3 |
|---|---|---|---|---|---|---|
| Base (k̄ = 29) | base | 0.653 | 0.653 | 0 | 0.639 | 0.602 |
| k̄ = 10 | sparse | 0.526 | 0.523 | **−0.004** [−0.006, −0.002] | 0.544 | 0.491 |
| k̄ = 15 | sparse | 0.554 | 0.560 | +0.006 [0.002, 0.010] | 0.581 | 0.516 |
| k̄ = 20 | sparse | 0.595 | 0.603 | +0.009 [0.004, 0.013] | 0.614 | 0.548 |
| N = 1,000 | small | 0.655 | 0.656 | +0.001 [−0.001, 0.002] | 0.651 | 0.620 |
| N = 2,000 / 5,000 | base | 0.669 / 0.664 | same | 0 | 0.648 / 0.645 | 0.620 / 0.613 |
| T = 3 | short | 0.500 | 0.509 | +0.009 [0.006, 0.013] | 0.531 | 0.478 |
| T = 8 / 10 (not available in the UI) | long | 0.715 / 0.737 | 0.716 / 0.738 | +0.001 | 0.694 / 0.724 | 0.637 / 0.655 |
| N = 2,000, T = 3 | short | 0.545 | 0.576 | **+0.032** [0.028, 0.036] | 0.570 | 0.536 |
| Fragmented | base | 0.601 | 0.601 | 0 | 0.585 | 0.549 |

Here k̄ is the average number of friends per voter, and "Smart (default)" is the default smart player.

**Claims:**
1. **Adapting helps a little overall:** +0.0045 [0.0036, 0.0054] pooled over all 12 conditions.
   - The pooled conditions are not independent, since they share seeds.
   - The gain comes almost entirely from short games and moderately sparse networks.
   - At k̄ = 10 the sparse branch hurts slightly. Its threshold was set on 40 training worlds with a coarse grid,
     and it was deliberately not re-tuned on the test results.
2. **The adaptive directions are the opposite of the Section 3 / Method 3 hypotheses.**
   - Short games and moderately sparse networks want **more** workers (12 and 6), not "mostly media".
   - Small electorates and long games want **fewer** workers (2), and the effect there is tiny.
   - One reading: with ₫500, front-loading workers is what converts quickly when there are few rounds left to
     spread through the network. This explanation was not separately tested.
3. **Where a computer still clearly wins.**
   - The default smart player beats both playbooks in sparse networks (k̄ ≤ 20) and in short games at N = 10⁴, by
     0.011–0.031.
   - In those settings, a human should prefer the advisor (Section 9.3).
   - The recommended smart player was not run in this sweep, because its compute cost was too high for 12
     conditions.

### 8.6 Caveats

- **Sample:** each result is one game per world, on 100 worlds. The 0.672 bound averages 20 streams.
- **Smart player's options:** they were chosen on the test worlds themselves.
  - The held-out run (Section 8.2, claim 2) shows no selection bias.
  - The playbook was tuned only on separate training worlds.
- **Settings:** only N ≤ 10,000, ₫500 and the default prices and vote threshold were tested. The robustness checks
  in Section 6 were not run.
- **UI replica:** the playbook's model of the UI was ported from the TypeScript code and its tests:
  - the dossier ranking;
  - the "Blend my hunches" formula;
  - slider rounding to whole percent.

  Node was not available, so the TypeScript and the Python replica were not run side by side.

---

## 9. How to play it yourself

**Two of the three strategies translate into actions you can take in the UI. The smart player does not.**
- **The playbook and the adaptive playbook are complete click-by-click rules.** Following them in the UI
  reproduces the tested player exactly. The UI rounds sliders to whole percent; this is already included in the
  tested player.
- **The smart player cannot be followed by hand.** Each decision comes from simulating many imagined electorates.
  For it, there is an **advisor** script that reads your game and suggests your next move (Section 9.3).

### 9.1 Playbook (use this by default: N = 10,000, 5 rounds, ₫500)

1. **Survey (₫40).** Ask 4 questions, one per region, all with Sector A = **Agriculture** and Sector B =
   **Industry**.
   - Typical "% preferring more Industry than Agriculture" by archetype:

     | Archetype | Typical answer |
     |---|---|
     | Agrarian heartland | ≈ 0% |
     | Welfare-reliant | ≈ 22% |
     | Balanced middle | ≈ 50% |
     | Green urban | ≈ 78% |
     | Knowledge hub | ≈ 98% |
     | Industrial belt | ≈ 100% |

2. **Hunches.** Open the region dossiers, starting with the region with the most voters and working down.
   - Set "Your hunch" to the archetype ranked first in that dossier.
   - If a larger region already has that archetype, take the next one in the list instead (two regions never
     share an archetype).
3. **Manifesto.**
   - Click **Blend my hunches**.
   - Tick **all four** regions.
   - Click **Apply blend**.
   - Click **Commit manifesto**. Do not adjust the sliders.
   - Before committing, note which region's hunch has the highest "Match" %. The adaptive rules use it.
4. **Round 1.**
   - Buy **1 worker in each of the 3 largest regions**.
   - Air **media in all 4 regions**.
5. **Round 2.** Air **media in all 4 regions**. This uses up the treasury.
6. **Rounds 3–5.** Launch each round with nothing selected.

Tested result: E = 0.653 and 95 wins out of 100. That is better than the default computer planner (0.639) and
0.016 below the best computer player.

> **This playbook is for a ₫500 treasury only.**
> - With more money, it puts nearly all the extra into last-round workers.
> - At ₫2,500 or more it no longer beats the default computer planner.
> - At ₫100,000 its final order is rejected and 99% of the money goes unspent.
> - For any budget other than ₫500, use the budget-aware playbook (Section 10.4).

### 9.2 Adaptive playbook (when the game settings differ)

**First, read the conditions.**
- N and the number of rounds T come from setup.
- Click each region on the campaign map and note its "Friends per voter". Take the average of the four.

**Then apply the first rule that matches.** Steps 1–3 of the playbook (survey, hunches, manifesto) stay the same in
every case.

| If | Round 1 | Round 2 | Later rounds |
|---|---|---|---|
| **N ≤ 1,500** (or T ≥ 8, which the UI does not allow) | 1 worker in each of the 2 largest regions + media in all 4 | Media in all 4 | Final round: 1 worker in the best-match region with the ₫20 left |
| **Average friends per voter < 22.5** | 6 workers split by region size (usually 2, 2, 1, 1 from largest to smallest) + media in all 4 | Nothing | Each round, divide the money left by the rounds left. Tick media for regions in order (fewest airings so far first, then largest first) while the price fits in that amount. Final round: spend everything, media first, then workers in the best-match region |
| **T = 3** | 12 workers split by region size (usually 4, 3, 3, 2) + media in all 4 | Nothing is affordable | Round 3: 1 worker in the best-match region |
| **Otherwise** | Base playbook (Section 9.1) | | |

**Splitting workers by region size**
1. Multiply the number of workers by each region's share of voters.
2. Give each region the whole-number part.
3. Give the leftover workers to the regions with the largest fractional parts.

The "usual" splits shown in the table match 92% (6 workers) and 70% (12 workers) of worlds.

**Parts that are not fully human-executable**
- The sparse rule does not help in very sparse networks (k̄ ≈ 10), where it is slightly worse.
- The T ≥ 8 branch cannot be reached in the UI, because the round limit is 5.
- Connectivity between regions is in the API but not shown in the UI. No branch needs it.
- The hunch rule can need an archetype that is not on screen.
  - A region's dossier shows only its top 3 archetypes.
  - If all three are already taken by larger regions, the next archetype in the ranking is not visible.
  - This is expected to be rare but was not measured.

### 9.3 Smart player via the advisor (suggestions, not a rulebook)

**What it does.** `scripts/advisor.py` reads your own observation and prints what the recommended smart player
would do next, in the UI's words. It only uses
`GET /api/games/{game_id}/players/{player_id}/observation`. It never acts on your game and never reads hidden
state. It prints:
- the next survey question ("Sector A = …, Sector B = …, Region = … → Ask");
- or the manifesto as slider percentages adding up to 100 → "Commit manifesto";
- or this round's workers per region and media regions → "Launch round k";
- its best guess of each region's archetype, which you can use for your hunches;
- its own forecast of your vote share.

**How to use it**
1. Start the game server as usual with `scripts/run_game.sh` (port 61008).
2. Find the game id. The UI does not show it. Either:
   - in the browser console, run `JSON.parse(sessionStorage["dumbocracy.v2.session"]).g`; or
   - copy it from any `/api/games/<id>/…` request in the Network tab.
3. Run `conda run -n mas python scripts/advisor.py --game-id <id>`.
   - Optional flags: `--device cuda:0` (faster), `--player bayes` (the default planner), `--api <url>`, `--json`.
4. Do what it says in the UI, then **run it again after every action**.

**Testing so far.** It was tested end-to-end on a temporary server: one full game at N = 2,000 and 3 rounds,
following every suggestion. All actions were accepted and the game was won (E = 0.54). It takes 0–3 s per
decision on CPU.

**When to use which**

| Situation | Use | Why |
|---|---|---|
| Playing unaided | Playbook | It needs no computer and scores 0.653, above the default planner |
| Advisor available | Advisor | It scores about 0.016 higher (0.669) |
| Sparse friendship networks or 3-round games | Advisor | The playbooks fall short there (Section 8.5) |

---

## 10. Does it generalise? Unseen worlds, electorate size and budget (2026-11-12)

### 10.1 Setup

**Run S4.** Files: [`s4.json`](s4.json), [`s4_tune.json`](s4_tune.json), `raw/s4*.jsonl`, `logs/s4*.log`.

**Worlds**
- Every result in this section is on **held-out worlds**: seeds 14000–14099.
- These worlds were never used to tune any strategy or to produce Sections 8–9.
- The base cell uses 100 worlds. Every other cell uses the first 50 (14000–14049).

**Settings**
- 5 rounds, default prices and default network.
- At most 10,000 voters, by instruction.

**Checking that the results line up with S1-holdout.** The smart player and the default reproduce
`raw/s1_holdout.jsonl` to within 4·10⁻⁷ per world (`s4.json → reproduction_vs_s1_holdout`).

**Keys:** `s4.json → cells.<cell>.players.<arm>`. Δ values are paired, with a 95% t-interval.

### 10.2 Verdict

| Question | Smart player (recommended) | Playbook | Adaptive playbook |
|---|---|---|---|
| **Unseen worlds** | **Yes:** +0.030 [0.026, 0.033] vs default | **Yes:** +0.018 [0.014, 0.022] vs default (+0.014 on the S2 test worlds) | Yes, but only because it plays exactly like the playbook here |
| **Electorate size, 1,000–10,000 voters** | **Yes**: +0.027 to +0.030 vs default at every size | **Yes** at N ≥ 2,000. **Partly** at N = 1,000: +0.009 [−0.000, 0.019] | **No added value.** Its small-N branch is −0.005 [−0.011, 0.002] at N = 1,000, and it plays exactly like the playbook elsewhere |
| **More than 10,000 voters** | **Not tested** (by instruction) | Not tested | Not tested |
| **Budget, ₫500–₫100,000** | **Yes, with a shrinking edge**: +0.030 at ₫500, +0.014 at ₫10,000, +0.005 (not significant) at ₫100,000 | **No**: its edge is gone by ₫2,500, and it **breaks at ₫100,000** | **No**: it plays exactly like the playbook at N = 10⁴, T = 5 |

### 10.3 Numbers

**Electorate size** (₫500):

| N | Smart | Default | Playbook | Smart Δ vs default | Playbook Δ vs default | Smart Δ vs playbook |
|---|---|---|---|---|---|---|
| 1,000 | 0.677 | 0.650 | 0.659 | +0.027 [0.021, 0.034] | +0.009 [−0.000, 0.019] | +0.018 [0.011, 0.025] |
| 2,000 | 0.677 | 0.650 | 0.666 | +0.027 [0.021, 0.034] | +0.016 [0.008, 0.025] | +0.011 [0.004, 0.018] |
| 5,000 | 0.673 | 0.643 | 0.661 | +0.030 [0.024, 0.037] | +0.018 [0.010, 0.026] | +0.012 [0.006, 0.019] |
| 10,000 | 0.679 | 0.650 | 0.668 | +0.030 [0.026, 0.033] | +0.018 [0.014, 0.022] | +0.012 [0.008, 0.016] |

Every player spends at least 98% of the treasury at every size.

**Budget** (N = 10⁴). The column "Budget-aware playbook" is the single rule of Section 10.4; the table version is in
brackets.

| Budget | Smart | Default | Playbook | Budget-aware playbook | Playbook Δ vs default | Budget-aware Δ vs playbook |
|---|---|---|---|---|---|---|
| ₫500 | 0.679 | 0.650 | 0.668 | 0.668 (identical) | +0.018 | 0 |
| ₫1,000 | 0.704 | 0.676 | 0.688 | 0.690 | +0.012 [0.005, 0.019] | +0.002 |
| ₫2,500 | 0.714 | 0.698 | 0.698 | 0.699 (table: 0.710) | +0.000 [−0.007, 0.007] | +0.001 (table: +0.011 [0.006, 0.017]) |
| ₫10,000 | 0.726 | 0.712 | 0.708 | 0.717 (table: 0.721) | −0.004 [−0.010, 0.002] | +0.009 (table: +0.013) |
| ₫100,000 | 0.847 | 0.842 | **0.688** | 0.834 | **−0.154** [−0.170, −0.137] | **+0.147** [0.133, 0.161] |

**Claims**

**1. Why the playbook degrades with money.**
- Media can be aired at most once per region per round, about ₫200 per round for all four regions.
- After rounds 1–2, the playbook's only outlet is the final round's worker order, placed in one region.
- So at ₫10,000 about 95% of the money becomes about 445 last-round workers.
- At ₫100,000 the order asks for about 4,950 workers in one region. That is more than the region has eligible
  voters, so the engine rejects it (INVALID_ALLOCATION) in **all 50 worlds** and **₫99,100 (99%) is left
  unspent**.
- Source: `cells.b100000.players.playbook.money`.

**2. The smart player never breaks.**
- It spends at least 98% of the treasury in every cell.
- At ₫100,000 it had 5 rejected actions and left ₫1,771 unspent. The default had 19 rejected actions and left ₫7,199.
- Its edge over the default shrinks as money grows. With a huge treasury, roughly half the electorate can simply
  be bought as workers, and that leaves little room for planning to matter.
- It is above the playbook in every cell, by +0.011 to +0.159.

**3. More round-1 workers lower E at every budget from ₫1,000 up.**
- This shows in both the tuning grid and a direct probe: at ₫10,000, E falls steadily from about 5 to about 450
  round-1 workers.
- Spare money is better spent on media in every round, plus workers bought in the last round.
- The mechanism was not identified.

### 10.4 Budget-aware playbook (human-executable)

**Build and tuning**
- Class `BudgetPlaybook` ([`experiments/x_playbook_budget.py`](../../../experiments/x_playbook_budget.py)).
- Tuned only on training seeds 13000–13049 at ₫500, ₫1,000, ₫2,500, ₫10,000 and ₫100,000, over 17 arms
  (`s4_tune.json → chosen`).
- Training mean E across the five budgets: 0.731, against 0.701 for the playbook.
- **At ₫500 it is exactly the playbook of Section 9.1**, so it is safe to use always.

**Steps**
1. **Survey, hunches and manifesto:** do exactly as in Section 9.1.
2. **Round 1:** buy 3 workers, split by region size (usually 1 in each of the 3 largest regions). Air media in all
   4 regions.
3. **Every later round while the money lasts:** air media in all 4 regions, about ₫200 per round.
4. **Last round:**
   1. Air media in all 4 regions.
   2. Spend everything left on workers, split by region size: (₫ left ÷ 20) × each region's share of voters.
   3. **Never put them all in one region.** A region cannot supply more workers than it has eligible voters.

**Optional extra for budgets around ₫2,500.**
- Buy 6 workers in round 1 instead of 3 (this is the per-budget table, `chosen_schedule`).
- It adds +0.011 at ₫2,500 on held-out worlds and +0.004 at ₫10,000 (5 workers). It makes no difference
  elsewhere.

**Held-out results**
- It never leaves money unspent. This is the fix for the ₫100,000 breakdown: it gains +0.147 over the playbook
  there.
- It beats the default by +0.005 to +0.009 at ₫10,000, and is level with it at ₫100,000 (−0.007 [−0.016, 0.001]).
- It is 0.005–0.016 below the smart player at budgets of ₫2,500 or more.

**Advice for a human player**
- At any budget, the budget-aware playbook is the hand-played choice.
- With more than about ₫1,000, the advisor (Section 9.3) gains more over hand play than it does at ₫500.

### 10.5 Caveats

- **Sample size:** the non-base cells use 50 worlds each.
- **Rejected actions:** S4 runs with `play_game(strict=False)`, so a rejected action forfeits that round.
  Rejections are counted in `rejected_actions`; they occurred only at ₫100,000.
- **Untested settings:**
  - only 5 rounds, default prices and the default network were used;
  - the conditions of Section 8.5 (sparse networks, 3-round games) were not crossed with budget.
- **More than 10,000 voters was not tested, by instruction.** The earlier calibration X14
  ([`../x14.json`](../x14.json)) gives context only:
  - The default planner fell from E 0.648 at N = 10⁴ to 0.587 at N = 10⁵, because its 1,600-voter imagined
    electorates scale every worker below one voter.
  - Full-size imagined electorates recovered +0.034 [0.025, 0.043].
  - The smart player's `plan_world_fraction=0.4` is capped at 20,000 imagined voters, so at N = 10⁵ it would only
    partly close that gap.
  - The playbooks are untested at that size.
