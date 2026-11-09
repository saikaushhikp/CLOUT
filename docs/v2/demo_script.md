# Dumbocracy v2: live demo script (about 9 minutes; 5-minute cut at the end)

**Primary seed: N = 10 000, seed 11, 5 rounds.** A come-from-behind win. The true expected share starts at 31 %,
and the game ends with 5 409 votes (54.1 %). The epistemic timeline shows **"knew a win exists" from round 1 but
"knew a winning plan" only from round 3**. This is the knowing-that vs knowing-how distinction.

**Backup seed: N = 10 000, seed 6006.** A loss (45.0 %) whose timeline shows **truth ≠ knowledge**. Winning was
truly impossible from round 1, yet the player "saw a chance" (M◇Win) in rounds 3–5.

## How the seeds were chosen

25 seeds were played through the live API on 2026-11-08: 101, 7, 11, 13, 17, 21, 42, 77, 123, 202, 314, 555, 777,
1001, 1234, 2024, 2026, 3141, 4242, 5005, 6006, 7007, 8008, 9009 and 1107. Each game was played by the
observation-only BayesPlanner: 2 information-gain queries, an optimised manifesto rounded to whole percent, and
planner allocations. Results:
- **22 wins.** 18 of them showed "knew a winning plan" from round 1, which is a dull story.
- **Interesting wins:**
  - seed 11: knew a winning plan from round 3, 53.9 %;
  - seed 555: from round 4, 53.9 %;
  - seed 314: never knew a winning plan, yet won 54.4 %.
- **Losses:** 4242 and 6006, both 43.9 %.

For seed 11 we then replaced the planner's irregular orders with a simple rule a presenter can explain: **all
workers in West, media rotating North → West → East → West, then a final push**. That rule does slightly better
(54.1 %) and tells the same epistemic story. Every number below comes from replaying exactly this script through
the live server (cuda:0) and through an in-process engine on cuda:2. All games were deleted afterwards.
- Answers, polls, round reports and votes were **bit-identical** across replays and across both GPUs.
- Debrief and epistemic floats differed only at about 10⁻⁸, with identical verdicts.
- Typing the manifesto in the UI sends exactly [0.24, 0.14, 0.13, 0.31, 0.18], which was checked.

> **Device caveat.** (N, seed) gives the same electorate and free statistics on any machine. Survey samples,
> cascades, polls and the vote, however, use the *engine's device* random generator. The numbers below hold when
> `/api/health` reports `"device": "cuda:…"`. On a CPU-only engine the same script still wins and tells the same
> story, but with different numbers. See the CPU column in the appendix.

---

## Pre-demo checklist

**The day before**
1. On the GPU host:
   ```bash
   cd /data/ai23btech11017/MAS/CLOUT
   scripts/run_game.sh
   ```
   This starts the engine on :61008 and Next on :61007. Skip it if they are already running, and do not restart
   a server someone else is using. To force a fresh production build, `rm -rf web/.next` first.
2. On the laptop:
   ```bash
   ssh -N -L 61007:127.0.0.1:61007 <user>@<host>
   ```
   Then open http://127.0.0.1:61007.
3. Health check: `curl -s http://127.0.0.1:61007/api/health` should return `{"status":"ok","device":"cuda:0",...}`.
   A `cuda:N` device is required for the exact numbers below. `active_games` must be below `max_games` (8).
   The server evicts the oldest idle game when it is full, so do not leave games open.
4. Run the **rehearsal check**. It needs plain python3 and no packages. It plays the primary script through the
   API, prints the key numbers, and deletes the game. It takes about 10 s.
   ```bash
   python3 - <<'EOF'
   import json, urllib.request
   B = "http://127.0.0.1:61007/api"
   def call(m, p, body=None):
       r = urllib.request.Request(B + p, None if body is None else json.dumps(body).encode(), {"Content-Type": "application/json"}, method=m)
       return json.load(urllib.request.urlopen(r, timeout=120))
   print(call("GET", "/health"))
   g = call("POST", "/games", {"num_voters": 10000, "seed": 11, "rounds": 5}); G = f"/games/{g['game_id']}/players/p0"
   try:
       for a, b, r in [("Welfare", "Env", "N"), ("Edu", "Agri", "S")]:
           q = call("POST", G + "/query", {"sector_a": a, "sector_b": b, "region": r})["answer"]
           print(f"query {a}->{b} {r}: {q['pct_b_over_a']:.1f}% ± {q['moe']:.1f}")
       call("POST", G + "/commit", {"manifesto": [0.24, 0.14, 0.13, 0.31, 0.18]})
       W = lambda x, n, m: {"workers_by_region": [0, 0, 0, n], "extra_workers": x, "media_regions": [k == m for k in "NSEW"]}
       for o in [W(13, 23, "N"), W(13, 23, "W"), W(13, 23, "E"), W(13, 23, "W"), W(22, 32, "-")]:
           obs = call("POST", G + "/allocate", o)["observation"]
       print("polls", [p["estimate"] for p in obs["polls"]], "result", obs["result"], "treasury", obs["treasury"])
       print(call("GET", G + "/epistemic")["headline"])
   finally:
       call("DELETE", f"/games/{g['game_id']}")
   EOF
   ```
   Expected output:
   - `query Welfare->Env N: 100.0% ± 0.9`
   - `query Edu->Agri S: 21.5% ± 4.3`
   - `polls [0.315, 0.33, 0.41, 0.45, 0.525, 0.5325]`
   - `votes 5409`, `victory True`, `treasury 2.64`
   - `Victory looked possible throughout; you knew a winning plan from round 3. You won with 54.1% of the vote.`
5. Do a full click-through rehearsal in the browser with a stopwatch, following the script below.
6. Prepare the mock fallback on the laptop, in a separate checkout of the repo. Do not run it inside the server's
   `web/`, because the scripts use port 61007 and a build would overwrite `web/.next`.
   ```bash
   cd web && npm ci && NEXT_PUBLIC_DUMBOCRACY_MOCK=1 npm run dev
   ```
   Check that it opens at http://127.0.0.1:61007 with the SSH tunnel *closed*. Mock numbers will differ from the
   ones below.

**Ten minutes before**
- Re-run step 3, the health check, and close every stale game tab.
- Use the browser at 100 % zoom in light theme, with the window about 1440 px wide; the screenshots were taken
  at that size. Close the devtools.
- Have `reports/v2/screenshots/` open in a file viewer, and `reports/v2/final_report.md` (§7 and §8) in a tab.
- Keep this script printed or on a second screen. The seed is **11**.

---

## The script (primary seed 11)

Times are cumulative. **[Click]** means what to do, **[See]** means what should appear, and **[Say]** gives the
talking points.

### 0:00–0:45 — Setup screen

- **[Say]** "Each of the 10 000 dots is an agent. They live in four regions of a social network, each holds a
  preferred budget split over five sectors, and they talk to their friends. We get 500 coins, five rounds, and
  very little information."
- **[Click]** Set the voters field to **10,000**, type **World seed = 11**, and leave **Campaign rounds = 5** and
  the round clock off. Click **Start campaign**. Creation takes about 1–2 s.
- **[Say]** "The seed is public on purpose. The same N and seed rebuild the identical electorate on any machine:
  the graph, preferences and zealots are drawn from CPU seed streams and then moved to the GPU. Anyone in the room
  can replay this exact game."

### 0:45–2:30 — Survey desk (Phase 1)

- **[See]** The treasury is **Đ500.00**. The free national top-3 panel shows **Education 74.3 %, Welfare 48.9 %,
  Agriculture 40.3 %, Industry 69.1 %, Environment 67.4 %**. The region dossiers show:

  | Region | Voters | Share | Average friends |
  |---|---|---|---|
  | North | 2,045 | 20 % | 26.8 |
  | South | 3,172 | 32 % | 27.8 |
  | East | 2,043 | 20 % | 25.7 |
  | West | 2,740 | 27 % | **38.2** |

- **[Say]** "The archetype library is public, but which region has which archetype is hidden. That is 360
  possible maps, and personalities add 24 more possibilities. The free statistic is exact but national, so it
  blurs the regions together."
- **[Click]** In **Ask a region**, set **Sector A = Welfare**, **Sector B = Environment**, **Region = North**, then
  click **Ask · Đ5.00**.
  - **[See]** **100.0 % ± 0.9** (sample 205). The North dossier's fit meter now points to **Green urban**.
- **[Click]** Set **Sector A = Education**, **Sector B = Agriculture**, **Region = South**, then click **Ask · Đ5.00**.
  - **[See]** **21.5 % ± 4.3** (sample 317). The South fit meter points to **Industrial belt**. Treasury **Đ490.00**.
- **[Say]**
  - "Each answer is a fresh 10 % random sample with a 95 % margin of error, using the finite-population
    correction. Asking again pools the samples."
  - "Our Bayesian player chose these two questions by maximum expected information gain over the 360 maps. Each
    question separates the candidate archetypes of one region almost perfectly."
  - "Together with the free statistic, the posterior over σ is now essentially a single point: entropy about
    3 × 10⁻⁵, which the timeline will show later. Across 40 games, gain-selected questions resolve σ in about
    2 queries; random ones need about 8 (E4)."
- **[Click, optional, 10 s]** Pencil in the hunches: North = Green urban, South = Industrial belt,
  East = Knowledge hub, West = Agrarian heartland. Then click **Ready — design the manifesto**.

### 2:30–3:45 — Manifesto

- **[Say]** "A voter supports us with probability σ(20·(S − 0.825)), where S is one minus the total-variation
  distance between their preference and our manifesto. They have to be within 17.5 points of us to be a coin
  flip. One manifesto cannot please four different regions, so we optimise."
- **[Click]** The sliders are linked: changing one rebalances the unlocked ones. Type each value and lock it:
  1. **Education 24** → lock;
  2. **Welfare 14** → lock;
  3. **Agriculture 13** → lock;
  4. **Industry 31** → Environment becomes **18** automatically.
  - **[See]** Σ 100 %.
- **[Say]**
  - "These numbers come from projected gradient ascent on the expected vote share. Each step is a gradient step
    followed by Euclidean projection onto the simplex, which is the prox of the simplex indicator from the
    lectures. Consensus ADMM, with each region as an agent holding a local copy and agreeing on z, gives the same
    answer to within 10⁻³."
  - "The key trick is that we optimise against the *anticipated* electorate. Five rounds of DeGroot averaging
    pull every voter toward their regional mean, so we aim at where opinions will be on election day, not where
    they are now."
- **[Optional contrast, 15 s]** Show what **Blend my hunches** proposes, about 22/15/18/25/20. Then re-enter
  24/14/13/31/18.
  - **[Say]** "In rehearsal, the naive blend with this same campaign wins only narrowly (51.4 %), and an even 20 %
    split loses (44.7 %). The optimised manifesto wins 54.1 %."
- **[Click]** **Commit & start round 1**, then confirm in the dialog.

### 3:45–6:15 — Campaign HQ, 5 rounds (about 30 s each)

**What to say once, in round 1**
- "Workers knock on 25 doors each, and every contact starts an SIR cascade: the voter tells friends with
  probability 0.03 per hop, recovers with probability 0.5, and the cascade runs for 3 hops."
- "The lecture threshold is τ·λ_max/ρ > 1. λ_max is at least the mean degree, and West has about 27 in-region
  friends per voter (36 898 internal ties over 2 740 voters). So in West x is at least 0.03 × 27 / 0.5 ≈ 1.6:
  supercritical. East's bound is only 0.77. So all our workers go West."
- "Media airings in the same region decay as 1/(1 + 0.5n), so we rotate them and every airing is close to full
  impact."
- "Network view: during play we only see the regions, the 'fog of war'. Opinions are revealed after the vote."

**Each round:** first set **Hire extra workers**, then type the West worker count, then toggle one media button,
then click **Launch round t**.

| Round | Poll shown at start | Extra workers | West workers | Media | Cost | Treasury after | Round report: contacted / word of mouth / saw ads |
|---|---|---|---|---|---|---|---|
| R1 | **31.5 % ± 4.6** (trailing) | 13 | 23 | **North** (Đ40.90) | Đ92.90 | Đ397.10 | 533 / 1,714 / 1,219 |
| R2 | **33.0 % ± 4.6** | 13 | 23 | **West** (Đ54.80) | Đ106.80 | Đ290.30 | 526 / 1,505 / 1,656 |
| R3 | **41.0 % ± 4.8** | 13 | 23 | **East** (Đ40.86) | Đ92.86 | Đ197.44 | 522 / 1,506 / 1,254 |
| R4 | **45.0 % ± 4.9** | 13 | 23 | **West** (2nd airing, shows "67 % impact") | Đ106.80 | Đ90.64 | 517 / 1,558 / 1,697 |
| R5 | **52.5 % ± 4.9** (first lead) | 22 | 32 | none | Đ88.00 | **Đ2.64** | 694 / 1,871 / 0 |

R5's button reads **Final push — go to the polls**. The election-eve poll is about **53 % ± 4.9**.

**Per-round talking points**
- **R1.** "Word of mouth reaches 3.2 times as many voters as the workers touch directly. That is the cascade
  above threshold."
- **R2.** "The poll barely moved, 31.5 → 33.0, and that is inside the ±4.6 margin. Polls of 400 voters wobble.
  Should we panic? Keep this in mind for the debrief."
- **R3.** "Now 41 %. Notice that we have learnt nothing new about which region is which. What we are learning
  from the polls is how responsive this electorate is, which depends on personalities and zealots that no survey
  reveals."
- **R4.** "The second West airing is weaker; the UI says 67 % impact."
- **R5.** "Spend everything. Unspent money has no value."

### 6:15–8:45 — Results and debrief (Phase 3)

- **[See]** **Victory: 5,409 of 10,000 votes, 54.1 %**, expected share **53.2 %**. Bullets include:
  - "Campaigning added about 12.7 points over simply letting opinions settle (40.6 % → 53.2 %)";
  - "South was your stronghold (65.1 %); North never warmed up (40.0 %)";
  - "You came from behind…".
- **The hidden map, revealed.** Point at each row:

  | Region | Archetype | Personality | Zealots | Expected share start → end |
  |---|---|---|---|---|
  | North | Green urban | Media-savvy | 100 | 20.7 % → 40.0 % |
  | South | Industrial belt | Receptive to canvassing | 156 | 49.1 % → 65.1 % |
  | East | Knowledge hub | Independent | 91 | 47.9 % → 60.7 % |
  | West | Agrarian heartland | Socially driven | 130 | **5.7 % → 43.9 %** |

  - **[Say]** "Our two questions identified North and South exactly, and the posterior got East and West right.
    The personalities were never observable. West was the battleground: it started at 6 % support, and the
    socially driven, densely connected region carried our cascades."
- **Truth vs your polls.** True expected share was 31.2 → 36.2 → 40.5 → 44.8 → 49.2 → 53.2 %. The polls were
  31.5, 33.0, 41.0, 45.0, 52.5 and about 53 %.
  - **[Say]** "In round 2 the poll under-read the truth by 3 points. The fog of war is quantified."
- **Your manifesto vs the oracle.** Oracle about **24/15/13/31/18** against ours, 24/14/13/31/18. With no
  campaign: you 40.6 %, oracle 40.6 %.
  - **[Say]** "Two survey questions and the optimiser got within a rounding error of full information. The rest
    of the win came from the campaign."
- **How opinion spread.** Click **Alignment**, then press **play**.
  - **[See]** Across the 1 500 sampled voters (3 487 ties), the share below 10 % falls from 30 % to 2 %, and the
    share at 60–90 % rises from 17 % to 42 %. Exact before → after counts are in the appendix.
  - **[Say]** "You can watch the red core in West turn grey, then blue, along the friendship edges. This is
    DeGroot averaging plus word-of-mouth cascades."
- **Epistemic timeline.** Computed on demand, which takes about 2–4 s.

  | Start of round | R1 | R2 | R3 | R4 | R5 |
  |---|---|---|---|---|---|
  | Truth: victory still possible | YES | YES | YES | YES | YES |
  | Possible (∃π∃w) | YES | YES | YES | YES | YES |
  | **Knew a win exists** (∀w∃π) | **YES** | **YES** | YES | YES | YES |
  | **Knew a winning plan** (∃π∀w) | **no** | **no** | **YES** | YES | YES |
  | Defeat inevitable | no | no | no | no | no |
  | Best plan, wins in | 62 % ("workers focused on W, media in W") | 81 % | 100 % ("workers by region size, no media") | 100 % | 100 % |

  Headline: **"Victory looked possible throughout; you knew a winning plan from round 3. You won with 54.1 % of
  the vote."**
  - **[Say]**
    - "Columns are Kripke models. For each round we rebuild 16 electorates that are consistent with everything we
      had seen at that moment: our two answers, the free statistics, every poll so far and our own orders. We then
      test 14 candidate plans in each."
    - "In rounds 1 and 2 **every** world had **some** winning plan, but no single plan won in all of them. We knew
      *that* we could win without knowing *how*: ∀w∃π but not ∃π∀w, the quantifier swap from the epistemic logic
      lecture."
    - "The polls of rounds 2 and 3 removed the pessimistic worlds, and from round 3 one plan wins everywhere: we
      knew how."
    - "Across 600 decision points, 'knew a winning plan' was never asserted when no plan could truly win, and when
      it held at round 1 the player won 100 % of the time (scenario table)."
    - Point at the scope note: "These are claims over 16 sampled worlds and a finite plan set, not proofs over all
      worlds." (The *Uncertainty* row is in bits = nats / ln 2; the API and report quote nats.)

### 8:45–9:30 — Wrap-up

- **[Say]** "One game, the whole course:
  - consensus and spectral gap: the drift that wins elections without a campaign (E1);
  - resilient consensus and zealots (E2);
  - the SIR threshold, which decided where the workers went (E3);
  - the prox and ADMM manifesto;
  - epistemic and temporal logic in the debrief.
  Over 200 seeds our Bayesian planner wins 91 % and full information 96 %. The results are in the report."
- Optional teaser for questions: offer to start seed **6006** (backup) to show a loss.

### 5-minute cut

1. Setup (0:30).
2. Survey: ask the two questions, but say only the first three survey talking points (1:00).
3. Manifesto: type the values without the blend contrast (0:45).
4. Campaign: run R1–R5 using only the table, with talking points only in R1 and R2 (1:15).
5. Debrief: show only the region reveal and the epistemic timeline (1:30).

---

## Backup seed 6006: a loss that shows truth ≠ knowledge

Use this if asked "what does a loss look like?", or if seed 11 misbehaves. The time is the same (about 7 min).

- **Setup.** N = 10,000, seed **6006**, 5 rounds.
- **Free top-3.** Edu 68.7, Welfare 63.7, Agri 51.4, Industry 56.2, Env 60.0.
- **Census.**

  | Region | Voters | Average friends | Media price |
  |---|---|---|---|
  | North | 2,968 | 26.5 | Đ59.36 |
  | South | 1,922 | 21.9 | Đ38.44 |
  | East | 2,163 | 31.9 | Đ43.26 |
  | West | 2,947 | 36.9 | Đ58.94 |

- **Queries.**
  1. **A = Welfare, B = Agriculture, Region = East** → **0.0 % ± 0.9** (sample 216). Fit meter: Welfare-reliant.
  2. **A = Education, B = Agriculture, Region = North** → **0.0 % ± 0.6** (sample 297). Fit meter is split between
     Knowledge hub (about 58 %) and Green urban (about 42 %); the free statistic in the Bayesian posterior settles
     it.
- **Manifesto.** **27 / 17 / 14 / 26 / 16** (Edu / Welfare / Agri / Industry / Env).
- **Orders.** All workers in West every round.

  | Round | Poll at start | Extra workers | West workers | Media | Cost | Treasury after |
  |---|---|---|---|---|---|---|
  | R1 | 27.0 % ± 4.4 | 13 | 23 | West | Đ110.94 | Đ379.06 |
  | R2 | 32.3 % ± 4.6 | 13 | 23 | East | Đ95.26 | Đ283.80 |
  | R3 | 32.8 % ± 4.6 | 13 | 23 | West (2nd) | Đ110.94 | Đ172.86 |
  | R4 | 36.3 % ± 4.7 | 13 | 23 | South | Đ90.44 | Đ82.42 |
  | R5 | 42.3 % ± 4.8 | 20 | 30 | none | Đ80.00 | Đ2.42 |

  The election-eve poll is about 42 %.
- **Result.** **Defeat: 4,503 votes (45.0 %)**, expected 44.9 %.
- **Debrief.**

  | Region | Archetype | Personality | Expected share start → end |
  |---|---|---|---|
  | North | Knowledge hub | Socially driven | 53.0 → 73.2 % |
  | South | Industrial belt | Receptive to canvassing | 32.5 → 47.2 % |
  | East | Welfare-reliant | Independent | 8.1 → 13.5 % |
  | West | Agrarian heartland | Media-savvy | 9.0 → 37.8 % |

  Oracle manifesto about 28/16/13/28/16. No-campaign share: oracle 34.8 %, yours 34.6 %.
  - **[Say]** "The manifesto was essentially optimal. This electorate simply could not be won."
- **Epistemic timeline.**

  | Start of round | R1 | R2 | R3 | R4 | R5 |
  |---|---|---|---|---|---|
  | **Truth: victory still possible** | **no** | **no** | **no** | **no** | **no** |
  | Possible (∃π∃w) | no | no | **YES** | **YES** | **YES** |
  | Knew a win exists / knew a winning plan | no | no | no | no | no |
  | Defeat inevitable | YES | YES | no | no | no |
  | Best plan, wins in | 0 % | 0 % | 6 % | 6 % | 6 % |

  Headline: *"From round 1 no plan won in any world you considered possible. In reality no plan could win from
  round 1 on, yet you still saw a chance in rounds 3, 4 and 5. You lost with 45.0 % of the vote."*
  - **[Say]** "Truth ≠ knowledge. From round 3 on, one of the 16 worlds we could not rule out had a winning
    plan, so 'possible' (M◇Win) became true. But the real world was never one of the winnable
    ones. The polls were noisy enough to keep a false hope alive. This is the optimistic error we measured in
    the scenario table: 'Possible' was asserted at 68 of the 136 decision points (out of 600) where no plan could truly
    win, while 'knowing how' was never wrong."

---

## Fallback plan

| Problem | What to do |
|---|---|
| Page does not load | Check the tunnel (`ssh -N -L 61007:…`) and `curl /api/health` on the host. If the engine is down and the host is yours to manage, run `scripts/run_game.sh`. |
| Health shows `"device":"cpu"` (GPU busy or unavailable) | Play the same script anyway: the electorate and free statistics are identical. Use the CPU column in the appendix for the expected values; the epistemic story is identical. |
| A number differs from the script | Carry on and narrate honestly. Small differences mean a different device or a different action order: queries are keyed by their order, so ask the two questions in the order given. |
| Epistemic timeline slow or showing 503 | It is computed on first request (target < 15 s). Talk through the truth-vs-polls chart while it loads. If it shows 503, the logic module is unavailable on that engine: switch to `screenshots/desktop-11-results-epistemic.png` and the table above. |
| Rejection toast ("not enough money", too many workers) | Nothing was charged, because rejections are atomic. Fix the input: set extra workers **before** the West count, since the West field is capped at the worker pool. |
| Server unusable | Switch to the **mock build** prepared on the laptop (`NEXT_PUBLIC_DUMBOCRACY_MOCK=1 npm run dev`). It is a simplified in-browser engine, so the numbers differ but the flow and every screen match. Or walk through the screenshots in this order: `desktop-01-setup`, `03-survey-answers-pooled`, `04-manifesto`, `06-campaign-round1`, `07-campaign-round2-planned`, `08-campaign-round5`, `09-results`, `10-results-network-alignment`, `11-results-epistemic` (in `reports/v2/screenshots/`). |
| Projector is narrow | The layout works down to phone width (`mobile-*.png`), but use 1440 px if possible so the epistemic table fits. |

**After the demo:** close the tab. The game stays in server memory until it is evicted. To clean up explicitly,
use `curl -X DELETE http://127.0.0.1:61007/api/games/<game_id>`; the game id is in the browser's network tab.

---

## Appendix: rehearsal record (seed 11, primary script)

| Quantity | CUDA engine (live server cuda:0 = in-process cuda:2) | CPU engine (in-process) |
|---|---|---|
| Query 1: Welfare → Env, North | 100.0 % ± 0.9 (n = 205) | 100.0 % ± 0.9 |
| Query 2: Edu → Agri, South | 21.5 % ± 4.3 (n = 317) | 19.6 % ± 4.2 |
| Polls R1 … R5, eve | 31.5, 33.0, 41.0, 45.0, 52.5, 53.25 | 32.75, 37.0, 41.75, 43.0, 45.75, 50.25 |
| Result | 5,409 votes (54.09 %), E = 0.5325, win | 5,334 votes (53.34 %), E = 0.5315, win |
| Epistemic headline | knew a winning plan from round 3 | knew a winning plan from round 3 (identical verdict rows) |
| Timeline compute | 1.9 s (cached repeat 0.005 s) | 3.9 s |
| Network bins at round 0 (<10 / 10–40 / 40–60 / 60–90 / >90 %) | 450 / 503 / 291 / 253 / 3 of 1 500 | — |
| Network bins after R5 | 33 / 420 / 391 / 636 / 20 | — |

**Manifesto comparison with the same campaign (CUDA):**

| Manifesto | Result | No-campaign share | Headline |
|---|---|---|---|
| Optimised 24/14/13/31/18 | 54.1 %, win | 40.6 % | knew a winning plan from round 3 |
| "Blend my hunches" 22/15/18/25/20 | 51.4 %, win | 37.9 % | never knew a winning plan |
| Uniform 20 × 5 | 44.7 %, loss | 33.6 % | defeat inevitable from round 1 |

**The original BayesPlanner run on seed 11,** with planner orders instead of the simple rule:
- orders: R1 workers N3/S3/E6 + 2 extra, media N; R2 W23 + 13, media W; R3 N2/S4/E2/W3 + 1, media S+E;
  R4 W24 + 14, media W; R5 W38 + 28;
- outcome: 53.9 % win; knew-that from round 1, knew-how from round 3.
