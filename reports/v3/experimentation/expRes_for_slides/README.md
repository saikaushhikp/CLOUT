# Experiment results for the slides

This folder holds four figures from three small experiments.

**Common settings**
- 50 held-out worlds, seeds 14000–14049. These were never used for tuning.
- Default rules, prices and network. 5% of voters are zealots.

**Reading the figures**
- Vote share is the expected vote share E: the mean over voters of each voter's chance of voting for the party.
- A win needs more than 50%.
- Lines are means over the 50 worlds; shaded bands are 95% confidence intervals.
- Starting shares vary a lot from world to world, so the bands are wide. The paired differences quoted below compare
  the same world and are the stronger evidence.

**Every figure comes as**
- a `.png` file (200 dpi);
- a `.svg` file;
- a `.json` file with all plotted numbers and provenance (exact command, seeds, device, wall time).

Logs are in `logs/`.

| Figure | Script | Wall time (cuda:0) |
|---|---|---|
| `worker_spread_touched_converted` and `worker_spread_vote_share` | `experiments/s5_worker_spread.py` | 23 s |
| `manifesto_no_campaign` (10,000 voters, 10 rounds) | `experiments/s6_manifesto_no_campaign.py` | 23 s |
| `media_vs_workers` | `experiments/s7_media_vs_workers.py` | 4 s |

Reruns use the `mas` conda environment, for example:

```
OMP_NUM_THREADS=4 python experiments/s5_worker_spread.py --seeds 50 --first-seed 14000 --device cuda:0
```

`s6` and `s7` take the same flags.

---

## 1. How influence spreads from workers bought only in round 1

**Setup**
- 10,000 voters, 5 rounds.
- k = 5, 10 or 20 workers, all bought in round 1. They are split by region size (largest remainder) and placed by
  the engine's normal rule.
- No later workers and no media.
- **Manifesto:** a blend of the four regions' true archetypes, weighted by size. This is what the UI's "Blend my
  hunches" produces when all four regions are ticked.

**Definitions**
- **Touched** means the voter received at least one conversion attempt from a worker or a convert. In the engine,
  this is the voter's `attempts` counter being above 0 (at most 3 attempts per voter).
- **Converted** means the voter was won over at some round up to t.
- Workers themselves are not counted in either.

### (a) `worker_spread_touched_converted.png`: voters reached over time

Cumulative number of voters, out of 10,000:

| Workers | | R1 | R2 | R3 | R4 | R5 |
|---|---|---|---|---|---|---|
| 5 | touched | 142 | 712 | 2,985 | 6,793 | **8,954** |
| 5 | converted | 20 | 129 | 724 | 2,147 | **3,237** |
| 10 | touched | 275 | 1,330 | 4,611 | 8,169 | **9,328** |
| 10 | converted | 39 | 245 | 1,186 | 2,695 | **3,411** |
| 20 | touched | 549 | 2,437 | 6,481 | 8,976 | **9,433** |
| 20 | converted | 78 | 477 | 1,819 | 3,087 | **3,423** |

**Takeaways**
- **A handful of workers starts a cascade that reaches almost everyone.** Five round-1 workers end up approaching
  about 9,000 of 10,000 voters and converting about 3,200.
- **More workers make it happen sooner, not bigger.** At round 3, 20 workers have converted 2.5 times as many voters
  as 5 workers. By round 5 all three settle at about 3,200–3,400.
- **The spread crosses region borders.** By round 5, about 37% of the largest region and about 30–32% of the other
  regions are converted (`worker_spread.json → regions_round_T`).

### (b) `worker_spread_vote_share.png`: vote share over time

Expected vote share, %:

| Workers | R0 | R1 | R2 | R3 | R4 | R5 | Win rate | Gain at R5 vs no workers (paired) |
|---|---|---|---|---|---|---|---|---|
| 0 | 34.6 | 36.7 | 38.8 | 40.9 | 42.9 | 44.9 | 28% | — |
| 5 | 34.6 | 36.8 | 39.3 | 43.6 | 51.4 | **58.4** | 82% | +13.5 pp [12.7, 14.2] |
| 10 | 34.6 | 36.9 | 39.9 | 45.5 | 54.1 | **59.4** | 84% | +14.5 pp [14.0, 15.0] |
| 20 | 34.6 | 37.2 | 40.8 | 48.3 | 56.2 | **59.6** | 86% | +14.7 pp [14.3, 15.1] |

**Takeaways**
- **Five round-1 workers turn a likely loss into a likely win:** 45% → 58% at the election, and the win rate goes
  from 28% to 82%.
- **Going from 5 to 20 workers adds only about 1 more point by round 5.** Extra workers mainly bring the gain
  forward.

**Caveats**
- **The no-worker line rises on its own.** Voters drift towards their friends' average, and the blended manifesto
  sits near the population average.
- **The end-point saturation depends on the setup.** It partly comes from the limit of 3 attempts per voter and from
  the 5-round game.
- **One game was played per world and worker count,** with no repeated cascade trials.

---

## 2. Five fixed manifestos with no campaign, over 10 rounds: `manifesto_no_campaign.png`

**Setup**
- **10,000 voters, 4 regions, 10 rounds.**
- No workers, no media and no surveys. Opinions only drift through friendships.
- The manifesto is fixed at round 0 and never changed.
- Stars mark the first round at which each mean line reaches 50%.

**Command**

```
python experiments/s6_manifesto_no_campaign.py --seeds 50 --first-seed 14000 --voters 10000 --rounds 10 --device cuda:0
```

- Wall time: 23 s.
- The engine-parity check passed (max |ΔE| = 9·10⁻⁸).

**The five manifestos** are built from archetype templates, as a player would:

| Manifesto | How it is built |
|---|---|
| 1 region | The largest region's archetype |
| 2 regions | UI "Blend my hunches" (size-weighted, whole percents) of the 2 largest regions |
| 3 regions | The same, for the 3 largest regions |
| 4 regions | The same, for all four regions |
| Balanced | 20% to each sector |

**Results.** Expected vote share, %. The brackets at R5 and R10 give the number of worlds (out of 50) at or above
50%.

| Manifesto | R0 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | R10 | 95% CI at R10 | Mean first reaches 50% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 region | 33.5 | 34.6 | 35.5 | 36.3 | 37.0 | 37.7 (3) | 38.3 | 38.8 | 39.3 | 39.8 | 40.2 (7) | 37.9–42.6 | Never (43.3 at R20) |
| 2 regions | 33.6 | 35.5 | 37.2 | 38.9 | 40.6 | 42.2 (7) | 43.6 | 45.0 | 46.4 | 47.6 | 48.8 (25) | 46.8–50.9 | After R10 (57.8 at R20) |
| 3 regions | 33.8 | 35.9 | 37.9 | 39.8 | 41.8 | 43.7 (11) | 45.5 | 47.2 | 48.8 | **50.4** | 51.9 (31) | 49.8–54.0 | **Round 9** |
| **4 regions** | **34.6** | 36.7 | 38.8 | 40.9 | 42.9 | 44.9 (12) | 46.9 | 48.7 | **50.5** | 52.2 | **53.8 (35)** | 51.8–55.8 | **Round 8** |
| Balanced | 31.2 | 32.9 | 34.6 | 36.3 | 37.9 | 39.5 (3) | 41.0 | 42.5 | 43.9 | 45.2 | 46.5 (17) | 44.4–48.6 | After R10 (56.4 at R20) |

Round 20 is in the JSON only.

**Takeaways**
- **Without a campaign, nothing reaches 50% within a 5-round game.** Drift alone adds about 2 points per round to
  the blends.
- **Given enough time, the broad blends win on their own.** On average, the 4-region blend passes 50% at round 8
  and the 3-region blend at round 9.
- **The more regions blended, the faster the gain.**
  - The 4-region blend gains about 19 points in 10 rounds and is the best manifesto in 43 of 50 worlds at round 10.
  - The 1-region manifesto gains only about 7 points and stalls near 40%.
- **The flat 20% split starts worst (31%).** It overtakes the 1-region manifesto by round 3 but stays below every
  blend.
- **Waiting is not a strategy.** Even at round 10 the best blend is below 50% in 15 of 50 worlds. Inside the real
  game the campaign is what wins (compare figure 1b).

**Consistency check.** The "4 regions" line matches the "no workers" line of figure 1(b) to within 0.02 points. Both
runs use the same seeds, 10,000 voters and the same manifesto.

**Caveats**
- In worlds whose largest region is "Balanced middle", the "1 region" and "Balanced" manifestos are the same.
- **Superseded files:** `manifesto_no_campaign_10r.{png,svg,json}` and `logs/manifesto_no_campaign_10r.log` are an
  earlier 2,500-voter run and are no longer used. They can be deleted.
  - That run gave the same picture within about 0.5 points: 4 regions passes 50% at round 8 and 3 regions at
    round 9.

---

## 3. All media vs all workers, same money: `media_vs_workers.png`

**Setup**
- 2,500 voters, all in **one region**, 3 rounds.
- **Balanced manifesto** (20% to each sector), no surveys.

**How the single region was made**
- The library always builds 4 regions, so the region split was forced with existing config options:
  `min_region_fraction=0` and `region_dirichlet_a=1e-3`. This puts all 2,500 voters in one region and leaves three
  empty.
- The script checks that every world is single-region. Average friends per voter is still about 30.
- The region's archetype varies from world to world: Industrial belt 13, Agrarian 12, Knowledge hub 8,
  Welfare-reliant 7, Green urban 5, Balanced middle 5.

**Money**
- One media airing in a region holding all voters costs ₫200, from the engine's own pricing.
- Airing in all 3 rounds therefore costs **M = ₫600**, which is the same as **30 workers** at ₫20, with ₫0 left over.
- The treasury was set to ₫600, because the default ₫500 cannot pay for 3 airings.

**The three arms**

| Arm | What it does |
|---|---|
| All media | Air media in rounds 1, 2 and 3 (₫600) |
| All workers | Buy 30 workers in round 1, nothing after (₫600) |
| No campaign | Spend nothing (baseline) |

**Results.** Expected vote share, %:

| Arm | R0 | R1 | R2 | R3 |
|---|---|---|---|---|
| No campaign | 28.0 | 28.5 | 28.9 | 29.2 |
| All media | 28.0 | **32.8** | 36.0 | 38.5 |
| All workers | 28.0 | 31.3 | **37.2** | **42.7** |

**Round 3, paired comparison**
- All workers minus all media: **+4.2 pp [1.8, 6.5]**, Wilcoxon p = 0.001.
- Workers are ahead in 31 of 50 worlds.
- Elections won: workers 12, media 6, no campaign 5.
- Gain over no campaign: media +9.4 pp, workers +13.5 pp.

**Takeaways**
- **With the same ₫600, 30 workers bought in round 1 beat 3 media airings** by about 4 points at round 3.
- **Media is ahead after round 1,** but the worker cascade overtakes it from round 2 onwards.
- **Both campaigns clearly beat doing nothing,** but on average both still end below 50%. A single region with a
  20% manifesto is a hard world.

**Caveats**
- **The "single region" is not a native game mode.** It is a 4-region world with three empty regions, and the
  budget had to be raised to ₫600.
- **5 worlds start already won.** In those worlds the region is "Balanced middle", which matches the manifesto, so
  they sit at about 85% with no campaign.
- **The result holds only for these settings:** ₫20 workers, 3 rounds, 2,500 voters.
- **Compare with the full game.** In the full game with 10,000 voters and 4 regions, the best plans buy only about 3
  workers and then put about 89% of the money into early media ([`../experiments.md`](../experiments.md) Section 9).
  Here, all workers beats all media, but neither arm is the best mix.
