# P9 — Experiment suite for the current game (ideation, for owner review)

Status: **IDEA DRAFT, 2026-11-10.** The owner edits this file. Implementation starts only after that review, and
the implementer re-reads this file first.

Rules under test: spec v2 with the post-v2.1 changes R16–R30. The main changes are:
- party workers are bought permanent nodes, pinned at b;
- they spread by an Independent Cascade one hop per round, with chance γ·P and at most 3 attempts per voter;
- there are no free workers; a worker costs ₫20 and a survey query ₫10;
- the treasury is chosen at ₫500–₫1000.

The old P7 results (E1–E7) were measured under the old worker rules and are superseded by this suite.

---

## 0. How we run every experiment (common protocol)

- **GPU first.**
  - Simulate many worlds at once as batched tensors (B worlds × N voters × 5 sectors) on one GPU, using the
    existing `strategies/worlds.WorldBatch` and `simulate.resolve_campaign_round` path.
  - Use `cuda:2` by default and check free memory first (`config.resolve_device`), since the GPUs are shared.
  - The CPU is only the reference for a small parity check per experiment (same seed gives the same numbers).
- **Paired seeds.**
  - Every arm of an experiment uses the same seeds, so the same electorates are compared.
  - Core experiments use 200 seeds at N = 10,000; sweeps use 100 seeds per cell.
- **Statistics.**
  - Win rates get 95% Wilson intervals; means get t-intervals.
  - Paired arms are compared with McNemar (win/loss) and Wilcoxon (share).
  - We report effect sizes, not only p-values.
- **Data flow:**
  - raw per-game rows go to `reports/v3/raw/<exp>.jsonl`;
  - summaries to `reports/v3/<exp>.json`;
  - figures to `reports/v3/figures/`;
  - one consolidated `reports/v3/results.md`.
  - Figures are drawn only from saved data, and every experiment runs from one command:
    `python experiments/x<k>_*.py --seeds … --device cuda:2`.
- **Every run records:** config, prices, seeds, git revision, device, and wall time.

---

## X0 — Balance gate at the new prices *(Core)*

- **How:** re-run the five balance checks (spec §12) at worker ₫20, query ₫10 and treasury ₫500.
  - Repeat at ₫750 and ₫1000, since the treasury is now the player's choice.
  - Policies: no campaign; workers only (spread over rounds); workers front-loaded in round 1; media only;
    split workers and media; oracle planner. Manifestos: uniform, top-3, informed, oracle.
- **Necessary?** Yes. Every later result is only meaningful if the game is not trivially won or lost. The
  price check found that cheap workers make "split workers and media" dominant (82–90% of games at ₫10–₫50), and
  ₫20 sits in that zone.
- **Infer:**
  - whether the game is balanced at the chosen prices;
  - which strategy dominates, if any;
  - whether a bigger treasury makes the game too easy.
  - If a check fails, we report it honestly and suggest the smallest rule or price change that would fix it.
- **Lecture link:** none directly. This is calibration.

## X1 — Network structure and consensus speed *(Core; Linear Algebra, MAS Algorithms / consensus)*

- **How:**
  - Vary the mean friend count (k̄ ∈ {10, 20, 30}) and cross-region tie strength (p_g range ∈ {[0.01, 0.05],
    [0.01, 0.2], [0.1, 0.4]}).
  - For each world compute the Laplacian's algebraic connectivity λ₂, the random-walk spectral gap 1 − |λ₂(W)|,
    and the cross-region conductance.
  - Run consensus-only dynamics (no campaign) and measure how fast opinions mix: rounds until regional means
    move halfway to the global mean.
  - Then play the standard strategies on the same worlds.
- **Necessary?** Yes. This is the main linear-algebra/consensus result, and the network is the one hidden thing
  the player can never survey.
- **Infer:**
  - whether mixing speed follows the spectral gap, as the theory predicts;
  - whether well-connected countries reward a single nationwide manifesto, while fragmented ones reward
    region-targeted play.
- **Lecture link:**
  - DeGroot consensus x(t+1) = W x(t), whose convergence rate is set by the second eigenvalue of W.
  - The Laplacian's λ₂ (Fiedler value) measures community structure.

## X2 — Stubborn agents: party workers vs zealots, and resilient consensus *(Core; MAS Algorithms: W-MSR / resilient consensus)*

- **How:**
  - Arm A: vary the zealot fraction ζ ∈ {0, 5, 15, 30%} under a fixed campaign.
  - Arm B: give the player exactly k workers (k ∈ {0, 5, 10, 20}) and switch the cascade off, so workers act only
    as pinned stubborn nodes in the averaging. This isolates the Friedkin–Johnsen effect of stubborn agents from
    the cascade.
  - Arm C: repeat A and B with the social rule set to W-MSR (F = 1, 2), where voters ignore their F most extreme
    friends.
  - Compare against the (r, s)-robustness bounds the analysis module already computes.
- **Necessary?** Yes. Workers pinned at b *are* stubborn agents. Their steady-state pull is the classic
  stubborn-agent result, and W-MSR is the lecture's defence against stubborn or malicious nodes.
- **Infer:**
  - how much of a worker's value comes from being a permanent "anchor" rather than from the cascade;
  - whether zealots cap the reachable share;
  - whether W-MSR protects voters from *your* workers, which treat the party as an adversary from the
    network's point of view;
  - what resilience costs in mixing speed.
- **Lecture link:** with stubborn nodes the final opinion of each normal voter is a convex mix of the stubborn
  opinions, weighted by the probability that a random walk from that voter ends at each stubborn node.
  W-MSR guarantees consensus among normal nodes under F-local/F-total threat models if the graph is
  (F+1, F+1)-robust.

## X3 — The worker cascade: threshold and reach *(Core; Infection Models)*

- **How:**
  - Seed k workers in one region (k ∈ {1, 3, 5, 10}) and record converts per round (one hop per round).
  - Vary the manifesto from "far from the region" to "ideal for the region", which sweeps the mean conversion
    chance E[γP].
  - Compute the predicted reproduction number per spreader, R₀ ≈ k̄ · E[γP] (mean friends × mean chance), plus its
    spectral version λ_max(A) · E[γP], and compare with the measured growth ratio converts(t+1)/converts(t).
  - Also check how often the 3-attempt cap binds and how many conversions cross region borders.
- **Necessary?** Yes. This replaces the old SIR threshold experiment (E3), which no longer applies. The cascade
  is the core new mechanic.
- **Infer:**
  - whether reach shows a sharp threshold near R₀ = 1, so that whether a cascade "takes off" depends on the
    manifesto;
  - how many rounds it takes to saturate a region;
  - whether a convert's spread is wasted when the voter drifts back after their turn.
- **Lecture link:**
  - Independent Cascade and SI/SIR models: the epidemic threshold is R₀ = 1 on random graphs, and τ·λ_max(A) = 1 on
    general graphs.
  - The attempt cap makes this a finite-exposure cascade, so its cascade size differs from plain SI.

## X4 — Worker timing and placement *(Recommended; Infection Models, MAS)*

- **How:**
  - Use the same total worker spend, but with different timing: all in round 1; even spread; all in the last
    round.
  - Use different placement: the largest region; the region best matched by the manifesto; split proportionally.
  - Ablation of the picking rule:
    - today's random voter with ≥ 3 friends;
    - the source document's "highest degree in a 10% sample";
    - the true highest degree as an upper bound.
- **Necessary?** Recommended. It turns the cascade theory into player advice, and it tests the owner's
  original "highest degree centrality" idea against the simpler random pick that was adopted.
- **Infer:**
  - how much "buy early" is worth in vote share;
  - whether region choice matters more than how many workers you buy;
  - how much degree-centrality picking would change outcomes, which tells us whether the random-pick
    simplification hides an important skill.
- **Lecture link:** influence maximisation under IC. Seeding high-degree or high-centrality nodes approximates the
  submodular greedy optimum.

## X5 — Value of information at ₫10 per question *(Core; Epistemic Logic)*

- **How:**
  - Vary the number of survey questions (0, 1, 2, 4, 8, 12) and how they are chosen: information-gain
    (Bayes) vs random.
  - Measure:
    - accuracy of the posterior over the 360 archetype maps;
    - the manifesto chosen and the resulting win rate;
    - net value (win-rate gain per ₫ spent);
    - the epistemic verdicts per round: "knows a win is possible", "knows a winning plan exists", "inevitable
      defeat".
  - Compare with the doubled query price (₫5 → ₫10).
- **Necessary?** Yes. The query price just doubled, and earlier work found archetypes cheap to learn (about 2
  questions). We need to know whether surveys are still worth buying.
- **Infer:**
  - the information-sufficiency curve, i.e. how many questions are enough;
  - the gap between "a win is possible" and "the player knows how to win" (knowing-that vs knowing-how);
  - whether ₫10 questions crowd out workers.
- **Lecture link:**
  - Kripke possible worlds = archetype maps consistent with the survey log.
  - K_p φ = φ holds in all worlds the player cannot rule out.
  - Knowing-that (∀w ∃π) vs knowing-how (∃π ∀w), measured per round.

## X6 — Strategy tournament, incl. the source document's strategies (§5.3) *(Core; Proximal Operator, ADMM, Distributed Learning)*

- **How:** play paired games with:
  - **I People Pleaser:** pick the 1–2 regions that together hold a majority and spend everything (media and
    workers) there, with the manifesto blended from those regions.
  - **II Balanced / decision-driven:** the Bayes planner decides whether to survey and whether to spend on workers
    and media, and optimises b.
  - **III Strong-Arm:** a random or fixed manifesto pushed with aggressive spending.
  - Baselines: uniform, top-3, random, and the oracle (full information).
  - Manifesto optimisers: projected gradient (simplex projection = prox of the indicator), consensus-ADMM across
    the 4 regions, and a "blend my hunches" equivalent.
  - **IV Opponent-aware:** deferred. There is no opponent yet. The engine is player-keyed, so a two-scripted-player
    variant is possible as an optional extension.
- **Necessary?** Yes. This is the headline "what wins" result, and the source document asks for exactly these
  strategies.
- **Infer:**
  - a ranking with confidence intervals;
  - whether getting the manifesto right beats spending money (II vs III);
  - whether concentrating on a coalition beats spreading thin (I vs II);
  - the gap to the oracle, i.e. the price of not knowing;
  - whether ADMM matches centralised projected gradient, which confirms the distributed optimisation is correct.
- **Lecture link:**
  - Projected gradient = proximal gradient with the simplex indicator.
  - Consensus-ADMM: each region optimises its own b_r, and they agree on a common b. This is distributed
    learning with a consensus constraint.
  - We report the convergence of the primal and dual residuals.

## X7 — Budget and price sensitivity *(Recommended)*

- **How:** run a grid of treasury {₫500, 750, 1000} × worker fee {₫10, 20, 50, 100} × query fee {₫5, 10}, using the
  oracle planner and the Balanced player. Record win rate, spending mix (workers / media / surveys) and the
  dominant pattern.
- **Necessary?** Recommended. It documents the effect of the owner's price choices (₫20 and ₫10) against
  alternatives, and shows the player what a bigger treasury buys.
- **Infer:**
  - diminishing returns of money;
  - where workers stop being worth it relative to media;
  - whether ₫20 workers create a dominant strategy, and at what treasury.
- **Lecture link:** none directly (economics of the design). It supports X0.

## X8 — Temporal properties of campaigns *(Recommended; Temporal Logics)*

- **How:**
  - Over the per-round traces from X6, evaluate these formulas:
    - ◇Lead: you lead at some round;
    - □(margin > −δ): you are never far behind;
    - Comeback = ¬Lead U (Lead ∧ ◇Victory);
    - STL robustness: the signed distance from 50% at election day.
  - Also check the CTL-style branching properties over the batched possible futures: "there is a future in which
    you win" (EF Victory) vs "every future wins" (AF Victory). Record the round at which defeat becomes
    inevitable.
- **Necessary?** Recommended. It is the main temporal-logic use, it is cheap (it reuses X6 traces), and it gives the
  report its "story" statistics.
- **Infer:**
  - how often games are decided early;
  - whether comebacks happen;
  - the "point of no return" round per strategy;
  - whether STL robustness predicts the final margin.
- **Lecture link:**
  - LTL operators ◇, □, U over linear traces.
  - CTL path quantifiers E/A over the branching tree of futures.
  - STL quantitative robustness.

## X9 — Scale and GPU performance *(Core, short)*

- **How:**
  - Run time per round and peak VRAM at N ∈ {10⁴, 10⁵, 10⁶}, single world and batched (B ∈ {1, 8, 32}), on
    cuda:2 against the CPU reference.
  - Check that the new cascade (edge-parallel hop with the attempt cap) stays O(edges) per round.
- **Necessary?** Yes, but short. It verifies the "GPU-first, up to 1M voters" requirement under the new mechanic.
- **Infer:** scaling curves; the largest batch that fits on one 24 GB GPU; the CPU→GPU speed-up.
- **Lecture link:** none directly. It supports the linear-algebra claim that the work is sparse matrix–vector
  products.

---

## Summary table

| ID | Name | Priority | Main lecture link | Approx. GPU cost |
|---|---|---|---|---|
| X0 | Balance gate at new prices | Core | (calibration) | low |
| X1 | Network structure & consensus speed | Core | Linear algebra, consensus | medium |
| X2 | Stubborn agents, zealots, W-MSR | Core | Resilient consensus (MAS algorithms) | medium |
| X3 | Worker cascade threshold | Core | Infection models | low |
| X4 | Worker timing & placement (incl. centrality ablation) | Recommended | Infection models, influence maximisation | medium |
| X5 | Value of information at ₫10 | Core | Epistemic logic | medium-high (logic is CPU-heavy) |
| X6 | Strategy tournament (§5.3 strategies) | Core | Prox / ADMM / distributed learning | high |
| X7 | Budget & price sensitivity | Recommended | (design economics) | medium |
| X8 | Temporal properties of campaigns | Recommended | Temporal logics | low (reuses X6) |
| X9 | Scale & GPU performance | Core (short) | (sparse linear algebra) | low |

**Lectures covered:**
- Epistemic Logic: X5.
- Temporal Logics: X8.
- Linear Algebra: X1, X9.
- Proximal Operator: X6.
- Distributed Learning / ADMM: X6.
- MAS Algorithms (consensus, W-MSR): X1, X2.
- Infection Model: X3, X4.

**Owner decisions requested:**
- keep, drop or modify each experiment;
- seeds per cell (200 core / 100 sweeps proposed);
- whether to run the optional two-player "opponent-aware" variant;
- whether the old P7 reports should be archived to `reports/archive/v2.1/`. (yes archive them)
