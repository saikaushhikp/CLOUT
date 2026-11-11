# v3 predictions (pre-registered)

Registered: **2026-11-10**, in Phase 0, before any v3 experiment was run (plan §1.3). Plan revision: `docs/v2/plans/implementation_plan.md` (FINAL, 2026-11-10), repository at `d8d9c4e+uncommitted`.

Each "Predict" line below is copied **unchanged** from the plan (the text between the markers is verbatim,
including its line breaks). The verdict line is filled in only after the experiment has run: confirmed /
refuted / partial / not tested, with the number and the JSON file it comes from (§14.3 item 11).

Experiments whose plan section has no "Predict" line say so; no prediction was invented for them.

## Phase 1 — Calibration

### X0 — Balance gate at the new prices *(Core; P9, unchanged in substance)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** at ₫20, "split" wins ≥ 80% (gate 4a fails), as the P8 price check found at ₫10–₫50.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X10 — Balance across the player-chosen N and T *(Core; new)*

<!-- verbatim from implementation_plan.md -->
- **Predict:**
  - Δshare per worker ∝ 1/N (log–log slope ≈ −1), while media's Δshare per ₫ is flat in N.
  - Workers dominate at N = 10³ (the "one worker converts most voters" problem) and are near-worthless at 10⁶.
  - T = 1 favours media, because a cascade gets only 1 hop.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X7 — Budget and price sensitivity *(Recommended; P9)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** money has diminishing returns; workers stop being worth it relative to media somewhere between
  ₫50 and ₫100 at N = 10⁴ (P8 table).
<!-- end verbatim -->

**Verdict:** not tested yet.

### X23 — Worker-rule ablations *(Recommended; new)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** cap ∞ and rule γ bring back cascade saturation (the manifesto stops mattering, as P8 saw with γ
  alone). A cap of 1 makes workers subcritical.
<!-- end verbatim -->

**Verdict:** not tested yet.

## Phase 2 — Network and cascade mechanics

### X1 — Network structure and consensus speed *(Core; P9, with a fix)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** the decay follows ā(1 − μ₂(D⁻¹A)), not λ₂(L)/k̄ (as E1 found). Under the P8 rules, high p_g raises
  the wins of a single nationwide manifesto.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X2 — Stubborn agents, zealots, W-MSR *(Core; P9 + polarisation metric)*

<!-- verbatim from implementation_plan.md -->
- **Predict:**
  - The worker anchor value is a sizeable share of the worker value: Arm B Δshare / full-cascade Δshare is
    between 0.2 and 0.6, to be measured.
  - Pinned agents stop consensus (disagreement does not decay to 0).
  - W-MSR trims workers, because they sit far from their friends, so it **lowers** the worker value. That
    makes W-MSR the network's defence against *your* campaign.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X3 — Worker cascade threshold and reach *(Core; P9, with a fix)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** growth crosses 1 near λ_max(A D_q) ≈ 1, more sharply than the naive estimates; the cap binds
  mainly above threshold.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X4 — Worker timing and placement *(Recommended; P9)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** buying early is worth more (more hops); centrality picking gives a modest gain under k̄ = 30 ER
  blocks (low degree variance).
<!-- end verbatim -->

**Verdict:** not tested yet.

### X11 — Worker × media interaction *(Core; new, N2)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** the interaction is positive, and larger in canvassing-receptive regions; "media first, then
  workers" beats the reverse.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X18 — Submodularity of the worker value curve *(Recommended; new, N9)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** roughly submodular in k, with possible early *super*-modularity in regions near threshold.
<!-- end verbatim -->

**Verdict:** not tested yet.

## Phase 3 — Strategies and optimisation

### X6 — Strategy tournament *(Core; P9, with a fix)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** II > I > III; the price of not knowing is ≤ 0.015 E.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X12 — Broad vs narrow ("viral") manifesto *(Core; new, N3)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** the best λ > 0 on seeds where some coalition region is canvassing-receptive. Joint optimisation
  gains a measurable +0.005–0.02 E over "b first, then plan" (to be measured).
<!-- end verbatim -->

**Verdict:** not tested yet.

### X13 — ORACLE optimality: open-loop plan search *(Core; new, N4)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** the search beats the ORACLE planner by ≤ 0.01 E on average, with a long tail of seeds where front-loading
  or concentration beats greedy by more.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X14 — Planner calibration *(Core; new, N5)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** positive bias (over-optimism) that shrinks with world size; "even" undervalues front-loaded worker
  buys.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X19 — Manifesto geometry and a third optimiser *(Recommended; new, N10)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** b* is closer to a 2–3-region coalition blend than to the mean or median; there are several local
  optima (multi-start matters); mirror descent converges in fewer iterations than PG on interior optima.
<!-- end verbatim -->

**Verdict:** not tested yet.

## Phase 4 — Information and logic

### X5 — Value of information at ₫10 *(Core; P9, with a fix)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** about 2 questions are still enough; beyond that every question is a net loss, larger than at ₫5.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X17 — Learning personalities by probing *(Optional; new, N8)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** probing lowers τ entropy measurably, but the 400-voter national poll (SE ≈ 2.4 pp) limits it.
  The net win-rate effect is ≈ 0 or negative.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X16 — Reliability and calibration of epistemic verdicts *(Recommended; new, N7)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** "guaranteed" becomes rarer and more reliable as the world count grows; p̂ is over-confident at 16
  worlds.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X8 — Temporal properties of campaigns *(Recommended; P9)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** most winners come from behind (as v2 found); the point of no return is round 2–3 for weak strategies.
<!-- end verbatim -->

**Verdict:** not tested yet.

## Phase 5 — Robustness and explanation

### X15 — Which elections are unwinnable? *(Core; new, N6; post-hoc, nearly free)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** losses concentrate where the two largest regions have distant archetypes and a fragmented network
  (low p_g); the largest region's personality decides the best channel.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X20 — Luck vs skill: variance decomposition *(Recommended; new, N11)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** world draw > worker picks > cascade > media ≈ election at N = 10⁴. Election-day noise is tiny
  (sd ≈ 0.005).
<!-- end verbatim -->

**Verdict:** not tested yet.

### X21 — Robustness to model misspecification *(Recommended; new, N12)*

<!-- verbatim from implementation_plan.md -->
- **Predict:** the planner is most sensitive to z and β_v (they define who votes); the least sensitive is κ.
<!-- end verbatim -->

**Verdict:** not tested yet.

### X22 — Global sensitivity analysis of the design *(Optional; new, N13)*

The plan gives no "Predict" line for this experiment; none is registered.

**Verdict:** not tested yet.

## Phase 6 — Scale

### X9 — Scale and GPU performance *(Core, short; P9)*

The plan gives no "Predict" line for this experiment; none is registered.

**Verdict:** not tested yet.

## Not scheduled

### X24 — Human playtest (+ LLM player) *(**Dropped, D8**; LLM player not approved, D2)*

Dropped (D8); no prediction registered.

**Verdict:** not tested yet.

### X25 — Two-party meta-game and manifesto convergence *(Deferred; new)*

Deferred (D1); no prediction registered.

**Verdict:** not tested yet.
