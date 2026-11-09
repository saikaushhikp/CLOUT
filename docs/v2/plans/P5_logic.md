# P5 — Temporal and epistemic reasoning (Days 7–8)

**Spec:** §10. **Owner area:** logic/verification (Psk). **Depends on:** P3 and P4 (policy set Π,
batch simulator).

## Deliverables
- `logic/worlds.py`: `consistent_thetas(knowledge_log)` (exact over the 8640 θ using the tolerance
  ε), `sample_worlds(knowledge_log, M, gen)` → a batched world tensor bundle; a rejection step that
  re-simulates each sampled world under the player's past actions and keeps the worlds whose answers
  fall within ε.
- `logic/epistemic.py`: `PossibleVictory`, `GuaranteedVictory`, `InevitableDefeat`, `K_p(φ)`,
  `M_p(φ)` (possibility = ¬K¬), `Viable_t` over (W, Π). Each result reports
  {status TRUE/FALSE/UNKNOWN, |W|, |Π|, scope text, witness policy / counter-world}.
- `logic/temporal.py`: a bounded LTL evaluator over per-round traces (atoms: Lead, Victory,
  BudgetLeft, Viable); STL robustness (min/max semantics) with the margin as the signal; CTL-style
  ∃◇ / ∀◇ over the batched worlds × policies tree (depth = rounds).
- `logic/report.py`: a per-game analysis bundle for the UI "why" panel and the final scenario table.
- `experiments/info_sufficiency.py`: queries spent vs fraction of seeds where
  K_p(GuaranteedVictory) holds, with an optimal query policy vs a random query policy.

## Tests
- Forced fixtures: a huge budget plus an aligned electorate gives GuaranteedVictory TRUE; zero budget
  plus an opposed electorate gives InevitableDefeat TRUE; two θ that are indistinguishable under the
  log but need different manifestos give Possible TRUE and Guaranteed FALSE.
- K/M duality: M_p φ == ¬K_p ¬φ on random fixtures.
- S5 sanity: the true world is always in the consistent set (reflexivity) when the tolerance exceeds
  the numeric error.
- LTL/STL known traces: the robustness sign matches Boolean satisfaction.

## Exit criteria
The scenario table (Possible / Guaranteed / Inevitable / K_p(Victory) / K_p(Defeat) across
information levels) is generated; the information-sufficiency curve is plotted.

## Handoff (2026-11-08)
- Owner decisions: R13 (information model unchanged; the focus is what stays hidden: τ, zealots, network)
  and R14 (logic in BOTH the web debrief timeline and the research layer).
- **Built (sub-agent I):** `logic/{knowledge,worlds,epistemic,temporal,report}.py`.
  - `observation_at`: the player's knowledge at each decision point, checked field by field against
    engine-captured observations.
  - `consistent_worlds`: Kripke information set I(K_p). σ comes from the posterior; τ is uniform over the 24
    permutations; micro-state is fresh. Each world is replayed through the player's own history and kept only
    if it matches every poll (|E_w − p̂| ≤ 2.5·(SE + 0.01)).
  - Epistemic properties over worlds × a finite policy set (14–16 stationary plans per round):
    - possible = ∃π∃w
    - guaranteed / knowing-how = ∃π∀w
    - knowing-that = ∀w∃π
    - inevitable = ∀π∀w¬
    - K/M with duality M = ¬K¬
    - ground-truth `truth_victory_possible` from the real world, used only for reporting.
  - Temporal: bounded LTL, STL min/max robustness, CTL-style helpers.
  - `epistemic_timeline(...)` per contract (`docs/v2/web_api_v2.md`, R14).
  - Experiments: `experiments/scenario_table.py`, `experiments/info_sufficiency.py`.
- **Built (sub-agent J):** `GET /api/games/{g}/players/{p}/epistemic` (409 before DONE, 503 if the logic module
  is unavailable; cached per game/player under the session lock) and the web debrief "Epistemic timeline"
  panel (truth row + Possible / knowing-that / knowing-how / defeat-inevitable badges per round, best plan,
  knowledge stats, explainer).
- **Lead:** fixed the headline's truth-vs-knowledge sentence. It now cites when winning really became impossible,
  separately from false-hope and missed-chance rounds, with a regression test. Verified live over :61007 on two
  10k games: 409 before DONE, 2–3 s computation, cached in ≈ 0.01 s.
- **Tests:** Python 232 passed (231 + the headline regression); web 39 passed; production build is the real engine.
- **Results** (`reports/v2/scenario_table.{json,md}`, `reports/v2/info_sufficiency.json`,
  `reports/v2/figures/info_sufficiency.png`; 60 seeds and 40 seeds per arm, N = 10k):
  - "Knows guaranteed" was never wrong: 0 of 600 decision points where no plan could truly win.
  - "Possible" errs on the optimistic side (≈ 35 false-possibles per 300 points, 0 missed).
  - P(win | guaranteed at round 1) = 100%.
  - BayesPlanner knows a winning plan at 70–73% of decision points, Top3 at 63–68%.
  - Information sufficiency: σ entropy reaches ≈ 0 after 2 information-gain queries (≈ 8 random queries
    needed), yet "guaranteed at round 1" only moves from 0.65 to ≈ 0.70. Knowledge then grows through the polls
    (≈ 0.68 → 0.80 by round 5), confirming R13: the hidden τ, zealots, network and campaign randomness limit
    *knowing how* to win.
  - The poll-consistency filter accepts 97–100% of drawn worlds; a 400-voter poll (SE ≈ 2.4 pp) barely
    identifies τ.
