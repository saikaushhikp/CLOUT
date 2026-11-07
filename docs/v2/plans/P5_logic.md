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
