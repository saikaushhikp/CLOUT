# Bounded Trace Diagnostics v1

**Scope:** deterministic diagnostics over one supplied finite completed episode trace. This is not a theorem prover, future-action search, or exhaustive epistemic model checker. Focused logic tests passed in the Week 4 coding-agent run; see [week4_handoff.md](week4_handoff.md) for the environment and full execution record. Reviewer rerun remains recommended.

## Trace time and audit events

`CanonicalTrace` contains exactly `H + 1` predicate snapshots. Index `0` is the initialized state; index `t` is the state after `t` accepted transitions; the configured terminal horizon is state index `H`. Bounds count transition distance and include the endpoint: `F_<=2 p` at time zero checks indices 0, 1, and 2. At `H=0`, the interval is just index 0.

Audit-event `sequence` is independent of temporal time. Rejected proposals remain safe audit events at their unchanged `round_index`; they do not add a predicate snapshot. The adapter `CanonicalTrace.from_game_state` accepts a completed fixed-horizon `GameState`, folds its accepted `TraceRecord`s by round, and folds rejection atomicity checks at that same state index. It exports only predicates, safe action/error summaries, and reproducibility metadata; it never emits voter arrays, susceptibilities, or private event attributes. The evaluator itself consumes only the canonical trace.

## Atoms and unknowns

Each state has exactly these boolean-or-null atoms:

| Atom | Meaning |
| --- | --- |
| `Terminal` | Fixed horizon reached at this state. |
| `Victory` | Defined only at terminal; true iff the engine's vote share is strictly greater than 0.5. |
| `BudgetNonnegative` | Remaining authoritative budget is nonnegative. |
| `StateValid` | Voter supports and susceptibilities are finite and in their M0 domains. |
| `ActionAtomic` | At least one rejected action was audited at this index and all such checks preserved the authoritative state fingerprint; `null` if no rejection check applies. |

`null` means unknown/not applicable, never false. `Terminal` is false at every preterminal state; `Victory` is null there. At a terminal state `Victory` must be true or false. `from_game_state` obtains initial state-validity from the engine's initial voter snapshot and transition atoms from engine trace records.

The evaluator uses strong Kleene three-valued boolean propagation: false dominates conjunction, true dominates disjunction, and otherwise unknown propagates. Implication is `not left or right`. For `G`, any false value violates; absent false, an unknown makes the result unknown. For `F`, any true value satisfies; absent true, an unknown makes the result unknown; otherwise it is violated. A missing terminal on an otherwise valid completed trace is therefore a falsified `F_<=H Terminal`, not malformed input.

## Supported closed formula AST

The API accepts only `Atom`, `And`, `Or`, `Not`, `Implies`, `Globally`, and `Eventually` nodes. Unknown atom names, invalid bounds, arbitrary Python objects, and malformed trace shapes raise `LogicInputError`; malformed input is not converted to a property violation.

The three M0 diagnostics are:

1. `G(BudgetNonnegative and StateValid)` over indices `0..H`.
2. `F_<=H Terminal` over indices `0..H`, including the terminal endpoint.
3. `Terminal -> (Victory or not Victory)` evaluated at every trace state (the
   trace-wide check is equivalent to `G` of the pointwise implication).

The third diagnostic checks that terminal outcome classification is defined; it is an exhaustiveness tautology, **not** a guarantee of victory. A trace where `Victory` is false at terminal satisfies it.

`LogicResult` reports stable formula/name, `SATISFIED`/`VIOLATED`/`UNKNOWN`, checked interval, first witness or counterexample where applicable, and a short explanation. `to_public_dict()` returns JSON-safe primitives. Repeated evaluation of an identical trace is deterministic.

## Fixtures and human-run check

Hand-computable fixtures are in `tests/fixtures/logic/`: valid terminal at H, safety failure, missing terminal, same-index rejection audit, unknown atoms, and exact-half defeat. The test module is `tests/unit/test_logic.py`. The coding-agent focused logic suite passed; a reviewer can rerun it with:

```bash
PYTHONPATH=src python -m unittest tests.unit.test_logic -v
```

Expected: all focused logic tests pass. The command was executed successfully by the coding agent; reviewers should rerun it to confirm in their environment.
