# Week 1 Hand-off

**Status:** M0 complete. Team approval was reported on 2026-10-19.

## What each workstream may use after approval

| Workstream | Contract provided | Readiness |
| --- | --- | --- |
| MAS/graph | Voter schema, sparse graph, synchronous equation | Ready |
| Resources | Action/ledger schema, validate-then-commit | Ready |
| Information | Hidden-state and detached-observation boundary | Ready |
| Game engine | Strict lifecycle, result/error schema | Ready |
| Strategies | Observation-only policy and scenario evaluation | Ready |
| UI prototype | Observation, legal-action, result payloads | Ready |
| Temporal model | Predicates and bounded LTL/knowledge semantics | Ready |

## API hand-off

Use the conceptual signatures and ownership table in
[data_contracts.md](data_contracts.md). The engine sequences interfaces but
must not expose `GameState` to strategies or UI. Rejected actions must leave
the ledger/state untouched. See the fixture README for toy data.

## Approved review record

The project owner confirmed approval from all five members on 2026-10-19.
Individual names and comments were not supplied, so the record preserves that
fact rather than inventing reviewer identities.

| Review lens | Reviewer identity supplied | Outcome |
| --- | --- | --- |
| MAS/graph | No | approve |
| Campaign/resource | No | approve |
| Information/game engine | No | approve |
| Logic/verification | No | approve |
| Strategy/experiment/UI | No | approve |

## Week 2 boundary

The contracts are frozen at M0. Week 2 may implement the interfaces and tests,
but may not change state/action/visibility semantics without a new decision.

