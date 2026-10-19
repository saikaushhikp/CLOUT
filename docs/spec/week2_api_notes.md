# Week 2 API Notes

These executable foundations implement the frozen M0 contracts. They do not
implement the Game Engine, observation projection, strategy API, or logic
evaluator.

## Modules and ownership

| Module | Public interface | Ownership / mutation guarantee |
| --- | --- | --- |
| `dumbocracy.config` | frozen `SeedConfig`, `GraphSpec`, `DynamicsConfig`, `ActionConfig`, `GameConfig` | Construction validates before a model is created. `SeedConfig.derive` uses BLAKE2b, not global RNG/hash. |
| `dumbocracy.graph` | frozen `ElectorateGraph`, `erdos_renyi_graph`, `load_edge_list_fixture` | Canonical index-pair edges only; no dense adjacency storage. |
| `dumbocracy.mas` | frozen `VoterState`, `advance_support`, `player_vote_share` | Transition returns a fresh voter tuple and never changes the input tuple. |
| `dumbocracy.campaign` | frozen action, ledger/context, validation and commit records | `validate_action` is pure. `commit_action` accepts only accepted validation and returns a new context. |

## Campaign hand-off

`ActionContext` carries ordered node IDs, immutable `ResourceLedger`, and the
round-local `broadcast_used` flag. The Week 3 engine creates a fresh context at
each round boundary, so it owns resetting that flag and clearing contributions
after calling `advance_support`. A `CampaignCommit` supplies a contribution
tuple aligned exactly to graph node order. Duplicate targeted IDs are rejected
with `INVALID_TARGET`; they are never normalized or double charged.

`no_op` shares the action union for convenience but is an engine-level zero-cost
action. It produces no contribution and does not implement round advancement;
that belongs to Week 3.

## Errors and observation boundary

Configuration/graph/dynamics failures use `DumbocracyError` with stable
`ErrorCode`. Rejected campaign actions return `ActionValidation(accepted=False,
error_code=...)`; they do not mutate inputs. No `Observation` constructor is
implemented in Week 2. Observation serialization and hidden-state leakage tests
remain a Week 3 integration gate. None of these modules imports a future engine,
UI, or strategy layer.

## Commands

Run the core suite with:

```bash
PYTHONPATH=src /data/ai23btech11017/miniconda3/envs/mas/bin/python -m unittest discover -s tests -v
/data/ai23btech11017/miniconda3/envs/mas/bin/python examples/week2_gpu_smoke.py
```

The smoke demo selects CUDA only when PyTorch reports it usable; otherwise it
runs the CPU reference and reports that state.
