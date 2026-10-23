# Observation-Only Strategy API v1

**Execution status:** implementation passed its focused contract tests and the
paired-strategy example completed 18 episodes in the 2026-10-23 coding-agent
run. See [week4_handoff.md](week4_handoff.md) for environment and full-suite
results; reviewer rerun is recommended.

## Contract

```python
Strategy.choose(
    observation: Observation,
    history: tuple[HistoryEntry, ...],
    legal_actions: tuple[ActionDescriptor, ...],
    rng: random.Random,
) -> CampaignAction
```

`observation` is the current detached public observation. `history` contains
only prior decision boundaries: each `HistoryEntry` has the public observation
at that boundary and a `PublicActionSummary`. It does not include the current
observation. The runner appends a history entry only after an accepted action;
a rejected proposal is captured in the experiment record and terminates that
episode as a policy failure. The runner never calls a policy on a terminal
state.

The policy receives only `Observation`, immutable `HistoryEntry` values, public
`ActionDescriptor`s, and a per-episode `random.Random` seeded from
`GameConfig.seeds.strategy`. It receives no `GameEngine`, `GameState`, voter
tuple, hidden seeds, private events, tensor, or result fields outside its
public history. Its returned `CampaignAction` is only a proposal; `GameEngine`
validates every proposal. A clear `StrategyError` is raised if a policy has no
usable supplied action rather than manufacturing an action.

## Baselines

- `NoOpStrategy`: chooses supplied `no_op` only.
- `FirstLegalStrategy`: stable priority `targeted_contact`, `broadcast`,
  `no_op`; for targeted contact it picks the lexicographically first eligible
  public ID.
- `SeededRandomLegalStrategy`: samples from supported descriptors sorted in
  that same stable priority, and samples a target from the sorted eligible ID
  list using only the passed RNG.

These are reproducibility and integration controls, not optimized policies.
Python RNG sequences are reproducible for a fixed Python/software version and
the recorded strategy seed; they are not promised identical across all Python
versions.

## Example

```python
import random
from dumbocracy.strategies import FirstLegalStrategy

proposal = FirstLegalStrategy().choose(
    observation=current_observation,
    history=prior_public_history,
    legal_actions=current_observation.legal_actions,
    rng=random.Random(strategy_seed),
)
response = engine.step(authoritative_state, proposal)  # engine validates it
```

The `engine` and authoritative state in this last line belong to the runner,
not the strategy. Tests are in `tests/unit/test_strategies.py` and the
human-run focused command is:

```bash
PYTHONPATH=src python -m unittest tests.unit.test_strategies -v
```
