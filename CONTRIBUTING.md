# Contributing to Clout

Thank you for helping improve Clout. This guide covers setup, project conventions, testing and how to open a pull
request. If anything here is unclear or out of date, opening an issue about it is a useful contribution too.

## Contents

- [Ways to contribute](#ways-to-contribute)
- [Set up your environment](#set-up-your-environment)
- [Project tour](#project-tour)
- [Ground rules](#ground-rules)
- [Running the tests](#running-the-tests)
- [Working on the web game](#working-on-the-web-game)
- [Adding or changing an experiment](#adding-or-changing-an-experiment)
- [Coding style](#coding-style)
- [Commits and pull requests](#commits-and-pull-requests)
- [Reporting bugs and suggesting features](#reporting-bugs-and-suggesting-features)

## Ways to contribute

- **Report a bug.** Include the seed and settings so we can replay it (see [below](#reporting-bugs-and-suggesting-features)).
- **Improve the game.** Clearer screens, better explanations, accessibility, mobile layout.
- **Improve the model or players.** New strategies, faster simulation, better planners.
- **Add an experiment.** Test a new question about campaigns on networks.
- **Improve the docs.** Fix anything confusing in the README, HOW_TO_PLAY or the code docstrings.

For anything larger than a small fix, **open an issue first** to agree on the approach.

## Set up your environment

```bash
git clone https://github.com/saikaushhikp/CLOUT.git
cd CLOUT

conda env create -f environment.yml      # the "mas" environment: Python, Node.js, PyTorch, FastAPI, pytest
conda activate mas                        # and an editable install of the dumbocracy package

cd web && npm ci && cd ..                 # web game dependencies
```

Python packages come from `requirements.txt`, which lists only direct dependencies with compatible version ranges.
If you add a dependency, add it to both `requirements.txt` and `pyproject.toml` with the same range. PyTorch comes
as the default build for your platform; for a specific CUDA version, run the command from
[pytorch.org](https://pytorch.org/get-started/locally/) inside the environment.

Check that everything works:

```bash
python -m pytest -q -m "not slow and not gpu"   # about a minute
scripts/run_game.sh                             # then open http://127.0.0.1:61007
```

**On a shared machine,** limit CPU threads, or electorate generation can slow down sharply:
`export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`.

## Project tour

| Path | What lives there | Read first |
|---|---|---|
| `src/dumbocracy/electorate/` | Regions, friendship network, preferences, personalities, zealots | `electorate.py` |
| `src/dumbocracy/dynamics/` | Opinion drift, worker cascade, media | `update.py`, `campaign.py` |
| `src/dumbocracy/voting.py` | Agreement, vote probability, the election | — |
| `src/dumbocracy/game/` | The rules engine: immutable states, actions, treasury, observations | `engine.py`, `actions.py` |
| `src/dumbocracy/information/` | Surveys, polls, the player's knowledge log | — |
| `src/dumbocracy/strategies/` | Oracle, Bayesian smart player, manifesto optimisation, planner | `smart.py` |
| `src/dumbocracy/logic/` | Possible worlds, epistemic and temporal logic, the debrief | `epistemic.py` |
| `src/dumbocracy/api/` | The FastAPI server used by the web game | `app.py` |
| `src/dumbocracy/config.py` | Every rule, price and default parameter | — |
| `web/` | Next.js web game | `lib/api.ts`, `components/GameApp.tsx` |
| `tests/` | `unit/`, `integration/` and `gpu/` tests | `conftest.py` |
| `experiments/` | Experiment scripts and shared helpers | `p7_common.py` |
| `docs/v2/` | Technical spec, decisions, API contract | `technical_spec_v2.md` |

## Ground rules

These keep the game fair and the results trustworthy. Pull requests that break them will not be merged.

1. **Hidden information stays hidden.** A player, human or computer, sees only its *observation*. Observations, API
   responses, logs and error messages are built by allow-listed serialisers. Never add a field that exposes voter
   preferences, susceptibilities, zealots, edges, the archetype or personality maps, block probabilities or seeds.
   The privacy leak test (`tests/privacy_scan.py`) must keep passing.
2. **Every game must replay exactly.** All randomness comes from **named seeded streams** in `rng.py`
   (`regions`, `graph`, `workers`, `cascade`, `media`, …). Never call the global `torch.rand` or `random` directly.
   A new random step gets its own stream name, so it can't shift the draws used elsewhere.
3. **Draw one random number per voter per step, whatever happens.** This keeps streams aligned across branches and
   lets the planners compare plans on identical luck.
4. **No Python loops over voters.** Write tensor operations, and accept an optional leading batch dimension, so the
   same code runs one world on a CPU or many on a GPU.
5. **CPU and GPU must agree.** On CPU the batched simulator is bit-identical to the single-world one. On GPU,
   discrete outcomes (hires, conversions, winners) must match. Add a GPU parity test when you touch the core.
6. **Money is exact.** Treasury amounts are `Decimal` with two places. Never use floats for money.
7. **Rejected actions change nothing.** The engine returns the same state with an error code. Keep it that way.
8. **Parameters live in `config.py`.** Don't hard-code a price, probability or threshold anywhere else.

## Running the tests

### Python

```bash
python -m pytest -q -m "not slow and not gpu"   # quick run, about a minute
python -m pytest -q                             # everything; GPU tests skip automatically without CUDA
python -m pytest tests/unit/test_voting.py -q   # one file
```

Markers: `gpu` needs a CUDA device; `slow` marks long statistical or scale tests.

**Add tests with every change.** New behaviour needs a unit test. Bug fixes need a test that failed before the fix.
For statistical code, test properties (means, bounds, invariants such as "preferences stay on the simplex") with fixed
seeds, not single exact numbers.

### Web

```bash
cd web
npm test                 # unit tests (Vitest)
npm run typecheck        # TypeScript
npm run e2e              # browser tests (Playwright); needs scripts/run_game.sh running
```

## Working on the web game

- Start the engine and the dev server together with `scripts/run_game.sh dev`. The web app proxies `/api/...` to the
  engine; set `DUMBOCRACY_API` to point it elsewhere.
- To work on the UI without Python, use the in-browser mock engine:
  `NEXT_PUBLIC_DUMBOCRACY_MOCK=1 npm run dev`.
- The API contract is [docs/v2/web_api_v2.md](docs/v2/web_api_v2.md). If you change a route, update that file, the
  types in `web/lib/types.ts`, and the mock in `web/lib/mockApi.ts` together.
- The app uses a recent Next.js with breaking changes. Read `web/AGENTS.md` before changing framework code.
- Check new screens in both light and dark themes and at phone width.

## Adding or changing an experiment

1. **Write down what you expect before running it.** Add a line to `reports/v3/predictions.md`.
2. **Use the shared helpers** in `experiments/p7_common.py` for device choice, 95% confidence intervals, paired tests
   and provenance.
3. **Compare arms on the same worlds.** Use the same seeds and common random numbers. Use a seed block that no other
   experiment uses.
4. **Save raw rows and a JSON summary** with the exact command, seeds, device, git revision and wall time. Draw
   figures only from those files.
5. **Run a short CPU-vs-GPU parity check** (three seeds) before the main GPU run.
6. **Report honestly.** Give means with confidence intervals, say which differences are significant after correction,
   and list the caveats.

## Coding style

- **Python:** follow the style of the surrounding code: type hints, `from __future__ import annotations`, frozen
  dataclasses for configuration and state, and docstrings that cite the relevant spec section.
- **Names:** use the model's terms: *manifesto*, *archetype*, *personality*, *worker*, *convert*, *zealot*,
  *observation*.
- **TypeScript:** strict types. Keep components small, and share colours through the CSS variables in
  `web/app/globals.css`.
- **Docs:** short sentences and plain words. A first-time player should be able to read `HOW_TO_PLAY.md`.

## Commits and pull requests

1. Create a branch from `main`: `git checkout -b fix/short-description`.
2. Make focused commits with clear messages in the imperative mood, for example
   `Cap media pull so preferences stay on the simplex`.
3. Before opening the pull request:
   - `python -m pytest -q -m "not slow and not gpu"` passes;
   - `cd web && npm test && npm run typecheck` passes, if you touched `web/`;
   - docs are updated if behaviour, rules or the API changed.
4. In the pull request, explain **what** changed and **why**, link the issue, and include screenshots for UI changes.
   If a change shifts any game numbers (prices, defaults, balance), say so: it changes results in the report.

Pull requests are reviewed by the maintainers. Small, focused pull requests get merged fastest.

## Reporting bugs and suggesting features

Open an issue on [GitHub](https://github.com/saikaushhikp/CLOUT/issues).

**For a bug, include:**
- what you did, what you expected and what happened;
- the **seed, number of voters, rounds and starting money**, so the world can be rebuilt exactly;
- whether the engine ran on CPU or GPU (random draws during play are device-specific);
- the error message or a screenshot.

**For a feature,** describe the problem it solves for players or researchers before the solution.

---

By contributing, you agree that your contributions are licensed under the project's [MIT License](LICENSE).
