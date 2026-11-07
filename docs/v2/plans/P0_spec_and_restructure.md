# P0 — Spec freeze and repository restructure (Day 1)

**Goal:** a clean v2 branch with the frozen spec, decision register, phase plans and an importable,
tested package skeleton. No model code yet.

## Tasks
1. Branch `v2-revamp` from `master@e3f7760`; tag `v1-final` (local; push only when the owner asks).
2. Archive v1: `docs/{spec,plans,decisions,Assumptions}` → `docs/archive/v1/`, `reports/*` →
   `reports/archive/v1/`; remove v1 `src/dumbocracy`, `tests`, `examples` (they stay in git history).
3. File sources under `docs/v2/sources/` (graph_construction.md, ChatGPT ideation html + extracted txt).
4. Write `docs/v2/technical_spec_v2.md`, `docs/v2/decisions_v2.md`, `docs/v2/plans/P0..P7`.
5. Package skeleton `src/dumbocracy/` with sub-packages from the master plan, plus two real modules:
   - `config.py`: frozen dataclasses for every ⚙ parameter, validation, `SeedStreams` (BLAKE2b →
     per-stream `torch.Generator`), device resolution (`auto|cpu|cuda:N`).
   - `sectors.py`: sector names, region names, archetype library, personality library.
6. `pyproject.toml`: core dependencies `numpy`, `torch`; extras `web` (fastapi, uvicorn), `analysis`
   (matplotlib, scipy), `test` (pytest, httpx), `llm` (litellm, optional); pytest config + markers.
7. Tests: `tests/unit/test_config.py`, `tests/unit/test_sectors.py`.
8. `docs/v2/README.md`: how to run, where things are.

## Exit criteria
- `conda run -n mas python -m pytest -q` passes.
- The owner has reviewed the spec (blocking for P1).

## Handoff (2026-11-07)
- Done: branch `v2-revamp`, local tag `v1-final` (not pushed), v1 archived via `git mv`/`git rm`
  (history intact), sources filed, spec v2.0 + decisions + P0–P7 plans written, package skeleton,
  `config.py` (all ⚙ defaults, validation, BLAKE2b seed streams → `torch.Generator`, device choice),
  `sectors.py` (libraries). pytest 9.2 installed into `mas`; package installed editable.
- Tests: `conda run -n mas python -m pytest -q` → **19 passed**.
- Finding: the GPUs are shared. At P0 time cuda:1 had ~0.1 GB free and cuda:0 ~10 GB, so probing a full
  device with `mem_get_info` raised CUDA OOM. `resolve_device("auto")` now reads free memory via
  `nvidia-smi` matched by GPU UUID (no CUDA context on busy devices) and picked `cuda:2`. Large runs
  must check free VRAM at runtime and should pin an explicit device.
- Not done (needs owner): commit/push, spec review sign-off. `.gitignore` left as the owner had it.
