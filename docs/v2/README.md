# Dumbocracy v2

| What | Where |
|---|---|
| Model + contracts (source of truth) | [technical_spec_v2.md](technical_spec_v2.md) |
| Decisions (owner U1–U17, implementer I1–I5, calibration changes) | [decisions_v2.md](decisions_v2.md) |
| Phase plans P0–P7 (tasks, tests, exit criteria, handoffs) | [plans/](plans/) |
| Original design inputs | [sources/](sources/) |
| Player FAQ (blending, hiring) | [faq.md](faq.md) |
| v1 (archived) | `docs/archive/v1/`, `reports/archive/v1/`, git tag `v1-final` |

```bash
conda activate mas
python -m pip install -e '.[web,analysis,test]'
python -m pytest -q            # GPU tests auto-skip without CUDA; add -m "not slow" for a quick run
```

Course material used by the model lives in `just_see/lectures/` (consensus, resilient consensus,
SI/SIR, prox/ADMM, temporal and epistemic logic).

## Play the game
```bash
scripts/run_game.sh        # engine :61008 + Next.js :61007 (production build)
scripts/run_game.sh dev    # hot-reload frontend
# remote: ssh -L 61007:127.0.0.1:61007 <user>@<host>  then open http://127.0.0.1:61007
```
API contract: [web_api_v2.md](web_api_v2.md).
