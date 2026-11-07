# P3 — Information model and player-keyed game engine (Days 4–5)

**Spec:** §6, §7, §8, §9. **Owner areas:** information/resources (Hem), engine (team). **Depends on:** P2.

## Deliverables
- `information/survey.py`: `global_top3(v)`, `answer_query(v, region_of, s1, s2, r)`,
  `run_poll(v, b, beta_v, z, n_poll, gen)`.
- `information/knowledge.py`: `KnowledgeEntry` (kind, round, payload) and `KnowledgeLog`
  (append-only, JSON-safe).
- `game/ledger.py`: Decimal treasury with an atomic debit.
- `game/actions.py`: `Query`, `Commit`, `Allocate`, `ForfeitRound` plus pure validators returning
  stable error codes.
- `game/state.py`: immutable `GameState` and `PlayerState` (per-player keyed), with phase/round,
  trace records and a public event log.
- `game/engine.py`: `GameEngine.new_game(config, player_ids)`, `apply(state, player_id, action) ->
  StepResult`, round resolution once all players are ready, automatic election, `replay(config, actions)`.
- `game/observation.py`: `Observation.for_player(state, pid)` built from the allowlist, plus
  `to_public_dict()`.
- `players/base.py` (`Player` protocol), `players/scripted.py` (wraps strategies),
  `players/llm_stub.py` (an interface that raises `NotImplementedError` and documents the prompt/obs
  contract for LiteLLM).
- ~~CLI game~~ — superseded by R11: the playable game is the Next.js web app (P3 now also delivers the
  FastAPI service `api/app.py` per `docs/v2/web_api_v2.md` and the first playable web UI).

## Tests
- Phase machine: illegal-phase actions rejected; commit is required before rounds start; the election
  runs after round T.
- Atomic rejection: state fingerprint, treasury and knowledge log are unchanged.
- Costs exact (Decimal), including media cost rounding; insufficient funds rejected.
- Query answer equals a brute-force count; duplicate query in the same round rejected.
- Poll MoE formula; reproducible per seed.
- **Privacy:** a recursive scan of `to_public_dict()` and every API payload finds no forbidden keys and
  no N-length numeric arrays.
- Replay determinism: the same actions reproduce an identical final state and result.
- Two-player smoke test (both scripted): a round resolves only when both are ready; per-player
  observations are isolated.
- Performance: an N = 10k round resolves in < 1 s on GPU.

## Exit criteria
A full game is playable in the browser on :61007 (via `scripts/run_game.sh`); all tests pass; the observation JSON schema is frozen
(`docs/v2/observation_schema.json`) and unblocks P6.

## Handoff (2026-11-07)
- **Backend (sub-agent C):** `information/{survey,knowledge}.py`; `game/{ledger,actions,state,engine,observation}.py`;
  `players/{base,llm_stub}.py`; `api/app.py` (FastAPI, every route in `docs/v2/web_api_v2.md`). The engine
  reproduces `simulate_world` exactly (same seed, manifesto and allocations), rejections are atomic, the
  privacy scan covers every payload, and a two-player smoke test passes. Contract ambiguities were resolved
  as listed in the agent report; the main ones:
  - media price per region is fixed;
  - the seed is shown to the player (they chose it);
  - `graph_sample` returns the whole topology when N ≤ max_nodes (topology is public, R3).
- **Frontend (sub-agent D):** Next.js 16 + React 19 + TypeScript app in `web/` (old Vite app deleted). Screens:
  setup, survey desk + manifesto designer, campaign HQ (region map, orders, polls with MoE band, optional
  round timer), results + debrief (region reveal, oracle comparison, alignment network animation). Uses
  sigma.js/graphology (WebGL) and a mock engine (`NEXT_PUBLIC_DUMBOCRACY_MOCK=1`); `npm test` 26 passed,
  `npm run build` OK.
- **Integration (lead):** `scripts/run_game.sh` starts the engine on :61008 and Next on :61007, capping CPU
  threads (the shared host had load ≈ 56). A full 10k-voter game was played over HTTP through the :61007
  proxy: new game ≈ 1 s, rounds 14–83 ms, election + debrief ≈ 0.2 s; rejections behave as specified.
- Tests: Python **167 passed** (≈ 2.5 min on the loaded host); web 26 passed.
- Still to check by a human: a browser walkthrough of every screen; WebGL network view and CSS animations
  (unverifiable in headless Firefox). The CLI deliverable is dropped (R11).
