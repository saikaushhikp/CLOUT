# Local web API v1

Run locally with `PYTHONPATH=src python -m uvicorn dumbocracy.webapp:app --host 127.0.0.1 --port 61007`. The service stores at most eight in-memory sessions and is a development beta, not multi-user production hosting. Sessions are not durable. `backend=auto` selects `cuda:0` only when PyTorch reports CUDA available; otherwise it selects CPU. Responses report the actual backend.

## Endpoints

- `GET /api/health`: service/API version and session limits.
- `POST /api/games`: JSON `{node_count,horizon_rounds,initial_budget,seed,backend}`.
  Defaults: 1000 nodes, five rounds, budget `20.00`, seed `20261019`, backend
  `auto`. Limits: 1–100000 nodes and 1–100 rounds. Returns `game_id`, public
  observation, actual backend, and safe public trajectory.
- `GET /api/games/{id}/observation`: detached v1 public observation.
- `POST /api/games/{id}/actions`: JSON `{"action_id":"no_op","target_ids":[]}`.
  For targeted contact, pass one public eligible ID, e.g.
  `{"action_id":"targeted_contact","target_ids":["v0"]}`. Broadcast takes no
  target. Rejections return `accepted:false`, stable error code, and the public
  state; rejected actions do not advance round or spend budget.
- `POST /api/games/{id}/reset`: reproducibly reinitialize the session.
- `GET /api/games/{id}/result`: JSON `null` before terminal, official public
  result afterward.
- `GET /api/games/{id}/analysis`: pending before terminal; afterward completed
  trace diagnostics with `evaluation_mode=trace_only` (not future guarantees).

## Copy/paste smoke flow

```bash
curl -fsS http://127.0.0.1:61007/api/health
GAME=$(curl -fsS -X POST http://127.0.0.1:61007/api/games \
  -H 'content-type: application/json' \
  -d '{"node_count":1000,"horizon_rounds":5,"initial_budget":"20.00","seed":20261019,"backend":"cuda:0"}' | python -c 'import json,sys; print(json.load(sys.stdin)["game_id"])')
curl -fsS "http://127.0.0.1:61007/api/games/$GAME/observation"
curl -fsS -X POST "http://127.0.0.1:61007/api/games/$GAME/actions" -H 'content-type: application/json' -d '{"action_id":"targeted_contact","target_ids":["v0"]}'
curl -fsS -X POST "http://127.0.0.1:61007/api/games/$GAME/actions" -H 'content-type: application/json' -d '{"action_id":"broadcast","target_ids":[]}'
curl -fsS -X POST "http://127.0.0.1:61007/api/games/$GAME/actions" -H 'content-type: application/json' -d '{"action_id":"no_op","target_ids":[]}'
curl -fsS -X POST "http://127.0.0.1:61007/api/games/$GAME/actions" -H 'content-type: application/json' -d '{"action_id":"no_op","target_ids":[]}'
curl -fsS -X POST "http://127.0.0.1:61007/api/games/$GAME/actions" -H 'content-type: application/json' -d '{"action_id":"no_op","target_ids":[]}'
curl -fsS "http://127.0.0.1:61007/api/games/$GAME/result"
curl -fsS "http://127.0.0.1:61007/api/games/$GAME/analysis"
curl -fsS -X POST "http://127.0.0.1:61007/api/games/$GAME/reset"
```

Use an eligible voter ID from the observation rather than assuming `v0` when IDs differ. The React app source and npm lockfile live under `web/`; build it before starting FastAPI so the production assets exist under `web/dist/`:

```bash
conda run -n mas node --version  # Node >=22.12
conda run -n mas npm --version   # npm >=10
conda run -n mas npm ci --prefix web
conda run -n mas npm --prefix web test
conda run -n mas npm --prefix web run build
conda run -n mas python -m uvicorn dumbocracy.webapp:app --host 127.0.0.1 --port 61007
```

If Node/npm are missing from `mas`, install Node 24 from conda-forge first: `conda install -n mas -c conda-forge nodejs=24`. React and the UI libraries are installed from `web/package-lock.json` by `npm ci`; they do not need separate global installs.

The production bundle is served by FastAPI. For iterative frontend development, run the API on port 61007 in one terminal, then `conda run -n mas npm --prefix web run dev` in a second and open `http://127.0.0.1:5173`; Vite proxies `/api` requests to the local API. The responsive dashboard shows the aggregate trajectory, public network summary, action costs/history, and terminal trace-only diagnostics. Target IDs can be filtered locally in the browser; only the chosen public ID is submitted. The UI makes no external font/CDN requests and never receives hidden edges/state. For a remote server, forward the port with `ssh -L 61007:127.0.0.1:61007 <user>@<host>` and open `http://127.0.0.1:61007` locally.

## Browser walkthrough

1. Start the API and open the page. Confirm the local-service indicator turns green and setup controls are usable at desktop and phone widths. For a quick smoke game, choose CPU, 100 nodes, three rounds, a `20.00` budget, and a fixed seed, then start the episode.
2. Confirm the round-zero aggregate, budget, graph summary, and actual compute device appear. In the public-target picker, search for an eligible ID and submit targeted contact. Check that one action is recorded, cost/budget update, round advances, and the chart and event feed gain a point.
3. Submit broadcast if the remaining budget covers its displayed cost. If it is unaffordable, confirm the rejection leaves round and budget unchanged. Choose no-op (wait) for any remaining rounds; the episode ends at its fixed horizon, not early on an apparent victory.
4. Read the official result and expand trace diagnostics. Confirm the UI labels the analysis `trace_only`, rather than presenting it as a guarantee about unseen future states.
5. Reset the game. Confirm round zero, original public aggregate and starting budget return. Start a second game if you want a clean playthrough.
6. In the browser Network panel inspect create/observation/action/result payloads. Only public aggregates, public eligible IDs and game results should be present; voter arrays, susceptibility, hidden edges and seed fields must not appear.

`web-ui/` is a separate experimental scaffold; it is not the production app. The checked-in source of truth is `web/src/`, and reviewers should run its tests and production build before testing the FastAPI-served app.

The API never returns GameState, voter support/susceptibility arrays, private events, hidden edges, device tensors, or seed fields. It uses existing public observation/result allowlists and derived trace-only analysis. For the terminal result, seed-bearing keys are removed from reproducibility metadata before serialization; result score, budget used, backend/runtime and non-seed config metadata remain public.
