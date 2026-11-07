# P6 — API and web UI (Days 6–9, parallel once the P3 schema is frozen)

**Spec:** §6.6, §7, §8; U16. **Owner area:** UI (Vis). **Depends on:** the frozen P3 observation schema
(mock JSON before that).

## API (`src/dumbocracy/api/app.py`, FastAPI)
| Route | Purpose |
|---|---|
| `GET /api/health` | status, device, session count |
| `POST /api/games` | create a game {N, T, seed, budget, device, social_rule}; returns game_id, player_id |
| `GET /api/games/{g}/players/{p}/observation` | allowlisted observation |
| `POST /api/games/{g}/players/{p}/query` | {s1, s2, region} |
| `POST /api/games/{g}/players/{p}/commit` | {b[5]} |
| `POST /api/games/{g}/players/{p}/allocate` | {workers_by_region, extra_workers, media_regions} |
| `POST /api/games/{g}/players/{p}/forfeit` | timer timeout |
| `GET /api/games/{g}/graph_sample` | ~1.5k-node stratified subsample: node region, layout xy, colour = **binned** S_i (5 bins, only after the game ends or as a "fog-of-war" view showing just region and degree during play — privacy decision, see below) |
| `GET /api/games/{g}/result`, `/analysis` | election result; post-game "why" bundle (P5 report, oracle gap) |

**Privacy decision for the graph view:** during play the sample shows topology and regions only
(public census level); per-node alignment colouring is revealed only at DONE (debrief). Recorded as I6
in decisions once the owner confirms.

## UI (`web/`, React 19 + Vite, sigma.js + graphology for WebGL)
Screens: Setup → Survey (query builder S1/S2/Region; free top-3 bars; knowledge list) → Manifesto (5
linked sliders constrained to sum to 100%, preview of public stats) → Campaign rounds (4-region SVG
map with known info per region; allocation panel: free/extra workers per region, media toggles with
live cost; poll chart with CI band; treasury; optional countdown) → Results (official vote, expected
share, region breakdown, network colouring animation across rounds, logic verdicts, oracle gap).

## Tests
- API integration via httpx TestClient: full game, rejection codes, recursive privacy scan.
- Vitest: slider simplex constraint, allocation cost math, timer → forfeit.
- Manual browser walkthrough at N = 10k over the SSH tunnel (desktop + phone width).

## Exit criteria
A full game is playable in the browser; tests pass; no hidden data appears in payloads during play.
