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
| `GET /api/games/{g}/players/{p}/debrief`, `/epistemic` | post-game "why" bundle (oracle gap, region reveal) and the P5 epistemic timeline (R14). The election result is `observation.result` — no separate `/result` route (superseded by `docs/v2/web_api_v2.md`) |

**Privacy decision for the graph view:** during play the sample shows topology and regions only
(public census level); per-node alignment colouring is revealed only at DONE (debrief). Confirmed by the owner as R3.

## UI (`web/`, Next.js 16 + React 19 + TypeScript (R11), sigma.js + graphology for WebGL)
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

## Handoff (2026-11-08)
- **Audit:**
  - Every contract route is served: health, games, observation, query, commit, allocate, forfeit,
    graph_sample, debrief, epistemic, delete.
  - All planned screens exist (setup, survey desk + manifesto designer, campaign HQ with region map, orders,
    poll chart + MoE band and optional round clock, results + debrief + epistemic timeline + network reveal).
  - Fixed the stale plan text (no `/result` route; Next.js not Vite; privacy decision = R3).
  - The remaining gap was a real-browser walkthrough; it is now closed.
- **Browser E2E (sub-agent K):** Playwright + headless Chrome with SwiftShader WebGL, in `web/e2e/` (`npm run e2e`,
  auto platform override on Ubuntu 20.04). Five specs against the real engine at N = 10k, all passing, with
  zero console errors or failed requests:
  - full game with a pooled repeat query;
  - rejection toasts (on-the-wire overspend, too many workers, stale-tab commit);
  - round-clock forfeit;
  - phone 390×844 with no horizontal scroll;
  - dark mode with contrast checks;
  - privacy of every `/api` payload before DONE;
  - WebGL network pixels during play and at DONE.
- **Bugs fixed:**
  1. The network never loaded during the campaign (`web/lib/game.tsx`, stage-keyed fetch).
  2. Network nodes were invisible (sigma `itemSizesReference: "screen"` + outlier layout fixes in
     `NetworkView.tsx`).
  3. A misleading alignment legend. The lead then changed the engine binning to each voter's
     vote-probability bins at 10/40/60/90% (R15; `engine.alignment_bins`, contract, mock, legend, tests). At
     DONE all five bins now appear instead of ~100% "supportive" in a lost election.
  4. Truth-row tint CSS.
- **Screenshots** (27, desktop + mobile, light + dark): `reports/v2/screenshots/`.
- **UI timings:** create ≈ 1.2–2 s; query ≤ 0.25 s; round 167 ms mean / 194 ms max; epistemic timeline ≈ 3.3 s.
- **Tests:** Python 233 passed; web unit 43 passed; typecheck clean; production build OK; e2e 5/5 on the live
  rebuilt server.
- **Known cosmetic leftovers (not fixed):**
  - single-poll chart in round 1;
  - toasts briefly covering buttons;
  - the mobile top bar hides the R1–R5 track;
  - the raw-amount overspend message;
  - a few low-degree stragglers in the network.

## Owner change set 2026-11-09 (R16–R20)
Decisions R16–R20 in `decisions_v2.md`; contract updated in `web_api_v2.md`; spec §15.
- **R16 permanent hires** (backend agent): one-time ₫4.00 fee per hire, after which the worker works every later
  round. Applied to the engine, simulate, planner, players, logic replay and balance policies. Engine and
  `simulate_world` still agree exactly.
- **R17 treasury choice:** ₫500–₫1000 at setup (`budget`, 422 outside the range); observation gains `initial_budget`.
- **R18 network only at the end:** no `graph_sample` request before DONE; the Campaign HQ network panel is
  replaced by a "Your manifesto" panel.
- **R19 round look-back:** past-round chips open a read-only review (poll, orders, hires, media, cost, outcomes
  from `round_reports`, questions, treasury before/after).
- **R20 blend picker:** choose 2–4 regions (shortcuts "2 largest" / "3 largest"), voter-weighted.
- **"How it works" popover** on hiring; the player FAQ is `docs/v2/faq.md`.
- **Tests:** Python 265 passed; web unit 60 passed; e2e 5/5 against the live rebuilt server (one locator fixed
  where the current-round chip is now in two places).
- **Stale:** the balance gate, the P7 experiments, the final report, the slide deck and the results page describe
  v2.1 rules (per-round hires, ₫500). The owner will re-run the experiments after further mechanics changes.
