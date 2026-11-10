# Dumbocracy v2 — Web API contract (P3)

**Binding contract between the Python engine (`src/dumbocracy/api/app.py`) and the Next.js game (`web/`).**
Spec references: technical_spec_v2.md §6 (information), §7 (flow), §8 (state/observation), §14 (defaults).

## Topology
- Browser → **Next.js on `127.0.0.1:61007`** (`web/`). Next rewrites `/api/:path*` →
  `${DUMBOCRACY_API:-http://127.0.0.1:61008}/api/:path*`.
- **FastAPI engine on `127.0.0.1:61008`** (`uvicorn dumbocracy.api.app:app`). Device `auto` (the CUDA device
  with the most free memory, else CPU). The electorate is generated from CPU seed streams (R8), so
  (num_voters, seed) always recreates the same initial electorate.
- `scripts/run_game.sh` starts both.
- All money values are **decimal strings with 2 dp** (`"500.00"`). Shares and percentages are floats:
  shares are in [0,1], `*_pct` fields are 0–100.

## Identifiers
Sectors `["Edu","Welfare","Agri","Industry","Env"]` (fixed order = manifesto vector order).
Regions `["N","S","E","W"]`. Phases `"SURVEY" | "ROUND" | "DONE"`. Player id is `"p0"` (single player now;
every route is player-keyed for a future second/LLM player).

## Routes
| Method & path | Body | Response |
|---|---|---|
| `GET /api/health` | – | `{status:"ok", device, active_games, max_games}` |
| `POST /api/games` | `{num_voters?: int=10000 (1000..1000000), seed?: int (omit → random, returned), rounds?: int=5 (1..10), budget?: string="500.00" ("500.00".."1000.00", 2 dp; R17)}` | `{game_id, player_id, observation}` |
| `GET /api/games/{g}/players/{p}/observation` | – | `Observation` |
| `POST /api/games/{g}/players/{p}/query` | `{sector_a, sector_b, region}` | `ActionResponse` (+ `answer`) |
| `POST /api/games/{g}/players/{p}/commit` | `{manifesto: number[5]}` (≥0, sums to 1 ± 1e-6) | `ActionResponse` |
| `POST /api/games/{g}/players/{p}/allocate` | `{hires_by_region: int[4], media_regions: bool[4]}` (P8/R21–R22: NEW workers bought this round, per region) | `ActionResponse` (+ `round_report`) |
| `POST /api/games/{g}/players/{p}/forfeit` | – (timer expired; = allocate nothing) | `ActionResponse` (+ `round_report`) |
| `GET /api/games/{g}/graph_sample?max_nodes=1500` | – | `GraphSample` |
| `GET /api/games/{g}/players/{p}/debrief` | – (only when phase = DONE, else 409) | `Debrief` |
| `DELETE /api/games/{g}` | – | `{deleted:true}` |

Errors: unknown game/player → 404 `{detail}`; invalid JSON body → 422. **Game-rule rejections are NOT HTTP
errors**: they return 200 with `accepted:false` and a stable `error_code`, and change nothing (atomic):
`INVALID_PHASE, INVALID_MANIFESTO, INVALID_ALLOCATION, INVALID_QUERY, INSUFFICIENT_FUNDS,
TERMINAL_STATE`. The server caps concurrent games (`max_games`, default 8; oldest idle game is evicted).

```ts
type ActionResponse = {
  accepted: boolean; error_code: string | null; message: string | null;
  observation: Observation;
  answer?: QueryAnswer;          // query only, when accepted
  round_report?: RoundReport;    // allocate/forfeit only, when accepted
};
type QueryAnswer = { round: number; phase: "SURVEY"|"ROUND"; sector_a: string; sector_b: string;
                     region: string;
                     pct_b_over_a: number;   // % of THIS query's random sample with v[b] > v[a]   (R12)
                     sample_size: number;    // m = round(0.10 · region voters)
                     moe: number;            // 95% margin of error of pct_b_over_a, percentage points (FPC)
                     pooled_pct: number;     // pooled over every repeat of (a, b, region) this phase/round
                     pooled_sample_size: number; pooled_moe: number;
                     repeat_index: number;   // 1 for the first ask of this triple this round, 2, 3, …
                     cost: string };
type RoundReport = { round: number /* the round just resolved */; cost: string;
                     hires_by_region: number[];   // [N,S,E,W] workers bought this round (P8)
                     team_by_region: number[];    // [N,S,E,W] permanent workers after this round's hires
                     spreaders: number;           // voters who tried to convert friends this round (new workers + last round's converts)
                     converted: number;           // voters converted to your manifesto this round (they spread next round)
                     media_exposed: number;
                     election_held: boolean };
type Observation = {
  schema_version: "2.2"; game_id: string; player_id: string;
  phase: "SURVEY"|"ROUND"|"DONE";
  round: number;            // 0 in SURVEY; 1..T = the round now being planned; T at DONE
  rounds: number;           // T
  num_voters: number; seed: number;
  treasury: string;         // remaining money
  initial_budget: string;   // the treasury chosen at setup (R17)
  workers: { by_region: Record<"N"|"S"|"E"|"W", number>; total: number };  // P8: permanent team (no free workers, R21)
  round_reports: RoundReport[];  // R19: one per resolved round, oldest first (for the round look-back)
  costs: { query: string; worker: string /* one-time fee per worker, P8 */;
           min_degree: number /* a bought worker has ≥ this many friends */; attempt_cap: number /* max conversion attempts per voter per game */;
           media_by_region: Record<"N"|"S"|"E"|"W", string> /* next airing price per region */ };
  sectors: {id: string; label: string}[];
  regions: {id: string; label: string; voters: number; intra_edges: number; mean_degree: number}[];
  inter_edges: {regions: [string, string]; edges: number}[];
  archetype_library: {id: string; name: string; mean: number[]}[];           // public library, assignment hidden
  personality_library: {id: string; name: string; social: number; media: number; campaign: number}[]; // mean susceptibilities
  free_stats: {round: number; top3_pct: Record<string, number>}[];           // history, newest last
  queries: QueryAnswer[];                                                     // this player's answers
  polls: {round: number; estimate: number; moe: number; sample_size: number}[]; // after commit; round = start-of-round t (T+1 = pre-election)
  manifesto: number[] | null;
  allocations: {round: number; hires_by_region: number[]; media_regions: boolean[]; cost: string}[];
  media_airings: Record<"N"|"S"|"E"|"W", number>;
  legal_actions: ("query"|"commit"|"allocate"|"forfeit")[];
  result: null | { votes: number; num_voters: number; vote_share: number; victory: boolean; expected_share: number };
  events: {seq: number; round: number; type: string; message: string}[];
};
type GraphSample = {
  revealed: boolean;          // false during play: topology + regions only (decision R3)
  nodes: {id: number; region: string; degree: number}[];   // stratified sample (≤ max_nodes)
  edges: [number, number][];  // induced edges between sampled nodes (indices into nodes[])
  alignment: number[][] | null;  // only when revealed: [round 0..T][node] vote-probability bin 0..4 (R15):
                                 //   0: P<10%  1: 10–40%  2: 40–60% (swing)  3: 60–90%  4: >90% likely to vote for you
};
type Debrief = {
  result: Observation["result"];
  regions: {id: string; archetype: string; personality: string; zealots: number;
            final_expected_share: number; initial_expected_share: number}[];
  share_trajectory: number[];   // true expected share before round 1 and after every round
  oracle: {manifesto: number[]; expected_share_no_campaign: number; note: "ORACLE: full-information benchmark"};
  your_manifesto_no_campaign_share: number;  // anticipated share of your manifesto without campaigning
};
```

## Semantics that both sides rely on
- SURVEY: `query` any number of times (each costs `costs.query`, default 10.00). Each query is a fresh 10% random sample of the region
  (R12); repeating the same triple in the same phase/round is allowed and narrows the pooled estimate. Then `commit`.
  Commit moves to ROUND 1 and immediately produces poll #1 (start of round 1) and a fresh `free_stats` entry.
- ROUND t: any number of `query`, then exactly one `allocate` (or `forfeit`) resolves the round:
  campaign → social dynamics → round t+1 (new free stat + poll). After round T resolves, a pre-election poll
  (round T+1) is taken and the election runs automatically → phase DONE, `result` filled.
- **P8 party workers (R21–R27; supersedes R16):** there are no free workers. `hires_by_region[r]` NEW workers are
  bought in region r this round for a one-time `costs.worker` each, and they stay in r for the rest of the game.
  Each one is a random region-r voter with ≥ `min_degree` friends, pinned at your manifesto. In the round it is bought,
  a worker tries to convert each of its friends once. Converted voters take your manifesto and try *their* friends
  in the next round (one hop per round). Each voter can be tried at most `attempt_cap` times per game. The
  conversion chance grows with how receptive the voter is and how much they already like your manifesto.
  cost = `Σ hires_by_region × worker` + Σ `media_by_region[r]` over aired regions, and it must be ≤ treasury.
  If a region does not have enough eligible voters, the allocate is rejected with `INVALID_ALLOCATION`. A rejected
  allocate buys nothing. `workers` already includes an accepted round's hires. All-zero hires and no media
  ("do nothing") is legal; earlier converts still spread that round.
- **R17 budget:** must be a JSON *string* (a JSON number → 422); omitted or `null` → `"500.00"`; at most 2 decimal
  places (`"750.5"` and `"1000"` are accepted and normalised to `"750.50"` / `"1000.00"`; `"500.001"` → 422).
- **R19 round_reports:** a forfeited round also has a report (cost `"0.00"`); the last one has `election_held: true`.
- During play nothing hidden is exposed (no per-voter values, archetype/personality assignment, edges of the
  full graph, p_r/p_rs, zealots). `graph_sample` exposes sampled topology only (public structure); alignment
  and the debrief are revealed only at DONE.

## Epistemic timeline (P5, decision R14)
| Method & path | Response |
|---|---|
| `GET /api/games/{g}/players/{p}/epistemic` | `EpistemicTimeline` — only at DONE (409 before; 503 if the logic module is unavailable on this engine). Computed on first request (target < 15 s at N = 10k), then cached per game/player. |

Computed by `dumbocracy.logic.report.epistemic_timeline(state, player_id, *, worlds=16, world_voters=1600, seed=0, device=None) -> dict`.
For each decision point t = 1..T (start of round t, after that round's poll; manifesto already committed) it
evaluates, over a finite set W_t of worlds consistent with what the player knew at that moment (the knowledge log
up to t: free stats, query answers, polls, own actions) and a finite policy set Π_t (stationary allocation plans
for the remaining rounds):
- `possible_victory` = ∃π ∃w Victory  (M_p ◇Victory: the player considers winning possible)
- `guaranteed_victory` = ∃π ∀w Victory  (knowing-HOW: one plan known to win in every world, i.e. K_p of a specific plan)
- `knows_win_exists` = ∀w ∃π Victory  (knowing-THAT: in every world some plan wins, but maybe a different one)
- `inevitable_defeat` = ∀π ∀w ¬Victory (K_p □¬Victory within Π)
- `truth_victory_possible` = ∃π Victory in the TRUE world (ground truth, not knowledge: truth ≠ knowledge)
Victory = election-day expected share > 0.5 (deterministic per world; spec I5).

```ts
type EpistemicTimeline = {
  scope: { worlds_per_round: number; policies_per_round: number; world_voters: number; note: string };
  rounds: {
    round: number; treasury: string;
    knowledge: { queries: number; polls: number; posterior_entropy: number; consistent_worlds: number };
    possible_victory: boolean; guaranteed_victory: boolean; knows_win_exists: boolean; inevitable_defeat: boolean;
    truth_victory_possible: boolean;
    best_plan: { label: string; win_fraction: number /* fraction of W_t it wins */; mean_share: number };
  }[];
  actual: { victory: boolean; vote_share: number; expected_share: number };
  headline: string;   // e.g. "Victory was possible throughout; you knew a winning plan from round 3."
};
```
