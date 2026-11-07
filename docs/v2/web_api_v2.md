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
| `POST /api/games` | `{num_voters?: int=10000 (1000..1000000), seed?: int (omit → random, returned), rounds?: int=5 (1..10)}` | `{game_id, player_id, observation}` |
| `GET /api/games/{g}/players/{p}/observation` | – | `Observation` |
| `POST /api/games/{g}/players/{p}/query` | `{sector_a, sector_b, region}` | `ActionResponse` (+ `answer`) |
| `POST /api/games/{g}/players/{p}/commit` | `{manifesto: number[5]}` (≥0, sums to 1 ± 1e-6) | `ActionResponse` |
| `POST /api/games/{g}/players/{p}/allocate` | `{workers_by_region: int[4], extra_workers: int, media_regions: bool[4]}` | `ActionResponse` (+ `round_report`) |
| `POST /api/games/{g}/players/{p}/forfeit` | – (timer expired; = allocate nothing) | `ActionResponse` (+ `round_report`) |
| `GET /api/games/{g}/graph_sample?max_nodes=1500` | – | `GraphSample` |
| `GET /api/games/{g}/players/{p}/debrief` | – (only when phase = DONE, else 409) | `Debrief` |
| `DELETE /api/games/{g}` | – | `{deleted:true}` |

Errors: unknown game/player → 404 `{detail}`; invalid JSON body → 422. **Game-rule rejections are NOT HTTP
errors**: they return 200 with `accepted:false` and a stable `error_code`, and change nothing (atomic):
`INVALID_PHASE, INVALID_MANIFESTO, INVALID_ALLOCATION, INVALID_QUERY, INSUFFICIENT_FUNDS, DUPLICATE_QUERY,
TERMINAL_STATE`. The server caps concurrent games (`max_games`, default 8; oldest idle game is evicted).

```ts
type ActionResponse = {
  accepted: boolean; error_code: string | null; message: string | null;
  observation: Observation;
  answer?: QueryAnswer;          // query only, when accepted
  round_report?: RoundReport;    // allocate/forfeit only, when accepted
};
type QueryAnswer = { round: number; phase: "SURVEY"|"ROUND"; sector_a: string; sector_b: string;
                     region: string; pct_b_over_a: number /* % of region voters with v[b] > v[a] */; cost: string };
type RoundReport = { round: number /* the round just resolved */; cost: string;
                     direct_contacts: number; word_of_mouth_reached: number; media_exposed: number;
                     election_held: boolean };
type Observation = {
  schema_version: "2.0"; game_id: string; player_id: string;
  phase: "SURVEY"|"ROUND"|"DONE";
  round: number;            // 0 in SURVEY; 1..T = the round now being planned; T at DONE
  rounds: number;           // T
  num_voters: number; seed: number;
  treasury: string;         // remaining money
  costs: { query: string; extra_worker: string; free_workers: number; contacts_per_worker: number;
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
  allocations: {round: number; workers_by_region: number[]; extra_workers: number; media_regions: boolean[]; cost: string}[];
  media_airings: Record<"N"|"S"|"E"|"W", number>;
  legal_actions: ("query"|"commit"|"allocate"|"forfeit")[];
  result: null | { votes: number; num_voters: number; vote_share: number; victory: boolean; expected_share: number };
  events: {seq: number; round: number; type: string; message: string}[];
};
type GraphSample = {
  revealed: boolean;          // false during play: topology + regions only (decision R3)
  nodes: {id: number; region: string; degree: number}[];   // stratified sample (≤ max_nodes)
  edges: [number, number][];  // induced edges between sampled nodes (indices into nodes[])
  alignment: number[][] | null;  // only when revealed: [round 0..T][node] similarity bin 0..4 (4 = closest to manifesto)
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
- SURVEY: `query` any number of times (5.00 each, duplicates within the same phase/round rejected), then `commit`.
  Commit moves to ROUND 1 and immediately produces poll #1 (start of round 1) and a fresh `free_stats` entry.
- ROUND t: any number of `query`, then exactly one `allocate` (or `forfeit`) resolves the round:
  campaign → social dynamics → round t+1 (new free stat + poll). After round T resolves, a pre-election poll
  (round T+1) is taken and the election runs automatically → phase DONE, `result` filled.
- `workers_by_region` must sum ≤ `free_workers + extra_workers`; cost = `extra_workers × extra_worker` +
  Σ `media_by_region[r]` over aired regions; must be ≤ treasury.
- During play nothing hidden is exposed (no per-voter values, archetype/personality assignment, edges of the
  full graph, p_r/p_rs, zealots). `graph_sample` exposes sampled topology only (public structure); alignment
  and the debrief are revealed only at DONE.
