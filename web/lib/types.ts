/**
 * Types of the binding HTTP/JSON contract (docs/v2/web_api_v2.md).
 * Keep these 1:1 with the contract; UI-only types live elsewhere.
 */

export type SectorId = "Edu" | "Welfare" | "Agri" | "Industry" | "Env";
export type RegionId = "N" | "S" | "E" | "W";
export type Phase = "SURVEY" | "ROUND" | "DONE";
export type LegalAction = "query" | "commit" | "allocate" | "forfeit";

export const SECTOR_IDS: readonly SectorId[] = ["Edu", "Welfare", "Agri", "Industry", "Env"] as const;
export const REGION_IDS: readonly RegionId[] = ["N", "S", "E", "W"] as const;

export type ErrorCode =
  | "INVALID_PHASE"
  | "INVALID_MANIFESTO"
  | "INVALID_ALLOCATION"
  | "INVALID_QUERY"
  | "INSUFFICIENT_FUNDS"
  | "DUPLICATE_QUERY"
  | "TERMINAL_STATE"
  | (string & {});

export type QueryAnswer = {
  round: number;
  phase: "SURVEY" | "ROUND";
  sector_a: string;
  sector_b: string;
  region: string;
  /** % of region voters with v[b] > v[a] (0-100) */
  pct_b_over_a: number;
  cost: string;
};

export type RoundReport = {
  /** the round just resolved */
  round: number;
  cost: string;
  direct_contacts: number;
  word_of_mouth_reached: number;
  media_exposed: number;
  election_held: boolean;
};

export type ElectionResult = {
  votes: number;
  num_voters: number;
  vote_share: number;
  victory: boolean;
  expected_share: number;
};

export type Observation = {
  schema_version: "2.0";
  game_id: string;
  player_id: string;
  phase: Phase;
  /** 0 in SURVEY; 1..T = the round now being planned; T at DONE */
  round: number;
  rounds: number;
  num_voters: number;
  seed: number;
  treasury: string;
  costs: {
    query: string;
    extra_worker: string;
    free_workers: number;
    contacts_per_worker: number;
    /** next airing price per region */
    media_by_region: Record<RegionId, string>;
  };
  sectors: { id: string; label: string }[];
  regions: { id: string; label: string; voters: number; intra_edges: number; mean_degree: number }[];
  inter_edges: { regions: [string, string]; edges: number }[];
  archetype_library: { id: string; name: string; mean: number[] }[];
  personality_library: { id: string; name: string; social: number; media: number; campaign: number }[];
  free_stats: { round: number; top3_pct: Record<string, number> }[];
  queries: QueryAnswer[];
  /** after commit; round = start-of-round t (T+1 = pre-election) */
  polls: { round: number; estimate: number; moe: number; sample_size: number }[];
  manifesto: number[] | null;
  allocations: {
    round: number;
    workers_by_region: number[];
    extra_workers: number;
    media_regions: boolean[];
    cost: string;
  }[];
  media_airings: Record<RegionId, number>;
  legal_actions: LegalAction[];
  result: null | ElectionResult;
  events: { seq: number; round: number; type: string; message: string }[];
};

export type ActionResponse = {
  accepted: boolean;
  error_code: ErrorCode | null;
  message: string | null;
  observation: Observation;
  /** query only, when accepted */
  answer?: QueryAnswer;
  /** allocate/forfeit only, when accepted */
  round_report?: RoundReport;
};

export type GraphSample = {
  /** false during play: topology + regions only */
  revealed: boolean;
  nodes: { id: number; region: string; degree: number }[];
  /** induced edges between sampled nodes (indices into nodes[]) */
  edges: [number, number][];
  /** only when revealed: [round 0..T][node] similarity bin 0..4 (4 = closest to manifesto) */
  alignment: number[][] | null;
};

export type Debrief = {
  result: Observation["result"];
  regions: {
    id: string;
    archetype: string;
    personality: string;
    zealots: number;
    final_expected_share: number;
    initial_expected_share: number;
  }[];
  /** true expected share before round 1 and after every round */
  share_trajectory: number[];
  oracle: {
    manifesto: number[];
    expected_share_no_campaign: number;
    note: "ORACLE: full-information benchmark";
  };
  your_manifesto_no_campaign_share: number;
};

export type Health = { status: "ok"; device: string; active_games: number; max_games: number };

export type CreateGameRequest = { num_voters?: number; seed?: number; rounds?: number };
export type CreateGameResponse = { game_id: string; player_id: string; observation: Observation };

export type QueryRequest = { sector_a: string; sector_b: string; region: string };
export type CommitRequest = { manifesto: number[] };
export type AllocateRequest = { workers_by_region: number[]; extra_workers: number; media_regions: boolean[] };
