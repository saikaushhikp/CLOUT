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
  | "TERMINAL_STATE"
  | (string & {});

export type QueryAnswer = {
  round: number;
  phase: "SURVEY" | "ROUND";
  sector_a: string;
  sector_b: string;
  region: string;
  /** % of THIS query's random sample with v[b] > v[a] (0-100) (R12) */
  pct_b_over_a: number;
  /** m = round(0.10 · region voters) */
  sample_size: number;
  /** 95% margin of error of pct_b_over_a, percentage points (finite-population corrected) */
  moe: number;
  /** pooled over every repeat of (a, b, region) this phase/round */
  pooled_pct: number;
  pooled_sample_size: number;
  pooled_moe: number;
  /** 1 for the first ask of this triple this round, 2, 3, … */
  repeat_index: number;
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
  /** remaining money */
  treasury: string;
  /** the treasury chosen at setup (R17) */
  initial_budget: string;
  /** R16: hired = permanent hires so far; available = free + hired */
  workers: { free: number; hired: number; available: number };
  /** R19: one per resolved round, oldest first (for the round look-back) */
  round_reports: RoundReport[];
  costs: {
    query: string;
    /** one-time hiring fee per permanent worker (R16) */
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

/** One decision point t (start of round t, after that round's poll; manifesto already committed). */
export type EpistemicRound = {
  round: number;
  treasury: string;
  knowledge: { queries: number; polls: number; posterior_entropy: number; consistent_worlds: number };
  /** ∃π ∃w Victory — M_p ◇Victory: winning was considered possible */
  possible_victory: boolean;
  /** ∃π ∀w Victory — knowing HOW: one plan wins in every consistent world */
  guaranteed_victory: boolean;
  /** ∀w ∃π Victory — knowing THAT: every world has some winning plan (maybe different ones) */
  knows_win_exists: boolean;
  /** ∀π ∀w ¬Victory — K_p □¬Victory within the plan set */
  inevitable_defeat: boolean;
  /** ∃π Victory in the TRUE world (ground truth, not knowledge) */
  truth_victory_possible: boolean;
  best_plan: { label: string; /** fraction of W_t it wins */ win_fraction: number; mean_share: number };
};

/** GET /api/games/{g}/players/{p}/epistemic (P5, decision R14) — DONE only. */
export type EpistemicTimeline = {
  scope: { worlds_per_round: number; policies_per_round: number; world_voters: number; note: string };
  rounds: EpistemicRound[];
  actual: { victory: boolean; vote_share: number; expected_share: number };
  headline: string;
};

export type Health = { status: "ok"; device: string; active_games: number; max_games: number };

/** `budget`: starting treasury, 2-dp decimal string "500.00".."1000.00" (R17). */
export type CreateGameRequest = { num_voters?: number; seed?: number; rounds?: number; budget?: string };
export type CreateGameResponse = { game_id: string; player_id: string; observation: Observation };

export type QueryRequest = { sector_a: string; sector_b: string; region: string };
export type CommitRequest = { manifesto: number[] };
/** `extra_workers` = NEW permanent hires this round (R16); sum(workers_by_region) ≤ workers.available + extra_workers. */
export type AllocateRequest = { workers_by_region: number[]; extra_workers: number; media_regions: boolean[] };
