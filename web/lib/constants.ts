import type { RegionId, SectorId } from "./types";

export const SECTOR_META: Record<SectorId, { label: string; short: string; color: string }> = {
  Edu: { label: "Education", short: "Edu", color: "var(--sector-edu)" },
  Welfare: { label: "Welfare", short: "Welf", color: "var(--sector-welfare)" },
  Agri: { label: "Agriculture", short: "Agri", color: "var(--sector-agri)" },
  Industry: { label: "Industry", short: "Ind", color: "var(--sector-industry)" },
  Env: { label: "Environment", short: "Env", color: "var(--sector-env)" },
};

/** Hex versions for canvas/WebGL consumers that cannot read CSS variables. */
export const REGION_HEX: Record<RegionId, string> = {
  N: "#6250d6",
  S: "#eb6834",
  E: "#1baf7a",
  W: "#2a78d6",
};

export const REGION_META: Record<RegionId, { label: string; compass: string; color: string }> = {
  N: { label: "North", compass: "N", color: "var(--region-n)" },
  S: { label: "South", compass: "S", color: "var(--region-s)" },
  E: { label: "East", compass: "E", color: "var(--region-e)" },
  W: { label: "West", compass: "W", color: "var(--region-w)" },
};

/** Alignment bins 0..4 (4 = closest to manifesto): diverging red → gray → blue. */
export const ALIGNMENT_HEX = ["#e34948", "#f0a3a0", "#9a9993", "#86b6ef", "#2a78d6"] as const;
/**
 * The engine bins each voter's probability of voting for you, P = σ(β(S − z)), at 10/40/60/90% (R15,
 * engine.alignment_bins). The bins therefore map directly onto the election result.
 */
export const ALIGNMENT_LABELS = ["<10%", "10–40%", "40–60%", "60–90%", ">90%"] as const;

export const SECTOR_ORDER: SectorId[] = ["Edu", "Welfare", "Agri", "Industry", "Env"];

export const sectorLabel = (id: string) => SECTOR_META[id as SectorId]?.label ?? id;
export const regionLabel = (id: string) => REGION_META[id as RegionId]?.label ?? id;

export const ERROR_COPY: Record<string, string> = {
  INVALID_PHASE: "That move isn't available in this phase.",
  INVALID_MANIFESTO: "The manifesto must be five non-negative shares summing to 100%.",
  INVALID_ALLOCATION: "That allocation breaks the rules — check your worker purchases (a region may be out of eligible voters).",
  INVALID_QUERY: "That query isn't valid — pick two different sectors and a region.",
  INSUFFICIENT_FUNDS: "Not enough money in the treasury for that.",
  TERMINAL_STATE: "The election is over. Start a new game to play again.",
};

export const N_MIN = 1_000;
export const N_MAX = 1_000_000;
export const N_DEFAULT = 10_000;
export const ROUNDS_MIN = 3; // R31: X10 found T = 1 and T = 10 unbalanced
export const ROUNDS_MAX = 5;
export const ROUNDS_DEFAULT = 5;
/** Starting treasury chosen at setup (R17), whole credits. */
export const BUDGET_MIN = 500;
export const BUDGET_MAX = 1000000;
export const BUDGET_DEFAULT = 500;
export const BUDGET_PRESETS = [1000, 10000, 100000, 1000000] as const;
/** Blend picker (R20): how many regions may be blended. */
export const BLEND_MIN = 2;
export const BLEND_MAX = 4;
/** Dirichlet concentration κ of the voter preference draw (spec §14, v2.1). Public rule. */
export const KAPPA = 60;
/** Fraction of a region's voters surveyed by one paid query (spec §6.3, R12). Public rule. */
export const QUERY_SAMPLE_FRACTION = 0.1;
