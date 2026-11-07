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
export const ALIGNMENT_LABELS = ["Far", "Distant", "Neutral", "Close", "Aligned"] as const;

export const SECTOR_ORDER: SectorId[] = ["Edu", "Welfare", "Agri", "Industry", "Env"];

export const sectorLabel = (id: string) => SECTOR_META[id as SectorId]?.label ?? id;
export const regionLabel = (id: string) => REGION_META[id as RegionId]?.label ?? id;

export const ERROR_COPY: Record<string, string> = {
  INVALID_PHASE: "That move isn't available in this phase.",
  INVALID_MANIFESTO: "The manifesto must be five non-negative shares summing to 100%.",
  INVALID_ALLOCATION: "That allocation breaks the rules — check your worker counts.",
  INVALID_QUERY: "That query isn't valid — pick two different sectors and a region.",
  INSUFFICIENT_FUNDS: "Not enough money in the treasury for that.",
  DUPLICATE_QUERY: "You already asked exactly that this round — no charge.",
  TERMINAL_STATE: "The election is over. Start a new game to play again.",
};

export const N_MIN = 1_000;
export const N_MAX = 1_000_000;
export const N_DEFAULT = 10_000;
export const ROUNDS_MIN = 1;
export const ROUNDS_MAX = 10;
export const ROUNDS_DEFAULT = 5;
/** Dirichlet concentration κ of the voter preference draw (spec §14, v2.1). Public rule. */
export const KAPPA = 60;
