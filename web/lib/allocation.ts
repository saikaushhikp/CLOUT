import type { Observation, RegionId } from "./types";
import { REGION_IDS } from "./types";
import { toCents } from "./money";

export type AllocationDraft = {
  /** NEW workers bought this round per region [N,S,E,W] (P8/R21–R22): one-time fee each, permanent, stay in that region */
  hires: number[];
  media: boolean[]; // per region
};

export const emptyDraft = (): AllocationDraft => ({ hires: [0, 0, 0, 0], media: [false, false, false, false] });

export type AllocationCheck = {
  ok: boolean;
  costCents: number;
  workerCostCents: number;
  mediaCostCents: number;
  treasuryCents: number;
  remainingCents: number;
  /** permanent team per region before this round's hires */
  teamBefore: number[];
  /** teamBefore + this round's hires, per region */
  teamAfter: number[];
  totalHires: number;
  /** most extra workers the player could still afford on top of the current plan */
  maxAffordableMore: number;
  errors: { code: "INVALID_ALLOCATION" | "INSUFFICIENT_FUNDS"; message: string }[];
};

type CostView = Pick<Observation, "treasury" | "costs"> & { workers?: Partial<Observation["workers"]> | null };

/** Permanent team per region [N,S,E,W] before this round's hires (P8 `workers.by_region`). */
export function teamByRegion(obs: CostView): number[] {
  const by = obs.workers?.by_region;
  return REGION_IDS.map((r) => {
    const x = by?.[r];
    return typeof x === "number" && Number.isFinite(x) ? x : 0;
  });
}

export function mediaCostCents(obs: CostView, media: boolean[]): number {
  return REGION_IDS.reduce((acc, r, i) => (media[i] ? acc + toCents(obs.costs.media_by_region[r as RegionId]) : acc), 0);
}

const sumHires = (hires: number[]) => hires.reduce((a, b) => a + Math.max(0, b), 0);

export function allocationCostCents(obs: CostView, draft: AllocationDraft): number {
  return sumHires(draft.hires) * toCents(obs.costs.worker) + mediaCostCents(obs, draft.media);
}

/**
 * Mirrors the engine's validation (web_api_v2.md "Semantics", P8): pure, never throws.
 * cost = Σ hires × worker + Σ aired media ≤ treasury. (Eligible-voter shortage is only known to the engine.)
 */
export function checkAllocation(obs: CostView, draft: AllocationDraft): AllocationCheck {
  const errors: AllocationCheck["errors"] = [];
  const isNonNegInt = (x: number) => Number.isInteger(x) && x >= 0;
  if (draft.hires.length !== 4 || draft.media.length !== 4) {
    errors.push({ code: "INVALID_ALLOCATION", message: "Allocation needs exactly four regions." });
  }
  if (!draft.hires.every(isNonNegInt)) {
    errors.push({ code: "INVALID_ALLOCATION", message: "New workers must be whole numbers ≥ 0." });
  }
  const totalHires = sumHires(draft.hires);
  const workerPrice = toCents(obs.costs.worker);
  const workerCostCents = totalHires * workerPrice;
  const mCost = mediaCostCents(obs, draft.media);
  const costCents = workerCostCents + mCost;
  const treasuryCents = toCents(obs.treasury);
  if (costCents > treasuryCents) {
    errors.push({ code: "INSUFFICIENT_FUNDS", message: "This plan costs more than your treasury." });
  }
  const teamBefore = teamByRegion(obs);
  const maxAffordableMore = workerPrice > 0 ? Math.max(0, Math.floor((treasuryCents - costCents) / workerPrice)) : 0;
  return {
    ok: errors.length === 0,
    costCents,
    workerCostCents,
    mediaCostCents: mCost,
    treasuryCents,
    remainingCents: treasuryCents - costCents,
    teamBefore,
    teamAfter: teamBefore.map((t, i) => t + Math.max(0, draft.hires[i] ?? 0)),
    totalHires,
    maxAffordableMore,
    errors,
  };
}

/** Set this round's new hires in one region (whole number ≥ 0). Affordability is reported by checkAllocation. */
export function setRegionHires(draft: AllocationDraft, region: number, value: number): AllocationDraft {
  const hires = draft.hires.slice();
  hires[region] = Math.max(0, Math.round(Number.isFinite(value) ? value : 0));
  return { ...draft, hires };
}
