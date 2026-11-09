import type { Observation, RegionId } from "./types";
import { REGION_IDS } from "./types";
import { toCents } from "./money";

export type AllocationDraft = {
  workers: number[]; // per region [N,S,E,W]
  /** NEW permanent hires this round (R16): one-time fee each, then they work every later round */
  extra: number;
  media: boolean[]; // per region
};

export const emptyDraft = (): AllocationDraft => ({ workers: [0, 0, 0, 0], extra: 0, media: [false, false, false, false] });

export type AllocationCheck = {
  ok: boolean;
  costCents: number;
  workerCostCents: number;
  mediaCostCents: number;
  treasuryCents: number;
  remainingCents: number;
  /** workers already on the team before this round's hires (free + earlier permanent hires) */
  workersBase: number;
  /** workersBase + this round's new hires */
  workersAvailable: number;
  workersAssigned: number;
  workersIdle: number;
  /** most new hires the player could still afford on top of the current media plan */
  maxAffordableExtra: number;
  errors: { code: "INVALID_ALLOCATION" | "INSUFFICIENT_FUNDS"; message: string }[];
};

type CostView = Pick<Observation, "treasury" | "costs"> & { workers?: Partial<Observation["workers"]> | null };

/**
 * Workers on the team before this round's new hires: `workers.available` (free + every earlier permanent
 * hire, R16). Falls back to the free workers for observations from engines that predate R16.
 */
export function baseWorkers(obs: CostView): number {
  const avail = obs.workers?.available;
  return typeof avail === "number" && Number.isFinite(avail) ? avail : obs.costs.free_workers;
}

export function mediaCostCents(obs: CostView, media: boolean[]): number {
  return REGION_IDS.reduce((acc, r, i) => (media[i] ? acc + toCents(obs.costs.media_by_region[r as RegionId]) : acc), 0);
}

export function allocationCostCents(obs: CostView, draft: Pick<AllocationDraft, "extra" | "media">): number {
  return Math.max(0, draft.extra) * toCents(obs.costs.extra_worker) + mediaCostCents(obs, draft.media);
}

/**
 * Mirrors the engine's validation (web_api_v2.md "Semantics", R16): pure, never throws.
 * sum(workers) ≤ workers.available + extra; cost = extra × extra_worker + Σ aired media ≤ treasury.
 */
export function checkAllocation(obs: CostView, draft: AllocationDraft): AllocationCheck {
  const errors: AllocationCheck["errors"] = [];
  const isNonNegInt = (x: number) => Number.isInteger(x) && x >= 0;
  if (draft.workers.length !== 4 || draft.media.length !== 4) {
    errors.push({ code: "INVALID_ALLOCATION", message: "Allocation needs exactly four regions." });
  }
  if (!draft.workers.every(isNonNegInt)) {
    errors.push({ code: "INVALID_ALLOCATION", message: "Worker counts must be whole numbers ≥ 0." });
  }
  if (!isNonNegInt(draft.extra)) {
    errors.push({ code: "INVALID_ALLOCATION", message: "New hires must be a whole number ≥ 0." });
  }
  const workersBase = baseWorkers(obs);
  const workersAvailable = workersBase + Math.max(0, draft.extra);
  const workersAssigned = draft.workers.reduce((a, b) => a + b, 0);
  if (workersAssigned > workersAvailable) {
    errors.push({
      code: "INVALID_ALLOCATION",
      message: `You assigned ${workersAssigned} workers but only have ${workersAvailable}.`,
    });
  }
  const workerCostCents = Math.max(0, draft.extra) * toCents(obs.costs.extra_worker);
  const mCost = mediaCostCents(obs, draft.media);
  const costCents = workerCostCents + mCost;
  const treasuryCents = toCents(obs.treasury);
  if (costCents > treasuryCents) {
    errors.push({ code: "INSUFFICIENT_FUNDS", message: "This plan costs more than your treasury." });
  }
  const workerPrice = toCents(obs.costs.extra_worker);
  const maxAffordableExtra = workerPrice > 0 ? Math.max(0, Math.floor((treasuryCents - mCost) / workerPrice)) : 0;
  return {
    ok: errors.length === 0,
    costCents,
    workerCostCents,
    mediaCostCents: mCost,
    treasuryCents,
    remainingCents: treasuryCents - costCents,
    workersBase,
    workersAvailable,
    workersAssigned,
    workersIdle: Math.max(0, workersAvailable - workersAssigned),
    maxAffordableExtra,
    errors,
  };
}

/** Clamp a worker stepper change so the total never exceeds the team (`base` = workers.available) + new hires. */
export function setRegionWorkers(draft: AllocationDraft, base: number, region: number, value: number): AllocationDraft {
  const others = draft.workers.reduce((a, b, i) => (i === region ? a : a + b), 0);
  const cap = Math.max(0, base + draft.extra - others);
  const v = Math.max(0, Math.min(cap, Math.round(value)));
  const workers = draft.workers.slice();
  workers[region] = v;
  return { ...draft, workers };
}

/** Change this round's new hires; if workers would exceed the pool, trim from the largest regions. */
export function setExtraWorkers(draft: AllocationDraft, base: number, extra: number): AllocationDraft {
  const e = Math.max(0, Math.round(extra));
  const workers = draft.workers.slice();
  let over = workers.reduce((a, b) => a + b, 0) - (base + e);
  while (over > 0) {
    const i = workers.indexOf(Math.max(...workers));
    workers[i] -= 1;
    over -= 1;
  }
  return { ...draft, extra: e, workers };
}
