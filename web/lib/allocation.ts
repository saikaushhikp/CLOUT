import type { Observation, RegionId } from "./types";
import { REGION_IDS } from "./types";
import { toCents } from "./money";

export type AllocationDraft = {
  workers: number[]; // per region [N,S,E,W]
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
  workersAvailable: number;
  workersAssigned: number;
  workersIdle: number;
  /** most extra workers the player could still afford on top of the current media plan */
  maxAffordableExtra: number;
  errors: { code: "INVALID_ALLOCATION" | "INSUFFICIENT_FUNDS"; message: string }[];
};

type CostView = Pick<Observation, "treasury" | "costs">;

export function mediaCostCents(obs: CostView, media: boolean[]): number {
  return REGION_IDS.reduce((acc, r, i) => (media[i] ? acc + toCents(obs.costs.media_by_region[r as RegionId]) : acc), 0);
}

export function allocationCostCents(obs: CostView, draft: Pick<AllocationDraft, "extra" | "media">): number {
  return Math.max(0, draft.extra) * toCents(obs.costs.extra_worker) + mediaCostCents(obs, draft.media);
}

/** Mirrors the engine's validation (web_api_v2.md "Semantics"): pure, never throws. */
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
    errors.push({ code: "INVALID_ALLOCATION", message: "Extra workers must be a whole number ≥ 0." });
  }
  const workersAvailable = obs.costs.free_workers + Math.max(0, draft.extra);
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
    workersAvailable,
    workersAssigned,
    workersIdle: Math.max(0, workersAvailable - workersAssigned),
    maxAffordableExtra,
    errors,
  };
}

/** Clamp a worker stepper change so the total never exceeds free + extra. */
export function setRegionWorkers(draft: AllocationDraft, freeWorkers: number, region: number, value: number): AllocationDraft {
  const others = draft.workers.reduce((a, b, i) => (i === region ? a : a + b), 0);
  const cap = Math.max(0, freeWorkers + draft.extra - others);
  const v = Math.max(0, Math.min(cap, Math.round(value)));
  const workers = draft.workers.slice();
  workers[region] = v;
  return { ...draft, workers };
}

/** Change the extra-worker purchase; if workers would exceed the pool, trim from the largest regions. */
export function setExtraWorkers(draft: AllocationDraft, freeWorkers: number, extra: number): AllocationDraft {
  const e = Math.max(0, Math.round(extra));
  const workers = draft.workers.slice();
  let over = workers.reduce((a, b) => a + b, 0) - (freeWorkers + e);
  while (over > 0) {
    const i = workers.indexOf(Math.max(...workers));
    workers[i] -= 1;
    over -= 1;
  }
  return { ...draft, extra: e, workers };
}
