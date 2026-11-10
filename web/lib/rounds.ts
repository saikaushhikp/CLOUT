/**
 * Round look-back (R19): rebuild what happened in a past round purely from the player's own observation
 * (polls, allocations, round_reports, queries, initial_budget). Nothing hidden is read.
 */
import type { Observation, QueryAnswer, RoundReport } from "./types";
import { toCents } from "./money";

type LookbackView = Pick<
  Observation,
  "rounds" | "polls" | "allocations" | "queries" | "costs" | "treasury"
> & {
  initial_budget?: string;
  round_reports?: RoundReport[];
};

export type RoundSummary = {
  round: number;
  /** poll taken at the start of the round (null if missing) */
  pollStart: Observation["polls"][number] | null;
  /** poll taken after the round resolved (start of round+1, or the pre-election poll) */
  pollAfter: Observation["polls"][number] | null;
  orders: Observation["allocations"][number] | null;
  report: RoundReport | null;
  queries: QueryAnswer[];
  /** permanent team (all regions) before this round's hires: every worker bought in earlier rounds */
  teamBefore: number;
  /** workers bought this round per region [N,S,E,W] */
  hiresByRegion: number[];
  newHires: number;
  /** permanent team per region after this round's hires */
  teamAfterByRegion: number[];
  queryCostCents: number;
  ordersCostCents: number;
  treasuryBeforeCents: number;
  treasuryAfterCents: number;
  /** false when this round was forfeited / had no orders (all zero) */
  campaigned: boolean;
};

/** Initial treasury, with a fallback for engines that predate R17: current treasury + everything spent. */
export function initialBudgetCents(obs: LookbackView): number {
  if (obs.initial_budget) return toCents(obs.initial_budget);
  const spent =
    obs.queries.reduce((a, q) => a + toCents(q.cost), 0) + obs.allocations.reduce((a, al) => a + toCents(al.cost), 0);
  return toCents(obs.treasury) + spent;
}

const isRoundQuery = (q: QueryAnswer, t: number) => q.phase === "ROUND" && q.round === t;

export function roundSummary(obs: LookbackView, t: number): RoundSummary {
  const orders = obs.allocations.find((a) => a.round === t) ?? null;
  const report = obs.round_reports?.find((r) => r.round === t) ?? null;
  const queries = obs.queries.filter((q) => isRoundQuery(q, t));
  const spentBefore =
    obs.queries.filter((q) => q.phase === "SURVEY" || q.round < t).reduce((a, q) => a + toCents(q.cost), 0) +
    obs.allocations.filter((a) => a.round < t).reduce((a, al) => a + toCents(al.cost), 0);
  const queryCostCents = queries.reduce((a, q) => a + toCents(q.cost), 0);
  const ordersCostCents = orders ? toCents(orders.cost) : report ? toCents(report.cost) : 0;
  const before = initialBudgetCents(obs) - spentBefore;
  const earlier = [0, 0, 0, 0];
  for (const al of obs.allocations) if (al.round < t) al.hires_by_region.forEach((h, i) => (earlier[i] += h));
  const hiresByRegion = (report?.hires_by_region ?? orders?.hires_by_region ?? [0, 0, 0, 0]).slice(0, 4);
  while (hiresByRegion.length < 4) hiresByRegion.push(0);
  const newHires = hiresByRegion.reduce((a, b) => a + b, 0);
  const teamAfterByRegion = report?.team_by_region?.slice(0, 4) ?? earlier.map((e, i) => e + hiresByRegion[i]);
  return {
    round: t,
    pollStart: obs.polls.find((p) => p.round === t) ?? null,
    pollAfter: obs.polls.find((p) => p.round === t + 1) ?? null,
    orders,
    report,
    queries,
    teamBefore: earlier.reduce((a, b) => a + b, 0),
    hiresByRegion,
    newHires,
    teamAfterByRegion,
    queryCostCents,
    ordersCostCents,
    treasuryBeforeCents: before,
    treasuryAfterCents: before - queryCostCents - ordersCostCents,
    campaigned: !!orders && (newHires > 0 || orders.media_regions.some(Boolean)),
  };
}

/** Rounds that can be reviewed: every resolved round (1 … round−1 while playing, 1 … T at DONE). */
export function reviewableRounds(obs: Pick<Observation, "phase" | "round" | "rounds">): number[] {
  const last = obs.phase === "DONE" ? obs.rounds : obs.phase === "ROUND" ? obs.round - 1 : 0;
  return Array.from({ length: Math.max(0, last) }, (_, k) => k + 1);
}
