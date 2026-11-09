/**
 * Public statistics of the sampled paid query (spec §6.3, decision R12). Shared by the mock engine
 * and the UI so both agree on sample sizes and margins of error.
 */
import type { QueryAnswer } from "./types";
import { QUERY_SAMPLE_FRACTION } from "./constants";

const Z95 = 1.96;

/** m = max(1, round(f · n_r)), capped at n_r. */
export function querySampleSize(regionVoters: number, fraction = QUERY_SAMPLE_FRACTION): number {
  return Math.min(Math.max(1, regionVoters), Math.max(1, Math.round(fraction * regionVoters)));
}

/** Smoothed proportion p̃ = (yes + 1)/(m + 2) — keeps the MoE strictly positive. */
const smoothed = (yes: number, m: number) => (yes + 1) / (m + 2);

/** 95% MoE (percentage points) of one sample of m out of a region of n voters, with finite-population correction. */
export function sampleMoe(yes: number, m: number, n: number): number {
  const p = smoothed(yes, m);
  const fpc = n > 1 ? Math.max(0, (n - m) / (n - 1)) : 0;
  return Z95 * 100 * Math.sqrt(((p * (1 - p)) / m) * fpc);
}

/** 95% MoE (percentage points) of the pooled estimate Σyes/Σm (no FPC, per spec). */
export function pooledMoe(yes: number, m: number): number {
  const p = smoothed(yes, m);
  return Z95 * 100 * Math.sqrt((p * (1 - p)) / m);
}

/** Key identifying answers that are pooled together: same triple, same phase, same round. */
export const poolKey = (q: Pick<QueryAnswer, "sector_a" | "sector_b" | "region" | "phase" | "round">) =>
  `${q.phase}:${q.round}:${q.region}:${q.sector_a}>${q.sector_b}`;

export type AnswerGroup = {
  key: string;
  /** every repeat of the triple this phase/round, in ask order */
  answers: QueryAnswer[];
  /** the most recent answer (carries the pooled estimate over all repeats so far) */
  latest: QueryAnswer;
  pct: number;
  moe: number;
  sampleSize: number;
};

/** Group answers by (triple, phase, round); the pooled value is the one reported with the latest repeat. */
export function groupAnswers(queries: QueryAnswer[]): AnswerGroup[] {
  const map = new Map<string, QueryAnswer[]>();
  for (const q of queries) {
    const k = poolKey(q);
    const list = map.get(k);
    if (list) list.push(q);
    else map.set(k, [q]);
  }
  return [...map.entries()].map(([key, answers]) => {
    const latest = answers.reduce((best, q) => (q.repeat_index >= best.repeat_index ? q : best), answers[0]);
    return { key, answers, latest, pct: latest.pooled_pct, moe: latest.pooled_moe, sampleSize: latest.pooled_sample_size };
  });
}

/** "62.4% ± 4.1 (n = 280)" */
export const formatEstimate = (pct: number, moe: number, n: number) =>
  `${pct.toFixed(1)}% ± ${moe.toFixed(1)} (n = ${n.toLocaleString("en-US")})`;
