/**
 * Client-side reasoning helpers built ONLY from public information (archetype library, rules, and the
 * player's own query answers). Nothing here reads hidden state.
 */
import type { Observation, QueryAnswer } from "./types";
import { SECTOR_IDS } from "./types";
import { KAPPA } from "./constants";
import { groupAnswers } from "./survey";

/** Abramowitz–Stegun erf approximation (|err| < 1.5e-7). */
export function erf(x: number): number {
  const s = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}
export const normalCdf = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2));

export const sectorIndex = (id: string) => SECTOR_IDS.indexOf(id as (typeof SECTOR_IDS)[number]);

/**
 * Predicted % of voters with v[b] > v[a] when v ~ Dirichlet(κ·μ) (normal approximation of the
 * difference of two Dirichlet marginals).
 */
export function predictPctBOverA(mean: number[], a: number, b: number, kappa = KAPPA): number {
  const ma = mean[a];
  const mb = mean[b];
  const variance = (ma * (1 - ma) + mb * (1 - mb) + 2 * ma * mb) / (kappa + 1);
  const sd = Math.sqrt(Math.max(variance, 1e-12));
  return 100 * normalCdf((mb - ma) / sd);
}

export type ArchetypeFit = {
  id: string;
  name: string;
  /** precision-weighted RMS gap between pooled estimates and the archetype's prediction (percentage points) */
  rmse: number;
  /** Gaussian log-likelihood of the region's evidence under this archetype */
  logLik: number;
  /** posterior probability under a uniform prior over the library (0..1) */
  posterior: number;
  /** number of independent pieces of evidence (pooled triple + round groups) */
  n: number;
  mean: number[];
};

/** Model slack (pp) added in quadrature to each sampling sd: Dirichlet approximation error + within-round drift. */
export const FIT_SLACK_PP = 2.5;

/**
 * Rank archetypes by how well they explain the player's answers about one region.
 * Repeats of the same triple in the same phase/round are pooled first (one piece of evidence each, using the
 * pooled estimate ± pooled MoE); each piece then enters a Gaussian likelihood with
 * sd = sqrt((moe/1.96)² + slack²), so wide-margin answers count for less.
 */
export function fitArchetypes(
  obs: Pick<Observation, "archetype_library">,
  answers: QueryAnswer[],
  region: string,
  slack = FIT_SLACK_PP,
): ArchetypeFit[] {
  const groups = groupAnswers(answers.filter((q) => q.region === region));
  const evidence = groups.map((g) => ({
    a: sectorIndex(g.latest.sector_a),
    b: sectorIndex(g.latest.sector_b),
    pct: g.pct,
    sd: Math.sqrt((g.moe / 1.96) ** 2 + slack ** 2),
  }));
  const fits = obs.archetype_library.map((arch) => {
    if (evidence.length === 0) return { id: arch.id, name: arch.name, rmse: NaN, logLik: 0, posterior: NaN, n: 0, mean: arch.mean };
    let ll = 0;
    let wse = 0;
    let w = 0;
    for (const e of evidence) {
      const gap = predictPctBOverA(arch.mean, e.a, e.b) - e.pct;
      ll += -0.5 * (gap / e.sd) ** 2 - Math.log(e.sd);
      wse += gap ** 2 / e.sd ** 2;
      w += 1 / e.sd ** 2;
    }
    return { id: arch.id, name: arch.name, rmse: Math.sqrt(wse / w), logLik: ll, posterior: 0, n: evidence.length, mean: arch.mean };
  });
  if (evidence.length > 0) {
    const top = Math.max(...fits.map((f) => f.logLik));
    const z = fits.reduce((s, f) => s + Math.exp(f.logLik - top), 0);
    fits.forEach((f) => (f.posterior = Math.exp(f.logLik - top) / z));
  }
  return fits.sort((x, y) => (Number.isNaN(y.posterior) ? 0 : y.posterior) - (Number.isNaN(x.posterior) ? 0 : x.posterior));
}

/** Voter-weighted blend of the archetypes the player has pencilled in per region. */
export function blendGuesses(
  obs: Pick<Observation, "regions" | "archetype_library">,
  guesses: Record<string, string | undefined>,
): number[] | null {
  const acc = [0, 0, 0, 0, 0];
  let w = 0;
  for (const r of obs.regions) {
    const g = guesses[r.id];
    const arch = obs.archetype_library.find((a) => a.id === g);
    if (!arch) continue;
    arch.mean.forEach((m, k) => (acc[k] += m * r.voters));
    w += r.voters;
  }
  if (w === 0) return null;
  return acc.map((x) => x / w);
}

/** Regions that have a hunch pencilled in (an archetype from the public library). */
export function regionsWithHunch(
  obs: Pick<Observation, "regions" | "archetype_library">,
  guesses: Record<string, string | undefined>,
): string[] {
  return obs.regions.filter((r) => obs.archetype_library.some((a) => a.id === guesses[r.id])).map((r) => r.id);
}

/** The k most populous regions that have a hunch (for the "2 largest" / "3 largest" shortcuts). */
export function largestHunchRegions(
  obs: Pick<Observation, "regions" | "archetype_library">,
  guesses: Record<string, string | undefined>,
  k: number,
): string[] {
  const eligible = new Set(regionsWithHunch(obs, guesses));
  return [...obs.regions]
    .filter((r) => eligible.has(r.id))
    .sort((a, b) => b.voters - a.voters)
    .slice(0, k)
    .map((r) => r.id);
}

/**
 * "Blend my hunches" (R20): take the archetype the player picked for each CHOSEN region and average their
 * budget profiles, weighted by how many voters live there. Needs 2–4 chosen regions that all have a hunch;
 * returns null otherwise.
 */
export function blendSelected(
  obs: Pick<Observation, "regions" | "archetype_library">,
  guesses: Record<string, string | undefined>,
  regionIds: string[],
  min = 2,
  max = 4,
): number[] | null {
  const chosen = [...new Set(regionIds)];
  if (chosen.length < min || chosen.length > max) return null;
  const sub: Record<string, string | undefined> = {};
  for (const id of chosen) {
    if (!obs.archetype_library.some((a) => a.id === guesses[id])) return null;
    sub[id] = guesses[id];
  }
  return blendGuesses(obs, sub);
}
