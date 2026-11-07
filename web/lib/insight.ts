/**
 * Client-side reasoning helpers built ONLY from public information (archetype library, rules, and the
 * player's own query answers). Nothing here reads hidden state.
 */
import type { Observation, QueryAnswer } from "./types";
import { SECTOR_IDS } from "./types";
import { KAPPA } from "./constants";

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

export type ArchetypeFit = { id: string; name: string; rmse: number; n: number; mean: number[] };

/** Rank archetypes by how well they explain the player's answers about one region (lower RMSE = better). */
export function fitArchetypes(obs: Pick<Observation, "archetype_library">, answers: QueryAnswer[], region: string): ArchetypeFit[] {
  const rel = answers.filter((q) => q.region === region);
  return obs.archetype_library
    .map((arch) => {
      if (rel.length === 0) return { id: arch.id, name: arch.name, rmse: NaN, n: 0, mean: arch.mean };
      let se = 0;
      for (const q of rel) {
        const pred = predictPctBOverA(arch.mean, sectorIndex(q.sector_a), sectorIndex(q.sector_b));
        se += (pred - q.pct_b_over_a) ** 2;
      }
      return { id: arch.id, name: arch.name, rmse: Math.sqrt(se / rel.length), n: rel.length, mean: arch.mean };
    })
    .sort((x, y) => (Number.isNaN(x.rmse) ? 0 : x.rmse) - (Number.isNaN(y.rmse) ? 0 : y.rmse));
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
