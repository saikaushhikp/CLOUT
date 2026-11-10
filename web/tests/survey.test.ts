import { describe, expect, it } from "vitest";
import { createMockClient } from "@/lib/mockApi";
import { fitArchetypes, predictPctBOverA, sectorIndex } from "@/lib/insight";
import { formatEstimate, groupAnswers, pooledMoe, querySampleSize, sampleMoe } from "@/lib/survey";
import type { QueryAnswer } from "@/lib/types";
import { centsToString, toCents } from "@/lib/money";

const none = { hires_by_region: [0, 0, 0, 0], media_regions: [false, false, false, false] };
const ask = { sector_a: "Edu", sector_b: "Agri", region: "N" };

describe("sampled query statistics (spec §6.3, R12)", () => {
  it("sample size is round(10% of region voters), at least 1", () => {
    expect(querySampleSize(2800)).toBe(280);
    expect(querySampleSize(2804)).toBe(280);
    expect(querySampleSize(2805)).toBe(281);
    expect(querySampleSize(4)).toBe(1);
  });

  it("MoE uses p̃ = (yes+1)/(m+2) and the finite-population correction", () => {
    const yes = 170;
    const m = 280;
    const n = 2800;
    const p = (yes + 1) / (m + 2);
    const expected = 1.96 * 100 * Math.sqrt(((p * (1 - p)) / m) * ((n - m) / (n - 1)));
    expect(sampleMoe(yes, m, n)).toBeCloseTo(expected, 10);
    expect(sampleMoe(0, m, n)).toBeGreaterThan(0); // never zero
    expect(sampleMoe(m, m, n)).toBeGreaterThan(0);
    expect(sampleMoe(m / 2, m, m)).toBe(0); // whole population surveyed
    // pooled: no FPC, shrinks as the pooled sample grows
    const pp = (340 + 1) / (560 + 2);
    expect(pooledMoe(340, 560)).toBeCloseTo(1.96 * 100 * Math.sqrt((pp * (1 - pp)) / 560), 10);
    expect(pooledMoe(510, 840)).toBeLessThan(pooledMoe(340, 560));
  });

  it("formats an estimate as 'p% ± moe (n = m)'", () => {
    expect(formatEstimate(62.43, 4.12, 280)).toBe("62.4% ± 4.1 (n = 280)");
    expect(formatEstimate(5, 1, 12_000)).toBe("5.0% ± 1.0 (n = 12,000)");
  });
});

describe("mock engine sampled queries", () => {
  it("answers vary between repeats, pool within the round, and reset next round — no DUPLICATE_QUERY", async () => {
    const c = createMockClient();
    const { game_id: g, player_id: p, observation: o0 } = await c.createGame({ num_voters: 10_000, seed: 11, rounds: 3 });
    const nr = o0.regions.find((r) => r.id === "N")!.voters;
    const m = Math.round(0.1 * nr);

    const answers: QueryAnswer[] = [];
    for (let k = 0; k < 4; k++) {
      const res = await c.query(g, p, ask);
      expect(res.accepted).toBe(true);
      expect(res.error_code).toBeNull();
      answers.push(res.answer!);
    }
    expect(new Set(answers.map((a) => a.pct_b_over_a)).size).toBeGreaterThan(1);
    answers.forEach((a, k) => {
      expect(a.phase).toBe("SURVEY");
      expect(a.sample_size).toBe(m);
      expect(a.repeat_index).toBe(k + 1);
      expect(a.pooled_sample_size).toBe((k + 1) * m);
      expect(a.moe).toBeGreaterThan(0);
      expect(a.moe).toBeLessThan(10);
      const yes = Math.round((a.pct_b_over_a / 100) * m);
      expect(a.moe).toBeCloseTo(sampleMoe(yes, m, nr), 1);
    });
    // pooled value = mean of the equal-size samples; pooled MoE shrinks
    const mean = answers.reduce((s, a) => s + a.pct_b_over_a, 0) / answers.length;
    expect(answers[3].pooled_pct).toBeCloseTo(mean, 1);
    expect(answers[0].pooled_pct).toBe(answers[0].pct_b_over_a);
    for (let k = 1; k < 4; k++) expect(answers[k].pooled_moe).toBeLessThan(answers[k - 1].pooled_moe);

    // a different triple starts its own pool
    const other = await c.query(g, p, { ...ask, region: "S" });
    expect(other.answer).toMatchObject({ repeat_index: 1 });
    expect(other.observation.treasury).toBe(centsToString(toCents(o0.treasury) - 5 * toCents(o0.costs.query)));
    expect(other.observation.queries).toHaveLength(5);

    // pooling resets each round
    await c.commit(g, p, { manifesto: [0.2, 0.2, 0.2, 0.2, 0.2] });
    const r1a = await c.query(g, p, ask);
    expect(r1a.answer).toMatchObject({ phase: "ROUND", round: 1, repeat_index: 1, pooled_sample_size: m });
    const r1b = await c.query(g, p, ask);
    expect(r1b.answer).toMatchObject({ repeat_index: 2, pooled_sample_size: 2 * m });
    await c.allocate(g, p, none);
    const r2 = await c.query(g, p, ask);
    expect(r2.answer).toMatchObject({ round: 2, repeat_index: 1, pooled_sample_size: m });
  });

  it("is deterministic for the same seed and query sequence", async () => {
    const run = async () => {
      const c = createMockClient();
      const { game_id: g, player_id: p } = await c.createGame({ num_voters: 8000, seed: 5 });
      const out: number[] = [];
      for (let k = 0; k < 3; k++) out.push((await c.query(g, p, ask)).answer!.pct_b_over_a);
      return out;
    };
    expect(await run()).toEqual(await run());
  });
});

describe("archetype matching weights evidence by margin of error", () => {
  const library = [
    { id: "A0", name: "Agrarian heartland", mean: [0.15, 0.2, 0.4, 0.1, 0.15] },
    { id: "A1", name: "Industrial belt", mean: [0.15, 0.15, 0.1, 0.45, 0.15] },
    { id: "A4", name: "Knowledge hub", mean: [0.4, 0.1, 0.1, 0.25, 0.15] },
  ];
  const obs = { archetype_library: library };
  const truth = library[0].mean;
  const answer = (a: string, b: string, moe: number, extra: Partial<QueryAnswer> = {}): QueryAnswer => {
    const pct = predictPctBOverA(truth, sectorIndex(a), sectorIndex(b));
    return {
      round: 0, phase: "SURVEY", sector_a: a, sector_b: b, region: "N",
      pct_b_over_a: pct, sample_size: 250, moe, pooled_pct: pct, pooled_sample_size: 250, pooled_moe: moe,
      repeat_index: 1, cost: "5.00", ...extra,
    };
  };

  it("posterior of the true archetype decreases monotonically as the MoE widens", () => {
    const moes = [1, 3, 6, 12, 25];
    const post = moes.map((moe) => {
      // a weakly discriminating question so the posterior does not saturate at 1
      const fits = fitArchetypes(obs, [answer("Env", "Welfare", moe)], "N");
      expect(fits[0].id).toBe("A0");
      return fits[0].posterior;
    });
    for (let k = 1; k < post.length; k++) expect(post[k]).toBeLessThan(post[k - 1]);
    // and the log-likelihood margin over the runner-up shrinks for a strongly discriminating pair too
    const margin = moes.map((moe) => {
      const fits = fitArchetypes(obs, [answer("Edu", "Agri", moe), answer("Industry", "Agri", moe)], "N");
      return fits[0].logLik - fits[1].logLik;
    });
    for (let k = 1; k < margin.length; k++) expect(margin[k]).toBeLessThan(margin[k - 1]);
    expect(post[post.length - 1]).toBeGreaterThan(1 / library.length);
  });

  it("a precise answer outweighs a contradictory noisy one", () => {
    const precise = answer("Edu", "Agri", 1.5);
    // a noisy sample about another triple that would point at the Industrial belt
    const misleadPct = predictPctBOverA(library[1].mean, sectorIndex("Agri"), sectorIndex("Industry"));
    const noisy = answer("Agri", "Industry", 30, { pct_b_over_a: misleadPct, pooled_pct: misleadPct });
    expect(fitArchetypes(obs, [precise, noisy], "N")[0].id).toBe("A0");
    const noisyPrecise = { ...noisy, moe: 1, pooled_moe: 1 };
    const sharpBoth = fitArchetypes(obs, [answer("Edu", "Agri", 30), noisyPrecise], "N");
    expect(sharpBoth[0].id).toBe("A1");
  });

  it("repeats of one triple in a round count once, at the pooled estimate", () => {
    const first = answer("Edu", "Agri", 5, { pct_b_over_a: 10, pooled_pct: 10 });
    const second = answer("Edu", "Agri", 5, { repeat_index: 2, pooled_moe: 3 });
    expect(groupAnswers([first, second])).toHaveLength(1);
    const fits = fitArchetypes(obs, [first, second], "N");
    expect(fits[0].n).toBe(1);
    const alone = fitArchetypes(obs, [{ ...second, repeat_index: 1 }], "N");
    expect(fits.map((f) => f.posterior)).toEqual(alone.map((f) => f.posterior));
    // a later round is separate evidence
    const later = answer("Edu", "Agri", 5, { round: 1, phase: "ROUND" });
    expect(fitArchetypes(obs, [first, second, later], "N")[0].n).toBe(2);
  });
});
