import { describe, expect, it } from "vitest";
import { apportion, percentToManifesto, rebalance, similarity, toIntegerSimplex } from "@/lib/simplex";

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

function lcg(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

describe("simplex sliders", () => {
  it("apportion is exact, integral and non-negative", () => {
    expect(apportion(100, [1, 1, 1])).toEqual([34, 33, 33]);
    expect(sum(apportion(7, [0.1, 0.2, 0.7]))).toBe(7);
    expect(apportion(10, [0, 0, 0, 0, 0])).toEqual([2, 2, 2, 2, 2]);
    expect(apportion(0, [3, 4])).toEqual([0, 0]);
  });

  it("moving one slider rebalances the others proportionally", () => {
    expect(rebalance([20, 20, 20, 20, 20], 0, 60)).toEqual([60, 10, 10, 10, 10]);
    expect(rebalance([40, 30, 10, 10, 10], 0, 70)).toEqual([70, 15, 5, 5, 5]);
    // others at zero share the remainder equally
    expect(rebalance([100, 0, 0, 0, 0], 0, 60)).toEqual([60, 10, 10, 10, 10]);
  });

  it("clamps out-of-range and garbage targets", () => {
    expect(rebalance([20, 20, 20, 20, 20], 2, 150)).toEqual([0, 0, 100, 0, 0]);
    expect(rebalance([20, 20, 20, 20, 20], 2, -30)).toEqual([25, 25, 0, 25, 25]);
    expect(sum(rebalance([20, 20, 20, 20, 20], 1, Number.NaN))).toBe(100);
  });

  it("respects locked sliders and caps the moved slider", () => {
    const locked = [false, true, false, true, false];
    const next = rebalance([20, 30, 10, 25, 15], 0, 90, locked);
    expect(next[1]).toBe(30);
    expect(next[3]).toBe(25);
    expect(next[0]).toBe(45); // 100 - 55 locked
    expect(sum(next)).toBe(100);
    // everything else locked: moved slider is pinned
    expect(rebalance([20, 20, 20, 20, 20], 0, 50, [false, true, true, true, true])).toEqual([20, 20, 20, 20, 20]);
  });

  it("fuzz: sum is always exactly 100 and nothing goes negative", () => {
    const rnd = lcg(42);
    let v = [20, 20, 20, 20, 20];
    for (let k = 0; k < 5000; k++) {
      const i = Math.floor(rnd() * 5);
      const t = Math.round(rnd() * 140 - 20);
      const locked = [0, 1, 2, 3, 4].map(() => rnd() < 0.2);
      const before = v.slice();
      v = rebalance(v, i, t, locked);
      expect(sum(v)).toBe(100);
      v.forEach((x) => {
        expect(Number.isInteger(x)).toBe(true);
        expect(x).toBeGreaterThanOrEqual(0);
      });
      locked.forEach((l, j) => {
        if (l && j !== i) expect(v[j]).toBe(before[j]);
      });
    }
  });

  it("converts to a valid manifesto vector", () => {
    for (const p of [[20, 20, 20, 20, 20], [33, 33, 34, 0, 0], [1, 2, 3, 4, 90], toIntegerSimplex([0.15, 0.2, 0.4, 0.1, 0.15])]) {
      const m = percentToManifesto(p);
      expect(Math.abs(sum(m) - 1)).toBeLessThan(1e-9);
      m.forEach((x) => expect(x).toBeGreaterThanOrEqual(0));
    }
    expect(similarity([0.2, 0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2])).toBeCloseTo(1);
    expect(similarity([1, 0, 0, 0, 0], [0, 1, 0, 0, 0])).toBeCloseTo(0);
  });
});
