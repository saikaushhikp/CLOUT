/**
 * Manifesto sliders live on the integer simplex: K non-negative integers (percent points) summing to
 * exactly `total` (100). Moving one slider rebalances the *unlocked* others proportionally to their
 * current values (largest-remainder rounding keeps the sum exact).
 */

export const SIMPLEX_TOTAL = 100;

/** Distribute `amount` (integer >= 0) over `weights` proportionally, as integers summing exactly to amount. */
export function apportion(amount: number, weights: number[]): number[] {
  const n = weights.length;
  if (n === 0) return [];
  const amt = Math.max(0, Math.round(amount));
  const w = weights.map((x) => (Number.isFinite(x) && x > 0 ? x : 0));
  const sumW = w.reduce((a, b) => a + b, 0);
  const basis = sumW > 0 ? w : w.map(() => 1);
  const sumB = sumW > 0 ? sumW : n;
  const raw = basis.map((x) => (amt * x) / sumB);
  const out = raw.map(Math.floor);
  let rest = amt - out.reduce((a, b) => a + b, 0);
  const order = raw
    .map((x, i) => ({ i, frac: x - Math.floor(x), w: basis[i] }))
    .sort((a, b) => b.frac - a.frac || b.w - a.w || a.i - b.i);
  for (let k = 0; rest > 0; k = (k + 1) % n) {
    out[order[k].i] += 1;
    rest -= 1;
  }
  return out;
}

/** Set slider `index` to `target` and rebalance the others. Locked indices never change. */
export function rebalance(
  values: number[],
  index: number,
  target: number,
  locked: boolean[] = [],
  total = SIMPLEX_TOTAL,
): number[] {
  const n = values.length;
  const isLocked = (i: number) => i !== index && Boolean(locked[i]);
  const lockedSum = values.reduce((a, v, i) => (isLocked(i) ? a + v : a), 0);
  const free = values.map((_, i) => i).filter((i) => i !== index && !isLocked(i));
  // With no free partner the moved slider is pinned to whatever keeps the sum exact.
  const maxForIndex = total - lockedSum;
  const minForIndex = free.length === 0 ? maxForIndex : 0;
  const t = Math.min(maxForIndex, Math.max(minForIndex, Math.round(Number.isFinite(target) ? target : 0)));
  const remaining = total - lockedSum - t;
  const parts = apportion(remaining, free.map((i) => values[i]));
  const next = values.slice();
  next[index] = t;
  free.forEach((i, k) => {
    next[i] = parts[k];
  });
  return next;
}

/** Normalise arbitrary non-negative weights into an integer simplex point. */
export function toIntegerSimplex(weights: number[], total = SIMPLEX_TOTAL): number[] {
  return apportion(total, weights);
}

/** Convert integer percentages into the manifesto vector sent to the engine (sums to 1 within 1e-12). */
export function percentToManifesto(values: number[]): number[] {
  const sum = values.reduce((a, b) => a + b, 0);
  const out = values.map((v) => v / sum);
  const drift = 1 - out.reduce((a, b) => a + b, 0);
  const iMax = out.indexOf(Math.max(...out));
  out[iMax] += drift;
  return out;
}

/** Similarity used by the voting model: S = 1 - ½‖a − b‖₁ (inputs are simplex points). */
export function similarity(a: number[], b: number[]): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]);
  return 1 - d / 2;
}
