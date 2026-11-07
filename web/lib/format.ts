export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");
export const fmtPct = (x: number, digits = 1) => `${x.toFixed(digits)}%`;
/** share in [0,1] → "47.3%" */
export const fmtShare = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
export function fmtCompact(n: number): string {
  if (n >= 1_000_000) return `${+(n / 1_000_000).toFixed(n % 1_000_000 === 0 ? 0 : 2)}M`;
  if (n >= 1_000) return `${+(n / 1_000).toFixed(n % 1_000 === 0 ? 0 : 1)}k`;
  return String(n);
}
