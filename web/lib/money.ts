/** Money arrives as 2-dp decimal strings ("500.00"). All UI arithmetic is done in integer cents. */

export function toCents(value: string | number): number {
  if (typeof value === "number") return Math.round(value * 100);
  const s = value.trim();
  const m = /^(-)?(\d+)(?:\.(\d{1,2}))?$/.exec(s);
  if (!m) {
    const n = Number(s);
    if (!Number.isFinite(n)) throw new Error(`Bad money value: ${value}`);
    return Math.round(n * 100);
  }
  const cents = Number(m[2]) * 100 + Number((m[3] ?? "0").padEnd(2, "0"));
  return m[1] ? -cents : cents;
}

export function centsToString(cents: number): string {
  const sign = cents < 0 ? "-" : "";
  const abs = Math.abs(Math.round(cents));
  return `${sign}${Math.floor(abs / 100)}.${String(abs % 100).padStart(2, "0")}`;
}

/** Display form, e.g. "Đ 1,234.50" — the game currency is "credits". */
export function formatMoney(value: string | number, opts: { symbol?: boolean } = {}): string {
  const cents = typeof value === "number" ? value : toCents(value);
  const sign = cents < 0 ? "-" : "";
  const abs = Math.abs(cents);
  const whole = Math.floor(abs / 100).toLocaleString("en-US");
  const frac = String(abs % 100).padStart(2, "0");
  return `${sign}${opts.symbol === false ? "" : "Đ"}${whole}.${frac}`;
}
