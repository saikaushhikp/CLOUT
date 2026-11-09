/**
 * Shared helpers for the real-engine browser walkthrough (P6). No mocks: every /api call goes through
 * Next.js to the FastAPI engine.
 */
import { expect, type Locator, type Page, type TestInfo } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

export const N = 10_000;
export const SHOT_DIR = path.resolve(__dirname, "../../reports/v2/screenshots");

/* ------------------------------------------------------------------ console / network diagnostics */
export type Diagnostics = {
  console: string[];
  pageErrors: string[];
  failedRequests: string[];
  httpErrors: string[];
};

/** Collect console errors/warnings, uncaught page errors, failed requests and HTTP ≥ 400 responses. */
export function watchDiagnostics(page: Page, opts: { allowHttp?: (url: string, status: number) => boolean } = {}): Diagnostics {
  const d: Diagnostics = { console: [], pageErrors: [], failedRequests: [], httpErrors: [] };
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") d.console.push(`[${m.type()}] ${m.text()}`);
  });
  page.on("pageerror", (e) => d.pageErrors.push(String(e)));
  page.on("requestfailed", (r) => {
    const why = r.failure()?.errorText ?? "";
    // Navigations/aborted fetches when a test ends are not app failures.
    if (why.includes("ERR_ABORTED") || why.includes("NS_BINDING_ABORTED")) return;
    d.failedRequests.push(`${r.method()} ${r.url()} — ${why}`);
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && !(opts.allowHttp?.(r.url(), r.status()) ?? false)) d.httpErrors.push(`${r.status()} ${r.request().method()} ${r.url()}`);
  });
  return d;
}

export function expectClean(d: Diagnostics) {
  expect.soft(d.pageErrors, "uncaught page errors").toEqual([]);
  expect.soft(d.console, "console errors/warnings").toEqual([]);
  expect.soft(d.failedRequests, "failed requests").toEqual([]);
  expect.soft(d.httpErrors, "HTTP error responses").toEqual([]);
}

/* ------------------------------------------------------------------ screenshots */
export async function shot(page: Page, name: string, opts: { fullPage?: boolean; keepToasts?: boolean } = {}) {
  fs.mkdirSync(SHOT_DIR, { recursive: true });
  // Transient toasts would cover controls in the report screenshots (they auto-dismiss after 4–6 s anyway).
  if (!opts.keepToasts) for (const b of await page.getByRole("button", { name: "Dismiss notification" }).all()) await b.click().catch(() => undefined);
  // Let entrance animations (toasts, cards) settle so the shots are representative.
  if (opts.fullPage ?? true) await page.evaluate(() => window.scrollTo(0, 0)); // sticky header stays on top
  await page.waitForTimeout(350);
  await page.screenshot({ path: path.join(SHOT_DIR, `${name}.png`), fullPage: opts.fullPage ?? true, animations: "disabled" });
}

/** No horizontal page scroll: the document is never wider than the viewport. */
export async function expectNoHorizontalScroll(page: Page, where: string) {
  const r = await page.evaluate(() => {
    const vw = window.innerWidth;
    const offenders: string[] = [];
    // Elements poking out of the viewport that are not inside a horizontally scrollable container.
    document.querySelectorAll("body *").forEach((el) => {
      const rect = el.getBoundingClientRect();
      if (rect.width === 0 || rect.right <= vw + 1) return;
      for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
        const ox = getComputedStyle(p).overflowX;
        if (ox === "auto" || ox === "scroll" || ox === "hidden" || ox === "clip") return;
      }
      const cls = typeof el.className === "string" ? el.className : "";
      offenders.push(`${el.tagName.toLowerCase()}${cls ? "." + cls.split(/\s+/).join(".") : ""} right=${Math.round(rect.right)}`);
    });
    return { vw, doc: document.documentElement.scrollWidth, body: document.body.scrollWidth, offenders: offenders.slice(0, 8) };
  });
  expect(r.doc, `${where}: documentElement.scrollWidth`).toBeLessThanOrEqual(r.vw);
  expect(r.body, `${where}: body.scrollWidth (${r.offenders.join(", ")})`).toBeLessThanOrEqual(r.vw);
  expect(r.offenders, `${where}: elements overflowing the viewport`).toEqual([]);
}

/* ------------------------------------------------------------------ game flow */
export async function startGame(page: Page, seed: number, rounds = 5, budget?: number) {
  await page.goto("/");
  await expect(page.getByText(/Engine online/)).toBeVisible();
  const exact = page.getByLabel("Exact number of voters");
  await exact.fill(String(N));
  await exact.blur();
  await page.locator("#seed").fill(String(seed));
  const roundsInput = page.getByRole("spinbutton", { name: "Campaign rounds" });
  await roundsInput.fill(String(rounds));
  if (budget !== undefined) {
    // R17: starting treasury ₫500–₫1000
    const field = page.getByLabel("Starting treasury in credits");
    await field.fill(String(budget));
    await field.blur();
    await expect(page.getByTestId("summary-budget")).toContainText(`Đ${budget.toLocaleString("en-US")}.00`);
  }
  await page.getByRole("button", { name: "Start campaign" }).click();
  await expect(page.getByRole("heading", { name: "Find out what the regions want" })).toBeVisible({ timeout: 60_000 });
}

/** The game id of the running session (sessionStorage), for cleanup through the API. */
export async function currentGameId(page: Page): Promise<string | null> {
  return page.evaluate(() => {
    try {
      const raw = sessionStorage.getItem("dumbocracy.v2.session");
      return raw ? (JSON.parse(raw) as { g: string }).g : null;
    } catch {
      return null;
    }
  });
}

export async function deleteGame(page: Page, gameId: string | null) {
  if (!gameId) return;
  await page.request.delete(`/api/games/${gameId}`).catch(() => undefined);
}

/** Ask one paid question through the query builder (`scope` = the panel holding the builder). */
export async function ask(scope: Locator, a: string, b: string, region: string) {
  await scope.getByLabel("Sector A").selectOption({ label: a });
  await scope.getByLabel("Sector B").selectOption({ label: b });
  await scope.getByRole("radio", { name: region }).click();
  const btn = scope.getByRole("button", { name: /^(Ask|Ask again) · Đ/ });
  await expect(btn).toBeEnabled();
  await btn.click();
  await expect(btn).toBeEnabled({ timeout: 20_000 }); // busy spinner gone
}

export async function setManifesto(page: Page, pct: number[]) {
  const names = ["Education", "Welfare", "Agriculture", "Industry", "Environment"];
  // Lock each sector after typing it so later sectors cannot rebalance it; the last unlocked one absorbs the rest.
  for (let i = 0; i < names.length - 1; i++) {
    await page.getByLabel(`${names[i]} percent`).fill(String(pct[i]));
    await page.getByRole("button", { name: `Lock ${names[i]}` }).click();
  }
  for (let i = 0; i < names.length - 1; i++) await page.getByRole("button", { name: `Unlock ${names[i]}` }).click();
}

export async function workers(page: Page, region: string, n: number) {
  const input = page.getByRole("spinbutton", { name: `Workers in ${region}` });
  await input.fill(String(n));
  await expect(input).toHaveValue(String(n));
}

/* ------------------------------------------------------------------ WebGL canvas check */
/**
 * Screenshot the network canvas and count pixels close to the given palette (decoded in the page via a
 * 2-D canvas: the WebGL buffers themselves are not readable without preserveDrawingBuffer).
 */
export async function paletteHits(page: Page, target: Locator, palette: readonly string[], savePath?: string) {
  const png = await target.screenshot({ animations: "disabled", ...(savePath ? { path: savePath } : {}) });
  return page.evaluate(
    async ({ b64, palette }) => {
      const img = new Image();
      img.src = `data:image/png;base64,${b64}`;
      await img.decode();
      const c = document.createElement("canvas");
      c.width = img.width;
      c.height = img.height;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      const { data } = ctx.getImageData(0, 0, c.width, c.height);
      const rgb = palette.map((h) => [1, 3, 5].map((k) => parseInt(h.slice(k, k + 2), 16)));
      const hits = palette.map(() => 0);
      const bg = [data[0], data[1], data[2]];
      let nonBg = 0;
      for (let i = 0; i < data.length; i += 4) {
        const px = [data[i], data[i + 1], data[i + 2]];
        if (Math.abs(px[0] - bg[0]) + Math.abs(px[1] - bg[1]) + Math.abs(px[2] - bg[2]) > 30) nonBg += 1;
        let best = -1;
        let bestD = 36;
        rgb.forEach((p, k) => {
          const d = Math.abs(px[0] - p[0]) + Math.abs(px[1] - p[1]) + Math.abs(px[2] - p[2]);
          if (d < bestD) {
            bestD = d;
            best = k;
          }
        });
        if (best >= 0) hits[best] += 1;
      }
      return { width: c.width, height: c.height, nonBg, hits };
    },
    { b64: png.toString("base64"), palette: [...palette] },
  );
}

export const REGION_HEX = ["#6250d6", "#eb6834", "#1baf7a", "#2a78d6"] as const;
export const ALIGNMENT_HEX = ["#e34948", "#f0a3a0", "#9a9993", "#86b6ef", "#2a78d6"] as const;

/* ------------------------------------------------------------------ privacy scan */
export type Recorded = { url: string; status: number; body: unknown };

/** Keys that must never appear in any payload during play (spec §6.6, web_api_v2.md "Semantics"). */
const FORBIDDEN_KEYS = new Set([
  "preferences", "preference", "pref", "prefs", "v", "v_i", "values",
  "susceptibility", "susceptibilities", "alpha", "beta", "gamma",
  "zealot", "zealots", "is_zealot", "zealot_mask",
  "archetype", "archetype_id", "archetypes", "personality", "personality_id", "personalities",
  "p_intra", "p_inter", "p_r", "p_rs", "sparsity", "block_probs",
  "sigma", "tau", "s", "opinions", "alignment_true", "true_share", "expected_share_true",
]);
const FORBIDDEN_TEXT = [/zealot/i, /susceptib/i, /p_intra/i, /p_inter/i, /sparsity/i, /preferences/i];

export function privacyViolations(rec: Recorded[], numVoters: number): string[] {
  const out: string[] = [];
  const walk = (x: unknown, at: string, url: string) => {
    if (Array.isArray(x)) {
      if (x.length >= numVoters) out.push(`${url} ${at}: array of length ${x.length} (≥ N)`);
      x.forEach((y, i) => walk(y, `${at}[${i}]`, url));
    } else if (x && typeof x === "object") {
      for (const [k, v] of Object.entries(x)) {
        if (FORBIDDEN_KEYS.has(k)) out.push(`${url} ${at}.${k}: forbidden key`);
        walk(v, `${at}.${k}`, url);
      }
    }
  };
  for (const r of rec) {
    const u = new URL(r.url).pathname;
    if (/\/(debrief|epistemic)$/.test(u)) out.push(`${u}: hidden-truth route requested before DONE`);
    walk(r.body, "$", u);
    const text = JSON.stringify(r.body);
    for (const re of FORBIDDEN_TEXT) if (re.test(text)) out.push(`${u}: text matches ${re}`);
    if (/graph_sample/.test(u)) {
      const g = r.body as { revealed?: boolean; alignment?: unknown; nodes?: Record<string, unknown>[] };
      if (g.revealed !== false) out.push(`${u}: revealed=${String(g.revealed)} before DONE`);
      if (g.alignment !== null) out.push(`${u}: alignment is not null before DONE`);
      const keys = new Set((g.nodes ?? []).flatMap((n) => Object.keys(n)));
      for (const k of keys) if (!["id", "region", "degree"].includes(k)) out.push(`${u}: node key ${k}`);
    }
    // Region entries of any observation carry the public census only.
    const obs = (r.body as { observation?: { regions?: Record<string, unknown>[] }; regions?: Record<string, unknown>[] })?.observation ?? r.body;
    for (const reg of ((obs as { regions?: Record<string, unknown>[] })?.regions ?? [])) {
      for (const k of Object.keys(reg)) if (!["id", "label", "voters", "intra_edges", "mean_degree"].includes(k)) out.push(`${u}: region key ${k}`);
    }
  }
  return out;
}

/**
 * Record every JSON /api response in arrival order. `duringPlay` keeps everything up to and including the
 * response that moved the game to DONE (later responses — the revealed graph, debrief, epistemic timeline —
 * are allowed to carry the hidden truth).
 */
export function recordApi(page: Page): { all: Recorded[]; duringPlay: () => Recorded[] } {
  const all: (Recorded & { seq: number })[] = [];
  let seq = 0;
  page.on("response", async (r) => {
    if (!new URL(r.url()).pathname.startsWith("/api/")) return;
    const mySeq = seq++;
    try {
      const body = await r.json();
      all.push({ url: r.url(), seq: mySeq, status: r.status(), body });
    } catch {
      /* non-JSON (e.g. 204) */
    }
  });
  const phaseOf = (b: unknown) => (b as { observation?: { phase?: string }; phase?: string })?.observation?.phase ?? (b as { phase?: string })?.phase;
  return {
    all,
    duringPlay: () => {
      const sorted = [...all].sort((a, b) => a.seq - b.seq);
      const k = sorted.findIndex((r) => phaseOf(r.body) === "DONE");
      return k < 0 ? sorted : sorted.slice(0, k + 1);
    },
  };
}

export function attachJson(testInfo: TestInfo, name: string, data: unknown) {
  return testInfo.attach(name, { body: JSON.stringify(data, null, 2), contentType: "application/json" });
}
