/**
 * (a) full game, (f) privacy of every payload during play, (g) WebGL network view at DONE only (R18) —
 * desktop 1440×900. Also: chosen starting treasury (R17), permanent hires (R16), round look-back (R19),
 * blend picker (R20), committed manifesto panel during the rounds.
 */
import { expect, test } from "@playwright/test";
import {
  ALIGNMENT_HEX,
  N,
  SHOT_DIR,
  ask,
  attachJson,
  currentGameId,
  deleteGame,
  expectClean,
  paletteHits,
  privacyViolations,
  recordApi,
  setManifesto,
  shot,
  startGame,
  watchDiagnostics,
  workers,
} from "./helpers";

const SEED = 20261108;
const BUDGET = 750;

test("full game: setup → survey → manifesto → 5 rounds (look-back) → results, private payloads, WebGL network at DONE", async ({ page }, testInfo) => {
  const diag = watchDiagnostics(page);
  const api = recordApi(page);
  const timings: Record<string, number> = {};
  let gameId: string | null = null;

  try {
    /* ---------------------------------------------------------------- setup */
    await page.goto("/");
    await expect(page.getByText(/Engine online/)).toBeVisible();
    await shot(page, "desktop-01-setup");
    let t0 = Date.now();
    await startGame(page, SEED, 5, BUDGET);
    timings.create_ms = Date.now() - t0;
    gameId = await currentGameId(page);
    expect(gameId).toBeTruthy();
    await expect(page.locator(".treasury-chip")).toContainText("Đ750.00");

    /* ---------------------------------------------------------------- survey */
    const top3 = page.getByRole("list", { name: "Share of all voters with each sector in their top three" });
    await expect(top3.getByRole("listitem")).toHaveCount(5);
    await expect(top3).toContainText("%");
    const builder = page.getByRole("form", { name: "Paid survey question" });
    await shot(page, "desktop-02-survey-start");

    const qTimes: number[] = [];
    for (const [a, b, r] of [
      ["Education", "Agriculture", "North"],
      ["Industry", "Environment", "East"],
      ["Education", "Welfare", "South"],
      ["Education", "Agriculture", "North"], // repeat → pooled
    ] as const) {
      t0 = Date.now();
      await ask(page.locator("section", { has: builder }), a, b, r);
      qTimes.push(Date.now() - t0);
    }
    timings.query_ms_max = Math.max(...qTimes);
    const flash = page.locator(".answer-flash");
    await expect(flash).toContainText(/Pooled over 2 samples/);
    await expect(flash).toContainText(/±/);
    const ledger = page.getByRole("list", { name: "Your survey answers" });
    await expect(ledger.getByText(/pooled ×2/)).toBeVisible();
    await expect(ledger.getByTestId("ledger-estimate").first()).toContainText(/%.*±/);
    await expect(page.getByText("4 questions · Đ20.00 spent")).toBeVisible();
    await page.getByLabel("Your hunch").nth(0).selectOption({ index: 1 });
    await page.getByLabel("Your hunch").nth(1).selectOption({ index: 2 });
    await shot(page, "desktop-03-survey-answers-pooled");

    /* ---------------------------------------------------------------- manifesto */
    await page.getByRole("tab", { name: /Manifesto/ }).click();
    await expect(page.getByRole("heading", { name: "Write the manifesto" })).toBeVisible();
    // R20: blend picker — choose which regions (2–4) to blend
    await page.getByRole("button", { name: /Blend my hunches/ }).click();
    const picker = page.getByRole("group", { name: "Blend which regions?" });
    await expect(picker).toBeVisible();
    await expect(picker.getByText(/averages their budget profiles, weighted by how many voters live there/)).toBeVisible();
    await expect(picker.getByRole("checkbox")).toHaveCount(4);
    await expect(picker.getByRole("checkbox", { disabled: true })).toHaveCount(2); // two regions without a hunch
    await picker.getByRole("button", { name: "2 largest" }).click();
    await expect(picker.getByRole("checkbox", { checked: true })).toHaveCount(2);
    await shot(page, "desktop-04a-blend-picker");
    await picker.getByRole("button", { name: /Apply blend/ }).click();
    await expect(picker).toHaveCount(0);
    await expect(page.locator(".sum-pill")).toHaveText("Σ 100%");
    await setManifesto(page, [30, 15, 25, 15, 15]);
    const sumPill = page.locator(".sum-pill");
    await expect(sumPill).toHaveText("Σ 100%");
    const values = await Promise.all(
      ["Education", "Welfare", "Agriculture", "Industry", "Environment"].map((s) => page.getByLabel(`${s} percent`).inputValue()),
    );
    expect(values.map(Number)).toEqual([30, 15, 25, 15, 15]);
    // dragging one slider keeps the total at 100
    await page.locator("#ms-Env").focus();
    await page.keyboard.press("ArrowRight");
    await expect(sumPill).toHaveText("Σ 100%");
    await page.getByLabel("Environment percent").fill("15");
    await expect(sumPill).toHaveText("Σ 100%");
    await shot(page, "desktop-04-manifesto");
    await page.getByRole("button", { name: "Commit manifesto" }).click();
    const dialog = page.getByRole("dialog", { name: "Lock in this manifesto?" });
    await expect(dialog).toBeVisible();
    await shot(page, "desktop-05-manifesto-confirm", { fullPage: false });
    await dialog.getByRole("button", { name: /Commit & start round 1/ }).click();

    /* ---------------------------------------------------------------- campaign rounds */
    await expect(page.getByRole("heading", { name: "Round 1 of 5" })).toBeVisible();
    const pollChart = page.getByRole("img", { name: /^Poll history/ });
    await expect(pollChart).toHaveAttribute("aria-label", /R1: poll/);

    // R18: no network panel (and no WebGL canvas) during the rounds; the committed manifesto is shown instead.
    await expect(page.getByRole("heading", { name: "Social network (sample)" })).toHaveCount(0);
    await expect(page.locator(".network-canvas")).toHaveCount(0);
    const manifestoPanel = page.getByTestId("manifesto-summary");
    await expect(manifestoPanel).toBeVisible();
    await expect(manifestoPanel.getByRole("img", { name: /^Manifesto: Education 30%, Welfare 15%, Agriculture 25%, Industry 15%, Environment 15%$/ })).toBeVisible();
    // "How it works" popover in the orders panel
    await page.getByText("How it works").click();
    await expect(page.getByRole("note", { name: "How workers are hired and used" })).toContainText("up to 3 hops");
    await page.getByText("How it works").click();
    await shot(page, "desktop-06-campaign-round1");

    // R16: hires are permanent, so the team grows: 10 → 12 → 15 → 15 → 19.
    const plans = [
      { w: { North: 4, South: 3, East: 2, West: 1 }, extra: 0, media: ["North"] },
      { w: { North: 3, South: 3, East: 3, West: 3 }, extra: 2, media: ["West"] },
      { w: { North: 5, South: 3, East: 3, West: 2 }, extra: 3, media: [] },
      { w: { North: 4, South: 4, East: 4, West: 3 }, extra: 0, media: ["East", "North"] },
      { w: { North: 5, South: 5, East: 5, West: 4 }, extra: 4, media: ["South"] },
    ];
    let team = 10;
    const roundTimes: number[] = [];
    let prevPollLabel = await pollChart.getAttribute("aria-label");
    for (let t = 1; t <= 5; t++) {
      await expect(page.getByRole("heading", { name: `Round ${t} of 5` })).toBeVisible();
      const plan = plans[t - 1];
      await expect(page.getByTestId("team-line")).toContainText(`= ${team} available`);
      if (plan.extra) {
        const extra = page.getByRole("spinbutton", { name: "New hires" });
        await extra.fill(String(plan.extra));
        await expect(extra).toHaveValue(String(plan.extra));
      }
      for (const [r, n] of Object.entries(plan.w)) await workers(page, r, n);
      for (const r of plan.media) {
        const toggle = page.getByRole("button", { name: new RegExp(`^Air media in ${r} for`) });
        await toggle.click();
        await expect(toggle).toHaveAttribute("aria-pressed", "true");
      }
      const orders = page.getByRole("form", { name: `Round ${t} allocation` });
      await expect(orders.getByText(new RegExp(`${Object.values(plan.w).reduce((a, b) => a + b, 0)}/${team + plan.extra} deployed`))).toBeVisible();
      await expect(orders.getByRole("alert")).toHaveCount(0);
      if (t === 2) {
        // one paid question mid-campaign, from the Intel panel
        await ask(page.locator("section", { has: page.getByRole("heading", { name: "Intel" }) }), "Industry", "Environment", "East");
        await shot(page, "desktop-07-campaign-round2-planned");
      }
      if (t === 3) {
        // R19: look back at round 2 from the round tracker (keyboard), then return to planning
        const tracker = page.getByRole("list", { name: /Campaign rounds/ });
        await tracker.getByRole("button", { name: "Review round 2" }).focus();
        await page.keyboard.press("Enter");
        const sheet = page.getByRole("dialog", { name: "Round 2 in review" });
        await expect(sheet).toBeVisible();
        await expect(sheet.getByText(/±\d+\.\d pts · n = /).first()).toBeVisible();
        await expect(sheet.getByText(/10 on staff \+ 2 new permanent hires · 12 deployed/)).toBeVisible();
        await expect(sheet.getByText("Media: West")).toBeVisible();
        await expect(sheet.getByText("Questions asked (1)")).toBeVisible();
        await expect(sheet.getByText("Word of mouth")).toBeVisible();
        await shot(page, "desktop-07b-round-lookback", { fullPage: false });
        await page.keyboard.press("Escape");
        await expect(sheet).toHaveCount(0);
        // the top-bar track opens it too; the current-round chip in the sheet returns to planning
        await page.getByRole("list", { name: "Game progress" }).getByRole("button", { name: "Review round 1" }).click();
        const review = page.getByRole("dialog", { name: "Round 1 in review" });
        await expect(review).toBeVisible();
        // the current-round chip exists in both the top bar and the sheet; use the sheet's
        await review.getByRole("button", { name: "Round 3 — back to planning" }).click();
        await expect(page.getByRole("dialog")).toHaveCount(0);
      }
      const submit = orders.getByRole("button", { name: t === 5 ? /Final push/ : `Launch round ${t}` });
      await expect(submit).toBeEnabled();
      t0 = Date.now();
      await submit.click();
      team += plan.extra;
      if (t < 5) {
        await expect(page.getByRole("heading", { name: `Round ${t + 1} of 5` })).toBeVisible({ timeout: 60_000 });
        roundTimes.push(Date.now() - t0);
        await expect(page.getByLabel(`Round ${t} report`)).toBeVisible();
        const label = await pollChart.getAttribute("aria-label");
        expect(label, "poll chart gained a point").not.toEqual(prevPollLabel);
        expect(label).toContain(`R${t + 1}: poll`);
        prevPollLabel = label;
        if (t === 4) await shot(page, "desktop-08-campaign-round5");
      }
    }

    /* ---------------------------------------------------------------- results */
    await expect(page.getByRole("heading", { name: /You (won|lost)\./ })).toBeVisible({ timeout: 60_000 });
    roundTimes.push(Date.now() - t0);
    timings.round_ms_max = Math.max(...roundTimes);
    timings.round_ms_mean = Math.round(roundTimes.reduce((a, b) => a + b, 0) / roundTimes.length);
    const tDone = Date.now();

    await expect(page.getByRole("heading", { name: "The hidden map, revealed" })).toBeVisible({ timeout: 60_000 });
    const cards = page.locator(".reveal-card");
    await expect(cards).toHaveCount(4);
    await expect(cards.getByText("Archetype")).toHaveCount(4, { timeout: 10_000 }); // all flipped
    await expect(page.getByRole("heading", { name: "Your manifesto vs the oracle" })).toBeVisible();
    await expect(page.getByText(/ORACLE: full-information benchmark/)).toBeVisible();
    await expect(page.getByRole("img", { name: /^Poll history/ }).first()).toHaveAttribute("aria-label", /true/);

    await expect(page.getByTestId("ep-headline")).toBeVisible({ timeout: 60_000 });
    timings.epistemic_ms = Date.now() - tDone;
    await expect(page.locator(".ep-table tbody").first().locator("tr")).toHaveCount(5);
    await expect(page.locator(".ep-table thead th")).toHaveCount(6);
    await shot(page, "desktop-09-results");

    // (g) at DONE: alignment colouring
    const doneNet = page.locator("section", { has: page.getByRole("heading", { name: "How opinion spread" }) });
    await expect(doneNet.getByRole("button", { name: "Alignment" })).toHaveAttribute("aria-pressed", "true", { timeout: 30_000 });
    await expect(doneNet.getByText(/Laying out/)).toHaveCount(0, { timeout: 30_000 });
    await doneNet.scrollIntoViewIfNeeded();
    await page.waitForTimeout(800);
    const atDone = await paletteHits(page, doneNet.locator(".network-canvas"), ALIGNMENT_HEX, `${SHOT_DIR}/desktop-10b-network-at-done-canvas.png`);
    await attachJson(testInfo, "network-at-done", atDone);
    console.log("NETWORK at done", JSON.stringify(atDone));
    expect(atDone.nonBg).toBeGreaterThan(2_000);
    expect(atDone.hits.filter((h) => h > 50).length, "several alignment bins drawn").toBeGreaterThanOrEqual(2);
    // the round slider animates the colouring back to the start
    await doneNet.getByRole("slider", { name: "Round shown" }).fill("0");
    await expect(doneNet.getByText("Start", { exact: true })).toBeVisible();
    await page.waitForTimeout(400);
    const atStart = await paletteHits(page, doneNet.locator(".network-canvas"), ALIGNMENT_HEX);
    expect(atStart.hits).not.toEqual(atDone.hits);
    await doneNet.getByRole("slider", { name: "Round shown" }).fill("5");
    await doneNet.screenshot({ path: `${SHOT_DIR}/desktop-10-results-network-alignment.png` });
    await page.locator(".ep-explainer summary").click();
    await page.getByRole("heading", { name: "Epistemic timeline" }).scrollIntoViewIfNeeded();
    await shot(page, "desktop-11-results-epistemic", { fullPage: false });

    // dark mode at DONE
    await page.getByRole("button", { name: "Switch to dark theme" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot(page, "desktop-12-results-dark");
    await page.getByRole("button", { name: "Switch to light theme" }).click();

    /* ---------------------------------------------------------------- privacy (f) */
    const during_ = api.duringPlay();
    expect(during_.length).toBeGreaterThanOrEqual(13); // health, create, 5 queries, commit, 5 allocations
    // R18: the graph sample is never requested before DONE
    expect(during_.some((r) => /graph_sample/.test(r.url))).toBe(false);
    // R17/R19: the observation carries the chosen budget and one report per resolved round
    const doneObs = (during_[during_.length - 1].body as { observation: { initial_budget: string; round_reports: unknown[]; workers: unknown } }).observation;
    expect(doneObs.initial_budget).toBe("750.00");
    expect(doneObs.round_reports).toHaveLength(5);
    expect(doneObs.workers).toMatchObject({ free: 10, hired: 9, available: 19 });
    const leaks = privacyViolations(during_, N);
    await attachJson(testInfo, "privacy-scan", { responses: during_.length, leaks });
    expect(leaks).toEqual([]);
    // after DONE the graph sample is revealed with one snapshot per round
    const revealed = api.all.find((r) => /graph_sample/.test(r.url) && (r.body as { revealed: boolean }).revealed);
    expect(revealed, "revealed graph sample after DONE").toBeTruthy();
    expect((revealed!.body as { alignment: number[][] }).alignment).toHaveLength(6);

    await attachJson(testInfo, "timings", timings);
    console.log("TIMINGS", JSON.stringify(timings));
    expectClean(diag);
  } finally {
    await deleteGame(page, gameId ?? (await currentGameId(page).catch(() => null)));
  }
});
