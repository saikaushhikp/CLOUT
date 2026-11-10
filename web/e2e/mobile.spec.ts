/** (d) Phone width 390×844: every screen usable, no horizontal page scroll; round look-back as a bottom sheet. */
import { expect, test } from "@playwright/test";
import {
  ask,
  currentGameId,
  deleteGame,
  expectClean,
  expectNoHorizontalScroll,
  setManifesto,
  shot,
  startGame,
  watchDiagnostics,
} from "./helpers";

test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 2 });

test("mobile: full game at 390×844 without horizontal scroll", async ({ page }) => {
  const diag = watchDiagnostics(page);
  let gameId: string | null = null;
  try {
    await page.goto("/");
    await expect(page.getByText(/Engine online/)).toBeVisible();
    await expectNoHorizontalScroll(page, "setup");
    await page.getByRole("button", { name: "Đ750" }).tap();
    await expect(page.getByTestId("summary-budget")).toContainText("Đ750.00");
    await expectNoHorizontalScroll(page, "setup with budget");
    await shot(page, "mobile-01-setup");

    await startGame(page, 909, 5, 1000);
    gameId = await currentGameId(page);
    await expectNoHorizontalScroll(page, "survey");
    const builder = page.locator("section", { has: page.getByRole("form", { name: "Paid survey question" }) });
    await ask(builder, "Education", "Environment", "West");
    await ask(builder, "Education", "Environment", "West");
    await ask(builder, "Welfare", "Industry", "South");
    await expect(page.locator(".answer-flash")).toBeVisible();
    await expectNoHorizontalScroll(page, "survey with answers");
    await shot(page, "mobile-02-survey");

    await page.getByRole("tab", { name: /Manifesto/ }).click();
    await setManifesto(page, [25, 20, 15, 15, 25]);
    await expect(page.locator(".sum-pill")).toHaveText("Σ 100%");
    await expectNoHorizontalScroll(page, "manifesto");
    await shot(page, "mobile-03-manifesto");
    await page.getByRole("button", { name: "Commit manifesto" }).click();
    const dialog = page.getByRole("dialog", { name: "Lock in this manifesto?" });
    await expect(dialog).toBeInViewport();
    const confirm = dialog.getByRole("button", { name: /Commit & start round 1/ });
    await expect(confirm).toBeInViewport();
    await shot(page, "mobile-04-manifesto-confirm", { fullPage: false });
    await confirm.tap();

    for (let t = 1; t <= 5; t++) {
      await expect(page.getByRole("heading", { name: `Round ${t} of 5` })).toBeVisible();
      // P8: buy one permanent worker in North each round (stepper +)
      await page.getByRole("button", { name: "Increase New workers in North" }).tap();
      await expect(page.getByTestId("hire-line-N")).toContainText(`team ${t - 1} · +1 new`);
      if (t % 2) await page.getByRole("button", { name: /^Air media in East for/ }).tap();
      await expectNoHorizontalScroll(page, `round ${t}`);
      if (t === 1) {
        // R18: no network during the rounds; the committed manifesto is shown instead
        await expect(page.locator(".network-canvas")).toHaveCount(0);
        await page.getByTestId("manifesto-summary").scrollIntoViewIfNeeded();
        await expect(page.getByTestId("manifesto-summary")).toBeInViewport();
        await shot(page, "mobile-05-campaign-round1");
        await page.getByRole("form", { name: "Round 1 allocation" }).scrollIntoViewIfNeeded();
        await shot(page, "mobile-06-campaign-orders", { fullPage: false });
      }
      if (t === 3) {
        // R19: tap a past round chip → read-only bottom sheet
        const chip = page.getByRole("list", { name: /Campaign rounds/ }).getByRole("button", { name: "Review round 2" });
        await chip.scrollIntoViewIfNeeded();
        await chip.tap();
        const sheet = page.getByRole("dialog", { name: "Round 2 in review" });
        await expect(sheet).toBeVisible();
        await expect(sheet.getByRole("button", { name: /Close round 2 review/ })).toBeInViewport();
        await expectNoHorizontalScroll(page, "look-back sheet");
        await shot(page, "mobile-06b-round-lookback", { fullPage: false });
        await sheet.getByRole("button", { name: "Round 3 — back to planning" }).tap();
        await expect(sheet).toHaveCount(0);
      }
      const submit = page.getByRole("button", { name: t === 5 ? /Final push/ : `Launch round ${t}` });
      await submit.scrollIntoViewIfNeeded();
      await expect(submit).toBeInViewport();
      await submit.tap();
    }

    await expect(page.getByRole("heading", { name: /You (won|lost)\./ })).toBeVisible({ timeout: 60_000 });
    await expect(page.locator(".reveal-card").getByText("Archetype")).toHaveCount(4, { timeout: 30_000 });
    await expect(page.getByTestId("ep-headline")).toBeVisible({ timeout: 60_000 });
    await expect(page.getByRole("button", { name: "Alignment" })).toBeVisible({ timeout: 30_000 });
    await expectNoHorizontalScroll(page, "results");
    // the wide epistemic table scrolls inside its own container
    const epScroll = page.getByRole("region", { name: /Epistemic timeline table/ });
    const sw = await epScroll.evaluate((el) => ({ sw: el.scrollWidth, cw: el.clientWidth }));
    expect(sw.cw).toBeLessThanOrEqual(390);
    await shot(page, "mobile-07-results");
    await epScroll.scrollIntoViewIfNeeded();
    await shot(page, "mobile-08-results-epistemic", { fullPage: false });
    expectClean(diag);
  } finally {
    await deleteGame(page, gameId);
  }
});
