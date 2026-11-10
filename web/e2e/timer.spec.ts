/** (c) The optional round clock: on expiry the client forfeits the round (allocate nothing). */
import { expect, test } from "@playwright/test";
import { currentGameId, deleteGame, expectClean, setManifesto, shot, startGame, watchDiagnostics, hire } from "./helpers";

test("round clock expiry forfeits the round", async ({ page }) => {
  test.setTimeout(150_000);
  const diag = watchDiagnostics(page);
  let gameId: string | null = null;
  try {
    await startGame(page, 555);
    gameId = await currentGameId(page);
    await page.getByRole("tab", { name: /Manifesto/ }).click();
    await setManifesto(page, [20, 20, 20, 20, 20]);
    await page.getByRole("button", { name: "Commit manifesto" }).click();
    await page.getByRole("button", { name: /Commit & start round 1/ }).click();
    await expect(page.getByRole("heading", { name: "Round 1 of 5" })).toBeVisible();

    await page.getByRole("combobox", { name: "Seconds per round" }).selectOption("30");
    const tOn = Date.now();
    await page.locator("label.switch", { hasText: "Round clock" }).click(); // users click the styled track/label
    await expect(page.getByRole("switch", { name: "Round clock" })).toBeChecked();
    const timer = page.getByRole("timer");
    await expect(timer).toContainText(/s left/);
    await hire(page, "North", 5); // a draft that is NOT submitted: expiry forfeits, it does not send the draft
    await page.waitForTimeout(1500);
    await shot(page, "desktop-14-round-clock-running", { fullPage: false });

    const forfeit = page.waitForRequest((r) => r.url().endsWith("/forfeit") && r.method() === "POST", { timeout: 45_000 });
    await forfeit;
    const waited = Date.now() - tOn;
    expect(waited).toBeGreaterThan(28_000); // 30 s clock (paused only while a request is in flight)
    expect(waited).toBeLessThan(40_000);
    await expect(page.getByRole("heading", { name: "Round 2 of 5" })).toBeVisible();
    await expect(page.locator(".toast-info")).toContainText("Time's up");
    await expect(page.locator(".treasury-chip")).toContainText("Đ500.00");
    await expect(page.getByLabel("Round 1 report")).toContainText("Đ0.00");
    // the clock restarts for the new round (and the North draft was cleared)
    await expect(timer).toContainText(/(29|30)s left/);
    await expect(page.getByRole("spinbutton", { name: "New workers in North" })).toHaveValue("0");
    // round-1 allocation recorded as nothing
    const obs = await (await page.request.get(`/api/games/${gameId}/players/p0/observation`)).json();
    expect(obs.allocations[0]).toMatchObject({ round: 1, hires_by_region: [0, 0, 0, 0], cost: "0.00" });
    expect(obs.workers).toMatchObject({ total: 0 }); // the unsent draft bought nothing
    await page.locator("label.switch", { hasText: "Round clock" }).click();
    expectClean(diag);
  } finally {
    await deleteGame(page, gameId);
  }
});
