/**
 * (b) Game-rule rejections from the REAL engine surface as friendly toasts and change nothing.
 * The UI already blocks overspending client-side, so the over-budget order is forged on the wire (as a
 * stale/tampered client would send it) to make the engine itself reject it.
 */
import { expect, test } from "@playwright/test";
import { currentGameId, deleteGame, expectClean, setManifesto, shot, startGame, watchDiagnostics, hire } from "./helpers";

test("overspend and stale-phase moves are rejected with friendly toasts, state unchanged", async ({ page, context }) => {
  const diag = watchDiagnostics(page);
  let gameId: string | null = null;
  try {
    await startGame(page, 31337);
    gameId = await currentGameId(page);
    await page.getByRole("tab", { name: /Manifesto/ }).click();
    await setManifesto(page, [20, 20, 20, 20, 20]);
    await page.getByRole("button", { name: "Commit manifesto" }).click();
    await page.getByRole("button", { name: /Commit & start round 1/ }).click();
    await expect(page.getByRole("heading", { name: "Round 1 of 5" })).toBeVisible();
    const treasury = page.locator(".treasury-chip");
    await expect(treasury).toContainText("Đ500.00");

    // client-side guard: a hire stepper cannot exceed what the treasury affords (fee read from the engine, P8)
    const obs0 = await (await page.request.get(`/api/games/${gameId}/players/p0/observation`)).json();
    const fee = Number(obs0.costs.worker);
    const affordable = Math.floor(500 / fee);
    const north = page.getByRole("spinbutton", { name: "New workers in North" });
    await north.fill("99999");
    await expect(north).toHaveValue(String(affordable));
    await north.fill("0");

    // server-side rejection: forge the body of an otherwise valid order (one worker more than affordable,
    // spread over the regions so no region runs out of eligible voters)
    await hire(page, "North", 2);
    const tooMany = affordable + 1;
    const spread = [0, 1, 2, 3].map((i) => Math.floor(tooMany / 4) + (i < tooMany % 4 ? 1 : 0));
    await page.route("**/players/*/allocate", async (route) => {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      await route.continue({ postData: JSON.stringify({ ...body, hires_by_region: spread }) });
    });
    await page.getByRole("button", { name: "Launch round 1" }).click();
    const toast = page.locator(".toast-error");
    await expect(toast).toBeVisible();
    await expect(toast).toContainText("Not enough money in the treasury for that.");
    await expect(toast).toContainText("INSUFFICIENT_FUNDS");
    await expect(toast).not.toContainText(/Traceback|undefined|\[object/);
    await shot(page, "desktop-13-rejection-toast", { fullPage: false, keepToasts: true });
    await expect(page.getByRole("heading", { name: "Round 1 of 5" })).toBeVisible();
    await expect(treasury).toContainText("Đ500.00");
    await page.unroute("**/players/*/allocate");
    await toast.getByRole("button", { name: "Dismiss notification" }).click();
    await expect(toast).toHaveCount(0);

    // invalid allocation (more workers than the region has eligible voters, R22), forged the same way
    await page.route("**/players/*/allocate", (route) =>
      route.continue({ postData: JSON.stringify({ hires_by_region: [1_000_000, 0, 0, 0], media_regions: [false, false, false, false] }) }),
    );
    await page.getByRole("button", { name: "Launch round 1" }).click();
    await expect(page.locator(".toast-error")).toContainText("That allocation breaks the rules");
    await page.unroute("**/players/*/allocate");

    // a stale second tab still showing the survey tries to commit again → INVALID_PHASE
    const stale = await context.newPage();
    const staleDiag = watchDiagnostics(stale);
    await stale.goto("/");
    await stale.evaluate((g) => sessionStorage.setItem("dumbocracy.v2.session", JSON.stringify({ g, p: "p0" })), gameId);
    await stale.route("**/players/*/observation", async (route) => {
      const res = await route.fetch();
      const obs = await res.json();
      await route.fulfill({ response: res, json: { ...obs, phase: "SURVEY", round: 0, legal_actions: ["query", "commit"], polls: [], manifesto: null } });
    });
    await stale.reload();
    await expect(stale.getByRole("heading", { name: "Find out what the regions want" })).toBeVisible();
    await stale.unroute("**/players/*/observation");
    await stale.getByRole("tab", { name: /Manifesto/ }).click();
    await stale.getByRole("button", { name: "Commit manifesto" }).click();
    await stale.getByRole("button", { name: /Commit & start round 1/ }).click();
    await expect(stale.locator(".toast-error")).toContainText("That move isn't available in this phase.");
    // the rejection carries the fresh observation, so the stale tab catches up to the real round
    await expect(stale.getByRole("heading", { name: "Round 1 of 5" })).toBeVisible();
    await stale.close();
    expectClean(staleDiag);

    // the real move still works afterwards
    await page.getByRole("button", { name: "Launch round 1" }).click();
    await expect(page.getByRole("heading", { name: "Round 2 of 5" })).toBeVisible();
    expectClean(diag);
  } finally {
    await deleteGame(page, gameId);
  }
});
