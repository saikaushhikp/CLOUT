/** (e) Dark mode toggle: switches the palette, persists across reloads, and every screen stays legible. */
import { expect, test, type Page } from "@playwright/test";
import { currentGameId, deleteGame, expectClean, setManifesto, shot, startGame, watchDiagnostics } from "./helpers";

const lum = (page: Page, sel: string, prop: "backgroundColor" | "color") =>
  page.locator(sel).first().evaluate((el, prop) => {
    const m = getComputedStyle(el)[prop as "color"].match(/[\d.]+/g)!.map(Number);
    const f = (c: number) => ((c /= 255) <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
    return 0.2126 * f(m[0]) + 0.7152 * f(m[1]) + 0.0722 * f(m[2]);
  }, prop);
const contrast = (a: number, b: number) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);

test("dark mode toggle", async ({ page }) => {
  const diag = watchDiagnostics(page);
  let gameId: string | null = null;
  try {
    await page.goto("/");
    await expect(page.locator("html")).not.toHaveAttribute("data-theme", "dark");
    const lightBg = await lum(page, "body", "backgroundColor");
    await page.getByRole("button", { name: "Switch to dark theme" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    const darkBg = await lum(page, "body", "backgroundColor");
    expect(darkBg).toBeLessThan(0.05);
    expect(lightBg).toBeGreaterThan(0.6);
    expect(contrast(await lum(page, "h1", "color"), darkBg)).toBeGreaterThan(7);
    await shot(page, "desktop-15-setup-dark");

    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await expect(page.getByRole("button", { name: "Switch to light theme" })).toBeVisible();

    await startGame(page, 4040);
    gameId = await currentGameId(page);
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    const panelBg = await lum(page, ".panel", "backgroundColor");
    expect(contrast(await lum(page, ".panel h2", "color"), panelBg)).toBeGreaterThan(4.5);
    await shot(page, "desktop-16-survey-dark");
    await page.getByRole("tab", { name: /Manifesto/ }).click();
    await setManifesto(page, [20, 20, 20, 20, 20]);
    await page.getByRole("button", { name: "Commit manifesto" }).click();
    await page.getByRole("button", { name: /Commit & start round 1/ }).click();
    await expect(page.getByRole("heading", { name: "Round 1 of 5" })).toBeVisible();
    // R18: no network during the rounds; the committed manifesto panel is themed too
    await expect(page.getByTestId("manifesto-summary")).toBeVisible();
    await shot(page, "desktop-17-campaign-dark");

    await page.getByRole("button", { name: "Switch to light theme" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    expect(await lum(page, "body", "backgroundColor")).toBeGreaterThan(0.6);
    expectClean(diag);
  } finally {
    await page.evaluate(() => localStorage.removeItem("dumbocracy.theme")).catch(() => undefined);
    await deleteGame(page, gameId);
  }
});
