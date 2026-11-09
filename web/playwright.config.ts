/**
 * Real-browser end-to-end walkthrough of the web game against a running engine (P6 exit criterion).
 *
 *   E2E_BASE_URL=http://127.0.0.1:61007 npm run e2e      # default base URL shown
 *
 * The specs drive the REAL engine (N = 10,000, fixed seeds) through the Next.js proxy, so both
 * `next start` (or `next dev`) and the FastAPI engine must be running (scripts/run_game.sh).
 * Screenshots of every screen are written to ../reports/v2/screenshots/.
 *
 * Ubuntu 20.04 hosts: Playwright no longer ships Chromium for focal; `npm run e2e` sets
 * PLAYWRIGHT_HOST_PLATFORM_OVERRIDE=ubuntu22.04-x64 there (the jammy headless shell runs fine on glibc 2.31).
 */
import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.E2E_BASE_URL ?? "http://127.0.0.1:61007";

export default defineConfig({
  testDir: "./e2e",
  testMatch: /.*\.spec\.ts$/,
  // One engine, a handful of concurrent games is fine (max_games = 8) — but the host is shared and loaded.
  workers: Number(process.env.E2E_WORKERS ?? 2),
  fullyParallel: false,
  retries: 0,
  timeout: 240_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  outputDir: "test-results",
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    actionTimeout: 20_000,
    navigationTimeout: 30_000,
    colorScheme: "light",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1440, height: 900 },
        launchOptions: {
          // Software WebGL so the sigma.js network view renders in headless mode.
          args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
        },
      },
    },
  ],
});
