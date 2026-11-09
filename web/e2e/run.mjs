#!/usr/bin/env node
// Launches `playwright test`, working around Playwright's missing Chromium build for Ubuntu 20.04:
// on focal hosts the Ubuntu 22.04 headless shell is used (it runs on glibc 2.31).
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const env = { ...process.env };
try {
  const osr = readFileSync("/etc/os-release", "utf8");
  if (!env.PLAYWRIGHT_HOST_PLATFORM_OVERRIDE && /^ID=ubuntu$/m.test(osr) && /^VERSION_ID="20\.04"$/m.test(osr)) {
    env.PLAYWRIGHT_HOST_PLATFORM_OVERRIDE = "ubuntu22.04-x64";
  }
} catch {
  /* not Linux */
}
const cli = createRequire(import.meta.url).resolve("@playwright/test/cli");
const r = spawnSync(process.execPath, [cli, "test", ...process.argv.slice(2)], { stdio: "inherit", env });
process.exit(r.status ?? 1);
