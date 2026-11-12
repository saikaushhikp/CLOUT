// Render the deck to high-resolution PNGs and one vector PDF with headless Chromium.
//
//   node render.mjs                 # light theme, PNGs at 2x (3840x2160) + deck PDF
//   node render.mjs --scale 3       # 5760x3240 PNGs
//   node render.mjs --theme dark
//   node render.mjs --only 04       # just slides whose file name starts with 04 (comma-separate several: 01,02)
//   node render.mjs --only 01,02 --out out/title+motivation   # custom output folder
//   node render.mjs --no-pdf | --no-png
//   node slides/render.mjs slides/01-title.html slides/02-motivation.html --out "out/title+motivation"   # just these files, PNG only
//
// Output: slides/out/<theme>/png/NN-name.png and slides/out/<theme>/clout-slides-<theme>.pdf
// Uses the Playwright already installed for the web app (../web/node_modules), so there is nothing extra to install.
import { createRequire } from "node:module";
import { mkdirSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import vm from "node:vm";
import { readFileSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(join(here, "../web/package.json"));
const { chromium } = require("playwright");

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const k = args.indexOf(`--${name}`);
  return k >= 0 ? args[k + 1] : dflt;
};
const theme = opt("theme", "light");
const scale = Number(opt("scale", "2"));
const only = opt("only", "");
// Slide files given by path (e.g. slides/01-title.html): render just those, PNG only.
const files = args.filter((a) => a.endsWith(".html")).map((a) => basename(a));
const doPng = !args.includes("--no-png");
const doPdf = !args.includes("--no-pdf") && files.length === 0;

const ctx = {};
vm.runInNewContext(readFileSync(join(here, "manifest.js"), "utf8"), { globalThis: ctx });
const prefixes = only ? only.split(",") : [""];
let slides = ctx.DECK.slides.filter((s) => prefixes.some((p) => s.file.startsWith(p)));
if (files.length) {
  const unknown = files.filter((f) => !ctx.DECK.slides.some((s) => s.file === f));
  if (unknown.length) {
    console.error(`not in manifest.js (page numbers would be missing): ${unknown.join(", ")}`);
    process.exit(1);
  }
  slides = ctx.DECK.slides.filter((s) => files.includes(s.file));
}

const outDir = opt("out", "") ? join(here, opt("out", "")) : join(here, "out", theme);
mkdirSync(join(outDir, "png"), { recursive: true });

const url = (file, query) => pathToFileURL(join(here, file)).href + query;

const browser = await chromium.launch();
try {
  if (doPng) {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: scale });
    for (const s of slides) {
      await page.goto(url(s.file, `?print&theme=${theme}`), { waitUntil: "networkidle", timeout: 120000 });
      await page.evaluate(() => document.fonts.ready);
      const png = join(outDir, "png", s.file.replace(/\.html$/, ".png"));
      await page.screenshot({ path: png, clip: { x: 0, y: 0, width: 1920, height: 1080 } });
      console.log("png ", png);
    }
    await page.close();
  }
  if (doPdf) {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    await page.goto(url("index.html", `?print&theme=${theme}` + (only ? `&only=${encodeURIComponent(only)}` : "")), { waitUntil: "networkidle", timeout: 120000 });
    // wait for every slide frame and its fonts
    for (const f of page.frames().slice(1)) {
      await f.waitForLoadState("networkidle");
      await f.evaluate(() => document.fonts.ready);
    }
    const pdf = join(outDir, `clout-slides-${theme}${only ? "-" + only.replace(/,/g, "-") : ""}.pdf`);
    await page.pdf({ path: pdf, width: "1920px", height: "1080px", printBackground: true, preferCSSPageSize: true });
    console.log("pdf ", pdf);
    await page.close();
  }
} finally {
  await browser.close();
}
