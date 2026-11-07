# Week 5 React UI validation and reviewer walkthrough

Automated checks were rerun on 2026-10-27 in `mas`. The HTTP/browser walkthrough is documented below for a reviewer; it was not completed in this executor because TCP binds are denied. No screenshot or manual click-through is claimed.

## Production frontend

The production React/Vite source, package manifest, lockfile, chart and tests are in `web/`. FastAPI serves the generated `web/dist/` directory. The production UI provides responsive episode setup, current metrics, support trajectory, searchable public target IDs, legal campaign actions and costs, action history, public network summary, event feed, official result and trace-only diagnostics. No runtime font/CDN is loaded.

The `mas` environment was verified using approved host access: Node.js 26.12.1, npm 11.20.0, React 19.3.1 and Vitest 4.1.12. The lockfile install added 155 packages and the final npm audit reported zero vulnerabilities. Dependencies are ignored in `web/node_modules/`; an earlier local Node runtime under the ignored `web/.runtime/` is only a fallback, not the primary setup.

## Commands and results

From the repository root:

```bash
conda run -n mas npm ci --prefix web
conda run -n mas npm --prefix web ls --depth=0
conda run -n mas npm --prefix web audit
conda run -n mas npm --prefix web test
conda run -n mas npm --prefix web run build
```

- Clean lockfile install: passed, 155 packages added.
- React/jsdom tests: passed, 3/3. Covered service-ready and game creation,public target selection/submission, accepted action and budget update,rejection with unchanged round/budget, fixed-horizon result and diagnostics,deterministic reset, and target filtering.
- Production build: passed with Vite 8.3.5. Initial app JS is 245.01 kB (77.11 kB gzip); the lazily loaded chart is 353.74 kB (102.77 kB gzip); CSS is 13.82 kB (3.84 kB gzip). No oversized chunk warning remains.
- Dependency tree/audit: passed with React/ReactDOM 19.3.1, Recharts 3.10.1, Vite 8.3.5, Vitest 4.1.12, Node 26.12.1 and npm 11.20.0; zero vulnerabilities.
- Vite preview smoke: production HTML and both JS assets returned HTTP 200.

The API integration test checks the built entrypoint and hashed assets, then tests create, rejected/accepted actions, terminal result, analysis, privacy, and reset. It passed in the final full and focused test runs. The first run found an outdated expected 560px responsive breakpoint; the stylesheet now uses 640px and the assertion checks that actual mobile breakpoint. The CSS also no longer imports Google Fonts, keeping the app offline-capable.

## Reviewer gameplay steps

1. Build assets, then start the local API with the command in
   `../docs/spec/web_api_v1.md` and open `http://127.0.0.1:61007`.
2. Choose CPU, 100 nodes, three rounds, a fixed seed, and a `20.00` budget;
   start the game and check the round-zero public aggregate and budget.
3. Search the eligible public IDs and perform targeted contact. Perform
   broadcast if the shown remaining budget allows; otherwise check that the
   rejection leaves the round and budget unchanged. Use no-op through the
   fixed horizon.
4. Review the official result and `trace_only` diagnostics, reset, and confirm
   the initial aggregate returns. Inspect Network responses for private fields.
5. Repeat at phone width and record browser/OS, checkout SHA, and findings.
