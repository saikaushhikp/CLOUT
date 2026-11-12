# Slides

Each slide is a standalone HTML page on a fixed 1920×1080 canvas, styled with the web app's theme
(`web/app/globals.css`: "morning paper" light palette, campaign-red accent, graph-paper grid, Google Sans).

| File | Role |
|---|---|
| `manifest.js` | Slide order and sections. Add, remove or reorder slides here only. |
| `theme.css` | Design tokens (light and dark), type scale, components (panel, stat, table, badge, takeaway…). |
| `deck.js` | Scales a slide to the window, sets the theme, fills in page numbers and the section pills. |
| `index.html` | Deck viewer. Keys: ←/→ to step, G for the grid overview, D for dark/light, F for fullscreen. |
| `render.mjs` | Exports PNGs (3840×2160 by default) and one vector PDF with headless Chromium. |
| `NN-*.html` | The real slides, in deck order. |
| `tpl-*.html` | Layout templates with `[placeholder]` text (section, bullets, stats, figure, table, columns, closing). Remove them from `manifest.js` before presenting. |

## View
Open `index.html` in a browser (works from `file://`). Add `?theme=dark` for the dark palette.

## Render
```bash
export PATH=/data/ai23btech11017/miniconda3/envs/mas/bin:$PATH   # node
node render.mjs                  # out/light/png/*.png + out/light/clout-slides-light.pdf
node render.mjs --scale 3        # 5760×3240 PNGs
node render.mjs --theme dark
node render.mjs --only 04 --no-pdf
```
This uses the Playwright already installed in `web/node_modules`. Fonts load from Google Fonts, so rendering needs network access.

## Add a slide
Copy the closest template, edit it, then list it in `manifest.js`. Inside `<section class="slide">`:
`header.topbar` (with an empty `nav.phase-track` to get the section pills), then `div.body`, then `footer.footer` (with `span.pageno`).
