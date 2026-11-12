// Slide order. One entry per slide file; `section` drives the pill track in each slide's top bar.
// Add, remove or reorder slides here only: page numbers, the deck viewer and render.mjs all read this list.
// Layout templates (tpl-*.html) are not in the deck; copy one to start a new slide.
globalThis.DECK = {
  title: "Clout",
  subtitle: "An Election Game",
  footer: "Multi-Agent Systems · IIT Hyderabad",
  sections: ["Intro", "Game", "Model", "Logic", "Results", "Wrap-up"],
  slides: [
    { file: "01-title.html", section: null },
    { file: "02-electorate-maths.html", section: "Model" },
    { file: "03-workers-media.html", section: "Model" },
    { file: "04-logic.html", section: "Logic" },
    { file: "05-exp-workers.html", section: "Results" },
    { file: "06-exp-manifesto.html", section: "Results" },
    { file: "07-exp-media-workers.html", section: "Results" },
    { file: "08-strategy.html", section: "Results" },
    { file: "09-contributions.html", section: "Wrap-up" },
    { file: "10-thanks.html", section: null },
  ],
};
