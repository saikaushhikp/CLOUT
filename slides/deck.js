// Runs inside every slide file (after manifest.js).
//  - scales the 1920x1080 .slide to fit the window (letterboxed), or leaves it 1:1 with ?print
//  - theme: ?theme=dark | ?theme=light (default light, the app's "morning paper")
//  - fills .pageno, .phase-track, [data-deck=title|subtitle|footer] from manifest.js
(function () {
  const q = new URLSearchParams(location.search);
  const root = document.documentElement;
  const theme = q.get("theme");
  if (theme === "dark" || theme === "light") root.dataset.theme = theme;
  if (q.has("print")) root.classList.add("is-print");

  const deck = globalThis.DECK || { slides: [], sections: [] };
  const file = decodeURIComponent(location.pathname.split("/").pop());
  const idx = deck.slides.findIndex((s) => s.file === file);
  const entry = deck.slides[idx] || {};

  function fill() {
    document.querySelectorAll("[data-deck]").forEach((el) => {
      const v = deck[el.dataset.deck];
      if (v != null && !el.textContent.trim()) el.textContent = v;
    });
    document.querySelectorAll(".pageno").forEach((el) => {
      if (idx >= 0) el.textContent = String(idx + 1).padStart(2, "0") + " / " + String(deck.slides.length).padStart(2, "0");
    });
    document.querySelectorAll(".phase-track:empty").forEach((track) => {
      const cur = deck.sections.indexOf(entry.section);
      deck.sections.forEach((name, i) => {
        const s = document.createElement("span");
        s.className = "phase-step" + (i < cur ? " is-done" : i === cur ? " is-current" : "");
        s.textContent = name;
        track.appendChild(s);
      });
    });
  }

  function fit() {
    if (root.classList.contains("is-print")) return;
    const el = document.querySelector(".slide");
    if (!el) return;
    const k = Math.min(innerWidth / 1920, innerHeight / 1080);
    const x = (innerWidth - 1920 * k) / 2;
    const y = (innerHeight - 1080 * k) / 2;
    el.style.transform = `translate(${x}px, ${y}px) scale(${k})`;
  }

  // standalone file: arrow keys step through the deck
  function keys(e) {
    if (window.top !== window || idx < 0) return;
    const go = (d) => {
      const n = deck.slides[idx + d];
      if (n) location.href = n.file + location.search;
    };
    if (["ArrowRight", "PageDown", " "].includes(e.key)) go(1);
    if (["ArrowLeft", "PageUp"].includes(e.key)) go(-1);
  }

  document.addEventListener("DOMContentLoaded", () => {
    fill();
    fit();
  });
  addEventListener("resize", fit);
  addEventListener("keydown", keys);
})();
