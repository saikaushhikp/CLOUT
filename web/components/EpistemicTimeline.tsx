"use client";

import { useState, type ReactNode } from "react";
import { AlertTriangle, Check, Loader2, Minus } from "lucide-react";
import { fmtShare } from "@/lib/format";
import { formatMoney } from "@/lib/money";
import type { EpistemicRound, EpistemicTimeline } from "@/lib/types";
import { Panel } from "./ui";

type Flag = "truth_victory_possible" | "possible_victory" | "knows_win_exists" | "guaranteed_victory" | "inevitable_defeat";

/** The verdict rows, in reading order: the ground truth first, then what you knew (weakest → strongest). */
const FLAG_ROWS: { key: Flag; label: string; formula: string; gloss: string; tone: "truth" | "good" | "bad" }[] = [
  {
    key: "truth_victory_possible",
    label: "Truth: victory still possible",
    formula: "∃π · Victory in the real world",
    gloss: "Some plan would have won in the actual electorate — a fact, whether or not you knew it.",
    tone: "truth",
  },
  {
    key: "possible_victory",
    label: "Win looked possible",
    formula: "M◇Win  =  ∃π ∃w",
    gloss: "In at least one world you could not rule out, some plan wins.",
    tone: "good",
  },
  {
    key: "knows_win_exists",
    label: "Knew a win exists",
    formula: "∀w ∃π  (knowing that)",
    gloss: "In every consistent world some plan wins — but maybe a different plan in each.",
    tone: "good",
  },
  {
    key: "guaranteed_victory",
    label: "Knew a winning plan",
    formula: "∃π ∀w  (knowing how)",
    gloss: "One single plan wins in every consistent world: you knew how to win.",
    tone: "good",
  },
  {
    key: "inevitable_defeat",
    label: "Defeat inevitable",
    formula: "K□¬Win  =  ∀π ∀w ¬Win",
    gloss: "No plan wins in any consistent world.",
    tone: "bad",
  },
];

function FlagBadge({ on, tone, label, round }: { on: boolean; tone: "truth" | "good" | "bad"; label: string; round: number }) {
  return (
    <span
      className={`ep-badge ${on ? `ep-on ep-${tone}` : "ep-off"}`}
      data-testid={`ep-${round}-${label}`}
      data-on={on ? "true" : "false"}
      aria-label={`Round ${round}: ${label} — ${on ? "yes" : "no"}`}
    >
      {on ? tone === "bad" ? <AlertTriangle size={12} aria-hidden /> : <Check size={12} aria-hidden /> : <Minus size={12} aria-hidden />}
      {on ? "Yes" : "No"}
    </span>
  );
}

function Explainer({ t }: { t: EpistemicTimeline }) {
  return (
    <div className="ep-explainer">
      <h3>How to read this</h3>
      <div className="stack" style={{ gap: 8 }}>
        <p>
          At the start of every round (after that round&apos;s poll) we freeze <b>what you knew</b>: the free statistics, your survey
          answers, the polls and your own moves. We then build {t.scope.worlds_per_round} <b>possible worlds</b> — whole electorates
          (of {t.scope.world_voters.toLocaleString("en-US")} voters each) that are <i>consistent with everything you had seen</i>. These are
          the worlds of a Kripke model: you cannot tell them apart, so you <b>know</b> something (K) only if it holds in all of them, and
          you consider it <b>possible</b> (M) if it holds in at least one.
        </p>
        <p>
          In each world we try {t.scope.policies_per_round} campaign plans for the remaining rounds. <b>Knowing that</b> a win exists
          (∀w ∃π) is weaker than <b>knowing how</b> to win (∃π ∀w): the first only promises that every world has <i>some</i> winning
          plan; the second names one plan that wins whatever the truth is.
        </p>
        <p>
          The <b>Truth</b> row is different: it asks the real electorate, which you never saw. Truth ≠ knowledge — victory can be
          possible in reality while you could not know it, and vice versa.
        </p>
        <p className="stat-sub">{t.scope.note}</p>
      </div>
    </div>
  );
}

function KnowledgeCell({ r }: { r: EpistemicRound }) {
  return (
    <span className="num">
      {r.knowledge.queries} q · {r.knowledge.polls} poll{r.knowledge.polls === 1 ? "" : "s"}
    </span>
  );
}

/**
 * `formal` rows only appear with "Show formal details" on. Rows with a `same` key draw neighbouring rounds
 * that share a value as one cell spanning those columns.
 */
type DetailRow = { label: string; hint?: string; formal?: boolean; same?: (r: EpistemicRound) => string; cell: (r: EpistemicRound) => ReactNode };
const DETAIL_ROWS: DetailRow[] = [
  { label: "Best plan", same: (r) => r.best_plan.label, cell: (r) => <span className="ep-plan">{r.best_plan.label}</span> },
  {
    label: "Wins in",
    same: (r) => String(Math.round(r.best_plan.win_fraction * 100)),
    hint: "share of consistent worlds the best plan wins",
    cell: (r) => (
      <span className="ep-frac">
        <span className="ep-frac-bar" aria-hidden>
          <span style={{ width: `${Math.round(r.best_plan.win_fraction * 100)}%` }} />
        </span>
        <span className="num">{Math.round(r.best_plan.win_fraction * 100)}%</span>
      </span>
    ),
  },
  { label: "Mean share", hint: "of the best plan, across worlds", cell: (r) => <span className="num">{fmtShare(r.best_plan.mean_share)}</span> },
  { label: "You had asked", cell: (r) => <KnowledgeCell r={r} /> },
  { label: "Consistent worlds", formal: true, cell: (r) => <span className="num">{r.knowledge.consistent_worlds}</span> },
  {
    label: "Uncertainty",
    formal: true,
    hint: "posterior entropy, bits — lower = you knew more",
    // The engine reports entropy in nats (natural log, strategies/bayes.py); show bits = nats / ln 2.
    cell: (r) => <span className="num">{(r.knowledge.posterior_entropy / Math.LN2).toFixed(2)} bits</span>,
  },
  { label: "Treasury", same: (r) => String(r.treasury), cell: (r) => <span className="num">{formatMoney(r.treasury)}</span> },
];

/** Runs of consecutive rounds with the same `same` value (every round on its own when there is no key). */
export function spans(rounds: EpistemicRound[], same?: (r: EpistemicRound) => string): { r: EpistemicRound; n: number }[] {
  const out: { r: EpistemicRound; n: number; k: string }[] = [];
  rounds.forEach((r, i) => {
    const k = same ? same(r) : `#${i}`;
    const last = out[out.length - 1];
    if (last && last.k === k) last.n += 1;
    else out.push({ r, n: 1, k });
  });
  return out;
}

/** Presentational timeline: truth row, epistemic verdict rows and per-round knowledge stats. */
export function EpistemicTimelineView({ timeline: t }: { timeline: EpistemicTimeline }) {
  const [formal, setFormal] = useState(false);
  return (
    <div className="stack" style={{ gap: 14 }}>
      <p className={`ep-headline ${t.actual.victory ? "ep-headline-win" : ""}`} data-testid="ep-headline">
        {t.headline}
      </p>
      <p className="stat-sub" style={{ margin: 0 }}>
        Each column is the start of a round. The first row uses the real electorate; the rest use only what you had seen by then.
      </p>
      <div className="ep-scroll" tabIndex={0} role="region" aria-label="Epistemic timeline table, scrolls horizontally on small screens">
        <table className="ep-table">
          <caption className="sr-only">
            Per round: whether victory was still possible in reality, and what you could know about it across the possible worlds.
          </caption>
          <thead>
            <tr>
              <th scope="col" className="ep-rowhead">
                Start of round
              </th>
              {t.rounds.map((r) => (
                <th key={r.round} scope="col" className="num">
                  R{r.round}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {FLAG_ROWS.map((f) => (
              <tr key={f.key} className={f.tone === "truth" ? "ep-truth-row" : undefined} data-testid={`ep-row-${f.key}`}>
                <th scope="row" className="ep-rowhead" title={f.gloss}>
                  <span className="ep-label">{f.label}</span>
                  {formal ? <span className="ep-formula">{f.formula}</span> : null}
                </th>
                {t.rounds.map((r) => (
                  <td key={r.round}>
                    <FlagBadge on={r[f.key]} tone={f.tone} label={f.label} round={r.round} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
          <tbody className="ep-details">
            {DETAIL_ROWS.filter((d) => formal || !d.formal).map((d) => (
              <tr key={d.label}>
                <th scope="row" className="ep-rowhead">
                  <span className="ep-label">{d.label}</span>
                  {formal && d.hint ? <span className="ep-formula">{d.hint}</span> : null}
                </th>
                {spans(t.rounds, d.same).map(({ r, n }) => (
                  <td key={r.round} colSpan={n > 1 ? n : undefined} className={n > 1 ? "ep-span" : undefined}>
                    {d.cell(r)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div>
        <button type="button" className="btn btn-sm btn-ghost" aria-expanded={formal} onClick={() => setFormal((f) => !f)}>
          {formal ? "Hide formal details" : "Show formal details"}
        </button>
      </div>
      {formal ? <Explainer t={t} /> : null}
    </div>
  );
}

/** The debrief panel: loading skeleton, error, or the timeline. */
export function EpistemicPanel({ timeline, error, onRetry }: { timeline: EpistemicTimeline | null; error: string | null; onRetry?: () => void }) {
  return (
    <Panel title="Epistemic timeline">
      {timeline ? (
        <EpistemicTimelineView timeline={timeline} />
      ) : error ? (
        <div className="empty" role="alert">
          The epistemic timeline is unavailable: {error}{" "}
          {onRetry ? (
            <button type="button" className="btn btn-sm" onClick={onRetry}>
              Try again
            </button>
          ) : null}
        </div>
      ) : (
        <div className="ep-skeleton" role="status" aria-live="polite" data-testid="ep-loading">
          <span className="row" style={{ gap: 8 }}>
            <Loader2 className="spin" size={18} aria-hidden /> Reasoning over possible worlds…
          </span>
          <span className="stat-sub">This takes about ten seconds.</span>
          <div className="ep-skeleton-rows" aria-hidden>
            {Array.from({ length: 5 }, (_, i) => (
              <i key={i} />
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}
