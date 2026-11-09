"use client";

import type { ReactNode } from "react";
import { AlertTriangle, BrainCircuit, Check, Loader2, Minus } from "lucide-react";
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
    label: "Possible",
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
    <details className="ep-explainer">
      <summary>How to read this</summary>
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
    </details>
  );
}

function KnowledgeCell({ r }: { r: EpistemicRound }) {
  return (
    <span className="num">
      {r.knowledge.queries} q · {r.knowledge.polls} poll{r.knowledge.polls === 1 ? "" : "s"}
    </span>
  );
}

const DETAIL_ROWS: { label: string; hint?: string; cell: (r: EpistemicRound) => ReactNode }[] = [
  { label: "Best plan", cell: (r) => <span className="ep-plan">{r.best_plan.label}</span> },
  {
    label: "Wins in",
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
  { label: "Consistent worlds", cell: (r) => <span className="num">{r.knowledge.consistent_worlds}</span> },
  {
    label: "Uncertainty",
    hint: "posterior entropy, bits — lower = you knew more",
    // The engine reports entropy in nats (natural log, strategies/bayes.py); show bits = nats / ln 2.
    cell: (r) => <span className="num">{(r.knowledge.posterior_entropy / Math.LN2).toFixed(2)} bits</span>,
  },
  { label: "Treasury", cell: (r) => <span className="num">{formatMoney(r.treasury)}</span> },
];

/** Presentational timeline: truth row, epistemic verdict rows and per-round knowledge stats. */
export function EpistemicTimelineView({ timeline: t }: { timeline: EpistemicTimeline }) {
  return (
    <div className="stack" style={{ gap: 14 }}>
      <p className="ep-headline" data-testid="ep-headline">
        {t.headline}
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
                  <span className="ep-formula">{f.formula}</span>
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
            {DETAIL_ROWS.map((d) => (
              <tr key={d.label}>
                <th scope="row" className="ep-rowhead">
                  <span className="ep-label">{d.label}</span>
                  {d.hint ? <span className="ep-formula">{d.hint}</span> : null}
                </th>
                {t.rounds.map((r) => (
                  <td key={r.round}>{d.cell(r)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="report-strip" style={{ gridTemplateColumns: "repeat(3, minmax(0,1fr))" }}>
        <div className="stat">
          <span className="stat-label">Outcome</span>
          <span className={`stat-value ${t.actual.victory ? "good" : "bad"}`}>{t.actual.victory ? "Won" : "Lost"}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Vote share</span>
          <span className="stat-value num">{fmtShare(t.actual.vote_share)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Expected share</span>
          <span className="stat-value num">{fmtShare(t.actual.expected_share)}</span>
        </div>
      </div>
      <Explainer t={t} />
    </div>
  );
}

/** The debrief panel: loading skeleton, error, or the timeline. */
export function EpistemicPanel({ timeline, error, onRetry }: { timeline: EpistemicTimeline | null; error: string | null; onRetry?: () => void }) {
  return (
    <Panel title="Epistemic timeline" icon={BrainCircuit} hint="truth vs what you could know, round by round">
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
          <span className="stat-sub">Rebuilding every electorate consistent with what you knew — this can take about ten seconds.</span>
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
