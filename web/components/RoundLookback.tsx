"use client";

import { useEffect, useRef } from "react";
import { ChevronLeft, ChevronRight, History, Megaphone, X } from "lucide-react";
import type { Observation, RegionId } from "@/lib/types";
import { REGION_IDS } from "@/lib/types";
import { REGION_META, sectorLabel } from "@/lib/constants";
import { formatMoney } from "@/lib/money";
import { fmtInt } from "@/lib/format";
import { reviewableRounds, roundSummary } from "@/lib/rounds";
import { RegionPip, Stat } from "./ui";

const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
const moe = (x: number) => `±${(x * 100).toFixed(1)}`;

/**
 * Read-only review of a resolved round (R19): poll at its start (± MoE), the orders placed, the outcomes
 * from `round_reports`, the queries asked that round and the treasury before/after. A modal dialog on
 * desktop, a bottom sheet on phones (CSS). Escape / the close button / the backdrop return to planning.
 */
export function RoundLookback({
  obs,
  round,
  onClose,
  onNavigate,
}: {
  obs: Observation;
  round: number;
  onClose: () => void;
  onNavigate: (round: number) => void;
}) {
  const s = roundSummary(obs, round);
  const rounds = reviewableRounds(obs);
  const prev = rounds.includes(round - 1) ? round - 1 : null;
  const next = rounds.includes(round + 1) ? round + 1 : null;
  const closeRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Focus the close button on open; give focus back to whatever opened the dialog on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    return () => {
      if (opener && document.contains(opener)) opener.focus();
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "Tab" && dialogRef.current) {
        // keep keyboard focus inside the dialog
        const items = dialogRef.current.querySelectorAll<HTMLElement>("button:not(:disabled), [href], [tabindex]:not([tabindex='-1'])");
        if (!items.length) return;
        const first = items[0];
        const last = items[items.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const titleId = `lookback-title-${round}`;
  const orders = s.orders;
  const aired = orders ? REGION_IDS.filter((_, i) => orders.media_regions[i]) : [];

  return (
    <div className="modal-backdrop lookback-backdrop" role="presentation" onClick={onClose}>
      <div
        ref={dialogRef}
        className="panel modal lookback"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        data-testid="round-lookback"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="row-between lookback-head">
          <div className="row" style={{ gap: 8 }}>
            <History size={18} aria-hidden className="panel-icon" />
            <h2 id={titleId}>Round {round} in review</h2>
            <span className="badge">read-only</span>
          </div>
          <button ref={closeRef} type="button" className="btn btn-ghost btn-icon" aria-label={`Close round ${round} review and return to planning`} onClick={onClose}>
            <X size={18} aria-hidden />
          </button>
        </div>

        <section aria-label="Poll" className="lookback-sec">
          <h3>Poll</h3>
          <div className="report-strip lookback-strip">
            <Stat
              label={`Start of R${round}`}
              value={s.pollStart ? pct(s.pollStart.estimate) : "—"}
              sub={s.pollStart ? `${moe(s.pollStart.moe)} pts · n = ${fmtInt(s.pollStart.sample_size)}` : "no poll"}
            />
            <Stat
              label={round >= obs.rounds ? "Eve of election" : `Start of R${round + 1}`}
              value={s.pollAfter ? pct(s.pollAfter.estimate) : "—"}
              sub={
                s.pollAfter && s.pollStart
                  ? `${s.pollAfter.estimate >= s.pollStart.estimate ? "+" : ""}${((s.pollAfter.estimate - s.pollStart.estimate) * 100).toFixed(1)} pts`
                  : undefined
              }
            />
            <Stat label="Treasury before" value={formatMoney(s.treasuryBeforeCents)} />
            <Stat label="Treasury after" value={formatMoney(s.treasuryAfterCents)} sub={`−${formatMoney(s.queryCostCents + s.ordersCostCents)}`} />
          </div>
        </section>

        <section aria-label="Orders" className="lookback-sec">
          <h3>Orders</h3>
          {!orders ? (
            <p className="stat-sub">No orders on record for this round.</p>
          ) : !s.campaigned ? (
            <p className="stat-sub">No campaign spending this round (forfeited or nothing bought). Earlier converts still spread.</p>
          ) : null}
          {orders ? (
            <>
              <table className="cost-table lookback-table">
                <caption className="sr-only">Workers bought, team after the round and media per region in round {round}</caption>
                <thead>
                  <tr>
                    <th scope="col">Region</th>
                    <th scope="col">Bought</th>
                    <th scope="col">Team after</th>
                    <th scope="col">Media</th>
                  </tr>
                </thead>
                <tbody>
                  {REGION_IDS.map((r, i) => (
                    <tr key={r}>
                      <th scope="row">
                        <span className="row" style={{ gap: 6 }}>
                          <RegionPip id={r} size={20} />
                          {REGION_META[r as RegionId].label}
                        </span>
                      </th>
                      <td className="num">{s.hiresByRegion[i] ? `+${s.hiresByRegion[i]}` : "0"}</td>
                      <td className="num">{s.teamAfterByRegion[i] ?? 0}</td>
                      <td>{orders.media_regions[i] ? <span className="badge"><Megaphone size={11} aria-hidden /> aired</span> : <span className="muted">—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <ul className="list-plain stat-sub lookback-facts">
                <li>
                  Team: {s.teamBefore} before{s.newHires ? ` + ${s.newHires} bought` : " · none bought"} ={" "}
                  {s.teamAfterByRegion.reduce((a, b) => a + b, 0)} permanent workers
                </li>
                <li>
                  Media: {aired.length ? aired.map((r) => REGION_META[r as RegionId].label).join(", ") : "none"}
                </li>
                <li>
                  Orders cost <b className="num">{formatMoney(s.ordersCostCents)}</b>
                  {s.queryCostCents ? <> · questions <b className="num">{formatMoney(s.queryCostCents)}</b></> : null}
                </li>
              </ul>
            </>
          ) : null}
        </section>

        <section aria-label="Outcomes" className="lookback-sec">
          <h3>Outcomes</h3>
          {s.report ? (
            <div className="report-strip lookback-strip">
              <Stat label="Spreaders" value={fmtInt(s.report.spreaders)} sub="new workers + last round's converts" />
              <Stat
                label="Converted"
                value={fmtInt(s.report.converted)}
                sub={s.report.election_held ? "final round: no spreading" : "spread next round"}
              />
              <Stat label="Saw your ads" value={fmtInt(s.report.media_exposed)} />
              <Stat label="Spent" value={formatMoney(s.report.cost)} />
            </div>
          ) : (
            <p className="stat-sub">The engine has no outcome report for this round.</p>
          )}
        </section>

        <section aria-label="Questions asked" className="lookback-sec">
          <h3>Questions asked ({s.queries.length})</h3>
          {s.queries.length ? (
            <ul className="list-plain stat-sub lookback-facts">
              {s.queries.map((q, k) => (
                <li key={k}>
                  {REGION_META[q.region as RegionId]?.label ?? q.region}: {q.pct_b_over_a.toFixed(1)}% ± {q.moe.toFixed(1)} (n = {fmtInt(q.sample_size)}) prefer{" "}
                  {sectorLabel(q.sector_b)} over {sectorLabel(q.sector_a)}
                </li>
              ))}
            </ul>
          ) : (
            <p className="stat-sub">No paid questions this round.</p>
          )}
        </section>

        <nav className="row-between lookback-nav" aria-label="Rounds">
          <button type="button" className="btn btn-sm btn-icon" disabled={prev === null} onClick={() => prev !== null && onNavigate(prev)} aria-label="Previous round">
            <ChevronLeft size={14} aria-hidden />
          </button>
          <ol className="round-dots list-plain">
            {rounds.map((t) => (
              <li key={t}>
                <button
                  type="button"
                  className={`round-dot done ${t === round ? "reviewing" : ""}`}
                  aria-label={`Review round ${t}`}
                  aria-current={t === round ? "true" : undefined}
                  onClick={() => onNavigate(t)}
                >
                  R{t}
                </button>
              </li>
            ))}
            {obs.phase === "ROUND" ? (
              <li>
                <button type="button" className="round-dot now" aria-label={`Round ${obs.round} — back to planning`} onClick={onClose}>
                  R{obs.round}
                </button>
              </li>
            ) : null}
          </ol>
          <button type="button" className="btn btn-sm btn-icon" disabled={next === null} onClick={() => next !== null && onNavigate(next)} aria-label="Next round">
            <ChevronRight size={14} aria-hidden />
          </button>
        </nav>
      </div>
    </div>
  );
}
