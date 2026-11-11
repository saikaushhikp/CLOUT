"use client";

import { Eraser, HelpCircle, Loader2, Megaphone, Radio, Send } from "lucide-react";
import type { Observation, RegionId } from "@/lib/types";
import { REGION_IDS } from "@/lib/types";
import { REGION_META } from "@/lib/constants";
import { checkAllocation, setRegionHires, teamByRegion, type AllocationDraft, emptyDraft } from "@/lib/allocation";
import { formatMoney } from "@/lib/money";
import { fmtInt } from "@/lib/format";
import { RegionPip, Stepper } from "./ui";

/** "How it works": how a bought party worker is chosen and how its influence spreads (P8, public rules only). */
export function WorkerHelp({ obs }: { obs: Observation }) {
  const minDegree = obs.costs.min_degree;
  const cap = obs.costs.attempt_cap;
  return (
    <details className="help-pop">
      <summary>
        <HelpCircle size={14} aria-hidden /> How it works
      </summary>
      <div className="help-pop-body" role="note" aria-label="How party workers work">
        <p>
          <b>If I buy a worker now, who is it and what does it do?</b>
        </p>
        <ul>
          <li>
            There are no free workers. You pay a <b>one-time fee</b> of {formatMoney(obs.costs.worker)} per worker. The worker is{" "}
            <b>permanent</b> and <b>stays in the region</b> you bought it in for the rest of the game.
          </li>
          <li>
            The worker is a random voter of that region with at least {minDegree} friends, pinned to your manifesto: friends and
            ads never change its mind.
          </li>
          <li>
            In the round you buy it, the worker tries to win over each of its friends once. Influence moves{" "}
            <b>one hop per round</b>: a friend it converts takes your manifesto and tries <i>their</i> friends next round, then
            becomes an ordinary voter again (who can drift away like anyone else).
          </li>
          <li>
            Each voter can be tried at most {cap} times in the whole game, failed tries included.
          </li>
          <li>
            The chance of converting someone = how receptive they are to canvassing × how much they already like your manifesto.
          </li>
          <li>
            <b>Buy early</b>: the cascade gets more rounds to spread. Voters converted in the final round have no round left to
            spread in.
          </li>
        </ul>
      </div>
    </details>
  );
}

/** Permanent team total, with this round's planned purchases (P8 `workers.by_region`). */
export function TeamStrip({ obs, hires }: { obs: Observation; hires?: number[] }) {
  const total = teamByRegion(obs).reduce((a, b) => a + b, 0);
  const newTotal = (hires ?? []).reduce((a, b) => a + b, 0);
  return (
    <span className="stat-sub" data-testid="team-line">
      Permanent team: {total} worker{total === 1 ? "" : "s"}
      {newTotal ? ` + ${newTotal} new = ${total + newTotal}` : ""}
    </span>
  );
}

export function AllocationPanel({
  obs,
  draft,
  onChange,
  onSubmit,
  busy,
}: {
  obs: Observation;
  draft: AllocationDraft;
  onChange: (d: AllocationDraft) => void;
  onSubmit: () => void;
  busy: boolean;
}) {
  const check = checkAllocation(obs, draft);
  const canAct = obs.legal_actions.includes("allocate");
  const isLast = obs.round >= obs.rounds;
  const fee = formatMoney(obs.costs.worker);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (check.ok && canAct && !busy) onSubmit();
      }}
      aria-label={`Round ${obs.round} allocation`}
    >
      <div className="row-between">
        <span className="field-label" id="hire-label">
          Buy party workers
        </span>
        <span className="stat-sub" id="hire-help">
          {fee} each, permanent
        </span>
      </div>
      <div className="row-between" style={{ alignItems: "flex-start" }}>
        <TeamStrip obs={obs} hires={draft.hires} />
        <WorkerHelp obs={obs} />
      </div>
      <div className="row" style={{ margin: "6px 0 4px" }}>
        <button type="button" className="btn btn-sm btn-ghost" onClick={() => onChange(emptyDraft())}>
          <Eraser size={14} aria-hidden /> Clear
        </button>
      </div>

      {REGION_IDS.map((r, i) => {
        const label = REGION_META[r as RegionId].label;
        const airings = obs.media_airings[r];
        const strength = 1 / (1 + 0.5 * airings);
        const add = draft.hires[i] ?? 0;
        return (
          <div className="alloc-row" key={r}>
            <div className="alloc-region">
              <RegionPip id={r} />
              <div style={{ minWidth: 0 }}>
                <b>{label}</b>
                <small className="num" data-testid={`hire-line-${r}`}>
                  team {check.teamBefore[i]}
                  {add ? ` · +${add} new` : ""}
                </small>
              </div>
            </div>
            <Stepper
              label={`New workers in ${label}`}
              value={add}
              max={add + check.maxAffordableMore}
              onChange={(v) => onChange(setRegionHires(draft, i, v))}
            />
            <button
              type="button"
              className="media-toggle"
              aria-pressed={draft.media[i]}
              aria-label={`Air media in ${label} for ${formatMoney(obs.costs.media_by_region[r])}${airings ? `, airing number ${airings + 1}, strength ${Math.round(strength * 100)} percent` : ""}`}
              title={airings ? `Repeat airing: ${Math.round(strength * 100)}% of first-airing impact` : "First airing: full impact"}
              onClick={() => onChange({ ...draft, media: draft.media.map((m, k) => (k === i ? !m : m)) })}
            >
              {draft.media[i] ? <Radio size={14} aria-hidden /> : <Megaphone size={14} aria-hidden />}
              <span className="num">{formatMoney(obs.costs.media_by_region[r])}</span>
              {airings ? <span className="num" style={{ opacity: 0.8 }}>· {Math.round(strength * 100)}% impact</span> : null}
            </button>
          </div>
        );
      })}

      <table className="cost-table" data-testid="cost-table">
        <tbody>
          <tr>
            <td>
              Workers {check.totalHires} × {fee} (one-time)
            </td>
            <td>{formatMoney(check.workerCostCents)}</td>
          </tr>
          <tr>
            <td>Media ({draft.media.filter(Boolean).length} {draft.media.filter(Boolean).length === 1 ? "region" : "regions"})</td>
            <td>{formatMoney(check.mediaCostCents)}</td>
          </tr>
          <tr className="total">
            <td>This round</td>
            <td>{formatMoney(check.costCents)}</td>
          </tr>
          <tr>
            <td className="muted">Left after this round</td>
            <td className={check.remainingCents < 0 ? "bad" : ""}>{formatMoney(check.remainingCents)}</td>
          </tr>
        </tbody>
      </table>
      <div className="meter" aria-hidden style={{ marginTop: 6 }}>
        <span
          style={{
            width: `${Math.min(100, (check.costCents / Math.max(1, check.treasuryCents)) * 100)}%`,
            background: check.remainingCents < 0 ? "var(--bad)" : "var(--accent)",
          }}
        />
      </div>

      {check.errors.length ? (
        <div className="errors" role="alert">
          {check.errors.map((e) => (
            <div key={e.message}>{e.message}</div>
          ))}
        </div>
      ) : check.totalHires === 0 && check.teamBefore.some((x) => x > 0) ? (
        <p className="stat-sub" style={{ marginTop: 8 }}>
          No new workers this round. Last round&apos;s converts still spread.
        </p>
      ) : null}

      <button type="submit" className="btn btn-primary btn-lg" style={{ width: "100%", marginTop: 12 }} disabled={!check.ok || !canAct || busy}>
        {busy ? <Loader2 className="spin" size={18} aria-hidden /> : <Send size={18} aria-hidden />}
        {isLast ? "Final push — go to the polls" : `Launch round ${obs.round}`}
      </button>
    </form>
  );
}
