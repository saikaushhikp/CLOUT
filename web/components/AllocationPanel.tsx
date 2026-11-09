"use client";

import { Eraser, HelpCircle, Loader2, Megaphone, Radio, Send, Users } from "lucide-react";
import type { Observation, RegionId } from "@/lib/types";
import { REGION_IDS } from "@/lib/types";
import { REGION_META, WOM_HOPS } from "@/lib/constants";
import { checkAllocation, setExtraWorkers, setRegionWorkers, type AllocationDraft, emptyDraft } from "@/lib/allocation";
import { formatMoney } from "@/lib/money";
import { fmtCompact, fmtInt } from "@/lib/format";
import { RegionPip, Stepper } from "./ui";

/** "How it works": how a (new) worker is chosen and used — public rules only (owner request). */
export function WorkerHelp({ obs }: { obs: Observation }) {
  const perWorker = obs.costs.contacts_per_worker;
  return (
    <details className="help-pop">
      <summary>
        <HelpCircle size={14} aria-hidden /> How it works
      </summary>
      <div className="help-pop-body" role="note" aria-label="How workers are hired and used">
        <p>
          <b>If I hire a worker now, how is he/she chosen and used?</b>
        </p>
        <ul>
          <li>
            A hire joins your <b>permanent team</b>: you pay the {formatMoney(obs.costs.extra_worker)} fee once and they work
            every remaining round, alongside your {obs.costs.free_workers} free workers.
          </li>
          <li>Each round you assign every worker to a region. Workers you leave unassigned sit idle that round.</li>
          <li>
            Each worker canvasses about 0.25% of the {fmtCompact(obs.num_voters)} voters ({fmtInt(perWorker)} doorsteps), picked
            uniformly at random from that region (with replacement, so the same door can be knocked twice).
          </li>
          <li>Voters with more campaign susceptibility move further toward your manifesto.</li>
          <li>
            Contacted voters can spread your message by word of mouth over friendship ties, for up to {WOM_HOPS} hops.
          </li>
        </ul>
      </div>
    </details>
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
  const free = obs.costs.free_workers;
  /** team before this round's hires: free + every earlier permanent hire (R16) */
  const base = check.workersBase;
  const hiredBefore = Math.max(0, base - free);
  const pool = base + draft.extra;
  const canAct = obs.legal_actions.includes("allocate");
  const isLast = obs.round >= obs.rounds;

  const spreadEvenly = () => {
    const per = Math.floor(pool / 4);
    const rest = pool - per * 4;
    // larger regions get the remainder
    const order = obs.regions.map((r, i) => ({ i, v: r.voters })).sort((a, b) => b.v - a.v);
    const workers = [per, per, per, per];
    for (let k = 0; k < rest; k++) workers[order[k].i] += 1;
    onChange({ ...draft, workers });
  };
  const byVoters = () => {
    const total = obs.regions.reduce((a, r) => a + r.voters, 0);
    const raw = obs.regions.map((r) => (pool * r.voters) / total);
    const workers = raw.map(Math.floor);
    let rest = pool - workers.reduce((a, b) => a + b, 0);
    raw.map((x, i) => ({ i, f: x - Math.floor(x) }))
      .sort((a, b) => b.f - a.f)
      .forEach(({ i }) => {
        if (rest > 0) {
          workers[i] += 1;
          rest -= 1;
        }
      });
    onChange({ ...draft, workers });
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (check.ok && canAct && !busy) onSubmit();
      }}
      aria-label={`Round ${obs.round} allocation`}
    >
      <div className="row-between">
        <span className="field-label">
          <Users size={14} aria-hidden style={{ verticalAlign: "-2px", marginRight: 6 }} />
          Party workers · {check.workersAssigned}/{pool} deployed
        </span>
        <span className="stat-sub">{fmtInt(obs.costs.contacts_per_worker)} doorsteps each</span>
      </div>
      <div className="row-between" style={{ alignItems: "flex-start" }}>
        <span className="stat-sub" data-testid="team-line">
          Team: {free} free{hiredBefore ? ` + ${hiredBefore} hired` : ""}
          {draft.extra ? ` + ${draft.extra} new hire${draft.extra > 1 ? "s" : ""}` : ""} = {pool} available
        </span>
        <WorkerHelp obs={obs} />
      </div>
      <div className="workers-pool" aria-hidden>
        {Array.from({ length: pool }, (_, k) => (
          <span key={k} className={`worker-dot ${k >= base ? "paid" : k >= free ? "hired" : ""} ${k < check.workersAssigned ? "used" : ""}`} />
        ))}
      </div>
      <div className="row" style={{ margin: "6px 0 4px" }}>
        <button type="button" className="btn btn-sm" onClick={spreadEvenly}>
          Even split
        </button>
        <button type="button" className="btn btn-sm" onClick={byVoters}>
          By population
        </button>
        <button type="button" className="btn btn-sm btn-ghost" onClick={() => onChange(emptyDraft())}>
          <Eraser size={14} aria-hidden /> Clear
        </button>
      </div>

      {REGION_IDS.map((r, i) => {
        const region = obs.regions.find((x) => x.id === r);
        const airings = obs.media_airings[r];
        const strength = 1 / (1 + 0.5 * airings);
        const others = draft.workers.reduce((a, b, k) => (k === i ? a : a + b), 0);
        return (
          <div className="alloc-row" key={r}>
            <div className="alloc-region">
              <RegionPip id={r} />
              <div style={{ minWidth: 0 }}>
                <b>{REGION_META[r as RegionId].label}</b>
                <small className="num">{fmtInt(region?.voters ?? 0)} voters</small>
              </div>
            </div>
            <Stepper
              label={`Workers in ${REGION_META[r as RegionId].label}`}
              value={draft.workers[i]}
              max={Math.max(draft.workers[i], pool - others)}
              onChange={(v) => onChange(setRegionWorkers(draft, base, i, v))}
            />
            <button
              type="button"
              className="media-toggle"
              aria-pressed={draft.media[i]}
              aria-label={`Air media in ${REGION_META[r as RegionId].label} for ${formatMoney(obs.costs.media_by_region[r])}${airings ? `, airing number ${airings + 1}, strength ${Math.round(strength * 100)} percent` : ""}`}
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

      <div className="row-between" style={{ marginTop: 12 }}>
        <div>
          <span className="field-label" id="hire-label">Hire extra workers</span>
          <div className="stat-sub" id="hire-help">
            {formatMoney(obs.costs.extra_worker)} each · one-time fee, works every remaining round
          </div>
        </div>
        <Stepper
          label="New hires"
          value={draft.extra}
          max={Math.max(draft.extra, check.maxAffordableExtra)}
          onChange={(v) => onChange(setExtraWorkers(draft, base, v))}
        />
      </div>

      <table className="cost-table">
        <tbody>
          <tr>
            <td>New hires × {draft.extra} (one-time)</td>
            <td>{formatMoney(check.workerCostCents)}</td>
          </tr>
          <tr>
            <td>Media in {draft.media.filter(Boolean).length} region(s)</td>
            <td>{formatMoney(check.mediaCostCents)}</td>
          </tr>
          <tr className="total">
            <td>This round</td>
            <td>{formatMoney(check.costCents)}</td>
          </tr>
          <tr>
            <td className="muted">Treasury after</td>
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
      ) : check.workersIdle > 0 ? (
        <p className="stat-sub" style={{ marginTop: 8 }}>
          {check.workersIdle} worker{check.workersIdle > 1 ? "s" : ""} unassigned — they sit idle this round (they stay on the team).
        </p>
      ) : null}

      <button type="submit" className="btn btn-primary btn-lg" style={{ width: "100%", marginTop: 12 }} disabled={!check.ok || !canAct || busy}>
        {busy ? <Loader2 className="spin" size={18} aria-hidden /> : <Send size={18} aria-hidden />}
        {isLast ? "Final push — go to the polls" : `Launch round ${obs.round}`}
      </button>
    </form>
  );
}
