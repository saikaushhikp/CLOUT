"use client";

import { useState } from "react";
import { Lock, TrendingDown, TrendingUp } from "lucide-react";
import { useGame, useObservation } from "@/lib/game";
import { emptyDraft, teamByRegion, type AllocationDraft } from "@/lib/allocation";
import { REGION_META, SECTOR_META } from "@/lib/constants";
import { formatMoney } from "@/lib/money";
import { fmtInt } from "@/lib/format";
import type { Observation, RegionId } from "@/lib/types";
import { REGION_IDS, SECTOR_IDS } from "@/lib/types";
import { toIntegerSimplex } from "@/lib/simplex";
import { reviewableRounds } from "@/lib/rounds";
import { AllocationPanel } from "./AllocationPanel";
import { RegionMap } from "./RegionMap";
import { PollChart } from "./PollChart";
import { RoundTimer } from "./RoundTimer";
import { QueryBuilder, RegionAnswers } from "./Intel";
import { Donut } from "./Donut";
import { RoundLookback } from "./RoundLookback";
import { Modal, Panel, SectorSwatch, StackBar, Stat } from "./ui";

function PollHud({ obs }: { obs: Observation }) {
  const latest = obs.polls[obs.polls.length - 1];
  const prev = obs.polls.length > 1 ? obs.polls[obs.polls.length - 2] : null;
  if (!latest) return null;
  const d = prev ? (latest.estimate - prev.estimate) * 100 : 0;
  const lead = latest.estimate - latest.moe > 0.5 ? "Clear lead" : latest.estimate + latest.moe < 0.5 ? "Trailing" : "Too close to call";
  return (
    <div>
      <div className="stat-label">Poll at the start of round {latest.round}</div>
      <div className="poll-hero num" aria-live="polite">
        {(latest.estimate * 100).toFixed(1)}%<small> ±{(latest.moe * 100).toFixed(1)}</small>
      </div>
      <div className="row" style={{ gap: 8, marginTop: 4 }}>
        <span className={`badge ${lead === "Clear lead" ? "" : "badge-live"}`}>{lead}</span>
        {prev ? (
          <span className={`num ${d >= 0 ? "good" : "bad"}`} style={{ fontSize: 13, fontWeight: 700 }}>
            {d >= 0 ? <TrendingUp size={14} aria-hidden /> : <TrendingDown size={14} aria-hidden />} {d >= 0 ? "+" : ""}
            {d.toFixed(1)} pts
          </span>
        ) : null}
      </div>
    </div>
  );
}

/**
 * Round tracker. Past rounds are buttons that open the read-only look-back (R19); the current round's chip
 * returns to planning; future rounds are inert.
 */
function RoundDots({ obs, reviewing, onReview }: { obs: Observation; reviewing: number | null; onReview: (t: number | null) => void }) {
  const past = new Set(reviewableRounds(obs));
  return (
    <ol className="round-dots list-plain" aria-label="Campaign rounds — tap a past round to review it">
      {Array.from({ length: obs.rounds }, (_, k) => k + 1).map((t) => {
        const cls = `round-dot ${t < obs.round ? "done" : t === obs.round ? "now" : ""} ${reviewing === t ? "reviewing" : ""}`;
        return (
          <li key={t}>
            {past.has(t) ? (
              <button
                type="button"
                className={cls}
                aria-label={`Review round ${t}`}
                aria-pressed={reviewing === t}
                title={`Review round ${t}`}
                onClick={() => onReview(reviewing === t ? null : t)}
              >
                R{t}
              </button>
            ) : t === obs.round ? (
              <button
                type="button"
                className={cls}
                aria-current="step"
                aria-label={`Round ${t} — planning now`}
                title="Back to planning this round"
                onClick={() => onReview(null)}
              >
                R{t}
              </button>
            ) : (
              <span className={cls}>R{t}</span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/** The committed manifesto, always visible during the rounds (owner request). */
export function ManifestoSummary({ manifesto }: { manifesto: number[] }) {
  const pcts = toIntegerSimplex(manifesto);
  const order = SECTOR_IDS.map((s, i) => ({ s, i })).sort((a, b) => pcts[b.i] - pcts[a.i]);
  return (
    <div className="manifesto-mini" data-testid="manifesto-summary">
      <div className="donut-wrap manifesto-mini-donut">
        <Donut values={pcts} size={128} thickness={22} />
        <div className="donut-center">
          <Lock size={16} aria-hidden className="muted" />
        </div>
      </div>
      <ul className="list-plain manifesto-mini-list" aria-label="Committed manifesto shares">
        {order.map(({ s, i }) => (
          <li key={s}>
            <SectorSwatch id={s} />
            <span>{SECTOR_META[s].label}</span>
            <span className="manifesto-mini-bar" aria-hidden>
              <span style={{ width: `${pcts[i]}%`, background: SECTOR_META[s].color }} />
            </span>
            <b className="num">{pcts[i]}%</b>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** The selected region beside the map: key numbers, your hunch, and its survey responses (in a dialog). */
function RegionDetail({ obs, region, draft }: { obs: Observation; region: RegionId; draft: AllocationDraft }) {
  const { hunches } = useGame();
  const [open, setOpen] = useState(false);
  const info = obs.regions.find((r) => r.id === region);
  const i = REGION_IDS.indexOf(region);
  const responses = obs.queries.filter((q) => q.region === region).length;
  const team = teamByRegion(obs)[i];
  const planned = draft.hires[i] ?? 0;
  const hunch = obs.archetype_library.find((a) => a.id === hunches[region]);
  const label = REGION_META[region].label;
  if (!info) return null;
  return (
    <div className="map-info">
      <h3>{label}</h3>
      <div className="region-detail">
        <Stat label="Voters" value={fmtInt(info.voters)} sub={`${((100 * info.voters) / obs.num_voters).toFixed(1)}% of all`} />
        <Stat label="Friends per voter" value={info.mean_degree.toFixed(1)} />
        <Stat label="Ads aired" value={obs.media_airings[region]} />
        <Stat label="Workers" value={team + planned} sub={planned ? `incl. ${planned} new` : undefined} />
      </div>
      <div className="stack" style={{ gap: 6 }}>
        <span className="stat-sub">
          Your hunch: <b className="ink2">{hunch?.name ?? "none yet"}</b>
        </span>
        {hunch ? <StackBar mean={hunch.mean} label={hunch.name} /> : null}
      </div>
      <button type="button" className="btn btn-sm" disabled={responses === 0} onClick={() => setOpen(true)}>
        {responses ? `Responses from ${label} (${responses})` : `No responses from ${label} yet`}
      </button>
      {open ? (
        <Modal title={`Responses from ${label}`} onClose={() => setOpen(false)} className="answers-modal">
          <RegionAnswers region={region} />
        </Modal>
      ) : null}
    </div>
  );
}

export function CampaignScreen() {
  const obs = useObservation();
  const { allocate, forfeit, busy, lastReport, reviewRound, setReviewRound } = useGame();
  // The draft belongs to one round: a new round starts from an empty plan in the same render (no stale frame
  // where last round's new hires would be counted on top of the now-larger permanent team).
  const [drafted, setDrafted] = useState<{ round: number; draft: AllocationDraft }>(() => ({ round: obs.round, draft: emptyDraft() }));
  const draft = drafted.round === obs.round ? drafted.draft : emptyDraft();
  const setDraft = (d: AllocationDraft) => setDrafted({ round: obs.round, draft: d });
  const [selected, setSelected] = useState<RegionId | null>("N");
  const [timerOn, setTimerOn] = useState(false);
  const [timerSecs, setTimerSecs] = useState(60);
  return (
    <div>
      <div className="subnav">
        <h1>
          Round {obs.round} of {obs.rounds}
        </h1>
        <RoundDots obs={obs} reviewing={reviewRound} onReview={setReviewRound} />
      </div>

      <div className="grid-campaign">
        <div className="stack">
          <Panel title="Polls" className="o-poll">
            <div className="hud">
              <PollHud obs={obs} />
              <div />
              <RoundTimer
                enabled={timerOn}
                seconds={timerSecs}
                round={obs.round}
                paused={busy !== null}
                onToggle={setTimerOn}
                onSeconds={setTimerSecs}
                onExpire={() => void forfeit()}
              />
            </div>
            <div style={{ marginTop: 12 }}>
              <PollChart obs={obs} />
            </div>
            {lastReport ? (
              <div className="report-strip" style={{ marginTop: 12 }} aria-label={`Round ${lastReport.round} report`}>
                <Stat label={`R${lastReport.round} spreaders`} value={fmtInt(lastReport.spreaders)} />
                <Stat label="Converted" value={fmtInt(lastReport.converted)} sub="spread next round" />
                <Stat label="Saw your ads" value={fmtInt(lastReport.media_exposed)} />
                <Stat label="Spent" value={formatMoney(lastReport.cost)} />
              </div>
            ) : null}
          </Panel>

          <Panel title="Map" className="o-map" hint="select a region">
            <div className="map-layout">
              <div className="map-wrap">
                <RegionMap obs={obs} selected={selected} onSelect={setSelected} overlay={{ workers: teamByRegion(obs).map((t, k) => t + (draft.hires[k] ?? 0)), media: draft.media }} />
              </div>
              {selected ? <RegionDetail obs={obs} region={selected} draft={draft} /> : null}
            </div>
          </Panel>

          {obs.manifesto ? (
            <Panel title="Your manifesto" className="o-manifesto">
              <ManifestoSummary manifesto={obs.manifesto} />
            </Panel>
          ) : null}
        </div>

        <div className="stack">
          <Panel title="Orders" className="o-orders">
            <AllocationPanel
              obs={obs}
              draft={draft}
              onChange={setDraft}
              busy={busy !== null}
              onSubmit={() =>
                void allocate({ hires_by_region: draft.hires, media_regions: draft.media })
              }
            />
          </Panel>

          <Panel title="Intel" className="o-intel">
            <QueryBuilder compact />
          </Panel>
        </div>
      </div>

      {reviewRound !== null && reviewableRounds(obs).includes(reviewRound) ? (
        <RoundLookback obs={obs} round={reviewRound} onClose={() => setReviewRound(null)} onNavigate={setReviewRound} />
      ) : null}
    </div>
  );
}
