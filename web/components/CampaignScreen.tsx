"use client";

import { useState } from "react";
import { Activity, ClipboardList, Lock, Map as MapIcon, Newspaper, Search, Stamp, Target, TrendingDown, TrendingUp } from "lucide-react";
import { useGame, useObservation } from "@/lib/game";
import { emptyDraft, teamByRegion, type AllocationDraft } from "@/lib/allocation";
import { REGION_META, SECTOR_META, sectorLabel } from "@/lib/constants";
import { formatMoney } from "@/lib/money";
import { fmtCompact, fmtInt } from "@/lib/format";
import { formatEstimate, groupAnswers } from "@/lib/survey";
import type { Observation, RegionId } from "@/lib/types";
import { REGION_IDS, SECTOR_IDS } from "@/lib/types";
import { toIntegerSimplex } from "@/lib/simplex";
import { reviewableRounds } from "@/lib/rounds";
import { AllocationPanel } from "./AllocationPanel";
import { RegionMap } from "./RegionMap";
import { PollChart } from "./PollChart";
import { RoundTimer } from "./RoundTimer";
import { KnowledgeLedger, QueryBuilder } from "./Intel";
import { Donut } from "./Donut";
import { RoundLookback } from "./RoundLookback";
import { Panel, SectorSwatch, StackBar, Stat } from "./ui";

function PollHud({ obs }: { obs: Observation }) {
  const latest = obs.polls[obs.polls.length - 1];
  const prev = obs.polls.length > 1 ? obs.polls[obs.polls.length - 2] : null;
  if (!latest) return null;
  const d = prev ? (latest.estimate - prev.estimate) * 100 : 0;
  const lead = latest.estimate - latest.moe > 0.5 ? "Clear lead" : latest.estimate + latest.moe < 0.5 ? "Trailing" : "Too close to call";
  return (
    <div>
      <div className="stat-label">Latest poll · start of round {latest.round}</div>
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
      <li>
        <span className="round-dot" role="img" aria-label="Election">
          🗳
        </span>
      </li>
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

function RegionDetail({ obs, region, draft }: { obs: Observation; region: RegionId; draft: AllocationDraft }) {
  const { hunches } = useGame();
  const info = obs.regions.find((r) => r.id === region);
  const i = REGION_IDS.indexOf(region);
  const answers = groupAnswers(obs.queries.filter((q) => q.region === region)).slice(-3).reverse();
  const team = teamByRegion(obs)[i];
  const planned = draft.hires[i] ?? 0;
  const hunch = obs.archetype_library.find((a) => a.id === hunches[region]);
  if (!info) return null;
  return (
    <div className="stack" style={{ gap: 10, marginTop: 12 }}>
      <div className="region-detail">
        <Stat label="Voters" value={fmtInt(info.voters)} sub={`${((100 * info.voters) / obs.num_voters).toFixed(1)}% of electorate`} />
        <Stat label="Friends / voter" value={info.mean_degree.toFixed(1)} sub={`${fmtInt(info.intra_edges)} local ties`} />
        <Stat label="Ad slot" value={formatMoney(obs.costs.media_by_region[region])} sub={`${obs.media_airings[region]} aired so far`} />
        <Stat label="Permanent team" value={team} sub={planned ? `+${planned} to buy now` : "workers in this region"} />
      </div>
      <div className="row-between">
        <span className="stat-sub">
          Your hunch: <b className="ink2">{hunch?.name ?? "none yet"}</b>
        </span>
        {hunch ? (
          <div style={{ width: 160 }}>
            <StackBar mean={hunch.mean} label={hunch.name} />
          </div>
        ) : null}
      </div>
      {answers.length ? (
        <ul className="list-plain stat-sub">
          {answers.map((g) => (
            <li key={g.key}>
              {g.latest.phase === "SURVEY" ? "Survey" : `R${g.latest.round}`}: {formatEstimate(g.pct, g.moe, g.sampleSize)} prefer{" "}
              {sectorLabel(g.latest.sector_b)} over {sectorLabel(g.latest.sector_a)}
              {g.answers.length > 1 ? ` (pooled ×${g.answers.length})` : ""}
            </li>
          ))}
        </ul>
      ) : (
        <span className="stat-sub">No paid answers about {REGION_META[region].label} yet.</span>
      )}
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
  const [intelTab, setIntelTab] = useState<"ask" | "ledger">("ask");

  const events = obs.events.slice(-30).reverse();
  return (
    <div>
      <div className="subnav">
        <div>
          <span className="kicker">Phase 2 · Campaign HQ</span>
          <h1>
            Round {obs.round} of {obs.rounds}
          </h1>
        </div>
        <RoundDots obs={obs} reviewing={reviewRound} onReview={setReviewRound} />
      </div>

      <div className="grid-campaign">
        <div className="stack">
          <Panel title="Polling desk" className="o-poll" icon={Activity}>
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

          <Panel title="The map" className="o-map" icon={MapIcon} hint="click a region">
            <div className="map-wrap">
              <RegionMap obs={obs} selected={selected} onSelect={setSelected} overlay={{ workers: teamByRegion(obs).map((t, k) => t + (draft.hires[k] ?? 0)), media: draft.media }} />
            </div>
            <div className="row" style={{ gap: 6, marginTop: 10 }} aria-label="Cross-region friendships">
              <span className="stat-sub">Cross-region ties:</span>
              {obs.inter_edges.map((e) => (
                <span key={e.regions.join("")} className="badge num" title={`${e.edges.toLocaleString("en-US")} friendships between ${e.regions.join(" and ")}`}>
                  {e.regions.join("–")} {fmtCompact(e.edges)}
                </span>
              ))}
            </div>
            {selected ? <RegionDetail obs={obs} region={selected} draft={draft} /> : null}
          </Panel>

          {obs.manifesto ? (
            <Panel title="Your manifesto" className="o-manifesto" icon={Stamp} hint="committed · locked">
              <ManifestoSummary manifesto={obs.manifesto} />
            </Panel>
          ) : null}
        </div>

        <div className="stack">
          <Panel title={`Round ${obs.round} orders`} className="o-orders" icon={Target}>
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

          <Panel
            title="Intel" className="o-intel"
            icon={intelTab === "ask" ? Search : ClipboardList}
            actions={
              <div className="seg" role="tablist" aria-label="Intel views">
                <button type="button" role="tab" aria-selected={intelTab === "ask"} onClick={() => setIntelTab("ask")}>
                  Ask
                </button>
                <button type="button" role="tab" aria-selected={intelTab === "ledger"} onClick={() => setIntelTab("ledger")}>
                  Ledger ({obs.queries.length})
                </button>
              </div>
            }
          >
            {intelTab === "ask" ? <QueryBuilder compact /> : <KnowledgeLedger />}
          </Panel>

          <Panel title="Wire feed" className="o-feed" icon={Newspaper}>
            <ul className="feed list-plain" aria-live="polite">
              {events.map((e) => (
                <li key={e.seq}>
                  <span className="r">{e.round === 0 ? "SRV" : `R${e.round}`}</span>
                  <span>{e.message}</span>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      </div>

      {reviewRound !== null && reviewableRounds(obs).includes(reviewRound) ? (
        <RoundLookback obs={obs} round={reviewRound} onClose={() => setReviewRound(null)} onNavigate={setReviewRound} />
      ) : null}
    </div>
  );
}
