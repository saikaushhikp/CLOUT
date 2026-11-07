"use client";

import { useEffect, useState } from "react";
import { Activity, ClipboardList, Map as MapIcon, Network, Newspaper, Search, Target, TrendingDown, TrendingUp } from "lucide-react";
import { useGame, useObservation } from "@/lib/game";
import { emptyDraft, type AllocationDraft } from "@/lib/allocation";
import { REGION_META, sectorLabel } from "@/lib/constants";
import { formatMoney } from "@/lib/money";
import { fmtCompact, fmtInt } from "@/lib/format";
import type { Observation, RegionId } from "@/lib/types";
import { REGION_IDS } from "@/lib/types";
import { AllocationPanel } from "./AllocationPanel";
import { RegionMap } from "./RegionMap";
import { PollChart } from "./PollChart";
import { RoundTimer } from "./RoundTimer";
import { NetworkView } from "./NetworkView";
import { KnowledgeLedger, QueryBuilder } from "./Intel";
import { Panel, StackBar, Stat } from "./ui";

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

function RoundDots({ obs }: { obs: Observation }) {
  return (
    <ol className="round-dots list-plain" aria-label="Campaign rounds">
      {Array.from({ length: obs.rounds }, (_, k) => k + 1).map((t) => (
        <li key={t} className={`round-dot ${t < obs.round ? "done" : t === obs.round ? "now" : ""}`} aria-current={t === obs.round ? "step" : undefined}>
          R{t}
        </li>
      ))}
      <li className="round-dot">🗳</li>
    </ol>
  );
}

function RegionDetail({ obs, region, draft }: { obs: Observation; region: RegionId; draft: AllocationDraft }) {
  const { hunches } = useGame();
  const info = obs.regions.find((r) => r.id === region);
  const i = REGION_IDS.indexOf(region);
  const answers = obs.queries.filter((q) => q.region === region).slice(-3).reverse();
  const sent = obs.allocations.reduce((a, al) => a + (al.workers_by_region[i] ?? 0), 0);
  const hunch = obs.archetype_library.find((a) => a.id === hunches[region]);
  if (!info) return null;
  return (
    <div className="stack" style={{ gap: 10, marginTop: 12 }}>
      <div className="region-detail">
        <Stat label="Voters" value={fmtInt(info.voters)} sub={`${((100 * info.voters) / obs.num_voters).toFixed(1)}% of electorate`} />
        <Stat label="Friends / voter" value={info.mean_degree.toFixed(1)} sub={`${fmtInt(info.intra_edges)} local ties`} />
        <Stat label="Ad slot" value={formatMoney(obs.costs.media_by_region[region])} sub={`${obs.media_airings[region]} aired so far`} />
        <Stat label="Workers sent" value={sent + draft.workers[i]} sub={`${draft.workers[i]} planned now`} />
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
          {answers.map((q, k) => (
            <li key={k}>
              {q.phase === "SURVEY" ? "Survey" : `R${q.round}`}: {q.pct_b_over_a.toFixed(1)}% prefer {sectorLabel(q.sector_b)} over {sectorLabel(q.sector_a)}
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
  const { allocate, forfeit, busy, lastReport, graph, graphError } = useGame();
  const [draft, setDraft] = useState<AllocationDraft>(emptyDraft());
  const [selected, setSelected] = useState<RegionId | null>("N");
  const [timerOn, setTimerOn] = useState(false);
  const [timerSecs, setTimerSecs] = useState(60);
  const [intelTab, setIntelTab] = useState<"ask" | "ledger">("ask");

  useEffect(() => {
    setDraft(emptyDraft());
  }, [obs.round]);

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
        <RoundDots obs={obs} />
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
                <Stat label={`R${lastReport.round} doorsteps`} value={fmtInt(lastReport.direct_contacts)} />
                <Stat label="Word of mouth" value={fmtInt(lastReport.word_of_mouth_reached)} />
                <Stat label="Saw your ads" value={fmtInt(lastReport.media_exposed)} />
                <Stat label="Spent" value={formatMoney(lastReport.cost)} />
              </div>
            ) : null}
          </Panel>

          <Panel title="The map" className="o-map" icon={MapIcon} hint="click a region">
            <div className="map-wrap">
              <RegionMap obs={obs} selected={selected} onSelect={setSelected} overlay={{ workers: draft.workers, media: draft.media }} />
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

          <Panel title="Social network (sample)" className="o-net" icon={Network} hint="regions only during play">
            <NetworkView sample={graph} error={graphError} totalVoters={obs.num_voters} />
          </Panel>
        </div>

        <div className="stack">
          <Panel title={`Round ${obs.round} orders`} className="o-orders" icon={Target}>
            <AllocationPanel
              obs={obs}
              draft={draft}
              onChange={setDraft}
              busy={busy !== null}
              onSubmit={() =>
                void allocate({ workers_by_region: draft.workers, extra_workers: draft.extra, media_regions: draft.media })
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
    </div>
  );
}
