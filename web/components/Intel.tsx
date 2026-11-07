"use client";

import { useMemo, useState } from "react";
import { ArrowLeftRight, Loader2, Search } from "lucide-react";
import { useGame, useObservation } from "@/lib/game";
import { REGION_META, SECTOR_META, regionLabel, sectorLabel } from "@/lib/constants";
import { fitArchetypes } from "@/lib/insight";
import { formatMoney, toCents } from "@/lib/money";
import { fmtInt } from "@/lib/format";
import type { RegionId, SectorId } from "@/lib/types";
import { REGION_IDS, SECTOR_IDS } from "@/lib/types";
import { Panel, RegionPip, SectorSwatch, StackBar } from "./ui";

/* ------------------------------------------------------------------ free global statistic */
export function Top3Bars() {
  const obs = useObservation();
  const latest = obs.free_stats[obs.free_stats.length - 1];
  const prev = obs.free_stats.length > 1 ? obs.free_stats[obs.free_stats.length - 2] : null;
  if (!latest) return <div className="empty">No statistics yet.</div>;
  return (
    <div>
      <ul className="hbars list-plain" aria-label="Share of all voters with each sector in their top three">
        {SECTOR_IDS.map((s) => {
          const v = latest.top3_pct[s] ?? 0;
          const d = prev ? v - (prev.top3_pct[s] ?? 0) : 0;
          return (
            <li className="hbar" key={s}>
              <span className="hbar-name">
                <SectorSwatch id={s} />
                {SECTOR_META[s].label}
              </span>
              <span className="hbar-track" aria-hidden>
                <span className="hbar-fill" style={{ width: `${Math.min(100, v)}%`, display: "block", background: SECTOR_META[s].color }} />
                <span
                  style={{ position: "absolute", left: "60%", top: 0, bottom: 0, width: 1, background: "var(--ink-3)", opacity: 0.6 }}
                />
              </span>
              <span className="num" style={{ textAlign: "right", fontWeight: 800 }}>
                {v.toFixed(1)}%
                {prev && Math.abs(d) >= 0.05 ? (
                  <span className={`hbar-delta ${d > 0 ? "good" : "bad"}`} style={{ display: "block" }}>
                    {d > 0 ? "▲" : "▼"} {Math.abs(d).toFixed(1)}
                  </span>
                ) : null}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="stat-sub" style={{ marginTop: 10 }}>
        % of all voters ranking the sector in their personal top 3 (sums to 300%). Tick = 60%, what a perfectly even
        electorate would show. {prev ? `Change vs. round ${prev.round}.` : "Refreshed free every round."}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ query builder */
export function QueryBuilder({ compact = false }: { compact?: boolean }) {
  const obs = useObservation();
  const { query, busy, lastAnswer } = useGame();
  const [a, setA] = useState<SectorId>("Edu");
  const [b, setB] = useState<SectorId>("Agri");
  const [region, setRegion] = useState<RegionId>("N");
  const allowed = obs.legal_actions.includes("query");
  const cost = toCents(obs.costs.query);
  const affordable = toCents(obs.treasury) >= cost;
  const phase = obs.phase;
  const duplicate = obs.queries.some(
    (q) => q.round === obs.round && q.phase === phase && q.sector_a === a && q.sector_b === b && q.region === region,
  );
  const reason = !allowed
    ? "Queries are closed."
    : a === b
      ? "Pick two different sectors."
      : duplicate
        ? "Already asked this round."
        : !affordable
          ? "Not enough money."
          : null;

  return (
    <div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!reason) void query(a, b, region);
        }}
        aria-label="Paid survey question"
      >
        <div className="qb">
          <div className="field">
            <label htmlFor="q-a">Sector A</label>
            <select id="q-a" className="select" value={a} onChange={(e) => setA(e.target.value as SectorId)}>
              {SECTOR_IDS.map((s) => (
                <option key={s} value={s}>
                  {sectorLabel(s)}
                </option>
              ))}
            </select>
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-icon qb-vs"
            aria-label="Swap sectors"
            title="Swap sectors"
            onClick={() => {
              setA(b);
              setB(a);
            }}
          >
            <ArrowLeftRight size={16} aria-hidden />
          </button>
          <div className="field">
            <label htmlFor="q-b">Sector B</label>
            <select id="q-b" className="select" value={b} onChange={(e) => setB(e.target.value as SectorId)}>
              {SECTOR_IDS.map((s) => (
                <option key={s} value={s}>
                  {sectorLabel(s)}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="field" style={{ marginTop: 10 }}>
          <span className="field-label" id="q-region-label">
            Region
          </span>
          <div className="seg" role="radiogroup" aria-labelledby="q-region-label">
            {REGION_IDS.map((r) => (
              <button key={r} type="button" role="radio" aria-checked={region === r} aria-pressed={region === r} onClick={() => setRegion(r)}>
                <span className="swatch" style={{ background: REGION_META[r].color, marginRight: 6 }} aria-hidden />
                {REGION_META[r].label}
              </button>
            ))}
          </div>
        </div>
        {!compact ? (
          <p className="qb-sentence">
            “What share of <b>{regionLabel(region)}</b> voters want more <b>{sectorLabel(b)}</b> than{" "}
            <b>{sectorLabel(a)}</b>?”
          </p>
        ) : null}
        <div className="row-between" style={{ marginTop: compact ? 10 : 0 }}>
          <span className="stat-sub" role="status">
            {reason ?? `Exact answer, current state · costs ${formatMoney(obs.costs.query)}`}
          </span>
          <button type="submit" className="btn btn-primary" disabled={Boolean(reason) || busy !== null}>
            {busy === "query" ? <Loader2 className="spin" size={16} aria-hidden /> : <Search size={16} aria-hidden />}
            Ask · {formatMoney(obs.costs.query)}
          </button>
        </div>
      </form>
      {lastAnswer ? (
        <div className="answer-flash" role="status" aria-live="polite">
          <span className="big num">{lastAnswer.pct_b_over_a.toFixed(1)}%</span>
          <span>
            of <b>{regionLabel(lastAnswer.region)}</b> voters rank <b>{sectorLabel(lastAnswer.sector_b)}</b> above{" "}
            <b>{sectorLabel(lastAnswer.sector_a)}</b>.
          </span>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ knowledge ledger */
export function KnowledgeLedger() {
  const obs = useObservation();
  const [filter, setFilter] = useState<string>("all");
  const items = obs.queries
    .map((q, i) => ({ q, i }))
    .filter(({ q }) => filter === "all" || q.region === filter)
    .reverse();
  return (
    <div>
      <div className="seg" role="group" aria-label="Filter answers by region" style={{ marginBottom: 10 }}>
        {["all", ...REGION_IDS].map((r) => (
          <button key={r} type="button" aria-pressed={filter === r} onClick={() => setFilter(r)}>
            {r === "all" ? `All (${obs.queries.length})` : r}
          </button>
        ))}
      </div>
      {items.length === 0 ? (
        <div className="empty">No answers yet. Every paid question lands here.</div>
      ) : (
        <ul className="ledger list-plain" aria-label="Your survey answers">
          {items.map(({ q, i }) => (
            <li className="ledger-item" key={i}>
              <RegionPip id={q.region} />
              <div style={{ minWidth: 0 }}>
                <div>
                  <b>{sectorLabel(q.sector_b)}</b> <span className="muted">over</span> {sectorLabel(q.sector_a)}
                </div>
                <div className="split-bar" aria-hidden>
                  <span style={{ width: `${q.pct_b_over_a}%`, background: SECTOR_META[q.sector_b as SectorId]?.color }} />
                  <span style={{ flex: 1, background: SECTOR_META[q.sector_a as SectorId]?.color, opacity: 0.45 }} />
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <b className="num">{q.pct_b_over_a.toFixed(1)}%</b>
                <div className="stat-sub mono">{q.phase === "SURVEY" ? "survey" : `R${q.round}`}</div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ region dossiers + archetype matcher */
export function RegionDossiers() {
  const obs = useObservation();
  const { hunches, setHunch } = useGame();
  return (
    <div className="dossier">
      {obs.regions.map((r) => {
        const answers = obs.queries.filter((q) => q.region === r.id);
        const fits = fitArchetypes(obs, obs.queries, r.id);
        const worst = Math.max(25, ...fits.map((f) => (Number.isNaN(f.rmse) ? 0 : f.rmse)));
        return (
          <article className="dossier-card" key={r.id} style={{ ["--rc" as string]: REGION_META[r.id as RegionId]?.color }}>
            <div className="row-between">
              <strong>
                {r.label} <span className="muted">({r.id})</span>
              </strong>
              <span className="badge">{answers.length} answer{answers.length === 1 ? "" : "s"}</span>
            </div>
            <div className="stat-sub num">
              {fmtInt(r.voters)} voters · {((100 * r.voters) / obs.num_voters).toFixed(0)}% · avg {r.mean_degree.toFixed(1)} friends
            </div>
            {answers.length === 0 ? (
              <div className="stat-sub">Ask about this region to see which archetypes fit.</div>
            ) : (
              <div className="fit-list" aria-label={`Archetype fit for ${r.label}`}>
                {fits.slice(0, 3).map((f, k) => (
                  <div className="fit-row" key={f.id}>
                    <span style={{ fontWeight: k === 0 ? 800 : 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {f.name}
                    </span>
                    <span className="meter" aria-hidden>
                      <span style={{ width: `${Math.max(4, 100 - (f.rmse / worst) * 100)}%`, background: k === 0 ? "var(--accent)" : "var(--ink-3)" }} />
                    </span>
                    <span className="num muted" title="Root-mean-square gap between your answers and what this archetype predicts (percentage points)">
                      ±{f.rmse.toFixed(0)}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <div className="field">
              <label htmlFor={`hunch-${r.id}`}>Your hunch</label>
              <select
                id={`hunch-${r.id}`}
                className="select"
                value={hunches[r.id] ?? ""}
                onChange={(e) => setHunch(r.id, e.target.value || undefined)}
              >
                <option value="">— undecided —</option>
                {obs.archetype_library.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.name}
                  </option>
                ))}
              </select>
            </div>
          </article>
        );
      })}
    </div>
  );
}

export function ArchetypeLibrary() {
  const obs = useObservation();
  const { hunches } = useGame();
  const claimed = useMemo(() => {
    const m = new Map<string, string[]>();
    Object.entries(hunches).forEach(([r, a]) => a && m.set(a, [...(m.get(a) ?? []), r]));
    return m;
  }, [hunches]);
  return (
    <div className="stack" style={{ gap: 12 }}>
      <div className="arch-grid">
        {obs.archetype_library.map((a) => (
          <article className="arch-card" key={a.id}>
            <div className="row-between">
              <h4>{a.name}</h4>
              {claimed.get(a.id)?.map((r) => <RegionPip key={r} id={r} size={20} />)}
            </div>
            <StackBar mean={a.mean} label={a.name} />
            <div className="stat-sub num">
              {SECTOR_IDS.map((s, i) => `${SECTOR_META[s].short} ${Math.round(a.mean[i] * 100)}`).join(" · ")}
            </div>
          </article>
        ))}
      </div>
      <p className="stat-sub">
        Each region secretly follows one of these six profiles (no two regions share one). Voters scatter around the
        profile mean — the fit meter compares your answers with what each profile would predict.
      </p>
    </div>
  );
}

export function PersonalityLibrary() {
  const obs = useObservation();
  return (
    <ul className="list-plain stack" style={{ gap: 8 }}>
      {obs.personality_library.map((p) => (
        <li key={p.id} className="ledger-item" style={{ gridTemplateColumns: "minmax(0,1fr) auto" }}>
          <b>{p.name}</b>
          <span className="stat-sub num" title="Mean susceptibility to friends / media / canvassing">
            friends {Math.round(p.social * 100)} · media {Math.round(p.media * 100)} · doorstep {Math.round(p.campaign * 100)}
          </span>
        </li>
      ))}
    </ul>
  );
}

