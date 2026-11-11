"use client";

import { useMemo, useState } from "react";
import { ArrowLeftRight, Loader2, Search } from "lucide-react";
import { useGame, useObservation } from "@/lib/game";
import { REGION_META, SECTOR_META, regionLabel, sectorLabel } from "@/lib/constants";
import { fitArchetypes } from "@/lib/insight";
import { formatEstimate, groupAnswers, querySampleSize, sampleMoe } from "@/lib/survey";
import { formatMoney, toCents } from "@/lib/money";
import { fmtInt } from "@/lib/format";
import type { RegionId, SectorId } from "@/lib/types";
import { REGION_IDS, SECTOR_IDS } from "@/lib/types";
import { Modal, RegionPip, SectorSwatch, StackBar } from "./ui";

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
        Share of voters with the sector in their top 3. The tick marks 60%, where it would sit if every sector were equally
        popular.{prev ? ` Change since round ${prev.round}.` : ""}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ query builder */
export function QueryBuilder({ compact = false }: { compact?: boolean }) {
  const obs = useObservation();
  const { query, busy } = useGame();
  const [a, setA] = useState<SectorId>("Edu");
  const [b, setB] = useState<SectorId>("Agri");
  const [region, setRegion] = useState<RegionId>("N");
  const allowed = obs.legal_actions.includes("query");
  const cost = toCents(obs.costs.query);
  const affordable = toCents(obs.treasury) >= cost;
  const phase = obs.phase;
  const regionVoters = obs.regions.find((r) => r.id === region)?.voters ?? 0;
  const m = querySampleSize(regionVoters);
  // worst-case (p = 50%) margin of a single sample of this size
  const typicalMoe = sampleMoe(m / 2, m, regionVoters);
  const pooled = phase === "DONE" ? undefined : groupAnswers(
    obs.queries.filter((q) => q.round === obs.round && q.phase === phase && q.sector_a === a && q.sector_b === b && q.region === region),
  )[0];
  const reason = !allowed
    ? "Queries are closed."
    : a === b
      ? "Pick two different sectors."
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
        className="qb-form"
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
          <div className="seg seg-fill" role="radiogroup" aria-labelledby="q-region-label">
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
        <p className="stat-sub" data-testid="qb-sample" style={{ margin: compact ? "10px 0 6px" : "0 0 6px" }}>
          Asks <b className="num">≈{fmtInt(m)}</b> random {regionLabel(region)} voters, accurate to{" "}
          <span className="num">±{typicalMoe.toFixed(1)}</span> pts.{" "}
          {pooled
            ? `Asked ${pooled.answers.length}× this round, pooled: ${formatEstimate(pooled.pct, pooled.moe, pooled.sampleSize)}. Ask again to narrow the margin.`
            : "Asking again this round pools the samples and narrows the estimate."}
        </p>
        <div className="row-between">
          <span className="stat-sub" role="status">
            {reason}
          </span>
          <button type="submit" className="btn btn-primary" disabled={Boolean(reason) || busy !== null}>
            {busy === "query" ? <Loader2 className="spin" size={16} aria-hidden /> : <Search size={16} aria-hidden />}
            {pooled ? "Ask again" : "Ask"} · {formatMoney(obs.costs.query)}
          </button>
        </div>
      </form>
    </div>
  );
}

/* ------------------------------------------------------------------ knowledge ledger */
/** Estimate ± margin drawn on a 0–100% track: band = 95% interval, tick = point estimate. */
export function ErrorBar({ pct, moe, color, thin = false, label }: { pct: number; moe: number; color?: string; thin?: boolean; label?: string }) {
  const lo = Math.max(0, pct - moe);
  const hi = Math.min(100, pct + moe);
  return (
    <div
      className={`errbar${thin ? " thin" : ""}`}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      title={`95% interval ${lo.toFixed(1)}–${hi.toFixed(1)}%`}
    >
      <span className="errbar-mid" />
      <span className="errbar-band" style={{ left: `${lo}%`, width: `${Math.max(0.5, hi - lo)}%`, background: color ?? "var(--accent)" }} />
      <span className="errbar-dot" style={{ left: `${Math.min(100, Math.max(0, pct))}%`, background: color ?? "var(--accent)" }} />
    </div>
  );
}

/** Responses per region: one button per region; each opens a dialog with that region's responses. */
export function KnowledgeLedger({ row = false }: { row?: boolean }) {
  const obs = useObservation();
  const [open, setOpen] = useState<RegionId | null>(null);
  return (
    <div>
      <div className={`region-answers${row ? " region-answers-row" : ""}`} role="group" aria-label="Responses by region">
        {REGION_IDS.map((r) => {
          const n = obs.queries.filter((q) => q.region === r).length;
          return (
            <button key={r} type="button" className="region-answer-btn" disabled={n === 0} onClick={() => setOpen(r)}>
              <RegionPip id={r} />
              <span>
                <b>{REGION_META[r].label}</b>{" "}
                <small>{n ? `${n} response${n === 1 ? "" : "s"}` : "No responses yet"}</small>
              </span>
            </button>
          );
        })}
      </div>
      {open ? (
        <Modal title={`Responses from ${REGION_META[open].label}`} onClose={() => setOpen(null)} className="answers-modal">
          <RegionAnswers region={open} />
        </Modal>
      ) : null}
    </div>
  );
}

/** Every question asked about a region, newest first: a short summary box over each, then its estimate and samples. */
export function RegionAnswers({ region }: { region: RegionId }) {
  const obs = useObservation();
  // one entry per (triple, phase, round); repeats are pooled. Newest activity first.
  const groups = groupAnswers(obs.queries.filter((q) => q.region === region)).reverse();
  if (!groups.length) return <div className="empty">No responses yet.</div>;
  return (
    <ul className="ledger list-plain" aria-label="Your survey answers">
      {groups.map((g) => {
        const q = g.latest;
        const color = SECTOR_META[q.sector_b as SectorId]?.color;
        const repeats = g.answers.length;
        return (
          <li className="ledger-entry" key={g.key}>
            <p className="answer-flash answer-flash-sm">
              <b className="num">{q.pct_b_over_a.toFixed(1)}%</b> of {fmtInt(q.sample_size)} <b>{regionLabel(region)}</b> voters prefer{" "}
              <b>{sectorLabel(q.sector_b)}</b> to <b>{sectorLabel(q.sector_a)}</b> <span className="num">(±{q.moe.toFixed(1)})</span>.
              {repeats > 1 ? (
                <>
                  {" "}
                  Pooled over {repeats} samples: <b className="num">{formatEstimate(g.pct, g.moe, g.sampleSize)}</b>
                </>
              ) : null}
            </p>
            <div className="ledger-item">
              <RegionPip id={q.region} />
              <div style={{ minWidth: 0 }}>
                <div>
                  <b>{sectorLabel(q.sector_b)}</b> <span className="muted">over</span> {sectorLabel(q.sector_a)}
                </div>
                <ErrorBar pct={g.pct} moe={g.moe} color={color} />
              </div>
              <div style={{ textAlign: "right" }}>
                <b className="num" data-testid="ledger-estimate">
                  {formatEstimate(g.pct, g.moe, g.sampleSize)}
                </b>
                <div className="stat-sub mono">
                  {q.phase === "SURVEY" ? "survey" : `R${q.round}`}
                  {repeats > 1 ? ` · pooled ×${repeats}` : ""}
                </div>
              </div>
              {repeats > 1 ? (
                <ul className="ledger-repeats list-plain" aria-label="Individual samples">
                  {g.answers.map((a) => (
                    <li key={a.repeat_index}>
                      <span className="muted mono">#{a.repeat_index}</span>
                      <ErrorBar pct={a.pct_b_over_a} moe={a.moe} color={color} thin />
                      <span className="num muted">{formatEstimate(a.pct_b_over_a, a.moe, a.sample_size)}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          </li>
        );
      })}
    </ul>
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
        return (
          <article className="dossier-card" key={r.id} style={{ ["--rc" as string]: REGION_META[r.id as RegionId]?.color }}>
            <div className="row-between">
              <strong>{r.label}</strong>
              {answers.length ? <span className="badge">{answers.length} answer{answers.length === 1 ? "" : "s"}</span> : null}
            </div>
            <div className="stat-sub num">
              {fmtInt(r.voters)} voters ({((100 * r.voters) / obs.num_voters).toFixed(0)}%)
            </div>
            {/* fixed-height evidence area: cards keep their size as answers arrive */}
            <div className="dossier-evidence">
              {answers.length === 0 ? (
                <div className="stat-sub">No answers yet.</div>
              ) : (
                <div className="fit-list" aria-label={`Archetype fit for ${r.label}`}>
                  {fits.slice(0, 3).map((f, k) => (
                    <div className="fit-row" key={f.id}>
                      <span style={{ fontWeight: k === 0 ? 800 : 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {f.name}
                      </span>
                      <span className="meter" aria-hidden>
                        <span style={{ width: `${Math.max(4, f.posterior * 100)}%`, background: k === 0 ? "var(--accent)" : "var(--ink-3)" }} />
                      </span>
                      <span
                        className="num muted"
                        title={`Plausibility given your answers, weighting each pooled estimate by its margin of error (typical gap ${f.rmse.toFixed(1)} pts)`}
                      >
                        {Math.round(f.posterior * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div className="field dossier-hunch">
              <label htmlFor={`hunch-${r.id}`}>Your hunch</label>
              <select
                id={`hunch-${r.id}`}
                className="select"
                value={hunches[r.id] ?? ""}
                onChange={(e) => setHunch(r.id, e.target.value || undefined)}
              >
                <option value="">Undecided</option>
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
              {claimed.get(a.id) ? (
                <span className="arch-pips">
                  {claimed.get(a.id)!.map((r) => (
                    <RegionPip key={r} id={r} size={20} />
                  ))}
                </span>
              ) : null}
            </div>
            <StackBar mean={a.mean} label={a.name} />
            <div className="stat-sub num">
              {SECTOR_IDS.map((s, i) => `${SECTOR_META[s].short} ${Math.round(a.mean[i] * 100)}`).join(" · ")}
            </div>
          </article>
        ))}
      </div>
      <p className="stat-sub">Each region follows one of these profiles, and no two regions share one.</p>
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
            friends {Math.round(p.social * 100)} · media {Math.round(p.media * 100)} · canvassing {Math.round(p.campaign * 100)}
          </span>
        </li>
      ))}
    </ul>
  );
}

