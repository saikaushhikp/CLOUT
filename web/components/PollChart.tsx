"use client";

import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Observation } from "@/lib/types";

type Row = { x: number; label: string; est?: number; band?: [number, number]; moe?: number; truth?: number };

export function roundLabel(x: number, rounds: number): string {
  return x > rounds ? "Eve" : `R${x}`;
}

/** Rows keyed by "start of round x" (x = T+1 is the pre-election poll). truth[i] = expected share before round i+1. */
export function buildPollRows(polls: Observation["polls"], rounds: number, truth?: number[]): Row[] {
  const rows: Row[] = [];
  for (let x = 1; x <= rounds + 1; x++) {
    const p = polls.find((q) => q.round === x);
    const t = truth?.[x - 1];
    if (!p && t === undefined) continue;
    rows.push({
      x,
      label: roundLabel(x, rounds),
      ...(p
        ? {
            est: +(p.estimate * 100).toFixed(2),
            moe: +(p.moe * 100).toFixed(2),
            band: [+(Math.max(0, p.estimate - p.moe) * 100).toFixed(2), +(Math.min(1, p.estimate + p.moe) * 100).toFixed(2)] as [number, number],
          }
        : {}),
      ...(t !== undefined ? { truth: +(t * 100).toFixed(2) } : {}),
    });
  }
  return rows;
}

function Tip({ active, payload }: { active?: boolean; payload?: { payload: Row }[] }) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  return (
    <div className="chart-tip">
      <b>{r.label === "Eve" ? "Election eve" : `Start of round ${r.x}`}</b>
      {r.est !== undefined ? (
        <div>
          Poll {r.est.toFixed(1)}% ± {r.moe?.toFixed(1)}
        </div>
      ) : null}
      {r.truth !== undefined ? <div>True expected {r.truth.toFixed(1)}%</div> : null}
    </div>
  );
}

export function PollChart({ obs, truth, height = 240 }: { obs: Observation; truth?: number[]; height?: number }) {
  const rows = buildPollRows(obs.polls, obs.rounds, truth);
  if (rows.length === 0) return <div className="empty">Polls start once the manifesto is committed.</div>;
  const values = rows.flatMap((r) => [...(r.band ?? []), r.truth ?? 50]);
  const lo = Math.max(0, Math.floor((Math.min(50, ...values) - 4) / 10) * 10);
  const hi = Math.min(100, Math.ceil((Math.max(50, ...values) + 4) / 10) * 10);
  const ticks = Array.from({ length: (hi - lo) / 10 + 1 }, (_, k) => lo + 10 * k);
  const summary = rows
    .map((r) => `${r.label}: ${r.est !== undefined ? `poll ${r.est.toFixed(1)}% ±${r.moe?.toFixed(1)}` : ""}${r.truth !== undefined ? ` true ${r.truth.toFixed(1)}%` : ""}`)
    .join("; ");
  return (
    <figure style={{ margin: 0 }}>
      <div className="chart-box" style={{ height }} role="img" aria-label={`Poll history. ${summary}`}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: -18 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="label" tickLine={false} axisLine={{ stroke: "var(--axis)" }} interval={0} />
            <YAxis domain={[lo, hi]} ticks={ticks} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v}%`} width={52} />
            <ReferenceLine y={50} stroke="var(--accent)" strokeDasharray="4 4" label={{ value: "50% to win", position: "insideTopRight", fill: "var(--ink-3)", fontSize: 11 }} />
            <Area dataKey="band" stroke="none" fill="var(--blue)" fillOpacity={0.16} isAnimationActive={false} connectNulls />
            <Line dataKey="est" stroke="var(--blue)" strokeWidth={2} dot={{ r: 4, fill: "var(--blue)", stroke: "var(--surface)", strokeWidth: 2 }} activeDot={{ r: 6 }} isAnimationActive={false} connectNulls />
            {truth ? (
              <Line dataKey="truth" stroke="var(--ink)" strokeWidth={2} strokeDasharray="6 3" dot={{ r: 4, fill: "var(--ink)", stroke: "var(--surface)", strokeWidth: 2 }} isAnimationActive={false} connectNulls />
            ) : null}
            <Tooltip content={<Tip />} cursor={{ stroke: "var(--axis)" }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="legend" style={{ marginTop: 6 }}>
        <span>
          <span className="swatch" style={{ background: "var(--blue)" }} aria-hidden /> Poll (400 voters) with 95% margin of error
        </span>
        {truth ? (
          <span>
            <span className="swatch" style={{ background: "var(--ink)" }} aria-hidden /> True expected share (revealed)
          </span>
        ) : null}
      </figcaption>
    </figure>
  );
}
