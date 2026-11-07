"use client";

import { useEffect, useRef, useState } from "react";
import { BarChart3, Lock, LockOpen, Loader2, Shuffle, Stamp, Wand2 } from "lucide-react";
import { useGame, useObservation } from "@/lib/game";
import { SECTOR_META, sectorLabel } from "@/lib/constants";
import { percentToManifesto, rebalance, similarity, toIntegerSimplex } from "@/lib/simplex";
import { blendGuesses } from "@/lib/insight";
import { SECTOR_IDS } from "@/lib/types";
import { Panel, SectorSwatch } from "./ui";
import { Donut } from "./Donut";

export function SimplexSliders({
  values,
  locked,
  onChange,
  onToggleLock,
}: {
  values: number[];
  locked: boolean[];
  onChange: (next: number[]) => void;
  onToggleLock: (i: number) => void;
}) {
  return (
    <div role="group" aria-label="Manifesto budget shares">
      {SECTOR_IDS.map((s, i) => {
        const id = `ms-${s}`;
        return (
          <div className="mslider" key={s} style={{ ["--sc" as string]: SECTOR_META[s].color }}>
            <label className="mslider-name" htmlFor={id}>
              <SectorSwatch id={s} />
              {SECTOR_META[s].label}
            </label>
            <input
              id={id}
              type="range"
              min={0}
              max={100}
              step={1}
              value={values[i]}
              disabled={locked[i]}
              aria-valuetext={`${values[i]} percent`}
              onChange={(e) => onChange(rebalance(values, i, Number(e.target.value), locked))}
            />
            <input
              className="input"
              type="number"
              min={0}
              max={100}
              value={values[i]}
              disabled={locked[i]}
              aria-label={`${SECTOR_META[s].label} percent`}
              onChange={(e) => onChange(rebalance(values, i, Number(e.target.value), locked))}
            />
            <button
              type="button"
              className="btn btn-ghost btn-icon btn-sm"
              aria-pressed={locked[i]}
              aria-label={`${locked[i] ? "Unlock" : "Lock"} ${SECTOR_META[s].label}`}
              title={locked[i] ? "Unlock" : "Lock while adjusting others"}
              onClick={() => onToggleLock(i)}
            >
              {locked[i] ? <Lock size={15} aria-hidden /> : <LockOpen size={15} aria-hidden />}
            </button>
          </div>
        );
      })}
    </div>
  );
}

export function ManifestoDesigner() {
  const obs = useObservation();
  const { commit, busy, hunches, pushToast } = useGame();
  const [values, setValues] = useState<number[]>([20, 20, 20, 20, 20]);
  const [locked, setLocked] = useState<boolean[]>([false, false, false, false, false]);
  const [confirming, setConfirming] = useState(false);
  const confirmRef = useRef<HTMLButtonElement>(null);
  const sum = values.reduce((a, b) => a + b, 0);
  const manifesto = percentToManifesto(values);
  const blend = blendGuesses(obs, hunches);

  useEffect(() => {
    if (confirming) confirmRef.current?.focus();
  }, [confirming]);
  useEffect(() => {
    if (!confirming) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setConfirming(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [confirming]);

  const top3 = obs.free_stats[obs.free_stats.length - 1]?.top3_pct;

  return (
    <div className="manifesto">
      <Panel title="Budget manifesto" icon={Stamp} hint={<span className="sum-pill num">Σ {sum}%</span>}>
        <SimplexSliders
          values={values}
          locked={locked}
          onChange={setValues}
          onToggleLock={(i) => setLocked((l) => l.map((x, k) => (k === i ? !x : x)))}
        />
        <div className="row" style={{ marginTop: 14 }}>
          <button type="button" className="btn btn-sm" onClick={() => setValues([20, 20, 20, 20, 20])}>
            <Shuffle size={14} aria-hidden /> Even split
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={!blend}
            title={blend ? "Voter-weighted average of the archetypes you pencilled in" : "Set a hunch for at least one region first"}
            onClick={() => blend && setValues(toIntegerSimplex(blend))}
          >
            <Wand2 size={14} aria-hidden /> Blend my hunches
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={!top3}
            onClick={() => {
              if (!top3) return;
              // Exaggerate differences above the 60% "even" line so the popular sectors stand out.
              setValues(toIntegerSimplex(SECTOR_IDS.map((s) => Math.max(1, (top3[s] ?? 60) - 30))));
              pushToast({ kind: "info", title: "Drafted from the free statistic", message: "A blunt starting point — the top-3 stat hides regional differences." });
            }}
          >
            <BarChart3 size={14} aria-hidden /> Draft from top-3 stats
          </button>
        </div>
        <p className="stat-sub" style={{ marginTop: 12 }}>
          Sliders are linked: moving one rebalances the unlocked others proportionally, so the total is always 100%.
          Voters back you when your budget is close to theirs (similarity = 1 − half the L1 distance). Tip: opinions drift
          toward their region&apos;s average over the campaign — aim at regions, not individuals.
        </p>
      </Panel>

      <div className="stack">
        <Panel title="Preview">
          <div className="donut-wrap">
            <Donut values={values} size={210} />
            <div className="donut-center">
              <div>
                <div className="stat-label">Largest</div>
                <div style={{ fontWeight: 900, fontSize: 18 }}>{sectorLabel(SECTOR_IDS[values.indexOf(Math.max(...values))])}</div>
                <div className="num muted">{Math.max(...values)}%</div>
              </div>
            </div>
          </div>
          <div className="stack" style={{ gap: 6, marginTop: 14 }}>
            <span className="stat-label">Closeness to each archetype</span>
            {obs.archetype_library
              .map((a) => ({ a, s: similarity(manifesto, a.mean) }))
              .sort((x, y) => y.s - x.s)
              .map(({ a, s }) => (
                <div className="match-row" key={a.id}>
                  <span>
                    {a.name}
                    {Object.entries(hunches).some(([, h]) => h === a.id) ? <span className="muted"> · hunch</span> : null}
                  </span>
                  <span className="meter" aria-hidden>
                    <span style={{ width: `${Math.max(0, (s - 0.5) / 0.5) * 100}%`, background: s >= 0.825 ? "var(--good)" : "var(--ink-3)" }} />
                  </span>
                  <span className="num" style={{ textAlign: "right", fontWeight: 700 }}>
                    {(s * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            <span className="stat-sub">Green = above the ~82.5% similarity where a typical voter flips to “likely yes”.</span>
          </div>
        </Panel>
        <button
          type="button"
          className="btn btn-primary btn-lg"
          disabled={!obs.legal_actions.includes("commit") || sum !== 100 || busy !== null}
          onClick={() => setConfirming(true)}
        >
          <Stamp size={18} aria-hidden /> Commit manifesto
        </button>
      </div>

      {confirming ? (
        <div className="modal-backdrop" role="presentation" onClick={() => setConfirming(false)}>
          <div
            className="panel modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="confirm-title"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 id="confirm-title">Lock in this manifesto?</h2>
            <p className="ink2">
              It can&apos;t be changed for the rest of the campaign. Round 1 starts immediately and the first poll lands.
            </p>
            <ul className="list-plain stack" style={{ gap: 4, margin: "12px 0" }}>
              {SECTOR_IDS.map((s, i) => (
                <li key={s} className="row-between">
                  <span className="row" style={{ gap: 6 }}>
                    <SectorSwatch id={s} />
                    {SECTOR_META[s].label}
                  </span>
                  <b className="num">{values[i]}%</b>
                </li>
              ))}
            </ul>
            <div className="row" style={{ justifyContent: "flex-end" }}>
              <button type="button" className="btn" onClick={() => setConfirming(false)}>
                Keep editing
              </button>
              <button
                ref={confirmRef}
                type="button"
                className="btn btn-primary"
                disabled={busy !== null}
                onClick={async () => {
                  const res = await commit(manifesto);
                  if (!res?.accepted) setConfirming(false);
                }}
              >
                {busy === "commit" ? <Loader2 className="spin" size={16} aria-hidden /> : <Stamp size={16} aria-hidden />}
                Commit &amp; start round 1
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
