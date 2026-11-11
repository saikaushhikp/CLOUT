"use client";

import { useEffect, useRef, useState } from "react";
import { BarChart3, ChevronDown, ChevronUp, Lock, LockOpen, Loader2, Shuffle, Stamp, Wand2 } from "lucide-react";
import { useGame, useObservation } from "@/lib/game";
import { BLEND_MAX, BLEND_MIN, REGION_META, SECTOR_META, sectorLabel } from "@/lib/constants";
import { percentToManifesto, rebalance, similarity, toIntegerSimplex } from "@/lib/simplex";
import { blendSelected, largestHunchRegions, regionsWithHunch } from "@/lib/insight";
import { fmtInt } from "@/lib/format";
import type { Observation, RegionId } from "@/lib/types";
import { SECTOR_IDS } from "@/lib/types";
import { Panel, RegionPip, SectorSwatch, StackBar } from "./ui";
import { KnowledgeLedger } from "./Intel";

export const BLEND_HELP =
  "Takes the archetype you picked for each chosen region and averages their budget profiles, weighted by how many voters live there.";

/**
 * "Blend my hunches" picker (R20): choose WHICH regions (2–4) to blend. Regions without a hunch are disabled.
 * Applying sets the sliders to the voter-weighted average of the chosen regions' hunch archetypes.
 */
export function BlendPicker({
  obs,
  hunches,
  onApply,
}: {
  obs: Pick<Observation, "regions" | "archetype_library">;
  hunches: Record<string, string | undefined>;
  onApply: (blend: number[], regions: string[]) => void;
}) {
  const eligible = regionsWithHunch(obs, hunches);
  const [chosen, setChosen] = useState<string[]>(() => eligible.slice(0, BLEND_MAX));
  // drop regions whose hunch was cleared
  const picked = chosen.filter((r) => eligible.includes(r));
  const blend = blendSelected(obs, hunches, picked, BLEND_MIN, BLEND_MAX);
  const totalVoters = picked.reduce((a, r) => a + (obs.regions.find((x) => x.id === r)?.voters ?? 0), 0);
  const toggle = (id: string) =>
    setChosen((c) => (c.includes(id) ? c.filter((x) => x !== id) : c.length >= BLEND_MAX ? c : [...c, id]));
  const tooFew = picked.length < BLEND_MIN;
  return (
    <fieldset className="blend-picker" aria-describedby="blend-help">
      <legend>Blend which regions?</legend>
      <p id="blend-help" className="stat-sub blend-help">
        {BLEND_HELP} Pick {BLEND_MIN} to {BLEND_MAX} regions that have a hunch.
      </p>
      <div className="row" style={{ gap: 6 }}>
        <span className="stat-sub">Shortcuts:</span>
        {[2, 3].map((k) => (
          <button
            key={k}
            type="button"
            className="btn btn-sm"
            disabled={eligible.length < k}
            onClick={() => setChosen(largestHunchRegions(obs, hunches, k))}
          >
            {k} largest
          </button>
        ))}
      </div>
      <ul className="list-plain blend-list">
        {obs.regions.map((r) => {
          const arch = obs.archetype_library.find((a) => a.id === hunches[r.id]);
          const on = picked.includes(r.id);
          const id = `blend-${r.id}`;
          const weight = on && totalVoters > 0 ? (100 * r.voters) / totalVoters : null;
          return (
            <li key={r.id} className={`blend-row ${arch ? "" : "is-disabled"}`}>
              <input
                id={id}
                type="checkbox"
                checked={on}
                disabled={!arch || (!on && picked.length >= BLEND_MAX)}
                onChange={() => toggle(r.id)}
                aria-describedby={`${id}-sub`}
              />
              <label htmlFor={id} className="blend-label">
                <RegionPip id={r.id} size={20} />
                <span>
                  <b>{REGION_META[r.id as RegionId]?.label ?? r.id}</b>
                  <small id={`${id}-sub`} className="stat-sub">
                    {fmtInt(r.voters)} voters · {arch ? `hunch: ${arch.name}` : "no hunch yet"}
                    {weight !== null ? ` · weight ${weight.toFixed(0)}%` : ""}
                  </small>
                </span>
              </label>
            </li>
          );
        })}
      </ul>
      {blend ? <StackBar mean={blend} label="Blend preview" showPct /> : null}
      <div className="row-between" style={{ marginTop: 8 }}>
        <span className={`stat-sub ${tooFew ? "bad" : ""}`} role="status">
          {tooFew ? `Choose at least ${BLEND_MIN} regions with a hunch.` : `${picked.length} regions selected.`}
        </span>
        <button type="button" className="btn btn-sm btn-primary" disabled={!blend} onClick={() => blend && onApply(blend, picked)}>
          <Wand2 size={14} aria-hidden /> Apply blend
        </button>
      </div>
    </fieldset>
  );
}
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
  const [blendOpen, setBlendOpen] = useState(false);
  const canBlend = regionsWithHunch(obs, hunches).length >= BLEND_MIN;

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
      <Panel title="Budget manifesto" className="manifesto-budget" hint={<span className="sum-pill num">Σ {sum}%</span>}>
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
            disabled={!canBlend}
            aria-expanded={blendOpen && canBlend}
            aria-controls="blend-picker"
            title={canBlend ? BLEND_HELP : `Set a hunch for at least ${BLEND_MIN} regions first`}
            onClick={() => setBlendOpen((o) => !o)}
          >
            <Wand2 size={14} aria-hidden /> Blend my hunches
            {blendOpen && canBlend ? <ChevronUp size={14} aria-hidden /> : <ChevronDown size={14} aria-hidden />}
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={!top3}
            onClick={() => {
              if (!top3) return;
              // Exaggerate differences above the 60% "even" line so the popular sectors stand out.
              setValues(toIntegerSimplex(SECTOR_IDS.map((s) => Math.max(1, (top3[s] ?? 60) - 30))));
              pushToast({ kind: "info", title: "Drafted from the national top 3", message: "This ignores differences between regions." });
            }}
          >
            <BarChart3 size={14} aria-hidden /> Draft from top-3 stats
          </button>
        </div>
        {blendOpen && canBlend ? (
          <div id="blend-picker" style={{ marginTop: 12 }}>
            <BlendPicker
              obs={obs}
              hunches={hunches}
              onApply={(b, regions) => {
                setValues(toIntegerSimplex(b));
                setBlendOpen(false);
                pushToast({
                  kind: "info",
                  title: "Blended your hunches",
                  message: `Voter-weighted blend of ${regions.map((r) => REGION_META[r as RegionId]?.label ?? r).join(", ")}.`,
                });
              }}
            />
          </div>
        ) : null}
        <p className="stat-sub" style={{ marginTop: 12 }}>
          The shares always add up to 100%: moving one slider adjusts the unlocked ones. Voters back you when your split is close
          to their own, and opinions drift toward their region&apos;s average, so aim at regions.
        </p>
      </Panel>

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
          <span className="stat-label">Match with each archetype</span>
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
          <span className="stat-sub">Green means a typical voter would lean your way (above about 82.5%).</span>
        </div>
      </Panel>

      <Panel title="Responses" className="manifesto-responses">
        <KnowledgeLedger row />
      </Panel>

      <button
        type="button"
        className="btn btn-primary btn-lg manifesto-commit"
        disabled={!obs.legal_actions.includes("commit") || sum !== 100 || busy !== null}
        onClick={() => setConfirming(true)}
      >
        <Stamp size={18} aria-hidden /> Commit manifesto
      </button>

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
              You can&apos;t change it later. Round 1 starts right away.
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
