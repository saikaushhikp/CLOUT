"use client";

import { useEffect, useId, useState, type ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, Minus, Moon, Plus, Sun, X, type LucideIcon } from "lucide-react";
import { useGame } from "@/lib/game";
import { REGION_META, SECTOR_META, sectorLabel } from "@/lib/constants";
import type { RegionId, SectorId } from "@/lib/types";
import { SECTOR_IDS } from "@/lib/types";

export function Panel({
  title,
  icon: Icon,
  hint,
  children,
  className = "",
  actions,
  id,
}: {
  title: ReactNode;
  icon?: LucideIcon;
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
  actions?: ReactNode;
  id?: string;
}) {
  const auto = useId();
  const headingId = id ?? auto;
  return (
    <section className={`panel ${className}`} aria-labelledby={headingId}>
      <div className="panel-head">
        {Icon ? <Icon size={18} className="panel-icon" aria-hidden /> : null}
        <h2 id={headingId}>{title}</h2>
        {hint ? <span className="hint">{hint}</span> : null}
        {actions ? <div style={{ marginLeft: hint ? 8 : "auto" }}>{actions}</div> : null}
      </div>
      {children}
    </section>
  );
}

export function Stepper({
  value,
  min = 0,
  max,
  onChange,
  label,
  disabled,
}: {
  value: number;
  min?: number;
  max: number;
  onChange: (v: number) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <div className="stepper" role="group" aria-label={label}>
      <button type="button" aria-label={`Decrease ${label}`} disabled={disabled || value <= min} onClick={() => onChange(value - 1)}>
        <Minus size={15} aria-hidden />
      </button>
      <input
        type="number"
        inputMode="numeric"
        aria-label={label}
        value={value}
        min={min}
        max={max}
        disabled={disabled}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, Math.round(v))));
        }}
      />
      <button type="button" aria-label={`Increase ${label}`} disabled={disabled || value >= max} onClick={() => onChange(value + 1)}>
        <Plus size={15} aria-hidden />
      </button>
    </div>
  );
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <label className="switch">
      <input type="checkbox" role="switch" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch-track" aria-hidden />
      <span>{label}</span>
    </label>
  );
}

export function RegionPip({ id, size = 26 }: { id: string; size?: number }) {
  const meta = REGION_META[id as RegionId];
  return (
    <span className="region-pip" style={{ background: meta?.color ?? "var(--ink-3)", width: size, height: size }} aria-hidden>
      {id}
    </span>
  );
}

export function SectorSwatch({ id }: { id: string }) {
  return <span className="swatch" style={{ background: SECTOR_META[id as SectorId]?.color }} aria-hidden />;
}

/** Five-sector stacked bar (shares in [0,1]); labelled for assistive tech. */
export function StackBar({ mean, label, showPct = false, height }: { mean: number[]; label: string; showPct?: boolean; height?: number }) {
  const total = mean.reduce((a, b) => a + b, 0) || 1;
  const text = SECTOR_IDS.map((s, i) => `${sectorLabel(s)} ${Math.round((mean[i] / total) * 100)}%`).join(", ");
  return (
    <div className="stackbar" role="img" aria-label={`${label}: ${text}`} style={height ? { height } : undefined}>
      {SECTOR_IDS.map((s, i) => (
        <span key={s} style={{ width: `${(mean[i] / total) * 100}%`, background: SECTOR_META[s].color }} title={`${sectorLabel(s)} ${Math.round((mean[i] / total) * 100)}%`}>
          {showPct && mean[i] / total >= 0.09 ? `${Math.round((mean[i] / total) * 100)}` : null}
        </span>
      ))}
    </div>
  );
}

export function SectorLegend() {
  return (
    <div className="legend" aria-hidden>
      {SECTOR_IDS.map((s) => (
        <span key={s}>
          <SectorSwatch id={s} />
          {sectorLabel(s)}
        </span>
      ))}
    </div>
  );
}

export function Stat({ label, value, sub }: { label: ReactNode; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className="stat-value num">{value}</span>
      {sub ? <span className="stat-sub">{sub}</span> : null}
    </div>
  );
}

export function Toasts() {
  const { toasts, dismissToast } = useGame();
  const icon = { error: AlertTriangle, info: Info, success: CheckCircle2 } as const;
  return (
    <div className="toasts" aria-live="polite" aria-relevant="additions">
      {toasts.map((t) => {
        const Icon = icon[t.kind];
        return (
          <div key={t.id} className={`toast toast-${t.kind}`} role={t.kind === "error" ? "alert" : "status"}>
            <Icon size={18} aria-hidden style={{ flex: "none", marginTop: 2 }} />
            <div>
              <strong>{t.title}</strong>
              {t.message ? <p>{t.message}</p> : null}
            </div>
            <button className="btn btn-ghost btn-icon btn-sm" aria-label="Dismiss notification" onClick={() => dismissToast(t.id)}>
              <X size={15} aria-hidden />
            </button>
          </div>
        );
      })}
    </div>
  );
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);
  useEffect(() => {
    const t = document.documentElement.dataset.theme;
    if (t === "light" || t === "dark") setTheme(t);
    else setTheme(window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }, []);
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="btn btn-ghost btn-icon"
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      onClick={() => {
        document.documentElement.dataset.theme = next;
        try {
          localStorage.setItem("dumbocracy.theme", next);
        } catch {
          /* ignore */
        }
        setTheme(next);
      }}
    >
      {theme === "dark" ? <Sun size={18} aria-hidden /> : <Moon size={18} aria-hidden />}
    </button>
  );
}
