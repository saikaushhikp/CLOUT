"use client";

import { Timer } from "lucide-react";
import { useCountdown } from "@/lib/useCountdown";
import { Switch } from "./ui";

export const TIMER_CHOICES = [30, 60, 90, 120] as const;

export function RoundTimer({
  enabled,
  seconds,
  round,
  paused,
  onToggle,
  onSeconds,
  onExpire,
}: {
  enabled: boolean;
  seconds: number;
  round: number;
  paused: boolean;
  onToggle: (v: boolean) => void;
  onSeconds: (s: number) => void;
  onExpire: () => void;
}) {
  const { remaining, fraction } = useCountdown({ enabled, seconds, resetKey: round, onExpire, paused });
  const r = 19;
  const circ = 2 * Math.PI * r;
  const mm = Math.floor(remaining / 60);
  const ss = String(remaining % 60).padStart(2, "0");
  return (
    <div className={`timer ${enabled && remaining <= 10 ? "urgent" : ""}`}>
      {enabled ? (
        <svg className="timer-ring" viewBox="0 0 46 46" aria-hidden>
          <circle cx="23" cy="23" r={r} fill="none" stroke="var(--surface-3)" strokeWidth="4" />
          <circle
            className="fg"
            cx="23"
            cy="23"
            r={r}
            fill="none"
            stroke="var(--accent)"
            strokeWidth="4"
            strokeLinecap="round"
            strokeDasharray={circ}
            strokeDashoffset={circ * (1 - fraction)}
            transform="rotate(-90 23 23)"
            style={{ transition: "stroke-dashoffset 1s linear" }}
          />
          <text x="23" y="27" textAnchor="middle" fontSize="11" fontWeight="800" fill="var(--ink)">
            {mm}:{ss}
          </text>
        </svg>
      ) : (
        <Timer size={22} aria-hidden className="muted" />
      )}
      <div className="stack" style={{ gap: 4 }}>
        <Switch checked={enabled} onChange={onToggle} label="Round clock" />
        {enabled ? (
          <span className="stat-sub" role="timer" aria-live={remaining <= 10 ? "assertive" : "off"} aria-label={`${remaining} seconds left this round`}>
            {remaining > 0 ? `${remaining}s left — then the round is forfeited` : "Time!"}
          </span>
        ) : (
          <label className="stat-sub">
            <span className="sr-only">Seconds per round</span>
            <select className="select" style={{ minHeight: 28, padding: "2px 28px 2px 8px", fontSize: 12 }} value={seconds} onChange={(e) => onSeconds(Number(e.target.value))}>
              {TIMER_CHOICES.map((s) => (
                <option key={s} value={s}>
                  {s}s per round
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
    </div>
  );
}
