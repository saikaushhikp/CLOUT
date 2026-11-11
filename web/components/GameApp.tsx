"use client";

import { Loader2, SquarePlus } from "lucide-react";
import { GameProvider, useGame } from "@/lib/game";
import type { DumbocracyClient } from "@/lib/api";
import type { Observation } from "@/lib/types";
import { formatMoney } from "@/lib/money";
import { fmtCompact } from "@/lib/format";
import { ThemeToggle, Toasts } from "./ui";
import { SetupScreen } from "./SetupScreen";
import { SurveyScreen } from "./SurveyScreen";
import { CampaignScreen } from "./CampaignScreen";
import { ResultsScreen } from "./ResultsScreen";

export function GameApp({
  client,
  initialObservation,
  howToPlay,
}: {
  client?: DumbocracyClient;
  initialObservation?: Observation;
  /** HOW_TO_PLAY.md, shown on the setup screen */
  howToPlay?: string;
}) {
  return (
    <GameProvider client={client} initialObservation={initialObservation}>
      <Shell howToPlay={howToPlay} />
    </GameProvider>
  );
}

function PhaseTrack({ obs }: { obs: Observation }) {
  const { reviewRound, setReviewRound } = useGame();
  const steps: { key: string; label: string; state: "done" | "current" | "todo"; round?: number }[] = [];
  const surveyState = obs.phase === "SURVEY" ? "current" : "done";
  steps.push({ key: "survey", label: "Survey & manifesto", state: surveyState });
  for (let t = 1; t <= obs.rounds; t++) {
    const state = obs.phase === "DONE" ? "done" : obs.phase === "ROUND" ? (t < obs.round ? "done" : t === obs.round ? "current" : "todo") : "todo";
    steps.push({ key: `r${t}`, label: `R${t}`, state, round: t });
  }
  steps.push({ key: "election", label: "Election", state: obs.phase === "DONE" ? "current" : "todo" });
  return (
    <ol className="phase-track list-plain" aria-label="Game progress">
      {steps.map((s) => {
        // While campaigning, past rounds open the read-only look-back (R19); the current round returns to planning.
        const clickable = obs.phase === "ROUND" && s.round !== undefined && s.state !== "todo";
        return (
          <li key={s.key} className={`phase-step is-${s.state} ${reviewRound === s.round ? "is-reviewing" : ""}`} aria-current={s.state === "current" ? "step" : undefined}>
            {clickable ? (
              <button
                type="button"
                className="phase-step-btn"
                aria-label={s.state === "current" ? `Round ${s.round} — back to planning` : `Review round ${s.round}`}
                onClick={() => setReviewRound(s.state === "current" || reviewRound === s.round ? null : (s.round as number))}
              >
                <span className="phase-dot" aria-hidden />
                {s.label}
              </button>
            ) : (
              <>
                <span className="phase-dot" aria-hidden />
                {s.label}
              </>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/** Engine health, shown in the top bar before a game starts; the device (e.g. cuda:2) is in the tooltip. */
function EngineStatus() {
  const { health, healthError } = useGame();
  return (
    <span className="engine-status" role="status" title={health ? `Running on ${health.device}` : healthError ?? undefined}>
      <span className={`engine-dot ${health ? "ok" : healthError ? "down" : ""}`} aria-hidden />
      {health ? "Engine online" : healthError ? "Engine offline" : "Checking engine…"}
      {health ? <span className="sr-only">, running on {health.device}</span> : null}
    </span>
  );
}

function Shell({ howToPlay }: { howToPlay?: string }) {
  const { obs, busy, leaveGame, client } = useGame();
  return (
    <div className="shell">
      <header className="topbar">
        {/* the setup screen shows the big title itself, so the top bar keeps only the engine state */}
        <div className={`brand${obs ? "" : " brand-hidden"}`}>
          <span className="brand-text">
            Clout
            {client.kind === "mock" ? <small>Mock engine</small> : null}
          </span>
        </div>
        {obs ? <PhaseTrack obs={obs} /> : null}
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          {obs ? (
            <>
              <span className="badge mono" title={`${fmtCompact(obs.num_voters)} voters, seed ${obs.seed}`}>
                Seed {obs.seed}
              </span>
              <div className="treasury-chip num" aria-live="polite">
                <span>Treasury</span>
                <span>{formatMoney(obs.treasury)}</span>
              </div>
              <button
                type="button"
                className="btn btn-ghost btn-icon"
                aria-label="Leave this game and start a new one"
                title="New game"
                onClick={() => {
                  if (obs.phase === "DONE" || window.confirm("Abandon this campaign and start a new game?")) leaveGame();
                }}
              >
                <SquarePlus size={18} aria-hidden />
              </button>
            </>
          ) : (
            <EngineStatus />
          )}
          <ThemeToggle />
        </div>
      </header>
      <main id="main">
        {!obs && busy === "load" ? (
          <div className="loading" role="status">
            <Loader2 className="spin" aria-hidden />
            Reconnecting…
          </div>
        ) : !obs ? (
          <SetupScreen howToPlay={howToPlay} />
        ) : obs.phase === "SURVEY" ? (
          <SurveyScreen />
        ) : obs.phase === "ROUND" ? (
          <CampaignScreen />
        ) : (
          <ResultsScreen />
        )}
      </main>
      <Toasts />
    </div>
  );
}
