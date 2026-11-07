"use client";

import { Loader2, LogOut, Vote } from "lucide-react";
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

export function GameApp({ client, initialObservation }: { client?: DumbocracyClient; initialObservation?: Observation }) {
  return (
    <GameProvider client={client} initialObservation={initialObservation}>
      <Shell />
    </GameProvider>
  );
}

function PhaseTrack({ obs }: { obs: Observation }) {
  const steps: { key: string; label: string; state: "done" | "current" | "todo" }[] = [];
  const surveyState = obs.phase === "SURVEY" ? "current" : "done";
  steps.push({ key: "survey", label: "Survey & manifesto", state: surveyState });
  for (let t = 1; t <= obs.rounds; t++) {
    const state = obs.phase === "DONE" ? "done" : obs.phase === "ROUND" ? (t < obs.round ? "done" : t === obs.round ? "current" : "todo") : "todo";
    steps.push({ key: `r${t}`, label: `R${t}`, state });
  }
  steps.push({ key: "election", label: "Election", state: obs.phase === "DONE" ? "current" : "todo" });
  return (
    <ol className="phase-track list-plain" aria-label="Game progress">
      {steps.map((s) => (
        <li key={s.key} className={`phase-step is-${s.state}`} aria-current={s.state === "current" ? "step" : undefined}>
          <span className="phase-dot" aria-hidden />
          {s.label}
        </li>
      ))}
    </ol>
  );
}

function Shell() {
  const { obs, busy, leaveGame, client } = useGame();
  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>
            <Vote size={18} />
          </span>
          <span className="brand-text">
            Dumbocracy
            <small>Campaign war room{client.kind === "mock" ? " · mock engine" : ""}</small>
          </span>
        </div>
        {obs ? <PhaseTrack obs={obs} /> : null}
        <div className="topbar-spacer" />
        <div className="topbar-actions">
          {obs ? (
            <>
              <span className="badge mono" title={`Seed ${obs.seed}`}>
                N {fmtCompact(obs.num_voters)} · #{obs.seed}
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
                <LogOut size={18} aria-hidden />
              </button>
            </>
          ) : null}
          <ThemeToggle />
        </div>
      </header>
      <main id="main">
        {!obs && busy === "load" ? (
          <div className="loading" role="status">
            <Loader2 className="spin" aria-hidden />
            Reconnecting to your campaign…
          </div>
        ) : !obs ? (
          <SetupScreen />
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
