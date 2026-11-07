"use client";

import { useState } from "react";
import { Dices, Loader2, Play, Users } from "lucide-react";
import { useGame } from "@/lib/game";
import { N_DEFAULT, N_MAX, N_MIN, ROUNDS_DEFAULT, ROUNDS_MAX, ROUNDS_MIN } from "@/lib/constants";
import { fmtCompact, fmtInt } from "@/lib/format";
import { Stepper } from "./ui";

const LOG_MIN = Math.log10(N_MIN);
const LOG_MAX = Math.log10(N_MAX);
const SLIDER_STEPS = 600;

/** Round to a "nice" number with 2 significant digits (1000, 1100 … 9900, 10000, 11000 …). */
export function niceVoters(n: number): number {
  const clamped = Math.min(N_MAX, Math.max(N_MIN, n));
  const mag = Math.pow(10, Math.floor(Math.log10(clamped)) - 1);
  return Math.min(N_MAX, Math.max(N_MIN, Math.round(clamped / mag) * mag));
}
export const sliderToVoters = (s: number) => niceVoters(Math.pow(10, LOG_MIN + ((LOG_MAX - LOG_MIN) * s) / SLIDER_STEPS));
export const votersToSlider = (n: number) => Math.round(((Math.log10(n) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * SLIDER_STEPS);

const PRESETS = [1_000, 10_000, 100_000, 1_000_000];

export function SetupScreen() {
  const { createGame, busy, health, healthError, client } = useGame();
  const [voters, setVoters] = useState(N_DEFAULT);
  const [votersText, setVotersText] = useState(String(N_DEFAULT));
  const [seedText, setSeedText] = useState("");
  const [rounds, setRounds] = useState(ROUNDS_DEFAULT);
  const seedValid = seedText.trim() === "" || /^\d{1,10}$/.test(seedText.trim());
  const creating = busy === "create";

  const commitVoters = (n: number) => {
    const v = Math.min(N_MAX, Math.max(N_MIN, Math.round(n)));
    setVoters(v);
    setVotersText(String(v));
  };

  const start = async () => {
    if (!seedValid) return;
    const seed = seedText.trim() === "" ? undefined : Number(seedText.trim());
    await createGame({ num_voters: voters, rounds, ...(seed === undefined ? {} : { seed }) });
  };

  return (
    <div className="hero">
      <div className="hero-copy">
        <span className="kicker">Election night is {rounds} rounds away</span>
        <h1>
          Win the vote.
          <br />
          <em>Know</em> your voters.
        </h1>
        <p className="lede">
          Four regions. Six possible archetypes. Hidden personalities. A treasury of Đ500 and a social network that
          gossips. Survey the electorate, commit a budget manifesto, then deploy party workers and media to pull voters
          your way before the ballots open.
        </p>
        <ol className="howto list-plain">
          <li>
            <span className="n">1</span>
            <span>
              <b>Survey</b>Free top-3 stats + paid head-to-head questions (Đ5) per region.
            </span>
          </li>
          <li>
            <span className="n">2</span>
            <span>
              <b>Commit</b>Split the budget across five sectors. No take-backs.
            </span>
          </li>
          <li>
            <span className="n">3</span>
            <span>
              <b>Campaign</b>Each round: workers, extra hires and regional media airtime.
            </span>
          </li>
          <li>
            <span className="n">4</span>
            <span>
              <b>Debrief</b>See the hidden truth, the oracle&apos;s manifesto and how opinion spread.
            </span>
          </li>
        </ol>
      </div>

      <form
        className="panel setup-card"
        aria-label="New game setup"
        onSubmit={(e) => {
          e.preventDefault();
          void start();
        }}
      >
        <div className="row-between">
          <span className="kicker">New campaign</span>
          <span className="engine-status" role="status">
            <span className={`engine-dot ${health ? "ok" : healthError ? "down" : ""}`} aria-hidden />
            {health
              ? `Engine online · ${health.device}`
              : healthError
                ? "Engine offline"
                : "Checking engine…"}
          </span>
        </div>

        <div className="field">
          <label htmlFor="voters-range">
            <Users size={14} aria-hidden style={{ verticalAlign: "-2px", marginRight: 6 }} />
            Electorate size
          </label>
          <div className="row-between">
            <span className="big-n num" aria-hidden>
              {fmtInt(voters)}
            </span>
            <span className="muted">voters</span>
          </div>
          <input
            id="voters-range"
            type="range"
            min={0}
            max={SLIDER_STEPS}
            step={1}
            value={votersToSlider(voters)}
            aria-valuetext={`${fmtInt(voters)} voters`}
            onChange={(e) => commitVoters(sliderToVoters(Number(e.target.value)))}
          />
          <div className="row-between">
            <div className="seg" role="group" aria-label="Electorate presets">
              {PRESETS.map((p) => (
                <button key={p} type="button" aria-pressed={voters === p} onClick={() => commitVoters(p)}>
                  {fmtCompact(p)}
                </button>
              ))}
            </div>
            <input
              className="input num"
              style={{ maxWidth: 130 }}
              aria-label="Exact number of voters"
              inputMode="numeric"
              value={votersText}
              onChange={(e) => setVotersText(e.target.value.replace(/[^\d]/g, ""))}
              onBlur={() => commitVoters(Number(votersText) || N_DEFAULT)}
            />
          </div>
          {voters >= 300_000 ? (
            <span className="stat-sub">Big electorate — generation takes a few seconds on the engine.</span>
          ) : null}
        </div>

        <div className="row" style={{ alignItems: "flex-end" }}>
          <div className="field" style={{ flex: 1 }}>
            <label htmlFor="seed">World seed</label>
            <input
              id="seed"
              className="input mono"
              placeholder="random"
              inputMode="numeric"
              value={seedText}
              aria-invalid={!seedValid}
              aria-describedby="seed-help"
              onChange={(e) => setSeedText(e.target.value.trim())}
            />
          </div>
          <button
            type="button"
            className="btn btn-icon"
            style={{ width: 40, height: 40 }}
            aria-label="Roll a random seed"
            title="Roll a random seed"
            onClick={() => setSeedText(String(Math.floor(Math.random() * 1_000_000_000)))}
          >
            <Dices size={18} aria-hidden />
          </button>
          <div className="field">
            <span className="field-label" id="rounds-label">
              Rounds
            </span>
            <Stepper label="Campaign rounds" value={rounds} min={ROUNDS_MIN} max={ROUNDS_MAX} onChange={setRounds} />
          </div>
        </div>
        <span id="seed-help" className="stat-sub">
          {seedValid
            ? "Same voters + same seed = the same electorate. Leave blank for a surprise."
            : "Seed must be a whole number."}
        </span>

        <button type="submit" className="btn btn-primary btn-lg" disabled={creating || !seedValid}>
          {creating ? <Loader2 className="spin" size={18} aria-hidden /> : <Play size={18} aria-hidden />}
          {creating ? "Drawing the electorate…" : "Start campaign"}
        </button>
        {client.kind === "mock" ? (
          <span className="stat-sub">Mock mode: a simplified in-browser engine stands in for the Python server.</span>
        ) : null}
      </form>
    </div>
  );
}
