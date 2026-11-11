"use client";

import { useState } from "react";
import { BookOpen, Dices, Loader2, Play } from "lucide-react";
import { useGame } from "@/lib/game";
import {
  BUDGET_DEFAULT,
  BUDGET_MAX,
  BUDGET_MIN,
  BUDGET_PRESETS,
  N_DEFAULT,
  N_MAX,
  N_MIN,
  ROUNDS_DEFAULT,
  ROUNDS_MAX,
  ROUNDS_MIN,
} from "@/lib/constants";
import { formatMoney } from "@/lib/money";
import { fmtCompact, fmtInt } from "@/lib/format";
import { Modal, Stepper } from "./ui";
import { HowToPlay } from "./HowToPlay";

const LOG_MIN = Math.log10(N_MIN);
const LOG_MAX = Math.log10(N_MAX);
const SLIDER_STEPS = 600;

/** Round to a "nice" number with 2 significant digits (1000, 1100 … 9900, 10000, 11000 …), kept in [lo, hi]. */
function nice2(n: number, lo: number, hi: number): number {
  const clamped = Math.min(hi, Math.max(lo, n));
  const mag = Math.pow(10, Math.floor(Math.log10(clamped)) - 1);
  return Math.min(hi, Math.max(lo, Math.round(clamped / mag) * mag));
}
export const niceVoters = (n: number) => nice2(n, N_MIN, N_MAX);
export const sliderToVoters = (s: number) => niceVoters(Math.pow(10, LOG_MIN + ((LOG_MAX - LOG_MIN) * s) / SLIDER_STEPS));
export const votersToSlider = (n: number) => Math.round(((Math.log10(n) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * SLIDER_STEPS);

// The money slider is logarithmic too: Đ500 → Đ1M spans three orders of magnitude.
const B_LOG_MIN = Math.log10(BUDGET_MIN);
const B_LOG_MAX = Math.log10(BUDGET_MAX);
export const sliderToBudget = (s: number) => nice2(Math.pow(10, B_LOG_MIN + ((B_LOG_MAX - B_LOG_MIN) * s) / SLIDER_STEPS), BUDGET_MIN, BUDGET_MAX);
export const budgetToSlider = (b: number) => Math.round(((Math.log10(b) - B_LOG_MIN) / (B_LOG_MAX - B_LOG_MIN)) * SLIDER_STEPS);

const PRESETS = [1_000, 10_000, 100_000, 1_000_000];

/** Starting treasury (R17): whole credits clamped to BUDGET_MIN–BUDGET_MAX; non-numbers fall back to the default. */
export function clampBudget(x: number): number {
  if (!Number.isFinite(x)) return BUDGET_DEFAULT;
  return Math.min(BUDGET_MAX, Math.max(BUDGET_MIN, Math.round(x)));
}
/** Is this typed text a valid starting treasury (whole credits within BUDGET_MIN–BUDGET_MAX)? */
export function budgetTextValid(text: string): boolean {
  if (!/^\d{1,7}$/.test(text.trim())) return false;
  const n = Number(text.trim());
  return n >= BUDGET_MIN && n <= BUDGET_MAX;
}
/** The wire form of the budget: a 2-dp decimal string ("750.00"). */
export const budgetToRequest = (credits: number): string => clampBudget(credits).toFixed(2);

export function SetupScreen({ howToPlay }: { howToPlay?: string }) {
  const { createGame, busy } = useGame();
  const [showHowTo, setShowHowTo] = useState(false);
  const [voters, setVoters] = useState(N_DEFAULT);
  const [votersText, setVotersText] = useState(String(N_DEFAULT));
  const [seedText, setSeedText] = useState("");
  const [rounds, setRounds] = useState(ROUNDS_DEFAULT);
  const [budget, setBudget] = useState(BUDGET_DEFAULT);
  const [budgetText, setBudgetText] = useState(String(BUDGET_DEFAULT));
  const budgetOk = budgetTextValid(budgetText);
  const seedValid = seedText.trim() === "" || /^\d{1,10}$/.test(seedText.trim());
  const creating = busy === "create";

  const commitVoters = (n: number) => {
    const v = Math.min(N_MAX, Math.max(N_MIN, Math.round(n)));
    setVoters(v);
    setVotersText(String(v));
  };

  const commitBudget = (n: number) => {
    const b = clampBudget(n);
    setBudget(b);
    setBudgetText(String(b));
  };

  const start = async () => {
    if (!seedValid || !budgetOk) return;
    const seed = seedText.trim() === "" ? undefined : Number(seedText.trim());
    await createGame({ num_voters: voters, rounds, budget: budgetToRequest(budget), ...(seed === undefined ? {} : { seed }) });
  };

  return (
    <div className="title-screen">
      <header className="title-head">
        <h1 className="hero-title">CLOUT</h1>
        {/* easter egg: reads as plain text, but it's a link */}
        <a className="hero-tagline" href="https://youtu.be/92cwKCU8Z5c?t=66&autoplay=1" target="_blank" rel="noopener noreferrer">
          The winner takes it all....
        </a>
      </header>

      <form
        className="panel setup-card"
        aria-label="New game setup"
        onSubmit={(e) => {
          e.preventDefault();
          void start();
        }}
      >

        <div className="field">
          <label htmlFor="voters-range">Voters</label>
          <div className="row-between">
            <span className="big-n num" aria-hidden>
              {fmtInt(voters)}
            </span>
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
            <span className="stat-sub">Large electorates take a few seconds to generate.</span>
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
            ? "Leave blank for a random electorate."
            : "Seed must be a whole number."}
        </span>

        <div className="field">
          <label htmlFor="budget-range">Starting money</label>
          <div className="row-between">
            <span className="big-n num" aria-hidden data-testid="summary-budget">
              {formatMoney(budget * 100)}
            </span>
          </div>
          <input
            id="budget-range"
            type="range"
            min={0}
            max={SLIDER_STEPS}
            step={1}
            value={budgetToSlider(budget)}
            aria-valuetext={`${fmtInt(budget)} credits`}
            onChange={(e) => commitBudget(sliderToBudget(Number(e.target.value)))}
          />
          <div className="row-between">
            <div className="seg" role="group" aria-label="Treasury presets">
              {BUDGET_PRESETS.map((p) => (
                <button key={p} type="button" aria-pressed={budget === p && budgetOk} onClick={() => commitBudget(p)}>
                  Đ{fmtCompact(p)}
                </button>
              ))}
            </div>
            <input
              className="input num"
              style={{ maxWidth: 130 }}
              aria-label="Starting treasury in credits"
              aria-invalid={!budgetOk}
              aria-describedby={budgetOk ? undefined : "budget-help"}
              inputMode="numeric"
              value={budgetText}
              onChange={(e) => {
                const t = e.target.value.replace(/[^\d]/g, "");
                setBudgetText(t);
                if (budgetTextValid(t)) setBudget(Number(t));
              }}
              onBlur={() => commitBudget(budgetText === "" ? BUDGET_DEFAULT : Number(budgetText))}
            />
          </div>
          {budgetOk ? null : (
            <span id="budget-help" className="stat-sub bad">
              Enter a whole amount from Đ{BUDGET_MIN.toLocaleString("en-US")} to Đ{BUDGET_MAX.toLocaleString("en-US")}.
            </span>
          )}
        </div>

        <button type="submit" className="btn btn-primary btn-lg" disabled={creating || !seedValid || !budgetOk}>
          {creating ? <Loader2 className="spin" size={18} aria-hidden /> : <Play size={18} aria-hidden />}
          {creating ? "Starting…" : "Start campaign"}
        </button>
      </form>

      {howToPlay ? (
        <button type="button" className="btn btn-ghost hero-howto" onClick={() => setShowHowTo(true)}>
          <BookOpen size={16} aria-hidden /> How to CLOUT?
        </button>
      ) : null}
      {showHowTo && howToPlay ? (
        <Modal title="How to CLOUT?" onClose={() => setShowHowTo(false)} className="howto-modal">
          <HowToPlay markdown={howToPlay} />
        </Modal>
      ) : null}
    </div>
  );
}
