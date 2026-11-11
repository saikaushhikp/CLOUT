"use client";

import { useEffect, useMemo, useState } from "react";
import { Check, Loader2, RotateCcw, X } from "lucide-react";
import { useGame, useObservation } from "@/lib/game";
import { REGION_META } from "@/lib/constants";
import { fmtInt, fmtShare } from "@/lib/format";
import { centsToString, formatMoney, toCents } from "@/lib/money";
import { initialBudgetCents } from "@/lib/rounds";
import { toIntegerSimplex } from "@/lib/simplex";
import type { Debrief, EpistemicTimeline, Observation, RegionId } from "@/lib/types";
import { EpistemicPanel } from "./EpistemicTimeline";
import { PollChart } from "./PollChart";
import { NetworkView } from "./NetworkView";
import { Panel, SectorLegend, StackBar } from "./ui";

/** The debrief may name archetypes/personalities by id ("A2") or by name; resolve both. */
function resolveLib<T extends { id: string; name: string }>(lib: T[], key: string): T | undefined {
  return lib.find((x) => x.id === key) ?? lib.find((x) => x.name.toLowerCase() === key.toLowerCase());
}

function Confetti() {
  const pieces = useMemo(
    () =>
      Array.from({ length: 36 }, (_, i) => ({
        left: (i * 37) % 100,
        delay: (i % 12) * 0.22,
        color: ["#fff", "#f2b84b", "#86b6ef", "#1baf7a", "#e87ba4"][i % 5],
        dur: 2.6 + (i % 5) * 0.35,
      })),
    [],
  );
  return (
    <div className="confetti" aria-hidden>
      {pieces.map((p, i) => (
        <i key={i} style={{ left: `${p.left}%`, background: p.color, animationDelay: `${p.delay}s`, animationDuration: `${p.dur}s` }} />
      ))}
    </div>
  );
}

function Verdict({ obs }: { obs: Observation }) {
  const r = obs.result;
  if (!r) return null;
  const margin = r.votes - r.num_voters / 2;
  // R17: the starting treasury is the player's choice (₫500–₫1000), so spend = initial_budget − treasury.
  const budget = initialBudgetCents(obs);
  const spent = budget - toCents(obs.treasury);
  return (
    <section className={`verdict ${r.victory ? "win" : ""}`} aria-live="polite">
      {r.victory ? <Confetti /> : null}
      <h1>{r.victory ? "You won." : "You lost."}</h1>
      <p style={{ margin: 0, fontSize: 17, opacity: 0.92 }}>
        {r.victory
          ? `A majority of ${fmtInt(Math.abs(Math.ceil(margin)))} votes over the 50% line.`
          : `${fmtInt(Math.abs(Math.floor(margin)) + 1)} votes short of a majority.`}
      </p>
      <div className="vote-bar" role="img" aria-label={`Vote share ${fmtShare(r.vote_share)}; 50% needed`}>
        <span style={{ width: `${Math.min(100, r.vote_share * 100)}%` }} />
      </div>
      <div className="verdict-stats">
        <div className="stat">
          <span className="stat-label">Votes</span>
          <span className="stat-value num">{fmtInt(r.votes)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Vote share</span>
          <span className="stat-value num">{fmtShare(r.vote_share)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Expected share</span>
          <span className="stat-value num">{fmtShare(r.expected_share)}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Money spent</span>
          <span className="stat-value num">{formatMoney(spent)}</span>
          <span className="stat-sub" style={{ color: "inherit", opacity: 0.85 }}>
            {spent >= budget ? "the whole budget" : `of ${formatMoney(budget)}`}
          </span>
        </div>
      </div>
    </section>
  );
}

function RevealCards({ obs, debrief }: { obs: Observation; debrief: Debrief }) {
  const { hunches } = useGame();
  const [flipped, setFlipped] = useState(0);
  useEffect(() => {
    if (flipped >= debrief.regions.length) return;
    const id = setTimeout(() => setFlipped((f) => f + 1), flipped === 0 ? 500 : 450);
    return () => clearTimeout(id);
  }, [flipped, debrief.regions.length]);
  return (
    <div className="reveal-grid">
      {debrief.regions.map((r, i) => {
        const arch = resolveLib(obs.archetype_library, r.archetype);
        const pers = resolveLib(obs.personality_library, r.personality);
        const region = obs.regions.find((x) => x.id === r.id);
        const hunch = hunches[r.id];
        const guessed = hunch ? hunch === arch?.id : null;
        const zealotText = r.zealots <= 1 && r.zealots > 0 && !Number.isInteger(r.zealots) ? fmtShare(r.zealots) : fmtInt(r.zealots);
        const delta = r.final_expected_share - r.initial_expected_share;
        return (
          <article
            key={r.id}
            className="reveal-card"
            aria-label={`${REGION_META[r.id as RegionId]?.label ?? r.id}: ${arch?.name ?? r.archetype}, ${pers?.name ?? r.personality}`}
          >
            {i >= flipped ? (
              <div className="reveal-face" style={{ ["--rc" as string]: REGION_META[r.id as RegionId]?.color }}>
                <strong>{REGION_META[r.id as RegionId]?.label ?? r.id}</strong>
                <span className="stat-sub">Hidden</span>
              </div>
            ) : (
              <div className="reveal-face reveal-back" style={{ ["--rc" as string]: REGION_META[r.id as RegionId]?.color }}>
                <div className="row-between">
                  <strong>{REGION_META[r.id as RegionId]?.label ?? r.id}</strong>
                  {guessed === null ? null : guessed ? (
                    <span className="badge" style={{ color: "var(--good-text)" }}>
                      <Check size={12} aria-hidden /> hunch right
                    </span>
                  ) : (
                    <span className="badge" style={{ color: "var(--bad)" }}>
                      <X size={12} aria-hidden /> hunch wrong
                    </span>
                  )}
                </div>
                <div>
                  <div className="stat-label">Archetype</div>
                  <div style={{ fontWeight: 800 }}>{arch?.name ?? r.archetype}</div>
                </div>
                {arch ? <StackBar mean={arch.mean} label={arch.name} /> : null}
                <div>
                  <div className="stat-label">Personality</div>
                  <div style={{ fontWeight: 700 }}>{pers?.name ?? r.personality}</div>
                </div>
                <div className="stat-sub num">
                  {fmtInt(region?.voters ?? 0)} voters · {zealotText} zealots
                </div>
                <div className="num" style={{ fontSize: 13 }}>
                  Support {fmtShare(r.initial_expected_share)} → <b>{fmtShare(r.final_expected_share)}</b>{" "}
                  <span className={delta >= 0 ? "good" : "bad"}>
                    ({delta >= 0 ? "+" : ""}
                    {(delta * 100).toFixed(1)})
                  </span>
                </div>
              </div>
            )}
          </article>
        );
      })}
    </div>
  );
}

/** Your manifesto next to the oracle's; collapses to one bar when they round to the same percentages. */
function ManifestoCompare({ manifesto, debrief }: { manifesto: number[] | null; debrief: Debrief }) {
  const mine = manifesto ? toIntegerSimplex(manifesto) : null;
  const oracle = toIntegerSimplex(debrief.oracle.manifesto);
  const same = mine !== null && mine.every((v, i) => v === oracle[i]);
  return (
    <div className="compare-bars">
      {same ? (
        <>
          <p style={{ margin: 0 }}>
            <b>Your manifesto matched the oracle&apos;s.</b>
          </p>
          <StackBar mean={debrief.oracle.manifesto} label="Your manifesto and the oracle's" showPct />
        </>
      ) : (
        <>
          <div className="compare-row">
            <span>You</span>
            {manifesto ? <StackBar mean={manifesto} label="Your manifesto" showPct /> : null}
          </div>
          <div className="compare-row">
            <span>Oracle</span>
            <StackBar mean={debrief.oracle.manifesto} label="Oracle manifesto" showPct />
          </div>
        </>
      )}
      <SectorLegend />
      {same ? (
        <div className="stat" style={{ marginTop: 6 }}>
          <span className="stat-label">Vote share with no campaign</span>
          <span className="stat-value num">{fmtShare(debrief.your_manifesto_no_campaign_share)}</span>
        </div>
      ) : (
        <div className="report-strip" style={{ gridTemplateColumns: "repeat(2, minmax(0,1fr))", marginTop: 6 }}>
          <div className="stat">
            <span className="stat-label">You, no campaign</span>
            <span className="stat-value num">{fmtShare(debrief.your_manifesto_no_campaign_share)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Oracle, no campaign</span>
            <span className="stat-value num">{fmtShare(debrief.oracle.expected_share_no_campaign)}</span>
          </div>
        </div>
      )}
      <span className="stat-sub">{debrief.oracle.note}.{same ? "" : " Both shares assume no campaign."}</span>
    </div>
  );
}

function lessons(obs: Observation, d: Debrief): string[] {
  const out: string[] = [];
  const traj = d.share_trajectory;
  const start = traj[0] ?? 0;
  const end = traj[traj.length - 1] ?? 0;
  const gapManifesto = d.oracle.expected_share_no_campaign - d.your_manifesto_no_campaign_share;
  if (gapManifesto > 0.02)
    out.push(
      `With no campaign, your manifesto would have drawn ${fmtShare(d.your_manifesto_no_campaign_share)} and the oracle's ${fmtShare(d.oracle.expected_share_no_campaign)}. Your manifesto cost you ${(gapManifesto * 100).toFixed(1)} points.`,
    );
  else if (gapManifesto < 0.0005) out.push("Your manifesto matched the oracle's.");
  else out.push(`Your manifesto was within ${(gapManifesto * 100).toFixed(1)} points of the oracle's.`);
  const lift = end - d.your_manifesto_no_campaign_share;
  out.push(
    lift > 0.005
      ? `Campaigning added about ${(lift * 100).toFixed(1)} points (${fmtShare(d.your_manifesto_no_campaign_share)} → ${fmtShare(end)}).`
      : `Campaigning made little difference (${fmtShare(d.your_manifesto_no_campaign_share)} without it, ${fmtShare(end)} with it).`,
  );
  const best = [...d.regions].sort((a, b) => b.final_expected_share - a.final_expected_share)[0];
  const worst = [...d.regions].sort((a, b) => a.final_expected_share - b.final_expected_share)[0];
  if (best && worst && best.id !== worst.id)
    out.push(
      `Best region: ${REGION_META[best.id as RegionId]?.label ?? best.id} (${fmtShare(best.final_expected_share)}). Worst: ${REGION_META[worst.id as RegionId]?.label ?? worst.id} (${fmtShare(worst.final_expected_share)}).`,
    );
  const pre = obs.polls.find((p) => p.round === obs.rounds + 1);
  if (pre)
    out.push(
      `The last poll said ${fmtShare(pre.estimate)} ± ${fmtShare(pre.moe)}; the true expected share was ${fmtShare(end)}.`,
    );
  if (start < 0.5 && end > 0.5) out.push("You started below 50% and finished above it.");
  return out;
}

/** Loads the P5 epistemic timeline lazily, once the debrief is on screen (the engine may need ~10 s). */
function useEpistemic(gameId: string, playerId: string, ready: boolean) {
  const { client } = useGame();
  const [timeline, setTimeline] = useState<EpistemicTimeline | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    setTimeline(null);
    setError(null);
    if (!ready) return;
    let alive = true;
    client
      .epistemic(gameId, playerId)
      .then((t) => alive && setTimeline(t))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [client, gameId, playerId, ready, attempt]);
  return { timeline, error, retry: () => setAttempt((a) => a + 1) };
}

export function ResultsScreen() {
  const obs = useObservation();
  const { debrief, graph, graphError, createGame, leaveGame, busy } = useGame();
  const epistemic = useEpistemic(obs.game_id, obs.player_id, debrief !== null);
  const roundsLabels = useMemo(() => ["Start", ...Array.from({ length: obs.rounds }, (_, k) => `After R${k + 1}`)], [obs.rounds]);
  return (
    <div className="stack">
      <Verdict obs={obs} />
      <div className="row">
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy !== null}
          onClick={() => {
            const req = { num_voters: obs.num_voters, seed: obs.seed, rounds: obs.rounds, budget: centsToString(initialBudgetCents(obs)) };
            leaveGame();
            void createGame(req);
          }}
        >
          <RotateCcw size={16} aria-hidden /> Replay seed {obs.seed}
        </button>
        <button type="button" className="btn" onClick={leaveGame}>
          New game
        </button>
      </div>

      {!debrief ? (
        <div className="loading" role="status" style={{ minHeight: 200 }}>
          <Loader2 className="spin" aria-hidden /> Loading results…
        </div>
      ) : (
        <>
          <Panel title="What happened">
            <ul className="list-plain lessons">
              {lessons(obs, debrief).map((l, i) => (
                <li key={i} className="lesson">
                  {l}
                </li>
              ))}
            </ul>
          </Panel>

          <Panel title="What each region was">
            <RevealCards obs={obs} debrief={debrief} />
          </Panel>

          <div className="grid-2">
            <Panel title="Polls vs actual">
              <PollChart obs={obs} truth={debrief.share_trajectory} height={260} />
            </Panel>
            <Panel title="Your manifesto vs the oracle">
              <ManifestoCompare manifesto={obs.manifesto} debrief={debrief} />
            </Panel>
          </div>

          <Panel title="How opinion spread" className="results-network">
            <NetworkView sample={graph} error={graphError} totalVoters={obs.num_voters} revealRounds={roundsLabels} />
          </Panel>

          <EpistemicPanel timeline={epistemic.timeline} error={epistemic.error} onRetry={epistemic.retry} />
        </>
      )}
    </div>
  );
}
