import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import {
  Activity, Check, CircleHelp, Cpu, Database, Gauge, Landmark, LoaderCircle,
  Play, RotateCcw, Search, ShieldCheck, Sparkles, Target, Waves, X,
} from "lucide-react";

const SupportChart = lazy(() => import("./SupportChart.jsx"));

const DEFAULTS = {
  node_count: "1000",
  horizon_rounds: "5",
  initial_budget: "20.00",
  seed: "20261019",
  backend: "auto",
};

const ACTION_META = {
  targeted_contact: { label: "Targeted contact", description: "Reach an eligible public node", icon: Target },
  broadcast: { label: "Broadcast", description: "Send a broad campaign message", icon: Waves },
  no_op: { label: "Wait / no-op", description: "Advance without spending", icon: Activity },
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: options.body ? { "content-type": "application/json", ...options.headers } : options.headers,
  });
  let payload;
  try { payload = await response.json(); } catch { throw new Error("The local service returned an unreadable response."); }
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status}).`);
  return payload;
}

function money(value) {
  const amount = Number(value);
  return Number.isFinite(amount) ? amount.toFixed(2) : String(value ?? "—");
}

function percent(value) {
  const amount = Number(value);
  return Number.isFinite(amount) ? `${(amount * 100).toFixed(1)}%` : "—";
}

function Notice({ notice, onDismiss }) {
  if (!notice) return null;
  const Icon = notice.type === "error" ? X : Check;
  return (
    <div className={`notice notice-${notice.type}`} role="status" aria-live="polite">
      <Icon size={16} aria-hidden="true" />
      <span>{notice.message}</span>
      <button className="icon-button" onClick={onDismiss} aria-label="Dismiss message"><X size={15} /></button>
    </div>
  );
}

function Metric({ label, value, sub, icon: Icon, tone = "mint" }) {
  return (
    <article className="metric-card">
      <div className={`metric-icon tone-${tone}`}><Icon size={17} strokeWidth={2} /></div>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-sub">{sub}</div>
    </article>
  );
}

function Timeline({ observations }) {
  const rows = observations.slice(1).map((current, index) => {
    const previous = observations[index];
    const event = (current.public_events || []).find(
      (item) => item.event_type === "action_accepted" && item.round_index === previous.round_index,
    );
    return {
      round: previous.round_index + 1,
      action: event?.action_id || "action",
      cost: Math.max(0, Number(previous.remaining_budget) - Number(current.remaining_budget)),
    };
  }).reverse();
  if (!rows.length) return <div className="empty-row">Your decisions and their costs will appear here.</div>;
  return (
    <ol className="timeline-list">
      {rows.map((row) => {
        const label = ACTION_META[row.action]?.label || row.action.replaceAll("_", " ");
        return <li key={`${row.round}-${row.action}`}><span>Round {row.round}</span><b>{label}</b><strong>−{row.cost.toFixed(2)}</strong></li>;
      })}
    </ol>
  );
}

function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [service, setService] = useState("checking");
  const [gameId, setGameId] = useState(null);
  const [game, setGame] = useState(null);
  const [result, setResult] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [targetSearch, setTargetSearch] = useState("");
  const [targetId, setTargetId] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [eventsOpen, setEventsOpen] = useState(false);

  useEffect(() => {
    let active = true;
    api("/api/health").then(() => active && setService("online")).catch(() => active && setService("offline"));
    return () => { active = false; };
  }, []);

  const observation = game?.observation;
  const timeline = game?.timeline?.length ? game.timeline : observation ? [observation] : [];
  const horizon = Number(form.horizon_rounds) || 1;
  const descriptors = observation?.legal_actions || [];
  const eligible = descriptors.find((item) => item.action_id === "targeted_contact")?.eligible_target_ids || [];
  const filteredTargets = useMemo(() => {
    const query = targetSearch.trim().toLowerCase();
    return eligible.filter((id) => id.toLowerCase().includes(query)).slice(0, 300);
  }, [eligible, targetSearch]);
  const graph = observation?.graph_summary || {};
  const publicEvents = observation?.public_events || [];
  const isTerminal = Boolean(observation?.terminal_status);

  useEffect(() => {
    if (!eligible.length) setTargetId("");
    else if (!eligible.includes(targetId)) setTargetId(filteredTargets[0] || "");
  }, [eligible, filteredTargets, targetId]);

  function updateField(event) {
    setForm((previous) => ({ ...previous, [event.target.name]: event.target.value }));
  }

  async function startGame(event) {
    event.preventDefault();
    setBusy(true); setNotice(null); setResult(null); setAnalysis(null);
    try {
      const created = await api("/api/games", {
        method: "POST",
        body: JSON.stringify({ ...form, node_count: Number(form.node_count), horizon_rounds: Number(form.horizon_rounds), seed: Number(form.seed) }),
      });
      setGameId(created.game_id); setGame(created);
      setNotice({ type: "success", message: `Episode ready · ${created.backend} selected.` });
    } catch (error) { setNotice({ type: "error", message: error.message }); }
    finally { setBusy(false); }
  }

  async function takeAction(actionId) {
    if (!gameId || !game) return;
    if (actionId === "targeted_contact" && !targetId) {
      setNotice({ type: "error", message: "Choose an eligible public target first." }); return;
    }
    setBusy(true); setNotice(null);
    try {
      const response = await api(`/api/games/${gameId}/actions`, {
        method: "POST",
        body: JSON.stringify({ action_id: actionId, target_ids: actionId === "targeted_contact" ? [targetId] : [] }),
      });
      setGame(response);
      if (!response.accepted) {
        setNotice({ type: "error", message: `Action rejected (${response.error_code || "rejected"}). Round and budget are unchanged.` });
      } else if (response.terminal) {
        const finalResult = response.result ?? await api(`/api/games/${gameId}/result`);
        setResult(finalResult);
        const finalAnalysis = await api(`/api/games/${gameId}/analysis`);
        setAnalysis(finalAnalysis);
        setNotice({ type: "success", message: "Fixed horizon complete. Your official result is ready." });
      } else {
        setNotice({ type: "success", message: `${ACTION_META[actionId]?.label || actionId} accepted · round advanced.` });
      }
    } catch (error) { setNotice({ type: "error", message: error.message }); }
    finally { setBusy(false); }
  }

  async function resetGame() {
    if (!gameId) return;
    setBusy(true); setNotice(null);
    try {
      const reset = await api(`/api/games/${gameId}/reset`, { method: "POST" });
      setGame(reset); setResult(null); setAnalysis(null); setEventsOpen(false);
      setNotice({ type: "success", message: "Episode reset. The seed reproduced the initial public state." });
    } catch (error) { setNotice({ type: "error", message: error.message }); }
    finally { setBusy(false); }
  }

  const events = [...publicEvents].reverse().slice(0, 30);
  const serviceMessage = service === "online" ? "Local service online" : service === "offline" ? "Service unavailable" : "Connecting";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Dumbocracy Campaign Lab home">
          <span className="brand-mark">D</span><span><b>Dumbocracy</b><small>CAMPAIGN LAB</small></span>
        </a>
        <div className="topbar-right">
          <span className={`service-status status-${service}`}><i />{serviceMessage}</span>
          <span className="privacy-chip"><ShieldCheck size={14} /> Private by design</span>
        </div>
      </header>

      <main>
        <section className="hero">
          <div>
            <div className="eyebrow"><Sparkles size={14} /> FINITE-WORLD STRATEGY SIMULATOR</div>
            <h1>Plan a campaign.<br /><em>Read the room.</em></h1>
            <p>Make deliberate moves with a fixed budget and public information. Every round gives you a new snapshot of the electorate.</p>
          </div>
          <div className="hero-note"><ShieldCheck size={18} /><span>Only aggregate observations are shown. Voter state and network connections stay inside the simulation.</span></div>
        </section>

        <section className="setup-card card" aria-labelledby="setup-heading">
          <div className="section-heading"><div><div className="eyebrow">01 / SETUP</div><h2 id="setup-heading">Start an episode</h2></div><span className="section-note">Replayable with the same seed</span></div>
          <form className="setup-form" onSubmit={startGame}>
            <label><span>Population <small>nodes</small></span><input name="node_count" type="number" min="1" max="100000" required value={form.node_count} onChange={updateField} /></label>
            <label><span>Horizon <small>rounds</small></span><input name="horizon_rounds" type="number" min="1" max="100" required value={form.horizon_rounds} onChange={updateField} /></label>
            <label><span>Budget <small>credits</small></span><input name="initial_budget" type="number" min="0" step="0.01" required value={form.initial_budget} onChange={updateField} /></label>
            <label><span>Seed <small>replayable</small></span><input name="seed" type="number" required value={form.seed} onChange={updateField} /></label>
            <label><span>Compute <small>device</small></span><select name="backend" value={form.backend} onChange={updateField}><option value="auto">Automatic</option><option value="cpu">CPU</option><option value="cuda:0">CUDA 0</option></select></label>
            <button className="button button-primary" type="submit" disabled={busy || service === "offline"}>{busy && !game ? <LoaderCircle className="spin" size={16} /> : <Play size={15} fill="currentColor" />} {busy && !game ? "Starting…" : "Start episode"}</button>
          </form>
          <div className="setup-help"><CircleHelp size={13} /> Automatic uses CUDA when available and reports the selected device below.</div>
        </section>

        <Notice notice={notice} onDismiss={() => setNotice(null)} />

        <section className="workspace" aria-label="Campaign dashboard">
          <div className="primary-column">
            <div className="metrics-grid">
              <Metric label="ROUND" value={observation ? `${observation.round_index} / ${horizon}` : "— / —"} sub={isTerminal ? "Horizon complete" : "Fixed horizon"} icon={Activity} />
              <Metric label="SUPPORT" value={observation ? percent(observation.aggregate_support) : "—"} sub="Public estimate" icon={Gauge} tone="gold" />
              <Metric label="BUDGET" value={observation ? money(observation.remaining_budget) : "—"} sub="Credits remaining" icon={Landmark} tone="blue" />
              <Metric label="DEVICE" value={game?.backend || "—"} sub={graph.node_count ? `${Number(graph.node_count).toLocaleString()} nodes` : "Waiting"} icon={Cpu} tone="purple" />
            </div>

            <section className="card panel chart-panel" aria-labelledby="chart-heading">
              <div className="section-heading"><div><div className="eyebrow">02 / TRACK</div><h2 id="chart-heading">Support trajectory</h2></div><span className="legend"><i /> Aggregate support</span></div>
              <p className="panel-desc">Watch the public signal change after each decision.</p>
              <div className="chart-frame">
                {timeline.length < 2 ? <div className="chart-empty"><Activity size={21} /><span>Play a round to see the trend.</span></div> : <Suspense fallback={<div className="chart-empty"><LoaderCircle className="spin" size={18} /> Loading chart…</div>}><SupportChart timeline={timeline} /></Suspense>}
              </div>
              <div className="chart-axis"><span>START</span><span>ROUND {horizon}</span></div>
            </section>

            <section className="card panel history-panel" aria-labelledby="history-heading">
              <div className="section-heading"><div><div className="eyebrow">03 / LOG</div><h2 id="history-heading">Your decisions</h2></div><span className="count-pill">{Math.max(0, timeline.length - 1)} actions</span></div>
              <Timeline observations={timeline} />
            </section>
          </div>

          <aside className="secondary-column">
            <section className="card panel action-panel" aria-labelledby="action-heading">
              <div className="action-header"><div><div className="eyebrow">YOUR NEXT MOVE</div><h2 id="action-heading">Choose an action</h2></div><span className="round-pill">{isTerminal ? "COMPLETE" : `ROUND ${observation ? observation.round_index + 1 : "—"}`}</span></div>
              <p className="panel-desc">Use the current public snapshot to decide where your budget goes.</p>
              <label className="target-label" htmlFor="target-search">Public target</label>
              <div className="search-box"><Search size={15} /><input id="target-search" type="search" aria-label="Public target" placeholder="Search eligible IDs…" value={targetSearch} onChange={(event) => setTargetSearch(event.target.value)} disabled={!eligible.length} /><span>{eligible.length ? `${eligible.length.toLocaleString()} eligible` : "—"}</span></div>
              <label className="sr-only" htmlFor="target-select">Choose a public target ID</label>
              <select id="target-select" className="target-select" value={targetId} onChange={(event) => setTargetId(event.target.value)} disabled={!eligible.length}>
                {filteredTargets.length ? filteredTargets.map((id) => <option key={id} value={id}>{id}</option>) : <option value="">{eligible.length ? "No matching public IDs" : "Start an episode first"}</option>}
              </select>
              <div className="action-list">
                {descriptors.map((descriptor) => {
                  const meta = ACTION_META[descriptor.action_id];
                  if (!meta) return null;
                  const Icon = meta.icon;
                  return <button key={descriptor.action_id} className="action-choice" onClick={() => takeAction(descriptor.action_id)} disabled={busy || isTerminal || (descriptor.action_id === "targeted_contact" && !eligible.length)}>
                    <span className="action-icon"><Icon size={17} /></span><span className="action-copy"><b>{meta.label}</b><small>{meta.description}</small></span><strong>{descriptor.cost}</strong>
                  </button>;
                })}
                {!descriptors.length && <div className="empty-row">Start an episode to see available actions.</div>}
              </div>
              <button className="button button-secondary reset-button" onClick={resetGame} disabled={!gameId || busy}><RotateCcw size={14} /> Reset & replay</button>
              <p className="action-footnote">Rejected actions do not advance the round or spend credits.</p>
            </section>

            <section className="card panel network-panel" aria-labelledby="network-heading">
              <div className="section-heading"><div><div className="eyebrow">PUBLIC SNAPSHOT</div><h2 id="network-heading">Network overview</h2></div><Database size={17} className="muted-icon" /></div>
              <dl className="network-facts">{Object.entries(graph).length ? Object.entries(graph).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{typeof value === "number" && !Number.isInteger(value) ? value.toFixed(4) : typeof value === "number" ? value.toLocaleString() : value}</dd></div>) : <div className="fact-empty"><dt>Status</dt><dd>Waiting for episode</dd></div>}</dl>
            </section>

            {result && <section className={`card panel result-panel ${result.victory ? "result-win" : "result-loss"}`} aria-labelledby="result-heading">
              <div className="eyebrow">FIXED HORIZON COMPLETE</div><h2 id="result-heading">{result.victory ? "Victory" : "Defeat"}</h2>
              <p>Final vote share <b>{percent(result.vote_share)}</b><br />Budget used <b>{money(result.budget_used)}</b></p>
              <details><summary>Trace-only diagnostics</summary><pre>{JSON.stringify(analysis, null, 2)}</pre></details>
            </section>}
          </aside>
        </section>

        <section className={`card event-panel ${eventsOpen ? "events-open" : ""}`}>
          <button className="event-toggle" onClick={() => setEventsOpen((open) => !open)} aria-expanded={eventsOpen}><span><Activity size={15} /> Public event feed</span><span>{publicEvents.length} {publicEvents.length === 1 ? "event" : "events"} {eventsOpen ? "−" : "+"}</span></button>
          {eventsOpen && <ol className="event-list">{events.length ? events.map((event, index) => <li key={`${event.sequence ?? event.round_index}-${index}`}><span>R{event.round_index}</span><b>{event.summary}</b></li>) : <li className="event-empty">Public events appear here as the episode progresses.</li>}</ol>}
        </section>
        <footer className="footer"><span>DUMBOCRACY · LOCAL RESEARCH BETA</span><span>Fixed horizon · Aggregate observations · Hidden state stays private</span></footer>
      </main>
    </div>
  );
}

export default App;
