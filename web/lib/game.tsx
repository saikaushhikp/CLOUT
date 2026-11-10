"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { DumbocracyClient } from "./api";
import { getClient } from "./api";
import { ApiError } from "./errors";
import type {
  ActionResponse,
  AllocateRequest,
  CreateGameRequest,
  Debrief,
  GraphSample,
  Health,
  Observation,
  QueryAnswer,
  RoundReport,
} from "./types";
import { ERROR_COPY } from "./constants";

export type Toast = { id: number; kind: "error" | "info" | "success"; title: string; message?: string };

type Busy = null | "create" | "query" | "commit" | "allocate" | "forfeit" | "load";

type GameContextValue = {
  client: DumbocracyClient;
  health: Health | null;
  healthError: string | null;
  obs: Observation | null;
  busy: Busy;
  lastAnswer: QueryAnswer | null;
  lastReport: RoundReport | null;
  graph: GraphSample | null;
  graphError: string | null;
  debrief: Debrief | null;
  hunches: Record<string, string | undefined>;
  setHunch: (region: string, archetypeId: string | undefined) => void;
  toasts: Toast[];
  pushToast: (t: Omit<Toast, "id">) => void;
  dismissToast: (id: number) => void;
  createGame: (req: CreateGameRequest) => Promise<boolean>;
  query: (sectorA: string, sectorB: string, region: string) => Promise<ActionResponse | null>;
  commit: (manifesto: number[]) => Promise<ActionResponse | null>;
  allocate: (req: AllocateRequest) => Promise<ActionResponse | null>;
  forfeit: () => Promise<ActionResponse | null>;
  leaveGame: () => void;
  refreshGraph: () => Promise<void>;
  /** Round look-back (R19): the past round under review, or null while planning the current round. */
  reviewRound: number | null;
  setReviewRound: (round: number | null) => void;
};

const GameContext = createContext<GameContextValue | null>(null);
const SESSION_KEY = "dumbocracy.v2.session";
const HUNCH_KEY = (g: string) => `dumbocracy.v2.hunches.${g}`;

function safeGet(key: string): string | null {
  try {
    return window.sessionStorage.getItem(key);
  } catch {
    return null;
  }
}
function safeSet(key: string, value: string | null) {
  try {
    if (value === null) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, value);
  } catch {
    /* storage unavailable: resume-on-refresh simply won't work */
  }
}

export function GameProvider({
  children,
  client: injected,
  initialObservation,
  resume = true,
}: {
  children: ReactNode;
  client?: DumbocracyClient;
  initialObservation?: Observation;
  resume?: boolean;
}) {
  const client = useMemo(() => injected ?? getClient(), [injected]);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [obs, setObs] = useState<Observation | null>(initialObservation ?? null);
  const [busy, setBusy] = useState<Busy>(null);
  const [lastAnswer, setLastAnswer] = useState<QueryAnswer | null>(null);
  const [lastReport, setLastReport] = useState<RoundReport | null>(null);
  const [graph, setGraph] = useState<GraphSample | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [debrief, setDebrief] = useState<Debrief | null>(null);
  const [hunches, setHunches] = useState<Record<string, string | undefined>>({});
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [reviewRound, setReviewRound] = useState<number | null>(null);
  const toastId = useRef(0);
  const busyRef = useRef<Busy>(null);

  const pushToast = useCallback((t: Omit<Toast, "id">) => {
    const id = ++toastId.current;
    setToasts((xs) => [...xs.slice(-3), { ...t, id }]);
    setTimeout(() => setToasts((xs) => xs.filter((x) => x.id !== id)), t.kind === "error" ? 6500 : 4200);
  }, []);
  const dismissToast = useCallback((id: number) => setToasts((xs) => xs.filter((x) => x.id !== id)), []);

  const reportError = useCallback(
    (err: unknown, title = "Something went wrong") => {
      const message = err instanceof ApiError ? err.message : err instanceof Error ? err.message : String(err);
      pushToast({ kind: "error", title, message });
    },
    [pushToast],
  );

  // Engine health (shown on the setup screen).
  useEffect(() => {
    let alive = true;
    client
      .health()
      .then((h) => alive && (setHealth(h), setHealthError(null)))
      .catch((e: unknown) => alive && setHealthError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [client]);

  // Resume a game after a page refresh.
  useEffect(() => {
    if (!resume || initialObservation) return;
    const raw = safeGet(SESSION_KEY);
    if (!raw) return;
    let alive = true;
    try {
      const { g, p } = JSON.parse(raw) as { g: string; p: string };
      setBusy("load");
      client
        .observation(g, p)
        .then((o) => alive && setObs(o))
        .catch(() => safeSet(SESSION_KEY, null))
        .finally(() => alive && setBusy(null));
    } catch {
      safeSet(SESSION_KEY, null);
    }
    return () => {
      alive = false;
    };
  }, [client, resume, initialObservation]);

  // Mock-only shortcut for UI work: /?demo=survey|campaign|done jumps straight to a phase.
  useEffect(() => {
    if (client.kind !== "mock" || initialObservation) return;
    let demo: string | null = null;
    try {
      demo = new URLSearchParams(window.location.search).get("demo");
    } catch {
      demo = null;
    }
    if (!demo || !["survey", "campaign", "done"].includes(demo)) return;
    let alive = true;
    (async () => {
      setBusy("load");
      try {
        const g = await client.createGame({ num_voters: 10_000, seed: 2026, rounds: 5, budget: "750.00" });
        let o = g.observation;
        const ask = async (a: string, b: string, r: string) => (o = (await client.query(g.game_id, g.player_id, { sector_a: a, sector_b: b, region: r })).observation);
        await ask("Edu", "Agri", "N");
        await ask("Edu", "Agri", "N"); // a repeat: pooled with the first sample
        await ask("Industry", "Env", "E");
        await ask("Welfare", "Edu", "S");
        if (demo !== "survey") {
          o = (await client.commit(g.game_id, g.player_id, { manifesto: [0.25, 0.2, 0.2, 0.2, 0.15] })).observation;
          const plans = [
            // small plans: the 5-round "done" demo must fit the ₫750 treasury at the observed worker fee
            { hires_by_region: [1, 0, 1, 0], media_regions: [true, false, false, false] },
            { hires_by_region: [0, 1, 0, 0], media_regions: [false, false, false, false] },
          ];
          const n = demo === "done" ? o.rounds : 2;
          for (let t = 0; t < n; t++) o = (await client.allocate(g.game_id, g.player_id, plans[t % 2])).observation;
        }
        if (alive) {
          safeSet(SESSION_KEY, JSON.stringify({ g: g.game_id, p: g.player_id }));
          setObs(o);
        }
      } finally {
        if (alive) setBusy(null);
      }
    })();
    return () => {
      alive = false;
    };
  }, [client, initialObservation]);

  // Hunches are UI-only notes, per game.
  useEffect(() => {
    if (!obs) return;
    const raw = safeGet(HUNCH_KEY(obs.game_id));
    try {
      setHunches(raw ? (JSON.parse(raw) as Record<string, string>) : {});
    } catch {
      setHunches({});
    }
  }, [obs?.game_id]); // eslint-disable-line react-hooks/exhaustive-deps

  const setHunch = useCallback(
    (region: string, archetypeId: string | undefined) => {
      setHunches((h) => {
        const next = { ...h, [region]: archetypeId };
        if (obs) safeSet(HUNCH_KEY(obs.game_id), JSON.stringify(next));
        return next;
      });
    },
    [obs],
  );

  const phase = obs?.phase;
  const gameId = obs?.game_id;
  const roundKey = obs ? `${obs.phase}:${obs.round}` : "";

  const refreshGraph = useCallback(async () => {
    if (!gameId) return;
    try {
      setGraph(await client.graphSample(gameId, 1500));
      setGraphError(null);
    } catch (e) {
      setGraphError(e instanceof Error ? e.message : String(e));
    }
  }, [client, gameId]);

  // Graph sample (R18): fetched and rendered ONLY in the post-election debrief — never during the survey or the
  // campaign rounds (saves the download and the WebGL layout while playing).
  const graphStage = gameId && phase === "DONE" ? "revealed" : null;
  useEffect(() => {
    if (!graphStage) {
      setGraph(null);
      setGraphError(null);
      return;
    }
    setGraph(null);
    void refreshGraph();
  }, [graphStage, refreshGraph]);

  useEffect(() => {
    if (!obs || obs.phase !== "DONE") {
      setDebrief(null);
      return;
    }
    let alive = true;
    client
      .debrief(obs.game_id, obs.player_id)
      .then((d) => alive && setDebrief(d))
      .catch((e: unknown) => alive && reportError(e, "Debrief unavailable"));
    return () => {
      alive = false;
    };
  }, [client, obs?.game_id, phase]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setLastAnswer(null);
    setReviewRound(null);
  }, [roundKey]);

  const run = useCallback(
    async (kind: Exclude<Busy, null>, fn: () => Promise<ActionResponse>): Promise<ActionResponse | null> => {
      if (busyRef.current) return null;
      busyRef.current = kind;
      setBusy(kind);
      try {
        const res = await fn();
        setObs(res.observation);
        if (!res.accepted) {
          const code = res.error_code ?? "REJECTED";
          pushToast({ kind: "error", title: ERROR_COPY[code] ?? "Move rejected", message: res.message ? `${code}: ${res.message}` : code });
        }
        return res;
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) {
          safeSet(SESSION_KEY, null);
          setObs(null);
        }
        reportError(e);
        return null;
      } finally {
        busyRef.current = null;
        setBusy(null);
      }
    },
    [pushToast, reportError],
  );

  const createGame = useCallback(
    async (req: CreateGameRequest) => {
      if (busyRef.current) return false;
      busyRef.current = "create";
      setBusy("create");
      try {
        const res = await client.createGame(req);
        setObs(res.observation);
        setLastReport(null);
        setGraph(null);
        safeSet(SESSION_KEY, JSON.stringify({ g: res.game_id, p: res.player_id }));
        return true;
      } catch (e) {
        reportError(e, "Could not start the game");
        return false;
      } finally {
        busyRef.current = null;
        setBusy(null);
      }
    },
    [client, reportError],
  );

  const query = useCallback(
    async (a: string, b: string, region: string) => {
      if (!obs) return null;
      const res = await run("query", () => client.query(obs.game_id, obs.player_id, { sector_a: a, sector_b: b, region }));
      if (res?.accepted && res.answer) setLastAnswer(res.answer);
      return res;
    },
    [client, obs, run],
  );

  const commit = useCallback(
    async (manifesto: number[]) => {
      if (!obs) return null;
      const res = await run("commit", () => client.commit(obs.game_id, obs.player_id, { manifesto }));
      if (res?.accepted) pushToast({ kind: "success", title: "Manifesto locked in", message: "Round 1 — the first poll just landed." });
      return res;
    },
    [client, obs, run, pushToast],
  );

  const afterRound = useCallback(
    (res: ActionResponse | null) => {
      if (res?.accepted && res.round_report) {
        setLastReport(res.round_report);
        if (res.round_report.election_held) pushToast({ kind: "info", title: "Polls closed", message: "The votes are in." });
      }
      return res;
    },
    [pushToast],
  );

  const allocate = useCallback(
    async (req: AllocateRequest) => {
      if (!obs) return null;
      return afterRound(await run("allocate", () => client.allocate(obs.game_id, obs.player_id, req)));
    },
    [client, obs, run, afterRound],
  );

  const forfeit = useCallback(async () => {
    if (!obs) return null;
    const res = afterRound(await run("forfeit", () => client.forfeit(obs.game_id, obs.player_id)));
    if (res?.accepted) pushToast({ kind: "info", title: "Time's up", message: "The round was forfeited — no campaign this round." });
    return res;
  }, [client, obs, run, afterRound, pushToast]);

  const leaveGame = useCallback(() => {
    if (obs) {
      client.deleteGame(obs.game_id).catch(() => undefined);
      safeSet(HUNCH_KEY(obs.game_id), null);
    }
    safeSet(SESSION_KEY, null);
    setObs(null);
    setGraph(null);
    setDebrief(null);
    setLastReport(null);
    setLastAnswer(null);
  }, [client, obs]);

  // After a refresh the last round report comes from the observation's history (R19).
  const latestReport = lastReport ?? (obs?.round_reports?.length ? obs.round_reports[obs.round_reports.length - 1] : null);

  const value: GameContextValue = {
    client,
    health,
    healthError,
    obs,
    busy,
    lastAnswer,
    lastReport: latestReport,
    graph,
    graphError,
    debrief,
    hunches,
    setHunch,
    toasts,
    pushToast,
    dismissToast,
    createGame,
    query,
    commit,
    allocate,
    forfeit,
    leaveGame,
    refreshGraph,
    reviewRound,
    setReviewRound,
  };
  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
}

export function useGame(): GameContextValue {
  const ctx = useContext(GameContext);
  if (!ctx) throw new Error("useGame must be used inside <GameProvider>");
  return ctx;
}

/** Same as useGame but asserts an active observation. */
export function useObservation(): Observation {
  const { obs } = useGame();
  if (!obs) throw new Error("No active game");
  return obs;
}
