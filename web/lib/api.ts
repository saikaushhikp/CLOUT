/**
 * Typed wrappers for every route of docs/v2/web_api_v2.md.
 * In the browser all calls go to same-origin `/api/...`; Next rewrites them to the FastAPI engine.
 * With NEXT_PUBLIC_DUMBOCRACY_MOCK=1 the in-browser mock engine (lib/mockApi.ts) is used instead.
 */
import type {
  ActionResponse,
  AllocateRequest,
  CommitRequest,
  CreateGameRequest,
  CreateGameResponse,
  Debrief,
  EpistemicTimeline,
  GraphSample,
  Health,
  Observation,
  QueryRequest,
} from "./types";
import { createMockClient } from "./mockApi";
import { ApiError } from "./errors";

export { ApiError };

export interface DumbocracyClient {
  readonly kind: "http" | "mock";
  health(): Promise<Health>;
  createGame(req: CreateGameRequest): Promise<CreateGameResponse>;
  observation(gameId: string, playerId: string): Promise<Observation>;
  query(gameId: string, playerId: string, req: QueryRequest): Promise<ActionResponse>;
  commit(gameId: string, playerId: string, req: CommitRequest): Promise<ActionResponse>;
  allocate(gameId: string, playerId: string, req: AllocateRequest): Promise<ActionResponse>;
  forfeit(gameId: string, playerId: string): Promise<ActionResponse>;
  graphSample(gameId: string, maxNodes?: number): Promise<GraphSample>;
  debrief(gameId: string, playerId: string): Promise<Debrief>;
  /** P5 epistemic timeline (DONE only; first call may take ~10 s on the engine, then cached). */
  epistemic(gameId: string, playerId: string): Promise<EpistemicTimeline>;
  deleteGame(gameId: string): Promise<{ deleted: boolean }>;
}

type FetchLike = typeof fetch;

function describeDetail(detail: unknown, status: number): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI 422: [{loc, msg, type}]
    return detail.map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : String(d))).join("; ");
  }
  if (status === 404) return "Game not found — it may have expired on the server.";
  return `Request failed (HTTP ${status})`;
}

export function createHttpClient(base = "", fetchImpl?: FetchLike): DumbocracyClient {
  const doFetch: FetchLike = fetchImpl ?? ((...args) => fetch(...args));
  const enc = encodeURIComponent;

  async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
    let res: Response;
    try {
      res = await doFetch(`${base}${path}`, {
        method,
        headers: body === undefined ? { Accept: "application/json" } : { "Content-Type": "application/json", Accept: "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        cache: "no-store",
      });
    } catch (err) {
      throw new ApiError(0, null, "Cannot reach the game engine. Is the Python server running on :61008?");
    }
    const text = await res.text();
    let data: unknown = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }
    }
    if (!res.ok) {
      const detail = data && typeof data === "object" && "detail" in data ? (data as { detail: unknown }).detail : data;
      const msg = res.status >= 500 && res.status < 600 && !detail
        ? "The game engine is unavailable (is it running on :61008?)."
        : describeDetail(detail, res.status);
      throw new ApiError(res.status, detail, msg);
    }
    return data as T;
  }

  const playerPath = (g: string, p: string) => `/api/games/${enc(g)}/players/${enc(p)}`;

  return {
    kind: "http",
    health: () => call<Health>("GET", "/api/health"),
    createGame: (req) => call<CreateGameResponse>("POST", "/api/games", req),
    observation: (g, p) => call<Observation>("GET", `${playerPath(g, p)}/observation`),
    query: (g, p, req) => call<ActionResponse>("POST", `${playerPath(g, p)}/query`, req),
    commit: (g, p, req) => call<ActionResponse>("POST", `${playerPath(g, p)}/commit`, req),
    allocate: (g, p, req) => call<ActionResponse>("POST", `${playerPath(g, p)}/allocate`, req),
    forfeit: (g, p) => call<ActionResponse>("POST", `${playerPath(g, p)}/forfeit`),
    graphSample: (g, maxNodes = 1500) => call<GraphSample>("GET", `/api/games/${enc(g)}/graph_sample?max_nodes=${maxNodes}`),
    debrief: (g, p) => call<Debrief>("GET", `${playerPath(g, p)}/debrief`),
    epistemic: (g, p) => call<EpistemicTimeline>("GET", `${playerPath(g, p)}/epistemic`),
    deleteGame: (g) => call<{ deleted: boolean }>("DELETE", `/api/games/${enc(g)}`),
  };
}

export const isMockMode = (): boolean => process.env.NEXT_PUBLIC_DUMBOCRACY_MOCK === "1";

let singleton: DumbocracyClient | null = null;

/** The app-wide client: mock engine when NEXT_PUBLIC_DUMBOCRACY_MOCK=1, otherwise same-origin HTTP. */
export function getClient(): DumbocracyClient {
  if (!singleton) {
    const demo = typeof window !== "undefined" && /[?&]demo=/.test(window.location.search);
    singleton = isMockMode() ? createMockClient({ latencyMs: demo ? 0 : 160 }) : createHttpClient("");
  }
  return singleton;
}
