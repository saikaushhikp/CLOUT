/**
 * In-browser MOCK of the Dumbocracy v2 engine, faithful to docs/v2/web_api_v2.md.
 *
 * It is a tiny fake engine, not the real model: each region is represented by a few hundred
 * "representative voters" (Dirichlet preferences around a hidden archetype, Beta susceptibilities
 * from a hidden personality, 5% zealots). Rules, costs, phases, rejection codes, polls, round
 * resolution, the election, the debrief, the epistemic timeline (a cheap heuristic stand-in for the
 * P5 possible-worlds computation) and the graph sample follow the contract so the UI can be
 * built and tested without the Python backend. Enabled with NEXT_PUBLIC_DUMBOCRACY_MOCK=1.
 */
import type { DumbocracyClient } from "./api";
import { ApiError } from "./errors";
import type {
  ActionResponse,
  AllocateRequest,
  CreateGameRequest,
  Debrief,
  EpistemicRound,
  EpistemicTimeline,
  ErrorCode,
  GraphSample,
  Observation,
  QueryAnswer,
  RegionId,
  RoundReport,
} from "./types";
import { REGION_IDS, SECTOR_IDS } from "./types";
import { centsToString, toCents } from "./money";
import { poolKey, pooledMoe, querySampleSize, sampleMoe } from "./survey";

// ---------------------------------------------------------------- public libraries (spec §2.3–2.4, §14)
const ARCHETYPES = [
  { id: "A0", name: "Agrarian heartland", mean: [0.15, 0.2, 0.4, 0.1, 0.15] },
  { id: "A1", name: "Industrial belt", mean: [0.15, 0.15, 0.1, 0.45, 0.15] },
  { id: "A2", name: "Green urban", mean: [0.25, 0.1, 0.1, 0.15, 0.4] },
  { id: "A3", name: "Welfare-reliant", mean: [0.15, 0.45, 0.15, 0.1, 0.15] },
  { id: "A4", name: "Knowledge hub", mean: [0.4, 0.1, 0.1, 0.25, 0.15] },
  { id: "A5", name: "Balanced middle", mean: [0.2, 0.2, 0.2, 0.2, 0.2] },
];
const PERSONALITIES = [
  { id: "P0", name: "Socially driven", a: [6, 3], b: [2, 5], g: [3, 4] },
  { id: "P1", name: "Media-savvy", a: [3, 4], b: [6, 3], g: [2, 5] },
  { id: "P2", name: "Independent", a: [2, 6], b: [2, 6], g: [2, 6] },
  { id: "P3", name: "Receptive to canvassing", a: [3, 4], b: [3, 4], g: [6, 3] },
];
const SECTOR_LABELS = ["Education", "Welfare", "Agriculture", "Industry", "Environment"];
const REGION_LABELS = ["North", "South", "East", "West"];

const P = {
  budget: "500.00", // default starting treasury (R17: the player may choose 500.00..1000.00)
  budgetMin: 50000, // cents
  budgetMax: 100000,
  query: "10.00",
  worker: "20.00",
  minDegree: 3,
  attemptCap: 3,
  /** share of a region's voters with ≥ min_degree friends, not zealots (eligible to be bought as workers) */
  eligibleShare: 0.9,
  kappa: 60,
  omega: 0.15,
  m0: 50,
  etaM0: 0.25,
  lambda: 0.5,
  pMedia: 0.6,
  betaV: 20,
  z: 0.825,
  zealot: 0.05,
  poll: 400,
  meanDegree: 30,
  inRegion: 0.74,
  repsPerRegion: 260,
};

// ---------------------------------------------------------------- deterministic RNG
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
type Rng = () => number;
function hashSeed(seed: number, stream: string): number {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  for (let i = 0; i < stream.length; i++) h = Math.imul(h ^ stream.charCodeAt(i), 0x01000193) >>> 0;
  return h;
}
const streamOf = (seed: number, name: string) => mulberry32(hashSeed(seed, name));
function gauss(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
function gamma(rng: Rng, k: number): number {
  if (k < 1) return gamma(rng, k + 1) * Math.pow(Math.max(rng(), 1e-12), 1 / k);
  const d = k - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = gauss(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
function dirichlet(rng: Rng, alpha: number[]): number[] {
  const g = alpha.map((a) => gamma(rng, a));
  const s = g.reduce((x, y) => x + y, 0);
  return g.map((x) => x / s);
}
const beta = (rng: Rng, a: number, b: number) => {
  const x = gamma(rng, a);
  return x / (x + gamma(rng, b));
};
function shuffle<T>(rng: Rng, xs: T[]): T[] {
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
/** Successes in m draws without replacement from a population of n containing k successes. */
function hypergeometric(rng: Rng, n: number, k: number, m: number): number {
  let good = Math.max(0, Math.min(n, k));
  let left = n;
  let yes = 0;
  for (let t = 0; t < m && left > 0; t++) {
    if (rng() * left < good) {
      yes++;
      good--;
    }
    left--;
  }
  return yes;
}
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const tv = (a: number[], b: number[]) => a.reduce((s, x, i) => s + Math.abs(x - b[i]), 0) / 2;
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x)) as T;
const r2 = (x: number) => Math.round(x * 100) / 100;

// ---------------------------------------------------------------- hidden world
type Rep = { region: number; v: number[]; alpha: number; beta: number; gamma: number; zealot: boolean };

type MockGame = {
  id: string;
  obs: Observation; // the player's observation (public + own knowledge)
  sizes: number[];
  archetypeOf: number[]; // region -> archetype index (hidden)
  personalityOf: number[]; // region -> personality index (hidden)
  reps: Rep[];
  repsByRegion: number[][];
  vHistory: number[][][]; // [snapshot][rep] preferences; 0 = at commit
  trajectory: number[];
  initialReps: number[][];
  eventSeq: number;
  lastTouched: number;
  /** P8 hidden cascade state, per region: permanent workers, last round's converts (spread next round), attempts used */
  team: number[];
  pendingSpreaders: number[];
  attempts: number[];
  /** pooled query tallies per (triple, phase, round) — hidden counts behind the public pooled estimate */
  pools: Map<string, { yes: number; m: number; count: number }>;
};

function regionSizes(n: number, rng: Rng): number[] {
  const m = Math.ceil(0.15 * n);
  const pi = dirichlet(rng, [2, 2, 2, 2]);
  const rest = n - 4 * m;
  const sizes = [m, m, m, m];
  // multinomial(rest, pi) via sequential binomial approximations (exact total)
  let left = rest;
  let mass = 1;
  for (let r = 0; r < 3; r++) {
    const p = Math.min(1, pi[r] / mass);
    const mean = left * p;
    const sd = Math.sqrt(Math.max(left * p * (1 - p), 0));
    const k = Math.max(0, Math.min(left, Math.round(mean + sd * gauss(rng))));
    sizes[r] += k;
    left -= k;
    mass -= pi[r];
  }
  sizes[3] += left;
  return sizes;
}

function mediaPrice(n: number, nr: number): string {
  // m0 · n_r / (N/4), rounded half-up to 0.01
  const cents = Math.floor((P.m0 * 100 * nr) / (n / 4) + 0.5 + 1e-9);
  return centsToString(cents);
}

function top3Pct(game: MockGame): Record<string, number> {
  const out: Record<string, number> = Object.fromEntries(SECTOR_IDS.map((s) => [s, 0]));
  game.repsByRegion.forEach((idxs, r) => {
    const w = game.sizes[r] / game.obs.num_voters / idxs.length;
    for (const i of idxs) {
      const v = game.reps[i].v;
      const order = v.map((x, k) => [x, k] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
      for (let t = 0; t < 3; t++) out[SECTOR_IDS[order[t][1]]] += 100 * w;
    }
  });
  for (const k of Object.keys(out)) out[k] = r2(out[k]);
  return out;
}

function expectedShare(game: MockGame, vs: number[][], b: number[]): number {
  let e = 0;
  game.repsByRegion.forEach((idxs, r) => {
    const w = game.sizes[r] / game.obs.num_voters / idxs.length;
    for (const i of idxs) e += w * sigmoid(P.betaV * (1 - tv(vs[i], b) - P.z));
  });
  return e;
}

function socialStep(game: MockGame, vs: number[][], reps: Rep[]): number[][] {
  const regionMean = game.repsByRegion.map((idxs) => {
    const m = [0, 0, 0, 0, 0];
    for (const i of idxs) vs[i].forEach((x, k) => (m[k] += x / idxs.length));
    return m;
  });
  const global = [0, 0, 0, 0, 0];
  regionMean.forEach((m, r) => m.forEach((x, k) => (global[k] += (x * game.sizes[r]) / game.obs.num_voters)));
  return vs.map((v, i) => {
    const rep = reps[i];
    if (rep.zealot) return v.slice();
    const a = P.omega * rep.alpha;
    const agg = regionMean[rep.region].map((x, k) => P.inRegion * x + (1 - P.inRegion) * global[k]);
    return v.map((x, k) => (1 - a) * x + a * agg[k]);
  });
}

/** Anticipated share of manifesto b with social dynamics only (no campaign) over T rounds. */
function noCampaignShare(game: MockGame, b: number[]): number {
  let vs = game.initialReps.map((v) => v.slice());
  for (let t = 0; t < game.obs.rounds; t++) vs = socialStep(game, vs, game.reps);
  return expectedShare(game, vs, b);
}

function oracleManifesto(game: MockGame): number[] {
  let vs = game.initialReps.map((v) => v.slice());
  for (let t = 0; t < game.obs.rounds; t++) vs = socialStep(game, vs, game.reps);
  // start from the voter-weighted mean and hill-climb in 1-point transfers
  let b = [0, 0, 0, 0, 0];
  game.repsByRegion.forEach((idxs, r) => {
    for (const i of idxs) vs[i].forEach((x, k) => (b[k] += (x * game.sizes[r]) / game.obs.num_voters / idxs.length));
  });
  let best = expectedShare(game, vs, b);
  for (const step of [0.04, 0.02, 0.01]) {
    let improved = true;
    let guard = 0;
    while (improved && guard++ < 60) {
      improved = false;
      for (let i = 0; i < 5; i++)
        for (let j = 0; j < 5; j++) {
          if (i === j || b[j] < step) continue;
          const c = b.slice();
          c[i] += step;
          c[j] -= step;
          const s = expectedShare(game, vs, c);
          if (s > best + 1e-9) {
            best = s;
            b = c;
            improved = true;
          }
        }
    }
  }
  const sum = b.reduce((x, y) => x + y, 0);
  return b.map((x) => r2((x / sum) * 100) / 100);
}

// ---------------------------------------------------------------- engine
/** One-sentence summary of a timeline (mirrors the engine's headline style). */
export function epistemicHeadline(rounds: EpistemicRound[], victory: boolean): string {
  const first = (k: keyof EpistemicRound) => rounds.find((r) => r[k] === true)?.round;
  const truthAll = rounds.every((r) => r.truth_victory_possible);
  const truthLost = rounds.find((r) => !r.truth_victory_possible)?.round;
  const how = first("guaranteed_victory");
  const that = first("knows_win_exists");
  const doom = first("inevitable_defeat");
  const a = truthAll ? "Victory was possible throughout" : truthLost === 1 ? "Victory was out of reach from the start" : `Victory slipped out of reach in round ${truthLost}`;
  const b = how
    ? `you knew a winning plan from round ${how}`
    : that
      ? `you knew a win existed from round ${that}, but never which single plan would deliver it`
      : doom
        ? `you could tell defeat was inevitable from round ${doom}`
        : "you never knew a plan that would surely win";
  return `${a}; ${b}${victory ? " — and you won." : "."}`;
}

export type MockOptions = { latencyMs?: number; maxGames?: number };

export function createMockClient(options: MockOptions = {}): DumbocracyClient {
  const latency = options.latencyMs ?? 0;
  const maxGames = options.maxGames ?? 8;
  const games = new Map<string, MockGame>();
  let counter = 0;

  const wait = <T,>(value: T): Promise<T> =>
    latency > 0 ? new Promise((res) => setTimeout(() => res(value), latency)) : Promise.resolve(value);
  const fail = (status: number, detail: string): Promise<never> =>
    latency > 0
      ? new Promise((_, rej) => setTimeout(() => rej(new ApiError(status, detail, detail)), latency))
      : Promise.reject(new ApiError(status, detail, detail));

  function getGame(g: string, p?: string): MockGame {
    const game = games.get(g);
    if (!game) throw new ApiError(404, "Unknown game", "Unknown game");
    if (p !== undefined && p !== game.obs.player_id) throw new ApiError(404, "Unknown player", "Unknown player");
    game.lastTouched = Date.now() + counter++;
    return game;
  }

  function event(game: MockGame, type: string, message: string) {
    game.obs.events.push({ seq: ++game.eventSeq, round: game.obs.round, type, message });
  }

  function legal(game: MockGame) {
    game.obs.legal_actions =
      game.obs.phase === "SURVEY" ? ["query", "commit"] : game.obs.phase === "ROUND" ? ["query", "allocate", "forfeit"] : [];
  }

  function refreshMediaPrices(game: MockGame) {
    REGION_IDS.forEach((r, i) => {
      game.obs.costs.media_by_region[r] = mediaPrice(game.obs.num_voters, game.sizes[i]);
    });
  }

  function takePoll(game: MockGame, round: number) {
    const b = game.obs.manifesto!;
    const e = expectedShare(game, game.reps.map((r) => r.v), b);
    const rng = streamOf(game.obs.seed, `poll:${round}`);
    let yes = 0;
    for (let i = 0; i < P.poll; i++) if (rng() < e) yes++;
    const p = yes / P.poll;
    game.obs.polls.push({ round, estimate: p, moe: 1.96 * Math.sqrt((p * (1 - p)) / P.poll), sample_size: P.poll });
  }

  function reject(game: MockGame, code: ErrorCode, message: string): ActionResponse {
    return { accepted: false, error_code: code, message, observation: clone(game.obs) };
  }

  function create(req: CreateGameRequest) {
    const n = req.num_voters ?? 10000;
    const rounds = req.rounds ?? 5;
    if (!Number.isInteger(n) || n < 1000 || n > 1_000_000) throw new ApiError(422, "num_voters must be 1000..1000000");
    if (!Number.isInteger(rounds) || rounds < 3 || rounds > 5) throw new ApiError(422, "rounds must be 3..5");
    const seed = req.seed ?? Math.floor(Math.random() * 2 ** 31);
    if (!Number.isInteger(seed) || seed < 0) throw new ApiError(422, "seed must be a non-negative integer");
    const budget = req.budget ?? P.budget;
    if (typeof budget !== "string" || !/^\d+(\.\d{1,2})?$/.test(budget.trim()))
      throw new ApiError(422, "budget must be a decimal string with at most 2 dp");
    const budgetCents = toCents(budget);
    if (budgetCents < P.budgetMin || budgetCents > P.budgetMax) throw new ApiError(422, "budget must be 500.00..1000.00");

    if (games.size >= maxGames) {
      const oldest = [...games.values()].sort((a, b) => a.lastTouched - b.lastTouched)[0];
      games.delete(oldest.id);
    }
    // world is a pure function of (n, seed)
    const sizes = regionSizes(n, streamOf(seed ^ n, "regions"));
    const archetypeOf = shuffle(streamOf(seed ^ n, "archetypes"), [0, 1, 2, 3, 4, 5]).slice(0, 4);
    const personalityOf = shuffle(streamOf(seed ^ n, "personalities"), [0, 1, 2, 3]);
    const prefRng = streamOf(seed ^ n, "preferences");
    const attrRng = streamOf(seed ^ n, "attributes");
    const reps: Rep[] = [];
    const repsByRegion: number[][] = [[], [], [], []];
    for (let r = 0; r < 4; r++) {
      const mu = ARCHETYPES[archetypeOf[r]].mean;
      const pers = PERSONALITIES[personalityOf[r]];
      for (let k = 0; k < P.repsPerRegion; k++) {
        const zealot = attrRng() < P.zealot;
        repsByRegion[r].push(reps.length);
        reps.push({
          region: r,
          v: dirichlet(prefRng, mu.map((m) => P.kappa * m)),
          alpha: zealot ? 0 : beta(attrRng, pers.a[0], pers.a[1]),
          beta: zealot ? 0 : beta(attrRng, pers.b[0], pers.b[1]),
          gamma: zealot ? 0 : beta(attrRng, pers.g[0], pers.g[1]),
          zealot,
        });
      }
    }
    // structural census (public)
    const gRng = streamOf(seed ^ n, "graph");
    const kbar = Math.min(P.meanDegree, n - 1);
    const intra = sizes.map((s) => Math.round((s * kbar * P.inRegion * (0.85 + 0.3 * gRng())) / 2));
    const pairs: [number, number][] = [];
    for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) pairs.push([a, b]);
    const pairW = pairs.map(([a, b]) => (0.05 + gRng()) * sizes[a] * sizes[b]);
    const totalInter = Math.round((n * kbar * (1 - P.inRegion)) / 2);
    const sumW = pairW.reduce((x, y) => x + y, 0);
    const inter = pairW.map((w) => Math.round((totalInter * w) / sumW));
    const degSum = sizes.map((_, r) => 2 * intra[r]);
    pairs.forEach(([a, b], k) => {
      degSum[a] += inter[k];
      degSum[b] += inter[k];
    });

    const id = `mock-${(seed >>> 0).toString(36)}-${Date.now().toString(36)}${(counter++).toString(36)}`;
    const obs: Observation = {
      schema_version: "2.2",
      game_id: id,
      player_id: "p0",
      phase: "SURVEY",
      round: 0,
      rounds,
      num_voters: n,
      seed,
      treasury: centsToString(budgetCents),
      initial_budget: centsToString(budgetCents),
      workers: { by_region: { N: 0, S: 0, E: 0, W: 0 }, total: 0 },
      round_reports: [],
      costs: {
        query: P.query,
        worker: P.worker,
        min_degree: P.minDegree,
        attempt_cap: P.attemptCap,
        media_by_region: { N: "0.00", S: "0.00", E: "0.00", W: "0.00" },
      },
      sectors: SECTOR_IDS.map((s, i) => ({ id: s, label: SECTOR_LABELS[i] })),
      regions: REGION_IDS.map((r, i) => ({
        id: r,
        label: REGION_LABELS[i],
        voters: sizes[i],
        intra_edges: intra[i],
        mean_degree: Math.round((degSum[i] / sizes[i]) * 100) / 100,
      })),
      inter_edges: pairs.map(([a, b], k) => ({ regions: [REGION_IDS[a], REGION_IDS[b]] as [string, string], edges: inter[k] })),
      archetype_library: ARCHETYPES.map((a) => ({ id: a.id, name: a.name, mean: a.mean.slice() })),
      personality_library: PERSONALITIES.map((p) => ({
        id: p.id,
        name: p.name,
        social: r2((p.a[0] / (p.a[0] + p.a[1])) * 100) / 100,
        media: r2((p.b[0] / (p.b[0] + p.b[1])) * 100) / 100,
        campaign: r2((p.g[0] / (p.g[0] + p.g[1])) * 100) / 100,
      })),
      free_stats: [],
      queries: [],
      polls: [],
      manifesto: null,
      allocations: [],
      media_airings: { N: 0, S: 0, E: 0, W: 0 },
      legal_actions: [],
      result: null,
      events: [],
    };
    const game: MockGame = {
      id,
      obs,
      sizes,
      archetypeOf,
      personalityOf,
      reps,
      repsByRegion,
      vHistory: [],
      trajectory: [],
      initialReps: reps.map((r) => r.v.slice()),
      eventSeq: 0,
      lastTouched: Date.now() + counter++,
      team: [0, 0, 0, 0],
      pendingSpreaders: [0, 0, 0, 0],
      attempts: [0, 0, 0, 0],
      pools: new Map(),
    };
    refreshMediaPrices(game);
    obs.free_stats.push({ round: 0, top3_pct: top3Pct(game) });
    legal(game);
    event(game, "game_created", `Electorate of ${n.toLocaleString("en-US")} voters generated (seed ${seed}). Survey phase open.`);
    games.set(id, game);
    return { game_id: id, player_id: "p0", observation: clone(obs) };
  }

  function query(game: MockGame, a: string, b: string, region: string): ActionResponse {
    const o = game.obs;
    if (o.phase === "DONE") return reject(game, "TERMINAL_STATE", "The game is over.");
    const ia = SECTOR_IDS.indexOf(a as never);
    const ib = SECTOR_IDS.indexOf(b as never);
    const ir = REGION_IDS.indexOf(region as never);
    if (ia < 0 || ib < 0 || ir < 0 || ia === ib) return reject(game, "INVALID_QUERY", "Pick two different sectors and a valid region.");
    const phase = o.phase as "SURVEY" | "ROUND";
    if (toCents(o.treasury) < toCents(o.costs.query)) return reject(game, "INSUFFICIENT_FUNDS", "Treasury too low for a query.");
    // R12: a fresh uniform sample (without replacement) of m = round(0.1·n_r) voters. The mock region is
    // a few hundred representatives; scale it to n_r voters and draw a hypergeometric sample.
    const idxs = game.repsByRegion[ir];
    const repYes = idxs.filter((i) => game.reps[i].v[ib] > game.reps[i].v[ia]).length;
    const nr = game.sizes[ir];
    const m = querySampleSize(nr);
    const yes = hypergeometric(streamOf(o.seed, `survey:${o.round}:${phase}:${o.queries.length}`), nr, Math.round((repYes / idxs.length) * nr), m);
    const key = poolKey({ sector_a: a, sector_b: b, region, phase, round: o.round });
    const pool = game.pools.get(key) ?? { yes: 0, m: 0, count: 0 };
    pool.yes += yes;
    pool.m += m;
    pool.count += 1;
    game.pools.set(key, pool);
    const answer: QueryAnswer = {
      round: o.round,
      phase,
      sector_a: a,
      sector_b: b,
      region,
      pct_b_over_a: r2((100 * yes) / m),
      sample_size: m,
      moe: r2(sampleMoe(yes, m, nr)),
      pooled_pct: r2((100 * pool.yes) / pool.m),
      pooled_sample_size: pool.m,
      pooled_moe: r2(pooledMoe(pool.yes, pool.m)),
      repeat_index: pool.count,
      cost: o.costs.query,
    };
    o.treasury = centsToString(toCents(o.treasury) - toCents(o.costs.query));
    o.queries.push(answer);
    event(
      game,
      "query",
      `Query: ${answer.pct_b_over_a.toFixed(1)}% ± ${answer.moe.toFixed(1)} of a ${m}-voter ${REGION_LABELS[ir]} sample prefer ${b} over ${a}` +
        (pool.count > 1 ? ` (pooled ${answer.pooled_pct.toFixed(1)}% ± ${answer.pooled_moe.toFixed(1)}, n = ${pool.m}).` : "."),
    );
    return { accepted: true, error_code: null, message: null, observation: clone(o), answer: clone(answer) };
  }

  function commit(game: MockGame, manifesto: unknown): ActionResponse {
    const o = game.obs;
    if (o.phase === "DONE") return reject(game, "TERMINAL_STATE", "The game is over.");
    if (o.phase !== "SURVEY") return reject(game, "INVALID_PHASE", "The manifesto is already committed.");
    if (
      !Array.isArray(manifesto) ||
      manifesto.length !== 5 ||
      !manifesto.every((x) => typeof x === "number" && Number.isFinite(x) && x >= 0) ||
      Math.abs((manifesto as number[]).reduce((s, x) => s + x, 0) - 1) > 1e-6
    )
      return reject(game, "INVALID_MANIFESTO", "Manifesto must be 5 non-negative shares summing to 1.");
    o.manifesto = (manifesto as number[]).slice();
    o.phase = "ROUND";
    o.round = 1;
    game.initialReps = game.reps.map((r) => r.v.slice());
    game.vHistory = [game.initialReps.map((v) => v.slice())];
    game.trajectory = [expectedShare(game, game.initialReps, o.manifesto)];
    o.free_stats.push({ round: 1, top3_pct: top3Pct(game) });
    takePoll(game, 1);
    legal(game);
    event(game, "commit", "Manifesto committed. Round 1 begins — first poll is in.");
    return { accepted: true, error_code: null, message: null, observation: clone(o) };
  }

  function resolve(game: MockGame, req: AllocateRequest, forfeit: boolean): ActionResponse {
    const o = game.obs;
    if (o.phase === "DONE") return reject(game, "TERMINAL_STATE", "The game is over.");
    if (o.phase !== "ROUND") return reject(game, "INVALID_PHASE", "Commit a manifesto before campaigning.");
    const h = req.hires_by_region;
    const okShape =
      Array.isArray(h) &&
      h.length === 4 &&
      h.every((x) => Number.isInteger(x) && x >= 0) &&
      Array.isArray(req.media_regions) &&
      req.media_regions.length === 4 &&
      req.media_regions.every((x) => typeof x === "boolean");
    if (!okShape) return reject(game, "INVALID_ALLOCATION", "Malformed allocation.");
    // R22: each hire is a distinct eligible voter of that region (degree ≥ min_degree, not a zealot, not a worker yet).
    for (let r = 0; r < 4; r++) {
      if (game.team[r] + h[r] > Math.floor(P.eligibleShare * game.sizes[r]))
        return reject(game, "INVALID_ALLOCATION", `Not enough eligible voters left in ${REGION_LABELS[r]} to hire ${h[r]} workers.`);
    }
    const costCents =
      h.reduce((s, x) => s + x, 0) * toCents(o.costs.worker) +
      REGION_IDS.reduce((s, r, i) => (req.media_regions[i] ? s + toCents(o.costs.media_by_region[r]) : s), 0);
    if (costCents > toCents(o.treasury)) return reject(game, "INSUFFICIENT_FUNDS", "Allocation costs more than the treasury.");

    const t = o.round;
    const b = o.manifesto!;
    const rng = streamOf(o.seed, `campaign:${t}`);
    // P8 cascade (a cheap aggregate stand-in for R24–R27): this round's spreaders are the new hires plus last
    // round's converts; each tries ~mean-degree friends once; chance = receptiveness × P(vote for b); ≤ cap tries each.
    let spreaders = 0;
    let converted = 0;
    let mediaExposed = 0;
    const convertP = [0, 0, 0, 0];
    const exposure = game.reps.map(() => 1); // product of (1 - e) for media
    for (let r = 0; r < 4; r++) {
      const nr = game.sizes[r];
      const idxs = game.repsByRegion[r];
      const sp = h[r] + game.pendingSpreaders[r];
      spreaders += sp;
      const chance =
        idxs.reduce((acc, i) => {
          const rep = game.reps[i];
          return rep.zealot ? acc : acc + rep.gamma * sigmoid(P.betaV * (1 - tv(rep.v, b) - P.z));
        }, 0) / idxs.length;
      const capLeft = Math.max(0, P.attemptCap * nr - game.attempts[r]);
      const tries = Math.min(capLeft, Math.round(sp * Math.min(P.meanDegree, nr - 1) * P.inRegion));
      game.attempts[r] += tries;
      const pool = Math.max(0, nr - game.team[r] - h[r]);
      const conv = Math.min(pool, Math.round(tries * chance * (capLeft / Math.max(1, P.attemptCap * nr))));
      converted += conv;
      game.pendingSpreaders[r] = t < o.rounds ? conv : 0; // R24: converts of the final round never spread
      game.team[r] += h[r];
      convertP[r] = Math.min(1, (conv + h[r]) / nr);
      const aired = req.media_regions[r];
      const airings = o.media_airings[REGION_IDS[r]];
      const etaM = P.etaM0 / (1 + P.lambda * airings);
      if (aired) mediaExposed += Math.round(nr * P.pMedia);
      for (const i of idxs) {
        const rep = game.reps[i];
        if (rep.zealot) continue;
        if (aired && rng() < P.pMedia) exposure[i] *= 1 - etaM * rep.beta;
      }
    }
    const before = game.reps.map((r) => r.v);
    const social = socialStep(game, before, game.reps);
    game.reps.forEach((rep, i) => {
      if (rep.zealot) return;
      const a = P.omega * rep.alpha;
      let c = 1 - exposure[i];
      // R25/R26: a converted voter (or new worker) takes v = b, then drifts with the social step like anyone else
      if (convertP[rep.region] > 0 && rng() < convertP[rep.region]) c = 1 - a;
      if (a + c > 1) c = 1 - a;
      // social part already mixes (1-a)v + a·agg; add campaign/media pull toward b
      const v = social[i].map((x, k) => (1 - c) * x + c * b[k]);
      const s = v.reduce((p, q) => p + q, 0);
      rep.v = v.map((x) => Math.max(0, x / s));
    });

    REGION_IDS.forEach((r, i) => {
      if (req.media_regions[i]) o.media_airings[r] += 1;
    });
    o.treasury = centsToString(toCents(o.treasury) - costCents);
    o.workers = {
      by_region: { N: game.team[0], S: game.team[1], E: game.team[2], W: game.team[3] },
      total: game.team.reduce((x, y) => x + y, 0),
    };
    o.allocations.push({
      round: t,
      hires_by_region: h.slice(),
      media_regions: req.media_regions.slice(),
      cost: centsToString(costCents),
    });
    game.vHistory.push(game.reps.map((r) => r.v.slice()));
    game.trajectory.push(expectedShare(game, game.reps.map((r) => r.v), b));
    event(
      game,
      forfeit ? "forfeit" : "allocate",
      forfeit
        ? `Round ${t}: clock ran out — campaign stood idle.`
        : `Round ${t}: ${spreaders.toLocaleString("en-US")} spreaders converted ${converted.toLocaleString("en-US")} voters, ${mediaExposed.toLocaleString("en-US")} saw ads.`,
    );

    const report: RoundReport = {
      round: t,
      cost: centsToString(costCents),
      hires_by_region: h.slice(),
      team_by_region: game.team.slice(),
      spreaders,
      converted,
      media_exposed: mediaExposed,
      election_held: false,
    };
    o.round_reports.push(report);
    if (t < o.rounds) {
      o.round = t + 1;
      o.free_stats.push({ round: t + 1, top3_pct: top3Pct(game) });
      takePoll(game, t + 1);
    } else {
      takePoll(game, t + 1); // pre-election poll
      const e = game.trajectory[game.trajectory.length - 1];
      const n = o.num_voters;
      const er = streamOf(o.seed, "election");
      const votes = Math.max(0, Math.min(n, Math.round(n * e + Math.sqrt(n * e * (1 - e)) * gauss(er))));
      o.result = { votes, num_voters: n, vote_share: votes / n, victory: votes > n / 2, expected_share: e };
      o.phase = "DONE";
      report.election_held = true;
      event(game, "election", `Election held: ${votes.toLocaleString("en-US")} votes (${((100 * votes) / n).toFixed(1)}%). ${votes > n / 2 ? "VICTORY" : "DEFEAT"}.`);
    }
    legal(game);
    return { accepted: true, error_code: null, message: null, observation: clone(o), round_report: clone(report) };
  }

  function graphSample(game: MockGame, maxNodes: number): GraphSample {
    const n = game.obs.num_voters;
    const total = Math.max(4, Math.min(maxNodes, n));
    const per = game.sizes.map((s) => Math.max(1, Math.round((total * s) / n)));
    const rng = streamOf(game.obs.seed, "graph_sample");
    const nodes: GraphSample["nodes"] = [];
    const repOf: number[] = [];
    const byRegion: number[][] = [[], [], [], []];
    per.forEach((count, r) => {
      for (let k = 0; k < count; k++) {
        byRegion[r].push(nodes.length);
        repOf.push(game.repsByRegion[r][k % game.repsByRegion[r].length]);
        nodes.push({ id: nodes.length * 7 + r, region: REGION_IDS[r], degree: 0 });
      }
    });
    const edges: [number, number][] = [];
    const seen = new Set<string>();
    const target = Math.round(nodes.length * 1.6);
    let guard = 0;
    while (edges.length < target && guard++ < target * 10) {
      const r = Math.floor(rng() * 4);
      const from = byRegion[r][Math.floor(rng() * byRegion[r].length)];
      const rr = rng() < P.inRegion ? r : Math.floor(rng() * 4);
      const to = byRegion[rr][Math.floor(rng() * byRegion[rr].length)];
      if (from === undefined || to === undefined || from === to) continue;
      const key = from < to ? `${from}-${to}` : `${to}-${from}`;
      if (seen.has(key)) continue;
      seen.add(key);
      edges.push(from < to ? [from, to] : [to, from]);
    }
    const degScale = game.obs.regions.map((r) => r.mean_degree);
    for (const [a, b] of edges) {
      nodes[a].degree++;
      nodes[b].degree++;
    }
    nodes.forEach((nd) => {
      const r = REGION_IDS.indexOf(nd.region as RegionId);
      nd.degree = Math.max(1, Math.round(nd.degree * (degScale[r] / 3.2) * (0.6 + 0.8 * rng())));
    });
    const revealed = game.obs.phase === "DONE";
    let alignment: number[][] | null = null;
    if (revealed && game.obs.manifesto) {
      const b = game.obs.manifesto;
      alignment = game.vHistory.map((snap) =>
        repOf.map((ri) => {
          // R15: bin each voter's vote probability at 10/40/60/90% (same as engine.alignment_bins).
          const p = sigmoid(P.betaV * (1 - tv(snap[ri], b) - P.z));
          return p > 0.9 ? 4 : p > 0.6 ? 3 : p > 0.4 ? 2 : p > 0.1 ? 1 : 0;
        }),
      );
    }
    return { revealed, nodes, edges, alignment };
  }

  function debrief(game: MockGame): Debrief {
    const o = game.obs;
    const b = o.manifesto!;
    const regions = REGION_IDS.map((r, i) => {
      const idxs = game.repsByRegion[i];
      const share = (vs: number[][]) => idxs.reduce((s, k) => s + sigmoid(P.betaV * (1 - tv(vs[k], b) - P.z)), 0) / idxs.length;
      return {
        id: r,
        archetype: ARCHETYPES[game.archetypeOf[i]].id,
        personality: PERSONALITIES[game.personalityOf[i]].id,
        zealots: Math.round((idxs.filter((k) => game.reps[k].zealot).length / idxs.length) * game.sizes[i]),
        initial_expected_share: share(game.vHistory[0]),
        final_expected_share: share(game.vHistory[game.vHistory.length - 1]),
      };
    });
    const oracle = oracleManifesto(game);
    return {
      result: clone(o.result),
      regions,
      share_trajectory: game.trajectory.slice(),
      oracle: { manifesto: oracle, expected_share_no_campaign: noCampaignShare(game, oracle), note: "ORACLE: full-information benchmark" },
      your_manifesto_no_campaign_share: noCampaignShare(game, b),
    };
  }

  /**
   * Mock P5 timeline. Not the real Kripke computation: each decision point samples W worlds whose
   * share sits around the true trajectory with a spread that shrinks as knowledge (queries + polls)
   * accumulates, then evaluates a small fixed plan set in each. Deterministic per seed; it respects
   * the contract invariants (guaranteed ⇒ knows-that ⇒ possible; inevitable_defeat = ¬possible).
   */
  function epistemic(game: MockGame): EpistemicTimeline {
    const o = game.obs;
    const T = o.rounds;
    const W = 16;
    const rng = streamOf(o.seed, "epistemic");
    const plans = [
      { label: "Hold back (no campaign)", lift: 0 },
      { label: "Buy workers, spread evenly", lift: 0.006 },
      { label: "Buy workers in the swing region", lift: 0.011 },
      { label: "Workers + media blitz", lift: 0.016 },
    ];
    const win = (share: number) => share > 0.5;
    const spentBefore = (t: number) =>
      o.queries.filter((q) => q.round < t).reduce((a, q) => a + toCents(q.cost), 0) +
      o.allocations.filter((a) => a.round < t).reduce((a, al) => a + toCents(al.cost), 0);
    const rounds: EpistemicRound[] = [];
    for (let t = 1; t <= T; t++) {
      const left = T - t + 1;
      const truthNow = game.trajectory[t - 1] ?? game.trajectory[0] ?? 0.5;
      const queries = o.queries.filter((q) => q.round < t).length;
      const polls = o.polls.filter((p) => p.round <= t).length;
      const spread = 0.07 / Math.sqrt(1 + 0.5 * queries + polls);
      const bias = (rng() - 0.5) * spread;
      const worlds = Array.from({ length: W }, () => truthNow + bias + spread * gauss(rng));
      const finals = plans.map((pl) => worlds.map((w) => w + pl.lift * left));
      const winFrac = finals.map((f) => f.filter(win).length / W);
      let best = 0;
      winFrac.forEach((f, i) => {
        const mean = (k: number) => finals[k].reduce((a, x) => a + x, 0) / W;
        if (f > winFrac[best] || (f === winFrac[best] && mean(i) > mean(best))) best = i;
      });
      const possible = winFrac.some((f) => f > 0);
      const knowsThat = worlds.every((_, w) => finals.some((f) => win(f[w])));
      const guaranteed = winFrac.some((f) => f === 1);
      // entropy (bits) of the posterior over W worlds, discretised into 0.5-pt share bins
      const bins = new Map<number, number>();
      worlds.forEach((w) => bins.set(Math.round(w * 200), (bins.get(Math.round(w * 200)) ?? 0) + 1));
      const entropy = [...bins.values()].reduce((h, c) => h - (c / W) * Math.log2(c / W), 0);
      rounds.push({
        round: t,
        treasury: centsToString(toCents(o.initial_budget) - spentBefore(t)),
        knowledge: { queries, polls, posterior_entropy: Math.round(entropy * 1000) / 1000, consistent_worlds: W },
        possible_victory: possible,
        guaranteed_victory: guaranteed,
        knows_win_exists: knowsThat || guaranteed,
        inevitable_defeat: !possible,
        truth_victory_possible: win(truthNow + plans[plans.length - 1].lift * left),
        best_plan: {
          label: plans[best].label,
          win_fraction: winFrac[best],
          mean_share: finals[best].reduce((a, x) => a + x, 0) / W,
        },
      });
    }
    const r = o.result!;
    return {
      scope: {
        worlds_per_round: W,
        policies_per_round: plans.length,
        world_voters: 1600,
        note: "MOCK: heuristic stand-in for the possible-worlds computation; finite worlds and plans only.",
      },
      rounds,
      actual: { victory: r.victory, vote_share: r.vote_share, expected_share: r.expected_share },
      headline: epistemicHeadline(rounds, r.victory),
    };
  }

  const guard = <T,>(fn: () => T): Promise<T> => {
    try {
      return wait(fn());
    } catch (err) {
      if (err instanceof ApiError) return fail(err.status, String(err.detail ?? err.message));
      return Promise.reject(err);
    }
  };

  return {
    kind: "mock",
    health: () => guard(() => ({ status: "ok" as const, device: "mock (in-browser)", active_games: games.size, max_games: maxGames })),
    createGame: (req) => guard(() => create(req ?? {})),
    observation: (g, p) => guard(() => clone(getGame(g, p).obs)),
    query: (g, p, req) => guard(() => query(getGame(g, p), req.sector_a, req.sector_b, req.region)),
    commit: (g, p, req) => guard(() => commit(getGame(g, p), req.manifesto)),
    allocate: (g, p, req) => guard(() => resolve(getGame(g, p), req, false)),
    forfeit: (g, p) =>
      guard(() => resolve(getGame(g, p), { hires_by_region: [0, 0, 0, 0], media_regions: [false, false, false, false] }, true)),
    graphSample: (g, maxNodes = 1500) => guard(() => graphSample(getGame(g), maxNodes)),
    debrief: (g, p) =>
      guard(() => {
        const game = getGame(g, p);
        if (game.obs.phase !== "DONE") throw new ApiError(409, "Debrief is available only after the election.");
        return debrief(game);
      }),
    epistemic: (g, p) =>
      guard(() => {
        const game = getGame(g, p);
        if (game.obs.phase !== "DONE") throw new ApiError(409, "The epistemic timeline is available only after the election.");
        return epistemic(game);
      }),
    deleteGame: (g) =>
      guard(() => {
        getGame(g);
        games.delete(g);
        return { deleted: true };
      }),
  };
}
