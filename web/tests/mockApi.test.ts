import { describe, expect, it } from "vitest";
import { createMockClient } from "@/lib/mockApi";
import { ApiError } from "@/lib/errors";
import { toCents } from "@/lib/money";

const none = { workers_by_region: [0, 0, 0, 0], extra_workers: 0, media_regions: [false, false, false, false] };

describe("mock engine", () => {
  it("(N, seed) recreates the same public electorate", async () => {
    const c = createMockClient();
    const a = await c.createGame({ num_voters: 12_345, seed: 7 });
    const b = await c.createGame({ num_voters: 12_345, seed: 7 });
    const d = await c.createGame({ num_voters: 12_345, seed: 8 });
    expect(a.observation.regions).toEqual(b.observation.regions);
    expect(a.observation.free_stats).toEqual(b.observation.free_stats);
    expect(a.observation.regions).not.toEqual(d.observation.regions);
    const sizes = a.observation.regions.map((r) => r.voters);
    expect(sizes.reduce((x, y) => x + y, 0)).toBe(12_345);
    sizes.forEach((s) => expect(s).toBeGreaterThanOrEqual(Math.ceil(0.15 * 12_345)));
    expect(a.observation.treasury).toBe("500.00");
    expect(a.observation.initial_budget).toBe("500.00");
    expect(a.observation.workers).toEqual({ free: 10, hired: 0, available: 10 });
    expect(a.observation.round_reports).toEqual([]);
    expect(a.observation.phase).toBe("SURVEY");
    expect(a.observation.legal_actions).toEqual(["query", "commit"]);
    const top3 = Object.values(a.observation.free_stats[0].top3_pct).reduce((x, y) => x + y, 0);
    expect(top3).toBeCloseTo(300, 0);
  });

  it("returns a random seed when omitted and validates inputs", async () => {
    const c = createMockClient();
    const g = await c.createGame({});
    expect(Number.isInteger(g.observation.seed)).toBe(true);
    expect(g.observation.num_voters).toBe(10_000);
    expect(g.observation.rounds).toBe(5);
    await expect(c.createGame({ num_voters: 10 })).rejects.toBeInstanceOf(ApiError);
    await expect(c.observation("nope", "p0")).rejects.toMatchObject({ status: 404 });
  });

  it("plays a full game to DONE with contract-shaped responses", async () => {
    const c = createMockClient();
    const { game_id: g, player_id: p, observation } = await c.createGame({ num_voters: 5000, seed: 123, rounds: 3 });
    expect(p).toBe("p0");

    // SURVEY: queries, repeats (allowed, R12), bad queries, wrong-phase allocate
    const q1 = await c.query(g, p, { sector_a: "Edu", sector_b: "Agri", region: "N" });
    expect(q1.accepted).toBe(true);
    expect(q1.answer?.pct_b_over_a).toBeGreaterThanOrEqual(0);
    expect(q1.answer?.pct_b_over_a).toBeLessThanOrEqual(100);
    expect(q1.observation.treasury).toBe("495.00");
    const rep = await c.query(g, p, { sector_a: "Edu", sector_b: "Agri", region: "N" });
    expect(rep).toMatchObject({ accepted: true, error_code: null });
    expect(rep.answer?.repeat_index).toBe(2);
    expect(rep.observation.treasury).toBe("490.00");
    expect((await c.query(g, p, { sector_a: "Edu", sector_b: "Edu", region: "N" })).error_code).toBe("INVALID_QUERY");
    expect((await c.allocate(g, p, none)).error_code).toBe("INVALID_PHASE");
    await expect(c.debrief(g, p)).rejects.toMatchObject({ status: 409 });
    expect((await c.graphSample(g)).revealed).toBe(false);

    // commit
    expect((await c.commit(g, p, { manifesto: [0.5, 0.5, 0.5, 0, 0] })).error_code).toBe("INVALID_MANIFESTO");
    const committed = await c.commit(g, p, { manifesto: [0.2, 0.2, 0.3, 0.2, 0.1] });
    expect(committed.accepted).toBe(true);
    expect(committed.observation).toMatchObject({ phase: "ROUND", round: 1 });
    expect(committed.observation.polls).toHaveLength(1);
    expect(committed.observation.polls[0].round).toBe(1);
    expect(committed.observation.free_stats).toHaveLength(2);
    expect((await c.commit(g, p, { manifesto: [0.2, 0.2, 0.2, 0.2, 0.2] })).error_code).toBe("INVALID_PHASE");

    // round 1: over-assign, then a real allocation with media + extras
    expect((await c.allocate(g, p, { ...none, workers_by_region: [11, 0, 0, 0] })).error_code).toBe("INVALID_ALLOCATION");
    expect((await c.allocate(g, p, { ...none, extra_workers: 200 })).error_code).toBe("INSUFFICIENT_FUNDS");
    const before = toCents(committed.observation.treasury);
    const media = toCents(committed.observation.costs.media_by_region.N);
    const r1 = await c.allocate(g, p, { workers_by_region: [4, 4, 3, 1], extra_workers: 2, media_regions: [true, false, false, false] });
    expect(r1.accepted).toBe(true);
    expect(r1.round_report).toMatchObject({ round: 1, election_held: false });
    expect(toCents(r1.round_report!.cost)).toBe(800 + media);
    expect(toCents(r1.observation.treasury)).toBe(before - 800 - media);
    expect(r1.observation.round).toBe(2);
    expect(r1.observation.media_airings.N).toBe(1);
    expect(r1.observation.allocations).toHaveLength(1);

    // round 2 forfeit, round 3 resolves the election
    const r2 = await c.forfeit(g, p);
    expect(r2.round_report).toMatchObject({ round: 2, cost: "0.00", direct_contacts: 0 });
    const r3 = await c.allocate(g, p, { workers_by_region: [10, 0, 0, 0], extra_workers: 0, media_regions: [false, false, false, false] });
    expect(r3.round_report?.election_held).toBe(true);
    const done = r3.observation;
    expect(done.phase).toBe("DONE");
    expect(done.round).toBe(3);
    expect(done.legal_actions).toEqual([]);
    expect(done.polls.map((x) => x.round)).toEqual([1, 2, 3, 4]);
    expect(done.result).not.toBeNull();
    expect(done.result!.victory).toBe(done.result!.votes > done.num_voters / 2);
    expect((await c.query(g, p, { sector_a: "Edu", sector_b: "Agri", region: "S" })).error_code).toBe("TERMINAL_STATE");

    // debrief + revealed graph
    const d = await c.debrief(g, p);
    expect(d.regions).toHaveLength(4);
    expect(d.share_trajectory).toHaveLength(4);
    expect(d.oracle.note).toBe("ORACLE: full-information benchmark");
    expect(d.oracle.manifesto.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 2);
    const archetypes = new Set(d.regions.map((r) => r.archetype));
    expect(archetypes.size).toBe(4);
    const gs = await c.graphSample(g, 400);
    expect(gs.revealed).toBe(true);
    expect(gs.nodes.length).toBeLessThanOrEqual(404);
    expect(gs.alignment).toHaveLength(4);
    gs.alignment!.forEach((row) => expect(row).toHaveLength(gs.nodes.length));
    gs.edges.forEach(([a, b]) => {
      expect(a).toBeLessThan(gs.nodes.length);
      expect(b).toBeLessThan(gs.nodes.length);
    });
    expect(observation.game_id).toBe(g);
    expect(await c.deleteGame(g)).toEqual({ deleted: true });
  });

  it("accepts a starting treasury of 500.00..1000.00 (R17)", async () => {
    const c = createMockClient();
    const g = await c.createGame({ num_voters: 2000, seed: 1, budget: "750.00" });
    expect(g.observation.treasury).toBe("750.00");
    expect(g.observation.initial_budget).toBe("750.00");
    expect((await c.createGame({ num_voters: 2000, seed: 1, budget: "1000.00" })).observation.treasury).toBe("1000.00");
    for (const bad of ["499.99", "1000.01", "abc", "750.123"]) {
      await expect(c.createGame({ num_voters: 2000, seed: 1, budget: bad })).rejects.toMatchObject({ status: 422 });
    }
  });

  it("hires are permanent: one-time fee, then they join the pool every later round (R16) + round_reports (R19)", async () => {
    const c = createMockClient();
    const { game_id: g, player_id: p } = await c.createGame({ num_voters: 5000, seed: 11, rounds: 3 });
    const committed = await c.commit(g, p, { manifesto: [0.2, 0.2, 0.2, 0.2, 0.2] });
    const t0 = toCents(committed.observation.treasury);
    const r1 = await c.allocate(g, p, { ...none, workers_by_region: [5, 5, 3, 0], extra_workers: 3 });
    expect(r1.accepted).toBe(true);
    expect(toCents(r1.observation.treasury)).toBe(t0 - 1200);
    expect(r1.observation.workers).toEqual({ free: 10, hired: 3, available: 13 });
    expect(r1.observation.round_reports).toHaveLength(1);
    expect(r1.observation.round_reports[0]).toEqual(r1.round_report);

    // round 2: 13 workers at no cost; 14 needs a new hire
    expect((await c.allocate(g, p, { ...none, workers_by_region: [14, 0, 0, 0] })).error_code).toBe("INVALID_ALLOCATION");
    const r2 = await c.allocate(g, p, { ...none, workers_by_region: [13, 0, 0, 0] });
    expect(r2.accepted).toBe(true);
    expect(r2.round_report?.cost).toBe("0.00");
    expect(r2.observation.treasury).toBe(r1.observation.treasury);
    expect(r2.observation.workers.available).toBe(13);
    const r3 = await c.allocate(g, p, { ...none, workers_by_region: [10, 4, 0, 0], extra_workers: 1 });
    expect(r3.accepted).toBe(true);
    expect(r3.observation.phase).toBe("DONE");
    expect(r3.observation.workers).toEqual({ free: 10, hired: 4, available: 14 });
    expect(r3.observation.round_reports.map((r) => r.round)).toEqual([1, 2, 3]);
    expect(r3.observation.round_reports[2].election_held).toBe(true);
  });

  it("never leaks hidden fields in observations", async () => {
    const c = createMockClient();
    const { observation } = await c.createGame({ num_voters: 2000, seed: 3 });
    const json = JSON.stringify(observation);
    for (const banned of ["archetypeOf", "personalityOf", "zealot", "alpha", "reps", "vHistory"]) expect(json).not.toContain(banned);
  });
});
