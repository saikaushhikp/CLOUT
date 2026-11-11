import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import { EpistemicPanel, EpistemicTimelineView } from "@/components/EpistemicTimeline";
import { GameApp } from "@/components/GameApp";
import { createMockClient } from "@/lib/mockApi";
import type { DumbocracyClient } from "@/lib/api";
import type { EpistemicTimeline } from "@/lib/types";
import { centsToString, toCents } from "@/lib/money";

vi.mock("@/components/NetworkView", () => ({
  NetworkView: () => <div data-testid="network-stub" />,
  default: () => <div data-testid="network-stub" />,
}));

const round = (t: number, flags: Partial<Record<string, boolean>>, plan = "Workers + media blitz", frac = 0.5) => ({
  round: t,
  treasury: (500 - 40 * t).toFixed(2),
  knowledge: { queries: 2 + t, polls: t, posterior_entropy: 3 - 0.5 * t, consistent_worlds: 16 },
  possible_victory: false,
  guaranteed_victory: false,
  knows_win_exists: false,
  inevitable_defeat: false,
  truth_victory_possible: false,
  ...flags,
  best_plan: { label: plan, win_fraction: frac, mean_share: 0.49 + 0.01 * t },
});

const FIXTURE: EpistemicTimeline = {
  scope: { worlds_per_round: 16, policies_per_round: 9, world_voters: 1600, note: "Finite worlds and plans." },
  rounds: [
    round(1, { truth_victory_possible: true, possible_victory: true }, "Hold back", 0.25),
    round(2, { truth_victory_possible: true, possible_victory: true, knows_win_exists: true }, "Canvass South", 0.75),
    round(3, { truth_victory_possible: true, possible_victory: true, knows_win_exists: true, guaranteed_victory: true }, "Blitz East", 1),
    round(4, { inevitable_defeat: true }, "Hold back", 0),
  ],
  actual: { victory: false, vote_share: 0.493, expected_share: 0.497 },
  headline: "Victory slipped out of reach in round 4; you knew a winning plan from round 3.",
};

const SCHEMA_ROUND_KEYS = [
  "best_plan",
  "guaranteed_victory",
  "inevitable_defeat",
  "knowledge",
  "knows_win_exists",
  "possible_victory",
  "round",
  "treasury",
  "truth_victory_possible",
];

function assertSchema(t: EpistemicTimeline, rounds: number) {
  expect(Object.keys(t).sort()).toEqual(["actual", "headline", "rounds", "scope"]);
  expect(Object.keys(t.scope).sort()).toEqual(["note", "policies_per_round", "world_voters", "worlds_per_round"]);
  expect(Object.keys(t.actual).sort()).toEqual(["expected_share", "victory", "vote_share"]);
  expect(typeof t.headline).toBe("string");
  expect(t.headline.length).toBeGreaterThan(10);
  expect(t.rounds.map((r) => r.round)).toEqual(Array.from({ length: rounds }, (_, i) => i + 1));
  for (const r of t.rounds) {
    expect(Object.keys(r).sort()).toEqual(SCHEMA_ROUND_KEYS);
    expect(Object.keys(r.knowledge).sort()).toEqual(["consistent_worlds", "polls", "posterior_entropy", "queries"]);
    expect(Object.keys(r.best_plan).sort()).toEqual(["label", "mean_share", "win_fraction"]);
    expect(r.treasury).toMatch(/^\d+\.\d{2}$/);
    expect(r.best_plan.win_fraction).toBeGreaterThanOrEqual(0);
    expect(r.best_plan.win_fraction).toBeLessThanOrEqual(1);
    expect(r.knowledge.posterior_entropy).toBeGreaterThanOrEqual(0);
    // modal invariants: knowing-how ⇒ knowing-that ⇒ possible; defeat inevitable ⇔ ¬possible
    if (r.guaranteed_victory) expect(r.knows_win_exists).toBe(true);
    if (r.knows_win_exists) expect(r.possible_victory).toBe(true);
    expect(r.inevitable_defeat).toBe(!r.possible_victory);
  }
}

describe("epistemic timeline view", () => {
  it("renders the headline, the truth row and a badge per round and modality", () => {
    render(<EpistemicTimelineView timeline={FIXTURE} />);
    expect(screen.getByTestId("ep-headline")).toHaveTextContent(FIXTURE.headline);
    expect(screen.getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["Start of round", "R1", "R2", "R3", "R4"]);

    const truth = screen.getByTestId("ep-row-truth_victory_possible");
    expect(within(truth).getByRole("rowheader")).toHaveTextContent(/truth: victory still possible/i);
    expect(within(truth).getAllByText(/^(yes|no)$/i).map((b) => b.textContent)).toEqual(["Yes", "Yes", "Yes", "No"]);

    const on = (key: string) =>
      within(screen.getByTestId(`ep-row-${key}`))
        .getAllByLabelText(/^Round \d/)
        .map((b) => b.getAttribute("data-on") === "true");
    expect(on("possible_victory")).toEqual([true, true, true, false]);
    expect(on("knows_win_exists")).toEqual([false, true, true, false]);
    expect(on("guaranteed_victory")).toEqual([false, false, true, false]);
    expect(on("inevitable_defeat")).toEqual([false, false, false, true]);
    expect(screen.getByLabelText("Round 3: Knew a winning plan — yes")).toBeInTheDocument();
    expect(screen.getByLabelText("Round 4: Defeat inevitable — yes")).toBeInTheDocument();
    expect(within(screen.getByTestId("ep-row-guaranteed_victory")).getByText(/knowing how/i)).toBeInTheDocument();
    expect(within(screen.getByTestId("ep-row-knows_win_exists")).getByText(/knowing that/i)).toBeInTheDocument();

    // per-round plan + knowledge stats
    expect(screen.getByText("Canvass South")).toBeInTheDocument();
    expect(screen.getByText("75%")).toBeInTheDocument();
    expect(screen.getByText("5 q · 3 polls")).toBeInTheDocument();
    expect(screen.getAllByText("16")).toHaveLength(4);
    expect(screen.getByText("1.44 bits")).toBeInTheDocument(); // 1.00 nats from the engine = 1.44 bits
    expect(screen.getByText("Lost")).toBeInTheDocument();

    // explainer ties worlds to K / M
    expect(screen.getByText("How to read this")).toBeInTheDocument();
    expect(screen.getByText(/consistent with everything you had seen/i)).toBeInTheDocument();
  });

  it("shows a loading state until the timeline arrives, and errors with a retry", () => {
    const { rerender } = render(<EpistemicPanel timeline={null} error={null} />);
    expect(screen.getByRole("heading", { name: "Epistemic timeline" })).toBeInTheDocument();
    expect(screen.getByTestId("ep-loading")).toHaveTextContent("Reasoning over possible worlds…");
    const retry = vi.fn();
    rerender(<EpistemicPanel timeline={null} error="boom" onRetry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("boom");
    screen.getByRole("button", { name: /try again/i }).click();
    expect(retry).toHaveBeenCalledOnce();
    rerender(<EpistemicPanel timeline={FIXTURE} error={null} />);
    expect(screen.queryByTestId("ep-loading")).toBeNull();
    expect(screen.getByTestId("ep-headline")).toBeInTheDocument();
  });
});

async function finishedGame(client: DumbocracyClient, seed = 31, rounds = 3) {
  const { game_id, player_id } = await client.createGame({ num_voters: 5000, seed, rounds });
  await client.query(game_id, player_id, { sector_a: "Edu", sector_b: "Env", region: "S" });
  let obs = (await client.commit(game_id, player_id, { manifesto: [0.2, 0.2, 0.2, 0.2, 0.2] })).observation;
  const plan = { hires_by_region: [1, 1, 0, 0], media_regions: [false, false, false, false] };
  for (let t = 0; t < rounds; t++) obs = (await client.allocate(game_id, player_id, plan)).observation;
  expect(obs.phase).toBe("DONE");
  return obs;
}

describe("mock engine — epistemic timeline", () => {
  it("is 409 before DONE, schema-valid and deterministic per seed after", async () => {
    const c = createMockClient();
    const { game_id, player_id } = await c.createGame({ num_voters: 5000, seed: 31, rounds: 3 });
    await expect(c.epistemic(game_id, player_id)).rejects.toMatchObject({ status: 409 });
    await expect(c.epistemic("nope", "p0")).rejects.toMatchObject({ status: 404 });

    const obs = await finishedGame(c);
    const t = await c.epistemic(obs.game_id, obs.player_id);
    assertSchema(t, 3);
    expect(t.actual.victory).toBe(obs.result!.victory);
    expect(t.rounds[0].treasury).toBe(centsToString(toCents(obs.initial_budget) - toCents(obs.costs.query)));
    expect(t.rounds[0].knowledge).toMatchObject({ queries: 1, polls: 1 });
    expect(t.rounds[2].knowledge.polls).toBe(3);

    const c2 = createMockClient();
    const o2 = await finishedGame(c2);
    expect(await c2.epistemic(o2.game_id, o2.player_id)).toEqual(t);
  });

  it("the results screen loads the timeline lazily after the debrief", async () => {
    const client = createMockClient();
    const obs = await finishedGame(client, 77, 3);
    let release: () => void = () => undefined;
    const gate = new Promise<void>((res) => (release = res));
    const real = client.epistemic.bind(client);
    const spy = vi.spyOn(client, "epistemic").mockImplementation(async (g, p) => {
      await gate;
      return real(g, p);
    });
    render(<GameApp client={client} initialObservation={obs} />);
    expect(await screen.findByText("The hidden map, revealed")).toBeInTheDocument();
    expect(await screen.findByTestId("ep-loading")).toHaveTextContent(/reasoning over possible worlds/i);
    expect(spy).toHaveBeenCalledWith(obs.game_id, obs.player_id);
    release();
    expect(await screen.findByTestId("ep-headline")).toBeInTheDocument();
    expect(screen.getByTestId("ep-row-truth_victory_possible")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByTestId("ep-loading")).toBeNull());
    expect(spy).toHaveBeenCalledTimes(1);
  });
});
