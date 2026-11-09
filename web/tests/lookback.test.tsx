import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { GameApp } from "@/components/GameApp";
import { createMockClient } from "@/lib/mockApi";
import { reviewableRounds, roundSummary } from "@/lib/rounds";
import type { Observation, QueryAnswer } from "@/lib/types";

vi.mock("@/components/NetworkView", () => ({ NetworkView: () => null, default: () => null }));

const q = (round: number, phase: "SURVEY" | "ROUND", region: string, pct: number): QueryAnswer => ({
  round,
  phase,
  sector_a: "Edu",
  sector_b: "Env",
  region,
  pct_b_over_a: pct,
  sample_size: 250,
  moe: 5.9,
  pooled_pct: pct,
  pooled_sample_size: 250,
  pooled_moe: 5.9,
  repeat_index: 1,
  cost: "5.00",
});

/** Fixture: a 750-credit game now planning round 3, with two resolved rounds. */
async function fixture(): Promise<Observation> {
  const client = createMockClient();
  const { observation } = await client.createGame({ num_voters: 10_000, seed: 5, rounds: 4, budget: "750.00" });
  return {
    ...observation,
    phase: "ROUND",
    round: 3,
    manifesto: [0.3, 0.1, 0.2, 0.2, 0.2],
    legal_actions: ["query", "allocate", "forfeit"],
    initial_budget: "750.00",
    // spent: survey 1 query (5) + R1 orders 50 + R2 query 5 + R2 orders 8 = 68
    treasury: "682.00",
    workers: { free: 10, hired: 2, available: 12 },
    queries: [q(0, "SURVEY", "N", 41.2), q(2, "ROUND", "E", 63.6)],
    polls: [
      { round: 1, estimate: 0.462, moe: 0.049, sample_size: 400 },
      { round: 2, estimate: 0.481, moe: 0.049, sample_size: 400 },
      { round: 3, estimate: 0.507, moe: 0.049, sample_size: 400 },
    ],
    allocations: [
      { round: 1, workers_by_region: [4, 3, 2, 1], extra_workers: 0, media_regions: [true, false, false, false], cost: "50.00" },
      { round: 2, workers_by_region: [6, 2, 2, 2], extra_workers: 2, media_regions: [false, false, false, false], cost: "8.00" },
    ],
    round_reports: [
      { round: 1, cost: "50.00", direct_contacts: 250, word_of_mouth_reached: 611, media_exposed: 1503, election_held: false },
      { round: 2, cost: "8.00", direct_contacts: 300, word_of_mouth_reached: 702, media_exposed: 0, election_held: false },
    ],
  };
}

describe("roundSummary (R19)", () => {
  it("rebuilds poll, orders, outcomes, queries and treasury for a past round", async () => {
    const obs = await fixture();
    expect(reviewableRounds(obs)).toEqual([1, 2]);
    const r1 = roundSummary(obs, 1);
    expect(r1.pollStart?.estimate).toBe(0.462);
    expect(r1.pollAfter?.estimate).toBe(0.481);
    expect(r1.report?.word_of_mouth_reached).toBe(611);
    expect(r1.queries).toEqual([]);
    expect(r1.treasuryBeforeCents).toBe(74_500); // 750 − 5 survey
    expect(r1.treasuryAfterCents).toBe(69_500);
    expect(r1.teamBefore).toBe(10);
    const r2 = roundSummary(obs, 2);
    expect(r2.treasuryBeforeCents).toBe(69_500);
    expect(r2.queryCostCents).toBe(500);
    expect(r2.treasuryAfterCents).toBe(68_200); // = current treasury
    expect(r2.newHires).toBe(2);
    expect(r2.workersAssigned).toBe(12);
    expect(r2.queries.map((x) => x.region)).toEqual(["E"]);
  });
});

describe("round look-back UI (R19)", () => {
  it("opens a read-only view of a past round from the round tracker and closes back to planning", async () => {
    const obs = await fixture();
    render(<GameApp client={createMockClient()} initialObservation={obs} />);
    expect(screen.getByRole("heading", { name: /round 3 of 4/i })).toBeInTheDocument();
    const tracker = screen.getByRole("list", { name: /campaign rounds/i });
    // future rounds are not buttons
    expect(within(tracker).queryByRole("button", { name: /review round 4/i })).toBeNull();

    const chip = within(tracker).getByRole("button", { name: "Review round 2" });
    chip.focus();
    fireEvent.click(chip);
    const dialog = screen.getByRole("dialog", { name: /round 2 in review/i });
    // poll at start ± MoE, next poll
    expect(within(dialog).getByText("48.1%")).toBeInTheDocument();
    expect(within(dialog).getByText(/±4\.9 pts · n = 400/)).toBeInTheDocument();
    expect(within(dialog).getByText("50.7%")).toBeInTheDocument();
    // treasury before/after
    expect(within(dialog).getByText("Đ695.00")).toBeInTheDocument();
    expect(within(dialog).getByText("Đ682.00")).toBeInTheDocument();
    // orders: workers per region, new hires, media, cost
    const north = within(dialog).getByRole("rowheader", { name: /north/i }).closest("tr")!;
    expect(north).toHaveTextContent("6");
    expect(within(dialog).getByText(/10 on staff \+ 2 new permanent hires · 12 deployed/)).toBeInTheDocument();
    expect(within(dialog).getByText(/Media: none/)).toBeInTheDocument();
    // outcomes from round_reports
    expect(within(dialog).getByText("702")).toBeInTheDocument();
    expect(within(dialog).getByText("300")).toBeInTheDocument();
    // queries asked that round
    expect(within(dialog).getByText(/Questions asked \(1\)/)).toBeInTheDocument();
    expect(within(dialog).getByText(/East: 63\.6% ± 5\.9 \(n = 250\) prefer Environment over Education/)).toBeInTheDocument();
    // focus moved into the dialog
    expect(within(dialog).getByRole("button", { name: /close round 2 review/i })).toHaveFocus();

    // navigate to round 1 inside the dialog
    fireEvent.click(within(dialog).getByRole("button", { name: "Review round 1" }));
    const d1 = screen.getByRole("dialog", { name: /round 1 in review/i });
    expect(within(d1).getByText(/Media: North/)).toBeInTheDocument();
    expect(within(d1).getByText("1,503")).toBeInTheDocument();
    expect(within(d1).getByText(/No paid questions this round/)).toBeInTheDocument();

    // Escape closes and returns to planning; focus goes back to the chip
    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(chip).toHaveFocus();
    expect(screen.getByRole("form", { name: /round 3 allocation/i })).toBeInTheDocument();

    // the current-round chip (inside the sheet, and in the tracker) returns to planning
    fireEvent.click(within(tracker).getByRole("button", { name: "Review round 1" }));
    const again = screen.getByRole("dialog", { name: /round 1 in review/i });
    fireEvent.click(within(again).getByRole("button", { name: /round 3 — back to planning/i }));
    expect(screen.queryByRole("dialog")).toBeNull();
    fireEvent.click(within(tracker).getByRole("button", { name: "Review round 2" }));
    expect(screen.getByRole("dialog", { name: /round 2 in review/i })).toBeInTheDocument();
    fireEvent.click(within(tracker).getByRole("button", { name: /round 3 — planning now/i }));
    expect(screen.queryByRole("dialog")).toBeNull();
    // keyboard: Enter on a chip opens it (native button)
    expect(within(tracker).getByRole("button", { name: "Review round 1" }).tagName).toBe("BUTTON");
  });

  it("the orders panel uses workers.available (free + earlier hires)", async () => {
    const obs = await fixture();
    render(<GameApp client={createMockClient()} initialObservation={obs} />);
    expect(screen.getByTestId("team-line")).toHaveTextContent("Team: 10 free + 2 hired = 12 available");
    expect(screen.getByText(/0\/12 deployed/)).toBeInTheDocument();
  });
});
