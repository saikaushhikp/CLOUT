import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { GameApp } from "@/components/GameApp";
import { budgetTextValid, budgetToRequest, clampBudget } from "@/components/SetupScreen";
import { createMockClient } from "@/lib/mockApi";
import type { DumbocracyClient } from "@/lib/api";
import type { Observation } from "@/lib/types";

// The WebGL network view is exercised in the browser, not jsdom.
vi.mock("@/components/NetworkView", () => ({
  NetworkView: () => <div data-testid="network-stub" />,
  default: () => <div data-testid="network-stub" />,
}));

async function roundObservation(client: DumbocracyClient): Promise<Observation> {
  const { game_id, player_id } = await client.createGame({ num_voters: 10_000, seed: 99, rounds: 3 });
  await client.query(game_id, player_id, { sector_a: "Edu", sector_b: "Env", region: "E" });
  const res = await client.commit(game_id, player_id, { manifesto: [0.25, 0.15, 0.2, 0.2, 0.2] });
  return res.observation;
}

describe("setup screen", () => {
  it("renders the setup form and starts a game", async () => {
    const client = createMockClient();
    const spy = vi.spyOn(client, "createGame");
    render(<GameApp client={client} />);
    expect(screen.getByRole("form", { name: /new game setup/i })).toBeInTheDocument();
    expect(screen.getAllByText("10,000")).toHaveLength(2); // big number + game summary
    await waitFor(() => expect(screen.getByText(/engine online/i)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "100k" }));
    expect(screen.getAllByText("100,000")).toHaveLength(2);
    fireEvent.change(screen.getByLabelText(/world seed/i), { target: { value: "4242" } });
    fireEvent.click(screen.getByRole("button", { name: /increase campaign rounds/i }));
    fireEvent.click(screen.getByRole("button", { name: /start campaign/i }));

    await waitFor(() => expect(spy).toHaveBeenCalledWith({ num_voters: 100_000, rounds: 6, seed: 4242, budget: "500.00" }));
    expect(await screen.findByText(/find out what the regions want/i)).toBeInTheDocument();
    expect(screen.getByText(/national top-3/i)).toBeInTheDocument();
  });

  it("lets the player choose the starting treasury (R17): presets, slider, typed value, summary, request", async () => {
    const client = createMockClient();
    const spy = vi.spyOn(client, "createGame");
    render(<GameApp client={client} />);
    const summary = () => screen.getByTestId("summary-budget");
    expect(summary()).toHaveTextContent("Đ500.00"); // default
    expect(screen.getByRole("button", { name: "Đ500" })).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(screen.getByRole("button", { name: "Đ750" }));
    expect(summary()).toHaveTextContent("Đ750.00");
    expect(screen.getByLabelText(/starting treasury in credits/i)).toHaveValue("750");

    const slider = screen.getByLabelText(/party fund/i);
    expect(slider).toHaveAttribute("min", "500");
    expect(slider).toHaveAttribute("max", "1000");
    fireEvent.change(slider, { target: { value: "1000" } });
    expect(summary()).toHaveTextContent("Đ1,000.00");

    // out-of-range typed values block the start button; blur clamps into range
    const field = screen.getByLabelText(/starting treasury in credits/i);
    fireEvent.change(field, { target: { value: "1200" } });
    expect(field).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("button", { name: /start campaign/i })).toBeDisabled();
    fireEvent.blur(field);
    expect(field).toHaveValue("1000");
    fireEvent.change(field, { target: { value: "820" } });
    expect(summary()).toHaveTextContent("Đ820.00");

    fireEvent.click(screen.getByRole("button", { name: /start campaign/i }));
    await waitFor(() => expect(spy).toHaveBeenCalledWith(expect.objectContaining({ budget: "820.00" })));
    expect(await screen.findByText(/find out what the regions want/i)).toBeInTheDocument();
    expect(screen.getByText("Đ820.00")).toBeInTheDocument(); // treasury chip
  });

  it("budget helpers clamp to ₫500–₫1000 and serialise with 2 dp", () => {
    expect(clampBudget(499)).toBe(500);
    expect(clampBudget(1001)).toBe(1000);
    expect(clampBudget(749.6)).toBe(750);
    expect(clampBudget(Number.NaN)).toBe(500);
    expect(budgetTextValid("500")).toBe(true);
    expect(budgetTextValid("1000")).toBe(true);
    expect(budgetTextValid("499")).toBe(false);
    expect(budgetTextValid("1000.5")).toBe(false);
    expect(budgetTextValid("")).toBe(false);
    expect(budgetToRequest(750)).toBe("750.00");
    expect(budgetToRequest(5000)).toBe("1000.00");
  });

  it("rejects a non-numeric seed", () => {
    render(<GameApp client={createMockClient()} />);
    fireEvent.change(screen.getByLabelText(/world seed/i), { target: { value: "abc" } });
    expect(screen.getByRole("button", { name: /start campaign/i })).toBeDisabled();
  });
});

describe("survey desk — sampled queries (R12)", () => {
  it("shows answers as '% ± MoE (n = m)', pools repeats and invites asking again", async () => {
    const client = createMockClient();
    const { game_id, player_id, observation } = await client.createGame({ num_voters: 10_000, seed: 99, rounds: 3 });
    const query = vi.spyOn(client, "query");
    render(<GameApp client={client} initialObservation={observation} />);
    const m = Math.round(0.1 * observation.regions.find((r) => r.id === "N")!.voters);

    // the builder announces the ~10% sample size of the selected region (North by default)
    expect(screen.getByTestId("qb-sample")).toHaveTextContent(`≈${m.toLocaleString("en-US")}`);
    expect(screen.getByTestId("qb-sample")).toHaveTextContent(/narrows the estimate/i);

    const askBtn = () => screen.getByRole("button", { name: /^ask( again)? · /i });
    fireEvent.click(askBtn());
    await waitFor(() => expect(query).toHaveBeenCalledTimes(1));
    const ledger = await screen.findByRole("list", { name: /your survey answers/i });
    await waitFor(() => expect(within(ledger).getByTestId("ledger-estimate")).toHaveTextContent(new RegExp(`^\\d+\\.\\d% ± \\d+\\.\\d \\(n = ${m}\\)$`)));

    // repeat: allowed, not blocked, and pooled into one ledger row
    expect(askBtn()).toHaveTextContent(/ask again/i);
    expect(askBtn()).toBeEnabled();
    expect(screen.getByTestId("qb-sample")).toHaveTextContent(/ask again to narrow the margin/i);
    fireEvent.click(askBtn());
    await waitFor(() => expect(query).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(within(ledger).getByTestId("ledger-estimate")).toHaveTextContent(`(n = ${2 * m})`));
    expect(within(ledger).getAllByTestId("ledger-estimate")).toHaveLength(1);
    expect(within(ledger).getByText(/pooled ×2/)).toBeInTheDocument();
    expect(within(ledger).getByRole("list", { name: /individual samples/i }).querySelectorAll("li")).toHaveLength(2);
    expect(query).toHaveBeenLastCalledWith(game_id, player_id, { sector_a: "Edu", sector_b: "Agri", region: "N" });
  });
});

describe("campaign screen", () => {
  it("renders HQ from an observation and submits an allocation", async () => {
    const client = createMockClient();
    const obs = await roundObservation(client);
    const allocate = vi.spyOn(client, "allocate");
    render(<GameApp client={client} initialObservation={obs} />);

    expect(screen.getByRole("heading", { name: /round 1 of 3/i })).toBeInTheDocument();
    expect(screen.getByRole("group", { name: /map of the four regions/i })).toBeInTheDocument();
    expect(screen.getByText(/latest poll/i)).toBeInTheDocument();
    // R18: no network panel during the rounds
    expect(screen.queryByTestId("network-stub")).toBeNull();
    expect(screen.queryByRole("heading", { name: /social network/i })).toBeNull();
    // the committed manifesto is always on screen
    const manifesto = screen.getByTestId("manifesto-summary");
    expect(within(manifesto).getByRole("img", { name: /manifesto: education 25%, welfare 15%, agriculture 20%, industry 20%, environment 20%/i })).toBeInTheDocument();
    expect(within(manifesto).getByText("Education").closest("li")).toHaveTextContent("25%");
    // hiring copy (R16)
    expect(screen.getByText(/one-time fee, works every remaining round/i)).toBeInTheDocument();
    expect(screen.getByTestId("team-line")).toHaveTextContent("Team: 10 free = 10 available");

    const north = screen.getByRole("group", { name: "Workers in North" });
    const inc = within(north).getByRole("button", { name: /increase/i });
    for (let i = 0; i < 12; i++) fireEvent.click(inc);
    // bounded by the 10 free workers
    expect(within(north).getByRole("spinbutton")).toHaveValue(10);
    expect(inc).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /increase new hires/i }));
    expect(screen.getByTestId("team-line")).toHaveTextContent("Team: 10 free + 1 new hire = 11 available");
    fireEvent.click(screen.getByRole("button", { name: /air media in south/i }));
    const southPrice = obs.costs.media_by_region.S;
    const expected = (4 + Number(southPrice)).toFixed(2);
    expect(screen.getByText("This round").nextSibling).toHaveTextContent(expected);

    fireEvent.click(screen.getByRole("button", { name: /launch round 1/i }));
    await waitFor(() =>
      expect(allocate).toHaveBeenCalledWith(obs.game_id, "p0", {
        workers_by_region: [10, 0, 0, 0],
        extra_workers: 1,
        media_regions: [false, true, false, false],
      }),
    );
    expect(await screen.findByRole("heading", { name: /round 2 of 3/i })).toBeInTheDocument();
    // R16: the hire is permanent — round 2 starts with 11 workers and no fee for them
    expect(screen.getByTestId("team-line")).toHaveTextContent("Team: 10 free + 1 hired = 11 available");
    expect(screen.getByText(/0\/11 deployed/)).toBeInTheDocument();
    expect(screen.getByTestId("manifesto-summary")).toBeInTheDocument();
  });

  it("explains how workers are hired and used", async () => {
    const client = createMockClient();
    const obs = await roundObservation(client);
    render(<GameApp client={client} initialObservation={obs} />);
    const summary = screen.getByText("How it works");
    const details = summary.closest("details")!;
    expect(details.open).toBe(false);
    fireEvent.click(summary);
    const note = within(details).getByRole("note", { name: /how workers are hired and used/i });
    expect(note).toHaveTextContent(/permanent team/i);
    expect(note).toHaveTextContent(/0\.25% of the 10k voters \(25 doorsteps\)/);
    expect(note).toHaveTextContent(/uniformly at random from that region \(with replacement/);
    expect(note).toHaveTextContent(/campaign susceptibility/);
    expect(note).toHaveTextContent(/up to 3 hops/);
  });

  it("shows a friendly toast when the engine rejects a move", async () => {
    const client = createMockClient();
    const obs = await roundObservation(client);
    vi.spyOn(client, "allocate").mockResolvedValue({
      accepted: false,
      error_code: "INSUFFICIENT_FUNDS",
      message: "Allocation costs more than the treasury.",
      observation: obs,
    });
    render(<GameApp client={client} initialObservation={obs} />);
    fireEvent.click(screen.getByRole("button", { name: /launch round 1/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/not enough money/i);
  });
});

describe("round timer", () => {
  it("calls /forfeit when the clock runs out", async () => {
    const client = createMockClient();
    const obs = await roundObservation(client);
    const forfeit = vi.spyOn(client, "forfeit");
    vi.useFakeTimers();
    try {
      render(<GameApp client={client} initialObservation={obs} />);
      fireEvent.change(screen.getByLabelText(/seconds per round/i), { target: { value: "30" } });
      fireEvent.click(screen.getByRole("switch", { name: /round clock/i }));
      expect(screen.getByRole("timer")).toHaveTextContent("30s left");
      await act(async () => {
        vi.advanceTimersByTime(29_000);
      });
      expect(forfeit).not.toHaveBeenCalled();
      await act(async () => {
        vi.advanceTimersByTime(1_000);
      });
      expect(forfeit).toHaveBeenCalledTimes(1);
      expect(forfeit).toHaveBeenCalledWith(obs.game_id, "p0");
      await act(async () => {
        await Promise.resolve();
        vi.advanceTimersByTime(50);
      });
    } finally {
      vi.useRealTimers();
    }
    expect(await screen.findByRole("heading", { name: /round 2 of 3/i })).toBeInTheDocument();
  });
});
