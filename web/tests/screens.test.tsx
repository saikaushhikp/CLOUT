import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { GameApp } from "@/components/GameApp";
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
    expect(screen.getByText("10,000")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText(/engine online/i)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "100k" }));
    expect(screen.getByText("100,000")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/world seed/i), { target: { value: "4242" } });
    fireEvent.click(screen.getByRole("button", { name: /increase campaign rounds/i }));
    fireEvent.click(screen.getByRole("button", { name: /start campaign/i }));

    await waitFor(() => expect(spy).toHaveBeenCalledWith({ num_voters: 100_000, rounds: 6, seed: 4242 }));
    expect(await screen.findByText(/find out what the regions want/i)).toBeInTheDocument();
    expect(screen.getByText(/national top-3/i)).toBeInTheDocument();
  });

  it("rejects a non-numeric seed", () => {
    render(<GameApp client={createMockClient()} />);
    fireEvent.change(screen.getByLabelText(/world seed/i), { target: { value: "abc" } });
    expect(screen.getByRole("button", { name: /start campaign/i })).toBeDisabled();
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
    expect(screen.getByTestId("network-stub")).toBeInTheDocument();

    const north = screen.getByRole("group", { name: "Workers in North" });
    const inc = within(north).getByRole("button", { name: /increase/i });
    for (let i = 0; i < 12; i++) fireEvent.click(inc);
    // bounded by the 10 free workers
    expect(within(north).getByRole("spinbutton")).toHaveValue(10);
    expect(inc).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /increase extra workers/i }));
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
