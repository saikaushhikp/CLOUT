import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { GameApp } from "@/components/GameApp";
import { createMockClient } from "@/lib/mockApi";

vi.mock("@/components/NetworkView", () => ({
  NetworkView: ({ sample }: { sample: unknown }) => <div data-testid="network-stub">{sample ? "sample" : "none"}</div>,
  default: () => null,
}));

// R18: the sampled network graph is fetched and rendered ONLY in the post-election debrief. No graph_sample
// request during the survey or the campaign rounds, and no network panel on those screens.
describe("graph sample fetch (R18)", () => {
  it("never requests graph_sample before DONE, then fetches it once for the debrief", async () => {
    const client = createMockClient({ latencyMs: 0 });
    const { observation } = await client.createGame({ num_voters: 10_000, seed: 7, rounds: 2 });
    const graphSample = vi.spyOn(client, "graphSample");
    render(<GameApp client={client} initialObservation={observation} />);
    expect(screen.queryByTestId("network-stub")).toBeNull();

    fireEvent.click(screen.getByRole("tab", { name: /manifesto/i }));
    fireEvent.click(screen.getByRole("button", { name: /commit manifesto/i }));
    fireEvent.click(await screen.findByRole("button", { name: /commit & start round 1/i }));

    // round 1: no network panel, no request
    expect(await screen.findByRole("heading", { name: /round 1 of 2/i })).toBeInTheDocument();
    expect(screen.queryByTestId("network-stub")).toBeNull();
    expect(screen.queryByRole("heading", { name: /social network/i })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /launch round 1/i }));

    // round 2: still nothing
    expect(await screen.findByRole("heading", { name: /round 2 of 2/i })).toBeInTheDocument();
    await new Promise((r) => setTimeout(r, 20));
    expect(graphSample).not.toHaveBeenCalled();
    expect(screen.queryByTestId("network-stub")).toBeNull();

    // DONE: the debrief fetches the (revealed) sample exactly once and renders the alignment network
    fireEvent.click(screen.getByRole("button", { name: /final push/i }));
    expect(await screen.findByRole("heading", { name: /you (won|lost)\./i })).toBeInTheDocument();
    await waitFor(() => expect(graphSample).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByTestId("network-stub")).toHaveTextContent("sample"));
    const sample = await graphSample.mock.results[0].value;
    expect(sample.revealed).toBe(true);
  });
});
