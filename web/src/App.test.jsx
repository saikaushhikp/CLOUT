import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";

vi.mock("./SupportChart.jsx", () => ({ default: () => <div data-testid="support-chart" /> }));

function descriptor(action_id, cost, eligible_target_ids = []) {
  return { action_id, cost, target_requirement: action_id === "targeted_contact" ? "eligible public IDs" : "no target", eligible_target_ids };
}

function observation(round_index = 0, remaining_budget = "20.00", terminal_status = false, events = []) {
  return {
    schema_version: "1.0",
    round_index,
    remaining_budget,
    aggregate_support: round_index === 0 ? 0.4 : 0.45,
    terminal_status,
    graph_summary: { node_count: 4, edge_count: 3, density: 0.5, min_degree: 1, max_degree: 2, mean_degree: 1.5, component_count: 1 },
    legal_actions: terminal_status ? [] : [descriptor("targeted_contact", "0.05 per target", ["v0", "v1", "v2"]), descriptor("broadcast", "1.00"), descriptor("no_op", "0.00")],
    public_events: events,
  };
}

function response(data, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => data };
}

function setupApi() {
  let currentRound = 0;
  let currentBudget = "20.00";
  let terminal = false;
  let events = [];
  let timeline = [observation()];
  const initialAggregate = timeline[0].aggregate_support;
  const fetchMock = vi.fn(async (url, options = {}) => {
    const method = options.method || "GET";
    if (url === "/api/health") return response({ status: "ok", api_version: "1.0" });
    if (url === "/api/games" && method === "POST") {
      return response({ game_id: "game-test", backend: "cpu", observation: observation(), timeline });
    }
    if (url.endsWith("/actions") && method === "POST") {
      const action = JSON.parse(options.body);
      if (action.action_id === "broadcast") {
        return response({ accepted: false, error_code: "INSUFFICIENT_BUDGET", terminal: false, backend: "cpu", observation: observation(currentRound, currentBudget, terminal, events), timeline });
      }
      currentRound += 1;
      terminal = currentRound >= 2;
      currentBudget = action.action_id === "targeted_contact" ? "19.95" : currentBudget;
      events = [...events, { sequence: currentRound, event_type: "action_accepted", round_index: currentRound - 1, action_id: action.action_id, summary: `${action.action_id} accepted` }];
      const next = observation(currentRound, currentBudget, terminal, events);
      timeline = [...timeline, next];
      return response({ accepted: true, terminal, backend: "cpu", observation: next, timeline, result: null });
    }
    if (url.endsWith("/result")) return response(terminal ? { victory: true, vote_share: 0.6, budget_used: "0.05" } : null);
    if (url.endsWith("/analysis")) return response({ evaluation_mode: "trace_only", status: "COMPLETE" });
    if (url.endsWith("/reset") && method === "POST") {
      currentRound = 0; currentBudget = "20.00"; terminal = false; events = []; timeline = [observation()];
      return response({ backend: "cpu", observation: observation(), timeline });
    }
    throw new Error(`Unexpected API request: ${method} ${url}`);
  });
  globalThis.fetch = fetchMock;
  return { fetchMock, initialAggregate };
}

describe("Dumbocracy React campaign UI", () => {
  beforeEach(() => setupApi());
  afterEach(() => { cleanup(); vi.restoreAllMocks(); });

  it("plays a complete episode, displays the result, and resets reproducibly", async () => {
    const user = userEvent.setup();
    const { fetchMock, initialAggregate } = setupApi();
    render(<App />);

    expect(await screen.findByText("Local service online")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /start episode/i }));
    expect(await screen.findByText(/Episode ready/)).toBeInTheDocument();
    expect(screen.getByText("0 / 5")).toBeInTheDocument();
    expect(screen.getByText("40.0%")).toBeInTheDocument();

    const targetPicker = screen.getByLabelText("Choose a public target ID");
    await user.selectOptions(targetPicker, "v1");
    await user.click(screen.getByRole("button", { name: /targeted contact/i }));
    expect(await screen.findByText("1 / 5")).toBeInTheDocument();
    expect(screen.getByText("19.95")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/games/game-test/actions", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ action_id: "targeted_contact", target_ids: ["v1"] }),
    }));

    await user.click(screen.getByRole("button", { name: /wait \/ no-op/i }));
    expect(await screen.findByRole("heading", { name: "Victory" })).toBeInTheDocument();
    expect(screen.getByText(/Final vote share/)).toHaveTextContent("60.0%");
    expect(screen.getByText(/trace_only/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /reset & replay/i }));
    expect(await screen.findByText("0 / 5")).toBeInTheDocument();
    expect(screen.getByText("40.0%")).toBeInTheDocument();
    expect(screen.getByText(/Episode reset/)).toBeInTheDocument();
    expect(initialAggregate).toBe(0.4);
  });

  it("communicates a rejected action without changing displayed round or funds", async () => {
    const user = userEvent.setup();
    setupApi();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: /start episode/i }));
    await user.click(screen.getByRole("button", { name: /broadcast/i }));
    expect(await screen.findByText(/Action rejected \(INSUFFICIENT_BUDGET\)/)).toBeInTheDocument();
    expect(screen.getByText("0 / 5")).toBeInTheDocument();
    expect(screen.getByText("20.00")).toBeInTheDocument();
  });

  it("filters the public target list and keeps the selected ID explicit", async () => {
    const user = userEvent.setup();
    setupApi();
    render(<App />);
    await user.click(await screen.findByRole("button", { name: /start episode/i }));
    const search = screen.getByRole("searchbox", { name: "Public target" });
    fireEvent.change(search, { target: { value: "v2" } });
    const select = screen.getByLabelText("Choose a public target ID");
    await waitFor(() => expect(within(select).getByRole("option")).toHaveTextContent("v2"));
    expect(within(select).queryByRole("option", { name: "v1" })).not.toBeInTheDocument();
  });
});
