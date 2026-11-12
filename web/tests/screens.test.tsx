import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { GameApp } from "@/components/GameApp";
import { budgetTextValid, budgetToRequest, budgetToSlider, clampBudget, sliderToBudget } from "@/components/SetupScreen";
import { createMockClient } from "@/lib/mockApi";
import type { DumbocracyClient } from "@/lib/api";
import type { Observation } from "@/lib/types";
import { centsToString, formatMoney, toCents } from "@/lib/money";

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
    expect(screen.getAllByText("10,000")).toHaveLength(1); // big number
    await waitFor(() => expect(screen.getByText(/engine online/i)).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: "100k" }));
    expect(screen.getAllByText("100,000")).toHaveLength(1);
    fireEvent.change(screen.getByLabelText(/world seed/i), { target: { value: "4242" } });
    fireEvent.click(screen.getByRole("button", { name: /decrease campaign rounds/i }));
    fireEvent.click(screen.getByRole("button", { name: /start campaign/i }));

    await waitFor(() => expect(spy).toHaveBeenCalledWith({ num_voters: 100_000, rounds: 4, seed: 4242, budget: "500.00" }));
    expect(await screen.findByText(/find out what the regions want/i)).toBeInTheDocument();
    expect(screen.getByText(/national top 3/i)).toBeInTheDocument();
  });

  it("lets the player choose the starting treasury (R17): presets, slider, typed value, summary, request", async () => {
    const client = createMockClient();
    const spy = vi.spyOn(client, "createGame");
    render(<GameApp client={client} />);
    const summary = () => screen.getByTestId("summary-budget");
    expect(summary()).toHaveTextContent("Đ500.00"); // default
    expect(screen.getByRole("button", { name: "Đ1k" })).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(screen.getByRole("button", { name: "Đ10k" }));
    expect(summary()).toHaveTextContent("Đ10,000.00");
    expect(screen.getByRole("button", { name: "Đ10k" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText(/starting treasury in credits/i)).toHaveValue("10000");

    // logarithmic slider: both ends reach the limits
    const slider = screen.getByLabelText(/starting money/i);
    fireEvent.change(slider, { target: { value: slider.getAttribute("max") } });
    expect(summary()).toHaveTextContent("Đ1,000,000.00");
    fireEvent.change(slider, { target: { value: "0" } });
    expect(summary()).toHaveTextContent("Đ500.00");

    // out-of-range typed values block the start button; blur clamps into range
    const field = screen.getByLabelText(/starting treasury in credits/i);
    fireEvent.change(field, { target: { value: "2000000" } });
    expect(field).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText(/enter a whole amount from Đ500 to Đ1,000,000/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start campaign/i })).toBeDisabled();
    fireEvent.blur(field);
    expect(field).toHaveValue("1000000");
    fireEvent.change(field, { target: { value: "820" } });
    expect(summary()).toHaveTextContent("Đ820.00");

    fireEvent.click(screen.getByRole("button", { name: /start campaign/i }));
    await waitFor(() => expect(spy).toHaveBeenCalledWith(expect.objectContaining({ budget: "820.00" })));
    expect(await screen.findByText(/find out what the regions want/i)).toBeInTheDocument();
    expect(screen.getByText("Đ820.00")).toBeInTheDocument(); // treasury chip
  });

  it("budget helpers clamp to ₫500–₫1,000,000 and serialise with 2 dp", () => {
    expect(clampBudget(499)).toBe(500);
    expect(clampBudget(1_000_001)).toBe(1_000_000);
    expect(clampBudget(749.6)).toBe(750);
    expect(clampBudget(Number.NaN)).toBe(500);
    expect(budgetTextValid("500")).toBe(true);
    expect(budgetTextValid("1000000")).toBe(true);
    expect(budgetTextValid("1000001")).toBe(false);
    expect(budgetTextValid("499")).toBe(false);
    expect(budgetTextValid("1000.5")).toBe(false);
    expect(budgetTextValid("")).toBe(false);
    expect(budgetToRequest(750)).toBe("750.00");
    expect(budgetToRequest(5_000_000)).toBe("1000000.00");
    // log slider: exact at both ends, presets land within one step (~1.3%), 2 significant digits in between
    expect(sliderToBudget(0)).toBe(500);
    expect(sliderToBudget(budgetToSlider(1_000_000))).toBe(1_000_000);
    for (const b of [1_000, 10_000, 100_000]) expect(Math.abs(sliderToBudget(budgetToSlider(b)) - b) / b).toBeLessThan(0.02);
    expect(sliderToBudget(1)).toBe(510);
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
    // the answer line stays empty until the question has been asked
    expect(screen.getByTestId("qb-sample")).toBeEmptyDOMElement();

    const askBtn = () => screen.getByRole("button", { name: /^ask( again)? · /i });
    const northBtn = () => screen.getByRole("button", { name: /^North \d+ responses?$/ });
    // before asking: every region's answers button is disabled
    expect(screen.getByRole("button", { name: "North No responses yet" })).toBeDisabled();
    fireEvent.click(askBtn());
    await waitFor(() => expect(query).toHaveBeenCalledTimes(1));
    // the answer appears, highlighted, under the question; all responses also open per region in a dialog
    await waitFor(() => expect(northBtn()).toHaveAccessibleName("North 1 response"));
    expect(screen.getByTestId("qb-sample")).toHaveClass("answer-flash");
    expect(screen.getByRole("button", { name: "South No responses yet" })).toBeDisabled();
    northBtn().focus(); // jsdom does not focus on click; a real click (or keyboard) does
    fireEvent.click(northBtn());
    let dialog = screen.getByRole("dialog", { name: "Responses from North" });
    let ledger = within(dialog).getByRole("list", { name: /your survey answers/i });
    expect(within(ledger).getByTestId("ledger-estimate")).toHaveTextContent(new RegExp(`^\\d+\\.\\d% ± \\d+\\.\\d \\(n = ${m}\\)$`));
    expect(within(dialog).getByText(new RegExp(`of ${m.toLocaleString("en-US")}`))).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Close" })).toHaveFocus();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(northBtn()).toHaveFocus(); // focus returns to the region button

    // repeat: allowed, not blocked, and pooled into one ledger row
    expect(askBtn()).toHaveTextContent(/ask again/i);
    expect(askBtn()).toBeEnabled();
    // once asked, the line under the question shows the answer itself
    expect(screen.getByTestId("qb-sample")).toHaveTextContent(new RegExp(`% of ${m.toLocaleString("en-US")} North voters prefer Agriculture to Education`));
    fireEvent.click(askBtn());
    await waitFor(() => expect(query).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(northBtn()).toHaveAccessibleName("North 2 responses"));
    fireEvent.click(northBtn());
    dialog = screen.getByRole("dialog", { name: "Responses from North" });
    ledger = within(dialog).getByRole("list", { name: /your survey answers/i });
    expect(within(ledger).getByTestId("ledger-estimate")).toHaveTextContent(`(n = ${2 * m})`);
    expect(within(ledger).getAllByTestId("ledger-estimate")).toHaveLength(1);
    expect(within(ledger).getByText(/pooled ×2/)).toBeInTheDocument();
    expect(within(ledger).getByRole("list", { name: /individual samples/i }).querySelectorAll("li")).toHaveLength(2);
    expect(within(dialog).getByText(/Pooled over 2 samples/)).toBeInTheDocument();
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
    expect(screen.getByText(/poll at the start of round 1/i)).toBeInTheDocument();
    // R18: no network panel during the rounds
    expect(screen.queryByTestId("network-stub")).toBeNull();
    expect(screen.queryByRole("heading", { name: /social network/i })).toBeNull();
    // the committed manifesto is always on screen
    const manifesto = screen.getByTestId("manifesto-summary");
    expect(within(manifesto).getByRole("img", { name: /manifesto: education 25%, welfare 15%, agriculture 20%, industry 20%, environment 20%/i })).toBeInTheDocument();
    expect(within(manifesto).getByText("Education").closest("li")).toHaveTextContent("25%");
    // buying copy (P8): one-time fee, permanent, no free workers
    expect(screen.getByText(/Buy party workers/)).toBeInTheDocument();
    const fee = formatMoney(obs.costs.worker); // read from the engine, never hard-coded
    expect(screen.getByText(`${fee} each, permanent`)).toBeInTheDocument();
    expect(screen.getByTestId("team-line")).toHaveTextContent("Permanent team: 0 workers");
    expect(screen.queryByText(/deployed|unassigned|available/i)).toBeNull(); // no reassignment UI

    const north = screen.getByRole("group", { name: "New workers in North" });
    const inc = within(north).getByRole("button", { name: /increase/i });
    fireEvent.click(inc);
    fireEvent.click(inc);
    expect(within(north).getByRole("spinbutton")).toHaveValue(2);
    expect(screen.getByTestId("hire-line-N")).toHaveTextContent("team 0 · +2 new");
    fireEvent.click(within(screen.getByRole("group", { name: "New workers in East" })).getByRole("button", { name: /increase/i }));
    expect(screen.getByTestId("team-line")).toHaveTextContent("Permanent team: 0 workers + 3 new = 3");
    fireEvent.click(screen.getByRole("button", { name: /air media in south/i }));
    const southPrice = obs.costs.media_by_region.S;
    const expected = formatMoney(3 * toCents(obs.costs.worker) + toCents(southPrice));
    expect(screen.getByText("This round").nextSibling).toHaveTextContent(expected);
    expect(screen.getByText(`Workers 3 × ${fee} (one-time)`).nextSibling).toHaveTextContent(formatMoney(3 * toCents(obs.costs.worker)));

    fireEvent.click(screen.getByRole("button", { name: /launch round 1/i }));
    await waitFor(() =>
      expect(allocate).toHaveBeenCalledWith(obs.game_id, "p0", {
        hires_by_region: [2, 0, 1, 0],
        media_regions: [false, true, false, false],
      }),
    );
    expect(await screen.findByRole("heading", { name: /round 2 of 3/i })).toBeInTheDocument();
    // P8: workers are permanent and stay in their region — round 2 shows the team, the steppers start at 0
    expect(screen.getByTestId("team-line")).toHaveTextContent("Permanent team: 3 workers");
    expect(screen.getByTestId("hire-line-N")).toHaveTextContent("team 2");
    expect(screen.getByTestId("hire-line-E")).toHaveTextContent("team 1");
    expect(within(screen.getByRole("group", { name: "New workers in North" })).getByRole("spinbutton")).toHaveValue(0);
    expect(screen.getByText("This round").nextSibling).toHaveTextContent("Đ0.00");
    // last round's outcome
    const strip = screen.getByLabelText("Round 1 report");
    expect(strip).toHaveTextContent("R1 spreaders");
    expect(strip).toHaveTextContent("Converted");
    expect(strip).toHaveTextContent("Saw your ads");
    expect(screen.getByTestId("manifesto-summary")).toBeInTheDocument();
  });

  it("caps the hire stepper at what the treasury affords and blocks over-budget plans", async () => {
    const client = createMockClient();
    const base = await roundObservation(client);
    // a treasury of exactly two workers: the stepper stops at 2, and any media then overspends
    const obs = { ...base, treasury: centsToString(2 * toCents(base.costs.worker)) };
    render(<GameApp client={client} initialObservation={obs} />);
    const north = screen.getByRole("group", { name: "New workers in North" });
    const inc = within(north).getByRole("button", { name: /increase/i });
    for (let i = 0; i < 5; i++) fireEvent.click(inc);
    expect(within(north).getByRole("spinbutton")).toHaveValue(2);
    expect(inc).toBeDisabled();
    const launch = screen.getByRole("button", { name: /launch round 1/i });
    expect(launch).toBeEnabled();
    // media pushes the plan over budget: error + disabled launch
    fireEvent.click(screen.getByRole("button", { name: /air media in north/i }));
    expect(screen.getByRole("alert")).toHaveTextContent(/costs more than your treasury/i);
    expect(launch).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /air media in north/i }));
    expect(launch).toBeEnabled();
  });

  it("explains how party workers are bought and how they spread", async () => {
    const client = createMockClient();
    const obs = await roundObservation(client);
    render(<GameApp client={client} initialObservation={obs} />);
    const summary = screen.getByText("How it works");
    const details = summary.closest("details")!;
    expect(details.open).toBe(false);
    fireEvent.click(summary);
    const note = within(details).getByRole("note", { name: /how party workers work/i });
    expect(note).toHaveTextContent(/no free workers/i);
    expect(note).toHaveTextContent(`one-time fee of ${formatMoney(obs.costs.worker)}`);
    expect(note).toHaveTextContent(/permanent/);
    expect(note).toHaveTextContent(/stays in the region/);
    expect(note).toHaveTextContent(/random voter of that region with at least 3 friends/);
    expect(note).toHaveTextContent(/pinned to your manifesto/);
    expect(note).toHaveTextContent(/one hop per round/);
    expect(note).toHaveTextContent(/next round, then\s+becomes an ordinary voter again/);
    expect(note).toHaveTextContent(/at most 3 times/);
    expect(note).toHaveTextContent(/how receptive they are to canvassing × how much they already like your manifesto/);
    expect(note).toHaveTextContent(/Buy early/);
    expect(note).toHaveTextContent(/converted in the final round have no round left to\s+spread/);
    expect(note).not.toHaveTextContent(/doorstep|hops/);
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
