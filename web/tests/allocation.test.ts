import { describe, expect, it } from "vitest";
import { allocationCostCents, checkAllocation, emptyDraft, setRegionHires, teamByRegion } from "@/lib/allocation";
import { centsToString, formatMoney, toCents } from "@/lib/money";

const obs = {
  treasury: "100.00",
  costs: {
    query: "5.00",
    worker: "4.00",
    min_degree: 3,
    attempt_cap: 3,
    media_by_region: { N: "42.50", S: "61.25", E: "50.00", W: "46.25" },
  },
};

describe("money", () => {
  it("parses and formats 2-dp decimal strings exactly", () => {
    expect(toCents("500.00")).toBe(50000);
    expect(toCents("0.1")).toBe(10);
    expect(toCents("61.25")).toBe(6125);
    expect(centsToString(6125)).toBe("61.25");
    expect(centsToString(5)).toBe("0.05");
    expect(formatMoney("1234.5")).toBe("Đ1,234.50");
  });
});

const none = [false, false, false, false];

describe("allocation math (P8 contract pricing)", () => {
  it("doing nothing is legal and free", () => {
    const c = checkAllocation(obs, emptyDraft());
    expect(c.ok).toBe(true);
    expect(c.costCents).toBe(0);
    expect(c.totalHires).toBe(0);
    expect(c.teamBefore).toEqual([0, 0, 0, 0]); // no free workers (R21)
  });

  it("cost = Σ hires × worker + Σ media_by_region over aired regions", () => {
    const draft = { hires: [2, 0, 1, 0], media: [true, false, false, true] };
    const c = checkAllocation({ ...obs, treasury: "200.00" }, draft);
    expect(c.totalHires).toBe(3);
    expect(c.workerCostCents).toBe(1200);
    expect(c.mediaCostCents).toBe(4250 + 4625);
    expect(c.costCents).toBe(1200 + 4250 + 4625);
    expect(allocationCostCents(obs, draft)).toBe(c.costCents);
    expect(c.ok).toBe(true);
    expect(c.remainingCents).toBe(20000 - c.costCents);
    expect(checkAllocation(obs, draft).errors.map((e) => e.code)).toEqual(["INSUFFICIENT_FUNDS"]);
  });

  it("over budget: workers alone can exceed the treasury; cost == treasury is fine", () => {
    const over = checkAllocation(obs, { hires: [26, 0, 0, 0], media: none });
    expect(over.ok).toBe(false);
    expect(over.remainingCents).toBe(-400);
    expect(over.errors.map((e) => e.code)).toEqual(["INSUFFICIENT_FUNDS"]);
    const exact = checkAllocation(obs, { hires: [10, 5, 5, 5], media: none });
    expect(exact.ok).toBe(true);
    expect(exact.remainingCents).toBe(0);
    const mediaExact = checkAllocation({ ...obs, treasury: "103.75" }, { hires: [0, 0, 0, 0], media: [true, true, false, false] });
    expect(mediaExact.ok).toBe(true);
  });

  it("rejects negative / fractional hires and wrong shapes", () => {
    expect(checkAllocation(obs, { hires: [-1, 0, 0, 0], media: none }).ok).toBe(false);
    expect(checkAllocation(obs, { hires: [1.5, 0, 0, 0], media: none }).errors[0].code).toBe("INVALID_ALLOCATION");
    expect(checkAllocation(obs, { hires: [0, 0, 0], media: none }).ok).toBe(false);
  });

  it("max affordable extra workers accounts for booked media and planned hires", () => {
    expect(checkAllocation(obs, emptyDraft()).maxAffordableMore).toBe(25);
    expect(checkAllocation(obs, { ...emptyDraft(), media: [true, false, false, false] }).maxAffordableMore).toBe(14); // (100-42.5)/4
    expect(checkAllocation(obs, { hires: [3, 2, 0, 0], media: none }).maxAffordableMore).toBe(20);
  });

  it("the hire stepper helper sets one region, rounding and flooring at 0", () => {
    let d = emptyDraft();
    d = setRegionHires(d, 0, 2);
    d = setRegionHires(d, 3, 1.6);
    expect(d.hires).toEqual([2, 0, 0, 2]);
    d = setRegionHires(d, 0, -3);
    expect(d.hires).toEqual([0, 0, 0, 2]);
    expect(emptyDraft().hires).toEqual([0, 0, 0, 0]); // helper never mutates
  });
});

describe("permanent team (workers.by_region)", () => {
  const veteran = { ...obs, workers: { by_region: { N: 3, S: 0, E: 1, W: 2 }, total: 6 } };

  it("reads the team per region; earlier workers are already paid for", () => {
    expect(teamByRegion(veteran)).toEqual([3, 0, 1, 2]);
    expect(teamByRegion(obs)).toEqual([0, 0, 0, 0]);
    const c = checkAllocation(veteran, emptyDraft());
    expect(c.costCents).toBe(0);
    expect(c.teamBefore).toEqual([3, 0, 1, 2]);
  });

  it("teamAfter = team + this round's hires", () => {
    const c = checkAllocation(veteran, { hires: [1, 2, 0, 0], media: none });
    expect(c.teamAfter).toEqual([4, 2, 1, 2]);
    expect(c.workerCostCents).toBe(3 * 400);
  });
});
