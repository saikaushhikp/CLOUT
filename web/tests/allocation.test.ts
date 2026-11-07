import { describe, expect, it } from "vitest";
import { allocationCostCents, checkAllocation, emptyDraft, setExtraWorkers, setRegionWorkers } from "@/lib/allocation";
import { centsToString, formatMoney, toCents } from "@/lib/money";

const obs = {
  treasury: "100.00",
  costs: {
    query: "5.00",
    extra_worker: "4.00",
    free_workers: 10,
    contacts_per_worker: 25,
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

describe("allocation math (contract pricing)", () => {
  it("free workers only costs nothing", () => {
    const c = checkAllocation(obs, { workers: [4, 3, 2, 1], extra: 0, media: [false, false, false, false] });
    expect(c.ok).toBe(true);
    expect(c.costCents).toBe(0);
    expect(c.workersAvailable).toBe(10);
    expect(c.workersIdle).toBe(0);
  });

  it("cost = extra × extra_worker + Σ media_by_region over aired regions", () => {
    const draft = { workers: [5, 5, 3, 0], extra: 3, media: [true, false, false, true] };
    const c = checkAllocation({ ...obs, treasury: "200.00" }, draft);
    expect(c.workerCostCents).toBe(1200);
    expect(c.mediaCostCents).toBe(4250 + 4625);
    expect(c.costCents).toBe(1200 + 4250 + 4625);
    expect(allocationCostCents(obs, draft)).toBe(c.costCents);
    expect(c.ok).toBe(true);
    expect(c.remainingCents).toBe(20000 - c.costCents);
    expect(checkAllocation(obs, draft).errors.map((e) => e.code)).toEqual(["INSUFFICIENT_FUNDS"]);
  });

  it("rejects more workers than free + extra", () => {
    const c = checkAllocation(obs, { workers: [5, 5, 1, 0], extra: 0, media: [false, false, false, false] });
    expect(c.ok).toBe(false);
    expect(c.errors[0].code).toBe("INVALID_ALLOCATION");
    expect(checkAllocation(obs, { workers: [5, 5, 1, 0], extra: 1, media: [false, false, false, false] }).ok).toBe(true);
  });

  it("rejects plans that cost more than the treasury (cost == treasury is fine)", () => {
    const over = checkAllocation(obs, { workers: [0, 0, 0, 0], extra: 0, media: [true, true, false, false] });
    expect(over.ok).toBe(false);
    expect(over.errors.map((e) => e.code)).toContain("INSUFFICIENT_FUNDS");
    const exact = checkAllocation({ ...obs, treasury: "103.75" }, { workers: [0, 0, 0, 0], extra: 0, media: [true, true, false, false] });
    expect(exact.ok).toBe(true);
    expect(exact.remainingCents).toBe(0);
  });

  it("rejects negative / fractional numbers", () => {
    expect(checkAllocation(obs, { workers: [-1, 0, 0, 0], extra: 0, media: [false, false, false, false] }).ok).toBe(false);
    expect(checkAllocation(obs, { workers: [1.5, 0, 0, 0], extra: 0, media: [false, false, false, false] }).ok).toBe(false);
    expect(checkAllocation(obs, { workers: [0, 0, 0, 0], extra: -2, media: [false, false, false, false] }).ok).toBe(false);
  });

  it("max affordable extra workers accounts for booked media", () => {
    expect(checkAllocation(obs, emptyDraft()).maxAffordableExtra).toBe(25);
    expect(checkAllocation(obs, { ...emptyDraft(), media: [true, false, false, false] }).maxAffordableExtra).toBe(14); // (100-42.5)/4
  });

  it("steppers never exceed the worker pool", () => {
    let d = emptyDraft();
    d = setRegionWorkers(d, 10, 0, 8);
    d = setRegionWorkers(d, 10, 1, 8);
    expect(d.workers).toEqual([8, 2, 0, 0]);
    d = setRegionWorkers(d, 10, 2, -3);
    expect(d.workers[2]).toBe(0);
    d = setExtraWorkers(d, 10, 5);
    d = setRegionWorkers(d, 10, 2, 99);
    expect(d.workers).toEqual([8, 2, 5, 0]);
    // dropping extra workers trims assignments back into the pool
    d = setExtraWorkers(d, 10, 1);
    expect(d.workers.reduce((a, b) => a + b, 0)).toBe(11);
    expect(checkAllocation(obs, d).ok).toBe(true);
  });
});
