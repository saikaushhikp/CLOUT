import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { BLEND_HELP, BlendPicker } from "@/components/ManifestoDesigner";
import { blendSelected, largestHunchRegions, regionsWithHunch } from "@/lib/insight";

const obs = {
  regions: [
    { id: "N", label: "North", voters: 4000, intra_edges: 0, mean_degree: 16 },
    { id: "S", label: "South", voters: 1000, intra_edges: 0, mean_degree: 16 },
    { id: "E", label: "East", voters: 3000, intra_edges: 0, mean_degree: 16 },
    { id: "W", label: "West", voters: 2000, intra_edges: 0, mean_degree: 16 },
  ],
  archetype_library: [
    { id: "A0", name: "Agrarian", mean: [0.1, 0.1, 0.6, 0.1, 0.1] },
    { id: "A1", name: "Industrial", mean: [0.1, 0.1, 0.1, 0.6, 0.1] },
    { id: "A2", name: "Green", mean: [0.2, 0.1, 0.1, 0.1, 0.5] },
  ],
};

describe("blend helpers (R20)", () => {
  const hunches = { N: "A0", S: "A1", E: "A2" }; // West has no hunch

  it("weights the chosen regions' archetypes by voters", () => {
    const b = blendSelected(obs, hunches, ["N", "E"])!;
    // (4000·A0 + 3000·A2) / 7000
    const expected = [0, 1, 2, 3, 4].map((k) => (4000 * obs.archetype_library[0].mean[k] + 3000 * obs.archetype_library[2].mean[k]) / 7000);
    b.forEach((x, k) => expect(x).toBeCloseTo(expected[k], 12));
    expect(b.reduce((a, c) => a + c, 0)).toBeCloseTo(1, 12);
    // a different choice of regions gives a different blend
    expect(blendSelected(obs, hunches, ["N", "S"])).not.toEqual(b);
  });

  it("needs at least 2 (and at most 4) regions, all with a hunch", () => {
    expect(blendSelected(obs, hunches, ["N"])).toBeNull();
    expect(blendSelected(obs, hunches, ["N", "N"])).toBeNull(); // duplicates count once
    expect(blendSelected(obs, hunches, ["N", "W"])).toBeNull(); // West has no hunch
    expect(blendSelected(obs, { ...hunches, W: "A1" }, ["N", "S", "E", "W"])).not.toBeNull();
  });

  it("shortcuts pick the k largest regions that have a hunch", () => {
    expect(regionsWithHunch(obs, hunches)).toEqual(["N", "S", "E"]);
    expect(largestHunchRegions(obs, hunches, 2)).toEqual(["N", "E"]);
    expect(largestHunchRegions(obs, hunches, 3)).toEqual(["N", "E", "S"]); // West (2000) skipped: no hunch
  });
});

describe("BlendPicker", () => {
  it("lets the player choose regions, disables those without a hunch, and enforces at least 2", () => {
    const onApply = vi.fn();
    render(<BlendPicker obs={obs} hunches={{ N: "A0", S: "A1", E: "A2" }} onApply={onApply} />);
    const group = screen.getByRole("group", { name: /blend which regions/i });
    expect(group).toHaveAccessibleDescription(new RegExp(BLEND_HELP.slice(0, 40)));
    const box = (name: RegExp) => within(group).getByRole("checkbox", { name });
    expect(box(/west/i)).toBeDisabled();
    expect(box(/north/i)).toBeChecked();
    expect(box(/south/i)).toBeChecked();
    expect(box(/east/i)).toBeChecked();

    // "2 largest" → North + East
    fireEvent.click(within(group).getByRole("button", { name: "2 largest" }));
    expect(box(/north/i)).toBeChecked();
    expect(box(/east/i)).toBeChecked();
    expect(box(/south/i)).not.toBeChecked();
    expect(within(group).getByText(/weight 57%/)).toBeInTheDocument(); // 4000 / 7000

    // dropping below 2 disables Apply
    fireEvent.click(box(/east/i));
    const apply = within(group).getByRole("button", { name: /apply blend/i });
    expect(apply).toBeDisabled();
    expect(within(group).getByText(/choose at least 2 regions/i)).toBeInTheDocument();

    fireEvent.click(box(/south/i));
    expect(apply).toBeEnabled();
    fireEvent.click(apply);
    expect(onApply).toHaveBeenCalledTimes(1);
    const [blend, regions] = onApply.mock.calls[0];
    expect(regions.sort()).toEqual(["N", "S"]);
    expect(blend[2]).toBeCloseTo((4000 * 0.6 + 1000 * 0.1) / 5000, 12);

    // "3 largest"
    fireEvent.click(within(group).getByRole("button", { name: "3 largest" }));
    expect([box(/north/i), box(/south/i), box(/east/i)].every((b) => (b as HTMLInputElement).checked)).toBe(true);
  });

  it("disables the shortcuts when too few regions have a hunch", () => {
    render(<BlendPicker obs={obs} hunches={{ N: "A0", S: "A1" }} onApply={() => undefined} />);
    expect(screen.getByRole("button", { name: "2 largest" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "3 largest" })).toBeDisabled();
  });
});
