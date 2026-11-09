import { describe, expect, it } from "vitest";
import Graph from "graphology";
import { reelInOutliers, settleIsolated } from "@/components/NetworkView";
import { ALIGNMENT_HEX, ALIGNMENT_LABELS } from "@/lib/constants";

// Regression (found by the e2e walkthrough): ForceAtlas2 flings nodes without a tie in the sample far away, so the
// camera zoomed out until the real network was a speck. They are put back inside their region's cloud.
describe("settleIsolated", () => {
  it("moves tie-less nodes into their region's cloud and leaves connected nodes alone", () => {
    const g = new Graph({ type: "undirected" });
    const pts: [string, string, number, number][] = [
      ["a", "N", 0, 0], ["b", "N", 2, 0], ["c", "N", 0, 2], ["d", "N", 2, 2],
      ["e", "S", 100, 100], ["f", "S", 102, 100],
    ];
    pts.forEach(([id, region, x, y]) => g.addNode(id, { region, x, y }));
    g.addEdge("a", "b"); g.addEdge("c", "d"); g.addEdge("a", "d"); g.addEdge("e", "f");
    g.addNode("lonelyN", { region: "N", x: 5000, y: -5000 });
    g.addNode("lonelyS", { region: "S", x: -9000, y: 9000 });

    expect(settleIsolated(g)).toBe(2);
    const n = g.getNodeAttributes("lonelyN");
    expect(Math.hypot(n.x - 1, n.y - 1)).toBeLessThanOrEqual(Math.SQRT2 + 1e-9);
    const s = g.getNodeAttributes("lonelyS");
    expect(Math.hypot(s.x - 101, s.y - 100)).toBeLessThanOrEqual(1 + 1e-9);
    expect(g.getNodeAttributes("a")).toMatchObject({ x: 0, y: 0 });
  });
});

describe("reelInOutliers", () => {
  it("pulls a far straggler back to the cloud's edge and keeps the cloud intact", () => {
    const g = new Graph({ type: "undirected" });
    for (let i = 0; i < 100; i++) g.addNode(String(i), { x: Math.cos(i) * (i % 10), y: Math.sin(i) * (i % 10) });
    g.addNode("far", { x: 1000, y: 0 });
    expect(reelInOutliers(g)).toBeGreaterThanOrEqual(1);
    const far = g.getNodeAttributes("far");
    expect(Math.hypot(far.x, far.y)).toBeLessThan(15);
    expect(far.y).toBeCloseTo(0, 0);
    expect(g.getNodeAttributes("5")).toMatchObject({ x: Math.cos(5) * 5, y: Math.sin(5) * 5 });
  });
});

// Regression (e2e walkthrough): the engine's bins are equal fifths of similarity, so qualitative names made a lost
// election look like "100% close/aligned". The labels must state the actual ranges.
describe("alignment legend", () => {
  it("labels the five bins with their similarity ranges", () => {
    expect(ALIGNMENT_LABELS).toHaveLength(ALIGNMENT_HEX.length);
    expect(ALIGNMENT_LABELS.join(" ")).toMatch(/<10%.*40–60%.*>90%/);
    expect(ALIGNMENT_LABELS.some((l) => /aligned|close/i.test(l))).toBe(false);
  });
});
