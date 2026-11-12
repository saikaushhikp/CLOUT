import { describe, expect, it } from "vitest";
import Graph from "graphology";
import { placeRegionClusters, reelInOutliers, settleIsolated } from "@/components/NetworkView";
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

// The four regions are drawn as separate clusters at their map positions (N top, E right, S bottom, W left).
describe("placeRegionClusters", () => {
  // all four regions laid out on top of each other (as one global layout would), with very different sizes
  const sizes: Record<string, number> = { N: 400, S: 300, E: 200, W: 100 };
  const pts = Object.entries(sizes).flatMap(([region, n]) =>
    Array.from({ length: n }, (_, i) => ({ region, x: 50 + Math.cos(i) * (i % 17), y: -20 + Math.sin(i) * (i % 17) })),
  );
  pts.push({ region: "W", x: 5000, y: 5000 }); // a far straggler
  const out = placeRegionClusters(pts);
  const cloud = (r: string) => out.filter((_, k) => pts[k].region === r);
  const centre = (ps: { x: number; y: number }[]) => [ps.reduce((s, p) => s + p.x, 0) / ps.length, ps.reduce((s, p) => s + p.y, 0) / ps.length];
  const radius = (ps: { x: number; y: number }[]) => { const [cx, cy] = centre(ps); return Math.max(...ps.map((p) => Math.hypot(p.x - cx, p.y - cy))); };

  it("centres each region at its map position", () => {
    const want: Record<string, [number, number]> = { N: [0, 1], E: [1, 0], S: [0, -1], W: [-1, 0] };
    for (const [r, [x, y]] of Object.entries(want)) {
      const [cx, cy] = centre(cloud(r));
      expect(Math.hypot(cx - x, cy - y)).toBeLessThan(0.1);
    }
  });

  it("sizes clusters by their share of voters and keeps them apart, stragglers included", () => {
    expect(radius(cloud("N"))).toBeGreaterThan(radius(cloud("W")));
    const [nx, ny] = centre(cloud("N"));
    const [ex, ey] = centre(cloud("E"));
    expect(radius(cloud("N")) + radius(cloud("E"))).toBeLessThan(Math.hypot(nx - ex, ny - ey));
    expect(radius(cloud("W"))).toBeLessThan(0.7); // the straggler was pulled in
  });
});
