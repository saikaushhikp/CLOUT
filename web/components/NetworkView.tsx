"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";
import type { GraphSample, RegionId } from "@/lib/types";
import { ALIGNMENT_HEX, ALIGNMENT_LABELS, REGION_HEX, REGION_META } from "@/lib/constants";
import { fmtInt } from "@/lib/format";

type Mode = "region" | "alignment";

const CENTERS: Record<RegionId, [number, number]> = { N: [0, 1], E: [1, 0], S: [0, -1], W: [-1, 0] };

/** Deterministic hash → [0,1) so the layout is identical on every render/refresh. */
function h01(n: number, salt: number): number {
  let x = (n * 2654435761 + salt * 40503) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 2246822507) >>> 0;
  x ^= x >>> 13;
  x = Math.imul(x, 3266489909) >>> 0;
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

export function computeLayout(sample: GraphSample): { x: number; y: number }[] {
  const counts: Record<string, number> = {};
  sample.nodes.forEach((n) => (counts[n.region] = (counts[n.region] ?? 0) + 1));
  const total = sample.nodes.length || 1;
  return sample.nodes.map((n, i) => {
    const [cx, cy] = CENTERS[n.region as RegionId] ?? [0, 0];
    const spread = 0.35 + 0.6 * Math.sqrt((counts[n.region] ?? 1) / total);
    const r = spread * Math.sqrt(h01(n.id + i, 1));
    const t = 2 * Math.PI * h01(n.id + i, 2);
    return { x: cx * 1.25 + r * Math.cos(t), y: cy * 1.25 + r * Math.sin(t) };
  });
}

type LayoutGraph = {
  forEachNode: (cb: (node: string, attrs: Record<string, unknown>) => void) => void;
  degree: (node: string) => number;
  mergeNodeAttributes: (node: string, attrs: Record<string, unknown>) => void;
};

/**
 * ForceAtlas2 repels nodes that have no tie inside the sample far away from everything else, which
 * zooms the camera out until the real network is a speck. Put them back inside their region's cloud.
 */
export function settleIsolated(graph: LayoutGraph): number {
  const byRegion = new Map<string, { x: number; y: number }[]>();
  const isolated: { node: string; region: string; k: number }[] = [];
  let k = 0;
  graph.forEachNode((node, a) => {
    const region = String(a.region ?? "");
    if (graph.degree(node) === 0) isolated.push({ node, region, k: k++ });
    else {
      const pts = byRegion.get(region) ?? [];
      pts.push({ x: Number(a.x), y: Number(a.y) });
      byRegion.set(region, pts);
    }
  });
  const all = [...byRegion.values()].flat();
  for (const it of isolated) {
    const pts = byRegion.get(it.region)?.length ? byRegion.get(it.region)! : all;
    if (!pts.length) return 0;
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    const d = pts.map((p) => Math.hypot(p.x - cx, p.y - cy)).sort((a, b) => a - b);
    const radius = d[Math.floor(d.length / 2)] || 1;
    const r = radius * Math.sqrt(h01(it.k, 3));
    const t = 2 * Math.PI * h01(it.k, 4);
    graph.mergeNodeAttributes(it.node, { x: cx + r * Math.cos(t), y: cy + r * Math.sin(t) });
  }
  reelInOutliers(graph);
  return isolated.length;
}

/** Pull the few stragglers (pendant chains, tiny components) beyond 1.15 × the 97th-percentile radius back to it. */
export function reelInOutliers(graph: LayoutGraph): number {
  const pts: { node: string; x: number; y: number }[] = [];
  graph.forEachNode((node, a) => pts.push({ node, x: Number(a.x), y: Number(a.y) }));
  if (pts.length < 20) return 0;
  // median centre: the stragglers must not drag the centre towards themselves
  const med = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
  const cx = med(pts.map((p) => p.x));
  const cy = med(pts.map((p) => p.y));
  const d = pts.map((p) => Math.hypot(p.x - cx, p.y - cy));
  const sorted = [...d].sort((a, b) => a - b);
  const limit = 1.15 * sorted[Math.floor(0.97 * (sorted.length - 1))];
  let moved = 0;
  pts.forEach((p, i) => {
    if (d[i] > limit && d[i] > 0) {
      const f = limit / d[i];
      graph.mergeNodeAttributes(p.node, { x: cx + (p.x - cx) * f, y: cy + (p.y - cy) * f });
      moved += 1;
    }
  });
  return moved;
}

export function NetworkView({
  sample,
  error,
  totalVoters,
  revealRounds,
}: {
  sample: GraphSample | null;
  error?: string | null;
  totalVoters: number;
  /** when the alignment is revealed, labels for each snapshot (round 0..T) */
  revealRounds?: string[];
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<{ kill: () => void; refresh: () => void } | null>(null);
  const graphRef = useRef<{ setNodeAttribute: (n: string, k: string, v: unknown) => void } | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "nowebgl">("idle");
  const revealed = Boolean(sample?.revealed && sample.alignment && sample.alignment.length > 0);
  const [mode, setMode] = useState<Mode>(revealed ? "alignment" : "region");
  const snapshots = sample?.alignment?.length ?? 0;
  const [frame, setFrame] = useState(Math.max(0, snapshots - 1));
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setMode(revealed ? "alignment" : "region");
    setFrame(Math.max(0, snapshots - 1));
  }, [revealed, snapshots]);

  const layout = useMemo(() => (sample ? computeLayout(sample) : []), [sample]);

  // Build sigma once per sample.
  useEffect(() => {
    const el = containerRef.current;
    if (!sample || !el) return;
    let cancelled = false;
    setStatus("loading");
    (async () => {
      try {
        const canvas = document.createElement("canvas");
        if (!canvas.getContext("webgl2") && !canvas.getContext("webgl")) {
          setStatus("nowebgl");
          return;
        }
        const [{ default: Graph }, { default: Sigma }, fa2mod] = await Promise.all([
          import("graphology"),
          import("sigma"),
          import("graphology-layout-forceatlas2"),
        ]);
        if (cancelled) return;
        const graph = new Graph({ type: "undirected", multi: false, allowSelfLoops: false });
        sample.nodes.forEach((n, i) => {
          graph.addNode(String(i), {
            x: layout[i].x,
            y: layout[i].y,
            size: Math.min(6, 1.6 + Math.sqrt(n.degree) / 3),
            color: REGION_HEX[n.region as RegionId] ?? "#888",
            region: n.region,
          });
        });
        sample.edges.forEach(([a, b]) => {
          const s = String(a);
          const t = String(b);
          if (a !== b && graph.hasNode(s) && graph.hasNode(t) && !graph.hasEdge(s, t)) graph.addEdge(s, t, { size: 0.5, color: "rgba(140,140,150,0.22)" });
        });
        const fa2 = (fa2mod as unknown as { default: { assign: (g: unknown, p: unknown) => void } }).default;
        fa2.assign(graph, {
          iterations: sample.nodes.length > 2000 ? 40 : 80,
          settings: { gravity: 1.5, scalingRatio: 6, barnesHutOptimize: sample.nodes.length > 800, slowDown: 4, strongGravityMode: false },
        });
        settleIsolated(graph);
        if (cancelled) return;
        const renderer = new Sigma(graph, el, {
          settings: {
            renderLabels: false,
            hideEdgesOnMove: true,
            enableNodeDrag: false,
            allowInvalidContainer: true,
            stagePadding: 18,
            // sigma v4 defaults to sizes in graph units; after ForceAtlas2 those are hundreds wide, which made
            // every node sub-pixel (only edges were visible). Our sizes are pixels.
            itemSizesReference: "screen",
          },
        });
        sigmaRef.current = renderer as unknown as { kill: () => void; refresh: () => void };
        graphRef.current = graph as unknown as { setNodeAttribute: (n: string, k: string, v: unknown) => void };
        setStatus("ready");
      } catch {
        if (!cancelled) setStatus("nowebgl");
      }
    })();
    return () => {
      cancelled = true;
      sigmaRef.current?.kill();
      sigmaRef.current = null;
      graphRef.current = null;
    };
  }, [sample, layout]);

  // Recolour without rebuilding.
  useEffect(() => {
    const g = graphRef.current;
    if (!g || !sample || status !== "ready") return;
    const bins = mode === "alignment" && sample.alignment ? sample.alignment[Math.min(frame, sample.alignment.length - 1)] : null;
    sample.nodes.forEach((n, i) => {
      const color = bins ? ALIGNMENT_HEX[Math.max(0, Math.min(4, bins[i] ?? 0))] : REGION_HEX[n.region as RegionId] ?? "#888";
      g.setNodeAttribute(String(i), "color", color);
    });
  }, [mode, frame, sample, status]);

  // Animation over rounds.
  useEffect(() => {
    if (!playing || snapshots < 2) return;
    const id = setInterval(() => {
      setFrame((f) => {
        if (f >= snapshots - 1) {
          setPlaying(false);
          return f;
        }
        return f + 1;
      });
    }, 900);
    return () => clearInterval(id);
  }, [playing, snapshots]);

  const counts = useMemo(() => {
    if (!sample?.alignment || mode !== "alignment") return null;
    const bins = sample.alignment[Math.min(frame, sample.alignment.length - 1)];
    const c = [0, 0, 0, 0, 0];
    bins.forEach((b) => (c[Math.max(0, Math.min(4, b))] += 1));
    return c;
  }, [sample, mode, frame]);

  return (
    <div className="stack" style={{ gap: 10 }}>
      <div className="network" aria-label={sample ? `Sampled social network: ${sample.nodes.length} voters and ${sample.edges.length} friendships` : "Social network"} role="img">
        <div ref={containerRef} className="network-canvas" />
        {!sample ? (
          <div className="network-msg">{error ? `Network unavailable: ${error}` : "Sampling the social network…"}</div>
        ) : status === "nowebgl" ? (
          <div className="network-msg">This browser can&apos;t draw the WebGL network view.</div>
        ) : status === "loading" ? (
          <div className="network-msg">Laying out {fmtInt(sample.nodes.length)} voters…</div>
        ) : null}
        {sample ? (
          <div className="network-overlay">
            <span className="badge">
              {fmtInt(sample.nodes.length)} of {fmtInt(totalVoters)} voters · {fmtInt(sample.edges.length)} ties
            </span>
            {!revealed ? <span className="badge">opinions hidden until the vote</span> : null}
          </div>
        ) : null}
      </div>
      <div className="row-between">
        {mode === "region" || !revealed ? (
          <div className="legend">
            {(Object.keys(REGION_META) as RegionId[]).map((r) => (
              <span key={r}>
                <span className="swatch" style={{ background: REGION_HEX[r] }} aria-hidden />
                {REGION_META[r].label}
              </span>
            ))}
          </div>
        ) : (
          <div className="legend" aria-label="Each voter's probability of voting for you">
            <span className="muted">Chance of voting for you:</span>
            {ALIGNMENT_LABELS.map((l, i) => (
              <span key={l}>
                <span className="swatch" style={{ background: ALIGNMENT_HEX[i] }} aria-hidden />
                {l}
                {counts ? <span className="muted num"> {Math.round((100 * counts[i]) / Math.max(1, sample!.nodes.length))}%</span> : null}
              </span>
            ))}
          </div>
        )}
        {revealed ? (
          <div className="seg" role="group" aria-label="Colour voters by">
            <button type="button" aria-pressed={mode === "alignment"} onClick={() => setMode("alignment")}>
              Alignment
            </button>
            <button type="button" aria-pressed={mode === "region"} onClick={() => setMode("region")}>
              Region
            </button>
          </div>
        ) : null}
      </div>
      {revealed && mode === "alignment" && snapshots > 1 ? (
        <div className="row" style={{ flexWrap: "nowrap" }}>
          <button
            type="button"
            className="btn btn-icon btn-sm"
            aria-label={playing ? "Pause animation" : "Play rounds animation"}
            onClick={() => {
              if (!playing && frame >= snapshots - 1) setFrame(0);
              setPlaying((p) => !p);
            }}
          >
            {playing ? <Pause size={15} aria-hidden /> : <Play size={15} aria-hidden />}
          </button>
          <input
            type="range"
            min={0}
            max={snapshots - 1}
            value={frame}
            aria-label="Round shown"
            aria-valuetext={revealRounds?.[frame] ?? `Snapshot ${frame}`}
            onChange={(e) => {
              setPlaying(false);
              setFrame(Number(e.target.value));
            }}
          />
          <span className="badge mono" style={{ minWidth: 92, justifyContent: "center" }}>
            {revealRounds?.[frame] ?? (frame === 0 ? "Start" : `After R${frame}`)}
          </span>
        </div>
      ) : null}
    </div>
  );
}

export default NetworkView;
