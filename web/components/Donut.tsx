import { SECTOR_META, sectorLabel } from "@/lib/constants";
import { SECTOR_IDS } from "@/lib/types";

/** Five-sector donut (values are percentages summing to 100). Pure SVG, segments separated by a surface gap. */
export function Donut({ values, size = 200, thickness = 30 }: { values: number[]; size?: number; thickness?: number }) {
  const r = (size - thickness) / 2;
  const c = size / 2;
  const total = values.reduce((a, b) => a + b, 0) || 1;
  let angle = -Math.PI / 2;
  const arcs = values.map((v, i) => {
    const sweep = (v / total) * Math.PI * 2;
    const start = angle;
    angle += sweep;
    return { i, v, start, sweep };
  });
  const label = SECTOR_IDS.map((s, i) => `${sectorLabel(s)} ${values[i]}%`).join(", ");
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Manifesto: ${label}`}>
      <circle cx={c} cy={c} r={r} fill="none" stroke="var(--surface-2)" strokeWidth={thickness} />
      {arcs.map(({ i, v, start, sweep }) => {
        if (v <= 0) return null;
        if (sweep >= Math.PI * 2 - 1e-6)
          return <circle key={i} cx={c} cy={c} r={r} fill="none" stroke={SECTOR_META[SECTOR_IDS[i]].color} strokeWidth={thickness} />;
        const end = start + sweep;
        const x1 = c + r * Math.cos(start);
        const y1 = c + r * Math.sin(start);
        const x2 = c + r * Math.cos(end);
        const y2 = c + r * Math.sin(end);
        const large = sweep > Math.PI ? 1 : 0;
        return (
          <path
            key={i}
            d={`M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`}
            fill="none"
            stroke={SECTOR_META[SECTOR_IDS[i]].color}
            strokeWidth={thickness}
            style={{ transition: "d 0.2s ease" }}
          >
            <title>{`${sectorLabel(SECTOR_IDS[i])} ${v}%`}</title>
          </path>
        );
      })}
      {/* surface-coloured separators between segments */}
      {arcs.map(({ i, v, start }) =>
        v > 0 && v < total ? (
          <line
            key={`sep-${i}`}
            x1={c + (r - thickness / 2 - 1) * Math.cos(start)}
            y1={c + (r - thickness / 2 - 1) * Math.sin(start)}
            x2={c + (r + thickness / 2 + 1) * Math.cos(start)}
            y2={c + (r + thickness / 2 + 1) * Math.sin(start)}
            stroke="var(--surface)"
            strokeWidth={2}
          />
        ) : null,
      )}
    </svg>
  );
}
