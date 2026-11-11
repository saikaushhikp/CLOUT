"use client";

import type { KeyboardEvent } from "react";
import type { Observation, RegionId } from "@/lib/types";
import { REGION_HEX, REGION_META } from "@/lib/constants";
import { fmtCompact } from "@/lib/format";

const S = 400;
const C = S / 2;
const SHAPES: Record<RegionId, string> = {
  N: `M 0 0 L ${S} 0 L ${C} ${C} Z`,
  E: `M ${S} 0 L ${S} ${S} L ${C} ${C} Z`,
  S: `M ${S} ${S} L 0 ${S} L ${C} ${C} Z`,
  W: `M 0 ${S} L 0 0 L ${C} ${C} Z`,
};
const ANCHOR: Record<RegionId, [number, number]> = { N: [C, 70], E: [S - 78, C], S: [C, S - 70], W: [78, C] };

export type MapOverlay = { workers?: number[]; media?: boolean[] };

export function RegionMap({
  obs,
  selected,
  onSelect,
  overlay,
}: {
  obs: Observation;
  selected: RegionId | null;
  onSelect: (r: RegionId) => void;
  overlay?: MapOverlay;
}) {
  const ids: RegionId[] = ["N", "E", "S", "W"];
  const onKey = (e: KeyboardEvent<SVGPathElement>, r: RegionId) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(r);
    }
  };
  return (
    <svg className="region-map" viewBox={`-12 -12 ${S + 24} ${S + 24}`} role="group" aria-label="Map of the four regions">
      <defs>
        <clipPath id="map-clip">
          <rect x={0} y={0} width={S} height={S} rx={26} />
        </clipPath>
        <pattern id="air-hatch" width="10" height="10" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="10" stroke="rgba(255,255,255,0.35)" strokeWidth="4" />
        </pattern>
      </defs>
      <g clipPath="url(#map-clip)">
        {ids.map((r) => {
          const region = obs.regions.find((x) => x.id === r);
          const i = ["N", "S", "E", "W"].indexOf(r);
          const aired = overlay?.media?.[i];
          return (
            <g key={r}>
              <path
                d={SHAPES[r]}
                fill={REGION_HEX[r]}
                className={`region-shape ${selected === r ? "is-selected" : ""}`}
                tabIndex={0}
                role="button"
                aria-pressed={selected === r}
                aria-label={`${REGION_META[r].label}: ${region?.voters ?? 0} voters, mean ${region?.mean_degree.toFixed(1)} friends${aired ? ", media booked" : ""}`}
                onClick={() => onSelect(r)}
                onKeyDown={(e) => onKey(e, r)}
              />
              {aired ? <path d={SHAPES[r]} fill="url(#air-hatch)" pointerEvents="none" /> : null}
            </g>
          );
        })}
      </g>
      {ids.map((r) => {
        const region = obs.regions.find((x) => x.id === r);
        const [x, y] = ANCHOR[r];
        const i = ["N", "S", "E", "W"].indexOf(r);
        const w = overlay?.workers?.[i] ?? 0;
        const aired = overlay?.media?.[i];
        return (
          <g key={`l-${r}`} aria-hidden>
            <text className="map-label" x={x} y={y - 14} textAnchor="middle" fontSize={22}>
              {REGION_META[r].label}
            </text>
            <text className="map-sub" x={x} y={y + 6} textAnchor="middle" fontSize={13}>
              {fmtCompact(region?.voters ?? 0)} voters
            </text>
            {w > 0 || aired ? (
              <text className="map-sub" x={x} y={y + 24} textAnchor="middle" fontSize={12}>
                {w > 0 ? `${w} worker${w > 1 ? "s" : ""}` : ""}
                {w > 0 && aired ? " · " : ""}
                {aired ? "ON AIR" : ""}
              </text>
            ) : null}
            {obs.media_airings[r] > 0 ? (
              <text className="map-sub" x={x} y={y + 40} textAnchor="middle" fontSize={11} opacity={0.85}>
                {obs.media_airings[r]} airing{obs.media_airings[r] > 1 ? "s" : ""} so far
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}
