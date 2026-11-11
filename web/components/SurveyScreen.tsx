"use client";

import { useState } from "react";
import { useObservation } from "@/lib/game";
import { formatMoney, toCents } from "@/lib/money";
import { ArchetypeLibrary, KnowledgeLedger, PersonalityLibrary, QueryBuilder, RegionDossiers, Top3Bars } from "./Intel";
import { ManifestoDesigner } from "./ManifestoDesigner";
import { Panel, SectorLegend } from "./ui";

export function SurveyScreen() {
  const obs = useObservation();
  const [tab, setTab] = useState<"research" | "manifesto">("research");
  const spent = obs.queries.reduce((a, q) => a + toCents(q.cost), 0);
  return (
    <div>
      <div className="subnav">
        <h1>{tab === "research" ? "Find out what the regions want" : "Write the manifesto"}</h1>
        <div className="row">
          <span className="stat-sub num">
            {obs.queries.length} questions · {formatMoney(spent)} spent
          </span>
          <div className="seg" role="tablist" aria-label="Survey steps">
            <button type="button" role="tab" aria-selected={tab === "research"} onClick={() => setTab("research")}>
              Research
            </button>
            <button type="button" role="tab" aria-selected={tab === "manifesto"} onClick={() => setTab("manifesto")}>
              Manifesto
            </button>
          </div>
        </div>
      </div>

      {tab === "research" ? (
        <div className="grid-survey" role="tabpanel">
          <div className="stack">
            <Panel title="National top 3" hint="free">
              <Top3Bars />
            </Panel>
            <Panel title="Ask a region">
              <QueryBuilder />
            </Panel>
          </div>
          <div className="stack">
            <Panel title="Regions">
              <RegionDossiers />
            </Panel>
            <Panel title="Responses">
              <KnowledgeLedger />
            </Panel>
          </div>
          <div className="stack">
            <Panel title="Archetypes">
              <SectorLegend />
              <div style={{ height: 10 }} />
              <ArchetypeLibrary />
            </Panel>
            <Panel title="Personalities">
              <PersonalityLibrary />
            </Panel>
          </div>
        </div>
      ) : (
        <div role="tabpanel">
          <ManifestoDesigner />
        </div>
      )}
    </div>
  );
}
