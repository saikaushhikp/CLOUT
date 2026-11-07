"use client";

import { useState } from "react";
import { BarChart3, BookOpen, Brain, ClipboardList, Map as MapIcon, Search, Stamp } from "lucide-react";
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
        <div>
          <span className="kicker">Phase 1 · Survey desk</span>
          <h1>{tab === "research" ? "Find out what the regions want" : "Write the manifesto"}</h1>
        </div>
        <div className="row">
          <span className="stat-sub num">
            {obs.queries.length} questions · {formatMoney(spent)} spent
          </span>
          <div className="seg" role="tablist" aria-label="Survey steps">
            <button type="button" role="tab" aria-selected={tab === "research"} onClick={() => setTab("research")}>
              <Search size={14} aria-hidden style={{ verticalAlign: "-2px", marginRight: 6 }} />
              1 · Research
            </button>
            <button type="button" role="tab" aria-selected={tab === "manifesto"} onClick={() => setTab("manifesto")}>
              <Stamp size={14} aria-hidden style={{ verticalAlign: "-2px", marginRight: 6 }} />
              2 · Manifesto
            </button>
          </div>
        </div>
      </div>

      {tab === "research" ? (
        <div className="grid-survey" role="tabpanel">
          <div className="stack">
            <Panel title="Free: national top-3" icon={BarChart3} hint="exact · free">
              <Top3Bars />
            </Panel>
            <Panel title="Ask a region" icon={Search}>
              <QueryBuilder />
            </Panel>
          </div>
          <div className="stack">
            <Panel title="Region dossiers" icon={MapIcon}>
              <RegionDossiers />
            </Panel>
            <Panel title="Knowledge ledger" icon={ClipboardList}>
              <KnowledgeLedger />
            </Panel>
          </div>
          <div className="stack">
            <Panel title="Archetype library" icon={BookOpen} actions={null}>
              <SectorLegend />
              <div style={{ height: 10 }} />
              <ArchetypeLibrary />
            </Panel>
            <Panel title="Personalities" icon={Brain} hint="assignment hidden">
              <PersonalityLibrary />
            </Panel>
            <button type="button" className="btn btn-primary btn-lg" onClick={() => setTab("manifesto")}>
              <Stamp size={18} aria-hidden /> Ready — design the manifesto
            </button>
          </div>
        </div>
      ) : (
        <div role="tabpanel">
          <ManifestoDesigner />
          <div style={{ marginTop: 16 }} className="grid-2">
            <Panel title="Your hunches & evidence" icon={MapIcon}>
              <RegionDossiers />
            </Panel>
            <Panel title="Still curious? Ask a region" icon={Search}>
              <QueryBuilder compact />
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
