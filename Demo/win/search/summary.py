"""Stage 4: write Demo/win/screening.jsonl and Demo/win/screening_summary.json from the stage outputs.

Usage: python Demo/win/search/summary.py --screening <screen.jsonl> --nocamp <nocamp.jsonl> --search <search.jsonl>
       --chosen S --why "..."
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WIN  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening", required=True)
    ap.add_argument("--nocamp", required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--chosen", type=int, required=True)
    ap.add_argument("--why", required=True)
    ap.add_argument("--extra-seed-source", default="")
    args = ap.parse_args()
    shutil.copyfile(args.screening, WIN / "screening.jsonl")
    rows = [json.loads(line) for line in open(args.screening)]
    full = [r for r in rows if not r["gated_out"]]
    nocamp = [r["E_none_oracle_cont"] for r in rows]
    srch = [json.loads(line) for line in open(args.search)]
    nc = [json.loads(line) for line in open(args.nocamp)]
    for s in srch:
        s["search_rank_key"] = s["E_fresh_mean"]
    summ = {
        "settings": {"num_voters": 100000, "rounds": 5, "budget": "850.00", "campaign_budget_after_2_queries": "830.00",
                     "player_id": "p0"},
        "requirement": ("Win with the scripted campaign (as large a share as possible) AND lose with no campaign "
                        "(same questions and manifesto, empty plans); strong version: even the best possible "
                        "no-campaign manifesto loses (target <= 0.46 expected share)."),
        "seeds_screened": len(rows),
        "seed_sources": {"contiguous": "1..4000", "from_Demo_lose_screening": args.extra_seed_source},
        "stage1_screen": {
            "script": "Demo/win/search/screen.py",
            "gate": "ORACLE manifesto no-campaign expected share (continuous) > 0.47 -> skipped",
            "passed_gate": len(full),
            "no_campaign_oracle_share_quantiles_0_5_25_50": np.quantile(nocamp, [0, 0.05, 0.25, 0.5]).tolist(),
            "share_with_nocamp_le_0.46": float(np.mean([x <= 0.46 for x in nocamp])),
            "campaign_proxy_E_best_among_nocamp_le_0.46": {
                "max": max(r["E_best"] for r in full if r["E_none_oracle_cont"] <= 0.46),
                "p90": float(np.percentile([r["E_best"] for r in full if r["E_none_oracle_cont"] <= 0.46], 90))},
            "best_screening_plan_counts": dict(Counter(r["best_template"] for r in full)),
            "top25_nocamp_le_0.46": [{"seed": r["seed"], "E_best": r["E_best"], "best_template": r["best_template"],
                                      "E_planner": r["E"]["planner"], "E_none_oracle": r["E_none_oracle_cont"],
                                      "sizes": r["sizes"], "sigma": r["sigma"]}
                                     for r in sorted((r for r in full if r["E_none_oracle_cont"] <= 0.46),
                                                     key=lambda r: -r["E_best"])[:25]],
        },
        "stage1b_best_no_campaign_manifesto": {"script": "Demo/win/search/nocamp_check.py", "rows": nc},
        "stage2_plan_search": {"script": "Demo/win/search/search.py", "seeds": srch},
        "chosen_seed": args.chosen, "why": args.why,
        "previous_choice": ("Seed 298 (first version of this demo) was replaced: its manifesto alone wins with ~57% "
                            "without any campaign, which the new requirement forbids."),
    }
    (WIN / "screening_summary.json").write_text(json.dumps(summ, indent=1))
    print("ok", len(rows))


if __name__ == "__main__":
    main()
