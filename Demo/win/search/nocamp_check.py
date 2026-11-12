"""Stage 1b: the strong no-campaign check on the shortlist.

Usage: python Demo/win/search/nocamp_check.py --screening <screen.jsonl> --top 20 --max-nocamp 0.46 --out <jsonl>

Takes the ``--top`` seeds by campaign proxy ``E_best`` among those whose ORACLE no-campaign share ≤ ``--max-nocamp``
and computes the best possible no-campaign share with ``common.best_no_campaign_manifesto`` (multi-start PG on the
5-round no-campaign forecast v_T, 64 random + 13 structured starts, plus the best whole-percent manifesto).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F401,F403

from dumbocracy.electorate import generate_electorate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening", required=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--max-nocamp", type=float, default=0.46)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    dev = torch.device(args.device)
    rows = [json.loads(line) for line in open(args.screening)]
    ok = [r for r in rows if not r["gated_out"] and r["E_none_oracle_cont"] <= args.max_nocamp]
    ok.sort(key=lambda r: -r["E_best"])
    with open(args.out, "a") as fh:
        for r in ok[:args.top]:
            t = time.time()
            cfg = cfg_for(r["seed"], dev)
            e = generate_electorate(cfg, device=dev)
            res = best_no_campaign_manifesto(e, cfg, random_starts=64)
            out = {"seed": r["seed"], "E_best": r["E_best"], "E_none_oracle_cont": r["E_none_oracle_cont"], **res,
                   "best_share_any": max(res["continuous_share"], r["E_none_oracle_cont"]), "seconds": time.time() - t}
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            print(out["seed"], round(out["E_best"], 4), round(out["E_none_oracle_cont"], 4), round(out["continuous_share"], 4),
                  out["integer_pct"], round(out["integer_share"], 4), f"{out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
