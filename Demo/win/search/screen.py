"""Stage 1: screen seeds for the WIN demo (no campaign loses, campaign wins big).

Usage: python Demo/win/search/screen.py --seeds 1-3200 --device cuda:0 --out <jsonl> [--part 0 --parts 2]
       python Demo/win/search/screen.py --seed-list 100117,104778 --device cuda:2 --out <jsonl>

Per seed (N = 100,000, ₫850 − ₫20 for 2 questions = ₫830, T = 5), on the canonical streams:
1. true electorate (canonical CPU draw → device), ORACLE manifesto (``oracle_manifesto``), anticipated
   election-day electorate v_T; ``E_none_oracle`` = exact no-campaign expected share of the ORACLE manifesto
   rounded to whole percent (the no-campaign v_T does not depend on the manifesto).
2. gate: if the continuous ORACLE no-campaign share > 0.47 the seed is recorded and skipped (no-campaign could win).
3. otherwise the campaign proxy (the multi-start PG best-no-campaign check, ``nocamp_check.py``, runs on the
   shortlist only): the ORACLE manifesto played by the ORACLE greedy planner and 7 scripted open-loop plans
   (workers in the friendliest region in round 1 + heavy early media) with ``x_batch.run_batch``
   (aggregation="exact" = the engine's own index_add, so CUDA results match the API). ``E_best`` = best of these.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F401,F403

import x_policies  # noqa: E402
from x_batch import run_batch  # noqa: E402

from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402

GATE = 0.47


def templates(e, b, cfg) -> dict:
    sup = region_support(e, b, cfg)
    r = max(range(4), key=lambda i: sup[i])
    Z = (0, 0, 0, 0)

    def one(k):
        return tuple(k if i == r else 0 for i in range(4))
    return {
        "w11_m123": Plan([one(11), Z, Z, Z, Z], [ALL, ALL, ALL, NONE, NONE]),
        "w6w5_m123": Plan([one(6), one(5), Z, Z, Z], [ALL, ALL, ALL, NONE, NONE]),
        "w21_m12": Plan([one(21), Z, Z, Z, Z], [ALL, ALL, NONE, NONE, NONE]),
        "w31_m1": Plan([one(31), Z, Z, Z, Z], [ALL, NONE, NONE, NONE, NONE]),
        "w41": Plan([one(41), Z, Z, Z, Z], [NONE] * 5),
        "w1_m1234": Plan([one(1), Z, Z, Z, Z], [ALL, ALL, ALL, ALL, NONE]),
        "w11_m124": Plan([one(11), Z, Z, Z, Z], [ALL, ALL, NONE, ALL, NONE]),
    }, r, sup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None, help="range a-b")
    ap.add_argument("--seed-list", default=None)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=12)
    args = ap.parse_args()
    if args.device in ("cuda:1", "1"):
        raise SystemExit("never use GPU 1")
    torch.set_num_threads(4)
    dev = torch.device(args.device)
    if args.seeds:
        a, b = (int(x) for x in args.seeds.split("-"))
        seeds = list(range(a, b + 1))
    else:
        seeds = [int(x) for x in args.seed_list.split(",")]
    seeds = seeds[args.part::args.parts]
    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)["seed"] for line in open(args.out)}
    seeds = [s for s in seeds if s not in done]
    t0 = time.time()
    cfg0 = cfg_for(0, dev)
    with open(args.out, "a") as fh:
        pending = []

        def flush(batch):
            es, bs, ss = [x["e"] for x in batch], [x["b"] for x in batch], [x["seed"] for x in batch]
            res = {s: {} for s in ss}
            for name in ("none", "planner"):
                outs = run_batch(es, cfg0, torch.stack(bs), x_policies.GATE_POLICIES[name], seeds=ss, aggregation="exact")
                for s, o in zip(ss, outs):
                    res[s][name] = o.result.expected_share
                    if name == "planner":
                        res[s]["planner_plan"] = [[list(a.hires_by_region), [int(x) for x in a.media_regions]]
                                                  for a in o.allocations]
            tps = [templates(e, b, cfg0) for e, b in zip(es, bs)]
            for tname in tps[0][0]:
                outs = run_batch(es, cfg0, torch.stack(bs), [tp[0][tname] for tp in tps], seeds=ss, aggregation="exact")
                for s, o in zip(ss, outs):
                    res[s][tname] = o.result.expected_share
            for x, tp in zip(batch, tps):
                s = x["seed"]
                sc = {k: v for k, v in res[s].items() if k != "planner_plan"}
                camp = {k: v for k, v in sc.items() if k != "none"}
                best = max(camp, key=lambda k: camp[k])
                row = {**x["row"], "gated_out": False, "friendly_region": tp[1],
                       "region_support_v0": [round(v, 4) for v in tp[2]],
                       "E": {k: round(v, 6) for k, v in sc.items()}, "best_template": best, "E_best": round(camp[best], 6),
                       "gap_best_minus_nocamp_best": round(camp[best] - x["row"]["nocamp_best"]["continuous_share"], 6),
                       "planner_plan": res[s]["planner_plan"]}
                fh.write(json.dumps(row) + "\n")
            fh.flush()

        for i, s in enumerate(seeds):
            cfg = cfg_for(s, dev)
            e = generate_electorate(cfg, device=dev)
            b0 = oracle_manifesto(e, cfg)
            v_t = anticipate(e, cfg)
            pct = int_percent(b0.tolist())
            b = torch.tensor(web_manifesto(pct), dtype=e.preferences.dtype, device=dev)
            row = {"seed": s, "sizes": [int(x) for x in e.sizes.tolist()],
                   "sigma": [int(x) for x in e.archetype_of_region.tolist()],
                   "tau": [int(x) for x in e.personality_of_region.tolist()],
                   "oracle_manifesto": [round(float(x), 6) for x in b0.tolist()], "manifesto_pct": pct,
                   "E_none_oracle_cont": share_of(v_t, b0, cfg), "E_none_oracle_int": share_of(v_t, b, cfg)}
            if row["E_none_oracle_cont"] > GATE:
                fh.write(json.dumps({**row, "gated_out": True}) + "\n")
            else:
                # cheap proxy for the best no-campaign share here; the multi-start PG check runs on the shortlist
                row["nocamp_best"] = {"continuous_share": row["E_none_oracle_cont"], "method": "oracle_manifesto (proxy)"}
                pending.append({"seed": s, "e": e, "b": b, "row": row})
                if len(pending) >= args.chunk:
                    flush(pending)
                    pending = []
                    torch.cuda.empty_cache() if dev.type == "cuda" else None
            if (i + 1) % 50 == 0:
                fh.flush()
                print(f"{i + 1}/{len(seeds)} seeds, {time.time() - t0:.0f}s", flush=True)
        if pending:
            flush(pending)
    print(f"done {len(seeds)} seeds in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
