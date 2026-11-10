"""E3 — party-worker cascades (P8 port; spec §3.1, §11).

TODO(P8, owner re-run): E3 MUST BE REDESIGNED for the Independent Cascade. The P7 version swept the SIR
transmission τ_c ∈ {0.005 … 0.12} against the spectral threshold τ λ_max/ρ = 1. P8 removed τ_c, ρ and L:
workers are permanent voter nodes and a cascade advances one hop per round with chance γ_j P_j (attempt cap 3).
The natural control parameter is now the branching number R0 ≈ mean over spreader edges of γP (it depends on
the manifesto via P, on personalities via γ and on degree), not a config knob. Until the owner's redesign this
script only keeps a runnable, minimal measurement so the P7 pipeline still imports:

* Probe: per region, ``WORKERS_PER_PROBE`` workers bought in round 1 (b = oracle manifesto), IC converts per hop
  for T hops on v(0) (no social step) and R0_r = k̄_r · E_r[γP], plus λ_max(A_rr).
* Outcomes (``simulate_world``): ORACLE manifesto + oracle planner; informed manifesto with pay_workers /
  front_workers / pay_media / pay_split (balance.py).

The P7 results in reports/v2/e3_cascades.json were produced by the SIR mechanism and are NOT reproducible with
this script. Do not re-run E3 without the owner's go-ahead.

Usage: python experiments/e3_cascades.py --seeds 200 --n 10000 --device auto --workers 6 --out reports/v2/e3_cascades.json
"""
from __future__ import annotations

import argparse
import time

import p7_common as pc

WORKERS_PER_PROBE = 5
TRIALS = 2
PATTERNS = ("pay_workers", "front_workers", "pay_media", "pay_split")


def run_task(task):
    import torch
    from balance import POLICIES
    from dumbocracy.analysis.spectral import region_lambda_max
    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.dynamics.campaign import conversion_chance, eligible_workers, ic_hop, pick_workers
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.simulate import simulate_world
    from dumbocracy.strategies.oracle import informed_manifesto, oracle_manifesto

    seed, n, device = task
    t0 = time.perf_counter()
    dev = torch.device(device)
    cfg = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=dev)
    zealot = e.susceptibility.zealot
    lam_rr = region_lambda_max(e.graph, e.region_of)
    bo = oracle_manifesto(e, cfg)
    bi = informed_manifesto(e, cfg)
    chance = conversion_chance(e.susceptibility.campaign, e.preferences, bo, cfg.voting.sensitivity, cfg.voting.threshold)
    rows_idx = e.graph.rows()
    eligible = eligible_workers(e.graph.degree, zealot, torch.zeros_like(zealot), cfg.workers.min_degree)
    probe = []
    for r in range(4):
        mask = e.region_of == r
        per_hop = [0] * cfg.rounds
        for trial in range(TRIALS):
            hires = [0] * 4
            hires[r] = WORKERS_PER_PROBE
            ids = torch.cat(pick_workers(e.region_of, eligible, hires,
                                         torch.Generator(device=dev).manual_seed(seed * 1000 + 10 * r + trial)))
            pinned = torch.zeros(n, dtype=torch.bool, device=dev)
            pinned[ids] = True
            spreaders, attempts = pinned.clone(), torch.zeros(n, dtype=torch.int64, device=dev)
            for h in range(cfg.rounds):
                gen = torch.Generator(device=dev).manual_seed(seed * 7919 + 100 * r + 10 * trial + h)
                spreaders, attempts = ic_hop(e.graph, spreaders, chance, attempts, pinned | zealot,
                                             cfg.workers.attempt_cap, gen, rows_idx)
                per_hop[h] += int(spreaders.sum())
        probe.append({"per_hop_converts": [x / TRIALS for x in per_hop], "lambda_max_rr": lam_rr[r],
                      "r0": float(e.graph.degree[mask].double().mean()) * float(chance[mask].mean())})
    runs = {}
    out = simulate_world(e, cfg, bo, POLICIES["planner"])
    runs["oracle/planner"] = {"expected": out.result.expected_share, "victory": out.result.victory,
                              "workers": sum(a.total_hires for a in out.allocations),
                              "media_airings": sum(sum(a.media_regions) for a in out.allocations),
                              "converted": out.converted}
    for pol in PATTERNS:
        out = simulate_world(e, cfg, bi, POLICIES[pol])
        runs[f"informed/{pol}"] = {"expected": out.result.expected_share, "victory": out.result.victory}
    return {"seed": seed, "sizes": e.sizes.tolist(), "probe": probe, "runs": runs,
            "best_paid_pattern": max(PATTERNS, key=lambda p: runs[f"informed/{p}"]["expected"]),
            "seconds": time.perf_counter() - t0}


def summarise(rows):
    out = {"r0": pc.mean_ci([p["r0"] for r in rows for p in r["probe"]]),
           "spearman_total_converts_vs_r0": pc.spearman([sum(p["per_hop_converts"]) for r in rows for p in r["probe"]],
                                                         [p["r0"] for r in rows for p in r["probe"]]),
           "best_paid_pattern": {p: sum(r["best_paid_pattern"] == p for r in rows) for p in PATTERNS},
           "oracle_planner_workers": pc.mean_ci([r["runs"]["oracle/planner"]["workers"] for r in rows])}
    for name in ("oracle/planner",) + tuple(f"informed/{p}" for p in PATTERNS):
        out[f"win_{name}"] = pc.rate([r["runs"][name]["victory"] for r in rows])
        out[f"expected_{name}"] = pc.mean_ci([r["runs"][name]["expected"] for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--first-seed", type=int, default=9500)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="reports/v2/e3_cascades.json")
    args = ap.parse_args()
    device = pc.pick_device(args.device)
    tasks = [(s, args.n, device) for s in range(args.first_seed, args.first_seed + args.seeds)]
    print(f"E3 (P8 IC port, needs redesign) device={device} tasks={len(tasks)}", flush=True)
    start = time.perf_counter()
    rows = pc.run_tasks(run_task, tasks, args.workers, "E3")
    report = {"experiment": "E3 party-worker IC cascades (P8 port; redesign pending)", "num_seeds": args.seeds,
              "num_voters": args.n, "workers_per_probe": WORKERS_PER_PROBE, "trials": TRIALS, **summarise(rows),
              "provenance": pc.provenance(args, device, time.perf_counter() - start, args.workers), "rows": rows}
    pc.save(args.out, report)


if __name__ == "__main__":
    main()
