"""E3 — Word-of-mouth cascades and the SIR spectral threshold (spec §3.1, §11; lecture: Infection Models —
linearised SIR  i̇ ≈ (βA − γI) i  grows iff  β λ_max(A)/γ > 1).

Factor: transmission τ_c ∈ {0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.07, 0.12} with ρ = 0.5, L = 3 hops (default
N = 10 000, k̄ = 30); the same seeds for every τ_c (paired).

Discrete-hop analogue of the lecture threshold: with I_h the infectious set at hop h, linearising the engine's
cascade (transmit w.p. τ per edge, then recover w.p. ρ) gives  New_{h+1} = τ A I_{h+1},  I_{h+1} = (1 − ρ) I_h + New_h,
so the per-hop growth factor is  g = 1 − ρ + τ λ_max  and the cascade grows hop-over-hop iff  x = τ λ_max/ρ > 1.
Mean-field reach per seed (first hop from a random seed uses the mean degree k̄):
    R_mf = τ k̄ · Σ_{h=0}^{L−1} g^h.
Per seed and τ_c:
* Reach probe: in each region, 250 random voters (= the 10 free workers' contacts) seed a cascade on the full
  graph (2 trials); reach per spreading seed, per-hop new counts (→ observed growth new_3/new_2), in-region share.
  Plotted against x_rr = τ λ_max(A_rr)/ρ (spec's regional threshold) and x_full = τ λ_max(A)/ρ.
* Outcomes: ORACLE manifesto + oracle planner (win rate); informed manifesto with pay_workers / pay_media /
  pay_split (balance.py): how often paid workers beat paid media, best pattern frequencies.

Usage: python experiments/e3_cascades.py --seeds 200 --n 10000 --device auto --workers 6 --out reports/v2/e3_cascades.json
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import p7_common as pc

TAUS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.07, 0.12)
SEEDS_PER_PROBE = 250
TRIALS = 2


def run_task(task):
    import torch
    from balance import POLICIES
    from dumbocracy.analysis.spectral import largest_eigenpair, region_lambda_max
    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.dynamics.campaign import sir_cascade
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.simulate import simulate_world
    from dumbocracy.strategies.oracle import informed_manifesto, oracle_manifesto

    seed, n, device = task
    t0 = time.perf_counter()
    dev = torch.device(device)
    base = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(base, device=dev)
    lam_rr = region_lambda_max(e.graph, e.region_of)
    lam_full = largest_eigenpair(e.graph)[0]
    kbar = 2 * e.graph.num_edges / n
    region_deg = [float(e.graph.degree[e.region_of == r].double().mean()) for r in range(4)]
    can_spread = ~e.susceptibility.zealot
    members = [torch.nonzero(e.region_of == r).squeeze(1) for r in range(4)]
    bo = oracle_manifesto(e, base)
    bi = informed_manifesto(e, base)
    rho, hops = base.workers.recovery, base.workers.hops
    row = {"seed": seed, "lambda_max_rr": lam_rr, "lambda_max_full": lam_full, "mean_degree": kbar,
           "region_mean_degree": region_deg, "sizes": e.sizes.tolist(),
           "personality": e.personality_of_region.tolist(), "taus": {}}
    for tau in TAUS:
        cfg = dataclasses.replace(base, workers=dataclasses.replace(base.workers, transmission=tau))
        probe = []
        for r in range(4):
            reached = spreading = 0
            per_hop = [0] * hops
            in_region = 0
            for trial in range(TRIALS):
                pick = torch.Generator().manual_seed(seed * 1000 + 10 * r + trial)
                chosen = members[r][torch.randperm(members[r].numel(), generator=pick)[:SEEDS_PER_PROBE].to(members[r].device)]
                seeds = torch.zeros(n, dtype=torch.bool, device=dev)
                seeds[chosen] = True
                gen = torch.Generator(device=dev).manual_seed(seed * 7919 + 100 * r + trial)
                res = sir_cascade(e.graph, seeds, can_spread, tau, rho, hops, gen)
                reached += int(res.reached.sum())
                in_region += int((res.reached & (e.region_of == r)).sum())
                spreading += int((seeds & can_spread).sum())
                per_hop = [a + b for a, b in zip(per_hop, res.per_hop_new)]
            probe.append({"reach_per_seed": reached / spreading, "per_hop_new": per_hop,
                          "in_region_share": in_region / max(1, reached),
                          "x_rr": tau * lam_rr[r] / rho, "x_full": tau * lam_full / rho})
        runs = {}
        out = simulate_world(e, cfg, bo, POLICIES["planner"])
        runs["oracle/planner"] = {"expected": out.result.expected_share, "victory": out.result.victory,
                                  "extra_workers": sum(a.extra_workers for a in out.allocations),
                                  "media_airings": sum(sum(a.media_regions) for a in out.allocations)}
        for pol in ("pay_workers", "pay_media", "pay_split"):
            out = simulate_world(e, cfg, bi, POLICIES[pol])
            runs[f"informed/{pol}"] = {"expected": out.result.expected_share, "victory": out.result.victory}
        row["taus"][str(tau)] = {"probe": probe, "runs": runs,
                                 "best_paid_pattern": max(("pay_workers", "pay_media", "pay_split"),
                                                          key=lambda p: runs[f"informed/{p}"]["expected"])}
    row["seconds"] = time.perf_counter() - t0
    return row


def mean_field(tau, lam, kbar, rho=0.5, hops=3):
    g = 1 - rho + tau * lam
    return tau * kbar * sum(g ** h for h in range(hops)), g


def summarise(rows):
    import numpy as np
    out = {"by_tau": {}, "collapse": {}, "paired": {}}
    for tau in TAUS:
        k = str(tau)
        rs = [r["taus"][k] for r in rows]
        probes = [p for r in rs for p in r["probe"]]
        hop = np.array([p["per_hop_new"] for p in probes], dtype=float)
        growth = [p["per_hop_new"][2] / p["per_hop_new"][1] for p in probes if p["per_hop_new"][1] > 0]
        mf = [mean_field(tau, r["lambda_max_full"], r["mean_degree"]) for r in rows]
        c = {"reach_per_seed": pc.mean_ci([p["reach_per_seed"] for p in probes]),
             "in_region_share": pc.mean_ci([p["in_region_share"] for p in probes]),
             "x_rr": pc.mean_ci([p["x_rr"] for p in probes]), "x_full": pc.mean_ci([p["x_full"] for p in probes]),
             "hop_growth_obs": pc.mean_ci(growth), "hop_growth_pred_full": pc.mean_ci([m[1] for m in mf]),
             "hop_growth_pred_rr": pc.mean_ci([1 - 0.5 + tau * lam for r in rows for lam in r["lambda_max_rr"]]),
             "reach_pred_mean_field": pc.mean_ci([m[0] for m in mf]),
             "per_hop_new_mean": hop.mean(0).tolist(),
             "pay_workers_beats_media": pc.rate([r["runs"]["informed/pay_workers"]["expected"] > r["runs"]["informed/pay_media"]["expected"] for r in rs]),
             "best_paid_pattern": {p: sum(r["best_paid_pattern"] == p for r in rs) for p in ("pay_workers", "pay_media", "pay_split")},
             "oracle_planner_extra_workers": pc.mean_ci([r["runs"]["oracle/planner"]["extra_workers"] for r in rs]),
             "oracle_planner_media_airings": pc.mean_ci([r["runs"]["oracle/planner"]["media_airings"] for r in rs])}
        for name in ("oracle/planner", "informed/pay_workers", "informed/pay_media", "informed/pay_split"):
            c[f"win_{name}"] = pc.rate([r["runs"][name]["victory"] for r in rs])
            c[f"expected_{name}"] = pc.mean_ci([r["runs"][name]["expected"] for r in rs])
        c["spearman_reach_vs_lambda_rr"] = pc.spearman([p["reach_per_seed"] for r in rows for p in r["taus"][k]["probe"]],
                                                        [lam for r in rows for lam in r["lambda_max_rr"]])
        out["by_tau"][k] = c
    # collapse: bin all (seed, region, τ) probes by x and compare spread of reach within bins, x_rr vs x_full
    pts = [(p["x_rr"], p["x_full"], p["reach_per_seed"], p["per_hop_new"]) for r in rows for k in r["taus"] for p in r["taus"][k]["probe"]]
    xr, xf, y = (np.array([q[i] for q in pts]) for i in range(3))
    for label, x in (("x_rr", xr), ("x_full", xf)):
        edges = np.quantile(x, np.linspace(0, 1, 25))
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, 23)
        cv = [float(np.std(y[idx == b]) / np.mean(y[idx == b])) for b in range(24) if (idx == b).sum() > 5]
        out["collapse"][label] = {"mean_within_bin_cv_reach": float(np.mean(cv)),
                                  "binned": [{"x": float(np.median(x[idx == b])), "reach": float(np.mean(y[idx == b])),
                                              "lo": float(np.quantile(y[idx == b], 0.1)), "hi": float(np.quantile(y[idx == b], 0.9))}
                                             for b in range(24) if (idx == b).sum() > 5]}
    # where does growth cross 1? (empirical crossing of hop growth vs x_full and x_rr)
    for label, x in (("x_rr", xr), ("x_full", xf)):
        g = np.array([q[3][2] / q[3][1] if q[3][1] > 0 else np.nan for q in pts])
        ok = ~np.isnan(g)
        order = np.argsort(x[ok])
        xs, gs = x[ok][order], g[ok][order]
        # running median crossing
        win = max(20, len(xs) // 40)
        med = np.array([np.median(gs[max(0, i - win):i + win]) for i in range(len(xs))])
        cross = xs[np.argmax(med >= 1.0)] if (med >= 1.0).any() else None
        out["collapse"][label]["growth_crosses_1_at_x"] = None if cross is None else float(cross)
    # paired: each τ vs default 0.03 for oracle planner wins, and pay_workers − pay_media difference
    base = "0.03"
    for tau in TAUS:
        k = str(tau)
        out["paired"][k] = {
            "oracle_expected_vs_0.03": pc.paired([r["taus"][k]["runs"]["oracle/planner"]["expected"] for r in rows],
                                                 [r["taus"][base]["runs"]["oracle/planner"]["expected"] for r in rows]),
            "oracle_win_mcnemar_vs_0.03": pc.mcnemar([r["taus"][k]["runs"]["oracle/planner"]["victory"] for r in rows],
                                                     [r["taus"][base]["runs"]["oracle/planner"]["victory"] for r in rows]),
            "workers_minus_media_expected": pc.paired([r["taus"][k]["runs"]["informed/pay_workers"]["expected"] for r in rows],
                                                      [r["taus"][k]["runs"]["informed/pay_media"]["expected"] for r in rows])}
    out["lambda_max_rr"] = pc.mean_ci([x for r in rows for x in r["lambda_max_rr"]])
    out["lambda_max_full"] = pc.mean_ci([r["lambda_max_full"] for r in rows])
    out["threshold_tau_rr"] = pc.mean_ci([0.5 / x for r in rows for x in r["lambda_max_rr"]])
    out["threshold_tau_full"] = pc.mean_ci([0.5 / r["lambda_max_full"] for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--first-seed", type=int, default=9500)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="reports/v2/e3_cascades.json")
    ap.add_argument("--resummarise", action="store_true", help="recompute the summary from the rows saved in --out")
    args = ap.parse_args()
    if args.resummarise:
        import json
        from pathlib import Path
        report = json.loads(Path(args.out).read_text())
        report.update(summarise(report["rows"]))
        pc.save(args.out, report)
        return
    device = pc.pick_device(args.device)
    tasks = [(s, args.n, device) for s in range(args.first_seed, args.first_seed + args.seeds)]
    print(f"E3 device={device} tasks={len(tasks)}", flush=True)
    start = time.perf_counter()
    rows = pc.run_tasks(run_task, tasks, args.workers, "E3")
    report = {"experiment": "E3 cascades vs SIR spectral threshold", "num_seeds": args.seeds, "num_voters": args.n,
              "taus": TAUS, "recovery": 0.5, "hops": 3, "seeds_per_probe": SEEDS_PER_PROBE, "trials": TRIALS,
              **summarise(rows), "provenance": pc.provenance(args, device, time.perf_counter() - start, args.workers),
              "rows": rows}
    pc.save(args.out, report)


if __name__ == "__main__":
    main()
