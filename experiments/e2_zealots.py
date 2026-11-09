"""E2 — Zealots × resilient consensus (spec §2.4, §4, §11; LeBlanc, Zhang, Koutsoukos & Sundaram, IEEE JSAC 2013).

Factors: zealot fraction ζ ∈ {0, .05, .15, .30} × social rule ∈ {DeGroot, W-MSR F=1, W-MSR F=2}; the same seeds
(and hence the same graph, preferences and susceptibilities of the non-zealots) in every cell.

Per seed and cell (default N = 10 000, k̄ = 30):
* ORACLE manifesto anticipated under that rule + oracle marginal planner (balance.py) vs the same manifesto with
  no campaign: win rate, campaign gain E_T − E_0, **persuasion** E_T(planner) − E_T(no campaign), and the mean
  similarity S of non-zealots to b per region. Persuasion is also measured with one fixed manifesto (the DeGroot
  oracle manifesto) so the rule effect is not confounded by a different b.
* **Zealot drag probe** (ζ > 0; a probe outside the game rules): zealots are re-set to the simplex vertex of
  the sector their region's archetype likes least (an extreme, "malicious-looking" stubborn value). Social-only
  dynamics for T = 5 and 30 rounds; drag = mean over regions of the TV distance between the non-zealot regional
  mean with extreme zealots and with the original (honest) zealots. First-order theory: a non-zealot i with
  z_i zealot neighbours feels zealot weight z_i/d_i under DeGroot and max(0, z_i − F)/(d_i − F) under W-MSR
  (extreme zealots are the farthest neighbours, so they are trimmed first) ⇒ predicted drag ratio
  W-MSR/DeGroot = mean[a_i max(0, z_i − F)/(d_i − F)] / mean[a_i z_i/d_i].
* **Theory (per graph)**: min degree; upper bound on r-robustness and on s for (F+1, s)-robustness
  (analysis.robustness, region-union and greedy certificates; exact check is coNP-hard); F-total model
  (needs (F+1, F+1)-robust and ≤ F adversaries in total) and F-local model ((2F+1)-robust sufficient,
  (F+1)-robust necessary, ≤ F adversarial neighbours per normal node): fraction of normal nodes with > F
  zealot neighbours.

Usage: python experiments/e2_zealots.py --seeds 200 --n 10000 --device auto --workers 6 --out reports/v2/e2_zealots.json
"""
from __future__ import annotations

import argparse
import dataclasses
import time

import p7_common as pc

ZETAS = (0.0, 0.05, 0.15, 0.30)
RULES = {"degroot": ("degroot", 1), "wmsr_F1": ("wmsr", 1), "wmsr_F2": ("wmsr", 2)}


def social_only(e, cfg, v, rounds):
    import torch
    from dumbocracy.dynamics.social import aggregate
    from dumbocracy.dynamics.update import step_preferences
    w = cfg.dynamics.social_step * e.susceptibility.social
    empty = torch.zeros((0, e.num_voters), dtype=v.dtype, device=v.device)
    no_b = torch.zeros((0, 5), dtype=v.dtype, device=v.device)
    for _ in range(rounds):
        agg = aggregate(v, e.graph, cfg.dynamics.social_rule, cfg.dynamics.wmsr_trim)
        v = step_preferences(v, agg, w, empty, no_b, e.susceptibility.zealot)
    return v


def regional_means(v, region, keep):
    import torch
    return torch.stack([v[(region == r) & keep].mean(0) for r in range(4)])


def run_task(task):
    import torch
    from balance import POLICIES
    from dumbocracy.analysis.robustness import robustness_upper_bound, rs_upper_bound
    from dumbocracy.config import DynamicsConfig, ElectorateConfig, GameConfig
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.sectors import archetype_means
    from dumbocracy.simulate import simulate_world
    from dumbocracy.strategies.oracle import oracle_manifesto
    from dumbocracy.voting import similarity

    seed, zeta, n, device, robust = task
    t0 = time.perf_counter()
    dev = torch.device(device)
    base = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n, zealot_fraction=zeta))
    e = generate_electorate(base, device=dev)
    g, region, zealot = e.graph, e.region_of, e.susceptibility.zealot
    normal = ~zealot
    rows_ = g.rows()
    zn = torch.bincount(rows_[zealot[g.indices]], minlength=n)  # zealot neighbours per node
    deg = g.degree
    a = base.dynamics.social_step * e.susceptibility.social.double()
    row = {"seed": seed, "zeta": zeta, "num_zealots": int(zealot.sum()), "theory": {}, "rules": {}}
    for F in (1, 2):
        m = normal & (deg > 0)
        row["theory"][f"F{F}"] = {
            "frac_normal_gt_F_zealot_nbrs": float((zn[m] > F).double().mean()),
            "F_total_violated": bool(int(zealot.sum()) > F),
            "pred_drag_ratio": float((a[m] * (zn[m] - F).clamp_min(0) / (deg[m] - F).clamp_min(1)).sum()
                                     / (a[m] * zn[m] / deg[m]).sum()) if zeta > 0 else None}
    row["theory"]["mean_zealot_nbrs"] = float(zn[normal].double().mean())
    row["theory"]["max_zealot_nbrs"] = int(zn[normal].max())
    if robust:
        b = robustness_upper_bound(g, region)
        row["robustness"] = {"r_upper": b.r_upper, "witness": b.witness, "min_degree": b.min_degree,
                             "s_upper_r2": rs_upper_bound(g, region, 2), "s_upper_r3": rs_upper_bound(g, region, 3)}
    # extreme zealot values: vertex of the sector least liked by the region's archetype
    mu = archetype_means(dtype=torch.float64)[e.archetype_of_region]          # [4, 5]
    worst = mu.argmin(1).to(dev)
    extreme = torch.nn.functional.one_hot(worst[region], 5).to(e.preferences.dtype)
    v_extreme = torch.where(zealot.unsqueeze(1), extreme, e.preferences)
    b_fixed = None
    for name, (rule, F) in RULES.items():
        cfg = dataclasses.replace(base, dynamics=DynamicsConfig(social_rule=rule, wmsr_trim=F))
        b = oracle_manifesto(e, cfg)
        if name == "degroot":
            b_fixed = b
        res = {"manifesto": [round(float(x), 4) for x in b]}
        for label, man in (("", b), ("fixed_b_", b_fixed)):
            plan = simulate_world(e, cfg, man, POLICIES["planner"])
            none = simulate_world(e, cfg, man, POLICIES["none"])
            res[f"{label}expected"] = plan.result.expected_share
            res[f"{label}victory"] = plan.result.victory
            res[f"{label}expected_none"] = none.result.expected_share
            res[f"{label}gain"] = plan.share_trajectory[-1] - plan.share_trajectory[0]
            res[f"{label}persuasion"] = plan.result.expected_share - none.result.expected_share
            if not label:
                res["trajectory"] = [round(x, 4) for x in plan.share_trajectory]
                res["trajectory_none"] = [round(x, 4) for x in none.share_trajectory]
                sim_p = similarity(plan.final_preferences, man.to(dev, plan.final_preferences.dtype))
                sim_n = similarity(none.final_preferences, man.to(dev, none.final_preferences.dtype))
                res["region_S"] = [float(sim_p[(region == r) & normal].mean()) for r in range(4)]
                res["region_S_none"] = [float(sim_n[(region == r) & normal].mean()) for r in range(4)]
                mp, mn = regional_means(plan.final_preferences, region, normal), regional_means(none.final_preferences, region, normal)
                res["campaign_shift_tv"] = float(0.5 * (mp - mn).abs().sum(1).mean())
        if zeta > 0:
            for T in (5, 30):
                h = social_only(e, cfg, e.preferences, T)
                x = social_only(e, cfg, v_extreme, T)
                res[f"drag_T{T}"] = float(0.5 * (regional_means(x, region, normal) - regional_means(h, region, normal)).abs().sum(1).mean())
        row["rules"][name] = res
    row["seconds"] = time.perf_counter() - t0
    return row


def summarise(rows):
    import numpy as np
    out = {"cells": {}, "paired": {}, "theory": {}}
    for z in ZETAS:
        rz = [r for r in rows if r["zeta"] == z]
        for name in RULES:
            rr = [r["rules"][name] for r in rz]
            c = {"n": len(rr), "win": pc.rate([x["victory"] for x in rr])}
            for k in ("expected", "expected_none", "gain", "persuasion", "fixed_b_persuasion", "fixed_b_expected",
                      "campaign_shift_tv"):
                c[k] = pc.mean_ci([x[k] for x in rr])
            c["region_S"] = pc.mean_ci([float(np.mean(x["region_S"])) for x in rr])
            c["region_S_none"] = pc.mean_ci([float(np.mean(x["region_S_none"])) for x in rr])
            c["trajectory"] = np.mean([x["trajectory"] for x in rr], 0).tolist()
            c["trajectory_none"] = np.mean([x["trajectory_none"] for x in rr], 0).tolist()
            if z > 0:
                for T in (5, 30):
                    c[f"drag_T{T}"] = pc.mean_ci([x[f"drag_T{T}"] for x in rr])
            out["cells"][f"{z}|{name}"] = c
        # paired rule comparisons within ζ
        deg = [r["rules"]["degroot"] for r in rz]
        for name in ("wmsr_F1", "wmsr_F2"):
            w = [r["rules"][name] for r in rz]
            p = {"persuasion_diff": pc.paired([x["persuasion"] for x in w], [x["persuasion"] for x in deg]),
                 "fixed_b_persuasion_diff": pc.paired([x["fixed_b_persuasion"] for x in w], [x["fixed_b_persuasion"] for x in deg]),
                 "expected_diff": pc.paired([x["expected"] for x in w], [x["expected"] for x in deg]),
                 "win_mcnemar": pc.mcnemar([x["victory"] for x in w], [x["victory"] for x in deg])}
            if z > 0:
                for T in (5, 30):
                    p[f"drag_T{T}_ratio"] = pc.mean_ci([x[f"drag_T{T}"] / d[f"drag_T{T}"] for x, d in zip(w, deg)])
                F = 1 if name == "wmsr_F1" else 2
                p["pred_drag_ratio"] = pc.mean_ci([r["theory"][f"F{F}"]["pred_drag_ratio"] for r in rz])
            out["paired"][f"{z}|{name}_vs_degroot"] = p
        th = {"frac_normal_gt_1_zealot_nbrs": pc.mean_ci([r["theory"]["F1"]["frac_normal_gt_F_zealot_nbrs"] for r in rz]),
              "frac_normal_gt_2_zealot_nbrs": pc.mean_ci([r["theory"]["F2"]["frac_normal_gt_F_zealot_nbrs"] for r in rz]),
              "mean_zealot_nbrs": pc.mean_ci([r["theory"]["mean_zealot_nbrs"] for r in rz]),
              "num_zealots": pc.mean_ci([r["num_zealots"] for r in rz])}
        out["theory"][str(z)] = th
    # paired ζ comparison under DeGroot vs ζ = 0
    base = {r["seed"]: r for r in rows if r["zeta"] == 0.0}
    for z in ZETAS[1:]:
        for name in RULES:
            rz = [r for r in rows if r["zeta"] == z]
            out["paired"][f"{z}_vs_0|{name}"] = {
                "persuasion_diff": pc.paired([r["rules"][name]["persuasion"] for r in rz],
                                             [base[r["seed"]]["rules"][name]["persuasion"] for r in rz]),
                "win_mcnemar": pc.mcnemar([r["rules"][name]["victory"] for r in rz],
                                          [base[r["seed"]]["rules"][name]["victory"] for r in rz])}
    rob = [r["robustness"] for r in rows if "robustness" in r]
    out["robustness"] = {
        "n_graphs": len(rob), "r_upper": pc.mean_ci([x["r_upper"] for x in rob]),
        "r_upper_min": min(x["r_upper"] for x in rob), "r_upper_max": max(x["r_upper"] for x in rob),
        "min_degree": pc.mean_ci([x["min_degree"] for x in rob]),
        "witness_counts": {w: sum(x["witness"].split(" {")[0].split(" (")[0] == w for x in rob)
                           for w in sorted({x["witness"].split(" {")[0].split(" (")[0] for x in rob})},
        "frac_r_upper_ge_3": sum(x["r_upper"] >= 3 for x in rob) / len(rob),
        "frac_r_upper_ge_5": sum(x["r_upper"] >= 5 for x in rob) / len(rob),
        "s_upper_r2_min": min(x["s_upper_r2"] for x in rob), "s_upper_r3_min": min(x["s_upper_r3"] for x in rob)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--first-seed", type=int, default=8000)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="reports/v2/e2_zealots.json")
    args = ap.parse_args()
    device = pc.pick_device(args.device)
    tasks = [(s, z, args.n, device, z == 0.0) for s in range(args.first_seed, args.first_seed + args.seeds) for z in ZETAS]
    print(f"E2 device={device} tasks={len(tasks)}", flush=True)
    start = time.perf_counter()
    rows = pc.run_tasks(run_task, tasks, args.workers, "E2")
    report = {"experiment": "E2 zealots x resilient consensus", "num_seeds": args.seeds, "num_voters": args.n,
              "zetas": ZETAS, "rules": RULES, **summarise(rows),
              "provenance": pc.provenance(args, device, time.perf_counter() - start, args.workers), "rows": rows}
    pc.save(args.out, report)


if __name__ == "__main__":
    main()
