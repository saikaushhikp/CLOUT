"""E1 — Topology and consensus speed (spec §2.2, §4, §11; lectures: MAS consensus, Linear-Algebra review,
Distributed Learning "spectral gap 1 − λ2(W) ⇒ faster mixing", Infection Model "convergence ∝ λ2(L)").

Factors: mean degree k̄ ∈ {8, 16, 30, 48} × inter-region ER probability range p_g ∈ {[.01,.05], [.01,.2]
(default), [.1,.4]}; the same seeds in every cell (paired).

Per seed and cell:
1. Spectrum on the giant component: λ2(L); eigenvalues μ2, μ_min of the random-walk matrix P = D⁻¹A (via the
   similar symmetric D^-½AD^-½); the |·| DeGroot gap; λ_max(A) and λ_max(A_rr).
2. Consensus speed. A zealot-free copy of the world (same graph, same α; ζ = 0) runs **social-only** dynamics
   x ← (1 − a)x + a·D⁻¹A x with a_i = ω·α_i for 400 rounds (float64). Theory:
     M = I − G L with G = diag(a/d) is similar to the symmetric I − G^½ L G^½, so the asymptotic per-round
     contraction of the disagreement x − 1c (c = Σ w_i x_i / Σ w_i, w ∝ d/a is the conserved weighting) is
       ρ_exact = 1 − ν2(G^½ L G^½)                         (exact operator),
       ρ_rw    = 1 − ā (1 − μ2(P))                          (homogeneous-step approximation, lecture form),
       ρ_L     = 1 − ā λ2(L)/k̄_real                         (Laplacian/λ2 approximation, regular-graph form).
   Observed: slope of log‖x(t) − 1c‖ over t ∈ [200, 400] → ρ_obs, and the game-horizon factor (e(5)/e(0))^{1/5}.
   Within-region variance W(t) (deviation from region means): round-1 i.i.d. bulk prediction
       W(1)/W(0) ≈ mean_i[(1 − a_i)² + a_i²/d_i].
3. Outcomes (default ζ = 0.05 world): ORACLE manifesto + oracle marginal planner (balance.py); informed
   manifesto with the three fixed paid patterns (best pattern frequency); BayesPlanner and Top3Player through
   the player-facing GameEngine (observations only).

Usage: python experiments/e1_topology.py --seeds 200 --n 10000 --device auto --workers 6 --out reports/v2/e1_topology.json
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time

import p7_common as pc

KBARS = (8, 16, 30, 48)
PG = {"low[.01,.05]": (0.01, 0.05), "default[.01,.2]": (0.01, 0.2), "high[.1,.4]": (0.1, 0.4)}
ROUNDS_LONG = 400
FIT = (3000, 4000)       # heterogeneous-step tail fit (slow: governed by the least-susceptible voters)
ROUNDS_HET = 4000
FIT_HOM = (150, 400)
SAVE_T = list(range(0, 11)) + list(range(20, ROUNDS_LONG + 1, 20))


def spectrum(graph, a_full):
    """Spectral quantities on the giant component (CPU, SciPy)."""
    import numpy as np
    import torch
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh

    n = graph.num_nodes
    rows = graph.rows().cpu().numpy()
    cols = graph.indices.cpu().numpy()
    A = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(A, directed=False)
    giant_label = np.bincount(labels).argmax()
    giant = labels == giant_label
    A = A[giant][:, giant]
    d = np.asarray(A.sum(1)).ravel()
    m = A.shape[0]
    L = diags(d) - A
    # λ2(L): largest of (c I − L)
    c = 2.0 * d.max() + 1.0
    top = eigsh(c * diags(np.ones(m)) - L, k=2, which="LA", return_eigenvectors=False, tol=1e-9)
    lam2 = float(c - min(top))
    # random-walk spectrum
    s = diags(1 / np.sqrt(d)) @ A @ diags(1 / np.sqrt(d))
    hi = np.sort(eigsh(s, k=4, which="LA", return_eigenvectors=False, tol=1e-9))[::-1]
    lo = eigsh(s, k=1, which="SA", return_eigenvectors=False, tol=1e-9)
    mu2, mumin = float(hi[1]), float(lo[0])
    mu234 = [float(x) for x in hi[1:4]]
    # exact operator: K = G^½ L G^½, G = a/d
    a = a_full[giant]
    g = np.sqrt(a / d)
    K = diags(g) @ L @ diags(g)
    kmax = float(eigsh(K, k=1, which="LA", return_eigenvectors=False, tol=1e-9)[0])
    cc = kmax * 1.01 + 1e-6
    top = eigsh(cc * diags(np.ones(m)) - K, k=2, which="LA", return_eigenvectors=False, tol=1e-10)
    nu2 = float(cc - min(top))
    return {"giant": torch.from_numpy(giant), "giant_size": int(m), "components": int(labels.max() + 1),
            "lambda2_L": lam2, "mu2_P": mu2, "mu234_P": mu234, "mumin_P": mumin, "degroot_abs_gap": 1 - max(abs(mu2), abs(mumin)),
            "nu2_operator": nu2, "mean_degree_giant": float(d.mean()), "min_degree": int(d.min()),
            "a_mean": float(a.mean()), "a_min": float(a.min()), "a_p01": float(np.quantile(a, 0.01)),
            "bulk_round1_ratio": float(np.mean((1 - a) ** 2 + a ** 2 / d))}


def consensus_run(e0, omega, giant, device, homogeneous=False, long_tail=True):
    """Social-only dynamics on the zealot-free world in float64: disagreement, within- and between-region variance.

    ``homogeneous`` replaces a_i = ω·α_i by its mean ā (the lecture's uniform-step setting)."""
    import numpy as np
    import torch
    from dumbocracy.dynamics.social import aggregate_degroot

    x = e0.preferences.to(device=device, dtype=torch.float64)
    a = (omega * e0.susceptibility.social).to(device=device, dtype=torch.float64)
    gmask = giant.to(device)
    if homogeneous:
        a = torch.full_like(a, float(a[gmask].mean()))
    deg = e0.graph.degree.to(device=device, dtype=torch.float64)
    w = torch.where(gmask, deg / a, torch.zeros_like(a))
    c = (w.unsqueeze(1) * x).sum(0) / w.sum()
    region = e0.region_of.to(device)
    onehot = torch.nn.functional.one_hot(region, 4).to(torch.float64)
    sizes = onehot.sum(0)

    def parts(xx):
        means = onehot.T @ xx / sizes.unsqueeze(1)
        overall = xx.mean(0)
        within = float(((xx - means[region]) ** 2).sum() / xx.shape[0])
        between = float((sizes.unsqueeze(1) * (means - overall) ** 2).sum() / xx.shape[0])
        return within, between

    errs, W, B = [], [], []
    rounds = ROUNDS_HET if (long_tail and not homogeneous) else ROUNDS_LONG
    for t in range(rounds + 1):
        errs.append(float(torch.linalg.vector_norm(x[gmask] - c)))
        if t <= 10:
            wv, bv = parts(x)
            W.append(wv)
            B.append(bv)
        if t == rounds:
            break
        agg = aggregate_degroot(x, e0.graph)
        x = (1 - a).unsqueeze(1) * x + a.unsqueeze(1) * agg
    errs = np.array(errs)
    lo, hi = (FIT_HOM if homogeneous else FIT)
    rho = None
    if rounds >= hi:
        rho = float(math.exp(np.polyfit(np.arange(lo, hi + 1), np.log(errs[lo:hi + 1]), 1)[0]))
    return {"rho_obs": rho, "rho_horizon5": float((errs[5] / errs[0]) ** 0.2),
            "err_rel": [float(errs[t] / errs[0]) for t in SAVE_T], "within_rel": [w_ / W[0] for w_ in W],
            "between_rel": [b_ / B[0] for b_ in B], "within_round1_ratio": W[1] / W[0],
            "rho_between5": float((B[5] / B[0]) ** 0.1),
            "rho_obs_400": float(math.exp(np.polyfit(np.arange(200, 401), np.log(errs[200:401]), 1)[0])),
            "err_rel_long": [float(errs[t] / errs[0]) for t in range(0, len(errs), 250)]}


def run_task(task):
    import torch
    from balance import POLICIES, policy_planner
    from dumbocracy.analysis.spectral import largest_eigenpair, region_lambda_max
    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.simulate import simulate_world
    from dumbocracy.strategies.oracle import informed_manifesto, oracle_manifesto
    from dumbocracy.strategies.players import BayesPlanner, Top3Player, play_game
    from strategies import spend_breakdown

    seed, kbar, pg_name, n, device, long_tail = task
    t0 = time.perf_counter()
    dev = torch.device(device)
    ecfg = ElectorateConfig(num_voters=n, mean_degree=float(kbar), inter_prob_range=PG[pg_name])
    cfg = GameConfig(seed=seed, device=device, electorate=ecfg)
    e = generate_electorate(cfg, device=dev)
    cfg0 = dataclasses.replace(cfg, electorate=dataclasses.replace(ecfg, zealot_fraction=0.0))
    e0 = generate_electorate(cfg0, device=dev)  # same graph and α; no zealots
    omega = cfg.dynamics.social_step
    a0 = (omega * e0.susceptibility.social).double().cpu().numpy()
    spec = spectrum(e0.graph, a0)
    giant = spec.pop("giant")
    cons = consensus_run(e0, omega, giant, dev, long_tail=long_tail)
    hom = consensus_run(e0, omega, giant, dev, homogeneous=True)
    kreal = 2 * e.graph.num_edges / n
    am = spec["a_mean"]
    pred = {"rho_exact": 1 - spec["nu2_operator"], "rho_rw": 1 - am * (1 - spec["mu2_P"]),
            "rho_L": 1 - am * spec["lambda2_L"] / spec["mean_degree_giant"],
            "rho_between_pred_mu2": 1 - am * (1 - spec["mu2_P"]),
            "rho_between_pred_mu234": 1 - am * (1 - sum(spec["mu234_P"]) / 3)}
    lam_rr = region_lambda_max(e.graph, e.region_of)
    lam_full = largest_eigenpair(e.graph)[0]
    census = e.census()
    intra = sum(r["intra_edges"] for r in census["regions"]) / max(1, e.graph.num_edges)
    # outcomes on the default-ζ world
    runs = {}
    bo = oracle_manifesto(e, cfg)
    for name, b, pol in (("oracle/planner", bo, "planner"), ("oracle/none", bo, "none")):
        out = simulate_world(e, cfg, b, POLICIES[pol])
        runs[name] = {"expected": out.result.expected_share, "victory": out.result.victory}
    bi = informed_manifesto(e, cfg)
    for pol in ("pay_workers", "pay_media", "pay_split"):
        out = simulate_world(e, cfg, bi, POLICIES[pol])
        runs[f"informed/{pol}"] = {"expected": out.result.expected_share, "victory": out.result.victory}
    ccpu = dataclasses.replace(cfg, device="cpu")
    state0 = GameEngine.new_game(cfg, electorate=e, game_id=f"e1s{seed}")
    for name, player in (("bayes", BayesPlanner(seed=seed, config=ccpu)), ("top3", Top3Player(config=ccpu))):
        final, _ = play_game(state0, player)
        obs = observation_for(final, "p0")
        runs[name] = {"expected": obs["result"]["expected_share"], "victory": obs["result"]["victory"],
                      "spend": spend_breakdown(obs, cfg)}
    best = max(("pay_workers", "pay_media", "pay_split"), key=lambda p: runs[f"informed/{p}"]["expected"])
    return {"seed": seed, "kbar": kbar, "pg": pg_name, "mean_degree_real": kreal, "intra_edge_fraction": intra,
            "lambda_max_rr": lam_rr, "lambda_max_full": lam_full, **spec, **pred, **cons,
            "hom_rho_obs": hom["rho_obs"], "hom_err_rel": hom["err_rel"], "runs": runs,
            "best_paid_pattern": best, "seconds": time.perf_counter() - t0}


def summarise(rows):
    import numpy as np
    cells = {}
    for kbar in KBARS:
        for pg in PG:
            rs = [r for r in rows if r["kbar"] == kbar and r["pg"] == pg]
            if not rs:
                continue
            s = {"n": len(rs)}
            for key in ("lambda2_L", "mu2_P", "degroot_abs_gap", "nu2_operator", "mean_degree_real", "intra_edge_fraction",
                        "rho_obs", "rho_exact", "rho_rw", "rho_L", "rho_horizon5", "within_round1_ratio",
                        "bulk_round1_ratio", "a_mean", "a_min", "a_p01", "lambda_max_full", "components", "giant_size",
                        "hom_rho_obs", "rho_obs_400", "rho_between5", "rho_between_pred_mu2", "rho_between_pred_mu234"):
                s[key] = pc.mean_ci([r[key] for r in rs])
            s["lambda2_L_over_k"] = pc.mean_ci([r["lambda2_L"] / r["mean_degree_giant"] for r in rs])
            s["lambda_max_rr"] = pc.mean_ci([x for r in rs for x in r["lambda_max_rr"]])
            # relative error of predicted decay rate (1 − ρ) vs observed
            for p in ("rho_rw", "rho_L"):  # lecture predictions vs the homogeneous-step run
                s[f"ratio_{p}_hom"] = pc.mean_ci([(1 - r[p]) / (1 - r["hom_rho_obs"]) for r in rs])
            s["ratio_rho_exact_obs"] = pc.mean_ci([(1 - r["rho_exact"]) / (1 - r["rho_obs"]) for r in rs if r["rho_obs"] is not None])
            for p in ("rho_between_pred_mu2", "rho_between_pred_mu234"):
                s[f"ratio_{p}"] = pc.mean_ci([(1 - r[p]) / (1 - r["rho_between5"]) for r in rs])
            s["hom_err_rel_curve"] = np.mean([r["hom_err_rel"] for r in rs], 0).tolist()
            s["between_rel_curve"] = np.mean([r["between_rel"] for r in rs], 0).tolist()
            s["err_rel_curve"] = np.mean([r["err_rel"] for r in rs], 0).tolist()
            s["within_rel_curve"] = np.mean([r["within_rel"] for r in rs], 0).tolist()
            for name in ("oracle/planner", "oracle/none", "bayes", "top3", "informed/pay_workers", "informed/pay_media",
                         "informed/pay_split"):
                s[f"win_{name}"] = pc.rate([r["runs"][name]["victory"] for r in rs])
                s[f"expected_{name}"] = pc.mean_ci([r["runs"][name]["expected"] for r in rs])
            s["best_paid_pattern"] = {p: sum(r["best_paid_pattern"] == p for r in rs) for p in ("pay_workers", "pay_media", "pay_split")}
            s["bayes_spend"] = {k: pc.mean_ci([r["runs"]["bayes"]["spend"][k] for r in rs])["mean"] for k in ("workers", "media", "queries")}
            cells[f"k{kbar}|{pg}"] = s
    # theory vs observation across all rows
    allr = rows
    hom = [1 - r["hom_rho_obs"] for r in allr]
    theory = {f"{p}_vs_homogeneous": {"spearman_decay": pc.spearman([1 - r[p] for r in allr], hom),
                                      "median_ratio_pred_over_obs": float(np.median([(1 - r[p]) / (1 - r["hom_rho_obs"]) for r in allr]))}
              for p in ("rho_rw", "rho_L")}
    tail = [r for r in allr if r["rho_obs"] is not None]
    het = [1 - r["rho_obs"] for r in tail]
    theory["heterogeneous_tail"] = {"n": len(tail),
        "spearman_obs_vs_lambda2": pc.spearman([r["lambda2_L"] for r in tail], het),
        "spearman_obs_vs_exact": pc.spearman([1 - r["rho_exact"] for r in tail], het),
        "median_ratio_exact_over_obs": float(np.median([(1 - r["rho_exact"]) / (1 - r["rho_obs"]) for r in tail])),
        "median_nu2_over_a_min": float(np.median([r["nu2_operator"] / r["a_min"] for r in allr])),
        "spearman_nu2_vs_a_min": pc.spearman([r["nu2_operator"] for r in allr], [r["a_min"] for r in allr]),
        "spearman_nu2_vs_lambda2": pc.spearman([r["nu2_operator"] for r in allr], [r["lambda2_L"] for r in allr])}
    btw = [1 - r["rho_between5"] for r in allr]
    theory["between_region_horizon"] = {
        p: {"spearman": pc.spearman([1 - r[p] for r in allr], btw),
            "median_ratio_pred_over_obs": float(np.median([(1 - r[p]) / (1 - r["rho_between5"]) for r in allr]))}
        for p in ("rho_between_pred_mu2", "rho_between_pred_mu234")}
    theory["within_round1_pred_vs_obs"] = {
        "mean_pred": pc.mean_ci([r["bulk_round1_ratio"] for r in allr]),
        "mean_obs": pc.mean_ci([r["within_round1_ratio"] for r in allr])}
    # paired outcome tests vs the default cell (k30|default) on the same seeds
    base = {r["seed"]: r for r in rows if r["kbar"] == 30 and r["pg"] == "default[.01,.2]"}
    tests = {}
    for key in cells:
        kbar, pg = key.split("|")
        rs = [r for r in rows if r["kbar"] == int(kbar[1:]) and r["pg"] == pg and r["seed"] in base]
        tests[key] = {name: {"expected_diff_vs_default": pc.paired([r["runs"][name]["expected"] for r in rs],
                                                                   [base[r["seed"]]["runs"][name]["expected"] for r in rs]),
                             "mcnemar_vs_default": pc.mcnemar([r["runs"][name]["victory"] for r in rs],
                                                              [base[r["seed"]]["runs"][name]["victory"] for r in rs])}
                      for name in ("oracle/planner", "bayes", "top3")}
    # does consensus speed predict how much social drift helps a fixed (oracle) manifesto?
    drift = [r["runs"]["oracle/none"]["expected"] for r in rows]
    theory["between_rate_vs_oracle_none_expected"] = pc.spearman([1 - r["rho_between5"] for r in rows], drift)
    return {"cells": cells, "theory": theory, "paired_vs_default": tests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--first-seed", type=int, default=7000)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tail-seeds", type=int, default=60, help="seeds that also run the 4000-round heterogeneous tail")
    ap.add_argument("--kbars", default=",".join(map(str, KBARS)))
    ap.add_argument("--out", default="reports/v2/e1_topology.json")
    args = ap.parse_args()
    device = pc.pick_device(args.device)
    kbars = [int(k) for k in args.kbars.split(",")]
    tasks = [(s, k, pg, args.n, device, s < args.first_seed + args.tail_seeds)
             for s in range(args.first_seed, args.first_seed + args.seeds) for k in kbars for pg in PG]
    print(f"E1 device={device} tasks={len(tasks)}", flush=True)
    start = time.perf_counter()
    rows = pc.run_tasks(run_task, tasks, args.workers, "E1")
    report = {"experiment": "E1 topology & consensus speed", "num_seeds": args.seeds, "num_voters": args.n,
              "kbars": kbars, "pg_ranges": PG, "fit_window": FIT, "save_t": SAVE_T, **summarise(rows),
              "provenance": pc.provenance(args, device, time.perf_counter() - start, args.workers), "rows": rows}
    pc.save(args.out, report)


if __name__ == "__main__":
    main()
