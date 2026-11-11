"""X1 — Network structure and consensus speed (plan ``docs/v2/plans/implementation_plan.md`` §4 X1; port of E1).

Command (reproduces reports/v3/x1.json, reports/v3/raw/x1.jsonl and reports/v3/logs/x1.log):
  OMP_NUM_THREADS=4 python experiments/x1_topology.py --seeds 50 --first-seed 11000 --device cuda:2 \
      --out reports/v3/x1.json 2>&1 | tee reports/v3/logs/x1.log

Grid: mean degree k̄ ∈ {10, 20, 30} × inter-region probability range p_g ∈ {[.01, .05], [.01, .2] (default),
[.1, .4]}; 50 paired seeds per cell (block 11000: the same seed gives the same regions, preferences and
susceptibilities in every cell; only the graph changes). N = 10⁴, T = 5, ₫500.

Per world:
1. Spectra on the zealot-free copy's giant component (``e1_topology.spectrum``, SciPy on the CPU: at N = 10⁴ it is
   5× faster than the GPU ``lobpcg`` path, measured in Phase 0 and again here, so this is the §1.1 "measured
   slower" exception; ``x_common.spectra_gpu`` is the checked alternative on 3 worlds): λ₂(L); μ₂, μ_min of
   D⁻¹A; ν₂ of the exact operator G^½ L G^½ (G = diag(a/d), a_i = ω α_i); λ_max(A), λ_max(A_rr) (GPU power
   iteration); cross-region conductance φ_r = cut(r) / min(vol r, vol V∖r); the inter-region edge share.
2. Long social-only runs (zealot-free copy, ζ = 0, same graph and α; float64; batched over the 50 worlds of a
   cell on the GPU): x ← (1 − a)x + a·D⁻¹A x, heterogeneous a_i = ω α_i (2,000 rounds) and homogeneous a_i = ā
   (400 rounds). Disagreement e(t) = ‖x − 1c‖ on the giant component (c = the conserved d/a-weighted mean);
   observed per-round contraction ρ_obs from the slope of log e(t) (hom: t ∈ [200, 400]; het: [200, 400] and
   [1500, 2000]); the game-horizon factor (e(5)/e(0))^{1/5}; and "rounds until the regional means move halfway to
   the global mean" (between-region TV dispersion ≤ ½ its start) — reported SEPARATELY from the 5-round game.
   Predictions: ρ_exact = 1 − ν₂, ρ_rw = 1 − ā(1 − μ₂(D⁻¹A)), ρ_L = 1 − ā λ₂(L)/k̄.
3. Game outcomes on the default-ζ world (T = 5; ``x_batch.run_batch`` on the GPU, the X policies of ``balance``):
   ORACLE manifesto with none / planner; the nationwide top-3 manifesto and the uniform one with the planner;
   the informed manifesto with pay_workers / pay_media / pay_split.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x0_balance import prepare_chunk  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_policies import GATE_POLICIES  # noqa: E402

from dumbocracy.analysis.spectral import largest_eigenpair, region_lambda_max  # noqa: E402
from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.worlds import _union_graph  # noqa: E402

KBARS = (10, 20, 30)
PG = {"low[.01,.05]": (0.01, 0.05), "default[.01,.2]": (0.01, 0.2), "high[.1,.4]": (0.1, 0.4)}
RUNS = [("oracle", "none"), ("oracle", "planner"), ("top3", "planner"), ("uniform", "planner"),
        ("informed", "pay_workers"), ("informed", "pay_media"), ("informed", "pay_split")]
PAID = ("pay_workers", "pay_media", "pay_split")
HET_T, HOM_T = 2000, 400
SAVE_T = list(range(0, 11)) + [20, 50, 100, 200, 300, 400, 1000, 1500, 2000]
RAW_PATH = xc.RAW / "x1.jsonl"


def cell_config(base: GameConfig, kbar: int, pg: str, zeta: float | None = None) -> GameConfig:
    e = dataclasses.replace(base.electorate, mean_degree=float(kbar), inter_prob_range=PG[pg])
    if zeta is not None:
        e = dataclasses.replace(e, zealot_fraction=zeta)
    return dataclasses.replace(base, electorate=e)


def conductance(graph, region_of) -> list[float]:
    rows, cols = graph.rows(), graph.indices
    deg = graph.degree.to(torch.float64)
    vol_total = float(deg.sum())
    out = []
    for r in range(4):
        m = region_of == r
        cut = float((m[rows] & ~m[cols]).sum())
        vol = float(deg[m].sum())
        out.append(cut / max(1e-12, min(vol, vol_total - vol)))
    return out


@torch.no_grad()
def long_runs(e0s: list, giants: list[torch.Tensor], omega: float, rounds: int, homogeneous: bool) -> dict:
    """Social-only DeGroot runs on the zealot-free copies, float64, all worlds of a cell at once (union CSR)."""
    B, n = len(e0s), e0s[0].num_voters
    dev = e0s[0].preferences.device
    g = e0s[0].graph if B == 1 else _union_graph([e.graph for e in e0s], n)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        adj = torch.sparse_csr_tensor(g.indptr, g.indices, torch.ones(g.indices.numel(), dtype=torch.float64, device=dev),
                                      (B * n, B * n))
    deg = g.degree.to(torch.float64)
    x = torch.cat([e.preferences for e in e0s]).to(torch.float64)
    a = torch.cat([omega * e.susceptibility.social for e in e0s]).to(torch.float64)
    gm = torch.cat(giants).to(dev)
    if homogeneous:
        abar = torch.stack([a.view(B, n)[b][gm.view(B, n)[b]].mean() for b in range(B)])
        a = abar.repeat_interleave(n)
    w = torch.where(gm, deg / a, torch.zeros_like(a)).view(B, n, 1)
    c = (w * x.view(B, n, 5)).sum(1) / w.sum(1)                                   # [B, 5] conserved mean
    region = torch.cat([e.region_of for e in e0s]).view(B, n)
    onehot = torch.nn.functional.one_hot(region, 4).to(torch.float64)            # [B, n, 4]
    cnt = onehot.sum(1)
    errs, between = [], []
    for t in range(rounds + 1):
        xv = x.view(B, n, 5)
        errs.append(torch.linalg.vector_norm((xv - c[:, None]) * gm.view(B, n, 1), dim=(1, 2)))
        means = torch.einsum("bnr,bnk->brk", onehot, xv) / cnt[..., None]
        glob = xv.mean(1)
        between.append((0.5 * (means - glob[:, None]).abs().sum(-1)).mean(1))
        if t == rounds:
            break
        agg = (adj @ x) / deg.clamp_min(1)[:, None]
        agg = torch.where((deg > 0)[:, None], agg, x)
        x = (1 - a)[:, None] * x + a[:, None] * agg
    E = torch.stack(errs, 1).cpu()                 # [B, T+1]
    D = torch.stack(between, 1).cpu()
    out = []
    import numpy as np

    def slope(e, lo, hi):
        y = np.log(np.maximum(e[lo:hi + 1], 1e-300))
        return float(math.exp(np.polyfit(np.arange(lo, hi + 1), y, 1)[0]))
    for b in range(B):
        e = E[b].numpy()
        d = D[b].numpy()
        half = next((t for t in range(len(d)) if d[t] <= 0.5 * d[0]), None)
        rec = {"rho_horizon5": float((e[5] / e[0]) ** 0.2), "rho_obs_200_400": slope(e, 200, 400),
               "err_rel": {str(t): float(e[t] / e[0]) for t in SAVE_T if t < len(e)},
               "between_rel": {str(t): float(d[t] / d[0]) for t in SAVE_T if t < len(d)},
               "t_half_between": half, "rho_between_200_400": slope(d, 200, 400)}
        if rounds >= 2000:
            rec["rho_obs_1500_2000"] = slope(e, 1500, 2000)
        out.append(rec)
    return out


def play(prepared, cfg: GameConfig, man: str, pol: str) -> list[dict]:
    outs = run_batch([p.electorate for p in prepared], cfg, torch.stack([p.manifestos[cfg.rounds][man] for p in prepared]),
                     GATE_POLICIES[pol], seeds=[p.seed for p in prepared])
    return [{"expected": o.result.expected_share, "victory": bool(o.result.victory),
             "workers": sum(a.total_hires for a in o.allocations),
             "airings": sum(sum(a.media_regions) for a in o.allocations), "converts": int(sum(o.converted))}
            for o in outs]


def summarise(rows: list[dict]) -> dict:
    import numpy as np
    from scipy.stats import spearmanr
    out: dict = {"cells": {}, "theory_vs_obs": {}, "nationwide": {}, "tests": {}}
    for kbar in KBARS:
        for pg in PG:
            rs = [r for r in rows if r["kbar"] == kbar and r["pg"] == pg]
            c = {"n": len(rs)}
            for k in ("mean_degree_real", "components", "giant_size", "lambda2_L", "mu2_P", "mumin_P", "degroot_abs_gap",
                      "nu2_operator", "lambda_max_full", "conductance_mean", "conductance_min", "inter_edge_share",
                      "a_mean", "rho_exact", "rho_rw", "rho_L", "t_half_pred_rw"):
                c[k] = xc.mean_ci([r[k] for r in rs])
            for mode in ("het", "hom"):
                for k in ("rho_obs_200_400", "rho_horizon5", "rho_between_200_400") + (("rho_obs_1500_2000",) if mode == "het" else ()):
                    c[f"{mode}_{k}"] = xc.mean_ci([r[mode][k] for r in rs])
                th = [r[mode]["t_half_between"] for r in rs]
                c[f"{mode}_t_half_between"] = {"reached": sum(t is not None for t in th), "n": len(th),
                                               "median": float(np.median([t for t in th if t is not None])) if any(
                                                   t is not None for t in th) else None}
                c[f"{mode}_err_rel_mean"] = {t: float(np.mean([r[mode]["err_rel"][t] for r in rs]))
                                             for t in rs[0][mode]["err_rel"]}
                c[f"{mode}_between_rel_mean"] = {t: float(np.mean([r[mode]["between_rel"][t] for r in rs]))
                                                 for t in rs[0][mode]["between_rel"]}
            for man, pol in RUNS:
                key = f"{man}/{pol}"
                c[key] = {"win": xc.rate([r["runs"][key]["victory"] for r in rs]),
                          "expected": xc.mean_ci([r["runs"][key]["expected"] for r in rs]),
                          "workers": xc.mean_ci([r["runs"][key]["workers"] for r in rs]),
                          "converts": xc.mean_ci([r["runs"][key]["converts"] for r in rs])}
            c["best_paid_pattern"] = {p: sum(r["best_paid_pattern"] == p for r in rs) for p in PAID}
            out["cells"][f"k{kbar}|{pg}"] = c
    # theory vs observation across all worlds: which prediction matches the observed contraction?
    for mode, obs_key in (("hom", "rho_obs_200_400"), ("het", "rho_obs_200_400"), ("het", "rho_obs_1500_2000"),
                          ("het", "rho_horizon5"), ("hom", "rho_between_200_400")):
        obs = np.array([1 - r[mode][obs_key] for r in rows])            # observed per-round decay rate 1 − ρ
        res = {}
        for pred in ("rho_exact", "rho_rw", "rho_L"):
            p = np.array([1 - r[pred] for r in rows])
            ok = (obs > 0) & (p > 0)
            lr = np.log(obs[ok] / p[ok])
            res[pred] = {"spearman_rho": float(spearmanr(p, obs).statistic), "log_ratio_obs_over_pred": xc.mean_ci(lr.tolist()),
                         "rmse_log": float(np.sqrt(np.mean(lr ** 2))), "n": int(ok.sum())}
        out["theory_vs_obs"][f"{mode}:{obs_key}"] = res
    # nationwide manifesto vs p_g (paired by seed across cells): high − low, per k̄ and pooled
    fam = []
    for key in ("top3/planner", "oracle/planner", "uniform/planner"):
        for kbar in list(KBARS) + ["all"]:
            sel = (lambda r: True) if kbar == "all" else (lambda r, k=kbar: r["kbar"] == k)
            hi = {(r["seed"], r["kbar"]): r for r in rows if r["pg"] == "high[.1,.4]" and sel(r)}
            lo = {(r["seed"], r["kbar"]): r for r in rows if r["pg"] == "low[.01,.05]" and sel(r)}
            ks = sorted(set(hi) & set(lo))
            t = {"run": key, "kbar": kbar,
                 "win_high": xc.rate([hi[k]["runs"][key]["victory"] for k in ks]),
                 "win_low": xc.rate([lo[k]["runs"][key]["victory"] for k in ks]),
                 "mcnemar": xc.mcnemar([hi[k]["runs"][key]["victory"] for k in ks], [lo[k]["runs"][key]["victory"] for k in ks]),
                 "expected_high_minus_low": xc.paired([hi[k]["runs"][key]["expected"] for k in ks],
                                                      [lo[k]["runs"][key]["expected"] for k in ks])}
            fam.append(t)
    for t, p in zip(fam, xc.holm([t["mcnemar"]["p"] for t in fam])):
        t["mcnemar_holm_p"] = p
    for t, p in zip(fam, xc.holm([t["expected_high_minus_low"]["wilcoxon_p"] for t in fam])):
        t["wilcoxon_holm_p"] = p
    out["nationwide"] = fam
    out["test_families"] = {"nationwide": "high − low p_g, paired by seed, for top3/planner, oracle/planner and "
                                          "uniform/planner × (k̄ = 10, 20, 30, pooled): McNemar on wins and Wilcoxon on "
                                          "expected share, each Holm over its 12 tests"}
    return out


def parity(seeds: list[int], device: torch.device, base: GameConfig) -> dict:
    def fn(seed, dev):
        cfg = dataclasses.replace(cell_config(base, 10, "low[.01,.05]"), device=str(dev))
        p = prepare_chunk([seed], cfg, dev, (cfg.rounds,))
        r = play(p, cfg, "oracle", "planner")[0]
        return {"expected": r["expected"], "converts_share_of_n": r["converts"] / cfg.electorate.num_voters}
    out = xc.parity_check(fn, seeds, device=device, abs_tol=0.01)
    out["cell"] = "k̄ = 10, p_g low; ORACLE manifesto + planner"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X1"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--out", default=str(xc.REPORTS / "x1.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    from e1_topology import spectrum
    device, dev_record = xc.select_device(args.device, need_bytes=6 * 2**30)
    if args.device.startswith("cuda") and str(device) != str(torch.device(args.device)):
        raise SystemExit(f"refusing to fall back from {args.device} to {device} (shared host: only the requested GPU)")
    if device.type == "cuda":
        torch.cuda.set_device(device)   # no stray context on the default GPU
    free0 = xc.free_bytes(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text("")
    rec: dict = {"timing": {}}
    rows: list[dict] = []
    start = time.perf_counter()
    omega = base.dynamics.social_step
    with xc.GpuMonitor(device) as monitor:
        for kbar in KBARS:
            for pg in PG:
                t0 = time.perf_counter()
                cfg = cell_config(base, kbar, pg)
                prepared = prepare_chunk(seeds, cfg, device, (cfg.rounds,))
                t_prep = time.perf_counter() - t0
                e0s = [generate_electorate(dataclasses.replace(cell_config(base, kbar, pg, 0.0), seed=s), device=device)
                       for s in seeds]
                t1 = time.perf_counter()
                specs, giants = [], []
                for e0 in e0s:
                    a0 = (omega * e0.susceptibility.social).double().cpu().numpy()
                    sp = spectrum(e0.graph, a0)
                    giants.append(sp.pop("giant"))
                    specs.append(sp)
                t_spec = time.perf_counter() - t1
                t2 = time.perf_counter()
                het = long_runs(e0s, giants, omega, HET_T, False)
                hom = long_runs(e0s, giants, omega, HOM_T, True)
                t_long = time.perf_counter() - t2
                t3 = time.perf_counter()
                runs = {f"{m}/{p}": play(prepared, cfg, m, p) for m, p in RUNS}
                t_games = time.perf_counter() - t3
                for i, (p, e0, sp) in enumerate(zip(prepared, e0s, specs)):
                    e = p.electorate
                    cond = conductance(e.graph, e.region_of)
                    census = e.census()
                    inter = sum(x["edges"] for x in census["inter_edges"]) / max(1, census["total_edges"])
                    am = sp["a_mean"]
                    rho_rw = 1 - am * (1 - sp["mu2_P"])
                    row = {"seed": p.seed, "kbar": kbar, "pg": pg, "mean_degree_real": 2 * e.graph.num_edges / args.n,
                           **sp, "rho_exact": 1 - sp["nu2_operator"], "rho_rw": rho_rw,
                           "rho_L": 1 - am * sp["lambda2_L"] / sp["mean_degree_giant"],
                           "t_half_pred_rw": math.log(2) / -math.log(rho_rw) if 0 < rho_rw < 1 else None,
                           "lambda_max_full": largest_eigenpair(e.graph)[0],
                           "lambda_max_rr": region_lambda_max(e.graph, e.region_of),
                           "conductance": cond, "conductance_mean": sum(cond) / 4, "conductance_min": min(cond),
                           "inter_edge_share": inter, "het": het[i], "hom": hom[i],
                           "runs": {k: v[i] for k, v in runs.items()}}
                    row["best_paid_pattern"] = max(PAID, key=lambda q: row["runs"][f"informed/{q}"]["expected"])
                    rows.append(row)
                    xc.write_row(RAW_PATH, {"exp": "X1", **row})
                rec["timing"][f"k{kbar}|{pg}"] = {"prepare": t_prep, "spectra_scipy": t_spec, "long_runs_gpu": t_long,
                                                  "games_gpu": t_games, "total": time.perf_counter() - t0}
                cr = [r for r in rows if r["kbar"] == kbar and r["pg"] == pg]
                print(f"[x1] k̄={kbar} {pg}: {time.perf_counter() - t0:.0f}s (spectra {t_spec:.0f}s, long {t_long:.0f}s); "
                      f"μ2 {sum(r['mu2_P'] for r in cr) / len(cr):.3f}; hom ρ_obs "
                      f"{sum(r['hom']['rho_obs_200_400'] for r in cr) / len(cr):.4f} vs ρ_rw "
                      f"{sum(r['rho_rw'] for r in cr) / len(cr):.4f} vs ρ_L {sum(r['rho_L'] for r in cr) / len(cr):.4f}; "
                      f"ORACLE win {sum(r['runs']['oracle/planner']['victory'] for r in cr) / len(cr):.2f}, top3 win "
                      f"{sum(r['runs']['top3/planner']['victory'] for r in cr) / len(cr):.2f}", flush=True)
                del prepared, e0s
                torch.cuda.empty_cache()
        # spectra: GPU lobpcg vs SciPy on 3 worlds of the default cell (§1.1 point 4 check)
        spec_check = []
        for s in seeds[:3]:
            e0 = generate_electorate(dataclasses.replace(cell_config(base, 30, "default[.01,.2]", 0.0), seed=s), device=device)
            t0 = time.perf_counter()
            g = xc.spectra_gpu(e0.graph)
            torch.cuda.synchronize(device)
            tg = time.perf_counter() - t0
            t0 = time.perf_counter()
            c = xc.spectra_scipy(e0.graph)
            tc = time.perf_counter() - t0
            spec_check.append({"seed": s, "gpu": g, "scipy": c, "gpu_seconds": tg, "scipy_seconds": tc,
                               "max_abs_diff": max(abs(g[k] - c[k]) for k in g)})
        rec["spectra_gpu_vs_scipy"] = spec_check
        summary = summarise(rows)
        par = parity(seeds[:args.parity_seeds], device, base)
        print(f"[x1] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    report = {"experiment": "X1", "question": "Network structure and consensus speed; nationwide manifesto vs p_g",
              "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
              "n": args.n, "rounds": base.rounds, "treasury": str(base.budget), "kbars": list(KBARS), "pg": PG,
              "long_rounds": {"het": HET_T, "hom": HOM_T},
              "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
              "metric_definitions": {
                  "rho_rw": "1 − ā(1 − μ₂(D⁻¹A)) (homogeneous-step, random-walk form)",
                  "rho_L": "1 − ā·λ₂(L)/k̄ (Laplacian form)", "rho_exact": "1 − ν₂(G^½ L G^½), G = diag(a/d)",
                  "rho_obs_200_400": "exp(slope of log e(t)), t ∈ [200, 400]; e = ‖x − 1c‖ on the giant component",
                  "rho_horizon5": "(e(5)/e(0))^{1/5} (the game horizon)",
                  "t_half_between": "first round where the between-region TV dispersion ≤ ½ its start (None if not by "
                                    "the end of the run)",
                  "t_half_pred_rw": "ln 2 / −ln ρ_rw",
                  "conductance": "cut(r, V∖r) / min(vol r, vol V∖r), per region",
                  "theory_vs_obs": "per world, decay rate 1 − ρ observed vs predicted: Spearman, mean log ratio, RMSE",
                  "nationwide": "win of the planner with a nationwide manifesto (top3; also ORACLE, uniform), p_g high − low"},
              **summary,
              "deviations": ["Spectra (λ₂, μ₂, ν₂) by SciPy on the CPU: at N = 10⁴ SciPy is ~5× faster than the GPU "
                             "lobpcg path (provenance.spectra_gpu_vs_scipy; §1.1 'measured slower' exception).",
                             "'Standard strategies' are the X0 policy set via the batched runner; the observation-only "
                             "players (BayesPlanner, Top-3 player) are X6's job and are not rerun here.",
                             f"Heterogeneous long runs are {HET_T} rounds (the asymptotic tail is slow), homogeneous "
                             f"{HOM_T} (plan: ≥ 400)."],
              "config": base.to_record(),
              "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=[len(seeds)], free_at_start=free0,
                                             monitor=monitor, device_record=dev_record, parity=par,
                                             extra={"timing": rec["timing"],
                                                    "spectra_gpu_vs_scipy": rec["spectra_gpu_vs_scipy"]})}
    xc.save(args.out, report)
    print(f"[x1] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
