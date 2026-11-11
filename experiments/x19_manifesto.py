"""X19 — Manifesto geometry and a third optimiser (plan ``docs/v2/plans/implementation_plan.md`` §5 X19).

Command (reproduces reports/v3/x19.json, reports/v3/raw/x19.jsonl and reports/v3/logs/x19.log):
  OMP_NUM_THREADS=4 python experiments/x19_manifesto.py --seeds 100 --first-seed 12000 --device cuda:0 \
      --out reports/v3/x19.json 2>&1 | tee reports/v3/logs/x19.log

Objective: f(b) = E(b) = mean_i σ(β(1 − TV(v_i, b) − z)) on the ANTICIPATED true electorate v_T (social-only
dynamics for T rounds, ``oracle.anticipate``), b ∈ Δ⁴. Optimisers ascend the pseudo-Huber surrogate (δ = 1e-3, as
``manifesto_opt``); every reported E is the exact (non-smoothed) objective.

Optimisers (all batched over starts ``[M, 5]`` on the GPU; the two new ones are defined HERE, additive only):
* PG   — projected gradient (prox of the simplex indicator = Euclidean projection), same update as
         ``manifesto_opt.optimise_manifesto``;
* ADMM — consensus ADMM across the 4 regions, same scaled-form updates as ``manifesto_opt.admm_manifesto``
         (δ = 0.02, ρ = 5, 10 inner gradient steps), vectorised over regions and starts so the E trace can be
         recorded every iteration (checked against the library on one start);
* MD   — :func:`mirror_descent`: exponentiated gradient, the entropic prox b ← b ⊙ exp(η∇f) / Σ;
* FW   — :func:`frank_wolfe`: linear maximisation over Δ (a vertex) and a line search of the exact objective
         over γ ∈ {0} ∪ 64 log-spaced points in [1e-6, 1] (batched).
Step sizes of PG and MD are tuned on 5 held-out seeds (12100–12104, not in the 100) from a grid, by the median
iterations to 1e-4; the library default PG step (0.02) is also reported.

Measured per seed: b* = the best exact E over every end point (all optimisers, all starts); iterations to reach
E* − 1e-4 (multi-start incumbent = max over the 8 standard starts), wall time, final E; local optima of PG and MD
from 64 flat-Dirichlet starts (end points clustered at TV < 0.01); b* vs mean, per-sector median (projected),
L1 geometric median (Weiszfeld-type iteration under TV), every 2-, 3-, 4-region blend of the anticipated regional
means and the ORACLE manifesto (TV distance and E); regions carried (regional mean P > 0.5) and coalition size.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.manifesto_opt import (  # noqa: E402
    admm_manifesto, default_starts, expected_share, optimise_manifesto, project_simplex,
)
from dumbocracy.strategies.oracle import anticipate, oracle_manifesto, top3_manifesto  # noqa: E402

RAW_PATH = xc.RAW / "x19.jsonl"
TOL = 1e-4
TUNE_SEEDS = list(range(12100, 12105))
PG_STEPS = (0.005, 0.01, 0.02, 0.05, 0.1)
MD_STEPS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
ITERS = 400
SMOOTH = 1e-3


# ============================================================================== objective
def exact(v: torch.Tensor, b: torch.Tensor, beta: float, z: float) -> torch.Tensor:
    return expected_share(v, b, beta, z, 0.0)


def grad(v: torch.Tensor, b: torch.Tensor, beta: float, z: float, smoothing: float = SMOOTH) -> torch.Tensor:
    b = b.detach().requires_grad_(True)
    (g,) = torch.autograd.grad(expected_share(v, b, beta, z, smoothing).sum(), b)
    return g


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================================== optimisers (batched over starts)
def projected_gradient(v, starts, beta, z, *, step=0.02, iterations=ITERS, tol=1e-7):
    """Same update as ``manifesto_opt.optimise_manifesto``; returns end points and the exact-E trace ``[it+1, M]``."""
    b = project_simplex(starts.to(v)).clone()
    trace = [exact(v, b, beta, z)]
    for _ in range(iterations):
        new_b = project_simplex(b + step * grad(v, b, beta, z))
        moved = float((new_b - b).abs().max())
        b = new_b
        trace.append(exact(v, b, beta, z))
        if moved < tol:
            break
    return b, torch.stack(trace)


def mirror_descent(v, starts, beta, z, *, step=0.1, iterations=ITERS, tol=1e-7, floor=1e-12):
    """Exponentiated gradient (entropic mirror ascent) on Δ: b ← b ⊙ exp(η ∇f(b)) / Σ. Starts are moved into the
    relative interior (mixed with 1e-6 of uniform), because a zero coordinate can never become positive."""
    k = starts.shape[-1]
    b = (1 - 1e-6) * project_simplex(starts.to(v)) + 1e-6 / k
    trace = [exact(v, b, beta, z)]
    for _ in range(iterations):
        g = grad(v, b, beta, z)
        logits = torch.log(b.clamp_min(floor)) + step * (g - g.max(-1, keepdim=True).values)
        new_b = torch.softmax(logits, -1)
        moved = float((new_b - b).abs().max())
        b = new_b
        trace.append(exact(v, b, beta, z))
        if moved < tol:
            break
    return b, torch.stack(trace)


def frank_wolfe(v, starts, beta, z, *, iterations=ITERS, grid=65, tol=1e-9):
    """Frank–Wolfe on Δ: s = vertex e_{argmax ∇f}, b ← b + γ(s − b) with γ from an exact line search of the exact
    objective on ``grid`` points: γ = 0 and a log-spaced grid on [1e-6, 1] (all starts and γ batched)."""
    b = project_simplex(starts.to(v)).clone()
    M, k = b.shape
    # γ = 0 plus a log-spaced grid on [1e-6, 1]: a uniform grid (smallest step 1/64) stalls near the optimum
    gammas = torch.cat([torch.zeros(1, dtype=v.dtype, device=v.device),
                        torch.logspace(-6, 0, grid - 1, dtype=v.dtype, device=v.device)])
    trace = [exact(v, b, beta, z)]
    for _ in range(iterations):
        g = grad(v, b, beta, z)
        s = torch.nn.functional.one_hot(g.argmax(-1), k).to(v.dtype)
        cand = b.unsqueeze(1) + gammas.view(1, -1, 1) * (s - b).unsqueeze(1)          # [M, G, 5]
        vals = exact(v, cand.reshape(-1, k), beta, z).view(M, grid)
        best = vals.argmax(1)
        new_b = cand[torch.arange(M), best]
        moved = float((new_b - b).abs().max())
        b = new_b
        trace.append(exact(v, b, beta, z))
        if moved < tol:
            break
    return b, torch.stack(trace)


def admm_batched(v, region_of, starts, beta, z, *, rho=5.0, iterations=300, inner_steps=10, inner_step=0.05,
                 smoothing=0.02, tol=1e-6, num_regions=4):
    """``manifesto_opt.admm_manifesto`` vectorised over starts ``[M, 5]`` and regions; records the exact E of the
    consensus z and the primal/dual residuals per iteration (per start)."""
    M, k = starts.shape
    dtype, dev = v.dtype, v.device
    n = v.shape[0]
    counts = torch.bincount(region_of, minlength=num_regions).to(dtype)
    weights = counts / n
    zc = project_simplex(starts.to(v)).clone()                                           # [M, 5]
    x = zc.unsqueeze(1).expand(M, num_regions, k).clone()                               # [M, R, 5]
    u = torch.zeros_like(x)
    trace, primal, dual = [exact(v, zc, beta, z)], [], []
    for _ in range(iterations):
        anchor = zc.unsqueeze(1) - u
        xr = x.clone()
        for _ in range(inner_steps):
            xr.requires_grad_(True)
            xv = xr[:, region_of]                                                        # [M, N, 5]
            diff = v.unsqueeze(0) - xv
            sim = 1 - 0.5 * (torch.sqrt(diff * diff + smoothing ** 2) - smoothing).sum(-1)
            prob = torch.sigmoid(beta * (sim - z))                                       # [M, N]
            share_r = torch.zeros(M, num_regions, dtype=dtype, device=dev).index_add(1, region_of, prob) / counts
            loss = (-(weights * share_r) + 0.5 * rho * ((xr - anchor) ** 2).sum(-1)).sum()
            (g,) = torch.autograd.grad(loss, xr)
            xr = (xr - inner_step * g).detach()
        x = xr
        z_prev = zc
        zc = project_simplex((x + u).mean(1))
        u = u + x - zc.unsqueeze(1)
        primal.append(torch.linalg.vector_norm(x - zc.unsqueeze(1), dim=(1, 2)))
        dual.append(rho * num_regions ** 0.5 * torch.linalg.vector_norm(zc - z_prev, dim=1))
        trace.append(exact(v, zc, beta, z))
        if bool((primal[-1] < tol).all() and (dual[-1] < tol).all()):
            break
    return zc, torch.stack(trace), torch.stack(primal), torch.stack(dual)


# ============================================================================== geometry
def tv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a - b).abs().sum(-1)


def geometric_median_tv(v: torch.Tensor, iterations: int = 200, eps: float = 1e-8) -> tuple[torch.Tensor, int]:
    """Weiszfeld-type iteration under TV: b ← Σ w_i v_i / Σ w_i with w_i = 1 / max(TV(v_i, b), eps), from the mean.
    Each iterate is a convex combination of voters, so it stays in Δ."""
    b = v.mean(0)
    it = 0
    for it in range(1, iterations + 1):
        w = 1.0 / tv(v, b).clamp_min(eps)
        new = (w.unsqueeze(-1) * v).sum(0) / w.sum()
        if float((new - b).abs().max()) < 1e-10:
            b = new
            break
        b = new
    return b, it


def cluster(points: torch.Tensor, scores: torch.Tensor, radius: float = 0.01) -> list[dict]:
    """Greedy clustering of end points at TV < ``radius`` (best E first)."""
    order = torch.argsort(scores, descending=True).tolist()
    centres: list[dict] = []
    for i in order:
        for c in centres:
            if float(tv(points[i], c["b"])) < radius:
                c["members"] += 1
                break
        else:
            centres.append({"b": points[i], "E": float(scores[i]), "members": 1})
    return centres


def regional_support(v: torch.Tensor, region_of: torch.Tensor, b: torch.Tensor, beta: float, z: float) -> list[float]:
    sim = 1 - tv(v, b)
    prob = torch.sigmoid(beta * (sim - z))
    sums = torch.zeros(4, dtype=prob.dtype, device=prob.device).index_add_(0, region_of, prob)
    return (sums / torch.bincount(region_of, minlength=4).to(prob.dtype)).tolist()


def first_hit(trace: torch.Tensor, target: float) -> int | None:
    """First iteration where the multi-start incumbent (max over starts) reaches ``target``; None if never."""
    best = trace.max(1).values if trace.dim() == 2 else trace
    hits = torch.nonzero(best >= target).flatten()
    return int(hits[0]) if hits.numel() else None


# ============================================================================== one seed
def standard_starts(v: torch.Tensor) -> torch.Tensor:
    """The library's default starts (uniform, mean, 5 pulled vertices) plus the top-3 statistic: 8 starts."""
    return project_simplex(torch.cat([default_starts(v), top3_manifesto(v).unsqueeze(0).to(v)]))


def run_methods(v, region_of, starts, beta, z, steps: dict, device) -> dict:
    out = {}
    for name, fn in (("PG", lambda: projected_gradient(v, starts, beta, z, step=steps["PG"])),
                     ("PG_lib_step", lambda: projected_gradient(v, starts, beta, z, step=0.02)),
                     ("MD", lambda: mirror_descent(v, starts, beta, z, step=steps["MD"])),
                     ("FW", lambda: frank_wolfe(v, starts, beta, z)),
                     ("ADMM", lambda: admm_batched(v, region_of, starts, beta, z))):
        _sync(device)
        t0 = time.perf_counter()
        res = fn()
        _sync(device)
        seconds = time.perf_counter() - t0
        b, trace = res[0], res[1]
        final = exact(v, b, beta, z)
        out[name] = {"b": b, "trace": trace, "final": final, "seconds": seconds, "iterations": trace.shape[0] - 1}
        if name == "ADMM":
            out[name]["primal"], out[name]["dual"] = res[2], res[3]
    return out


def seed_task(task: tuple) -> dict:
    seed, n, device, steps, dirichlet = task
    torch.set_num_threads(4)
    dev = torch.device(device)
    start = time.perf_counter()
    cfg = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=dev)
    v = anticipate(e, cfg)
    beta, z = cfg.voting.sensitivity, cfg.voting.threshold
    region_of = e.region_of
    sizes = [int(x) for x in e.sizes.tolist()]
    starts = standard_starts(v)
    meth = run_methods(v, region_of, starts, beta, z, steps, dev)
    # 64 Dirichlet starts (PG with the tuned step and MD): local optima
    g = torch.Generator(device="cpu").manual_seed(int(seed) * 7 + 19)
    x = -torch.log(torch.rand(dirichlet, 5, generator=g, dtype=torch.float64))      # flat Dirichlet(1) draws
    dstarts = (x / x.sum(-1, keepdim=True)).to(device=dev, dtype=v.dtype)
    local = {}
    for name, fn in (("PG", lambda: projected_gradient(v, dstarts, beta, z, step=steps["PG"])),
                     ("MD", lambda: mirror_descent(v, dstarts, beta, z, step=steps["MD"]))):
        b_end, trace = fn()
        local[name] = {"b": b_end, "E": exact(v, b_end, beta, z)}
    # b* = best exact E over every end point
    cands = [meth[m]["b"] for m in meth] + [local[m]["b"] for m in local]
    allb = torch.cat(cands)
    allE = exact(v, allb, beta, z)
    k = int(torch.argmax(allE))
    b_star, E_star = allb[k], float(allE[k])
    row: dict = {"exp": "X19", "seed": seed, "sizes": sizes, "sigma": e.archetype_of_region.tolist(),
                 "tau": e.personality_of_region.tolist(), "b_star": [float(x) for x in b_star.tolist()], "E_star": E_star}
    # optimiser comparison
    opt = {}
    for name, m in meth.items():
        hit = first_hit(m["trace"], E_star - TOL)
        per_it = m["seconds"] / max(1, m["iterations"])
        best_i = int(torch.argmax(m["final"]))
        opt[name] = {"iterations_run": m["iterations"], "iterations_to_tol": hit,
                     "seconds_total": m["seconds"], "seconds_to_tol": None if hit is None else hit * per_it,
                     "final_E_best_start": float(m["final"].max()), "final_E_mean_starts": float(m["final"].mean()),
                     "gap_to_star": E_star - float(m["final"].max()),
                     "starts_within_tol": int((m["final"] >= E_star - TOL).sum()),
                     "b_best": [float(x) for x in m["b"][best_i].tolist()]}
        if name == "ADMM":
            opt[name]["primal_final"] = [float(x) for x in m["primal"][-1].tolist()]
            opt[name]["dual_final"] = [float(x) for x in m["dual"][-1].tolist()]
            opt[name]["primal_trace_start0"] = [float(x) for x in m["primal"][:, 0].tolist()]
            opt[name]["dual_trace_start0"] = [float(x) for x in m["dual"][:, 0].tolist()]
        opt[name]["incumbent_trace"] = [round(float(x), 7) for x in m["trace"].max(1).values.tolist()]
    row["optimisers"] = opt
    # local optima
    row["local_optima"] = {}
    for name, l in local.items():
        cl = cluster(l["b"], l["E"])
        row["local_optima"][name] = {"starts": dirichlet, "distinct": len(cl),
                                     "cluster_E": [c["E"] for c in cl], "cluster_members": [c["members"] for c in cl],
                                     "frac_starts_at_star": float((l["E"] >= E_star - TOL).float().mean()),
                                     "best_E": float(l["E"].max()), "E_spread": float(l["E"].max() - l["E"].min()),
                                     "distinct_E_ge_star_minus_0.01": sum(1 for c in cl if c["E"] >= E_star - 0.01)}
    # geometry: references
    regional_means = torch.stack([v[region_of == r].mean(0) for r in range(4)])
    refs: dict[str, torch.Tensor] = {"mean": v.mean(0)}
    refs["median_projected"] = project_simplex(v.median(0).values.unsqueeze(0))[0]
    gm, gm_it = geometric_median_tv(v)
    refs["geomedian_tv"] = gm
    refs["oracle_manifesto"] = oracle_manifesto(e, cfg).to(v)
    for size in (2, 3, 4):
        for comb in itertools.combinations(range(4), size):
            w = torch.tensor([sizes[r] for r in comb], dtype=v.dtype, device=v.device)
            refs["blend_" + "".join(str(r) for r in comb)] = (w.unsqueeze(-1) * regional_means[list(comb)]).sum(0) / w.sum()
    geo = {}
    for name, b in refs.items():
        geo[name] = {"tv_to_star": float(tv(b, b_star)), "E": float(exact(v, b, beta, z)),
                     "b": [round(float(x), 6) for x in b.tolist()]}
        if name.startswith("blend_"):
            members = [int(c) for c in name[6:]]
            geo[name]["coalition_share"] = sum(sizes[r] for r in members) / sum(sizes)
    row["geometry"] = geo
    row["geomedian_iterations"] = gm_it
    blends23 = [k for k in geo if k.startswith("blend_") and len(k) - 6 in (2, 3)]
    nearest = min(geo, key=lambda k: geo[k]["tv_to_star"] if k != "oracle_manifesto" else 9)
    row["nearest_reference"] = nearest
    row["tv_star_nearest_blend23"] = min(geo[k]["tv_to_star"] for k in blends23)
    row["nearest_blend23"] = min(blends23, key=lambda k: geo[k]["tv_to_star"])
    support = regional_support(v, region_of, b_star, beta, z)
    carried = [r for r in range(4) if support[r] > 0.5]
    share_c = sum(sizes[r] for r in carried) / sum(sizes)
    minimal = share_c > 0.5 and all(share_c - sizes[r] / sum(sizes) <= 0.5 for r in carried)
    row["carried"] = {"support_by_region": support, "regions": carried, "count": len(carried),
                      "voter_share": share_c, "minimal_winning": bool(minimal)}
    row["seconds"] = time.perf_counter() - start
    row["device"] = str(dev)
    return row


# ============================================================================== tuning and checks
def tune_steps(device: torch.device, n: int) -> dict:
    """Pick the PG and MD steps by the median iterations to 1e-4 (incumbent over the 8 standard starts) on the
    held-out TUNE_SEEDS; the target E* is the best end point over all grid runs of that seed."""
    res = {"PG": {s: [] for s in PG_STEPS}, "MD": {s: [] for s in MD_STEPS}}
    for seed in TUNE_SEEDS:
        cfg = GameConfig(seed=seed, device=str(device), electorate=ElectorateConfig(num_voters=n))
        e = generate_electorate(cfg, device=device)
        v = anticipate(e, cfg)
        beta, z = cfg.voting.sensitivity, cfg.voting.threshold
        starts = standard_starts(v)
        runs = [("PG", s, projected_gradient(v, starts, beta, z, step=s)) for s in PG_STEPS] + \
               [("MD", s, mirror_descent(v, starts, beta, z, step=s)) for s in MD_STEPS]
        star = max(float(r[2][1].max()) for r in runs)
        for name, s, (_, trace) in runs:
            hit = first_hit(trace, star - TOL)
            res[name][s].append(ITERS + 1 if hit is None else hit)
    chosen, table = {}, {}
    for name, grid in res.items():
        med = {s: float(np.median(v)) for s, v in grid.items()}
        table[name] = {str(s): {"median_iterations": med[s], "per_seed": grid[s]} for s in grid}
        chosen[name] = min(med, key=lambda s: (med[s], s))
    return {"seeds": TUNE_SEEDS, "chosen": chosen, "grid": table, "criterion": "median iterations to E*−1e-4 "
            "(misses count as 401); ties → smaller step"}


def library_check(device: torch.device, n: int, seed: int) -> dict:
    """Experiment-side PG / ADMM reproduce the library functions (same start, same parameters)."""
    cfg = GameConfig(seed=seed, device=str(device), electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=device)
    v = anticipate(e, cfg)
    beta, z = cfg.voting.sensitivity, cfg.voting.threshold
    starts = default_starts(v)
    lib = optimise_manifesto(v, beta, z, starts=starts)
    b, trace = projected_gradient(v, starts, beta, z, step=0.02)
    mine = exact(v, b, beta, z)
    uni = torch.full((5,), 0.2, dtype=v.dtype, device=v.device)
    lib_admm = admm_manifesto(v, e.region_of, beta, z, start=uni)
    za, ta, pa, da = admm_batched(v, e.region_of, uni.unsqueeze(0), beta, z)
    return {"seed": seed, "pg_library_share": lib.share, "pg_experiment_best_share": float(mine.max()),
            "pg_abs_diff": abs(lib.share - float(mine.max())),
            "admm_library_share": lib_admm.share, "admm_experiment_share": float(exact(v, za, beta, z)[0]),
            "admm_library_iterations": lib_admm.iterations, "admm_experiment_iterations": int(ta.shape[0] - 1),
            "admm_tv_between_solutions": float(tv(za[0], lib_admm.manifesto)),
            "admm_primal_final": [lib_admm.primal_residuals[-1], float(pa[-1, 0])],
            "admm_dual_final": [lib_admm.dual_residuals[-1], float(da[-1, 0])]}


def parity(seeds: list[int], device: torch.device, n: int, steps: dict) -> dict:
    def fn(seed, dev):
        r = seed_task((seed, n, str(dev), steps, 64))
        return {"E_star": r["E_star"], "PG_distinct": r["local_optima"]["PG"]["distinct"],
                "carried": r["carried"]["count"], "tv_star_mean": r["geometry"]["mean"]["tv_to_star"]}

    out = xc.parity_check(fn, seeds, device=device, abs_tol=0.002)
    # counts are integers: tolerance 1 (deterministic optimisation, float32 sums differ CPU vs GPU)
    for key in ("PG_distinct", "carried"):
        m = out["metrics"][key]
        m["tolerance"] = max(m["tolerance"], 1.0)
        m["passed"] = abs(m["mean_diff"]) <= m["tolerance"]
        m["note"] = "integer count, tolerance 1"
    out["passed"] = all(m["passed"] for m in out["metrics"].values())
    return out


# ============================================================================== statistics
def summarise(rows: list[dict]) -> dict:
    out: dict = {"n_seeds": len(rows), "E_star": xc.mean_ci([r["E_star"] for r in rows])}
    methods = list(rows[0]["optimisers"])
    opt = {}
    for m in methods:
        rs = [r["optimisers"][m] for r in rows]
        hits = [x["iterations_to_tol"] for x in rs]
        reached = [h for h in hits if h is not None]
        opt[m] = {"reached_tol": xc.rate([h is not None for h in hits]),
                  "iterations_to_tol_median": float(np.median(reached)) if reached else None,
                  "iterations_to_tol_iqr": [float(np.percentile(reached, 25)), float(np.percentile(reached, 75))]
                  if reached else None,
                  "iterations_to_tol_mean_reached": xc.mean_ci(reached) if reached else None,
                  "seconds_to_tol_median": float(np.median([x["seconds_to_tol"] for x in rs if x["seconds_to_tol"] is not None]))
                  if reached else None,
                  "seconds_total": xc.mean_ci([x["seconds_total"] for x in rs]),
                  "iterations_run": xc.mean_ci([x["iterations_run"] for x in rs]),
                  "final_E_best_start": xc.mean_ci([x["final_E_best_start"] for x in rs]),
                  "gap_to_star": xc.mean_ci([x["gap_to_star"] for x in rs]),
                  "gap_to_star_max": max(x["gap_to_star"] for x in rs),
                  "starts_within_tol_mean": sum(x["starts_within_tol"] for x in rs) / len(rs)}
        if m == "ADMM":
            opt[m]["primal_final_max"] = max(max(x["primal_final"]) for x in rs)
            opt[m]["dual_final_max"] = max(max(x["dual_final"]) for x in rs)
            opt[m]["primal_final_median"] = float(np.median([np.median(x["primal_final"]) for x in rs]))
            opt[m]["dual_final_median"] = float(np.median([np.median(x["dual_final"]) for x in rs]))
    out["optimisers"] = opt
    # paired tests: iterations to tol (misses = run length + 1) MD vs PG, and others vs PG; final E vs PG
    big = ITERS + 1

    def its(m):
        return [r["optimisers"][m]["iterations_to_tol"] if r["optimisers"][m]["iterations_to_tol"] is not None else big
                for r in rows]

    fam = []
    for a, b in (("MD", "PG"), ("FW", "PG"), ("ADMM", "PG"), ("MD", "FW"), ("PG_lib_step", "PG")):
        fam.append({"a": a, "b": b, "iterations_to_tol": xc.paired(its(a), its(b)),
                    "final_E": xc.paired([r["optimisers"][a]["final_E_best_start"] for r in rows],
                                         [r["optimisers"][b]["final_E_best_start"] for r in rows])})
    for x, p in zip(fam, xc.holm([x["iterations_to_tol"]["wilcoxon_p"] for x in fam])):
        x["iterations_to_tol"]["holm_p"] = p
    for x, p in zip(fam, xc.holm([x["final_E"]["wilcoxon_p"] for x in fam])):
        x["final_E"]["holm_p"] = p
    out["optimiser_tests"] = {"family": "paired per seed, misses counted as 401 iterations; Wilcoxon, Holm per metric",
                              "tests": fam}
    # interior vs boundary optima (prediction: MD faster on interior optima)
    interior = [r for r in rows if min(r["b_star"]) > 1e-3]
    boundary = [r for r in rows if min(r["b_star"]) <= 1e-3]
    out["interior_vs_boundary"] = {"interior_n": len(interior), "boundary_n": len(boundary)}
    for label, rs in (("interior", interior), ("boundary", boundary)):
        if len(rs) >= 2:
            a = [r["optimisers"]["MD"]["iterations_to_tol"] or big for r in rs]
            b = [r["optimisers"]["PG"]["iterations_to_tol"] or big for r in rs]
            out["interior_vs_boundary"][label] = {"MD_median": float(np.median(a)), "PG_median": float(np.median(b)),
                                                  "MD_minus_PG": xc.paired(a, b)}
    # local optima
    out["local_optima"] = {}
    for m in ("PG", "MD"):
        d = [r["local_optima"][m]["distinct"] for r in rows]
        out["local_optima"][m] = {"distinct": xc.mean_ci(d), "distinct_median": float(np.median(d)),
                                  "distinct_max": max(d), "seeds_with_gt1": sum(1 for x in d if x > 1),
                                  "frac_starts_at_star": xc.mean_ci([r["local_optima"][m]["frac_starts_at_star"] for r in rows]),
                                  "E_spread": xc.mean_ci([r["local_optima"][m]["E_spread"] for r in rows]),
                                  "near_optimal_distinct": xc.mean_ci([r["local_optima"][m]["distinct_E_ge_star_minus_0.01"]
                                                                       for r in rows])}
    # geometry
    names = list(rows[0]["geometry"])
    geo = {}
    for nme in names:
        geo[nme] = {"tv_to_star": xc.mean_ci([r["geometry"][nme]["tv_to_star"] for r in rows]),
                    "E": xc.mean_ci([r["geometry"][nme]["E"] for r in rows]),
                    "E_gap_to_star": xc.mean_ci([r["E_star"] - r["geometry"][nme]["E"] for r in rows])}
    out["geometry"] = geo
    near = [r["tv_star_nearest_blend23"] for r in rows]
    fam = []
    for ref in ("mean", "median_projected", "geomedian_tv"):
        other = [r["geometry"][ref]["tv_to_star"] for r in rows]
        fam.append({"a": "nearest 2–3-region blend", "b": ref, "tv_difference": xc.paired(near, other),
                    "seeds_blend_closer": sum(1 for x, y in zip(near, other) if x < y)})
    for x, p in zip(fam, xc.holm([x["tv_difference"]["wilcoxon_p"] for x in fam])):
        x["tv_difference"]["holm_p"] = p
    out["geometry_tests"] = {"family": "TV(b*, nearest 2–3-region blend) − TV(b*, reference); Wilcoxon, Holm",
                             "tests": fam}
    counts: dict[str, int] = {}
    for r in rows:
        key = r["nearest_reference"]
        key = ("blend_" + str(len(key) - 6)) if key.startswith("blend_") else key
        counts[key] = counts.get(key, 0) + 1
    out["nearest_reference_counts"] = counts
    out["carried"] = {"count_distribution": {str(k): sum(1 for r in rows if r["carried"]["count"] == k) for k in range(5)},
                      "count": xc.mean_ci([r["carried"]["count"] for r in rows]),
                      "voter_share": xc.mean_ci([r["carried"]["voter_share"] for r in rows]),
                      "minimal_winning": xc.rate([r["carried"]["minimal_winning"] for r in rows]),
                      "wins_without_majority_of_regions": sum(1 for r in rows if r["carried"]["voter_share"] <= 0.5)}
    out["b_star_min_coordinate"] = xc.mean_ci([min(r["b_star"]) for r in rows])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X19"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dirichlet", type=int, default=64)
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--out", default=str(xc.REPORTS / "x19.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=2 * 2**30)
    free0 = xc.free_bytes(device)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    deviations = [f"Device {device} instead of the plan default cuda:2 (cuda:2 is used by the Phase 2 agent)."]
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        tuning = tune_steps(device, args.n)
        steps = tuning["chosen"]
        print(f"[x19] tuned steps {steps} ({time.perf_counter() - start:.0f}s)", flush=True)
        check = library_check(device, args.n, TUNE_SEEDS[0])
        print(f"[x19] library check {check}", flush=True)
        pilot = seed_task((seeds[0], args.n, str(device), steps, args.dirichlet))
        est = pilot["seconds"] * (len(seeds) - 1) * 1.1
        print(f"[x19] pilot seed {seeds[0]}: {pilot['seconds']:.1f}s; estimated remaining {est / 60:.1f} min", flush=True)
        if args.pilot_only:
            print({k: (v["iterations_to_tol"], round(v["final_E_best_start"], 5)) for k, v in pilot["optimisers"].items()})
            return
        rows = [pilot]
        for i, s in enumerate(seeds[1:]):
            rows.append(seed_task((s, args.n, str(device), steps, args.dirichlet)))
            if (i + 2) % 10 == 0:
                print(f"[x19] [{i + 2}/{len(seeds)}] {time.perf_counter() - start:.0f}s", flush=True)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text("")
        for r in rows:
            xc.write_row(RAW_PATH, r)
        par = parity(seeds[:args.parity_seeds], device, args.n, steps)
        print(f"[x19] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    report = {
        "experiment": "X19", "question": "Manifesto geometry and a third optimiser",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds}, "n": args.n, "rounds": base.rounds,
        "objective": "exact expected share on the anticipated true electorate (social-only T rounds)",
        "tolerance": TOL, "max_iterations": {"PG": ITERS, "MD": ITERS, "FW": ITERS, "ADMM": 300},
        "starts": "8 standard starts (library default_starts + top-3); local optima: 64 flat-Dirichlet starts",
        "step_tuning": tuning, "library_check": check,
        **summarise(rows), "deviations": deviations, "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, batch_sizes={"starts_M": 8, "dirichlet_M": args.dirichlet,
                                                                           "fw_line_search_grid": 65},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       grad_mode="autograd (optimisers)"),
    }
    xc.save(args.out, report)
    print(f"[x19] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
