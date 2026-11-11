"""X3 — Worker cascade threshold and reach (plan ``docs/v2/plans/implementation_plan.md`` §4 X3; P9 X3).

Command (reproduces reports/v3/x3.json, reports/v3/raw/x3.jsonl and reports/v3/logs/x3.log):
  OMP_NUM_THREADS=4 python experiments/x3_cascade.py --seeds 50 --first-seed 11000 --device cuda:2 \
      --out reports/v3/x3.json 2>&1 | tee reports/v3/logs/x3.log

Design (50 worlds × 9 manifesto points × k ∈ {1, 3, 5, 10}, 10 cascade trials each; N = 10⁴, T = 5, ₫500):
* target region r = the largest region of each world; k workers bought in round 1 in r (``WorkerTiming``,
  "front", "region:r"), no media, nothing else. A k = 0 run per (world, θ) is the no-worker baseline.
* manifesto sweep b(θ) = (1 − θ)·b_far + θ·b_ideal, θ ∈ {0, 1/8, …, 1}: b_ideal = the region's mean v(0),
  b_far = the archetype mean farthest (TV) from it.
* 10 cascade trials = 10 salted ``workers`` + ``cascade`` streams (``x_streams.SaltedStreams``, salt = trial;
  trial 0 = the game's own streams); every other stream is the world's own. All trials and worlds of one
  (θ, k) cell are ONE batched ``x_batch.run_batch`` call on the GPU (full game dynamics: one IC hop per round,
  social step, workers pinned, zealots frozen).

Theory (per world and θ; q_j = γ_j·P_j(b) on v(0), q = 0 for zealots):
* weighted threshold λ_max(A D_q) = λ_max(D_q^{1/2} A D_q^{1/2}) (symmetric, same spectrum) by batched GPU power
  iteration (all worlds × θ columns at once, per-world normalisation on the disjoint-union graph); globally and
  restricted to the target region (A_rr D_q,r). SciPy ``eigsh`` on the CPU is the reference (checked on 3 worlds).
* naive estimates: k̄_r·E_r[q] and λ_max(A_rr)·E_r[q] (region r), and the global k̄·E[q], λ_max(A)·E[q].
* cap-corrected branching (observed hops): Σ_j [1 − (1 − q_j)^{eff_j}] / spreaders with eff_j = min(count_j,
  cap − attempts_j) (q on v at the start of the round), next to the uncapped Σ_j [1 − (1 − q_j)^{count_j}].

Observed: converts per hop; R_obs = pooled branching ratio of the converts (Σ_{t≥2} c_t / Σ_{t≤T−1} c_t, i.e.
converts produced per convert-spreader; the worker hop is reported separately as c_1/k); distinct converts and
their share outside region r (cross-region); cap binding (share of targeted voters with final attempts = cap);
drift-back (final P of converted, non-worker voters by the number of social steps since their last
conversion; P right after conversion is P(b, b) = σ(β_v(1 − z)) = 0.971).
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
from x_batch import conversion_chance_rows, run_batch  # noqa: E402
from x_policies import WorkerTiming  # noqa: E402
from x_streams import SaltedStreams  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.dynamics.campaign import spreader_edge_counts  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import PERSONALITIES, archetype_means  # noqa: E402
from dumbocracy.strategies.worlds import _union_graph  # noqa: E402
from dumbocracy.voting import similarity, vote_probability  # noqa: E402

THETAS = [i / 8 for i in range(9)]
KS = (1, 3, 5, 10)
EXT_TARGETS = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0)   # extension points (λ_max(A D_q) targets)
RAW_PATH = xc.RAW / "x3.jsonl"
POWER_LOG: list[dict] = []     # convergence record of every batched power iteration (provenance)


# ============================================================================== weighted spectral threshold
@torch.no_grad()
def batched_lambda_max(graphs, weights: torch.Tensor, *, iters: int = 3000, tol: float = 1e-9) -> tuple[torch.Tensor, int]:
    """λ_max(D_w^{1/2} A D_w^{1/2}) per world and column, by power iteration on the GPU.

    ``graphs``: B graphs of equal N; ``weights``: [B, N, C] non-negative (one column per operator). The B graphs
    are one disjoint union (block diagonal), the C columns one dense right-hand side of a sparse mat-mul; each
    (world, column) vector is normalised on its own block, so every block converges to its own Perron root.
    The iteration uses the shift M + I (all eigenvalues of M + I lie in (−ρ+1, ρ+1], so the Perron root
    dominates in modulus even on near-bipartite blocks). Returns ([B, C] float64, iterations used)."""
    B, n, C = weights.shape
    dev = weights.device
    union = graphs[0] if B == 1 else _union_graph(list(graphs), n)
    ones = torch.ones(union.indices.numel(), dtype=torch.float64, device=dev)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        adj = torch.sparse_csr_tensor(union.indptr, union.indices, ones, (B * n, B * n))
    s = weights.to(torch.float64).sqrt().reshape(B * n, C)
    x = s.clone().view(B, n, C)
    x = (x + 1e-12) / torch.linalg.vector_norm(x + 1e-12, dim=1, keepdim=True)
    x = x.reshape(B * n, C)
    value = torch.zeros(B, C, dtype=torch.float64, device=dev)
    it = 0
    for it in range(1, iters + 1):
        y = s * (adj @ (s * x)) + x                                          # (M + I) x
        new = (x * y).view(B, n, C).sum(1)                                   # Rayleigh quotient (x unit)
        y = y.view(B, n, C)
        y = y / torch.linalg.vector_norm(y, dim=1, keepdim=True).clamp_min(1e-300)
        x = y.reshape(B * n, C)
        change = ((new - value).abs() / new.abs().clamp_min(1.0)).max()
        value = new
        if it > 10 and float(change) <= tol:
            break
    POWER_LOG.append({"iterations": it, "max_rel_change_last_iter": float(change), "columns": int(B * C)})
    return value - 1.0, it


def scipy_lambda_max(graph, w: torch.Tensor) -> float:
    """CPU reference: largest eigenvalue of D_w^{1/2} A D_w^{1/2} with SciPy ``eigsh``."""
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh
    rows = graph.rows().cpu().numpy()
    cols = graph.indices.cpu().numpy()
    a = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(graph.num_nodes, graph.num_nodes))
    d = diags(np.sqrt(w.double().cpu().numpy()))
    return float(eigsh(d @ a @ d, k=1, which="LA", return_eigenvectors=False, tol=1e-10)[0])


# ============================================================================== worlds and manifestos
def prepare(seeds, base: GameConfig, device: torch.device) -> list[dict]:
    """Electorates, target region (largest), b_ideal, b_far, and the 9 plan manifestos b(θ). ``vertex`` = the
    simplex vertex of the sector with the lowest mean v(0) over all voters (the start of the extension path)."""
    means = archetype_means(dtype=torch.float64)
    out = []
    for s in seeds:
        e = generate_electorate(dataclasses.replace(base, seed=s), device=device)
        sizes = [int(x) for x in e.sizes.tolist()]
        r = max(range(4), key=lambda i: sizes[i])
        ideal = e.preferences[e.region_of == r].to(torch.float64).mean(0).cpu()
        ideal = ideal / ideal.sum()
        tv = 0.5 * (means - ideal).abs().sum(1)
        far_idx = int(tv.argmax())
        far = means[far_idx]
        vertex = torch.nn.functional.one_hot(e.preferences.to(torch.float64).mean(0).argmin().cpu(), 5).to(torch.float64)
        bs = torch.stack([(1 - th) * far + th * ideal for th in THETAS]).to(device=device, dtype=e.preferences.dtype)
        out.append({"seed": int(s), "e": e, "region": r, "sizes": sizes, "far_archetype": far_idx,
                    "tv_far_ideal": float(tv[far_idx]), "personality": PERSONALITIES[int(e.personality_of_region[r])].id,
                    "manifestos": bs, "far": far, "vertex": vertex, "phis": [], "points": [("theta", th) for th in THETAS]})
    return out


def _q_of(w: dict, bs: torch.Tensor, cfg: GameConfig) -> torch.Tensor:
    """q = γ·P(b) on v(0), 0 for zealots, float64 ``[len(bs), N]``."""
    e = w["e"]
    sim = similarity(e.preferences, bs.to(e.preferences.device, e.preferences.dtype))
    q = (e.susceptibility.campaign * vote_probability(sim, cfg.voting.sensitivity, cfg.voting.threshold)).clamp(0, 1)
    return torch.where(e.susceptibility.zealot.to(torch.bool), torch.zeros_like(q), q).to(torch.float64)


def extension(worlds: list[dict], cfg: GameConfig, targets=EXT_TARGETS, grid: int = 41) -> dict:
    """Addition to the plan sweep (recorded as such): per world, manifestos b(φ) = (1 − φ)·vertex + φ·b_far whose
    weighted operator hits λ_max(A D_q) ≈ each target (so the threshold region λ ≈ 1 is actually sampled).
    One batched power iteration over a 41-point φ grid, log-linear interpolation, then the chosen points are
    appended to each world's manifestos (their exact λ is computed with the main theory call)."""
    phis = torch.linspace(0, 1, grid, dtype=torch.float64)
    cols = []
    for w in worlds:
        bs = torch.stack([(1 - p) * w["vertex"] + p * w["far"] for p in phis])
        cols.append(_q_of(w, bs, cfg).T)
    lam, iters = batched_lambda_max([w["e"].graph for w in worlds], torch.stack(cols))
    lam = lam.cpu()
    for b, w in enumerate(worlds):
        curve = lam[b].clamp_min(1e-12).log()
        chosen = []
        for tgt in targets:
            lt = math.log(tgt)
            idx = next((j for j in range(1, grid) if curve[j - 1] <= lt <= curve[j]), None)
            if idx is None:     # target outside the path's range: nearest grid point
                idx2 = int((curve - lt).abs().argmin())
                chosen.append(float(phis[idx2]))
                continue
            f = float((lt - curve[idx - 1]) / (curve[idx] - curve[idx - 1]).clamp_min(1e-12))
            chosen.append(float(phis[idx - 1] + f * (phis[idx] - phis[idx - 1])))
        w["phis"] = chosen
        ext = torch.stack([(1 - p) * w["vertex"] + p * w["far"] for p in chosen]).to(w["manifestos"].device,
                                                                                    w["manifestos"].dtype)
        w["manifestos"] = torch.cat([w["manifestos"], ext])
        w["points"] = w["points"] + [("ext", t) for t in targets]
    return {"grid": grid, "iterations": iters, "targets": list(targets),
            "lambda_range_on_path": [[float(lam[b].min()), float(lam[b].max())] for b in range(len(worlds))]}


def theory(worlds: list[dict], cfg: GameConfig, device: torch.device) -> tuple[list[dict], int]:
    """Weighted and naive threshold estimates per world and manifesto point (GPU power iteration, one call)."""
    cols = []
    m = worlds[0]["manifestos"].shape[0]
    for w in worlds:
        e = w["e"]
        q = _q_of(w, w["manifestos"], cfg)                                                   # [M, N]
        in_r = (e.region_of == w["region"]).to(torch.float64)
        ones = torch.ones_like(in_r)
        w["q"] = q
        cols.append(torch.cat([q.T, (q * in_r).T, ones[:, None], in_r[:, None]], 1))         # [N, 2M + 2]
    weights = torch.stack(cols)                                                               # [B, N, 2M + 2]
    lam, iters = batched_lambda_max([w["e"].graph for w in worlds], weights)
    lam = lam.cpu()
    out = []
    for b, w in enumerate(worlds):
        e = w["e"]
        in_r = e.region_of == w["region"]
        deg = e.graph.degree.to(torch.float64)
        # k̄_rr: mean degree INSIDE region r (A_rr edges); k̄: mean total degree; both reported
        rows, cols_ = e.graph.rows(), e.graph.indices
        intra = torch.bincount(rows[in_r[rows] & in_r[cols_]], minlength=e.num_voters).to(torch.float64)
        kbar_rr = float(intra[in_r].mean())
        kbar = float(deg.mean())
        lam_a, lam_rr = float(lam[b, 2 * m]), float(lam[b, 2 * m + 1])
        per_point = []
        for i, (kind, val) in enumerate(w["points"]):
            q = w["q"][i]
            eq_r, eq = float(q[in_r].mean()), float(q.mean())
            per_point.append({"kind": kind, "theta": val if kind == "theta" else None,
                              "target": val if kind == "ext" else None,
                              "phi": w["phis"][i - len(THETAS)] if kind == "ext" else None,
                              "E_q_region": eq_r, "E_q": eq,
                              "lambda_ADq": float(lam[b, i]), "lambda_ADq_region": float(lam[b, m + i]),
                              "naive_kbar_rr_Eq": kbar_rr * eq_r, "naive_kbar_Eq_region": kbar * eq_r,
                              "naive_lambdaArr_Eq": lam_rr * eq_r, "naive_kbar_Eq_global": kbar * eq,
                              "naive_lambdaA_Eq_global": lam_a * eq})
        out.append({"lambda_A": lam_a, "lambda_Arr": lam_rr, "kbar": kbar, "kbar_rr": kbar_rr,
                    "per_theta": per_point})
    return out, iters


# ============================================================================== cascade runs
def play(cells: list[tuple[dict, int, int]], cfg: GameConfig, k: int) -> list[dict]:
    """One batched call: ``cells`` = [(world, θ index, trial)]. Returns per-game metric rows."""
    B = len(cells)
    es = [w["e"] for w, _, _ in cells]
    n = es[0].num_voters
    T, cap = cfg.rounds, cfg.workers.attempt_cap
    ms = torch.stack([w["manifestos"][i] for w, i, _ in cells])
    pols = [WorkerTiming(k, "front", f"region:{w['region']}") for w, _, _ in cells]
    streams = [SaltedStreams(w["seed"], {"workers": t, "cascade": t}) for w, _, t in cells]
    target = torch.tensor([w["region"] for w, _, _ in cells], device=es[0].device)
    st: dict = {}
    vcfg = cfg.voting

    def observer(t: int, info: dict) -> None:
        game = info["game"]
        if t == 0:
            st["v_prev"] = game.v0
            st["att_prev"] = torch.zeros_like(info["attempts"])
            st["last"] = torch.full((B * n,), -1, dtype=torch.int64, device=game.device)
            st["exp_capped"], st["exp_uncapped"], st["att_events"] = [], [], []
            st["in_target"] = game.region_of == target[game.world_of]
        q = conversion_chance_rows(game.campaign, st["v_prev"], game.b_rows, vcfg.sensitivity, vcfg.threshold,
                                   cfg.workers.conversion_rule).to(torch.float64)
        counts = spreader_edge_counts(game.graph, info["spreaders"], game.rows)
        blocked = info["pinned"] | game.zealot
        eff = torch.where(blocked, torch.zeros_like(counts), torch.minimum(counts, (cap - st["att_prev"]).clamp_min(0)))
        assert torch.equal(st["att_prev"] + eff, info["attempts"]), "observer eff disagrees with run_batch"
        unc = torch.where(blocked | game.zealot, torch.zeros_like(counts), counts)
        st["exp_capped"].append((1 - (1 - q).pow(eff.to(q.dtype))).view(B, n).sum(1))
        st["exp_uncapped"].append((1 - (1 - q).pow(unc.to(q.dtype))).view(B, n).sum(1))
        st["att_events"].append(eff.view(B, n).sum(1))
        st["last"] = torch.where(info["converted"], torch.full_like(st["last"], t), st["last"])
        st["v_prev"], st["att_prev"] = info["v"], info["attempts"]
        if t == T - 1:
            v = info["v"]
            p_final = vote_probability(1 - 0.5 * (v - game.b_rows).abs().sum(-1), vcfg.sensitivity, vcfg.threshold)
            ever = (st["last"] >= 0) & ~info["pinned"]
            steps = (T - st["last"]).clamp(0, T)                     # social steps since last conversion
            att = info["attempts"]
            targeted = att > 0
            rows = [ever.view(B, n).sum(1), (ever & ~st["in_target"]).view(B, n).sum(1),
                    targeted.view(B, n).sum(1), (targeted & (att >= cap)).view(B, n).sum(1),
                    info["pinned"].view(B, n).sum(1)]
            p_by_steps, n_by_steps = [], []
            for s in range(1, T + 1):
                m = ever & (steps == s)
                n_by_steps.append(m.view(B, n).sum(1))
                p_by_steps.append(torch.where(m, p_final, torch.zeros_like(p_final)).view(B, n).sum(1))
            st["final"] = torch.stack([x.to(torch.float64) for x in rows], 1).cpu()
            st["p_steps"] = torch.stack(p_by_steps, 1).to(torch.float64).cpu()
            st["n_steps"] = torch.stack(n_by_steps, 1).to(torch.float64).cpu()
            st["caps"] = torch.stack(st["exp_capped"], 1).cpu()
            st["uncaps"] = torch.stack(st["exp_uncapped"], 1).cpu()
            st["atts"] = torch.stack(st["att_events"], 1).to(torch.float64).cpu()

    outs = run_batch(es, cfg, ms, pols, streams=streams, observer=observer)
    rows = []
    for b, ((w, i, trial), o) in enumerate(zip(cells, outs)):
        ever, cross, targeted, at_cap, workers = (float(x) for x in st["final"][b].tolist())
        rows.append({"seed": w["seed"], "theta_index": i, "point": list(w["points"][i]), "k": k, "trial": trial,
                     "region": w["region"], "workers": int(workers), "expected": o.result.expected_share,
                     "victory": bool(o.result.victory), "converted_by_round": o.converted,
                     "spreaders_by_round": o.spreaders, "distinct_converts": int(ever), "cross_region": int(cross),
                     "targeted": int(targeted), "at_cap": int(at_cap),
                     "expected_converts_capped": [round(x, 4) for x in st["caps"][b].tolist()],
                     "expected_converts_uncapped": [round(x, 4) for x in st["uncaps"][b].tolist()],
                     "attempt_events": [int(x) for x in st["atts"][b].tolist()],
                     "p_final_sum_by_steps": [round(x, 4) for x in st["p_steps"][b].tolist()],
                     "converts_by_steps": [int(x) for x in st["n_steps"][b].tolist()]})
    return rows


# ============================================================================== summary
def _logistic_crossing(x: list[float], y: list[bool]) -> dict:
    """Fit P(R_obs > 1) = σ(a + b·log x); crossing x* = exp(−a/b) (where the cascade becomes supercritical)."""
    import numpy as np
    from scipy.optimize import minimize
    lx = np.log(np.asarray(x, float))
    yy = np.asarray(y, float)

    def nll(p):
        z = p[0] + p[1] * lx
        return float(np.sum(np.logaddexp(0, z) - yy * z))
    res = minimize(nll, np.array([0.0, 1.0]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
    a, b = res.x
    return {"a": float(a), "slope_b": float(b), "x_star": float(math.exp(-a / b)) if b != 0 else None,
            "converged": bool(res.success), "n": int(len(yy)), "share_supercritical": float(yy.mean())}


ESTIMATORS = ("lambda_ADq", "lambda_ADq_region", "naive_kbar_rr_Eq", "naive_kbar_Eq_region", "naive_lambdaArr_Eq",
              "naive_kbar_Eq_global", "naive_lambdaA_Eq_global")


def summarise(rows: list[dict], worlds: list[dict], th: list[dict], cfg: GameConfig) -> dict:
    import numpy as np
    from scipy.stats import mannwhitneyu, spearmanr
    by_world = {w["seed"]: (w, t) for w, t in zip(worlds, th)}
    base = {(r["seed"], r["theta_index"]): r["expected"] for r in rows if r["k"] == 0}
    points = []                     # one per (world, manifesto point, k): estimators and pooled observations
    groups: dict = {}
    for r in rows:
        if r["k"] == 0:
            continue
        groups.setdefault((r["seed"], r["theta_index"], r["k"]), []).append(r)
    for (seed, i, k), rs in groups.items():
        w, t = by_world[seed]
        est = t["per_theta"][i]
        c = np.array([x["converted_by_round"] for x in rs], float)          # [trials, T]
        num, den = c[:, 1:].sum(), c[:, :-1].sum()
        caps = np.array([x["expected_converts_capped"] for x in rs], float)
        uncaps = np.array([x["expected_converts_uncapped"] for x in rs], float)
        spr = np.array([x["spreaders_by_round"] for x in rs], float)
        dist = sum(x["distinct_converts"] for x in rs)
        tgt = sum(x["targeted"] for x in rs)
        points.append({
            "seed": seed, "theta_index": i, "kind": est["kind"], "theta": est["theta"], "target": est["target"],
            "k": k, **{e_: est[e_] for e_ in ESTIMATORS},
            "R_obs": num / den if den > 0 else None, "R_den": den,
            "R_early": c[:, 1].sum() / c[:, 0].sum() if c[:, 0].sum() > 0 else None, "R_early_den": c[:, 0].sum(),
            "R_hop1_per_worker": float(c[:, 0].mean() / k),
            "R_exp_capped": float(caps[:, 1:].sum() / spr[:, 1:].sum()) if spr[:, 1:].sum() > 0 else None,
            "R_exp_uncapped": float(uncaps[:, 1:].sum() / spr[:, 1:].sum()) if spr[:, 1:].sum() > 0 else None,
            "R_exp_capped_hop1": float(caps[:, 0].sum() / spr[:, 0].sum()),
            "R_exp_uncapped_hop1": float(uncaps[:, 0].sum() / spr[:, 0].sum()),
            "distinct": float(np.mean([x["distinct_converts"] for x in rs])),
            "cross_share": sum(x["cross_region"] for x in rs) / dist if dist else None,
            "cap_bind": sum(x["at_cap"] for x in rs) / tgt if tgt else None,
            "delta_expected": float(np.mean([x["expected"] for x in rs]) - base[(seed, i)]),
            "converts_by_round": c.mean(0).tolist(), "personality": w["personality"]})

    def cell_of(ps, rs, head):
        cell = {**head, "n_worlds": len(ps), "trials": len(rs) // max(1, len(ps))}
        for key in ESTIMATORS + ("R_obs", "R_early", "R_hop1_per_worker", "R_exp_capped", "R_exp_uncapped", "distinct",
                                 "cross_share", "cap_bind", "delta_expected"):
            xs = [p[key] for p in ps if p[key] is not None]
            cell[key] = xc.mean_ci(xs) if len(xs) > 1 else ({"mean": xs[0]} if xs else None)
        cell["converts_by_round_mean"] = np.mean([p["converts_by_round"] for p in ps], 0).tolist()
        cell["win"] = xc.rate([r["victory"] for r in rs])
        psum = np.sum([r["p_final_sum_by_steps"] for r in rs], 0)
        nsum = np.sum([r["converts_by_steps"] for r in rs], 0)
        cell["drift_final_P_by_steps"] = [float(a / b) if b else None for a, b in zip(psum, nsum)]
        cell["drift_converts_by_steps"] = [int(x) for x in nsum]
        return cell

    n_points = len(worlds[0]["points"])
    cells, ext_cells = {}, {}
    for i in range(n_points):
        kind, val = worlds[0]["points"][i]
        for k in KS:
            ps = [p for p in points if p["theta_index"] == i and p["k"] == k]
            rs = [r for r in rows if r["theta_index"] == i and r["k"] == k]
            if kind == "theta":
                cells[f"theta{i}_k{k}"] = cell_of(ps, rs, {"theta": val, "k": k})
            else:
                ext_cells[f"target{val}_k{k}"] = cell_of(ps, rs, {"target_lambda_ADq": val, "k": k})

    # threshold location and calibration per estimator, for two observed growth measures
    def threshold_for(measure: str, den_key: str, min_den: float) -> dict:
        good = [p for p in points if p[measure] is not None and p[den_key] >= min_den]
        out = {}
        for est in ESTIMATORS + ("R_exp_capped", "R_exp_uncapped"):
            sel = [p for p in good if p[est] is not None and p[est] > 0 and p[measure] > 0]
            if len(sel) < 10:
                continue
            xs, ys = [p[est] for p in sel], [p[measure] for p in sel]
            lr = np.log(np.asarray(ys)) - np.log(np.asarray(xs))
            rho = spearmanr(xs, ys)
            out[est] = {"crossing": _logistic_crossing(xs, [y > 1 for y in ys]),
                        "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
                        "log_ratio_obs_over_est": xc.mean_ci(lr.tolist()), "rmse_log": float(np.sqrt(np.mean(lr ** 2))),
                        "n_points": len(xs)}
        for e_, p in zip(list(out), xc.holm([out[e_]["spearman_p"] for e_ in out])):
            out[e_]["spearman_holm_p"] = p
        return {"min_denominator": min_den, "estimators": out,
                "ranking_by_rmse_log": sorted(out, key=lambda e_: out[e_]["rmse_log"]),
                "ranking_by_crossing_distance_from_1": sorted(
                    [e_ for e_ in out if out[e_]["crossing"]["x_star"]],
                    key=lambda e_: abs(math.log(out[e_]["crossing"]["x_star"])))}

    thresh = {"R_obs": threshold_for("R_obs", "R_den", 10), "R_early": threshold_for("R_early", "R_early_den", 10)}
    bins = [0, 0.25, 0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2, 3, 5, 10, 1e9]
    binned = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        ps = [p for p in points if lo <= p["lambda_ADq"] < hi]
        if not ps:
            continue
        b = {"lo": lo, "hi": hi, "n": len(ps)}
        for key in ("R_obs", "R_early", "R_hop1_per_worker", "cap_bind", "distinct", "cross_share"):
            xs = [p[key] for p in ps if p[key] is not None]
            b[key] = xc.mean_ci(xs) if len(xs) > 1 else ({"mean": xs[0]} if xs else None)
        rr = [p["R_obs"] for p in ps if p["R_obs"] is not None and p["R_den"] >= 10]
        b["share_R_obs_gt_1"] = sum(x > 1 for x in rr) / len(rr) if rr else None
        b["n_R_obs"] = len(rr)
        binned.append(b)
    below = [p["cap_bind"] for p in points if p["cap_bind"] is not None and p["lambda_ADq"] < 1]
    above = [p["cap_bind"] for p in points if p["cap_bind"] is not None and p["lambda_ADq"] >= 1]
    cap_test = {"below_threshold": xc.mean_ci(below) if len(below) > 1 else None,
                "above_threshold": xc.mean_ci(above) if len(above) > 1 else None,
                "mannwhitney_p": float(mannwhitneyu(above, below).pvalue) if len(below) > 1 and len(above) > 1 else None,
                "definition": "per (world, point, k) pooled cap-binding share, split at λ_max(A D_q) = 1"}
    pers = {}
    for p in points:
        if p["theta_index"] == len(THETAS) - 1 and p["k"] == 5 and p["R_obs"] is not None:
            pers.setdefault(p["personality"], []).append(p["R_obs"])
    return {"cells": cells, "extension_cells": ext_cells, "threshold": thresh, "binned_by_lambda_ADq": binned,
            "cap_binding_vs_threshold": cap_test,
            "R_obs_by_target_personality_theta1_k5": {k_: xc.mean_ci(v) if len(v) > 1 else {"mean": v[0]}
                                                      for k_, v in sorted(pers.items())},
            "points": points}


# ============================================================================== parity
def parity(seeds: list[int], device: torch.device, base: GameConfig) -> dict:
    def fn(seed, dev):
        cfg = dataclasses.replace(base, device=str(dev))
        w = prepare([seed], cfg, dev)[0]
        r = play([(w, 8, 0)], cfg, 5)[0]
        n = cfg.electorate.num_voters
        return {"expected": r["expected"], "distinct_converts_share_of_n": r["distinct_converts"] / n,
                "cross_region_share_of_n": r["cross_region"] / n,
                "cap_bind_share": r["at_cap"] / r["targeted"] if r["targeted"] else 0.0}
    out = xc.parity_check(fn, seeds, device=device, abs_tol=0.01)
    out["cell"] = "theta = 1 (ideal), k = 5, trial 0"
    out["note"] = ("CPU and GPU draw different random numbers (spec §9): metrics compared as means over the seeds, "
                   "tolerance max(0.01, 3·SE of the paired difference); counts are scaled by N.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X3"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--out", default=str(xc.REPORTS / "x3.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
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
    record: dict = {}
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        worlds = prepare(seeds, base, device)
        t0 = time.perf_counter()
        record["extension"] = extension(worlds, base)
        th, iters = theory(worlds, base, device)
        record["power_iteration"] = {"iterations": iters, "seconds": time.perf_counter() - t0,
                                     "columns_per_world": 2 * len(worlds[0]["points"]) + 2, "worlds": len(worlds)}
        # SciPy reference on 3 worlds (θ = 0.5 weighted operator, global)
        ref = []
        for w, t in list(zip(worlds, th))[:3]:
            lam_ref = scipy_lambda_max(w["e"].graph, w["q"][4])
            ref.append({"seed": w["seed"], "gpu": t["per_theta"][4]["lambda_ADq"], "scipy": lam_ref,
                        "rel_err": abs(t["per_theta"][4]["lambda_ADq"] - lam_ref) / lam_ref})
        for w, t in list(zip(worlds, th))[:3]:   # and at the extension point λ ≈ 1 (index 12)
            j = len(THETAS) + list(EXT_TARGETS).index(1.0)
            lam_ref = scipy_lambda_max(w["e"].graph, w["q"][j])
            ref.append({"seed": w["seed"], "point": "ext 1.0", "gpu": t["per_theta"][j]["lambda_ADq"], "scipy": lam_ref,
                        "rel_err": abs(t["per_theta"][j]["lambda_ADq"] - lam_ref) / lam_ref})
        record["power_log"] = POWER_LOG
        record["power_vs_scipy"] = {"rows": ref, "max_rel_err": max(r["rel_err"] for r in ref),
                                    "passed": max(r["rel_err"] for r in ref) < 1e-5}
        print(f"[x3] theory done ({iters} power iterations, {time.perf_counter() - t0:.1f}s); "
              f"GPU vs SciPy max rel err {record['power_vs_scipy']['max_rel_err']:.2e}", flush=True)
        trials = list(range(args.trials))
        cal = xc.calibrate_batch(lambda B: play([(worlds[j % len(worlds)], 8, 0) for j in range(B)], base, 10),
                                 device, sizes=(20, 40), max_batch=len(worlds) * len(trials))
        record["calibration"] = cal
        print(f"[x3] batch {cal['batch']}", flush=True)
        rows: list[dict] = []
        for i in range(len(worlds[0]["points"])):
            t0 = time.perf_counter()
            jobs = [("k0", 0, [(w, i, 0) for w in worlds])] + \
                   [(f"k{k}", k, [(w, i, t) for t in trials for w in worlds]) for k in KS]
            for _, k, cells in jobs:
                out = xc.run_chunks(lambda ch: play(ch, base, k), cells, cal["batch"], record=record)
                for r in out:
                    xc.write_row(RAW_PATH, {"exp": "X3", **r})
                rows += out
            last = [r for r in rows if r["theta_index"] == i and r["k"] == 5]
            lam = sum(t["per_theta"][i]["lambda_ADq"] for t in th) / len(th)
            print(f"[x3] point {worlds[0]['points'][i]}: {time.perf_counter() - t0:.1f}s; mean λ(A D_q) {lam:.2f}; "
                  f"k=5 mean distinct converts {sum(r['distinct_converts'] for r in last) / len(last):.0f}", flush=True)
        summary = summarise(rows, worlds, th, base)
        par = parity(seeds[:args.parity_seeds], device, base)
        print(f"[x3] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    world_meta = [{"seed": w["seed"], "region": w["region"], "sizes": w["sizes"], "personality": w["personality"],
                   "far_archetype": w["far_archetype"], "tv_far_ideal": w["tv_far_ideal"], "phis": w["phis"],
                   "vertex_sector": int(w["vertex"].argmax()),
                   "manifestos": [[round(float(x), 5) for x in b] for b in w["manifestos"]], **t}
                  for w, t in zip(worlds, th)]
    report = {"experiment": "X3", "question": "Worker cascade threshold and reach; weighted operator λ_max(A D_q)",
              "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
              "n": args.n, "rounds": base.rounds, "treasury": str(base.budget), "trials": args.trials,
              "thetas": THETAS, "ks": list(KS), "extension_targets": list(EXT_TARGETS),
              "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
              "metric_definitions": {
                  "q": "q_j = γ_j·P_j(b) on v(0); 0 for zealots",
                  "lambda_ADq": "λ_max(A D_q) = λ_max(D_q^½ A D_q^½), whole graph (GPU power iteration)",
                  "lambda_ADq_region": "same with q restricted to the target region (= λ_max(A_rr D_q,r))",
                  "naive_kbar_rr_Eq": "k̄_rr (mean within-region degree in r) · E_r[q]",
                  "naive_kbar_Eq_region": "k̄ (mean total degree) · E_r[q]",
                  "naive_lambdaArr_Eq": "λ_max(A_rr) · E_r[q]",
                  "naive_kbar_Eq_global": "k̄ · E[q] (all voters)", "naive_lambdaA_Eq_global": "λ_max(A) · E[q]",
                  "R_obs": "Σ_{t≥2} converts_t / Σ_{t≤T−1} converts_t, pooled over the 10 trials (convert → convert "
                           "branching; each voter spreads for one hop); points with < 20 convert-spreaders dropped",
                  "R_hop1_per_worker": "converts in hop 1 / k (worker → convert)",
                  "R_early": "converts in hop 2 / converts in hop 1, pooled over the trials (first convert generation)",
                  "R_exp_capped": "Σ_{t≥2} Σ_j [1 − (1 − q_j)^{eff_j}] / Σ_{t≥2} spreaders_t, eff_j = min(count_j, cap − "
                                  "attempts_j), q on v at the start of the round (cap-corrected expected branching)",
                  "R_exp_uncapped": "same with eff_j = count_j (no cap)",
                  "cap_bind": "voters with final attempts = cap / voters with attempts > 0, pooled over trials",
                  "cross_share": "distinct converts outside the target region / all distinct converts",
                  "drift_final_P_by_steps": "mean final P_j of converted non-worker voters, by social steps since their "
                                            "last conversion (1 = converted in round T); P right after conversion "
                                            "= P(b, b) = 0.971",
                  "delta_expected": "expected share with k workers − with 0 workers, same world and θ",
                  "crossing": "logistic fit P(R_obs > 1) = σ(a + b log x); x* = exp(−a/b)"},
              **{k: v for k, v in summary.items() if k != "points"},
              "worlds": world_meta,
              "deviations": [
                  "Addition to the plan's 9-point sweep: 8 extension points per world on the path b(φ) = (1 − φ)·vertex + "
                  "φ·b_far (vertex = the simplex vertex of the least-liked sector), chosen so that λ_max(A D_q) ≈ "
                  "0.5 … 2.0. The pilot showed the plan sweep never goes below λ_max(A D_q) ≈ 3 (b_far is another "
                  "region's archetype, liked there), so without them the threshold λ ≈ 1 is not sampled.",
                  "Target region = the largest region of each world (the plan says 'one region').",
                  "Observed growth: R_obs pools hops 2..T (convert → convert); R_early = converts(hop 2)/converts(hop 1) "
                  "is reported next to it because depletion and the cap lower later hops."],
              "config": base.to_record(),
              "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=record.get("batch_sizes"),
                                             free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                             extra={k: record.get(k) for k in ("calibration", "power_iteration",
                                                                               "power_vs_scipy", "power_log",
                                                                               "extension")})}
    xc.save(args.out, report)
    for p in summary["points"]:
        xc.write_row(RAW_PATH, {"exp": "X3", "kind": "point", **p})
    print(f"[x3] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
