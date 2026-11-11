"""R1 — round-0 manifesto atlas: what does the ideal manifesto look like on the INITIAL electorate?

Command (reproduces everything under reports/v3/experimentation/ for R1):
  OMP_NUM_THREADS=4 python experiments/r1_manifesto_round0.py --seeds 100 --first-seed 12000 --device cuda:2
  python experiments/r1_manifesto_round0.py --plot-only          # redraw the figures from r1.json

Everything is scored on v(0) = electorate.preferences only: no campaign, no social drift. Default GameConfig
(N = 10,000, β_v = 20, z = 0.825, zealots 5%; zealots vote like everyone else). Seeds 12000–12099 are the X19 worlds.

Regions are named by SIZE RANK per seed (R1 = largest … R4 = smallest), because the compass letters are
arbitrary across seeds. Manifestos scored per seed:
  * single  — each region's mean preference (4);
  * group   — every 2- and 3-region size-weighted blend of regional means (6 + 4), flagged when they hold > 50%;
  * average — population mean (= 4-region blend), per-sector median projected to Δ, L1/TV geometric median
              (x19_manifesto.geometric_median_tv), uniform 0.2, top-3 statistic (oracle.top3_manifesto);
  * optimum — b*_0: projected gradient ascent on v(0) (same update as manifesto_opt.optimise_manifesto: step 0.02,
              pseudo-Huber δ = 1e-3, ≤ 400 iterations) from 27 starts = the library's 7 default starts + top-3 +
              the 19 variants above, batched over seeds × starts; b*_0 = best EXACT E over every iterate;
  * reference — the ORACLE election-day manifesto (oracle_manifesto: optimised on the 5-round social forecast)
              scored at round 0.
Metrics: E = mean_i σ(β(S_i − z)); share won over (S_i ≥ z); mean P per region; regions carried (regional mean
P > 0.5) and the voter share they hold; TV to b*_0; 40-bin agreement histogram over S ∈ [0, 1].
Scenario split: TV between the archetype means of the two largest regions ≥ 0.35 → "divided", else "aligned".
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x19_manifesto import cluster, geometric_median_tv  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import REGIONS, SECTORS, archetype_means  # noqa: E402
from dumbocracy.strategies.manifesto_opt import default_starts, optimise_manifesto, project_simplex  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto, top3_manifesto  # noqa: E402

OUT = xc.REPORTS / "experimentation"
JSON_PATH = OUT / "r1.json"
RAW_PATH = OUT / "raw" / "r1.jsonl"
FIG = OUT / "figures"
LOG_PATH = OUT / "logs" / "r1.log"

BINS = 40
STEP = 0.02          # library default PG step
ITERS = 400          # library default
SMOOTH = 1e-3        # library default pseudo-Huber δ
MOVE_TOL = 1e-7      # library default stop rule
TOL = 1e-4           # "reached the optimum" for convergence counts
DIVIDED_TV = 0.35    # X15 threshold
SHOW_SEED = 12000
GRID_RES = 100       # ternary grid: i + j + k = GRID_RES
N_STANDARD = 8       # 7 library default starts + top-3

AVERAGES = ("mean", "median", "geomedian", "uniform", "top3")
LABELS = {"mean": "Population mean", "median": "Per-sector median (proj.)", "geomedian": "TV geometric median",
          "uniform": "Uniform 0.2", "top3": "Top-3 statistic", "bstar0": "b*0 (round-0 optimum)",
          "oracle": "ORACLE election-day, at round 0"}

_LOG = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


# ============================================================================== helpers (unit-tested)
def label_of(name: str) -> str:
    if name in LABELS:
        return LABELS[name]
    kind, ranks = name.split("_", 1)
    return ("Region " if kind == "single" else "") + "+".join("R" + c for c in ranks.split("R")[1:])


def size_order(sizes: list[int]) -> list[int]:
    """Region indices sorted by size, largest first (ties by index)."""
    return sorted(range(len(sizes)), key=lambda r: (-sizes[r], r))


def regional_means(v: torch.Tensor, region_of: torch.Tensor, num_regions: int = 4) -> torch.Tensor:
    return torch.stack([v[region_of == r].mean(0) for r in range(num_regions)])


def blend(means: torch.Tensor, sizes: torch.Tensor, members: list[int]) -> torch.Tensor:
    """Size-weighted blend of the regional means of ``members`` (a point of Δ when each mean is)."""
    w = sizes[members].to(means.dtype)
    return (w.unsqueeze(-1) * means[members]).sum(0) / w.sum()


def scenario_manifestos(v: torch.Tensor, region_of: torch.Tensor, sizes: list[int]) -> tuple[list[str], torch.Tensor, dict]:
    """The 19 variant manifestos of one seed, in a fixed name order: 4 single, 6 pairs, 4 triples, 5 averages.
    Names use size ranks (``single_R1``, ``pair_R1R3``, ``triple_R1R2R4``)."""
    order = size_order(sizes)
    means = regional_means(v, region_of, len(sizes))
    sz = torch.tensor(sizes, dtype=v.dtype, device=v.device)
    total = sum(sizes)
    names, bs, meta = [], [], {}
    for k, kind in ((1, "single"), (2, "pair"), (3, "triple")):
        for comb in itertools.combinations(range(len(sizes)), k):
            regions = [order[c] for c in comb]
            name = kind + "_" + "".join(f"R{c + 1}" for c in comb)
            names.append(name)
            bs.append(blend(means, sz, regions))
            share = sum(sizes[r] for r in regions) / total
            meta[name] = {"kind": "single" if k == 1 else "group", "ranks": [c + 1 for c in comb],
                          "regions": [REGIONS[r] for r in regions], "voter_share": share, "majority": share > 0.5}
    gm, _ = geometric_median_tv(v)
    avg = {"mean": v.mean(0), "median": project_simplex(v.median(0).values.unsqueeze(0))[0], "geomedian": gm,
           "uniform": torch.full((v.shape[-1],), 1.0 / v.shape[-1], dtype=v.dtype, device=v.device),
           "top3": top3_manifesto(v).to(v)}
    for name in AVERAGES:
        names.append(name)
        bs.append(avg[name])
        meta[name] = {"kind": "average"}
    return names, torch.stack(bs), meta


def exact_share(v: torch.Tensor, b: torch.Tensor, beta: float, z: float) -> torch.Tensor:
    """E for voters ``v`` [B, N, 5] and manifestos ``b`` [B, M, 5] → [B, M] (= manifesto_opt.expected_share)."""
    sim = 1.0 - 0.5 * (v.unsqueeze(1) - b.unsqueeze(2)).abs().sum(-1)
    return torch.sigmoid(beta * (sim - z)).mean(-1)


def smoothed_share(v: torch.Tensor, b: torch.Tensor, beta: float, z: float, delta: float = SMOOTH) -> torch.Tensor:
    d = v.unsqueeze(1) - b.unsqueeze(2)
    sim = 1.0 - 0.5 * (torch.sqrt(d * d + delta * delta) - delta).sum(-1)
    return torch.sigmoid(beta * (sim - z)).mean(-1)


def score_batch(v: torch.Tensor, region_of: torch.Tensor, b: torch.Tensor, beta: float, z: float,
                bins: int = BINS, num_regions: int = 4) -> dict:
    """All metrics of manifestos ``b`` [B, M, 5] on voters ``v`` [B, N, 5] in one tensor op.

    Returns E [B, M], won (share with S ≥ z) [B, M], region_P [B, M, R] and hist [B, M, bins] (counts over S ∈
    [0, 1], S = 1 in the last bin; each row sums to N)."""
    sim = 1.0 - 0.5 * (v.unsqueeze(1) - b.unsqueeze(2)).abs().sum(-1)                    # [B, M, N]
    prob = torch.sigmoid(beta * (sim - z))
    onehot = F.one_hot(region_of, num_regions).to(prob.dtype)                          # [B, N, R]
    region_p = torch.einsum("bmn,bnr->bmr", prob, onehot) / onehot.sum(1).unsqueeze(1)
    idx = (sim * bins).floor().long().clamp(0, bins - 1)
    hist = torch.zeros(*sim.shape[:2], bins, dtype=torch.long, device=sim.device).scatter_add_(-1, idx, torch.ones_like(idx))
    return {"E": prob.mean(-1), "won": (sim >= z).to(prob.dtype).mean(-1), "region_P": region_p, "hist": hist}


@torch.no_grad()
def _exact_ng(v, b, beta, z):
    return exact_share(v, b, beta, z)


def pg_ascent(v: torch.Tensor, starts: torch.Tensor, beta: float, z: float, *, step: float = STEP,
              iterations: int = ITERS, smoothing: float = SMOOTH, tol: float = MOVE_TOL) -> tuple[torch.Tensor, torch.Tensor]:
    """Projected gradient ascent batched over seeds and starts (``v`` [B, N, 5], ``starts`` [B, M, 5]); same update
    and stop rule as ``optimise_manifesto``. Returns the iterate path [T+1, B, M, 5] and its exact E [T+1, B, M]."""
    b = project_simplex(starts.to(v)).clone()
    path, trace = [b], [_exact_ng(v, b, beta, z)]
    for _ in range(iterations):
        bb = b.detach().requires_grad_(True)
        (g,) = torch.autograd.grad(smoothed_share(v, bb, beta, z, smoothing).sum(), bb)
        with torch.no_grad():
            new = project_simplex(b + step * g)
        moved = float((new - b).abs().max())
        b = new
        path.append(b)
        trace.append(_exact_ng(v, b, beta, z))
        if moved < tol:
            break
    return torch.stack(path), torch.stack(trace)


def best_iterate(path: torch.Tensor, trace: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per seed, the iterate with the highest exact E over all iterations and starts: (b [B,5], E [B], t [B], m [B])."""
    T1, B, M = trace.shape
    flat = trace.permute(1, 0, 2).reshape(B, T1 * M)
    E, idx = flat.max(1)
    t, m = idx // M, idx % M
    return path[t, torch.arange(B, device=path.device), m], E, t, m


def ternary_points(res: int) -> torch.Tensor:
    """All barycentric points (i, j, k)/res with i + j + k = res, as [P, 3] float64."""
    pts = [(i, j, res - i - j) for i in range(res + 1) for j in range(res + 1 - i)]
    return torch.tensor(pts, dtype=torch.float64) / res


def tv_two_largest(sizes: list[int], sigma: list[int]) -> float:
    means = archetype_means(dtype=torch.float64)[torch.tensor(sigma)]
    a, b = size_order(sizes)[:2]
    return float(0.5 * (means[a] - means[b]).abs().sum())


# ============================================================================== one batch of seeds
def prepare(seed: int, n: int, device: torch.device) -> dict:
    cfg = GameConfig(seed=seed, device=str(device), electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=device)
    v0 = e.preferences
    beta, z = cfg.voting.sensitivity, cfg.voting.threshold
    sizes = [int(x) for x in e.sizes.tolist()]
    sigma = [int(x) for x in e.archetype_of_region.tolist()]
    lib = optimise_manifesto(v0, beta, z)
    item = {"seed": seed, "v": v0, "region_of": e.region_of, "sizes": sizes, "sigma": sigma,
            "tau": [int(x) for x in e.personality_of_region.tolist()],
            "oracle": oracle_manifesto(e, cfg).to(v0), "E_library": lib.share, "beta": beta, "z": z,
            "tv_two_largest": tv_two_largest(sizes, sigma)}
    return item


def landscape(v: torch.Tensor, bstar: torch.Tensor, means_ranked: torch.Tensor, beta: float, z: float,
              res: int = GRID_RES, chunk: int = 512) -> dict:
    """E over the 3-sector slice through b*_0 (the 3 sectors of largest variance across regional means move; the
    other 2 are fixed at their b*_0 values)."""
    var = means_ranked.var(0)
    s3 = sorted(torch.topk(var, 3).indices.tolist())
    rest = [s for s in range(5) if s not in s3]
    mass = float(bstar[s3].sum())
    bary = ternary_points(res).to(v)
    pts = torch.zeros(bary.shape[0], 5, dtype=v.dtype, device=v.device)
    pts[:, s3] = mass * bary
    pts[:, rest] = bstar[rest]
    E = torch.cat([exact_share(v.unsqueeze(0), pts[i:i + chunk].unsqueeze(0), beta, z)[0]
                   for i in range(0, pts.shape[0], chunk)])
    best = int(E.argmax())
    return {"sectors": s3, "sector_names": [SECTORS[s] for s in s3], "fixed_sectors": [SECTORS[s] for s in rest],
            "fixed_values": [float(bstar[s]) for s in rest], "free_mass": mass, "res": res,
            "bary": [[round(float(x), 4) for x in p] for p in bary.tolist()],
            "E": [round(float(x), 6) for x in E.tolist()], "grid_max_E": float(E[best]),
            "grid_argmax_bary": [float(x) for x in bary[best].tolist()],
            "variance_by_sector": [float(x) for x in var.tolist()]}


def to_bary(x: torch.Tensor, s3: list[int]) -> list[float]:
    p = x[s3].clamp_min(0)
    return [float(t) for t in (p / p.sum()).tolist()]


def process_batch(items: list[dict], show_seed: int | None = SHOW_SEED) -> tuple[list[dict], dict | None]:
    beta, z = items[0]["beta"], items[0]["z"]
    v = torch.stack([it["v"] for it in items])
    reg = torch.stack([it["region_of"] for it in items])
    B = len(items)
    scen = [scenario_manifestos(it["v"], it["region_of"], it["sizes"]) for it in items]
    names, meta = scen[0][0], scen[0][2]
    variants = torch.stack([s[1] for s in scen])                                           # [B, 19, 5]
    standard = torch.stack([project_simplex(torch.cat([default_starts(it["v"]), top3_manifesto(it["v"]).unsqueeze(0).to(it["v"])]))
                            for it in items])                                                # [B, 8, 5]
    start_names = ["uniform*", "mean*"] + [f"vertex_{s}*" for s in SECTORS] + ["top3*"] + names
    starts = torch.cat([standard, variants], 1)
    path, trace = pg_ascent(v, starts, beta, z)
    bstar, Estar, tbest, mbest = best_iterate(path, trace)
    E_std = trace[:, :, :N_STANDARD].amax((0, 2))
    all_names = names + ["bstar0", "oracle"]
    allb = torch.cat([variants, bstar.unsqueeze(1), torch.stack([it["oracle"] for it in items]).unsqueeze(1)], 1)
    sc = score_batch(v, reg, allb, beta, z)
    tv_star = 0.5 * (allb - bstar.unsqueeze(1)).abs().sum(-1)
    incumbent = trace.amax(2).cummax(0).values                                              # [T+1, B]
    rows, show = [], None
    for i, it in enumerate(items):
        order = size_order(it["sizes"])
        total = sum(it["sizes"])
        man = {}
        for j, name in enumerate(all_names):
            sup = sc["region_P"][i, j]
            sup_rank = [float(sup[r]) for r in order]
            carried = [k + 1 for k in range(4) if sup_rank[k] > 0.5]
            man[name] = {"b": [round(float(x), 6) for x in allb[i, j].tolist()], "E": float(sc["E"][i, j]),
                         "won_over": float(sc["won"][i, j]), "support_by_rank": sup_rank, "carried_ranks": carried,
                         "carried_count": len(carried),
                         "carried_share": sum(it["sizes"][order[k - 1]] for k in carried) / total,
                         "tv_to_bstar0": float(tv_star[i, j]), "hist": [int(x) for x in sc["hist"][i, j].tolist()]}
            if name in meta and meta[name]["kind"] != "average":
                man[name].update({k: meta_v for k, meta_v in scen[i][2][name].items() if k != "kind"})
        hit = torch.nonzero(incumbent[:, i] >= Estar[i] - TOL).flatten()
        final = path[-1, i]
        cl = cluster(final, trace[-1, i])
        row = {"exp": "R1", "seed": it["seed"], "sizes": it["sizes"], "size_order": [REGIONS[r] for r in order],
               "sigma": it["sigma"], "tau": it["tau"], "tv_two_largest": it["tv_two_largest"],
               "scenario": "divided" if it["tv_two_largest"] >= DIVIDED_TV else "aligned",
               "manifestos": man,
               "optimum": {"E_star": float(Estar[i]), "E_library_default": it["E_library"],
                           "E_standard_starts_only": float(E_std[i]), "best_start": start_names[int(mbest[i])],
                           "best_iteration": int(tbest[i]), "iterations_run": trace.shape[0] - 1,
                           "iterations_to_tol": int(hit[0]) if hit.numel() else None,
                           "distinct_endpoints": len(cl), "endpoint_E": [round(c["E"], 6) for c in cl],
                           "incumbent_trace": [round(float(x), 6) for x in incumbent[:, i].tolist()]}}
        rows.append(row)
        if show_seed is not None and it["seed"] == show_seed:
            means_ranked = regional_means(it["v"], it["region_of"])[order]
            land = landscape(it["v"], bstar[i], means_ranked, beta, z)
            s3 = land["sectors"]
            T = int(trace.shape[0])
            mb = int(mbest[i])
            show = {"seed": it["seed"], "start_names": start_names,
                    "per_start_traces": [[round(float(x), 6) for x in trace[:, i, m].tolist()] for m in range(starts.shape[1])],
                    "paths": [[[round(float(x), 5) for x in path[t, i, m].tolist()] for t in range(T)]
                              for m in range(starts.shape[1])],
                    "best_start_index": mb, "bstar0": [float(x) for x in bstar[i].tolist()],
                    "regional_means_ranked": [[float(x) for x in r.tolist()] for r in means_ranked],
                    "landscape": land, "bstar0_bary": to_bary(bstar[i], s3),
                    "regional_means_bary": [to_bary(r, s3) for r in means_ranked],
                    "best_path_bary": [to_bary(path[t, i, mb], s3) for t in range(T)],
                    "bstar0_E": float(Estar[i])}
    return rows, show


# ============================================================================== summary statistics
def summarise(rows: list[dict]) -> dict:
    from scipy.stats import mannwhitneyu
    names = list(rows[0]["manifestos"].keys())
    kinds = {}
    for n in names:
        kinds[n] = ("single" if n.startswith("single") else "group" if n.startswith(("pair", "triple"))
                    else "optimum" if n == "bstar0" else "reference" if n == "oracle" else "average")

    def stats_for(sub: list[dict]) -> dict:
        star = [r["manifestos"]["bstar0"]["E"] for r in sub]
        out, pvals, keys = {}, [], []
        for n in names:
            ms = [r["manifestos"][n] for r in sub]
            E = [m["E"] for m in ms]
            d = {"kind": kinds[n], "label": label_of(n), "E": xc.mean_ci(E),
                 "won_over": xc.mean_ci([m["won_over"] for m in ms]),
                 "carried_count": xc.mean_ci([m["carried_count"] for m in ms]),
                 "carried_share": xc.mean_ci([m["carried_share"] for m in ms]),
                 "tv_to_bstar0": xc.mean_ci([m["tv_to_bstar0"] for m in ms]),
                 "win_alone_rate": xc.rate([e > 0.5 for e in E]),
                 "carries_majority_of_voters_rate": xc.rate([m["carried_share"] > 0.5 for m in ms]),
                 "support_by_rank": [xc.mean_ci([m["support_by_rank"][k] for m in ms]) for k in range(4)]}
            if "majority" in ms[0]:
                d["coalition_holds_majority_rate"] = xc.rate([m["majority"] for m in ms])
                d["coalition_voter_share"] = xc.mean_ci([m["voter_share"] for m in ms])
            if n != "bstar0":
                d["gap_vs_bstar0"] = xc.paired(E, star)          # E_m − E_b*0 (negative = worse)
                pvals.append(d["gap_vs_bstar0"]["wilcoxon_p"])
                keys.append(n)
            out[n] = d
        for k, p in zip(keys, xc.holm(pvals)):
            out[k]["gap_vs_bstar0"]["holm_p"] = p
        # per-seed best of a family
        derived = {}
        fam = {"best_single": [n for n in names if kinds[n] == "single"],
               "best_group": [n for n in names if kinds[n] == "group"],
               "best_average": [n for n in names if kinds[n] == "average"]}
        for key, members in fam.items():
            vals = [max(r["manifestos"][n]["E"] for n in members) for r in sub]
            arg = [max(members, key=lambda n: r["manifestos"][n]["E"]) for r in sub]
            derived[key] = {"E": xc.mean_ci(vals), "gap_vs_bstar0": xc.paired(vals, star),
                            "which": {n: arg.count(n) for n in members if arg.count(n)}}
        maj = [max([r["manifestos"][n]["E"] for n in fam["best_group"] if r["manifestos"][n]["majority"]] or [float("nan")])
               for r in sub]
        ok = [(a, s) for a, s in zip(maj, star) if not math.isnan(a)]
        derived["best_majority_group"] = {"E": xc.mean_ci([a for a, _ in ok]), "n": len(ok),
                                          "gap_vs_bstar0": xc.paired([a for a, _ in ok], [s for _, s in ok])}
        hp = xc.holm([derived[k]["gap_vs_bstar0"]["wilcoxon_p"] for k in derived])
        for k, p in zip(derived, hp):
            derived[k]["gap_vs_bstar0"]["holm_p"] = p
        return {"n": len(sub), "manifestos": out, "derived": derived}

    full = stats_for(rows)
    hist = {n: [sum(r["manifestos"][n]["hist"][k] for r in rows) for k in range(BINS)] for n in names}
    split = {}
    for g in ("divided", "aligned"):
        sub = [r for r in rows if r["scenario"] == g]
        split[g] = stats_for(sub) if len(sub) >= 3 else {"n": len(sub)}
    # does the gap to b*0 differ between the two kinds of world? (Mann–Whitney, Holm over manifestos)
    between, pv = {}, []
    div = [r for r in rows if r["scenario"] == "divided"]
    ali = [r for r in rows if r["scenario"] == "aligned"]
    if len(div) >= 3 and len(ali) >= 3:
        for n in names:
            if n == "bstar0":
                continue
            a = [r["manifestos"][n]["E"] - r["manifestos"]["bstar0"]["E"] for r in div]
            b = [r["manifestos"][n]["E"] - r["manifestos"]["bstar0"]["E"] for r in ali]
            p = float(mannwhitneyu(a, b).pvalue)
            between[n] = {"gap_divided": sum(a) / len(a), "gap_aligned": sum(b) / len(b), "mannwhitney_p": p}
            pv.append(p)
        for n, p in zip(between, xc.holm(pv)):
            between[n]["holm_p"] = p
    split["E_star_divided_vs_aligned"] = {
        "divided": xc.mean_ci([r["optimum"]["E_star"] for r in div]),
        "aligned": xc.mean_ci([r["optimum"]["E_star"] for r in ali]),
        "mannwhitney_p": float(mannwhitneyu([r["optimum"]["E_star"] for r in div],
                                            [r["optimum"]["E_star"] for r in ali]).pvalue) if div and ali else None}
    split["gap_between_scenarios"] = between
    opt = [r["optimum"] for r in rows]
    its = sorted(o["iterations_to_tol"] for o in opt if o["iterations_to_tol"] is not None)

    def q(p):
        return its[min(len(its) - 1, int(p * (len(its) - 1) + 0.5))] if its else None

    T = max(len(o["incumbent_trace"]) for o in opt)
    gap_curve = []
    for t in range(T):
        g = [o["E_star"] - o["incumbent_trace"][min(t, len(o["incumbent_trace"]) - 1)] for o in opt]
        g.sort()
        gap_curve.append({"median": g[len(g) // 2], "p90": g[int(0.9 * (len(g) - 1))], "max": g[-1]})
    optimum = {
        "E_star": xc.mean_ci([o["E_star"] for o in opt]),
        "win_alone_rate": xc.rate([o["E_star"] > 0.5 for o in opt]),
        "iterations_to_tol": {"median": q(0.5), "q25": q(0.25), "q75": q(0.75), "max": its[-1] if its else None,
                              "reached": len(its), "n": len(opt)},
        "distinct_endpoints": xc.mean_ci([o["distinct_endpoints"] for o in opt]),
        "seeds_with_more_than_one_endpoint": xc.rate([o["distinct_endpoints"] > 1 for o in opt]),
        "variant_starts_gain": xc.mean_ci([o["E_star"] - o["E_standard_starts_only"] for o in opt]),
        "variant_starts_gain_gt_tol_rate": xc.rate([o["E_star"] - o["E_standard_starts_only"] > TOL for o in opt]),
        "library_check": {"max_E_library_minus_bstar0": max(o["E_library_default"] - o["E_star"] for o in opt),
                          "mean_E_star_minus_library": sum(o["E_star"] - o["E_library_default"] for o in opt) / len(opt)},
        "best_start_counts": {s: sum(1 for o in opt if o["best_start"] == s) for s in sorted({o["best_start"] for o in opt})},
        "incumbent_gap_curve": gap_curve,
    }
    return {"overall": full, "histograms": {"bins": BINS, "edges": [k / BINS for k in range(BINS + 1)],
                                            "counts": hist}, "scenario_split": split, "optimum": optimum,
            "scenario_counts": {"divided": len(div), "aligned": len(ali)}}


# ============================================================================== CPU↔GPU parity
def parity(seeds: list[int], device: torch.device, n: int) -> dict:
    def fn(seed, dev):
        rows, _ = process_batch([prepare(seed, n, dev)], show_seed=None)
        m = rows[0]["manifestos"]
        return {"E_bstar0": m["bstar0"]["E"], "E_mean": m["mean"]["E"], "E_single_R1": m["single_R1"]["E"],
                "E_oracle": m["oracle"]["E"], "won_bstar0": m["bstar0"]["won_over"],
                "carried_bstar0": m["bstar0"]["carried_count"], "tv_mean_bstar0": m["mean"]["tv_to_bstar0"]}
    return xc.parity_check(fn, seeds, device=device, abs_tol=0.002)


# ============================================================================== figures (from the saved JSON only)
PAL = {"single": "#2a78d6", "group": "#eb6834", "average": "#1baf7a", "optimum": "#eda100", "reference": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KIND_ORDER = ("single", "group", "average", "optimum", "reference")
KIND_LABEL = {"single": "Single region", "group": "Group of regions", "average": "Average",
              "optimum": "Round-0 optimum", "reference": "ORACLE reference"}


def _style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "legend.frameon": False})


def _ordered(man: dict) -> list[str]:
    out = []
    for k in KIND_ORDER:
        out += [n for n, d in man.items() if d["kind"] == k]
    return out


def plot_all(rep: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import numpy as np
    from matplotlib.patches import Patch
    _style(plt)
    FIG.mkdir(parents=True, exist_ok=True)
    man = rep["overall"]["manifestos"]
    order = _ordered(man)
    saved = []

    def save(fig, name):
        p = FIG / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p.relative_to(xc.ROOT)))

    # (a) mean E per manifesto
    fig, ax = plt.subplots(figsize=(7.5, 7))
    y = np.arange(len(order))[::-1]
    for yi, n in zip(y, order):
        d = man[n]["E"]
        ax.barh(yi, d["mean"], height=0.7, color=PAL[man[n]["kind"]])
        ax.errorbar(d["mean"], yi, xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]], color=INK, lw=1, capsize=2)
        ax.text(d["ci95"][1] + 0.005, yi, f"{d['mean']:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [man[n]["label"] for n in order])
    ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Expected vote share E at round 0 (mean over seeds, 95% t-CI)")
    ax.set_title("R1: which manifesto wins round 0?", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=PAL[k], label=KIND_LABEL[k]) for k in KIND_ORDER], loc="upper center",
              bbox_to_anchor=(0.4, -0.07), ncol=3, fontsize=8)
    save(fig, "r1_share_by_manifesto.png")

    # (b) heat map manifesto × region (size rank) of mean support
    M = np.array([[man[n]["support_by_rank"][k]["mean"] for k in range(4)] for n in order])
    fig, ax = plt.subplots(figsize=(5.6, 7.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for k in range(4):
            ax.text(k, i, f"{M[i, k]:.2f}", ha="center", va="center", fontsize=7,
                    color="#fcfcfb" if M[i, k] > 0.55 else INK)
    ax.set_xticks(range(4), ["R1\n(largest)", "R2", "R3", "R4\n(smallest)"])
    ax.set_yticks(range(len(order)), [man[n]["label"] for n in order])
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Mean vote probability in region")
    ax.set_title("Support by region (regions ranked by size per seed)", loc="left", color=INK)
    save(fig, "r1_region_support_heatmap.png")

    # (c) agreement histograms
    hist = rep["histograms"]
    best_group = max((n for n in order if man[n]["kind"] == "group"), key=lambda n: man[n]["E"]["mean"])
    show = [("single_R1", PAL["single"]), (best_group, PAL["group"]), ("mean", PAL["average"]), ("bstar0", PAL["optimum"])]
    edges = np.array(hist["edges"])
    centres = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True, sharey=True)
    z = rep["config"]["voting"]["threshold"]
    for ax, (n, col) in zip(axes.ravel(), show):
        c = np.array(hist["counts"][n], dtype=float)
        c /= c.sum()
        ax.bar(centres, c, width=1 / hist["bins"] * 0.9, color=col)
        ax.axvline(z, color=INK, lw=0.9, ls="--")
        ax.set_title(f"{man[n]['label']}  (won over {man[n]['won_over']['mean']:.1%})", loc="left", fontsize=8.5, color=INK)
        ax.set_xlim(0.4, 1.0)
    for ax in axes[1]:
        ax.set_xlabel("Agreement S = 1 − TV(v, b)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of voters")
    fig.suptitle(f"Agreement with the manifesto, all seeds pooled (dashed: z = {z})", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    save(fig, "r1_agreement_histograms.png")

    # (d) convergence + path in a 2-D projection
    sh = rep["showcase"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.2))
    names = sh["start_names"]
    estar = sh["bstar0_E"]
    for m, tr in enumerate(sh["per_start_traces"]):
        is_best = m == sh["best_start_index"]
        a1.plot(range(len(tr)), tr, lw=2 if is_best else 0.8, color=PAL["optimum"] if is_best else "#9ec5f4",
                alpha=1 if is_best else 0.8, zorder=3 if is_best else 1)
    a1.axhline(estar, color=INK2, lw=0.8, ls="--")
    a1.set_xscale("symlog", linthresh=10)
    a1.set_xlim(left=0)
    a1.set_xlabel("Iteration")
    a1.set_ylabel("Exact E")
    a1.set_title(f"Seed {sh['seed']}: E per start ({len(names)} starts; bold = best)", loc="left", fontsize=8.5, color=INK)
    gc = rep["optimum"]["incumbent_gap_curve"]
    xs = np.arange(len(gc))
    for key, col, lab in (("median", PAL["single"], "median"), ("p90", PAL["group"], "90th pct"), ("max", PAL["reference"], "max")):
        a2.plot(xs, [max(g[key], 1e-7) for g in gc], color=col, lw=1.6, label=lab)
    a2.axhline(TOL, color=INK2, lw=0.8, ls="--")
    a2.set_yscale("log")
    a2.set_xscale("symlog", linthresh=10)
    a2.set_xlim(left=0)
    a2.set_xlabel("Iteration")
    a2.set_ylabel("E*0 − best E so far (floored at 1e-7)")
    a2.set_title(f"All {rep['n_seeds']} seeds: multi-start incumbent gap", loc="left", fontsize=8.5, color=INK)
    a2.legend(fontsize=7.5)
    means = np.array(sh["regional_means_ranked"])
    centre = means.mean(0)
    _, _, vt = np.linalg.svd(means - centre)
    P = vt[:2].T
    for m, pth in enumerate(sh["paths"]):
        q2 = (np.array(pth) - centre) @ P
        is_best = m == sh["best_start_index"]
        a3.plot(q2[:, 0], q2[:, 1], lw=2 if is_best else 0.7, color=PAL["optimum"] if is_best else "#9ec5f4",
                zorder=3 if is_best else 1)
        a3.plot(q2[0, 0], q2[0, 1], "o", ms=3, color=INK2, zorder=2)
    mq = (means - centre) @ P
    a3.scatter(mq[:, 0], mq[:, 1], s=60, color=PAL["single"], edgecolor="#fcfcfb", lw=1.5, zorder=4)
    for k in range(4):
        a3.annotate(f"R{k + 1}", mq[k], xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
    bq = (np.array(sh["bstar0"]) - centre) @ P
    a3.scatter([bq[0]], [bq[1]], marker="*", s=180, color=PAL["optimum"], edgecolor=INK, lw=0.8, zorder=5)
    a3.annotate("b*0", bq, xytext=(6, -10), textcoords="offset points", fontsize=8, color=INK)
    a3.set_xlabel("PC1 of the regional means")
    a3.set_ylabel("PC2")
    a3.set_title("Gradient paths (grey dots = starts)", loc="left", fontsize=8.5, color=INK)
    fig.tight_layout()
    save(fig, "r1_convergence.png")

    # (e) ternary landscape
    land = sh["landscape"]
    bary = np.array(land["bary"])

    def cart(b):
        b = np.atleast_2d(b)
        return b[:, 1] + 0.5 * b[:, 2], (math.sqrt(3) / 2) * b[:, 2]
    X, Y = cart(bary)
    tri = mtri.Triangulation(X, Y)
    fig, ax = plt.subplots(figsize=(6.6, 6))
    cs = ax.tricontourf(tri, land["E"], levels=24, cmap="Blues")
    has_half = max(land["E"]) >= 0.5
    if has_half:
        ax.tricontour(tri, land["E"], levels=[0.5], colors=[INK], linewidths=0.9, linestyles="--")
    fig.colorbar(cs, ax=ax, fraction=0.04, label="Expected share E (round 0)")
    px, py = cart(np.array(sh["best_path_bary"]))
    ax.plot(px, py, color=PAL["optimum"], lw=1.8, label="gradient path (best start)")
    ax.plot(px[0], py[0], "o", color=PAL["optimum"], ms=5)
    rx, ry = cart(np.array(sh["regional_means_bary"]))
    ax.scatter(rx, ry, s=55, color=PAL["group"], edgecolor="#fcfcfb", lw=1.5, zorder=4, label="regional means (projected)")
    for k in range(4):
        ax.annotate(f"R{k + 1}", (rx[k], ry[k]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
    bx, by = cart(np.array(sh["bstar0_bary"]))
    ax.scatter(bx, by, marker="*", s=200, color=PAL["optimum"], edgecolor=INK, lw=0.8, zorder=5, label="b*0")
    ax.plot([0, 1, 0.5, 0], [0, 0, math.sqrt(3) / 2, 0], color=INK2, lw=0.8)
    s = land["sector_names"]
    ax.text(0, -0.04, f"{s[0]} = all", ha="center", va="top", fontsize=8.5, color=INK)
    ax.text(1, -0.04, f"{s[1]} = all", ha="center", va="top", fontsize=8.5, color=INK)
    ax.text(0.5, math.sqrt(3) / 2 + 0.03, f"{s[2]} = all", ha="center", fontsize=8.5, color=INK)
    ax.set_aspect("equal")
    ax.axis("off")
    fx = ", ".join(f"{n} = {v:.3f}" for n, v in zip(land["fixed_sectors"], land["fixed_values"]))
    ax.set_title(f"Seed {sh['seed']}: E over {'/'.join(s)} (free mass {land['free_mass']:.3f}); fixed {fx}\n"
                 + ("dashed contour: E = 0.5" if has_half else f"max E on this slice {max(land['E']):.3f} < 0.5"),
                 loc="left", fontsize=8.5, color=INK, pad=22)
    ax.legend(loc="upper right", fontsize=7.5, bbox_to_anchor=(1.02, 1.0))
    save(fig, "r1_ternary_landscape.png")

    # (f) divided vs aligned
    sp = rep["scenario_split"]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    y = np.arange(len(order))[::-1]
    for off, g, col in ((0.17, "divided", PAL["single"]), (-0.17, "aligned", PAL["group"])):
        if "manifestos" not in sp[g]:
            continue
        mm = sp[g]["manifestos"]
        xs_ = [mm[n]["E"]["mean"] for n in order]
        lo = [mm[n]["E"]["mean"] - mm[n]["E"]["ci95"][0] for n in order]
        hi = [mm[n]["E"]["ci95"][1] - mm[n]["E"]["mean"] for n in order]
        ax.errorbar(xs_, y + off, xerr=[lo, hi], fmt="o", ms=5, color=col, lw=1, capsize=2,
                    label=f"{g} worlds (n = {sp[g]['n']})")
    ax.set_yticks(y, [man[n]["label"] for n in order])
    ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Expected vote share E at round 0 (mean, 95% t-CI)")
    ax.set_title(f"Divided (TV of two largest regions ≥ {DIVIDED_TV}) vs aligned-majority worlds", loc="left", color=INK, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.07), ncol=2)
    save(fig, "r1_divided_vs_aligned.png")
    return saved


# ============================================================================== main
def main() -> None:
    global _LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=12000)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default=xc.DEFAULT_DEVICE)
    parser.add_argument("--seed-batch", type=int, default=25, help="seeds per batched PG/scoring call")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        rep = json.loads(JSON_PATH.read_text())
        for p in plot_all(rep):
            print("figure", p)
        return
    torch.set_num_threads(4)
    for d in (OUT, RAW_PATH.parent, FIG, LOG_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
    _LOG = LOG_PATH.open("w")
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    log(f"[r1] device {device} ({dev_record})")
    free0 = xc.free_bytes(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    deviations = []
    start = time.perf_counter()
    batch_record: dict = {}
    with xc.GpuMonitor(device) as monitor:
        items = []
        for i, s in enumerate(seeds):
            items.append(prepare(s, args.n, device))
            if (i + 1) % 10 == 0:
                log(f"[r1] prepared {i + 1}/{len(seeds)} electorates (+ORACLE, library PG) {time.perf_counter() - start:.0f}s")
        t_prep = time.perf_counter() - start
        rows: list[dict] = []
        show_box: list = []

        def run(chunk):
            r, sh = process_batch(chunk)
            if sh is not None:
                show_box.append(sh)
            log(f"[r1] batch of {len(chunk)} seeds done {time.perf_counter() - start:.0f}s")
            return r
        t0 = time.perf_counter()
        rows = xc.run_chunks(run, items, args.seed_batch, log=log, record=batch_record)
        t_main = time.perf_counter() - t0
        RAW_PATH.write_text("")
        for r in rows:
            xc.write_row(RAW_PATH, r)
        del items
        par = parity(seeds[:args.parity_seeds], device, args.n)
        log(f"[r1] parity passed: {par['passed']}")
    seconds = time.perf_counter() - start
    if device.type == "cuda" and device.index != 2:
        deviations.append(f"Device {device} instead of cuda:2 (selection record in provenance).")
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    summary = summarise(rows)
    show = show_box[0] if show_box else None
    report = {
        "experiment": "R1", "question": "What does the ideal manifesto look like at round 0 (no campaign, no drift)?",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds}, "n_seeds": len(seeds), "n": args.n,
        "objective": "exact expected share on v(0) = electorate.preferences",
        "optimiser": {"method": "projected gradient (as manifesto_opt.optimise_manifesto)", "step": STEP,
                      "max_iterations": ITERS, "smoothing": SMOOTH, "move_tol": MOVE_TOL,
                      "starts": "7 library default_starts + top-3 + 19 scenario variants = 27",
                      "b_star0": "best exact E over every iterate of every start", "tol_for_convergence": TOL},
        "divided_threshold": DIVIDED_TV, "hist_bins": BINS,
        **summary, "showcase": show, "deviations": deviations, "config": base.to_record(),
        "timing": {"prepare_seconds": t_prep, "batched_pg_and_scoring_seconds": t_main},
        "provenance": xc.provenance_v3(args, device, seconds, batch_sizes={"seeds_per_batch": batch_record.get("batch_sizes"),
                                                                           "starts_M": 27, "manifestos_scored_M": 21,
                                                                           "voters_N": args.n, "ternary_grid_points": (GRID_RES + 1) * (GRID_RES + 2) // 2,
                                                                           **{k: v for k, v in batch_record.items() if k != "batch_sizes"}},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       grad_mode="autograd (PG), no_grad (scoring)"),
    }
    xc.save(str(JSON_PATH), report)
    for p in plot_all(report):
        log(f"[r1] figure {p}")
    pk = report["provenance"]["compute"]["peak_bytes"]
    log(f"[r1] done in {seconds:.1f}s; peak {pk / 2**30 if pk else 0:.2f} GiB")
    _LOG.close()


if __name__ == "__main__":
    main()
