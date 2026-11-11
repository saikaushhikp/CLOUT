"""R4 — does the best manifesto stay the best as opinions drift?

Command (reproduces everything under reports/v3/experimentation/ for R4):
  OMP_NUM_THREADS=4 python experiments/r4_best_over_time.py --seeds 100 --first-seed 12000 --device cuda:2
  python experiments/r4_best_over_time.py --plot-only          # redraw the figures from r4.json

Worlds: seeds 12000–12099 (the R1 worlds), N = 10,000, default config. Two paired populations per seed, built as in
R2 (``r2_drift_no_campaign.world_pair``): "zealots" (default ζ = 0.05, primary) and "normal" (ζ = 0, secondary);
v(0), graph, regions and non-zealot α are identical.

No campaign. Each world drifts with the game's own social update (``r2.social_run`` = ``oracle.anticipate``) and the
electorate is recorded at ROUNDS = 0…10, 20, 50. For EVERY recorded round s (a superset of the requested design
rounds {0,1,2,3,4,5,7,10,20,50}) the best manifesto b*_s for v(s) is found with R1's batched multi-start projected
gradient (``r1.pg_ascent``: step 0.02, pseudo-Huber δ = 1e-3, ≤ 400 iterations; starts = 7 library defaults + top-3
+ R1's 19 scenario variants computed on v(s) + a warm start at b*_{previous s} = 28); b*_s = best EXACT E over every
iterate (``r1.best_iterate``). Every b*_s and the library ORACLE manifesto (``oracle_manifesto``, optimised on the
5-round forecast) are then scored at every recorded round t → E[s, t]; E*(t) = E[t, t];
regret[s, t] = E*(t) − E[s, t].
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from r1_manifesto_round0 import (  # noqa: E402
    DIVIDED_TV, INK, INK2, N_STANDARD, PAL, _style, best_iterate, pg_ascent, regional_means, scenario_manifestos,
    score_batch, size_order, tv_two_largest,
)
from r2_drift_no_campaign import social_run, tv, world_pair  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.sectors import SECTORS  # noqa: E402
from dumbocracy.strategies.manifesto_opt import default_starts, project_simplex  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto, top3_manifesto  # noqa: E402
from dumbocracy.strategies.worlds import _union_graph  # noqa: E402

OUT = xc.REPORTS / "experimentation"
JSON_PATH = OUT / "r4.json"
RAW_PATH = OUT / "raw" / "r4.jsonl"
FIG = OUT / "figures"
LOG_PATH = OUT / "logs" / "r4.log"

ROUNDS = tuple(range(11)) + (20, 50)
REQUESTED_DESIGN = (0, 1, 2, 3, 4, 5, 7, 10, 20, 50)
POPS = ("zealots", "normal")
TV_MOVE = 0.05
NEG_TOL = 1e-4          # regret below −NEG_TOL counts as an optimiser miss
SHOW_SEED = 12000
GAME_LENGTHS = (3, 4, 5)
SCEN_LABEL = {"divided": "divided", "united": "united"}

_LOG = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


# ============================================================================== core (unit-tested)
def record_states(v0: torch.Tensor, graph, weight: torch.Tensor, frozen: torch.Tensor, rounds=ROUNDS,
                  rule: str = "degroot", trim: int = 0) -> dict[int, torch.Tensor]:
    """Run social-only drift to max(rounds) and keep v(t) for t in ``rounds`` (shape of ``v0``)."""
    keep: dict[int, torch.Tensor] = {}
    want = set(rounds)
    social_run(v0, graph, weight, frozen, max(rounds), rule, trim,
               on_round=lambda t, v: keep.__setitem__(t, v.clone()) if t in want else None)
    return keep


def starts_for(v: torch.Tensor, region_of: torch.Tensor, sizes: list[int]) -> torch.Tensor:
    """R1's 27 starts for one electorate: 7 library defaults + top-3 + the 19 scenario variants → [27, 5]."""
    std = project_simplex(torch.cat([default_starts(v), top3_manifesto(v).unsqueeze(0).to(v)]))
    return torch.cat([std, scenario_manifestos(v, region_of, sizes)[1]])


def optimise_rounds(states: dict[int, torch.Tensor], region_of: torch.Tensor, sizes: list[list[int]], beta: float,
                    z: float, design=ROUNDS, iterations: int = 400) -> dict:
    """b*_s for every s in ``design`` (ascending): batched PG over seeds × 28 starts on v(s) [B, N, 5].

    Returns ``b`` [S, B, 5], ``E`` [S, B], ``best_start`` [S, B] (27 = the warm start)."""
    bs, Es, ms = [], [], []
    prev = None
    for s in design:
        v = states[s]
        st = torch.stack([starts_for(v[i], region_of[i], sizes[i]) for i in range(v.shape[0])])     # [B, 27, 5]
        warm = (prev if prev is not None else st[:, 1]).unsqueeze(1)                              # round 0: the mean
        path, trace = pg_ascent(v, torch.cat([st, warm], 1), beta, z, iterations=iterations)
        b, E, _, m = best_iterate(path, trace)
        bs.append(b)
        Es.append(E)
        ms.append(m)
        prev = b
        del path, trace
    return {"b": torch.stack(bs), "E": torch.stack(Es), "best_start": torch.stack(ms)}


@torch.no_grad()
def evaluate(states: dict[int, torch.Tensor], b: torch.Tensor, beta: float, z: float, rounds=ROUNDS) -> torch.Tensor:
    """Exact E of manifestos ``b`` [B, M, 5] on v(t) for each t → [M, T, B]."""
    out = []
    for t in rounds:
        s = 1.0 - 0.5 * (states[t].unsqueeze(1) - b.unsqueeze(2)).abs().sum(-1)                  # [B, M, N]
        out.append(torch.sigmoid(beta * (s - z)).mean(-1))
    return torch.stack(out).permute(2, 0, 1)


def regret_matrix(E: torch.Tensor, design_idx: list[int]) -> torch.Tensor:
    """E [S, T, ...] with design s evaluated at rounds t; design s optimised at round index design_idx[s].
    Regret[s, t] = E[s*, t] − E[s, t] where s* is the design made at round t (defined only for t in the design
    set; other columns are NaN)."""
    S, T = E.shape[:2]
    R = torch.full_like(E, float("nan"))
    for s2, t in enumerate(design_idx):
        R[:, t] = E[s2, t].unsqueeze(0) - E[:, t]
    return R


def first_crossover(a: list[float], b: list[float], rounds: list[int], eps: float = 0.0) -> int | None:
    """First round at which series ``b`` is strictly above series ``a`` (by more than ``eps``); None if never."""
    for t, x, y in zip(rounds, a, b):
        if y > x + eps:
            return t
    return None


def first_reach(xs: list[float], rounds: list[int], thr: float) -> int | None:
    for t, x in zip(rounds, xs):
        if x >= thr:
            return t
    return None


# ============================================================================== one batch of seeds
def build(seed: int, device: torch.device, base: GameConfig) -> dict:
    w = world_pair(seed, device, base)
    cfg_n = dataclasses.replace(w["cfg"], electorate=dataclasses.replace(w["cfg"].electorate, zealot_fraction=0.0))
    ez = w["ez"]
    sizes = [int(x) for x in ez.sizes.tolist()]
    sigma = [int(x) for x in ez.archetype_of_region.tolist()]
    t2 = tv_two_largest(sizes, sigma)
    return {"seed": seed, "cfg": w["cfg"], "ez": ez, "en": w["en"], "check": w["check"], "v5_ref": w["v5_ref"],
            "b_lib0": w["b"][0], "oracle": {"zealots": w["b"][1], "normal": oracle_manifesto(w["en"], cfg_n).to(w["b"])},
            "sizes": sizes, "sigma": sigma, "tv_two_largest": t2,
            "scenario": "divided" if t2 >= DIVIDED_TV else "united"}


def process_batch(worlds: list[dict], design=ROUNDS, pops=POPS, iterations: int = 400) -> list[dict]:
    cfg = worlds[0]["cfg"]
    B, n = len(worlds), worlds[0]["ez"].num_voters
    beta, z = cfg.voting.sensitivity, cfg.voting.threshold
    graph = worlds[0]["ez"].graph if B == 1 else _union_graph([w["ez"].graph for w in worlds], n)
    v0 = torch.cat([w["ez"].preferences for w in worlds])
    key = {"zealots": "ez", "normal": "en"}
    weight = torch.stack([cfg.dynamics.social_step * torch.cat([w[key[p]].susceptibility.social for w in worlds])
                          for p in pops])
    frozen = torch.stack([torch.cat([w[key[p]].susceptibility.zealot.to(torch.bool) for w in worlds]) for p in pops])
    rounds = sorted(set(ROUNDS) | set(design))
    flat = record_states(torch.stack([v0] * len(pops)), graph, weight, frozen, rounds,
                         cfg.dynamics.social_rule, cfg.dynamics.wmsr_trim)
    region_of = torch.stack([w["ez"].region_of for w in worlds])
    v5_diff = None
    if "zealots" in pops and 5 in flat:
        v5 = flat[5][pops.index("zealots")].view(B, n, -1)
        v5_diff = float((v5 - torch.stack([w["v5_ref"] for w in worlds])).abs().max())
    didx = [rounds.index(s) for s in design]
    rows = []
    for p, pop in enumerate(pops):
        states = {t: flat[t][p].view(B, n, -1) for t in rounds}
        opt = optimise_rounds(states, region_of, [w["sizes"] for w in worlds], beta, z, design, iterations)
        orc = torch.stack([w["oracle"][pop] for w in worlds]).unsqueeze(1).to(v0)                  # [B, 1, 5]
        allb = torch.cat([opt["b"].permute(1, 0, 2), orc, torch.stack([w["b_lib0"] for w in worlds]).unsqueeze(1).to(v0)], 1)
        E = evaluate(states, allb, beta, z, rounds)                                                # [S+2, T, B]
        # support by region (size rank) of b*_s on its own round s
        sup = []
        for k, s in enumerate(design):
            sc = score_batch(states[s], region_of, opt["b"][k].unsqueeze(1), beta, z, bins=4)
            sup.append(sc["region_P"][:, 0])                                                       # [B, 4]
        sup = torch.stack(sup).cpu()                                                               # [S, B, 4]
        E, Eopt, bst, bs = E.cpu(), opt["E"].cpu(), opt["best_start"].cpu(), opt["b"].cpu()
        orc_c = orc[:, 0].cpu()
        for i, w in enumerate(worlds):
            order = size_order(w["sizes"])
            total = sum(w["sizes"])
            Ei = E[:, :, i]
            S = len(design)
            reg = regret_matrix(Ei[:S], didx)
            b0, b_ = bs[0, i], bs[:, i]
            tvs = [float(tv(b_[k], b0)) for k in range(S)]
            sup_rank = [[float(sup[k, i, r]) for r in order] for k in range(S)]
            carried = [[r + 1 for r in range(4) if sr[r] > 0.5] for sr in sup_rank]
            row = {"exp": "R4", "seed": w["seed"], "population": pop, "sizes": w["sizes"], "sigma": w["sigma"],
                   "tv_two_largest": w["tv_two_largest"], "scenario": w["scenario"],
                   "rounds": rounds, "design_rounds": list(design),
                   "bstar": {str(s): [round(float(x), 6) for x in b_[k].tolist()] for k, s in enumerate(design)},
                   "E_star_pg": [float(x) for x in Eopt[:, i].tolist()],
                   "best_start": [int(x) for x in bst[:, i].tolist()],
                   "E": {str(s): [float(x) for x in Ei[k].tolist()] for k, s in enumerate(design)},
                   "E_oracle": [float(x) for x in Ei[S].tolist()],
                   "E_library_b0": [float(x) for x in Ei[S + 1].tolist()],
                   "regret": {str(s): [None if math.isnan(x) else float(x) for x in reg[k].tolist()]
                              for k, s in enumerate(design)},
                   "oracle_b": [round(float(x), 6) for x in orc_c[i].tolist()],
                   "tv_to_bstar0": tvs,
                   "tv_oracle_to_bstar5": float(tv(orc_c[i], b_[design.index(5)])) if 5 in design else None,
                   "support_by_rank": sup_rank, "carried_ranks": carried,
                   "carried_share": [sum(w["sizes"][order[r - 1]] for r in c) / total for c in carried]}
            if 0 in design and 5 in design:
                k0, k5 = design.index(0), design.index(5)
                row["crossover_b5_over_b0"] = first_crossover(Ei[k0].tolist(), Ei[k5].tolist(), rounds)
            if 0 in design:
                row["first_round_tv_ge_0.05"] = first_reach(tvs, list(design), TV_MOVE)
            row["min_regret"] = float(torch.nan_to_num(reg, nan=0.0).min())
            if pop == "zealots":
                row["identity_check"] = w["check"]
            rows.append(row)
        del states
    if v5_diff is not None:
        for r in rows:
            r["v5_vs_anticipate_max_abs"] = v5_diff
    return rows


# ============================================================================== summary
def _col(rows, fn):
    return [fn(r) for r in rows]


def summarise_pop(rows: list[dict]) -> dict:
    rounds, design = rows[0]["rounds"], rows[0]["design_rounds"]
    ti = {t: rounds.index(t) for t in rounds}
    di = {s: design.index(s) for s in design}
    S = len(design)
    out: dict = {"n": len(rows)}
    out["E_star"] = {str(t): xc.mean_ci(_col(rows, lambda r: r["E"][str(t)][ti[t]])) for t in design}
    out["E_by_design"] = {str(s): {str(t): xc.mean_ci(_col(rows, lambda r: r["E"][str(s)][ti[t]])) for t in rounds}
                          for s in design}
    out["E_oracle"] = {str(t): xc.mean_ci(_col(rows, lambda r: r["E_oracle"][ti[t]])) for t in rounds}
    out["regret"] = {str(s): {str(t): xc.mean_ci(_col(rows, lambda r: r["regret"][str(s)][ti[t]])) for t in design}
                     for s in design}
    out["regret_votes_per_10k"] = {s: {t: round(10_000 * c["mean"], 1) for t, c in d.items()} for s, d in out["regret"].items()}
    neg = [(r["seed"], s, t, r["regret"][str(s)][ti[t]]) for r in rows for s in design for t in design
           if r["regret"][str(s)][ti[t]] is not None and r["regret"][str(s)][ti[t]] < -NEG_TOL]
    allreg = [r["regret"][str(s)][ti[t]] for r in rows for s in design for t in design]
    out["negative_regret"] = {"tolerance": NEG_TOL, "count_below_minus_tol": len(neg),
                              "cells": len(allreg), "min": min(allreg),
                              "worst": sorted(neg, key=lambda x: x[3])[:10]}
    out["tv_to_bstar0"] = {str(s): xc.mean_ci(_col(rows, lambda r: r["tv_to_bstar0"][di[s]])) for s in design}
    out["tv_to_bstar0_median"] = {str(s): sorted(_col(rows, lambda r: r["tv_to_bstar0"][di[s]]))[len(rows) // 2]
                                  for s in design}
    out["rate_tv_ge_0.05"] = {str(s): xc.rate(_col(rows, lambda r: r["tv_to_bstar0"][di[s]] >= TV_MOVE)) for s in design}
    fr = _col(rows, lambda r: r.get("first_round_tv_ge_0.05"))
    out["first_round_tv_ge_0.05"] = {"counts": {str(k): fr.count(k) for k in sorted({x for x in fr if x is not None})},
                                     "never": fr.count(None), "n": len(fr),
                                     "median": _median([x for x in fr if x is not None] + [math.inf] * fr.count(None)),
                                     "mean_among_reached": xc.mean_ci([x for x in fr if x is not None])}
    # round at which the MEAN TV curve reaches 0.05
    out["mean_tv_first_round_ge_0.05"] = first_reach([out["tv_to_bstar0"][str(s)]["mean"] for s in design], design, TV_MOVE)
    cr = _col(rows, lambda r: r.get("crossover_b5_over_b0"))
    out["crossover_b5_over_b0"] = {"counts": {str(k): cr.count(k) for k in sorted({x for x in cr if x is not None})},
                                   "never_by_50": cr.count(None), "n": len(cr),
                                   "median": _median([x for x in cr if x is not None] + [math.inf] * cr.count(None)),
                                   "by_round_3_rate": xc.rate([x is not None and x <= 3 for x in cr]),
                                   "by_round_5_rate": xc.rate([x is not None and x <= 5 for x in cr])}
    out["support_by_rank"] = {str(s): [xc.mean_ci(_col(rows, lambda r: r["support_by_rank"][di[s]][k])) for k in range(4)]
                              for s in design}
    out["carried_count"] = {str(s): xc.mean_ci(_col(rows, lambda r: len(r["carried_ranks"][di[s]]))) for s in design}
    out["carried_share"] = {str(s): xc.mean_ci(_col(rows, lambda r: r["carried_share"][di[s]])) for s in design}
    out["sector_abs_change_from_b0"] = {str(s): {SECTORS[k]: xc.mean_ci(_col(rows, lambda r: abs(r["bstar"][str(s)][k] - r["bstar"]["0"][k])))
                                                 for k in range(5)} for s in design}
    out["sector_mean_share"] = {str(s): {SECTORS[k]: xc.mean_ci(_col(rows, lambda r: r["bstar"][str(s)][k])) for k in range(5)}
                                for s in design}
    out["tv_oracle_to_bstar5"] = xc.mean_ci(_col(rows, lambda r: r["tv_oracle_to_bstar5"]))
    out["oracle_minus_Estar"] = {str(t): xc.mean_ci(_col(rows, lambda r: r["E_oracle"][ti[t]] - r["E"][str(t)][ti[t]]))
                                 for t in design}
    out["win_alone_rate_Estar"] = {str(t): xc.rate(_col(rows, lambda r: r["E"][str(t)][ti[t]] > 0.5)) for t in design}
    out["warm_start_won_rate"] = {str(s): xc.rate(_col(rows, lambda r: r["best_start"][di[s]] == N_STANDARD + 19))
                                  for s in design}
    # decision outputs
    dec: dict = {}
    for s in (0, 3, 4, 5):
        for t in GAME_LENGTHS:
            dec[f"regret_design{s}_vote{t}"] = out["regret"][str(s)][str(t)]
    worst = {s: _col(rows, lambda r, s=s: max(r["regret"][str(s)][ti[t]] for t in GAME_LENGTHS)) for s in design}
    avg = {s: _col(rows, lambda r, s=s: sum(r["regret"][str(s)][ti[t]] for t in GAME_LENGTHS) / len(GAME_LENGTHS))
           for s in design}
    dec["worst_case_regret_over_T3to5"] = {str(s): xc.mean_ci(worst[s]) for s in design}
    dec["mean_regret_over_T3to5"] = {str(s): xc.mean_ci(avg[s]) for s in design}
    dec["best_design_by_mean_worst_case"] = min(design, key=lambda s: dec["worst_case_regret_over_T3to5"][str(s)]["mean"])
    dec["best_design_by_mean_regret"] = min(design, key=lambda s: dec["mean_regret_over_T3to5"][str(s)]["mean"])
    out["decision"] = dec
    # tests (Wilcoxon, Holm over the family)
    tests = {}

    def reg(s, t):
        return _col(rows, lambda r: r["regret"][str(s)][ti[t]])
    zero = [0.0] * len(rows)
    for s, t in ((0, 3), (0, 4), (0, 5), (5, 3), (3, 5), (4, 3), (4, 5), (5, 4), (3, 4)):
        tests[f"regret[{s},{t}]>0"] = xc.paired(reg(s, t), zero)
    tests["regret[0,3]-regret[5,3]"] = xc.paired(reg(0, 3), reg(5, 3))
    tests["regret[0,5]-regret[3,5]"] = xc.paired(reg(0, 5), reg(3, 5))
    tests["worst3to5[design5]-worst3to5[design4]"] = xc.paired(worst[5], worst[4])
    tests["worst3to5[design3]-worst3to5[design4]"] = xc.paired(worst[3], worst[4])
    tests["worst3to5[design0]-worst3to5[design4]"] = xc.paired(worst[0], worst[4])
    tests["E_oracle(5)-E*(5)"] = xc.paired(_col(rows, lambda r: r["E_oracle"][ti[5]]), _col(rows, lambda r: r["E"]["5"][ti[5]]))
    for k, p in zip(tests, xc.holm([v["wilcoxon_p"] for v in tests.values()])):
        tests[k]["holm_p"] = p
    out["tests"] = tests
    return out


def _median(xs):
    if not xs:
        return None
    xs = sorted(xs)
    m = xs[len(xs) // 2]
    return None if m == math.inf else m


def summarise(rows: list[dict]) -> dict:
    from scipy.stats import mannwhitneyu
    out = {}
    for pop in POPS:
        sub = [r for r in rows if r["population"] == pop]
        if not sub:
            continue
        d = summarise_pop(sub)
        split = {}
        for g in ("divided", "united"):
            gs = [r for r in sub if r["scenario"] == g]
            split[g] = summarise_pop(gs) if len(gs) >= 5 else {"n": len(gs)}
        div = [r for r in sub if r["scenario"] == "divided"]
        uni = [r for r in sub if r["scenario"] == "united"]
        ti = {t: sub[0]["rounds"].index(t) for t in sub[0]["rounds"]}
        between, pv = {}, []
        if len(div) >= 3 and len(uni) >= 3:
            for name, fn in (("regret[0,3]", lambda r: r["regret"]["0"][ti[3]]),
                             ("regret[0,5]", lambda r: r["regret"]["0"][ti[5]]),
                             ("regret[5,3]", lambda r: r["regret"]["5"][ti[3]]),
                             ("tv_bstar5_to_bstar0", lambda r: r["tv_to_bstar0"][r["design_rounds"].index(5)]),
                             ("E*(5)", lambda r: r["E"]["5"][ti[5]])):
                a, b = _col(div, fn), _col(uni, fn)
                p = float(mannwhitneyu(a, b).pvalue)
                between[name] = {"divided": xc.mean_ci(a), "united": xc.mean_ci(b), "mannwhitney_p": p}
                pv.append(p)
            for k, p in zip(between, xc.holm(pv)):
                between[k]["holm_p"] = p
        split["between"] = between
        d["scenario_split"] = split
        out[pop] = d
    # zealots vs normal (paired by seed) on the key regrets
    z = {r["seed"]: r for r in rows if r["population"] == "zealots"}
    nm = {r["seed"]: r for r in rows if r["population"] == "normal"}
    common = sorted(set(z) & set(nm))
    if common:
        ti = {t: rows[0]["rounds"].index(t) for t in rows[0]["rounds"]}
        cmp_, pv = {}, []
        for name, fn in (("regret[0,3]", lambda r: r["regret"]["0"][ti[3]]),
                         ("regret[0,5]", lambda r: r["regret"]["0"][ti[5]]),
                         ("regret[5,3]", lambda r: r["regret"]["5"][ti[3]]),
                         ("regret[0,50]", lambda r: r["regret"]["0"][ti[50]]),
                         ("tv_bstar5_to_bstar0", lambda r: r["tv_to_bstar0"][r["design_rounds"].index(5)]),
                         ("E*(5)", lambda r: r["E"]["5"][ti[5]]), ("E*(50)", lambda r: r["E"]["50"][ti[50]])):
            cmp_[name] = xc.paired([fn(z[s]) for s in common], [fn(nm[s]) for s in common])
            pv.append(cmp_[name]["wilcoxon_p"])
        for k, p in zip(cmp_, xc.holm(pv)):
            cmp_[k]["holm_p"] = p
        out["zealots_minus_normal"] = cmp_
    return out


def showcase(rows: list[dict], seed: int = SHOW_SEED) -> dict | None:
    r = next((r for r in rows if r["seed"] == seed and r["population"] == "zealots"), None)
    if r is None:
        return None
    return {"seed": seed, "design_rounds": r["design_rounds"], "bstar": r["bstar"], "oracle_b": r["oracle_b"],
            "E": r["E"], "E_oracle": r["E_oracle"], "tv_to_bstar0": r["tv_to_bstar0"], "sizes": r["sizes"],
            "scenario": r["scenario"], "carried_ranks": r["carried_ranks"]}


# ============================================================================== CPU↔GPU parity
def parity(seeds: list[int], device: torch.device, n: int) -> dict:
    design = (0, 3, 5)

    def fn(seed, dev):
        base = GameConfig(device=str(dev), electorate=ElectorateConfig(num_voters=n))
        r = process_batch([build(seed, dev, base)], design=design, pops=("zealots",))[0]
        ti = {t: r["rounds"].index(t) for t in r["rounds"]}
        return {"E*0": r["E"]["0"][0], "E*5": r["E"]["5"][ti[5]], "E_b0_at5": r["E"]["0"][ti[5]],
                "regret_0_5": r["regret"]["0"][ti[5]], "regret_5_3": r["regret"]["5"][ti[3]],
                "tv_b5_b0": r["tv_to_bstar0"][2], "E_oracle_at5": r["E_oracle"][ti[5]]}
    return xc.parity_check(fn, seeds, device=device, abs_tol=0.002)


# ============================================================================== figures (from the saved JSON only)
def plot_all(rep: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _style(plt)
    FIG.mkdir(parents=True, exist_ok=True)
    saved = []
    Z = rep["summary"]["zealots"]
    rounds, design = rep["rounds"], rep["design_rounds"]
    xpos = {t: i for i, t in enumerate(rounds)}
    xt = [xpos[t] for t in rounds]

    def save(fig, name):
        p = FIG / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p.relative_to(xc.ROOT)))

    def xaxis(ax, ts=rounds):
        ax.set_xticks([xpos[t] for t in ts], [str(t) for t in ts])
        ax.set_xlabel("Round (0–10 every round, then 20 and 50; axis not to scale after 10)")

    def band(ax, d, ts, color, label, ls="-", marker=None):
        m = [d[str(t)]["mean"] for t in ts]
        lo = [d[str(t)]["ci95"][0] for t in ts]
        hi = [d[str(t)]["ci95"][1] for t in ts]
        x = [xpos[t] for t in ts]
        ax.plot(x, m, color=color, lw=2, ls=ls, label=label, marker=marker, ms=4)
        ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)

    def panel_a(ax, S, title):
        band(ax, S["E_star"], design, PAL["reference"], "Re-optimised best E*(t)", marker="o")
        band(ax, S["E_by_design"]["0"], rounds, PAL["single"], "b*0 (best for round 0), kept")
        band(ax, S["E_by_design"]["5"], rounds, PAL["optimum"], "b*5 (best for round 5 = election day), kept", ls="--")
        ax.axvline(xpos[5], color=INK2, lw=0.7, ls=":")
        ax.set_ylabel("Expected vote share E")
        ax.set_title(title, loc="left", color=INK, fontsize=9)
        xaxis(ax)

    # (a)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    panel_a(ax, Z, f"R4 (a): vote share as voters drift, zealot world ({Z['n']} seeds, mean and 95% CI)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "r4_share_over_rounds.png")

    # (b) regret heat map
    M = np.array([[Z["regret"][str(s)][str(t)]["mean"] * 10_000 for t in design] for s in design])
    fig, ax = plt.subplots(figsize=(7.4, 6))
    vmax = float(np.nanmax(M))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="#fcfcfb" if M[i, j] > 0.55 * vmax else INK)
    ax.set_xticks(range(len(design)), [str(t) for t in design])
    ax.set_yticks(range(len(design)), [str(s) for s in design])
    ax.set_xlabel("Voting round t")
    ax.set_ylabel("Design round s (manifesto b*_s built for round s)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Mean regret, votes lost per 10,000 voters")
    ax.set_title("R4 (b): regret of keeping b*_s when the vote is at round t (zealot world)", loc="left",
                 color=INK, fontsize=9)
    save(fig, "r4_regret_heatmap.png")

    # (c) TV from b*_0
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for pop, col in (("zealots", PAL["single"]), ("normal", PAL["group"])):
        if pop in rep["summary"]:
            band(ax, rep["summary"][pop]["tv_to_bstar0"], design, col,
                 "zealot world (5%)" if pop == "zealots" else "no-zealot world", marker="o")
    ax.axhline(TV_MOVE, color=INK2, lw=0.8, ls="--")
    ax.text(0.1, TV_MOVE + 0.003, f"TV = {TV_MOVE}", fontsize=7.5, color=INK2)
    ax.set_ylabel("TV(b*_t, b*_0)")
    ax.set_title("R4 (c): how far the best manifesto moves away from the round-0 best", loc="left", color=INK, fontsize=9)
    xaxis(ax, design)
    ax.legend(fontsize=8)
    save(fig, "r4_optimum_drift.png")

    # (d) support by region size rank
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cols = [PAL["single"], PAL["group"], PAL["average"], PAL["reference"]]
    for k in range(4):
        d = {s: Z["support_by_rank"][s][k] for s in Z["support_by_rank"]}
        band(ax, d, design, cols[k], f"R{k + 1}" + (" (largest)" if k == 0 else " (smallest)" if k == 3 else ""), marker="o")
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_ylabel("Mean vote probability in region under b*_t at round t")
    ax.set_title("R4 (d): support by region (ranked by size) for the best manifesto of each round", loc="left",
                 color=INK, fontsize=9)
    xaxis(ax, design)
    ax.legend(fontsize=8, ncol=4, loc="lower center")
    save(fig, "r4_region_support.png")

    # (e) showcase sector shares
    sh = rep["showcase"]
    if sh:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        scol = [PAL["single"], PAL["group"], PAL["average"], PAL["optimum"], PAL["reference"]]
        for k, sec in enumerate(rep["sectors"]):
            ax.plot([xpos[s] for s in design], [sh["bstar"][str(s)][k] for s in design], marker="o", ms=4, lw=2,
                    color=scol[k], label=sec)
            ax.plot([xpos[5]], [sh["oracle_b"][k]], marker="x", ms=8, color=scol[k], ls="")
        ax.plot([], [], marker="x", color=INK2, ls="", label="ORACLE manifesto (at round 5)")
        ax.set_ylabel("Share of the manifesto")
        ax.set_title(f"R4 (e): seed {sh['seed']}, the best manifesto's 5 sector shares by round ({sh['scenario']} world)",
                     loc="left", color=INK, fontsize=9)
        xaxis(ax, design)
        ax.legend(fontsize=8, ncol=3)
        save(fig, "r4_showcase_sectors.png")

    # (f) crossover histogram
    cr = Z["crossover_b5_over_b0"]
    keys = [str(t) for t in rounds if str(t) in cr["counts"]] + ["never"]
    vals = [cr["counts"].get(k, 0) for k in keys[:-1]] + [cr["never_by_50"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(range(len(keys)), vals, color=PAL["optimum"], width=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(keys)), keys)
    ax.set_xlabel("First round at which b*5 gets more votes than b*0")
    ax.set_ylabel("Number of seeds")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"R4 (f): crossover round per seed (zealot world, n = {cr['n']})", loc="left", color=INK, fontsize=9)
    save(fig, "r4_crossover_hist.png")

    # (g) divided vs united
    sp = Z["scenario_split"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, g in zip(axes, ("divided", "united")):
        if "E_star" in sp[g]:
            panel_a(ax, sp[g], f"{g} worlds (n = {sp[g]['n']})")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"R4 (g): divided (two largest regions' archetypes TV ≥ {DIVIDED_TV}) vs united worlds",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, "r4_divided_vs_united.png")
    return saved


# ============================================================================== main
def main() -> None:
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--first-seed", type=int, default=12000)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default=xc.DEFAULT_DEVICE)
    ap.add_argument("--seed-batch", type=int, default=50)
    ap.add_argument("--parity-seeds", type=int, default=3)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        for p in plot_all(json.loads(JSON_PATH.read_text())):
            print("figure", p)
        return
    torch.set_num_threads(4)
    for d in (OUT, RAW_PATH.parent, FIG, LOG_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
    _LOG = LOG_PATH.open("w")
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    if device.type == "cuda" and device.index == 1:
        raise RuntimeError("GPU 1 must not be used")
    log(f"[r4] device {device} ({dev_record})")
    free0 = xc.free_bytes(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    start = time.perf_counter()
    rec: dict = {}
    with xc.GpuMonitor(device) as monitor:
        worlds = []
        for i, s in enumerate(seeds):
            worlds.append(build(s, device, base))
            if (i + 1) % 10 == 0:
                log(f"[r4] built {i + 1}/{len(seeds)} world pairs (+2 ORACLE manifestos) {time.perf_counter() - start:.0f}s")
        t_build = time.perf_counter() - start
        identity = all(all(v for k, v in w["check"].items() if isinstance(v, bool)) for w in worlds)
        if not identity:
            raise RuntimeError("populations are not paired")

        def run(chunk):
            r = process_batch(chunk)
            log(f"[r4] batch of {len(chunk)} seeds × {len(POPS)} populations done {time.perf_counter() - start:.0f}s")
            return r
        t0 = time.perf_counter()
        rows = xc.run_chunks(run, worlds, args.seed_batch, log=log, record=rec)
        t_main = time.perf_counter() - t0
        peak_main = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        RAW_PATH.write_text("")
        for r in rows:
            xc.write_row(RAW_PATH, r)
        del worlds
        t1 = time.perf_counter()
        par = parity(seeds[:args.parity_seeds], device, args.n) if args.parity_seeds else None
        log(f"[r4] parity passed: {par['passed'] if par else None} ({time.perf_counter() - t1:.0f}s)")
    seconds = time.perf_counter() - start
    deviations = [f"Design rounds = every recorded round {list(ROUNDS)} (a superset of the requested "
                  f"{list(REQUESTED_DESIGN)}), so E*(t) and regret are available at 6, 8 and 9 as well.",
                  "R1's label 'aligned' is called 'united' here (same threshold, same worlds)."]
    if device.type == "cuda" and device.index != 2:
        deviations.append(f"Device {device} instead of cuda:2.")
    # consistency with R1: b*_0 on the zealot world should reproduce R1's E*_0 (R4 adds one warm start = the mean)
    r1_check = None
    r1_raw = OUT / "raw" / "r1.jsonl"
    if r1_raw.exists():
        r1 = {r["seed"]: r["optimum"]["E_star"] for r in xc.read_rows(r1_raw)}
        diffs = [r["E"]["0"][0] - r1[r["seed"]] for r in rows if r["population"] == "zealots" and r["seed"] in r1]
        if diffs:
            r1_check = {"n": len(diffs), "max_abs_diff_E_star0_vs_r1": max(abs(d) for d in diffs),
                        "mean_diff": sum(diffs) / len(diffs)}
    lib = [r["E"]["0"][0] - r["E_library_b0"][0] for r in rows if r["population"] == "zealots"]
    v5 = max(r["v5_vs_anticipate_max_abs"] for r in rows if "v5_vs_anticipate_max_abs" in r)
    report = {
        "experiment": "R4", "question": "Does the best manifesto stay the best as opinions drift (no campaign)?",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds}, "n_seeds": len(seeds), "n": args.n,
        "rounds": list(ROUNDS), "design_rounds": list(ROUNDS), "requested_design_rounds": list(REQUESTED_DESIGN),
        "populations": {"zealots": "default zealot_fraction = 0.05 (primary)", "normal": "zealot_fraction = 0 (secondary)"},
        "sectors": list(SECTORS), "divided_threshold": DIVIDED_TV, "tv_move_threshold": TV_MOVE,
        "optimiser": {"method": "r1.pg_ascent (projected gradient as manifesto_opt.optimise_manifesto)", "step": 0.02,
                      "max_iterations": 400, "smoothing": 1e-3,
                      "starts": "7 library default_starts + top-3 + 19 R1 scenario variants on v(s) + warm start "
                                "b*_{previous s} (round 0: population mean) = 28",
                      "b_star": "best exact E over every iterate of every start"},
        "definitions": {"E[s,t]": "exact expected vote share of b*_s on the round-t electorate",
                        "E*(t)": "E[t,t], the re-optimised best at round t",
                        "regret[s,t]": "E*(t) − E[s,t] (vote share lost by keeping b*_s when the vote is at t)",
                        "crossover": "first recorded round at which E(b*_5) > E(b*_0)",
                        "TV": "½‖x − y‖₁",
                        "divided": f"TV between the archetype means of the two largest regions ≥ {DIVIDED_TV}"},
        "checks": {"v5_vs_oracle_anticipate_max_abs": v5, "r1_consistency": r1_check,
                   "E_star0_minus_library_b0": {"min": min(lib), "mean": sum(lib) / len(lib)},
                   "identity_check_all": identity},
        "summary": summarise(rows), "showcase": showcase(rows), "deviations": deviations, "config": base.to_record(),
        "timing": {"build_worlds_seconds": t_build, "drift_optimise_score_seconds": t_main},
        "provenance": xc.provenance_v3(args, device, seconds,
                                       batch_sizes={"seeds_per_batch": rec.get("batch_sizes"), "populations": len(POPS),
                                                    "starts_M": 28, "voters_N": args.n,
                                                    "design_rounds": len(ROUNDS),
                                                    "drift_layout": f"[{len(POPS)}, seeds·N, 5] on one disjoint-union graph",
                                                    "pg_layout": "[seeds, 28 starts, N, 5] per design round",
                                                    **{k: v for k, v in rec.items() if k != "batch_sizes"}},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       grad_mode="autograd (PG), no_grad (drift, scoring)",
                                       extra={"peak_bytes_main_run": peak_main}),
    }
    xc.save(str(JSON_PATH), report)
    for p in plot_all(report):
        log(f"[r4] figure {p}")
    pk = report["provenance"]["compute"]["peak_bytes"]
    log(f"[r4] done in {seconds:.1f}s; peak {pk / 2**30 if pk else 0:.2f} GiB")
    _LOG.close()


if __name__ == "__main__":
    main()
