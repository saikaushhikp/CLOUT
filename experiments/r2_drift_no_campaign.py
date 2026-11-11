"""R2 — Opinion drift with no campaign: a normal vs a zealot-included population.

Owner's question: "How does the voter opinion change without the influence of media and party workers, after a
certain number of rounds? Having the analysis on a normal voter population and a zealot-included one is
sufficient."

Design (paired):
  * seeds 10000–10049 (the XC worlds), N = 10⁴, default config otherwise;
  * per seed two populations from the SAME seed: "normal" (ζ = 0) and "zealots" (ζ = 0.05, the default). The
    zealot draw has its own seed stream, so regions, graph, v(0) and the susceptibilities of non-zealots are
    identical (checked and recorded in ``identity_check``); zealots have α = β = γ = 0 and never move;
  * social-only dynamics (``dynamics.social.aggregate`` + ``dynamics.update.step_preferences`` with no campaign,
    exactly ``strategies.oracle.anticipate``) for 200 rounds; every round is measured, the tables use rounds
    {0,1,2,3,5,10,20,50,100,200};
  * two FIXED manifestos per seed (same b for both populations, computed from the zealot world): b*_0 =
    ``optimise_manifesto`` on v(0) and the ORACLE manifesto ``oracle_manifesto(electorate, config)`` (T = 5);
  * long-run references: the Friedkin–Johnsen fixed point of the zealot world (zealots stubborn at v(0); solved
    by ``x2_stubborn.fj_solve``, CG on the GPU), and the consensus point of the zealot-free world
    (c = Σ π_i v_i(0), π_i ∝ d_i/a_i, the stationary law of the lazy heterogeneous-step DeGroot walk).

All 50 seeds × 2 populations run as one batch: the 50 graphs are joined into one disjoint-union graph
(``strategies.worlds._union_graph``) and the two populations are a leading batch dimension [2, 50·N, 5].

usage: OMP_NUM_THREADS=4 python experiments/r2_drift_no_campaign.py --seeds 50 --first-seed 10000 --rounds 200 \
           --device cuda:2      (add --plot-only to redraw the figures from r2.json)
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

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.dynamics.social import aggregate  # noqa: E402
from dumbocracy.dynamics.update import step_preferences  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import SECTORS  # noqa: E402
from dumbocracy.strategies.manifesto_opt import optimise_manifesto  # noqa: E402
from dumbocracy.strategies.oracle import anticipate, oracle_manifesto  # noqa: E402
from dumbocracy.strategies.worlds import _union_graph  # noqa: E402

OUT_DIR = xc.REPORTS / "experimentation"
OUT_JSON = OUT_DIR / "r2.json"
RAW_JSONL = OUT_DIR / "raw" / "r2.jsonl"
LOG = OUT_DIR / "logs" / "r2.log"
FIG_DIR = OUT_DIR / "figures"
POPS = ("normal", "zealots")
RECORD = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200)
PCA_ROUNDS = (0, 5, 20, 50, 200)
PCA_SAMPLE = 1500
MANIFESTOS = ("b0", "oracle")
SCALARS = ("disagreement", "within", "between", "disagreement_free", "mean_shift", "moved", "region_drift",
           "E_b0", "won_b0", "E_oracle", "won_oracle")
TESTED = ("disagreement", "within", "between", "E_b0", "won_b0", "E_oracle", "won_oracle")
DEFINITIONS = {
    "disagreement": "mean over all voters of TV(v_i, population mean) — the XC 'spread'",
    "within": "within-region spread: mean over all voters of TV(v_i, own region mean)",
    "between": "between-region dispersion: PLAIN (unweighted) mean of TV over the 6 pairs of region means",
    "disagreement_free": "disagreement restricted to the voters that are NOT zealots in the ζ = 0.05 world (same "
                         "voter set in both populations), distance to the full population mean",
    "mean_shift": "TV(population mean at t, population mean at 0)",
    "moved": "mean over voters of TV(v_i(t), v_i(0))",
    "region_drift": "mean over the 4 regions of TV(region mean at t, region mean at 0)",
    "E_<m>": "expected vote share mean σ(β(S_i − z)), S_i = 1 − TV(v_i, b), for fixed manifesto m",
    "won_<m>": "share of voters with S_i ≥ z = 0.825",
    "region_mean": "per-region mean preference (4 × 5), all voters of the region (zealots included)",
    "TV": "½‖x − y‖₁",
}

_LOG_HANDLE = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _LOG_HANDLE is not None:
        _LOG_HANDLE.write(msg + "\n")
        _LOG_HANDLE.flush()


# ============================================================================== metrics (unit-tested)
def tv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x - y).abs().sum(-1)


@torch.no_grad()
def drift_metrics(v: torch.Tensor, v0: torch.Tensor, region_of: torch.Tensor, b: torch.Tensor, free: torch.Tensor,
                  beta: float, z: float, regmean0: torch.Tensor | None = None,
                  popmean0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Metrics for P populations × B equal-sized worlds.

    v, v0: [P, B, n, K]; region_of: [B, n] in 0..3; b: [B, M, K] fixed manifestos; free: [B, n] bool voter mask
    for ``disagreement_free``. Returns scalars [P, B] (``E_<j>``/``won_<j>`` indexed by manifesto j = 0..M−1),
    ``region_mean`` [P, B, 4, K] and ``pop_mean`` [P, B, K]."""
    onehot = torch.nn.functional.one_hot(region_of, 4).to(v.dtype)          # [B, n, 4]
    counts = onehot.sum(1).clamp_min(1)                                      # [B, 4]
    regmean = torch.einsum("pbnk,bnr->pbrk", v, onehot) / counts[None, :, :, None]
    own = torch.einsum("bnr,pbrk->pbnk", onehot, regmean)
    popmean = v.mean(2)                                                      # [P, B, K]
    d_pop = tv(v, popmean[:, :, None])                                       # [P, B, n]
    pair = tv(regmean[:, :, :, None], regmean[:, :, None])                   # [P, B, 4, 4]
    iu = torch.triu_indices(4, 4, 1, device=v.device)
    fm = free.to(v.dtype)
    out = {"disagreement": d_pop.mean(-1),
           "within": tv(v, own).mean(-1),
           "between": pair[:, :, iu[0], iu[1]].mean(-1),
           "disagreement_free": (d_pop * fm).sum(-1) / fm.sum(-1).clamp_min(1),
           "moved": tv(v, v0).mean(-1),
           "mean_shift": tv(popmean, v0.mean(2) if popmean0 is None else popmean0),
           "region_drift": tv(regmean, regmean if regmean0 is None else regmean0).mean(-1),
           "region_mean": regmean, "pop_mean": popmean}
    s = 1.0 - tv(v[:, :, :, None], b[None, :, None])                         # [P, B, n, M]
    for j in range(b.shape[1]):
        out[f"E_{j}"] = torch.sigmoid(beta * (s[..., j] - z)).mean(-1)
        out[f"won_{j}"] = (s[..., j] >= z).to(v.dtype).mean(-1)
    return out


@torch.no_grad()
def social_run(v0: torch.Tensor, graph, weight: torch.Tensor, frozen: torch.Tensor, rounds: int, rule: str = "degroot",
               trim: int = 0, on_round=None) -> torch.Tensor:
    """``rounds`` steps of social-only dynamics (as ``oracle.anticipate``) for a leading batch [P, M, K] sharing one
    graph; ``weight`` and ``frozen`` are [P, M]. ``on_round(t, v)`` is called for t = 0..rounds."""
    empty = torch.zeros((0, v0.shape[-2]), dtype=v0.dtype, device=v0.device)
    no_b = torch.zeros((0, v0.shape[-1]), dtype=v0.dtype, device=v0.device)
    v = v0
    if on_round is not None:
        on_round(0, v)
    for t in range(1, rounds + 1):
        v = step_preferences(v, aggregate(v, graph, rule, trim), weight, empty, no_b, frozen)
        if on_round is not None:
            on_round(t, v)
    return v


@torch.no_grad()
def consensus_limit(v0: torch.Tensor, graph, weight: torch.Tensor, B: int, n: int) -> torch.Tensor:
    """Zealot-free long-run state: each world's non-isolated voters go to c = Σ π_i v_i(0), π_i ∝ d_i / a_i (left
    Perron vector of P = I − diag(a)(I − D⁻¹A)); isolated voters keep v_i(0). Assumes each world's non-isolated part
    is connected. v0: [B·n, K], weight: [B·n]. Returns [B·n, K] (float64)."""
    deg = graph.degree.to(torch.float64)
    pi = torch.where(graph.degree > 0, deg / weight.to(torch.float64).clamp_min(1e-12), torch.zeros_like(deg))
    pi = pi.view(B, n)
    pi = pi / pi.sum(1, keepdim=True)
    c = torch.einsum("bn,bnk->bk", pi, v0.to(torch.float64).view(B, n, -1))
    lim = c[:, None].expand(B, n, v0.shape[-1]).reshape(B * n, -1)
    return torch.where((graph.degree > 0)[:, None], lim, v0.to(torch.float64))


# ============================================================================== worlds
def world_pair(seed: int, device: torch.device, base: GameConfig) -> dict:
    cfg_z = dataclasses.replace(base, seed=seed)
    cfg_n = dataclasses.replace(cfg_z, electorate=dataclasses.replace(base.electorate, zealot_fraction=0.0))
    ez = generate_electorate(cfg_z, device=device)
    en = generate_electorate(cfg_n, device=device)
    zealot = ez.susceptibility.zealot.to(torch.bool)
    check = {"v0_equal": bool(torch.equal(ez.preferences, en.preferences)),
             "graph_equal": bool(torch.equal(ez.graph.indptr, en.graph.indptr) and torch.equal(ez.graph.indices, en.graph.indices)),
             "regions_equal": bool(torch.equal(ez.region_of, en.region_of)),
             "alpha_equal_on_non_zealots": bool(torch.equal(ez.susceptibility.social[~zealot], en.susceptibility.social[~zealot])),
             "zealot_alpha_zero": bool((ez.susceptibility.social[zealot] == 0).all()),
             "normal_has_no_zealots": int(en.susceptibility.zealot.sum()) == 0,
             "zealots": int(zealot.sum()), "isolated": int((ez.graph.degree == 0).sum())}
    b0 = optimise_manifesto(ez.preferences, cfg_z.voting.sensitivity, cfg_z.voting.threshold).manifesto
    bor = oracle_manifesto(ez, cfg_z).to(device)
    v5 = anticipate(ez, cfg_z)
    return {"seed": seed, "cfg": cfg_z, "ez": ez, "en": en, "b": torch.stack([b0, bor]).to(ez.preferences.dtype),
            "check": check, "v5_ref": v5}


@torch.no_grad()
def simulate_block(worlds: list[dict], rounds: int, *, showcase: int | None = None, fj: bool = True) -> dict:
    """Run one batch of worlds (both populations at once). Returns CPU tensors: ``series`` {metric: [T+1, P, B]},
    ``region_mean`` [T+1, P, B, 4, K], ``pca`` (showcase), ``fj``/``limit`` metrics, ``v5_maxdiff``."""
    cfg = worlds[0]["cfg"]
    B, n = len(worlds), worlds[0]["ez"].num_voters
    es_z = [w["ez"] for w in worlds]
    graph = es_z[0].graph if B == 1 else _union_graph([e.graph for e in es_z], n)
    dev = graph.indices.device
    v0 = torch.cat([e.preferences for e in es_z])                                        # [B·n, K] identical in both
    zealot = torch.cat([e.susceptibility.zealot.to(torch.bool) for e in es_z])
    w_n = cfg.dynamics.social_step * torch.cat([w["en"].susceptibility.social for w in worlds])
    w_z = cfg.dynamics.social_step * torch.cat([e.susceptibility.social for e in es_z])
    weight = torch.stack([w_n, w_z])                                                     # [2, B·n]
    frozen = torch.stack([torch.zeros_like(zealot), zealot])
    region_of = torch.stack([e.region_of for e in es_z])                                 # [B, n]
    b = torch.stack([w["b"] for w in worlds])                                            # [B, M, K]
    free = ~zealot.view(B, n)
    beta, z = cfg.voting.sensitivity, cfg.voting.threshold
    V0 = torch.stack([v0, v0])                                                           # [2, B·n, K]
    V0b = V0.view(2, B, n, -1)
    base = drift_metrics(V0b, V0b, region_of, b, free, beta, z)
    reg0, pop0 = base["region_mean"], base["pop_mean"]

    series: dict[str, list] = {k: [] for k in SCALARS}
    regs, popm = [], []
    pca_states: dict[int, torch.Tensor] = {}
    v5_maxdiff = []

    def on_round(t: int, v: torch.Tensor) -> None:
        vb = v.view(2, B, n, -1)
        m = drift_metrics(vb, V0b, region_of, b, free, beta, z, reg0, pop0)
        for k in ("disagreement", "within", "between", "disagreement_free", "mean_shift", "moved", "region_drift"):
            series[k].append(m[k])
        for j, name in enumerate(MANIFESTOS):
            series[f"E_{name}"].append(m[f"E_{j}"])
            series[f"won_{name}"].append(m[f"won_{j}"])
        regs.append(m["region_mean"])
        popm.append(m["pop_mean"])
        if showcase is not None and t in PCA_ROUNDS:
            pca_states[t] = vb[:, showcase].detach().cpu()
        if t == cfg.rounds:   # the zealot population's prefix must equal oracle.anticipate (the game's T = 5)
            ref = torch.stack([w["v5_ref"] for w in worlds])
            v5_maxdiff.append(float((vb[1] - ref).abs().max()))

    vT = social_run(V0, graph, weight, frozen, rounds, cfg.dynamics.social_rule, cfg.dynamics.wmsr_trim, on_round)
    out = {"series": {k: torch.stack(x).cpu() for k, x in series.items()},
           "region_mean": torch.stack(regs).cpu(), "pop_mean": torch.stack(popm).cpu(), "pca_states": pca_states,
           "v5_maxdiff": max(v5_maxdiff) if v5_maxdiff else None}

    # long-run references, scored with the same metrics (float64)
    def score(vstar: torch.Tensor) -> dict:
        vs = vstar.to(torch.float64).view(1, B, n, -1)
        m = drift_metrics(vs, V0b[:1].double(), region_of, b.double(), free, beta, z, reg0[:1].double(), pop0[:1].double())
        res = {k: m[k][0].cpu() for k in ("disagreement", "within", "between", "disagreement_free", "mean_shift",
                                           "moved", "region_drift")}
        for j, name in enumerate(MANIFESTOS):
            res[f"E_{name}"], res[f"won_{name}"] = m[f"E_{j}"][0].cpu(), m[f"won_{j}"][0].cpu()
        res["region_mean"] = m["region_mean"][0].cpu()
        res["pop_mean"] = m["pop_mean"][0].cpu()
        return res

    lim = consensus_limit(v0, graph, w_n, B, n)
    out["limit_normal"] = score(lim)
    out["limit_normal"]["max_tv_vT_to_limit"] = tv(vT[0].double(), lim).view(B, n).max(1).values.cpu()
    if fj:
        from x2_stubborn import fj_solve
        vstar, info = fj_solve([{"es": {"z": e}} for e in es_z], "z", torch.zeros(B * n, dtype=torch.bool, device=dev),
                               torch.zeros(B * n, v0.shape[-1], dtype=torch.float64, device=dev))
        out["fj_zealots"] = score(vstar)
        out["fj_zealots"]["max_tv_vT_to_fj"] = tv(vT[1].double(), vstar).view(B, n).max(1).values.cpu()
        out["fj_info"] = info
    return out


# ============================================================================== run
def _r(x: float, digits: int = 6) -> float:
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def parity_fn(base: GameConfig, rounds: int):
    def fn(seed: int, device: torch.device) -> dict:
        w = world_pair(seed, device, dataclasses.replace(base, device=str(device)))
        r = simulate_block([w], rounds, fj=False)
        out = {}
        for p, pop in enumerate(POPS):
            for k in ("disagreement", "within", "between", "E_b0", "E_oracle"):
                out[f"{pop}.{k}@{rounds}"] = float(r["series"][k][rounds, p, 0])
        return out
    return fn


def run(args) -> dict:
    global _LOG_HANDLE
    t_start = time.time()
    device, dev_record = xc.select_device(args.device, need_bytes=3 * 2**30)
    if device.type == "cuda" and device.index == 1:
        raise RuntimeError("GPU 1 must not be used")
    free_at_start = xc.free_bytes(device)
    torch.set_num_threads(4)
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    T = args.rounds
    record = [t for t in RECORD if t <= T]
    log(f"[r2] device {device} ({dev_record}); seeds {seeds[0]}..{seeds[-1]}; N={args.n}; T={T}; batch {args.batch}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with xc.GpuMonitor(device) as monitor:
        t0 = time.time()
        worlds = []
        for k, seed in enumerate(seeds):
            worlds.append(world_pair(seed, device, base))
            if (k + 1) % 10 == 0:
                log(f"[r2] built {k + 1}/{len(seeds)} world pairs + manifestos, {time.time() - t0:.0f}s")
        t_build = time.time() - t0
        identity = {key: all(w["check"][key] for w in worlds) for key in
                    ("v0_equal", "graph_equal", "regions_equal", "alpha_equal_on_non_zealots", "zealot_alpha_zero",
                     "normal_has_no_zealots")}
        log(f"[r2] identity check (normal vs zealot world): {identity}")
        if not all(identity.values()):
            raise RuntimeError(f"populations are not paired: {identity}")

        t1 = time.time()
        rec: dict = {}
        blocks = xc.run_chunks(lambda chunk: [simulate_block(chunk, T, showcase=0 if chunk[0] is worlds[0] else None)],
                               worlds, args.batch, log=log, record=rec)
        t_sim = time.time() - t1
        log(f"[r2] simulated {T} rounds for {len(seeds)} seeds × 2 populations in {t_sim:.1f}s "
            f"(batch sizes {rec.get('batch_sizes')})")

    series = {k: torch.cat([bk["series"][k] for bk in blocks], dim=2) for k in SCALARS}          # [T+1, P, S]
    region_mean = torch.cat([bk["region_mean"] for bk in blocks], dim=2)                          # [T+1, P, S, 4, K]
    pop_mean = torch.cat([bk["pop_mean"] for bk in blocks], dim=2)
    lim = {k: torch.cat([bk["limit_normal"][k] for bk in blocks], dim=0) for k in blocks[0]["limit_normal"]}
    fj = {k: torch.cat([bk["fj_zealots"][k] for bk in blocks], dim=0) for k in blocks[0]["fj_zealots"]}
    fj_info = [bk["fj_info"] for bk in blocks]
    v5_maxdiff = max(bk["v5_maxdiff"] for bk in blocks)
    log(f"[r2] T=5 prefix vs oracle.anticipate: max |Δv| = {v5_maxdiff:.2e}; FJ: {fj_info}")

    # ---------------------------------------------------------------- raw rows (one per seed × population)
    RAW_JSONL.parent.mkdir(parents=True, exist_ok=True)
    RAW_JSONL.write_text("")
    for s, w in enumerate(worlds):
        for p, pop in enumerate(POPS):
            row = {"seed": w["seed"], "population": pop, "zealot_fraction": 0.0 if p == 0 else base.electorate.zealot_fraction,
                   "zealots": 0 if p == 0 else w["check"]["zealots"], "isolated": w["check"]["isolated"],
                   "b0": [_r(x) for x in w["b"][0].tolist()], "b_oracle": [_r(x) for x in w["b"][1].tolist()],
                   "series": {k: [_r(x) for x in series[k][:, p, s].tolist()] for k in SCALARS},
                   "region_mean": {str(t): [[_r(x) for x in row4] for row4 in region_mean[t, p, s].tolist()] for t in record},
                   "pop_mean": {str(t): [_r(x) for x in pop_mean[t, p, s].tolist()] for t in record},
                   "sizes": [int(x) for x in w["ez"].sizes.tolist()],
                   "archetype_of_region": [int(x) for x in w["ez"].archetype_of_region.tolist()]}
            if p == 0:
                row["limit"] = {k: _r(lim[k][s]) for k in lim if k not in ("region_mean", "pop_mean")}
            else:
                row["fj"] = {k: _r(fj[k][s]) for k in fj if k not in ("region_mean", "pop_mean")}
            xc.write_row(RAW_JSONL, row)

    # ---------------------------------------------------------------- summary statistics
    S = len(seeds)
    per_round = {pop: {k: {str(t): xc.mean_ci(series[k][t, p].tolist()) for t in record} for k in SCALARS}
                 for p, pop in enumerate(POPS)}
    dense = {pop: {k: [[_r(c["mean"]), _r(c["ci95"][0]), _r(c["ci95"][1])] for c in
                       (xc.mean_ci(series[k][t, p].tolist()) for t in range(T + 1))]
                   for k in ("disagreement", "within", "between", "disagreement_free", "E_b0", "won_b0", "E_oracle",
                             "won_oracle", "region_drift", "mean_shift", "moved")}
             for p, pop in enumerate(POPS)}
    tests, pvals = [], []
    for k in TESTED:
        for t in record[1:]:
            d = xc.paired(series[k][t, 1].tolist(), series[k][t, 0].tolist())
            tests.append({"metric": k, "round": t, "diff_zealots_minus_normal": d})
            pvals.append(d["wilcoxon_p"])
    for test, adj in zip(tests, xc.holm(pvals)):
        test["holm_p"] = adj
    identical_r0 = {k: float((series[k][0, 1] - series[k][0, 0]).abs().max()) for k in TESTED}

    # long-run floor: decay over the last half of the run, relative to the FJ floor
    def slope(p: int, t_a: int, t_b: int, floor: torch.Tensor | None = None) -> list[float]:
        a, bb = series["disagreement"][t_a, p].double(), series["disagreement"][t_b, p].double()
        if floor is not None:
            a, bb = a - floor, bb - floor
        return [((math.log(y) - math.log(x)) / (t_b - t_a)) if x > 0 and y > 0 else float("nan")
                for x, y in zip(a.tolist(), bb.tolist())]
    t_half = T // 2
    floor = {"window": [t_half, T]}
    for p, pop in enumerate(POPS):
        rate = slope(p, t_half, T)
        floor[pop] = {"log_slope_per_round": xc.mean_ci(rate),
                      "factor_per_round": xc.mean_ci([math.exp(r) for r in rate]),
                      "half_life_rounds": xc.mean_ci([math.log(2) / -r if r < 0 else float("nan") for r in rate]),
                      f"ratio_d{T}_over_d{t_half}": xc.mean_ci((series["disagreement"][T, p] / series["disagreement"][t_half, p]).tolist())}
    fjd = fj["disagreement"].float()
    floor["zealots"]["fj_disagreement"] = xc.mean_ci(fjd.tolist())
    floor["zealots"][f"ratio_d{T}_over_fj"] = xc.mean_ci((series["disagreement"][T, 1] / fjd).tolist())
    floor["zealots"][f"ratio_d{t_half}_over_fj"] = xc.mean_ci((series["disagreement"][t_half, 1] / fjd).tolist())
    excess = slope(1, t_half, T, fjd.double())
    floor["zealots"]["excess_over_fj_log_slope_per_round"] = xc.mean_ci(excess)
    floor["zealots"]["excess_over_fj_half_life_rounds"] = xc.mean_ci([math.log(2) / -r if r < 0 else float("nan") for r in excess])
    floor["zealots"]["max_tv_vT_to_fj"] = xc.mean_ci(fj["max_tv_vT_to_fj"].tolist())
    floor["normal"]["limit_disagreement"] = xc.mean_ci(lim["disagreement"].tolist())
    floor["normal"]["max_tv_vT_to_limit"] = xc.mean_ci(lim["max_tv_vT_to_limit"].tolist())
    floor["normal"]["tv_popmean_T_to_consensus"] = xc.mean_ci(
        tv(pop_mean[T, 0].double(), lim["pop_mean"].double()).tolist())
    limits = {"normal_consensus": {k: xc.mean_ci(lim[k].tolist()) for k in ("disagreement", "within", "between",
                                                                            "E_b0", "won_b0", "E_oracle", "won_oracle",
                                                                            "mean_shift", "region_drift")},
              "zealots_fj": {k: xc.mean_ci(fj[k].tolist()) for k in ("disagreement", "within", "between", "E_b0",
                                                                     "won_b0", "E_oracle", "won_oracle", "mean_shift",
                                                                     "region_drift")},
              "fj_solver": fj_info,
              "definitions": {"normal_consensus": "c = Σ π_i v_i(0), π_i ∝ d_i/a_i; isolated voters keep v_i(0)",
                              "zealots_fj": "Friedkin–Johnsen fixed point, zealots stubborn at v(0) (x2_stubborn.fj_solve, "
                                            "Jacobi-CG on the GPU, float64)"}}
    # where the zealot world's long-run state sits: E at FJ vs E at round T (paired)
    limits["zealots_E_oracle_T_vs_fj"] = xc.paired(series["E_oracle"][T, 1].tolist(), fj["E_oracle"].tolist())

    # ---------------------------------------------------------------- showcase (seed index 0): region lines + PCA
    show = worlds[0]
    pca = _pca_showcase(show, blocks[0]["pca_states"], series, region_mean)
    showcase = {"seed": show["seed"], "sizes": [int(x) for x in show["ez"].sizes.tolist()],
                "archetype_of_region": [int(x) for x in show["ez"].archetype_of_region.tolist()],
                "region_mean": {pop: [[[_r(x) for x in r4] for r4 in region_mean[t, p, 0].tolist()] for t in range(T + 1)]
                                for p, pop in enumerate(POPS)},
                "fj_region_mean": [[_r(x) for x in r4] for r4 in fj["region_mean"][0].tolist()],
                "limit_region_mean": [[_r(x) for x in r4] for r4 in lim["region_mean"][0].tolist()],
                "zealots_by_region": [int(x) for x in torch.bincount(show["ez"].region_of[show["ez"].susceptibility.zealot.bool()], minlength=4).tolist()],
                "pca": pca}

    seconds = time.time() - t_start
    prov = xc.provenance_v3(args, device, seconds, batch_sizes=rec.get("batch_sizes"), free_at_start=free_at_start,
                            monitor=monitor, device_record=dev_record,
                            extra={"timing_s": {"build_worlds_and_manifestos": t_build, "simulate_and_references": t_sim},
                                   "batch_record": rec,
                                   "batch_layout": f"[2 populations, {args.batch} worlds × N voters, 5] on one disjoint-union graph"})
    return {"experiment": "R2", "title": "Opinion drift with no campaign: normal vs zealot population",
            "question": "How does voter opinion change without media and party workers over many rounds, in a normal "
                        "and a zealot-included population?",
            "seeds": {"first": seeds[0], "count": S, "list": seeds}, "n": args.n, "rounds": T, "recorded_rounds": record,
            "populations": {"normal": "zealot_fraction = 0", "zealots": f"zealot_fraction = {base.electorate.zealot_fraction}"},
            "config": base.to_record(), "sectors": list(SECTORS), "manifestos": {
                "b0": "optimise_manifesto on v(0) of the zealot world (round-0 optimum)",
                "oracle": "oracle_manifesto(electorate, config): optimum on the T = 5 social-only forecast of the zealot world"},
            "metric_definitions": DEFINITIONS, "identity_check": identity,
            "identity_per_seed": [{"seed": w["seed"], **w["check"]} for w in worlds],
            "t5_prefix_vs_anticipate_max_abs": v5_maxdiff,
            "zealots": xc.mean_ci([w["check"]["zealots"] for w in worlds]),
            "per_round": per_round, "dense": dense, "tests": {"family": f"{len(TESTED)} metrics × {len(record) - 1} rounds, Holm",
                                                           "round0_max_abs_diff": identical_r0, "rows": tests},
            "floor": floor, "limits": limits, "showcase": showcase, "provenance": prov}


def _pca_showcase(show: dict, states: dict[int, torch.Tensor], series, region_mean) -> dict:
    e = show["ez"]
    g = torch.Generator().manual_seed(show["seed"])
    idx = torch.randperm(e.num_voters, generator=g)[:PCA_SAMPLE]
    x0 = states[0][0].double()
    mu = x0.mean(0)
    _, sv, V = torch.linalg.svd(x0 - mu, full_matrices=False)
    P = V[:2].T
    proj = lambda x: [[_r(a, 5), _r(b, 5)] for a, b in ((x.double() - mu) @ P).tolist()]  # noqa: E731
    out = {"rounds": sorted(states), "region": e.region_of.cpu()[idx].tolist(),
           "zealot": e.susceptibility.zealot.cpu()[idx].tolist(),
           "explained_variance": (sv[:2] ** 2 / (sv ** 2).sum()).tolist(),
           "b": proj(show["b"].cpu()), "points": {}}
    for p, pop in enumerate(POPS):
        out["points"][pop] = {str(t): proj(states[t][p][idx]) for t in sorted(states)}
    return out


# ============================================================================== figures (from JSON only)
def plot(d: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_v3 import INK2, SERIES, _style, save_figure
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    T = d["rounds"]
    ts = list(range(T + 1))
    pop_color = {"normal": SERIES[0], "zealots": SERIES[1]}
    pop_label = {"normal": "normal (no zealots)", "zealots": "zealots (ζ = 5%)"}
    reg_colors = [SERIES[2], SERIES[3], SERIES[6], SERIES[7]]
    rname = ["North", "South", "East", "West"]

    def xaxis(ax):
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, T)
        ticks = [t for t in (0, 1, 2, 5, 10, 20, 50, 100, 200) if t <= T]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel("round (log scale)")

    def band(ax, rows, color, label, ls="-"):
        m = [r[0] for r in rows]
        ax.plot(ts, m, color=color, lw=1.6, ls=ls, label=label)
        ax.fill_between(ts, [r[1] for r in rows], [r[2] for r in rows], color=color, alpha=0.18, lw=0)

    # (a) region means per sector, showcase seed; + region drift over all seeds
    sc = d["showcase"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    for k, ax in enumerate(axes.flat[:5]):
        for r in range(4):
            for pop, ls in (("normal", "-"), ("zealots", "--")):
                ax.plot(ts, [sc["region_mean"][pop][t][r][k] for t in ts], color=reg_colors[r], ls=ls, lw=1.5)
            ax.plot([T], [sc["fj_region_mean"][r][k]], marker="o", ms=5, mfc="none", mec=reg_colors[r], clip_on=False)
        ax.set_title(f"{d['sectors'][k]}: region mean preference", fontsize=10)
        xaxis(ax)
        _style(ax)
    for r in range(4):
        axes[0, 0].plot([], [], color=reg_colors[r], lw=2, label=f"{rname[r]} (n = {sc['sizes'][r]:,})")
    axes[0, 0].plot([], [], color=INK2, ls="-", label="normal")
    axes[0, 0].plot([], [], color=INK2, ls="--", label="zealots")
    axes[0, 0].plot([], [], color=INK2, marker="o", mfc="none", ls="", label="FJ long-run (zealots)")
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=1)
    ax = axes[1, 2]
    for pop in ("normal", "zealots"):
        band(ax, d["dense"][pop]["region_drift"], pop_color[pop], pop_label[pop])
    ax.set_title(f"Region means: TV from round 0 (all {d['seeds']['count']} seeds)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    xaxis(ax)
    _style(ax)
    fig.suptitle(f"R2 (a). Showcase seed {sc['seed']}: each region's mean preference by sector, no campaign",
                 fontsize=11, x=0.01, ha="left")
    out.append(save_figure(fig, out_dir / "r2_region_sectors.png"))

    # (b) disagreement, within, between
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    titles = {"disagreement": "Overall disagreement (mean TV to population mean)",
              "within": "Within-region spread (mean TV to own region mean)",
              "between": "Between-region dispersion (mean pairwise TV of region means)"}
    for ax, k in zip(axes, titles):
        for pop in ("normal", "zealots"):
            band(ax, d["dense"][pop][k], pop_color[pop], pop_label[pop])
        fjm = d["limits"]["zealots_fj"][k]["mean"]
        ax.axhline(fjm, color=pop_color["zealots"], ls=":", lw=1.1)
        ax.annotate(f"FJ floor {fjm:.3f}", (1, fjm), xytext=(2, 3), textcoords="offset points", fontsize=7.5,
                    color=pop_color["zealots"])
        ax.set_yscale("log")
        ax.set_title(titles[k], fontsize=10)
        xaxis(ax)
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"R2 (b). Disagreement without a campaign ({d['seeds']['count']} seeds, N = {d['n']:,}, mean and 95% CI; "
                 "log y)", fontsize=11, x=0.01, ha="left")
    out.append(save_figure(fig, out_dir / "r2_disagreement.png"))

    # (c) PCA snapshots
    pca = sc["pca"]
    rounds = pca["rounds"]
    fig, axes = plt.subplots(2, len(rounds), figsize=(3.2 * len(rounds), 6.8), sharex=True, sharey=True)
    zmask = pca["zealot"]
    for i, pop in enumerate(("normal", "zealots")):
        for j, t in enumerate(rounds):
            ax = axes[i, j]
            pts = pca["points"][pop][str(t)]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=3, c=[reg_colors[r] for r in pca["region"]],
                       alpha=0.5, lw=0)
            if pop == "zealots":
                zp = [p for p, f in zip(pts, zmask) if f]
                ax.scatter([p[0] for p in zp], [p[1] for p in zp], s=9, marker="x", color="#0b0b0b", lw=0.6)
            for m, mk in zip(pca["b"], ("*", "P")):
                ax.scatter([m[0]], [m[1]], marker=mk, s=110, facecolor="none", edgecolor=INK2, lw=1.0, zorder=5)
            ax.set_title(f"{pop_label[pop]}, round {t}", fontsize=9)
            _style(ax)
            if j == 0:
                ax.set_ylabel("PC 2")
            if i == 1:
                ax.set_xlabel("PC 1")
    for r in range(4):
        axes[0, 0].scatter([], [], s=12, color=reg_colors[r], label=rname[r])
    axes[0, 0].scatter([], [], marker="x", color="#0b0b0b", label="zealot")
    axes[0, 0].scatter([], [], marker="*", s=60, facecolor="none", edgecolor=INK2, label="b*₀")
    axes[0, 0].scatter([], [], marker="P", s=60, facecolor="none", edgecolor=INK2, label="ORACLE b")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"R2 (c). Seed {sc['seed']}: {len(pca['region']):,} sampled voters in the round-0 PCA plane "
                 f"(PC1+PC2 explain {100 * sum(pca['explained_variance']):.0f}% of v(0) variance)",
                 fontsize=11, x=0.01, ha="left")
    out.append(save_figure(fig, out_dir / "r2_pca.png"))

    # (d) E and won-over share for the two fixed manifestos
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, kind, title in ((axes[0], "E", "Expected vote share E"), (axes[1], "won", "Voters won over (S ≥ z)")):
        for pop in ("normal", "zealots"):
            for mname, ls, lab in (("b0", "-", "b*₀ (round-0 optimum)"), ("oracle", "--", "ORACLE b (T = 5 forecast)")):
                band(ax, d["dense"][pop][f"{kind}_{mname}"], pop_color[pop], f"{pop_label[pop]}, {lab}", ls=ls)
        ax.axvline(5, color=INK2, lw=0.7, ls=":")
        ax.text(5.2, ax.get_ylim()[1], "game ends (T = 5)", fontsize=7.5, color=INK2, va="top")
        ax.set_title(title, fontsize=10)
        xaxis(ax)
        _style(ax)
    axes[0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle("R2 (d). Fixed manifestos scored on the drifting electorate (no campaign; mean and 95% CI)",
                 fontsize=11, x=0.01, ha="left")
    out.append(save_figure(fig, out_dir / "r2_manifesto_share.png"))
    return out


# ============================================================================== main
def main() -> None:
    global _LOG_HANDLE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=10000)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=50, help="worlds per GPU batch (both populations each)")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--parity-seeds", type=int, default=3)
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.rounds < max(PCA_ROUNDS):
        raise SystemExit(f"--rounds must be >= {max(PCA_ROUNDS)}")
    if not args.plot_only:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        _LOG_HANDLE = LOG.open("w")
        d = run(args)
        if args.parity_seeds > 0:
            t0 = time.time()
            base = GameConfig(electorate=ElectorateConfig(num_voters=args.n))
            seeds = list(range(args.first_seed, args.first_seed + args.parity_seeds))
            par = xc.parity_check(parity_fn(base, 50), seeds, device=d["provenance"]["compute"]["device"])
            par["rounds"] = 50
            par["seconds"] = time.time() - t0
            d["provenance"]["parity"] = par
            log(f"[r2] CPU↔GPU parity on {seeds}: passed={par['passed']} "
                f"(max |mean diff| {max(abs(m['mean_diff']) for m in par['metrics'].values()):.2e})")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(d, indent=1, default=xc._json_default))
        log(f"[r2] saved {args.out}; total {d['provenance']['compute']['wall_seconds']:.1f}s, "
            f"peak {d['provenance']['compute']['peak_bytes']} bytes")
    d = json.loads(Path(args.out).read_text())
    for p in plot(d, FIG_DIR):
        log(f"[r2] wrote {p}")
    if _LOG_HANDLE is not None:
        _LOG_HANDLE.close()


if __name__ == "__main__":
    main()
