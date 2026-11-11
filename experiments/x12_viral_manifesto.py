"""X12 — Broad vs narrow ("viral") manifesto (plan ``docs/v2/plans/implementation_plan.md`` §5 X12).

Command (reproduces reports/v3/x12.json, reports/v3/raw/x12.jsonl and reports/v3/logs/x12.log; run after X13):
  OMP_NUM_THREADS=4 python experiments/x12_viral_manifesto.py --seeds 50 --joint-seeds 100 --first-seed 12000 \
      --device cuda:0 --out reports/v3/x12.json 2>&1 | tee reports/v3/logs/x12.log

Interpolation arm (50 seeds):
* b₀ = ORACLE manifesto (``oracle_manifesto``, tuned to the anticipated true electorate v_T);
* coalition blend b_C for every 2- and 3-region coalition C with > 50 % of voters: the size-weighted mean of the
  TRUE anticipated regional means of v_T;
* b(λ) = (1 − λ) b₀ + λ b_C, λ ∈ {0, 0.25, 0.5, 0.75, 1};
* (a) every (C, λ) is played by the ORACLE planner (``balance.policy_planner``, closed loop on the true state);
  (b) for the best coalition C* of each seed (highest mean ORACLE-planner E over λ > 0) every λ is also played by
  X13's open-loop plan search (``x13_plan_search.search_plan``, same settings as X13);
* every game is evaluated on the same 20 fresh stream sets as X13 (``SaltedStreams`` salts 1..20, ``run_batch``);
* recorded: E, wins, cascade reach (converts) per region, and per region λ_max(A_rr D_q) with
  q_j = γ_j P_j(v(0), b) — the branching factor of the worker cascade inside region r (symmetrised
  D_q^{1/2} A_rr D_q^{1/2}, power iteration on the GPU).

Joint-optimisation arm (100 seeds; campaign-aware b): start from b₀ and the ORACLE planner's executed plan on the
canonical streams; alternate 3 times (i) fix the plan (the realised worker picks per round and the media schedule)
and optimise b by projected gradient on the EXPECTED-VALUE campaign — conversions replaced by their probability
1 − (1 − γP)^{E[count]} (differentiable in b through γP), expected media intensity p_m η β, workers pinned at b,
the social step of the game — then (ii) re-plan with the new b (ORACLE planner, canonical streams). The final b is
played by the ORACLE planner and by the X13 search on the 20 fresh streams and compared with "b first, then plan"
(b₀; for the search arm the X13 per-seed result, same code, seeds and streams).
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
import x_policies  # noqa: E402
import x13_plan_search as x13  # noqa: E402
from x_batch import run_batch  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.manifesto_opt import expected_share, project_simplex  # noqa: E402
from dumbocracy.strategies.oracle import anticipate, oracle_manifesto  # noqa: E402

RAW_PATH = xc.RAW / "x12.jsonl"
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FRESH = 20


# ============================================================================== coalitions and spectra
def coalitions(sizes: list[int]) -> list[tuple[int, ...]]:
    total = sum(sizes)
    return [c for k in (2, 3) for c in itertools.combinations(range(4), k) if 2 * sum(sizes[r] for r in c) > total]


def blend(v_t: torch.Tensor, region_of: torch.Tensor, sizes: list[int], c: tuple[int, ...]) -> torch.Tensor:
    means = torch.stack([v_t[region_of == r].mean(0) for r in c])
    w = torch.tensor([sizes[r] for r in c], dtype=v_t.dtype, device=v_t.device)
    return (w.unsqueeze(-1) * means).sum(0) / w.sum()


def region_branching(e, b: torch.Tensor, cfg: GameConfig, iterations: int = 300) -> list[float]:
    """λ_max(A_rr D_q) per region, q_j = γ_j P_j(v(0), b), via power iteration on the block-diagonal symmetric
    matrix D_q^{1/2} A_intra D_q^{1/2} (each region block normalised separately)."""
    g, region = e.graph, e.region_of
    rows, cols = g.rows(), g.indices
    keep = region[rows] == region[cols]
    r_, c_ = rows[keep], cols[keep]
    v0 = e.preferences
    sim = 1 - 0.5 * (v0 - b).abs().sum(-1)
    prob = torch.sigmoid(cfg.voting.sensitivity * (sim - cfg.voting.threshold))
    q = (e.susceptibility.campaign * prob).clamp(0, 1).to(torch.float64)
    q = torch.where(e.susceptibility.zealot.to(torch.bool), torch.zeros_like(q), q)
    w = torch.sqrt(q[r_] * q[c_])
    n = v0.shape[0]
    x = torch.ones(n, dtype=torch.float64, device=v0.device)
    lam = torch.zeros(4, dtype=torch.float64, device=v0.device)
    for _ in range(iterations):
        y = torch.zeros_like(x).index_add_(0, r_, w * x[c_])
        num = torch.zeros(4, dtype=x.dtype, device=x.device).index_add_(0, region, x * y)
        den = torch.zeros(4, dtype=x.dtype, device=x.device).index_add_(0, region, x * x)
        lam = num / den.clamp_min(1e-300)
        norm = torch.sqrt(torch.zeros(4, dtype=x.dtype, device=x.device).index_add_(0, region, y * y)).clamp_min(1e-300)
        x = y / norm[region]
    return [float(v) for v in lam.tolist()]


# ============================================================================== evaluation helpers
def play_oracle(e, cfg: GameConfig, manifestos: torch.Tensor, streams: list) -> list[dict]:
    """ORACLE planner for each manifesto row on every stream set: one ``run_batch`` over M × S copies; returns per
    manifesto {E mean, win rate, converts per region (mean over streams)}."""
    M, S = manifestos.shape[0], len(streams)
    reach = torch.zeros(M * S, 4, dtype=torch.float64, device=e.device)

    def chunk(idx: list[int]) -> list:
        # manifestos idx[0]..idx[-1] (contiguous), all streams; the observer indexes the union by global world id
        lo = idx[0]
        def obs_shift(t, info):
            conv = info["converted"]
            w = torch.arange(conv.numel(), device=e.device) // e.num_voters + lo * S
            reach.view(-1).index_add_(0, (w * 4 + e.region_of.repeat(len(idx) * S))[conv],
                                      torch.ones(int(conv.sum()), dtype=torch.float64, device=e.device))
        out = run_batch([e] * (len(idx) * S), cfg, manifestos[idx].repeat_interleave(S, 0), x_policies.policy_planner,
                        streams=streams * len(idx), observer=obs_shift)
        return [out[i * S:(i + 1) * S] for i in range(len(idx))]

    # OOM back-off over manifestos (§1.1 point 3); 12 manifestos × 20 streams ≈ 2.4M voters per call
    grouped = xc.run_chunks(chunk, list(range(M)), max(1, min(M, 12)))
    outs = [o for g in grouped for o in g]
    res = []
    reach = reach.view(M, S, 4).mean(1).cpu().tolist()
    for m in range(M):
        os_ = outs[m * S:(m + 1) * S]
        res.append({"E": float(np.mean([o.result.expected_share for o in os_])),
                    "win": float(np.mean([o.result.victory for o in os_])),
                    "E_streams": [o.result.expected_share for o in os_], "reach_by_region": reach[m],
                    "workers": float(np.mean([sum(sum(a.hires_by_region) for a in o.allocations) for o in os_])),
                    "airings": float(np.mean([sum(sum(a.media_regions) for a in o.allocations) for o in os_]))})
    return res


def play_search(e, cfg: GameConfig, b: torch.Tensor, seed: int, streams: list, settings) -> dict:
    warm = x13.warm_plans(e, cfg, b, seed)
    found = x13.search_plan(e, cfg, b, seed, [(h, m) for h, m, _ in warm.values()], settings)
    ev = x13.evaluate_policies(e, cfg, b, {"search": x13.policy_of(found["hires"], found["media"])}, streams)["search"]
    pricing = x13.Pricing([int(x) for x in e.sizes.tolist()], cfg)
    return {"E": float(np.mean(ev["expected"])), "win": float(np.mean(ev["victory"])), "E_streams": ev["expected"],
            "plan": {"hires": found["hires"].tolist(), "media": found["media"].astype(int).tolist()},
            "features": x13.plan_features(found["hires"], found["media"], pricing), "seconds": found["seconds"]}


# ============================================================================== expected-value campaign (joint arm)
def oracle_canonical_plan(e, cfg: GameConfig, b: torch.Tensor, seed: int) -> dict:
    """The ORACLE planner's executed plan on the canonical streams, with the realised worker picks per round."""
    picks = []

    def observer(t, info):
        picks.append(info["new"].clone())

    o = run_batch([e], cfg, b.reshape(1, -1), x_policies.policy_planner, seeds=[seed], observer=observer)[0]
    return {"new_workers": picks, "media": [list(a.media_regions) for a in o.allocations],
            "hires": [list(a.hires_by_region) for a in o.allocations], "expected": o.result.expected_share}


class EVCampaign:
    """Expected-value campaign on the true electorate for a FIXED plan (worker picks per round + media schedule);
    differentiable in b. Follows the P8 round order of ``x_batch.run_batch`` with every random event replaced by
    its expectation."""

    def __init__(self, e, cfg: GameConfig, plan: dict):
        self.e, self.cfg = e, cfg
        sus = e.susceptibility
        self.zealot = sus.zealot.to(torch.bool)
        self.a = (cfg.dynamics.social_step * sus.social).to(e.preferences.dtype)
        self.gamma = sus.campaign
        self.media_sus = sus.media
        g = e.graph
        self.rows, self.cols = g.rows(), g.indices
        self.degree = g.degree.to(e.preferences.dtype)
        self.new = [x.to(torch.bool) for x in plan["new_workers"]]
        self.media = plan["media"]

    def _agg(self, v):
        total = torch.zeros_like(v).index_add(0, self.rows, v.index_select(0, self.cols))
        mean = total / self.degree.clamp_min(1).unsqueeze(-1)
        return torch.where((self.degree > 0).unsqueeze(-1), mean, v)

    def final_share(self, b: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
        cfg, e = self.cfg, self.e
        beta, z = cfg.voting.sensitivity, cfg.voting.threshold
        mcfg, cap = cfg.media, cfg.workers.attempt_cap
        v = e.preferences
        n = v.shape[0]
        pinned = torch.zeros(n, dtype=torch.bool, device=v.device)
        spread = torch.zeros(n, dtype=v.dtype, device=v.device)
        attempts = torch.zeros(n, dtype=v.dtype, device=v.device)
        airings = [0, 0, 0, 0]
        for t in range(cfg.rounds):
            new = self.new[t] if t < len(self.new) else torch.zeros_like(pinned)
            blocked = pinned | new | self.zealot
            pinned = pinned | new
            spreaders = torch.maximum(spread, new.to(v.dtype))
            sim = 1 - 0.5 * (v - b).abs().sum(-1)
            prob = torch.sigmoid(beta * (sim - z))
            q = (self.gamma * prob).clamp(0, 1)
            counts = torch.zeros(n, dtype=v.dtype, device=v.device).index_add(0, self.rows, spreaders[self.cols])
            eff = torch.where(blocked, torch.zeros_like(counts), torch.minimum(counts, (cap - attempts).clamp_min(0)))
            p_conv = 1 - torch.pow((1 - q).clamp_min(1e-12), eff)
            attempts = attempts + eff
            aired = self.media[t] if t < len(self.media) else [False] * 4
            eta = torch.tensor([mcfg.strength / (1 + mcfg.decay * airings[r]) if aired[r] else 0.0 for r in range(4)],
                               dtype=v.dtype, device=v.device)
            c = (mcfg.exposure_prob * (eta[e.region_of] * self.media_sus).clamp(0, 1))
            c = torch.where(self.zealot, torch.zeros_like(c), c)
            airings = [x + int(f) for x, f in zip(airings, aired)]
            v = torch.where(new.unsqueeze(-1), b.expand_as(v), v)
            v = (1 - p_conv).unsqueeze(-1) * v + p_conv.unsqueeze(-1) * b
            agg = self._agg(v)
            a = self.a
            over = a + c > 1
            c = torch.where(over, (1 - a), c)
            new_v = (1 - a - c).unsqueeze(-1) * v + a.unsqueeze(-1) * agg + c.unsqueeze(-1) * b
            new_v = new_v.clamp_min(0)
            new_v = new_v / new_v.sum(-1, keepdim=True).clamp_min(1e-30)
            new_v = torch.where(pinned.unsqueeze(-1), b.expand_as(v), new_v)
            v = torch.where(self.zealot.unsqueeze(-1), v, new_v)
            spread = p_conv
        return expected_share(v, b, beta, z, smoothing)


def optimise_b_ev(ev: EVCampaign, b_init: torch.Tensor, *, steps: int = 150, lr: float = 0.01, smoothing: float = 1e-3) -> tuple[torch.Tensor, float, float]:
    """Projected gradient on the EV objective from ``b_init``; returns the best iterate (exact EV objective)."""
    b = b_init.detach().clone()
    best_b, best = b.clone(), float(ev.final_share(b))
    start_val = best
    for _ in range(steps):
        b.requires_grad_(True)
        (g,) = torch.autograd.grad(ev.final_share(b, smoothing), b)
        with torch.no_grad():
            b = project_simplex((b + lr * g).unsqueeze(0))[0]
            val = float(ev.final_share(b))
            if val > best:
                best, best_b = val, b.clone()
    return best_b, best, start_val


def joint_seed(seed: int, e, cfg: GameConfig, b0: torch.Tensor, alternations: int = 3) -> dict:
    b = b0.clone()
    history = []
    plan = oracle_canonical_plan(e, cfg, b, seed)
    for k in range(alternations):
        ev = EVCampaign(e, cfg, plan)
        with torch.enable_grad():
            new_b, ev_val, ev_start = optimise_b_ev(ev, b)
        new_plan = oracle_canonical_plan(e, cfg, new_b, seed)
        history.append({"alternation": k + 1, "ev_before": ev_start, "ev_after": ev_val,
                        "realised_before": plan["expected"], "realised_after_replan": new_plan["expected"],
                        "b": [round(float(x), 6) for x in new_b.tolist()], "tv_step": float(0.5 * (new_b - b).abs().sum()),
                        "workers": int(sum(sum(h) for h in new_plan["hires"])),
                        "airings": int(sum(sum(m) for m in new_plan["media"]))})
        b, plan = new_b, new_plan
    return {"b_joint": b, "history": history}


# ============================================================================== per seed
def seed_task(task: tuple) -> dict:
    seed, n, device, do_interp, do_joint, settings_dict = task
    torch.set_num_threads(4)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.zeros(1, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    settings = x13.SearchSettings(**settings_dict)
    cfg = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=dev)
    v_t = anticipate(e, cfg)
    b0 = oracle_manifesto(e, cfg)
    sizes = [int(x) for x in e.sizes.tolist()]
    streams = x13.fresh_streams(seed, FRESH)
    beta, z = cfg.voting.sensitivity, cfg.voting.threshold
    row: dict = {"exp": "X12", "seed": seed, "sizes": sizes, "b0": [float(x) for x in b0.tolist()],
                 "sigma": e.archetype_of_region.tolist(), "tau": e.personality_of_region.tolist()}
    if do_interp:
        coal = coalitions(sizes)
        bs, keys = [b0], [("b0", 0.0)]
        blends = {}
        for c in coal:
            bc = blend(v_t, e.region_of, sizes, c)
            blends[c] = bc
            for lam in LAMBDAS[1:]:
                bs.append((1 - lam) * b0 + lam * bc)
                keys.append(("".join(map(str, c)), lam))
        bs_t = torch.stack(bs)
        oracle = play_oracle(e, cfg, bs_t, streams)
        grid = []
        for (cname, lam), b, res in zip(keys, bs, oracle):
            grid.append({"coalition": cname, "lambda": lam, "b": [round(float(x), 6) for x in b.tolist()],
                         "static_E": float(expected_share(v_t, b, beta, z)),
                         "lambda_max_region": region_branching(e, b, cfg),
                         "coalition_share": (sum(sizes[int(r)] for r in cname) / sum(sizes)) if cname != "b0" else None,
                         "oracle": {k: v for k, v in res.items() if k != "E_streams"},
                         "oracle_E_streams": res["E_streams"]})
        row["grid"] = grid
        # best coalition for the ORACLE planner (mean over λ > 0) → search arm
        best_c = max((g["coalition"] for g in grid if g["coalition"] != "b0"),
                     key=lambda c: np.mean([g["oracle"]["E"] for g in grid if g["coalition"] == c]))
        row["best_coalition"] = best_c
        search = []
        for lam in LAMBDAS:
            b = b0 if lam == 0 else (1 - lam) * b0 + lam * blends[tuple(int(x) for x in best_c)]
            res = play_search(e, cfg, b, seed, streams, settings)
            search.append({"lambda": lam, **{k: v for k, v in res.items()}})
        row["search_best_coalition"] = search
    if do_joint:
        j = joint_seed(seed, e, cfg, b0)
        bj = j["b_joint"]
        res = play_oracle(e, cfg, torch.stack([b0, bj]), streams)
        sr = play_search(e, cfg, bj, seed, streams, settings)
        row["joint"] = {"history": j["history"], "b_joint": [float(x) for x in bj.tolist()],
                        "tv_b0_bjoint": float(0.5 * (bj - b0).abs().sum()),
                        "static_E_b0": float(expected_share(v_t, b0, beta, z)),
                        "static_E_bjoint": float(expected_share(v_t, bj, beta, z)),
                        "oracle_b0": {k: v for k, v in res[0].items() if k != "E_streams"},
                        "oracle_bjoint": {k: v for k, v in res[1].items() if k != "E_streams"},
                        "oracle_b0_E_streams": res[0]["E_streams"], "oracle_bjoint_E_streams": res[1]["E_streams"],
                        "search_bjoint": {k: v for k, v in sr.items() if k != "E_streams"},
                        "lambda_max_b0": region_branching(e, b0, cfg), "lambda_max_bjoint": region_branching(e, bj, cfg)}
    row["seconds"] = time.perf_counter() - start
    row["peak_bytes"] = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    if dev.type == "cuda":
        torch.cuda.empty_cache()          # release cached blocks between seeds (shared GPU, 2 processes)
    return row


# ============================================================================== parity
def parity(seeds: list[int], device: torch.device, n: int) -> dict:
    def fn(seed, dev):
        cfg = GameConfig(seed=seed, device=str(dev), electorate=ElectorateConfig(num_voters=n))
        e = generate_electorate(cfg, device=dev)
        v_t = anticipate(e, cfg)
        b0 = oracle_manifesto(e, cfg)
        sizes = [int(x) for x in e.sizes.tolist()]
        c = coalitions(sizes)[0]
        b1 = blend(v_t, e.region_of, sizes, c)
        res = play_oracle(e, cfg, torch.stack([b0, b1]), x13.fresh_streams(seed, 5))
        plan = oracle_canonical_plan(e, cfg, b0, seed)
        ev = float(EVCampaign(e, cfg, plan).final_share(b0))
        lam = region_branching(e, b1, cfg)
        return {"E_oracle_b0": res[0]["E"], "E_oracle_blend": res[1]["E"], "ev_objective_b0": ev,
                "lambda_max_mean_blend": sum(lam) / 4}

    out = xc.parity_check(fn, seeds, device=device, abs_tol=0.01)
    out["metrics"]["lambda_max_mean_blend"]["note"] = "deterministic up to float order; tolerance 0.01 absolute"
    return out


# ============================================================================== statistics
def summarise(rows: list[dict], x13_data: dict | None) -> dict:
    out: dict = {}
    interp = [r for r in rows if "grid" in r]
    if interp:
        # ORACLE arm: per λ, best coalition per seed (C*), and over all coalitions
        lam_tab = {}
        for lam in LAMBDAS:
            if lam == 0:
                vals = [next(g for g in r["grid"] if g["coalition"] == "b0")["oracle"] for r in interp]
            else:
                vals = [next(g for g in r["grid"] if g["coalition"] == r["best_coalition"] and g["lambda"] == lam)["oracle"]
                        for r in interp]
            lam_tab[str(lam)] = {"E": xc.mean_ci([x["E"] for x in vals]), "win": xc.mean_ci([x["win"] for x in vals])}
        out["oracle_arm_best_coalition_by_lambda"] = lam_tab
        all_c = {}
        for lam in LAMBDAS[1:]:
            vals = [g["oracle"]["E"] for r in interp for g in r["grid"] if g["lambda"] == lam]
            all_c[str(lam)] = xc.mean_ci(vals)
        out["oracle_arm_all_coalitions_by_lambda"] = all_c
        # best λ per seed (ORACLE arm, best coalition)
        best_lams = []
        for r in interp:
            cand = [(0.0, next(g for g in r["grid"] if g["coalition"] == "b0")["oracle"]["E"])] + \
                   [(g["lambda"], g["oracle"]["E"]) for g in r["grid"] if g["coalition"] == r["best_coalition"]]
            best_lams.append(max(cand, key=lambda x: x[1])[0])
        out["oracle_arm_best_lambda_counts"] = {str(l): best_lams.count(l) for l in LAMBDAS}
        # search arm
        s_tab = {}
        for i, lam in enumerate(LAMBDAS):
            s_tab[str(lam)] = {"E": xc.mean_ci([r["search_best_coalition"][i]["E"] for r in interp]),
                               "win": xc.mean_ci([r["search_best_coalition"][i]["win"] for r in interp])}
        out["search_arm_best_coalition_by_lambda"] = s_tab
        s_best = [LAMBDAS[int(np.argmax([x["E"] for x in r["search_best_coalition"]]))] for r in interp]
        out["search_arm_best_lambda_counts"] = {str(l): s_best.count(l) for l in LAMBDAS}
        # paired tests: λ vs 0 for each arm (best coalition), Holm within arm
        tests = {}
        for arm in ("oracle", "search"):
            fam = []
            for i, lam in enumerate(LAMBDAS[1:], start=1):
                if arm == "oracle":
                    a = [next(g for g in r["grid"] if g["coalition"] == r["best_coalition"] and g["lambda"] == lam)["oracle"]["E"]
                         for r in interp]
                    b = [next(g for g in r["grid"] if g["coalition"] == "b0")["oracle"]["E"] for r in interp]
                else:
                    a = [r["search_best_coalition"][i]["E"] for r in interp]
                    b = [r["search_best_coalition"][0]["E"] for r in interp]
                fam.append({"lambda": lam, "E_minus_lambda0": xc.paired(a, b)})
            for x, p in zip(fam, xc.holm([x["E_minus_lambda0"]["wilcoxon_p"] for x in fam])):
                x["E_minus_lambda0"]["holm_p"] = p
            tests[arm] = fam
        out["tests_lambda_vs_0"] = {"family": "per arm: E(λ) − E(0) for the best coalition; Wilcoxon, Holm within arm "
                                              "(ORACLE-arm selection of C* uses the same streams: optimistic)",
                                    "tests": tests}
        # canvassing receptivity: does the best λ > 0 coincide with a high λ_max in the coalition's regions?
        rec, best_pos = [], []
        for r, bl in zip(interp, best_lams):
            c = [int(x) for x in r["best_coalition"]]
            g1 = next(g for g in r["grid"] if g["coalition"] == r["best_coalition"] and g["lambda"] == 1.0)
            rec.append(max(g1["lambda_max_region"][k] for k in c))
            best_pos.append(bl > 0)
        out["receptivity"] = {
            "definition": "max over the best coalition's regions of λ_max(A_rr D_q) at b_C (λ = 1)",
            "lambda_max_when_best_lambda_gt0": xc.mean_ci([x for x, p in zip(rec, best_pos) if p]) if any(best_pos) else None,
            "lambda_max_when_best_lambda_eq0": xc.mean_ci([x for x, p in zip(rec, best_pos) if not p]) if not all(best_pos) else None,
            "seeds_supercritical_region": sum(1 for x in rec if x > 1),
            "spearman_lambda_max_vs_gain": xc.spearman(
                rec, [next(g for g in r["grid"] if g["coalition"] == r["best_coalition"] and g["lambda"] == 1.0)["oracle"]["E"]
                      - next(g for g in r["grid"] if g["coalition"] == "b0")["oracle"]["E"] for r in interp])}
        # λ_max vs λ (mean over seeds, best coalition, max over coalition regions and over all regions)
        lm = {}
        for lam in LAMBDAS:
            vals = []
            for r in interp:
                g = next(g for g in r["grid"] if g["coalition"] == "b0") if lam == 0 else \
                    next(g for g in r["grid"] if g["coalition"] == r["best_coalition"] and g["lambda"] == lam)
                c = [int(x) for x in r["best_coalition"]]
                vals.append((max(g["lambda_max_region"][k] for k in c), max(g["lambda_max_region"]),
                             sum(g["oracle"]["reach_by_region"][k] for k in c), sum(g["oracle"]["reach_by_region"]),
                             g["static_E"]))
            lm[str(lam)] = {"lambda_max_coalition_max": xc.mean_ci([v[0] for v in vals]),
                            "lambda_max_all_max": xc.mean_ci([v[1] for v in vals]),
                            "reach_coalition": xc.mean_ci([v[2] for v in vals]), "reach_total": xc.mean_ci([v[3] for v in vals]),
                            "static_E": xc.mean_ci([v[4] for v in vals])}
        out["spectra_and_reach_by_lambda"] = lm
        out["coalitions_per_seed"] = xc.mean_ci([len({g["coalition"] for g in r["grid"]}) - 1 for r in interp])
        out["best_coalition_size_counts"] = {str(k): sum(1 for r in interp if len(r["best_coalition"]) == k) for k in (2, 3)}
    joint = [r for r in rows if "joint" in r]
    if joint:
        j = [r["joint"] for r in joint]
        out["joint"] = {
            "n": len(j),
            "oracle_planner": {"b0": xc.mean_ci([x["oracle_b0"]["E"] for x in j]),
                               "b_joint": xc.mean_ci([x["oracle_bjoint"]["E"] for x in j]),
                               "joint_minus_b0": xc.paired([x["oracle_bjoint"]["E"] for x in j], [x["oracle_b0"]["E"] for x in j]),
                               "win_b0": xc.mean_ci([x["oracle_b0"]["win"] for x in j]),
                               "win_b_joint": xc.mean_ci([x["oracle_bjoint"]["win"] for x in j])},
            "static_E": {"b0": xc.mean_ci([x["static_E_b0"] for x in j]), "b_joint": xc.mean_ci([x["static_E_bjoint"] for x in j]),
                         "joint_minus_b0": xc.paired([x["static_E_bjoint"] for x in j], [x["static_E_b0"] for x in j])},
            "tv_b0_bjoint": xc.mean_ci([x["tv_b0_bjoint"] for x in j]),
            "ev_gain_per_alternation": {str(k + 1): xc.mean_ci([x["history"][k]["ev_after"] - x["history"][k]["ev_before"] for x in j])
                                        for k in range(len(j[0]["history"]))},
            "ev_vs_realised_b0": xc.spearman([x["history"][0]["ev_before"] for x in j],
                                             [x["history"][0]["realised_before"] for x in j]),
            "ev_bias_b0": xc.mean_ci([x["history"][0]["ev_before"] - x["history"][0]["realised_before"] for x in j]),
        }
        if x13_data is not None:
            base = [x13_data["E20_search_by_seed"].get(str(r["seed"])) for r in joint]
            ok = [i for i, v in enumerate(base) if v is not None]
            out["joint"]["search_planner"] = {
                "b0_x13": xc.mean_ci([base[i] for i in ok]),
                "b_joint": xc.mean_ci([j[i]["search_bjoint"]["E"] for i in ok]),
                "joint_minus_b0": xc.paired([j[i]["search_bjoint"]["E"] for i in ok], [base[i] for i in ok])}
        fam = [{"arm": "oracle_planner", **out["joint"]["oracle_planner"]["joint_minus_b0"]}]
        if "search_planner" in out["joint"]:
            fam.append({"arm": "search_planner", **out["joint"]["search_planner"]["joint_minus_b0"]})
        for x, p in zip(fam, xc.holm([x["wilcoxon_p"] for x in fam])):
            x["holm_p"] = p
        out["joint"]["tests"] = {"family": "joint − b-first, per planner; Wilcoxon, Holm", "tests": fam}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50, help="interpolation arm")
    parser.add_argument("--joint-seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X12"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--budget-hours", type=float, default=2.5)
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--out", default=str(xc.REPORTS / "x12.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=6 * 2**30)
    free0 = xc.free_bytes(device)
    seeds = list(range(args.first_seed, args.first_seed + max(args.seeds, args.joint_seeds)))
    settings = dataclasses.asdict(x13.SearchSettings())
    tasks = [(s, args.n, str(device), i < args.seeds, i < args.joint_seeds, settings) for i, s in enumerate(seeds)]
    deviations = [f"Device {device} instead of the plan default cuda:2 (cuda:2 is used by the Phase 2 agent).",
                  "Search arm (b): X13's plan search is run for the best coalition per seed only (5 λ points), not for "
                  "every coalition (≈ 7 per seed): the ORACLE-planner arm covers all coalitions."]
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        print(f"[x12] device {device}, interpolation seeds {args.seeds}, joint seeds {args.joint_seeds}", flush=True)
        pilot = seed_task(tasks[0])
        est = pilot["seconds"] * (args.seeds - 1) / max(1, args.workers) * 1.15
        print(f"[x12] pilot seed {seeds[0]}: {pilot['seconds']:.0f}s; estimated remaining ≤ {est / 60:.0f} min", flush=True)
        if args.pilot_only:
            print(json.dumps({"best": pilot.get("best_coalition"),
                              "search": [(x["lambda"], round(x["E"], 4)) for x in pilot.get("search_best_coalition", [])],
                              "joint": {k: pilot["joint"][k] for k in ("tv_b0_bjoint",)} if "joint" in pilot else None,
                              "joint_oracle": [pilot["joint"]["oracle_b0"]["E"], pilot["joint"]["oracle_bjoint"]["E"],
                                               pilot["joint"]["search_bjoint"]["E"]] if "joint" in pilot else None,
                              "hist": pilot["joint"]["history"] if "joint" in pilot else None}, indent=1))
            return
        if est > args.budget_hours * 3600:
            raise SystemExit("estimate exceeds the time budget; reduce and record the deviation")
        torch.cuda.empty_cache() if device.type == "cuda" else None   # the parent idles while the workers run
        rows = [pilot] + xc.run_tasks(seed_task, tasks[1:], args.workers, label="[x12]")
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text("")
        for r in rows:
            xc.write_row(RAW_PATH, r)
        par = parity(seeds[:args.parity_seeds], device, args.n)
        print(f"[x12] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    x13_path = xc.REPORTS / "x13.json"
    x13_data = json.loads(x13_path.read_text()) if x13_path.exists() else None
    summary = summarise(rows, x13_data)
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    report = {
        "experiment": "X12", "question": "Broad vs narrow ('viral') manifesto; campaign-aware b",
        "seeds": {"first": seeds[0], "interpolation": args.seeds, "joint": args.joint_seeds, "list": seeds},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget), "lambdas": list(LAMBDAS),
        "fresh_streams": FRESH, "search_settings": settings,
        "definitions": {"b0": "oracle_manifesto", "b_C": "size-weighted mean of the true anticipated regional means over "
                        "coalition C (2- and 3-region coalitions with > 50% of voters)",
                        "lambda_max": "λ_max(A_rr D_q), q = γ P(v(0), b), zealots 0; power iteration",
                        "best_coalition": "per seed, highest mean ORACLE-planner E over λ ∈ {0.25, 0.5, 0.75, 1}",
                        "joint": "3 alternations: PG (150 steps, lr 0.01) on the expected-value campaign for the fixed "
                                 "executed ORACLE plan, then re-plan with the ORACLE planner"},
        **summary, "pilot_seconds": pilot["seconds"], "deviations": deviations, "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, workers=args.workers,
                                       batch_sizes={"oracle_arm_run_batch_B": "manifestos × 20 streams per seed",
                                                    "search": "as X13 (C = 256, K = 4 / 8)"},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       grad_mode="no_grad (simulation), autograd (EV manifesto optimiser)",
                                       extra={"child_peak_bytes_max": max((r["peak_bytes"] or 0) for r in rows)}),
    }
    xc.save(args.out, report)
    print(f"[x12] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
