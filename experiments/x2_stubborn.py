"""X2 — Stubborn agents, zealots, W-MSR (+ polarisation metric) (plan ``docs/v2/plans/implementation_plan.md`` §4 X2).

Command (reproduces reports/v3/x2.json, reports/v3/raw/x2.jsonl and reports/v3/logs/x2.log):
  OMP_NUM_THREADS=4 python experiments/x2_stubborn.py --seeds 50 --first-seed 11000 --device cuda:2 \
      --out reports/v3/x2.json 2>&1 | tee reports/v3/logs/x2.log

N = 10⁴, T = 5, ₫500, ₫20 workers, 50 paired seeds (block 11000). One FIXED manifesto per seed in every cell: the
ORACLE manifesto of the default world (ζ = 0.05, DeGroot), so rule and zealot effects are not confounded with a
different b. Social rule ∈ {DeGroot, W-MSR F = 1, W-MSR F = 2} (Arm C = Arms A and B under each W-MSR rule).

* Arm A — zealots ζ ∈ {0, 0.05, 0.15, 0.30} (the zealot draw has its own stream, so graph and preferences are
  identical across ζ): policies none, pay_split (fixed campaign) and the ORACLE planner (``balance.policy_planner``).
* Arm B — exactly k ∈ {0, 5, 10, 20} workers bought in round 1 (proportional to region size), nothing else;
  cascade OFF (``workers.attempt_cap = 0``: workers stay pinned, nobody converts) vs ON (the game rule).
  Anchor share of the worker value = Δ_off(k) / Δ_on(k), Δ = E(k) − E(0).
* Long runs (experiment-side, outside the game's T ≤ 10): the cascade-off dynamics continued to T = 200 with the
  SAME pinned workers (taken from the T = 5 game's ``WorkerState``), no campaign; per rule, for the default world
  (ζ = 0.05, k ∈ {0, 5, 10, 20}) and for a zealot-free world (ζ = 0, k ∈ {0, 20}). The T = 5 prefix of the long
  run reproduces the game's expected share (checked).
* Friedkin–Johnsen prediction (DeGroot): with stubborn set S = workers (at b) ∪ zealots (at v(0)) the update's
  fixed point solves L_ff v_f = A_fs v_s (harmonic extension: every free voter ends at the random-walk-absorption-
  weighted mix of the stubborn opinions; the step sizes a_i only set the speed). Solved by Jacobi-preconditioned
  conjugate gradients on the GPU (all worlds in one block-diagonal system, 5 right-hand sides); SciPy ``spsolve``
  on the CPU is the reference (3 worlds; SciPy ``cg``, as ``spsolve`` fill-in is too slow). Predicted long-run share vs the simulated share at T = 5 and T = 200.
* Polarisation (N15), at election day and in the long runs: between-region dispersion = mean over regions of
  TV(regional mean, global mean); within-region = mean over voters of TV(v_i, regional mean); bimodality = share of
  voters with P < 0.1 or P > 0.9 (P for the fixed b). All over non-zealot voters.
* (r, s)-robustness (``analysis.robustness``): upper bound on r (region-union and greedy certificates), upper
  bounds on s for (2, s) and (3, s); and the F-local exposure = share of normal voters with more than F stubborn
  neighbours (zealots ∪ workers), the condition W-MSR's guarantee needs to hold (≤ F per neighbourhood).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
import x_policies  # noqa: E402
from x_batch import BatchGame, run_batch  # noqa: E402
from x_policies import WorkerTiming  # noqa: E402

from dumbocracy.analysis.robustness import robustness_upper_bound, rs_upper_bound  # noqa: E402
from dumbocracy.config import DynamicsConfig, ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402
from dumbocracy.strategies.worlds import _union_graph  # noqa: E402
from dumbocracy.voting import vote_probability  # noqa: E402

ZETAS = (0.0, 0.05, 0.15, 0.30)
KS = (0, 5, 10, 20)
RULES = {"degroot": ("degroot", 1), "wmsr_F1": ("wmsr", 1), "wmsr_F2": ("wmsr", 2)}
A_POLICIES = ("none", "pay_split", "planner")
LONG_T = 200
LONG_CHECKPOINTS = (0, 1, 5, 10, 20, 50, 100, 200)
RAW_PATH = xc.RAW / "x2.jsonl"


def rule_cfg(base: GameConfig, rule: str, zeta: float | None = None, cascade: bool = True) -> GameConfig:
    name, F = RULES[rule]
    cfg = dataclasses.replace(base, dynamics=DynamicsConfig(social_step=base.dynamics.social_step, social_rule=name,
                                                            wmsr_trim=F))
    if zeta is not None:
        cfg = dataclasses.replace(cfg, electorate=dataclasses.replace(cfg.electorate, zealot_fraction=zeta))
    if not cascade:
        cfg = dataclasses.replace(cfg, workers=dataclasses.replace(cfg.workers, attempt_cap=0))
    return cfg


# ============================================================================== metrics
def polarisation(v: torch.Tensor, b_rows: torch.Tensor, region_of: torch.Tensor, keep: torch.Tensor, B: int, n: int,
                 cfg: GameConfig) -> torch.Tensor:
    """[B, 3] = (between-region dispersion, within-region dispersion, bimodality) over voters with ``keep``."""
    v = v.to(torch.float64).view(B, n, 5)
    keep = keep.view(B, n)
    reg = region_of.view(B, n)
    w = keep.to(torch.float64)
    glob = (v * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp_min(1)                       # [B, 5]
    onehot = torch.nn.functional.one_hot(reg, 4).to(torch.float64) * w[..., None]                 # [B, n, 4]
    cnt = onehot.sum(1)                                                                           # [B, 4]
    means = torch.einsum("bnr,bnk->brk", onehot, v) / cnt[..., None].clamp_min(1)                 # [B, 4, 5]
    between = (0.5 * (means - glob[:, None]).abs().sum(-1)).mean(1)
    own = means.gather(1, reg[..., None].expand(B, n, 5))
    within = ((0.5 * (v - own).abs().sum(-1)) * w).sum(1) / w.sum(1).clamp_min(1)
    sim = 1 - 0.5 * (v - b_rows.to(torch.float64).view(B, n, 5)).abs().sum(-1)
    p = vote_probability(sim, cfg.voting.sensitivity, cfg.voting.threshold)
    bimodal = (((p < 0.1) | (p > 0.9)).to(torch.float64) * w).sum(1) / w.sum(1).clamp_min(1)
    return torch.stack([between, within, bimodal], 1)


def play(worlds: list[dict], cfg: GameConfig, policy, *, zeta_key: str) -> list[dict]:
    """Batched T = 5 games; returns rows with share, win, workers, converts and the election-day polarisation."""
    es = [w["es"][zeta_key] for w in worlds]
    B, n = len(es), es[0].num_voters
    ms = torch.stack([w["b"] for w in worlds])
    outs = run_batch(es, cfg, ms, policy, seeds=[w["seed"] for w in worlds], keep_final=True)
    v = torch.cat([o.final_preferences for o in outs])
    zealot = torch.cat([e.susceptibility.zealot.to(torch.bool) for e in es])
    region = torch.cat([e.region_of for e in es])
    pol = polarisation(v, ms.repeat_interleave(n, 0), region, ~zealot, B, n, cfg).cpu()
    rows = []
    for b, (w, o) in enumerate(zip(worlds, outs)):
        rows.append({"seed": w["seed"], "expected": o.result.expected_share, "victory": bool(o.result.victory),
                     "workers": int(o.workers.pinned.sum()), "converts": int(sum(o.converted)),
                     "spent": float(o.spent), "between": float(pol[b, 0]), "within": float(pol[b, 1]),
                     "bimodal": float(pol[b, 2]), "trajectory": [round(x, 6) for x in o.share_trajectory],
                     "_pinned": o.workers.pinned})
    return rows


@torch.no_grad()
def long_run(worlds: list[dict], cfg: GameConfig, zeta_key: str, pinned: list[torch.Tensor]) -> dict:
    """Cascade-off, campaign-free dynamics for LONG_T rounds with the given pinned workers (taking v = b at round 1,
    as bought workers do). Returns per-world checkpoints of share and polarisation, plus the final v."""
    es = [w["es"][zeta_key] for w in worlds]
    B, n = len(es), es[0].num_voters
    ms = torch.stack([w["b"] for w in worlds])
    game = BatchGame(es, cfg, ms)
    pin = torch.cat(pinned)
    v = game.v0.clone()
    keep = ~game.zealot
    zero = torch.zeros(B * n, dtype=v.dtype, device=v.device)
    checkpoints = {}

    def snap(t, v_):
        sim = 1 - 0.5 * (v_ - game.b_rows).abs().sum(-1)
        share = vote_probability(sim, cfg.voting.sensitivity, cfg.voting.threshold).view(B, n).mean(1)
        pol = polarisation(v_, game.b_rows, game.region_of, keep, B, n, cfg)
        free = keep & ~pin
        glob = (v_.to(torch.float64).view(B, n, 5) * free.view(B, n, 1)).sum(1) / free.view(B, n).sum(1, keepdim=True)
        disagreement = (0.5 * (v_.to(torch.float64).view(B, n, 5) - glob[:, None]).abs().sum(-1) * free.view(B, n)).sum(1) \
            / free.view(B, n).sum(1)
        checkpoints[t] = torch.cat([share.to(torch.float64)[:, None], pol, disagreement[:, None]], 1).cpu()

    snap(0, v)
    for t in range(1, LONG_T + 1):
        if t == 1:
            v = torch.where(pin.unsqueeze(-1), game.b_rows, v)
        v = game.step(v, game.aggregate(v), zero, pin)
        if t in LONG_CHECKPOINTS:
            snap(t, v)
    return {"checkpoints": checkpoints, "v": v, "game": game, "pinned": pin}


# ============================================================================== Friedkin–Johnsen
@torch.no_grad()
def fj_solve(worlds: list[dict], zeta_key: str, pinned: torch.Tensor, b_rows: torch.Tensor, *, tol: float = 1e-10,
             max_iter: int = 5000) -> tuple[torch.Tensor, dict]:
    """Fixed point of the DeGroot update with stubborn S = workers (at b) ∪ zealots (at v(0)) ∪ isolated voters:
    L_ff x = A_fs v_s, by Jacobi-preconditioned CG on the GPU over the block-diagonal union (5 right-hand sides)."""
    es = [w["es"][zeta_key] for w in worlds]
    B, n = len(es), es[0].num_voters
    g = es[0].graph if B == 1 else _union_graph([e.graph for e in es], n)
    dev = g.indices.device
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        adj = torch.sparse_csr_tensor(g.indptr, g.indices, torch.ones(g.indices.numel(), dtype=torch.float64, device=dev),
                                      (B * n, B * n))
    zealot = torch.cat([e.susceptibility.zealot.to(torch.bool) for e in es])
    v0 = torch.cat([e.preferences for e in es]).to(torch.float64)
    deg = g.degree.to(torch.float64)
    stub = zealot | pinned | (g.degree == 0)
    free = ~stub
    vs = torch.where(pinned[:, None], b_rows.to(torch.float64), v0)
    vs = torch.where(stub[:, None], vs, torch.zeros_like(vs))
    f = free.to(torch.float64)[:, None]
    rhs = (adj @ vs) * f

    def matvec(x):
        return (deg[:, None] * x - adj @ (x * f)) * f
    x = torch.zeros_like(rhs)
    r = rhs - matvec(x)
    minv = torch.where(free, 1.0 / deg.clamp_min(1), torch.zeros_like(deg))[:, None]
    z = minv * r
    p = z.clone()
    rz = (r * z).sum(0)
    norm0 = torch.linalg.vector_norm(rhs, dim=0).clamp_min(1e-300)
    it = 0
    for it in range(1, max_iter + 1):
        ap = matvec(p)
        alpha = rz / (p * ap).sum(0).clamp_min(1e-300)
        x = x + alpha * p
        r = r - alpha * ap
        res = float((torch.linalg.vector_norm(r, dim=0) / norm0).max())
        if res < tol:
            break
        z = minv * r
        rz_new = (r * z).sum(0)
        p = z + (rz_new / rz) * p
        rz = rz_new
    v_star = torch.where(free[:, None], x, vs)
    return v_star, {"iterations": it, "max_rel_residual": res, "stubborn_share": float(stub.double().mean())}


def fj_scipy(e, pinned: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """CPU reference for one world: ``scipy.sparse.linalg.cg`` (Jacobi preconditioner, rtol 1e-12) on the reduced
    Laplacian (``spsolve`` was tried first: its LU fill-in on a k̄ = 30 graph took minutes per world)."""
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import cg
    n = e.num_voters
    rows, cols = e.graph.rows().cpu().numpy(), e.graph.indices.cpu().numpy()
    a = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(n, n))
    deg = np.asarray(a.sum(1)).ravel()
    stub = (e.susceptibility.zealot.to(torch.bool) | pinned).cpu().numpy() | (deg == 0)
    free = ~stub
    v0 = e.preferences.double().cpu().numpy()
    vs = np.where(pinned.cpu().numpy()[:, None], b.double().cpu().numpy()[None], v0)
    lap = (diags(deg) - a).tocsr()
    lff = lap[free][:, free].tocsr()
    rhs = a[free][:, stub] @ vs[stub]
    out = vs.copy()
    pre = diags(1.0 / lff.diagonal())
    out[free] = np.column_stack([cg(lff, rhs[:, j], rtol=1e-12, atol=0.0, M=pre, maxiter=20000)[0] for j in range(5)])
    return torch.from_numpy(out)


# ============================================================================== robustness
def robustness(worlds: list[dict], pinned_k20: list[torch.Tensor]) -> list[dict]:
    out = []
    for w, pin in zip(worlds, pinned_k20):
        e = w["es"]["0.05"]
        g = e.graph
        bound = robustness_upper_bound(g, e.region_of)
        rows = g.rows()
        rec = {"seed": w["seed"], "r_upper": bound.r_upper, "witness": bound.witness, "min_degree": bound.min_degree,
               "s_upper_r2": rs_upper_bound(g, e.region_of, 2), "s_upper_r3": rs_upper_bound(g, e.region_of, 3)}
        zealot = e.susceptibility.zealot.to(torch.bool)
        for name, stub in (("zealots", zealot), ("workers20", pin), ("zealots_workers20", zealot | pin)):
            cnt = torch.bincount(rows[stub[g.indices]], minlength=e.num_voters)
            normal = ~stub
            for F in (1, 2):
                rec[f"frac_normal_gt_{F}_{name}_nbrs"] = float((cnt[normal] > F).double().mean())
            rec[f"mean_{name}_nbrs"] = float(cnt[normal].double().mean())
        out.append(rec)
    return out


# ============================================================================== summary
def summarise(res: dict, seeds: list[int], long: dict, fj: dict, rob: list[dict]) -> dict:
    import numpy as np
    from scipy.stats import wilcoxon
    out: dict = {"arm_A": {}, "arm_B": {}, "anchor_share": {}, "wmsr_effect": {}, "long_runs": {}, "fj": {},
                 "robustness": {}, "tests": {}}

    def stats(rows):
        return {"expected": xc.mean_ci([r["expected"] for r in rows]), "win": xc.rate([r["victory"] for r in rows]),
                "workers": xc.mean_ci([r["workers"] for r in rows]), "converts": xc.mean_ci([r["converts"] for r in rows]),
                "between": xc.mean_ci([r["between"] for r in rows]), "within": xc.mean_ci([r["within"] for r in rows]),
                "bimodal": xc.mean_ci([r["bimodal"] for r in rows])}
    for key, rows in res.items():
        arm = key.split("|")[0]
        (out["arm_A"] if arm == "A" else out["arm_B"])[key] = stats(rows)
    tests_anchor, tests_wmsr, tests_zeta = [], [], []
    for rule in RULES:
        # persuasion under zealots: E(policy) − E(none), and change vs ζ = 0
        for zeta in ZETAS:
            for pol in ("pay_split", "planner"):
                a, z = res[f"A|{rule}|z{zeta}|{pol}"], res[f"A|{rule}|z{zeta}|none"]
                out["arm_A"][f"A|{rule}|z{zeta}|{pol}"]["persuasion"] = xc.paired([x["expected"] for x in a],
                                                                                  [x["expected"] for x in z])
                if zeta > 0:
                    a0, z0 = res[f"A|{rule}|z0.0|{pol}"], res[f"A|{rule}|z0.0|none"]
                    d = xc.paired([x["expected"] - y["expected"] for x, y in zip(a, z)],
                                  [x["expected"] - y["expected"] for x, y in zip(a0, z0)])
                    tests_zeta.append({"rule": rule, "zeta": zeta, "policy": pol, "measure": "persuasion(ζ) − persuasion(0)", **d})
        # anchor share Δ_off / Δ_on
        for k in KS[1:]:
            on, off = res[f"B|{rule}|on|k{k}"], res[f"B|{rule}|off|k{k}"]
            on0, off0 = res[f"B|{rule}|on|k0"], res[f"B|{rule}|off|k0"]
            d_on = [x["expected"] - y["expected"] for x, y in zip(on, on0)]
            d_off = [x["expected"] - y["expected"] for x, y in zip(off, off0)]
            rng = np.random.default_rng(k)
            boots = []
            for _ in range(4000):
                idx = rng.integers(0, len(d_on), len(d_on))
                boots.append(np.mean(np.asarray(d_off)[idx]) / np.mean(np.asarray(d_on)[idx]))
            out["anchor_share"][f"{rule}|k{k}"] = {
                "delta_on": xc.mean_ci(d_on), "delta_off": xc.mean_ci(d_off),
                "ratio_of_means": float(np.mean(d_off) / np.mean(d_on)),
                "ratio_boot_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "per_seed_ratio_median": float(np.median(np.asarray(d_off) / np.asarray(d_on)))}
            tests_anchor.append({"rule": rule, "k": k, "measure": "Δ_off (anchor only) > 0", **xc.paired(d_off, [0.0] * len(d_off))})
            if rule != "degroot":
                for mode in ("on", "off"):
                    dr = [x["expected"] - y["expected"] for x, y in zip(res[f"B|{rule}|{mode}|k{k}"], res[f"B|{rule}|{mode}|k0"])]
                    dd = [x["expected"] - y["expected"] for x, y in zip(res[f"B|degroot|{mode}|k{k}"], res[f"B|degroot|{mode}|k0"])]
                    t = {"rule": rule, "k": k, "cascade": mode, "measure": "worker value Δ(k): W-MSR − DeGroot", **xc.paired(dr, dd)}
                    tests_wmsr.append(t)
                    out["wmsr_effect"][f"{rule}|{mode}|k{k}"] = t
    for name, fam in (("anchor_positive", tests_anchor), ("wmsr_minus_degroot_worker_value", tests_wmsr),
                      ("zealot_persuasion_change", tests_zeta)):
        for t, p in zip(fam, xc.holm([t["wilcoxon_p"] for t in fam])):
            t["holm_p"] = p
        out["tests"][name] = fam
    out["test_families"] = {
        "anchor_positive": "Δ_off(k) = E(k workers, cascade off) − E(0) vs 0, Wilcoxon, 3 rules × 3 k, Holm (9)",
        "wmsr_minus_degroot_worker_value": "worker value Δ(k) under W-MSR F − under DeGroot, 2 rules × 3 k × "
                                           "{cascade on, off}, Wilcoxon, Holm (12)",
        "zealot_persuasion_change": "persuasion (policy − none) at ζ minus at ζ = 0, 3 rules × 3 ζ × {pay_split, planner}, "
                                    "Wilcoxon, Holm (18)"}
    # long runs
    names = ("share", "between", "within", "bimodal", "disagreement")
    for key, lr in long.items():
        cps = lr["checkpoints"]
        out["long_runs"][key] = {name: {str(t): xc.mean_ci(cps[t][:, j].tolist()) for t in cps}
                                 for j, name in enumerate(names)}
        out["long_runs"][key]["T5_matches_game"] = lr.get("t5_check")
        # disagreement decay rate per round over t ∈ [100, 200]
        d100, d200 = cps[100][:, 4].clamp_min(1e-300), cps[200][:, 4].clamp_min(1e-300)
        out["long_runs"][key]["disagreement_ratio_200_over_100"] = xc.mean_ci((d200 / d100).tolist())
    out["fj"] = fj
    out["robustness"] = {
        "per_world": rob, "r_upper": xc.mean_ci([r["r_upper"] for r in rob]),
        "r_upper_min": min(r["r_upper"] for r in rob), "r_upper_max": max(r["r_upper"] for r in rob),
        "min_degree": xc.mean_ci([r["min_degree"] for r in rob]),
        "s_upper_r2_min": min(r["s_upper_r2"] for r in rob), "s_upper_r3_min": min(r["s_upper_r3"] for r in rob),
        "witness_counts": {wit: sum(r["witness"].split(" {")[0] == wit for r in rob)
                           for wit in sorted({r["witness"].split(" {")[0] for r in rob})},
        **{k: xc.mean_ci([r[k] for r in rob]) for k in rob[0] if k.startswith("frac_") or k.startswith("mean_")}}
    return out


# ============================================================================== parity
def parity(seeds: list[int], device: torch.device, base: GameConfig) -> dict:
    def fn(seed, dev):
        cfg = dataclasses.replace(rule_cfg(base, "wmsr_F1", cascade=True), device=str(dev))
        e = generate_electorate(dataclasses.replace(cfg, seed=seed), device=dev)
        b = oracle_manifesto(e, dataclasses.replace(base, seed=seed, device=str(dev)))
        w = {"seed": seed, "es": {"0.05": e}, "b": b}
        r = play([w], cfg, WorkerTiming(10, "front", "proportional"), zeta_key="0.05")[0]
        return {"expected": r["expected"], "converts_share_of_n": r["converts"] / e.num_voters,
                "within": r["within"], "between": r["between"], "bimodal": r["bimodal"]}
    out = xc.parity_check(fn, seeds, device=device, abs_tol=0.01)
    out["cell"] = "W-MSR F = 1, k = 10 workers in round 1, cascade on"
    out["note"] = "statistical tolerance max(0.01, 3·SE); converts scaled by N"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X2"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--out", default=str(xc.REPORTS / "x2.json"))
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
    record: dict = {"timing": {}}
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        worlds = []
        for s in seeds:
            es = {str(z): generate_electorate(dataclasses.replace(base, seed=s, electorate=dataclasses.replace(
                base.electorate, zealot_fraction=z)), device=device) for z in ZETAS}
            worlds.append({"seed": s, "es": es, "b": oracle_manifesto(es["0.05"], dataclasses.replace(base, seed=s))})
        record["setup_seconds"] = time.perf_counter() - start
        print(f"[x2] setup {record['setup_seconds']:.0f}s", flush=True)
        res: dict[str, list[dict]] = {}
        pinned: dict[str, list[torch.Tensor]] = {}

        def run(key, cfg, policy, zkey):
            t0 = time.perf_counter()
            rows = play(worlds, cfg, policy, zeta_key=zkey)
            for r in rows:
                xc.write_row(RAW_PATH, {"exp": "X2", "cell": key, **{k: v for k, v in r.items() if k != "_pinned"}})
            pinned[key] = [r.pop("_pinned") for r in rows]
            res[key] = rows
            record["timing"][key] = time.perf_counter() - t0
            print(f"[x2] {key}: {record['timing'][key]:.1f}s; E {sum(r['expected'] for r in rows) / len(rows):.4f} "
                  f"within {sum(r['within'] for r in rows) / len(rows):.4f}", flush=True)

        for rule in RULES:
            for zeta in ZETAS:
                cfg = rule_cfg(base, rule, zeta)
                for pol in A_POLICIES:
                    run(f"A|{rule}|z{zeta}|{pol}", cfg, x_policies.GATE_POLICIES[pol], str(zeta))
            for mode in ("on", "off"):
                cfg = rule_cfg(base, rule, cascade=(mode == "on"))
                for k in KS:
                    run(f"B|{rule}|{mode}|k{k}", cfg, WorkerTiming(k, "front", "proportional"), "0.05")
        # zealot-free pinned set for the consensus contrast (DeGroot rules only differ in dynamics, not the pick)
        for rule in RULES:
            run(f"B0|{rule}|off|k20", dataclasses.replace(rule_cfg(base, rule, 0.0, cascade=False)),
                WorkerTiming(20, "front", "proportional"), "0.0")
        # long runs and the T = 5 consistency check
        long: dict = {}
        for rule in RULES:
            cfg = rule_cfg(base, rule, cascade=False)
            for zkey, ks in (("0.05", KS), ("0.0", (0, 20))):
                for k in ks:
                    src = f"B|{rule}|off|k{k}" if zkey == "0.05" else (f"B0|{rule}|off|k20" if k else None)
                    pins = pinned[src] if src else [torch.zeros(args.n, dtype=torch.bool, device=device)] * len(worlds)
                    t0 = time.perf_counter()
                    lr = long_run(worlds, dataclasses.replace(cfg, electorate=dataclasses.replace(
                        cfg.electorate, zealot_fraction=float(zkey))), zkey, pins)
                    key = f"{rule}|z{zkey}|k{k}"
                    if src:
                        game_e = torch.tensor([r["expected"] for r in res[src]], dtype=torch.float64)
                        lr["t5_check"] = {"max_abs_diff_share_T5": float((lr["checkpoints"][5][:, 0] - game_e).abs().max())}
                    if rule == "degroot" and (k > 0 or zkey == "0.05"):
                        v_star, info = fj_solve(worlds, zkey, lr["pinned"], lr["game"].b_rows)
                        g = lr["game"]
                        B, n = g.B, g.n
                        sim = 1 - 0.5 * (v_star - g.b_rows.to(torch.float64)).abs().sum(-1)
                        share_star = vote_probability(sim, base.voting.sensitivity, base.voting.threshold).view(B, n).mean(1).cpu()
                        free = (~g.zealot & ~lr["pinned"]).view(B, n)
                        tv200 = (0.5 * (lr["v"].to(torch.float64) - v_star).abs().sum(-1)).view(B, n)
                        pol_star = polarisation(v_star, g.b_rows, g.region_of, ~g.zealot, B, n, base).cpu()
                        glob = (v_star.view(B, n, 5) * free[..., None]).sum(1) / free.sum(1, keepdim=True)
                        dis_star = ((0.5 * (v_star.view(B, n, 5) - glob[:, None]).abs().sum(-1)) * free).sum(1) / free.sum(1)
                        fjrec = {"solver": info, "share_fj": xc.mean_ci(share_star.tolist()),
                                 "share_T5": xc.mean_ci(lr["checkpoints"][5][:, 0].tolist()),
                                 "share_T200": xc.mean_ci(lr["checkpoints"][200][:, 0].tolist()),
                                 "fj_minus_T5": xc.paired(share_star.tolist(), lr["checkpoints"][5][:, 0].tolist()),
                                 "fj_minus_T200": xc.paired(share_star.tolist(), lr["checkpoints"][200][:, 0].tolist()),
                                 "mean_tv_v200_to_fj_free": xc.mean_ci(((tv200 * free).sum(1) / free.sum(1)).cpu().tolist()),
                                 "between_fj": xc.mean_ci(pol_star[:, 0].tolist()), "within_fj": xc.mean_ci(pol_star[:, 1].tolist()),
                                 "bimodal_fj": xc.mean_ci(pol_star[:, 2].tolist()),
                                 "disagreement_fj": xc.mean_ci(dis_star.cpu().tolist())}
                        if zkey == "0.05" and k == 10:   # SciPy reference on 3 worlds
                            ref = []
                            for j in range(min(3, len(worlds))):
                                sl = slice(j * n, (j + 1) * n)
                                cpu = fj_scipy(worlds[j]["es"][zkey], lr["pinned"][sl], worlds[j]["b"])
                                ref.append(float((cpu - v_star[sl].cpu()).abs().max()))
                            fjrec["scipy_reference_max_abs_diff"] = ref
                        long.setdefault("_fj", {})[key] = fjrec
                        print(f"[x2] FJ {key}: CG {info['iterations']} it, share FJ {fjrec['share_fj']['mean']:.4f} "
                              f"T5 {fjrec['share_T5']['mean']:.4f} T200 {fjrec['share_T200']['mean']:.4f}", flush=True)
                    del lr["v"], lr["game"], lr["pinned"]
                    long[key] = lr
                    print(f"[x2] long {key}: {time.perf_counter() - t0:.1f}s; share T200 "
                          f"{float(lr['checkpoints'][200][:, 0].mean()):.4f}; disagreement T200 "
                          f"{float(lr['checkpoints'][200][:, 4].mean()):.4f}", flush=True)
        fj = long.pop("_fj")
        t0 = time.perf_counter()
        rob = robustness(worlds, pinned["B|degroot|off|k20"])
        record["robustness_seconds"] = time.perf_counter() - t0
        summary = summarise(res, seeds, long, fj, rob)
        par = parity(seeds[:args.parity_seeds], device, base)
        print(f"[x2] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    report = {"experiment": "X2", "question": "Stubborn agents (workers, zealots), W-MSR, Friedkin–Johnsen, polarisation",
              "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
              "n": args.n, "rounds": base.rounds, "treasury": str(base.budget), "zetas": list(ZETAS), "ks": list(KS),
              "rules": RULES, "long_T": LONG_T, "long_checkpoints": list(LONG_CHECKPOINTS),
              "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
              "metric_definitions": {
                  "manifesto": "fixed per seed: ORACLE manifesto of the default world (ζ = 0.05, DeGroot)",
                  "persuasion": "E(policy) − E(none), same cell",
                  "anchor_share": "Δ_off(k) / Δ_on(k), Δ = E(k workers) − E(0 workers); off = attempt_cap 0",
                  "between": "mean over regions of TV(regional mean v, global mean v), non-zealots",
                  "within": "mean over non-zealot voters of TV(v_i, own regional mean)",
                  "bimodal": "share of non-zealot voters with P < 0.1 or P > 0.9 (P for the fixed b)",
                  "disagreement": "long runs: mean TV(v_i, mean v) over free voters (not zealot, not worker)",
                  "fj": "fixed point of the cascade-off DeGroot dynamics (harmonic extension of the stubborn opinions)"},
              **summary,
              "deviations": ["Arm A 'fixed campaign' is played with two policies (pay_split and the ORACLE planner), "
                             "each against 'none' in the same cell.",
                             "One fixed manifesto per seed in every cell (the DeGroot ζ = 0.05 ORACLE manifesto), not "
                             "re-optimised per rule or ζ.",
                             f"Long runs (T = {LONG_T}) are experiment-side (the game allows T ≤ 10), cascade off, no "
                             "campaign after the round-1 workers."],
              "config": base.to_record(),
              "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=[len(seeds)], free_at_start=free0,
                                             monitor=monitor, device_record=dev_record, parity=par,
                                             extra={k: record.get(k) for k in ("setup_seconds", "timing",
                                                                               "robustness_seconds")})}
    xc.save(args.out, report)
    print(f"[x2] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
