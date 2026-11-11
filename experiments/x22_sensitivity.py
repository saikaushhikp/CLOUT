"""X22 — Global sensitivity analysis of the design: Morris elementary effects (plan §7 X22).

Command (reproduces reports/v3/x22.json, reports/v3/raw/x22.jsonl, reports/v3/logs/x22.log and the figures):
  OMP_NUM_THREADS=4 python experiments/x22_sensitivity.py --seeds 50 --first-seed 14000 --trajectories 10 \
      --device cuda:2 --out reports/v3/x22.json 2>&1 | tee reports/v3/logs/x22.log

Factors (10), each in a ±30% box around the game default: β_v (vote sensitivity), z (vote threshold, see below),
κ (Dirichlet concentration), ω (social step), ζ (zealot fraction), k̄ (mean degree), p_m (media exposure), η_m0
(media strength), λ (media decay), worker fee (₫20, rounded to the cent).
z: the literal box 0.825 × [0.7, 1.3] reaches 1.07 > 1 (invalid; S ≤ 1, so z ≥ 1 means nobody votes). The box is
put on the margin 1 − z instead: z = 1 − 0.175 × [0.7, 1.3] ∈ [0.7725, 0.8775] (deviation, recorded).

Design: Morris (1991) one-at-a-time trajectories, numpy implementation (SALib is not installed and not added),
p = 4 levels {0, 1/3, 2/3, 1}, Δ = 2/3, r = 10 trajectories × (k + 1) = 11 points = 110 design points, random start,
random factor order, each step ±Δ (rng seed 14000). Unit level u maps to default × (0.7 + 0.6u).

Outputs per design point, ORACLE-only on the GPU (oracle manifesto on the true world; ``x_batch.run_batch`` with all
5 policies × 50 paired seeds (block 14000) in one batched call):
  oracle_win     ORACLE planner win rate;   split_win  "pay_split" win rate;
  worker_best    best-channel share: share of seeds whose best paid pattern (by E, among pay_workers,
                 front_workers, pay_media, pay_split) is a worker-only pattern (pay_workers / front_workers);
  media_best     the same for pay_media (reported, not ranked);  oracle_E, split_E  mean expected shares (extra).
Elementary effect EE_i = [y(x + Δe_i) − y(x)] / Δ (unit scale, so EE = change over 2/3 of the box). μ, μ*, σ per
factor and output; 95% CI of μ* by bootstrap over trajectories (2,000); noise floor = mean over EEs of the
seed-sampling SE (sd of the 50 per-seed differences / √50 / Δ; seeds are common to both points).
Electorates are reused between consecutive points when the generation factors (κ, ζ, k̄) are unchanged, and the
manifestos when (β_v, z, κ, ω, ζ, k̄) are unchanged.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
import x_parity  # noqa: E402
import x_phase5a as p5  # noqa: E402
import x_policies  # noqa: E402
from x_batch import run_batch  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402

D = GameConfig()
FACTORS = {  # name: (symbol, config path, default)
    "beta_v": ("β_v", "voting.sensitivity", D.voting.sensitivity),
    "z": ("z", "voting.threshold", D.voting.threshold),
    "kappa": ("κ", "electorate.dirichlet_concentration", D.electorate.dirichlet_concentration),
    "omega": ("ω", "dynamics.social_step", D.dynamics.social_step),
    "zeta": ("ζ", "electorate.zealot_fraction", D.electorate.zealot_fraction),
    "kbar": ("k̄", "electorate.mean_degree", D.electorate.mean_degree),
    "p_m": ("p_m", "media.exposure_prob", D.media.exposure_prob),
    "eta_m0": ("η_m0", "media.strength", D.media.strength),
    "lambda": ("λ", "media.decay", D.media.decay),
    "worker_fee": ("worker fee", "workers.worker_cost", float(D.workers.worker_cost)),
}
NAMES = list(FACTORS)
K = len(NAMES)
GEN_FACTORS = {"kappa", "zeta", "kbar"}
MANIFESTO_FACTORS = GEN_FACTORS | {"beta_v", "z", "omega"}
PAID = ("pay_workers", "front_workers", "pay_media", "pay_split")
RUN_POLICIES = ("planner",) + PAID
OUTPUTS = ("oracle_win", "split_win", "worker_best", "media_best", "oracle_E", "split_E")
RANKED = ("oracle_win", "worker_best", "split_win")
LEVELS = 4
DELTA = LEVELS / (2 * (LEVELS - 1))   # 2/3


def factor_value(name: str, u: float):
    _, _, d = FACTORS[name]
    scale = 0.7 + 0.6 * u
    if name == "z":
        return round(1 - (1 - d) * scale, 10)
    if name == "worker_fee":
        return Decimal(str(round(d * scale, 2))).quantize(Decimal("0.01"))
    return round(d * scale, 10)


def point_config(base: GameConfig, x: np.ndarray) -> GameConfig:
    cfg = base
    for name, u in zip(NAMES, x):
        cfg = xc.override(cfg, FACTORS[name][1], str(factor_value(name, float(u))))
    return cfg


def morris_trajectories(r: int, rng: np.random.Generator) -> list[dict]:
    grid = np.arange(LEVELS) / (LEVELS - 1)
    trajs = []
    for _ in range(r):
        x = rng.choice(grid, size=K)
        order = rng.permutation(K)
        pts, steps = [x.copy()], []
        for i in order:
            if x[i] + DELTA <= 1 + 1e-9 and (x[i] - DELTA < -1e-9 or rng.random() < 0.5):
                x[i] = round(x[i] + DELTA, 10)
                sign = 1
            else:
                x[i] = round(x[i] - DELTA, 10)
                sign = -1
            pts.append(x.copy())
            steps.append((int(i), sign))
        trajs.append({"points": [p.tolist() for p in pts], "steps": steps})
    return trajs


# ============================================================================== one design point
def evaluate(cfg: GameConfig, seeds, device, cache: dict, B: int, record: dict) -> dict:
    gen_key = tuple(getattr(getattr(cfg, h), f) for h, f in (p.split(".") for _, p, _ in
                    (FACTORS[n] for n in sorted(GEN_FACTORS))))
    man_key = gen_key + (cfg.voting.sensitivity, cfg.voting.threshold, cfg.dynamics.social_step)
    if cache.get("gen_key") != gen_key:
        cache.clear()
        cache["gen_key"] = gen_key
        cache["es"] = [generate_electorate(dataclasses.replace(cfg, seed=s), device=device) for s in seeds]
        record["electorate_draws"] = record.get("electorate_draws", 0) + 1
    if cache.get("man_key") != man_key:
        cache["man_key"] = man_key
        cache["ms"] = torch.stack([oracle_manifesto(e, dataclasses.replace(cfg, seed=s))
                                   for e, s in zip(cache["es"], seeds)])
        record["manifesto_draws"] = record.get("manifesto_draws", 0) + 1
    es, ms = cache["es"], cache["ms"]
    W = len(seeds)
    items = [(p, i) for p in RUN_POLICIES for i in range(W)]

    def fn(ch):
        return list(zip(ch, run_batch([es[i] for _, i in ch], cfg, ms[[i for _, i in ch]],
                                      [x_policies.GATE_POLICIES[p] for p, _ in ch], seeds=[seeds[i] for _, i in ch])))
    E = {p: [0.0] * W for p in RUN_POLICIES}
    win = {p: [False] * W for p in RUN_POLICIES}
    for (p, i), o in xc.run_chunks(fn, items, B, record=record):
        E[p][i] = float(o.result.expected_share)
        win[p][i] = bool(o.result.victory)
    best = [max(PAID, key=lambda p: E[p][i]) for i in range(W)]
    per_seed = {"oracle_win": [float(x) for x in win["planner"]], "split_win": [float(x) for x in win["pay_split"]],
                "worker_best": [float(b in ("pay_workers", "front_workers")) for b in best],
                "media_best": [float(b == "pay_media") for b in best],
                "oracle_E": E["planner"], "split_E": E["pay_split"]}
    return {"per_seed": per_seed, "best": best, "E": E, "win": win,
            "y": {k: float(np.mean(v)) for k, v in per_seed.items()}}


# ============================================================================== Morris statistics
def morris_stats(trajs, results, n_boot: int, rng) -> dict:
    out = {}
    for o in OUTPUTS:
        ee = {n: [] for n in NAMES}      # per trajectory
        se = []
        for t, traj in enumerate(trajs):
            for s, (i, sign) in enumerate(traj["steps"]):
                a, b = results[t][s], results[t][s + 1]
                d = (np.array(b["per_seed"][o]) - np.array(a["per_seed"][o])) * sign / DELTA
                ee[NAMES[i]].append(float(d.mean()))
                se.append(float(d.std(ddof=1) / math.sqrt(len(d))))
        stats = {}
        for n in NAMES:
            v = np.array(ee[n])
            boot = [np.abs(v[rng.integers(0, len(v), len(v))]).mean() for _ in range(n_boot)]
            stats[n] = {"mu": float(v.mean()), "mu_star": float(np.abs(v).mean()), "sigma": float(v.std(ddof=1)),
                        "mu_star_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                        "ee": [float(x) for x in v]}
        rank = sorted(NAMES, key=lambda n: -stats[n]["mu_star"])
        floor = float(np.mean(se))
        out[o] = {"factors": stats, "ranking_mu_star": rank, "noise_floor_ee_se": floor,
                  "above_noise": [n for n in rank if stats[n]["mu_star_ci95"][0] > 2 * floor]}
    return out


# ============================================================================== parity
def parity(base: GameConfig, seeds, device) -> dict:
    """CPU↔GPU at the default (centre) design point, 3 seeds, canonical CPU draw; ORACLE planner and pay_split."""
    def fn(seed, dev):
        cfg = dataclasses.replace(base, seed=seed, device=str(dev), generation="cpu")
        e = generate_electorate(cfg, device=dev)
        m = oracle_manifesto(e, cfg)
        outs = run_batch([e, e], cfg, torch.stack([m, m]), [x_policies.GATE_POLICIES["planner"],
                                                            x_policies.GATE_POLICIES["pay_split"]], seeds=[seed, seed])
        return {"E_planner": outs[0].result.expected_share, "E_split": outs[1].result.expected_share,
                "win_planner": float(outs[0].result.victory)}
    return x_parity.parity_rel(fn, seeds, device=device, abs_tol=0.01)


# ============================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X22"])
    ap.add_argument("--trajectories", type=int, default=10)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--cap-gib", type=float, default=10.0)
    ap.add_argument("--parity-seeds", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--max-points", type=int, default=None, help="pilot only: stop after this many points")
    ap.add_argument("--out", default=str(xc.REPORTS / "x22.json"))
    ap.add_argument("--raw", default=str(xc.RAW / "x22.jsonl"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    device, dev_record = p5.pin_device(args.device, need_bytes=int(args.cap_gib * 2**30))
    free0 = xc.free_bytes(device)
    torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(device=str(device), generation="device", electorate=ElectorateConfig(num_voters=args.n))
    trajs = morris_trajectories(args.trajectories, np.random.default_rng(args.first_seed))
    raw = Path(args.raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("")
    record: dict = {}
    cache: dict = {}
    results: list[list[dict]] = []
    start = time.perf_counter()
    done = 0
    with xc.GpuMonitor(device) as monitor:
        # batch: 5 policies × W worlds per call, capped at cap-gib
        cfg0 = point_config(base, np.array(trajs[0]["points"][0]))
        es = [generate_electorate(dataclasses.replace(cfg0, seed=s), device=device) for s in seeds[:8]]
        ms = torch.stack([oracle_manifesto(e, dataclasses.replace(cfg0, seed=s)) for e, s in zip(es, seeds[:8])])
        cal = p5.capped_batch(lambda b: run_batch(es[:b], cfg0, ms[:b], x_policies.policy_planner, seeds=seeds[:b]),
                              device, sizes=(4, 8), max_batch=len(RUN_POLICIES) * len(seeds),
                              cap_bytes=int(args.cap_gib * 2**30))
        del es, ms
        B = cal["batch"]
        print(f"[x22] device {device}; batch B = {B}; {len(trajs)} trajectories × {K + 1} points × {len(seeds)} seeds",
              flush=True)
        for t, traj in enumerate(trajs):
            results.append([])
            for s, x in enumerate(traj["points"]):
                if args.max_points is not None and done >= args.max_points:
                    break
                t0 = time.perf_counter()
                cfg = point_config(base, np.array(x))
                res = evaluate(cfg, seeds, device, cache, B, record)
                results[t].append(res)
                values = {n: (str(factor_value(n, u)) if n == "worker_fee" else factor_value(n, u))
                          for n, u in zip(NAMES, x)}
                xc.write_row(raw, {"exp": "X22", "trajectory": t, "step": s, "unit": x, "values": values,
                                   "changed": None if s == 0 else NAMES[traj["steps"][s - 1][0]], **res["y"],
                                   "per_seed": {**res["per_seed"], "best": res["best"]}, "seeds": seeds,
                                   "E": res["E"], "seconds": time.perf_counter() - t0})
                done += 1
                print(f"[x22] point {done}/{len(trajs) * (K + 1)} (traj {t} step {s}): "
                      + " ".join(f"{k}={v:.3f}" for k, v in res["y"].items())
                      + f" [{time.perf_counter() - t0:.1f}s, total {time.perf_counter() - start:.0f}s]", flush=True)
        games_seconds = time.perf_counter() - start
        cache.clear()
        torch.cuda.empty_cache()
        par = parity(base, seeds[:args.parity_seeds], device)
        print(f"[x22] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    complete = all(len(r) == K + 1 for r in results)
    stats = morris_stats(trajs, results, args.bootstrap, np.random.default_rng(14000)) if complete else None
    report = {
        "experiment": "X22", "title": "Global sensitivity analysis of the design (Morris)",
        "question": "Which parameters is the game's balance most sensitive to?",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True,
                  "note": "the same 50 seeds at every design point (common random numbers)"},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget),
        "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
        "factors": {n: {"symbol": FACTORS[n][0], "path": FACTORS[n][1],
                        "default": str(FACTORS[n][2]) if n == "worker_fee" else FACTORS[n][2],
                        "range": [str(factor_value(n, 0.0)), str(factor_value(n, 1.0))] if n == "worker_fee"
                        else [factor_value(n, 0.0), factor_value(n, 1.0)]} for n in NAMES},
        "design": {"method": "Morris elementary effects (numpy; SALib not installed)", "levels": LEVELS,
                   "delta_unit": DELTA, "trajectories": len(trajs), "points": len(trajs) * (K + 1),
                   "unit_to_value": "default × (0.7 + 0.6u); z: 1 − (1 − z₀)(0.7 + 0.6u); fee rounded to the cent",
                   "rng_seed": args.first_seed, "trajectory_list": trajs},
        "metric_definitions": {
            "oracle_win": "ORACLE manifesto + ORACLE planner win rate over the 50 seeds",
            "split_win": "ORACLE manifesto + pay_split win rate",
            "worker_best": "best-channel share: share of seeds whose best paid pattern (by E, among pay_workers, "
                           "front_workers, pay_media, pay_split; ORACLE manifesto) is pay_workers or front_workers",
            "media_best": "share of seeds whose best paid pattern is pay_media",
            "oracle_E": "mean expected share, ORACLE planner", "split_E": "mean expected share, pay_split",
            "EE": "[y(x ± Δe_i) − y(x)]·(±1)/Δ with Δ = 2/3 of the box (unit scale)",
            "mu_star": "mean |EE| over the r trajectories; mu = signed mean; sigma = sd (interaction/non-linearity)",
            "noise_floor_ee_se": "mean over all EEs of sd(per-seed EE)/√50 (seed-sampling SE of one EE)",
            "above_noise": "factors whose μ* bootstrap CI lower bound exceeds 2 × noise floor"},
        "outputs_ranked": list(RANKED),
        "points": [[r["y"] for r in rs] for rs in results],
        "morris": stats, "complete": complete,
        "prediction": "none registered (predictions.md: the plan gives no Predict line for X22)",
        "deviations": [
            "z box on the margin 1 − z (z ∈ [0.7725, 0.8775]) because the literal ±30% box reaches z = 1.07 > 1.",
            "'best-channel share' operationalised as the share of seeds where a worker-only pattern is the best paid "
            "pattern (media-best share also reported).",
            "Sobol indices (optional in the plan) not computed.",
            "Electorates drawn on the GPU (generation='device'); parity uses the canonical CPU draw."],
        "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=record.get("batch_sizes"), free_at_start=free0,
                                       monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"calibration": cal, "games_seconds": games_seconds,
                                              "electorate_draws": record.get("electorate_draws"),
                                              "manifesto_draws": record.get("manifesto_draws"),
                                              "oom_backoffs": record.get("oom_backoffs", []),
                                              "command": "python experiments/x22_sensitivity.py --seeds %d --first-seed %d "
                                                         "--trajectories %d --device %s --out %s" % (
                                                             len(seeds), seeds[0], len(trajs), args.device, args.out)}),
    }
    xc.save(args.out, report)
    if complete and not args.no_plots:
        plots(report)
        xc.save(args.out, report)
    if stats:
        for o in RANKED:
            print(f"[x22] {o}: ranking {stats[o]['ranking_mu_star']} (noise floor {stats[o]['noise_floor_ee_se']:.3f})",
                  flush=True)
    c = report["provenance"]["compute"]
    print(f"[x22] wall {seconds:.0f}s, peak {c['peak_bytes'] / 2**30:.2f} GiB, util {c.get('gpu_util_mean')}", flush=True)


def plots(report: dict) -> None:
    figs = []
    fig, axes = p5.new_figure(1, 3, figsize=(13, 4.2))
    for ax, o in zip(axes, RANKED):
        p5.style(ax)
        ax.grid(axis="x", color=p5.GRID, linewidth=0.6)
        st = report["morris"][o]["factors"]
        for j, n in enumerate(NAMES):
            ax.scatter(st[n]["mu_star"], st[n]["sigma"], color=p5.SERIES[j % len(p5.SERIES)],
                       marker="o" if j < 8 else "s", s=28, zorder=3)
            ax.annotate(FACTORS[n][0], (st[n]["mu_star"], st[n]["sigma"]), fontsize=7, color=p5.INK2,
                        xytext=(3, 3), textcoords="offset points")
        floor = report["morris"][o]["noise_floor_ee_se"]
        ax.axvline(2 * floor, color=p5.INK2, linestyle=":", linewidth=0.8)
        ax.text(2 * floor, ax.get_ylim()[1], " 2× noise", fontsize=7, color=p5.INK2, va="top")
        ax.set_xlabel("μ* (mean |elementary effect|)", fontsize=8, color=p5.INK2)
        ax.set_ylabel("σ (sd of elementary effects)", fontsize=8, color=p5.INK2)
        ax.set_title(o, fontsize=9, color=p5.INK)
    fig.suptitle("X22: Morris screening, 10 factors in ±30% boxes (r = 10, 50 seeds per point)", fontsize=10)
    figs.append(p5.save_figure(fig, "x22_morris.png"))

    fig, axes = p5.new_figure(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, o in zip(axes, RANKED):
        p5.style(ax)
        ax.grid(axis="x", color=p5.GRID, linewidth=0.6)
        st = report["morris"][o]["factors"]
        order = report["morris"][o]["ranking_mu_star"]
        ys = list(range(len(NAMES)))
        mu = [st[n]["mu"] for n in NAMES]
        ax.barh(ys, mu, color=[p5.SERIES[0] if m >= 0 else p5.SERIES[1] for m in mu], height=0.7)
        ax.axvline(0, color=p5.INK2, linewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([FACTORS[n][0] for n in NAMES], fontsize=8)
        ax.set_title(f"{o}: signed μ (rank: {', '.join(FACTORS[n][0] for n in order[:3])})", fontsize=8, color=p5.INK)
        ax.set_xlabel("μ: change over 2/3 of the box (+ = raising the factor raises the output)", fontsize=7,
                      color=p5.INK2)
    axes[0].invert_yaxis()
    figs.append(p5.save_figure(fig, "x22_direction.png"))
    report["figures"] = figs


if __name__ == "__main__":
    main()
