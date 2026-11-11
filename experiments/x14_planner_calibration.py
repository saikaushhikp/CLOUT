"""X14 — Planner calibration: prediction vs reality (plan ``docs/v2/plans/implementation_plan.md`` §5 X14).

Command (reproduces reports/v3/x14.json, reports/v3/raw/x14.jsonl and reports/v3/logs/x14.log):
  OMP_NUM_THREADS=4 python experiments/x14_planner_calibration.py --seeds 50 --first-seed 12000 --device cuda:0 \
      --out reports/v3/x14.json 2>&1 | tee reports/v3/logs/x14.log

BayesPlanner predicts, every round, the election-day expected share of its best candidate (``best_score``: the mean
over ``num_worlds`` posterior worlds of ``world_voters`` voters, rolled out to the election with the ``rollout``
continuation). We compare that prediction with the realised E of the same game (``result.expected_share``).

Design (one factor at a time around the library default, at each N ∈ {10⁴, 10⁵}; 50 paired seeds per arm, D3):
  default        world_voters 1600, num_worlds 8, rollout "even"
  world_voters   800 | 4000 | N_true            (8 worlds, even)
  num_worlds     4 | 16 | 32                    (1600 voters, even)
  rollout        "none" | "front"               (1600 voters, 8 worlds); "front" is the approved optional
                                                 continuation added to ``strategies/planner.py`` (D2, §9)
Every arm commits the SAME manifesto: the default BayesPlanner's posterior-optimised manifesto (8 manifesto worlds
of 1,600 voters), whatever the arm's world settings (plan: "same seeds and the same manifesto"). Surveys are the
library default (they do not depend on the arm's world settings), so the arms differ only in the planner.

Metrics per arm and decision round (1..5) and pooled: bias = mean(predicted − realised), RMSE, calibration slope
(OLS realised ~ predicted over seeds), win rate, mean expected share, decision latency (allocate seconds); also the
worker money spent in round 1 (front-loading) and the fraction of rounds whose planned hires were scaled to zero
in the planner's worlds (expected scaled hires per world < 1).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.players import BayesPlanner  # noqa: E402

RAW_PATH = xc.RAW / "x14.jsonl"
NS = (10_000, 100_000)
DEFAULT = {"world_voters": 1600, "num_worlds": 8, "rollout": "even"}


def arms_for(n: int) -> dict[str, dict]:
    arms = {"default": dict(DEFAULT)}
    for wv in (800, 4000, "N"):
        arms[f"wv_{wv}"] = {**DEFAULT, "world_voters": n if wv == "N" else wv}
    for nw in (4, 16, 32):
        arms[f"nw_{nw}"] = {**DEFAULT, "num_worlds": nw}
    for ro in ("none", "front"):
        arms[f"ro_{ro}"] = {**DEFAULT, "rollout": ro}
    return arms


@dataclass
class FixedManifestoPlanner(BayesPlanner):
    """BayesPlanner whose manifesto is always the DEFAULT planner's (same seed, device and config), so every arm
    commits the same b; only the allocation planner's world settings / rollout change."""

    def manifesto(self, observation, posterior=None):
        ref = BayesPlanner(seed=self.seed, device=self.device, config=self.config)
        return ref.manifesto(observation, posterior)


def seed_task(task: tuple) -> dict:
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.strategies.players import play_game
    from strategies import spend_breakdown
    seed, n, device, arm_names = task
    torch.set_num_threads(4)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.zeros(1, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    cfg = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=dev)
    state0 = GameEngine.new_game(cfg, electorate=e, game_id=f"x14_{seed}")
    arms = arms_for(n)
    rows = []
    for name in arm_names:
        params = arms[name]
        player = FixedManifestoPlanner(seed=seed, device=device, config=dataclasses.replace(cfg, device="cpu"),
                                       **params)
        g0 = time.perf_counter()
        final, log = play_game(state0, player)
        obs = observation_for(final, "p0")
        res = obs["result"]
        allocs = sorted(obs["allocations"], key=lambda a: a["round"])
        n_world = min(params["world_voters"], n)
        scale = min(1.0, n_world / n)
        rows.append({"exp": "X14", "seed": seed, "n": n, "arm": name, **params,
                     "expected": res["expected_share"], "victory": bool(res["victory"]), "vote_share": res["vote_share"],
                     "manifesto": [round(float(x), 6) for x in obs["manifesto"]],
                     "predicted": [d["best_score"] for d in player.diagnostics],
                     "decision_rounds": [int(d["round"]) for d in player.diagnostics],
                     "planner_seconds": [d["seconds"] for d in player.diagnostics],
                     "allocate_seconds": player.latency.get("allocate", []),
                     "hires_by_round": [sum(a["hires_by_region"]) for a in allocs],
                     "media_by_round": [sum(bool(m) for m in a["media_regions"]) for a in allocs],
                     "expected_scaled_hires_per_world": [sum(a["hires_by_region"]) * scale for a in allocs],
                     "share_trajectory": [float(x) for x in final.player("p0").share_trajectory],
                     "spend": spend_breakdown(obs, cfg), "game_seconds": time.perf_counter() - g0})
    peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    return {"seed": seed, "n": n, "rows": rows, "seconds": time.perf_counter() - start, "peak_bytes": peak}


# ============================================================================== statistics
def calib(pred: list[float], real: list[float]) -> dict:
    p, r = np.array(pred, dtype=float), np.array(real, dtype=float)
    d = p - r
    out = {"n": len(p), "bias": xc.mean_ci(d.tolist()), "rmse": float(np.sqrt((d ** 2).mean())),
           "mae": float(np.abs(d).mean()), "pred_mean": float(p.mean()), "real_mean": float(r.mean())}
    if len(p) > 2 and p.std() > 0:
        A = np.column_stack([np.ones_like(p), p])
        coef, res, *_ = np.linalg.lstsq(A, r, rcond=None)
        resid = r - A @ coef
        se = math.sqrt((resid ** 2).sum() / (len(p) - 2) / ((p - p.mean()) ** 2).sum())
        out["slope"] = {"estimate": float(coef[1]), "se": se, "ci95": [float(coef[1] - 1.96 * se), float(coef[1] + 1.96 * se)],
                        "intercept": float(coef[0])}
        out["pearson_r"] = float(np.corrcoef(p, r)[0, 1])
    return out


def summarise(rows: list[dict], seeds: list[int]) -> dict:
    out: dict = {}
    for n in NS:
        arms = arms_for(n)
        rs_n = [r for r in rows if r["n"] == n]
        if not rs_n:
            continue
        by = {a: {r["seed"]: r for r in rs_n if r["arm"] == a} for a in arms}
        cell = {}
        for a in arms:
            rs = [by[a][s] for s in seeds if s in by[a]]
            if not rs:
                continue
            T = max(len(r["predicted"]) for r in rs)
            per_round = {}
            for t in range(T):
                pr = [(r["predicted"][t], r["expected"]) for r in rs if len(r["predicted"]) > t]
                per_round[str(t + 1)] = calib([x for x, _ in pr], [y for _, y in pr])
            pooled = calib([p for r in rs for p in r["predicted"]], [r["expected"] for r in rs for _ in r["predicted"]])
            alloc = [x for r in rs for x in r["allocate_seconds"]]
            cell[a] = {"params": arms[a], "n_games": len(rs), "per_round": per_round, "pooled": pooled,
                       "round1": per_round.get("1"), "win": xc.rate([r["victory"] for r in rs]),
                       "expected": xc.mean_ci([r["expected"] for r in rs]),
                       "latency_allocate_seconds": xc.mean_ci(alloc), "game_seconds": xc.mean_ci([r["game_seconds"] for r in rs]),
                       "workers_round1": xc.mean_ci([r["hires_by_round"][0] if r["hires_by_round"] else 0 for r in rs]),
                       "workers_total": xc.mean_ci([sum(r["hires_by_round"]) for r in rs]),
                       "spend": {k: xc.mean_ci([r["spend"][k] for r in rs]) for k in ("workers", "media", "queries", "total")},
                       "rounds_scaled_hires_lt1": float(np.mean([x < 1 for r in rs for h, x in
                                                                 zip(r["hires_by_round"], r["expected_scaled_hires_per_world"])
                                                                 if h > 0])) if any(sum(r["hires_by_round"]) for r in rs) else None}
        # paired tests vs default (bias in round 1, realised E, wins)
        fam = []
        for a in arms:
            if a == "default" or a not in cell:
                continue
            common = [s for s in seeds if s in by[a] and s in by["default"]]
            bias_a = [by[a][s]["predicted"][0] - by[a][s]["expected"] for s in common]
            bias_d = [by["default"][s]["predicted"][0] - by["default"][s]["expected"] for s in common]
            fam.append({"arm": a, "abs_bias_round1": xc.paired([abs(x) for x in bias_a], [abs(x) for x in bias_d]),
                        "expected": xc.paired([by[a][s]["expected"] for s in common],
                                              [by["default"][s]["expected"] for s in common]),
                        "mcnemar_wins": xc.mcnemar([by[a][s]["victory"] for s in common],
                                                   [by["default"][s]["victory"] for s in common])})
        for key in ("abs_bias_round1", "expected"):
            for x, p in zip(fam, xc.holm([x[key]["wilcoxon_p"] for x in fam])):
                x[key]["holm_p"] = p
        # bias trend with world size: Spearman over (world_voters, round-1 bias) pairs
        size_arms = [a for a in ("wv_800", "default", "wv_4000", "wv_N") if a in cell]
        pts = [(arms[a]["world_voters"], by[a][s]["predicted"][0] - by[a][s]["expected"]) for a in size_arms
               for s in seeds if s in by[a]]
        trend = xc.spearman([x for x, _ in pts], [y for _, y in pts]) if len(pts) > 3 else None
        out[f"N{n}"] = {"arms": cell, "tests_vs_default": {"family": "arm − default per N: |round-1 bias|, realised E "
                                                                     "(Wilcoxon, Holm per metric), wins (McNemar)",
                                                          "tests": fam},
                        "bias_vs_world_voters_spearman": trend}
    return out


def parity(seeds: list[int], device: torch.device) -> dict:
    cache: dict = {}

    def fn(seed, dev):
        key = (seed, str(dev))
        if key not in cache:
            r = seed_task((seed, 10_000, str(dev), ["default"]))["rows"][0]
            cache[key] = {"expected": r["expected"], "predicted_round1": r["predicted"][0],
                          "bias_round1": r["predicted"][0] - r["expected"]}
        return cache[key]

    return xc.parity_check(fn, seeds, device=device, abs_tol=0.02)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X14"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ns", default="10000,100000")
    parser.add_argument("--seeds-heavy", type=int, default=None,
                        help="seeds for the N=1e5 world_voters=N_true arm if the time budget requires it")
    parser.add_argument("--budget-hours", type=float, default=2.5)
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--out", default=str(xc.REPORTS / "x14.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=6 * 2**30)
    free0 = xc.free_bytes(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    ns = [int(x) for x in args.ns.split(",")]
    deviations = [f"Device {device} instead of the plan default cuda:2 (cuda:2 is used by the Phase 2 agent).",
                  "One-factor-at-a-time design around the default (1600, 8, even) instead of the full 4×4×3 factorial "
                  "(9 arms per N instead of 48)."]
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        print(f"[x14] device {device}, seeds {seeds[0]}..{seeds[-1]}, N {ns}", flush=True)
        pilots = {}
        for n in ns:
            p = seed_task((seeds[0], n, str(device), list(arms_for(n))))
            pilots[n] = p
            per = {r["arm"]: round(r["game_seconds"], 1) for r in p["rows"]}
            print(f"[x14] pilot N={n}: {p['seconds']:.0f}s {per}", flush=True)
        est = sum(p["seconds"] for p in pilots.values()) * (len(seeds) - 1) / max(1, args.workers) * 1.15
        print(f"[x14] estimated remaining {est / 60:.0f} min with {args.workers} processes", flush=True)
        if args.pilot_only:
            for p in pilots.values():
                print(json.dumps({r["arm"]: [round(r["predicted"][0], 4), round(r["expected"], 4)] for r in p["rows"]}))
            return
        heavy_seeds = seeds
        if est > args.budget_hours * 3600:
            heavy = pilots[100_000]["rows"]
            heavy_cost = next(r["game_seconds"] for r in heavy if r["arm"] == "wv_N")
            rest_cost = est - heavy_cost * (len(seeds) - 1) / max(1, args.workers) * 1.15
            k = max(10, int((args.budget_hours * 3600 - rest_cost) / (heavy_cost / max(1, args.workers) * 1.15)))
            k = args.seeds_heavy or min(len(seeds), k)
            heavy_seeds = seeds[:k]
            deviations.append(f"N=1e5 world_voters=N_true arm: {k} seeds instead of {len(seeds)} (time budget; pilot "
                              f"{heavy_cost:.0f} s per game)")
            print(f"[x14] heavy arm reduced to {k} seeds", flush=True)
        tasks = []
        for s in seeds[1:]:
            for n in ns:
                names = list(arms_for(n))
                if n == 100_000 and s not in heavy_seeds:
                    names.remove("wv_N")
                tasks.append((s, n, str(device), names))
        tasks.sort(key=lambda t: -t[1])                       # heavy N first
        results = list(pilots.values()) + xc.run_tasks(seed_task, tasks, args.workers, label="[x14]")
        rows = [r for res in results for r in res["rows"]]
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text("")
        for r in rows:
            xc.write_row(RAW_PATH, r)
        par = parity(seeds[:args.parity_seeds], device)
        print(f"[x14] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    summary = summarise(rows, seeds)
    base = GameConfig(device=str(device))
    report = {
        "experiment": "X14", "question": "Planner calibration: BayesPlanner's predicted best_score vs realised E",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "heavy_arm_seeds": len(heavy_seeds)},
        "ns": ns, "rounds": base.rounds, "treasury": str(base.budget),
        "arms": {f"N{n}": arms_for(n) for n in ns},
        "front_rollout": "strategies.planner.front_allocation (approved D2): next round spends all the remaining "
                         "worker money (treasury minus a media reserve of (rounds left − 1) × cap/2); media only "
                         "afterwards. Default rollout unchanged (tests/unit/test_planner_front_rollout.py).",
        "manifesto": "every arm commits the default BayesPlanner's manifesto (FixedManifestoPlanner)",
        **summary,
        "pilot_seconds": {str(n): p["seconds"] for n, p in pilots.items()},
        "estimated_seconds": est, "deviations": deviations, "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, workers=args.workers,
                                       batch_sizes={"num_worlds": [4, 8, 16, 32], "world_voters": [800, 1600, 4000, "N"]},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"child_peak_bytes_max": max((r["peak_bytes"] or 0) for r in results)}),
    }
    xc.save(args.out, report)
    print(f"[x14] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
