"""X10 — Balance across the player-chosen N and T (plan §3 X10; D3: 50 seeds per sweep cell; D7 checkpoint).

Command (reproduces reports/v3/x10.json and reports/v3/raw/x10.jsonl):
  OMP_NUM_THREADS=4 python experiments/x10_n_t_balance.py --seeds 50 --device cuda:2 --treasury 500 \
      --out reports/v3/x10.json

Grid N ∈ {10³, 3·10³, 10⁴, 10⁵, 10⁶} × T ∈ {1, 3, 5, 10} at treasury ₫500, prices ₫20 / ₫10 (D5). Seeds: the
X10 block (``seed_block("X10", 50)`` = 10000…10049), the same list in every cell and every arm (paired).
Every cell plays the X0 policy × manifesto set (``x0_balance.RUNS``, i.e. ``balance.run``) and evaluates the
five §12 gate criteria with ``x0_balance.gate_of``.

Single-channel arms (same seeds): workers-only = ``informed/pay_workers``, media-only = ``informed/pay_media``,
none = ``informed/free_even`` (≡ ``none`` under P8). Per seed:
  Δshare_W = E[pay_workers] − E[none];  Δshare per worker = Δshare_W / workers hired;  per ₫ = Δshare_W / ₫ spent;
  converts per worker = cascade converts (absolute voters) / workers hired;
  Δshare_M per ₫ = (E[pay_media] − E[none]) / ₫ spent on media.
(E = expected vote share, the smooth outcome; paired across arms.) Log–log slopes vs N are fitted per T on the
cell means with a seed-bootstrap 95% CI.

Electorates: canonical CPU draws (R8, ``generation="cpu"``) for N ≤ 10⁵; **N = 10⁶ uses ``generation="device"``**
(GPU draws, plan §1.1 point 3), stated in the output. Runtime: a pilot (``--pilot-seeds`` per N, all T) is timed
first and the full-grid estimate printed; if it exceeds ``--gpu-hours`` the N = 10⁶ seed count is reduced
(never below 25) before anything else, and the reduction is recorded as a deviation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import time
from decimal import Decimal
from pathlib import Path

import torch

import x_common as xc
from balance import PAID_PATTERNS
from dumbocracy.config import ElectorateConfig, GameConfig
from x0_balance import RUNS, chunk_plan, gate_of, parity, play_runs, prepare_chunk

NS = [1_000, 3_000, 10_000, 100_000, 1_000_000]
TS = [1, 3, 5, 10]
DEVICE_GENERATION_FROM = 1_000_000
MIN_SEEDS_1E6 = 25


def base_config(n: int, treasury: str, device: torch.device) -> GameConfig:
    gen = "device" if n >= DEVICE_GENERATION_FROM else "cpu"
    return GameConfig(device=str(device), budget=Decimal(treasury), generation=gen,
                      electorate=ElectorateConfig(num_voters=n))


def run_n(n: int, seeds: list[int], rounds: list[int], treasury: str, device: torch.device, *, raw: Path | None,
          record: dict, log=print) -> tuple[dict[int, list[dict]], dict]:
    """All T for one N: per T a list of per-seed rows ``{"seed", "personality", "sizes", "runs"}``."""
    base = base_config(n, treasury, device)
    plan = chunk_plan(base, device, len(seeds), seeds, max(rounds), log=log)
    rows: dict[int, list[dict]] = {T: [] for T in rounds}
    t0 = time.perf_counter()
    for i in range(0, len(seeds), plan["chunk"]):
        chunk = prepare_chunk(seeds[i:i + plan["chunk"]], base, device, tuple(rounds))
        for T in rounds:
            res = play_runs(chunk, base, T, batch=plan["batch"], record=record)
            for p in chunk:
                rows[T].append({"seed": p.seed, **p.meta, "runs": res[p.seed]})
                if raw is not None:
                    for key, row in res[p.seed].items():
                        man, pol = key.split("/")
                        xc.write_row(raw, {"experiment": "X10", "n": n, "rounds": T, "treasury": treasury,
                                           "generation": base.generation, "seed": p.seed, "manifesto": man,
                                           "policy": pol, "personality": p.meta["personality"],
                                           "sizes": p.meta["sizes"], **row})
        log(f"[x10] N={n}: seeds {i + len(chunk)}/{len(seeds)}, {time.perf_counter() - t0:.0f}s")
        del chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, plan


# ============================================================================== per-cell channel metrics
def channel_metrics(rows: list[dict]) -> dict:
    per_seed = []
    for r in rows:
        runs = r["runs"]
        base = runs["informed/free_even"]["expected"]
        w, m, f = runs["informed/pay_workers"], runs["informed/pay_media"], runs["informed/front_workers"]
        sw, sm = float(w["spent_workers"]), float(m["spent_media"])
        d_w, d_m, d_f = w["expected"] - base, m["expected"] - base, f["expected"] - base
        per_seed.append({
            "seed": r["seed"], "dshare_workers": d_w, "dshare_media": d_m, "dshare_front": d_f,
            "workers": w["workers"], "front_workers": f["workers"], "spent_workers": sw, "spent_media": sm,
            "airings": m["airings"],
            "dshare_per_worker": d_w / w["workers"] if w["workers"] else None,
            "dshare_per_rupee_workers": d_w / sw if sw else None,
            "dshare_per_rupee_media": d_m / sm if sm else None,
            "converts_per_worker": w["converted_total"] / w["workers"] if w["workers"] else None,
            "converts_total_workers": w["converted_total"],
            "front_converts_per_worker": f["converted_total"] / f["workers"] if f["workers"] else None,
        })

    def col(k):
        return [s[k] for s in per_seed if s[k] is not None]
    out = {k: xc.mean_ci(col(k)) for k in ("dshare_workers", "dshare_media", "dshare_front", "dshare_per_worker",
                                           "dshare_per_rupee_workers", "dshare_per_rupee_media", "converts_per_worker",
                                           "converts_total_workers", "front_converts_per_worker", "workers",
                                           "spent_workers", "spent_media", "airings")}
    w = [r["runs"]["informed/pay_workers"]["expected"] for r in rows]
    m = [r["runs"]["informed/pay_media"]["expected"] for r in rows]
    workers_vs_media = xc.paired(w, m)
    workers_vs_media["workers_better_rate"] = xc.rate([a > b for a, b in zip(w, m)])
    workers_vs_media["mcnemar_wins"] = xc.mcnemar([r["runs"]["informed/pay_workers"]["victory"] for r in rows],
                                                  [r["runs"]["informed/pay_media"]["victory"] for r in rows])
    rpw, rpm = out["dshare_per_rupee_workers"]["mean"], out["dshare_per_rupee_media"]["mean"]
    out["workers_vs_media_expected"] = workers_vs_media
    out["per_rupee_ratio_workers_over_media"] = rpw / rpm if rpm else None
    frac = workers_vs_media["workers_better_rate"]["rate"]
    out["dominant_single_channel"] = "workers" if frac > 0.5 else ("media" if frac < 0.5 else "tie")
    out["per_seed"] = per_seed
    return out


def fit_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def loglog_slope(cells: dict[int, list[float]], boot: int = 2000, seed: int = 0) -> dict:
    """Slope of log10(mean metric) vs log10 N over the N's with a positive mean; seed-bootstrap percentile CI
    (seeds resampled independently within each cell)."""
    ns = sorted(n for n, v in cells.items() if v)
    means = {n: sum(cells[n]) / len(cells[n]) for n in ns}
    used = [n for n in ns if means[n] > 0]
    if len(used) < 2:
        return {"slope": None, "ci95": [None, None], "ns": used, "note": "fewer than 2 positive cell means"}
    xs = [math.log10(n) for n in used]
    slope = fit_slope(xs, [math.log10(means[n]) for n in used])
    rng = random.Random(seed)
    slopes, dropped = [], 0
    for _ in range(boot):
        ys = []
        for n in used:
            v = cells[n]
            mean = sum(v[rng.randrange(len(v))] for _ in range(len(v))) / len(v)
            if mean <= 0:
                break
            ys.append(math.log10(mean))
        if len(ys) < len(used):
            dropped += 1
            continue
        slopes.append(fit_slope(xs, ys))
    slopes.sort()
    lo = slopes[int(0.025 * len(slopes))] if slopes else None
    hi = slopes[min(len(slopes) - 1, int(0.975 * len(slopes)))] if slopes else None
    return {"slope": slope, "ci95": [lo, hi], "ns": used, "cell_means": {str(n): means[n] for n in ns},
            "bootstrap": boot, "bootstrap_dropped_nonpositive": dropped,
            "ci_contains_minus_1": (lo is not None and lo <= -1 <= hi), "ci_contains_0": (lo is not None and lo <= 0 <= hi)}


def summarize(cells: dict[tuple[int, int], list[dict]], ns: list[int], ts: list[int]) -> dict:
    out_cells, gate_map, dominance, best_map = {}, {}, {}, {}
    for (n, T), rows in sorted(cells.items()):
        g = gate_of(rows)
        ch = channel_metrics(rows)
        key = f"{n}|{T}"
        out_cells[key] = {"n": n, "rounds": T, "num_seeds": len(rows), **{k: v for k, v in g.items()
                                                                           if k != "best_paid_pattern_per_seed"},
                          "channels": ch}
        gate_map[key] = {"passed_all": g["gate_passed_all"], **g["gate"]}
        best = max(PAID_PATTERNS, key=lambda p: g["best_paid_pattern_counts"][p])
        best_map[key] = {"most_often_best": best, "counts": g["best_paid_pattern_counts"]}
        dominance[key] = {"single_channel": ch["dominant_single_channel"],
                          "workers_better_rate": ch["workers_vs_media_expected"]["workers_better_rate"]["rate"],
                          "mean_diff_workers_minus_media": ch["workers_vs_media_expected"]["mean"],
                          "per_rupee_ratio_workers_over_media": ch["per_rupee_ratio_workers_over_media"],
                          "most_often_best_paid_pattern": best}
    keys = [f"{n}|{T}" for n in ns for T in ts if f"{n}|{T}" in out_cells]
    for fam, getter in (("informed_vs_top3", lambda c: c["informed_vs_top3"]["wilcoxon_p"]),
                        ("workers_vs_media", lambda c: c["channels"]["workers_vs_media_expected"]["wilcoxon_p"])):
        adj = xc.holm([getter(out_cells[k]) for k in keys])
        for k, p in zip(keys, adj):
            if fam == "informed_vs_top3":
                out_cells[k]["informed_vs_top3"]["wilcoxon_p_holm"] = p
            else:
                out_cells[k]["channels"]["workers_vs_media_expected"]["wilcoxon_p_holm"] = p
    slopes = {}
    for metric in ("dshare_per_worker", "converts_per_worker", "dshare_per_rupee_media", "dshare_per_rupee_workers"):
        slopes[metric] = {}
        for T in ts:
            per_n = {n: [s[metric] for s in out_cells[f"{n}|{T}"]["channels"]["per_seed"] if s[metric] is not None]
                     for n in ns if f"{n}|{T}" in out_cells}
            slopes[metric][str(T)] = loglog_slope(per_n, seed=1000 * T + len(metric))
    return {"cells": out_cells, "gate_map": gate_map, "channel_dominance": dominance, "best_pattern_map": best_map,
            "slopes": slopes,
            "holm_families": {"informed_vs_top3": f"one-sided Wilcoxon per cell ({len(keys)} tests)",
                              "workers_vs_media": f"two-sided Wilcoxon of E[pay_workers] − E[pay_media] per cell ({len(keys)} tests)"}}


# ============================================================================== pilot
def pilot(ns, ts, treasury, device, seeds, log=print) -> dict:
    """Time ``len(seeds)`` seeds per N (all T, all runs), excluding the one-off chunk planning."""
    per_seed = {}
    for n in ns:
        base = base_config(n, treasury, device)
        t0 = time.perf_counter()
        chunk = prepare_chunk(seeds, base, device, tuple(ts))
        for T in ts:
            play_runs(chunk, base, T)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        per_seed[n] = (time.perf_counter() - t0) / len(seeds)
        del chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"[pilot] N={n}: {per_seed[n]:.2f} s per seed (all {len(ts)} T, {len(RUNS)} runs each)")
    return per_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--seeds-1e6", type=int, default=None, help="seeds at N = 10^6 (default: --seeds, auto-reduced)")
    parser.add_argument("--ns", type=int, nargs="+", default=NS)
    parser.add_argument("--rounds", type=int, nargs="+", default=TS)
    parser.add_argument("--treasury", default="500")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--gpu-hours", type=float, default=3.0, help="runtime budget for the full grid")
    parser.add_argument("--pilot-seeds", type=int, default=2)
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--summarize-only", action="store_true", help="rebuild x10.json from the raw JSONL")
    parser.add_argument("--out", default=str(xc.REPORTS / "x10.json"))
    parser.add_argument("--raw", default=str(xc.RAW / "x10.jsonl"))
    args = parser.parse_args()
    start = time.perf_counter()
    raw = Path(args.raw)
    ns, ts = sorted(args.ns), sorted(args.rounds)
    seeds = xc.seed_block("X10", args.seeds)

    if args.summarize_only:
        prior = json.loads(Path(args.out).read_text())
        cells: dict[tuple[int, int], dict[int, dict]] = {}
        for row in xc.read_rows(raw):
            c = cells.setdefault((row["n"], row["rounds"]), {})
            s = c.setdefault(row["seed"], {"seed": row["seed"], "personality": row["personality"], "sizes": row["sizes"],
                                           "runs": {}})
            s["runs"][f"{row['manifesto']}/{row['policy']}"] = row
        report = {**prior, **summarize({k: [v[s] for s in sorted(v)] for k, v in cells.items()}, ns, ts)}
        xc.save(args.out, report)
        return

    device, dev_record = xc.select_device(args.device, need_bytes=8 * 2**30)
    free_at_start = xc.free_bytes(device)
    record: dict = {}
    deviations = []
    with xc.GpuMonitor(device) as monitor:
        par = parity(base_config(10_000, args.treasury, device), seeds[:args.parity_seeds], device)
        print(f"[x10] parity passed={par['passed']}", flush=True)
        # ---- pilot and runtime estimate
        pilot_seeds = seeds[:args.pilot_seeds]
        per_seed = pilot(ns, ts, args.treasury, device, pilot_seeds)
        seeds_by_n = {n: (args.seeds_1e6 or args.seeds) if n >= 1_000_000 else args.seeds for n in ns}
        est = sum(per_seed[n] * seeds_by_n[n] for n in ns)
        budget = args.gpu_hours * 3600
        print(f"[x10] pilot estimate for the full grid: {est / 60:.1f} min ({est / 3600:.2f} h; budget "
              f"{args.gpu_hours} h)", flush=True)
        if est > budget and 1_000_000 in seeds_by_n:
            rest = est - per_seed[1_000_000] * seeds_by_n[1_000_000]
            fit = int((budget - rest) // per_seed[1_000_000])
            reduced = max(MIN_SEEDS_1E6, min(seeds_by_n[1_000_000], fit))
            deviations.append(f"N=10^6 seeds reduced {seeds_by_n[1_000_000]} -> {reduced} (pilot estimate "
                              f"{est / 3600:.2f} h > {args.gpu_hours} h)")
            seeds_by_n[1_000_000] = reduced
            print(f"[x10] DEVIATION: {deviations[-1]}", flush=True)
        if any(seeds_by_n[n] < args.seeds for n in ns) and not deviations:
            deviations.append(f"seeds per N set by hand: {seeds_by_n}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        # ---- full grid
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("")
        cells: dict[tuple[int, int], list[dict]] = {}
        plans, seconds_by_n = {}, {}
        for n in ns:
            t0 = time.perf_counter()
            rows, plans[n] = run_n(n, seeds[:seeds_by_n[n]], ts, args.treasury, device, raw=raw, record=record)
            seconds_by_n[n] = time.perf_counter() - t0
            for T in ts:
                cells[(n, T)] = rows[T]
            print(f"[x10] N={n} done in {seconds_by_n[n]:.0f}s", flush=True)
    summary = summarize(cells, ns, ts)
    seconds = time.perf_counter() - start
    prov = xc.provenance_v3(args, device, seconds,
                            batch_sizes={"prepare_chunk_by_n": {str(n): p["chunk"] for n, p in plans.items()},
                                         "run_batch_B_used": record.get("batch_sizes")},
                            free_at_start=free_at_start, monitor=monitor, device_record=dev_record, parity=par,
                            extra={"command": f"python experiments/x10_n_t_balance.py --seeds {args.seeds} --device "
                                              f"{args.device} --treasury {args.treasury} --out {args.out}",
                                   "pilot": {"seeds": pilot_seeds, "seconds_per_seed_by_n": {str(k): v for k, v in per_seed.items()},
                                             "estimated_full_seconds": est, "budget_seconds": budget},
                                   "seconds_by_n": {str(k): v for k, v in seconds_by_n.items()},
                                   "batch_plans": {str(k): v for k, v in plans.items()},
                                   "oom_backoffs": record.get("oom_backoffs", [])})
    report = {
        "experiment": "X10", "title": "Balance across the player-chosen N and T",
        "prices": {"worker": "20.00", "query": "10.00"}, "treasury": args.treasury, "ns": ns, "rounds": ts,
        "seeds": seeds, "seeds_by_n": {str(k): v for k, v in seeds_by_n.items()},
        "generation_by_n": {str(n): base_config(n, args.treasury, device).generation for n in ns},
        "generation_note": "N = 10^6 uses generation=\"device\" (GPU draws, plan §1.1); smaller N use the canonical CPU draws (R8)",
        "runs": [f"{m}/{p}" for m, p in RUNS],
        "single_channel_arms": {"workers_only": "informed/pay_workers", "media_only": "informed/pay_media",
                                "none": "informed/free_even"},
        "deviations": deviations, **summary,
        "config_example": base_config(10_000, args.treasury, device).to_record(), "provenance": prov,
    }
    xc.save(args.out, report)
    for n in ns:
        print(" ".join(f"N={n} T={T}: {'PASS' if summary['gate_map'][f'{n}|{T}']['passed_all'] else 'fail'} "
                       f"{summary['channel_dominance'][f'{n}|{T}']['single_channel']}" for T in ts), flush=True)
    for metric, by_t in summary["slopes"].items():
        print(metric, {T: (round(v["slope"], 3) if v["slope"] is not None else None,
                           [round(x, 3) if x is not None else None for x in v["ci95"]]) for T, v in by_t.items()})
    c = prov["compute"]
    print(f"[x10] wall {seconds:.0f}s, peak {(c['peak_bytes'] or 0) / 2**30:.2f} GiB, GPU util mean {c.get('gpu_util_mean')}")


if __name__ == "__main__":
    main()
