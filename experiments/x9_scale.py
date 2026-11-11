"""X9 — Scale and GPU performance (plan ``docs/v2/plans/implementation_plan.md`` §8 X9; port of E7).

Command (reproduces reports/v3/x9.json, reports/v3/raw/x9.jsonl and reports/v3/logs/x9.log):
  OMP_NUM_THREADS=4 python experiments/x9_scale.py --device cuda:2 --repeats 3 --out reports/v3/x9.json \
      2>&1 | tee reports/v3/logs/x9.log

Canonical seed 20261107 (§1.5), 3 timed repeats after one warm-up. Each (N, device, B) runs in a FRESH subprocess so
that the peak memory is clean (GPU: ``torch.cuda.max_memory_allocated``; host: ``ru_maxrss``).
* N ∈ {10⁴, 10⁵, 10⁶}; batched B ∈ {1, 8, 16, 32} (16 added) worlds per ``x_batch.run_batch`` call on cuda:2 (B copies of the
  canonical world joined into one disjoint-union graph: the timing does not depend on which worlds are joined);
  CPU reference B ∈ {1, 8} at N ≤ 10⁵ and B = 1 at 10⁶. A configuration that does not fit is recorded as OOM.
* Game: T = 5, policy pay_split (workers AND media every round, so the cascade hop, media and social step all run),
  manifesto = global top-3 of v(0) (no optimiser in the timing). Reported: ms per round (per call) and per
  world-round, games per second, generation time (GPU: ``generation="device"``; CPU: the canonical CPU draw).
  The single-world library path ``simulate_world`` is timed next to ``run_batch`` at B = 1.
* Cascade hop O(edges): ``spreader_edge_counts`` + ``ic_from_counts`` timed alone on the canonical world for spreader
  fractions {0.001, 0.01, 0.1} (20 synchronised repetitions); the log–log slope of hop time vs the number of edges
  over N is the scaling check (1 = linear in edges).
* CPU↔GPU parity at N = 10⁴ on 3 seeds (20261107–20261109), pay_split, metric-appropriate tolerance.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402

SEED = 20261107
NS = (10_000, 100_000, 1_000_000)
GPU_BS = (1, 8, 16, 32)   # plan: {1, 8, 32}; 16 added to locate the largest batch that fits at 10⁶
CPU_PLAN = {10_000: (1, 8), 100_000: (1, 8), 1_000_000: (1,)}
FRACTIONS = (0.001, 0.01, 0.1)
RAW_PATH = xc.RAW / "x9.jsonl"


def child(n: int, device: str, B: int, repeats: int) -> dict:
    import dataclasses
    import resource

    import torch
    import x_policies
    from x_batch import run_batch

    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.dynamics.campaign import conversion_chance, ic_from_counts, spreader_edge_counts
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.simulate import simulate_world
    from dumbocracy.strategies.oracle import top3_manifesto

    torch.set_num_threads(4)
    dev = torch.device(device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.set_device(dev)

    def sync():
        if cuda:
            torch.cuda.synchronize(dev)
    cfg = GameConfig(seed=SEED, device=device, generation="device" if cuda else "cpu",
                     electorate=ElectorateConfig(num_voters=n))
    out: dict = {"n": n, "device": device, "B": B, "repeats": repeats, "generation": cfg.generation}
    try:
        sync()
        t0 = time.perf_counter()
        e = generate_electorate(cfg, device=dev)
        sync()
        out["generation_seconds"] = time.perf_counter() - t0
        out["edges_undirected"] = int(e.graph.num_edges)
        out["mean_degree"] = 2 * e.graph.num_edges / n
        b = top3_manifesto(e.preferences)
        policy = x_policies.GATE_POLICIES["pay_split"]
        if cuda:
            torch.cuda.reset_peak_memory_stats(dev)
            base_mem = torch.cuda.memory_allocated(dev)
            out["input_bytes"] = int(base_mem)
        times = []
        for rep in range(repeats + 1):            # rep 0 = warm-up
            sync()
            t0 = time.perf_counter()
            outs = run_batch([e] * B, cfg, b.expand(B, 5), policy, seeds=[SEED] * B)
            sync()
            if rep > 0:
                times.append(time.perf_counter() - t0)
        T = cfg.rounds
        out["run_batch_seconds"] = times
        out["ms_per_round"] = 1000 * min(times) / T
        out["ms_per_world_round"] = 1000 * min(times) / (T * B)
        out["games_per_second"] = B / min(times)
        out["expected_share"] = outs[0].result.expected_share
        out["converts"] = int(sum(outs[0].converted))
        if cuda:
            out["peak_bytes"] = int(torch.cuda.max_memory_allocated(dev))
            out["peak_bytes_above_inputs"] = int(torch.cuda.max_memory_allocated(dev) - base_mem)
        if B == 1:      # library single-world path
            st = []
            for rep in range(repeats + 1):
                sync()
                t0 = time.perf_counter()
                simulate_world(e, cfg, b, policy)
                sync()
                if rep > 0:
                    st.append(time.perf_counter() - t0)
            out["simulate_world_ms_per_round"] = 1000 * min(st) / T
            # cascade hop alone: O(edges)?
            rows = e.graph.rows()
            chance = conversion_chance(e.susceptibility.campaign, e.preferences, b, cfg.voting.sensitivity,
                                       cfg.voting.threshold)
            hop = {}
            for f in FRACTIONS:
                g = torch.Generator(device=dev).manual_seed(1)
                spreaders = torch.rand(n, generator=g, device=dev) < f
                attempts = torch.zeros(n, dtype=torch.int64, device=dev)
                blocked = e.susceptibility.zealot.to(torch.bool)
                ts = []
                for rep in range(21):
                    gen = torch.Generator(device=dev).manual_seed(rep)
                    sync()
                    t0 = time.perf_counter()
                    counts = spreader_edge_counts(e.graph, spreaders, rows)
                    ic_from_counts(counts, chance, attempts, blocked, cfg.workers.attempt_cap, gen)
                    sync()
                    if rep > 0:
                        ts.append(time.perf_counter() - t0)
                ts.sort()
                hop[str(f)] = {"median_ms": 1000 * ts[len(ts) // 2], "min_ms": 1000 * ts[0],
                               "spreader_edges": int(spreaders[rows].sum())}
            out["cascade_hop"] = hop
        out["oom"] = False
    except torch.cuda.OutOfMemoryError as exc:
        out.update(oom=True, error=str(exc).splitlines()[0])
    except RuntimeError as exc:                   # e.g. CPU allocation failure
        out.update(oom=True, error=str(exc).splitlines()[0])
    out["host_peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return out


def parity(device: str) -> dict:
    import dataclasses

    import torch
    import x_policies
    from x_batch import run_batch
    from x_parity import parity_rel

    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.strategies.oracle import top3_manifesto

    def fn(seed, dev):
        cfg = GameConfig(seed=seed, device=str(dev), electorate=ElectorateConfig(num_voters=10_000))
        e = generate_electorate(cfg, device=dev)
        o = run_batch([e], cfg, top3_manifesto(e.preferences)[None], x_policies.GATE_POLICIES["pay_split"], seeds=[seed])[0]
        return {"expected": o.result.expected_share, "converts_share_of_n": sum(o.converted) / 10_000}
    out = parity_rel(fn, [SEED, SEED + 1, SEED + 2], device=torch.device(device))
    out["cell"] = "N = 10⁴, pay_split, top-3 manifesto, canonical CPU generation"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ns", type=int, nargs="*", default=list(NS))
    parser.add_argument("--out", default=str(xc.REPORTS / "x9.json"))
    parser.add_argument("--child", nargs=4, metavar=("N", "DEVICE", "B", "REPEATS"))
    args = parser.parse_args()
    if args.child:
        n, dev, B, rep = args.child
        print(json.dumps(child(int(n), dev, int(B), int(rep))))
        return
    import torch
    device, dev_record = xc.select_device(args.device, need_bytes=1 * 2**30)
    if args.device.startswith("cuda") and str(device) != str(torch.device(args.device)):
        raise SystemExit(f"refusing to fall back from {args.device} to {device} (shared host: only the requested GPU)")
    if device.type == "cuda":
        torch.cuda.set_device(device)   # no stray context on the default GPU
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text("")
    start = time.perf_counter()
    results = []
    plan = [(n, str(device), B) for n in args.ns for B in GPU_BS] + \
           [(n, "cpu", B) for n in args.ns for B in CPU_PLAN.get(n, (1,))]
    with xc.GpuMonitor(device) as monitor:
        for n, dev, B in plan:
            t0 = time.perf_counter()
            proc = subprocess.run([sys.executable, __file__, "--child", str(n), dev, str(B), str(args.repeats)],
                                  capture_output=True, text=True)
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = {"n": n, "device": dev, "B": B, "oom": True, "error": (proc.stderr or "no output")[-500:]}
            if proc.returncode != 0 and "ms_per_round" not in row:
                row.update(oom=True, error=(proc.stderr or "")[-500:])
            row["wall_seconds"] = time.perf_counter() - t0
            results.append(row)
            xc.write_row(RAW_PATH, {"exp": "X9", **row})
            print(f"[x9] N={n} {dev} B={B}: " + ("OOM/failed" if row.get("oom") else
                  f"{row['ms_per_round']:.1f} ms/round, {row['ms_per_world_round']:.2f} ms/world-round, peak "
                  f"{(row.get('peak_bytes') or 0) / 2**30:.2f} GiB"), flush=True)
        par = parity(str(device))
        print(f"[x9] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    # summaries
    import numpy as np
    ok = [r for r in results if not r.get("oom")]
    table = {f"{r['device']}|N{r['n']}|B{r['B']}": {k: r.get(k) for k in (
        "ms_per_round", "ms_per_world_round", "games_per_second", "peak_bytes", "peak_bytes_above_inputs",
        "generation_seconds", "simulate_world_ms_per_round", "host_peak_rss_mib", "edges_undirected")}
        for r in ok}
    speedup = {}
    for n in args.ns:
        g1 = next((r for r in ok if r["n"] == n and r["device"] != "cpu" and r["B"] == 1), None)
        c1 = next((r for r in ok if r["n"] == n and r["device"] == "cpu" and r["B"] == 1), None)
        gb = [r for r in ok if r["n"] == n and r["device"] != "cpu"]
        cb = [r for r in ok if r["n"] == n and r["device"] == "cpu"]
        if g1 and c1:
            speedup[f"N{n}"] = {
                "single_world_gpu_over_cpu": c1["ms_per_round"] / g1["ms_per_round"],
                "best_batched_gpu_world_round_ms": min(r["ms_per_world_round"] for r in gb),
                "best_cpu_world_round_ms": min(r["ms_per_world_round"] for r in cb),
                "batched_gpu_over_best_cpu": min(r["ms_per_world_round"] for r in cb) / min(r["ms_per_world_round"] for r in gb),
                "largest_B_that_fit": max(r["B"] for r in gb)}
    hop = {}
    for dev_kind in ("gpu", "cpu"):
        rs = [r for r in ok if r["B"] == 1 and (r["device"] == "cpu") == (dev_kind == "cpu") and "cascade_hop" in r]
        for f in FRACTIONS:
            e = np.array([r["edges_undirected"] for r in rs], float)
            t = np.array([r["cascade_hop"][str(f)]["median_ms"] for r in rs], float)
            if len(rs) >= 2:
                slope = float(np.polyfit(np.log(e), np.log(t), 1)[0])
                hop[f"{dev_kind}|f{f}"] = {"loglog_slope_time_vs_edges": slope, "ns": [r["n"] for r in rs],
                                           "median_ms": t.tolist(), "ns_per_edge": (1e6 * t / (2 * e)).tolist()}
    report = {"experiment": "X9", "question": "Scale and GPU performance; is the cascade hop O(edges)?",
              "seeds": {"canonical": SEED, "repeats": args.repeats, "parity": [SEED, SEED + 1, SEED + 2]},
              "ns": list(args.ns), "gpu_batches": list(GPU_BS), "cpu_plan": {str(k): v for k, v in CPU_PLAN.items()},
              "metric_definitions": {
                  "ms_per_round": "min over repeats of run_batch wall time / T (one call, B worlds)",
                  "ms_per_world_round": "ms_per_round / B", "peak_bytes": "torch.cuda.max_memory_allocated in the child",
                  "cascade_hop": "spreader_edge_counts + ic_from_counts, median of 20 synchronised repetitions",
                  "loglog_slope_time_vs_edges": "slope of log(hop ms) on log(edges) over N (1 = O(edges))"},
              "results": results, "table": table, "speedup": speedup, "cascade_hop_scaling": hop,
              "deviations": ["Batched B uses B copies of the canonical world (timing does not depend on which worlds are "
                             "joined); GPU generation uses generation='device' (§1.1 point 3), CPU the canonical CPU draw.",
                             "CPU batched runs only up to B = 8 and N ≤ 10⁵ (CPU time), B = 1 at 10⁶."],
              "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=list(GPU_BS), monitor=monitor,
                                             device_record=dev_record, parity=par,
                                             extra={"note": "peak memory per configuration is in results[*].peak_bytes "
                                                            "(fresh subprocess each)"})}
    xc.save(args.out, report)
    print(f"[x9] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
