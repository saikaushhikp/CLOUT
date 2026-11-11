"""Phase 0 GPU benchmarks (plan §1.1 points 3–4, §10 Phase 0) → ``reports/v3/gpu_benchmarks.json``.

Command:
  OMP_NUM_THREADS=4 python experiments/x_gpu_benchmarks.py --device cuda:2 --out reports/v3/gpu_benchmarks.json

Sections:
* ``x_batch``: the batched true-world runner (``x_batch.run_batch``) at N ∈ {1e4, 1e5} and B ∈ {1, 8, 32, 64} on
  the GPU vs single-world ``simulate_world`` on the CPU (and on the GPU), with peak memory and the
  memory-based batch-size estimate;
* ``parity``: CPU bit-exactness of the batch vs ``simulate_world`` (20 seeds), GPU agreement of the batch with
  ``simulate_world`` on the same device (20 seeds), and the statistical CPU↔GPU check (``parity_check``);
* ``bayes_planner``: BayesPlanner decision latency with its synthetic worlds on CPU vs GPU, batched worlds
  ``num_worlds`` ∈ {8, 32} × ``world_voters`` ∈ {1600, 4000};
* ``logic``: epistemic timeline (``consistent_worlds`` + plan evaluation) with 16 / 64 worlds on CPU vs GPU;
* ``spectra``: ``x_common.spectra_gpu`` (torch.lobpcg) vs the SciPy reference at N ∈ {1e4, 1e5}.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import statistics
import time
from pathlib import Path

import torch

import x_common as xc
import x_policies
from x_batch import run_batch

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.electorate import generate_electorate
from dumbocracy.simulate import simulate_world
from dumbocracy.strategies.oracle import oracle_manifesto, top3_manifesto


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device: torch.device):
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - start


def _same(a, b) -> bool:
    return (a.result == b.result and a.share_trajectory == b.share_trajectory and a.converted == b.converted
            and a.spreaders == b.spreaders and a.allocations == b.allocations and a.spent == b.spent
            and torch.equal(a.final_preferences.cpu(), b.final_preferences.cpu()))


def _discrete(a, b) -> bool:
    return (a.result.votes == b.result.votes and a.result.victory == b.result.victory and a.converted == b.converted
            and a.spreaders == b.spreaders and a.allocations == b.allocations and a.spent == b.spent)


# ------------------------------------------------------------------------------------------ x_batch
def bench_batch(n: int, batches: list[int], device: torch.device, policies: list[str], cpu_worlds: int,
                first_seed: int) -> dict:
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=n))
    bmax = max(batches)
    seeds = list(range(first_seed, first_seed + bmax))
    start = time.perf_counter()
    cpu_es = [generate_electorate(dataclasses.replace(base, seed=s), device="cpu") for s in seeds[:cpu_worlds]]
    gpu_es = [generate_electorate(dataclasses.replace(base, seed=s), device=device) for s in seeds]
    gen_seconds = time.perf_counter() - start
    ms = [top3_manifesto(e.preferences) for e in gpu_es]
    out = {"n": n, "seeds": [seeds[0], seeds[-1]], "manifesto": "top3", "generation": "cpu (R8), moved to device",
           "electorate_generation_seconds_total": gen_seconds, "policies": {}}
    for name in policies:
        policy = x_policies.GATE_POLICIES[name]
        # CPU reference: single-world simulate_world
        cpu_times = []
        for e, s, m in zip(cpu_es, seeds, ms):
            _, sec = _timed(lambda: simulate_world(e, dataclasses.replace(base, seed=s, device="cpu"), m.cpu(), policy),
                            torch.device("cpu"))
            cpu_times.append(sec)
        cpu_per_world = statistics.mean(cpu_times)
        # GPU single world (simulate_world on the GPU), warm-up first
        simulate_world(gpu_es[0], dataclasses.replace(base, seed=seeds[0]), ms[0], policy)
        gpu_single = []
        for e, s, m in list(zip(gpu_es, seeds, ms))[:min(4, bmax)]:
            _, sec = _timed(lambda: simulate_world(e, dataclasses.replace(base, seed=s), m, policy), device)
            gpu_single.append(sec)
        rows = []
        for B in batches:
            gc.collect()
            torch.cuda.empty_cache()
            run_batch(gpu_es[:1], base, ms[0][None], policy, seeds=seeds[:1])     # warm-up
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
            try:
                _, sec = _timed(lambda: run_batch(gpu_es[:B], base, torch.stack(ms[:B]), policy, seeds=seeds[:B]), device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append({"B": B, "oom": True})
                continue
            peak = torch.cuda.max_memory_allocated(device) - before
            rows.append({"B": B, "seconds": sec, "seconds_per_world": sec / B,
                         "speedup_vs_cpu_single": cpu_per_world / (sec / B),
                         "speedup_vs_gpu_single": statistics.mean(gpu_single) / (sec / B),
                         "peak_bytes_above_inputs": int(peak), "peak_gib_above_inputs": peak / 2**30})
            print(f"N={n} {name} B={B}: {sec:.2f}s ({sec / B * 1000:.0f} ms/world), x{cpu_per_world / (sec / B):.1f} vs CPU,"
                  f" peak {peak / 2**30:.2f} GiB", flush=True)
        out["policies"][name] = {"cpu_simulate_world_seconds_per_world": cpu_per_world, "cpu_worlds_timed": len(cpu_times),
                                 "gpu_simulate_world_seconds_per_world": statistics.mean(gpu_single), "batched": rows}
    # memory-based batch sizer (calibration at B = 1, 2), the inputs (electorates) excluded
    sizer = xc.calibrate_batch(lambda B: run_batch(gpu_es[:B], base, torch.stack(ms[:B]), x_policies.policy_pay_split,
                                                   seeds=seeds[:B]), device, sizes=(2, 4))
    per_world_input = sum(t.numel() * t.element_size() for t in (
        gpu_es[0].graph.src, gpu_es[0].graph.dst, gpu_es[0].graph.indices, gpu_es[0].graph.indptr, gpu_es[0].graph.degree,
        gpu_es[0].preferences, gpu_es[0].region_of, gpu_es[0].susceptibility.social, gpu_es[0].susceptibility.media,
        gpu_es[0].susceptibility.campaign, gpu_es[0].susceptibility.zealot))
    sizer["input_bytes_per_world"] = int(per_world_input)
    sizer["recommended_batch_incl_inputs"] = int(0.7 * sizer["free_bytes"] // (sizer["per_item_bytes"] + per_world_input))
    out["batch_sizer"] = sizer
    del gpu_es, cpu_es
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------ parity
def parity(device: torch.device, seeds: list[int], n: int) -> dict:
    base = GameConfig(electorate=ElectorateConfig(num_voters=n))
    policy = x_policies.policy_planner
    out: dict = {"n": n, "seeds": [seeds[0], seeds[-1]], "count": len(seeds), "policy": "planner (ORACLE)",
                 "manifesto": "oracle"}
    # CPU: batch of all seeds vs per-world simulate_world, bit for bit
    cfg = dataclasses.replace(base, device="cpu")
    es = [generate_electorate(dataclasses.replace(cfg, seed=s), device="cpu") for s in seeds]
    ms = [oracle_manifesto(e, dataclasses.replace(cfg, seed=s)) for e, s in zip(es, seeds)]
    batch = run_batch(es, cfg, torch.stack(ms), policy, seeds=seeds, keep_final=True)
    ref = [simulate_world(e, dataclasses.replace(cfg, seed=s), m, policy) for e, s, m in zip(es, seeds, ms)]
    out["cpu_batch_vs_simulate_world"] = {"B": len(seeds), "bit_identical_worlds": sum(_same(a, b) for a, b in zip(ref, batch)),
                                          "aggregation": "exact"}
    # GPU: batch vs simulate_world on the same device; and simulate_world vs itself
    gcfg = dataclasses.replace(base, device=str(device))
    ges = [generate_electorate(dataclasses.replace(gcfg, seed=s), device=device) for s in seeds]
    gms = [m.to(device) for m in ms]
    gref = [simulate_world(e, dataclasses.replace(gcfg, seed=s), m, policy) for e, s, m in zip(ges, seeds, gms)]
    gref2 = [simulate_world(e, dataclasses.replace(gcfg, seed=s), m, policy) for e, s, m in zip(ges, seeds, gms)]
    rec = {}
    for agg in ("csr", "exact"):
        gb = run_batch(ges, gcfg, torch.stack(gms), policy, seeds=seeds, aggregation=agg, keep_final=True)
        rec[agg] = {"discrete_identical_worlds": sum(_discrete(a, b) for a, b in zip(gref, gb)),
                    "bit_identical_worlds": sum(_same(a, b) for a, b in zip(gref, gb)),
                    "max_abs_diff_expected_share": max(abs(a.result.expected_share - b.result.expected_share)
                                                       for a, b in zip(gref, gb)),
                    "max_abs_diff_trajectory": max(max(abs(x - y) for x, y in zip(a.share_trajectory, b.share_trajectory))
                                                   for a, b in zip(gref, gb))}
    out["gpu_batch_vs_simulate_world"] = rec
    out["gpu_simulate_world_run_to_run"] = {
        "bit_identical_worlds": sum(_same(a, b) for a, b in zip(gref, gref2)),
        "discrete_identical_worlds": sum(_discrete(a, b) for a, b in zip(gref, gref2)),
        "note": "CUDA index_add_ (DeGroot aggregation) uses float atomics, so simulate_world itself is not bitwise "
                "reproducible on the GPU; only discrete outcomes are compared there."}
    # CPU vs GPU, statistical (different random streams per device)
    cpu_by_seed = {s: o for s, o in zip(seeds, ref)}
    gpu_by_seed = {s: o for s, o in zip(seeds, gref)}

    def fn(seed, dev):
        o = cpu_by_seed[seed] if dev.type == "cpu" else gpu_by_seed[seed]
        return {"expected": o.result.expected_share, "victory": o.result.victory, "converted": sum(o.converted)}

    chk = xc.parity_check(fn, seeds, device=device, abs_tol=0.01)
    chk["metrics"]["converted"]["note"] = "tolerance in voters"
    chk.pop("rows")
    out["cpu_vs_gpu_statistical"] = chk
    del ges
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------ BayesPlanner / logic
def _game(seed: int, n: int, device: torch.device):
    from dumbocracy.game import GameEngine
    cfg = GameConfig(seed=seed, device=str(device), electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=device)
    return GameEngine.new_game(cfg, electorate=e, game_id=f"b{seed}")


def _gpu_planner_class():
    """``BayesPlanner(device="cuda")`` crashes in ``manifesto()`` (players.py: the free-stat start is built on the
    CPU and stacked with GPU tensors). Experiment-side workaround, no library change: the manifesto worlds stay
    on the CPU (one call per game), the per-round rollout worlds use ``device``."""
    from dumbocracy.strategies.players import BayesPlanner

    class GpuRolloutBayesPlanner(BayesPlanner):
        def manifesto(self, observation, posterior=None):
            device, self.device = self.device, "cpu"
            try:
                return super().manifesto(observation, posterior)
            finally:
                self.device = device

    return GpuRolloutBayesPlanner


def bench_bayes(device: torch.device, seeds: list[int], settings: list[tuple[int, int]]) -> list[dict]:
    from dumbocracy.strategies.players import BayesPlanner, play_game
    gpu_planner = _gpu_planner_class()
    rows = []
    for worlds, voters in settings:
        for dev in ("cpu", str(device)):
            lat, plan_lat, wins = [], [], []
            for s in seeds:
                state0 = _game(s, 10_000, device)
                cls = BayesPlanner if dev == "cpu" else gpu_planner
                player = cls(seed=s, num_worlds=worlds, manifesto_worlds=8, world_voters=voters, device=dev)
                final, _ = play_game(state0, player)
                lat += [x for xs in player.latency.values() for x in xs]
                plan_lat += [d["seconds"] for d in player.diagnostics]
            rows.append({"num_worlds": worlds, "world_voters": voters, "device": dev, "games": len(seeds),
                         "player": "BayesPlanner" if dev == "cpu" else "BayesPlanner, manifesto worlds on CPU (workaround)",
                         "decision_seconds_mean": statistics.mean(lat), "allocate_seconds_mean": statistics.mean(plan_lat)})
            print(f"BayesPlanner worlds={worlds} voters={voters} {dev}: allocate {statistics.mean(plan_lat):.3f}s", flush=True)
    return rows


def bench_logic(device: torch.device, seeds: list[int], world_counts: list[int]) -> list[dict]:
    from dumbocracy.logic.report import timeline_with_diagnostics
    from dumbocracy.strategies.players import Top3Player, play_game
    finals = [play_game(_game(s, 10_000, device), Top3Player())[0] for s in seeds]
    rows = []
    for worlds in world_counts:
        for dev in ("cpu", str(device)):
            times = []
            for final in finals:
                _, sec = _timed(lambda: timeline_with_diagnostics(final, "p0", worlds=worlds, world_voters=1600, device=dev),
                                device)
                times.append(sec)
            rows.append({"worlds": worlds, "world_voters": 1600, "device": dev, "games": len(finals),
                         "timeline_seconds_mean": statistics.mean(times)})
            print(f"logic timeline worlds={worlds} {dev}: {statistics.mean(times):.2f}s", flush=True)
    return rows


def bench_spectra(device: torch.device, sizes: list[int]) -> list[dict]:
    rows = []
    for n in sizes:
        cfg = GameConfig(seed=10000, electorate=ElectorateConfig(num_voters=n))
        e = generate_electorate(cfg, device=device)
        xc.spectra_gpu(e.graph)  # warm-up
        gpu, sec_gpu = _timed(lambda: xc.spectra_gpu(e.graph), device)
        cpu_graph = generate_electorate(cfg, device="cpu").graph
        ref, sec_cpu = _timed(lambda: xc.spectra_scipy(cpu_graph), torch.device("cpu"))
        rows.append({"n": n, "gpu_lobpcg_seconds": sec_gpu, "scipy_seconds": sec_cpu,
                     "lambda2_L": [gpu["lambda2_L"], ref["lambda2_L"]], "mu2": [gpu["mu2"], ref["mu2"]],
                     "mu_min": [gpu["mu_min"], ref["mu_min"]]})
        print(f"spectra N={n}: gpu {sec_gpu:.2f}s scipy {sec_cpu:.2f}s", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=xc.DEFAULT_DEVICE)
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    parser.add_argument("--sections", default="x_batch,parity,bayes,logic,spectra")
    parser.add_argument("--out", default=str(xc.REPORTS / "gpu_benchmarks.json"))
    parser.add_argument("--merge", action="store_true", help="update only the chosen sections of an existing --out")
    args = parser.parse_args()
    device, record = xc.select_device(args.device, need_bytes=8 * 2**30)
    free0 = xc.free_bytes(device)
    sections = set(args.sections.split(","))
    first = xc.seed_block("X9", 1)[0]
    report: dict = {"device_selection": record}
    if args.merge and Path(args.out).exists():
        report = json.loads(Path(args.out).read_text())
        report.setdefault("section_provenance", {})
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        if "x_batch" in sections:
            grid = [(10_000, [1, 8], 2), (100_000, [1, 8], 1)] if args.quick else \
                [(10_000, [1, 8, 32, 64], 8), (100_000, [1, 8, 32, 64], 3)]
            report["x_batch"] = [bench_batch(n, bs, device, ["pay_split", "planner"], cpu_w, first) for n, bs, cpu_w in grid]
        if "parity" in sections:
            report["parity"] = parity(device, xc.seed_block("X0", 4 if args.quick else 20), 10_000)
        if "bayes" in sections:
            settings = [(8, 1600)] if args.quick else [(8, 1600), (32, 1600), (8, 4000), (32, 4000)]
            report["bayes_planner"] = bench_bayes(device, [first + 1] if args.quick else [first + 1, first + 2], settings)
        if "logic" in sections:
            report["logic"] = bench_logic(device, [first + 1] if args.quick else [first + 1, first + 2],
                                          [16] if args.quick else [16, 64])
        if "spectra" in sections:
            report["spectra"] = bench_spectra(device, [10_000] if args.quick else [10_000, 100_000])
    seconds = time.perf_counter() - start
    prov = xc.provenance_v3(args, device, seconds, free_at_start=free0, monitor=monitor,
                            device_record=record, batch_sizes="see x_batch[*].batched[*].B")
    if args.merge and "section_provenance" in report:
        report["section_provenance"][args.sections] = prov
    else:
        report["provenance"] = prov
    xc.save(args.out, report)


if __name__ == "__main__":
    main()
