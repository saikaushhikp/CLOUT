"""X18 — Submodularity of the worker value curve (plan ``docs/v2/plans/implementation_plan.md`` §4 X18; new, N9).

Command (reproduces reports/v3/x18.json, reports/v3/raw/x18.jsonl and reports/v3/logs/x18.log):
  OMP_NUM_THREADS=4 python experiments/x18_submodularity.py --seeds 50 --first-seed 11000 --device cuda:2 \
      --out reports/v3/x18.json 2>&1 | tee reports/v3/logs/x18.log

N = 10⁴, T = 5, ORACLE manifesto, ₫20 workers, treasury ₫1000 (so that 40 workers, ₫800, are affordable; nothing
else is bought, so the treasury has no other effect). 50 paired seeds (block 11000).

1. Value curve. k workers are bought in round 1 in ONE region (each of the 4 regions in turn, indexed by size rank
   0 = largest), k ∈ {0, 1, …, 10, 15, 20, …, 40}. The library pick (R22) draws one ``randperm`` of the region's
   eligible voters from the round-1 ``workers`` stream and takes its first k entries, so the worker sets are NESTED
   in k (checked) and all other randomness is common (CRN): the curve is a paired, within-world curve.
   Marginal gain per worker g(k_i) = [E(k_i) − E(k_{i−1})] / (k_i − k_{i−1}). For each adjacent pair (g_i, g_{i+1})
   and size rank, a paired bootstrap over seeds (4,000 resamples) of mean(g_{i+1} − g_i): a pair VIOLATES
   diminishing returns "beyond noise" if the 95% CI lies above 0, and is DECREASING if it lies below 0.
   Units are also split by the regional weighted threshold λ_max(A_rr D_q,r) (X3's operator, q = γ·P(b) on v(0))
   to look for early super-modularity near threshold.
2. CELF reference (experiment-side, OUTSIDE THE GAME RULES: the workers are chosen node by node with full knowledge
   of the graph): 10 seeds × the largest region × k ≤ 10. Objective f(S) = mean expected share over R = 4 salted
   ``cascade`` streams with the workers S bought in round 1 (``pick_fn`` hook). Lazy greedy (CELF) with batched
   re-evaluation: the M = 64 stale candidates with the largest upper bounds are re-evaluated in one GPU batch;
   the top candidate is accepted when its gain is fresh. The candidate pool is every eligible voter of the region.
   Compared on the same objective with the random pick (R22) and the top-degree pick.
"""
from __future__ import annotations

import argparse
import dataclasses
import heapq
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x3_cascade import batched_lambda_max  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_parity import parity_rel  # noqa: E402
from x_policies import WorkerTiming  # noqa: E402
from x_streams import SaltedStreams  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.dynamics.campaign import eligible_workers  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import NUM_REGIONS, PERSONALITIES  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402
from dumbocracy.voting import similarity, vote_probability  # noqa: E402

KS = [0] + list(range(1, 11)) + [15, 20, 25, 30, 35, 40]
CELF_K, CELF_SEEDS, CELF_R, CELF_M = 10, 10, 4, 64
RAW_PATH = xc.RAW / "x18.jsonl"


def prepare(seeds, base: GameConfig, device) -> list[dict]:
    out = []
    for s in seeds:
        e = generate_electorate(dataclasses.replace(base, seed=s), device=device)
        sizes = [int(x) for x in e.sizes.tolist()]
        rank = sorted(range(4), key=lambda r: -sizes[r])            # rank[i] = region with size rank i
        out.append({"seed": s, "e": e, "b": oracle_manifesto(e, dataclasses.replace(base, seed=s)), "sizes": sizes,
                    "rank": rank, "personality": [PERSONALITIES[int(i)].id for i in e.personality_of_region]})
    return out


def regional_lambda(worlds: list[dict], cfg: GameConfig) -> None:
    """λ_max(A_rr D_q,r) per world and region for the ORACLE b (X3's weighted operator, batched power iteration)."""
    cols = []
    for w in worlds:
        e = w["e"]
        sim = similarity(e.preferences, w["b"])
        q = (e.susceptibility.campaign * vote_probability(sim, cfg.voting.sensitivity, cfg.voting.threshold)).clamp(0, 1)
        q = torch.where(e.susceptibility.zealot.to(torch.bool), torch.zeros_like(q), q).to(torch.float64)
        cols.append(torch.stack([q * (e.region_of == r) for r in range(4)], 1))
    lam, _ = batched_lambda_max([w["e"].graph for w in worlds], torch.stack(cols))
    for w, row in zip(worlds, lam.cpu().tolist()):
        w["lambda_region"] = row


# ============================================================================== value curve
def play_curve(units: list[tuple[dict, int]], cfg: GameConfig, k: int) -> list[dict]:
    """``units`` = [(world, size rank)]; k workers in round 1 in that region (library pick)."""
    es = [w["e"] for w, _ in units]
    pols = [WorkerTiming(k, "front", f"region:{w['rank'][i]}") for w, i in units]
    outs = run_batch(es, cfg, torch.stack([w["b"] for w, _ in units]), pols, seeds=[w["seed"] for w, _ in units])
    rows = []
    for (w, i), o in zip(units, outs):
        r = w["rank"][i]
        rows.append({"seed": w["seed"], "size_rank": i, "region": r, "personality": w["personality"][r],
                     "lambda_region": w["lambda_region"][r], "k": k, "expected": o.result.expected_share,
                     "victory": bool(o.result.victory), "converts": int(sum(o.converted)),
                     "worker_ids": [int(x) for x in o.workers.worker_ids[r].tolist()]})
    return rows


def boot_ci(x: np.ndarray, reps: int = 4000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (reps, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarise_curve(rows: list[dict], seeds: list[int]) -> dict:
    E = {(r["seed"], r["size_rank"], r["k"]): r for r in rows}
    out: dict = {"curve": {}, "pairs": {}, "violations": {}, "nested_check": None}
    # nested worker sets
    nested = all(set(E[(s, i, KS[j])]["worker_ids"]) <= set(E[(s, i, KS[j + 1])]["worker_ids"])
                 for s in seeds for i in range(4) for j in range(len(KS) - 1))
    out["nested_check"] = nested
    groups = {"rank0": [0], "rank1": [1], "rank2": [2], "rank3": [3], "all": [0, 1, 2, 3]}
    for g, ranks in groups.items():
        units = [(s, i) for s in seeds for i in ranks]
        curve = {}
        gains = {}
        for j, k in enumerate(KS):
            vals = np.array([E[(s, i, k)]["expected"] - E[(s, i, 0)]["expected"] for s, i in units])
            curve[str(k)] = {"delta_mean": float(vals.mean()), "ci95": xc.mean_ci(vals.tolist())["ci95"],
                             "win": xc.rate([E[(s, i, k)]["victory"] for s, i in units])["rate"]}
            if j > 0:
                gains[k] = np.array([(E[(s, i, k)]["expected"] - E[(s, i, KS[j - 1])]["expected"]) / (k - KS[j - 1])
                                     for s, i in units])
                curve[str(k)]["marginal_gain_per_worker"] = float(gains[k].mean())
        pairs = []
        ks = KS[1:]
        for a, b in zip(ks[:-1], ks[1:]):
            d = gains[b] - gains[a]
            # bootstrap over seeds (clusters) for pooled groups: resample seeds, keep their units
            if len(ranks) > 1:
                per_seed = np.array([d[[u for u, (s, _) in enumerate(units) if s == s0]].mean() for s0 in seeds])
                ci = boot_ci(per_seed, seed=a)
            else:
                ci = boot_ci(d, seed=a)
            pairs.append({"k": a, "k_next": b, "gain_k": float(gains[a].mean()), "gain_next": float(gains[b].mean()),
                          "diff_mean": float(d.mean()), "ci95": ci, "violation": ci[0] > 0, "decreasing": ci[1] < 0,
                          "share_units_increasing": float((d > 0).mean())})
        out["curve"][g] = curve
        out["pairs"][g] = pairs
        out["violations"][g] = {"pairs": len(pairs), "violating": sum(p["violation"] for p in pairs),
                                "decreasing": sum(p["decreasing"] for p in pairs),
                                "share_violating": sum(p["violation"] for p in pairs) / len(pairs)}
    # near-threshold units: per-unit early super-modularity g(2) > g(1)? split by regional λ
    near = {}
    for label, lo, hi in (("lambda_region<1.5", 0, 1.5), ("1.5≤lambda_region<3", 1.5, 3), ("lambda_region≥3", 3, 1e9)):
        units = [(s, i) for s in seeds for i in range(4) if lo <= E[(s, i, 1)]["lambda_region"] < hi]
        if not units:
            near[label] = {"units": 0}
            continue
        g1 = np.array([E[(s, i, 1)]["expected"] - E[(s, i, 0)]["expected"] for s, i in units])
        g2 = np.array([E[(s, i, 2)]["expected"] - E[(s, i, 1)]["expected"] for s, i in units])
        g3 = np.array([E[(s, i, 3)]["expected"] - E[(s, i, 2)]["expected"] for s, i in units])
        near[label] = {"units": len(units), "g1": float(g1.mean()), "g2": float(g2.mean()), "g3": float(g3.mean()),
                       "g2_minus_g1_ci95": boot_ci(g2 - g1, seed=7) if len(units) > 1 else None,
                       "share_units_g2_gt_g1": float((g2 > g1).mean()),
                       "delta_k10": float(np.mean([E[(s, i, 10)]["expected"] - E[(s, i, 0)]["expected"] for s, i in units]))}
    out["near_threshold"] = near
    out["lambda_region_distribution"] = xc.mean_ci([r["lambda_region"] for r in rows if r["k"] == 0])
    out["lambda_region_min"] = min(r["lambda_region"] for r in rows if r["k"] == 0)
    return out


# ============================================================================== CELF
class SetPick:
    """``pick_fn`` returning a fixed worker set per batch position (all in one region)."""

    def __init__(self, sets: list[torch.Tensor], region: list[int]):
        self.sets, self.region = sets, region

    def __call__(self, region_of, eligible, hires_by_region, generator, *, world, electorate):
        out = []
        for r in range(NUM_REGIONS):
            if r == self.region[world] and hires_by_region[r]:
                assert int(hires_by_region[r]) == self.sets[world].numel()
                out.append(self.sets[world])
            else:
                out.append(torch.zeros(0, dtype=torch.int64, device=region_of.device))
        return tuple(out)


def evaluate_sets(w: dict, cfg: GameConfig, region: int, sets: list[torch.Tensor], batch: int) -> list[float]:
    """f(S) = mean over CELF_R salted cascade streams of the expected share, for each set (batched)."""
    jobs = [(j, r) for j in range(len(sets)) for r in range(CELF_R)]

    def fn(chunk):
        ss = [sets[j] for j, _ in chunk]
        pols = [WorkerTiming(int(s.numel()), "front", f"region:{region}") for s in ss]
        outs = run_batch([w["e"]] * len(chunk), cfg, w["b"].expand(len(chunk), 5), pols,
                         streams=[SaltedStreams(w["seed"], {"cascade": r}) for _, r in chunk],
                         pick_fn=SetPick(ss, [region] * len(chunk)))
        return [o.result.expected_share for o in outs]
    vals = xc.run_chunks(fn, jobs, batch)
    out = np.zeros(len(sets))
    for (j, _), v in zip(jobs, vals):
        out[j] += v / CELF_R
    return out.tolist()


def celf(w: dict, cfg: GameConfig, batch: int, log=print) -> dict:
    e = w["e"]
    region = w["rank"][0]
    dev = e.preferences.device
    elig = eligible_workers(e.graph.degree, e.susceptibility.zealot, torch.zeros_like(e.susceptibility.zealot),
                            cfg.workers.min_degree) & (e.region_of == region)
    cand = torch.nonzero(elig).flatten()
    t0 = time.perf_counter()
    empty = torch.zeros(0, dtype=torch.int64, device=dev)
    f0 = evaluate_sets(w, cfg, region, [empty], batch)[0]
    first = evaluate_sets(w, cfg, region, [c.view(1) for c in cand], batch)
    evals = len(cand) + 1
    heap = [(-(f - f0), int(c), 1) for f, c in zip(first, cand.tolist())]      # (−gain, id, step evaluated)
    heapq.heapify(heap)
    chosen: list[int] = []
    values = [f0]
    for step in range(1, CELF_K + 1):
        while True:
            top = heap[0]
            if top[2] == step:                       # fresh: accept
                heapq.heappop(heap)
                chosen.append(top[1])
                values.append(values[-1] - top[0])
                break
            stale = [heapq.heappop(heap) for _ in range(min(CELF_M, len(heap)))]
            stale = [s for s in stale if s[2] != step] + [s for s in stale if s[2] == step]
            todo = [s for s in stale if s[2] != step]
            sets = [torch.tensor(chosen + [s[1]], dtype=torch.int64, device=dev) for s in todo]
            fv = evaluate_sets(w, cfg, region, sets, batch)
            evals += len(sets)
            for s, f in zip(todo, fv):
                heapq.heappush(heap, (-(f - values[-1]), s[1], step))
            for s in stale:
                if s[2] == step and s not in todo:
                    heapq.heappush(heap, s)
        if step == 1:
            pass
    # comparisons on the same objective: random pick (library, nested) and top-degree pick, k = 1..10
    gen_sets_random, gen_sets_top = [], []
    deg = e.graph.degree[cand].to(torch.float64) - cand.to(torch.float64) * 1e-9
    top_order = cand[torch.argsort(deg, descending=True)]
    from dumbocracy.config import SeedStreams
    g = SeedStreams(w["seed"]).generator("workers", 0, "p0", device=dev)
    perm = cand[torch.randperm(cand.numel(), generator=g, device=dev)]          # == R22's round-1 pick (nested)
    for k in range(1, CELF_K + 1):
        gen_sets_random.append(perm[:k])
        gen_sets_top.append(top_order[:k])
    f_rand = evaluate_sets(w, cfg, region, gen_sets_random, batch)
    f_top = evaluate_sets(w, cfg, region, gen_sets_top, batch)
    deg_all = e.graph.degree
    rec = {"seed": w["seed"], "region": region, "candidates": int(cand.numel()), "f0": f0,
           "celf_values": values, "celf_set": chosen, "celf_degrees": [int(deg_all[i]) for i in chosen],
           "random_values": [f0] + f_rand, "top_degree_values": [f0] + f_top,
           "random_degrees": [int(deg_all[i]) for i in perm[:CELF_K].tolist()],
           "evaluations": evals, "seconds": time.perf_counter() - t0,
           "celf_marginal_gains": [values[i + 1] - values[i] for i in range(CELF_K)]}
    log(f"[x18] CELF seed {w['seed']}: {evals} evaluations × {CELF_R} streams, {rec['seconds']:.0f}s; f(S_10): CELF "
        f"{values[-1]:.4f}, random {f_rand[-1]:.4f}, top-degree {f_top[-1]:.4f}, none {f0:.4f}")
    return rec


def summarise_celf(recs: list[dict]) -> dict:
    out = {}
    for k in (1, 2, 3, 5, 10):
        c = [r["celf_values"][k] - r["f0"] for r in recs]
        ra = [r["random_values"][k] - r["f0"] for r in recs]
        tp = [r["top_degree_values"][k] - r["f0"] for r in recs]
        out[f"k{k}"] = {"celf_delta": xc.mean_ci(c), "random_delta": xc.mean_ci(ra), "top_degree_delta": xc.mean_ci(tp),
                        "random_over_celf": float(np.mean(ra) / np.mean(c)) if np.mean(c) else None,
                        "top_degree_over_celf": float(np.mean(tp) / np.mean(c)) if np.mean(c) else None,
                        "celf_minus_random": xc.paired(c, ra)}
    mg = np.array([r["celf_marginal_gains"] for r in recs])
    out["celf_marginal_gains_mean"] = mg.mean(0).tolist()
    out["celf_marginal_gain_increases"] = int((np.diff(mg, axis=1) > 0).sum())
    out["celf_marginal_gain_pairs"] = int(mg.shape[0] * (mg.shape[1] - 1))
    out["celf_worker_degree_mean"] = float(np.mean([np.mean(r["celf_degrees"]) for r in recs]))
    out["random_worker_degree_mean"] = float(np.mean([np.mean(r["random_degrees"]) for r in recs]))
    out["evaluations_mean"] = float(np.mean([r["evaluations"] for r in recs]))
    return out


def parity(seeds: list[int], device: torch.device, base: GameConfig) -> dict:
    def fn(seed, dev):
        cfg = dataclasses.replace(base, device=str(dev))
        w = prepare([seed], cfg, dev)[0]
        w["lambda_region"] = [0.0] * 4
        r = play_curve([(w, 0)], cfg, 10)[0]
        return {"expected": r["expected"], "converts_share_of_n": r["converts"] / cfg.electorate.num_voters}
    out = parity_rel(fn, seeds, device=device)
    out["cell"] = "k = 10 workers in the largest region, round 1"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X18"])
    parser.add_argument("--celf-seeds", type=int, default=CELF_SEEDS)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--out", default=str(xc.REPORTS / "x18.json"))
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
    base = GameConfig(device=str(device), budget=Decimal("1000.00"), electorate=ElectorateConfig(num_voters=args.n))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text("")
    rec: dict = {}
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        worlds = prepare(seeds, base, device)
        regional_lambda(worlds, base)
        rec["setup_seconds"] = time.perf_counter() - start
        units = [(w, i) for w in worlds for i in range(4)]
        rows = []
        t0 = time.perf_counter()
        for k in KS:
            out = play_curve(units, base, k)
            rows += out
            for r in out:
                xc.write_row(RAW_PATH, {"exp": "X18", "kind": "curve", **r})
        rec["curve_seconds"] = time.perf_counter() - t0
        curve = summarise_curve(rows, seeds)
        print(f"[x18] curve done in {rec['curve_seconds']:.0f}s; nested {curve['nested_check']}; violations "
              f"{curve['violations']['all']}", flush=True)
        cal = xc.calibrate_batch(lambda B: evaluate_sets(worlds[0], base, worlds[0]["rank"][0],
                                                         [torch.arange(10, device=device)] * max(1, B // CELF_R), B),
                                 device, sizes=(16, 32), max_batch=1024)
        rec["celf_calibration"] = cal
        celf_recs = []
        t0 = time.perf_counter()
        for w in worlds[:args.celf_seeds]:
            r = celf(w, base, cal["batch"])
            celf_recs.append(r)
            xc.write_row(RAW_PATH, {"exp": "X18", "kind": "celf", **r})
        rec["celf_seconds"] = time.perf_counter() - t0
        celf_summary = summarise_celf(celf_recs)
        par = parity(seeds[:args.parity_seeds], device, base)
        print(f"[x18] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    report = {"experiment": "X18", "question": "Submodularity of the worker value curve; CELF reference",
              "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True,
                        "celf": seeds[:args.celf_seeds]},
              "n": args.n, "rounds": base.rounds, "treasury": str(base.budget), "ks": KS,
              "celf": {"k_max": CELF_K, "streams": CELF_R, "batch_reeval": CELF_M, "region": "largest"},
              "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
              "metric_definitions": {
                  "delta": "E(k workers) − E(0), same world and streams",
                  "marginal_gain_per_worker": "[E(k_i) − E(k_{i−1})] / (k_i − k_{i−1})",
                  "violation": "adjacent pair with 95% bootstrap CI of mean(g_{i+1} − g_i) above 0 (increasing returns "
                               "beyond noise); 'decreasing' = CI below 0; pooled groups resample seeds",
                  "lambda_region": "λ_max(A_rr D_q,r), q = γ·P(b) on v(0) (X3 operator)",
                  "celf_objective": f"mean expected share over {CELF_R} salted cascade streams"},
              **curve, "celf_summary": celf_summary, "celf_runs": celf_recs,
              "deviations": ["Treasury ₫1000 instead of ₫500 so that k = 40 workers (₫800) are affordable; nothing else "
                             "is bought, so the treasury has no other effect.",
                             "All 4 regions per seed (indexed by size rank) on the curve; CELF on the largest region.",
                             "CELF and the top-degree pick are outside the game rules (labelled)."],
              "config": base.to_record(),
              "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=[len(units), cal["batch"]],
                                             free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                             extra=rec)}
    xc.save(args.out, report)
    print(f"[x18] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
