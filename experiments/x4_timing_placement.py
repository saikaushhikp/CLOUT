"""X4 — Worker timing and placement, plus the pick-rule ablation (plan ``docs/v2/plans/implementation_plan.md`` §4 X4).

Command (reproduces reports/v3/x4.json, reports/v3/raw/x4.jsonl and reports/v3/logs/x4.log):
  OMP_NUM_THREADS=4 python experiments/x4_timing_placement.py --seeds 50 --first-seed 11000 --device cuda:2 \
      --out reports/v3/x4.json 2>&1 | tee reports/v3/logs/x4.log

N = 10⁴, T = 5, ₫500, ₫20 workers, ORACLE manifesto, 50 paired seeds (block 11000). Workers only (no media).

* Timing × placement at EQUAL worker spend: a fixed total of K workers (K ∈ {4, 10}, i.e. ₫80 / ₫200) bought
  "front" (all in round 1), "even" (spread over the 5 rounds) or "late" (all in round 5), placed in the
  "largest" region, the "best_matched" region (ORACLE: highest true mean support for b at the time of buying) or
  "proportional" to region size (``x_policies.WorkerTiming``). Baseline: no campaign.
* Pick-rule ablation (EXPERIMENT-SIDE, OUTSIDE THE GAME RULES; labelled): the ``pick_fn`` hook of
  ``x_batch.run_batch`` replaces R22's pick for these arms only:
    - "random"      : a uniformly random eligible voter (degree ≥ 3, not a zealot or worker) — the rule in force;
    - "sample10"    : the highest-degree voter in a 10% random sample of the region's eligible voters (the source
                      document's rule; the sample is drawn from the same ``workers`` stream);
    - "top_degree"  : the true highest-degree eligible voters of the region (upper bound; ties by voter id).
  Played at K = 10 with timing front / even and placement proportional, and at K = 4 front / largest.
Paired tests (Wilcoxon on expected share, exact McNemar on wins) with Holm within each family.
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
from x_batch import run_batch  # noqa: E402
from x_parity import parity_rel  # noqa: E402
from x_policies import GATE_POLICIES, WorkerTiming  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import NUM_REGIONS  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402

KS = (4, 10)
TIMINGS = ("front", "even", "late")
PLACEMENTS = ("largest", "best_matched", "proportional")
PICK_RULES = ("random", "sample10", "top_degree")
PICK_ARMS = [(10, "front", "proportional"), (10, "even", "proportional"), (4, "front", "largest")]
RAW_PATH = xc.RAW / "x4.jsonl"


# ============================================================================== pick rules (outside the game rules)
def make_pick(rule: str):
    """``pick_fn(region_of, eligible, hires_by_region, generator, *, world, electorate)`` for ``run_batch``."""
    def pick(region_of, eligible, hires_by_region, generator, *, world, electorate):
        degree = electorate.graph.degree
        out = []
        for r in range(NUM_REGIONS):
            k = int(hires_by_region[r])
            if k == 0:
                out.append(torch.zeros(0, dtype=torch.int64, device=region_of.device))
                continue
            cand = torch.nonzero(eligible & (region_of == r)).flatten()
            if cand.numel() < k:
                raise ValueError(f"region {r}: only {cand.numel()} eligible voters for {k}")
            if rule == "random":            # == dynamics.campaign.pick_workers (same draw)
                perm = torch.randperm(cand.numel(), generator=generator, device=cand.device)[:k]
                out.append(cand[perm])
            elif rule == "sample10":        # top degree within a 10% sample (at least k voters)
                m = max(k, int(round(0.10 * cand.numel())))
                sample = cand[torch.randperm(cand.numel(), generator=generator, device=cand.device)[:m]]
                d = degree[sample].to(torch.float64) - sample.to(torch.float64) * 1e-9   # ties: lower id first
                out.append(sample[torch.topk(d, k).indices])
            elif rule == "top_degree":
                d = degree[cand].to(torch.float64) - cand.to(torch.float64) * 1e-9
                out.append(cand[torch.topk(d, k).indices])
            else:
                raise ValueError(rule)
        return tuple(out)
    return pick


def play(worlds: list[dict], cfg: GameConfig, policy, pick_rule: str | None = None) -> list[dict]:
    es = [w["e"] for w in worlds]
    B, n = len(es), es[0].num_voters
    st: dict = {"deg_new": torch.zeros(B, dtype=torch.float64), "n_new": torch.zeros(B, dtype=torch.float64)}

    def observer(t, info):
        game = info["game"]
        new = info["new"].view(B, n)
        st["deg_new"] += (game.degree.view(B, n).to(torch.float64) * new).sum(1).cpu()
        st["n_new"] += new.sum(1).cpu().to(torch.float64)
        if t == cfg.rounds - 1:
            st["ever"] = None

    pick_fn = None if pick_rule is None else make_pick(pick_rule)
    outs = run_batch(es, cfg, torch.stack([w["b"] for w in worlds]), policy, seeds=[w["seed"] for w in worlds],
                     pick_fn=pick_fn, observer=observer)
    rows = []
    for b, (w, o) in enumerate(zip(worlds, outs)):
        hires = [list(a.hires_by_region) for a in o.allocations]
        rows.append({"seed": w["seed"], "expected": o.result.expected_share, "victory": bool(o.result.victory),
                     "workers": int(sum(sum(h) for h in hires)), "hires_by_round": [sum(h) for h in hires],
                     "hires_by_region": [sum(h[r] for h in hires) for r in range(4)],
                     "converts": int(sum(o.converted)), "converted_by_round": o.converted, "spent": float(o.spent),
                     "worker_mean_degree": float(st["deg_new"][b] / st["n_new"][b]) if st["n_new"][b] > 0 else None,
                     "trajectory": [round(x, 6) for x in o.share_trajectory]})
    return rows


def summarise(res: dict[str, list[dict]], seeds: list[int]) -> dict:
    out: dict = {"arms": {}, "tests": {}}
    base = res["none"]
    for key, rows in res.items():
        out["arms"][key] = {"expected": xc.mean_ci([r["expected"] for r in rows]),
                            "win": xc.rate([r["victory"] for r in rows]),
                            "delta_vs_none": xc.mean_ci([r["expected"] - b["expected"] for r, b in zip(rows, base)]),
                            "workers": xc.mean_ci([r["workers"] for r in rows]),
                            "converts": xc.mean_ci([r["converts"] for r in rows]),
                            "worker_mean_degree": xc.mean_ci([r["worker_mean_degree"] for r in rows
                                                              if r["worker_mean_degree"] is not None])
                            if any(r["worker_mean_degree"] is not None for r in rows) else None,
                            "converted_by_round_mean": [sum(r["converted_by_round"][t] for r in rows) / len(rows)
                                                        for t in range(len(rows[0]["converted_by_round"]))]}

    def test(a, b, fam, **meta):
        ra, rb = res[a], res[b]
        t = {"a": a, "b": b, **meta, **xc.paired([x["expected"] for x in ra], [x["expected"] for x in rb]),
             "mcnemar": xc.mcnemar([x["victory"] for x in ra], [x["victory"] for x in rb])}
        out["tests"].setdefault(fam, []).append(t)
    for K in KS:
        for pl in PLACEMENTS:
            test(f"K{K}|front|{pl}", f"K{K}|even|{pl}", "timing", K=K, placement=pl)
            test(f"K{K}|front|{pl}", f"K{K}|late|{pl}", "timing", K=K, placement=pl)
            test(f"K{K}|even|{pl}", f"K{K}|late|{pl}", "timing", K=K, placement=pl)
        for tm in TIMINGS:
            test(f"K{K}|{tm}|best_matched", f"K{K}|{tm}|largest", "placement", K=K, timing=tm)
            test(f"K{K}|{tm}|proportional", f"K{K}|{tm}|largest", "placement", K=K, timing=tm)
            test(f"K{K}|{tm}|best_matched", f"K{K}|{tm}|proportional", "placement", K=K, timing=tm)
    for tm in TIMINGS:
        for pl in PLACEMENTS:
            test(f"K10|{tm}|{pl}", f"K4|{tm}|{pl}", "amount", timing=tm, placement=pl)
    for K, tm, pl in PICK_ARMS:
        for rule in ("sample10", "top_degree"):
            test(f"pick|{rule}|K{K}|{tm}|{pl}", f"pick|random|K{K}|{tm}|{pl}", "pick_rule", K=K, timing=tm, placement=pl)
    for fam, items in out["tests"].items():
        for t, p in zip(items, xc.holm([t["wilcoxon_p"] for t in items])):
            t["holm_p"] = p
        for t, p in zip(items, xc.holm([t["mcnemar"]["p"] for t in items])):
            t["mcnemar_holm_p"] = p
    # spread of the 9 timing × placement arms vs the effect of the amount K (does region choice matter more?)
    spread = {}
    for K in KS:
        vals = {f"{tm}|{pl}": out["arms"][f"K{K}|{tm}|{pl}"]["expected"]["mean"] for tm in TIMINGS for pl in PLACEMENTS}
        by_pl = {pl: max(out["arms"][f"K{K}|{tm}|{pl}"]["expected"]["mean"] for tm in TIMINGS) for pl in PLACEMENTS}
        spread[f"K{K}"] = {"best_arm": max(vals, key=vals.get), "worst_arm": min(vals, key=vals.get),
                           "range": max(vals.values()) - min(vals.values()),
                           "placement_range_at_best_timing": max(by_pl.values()) - min(by_pl.values())}
    spread["amount_effect_front"] = {pl: out["arms"][f"K10|front|{pl}"]["expected"]["mean"]
                                     - out["arms"][f"K4|front|{pl}"]["expected"]["mean"] for pl in PLACEMENTS}
    out["spread"] = spread
    out["test_families"] = {
        "timing": "front vs even, front vs late, even vs late × 3 placements × 2 K (18), Wilcoxon + McNemar, Holm",
        "placement": "best_matched vs largest, proportional vs largest, best_matched vs proportional × 3 timings × 2 K (18)",
        "amount": "K = 10 vs K = 4 for each of the 9 timing × placement arms (9)",
        "pick_rule": "sample10 vs random and top_degree vs random × 3 pick arms (6)"}
    return out


def parity(seeds: list[int], device: torch.device, base: GameConfig) -> dict:
    def fn(seed, dev):
        cfg = dataclasses.replace(base, device=str(dev))
        e = generate_electorate(dataclasses.replace(cfg, seed=seed), device=dev)
        w = {"seed": seed, "e": e, "b": oracle_manifesto(e, dataclasses.replace(cfg, seed=seed))}
        r = play([w], cfg, WorkerTiming(10, "even", "proportional"))[0]
        rt = play([w], cfg, WorkerTiming(10, "front", "proportional"), "top_degree")[0]
        return {"expected": r["expected"], "converts_share_of_n": r["converts"] / cfg.electorate.num_voters,
                "expected_top_degree": rt["expected"]}
    out = parity_rel(fn, seeds, device=device)
    out["cell"] = "K = 10 even / proportional (random pick) and K = 10 front / proportional (top_degree pick)"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X4"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--out", default=str(xc.REPORTS / "x4.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=3 * 2**30)
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
    start = time.perf_counter()
    res: dict[str, list[dict]] = {}
    with xc.GpuMonitor(device) as monitor:
        worlds = []
        for s in seeds:
            e = generate_electorate(dataclasses.replace(base, seed=s), device=device)
            worlds.append({"seed": s, "e": e, "b": oracle_manifesto(e, dataclasses.replace(base, seed=s))})
        setup = time.perf_counter() - start

        def run(key, policy, pick=None, **meta):
            t0 = time.perf_counter()
            rows = play(worlds, base, policy, pick)
            res[key] = rows
            for r in rows:
                xc.write_row(RAW_PATH, {"exp": "X4", "arm": key, **meta, **r})
            print(f"[x4] {key}: {time.perf_counter() - t0:.1f}s; E {sum(r['expected'] for r in rows) / len(rows):.4f}",
                  flush=True)
        run("none", GATE_POLICIES["none"])
        for K in KS:
            for tm in TIMINGS:
                for pl in PLACEMENTS:
                    run(f"K{K}|{tm}|{pl}", WorkerTiming(K, tm, pl), K=K, timing=tm, placement=pl)
        for K, tm, pl in PICK_ARMS:
            for rule in PICK_RULES:
                run(f"pick|{rule}|K{K}|{tm}|{pl}", WorkerTiming(K, tm, pl), rule, K=K, timing=tm, placement=pl,
                    pick_rule=rule, outside_game_rules=True)
        # sanity: the "random" pick_fn reproduces the library rule exactly
        same = all(a["expected"] == b["expected"] for a, b in zip(res["pick|random|K10|front|proportional"],
                                                                  res["K10|front|proportional"]))
        print(f"[x4] random pick_fn == library pick: {same}", flush=True)
        summary = summarise(res, seeds)
        par = parity(seeds[:args.parity_seeds], device, base)
        print(f"[x4] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    report = {"experiment": "X4", "question": "Worker timing and placement at equal spend; pick-rule ablation",
              "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
              "n": args.n, "rounds": base.rounds, "treasury": str(base.budget), "ks": list(KS),
              "timings": list(TIMINGS), "placements": list(PLACEMENTS), "pick_rules": list(PICK_RULES),
              "pick_arms": [list(x) for x in PICK_ARMS],
              "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
              "metric_definitions": {
                  "delta_vs_none": "expected share − that of no campaign (same world)",
                  "worker_mean_degree": "mean degree of the workers bought (per world)",
                  "best_matched": "ORACLE: region with the highest true mean P for b when buying",
                  "pick rules": "experiment-side, outside the game rules (R22 = random)"},
              "random_pick_fn_equals_library": same,
              **summary,
              "deviations": ["Two spend levels K ∈ {4, 10} workers (the plan fixes only 'equal total spend').",
                             "The pick-rule arms replace R22 through x_batch's pick_fn hook (experiment-side, outside "
                             "the game rules); the 'random' pick_fn reproduces the library pick bit for bit (checked)."],
              "config": base.to_record(),
              "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=[len(seeds)], free_at_start=free0,
                                             monitor=monitor, device_record=dev_record, parity=par,
                                             extra={"setup_seconds": setup})}
    xc.save(args.out, report)
    print(f"[x4] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
