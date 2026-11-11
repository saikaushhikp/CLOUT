"""X23 — Worker-rule ablations (plan ``docs/v2/plans/implementation_plan.md`` §3 X23).

Command (reproduces reports/v3/x23.json, reports/v3/raw/x23.jsonl and reports/v3/logs/x23.log):
  OMP_NUM_THREADS=4 python experiments/x23_worker_rules.py --seeds 50 --first-seed 10000 --device cuda:0 \
      --out reports/v3/x23.json 2>&1 | tee reports/v3/logs/x23.log

One rule changed at a time from the game default (cap 3, min degree 3, γ·P), N = 10⁴, T = 5, ₫500, 50 paired seeds
(block 10000, §1.5):
  workers.attempt_cap ∈ {1, 2, 3, 5, 1000 (≈ ∞)};  workers.min_degree ∈ {1, 3, 10};
  workers.conversion_rule ∈ {gamma, gamma_p (default), gamma_p2}   (approved library knob, D2).
The three default values are one shared arm ("default").

Per arm, everything runs batched on the GPU (``x_batch.run_batch``, B worlds per call):
* the X0 policies with the ORACLE manifesto: none, pay_workers, front_workers, pay_media, pay_split, planner (ORACLE);
* the extra (manifesto, policy) runs that ``balance.py`` needs for its five gate criteria: uniform/none,
  uniform/planner, top3/planner, informed/planner and informed/{pay_workers, front_workers, pay_media, pay_split}.
Gate criteria are computed exactly as ``balance.run`` does (same thresholds, same Wilcoxon "greater" test).

Cascade metrics per world, from an ``observer`` on the union tensors (no library change):
* reach per worker = distinct voters ever converted / workers bought (None when no worker was bought);
* branching ratio = Σ_t converts_t / Σ_t spreaders_t (each spreader spreads for one hop; < 1 = subcritical);
* cap binding = share of targeted voters (attempts > 0) whose final attempts_j == cap, from the final
  ``WorkerState.attempts``; plus the share of worlds where any voter hit the cap.
Paired tests against the default arm, Holm-corrected within each family (stated in the JSON).
Note: the ORACLE planner's worker-value model assumes γ·P and no cap; it is not re-tuned per arm (rule-unaware).
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
from x_batch import run_batch  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.oracle import informed_manifesto, oracle_manifesto, top3_manifesto  # noqa: E402

ARMS: dict[str, list[str]] = {
    "default": [],
    "cap1": ["workers.attempt_cap=1"], "cap2": ["workers.attempt_cap=2"], "cap5": ["workers.attempt_cap=5"],
    "cap1000": ["workers.attempt_cap=1000"],
    "min_degree1": ["workers.min_degree=1"], "min_degree10": ["workers.min_degree=10"],
    "rule_gamma": ["workers.conversion_rule=gamma"], "rule_gamma_p2": ["workers.conversion_rule=gamma_p2"],
}
FAMILIES = {"attempt_cap": ["cap1", "cap2", "default", "cap5", "cap1000"],
            "min_degree": ["min_degree1", "default", "min_degree10"],
            "conversion_rule": ["rule_gamma", "default", "rule_gamma_p2"]}
ORACLE_POLICIES = ("none", "pay_workers", "front_workers", "pay_media", "pay_split", "planner")
WORKER_POLICIES = ("pay_workers", "front_workers", "pay_split", "planner")
PAID = x_policies.balance.PAID_PATTERNS
GATE_RUNS = [("uniform", "none"), ("uniform", "planner"), ("top3", "planner"), ("informed", "planner")] + \
            [("informed", p) for p in PAID]
RUNS = [("oracle", p) for p in ORACLE_POLICIES] + GATE_RUNS
RAW_PATH = xc.RAW / "x23.jsonl"


def play(es, cfg: GameConfig, ms: torch.Tensor, policy, seeds: list[int]) -> list[dict]:
    """One batched call; returns per-world outcome summaries with the cascade metrics."""
    B, n = len(es), es[0].num_voters
    state: dict = {}

    def observer(t: int, info: dict) -> None:
        conv = info["converted"]
        state["ever"] = conv.clone() if "ever" not in state else state["ever"] | conv
        if t == cfg.rounds - 1:
            att = info["attempts"].view(B, n)
            cap = cfg.workers.attempt_cap
            targeted = att > 0
            state["final"] = torch.stack([info["pinned"].view(B, n).sum(1), state["ever"].view(B, n).sum(1),
                                          targeted.sum(1), (targeted & (att >= cap)).sum(1), att.sum(1)], 1).cpu()

    outs = run_batch(es, cfg, ms, policy, seeds=seeds, observer=observer)
    rows = []
    for b, o in enumerate(outs):
        workers, distinct, targeted, at_cap, attempts = (int(x) for x in state["final"][b].tolist())
        assert workers == int(o.workers.pinned.sum()), "observer and final WorkerState disagree"
        conv, spr = o.converted, o.spreaders
        rows.append({"expected": o.result.expected_share, "vote_share": o.result.vote_share,
                     "victory": bool(o.result.victory), "spent": float(o.spent), "workers": workers,
                     "distinct_converts": distinct, "conversion_events": sum(conv),
                     "reach_per_worker": distinct / workers if workers else None,
                     "branching_ratio": sum(conv) / sum(spr) if sum(spr) else None,
                     "converted_by_round": conv, "spreaders_by_round": spr,
                     "targeted": targeted, "at_cap": at_cap, "attempts": attempts,
                     "cap_bind_share": at_cap / targeted if targeted else None,
                     "trajectory": [round(x, 6) for x in o.share_trajectory]})
    return rows


def gate(runs: dict[str, list[dict]]) -> dict:
    """The five ``balance.run`` gate criteria (same thresholds and test), from per-seed rows of one arm."""
    from scipy.stats import wilcoxon
    n = len(runs["uniform/none"])
    rate = {k: sum(r["victory"] for r in v) / n for k, v in runs.items()}
    diffs = [a["expected"] - b["expected"] for a, b in zip(runs["informed/planner"], runs["top3/planner"])]
    p_value = float(wilcoxon(diffs, alternative="greater").pvalue) if any(d != 0 for d in diffs) else 1.0
    best = {p: 0 for p in PAID}
    for i in range(n):
        best[max(PAID, key=lambda p: runs[f"informed/{p}"][i]["expected"])] += 1
    max_rate = max(rate[f"informed/{p}"] for p in PAID)
    return {"criteria": {
        "1_uniform_no_campaign_le_30pct": rate["uniform/none"] <= 0.30,
        "2_oracle_planner_ge_85pct": rate["oracle/planner"] >= 0.85,
        "3_informed_beats_top3_p_lt_0.01": p_value < 0.01 and sum(diffs) > 0,
        "4a_no_paid_pattern_ge_80pct": max_rate < 0.80,
        "4b_workers_and_media_each_best_sometimes": (best["pay_workers"] + best["front_workers"]) > 0
                                                    and best["pay_media"] > 0},
        "values": {"uniform_none_win": rate["uniform/none"], "oracle_planner_win": rate["oracle/planner"],
                   "informed_minus_top3_mean": sum(diffs) / n, "informed_vs_top3_wilcoxon_greater_p": p_value,
                   "max_paid_pattern_win": max_rate, "best_paid_pattern_counts": best}}


def summarise(results: dict[str, dict[str, list[dict]]], seeds: list[int]) -> dict:
    def opt(xs):
        xs = [x for x in xs if x is not None]
        return xc.mean_ci(xs) if xs else None

    arms = {}
    for arm, runs in results.items():
        per_run = {}
        for key, rs in runs.items():
            tgt = sum(r["targeted"] for r in rs)
            per_run[key] = {
                "win": xc.rate([r["victory"] for r in rs]), "expected": xc.mean_ci([r["expected"] for r in rs]),
                "workers": xc.mean_ci([r["workers"] for r in rs]),
                "distinct_converts": xc.mean_ci([r["distinct_converts"] for r in rs]),
                "reach_per_worker": opt([r["reach_per_worker"] for r in rs]),
                "reach_per_worker_pooled": (sum(r["distinct_converts"] for r in rs) / sum(r["workers"] for r in rs)
                                            if sum(r["workers"] for r in rs) else None),
                "branching_ratio": opt([r["branching_ratio"] for r in rs]),
                "branching_ratio_pooled": (sum(r["conversion_events"] for r in rs)
                                           / max(1, sum(sum(r["spreaders_by_round"]) for r in rs))),
                "cap_bind_share_pooled": sum(r["at_cap"] for r in rs) / tgt if tgt else None,
                "cap_bind_share": opt([r["cap_bind_share"] for r in rs]),
                "worlds_with_cap_hit": xc.rate([r["at_cap"] > 0 for r in rs]),
                "converted_by_round_mean": [sum(r["converted_by_round"][t] for r in rs) / len(rs)
                                            for t in range(len(rs[0]["converted_by_round"]))]}
        g = gate(runs)
        manif = {"oracle_minus_uniform_planner": xc.paired([r["expected"] for r in runs["oracle/planner"]],
                                                          [r["expected"] for r in runs["uniform/planner"]]),
                 "informed_minus_top3_planner": xc.paired([r["expected"] for r in runs["informed/planner"]],
                                                         [r["expected"] for r in runs["top3/planner"]])}
        arms[arm] = {"overrides": ARMS[arm], "runs": per_run, "gate": g, "manifesto_effect": manif}

    # paired tests vs default, Holm within each family
    tests: dict = {"expected_share": [], "victory": [], "reach_per_worker": [], "manifesto_effect": []}
    base = results["default"]
    for arm in ARMS:
        if arm == "default":
            continue
        for pol in ORACLE_POLICIES:
            key = f"oracle/{pol}"
            a, b = results[arm][key], base[key]
            tests["expected_share"].append({"arm": arm, "run": key, **xc.paired([r["expected"] for r in a],
                                                                                 [r["expected"] for r in b])})
            tests["victory"].append({"arm": arm, "run": key, **xc.mcnemar([r["victory"] for r in a],
                                                                           [r["victory"] for r in b])})
            if pol in WORKER_POLICIES:
                pairs = [(x["reach_per_worker"], y["reach_per_worker"]) for x, y in zip(a, b)
                         if x["reach_per_worker"] is not None and y["reach_per_worker"] is not None]
                tests["reach_per_worker"].append({"arm": arm, "run": key, "pairs": len(pairs),
                                                  **xc.paired([p[0] for p in pairs], [p[1] for p in pairs])})
        # does the manifesto matter less? (oracle − uniform, planner) in arm minus the same in default
        eff = lambda rs: [x["expected"] - y["expected"] for x, y in zip(rs["oracle/planner"], rs["uniform/planner"])]  # noqa: E731
        tests["manifesto_effect"].append({"arm": arm, "measure": "(oracle − uniform manifesto, ORACLE planner): arm − default",
                                          **xc.paired(eff(results[arm]), eff(base))})
    for fam, items in tests.items():
        key = "p" if fam == "victory" else "wilcoxon_p"
        for x, p in zip(items, xc.holm([x[key] for x in items])):
            x["holm_p"] = p
    return {"arms": arms, "tests": tests,
            "test_families": {
                "expected_share": "8 non-default arms × 6 ORACLE-manifesto policies, arm − default, Wilcoxon, Holm (48)",
                "victory": "same 48 pairs, exact McNemar, Holm",
                "reach_per_worker": "8 arms × 4 worker-buying policies, arm − default over seeds with workers in both, "
                                    "Wilcoxon, Holm (32)",
                "manifesto_effect": "8 arms: (oracle − uniform manifesto under the ORACLE planner) arm − default, "
                                    "Wilcoxon, Holm (8)"}}


def parity(seeds: list[int], device: torch.device, n: int) -> dict:
    """CPU↔GPU, 3 seeds, arm rule_gamma (exercises the conversion-rule knob), ORACLE manifesto + planner."""
    def fn(seed, dev):
        cfg = xc.apply_overrides(GameConfig(seed=seed, device=str(dev), electorate=ElectorateConfig(num_voters=n)),
                                 ARMS["rule_gamma"])
        e = generate_electorate(cfg, device=dev)
        r = play([e], cfg, oracle_manifesto(e, cfg)[None], x_policies.policy_planner, [seed])[0]
        return {"expected": r["expected"], "victory": r["victory"], "converts_share_of_n": r["distinct_converts"] / n,
                "cap_bind_share": r["cap_bind_share"] or 0.0}

    out = xc.parity_check(fn, seeds, device=device, abs_tol=0.01)
    out["arm"] = "rule_gamma"
    out["metrics"]["converts_share_of_n"]["note"] = ("distinct converts / N (reach on the electorate scale, so one "
                                                     "absolute tolerance of 0.01 applies to every metric)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X23"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--out", default=str(xc.REPORTS / "x23.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    free0 = xc.free_bytes(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text("")                      # this script's own output, regenerated per run
    record: dict = {}
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        es = [generate_electorate(dataclasses.replace(base, seed=s), device=device) for s in seeds]
        ms = {"uniform": torch.full((len(seeds), 5), 0.2, dtype=es[0].preferences.dtype, device=device),
              "top3": torch.stack([top3_manifesto(e.preferences) for e in es]),
              "informed": torch.stack([informed_manifesto(e, dataclasses.replace(base, seed=s)) for e, s in zip(es, seeds)]),
              "oracle": torch.stack([oracle_manifesto(e, dataclasses.replace(base, seed=s)) for e, s in zip(es, seeds)])}
        record["setup_seconds"] = time.perf_counter() - start
        nz = [~e.susceptibility.zealot.to(torch.bool) for e in es]
        record["degree_check"] = {
            "min_degree_over_worlds": int(min(int(e.graph.degree.min()) for e in es)),
            "share_non_zealots_degree_lt_3": sum(int(((e.graph.degree < 3) & m).sum()) for e, m in zip(es, nz))
                                             / sum(int(m.sum()) for m in nz),
            "share_non_zealots_degree_lt_10": sum(int(((e.graph.degree < 10) & m).sum()) for e, m in zip(es, nz))
                                              / sum(int(m.sum()) for m in nz)}
        print(f"[x23] degree check {record['degree_check']}", flush=True)
        cal = xc.calibrate_batch(lambda B: play(es[:B], base, ms["oracle"][:B], x_policies.policy_planner, seeds[:B]),
                                 device, sizes=(2, 4), max_batch=len(seeds))
        record["calibration"] = cal
        idx = list(range(len(seeds)))
        results: dict[str, dict[str, list[dict]]] = {}
        for arm, items in ARMS.items():
            cfg = xc.apply_overrides(base, items)
            results[arm] = {}
            t0 = time.perf_counter()
            for mname, pname in RUNS:
                policy = x_policies.GATE_POLICIES[pname]
                rows = xc.run_chunks(lambda ch: play([es[i] for i in ch], cfg, ms[mname][ch], policy,
                                                     [seeds[i] for i in ch]), idx, cal["batch"], record=record)
                key = f"{mname}/{pname}"
                results[arm][key] = rows
                for s, r in zip(seeds, rows):
                    xc.write_row(RAW_PATH, {"exp": "X23", "seed": s, "arm": arm, "overrides": items, "manifesto": mname,
                                            "policy": pname, **r})
            op = results[arm]["oracle/planner"]
            print(f"[x23] {arm}: {time.perf_counter() - t0:.0f}s; ORACLE planner win "
                  f"{sum(r['victory'] for r in op) / len(op):.2f}, reach/worker "
                  f"{sum(r['distinct_converts'] for r in op) / max(1, sum(r['workers'] for r in op)):.1f}", flush=True)
        par = parity(seeds[:args.parity_seeds], device, args.n)
        print(f"[x23] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    report = {"experiment": "X23", "question": "Worker-rule ablations: cap, min degree, conversion rule",
              "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
              "n": args.n, "rounds": base.rounds, "treasury": str(base.budget),
              "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
              "arms": ARMS, "families": FAMILIES, "runs": [f"{m}/{p}" for m, p in RUNS],
              "metric_definitions": {
                  "reach_per_worker": "distinct voters converted at least once / workers bought (per world)",
                  "branching_ratio": "Σ_t converts_t / Σ_t spreaders_t; a spreader spreads for one hop; < 1 subcritical",
                  "cap_bind_share": "voters with final attempts_j == cap / voters with attempts_j > 0 (final WorkerState)",
                  "gate": "balance.run criteria 1, 2, 3, 4a, 4b (same thresholds and test)"},
              "notes": ["The ORACLE planner (balance.policy_planner) values workers with γ·P and no cap in every arm "
                        "(rule-unaware); its spend mix is therefore not re-optimised per rule.",
                        "none, pay_media and uniform/none buy no workers, so they are rule-invariant (sanity check)."],
              **summarise(results, seeds),
              "deviations": [f"Device {device} instead of the plan default cuda:2 (cuda:2 runs X0/X10 for another agent)."],
              "config": base.to_record(),
              "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=record.get("batch_sizes"),
                                             free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                             extra={"calibration": record.get("calibration"),
                                                    "degree_check": record.get("degree_check"),
                                                    "setup_seconds": record.get("setup_seconds")})}
    xc.save(args.out, report)
    print(f"[x23] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
