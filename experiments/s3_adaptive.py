"""S3 — Adaptive playbook (strategies.md Method 3): branch the playbook's spending on visible conditions.

Commands (write under reports/v3/experimentation/; run after s2_playbook.py --stage tune):
  OMP_NUM_THREADS=4 python experiments/s3_adaptive.py --stage tune --seeds 50 --first-seed 13000 --device cuda:0
  OMP_NUM_THREADS=4 python experiments/s3_adaptive.py --stage test --seeds 100 --first-seed 12000 --device cuda:0

Conditions (never more than 10⁴ voters; T outside 3–5 cannot be chosen in the web UI, ROUNDS_MAX = 5, so T = 8/10
are model results only):
  base N=10⁴ T=5 k̄=30 · sparse k̄ ∈ {10, 15, 20} · small N ∈ {1000, 2000, 5000} · short T=3 · long T ∈ {8, 10}
  · small+short N=2000 T=3 · fragmented (inter-region probability range [.01, .05], the X1 "low" cell).
Survey and manifesto are the S2-tuned ones (s2_tune.json → chosen) in every arm; only spending dials vary.

Tune (TRAINING seeds 13000+, disjoint from 12000–12099): every spending arm of a grid (workers, placement, media
regions in rounds 1/2, pace of later rounds) on every condition × seed. Branch dial sets = the grid arm with the
highest mean E over the training worlds of the conditions the branch targets; thresholds = the candidates that
maximise the mean over all conditions of the per-condition mean E when every world is played by the branch its
own visible numbers select (world-level k̄ from the census). The chosen AdaptivePlaybook parameters are written
to s3_tune.json → ``chosen``.
Test (seeds 12000–12099): non-adaptive playbook (S2 tuned) vs adaptive playbook vs top3 vs bayes (default
BayesPlanner) on the same worlds of every condition; paired differences with 95% CIs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_playbook as s2  # noqa: E402
import x_common as xc  # noqa: E402
import x_playbook as xp  # noqa: E402

OUT = s2.OUT
CONDITIONS = {
    "base": {"n": 10_000, "rounds": 5},
    "k10": {"n": 10_000, "rounds": 5, "mean_degree": 10},
    "k15": {"n": 10_000, "rounds": 5, "mean_degree": 15},
    "k20": {"n": 10_000, "rounds": 5, "mean_degree": 20},
    "n1000": {"n": 1_000, "rounds": 5},
    "n2000": {"n": 2_000, "rounds": 5},
    "n5000": {"n": 5_000, "rounds": 5},
    "t3": {"n": 10_000, "rounds": 3},
    "t8": {"n": 10_000, "rounds": 8},
    "t10": {"n": 10_000, "rounds": 10},
    "n2000_t3": {"n": 2_000, "rounds": 3},
    "frag": {"n": 10_000, "rounds": 5, "inter_prob_range": [0.01, 0.05]},
}
BRANCH_TARGETS = {"sparse": ("k10", "k15"), "small_or_long": ("n1000", "n2000", "t8", "t10"),
                  "short_large": ("t3",), "fragmented": ("frag",)}
SPEND = ("workers", "worker_place", "media1", "media2", "later")
THRESHOLDS = {"sparse_k": (12.5, 17.5, 22.5), "small_n": (1500, 3000, 7500), "long_t": (8, 6),
              "large_n": (5000, 1500)}


def spend_grid():
    # placement fixed to "size" (the S2 winner; placement mattered ≤ 0.0002 E there) to keep the grid small
    for W, place, m1, m2, later in itertools.product((0, 1, 2, 3, 4, 6, 8, 12, 16), ("size",), (2, 4),
                                                     (0, 2, 4), ("front", "even")):
        yield {"workers": W, "worker_place": place, "media1": m1, "media2": m2, "later": later}


def key_of(d: dict) -> str:
    return ",".join(f"{k}={d[k]}" for k in SPEND)


def run_conditions(names, seeds, arms, device, workers, label):
    rows, feats = [], {}
    for c in names:
        r, f, _ = s2.run_arms(seeds, {"name": c, **CONDITIONS[c]}, arms, device, workers, f"{label} {c}")
        rows += r
        feats[c] = f
    return rows, feats


def branch_of(features: dict, cond: dict, p: dict) -> str:
    """Same decision as ``AdaptivePlaybook.branch`` from the world's visible numbers (k̄, N, T)."""
    n, T = cond["n"], cond["rounds"]
    if n <= p["small_n"] or T >= p["long_t"]:
        return "small_or_long"
    if features["mean_friends"] < p["sparse_k"]:
        return "sparse"
    if T <= p["short_t"] and n >= p["large_n"]:
        return "short_large"
    if p.get("fragmented_share") is not None and features["inter_share"] < p["fragmented_share"]:
        return "fragmented"
    return "base"


def stage_tune(args, log, device) -> dict:
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    assert not set(seeds) & set(range(12000, 12100))
    chosen_s2 = json.loads((OUT / "s2_tune.json").read_text())["chosen"]
    fixed = {k: v for k, v in chosen_s2.items() if k not in SPEND}
    grid = list(spend_grid())
    arms = [(key_of(g), "playbook", {**fixed, **g}) for g in grid]
    names = list(CONDITIONS)
    log(f"[s3 tune] {len(arms)} spending arms × {len(names)} conditions × {len(seeds)} seeds; fixed {fixed}")
    rows, feats = run_conditions(names, seeds, arms, str(device), args.workers, "[s3 tune]")
    E = {}   # E[cond][arm][seed]
    for r in rows:
        E.setdefault(r["cond"], {}).setdefault(r["arm"], {})[r["seed"]] = r["expected"]
    mean = lambda c, a, ss=seeds: sum(E[c][a][s] for s in ss) / len(ss)  # noqa: E731
    base_key = key_of({k: chosen_s2[k] for k in SPEND})
    per_cond = {}
    for c in names:
        ranked = sorted(E[c], key=lambda a: -mean(c, a))
        best_by_w = {}
        for g in grid:
            k = key_of(g)
            w = g["workers"]
            if w not in best_by_w or mean(c, k) > mean(c, best_by_w[w]):
                best_by_w[w] = k
        per_cond[c] = {"best": ranked[0], "best_E": mean(c, ranked[0]), "s2_tuned_E": mean(c, base_key),
                       "top10": {a: mean(c, a) for a in ranked[:10]},
                       "best_by_workers": {str(w): {"arm": k, "E": mean(c, k)} for w, k in sorted(best_by_w.items())},
                       "mean_friends": xc.mean_ci([feats[c][s]["mean_friends"] for s in seeds]),
                       "inter_share": xc.mean_ci([feats[c][s]["inter_share"] for s in seeds])}
        log(f"[s3 tune] {c}: best {ranked[0]} E={mean(c, ranked[0]):.4f}; s2 tuned {mean(c, base_key):.4f}")
    # branch dial sets: best grid arm over the branch's target conditions (equal weight per condition)
    branch_sets = {}
    for b, targets in BRANCH_TARGETS.items():
        best = max(E["base"], key=lambda a: sum(mean(c, a) for c in targets))
        branch_sets[b] = best
        log(f"[s3 tune] branch {b} ← {best} (mean over {targets}: "
            f"{sum(mean(c, best) for c in targets) / len(targets):.4f})")
    dials_of = {key_of(g): g for g in grid}

    def score(p: dict, use_frag: bool) -> float:
        tot = 0.0
        for c in names:
            vals = []
            for s in seeds:
                b = branch_of(feats[c][s], CONDITIONS[c], {**p, "fragmented_share": p["fragmented_share"] if use_frag else None})
                a = base_key if b == "base" else branch_sets[b]
                vals.append(E[c][a][s])
            tot += sum(vals) / len(vals)
        return tot / len(names)

    cands = []
    for sk, sn, lt, ln in itertools.product(*THRESHOLDS.values()):
        cands.append({"sparse_k": sk, "small_n": sn, "long_t": lt, "short_t": 3, "large_n": ln,
                      "fragmented_share": None})
    scored = sorted(((score(p, False), p) for p in cands), key=lambda x: -x[0])
    best_p = dict(scored[0][1])
    frag_cut = sorted(feats["base"][s]["inter_share"] for s in seeds)[len(seeds) // 3]
    frag_p = {**best_p, "fragmented_share": frag_cut}
    frag_score = score(frag_p, True)
    non_adaptive = sum(mean(c, base_key) for c in names) / len(names)
    oracle_per_cond = sum(per_cond[c]["best_E"] for c in names) / len(names)
    branches = {b: {k: dials_of[branch_sets[b]][k] for k in SPEND} for b in ("sparse", "small_or_long", "short_large")}
    chosen = {**fixed, **{k: chosen_s2[k] for k in SPEND}, **{k: v for k, v in best_p.items()}, "branches": branches}
    log(f"[s3 tune] thresholds {best_p}: train mean E over conditions {scored[0][0]:.4f} "
        f"(non-adaptive {non_adaptive:.4f}, per-condition best {oracle_per_cond:.4f}, with fragmented branch "
        f"{frag_score:.4f})")
    raw = OUT / "raw" / f"s3_tune{args.tag}.jsonl"
    raw.write_text("")
    for r in rows:
        xc.write_row(raw, {"exp": "S3-tune", **{k: v for k, v in r.items() if k not in ("allocations", "manifesto")}})
    return {"conditions": CONDITIONS, "per_condition": per_cond, "branch_targets": BRANCH_TARGETS,
            "branch_dials": {b: dials_of[a] for b, a in branch_sets.items()},
            "threshold_candidates_top10": [{"score": s, **p} for s, p in scored[:10]],
            "train_mean_E_over_conditions": {"adaptive": scored[0][0], "non_adaptive_s2": non_adaptive,
                                             "per_condition_best_arm": oracle_per_cond,
                                             "adaptive_plus_fragmented_branch_nonhuman": frag_score},
            "fragmented_branch": {"inter_share_cut": frag_cut, "dials": dials_of[branch_sets["fragmented"]],
                                  "human_executable": False,
                                  "note": "inter_edges is in the observation/API but not shown in the web UI"},
            "chosen": chosen, "s2_chosen": chosen_s2,
            "seeds": {"first": seeds[0], "count": len(seeds), "role": "training (disjoint from 12000–12099)"},
            "num_spend_arms": len(arms), "games": len(rows)}


def stage_test(args, log, device) -> dict:
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    tune = json.loads((OUT / "s3_tune.json").read_text())
    chosen, chosen_s2 = tune["chosen"], tune["s2_chosen"]
    adaptive_kw = {k: v for k, v in chosen.items()}
    arms = [("playbook", "playbook", dict(chosen_s2)), ("adaptive", "adaptive", adaptive_kw),
            ("top3", "top3", {})]
    frag = tune["fragmented_branch"]
    if {k: frag["dials"][k] for k in SPEND} != {k: chosen_s2[k] for k in SPEND}:   # NOT human (inter_edges)
        arms.append(("adaptive_nonhuman_fragmented", "adaptive",
                     {**adaptive_kw, "fragmented_share": frag["inter_share_cut"],
                      "branches": {**chosen["branches"], "fragmented": {k: frag["dials"][k] for k in SPEND}}}))
    names = list(CONDITIONS) if not args.conditions else args.conditions.split(",")
    log(f"[s3 test] arms {[a for a, _, _ in arms]} (+ bayes: {not args.no_bayes}) × {names} × {len(seeds)} seeds")
    rows, feats = run_conditions(names, seeds, arms, str(device), args.workers, "[s3 test]")
    if not args.no_bayes:
        brows, _ = run_conditions(names, seeds, [("bayes", "bayes", {})], str(device), args.workers, "[s3 test bayes]")
        rows += brows
    by: dict = {}
    for r in rows:
        by.setdefault(r["cond"], {}).setdefault(r["arm"], {})[r["seed"]] = r
    arm_names = sorted({r["arm"] for r in rows}, key=lambda a: [x for x, _, _ in arms].index(a) if a in [x for x, _, _ in arms] else 99)
    out = {}
    for c in names:
        v = lambda a, k="expected": [by[c][a][s][k] for s in seeds]  # noqa: E731
        d = {"condition": CONDITIONS[c], "human_playable_in_ui": 3 <= CONDITIONS[c]["rounds"] <= 5,
             "mean_friends": xc.mean_ci([feats[c][s]["mean_friends"] for s in seeds]),
             "inter_share": xc.mean_ci([feats[c][s]["inter_share"] for s in seeds]),
             "adaptive_branch_counts": {}, "players": {}}
        for s in seeds:
            b = by[c]["adaptive"][s].get("record", {}).get("branch", "base")
            d["adaptive_branch_counts"][b] = d["adaptive_branch_counts"].get(b, 0) + 1
        for a in arm_names:
            p = {"expected": xc.mean_ci(v(a)), "win": xc.rate(v(a, "victory")),
                 "vs_playbook": xc.paired(v(a), v("playbook")), "vs_top3": xc.paired(v(a), v("top3")),
                 "spend": {k: xc.mean_ci([by[c][a][s]["spend"].get(k, 0.0) for s in seeds])
                           for k in ("workers", "media", "queries")},
                 "rejected_actions": sum(by[c][a][s]["rejected"] for s in seeds)}
            if "bayes" in by[c]:
                p["vs_bayes"] = xc.paired(v(a), v("bayes"))
            d["players"][a] = p
        out[c] = d
        log(f"[s3 test] {c}: " + ", ".join(f"{a} {d['players'][a]['expected']['mean']:.4f}" for a in arm_names)
            + f"; adaptive−playbook {d['players']['adaptive']['vs_playbook']['mean']:+.4f} "
              f"[{d['players']['adaptive']['vs_playbook']['ci95'][0]:+.4f}, {d['players']['adaptive']['vs_playbook']['ci95'][1]:+.4f}]")
    raw = OUT / "raw" / f"s3{args.tag}.jsonl"
    raw.write_text("")
    for r in rows:
        xc.write_row(raw, {"exp": "S3", **r})
    pooled = {"adaptive_minus_playbook_all_conditions": xc.paired(
        [by[c]["adaptive"][s]["expected"] for c in names for s in seeds],
        [by[c]["playbook"][s]["expected"] for c in names for s in seeds]),
        "note": "pooled over condition × seed pairs (not independent across conditions: same seeds)"}
    return {"conditions": out, "pooled": pooled, "chosen": chosen, "tune_file": "s3_tune.json",
            "arms": {a: {"kind": k, "dials": kw} for a, k, kw in arms},
            "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True,
                      "role": "test (X6 block seeds; worlds differ per condition)"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tune", "test"), required=True)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--conditions", default="")
    parser.add_argument("--no-bayes", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir)
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if args.first_seed is None:
        args.first_seed = 13000 if args.stage == "tune" else 12000
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=2 * 2**30)
    name = ("s3_tune" if args.stage == "tune" else "s3") + args.tag
    log_path = OUT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    log = s2.Log(log_path)
    free0 = xc.free_bytes(device)
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        report = stage_tune(args, log, device) if args.stage == "tune" else stage_test(args, log, device)
    seconds = time.perf_counter() - start
    report = {"experiment": name.upper(), "question": "Adaptive playbook (Method 3): spending branches on visible "
              "conditions (friends per voter, N, T) vs the non-adaptive playbook", **report,
              "provenance": xc.provenance_v3(args, device, seconds, workers=args.workers, free_at_start=free0,
                                             monitor=monitor, device_record=dev_record)}
    report["provenance"]["command"] = "OMP_NUM_THREADS=4 " + report["provenance"]["command"]
    xc.save(str(OUT / f"{name}.json"), report)
    log(f"[{name}] done in {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
