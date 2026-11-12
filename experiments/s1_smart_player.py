"""S1 — the smart player (Method 4 of reports/v3/experimentation/strategies.md): BayesPlanner touch-ups, ablated.

Command (reproduces reports/v3/experimentation/s1.json, raw/s1.jsonl and logs/s1.log):
  OMP_NUM_THREADS=4 python experiments/s1_smart_player.py --seeds 100 --first-seed 12000 --device cuda:0
Held-out check (s1_holdout.json, raw/s1_holdout.jsonl, logs/s1_holdout.log; seeds never used for choosing options):
  OMP_NUM_THREADS=4 python experiments/s1_smart_player.py --seeds 100 --first-seed 14000 --device cuda:0 \
      --arms bayes,smart --tag holdout

Worlds: exactly the X6 strategy-tournament games (seed block 12000, N = 10⁴, T = 5, ₫500, default prices): per seed
``GameConfig(seed, device, N)`` → ``generate_electorate`` → ``GameEngine.new_game`` as in
``x6_tournament.seed_task``; every arm plays a full engine game on observation dicts only (``play_game``), all arms
on the same electorate (paired). Players run on the GPU like X6.

Arms (``dumbocracy.strategies.smart.SmartPlayer`` options; ``bayes`` is the library ``BayesPlanner`` itself):
  bayes         BayesPlanner defaults (must reproduce X6 ``bayes``: E = 0.639)
  cands         + the X13 shape as candidates (few workers in one good region + media in 2–4 regions, whole treasury)
  taper         rollout continuation "taper" (no more workers, front-loaded media) instead of "even"
  worlds        planning worlds ≥ 40 % of N voters (4,000 at N = 10⁴) instead of 1,600
  admm          consensus-ADMM manifesto (= X6 ``bayes_admm``) instead of projected gradient
  cands_taper   cands + taper
  cands_worlds  cands + worlds
  planner_all   cands + taper + worlds (PG manifesto)
  smart_no_taper cands + worlds + admm
  smart         all four

Reported: mean expected share E (``result.expected_share``) and win rate per arm; paired differences vs ``bayes``
(mean, t 95 % CI, Wilcoxon p with Holm over the arm family; exact McNemar on wins); the same split by world type —
"divided" if the TV distance between the archetype means of the two largest regions is ≥ 0.35 (the R1/X15 threshold,
``r1_manifesto_round0.tv_two_largest``), else "aligned"; the gap to the X13 full-information plan-search bound
(``reports/v3/x13.json`` E20_search_by_seed); spend shape; seconds per game; and the per-seed reproduction check of
``bayes`` against ``reports/v3/raw/x6.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from r1_manifesto_round0 import DIVIDED_TV, tv_two_largest  # noqa: E402
from x6_tournament import trace_of  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.manifesto_opt import expected_share  # noqa: E402
from dumbocracy.strategies.oracle import anticipate  # noqa: E402

OUT = xc.REPORTS / "experimentation"
JSON_PATH = OUT / "s1.json"
RAW_PATH = OUT / "raw" / "s1.jsonl"
LOG_PATH = OUT / "logs" / "s1.log"
X6_RAW = xc.RAW / "x6.jsonl"
X13_PATH = xc.REPORTS / "x13.json"

CANDS = {"front_candidates": True}
TAPER = {"rollout": "taper"}
WORLDS = {"plan_world_fraction": 0.4}
ADMM = {"manifesto_method": "admm"}
ARMS: dict[str, dict | None] = {
    "bayes": None,
    "cands": CANDS,
    "taper": TAPER,
    "worlds": WORLDS,
    "admm": ADMM,
    "cands_taper": {**CANDS, **TAPER},
    "cands_worlds": {**CANDS, **WORLDS},
    "planner_all": {**CANDS, **TAPER, **WORLDS},
    "smart_no_taper": {**CANDS, **WORLDS, **ADMM},
    "smart": {**CANDS, **TAPER, **WORLDS, **ADMM},
}

_LOG = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _LOG is not None:
        _LOG.write(msg + "\n")
        _LOG.flush()


def make_player(name: str, seed: int, device: str):
    from dumbocracy.strategies.players import BayesPlanner
    from dumbocracy.strategies.smart import SmartPlayer
    opts = ARMS[name]
    return BayesPlanner(seed=seed, device=device) if opts is None else SmartPlayer(seed=seed, device=device, **opts)


def seed_task(task: tuple) -> dict:
    """All arms on one electorate (spawned process). World construction = x6_tournament.seed_task."""
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.strategies.players import play_game
    from strategies import spend_breakdown
    seed, n, device, arms = task
    torch.set_num_threads(4)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.zeros(1, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    cfg = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=dev)
    v_t = anticipate(e, cfg)
    state0 = GameEngine.new_game(cfg, electorate=e, game_id=f"x6_{seed}")
    sizes = [int(x) for x in e.sizes.tolist()]
    sigma = [int(x) for x in e.archetype_of_region.tolist()]
    tv = tv_two_largest(sizes, sigma)
    rows = []
    for name in arms:
        player = make_player(name, seed, device)
        g0 = time.perf_counter()
        final, log_ = play_game(state0, player)
        game_seconds = time.perf_counter() - g0
        obs = observation_for(final, "p0")
        res = obs["result"]
        b = torch.tensor(obs["manifesto"], dtype=v_t.dtype, device=v_t.device)
        decisions = [x for xs in player.latency.values() for x in xs]
        rows.append({"exp": "S1", "seed": seed, "player": name, "options": ARMS[name] or "BayesPlanner defaults",
                     "expected": res["expected_share"], "vote_share": res["vote_share"],
                     "victory": bool(res["victory"]),
                     "static_share_anticipated": float(expected_share(v_t, b, cfg.voting.sensitivity,
                                                                      cfg.voting.threshold)),
                     "spend": spend_breakdown(obs, cfg), "rejected": sum(1 for *_, ok, _c in log_ if not ok),
                     "decision_seconds_mean": sum(decisions) / len(decisions), "game_seconds": game_seconds,
                     "tv_two_largest": tv, "scenario": "divided" if tv >= DIVIDED_TV else "aligned",
                     "planner_diagnostics": [{k: d[k] for k in ("round", "best_score", "num_candidates", "seconds")}
                                             for d in player.diagnostics],
                     "trace": trace_of(final, obs)})
    peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    return {"seed": seed, "rows": rows, "seconds": time.perf_counter() - start, "peak_bytes": peak,
            "sizes": sizes, "sigma": sigma}


# ============================================================================== statistics
def _shape(rows: list[dict]) -> dict:
    """Mean campaign shape: workers and airings per round, and the share of allocation money spent in rounds 1–2."""
    T = max(len(r["trace"]["allocations"]) for r in rows)
    workers = [sum(sum(r["trace"]["allocations"][t]["hires"]) for r in rows if t < len(r["trace"]["allocations"]))
               / len(rows) for t in range(T)]
    airings = [sum(sum(r["trace"]["allocations"][t]["media"]) for r in rows if t < len(r["trace"]["allocations"]))
               / len(rows) for t in range(T)]
    early = []
    for r in rows:
        costs = [float(a["cost"]) for a in r["trace"]["allocations"]]
        early.append(sum(costs[:2]) / sum(costs) if sum(costs) > 0 else 0.0)
    return {"workers_by_round": workers, "airings_by_round": airings, "early_spend_share_rounds_1_2": xc.mean_ci(early)}


def summarise(rows: list[dict], seeds: list[int], arms: list[str], x13: dict | None, x6_ref: dict) -> dict:
    by: dict[str, dict[int, dict]] = {}
    for r in rows:
        by.setdefault(r["player"], {})[r["seed"]] = r

    def vals(p, key="expected", subset=None):
        return [by[p][s][key] for s in (subset or seeds)]

    scen = {s: by[arms[0]][s]["scenario"] for s in seeds}
    groups = {g: [s for s in seeds if scen[s] == g] for g in ("aligned", "divided")}
    out_arms = {}
    for p in arms:
        rs = [by[p][s] for s in seeds]
        a = {"options": rs[0]["options"], "expected": xc.mean_ci(vals(p)), "win": xc.rate(vals(p, "victory")),
             "static_share_anticipated": xc.mean_ci(vals(p, "static_share_anticipated")),
             "spend": {k: xc.mean_ci([r["spend"].get(k, 0.0) for r in rs]) for k in ("workers", "media", "queries",
                                                                                      "total")},
             "mean_queries": sum(r["spend"].get("num_queries", 0) for r in rs) / len(rs),
             "shape": _shape(rs), "rejected_actions": sum(r["rejected"] for r in rs),
             "game_seconds": xc.mean_ci([r["game_seconds"] for r in rs]),
             "decision_seconds": xc.mean_ci([r["decision_seconds_mean"] for r in rs]),
             "candidates_per_decision": xc.mean_ci([d["num_candidates"] for r in rs for d in r["planner_diagnostics"]]),
             "by_scenario": {}}
        if p != "bayes":
            a["vs_bayes"] = {"expected": xc.paired(vals(p), vals("bayes")),
                             "mcnemar_wins": xc.mcnemar(vals(p, "victory"), vals("bayes", "victory"))}
        if x13 is not None and all(str(s) in x13["E20_search_by_seed"] for s in seeds):
            a["gap_to_x13_bound"] = xc.paired([x13["E20_search_by_seed"][str(s)] for s in seeds], vals(p))
        for g, ss in groups.items():
            if not ss:
                continue
            row = {"n": len(ss), "expected": xc.mean_ci(vals(p, subset=ss)), "win": xc.rate(vals(p, "victory", ss))}
            if p != "bayes" and len(ss) > 1:
                row["vs_bayes"] = xc.paired(vals(p, subset=ss), vals("bayes", subset=ss))
            a["by_scenario"][g] = row
        out_arms[p] = a
    fam = [p for p in arms if p != "bayes"]
    for p, h in zip(fam, xc.holm([out_arms[p]["vs_bayes"]["expected"]["wilcoxon_p"] for p in fam])):
        out_arms[p]["vs_bayes"]["expected"]["holm_p"] = h
    extra = {}
    if "smart" in arms and "planner_all" in arms:
        extra["smart_minus_planner_all"] = xc.paired(vals("smart"), vals("planner_all"))
    if "smart" in arms and "admm" in arms:
        extra["smart_minus_admm"] = xc.paired(vals("smart"), vals("admm"))
    if "cands_taper" in arms and "cands" in arms:
        extra["cands_taper_minus_cands"] = xc.paired(vals("cands_taper"), vals("cands"))
    if "planner_all" in arms and "cands_worlds" in arms:
        extra["planner_all_minus_cands_worlds"] = xc.paired(vals("planner_all"), vals("cands_worlds"))
    if "smart" in arms and "smart_no_taper" in arms:
        extra["smart_minus_smart_no_taper"] = xc.paired(vals("smart"), vals("smart_no_taper"))
    repro = {}
    if "bayes" in arms:
        ref = [x6_ref.get(s) for s in seeds]
        diffs = [abs(a - b) for a, b in zip(vals("bayes"), ref) if b is not None]
        repro = {"x6_bayes_expected_mean_same_seeds": (sum(b for b in ref if b is not None) / len(diffs)
                                                       if diffs else None),
                 "s1_bayes_expected_mean": xc.mean_ci(vals("bayes"))["mean"], "seeds_compared": len(diffs),
                 "max_abs_diff": max(diffs) if diffs else None,
                 "identical_within_1e-6": sum(1 for d in diffs if d <= 1e-6)}
    return {"arms": out_arms, "tests_note": "vs_bayes = arm − bayes, paired over seeds (t CI, Wilcoxon; Holm over the "
                                            f"{len(fam)} arms); McNemar exact on wins",
            "scenario_counts": {g: len(ss) for g, ss in groups.items()}, "comparisons": extra,
            "reproduction_vs_x6": repro, "ranking_by_mean_E": sorted(arms, key=lambda p: -out_arms[p]["expected"]["mean"])}


def main() -> None:
    global _LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=12000)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", default=None, help="JSON path (default s1.json, or s1_<tag>.json with --tag)")
    parser.add_argument("--tag", default=None, help="name suffix for the JSON/raw/log files, e.g. 'holdout'")
    args = parser.parse_args()
    global RAW_PATH, LOG_PATH
    if args.tag:
        RAW_PATH = OUT / "raw" / f"s1_{args.tag}.jsonl"
        LOG_PATH = OUT / "logs" / f"s1_{args.tag}.log"
    if args.out is None:
        args.out = str(OUT / f"s1_{args.tag}.json") if args.tag else str(JSON_PATH)
    if args.n > 10_000:
        raise SystemExit("S1 is only run up to N = 10,000 voters")
    torch.set_num_threads(4)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG = open(LOG_PATH, "w")
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    free0 = xc.free_bytes(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    arms = [a for a in args.arms.split(",") if a]
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        log(f"[s1] device {device}, seeds {seeds[0]}..{seeds[-1]}, N {args.n}, arms {arms}")
        results = xc.run_tasks(seed_task, [(s, args.n, str(device), tuple(arms)) for s in seeds], args.workers,
                               label="[s1]")
    seconds = time.perf_counter() - start
    rows = [r for res in results for r in res["rows"]]
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text("")
    for r in rows:
        xc.write_row(RAW_PATH, r)
    x13 = json.loads(X13_PATH.read_text()) if X13_PATH.exists() else None
    x6_ref = {r["seed"]: r["expected"] for r in xc.read_rows(X6_RAW) if r.get("player") == "bayes"} \
        if X6_RAW.exists() else {}
    summary = summarise(rows, seeds, arms, x13, x6_ref)
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    report = {
        "experiment": "S1", "question": "Smart player (BayesPlanner) touch-ups of strategies.md Method 4, ablated on "
                                        "the X6 tournament worlds",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget),
        "arm_options": {a: ARMS[a] or "BayesPlanner defaults" for a in arms},
        "definitions": {"expected": "result.expected_share of the engine game (true E at the election)",
                        "scenario": f"divided if TV(archetype means of the two largest regions) >= {DIVIDED_TV} "
                                    "(r1_manifesto_round0.tv_two_largest), else aligned",
                        "gap_to_x13_bound": "X13 full-information plan search (mean E over 20 fresh streams) − arm",
                        "plan_world_fraction": "planning worlds use max(1600, ceil(0.4 N)) voters (capped at N)"},
        **summary,
        "per_seed_seconds": xc.mean_ci([res["seconds"] for res in results]),
        "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, workers=args.workers,
                                       batch_sizes={"bayes_planner_num_worlds": 8, "bayes_world_voters": 1600,
                                                    "smart_plan_world_voters": 4000},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record,
                                       extra={"child_peak_bytes_max": max((r["peak_bytes"] or 0) for r in results),
                                              "env": {"OMP_NUM_THREADS": "4"}}),
    }
    xc.save(args.out, report)
    for p in arms:
        a = summary["arms"][p]
        d = a.get("vs_bayes", {}).get("expected")
        diff = f" Δ {d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d else ""
        log(f"[s1] {p:12s} E {a['expected']['mean']:.4f} win {a['win']['rate']:.2f}{diff} "
            f"game {a['game_seconds']['mean']:.1f}s")
    log(f"[s1] reproduction vs x6: {json.dumps(summary['reproduction_vs_x6'])}")
    log(f"[s1] done in {seconds / 60:.1f} min")
    _LOG.close()


if __name__ == "__main__":
    main()
