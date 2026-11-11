"""X6 — Strategy tournament (plan ``docs/v2/plans/implementation_plan.md`` §5 X6).

Command (reproduces reports/v3/x6.json, reports/v3/raw/x6.jsonl and reports/v3/logs/x6.log; run after X13):
  OMP_NUM_THREADS=4 python experiments/x6_tournament.py --seeds 100 --first-seed 12000 --device cuda:0 \
      --out reports/v3/x6.json 2>&1 | tee reports/v3/logs/x6.log

Paired games (same 100 electorates, seed block 12000, N = 10⁴, T = 5, ₫500, ₫20 workers, ₫10 queries). Every
player plays full ``GameEngine`` games on observation dicts only (players run on the GPU: engine state, posterior
worlds and rollouts on cuda; D12 fixed BayesPlanner on the GPU):

  uniform, top3, random           library baselines (``strategies.players``)
  I   people_pleaser              ``x_players.PeoplePleaser``: posterior + blend of the majority coalition, even pace
  II  bayes                       ``BayesPlanner`` (manifesto by projected gradient = PG; rollout planner)
      bayes_admm                  BayesPlanner with the consensus-ADMM manifesto (primal/dual residuals recorded)
      bayes_blend                 BayesPlanner with player I's blend manifesto
  III strong_arm                  ``x_players.StrongArm`` fixed b (0.6 on Education), split at 1.5× pace, no queries
      strong_arm_random           the same with a random (flat Dirichlet, keyed by seed) b

ORACLE (reference, not a player): ``oracle_manifesto`` + ``balance.policy_planner`` on the TRUE state, batched with
``x_batch.run_batch`` on the canonical streams of each seed (= ``simulate_world``). **The ORACLE is a heuristic**:
its manifesto ignores the campaign and its planner is greedy first-order. X13's open-loop plan search with the
same manifesto is the upper bound reported alongside (per-seed mean E over 20 fresh streams,
``reports/v3/x13.json``).

Manifesto optimisers (arms PG / ADMM / blend) are compared in two ways: the full game (bayes vs bayes_admm vs
bayes_blend) and the committed manifesto's static expected share on the anticipated TRUE electorate.

Per-round traces for X8 are saved in raw/x6.jsonl (``trace``): the true expected share E after each round
(``share_trajectory``, hidden, read from the final engine state after the game), the public polls, the
margin E − 0.5, the treasury, and every action (queries, commit, allocations with cost).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
import x_policies  # noqa: E402
import x_players  # noqa: E402
from x_batch import run_batch  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.manifesto_opt import expected_share  # noqa: E402
from dumbocracy.strategies.oracle import anticipate, oracle_manifesto  # noqa: E402

PLAYERS = ("uniform", "top3", "random", "people_pleaser", "bayes", "bayes_admm", "bayes_blend", "strong_arm",
           "strong_arm_random")
ROLE = {"people_pleaser": "I", "bayes": "II", "strong_arm": "III"}
RAW_PATH = xc.RAW / "x6.jsonl"
X13_PATH = xc.REPORTS / "x13.json"


def make_player(name: str, seed: int, device: str):
    from dumbocracy.strategies.players import BayesPlanner, RandomPlayer, Top3Player, UniformPlayer
    if name == "uniform":
        return UniformPlayer()
    if name == "top3":
        return Top3Player()
    if name == "random":
        return RandomPlayer(seed=seed)
    if name == "bayes":
        return BayesPlanner(seed=seed, device=device)
    if name == "bayes_admm":
        return x_players.AdmmBayesPlanner(seed=seed, device=device)
    if name == "bayes_blend":
        return x_players.BlendBayesPlanner(seed=seed, device=device)
    if name == "people_pleaser":
        return x_players.PeoplePleaser(seed=seed, device=device)
    if name == "strong_arm":
        return x_players.StrongArm(seed=seed, kind="fixed")
    if name == "strong_arm_random":
        return x_players.StrongArm(seed=seed, kind="random")
    raise ValueError(name)


def trace_of(final, obs: dict) -> dict:
    """Per-round trace for X8 (true E per round is hidden state, read after the game)."""
    ps = final.player("p0")
    traj = [float(x) for x in ps.share_trajectory]
    polls = [{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in p.items()} for p in obs["polls"]]
    allocs = [{"round": int(a["round"]), "hires": [int(h) for h in a["hires_by_region"]],
               "media": [bool(m) for m in a["media_regions"]], "cost": str(a["cost"])}
              for a in sorted(obs["allocations"], key=lambda a: a["round"])]
    queries = [{k: q[k] for k in ("phase", "round", "sector_a", "sector_b", "region", "cost") if k in q}
               for q in obs["queries"]]
    treasury, spent = [], Decimal(obs["initial_budget"]) - sum((Decimal(q["cost"]) for q in obs["queries"]
                                                                if q.get("phase") == "SURVEY"), Decimal("0"))
    for a in allocs:
        spent -= Decimal(a["cost"]) + sum((Decimal(q["cost"]) for q in obs["queries"]
                                           if q.get("phase") != "SURVEY" and int(q.get("round", -1)) == a["round"]),
                                          Decimal("0"))
        treasury.append(float(spent))
    return {"expected_by_round": traj, "margin_by_round": [x - 0.5 for x in traj], "polls": polls,
            "allocations": allocs, "queries": queries, "treasury_after_round": treasury,
            "manifesto": [float(x) for x in obs["manifesto"]]}


def seed_task(task: tuple) -> dict:
    """All players on one electorate (spawned process)."""
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.strategies.players import play_game
    from strategies import spend_breakdown
    seed, n, device, players = task
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
    rows = []
    for name in players:
        player = make_player(name, seed, device)
        g0 = time.perf_counter()
        final, log = play_game(state0, player)
        obs = observation_for(final, "p0")
        res = obs["result"]
        b = torch.tensor(obs["manifesto"], dtype=v_t.dtype, device=v_t.device)
        decisions = [x for xs in player.latency.values() for x in xs]
        row = {"exp": "X6", "seed": seed, "player": name, "role": ROLE.get(name), "expected": res["expected_share"],
               "vote_share": res["vote_share"], "victory": bool(res["victory"]),
               "static_share_anticipated": float(expected_share(v_t, b, cfg.voting.sensitivity, cfg.voting.threshold)),
               "spend": spend_breakdown(obs, cfg), "rejected": sum(1 for *_, ok, _c in log if not ok),
               "decision_seconds_mean": sum(decisions) / len(decisions), "game_seconds": time.perf_counter() - g0,
               "trace": trace_of(final, obs)}
        if getattr(player, "diagnostics", None):
            row["planner_diagnostics"] = [{k: d[k] for k in ("round", "best_score", "num_candidates", "seconds", "entropy")}
                                          for d in player.diagnostics]
        if getattr(player, "admm_records", None):
            row["admm"] = player.admm_records
        rows.append(row)
    peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    return {"seed": seed, "rows": rows, "seconds": time.perf_counter() - start, "peak_bytes": peak,
            "sizes": [int(x) for x in e.sizes.tolist()], "sigma": e.archetype_of_region.tolist()}


def oracle_rows(seeds: list[int], n: int, device: torch.device, record: dict) -> list[dict]:
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=n))
    es = [generate_electorate(dataclasses.replace(base, seed=s), device=device) for s in seeds]
    ms = torch.stack([oracle_manifesto(e, dataclasses.replace(base, seed=s)) for e, s in zip(es, seeds)])
    cal = xc.calibrate_batch(lambda B: run_batch(es[:B], base, ms[:B], x_policies.policy_planner, seeds=seeds[:B]),
                             device, sizes=(2, 4), max_batch=len(seeds))
    record["calibration"] = cal
    idx = list(range(len(seeds)))
    outs = xc.run_chunks(lambda ch: run_batch([es[i] for i in ch], base, ms[ch], x_policies.policy_planner,
                                              seeds=[seeds[i] for i in ch]), idx, cal["batch"], record=record)
    rows = []
    for i, o in zip(idx, outs):
        cfg = dataclasses.replace(base, seed=seeds[i])
        v_t = anticipate(es[i], cfg)
        traj = [float(x) for x in o.share_trajectory]
        spend_w = float(sum(sum(a.hires_by_region) for a in o.allocations) * base.workers.worker_cost)
        rows.append({"exp": "X6", "seed": seeds[i], "player": "ORACLE", "role": "reference",
                     "expected": o.result.expected_share, "vote_share": o.result.vote_share,
                     "victory": bool(o.result.victory),
                     "static_share_anticipated": float(expected_share(v_t, ms[i], base.voting.sensitivity,
                                                                      base.voting.threshold)),
                     "spend": {"total": float(o.spent), "workers": spend_w, "media": float(o.spent) - spend_w,
                               "queries": 0.0, "num_queries": 0},
                     "trace": {"expected_by_round": traj, "margin_by_round": [x - 0.5 for x in traj], "polls": [],
                               "allocations": [{"round": t + 1, "hires": list(a.hires_by_region),
                                                "media": list(a.media_regions)} for t, a in enumerate(o.allocations)],
                               "queries": [], "manifesto": [float(x) for x in ms[i].tolist()]}})
    return rows


def parity(seeds: list[int], device: torch.device, n: int) -> dict:
    """CPU↔GPU on 3 seeds for bayes, people_pleaser and strong_arm (engine + players on each device)."""
    cache: dict = {}

    def fn(seed, dev):
        key = (seed, str(dev))
        if key not in cache:
            res = seed_task((seed, n, str(dev), ("bayes", "people_pleaser", "strong_arm")))
            cache[key] = {f"E_{r['player']}": r["expected"] for r in res["rows"]}
        return cache[key]

    return xc.parity_check(fn, seeds, device=device, abs_tol=0.02)


# ============================================================================== statistics
def summarise(rows: list[dict], seeds: list[int], x13: dict | None) -> dict:
    by: dict[str, dict[int, dict]] = {}
    for r in rows:
        by.setdefault(r["player"], {})[r["seed"]] = r
    names = list(PLAYERS) + ["ORACLE"]

    def vals(p, key="expected"):
        return [by[p][s][key] for s in seeds]

    players = {}
    for p in names:
        rs = [by[p][s] for s in seeds]
        out = {"role": ROLE.get(p, "reference" if p == "ORACLE" else "baseline"), "win": xc.rate(vals(p, "victory")),
               "expected": xc.mean_ci(vals(p)), "static_share_anticipated": xc.mean_ci(vals(p, "static_share_anticipated")),
               "spend": {k: xc.mean_ci([r["spend"].get(k, 0.0) for r in rs]) for k in ("workers", "media", "queries", "total")},
               "mean_queries": sum(r["spend"].get("num_queries", 0) for r in rs) / len(rs),
               "rejected_actions": sum(r.get("rejected", 0) for r in rs)}
        if p != "ORACLE":
            out["gap_to_oracle"] = xc.paired(vals("ORACLE"), vals(p))
            out["decision_seconds"] = xc.mean_ci([r["decision_seconds_mean"] for r in rs])
            out["game_seconds"] = xc.mean_ci([r["game_seconds"] for r in rs])
        if x13 is not None:
            bound = [x13["E20_search_by_seed"][str(s)] for s in seeds]
            out["gap_to_x13_bound"] = xc.paired(bound, vals(p))
        players[p] = out
    tests: dict = {}
    # family P: the prediction II > I > III (+ II > III), one-sided direction reported, two-sided p, Holm
    fam = []
    for a, b in (("bayes", "people_pleaser"), ("people_pleaser", "strong_arm"), ("bayes", "strong_arm")):
        fam.append({"a": a, "b": b, "expected": xc.paired(vals(a), vals(b)),
                    "mcnemar_wins": xc.mcnemar(vals(a, "victory"), vals(b, "victory"))})
    for x, p in zip(fam, xc.holm([x["expected"]["wilcoxon_p"] for x in fam])):
        x["expected"]["holm_p"] = p
    for x, p in zip(fam, xc.holm([x["mcnemar_wins"]["p"] for x in fam])):
        x["mcnemar_wins"]["holm_p"] = p
    tests["P_ordering_II_I_III"] = {"family": "E and wins, II−I, I−III, II−III; Wilcoxon / exact McNemar, Holm per "
                                              "metric", "tests": fam}
    # family M: manifesto optimisers (all with the rollout planner) + static share
    fam = []
    for a, b in (("bayes", "bayes_admm"), ("bayes", "bayes_blend"), ("bayes_admm", "bayes_blend")):
        fam.append({"a": a, "b": b, "expected": xc.paired(vals(a), vals(b)),
                    "static_share": xc.paired(vals(a, "static_share_anticipated"), vals(b, "static_share_anticipated")),
                    "mcnemar_wins": xc.mcnemar(vals(a, "victory"), vals(b, "victory"))})
    for x, p in zip(fam, xc.holm([x["expected"]["wilcoxon_p"] for x in fam])):
        x["expected"]["holm_p"] = p
    for x, p in zip(fam, xc.holm([x["static_share"]["wilcoxon_p"] for x in fam])):
        x["static_share"]["holm_p"] = p
    tests["M_manifesto_optimisers"] = {"family": "PG (bayes) vs ADMM vs blend, game E and static share on the "
                                                 "anticipated true electorate; Holm per metric", "tests": fam}
    # family B: every player vs the baselines uniform / top3
    fam = []
    for p in PLAYERS:
        for base in ("uniform", "top3"):
            if p != base:
                fam.append({"a": p, "b": base, "expected": xc.paired(vals(p), vals(base)),
                            "mcnemar_wins": xc.mcnemar(vals(p, "victory"), vals(base, "victory"))})
    for x, p in zip(fam, xc.holm([x["expected"]["wilcoxon_p"] for x in fam])):
        x["expected"]["holm_p"] = p
    tests["B_vs_baselines"] = {"family": "each player − uniform / top3 (E Wilcoxon), Holm", "tests": fam}
    # ADMM residuals
    admm = [a for s in seeds for a in by["bayes_admm"][s].get("admm", [])]
    admm_summary = {}
    if admm:
        admm_summary = {"solves": len(admm), "iterations": xc.mean_ci([a["iterations"] for a in admm]),
                        "converged_tol_1e-6": sum(1 for a in admm if a["iterations"] < 300),
                        "primal_final": xc.mean_ci([a["primal_final"] for a in admm]),
                        "dual_final": xc.mean_ci([a["dual_final"] for a in admm]),
                        "primal_final_max": max(a["primal_final"] for a in admm),
                        "dual_final_max": max(a["dual_final"] for a in admm),
                        "seconds": xc.mean_ci([a["seconds"] for a in admm]),
                        "median_trace_first_seed": {"primal": admm[0]["primal"], "dual": admm[0]["dual"]}}
        lens = [len(a["primal"]) for a in admm]
        L = max(lens)
        admm_summary["mean_trace"] = {
            "primal": [sum(a["primal"][min(i, len(a["primal"]) - 1)] for a in admm) / len(admm) for i in range(L)],
            "dual": [sum(a["dual"][min(i, len(a["dual"]) - 1)] for a in admm) / len(admm) for i in range(L)]}
    price_of_not_knowing = xc.paired(vals("ORACLE"), vals("bayes"))
    out = {"players": players, "tests": tests, "admm_residuals": admm_summary,
           "price_of_not_knowing_oracle_minus_bayes": price_of_not_knowing,
           "ranking_by_mean_E": sorted(names, key=lambda p: -players[p]["expected"]["mean"])}
    if x13 is not None:
        o20 = [x13["E20_oracle_by_seed"][str(s)] for s in seeds]
        out["x13_bound"] = {"E20_search": xc.mean_ci([x13["E20_search_by_seed"][str(s)] for s in seeds]),
                            "E20_oracle": xc.mean_ci(o20),
                            "oracle_single_stream_vs_E20": xc.paired(vals("ORACLE"), o20),
                            "note": "bound = X13 open-loop search with the ORACLE manifesto, mean E over 20 fresh "
                                    "streams; players and ORACLE here are single canonical-stream games"}
        out["price_of_not_knowing_x13_minus_bayes"] = xc.paired([x13["E20_search_by_seed"][str(s)] for s in seeds],
                                                                vals("bayes"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X6"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--players", default=",".join(PLAYERS))
    parser.add_argument("--budget-hours", type=float, default=2.5)
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--out", default=str(xc.REPORTS / "x6.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    free0 = xc.free_bytes(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    players = tuple(p for p in args.players.split(",") if p)
    deviations = [f"Device {device} instead of the plan default cuda:2 (cuda:2 is used by the Phase 2 agent)."]
    record: dict = {}
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        print(f"[x6] device {device}, seeds {seeds[0]}..{seeds[-1]}, players {players}", flush=True)
        pilot = seed_task((seeds[0], args.n, str(device), players))
        est = pilot["seconds"] * (len(seeds) - 1) / max(1, args.workers) * 1.15
        per = {r["player"]: round(r["game_seconds"], 2) for r in pilot["rows"]}
        print(f"[x6] pilot seed {seeds[0]}: {pilot['seconds']:.1f}s {per}; estimated remaining {est / 60:.1f} min",
              flush=True)
        if args.pilot_only:
            print(json.dumps({r["player"]: r["expected"] for r in pilot["rows"]}))
            return
        if est > args.budget_hours * 3600:
            raise SystemExit("estimate exceeds the time budget; reduce and record the deviation")
        results = [pilot] + xc.run_tasks(seed_task, [(s, args.n, str(device), players) for s in seeds[1:]],
                                         args.workers, label="[x6]")
        rows = [r for res in results for r in res["rows"]]
        t_or = time.perf_counter()
        rows += oracle_rows(seeds, args.n, device, record)
        record["seconds"] = time.perf_counter() - t_or
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text("")
        for r in rows:
            xc.write_row(RAW_PATH, r)
        par = parity(seeds[:args.parity_seeds], device, args.n)
        print(f"[x6] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    x13 = json.loads(X13_PATH.read_text()) if X13_PATH.exists() else None
    if x13 is None:
        deviations.append("reports/v3/x13.json missing: no gap to the X13 bound")
    summary = summarise(rows, seeds, x13)
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    report = {
        "experiment": "X6", "question": "Strategy tournament (players I/II/III, baselines, manifesto optimisers)",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget),
        "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
        "players": {p: make_player(p, 0, "cpu").__class__.__name__ for p in players},
        "oracle_note": "ORACLE = oracle_manifesto + balance.policy_planner on the true state (canonical streams). "
                       "It is a HEURISTIC, not an optimum: the manifesto ignores the campaign and the planner is "
                       "greedy; X13's search is the upper bound.",
        **summary,
        "per_seed_seconds": xc.mean_ci([res["seconds"] for res in results]),
        "pilot": {"seed": seeds[0], "seconds": pilot["seconds"], "estimated_remaining_seconds": est},
        "oracle_run": record, "deviations": deviations, "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, workers=args.workers,
                                       batch_sizes={"oracle_run_batch": record.get("batch_sizes"),
                                                    "bayes_planner_num_worlds": 8, "bayes_world_voters": 1600},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"child_peak_bytes_max": max((r["peak_bytes"] or 0) for r in results)}),
    }
    xc.save(args.out, report)
    print(json.dumps({p: [round(v["win"]["rate"], 3), round(v["expected"]["mean"], 4)]
                      for p, v in summary["players"].items()}), flush=True)
    print(f"[x6] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
