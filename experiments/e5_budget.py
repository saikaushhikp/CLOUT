"""E5 — Budget: P(Victory | B) (spec §3, §7, U14; lectures: optimisation under a resource constraint — the
oracle planner ranks options by marginal expected-vote gain per unit cost, a greedy knapsack/Lagrangian rule).

Factor: starting treasury B ∈ {250, 500 (default), 1000}; the same seeds (same electorate) for every B (paired).
Per seed and B:
* BayesPlanner (observation-only; sampled surveys + posterior manifesto + rollout planner) and Top3Player through
  the player-facing GameEngine; spend mix (workers / media / queries) from the player's own public history.
* ORACLE: oracle manifesto (anticipated true electorate; independent of B) + oracle marginal planner via
  simulate_world; spend mix from its allocations.
Theory: the marginal value of money is decreasing (media has diminishing returns η_m0/(1 + λn); worker cascades
compound multiplicatively), so P(Victory | B) should be concave in log B with diminishing gains per doubling.

Usage: python experiments/e5_budget.py --seeds 200 --n 10000 --device auto --workers 6 --out reports/v2/e5_budget.json
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from decimal import Decimal

import p7_common as pc

BUDGETS = ("250.00", "500.00", "1000.00")


def run_task(task):
    import torch
    from balance import POLICIES
    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.simulate import simulate_world
    from dumbocracy.strategies.oracle import oracle_manifesto
    from dumbocracy.strategies.players import BayesPlanner, Top3Player, play_game
    from strategies import spend_breakdown

    seed, n, device = task
    t0 = time.perf_counter()
    dev = torch.device(device)
    base = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(base, device=dev)
    bo = oracle_manifesto(e, base)
    row = {"seed": seed, "budgets": {}}
    for B in BUDGETS:
        cfg = dataclasses.replace(base, budget=Decimal(B))
        runs = {}
        out = simulate_world(e, cfg, bo, POLICIES["planner"])
        w = sum(a.total_hires for a in out.allocations) * float(cfg.workers.worker_cost)
        runs["oracle"] = {"expected": out.result.expected_share, "victory": out.result.victory,
                          "spend": {"workers": w, "media": float(out.spent) - w, "queries": 0.0, "total": float(out.spent)}}
        ccpu = dataclasses.replace(cfg, device="cpu")
        state0 = GameEngine.new_game(cfg, electorate=e, game_id=f"e5s{seed}b{B}")
        for name, player in (("bayes", BayesPlanner(seed=seed, config=ccpu)), ("top3", Top3Player(config=ccpu))):
            final, _ = play_game(state0, player)
            obs = observation_for(final, "p0")
            runs[name] = {"expected": obs["result"]["expected_share"], "victory": obs["result"]["victory"],
                          "spend": spend_breakdown(obs, cfg)}
        row["budgets"][B] = runs
    row["seconds"] = time.perf_counter() - t0
    return row


def summarise(rows):
    out = {"by_budget": {}, "paired": {}}
    for B in BUDGETS:
        c = {}
        for name in ("oracle", "bayes", "top3"):
            rs = [r["budgets"][B][name] for r in rows]
            c[name] = {"win": pc.rate([x["victory"] for x in rs]), "expected": pc.mean_ci([x["expected"] for x in rs]),
                       "spend": {k: pc.mean_ci([x["spend"][k] for x in rs]) for k in ("workers", "media", "queries", "total")}}
            tot = sum(x["spend"]["total"] for x in rs)
            c[name]["spend_share"] = {k: (sum(x["spend"][k] for x in rs) / tot if tot else 0.0) for k in ("workers", "media", "queries")}
        out["by_budget"][B] = c
    for a, b in (("250.00", "500.00"), ("1000.00", "500.00"), ("1000.00", "250.00")):
        for name in ("oracle", "bayes", "top3"):
            out["paired"][f"{name}|{a}_vs_{b}"] = {
                "expected": pc.paired([r["budgets"][a][name]["expected"] for r in rows], [r["budgets"][b][name]["expected"] for r in rows]),
                "win_mcnemar": pc.mcnemar([r["budgets"][a][name]["victory"] for r in rows], [r["budgets"][b][name]["victory"] for r in rows])}
    for B in BUDGETS:
        out["paired"][f"oracle_minus_bayes|{B}"] = pc.paired([r["budgets"][B]["oracle"]["expected"] for r in rows],
                                                             [r["budgets"][B]["bayes"]["expected"] for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--first-seed", type=int, default=5000)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="reports/v2/e5_budget.json")
    args = ap.parse_args()
    device = pc.pick_device(args.device)
    tasks = [(s, args.n, device) for s in range(args.first_seed, args.first_seed + args.seeds)]
    print(f"E5 device={device} tasks={len(tasks)}", flush=True)
    start = time.perf_counter()
    rows = pc.run_tasks(run_task, tasks, args.workers, "E5")
    report = {"experiment": "E5 budget", "num_seeds": args.seeds, "num_voters": args.n, "budgets": BUDGETS,
              **summarise(rows), "provenance": pc.provenance(args, device, time.perf_counter() - start, args.workers),
              "rows": rows}
    pc.save(args.out, report)


if __name__ == "__main__":
    main()
