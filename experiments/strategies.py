"""P4 strategy comparison (spec §10.2, §12 criterion 2 re-check; plan P4 exit criteria).

Every player plays full games through the player-facing ``GameEngine`` (observation dicts only):
  uniform      UniformPlayer   — no queries, uniform manifesto, free workers only
  top3         Top3Player      — free top-3 statistic manifesto, even split spending
  random       RandomPlayer    — seeded random legal actions
  bayes_even   BayesPlanner(planning=False) — Bayesian survey + posterior manifesto, even split spending (ablation)
  bayes        BayesPlanner    — Bayesian survey + posterior manifesto + rollout planner
plus the ORACLE reference (labelled, never mixed with players): oracle manifesto (true v, anticipated) +
the oracle marginal-value planner of ``experiments/balance.py`` via ``simulate_world`` (true state).

Queries follow R12 (sampled 10% surveys, repeats allowed and pooled); ``reports/v2/strategies.json`` holds the
earlier exact-query run (same seeds) — ``--compare`` adds paired by-seed tests against it.

Usage:
  python experiments/strategies.py --seeds 200 --n 10000 --device cpu --out reports/v2/strategies_r12.json \
      --compare reports/v2/strategies.json
  python experiments/strategies.py --seeds 20 --out reports/v2/strategies_quick.json
  python experiments/strategies.py --info-curve 0,1,2,4,8,12 --seeds 60 --out reports/v2/info_curve_r12.json
Then: python experiments/plot_strategies.py --in reports/v2/strategies_r12.json --suffix _r12
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from balance import policy_planner, wilson  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.game import GameEngine  # noqa: E402
from dumbocracy.game.observation import observation_for  # noqa: E402
from dumbocracy.simulate import simulate_world  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402
from dumbocracy.strategies.players import (  # noqa: E402
    BayesPlanner, RandomPlayer, Top3Player, UniformPlayer, play_game,
)

PLAYERS = ("uniform", "top3", "random", "bayes_even", "bayes")
ORACLE = "ORACLE"


def forced_bayes(seed: int, k: int) -> BayesPlanner:
    """BayesPlanner asking exactly k survey queries (every stop rule disabled; no in-round queries)."""
    return BayesPlanner(seed=seed, num_queries=k, stop_entropy=None, min_gain_per_coin=None, round_query_gain=None,
                        survey_budget=Decimal("500.00"))


def make_player(name: str, seed: int):
    if name.startswith("bayes_k"):
        return forced_bayes(seed, int(name[len("bayes_k"):]))
    if name == "uniform":
        return UniformPlayer()
    if name == "top3":
        return Top3Player()
    if name == "random":
        return RandomPlayer(seed=seed)
    if name == "bayes_even":
        return BayesPlanner(seed=seed, planning=False)
    if name == "bayes":
        return BayesPlanner(seed=seed)
    raise ValueError(name)


def pick_device(requested: str) -> torch.device:
    """``auto``: cuda:2 if it has ≥ 4 GB free and < 50 % utilisation, else CPU (shared host)."""
    if requested != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        return torch.device("cpu")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            idx, free, util = (int(x.strip()) for x in line.split(","))
            if idx == 2 and free >= 4096 and util < 50:
                return torch.device("cuda:2")
    except Exception:  # noqa: BLE001
        pass
    return torch.device("cpu")


def spend_breakdown(observation: dict, config: GameConfig) -> dict:
    """Money by category from the player's own (public) history."""
    worker_unit = config.workers.worker_cost
    workers = sum((sum(a["hires_by_region"]) * worker_unit for a in observation["allocations"]), Decimal("0.00"))
    total_alloc = sum((Decimal(a["cost"]) for a in observation["allocations"]), Decimal("0.00"))
    queries = sum((Decimal(q["cost"]) for q in observation["queries"]), Decimal("0.00"))
    return {"workers": float(workers), "media": float(total_alloc - workers), "queries": float(queries),
            "total": float(total_alloc + queries), "num_queries": len(observation["queries"]),
            "survey_queries": sum(1 for q in observation["queries"] if q["phase"] == "SURVEY"),
            "repeat_queries": sum(1 for q in observation["queries"] if q.get("repeat_index", 1) > 1),
            "media_airings": sum(observation["media_airings"].values())}


def run_seed(seed: int, n: int, device: torch.device, players: tuple[str, ...], oracle: bool = True) -> dict:
    cfg = GameConfig(seed=seed, device=str(device), electorate=ElectorateConfig(num_voters=n))
    electorate = generate_electorate(cfg, device=device)
    state0 = GameEngine.new_game(cfg, electorate=electorate, game_id=f"s{seed}")
    row: dict = {"seed": seed, "runs": {}}
    for name in players:
        player = make_player(name, seed)
        start = time.perf_counter()
        final, log = play_game(state0, player)
        seconds = time.perf_counter() - start
        obs = observation_for(final, "p0")
        result = obs["result"]
        decisions = [x for xs in player.latency.values() for x in xs]
        row["runs"][name] = {
            "expected": result["expected_share"], "vote_share": result["vote_share"], "victory": result["victory"],
            "manifesto": [round(x, 4) for x in obs["manifesto"]], "spend": spend_breakdown(obs, cfg),
            "rejected": sum(1 for *_, ok, _code in log if not ok), "actions": len(log),
            "decision_seconds_mean": sum(decisions) / len(decisions), "decision_seconds_max": max(decisions),
            "allocate_seconds_mean": (sum(player.latency.get("allocate", [0])) / max(1, len(player.latency.get("allocate", [])))),
            "commit_seconds": sum(player.latency.get("commit", [0.0])), "game_seconds": seconds,
            "polls": [round(p["estimate"], 4) for p in obs["polls"]],
        }
    row["sigma"] = electorate.archetype_of_region.tolist()
    row["tau"] = electorate.personality_of_region.tolist()
    row["sizes"] = electorate.sizes.tolist()
    if not oracle:
        return row
    # ORACLE reference (true state; labelled, never mixed with player results)
    start = time.perf_counter()
    b = oracle_manifesto(electorate, cfg)
    out = simulate_world(electorate, cfg, b, policy_planner)
    row["oracle"] = {"label": ORACLE, "expected": out.result.expected_share, "vote_share": out.result.vote_share,
                     "victory": out.result.victory, "spent": float(out.spent), "seconds": time.perf_counter() - start}
    return row


def paired(rows: list[dict], a: str, b: str) -> dict:
    from scipy.stats import wilcoxon

    def value(row, name):
        return row["oracle"]["expected"] if name == ORACLE else row["runs"][name]["expected"]
    diffs = [value(r, a) - value(r, b) for r in rows]
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 1:
        p_two = p_greater = 1.0
    else:
        p_two = float(wilcoxon(diffs, alternative="two-sided").pvalue)
        p_greater = float(wilcoxon(diffs, alternative="greater").pvalue)
    wins_a = sum(1 for d in diffs if d > 0)
    mean = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(1, len(diffs) - 1))
    return {"a": a, "b": b, "mean_diff": mean, "sd_diff": sd, "a_better_seeds": wins_a, "n": len(diffs),
            "wilcoxon_p_two_sided": p_two, "wilcoxon_p_greater": p_greater}


def summarise(rows: list[dict], players: tuple[str, ...]) -> dict:
    n = len(rows)
    summary = {}
    for name in players:
        runs = [r["runs"][name] for r in rows]
        wins = sum(x["victory"] for x in runs)
        exp_wins = sum(x["expected"] > 0.5 for x in runs)
        mean = lambda key: sum(x[key] for x in runs) / n  # noqa: E731
        summary[name] = {
            "win_rate": wins / n, "win_ci95": list(wilson(wins, n)),
            "expected_gt_half_rate": exp_wins / n, "expected_gt_half_ci95": list(wilson(exp_wins, n)),
            "mean_expected_share": mean("expected"),
            "sd_expected_share": math.sqrt(sum((x["expected"] - mean("expected")) ** 2 for x in runs) / max(1, n - 1)),
            "mean_spend": {k: sum(x["spend"][k] for x in runs) / n for k in ("workers", "media", "queries", "total")},
            "mean_queries": sum(x["spend"]["num_queries"] for x in runs) / n,
            "mean_survey_queries": sum(x["spend"]["survey_queries"] for x in runs) / n,
            "mean_repeat_queries": sum(x["spend"].get("repeat_queries", 0) for x in runs) / n,
            "mean_media_airings": sum(x["spend"]["media_airings"] for x in runs) / n,
            "rejected_actions": sum(x["rejected"] for x in runs),
            "mean_decision_seconds": mean("decision_seconds_mean"),
            "max_decision_seconds": max(x["decision_seconds_max"] for x in runs),
            "mean_allocate_seconds": mean("allocate_seconds_mean"),
            "mean_commit_seconds": mean("commit_seconds"),
            "oracle_gap_mean_expected": sum(r["oracle"]["expected"] - r["runs"][name]["expected"] for r in rows) / n,
        }
    o_wins = sum(r["oracle"]["victory"] for r in rows)
    summary[ORACLE] = {"label": "ORACLE: full-information benchmark (true v), not a player",
                       "win_rate": o_wins / n, "win_ci95": list(wilson(o_wins, n)),
                       "expected_gt_half_rate": sum(r["oracle"]["expected"] > 0.5 for r in rows) / n,
                       "mean_expected_share": sum(r["oracle"]["expected"] for r in rows) / n,
                       "mean_spend": {"total": sum(r["oracle"]["spent"] for r in rows) / n}}
    comparisons = []
    for name in players:
        for base in ("uniform", "top3"):
            if name != base and base in players:
                comparisons.append(paired(rows, name, base))
    if "bayes" in players and "bayes_even" in players:
        comparisons.append(paired(rows, "bayes", "bayes_even"))
    if "bayes" in players:
        comparisons.append(paired(rows, ORACLE, "bayes"))
    return {"summary": summary, "paired": comparisons}


def compare_runs(rows: list[dict], old: dict, players: tuple[str, ...]) -> list[dict]:
    """Paired by seed against an earlier report (e.g. the exact-query run): Wilcoxon on expected share and
    McNemar (exact binomial on discordant pairs) on victory, for every player present in both."""
    from scipy.stats import binomtest, wilcoxon
    old_rows = {r["seed"]: r for r in old["rows"]}
    out = []
    for name in players:
        pairs = [(r["runs"][name], old_rows[r["seed"]]["runs"][name]) for r in rows
                 if r["seed"] in old_rows and name in old_rows[r["seed"]]["runs"]]
        if not pairs:
            continue
        diffs = [a["expected"] - b["expected"] for a, b in pairs]
        nonzero = [d for d in diffs if d != 0]
        p = float(wilcoxon(diffs).pvalue) if len(nonzero) >= 1 else 1.0
        gained = sum(1 for a, b in pairs if a["victory"] and not b["victory"])
        lost = sum(1 for a, b in pairs if b["victory"] and not a["victory"])
        p_win = float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0
        out.append({"player": name, "n": len(pairs), "mean_expected_new": sum(a["expected"] for a, _ in pairs) / len(pairs),
                    "mean_expected_old": sum(b["expected"] for _, b in pairs) / len(pairs),
                    "mean_diff": sum(diffs) / len(diffs), "identical_seeds": len(diffs) - len(nonzero),
                    "wilcoxon_p_two_sided": p, "win_rate_new": sum(a["victory"] for a, _ in pairs) / len(pairs),
                    "win_rate_old": sum(b["victory"] for _, b in pairs) / len(pairs),
                    "wins_gained": gained, "wins_lost": lost, "mcnemar_exact_p": p_win})
    return out


def info_curve(ks: list[int], seeds: int, first_seed: int, n: int, device: torch.device, out_path: str) -> None:
    """Mean expected share and win rate of BayesPlanner forced to ask exactly k survey queries."""
    players = tuple(f"bayes_k{k}" for k in ks)
    print(f"info curve: device={device} seeds={seeds} n={n} k={ks}", flush=True)
    rows, start = [], time.perf_counter()
    for i, seed in enumerate(range(first_seed, first_seed + seeds)):
        rows.append(run_seed(seed, n, device, players, oracle=False))
        if (i + 1) % 10 == 0 or i + 1 == seeds:
            print(f"[{i + 1}/{seeds}] {time.perf_counter() - start:.0f}s", flush=True)
    curve = []
    for k, name in zip(ks, players):
        runs = [r["runs"][name] for r in rows]
        wins = sum(x["victory"] for x in runs)
        mean = sum(x["expected"] for x in runs) / len(runs)
        curve.append({"k": k, "mean_expected_share": mean,
                      "sd_expected_share": math.sqrt(sum((x["expected"] - mean) ** 2 for x in runs) / max(1, len(runs) - 1)),
                      "win_rate": wins / len(runs), "win_ci95": list(wilson(wins, len(runs))),
                      "mean_query_spend": sum(x["spend"]["queries"] for x in runs) / len(runs),
                      "mean_repeat_queries": sum(x["spend"]["repeat_queries"] for x in runs) / len(runs),
                      "mean_decision_seconds": sum(x["decision_seconds_mean"] for x in runs) / len(runs),
                      "max_decision_seconds": max(x["decision_seconds_max"] for x in runs)})
    base = players[0]
    paired_vs_first = []
    for k, name in zip(ks[1:], players[1:]):
        p = paired([{"runs": r["runs"], "oracle": {"expected": 0.0}} for r in rows], name, base)
        paired_vs_first.append({"k": k, **{kk: p[kk] for kk in ("mean_diff", "sd_diff", "a_better_seeds", "wilcoxon_p_two_sided")}})
    report = {"num_seeds": seeds, "num_voters": n, "first_seed": first_seed, "device": str(device),
              "seconds": time.perf_counter() - start, "query_rules": "R12 sampled surveys (10% of region, repeats pooled)",
              "player": "BayesPlanner(num_queries=k, stop_entropy=None, min_gain_per_coin=None, round_query_gain=None)",
              "curve": curve, f"paired_vs_k{ks[0]}": paired_vs_first, "rows": rows}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1))
    print(json.dumps([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in curve], indent=1))
    print(json.dumps(paired_vs_first, indent=1))
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=5000)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="auto", help="auto = cuda:2 if idle enough, else cpu")
    parser.add_argument("--players", default=",".join(PLAYERS))
    parser.add_argument("--out", default="reports/v2/strategies_r12.json")
    parser.add_argument("--compare", default=None, help="earlier report to compare against, paired by seed")
    parser.add_argument("--info-curve", default=None, help="comma-separated k values: forced-query BayesPlanner curve")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    device = pick_device(args.device)
    if args.info_curve:
        info_curve([int(k) for k in args.info_curve.split(",")], args.seeds, args.first_seed, args.n, device, args.out)
        return
    players = tuple(p for p in args.players.split(",") if p)
    print(f"device={device} seeds={args.seeds} n={args.n} players={players}", flush=True)
    rows = []
    start = time.perf_counter()
    for i, seed in enumerate(range(args.first_seed, args.first_seed + args.seeds)):
        rows.append(run_seed(seed, args.n, device, players))
        if (i + 1) % 10 == 0 or i + 1 == args.seeds:
            means = {p: round(sum(r["runs"][p]["expected"] for r in rows) / len(rows), 4) for p in players}
            means[ORACLE] = round(sum(r["oracle"]["expected"] for r in rows) / len(rows), 4)
            print(f"[{i + 1}/{args.seeds}] {time.perf_counter() - start:.0f}s mean expected {means}", flush=True)
    report = {"num_seeds": len(rows), "num_voters": args.n, "first_seed": args.first_seed, "device": str(device),
              "seconds": time.perf_counter() - start, "players": list(players),
              "query_rules": "R12 sampled surveys (10% of region, repeats pooled)",
              **summarise(rows, players),
              "environment": {"torch": torch.__version__, "python": platform.python_version(),
                              "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                                    text=True).stdout.strip()},
              "config": dataclasses.replace(GameConfig(), device=str(device),
                                            electorate=ElectorateConfig(num_voters=args.n)).to_record(),
              "notes": ["Players act only on observation dicts through GameEngine; ORACLE uses the true state "
                        "(oracle manifesto on the anticipated true electorate + balance.policy_planner) and is "
                        "a benchmark, not a player.",
                        "expected = deterministic expected share E = mean_i P_i at the election (spec §5); "
                        "victory = official sampled vote > N/2."],
              "rows": rows}
    if args.compare:
        report["compare"] = {"against": args.compare,
                             "paired": compare_runs(rows, json.loads(Path(args.compare).read_text()), players)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)
    table = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()
                 if kk in ("win_rate", "win_ci95", "mean_expected_share", "oracle_gap_mean_expected",
                           "mean_queries", "mean_decision_seconds")}
             for k, v in report["summary"].items()}
    print(json.dumps(table, indent=1))
    print(json.dumps([{k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()} for c in report["paired"]], indent=1))
    if args.compare:
        print(json.dumps(report["compare"]["paired"], indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
