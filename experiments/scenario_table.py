"""P5 scenario table: how often each epistemic property holds, per round, and truth vs knowledge (spec §10; R14).

Players play full games through the player-facing ``GameEngine`` (observation dicts only):
  bayes  BayesPlanner — Bayesian sampled surveys, posterior manifesto, rollout planner
  top3   Top3Player   — no queries, free top-3 manifesto, even spending
Every finished game gets an epistemic timeline (``dumbocracy.logic.report``) with the default settings
(16 consistent worlds per round, 1600-voter worlds). We aggregate, per player and round, the frequency of
possible / guaranteed / knows-that / inevitable-defeat / truth-possible, the truth-vs-knowledge confusion
tables (truth_victory_possible vs possible_victory and vs guaranteed_victory), temporal properties of the
per-game traces (bounded LTL + STL robustness, ``dumbocracy.logic.temporal``) and how knowledge relates to
the actual outcome.

Usage:
  python experiments/scenario_table.py --seeds 60 --n 10000 --workers 3
Outputs reports/v2/scenario_table.json and reports/v2/scenario_table.md.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from decimal import Decimal
from typing import Any

from dumbocracy.sectors import REGIONS, SECTORS
from dumbocracy.strategies.players import BayesPlanner, Top3Player

ROOT = Path(__file__).resolve().parents[1]
PROPS = ("possible_victory", "guaranteed_victory", "knows_win_exists", "inevitable_defeat", "truth_victory_possible")
SHORT = {"possible_victory": "Possible (∃π∃w)", "guaranteed_victory": "Guaranteed (∃π∀w)",
         "knows_win_exists": "Knows-that (∀w∃π)", "inevitable_defeat": "Inevitable defeat (∀π∀w¬)",
         "truth_victory_possible": "Truth: win possible"}


def _setup_worker(threads: int) -> None:
    import torch
    torch.set_num_threads(threads)


class RandomQueryPlanner(BayesPlanner):
    """BayesPlanner whose survey queries are uniformly random triples (same count, same later play)."""

    def _maybe_query(self, observation, posterior, survey):
        if not survey or "query" not in observation["legal_actions"]:
            return None
        asked = [q for q in observation.get("queries", []) if q.get("phase") == "SURVEY"]
        if self.num_queries is not None and len(asked) >= self.num_queries:
            return None
        rng = random.Random(f"{self.seed}:{observation['seed']}:{len(asked)}")
        a, b = rng.sample(SECTORS, 2)
        return {"type": "query", "sector_a": a, "sector_b": b, "region": rng.choice(REGIONS)}


def make_player(name: str, seed: int):
    """``bayes`` | ``top3`` | ``forced-<k>:gain`` | ``forced-<k>:random`` (exactly k survey queries, no in-round
    queries: knowledge then grows from polls only)."""
    if name == "bayes":
        return BayesPlanner(seed=seed)
    if name == "top3":
        return Top3Player()
    if name.startswith("forced"):
        k, policy = name.split("-", 1)[1].split(":")
        cls = BayesPlanner if policy == "gain" else RandomQueryPlanner
        return cls(seed=seed, num_queries=int(k), stop_entropy=None, min_gain_per_coin=None, round_query_gain=None,
                   survey_budget=Decimal("500.00"))
    raise ValueError(name)


def run_game(task: tuple[str, int, int, int]) -> dict[str, Any]:
    """Play one game and compute its timeline; returns a JSON-safe row (aggregated knowledge, no hidden data)."""
    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.logic.report import timeline_with_diagnostics
    from dumbocracy.logic.temporal import And, Atom, F, G, Implies, Not, Pred, U, game_trace, holds, robustness
    from dumbocracy.strategies.players import play_game

    player_name, seed, n, worlds = task
    cfg = GameConfig(seed=seed, device="cpu", electorate=ElectorateConfig(num_voters=n))
    t0 = time.perf_counter()
    state, _ = play_game(GameEngine.new_game(cfg, game_id=f"s{seed}"), make_player(player_name, seed))
    play_s = time.perf_counter() - t0
    timeline, diag = timeline_with_diagnostics(state, "p0", worlds=worlds, seed=0)
    obs = observation_for(state, "p0")
    trace = game_trace(obs, timeline)
    lead = Pred("margin", 0.0)
    formulas = {
        "F Lead": F(lead), "G Lead": G(lead), "F Victory": F(Atom("Victory")),
        "G Viable": G(Atom("Viable")), "F Guaranteed": F(Atom("Guaranteed")),
        "G(Viable) and not F(Victory)": And(G(Atom("Viable")), Not(F(Atom("Victory")))),
        "Lead U Victory": U(lead, Atom("Victory")),
        "G(BudgetLeft -> Viable)": G(Implies(Atom("BudgetLeft"), Atom("Viable"))),
    }
    ltl = {k: holds(f, trace) for k, f in formulas.items()}
    stl = {"rho F(margin>0)": robustness(F(lead), trace), "rho G(margin>0)": robustness(G(lead), trace),
           "rho G[<=2](margin>0) from round 3": robustness(G(lead, 2), trace, min(2, len(trace) - 1))}
    queries = len(obs["queries"])
    return {"player": player_name, "seed": seed, "n": n, "queries": queries,
            "survey_queries": sum(1 for q in obs["queries"] if q["phase"] == "SURVEY"),
            "timeline": timeline, "ltl": ltl, "stl": stl,
            "acceptance": [r["accepted"] / r["draws"] for r in diag["rounds"]],
            "fallback_rounds": sum(1 for r in diag["rounds"] if r["fallback"]),
            "play_seconds": play_s, "timeline_seconds": diag["seconds"]}


def run_all(tasks: list[tuple[str, int, int, int]], workers: int, threads: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    if workers <= 1:
        _setup_worker(threads)
        for i, task in enumerate(tasks):
            rows.append(run_game(task))
            print(f"[{i + 1}/{len(tasks)}] {task[0]} seed {task[1]} {time.perf_counter() - start:.0f}s", flush=True)
        return rows
    import multiprocessing as mp
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_setup_worker,
                             initargs=(threads,)) as pool:
        for i, row in enumerate(pool.map(run_game, tasks)):
            rows.append(row)
            print(f"[{i + 1}/{len(tasks)}] {row['player']} seed {row['seed']} {time.perf_counter() - start:.0f}s", flush=True)
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def confusion(pairs: list[tuple[bool, bool]]) -> dict[str, int]:
    return {"truth_T_know_T": sum(1 for t, k in pairs if t and k), "truth_T_know_F": sum(1 for t, k in pairs if t and not k),
            "truth_F_know_T": sum(1 for t, k in pairs if not t and k), "truth_F_know_F": sum(1 for t, k in pairs if not t and not k)}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for player in sorted({r["player"] for r in rows}):
        sub = [r for r in rows if r["player"] == player]
        rounds = sorted({x["round"] for r in sub for x in r["timeline"]["rounds"]})
        per_round = {}
        for t in rounds:
            cells = [x for r in sub for x in r["timeline"]["rounds"] if x["round"] == t]
            per_round[t] = {p: sum(bool(c[p]) for c in cells) / len(cells) for p in PROPS}
            per_round[t]["n"] = len(cells)
            per_round[t]["mean_entropy"] = sum(c["knowledge"]["posterior_entropy"] for c in cells) / len(cells)
            per_round[t]["mean_best_win_fraction"] = sum(c["best_plan"]["win_fraction"] for c in cells) / len(cells)
        cells = [x for r in sub for x in r["timeline"]["rounds"]]
        wins = [r["timeline"]["actual"]["victory"] for r in sub]
        g1 = [r["timeline"]["rounds"][0]["guaranteed_victory"] for r in sub]
        any_g = [any(x["guaranteed_victory"] for x in r["timeline"]["rounds"]) for r in sub]
        lastp = [r["timeline"]["rounds"][-1]["possible_victory"] for r in sub]
        out[player] = {
            "games": len(sub),
            "win_rate": sum(wins) / len(sub), "win_rate_ci": wilson(sum(wins), len(sub)),
            "mean_queries": sum(r["queries"] for r in sub) / len(sub),
            "per_round": per_round,
            "confusion_possible": confusion([(c["truth_victory_possible"], c["possible_victory"]) for c in cells]),
            "confusion_guaranteed": confusion([(c["truth_victory_possible"], c["guaranteed_victory"]) for c in cells]),
            "knowledge_vs_outcome": {
                "P(win | guaranteed at round 1)": _cond(wins, g1),
                "P(win | guaranteed at some round)": _cond(wins, any_g),
                "P(win | never guaranteed)": _cond(wins, [not x for x in any_g]),
                "P(win | possible at round T)": _cond(wins, lastp),
                "P(win | not possible at round T)": _cond(wins, [not x for x in lastp]),
            },
            "ltl_frequency": {k: sum(r["ltl"][k] for r in sub) / len(sub) for k in sub[0]["ltl"]},
            "stl_mean_robustness": {k: _finite_mean([r["stl"][k] for r in sub]) for k in sub[0]["stl"]},
            "mean_acceptance": sum(sum(r["acceptance"]) / len(r["acceptance"]) for r in sub) / len(sub),
            "fallback_rounds": sum(r["fallback_rounds"] for r in sub),
            "timeline_seconds": {"mean": sum(r["timeline_seconds"] for r in sub) / len(sub),
                                 "max": max(r["timeline_seconds"] for r in sub)},
        }
    return out


def _finite_mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def _cond(wins: list[bool], cond: list[bool]) -> dict[str, Any]:
    sel = [w for w, c in zip(wins, cond) if c]
    return {"n": len(sel), "p": (sum(sel) / len(sel)) if sel else None}


def markdown(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = [f"# P5 scenario table ({meta['seeds']} seeds, N = {meta['n']}, {meta['worlds']} worlds/round)", "",
             "Frequency over games of each property at decision point t (start of round t after the poll and the "
             "player's round-t queries). Knowledge columns use only the player's own information over sampled "
             "consistent worlds × a finite plan set; 'Truth' evaluates the same plans in the true world.", ""]
    for player, s in summary.items():
        lines += [f"## {player} — win rate {100 * s['win_rate']:.1f}% (95% CI {100 * s['win_rate_ci'][0]:.0f}–"
                  f"{100 * s['win_rate_ci'][1]:.0f}%), mean queries {s['mean_queries']:.1f}", "",
                  "| round | " + " | ".join(SHORT[p] for p in PROPS) + " | σ entropy (nats) | best-plan win frac |",
                  "|---|" + "---|" * (len(PROPS) + 2)]
        for t, row in s["per_round"].items():
            lines.append(f"| {t} | " + " | ".join(f"{100 * row[p]:.0f}%" for p in PROPS)
                         + f" | {row['mean_entropy']:.2f} | {row['mean_best_win_fraction']:.2f} |")
        cp, cg = s["confusion_possible"], s["confusion_guaranteed"]
        lines += ["", "Truth vs knowledge (all rounds pooled):", "",
                  "| | knows Possible | ¬Possible | knows Guaranteed | ¬Guaranteed |", "|---|---|---|---|---|",
                  f"| truth: win possible | {cp['truth_T_know_T']} | {cp['truth_T_know_F']} | {cg['truth_T_know_T']} | {cg['truth_T_know_F']} |",
                  f"| truth: no win possible | {cp['truth_F_know_T']} | {cp['truth_F_know_F']} | {cg['truth_F_know_T']} | {cg['truth_F_know_F']} |",
                  "", "Knowledge vs outcome: " + "; ".join(
                      f"{k} = {('n/a' if v['p'] is None else f'{100 * v['p']:.0f}%')} (n={v['n']})"
                      for k, v in s["knowledge_vs_outcome"].items()), "",
                  "Temporal (per game trace): " + "; ".join(f"{k}: {100 * v:.0f}%" for k, v in s["ltl_frequency"].items()),
                  "", "STL mean robustness: " + "; ".join(
                      f"{k} = {'n/a' if v is None else f'{v:+.3f}'}" for k, v in s["stl_mean_robustness"].items()),
                  "", f"Poll-consistency acceptance rate {100 * s['mean_acceptance']:.0f}% (fallback rounds: "
                      f"{s['fallback_rounds']}); timeline {s['timeline_seconds']['mean']:.1f}s mean, "
                      f"{s['timeline_seconds']['max']:.1f}s max.", ""]
    return "\n".join(lines)


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = None
    return {"git": rev, "torch": torch.__version__, "python": platform.python_version(), "device": "cpu",
            "args": vars(args), "host_threads": os.environ.get("OMP_NUM_THREADS")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=60)
    parser.add_argument("--first-seed", type=int, default=7000)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--players", default="bayes,top3")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="reports/v2/scenario_table.json")
    parser.add_argument("--md", default="reports/v2/scenario_table.md")
    args = parser.parse_args()
    tasks = [(p, s, args.n, args.worlds) for s in range(args.first_seed, args.first_seed + args.seeds)
             for p in args.players.split(",")]
    start = time.perf_counter()
    rows = run_all(tasks, args.workers, args.threads)
    summary = summarise(rows)
    meta = {"seeds": args.seeds, "n": args.n, "worlds": args.worlds, "seconds": time.perf_counter() - start}
    report = {"meta": meta, "provenance": provenance(args), "summary": summary, "games": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    Path(args.md).write_text(markdown(summary, meta))
    print(markdown(summary, meta))


if __name__ == "__main__":
    main()
