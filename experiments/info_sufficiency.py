"""P5 information sufficiency (spec §10.2 "Information sufficiency"; R13, R14).

How much information makes victory *known*? A BayesPlanner is forced to ask exactly k survey queries
(k ∈ {0, 1, 2, 4, 8}; every stop rule disabled) chosen either by information gain ("gain") or uniformly at
random ("random"), and asks nothing during the rounds — so after the survey its knowledge grows from the
free polls (and statistics) only. For each finished game the epistemic timeline is computed and we report,
per (policy, k): the fraction of seeds where guaranteed_victory (∃π∀w — a plan known to win) holds at
round 1 and at each round t, the same for possible / knows-that, the posterior entropy over σ at round 1,
the actual win rate and P(win | guaranteed at round 1) vs P(win | not guaranteed).

Usage:
  python experiments/info_sufficiency.py --seeds 40 --n 10000 --workers 4
Outputs reports/v2/info_sufficiency.json and reports/v2/figures/info_sufficiency.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenario_table import provenance, run_all, wilson  # noqa: E402

KS = (0, 1, 2, 4, 8)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for policy in ("gain", "random"):
        out[policy] = {}
        for k in sorted({int(r["player"].split("-")[1].split(":")[0]) for r in rows}):
            sub = [r for r in rows if r["player"] == f"forced-{k}:{policy}"]
            if not sub:
                continue
            T = len(sub[0]["timeline"]["rounds"])
            wins = [r["timeline"]["actual"]["victory"] for r in sub]
            g1 = [r["timeline"]["rounds"][0]["guaranteed_victory"] for r in sub]

            def frac(key: str, t: int) -> float:
                return sum(r["timeline"]["rounds"][t]["guaranteed_victory" if key == "g" else key] for r in sub) / len(sub)

            def cond(c: list[bool]) -> dict[str, Any]:
                sel = [w for w, x in zip(wins, c) if x]
                return {"n": len(sel), "p": sum(sel) / len(sel) if sel else None}

            out[policy][k] = {
                "games": len(sub),
                "win_rate": sum(wins) / len(sub), "win_rate_ci": wilson(sum(wins), len(sub)),
                "guaranteed_by_round": [frac("g", t) for t in range(T)],
                "possible_by_round": [frac("possible_victory", t) for t in range(T)],
                "knows_that_by_round": [frac("knows_win_exists", t) for t in range(T)],
                "truth_possible_by_round": [frac("truth_victory_possible", t) for t in range(T)],
                "guaranteed_round1": frac("g", 0), "guaranteed_round1_ci": wilson(sum(g1), len(sub)),
                "guaranteed_last_round": frac("g", T - 1),
                "entropy_round1": sum(r["timeline"]["rounds"][0]["knowledge"]["posterior_entropy"] for r in sub) / len(sub),
                "best_win_fraction_round1": sum(r["timeline"]["rounds"][0]["best_plan"]["win_fraction"] for r in sub) / len(sub),
                "P(win | guaranteed at round 1)": cond(g1),
                "P(win | not guaranteed at round 1)": cond([not x for x in g1]),
                "P(win | guaranteed at last round)": cond([r["timeline"]["rounds"][-1]["guaranteed_victory"] for r in sub]),
                "mean_acceptance": sum(sum(r["acceptance"]) / len(r["acceptance"]) for r in sub) / len(sub),
            }
    return out


def plot(summary: dict[str, Any], path: str, meta: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]          # sequential blue: k small → large
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=surface)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted)
        ax.set_ylim(-0.02, 1.02)

    ax = axes[0]
    series = [("gain", "guaranteed_round1", blue, "-", "o", "Guaranteed at round 1 (info-gain queries)"),
              ("random", "guaranteed_round1", orange, "--", "s", "Guaranteed at round 1 (random queries)"),
              ("gain", "win_rate", aqua, "-", "^", "Actual win rate (info-gain queries)")]
    for policy, key, color, style, marker, label in series:
        ks = sorted(summary[policy], key=int)
        ax.plot([int(k) for k in ks], [summary[policy][k][key] for k in ks], style, color=color, linewidth=2,
                marker=marker, markersize=8, markeredgecolor=surface, markeredgewidth=2, label=label)
    ax.set_xticks(sorted({int(k) for k in summary["gain"]}))
    ax.set_xlabel("Survey queries asked (k)", color=ink)
    ax.set_ylabel("Fraction of seeds", color=ink)
    ax.set_title("Knowing a winning plan vs survey effort", color=ink, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="best")

    ax = axes[1]
    ks = sorted(summary["gain"], key=int)
    for color, k in zip(ramp, ks):
        ys = summary["gain"][k]["guaranteed_by_round"]
        ax.plot(range(1, len(ys) + 1), ys, "-", color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=surface, markeredgewidth=2, label=f"k = {k}")
    ax.set_xticks(range(1, len(summary["gain"][ks[0]]["guaranteed_by_round"]) + 1))
    ax.set_xlabel("Round t (knowledge from polls only after the survey)", color=ink)
    ax.set_title("Guaranteed victory (∃π∀w) by round, info-gain queries", color=ink, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, title="survey queries", title_fontsize=8)
    fig.suptitle(f"Information sufficiency — {meta['seeds']} seeds per arm, N = {meta['n']}, "
                 f"{meta['worlds']} sampled worlds per round", color=muted, fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--first-seed", type=int, default=9000)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--ks", default=",".join(map(str, KS)))
    parser.add_argument("--policies", default="gain,random")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="reports/v2/info_sufficiency.json")
    parser.add_argument("--fig", default="reports/v2/figures/info_sufficiency.png")
    args = parser.parse_args()
    ks = [int(k) for k in args.ks.split(",")]
    tasks = [(f"forced-{k}:{p}", s, args.n, args.worlds) for s in range(args.first_seed, args.first_seed + args.seeds)
             for p in args.policies.split(",") for k in ks]
    start = time.perf_counter()
    rows = run_all(tasks, args.workers, args.threads)
    summary = summarise(rows)
    meta = {"seeds": args.seeds, "n": args.n, "worlds": args.worlds, "ks": ks, "seconds": time.perf_counter() - start}
    games = [{k: r[k] for k in ("player", "seed", "queries", "acceptance", "timeline_seconds")}
             | {"rounds": [{p: x[p] for p in ("round", "possible_victory", "guaranteed_victory", "knows_win_exists",
                                               "truth_victory_possible")} | {"entropy": x["knowledge"]["posterior_entropy"]}
                           for x in r["timeline"]["rounds"]],
                "victory": r["timeline"]["actual"]["victory"]} for r in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "provenance": provenance(args), "summary": summary,
                                          "games": games}, indent=1))
    plot(summary, args.fig, meta)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
