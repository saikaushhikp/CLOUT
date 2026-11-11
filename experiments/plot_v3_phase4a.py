"""Figures for X5 and X17 (Phase 4), drawn ONLY from the saved summaries reports/v3/x5.json and x17.json
(plan §1.4, §14.4). Kept separate from ``plot_v3.py`` (edited concurrently by other agents); reuses its style.

Command:
  python experiments/plot_v3_phase4a.py [--only x5,x17] [--reports reports/v3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_v3 import INK2, REPORTS, SERIES, _style, figure_path, new_figure, save_figure  # noqa: E402


def _ci(stat: dict) -> tuple[float, float, float]:
    if "rate" in stat:
        m, (lo, hi) = stat["rate"], stat["ci95"]
    else:
        m, (lo, hi) = stat["mean"], stat["ci95"]
    return m, m - lo, hi - m


def plot_x5(data: dict, out_dir: Path) -> list[Path]:
    out = []
    ks = data["ks"]
    arms = data["arms"]
    series = [("10.00", "gain", SERIES[0], "-", "o"), ("10.00", "random", SERIES[0], "--", "s"),
              ("5.00", "gain", SERIES[1], "-", "o"), ("5.00", "random", SERIES[1], "--", "s")]

    def name(price, k, rule):
        return f"p{int(float(price))}_k{k}_{'none' if k == 0 else rule}"

    fig, axes = new_figure(1, 2, figsize=(10, 3.8))
    for price, rule, color, ls, mk in series:
        for ax, key in zip(axes, ("expected", "win_rate")):
            vals = [_ci(arms[name(price, k, rule)][key]) for k in ks]
            off = (0.08 if price == "5.00" else -0.08) + (0.04 if rule == "random" else 0)
            ax.errorbar([i + off for i in range(len(ks))], [v[0] for v in vals],
                        yerr=[[v[1] for v in vals], [v[2] for v in vals]], color=color, ls=ls, marker=mk, ms=4,
                        capsize=2, lw=1.4, label=f"₫{int(float(price))} per query, {rule}")
    for ax, label in zip(axes, ("mean expected share E (95% CI)", "win rate (Wilson 95% CI)")):
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_xlabel("survey queries k", color=INK2)
        ax.set_ylabel(label, color=INK2)
        _style(ax)
    axes[0].axhline(0.5, color=INK2, lw=0.6, ls=":")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("Outcome vs number of survey queries", fontsize=10, loc="left")
    axes[1].set_title(f"{data['seeds']['count']} paired seeds per arm", fontsize=10, loc="left")
    out.append(save_figure(fig, figure_path(out_dir, "x5", "curve")))

    fig, axes = new_figure(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for j, price in enumerate(("10.00", "5.00")):
        curve = data["value_of_information"][f"p{int(float(price))}_gain"]["curve"]
        vals = [_ci(c["marginal_vs_prev_k"]) for c in curve]
        xs = [i + (j - 0.5) * 0.3 for i in range(len(curve))]
        ax.bar(xs, [v[0] for v in vals], 0.3, yerr=[[v[1] for v in vals], [v[2] for v in vals]], capsize=2,
               color=SERIES[j], label=f"₫{int(float(price))} per query (gain)")
    ax.set_xticks(range(len(ks) - 1), [f"{a}→{b}" for a, b in zip(ks[:-1], ks[1:])])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("step on the k grid", color=INK2)
    ax.set_ylabel("paired ΔE of the step (95% CI)", color=INK2)
    ax.set_title("Marginal value of more queries", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    for j, rule in enumerate(("gain", "random")):
        vals = [_ci(arms[name("10.00", k, rule)]["posterior"]["p_true_sigma"]) for k in ks]
        ax.errorbar(range(len(ks)), [v[0] for v in vals], yerr=[[v[1] for v in vals], [v[2] for v in vals]],
                    color=SERIES[2 + j], marker="o", ms=4, capsize=2, lw=1.4, label=f"{rule} queries")
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("survey queries k", color=INK2)
    ax.set_ylabel("posterior mass on the true σ", color=INK2)
    ax.set_title("Posterior accuracy after the survey", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    out.append(save_figure(fig, figure_path(out_dir, "x5", "marginal")))

    fig, axes = new_figure(1, 2, figsize=(10, 3.8))
    for ax, key, title in zip(axes, ("guaranteed", "possible"), ("K(Guaranteed) = ∃π∀w win", "Possible = ∃π∃w win")):
        for j, k in enumerate(ks):
            v = arms[name("10.00", k, "gain")]["verdicts_by_round"][key]
            ax.plot(range(1, len(v) + 1), [x["rate"] for x in v], marker="o", ms=4, lw=1.4,
                    color=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b", "#06203f"][j], label=f"k = {k}")
        ax.set_xticks(range(1, len(v) + 1))
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("decision point (round)", color=INK2)
        ax.set_ylabel("fraction of games", color=INK2)
        ax.set_title(f"{title} (₫10, gain)", fontsize=10, loc="left")
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    out.append(save_figure(fig, figure_path(out_dir, "x5", "verdicts")))
    return out


def plot_x17(data: dict, out_dir: Path) -> list[Path]:
    out = []
    arms = data["arms"]
    names = {"bayes": "BayesPlanner", "poll_planner": "poll-consistent planner", "probe": "probe player"}
    fig, axes = new_figure(1, 3, figsize=(12, 3.6))
    for ax, key, title, prior in zip(axes, ("H_perm", "H_soft", "H_probe_region"),
                                     ("τ entropy, hard poll filter (plan metric)", "τ entropy, Gaussian poll likelihood",
                                      "personality entropy of the probed region (hard)"),
                                     (3.178, 3.178, 1.386)):
        for j, a in enumerate(names):
            vals = [_ci(x) for x in arms[a]["tau_by_round"][key]]
            ax.errorbar([t + 1 + (j - 1) * 0.06 for t in range(len(vals))], [v[0] for v in vals],
                        yerr=[[v[1] for v in vals], [v[2] for v in vals]], color=SERIES[j], marker="o", ms=4,
                        capsize=2, lw=1.4, label=names[a])
        ax.axhline(prior, color=INK2, lw=0.6, ls=":")
        ax.set_xlabel("decision point (round)", color=INK2)
        ax.set_ylabel("entropy (nats)", color=INK2)
        ax.set_title(title, fontsize=9, loc="left")
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    out.append(save_figure(fig, figure_path(out_dir, "x17", "entropy")))

    fig, axes = new_figure(1, 3, figsize=(12, 3.6))
    for ax, key, label in zip(axes[:2], ("expected", "win_rate"), ("mean expected share E (95% CI)", "win rate (Wilson 95% CI)")):
        vals = [_ci(arms[a][key]) for a in names]
        ax.bar(range(len(names)), [v[0] for v in vals], 0.6, yerr=[[v[1] for v in vals], [v[2] for v in vals]],
               capsize=3, color=SERIES[:3])
        ax.set_xticks(range(len(names)), list(names.values()), fontsize=7)
        ax.set_ylabel(label, color=INK2)
        if key == "expected":
            ax.set_ylim(0.4, None)
        _style(ax)
    ax = axes[2]
    for j, a in enumerate(names):
        v = arms[a]["guaranteed_by_round"]
        ax.plot(range(1, len(v) + 1), [x["rate"] for x in v], marker="o", ms=4, lw=1.4, color=SERIES[j], label=names[a])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("decision point (round)", color=INK2)
    ax.set_ylabel("fraction with K(Guaranteed)", color=INK2)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    axes[0].set_title(f"Outcome ({data['seeds']['count']} paired seeds)", fontsize=10, loc="left")
    out.append(save_figure(fig, figure_path(out_dir, "x17", "outcome")))
    return out


PLOTS = {"x5": plot_x5, "x17": plot_x17}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="x5,x17")
    parser.add_argument("--reports", default=str(REPORTS))
    parser.add_argument("--figures", default=None)
    args = parser.parse_args()
    reports = Path(args.reports)
    out_dir = Path(args.figures) if args.figures else reports / "figures"
    for exp in args.only.split(","):
        path = reports / f"{exp}.json"
        if not path.exists():
            print(f"skip {exp}: {path} missing")
            continue
        PLOTS[exp](json.loads(path.read_text()), out_dir)


if __name__ == "__main__":
    main()
