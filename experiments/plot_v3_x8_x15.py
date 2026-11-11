"""Figures for X8 and X15, drawn ONLY from reports/v3/x8.json and reports/v3/x15.json (plan §1.4, §14.4).

Command:
  python experiments/plot_v3_x8_x15.py [--only x8,x15]

Kept apart from plot_v3.py (not edited here); it reuses its style helpers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_v3 import (  # noqa: E402
    GRID, INK, INK2, REPORTS, SERIES, SURFACE, _style, figure_path, new_figure, save_figure,
)

LABEL = {"ORACLE": "ORACLE", "bayes": "II Bayes (PG)", "bayes_admm": "II + ADMM", "bayes_blend": "II + blend",
         "top3": "top-3", "people_pleaser": "I People Pleaser", "uniform": "uniform", "random": "random",
         "strong_arm": "III Strong-Arm", "strong_arm_random": "III random b"}
ORDER = ["ORACLE", "bayes_admm", "bayes", "top3", "bayes_blend", "people_pleaser", "uniform", "strong_arm_random",
         "random", "strong_arm"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]   # sequential ramp, light → dark


def plot_x8(d: dict, out: Path) -> list[Path]:
    paths = []
    T = d["rounds"]
    # 1. viability: EF Victory by decision round (selected players; fixed hue per player)
    shown = ["ORACLE", "bayes", "top3", "people_pleaser", "uniform", "random", "strong_arm"]
    fig, axes = new_figure(1, 2, figsize=(10, 3.8))
    for ax, key, title in ((axes[0], "EF_by_round", "EF Victory (some sampled future wins)"),
                           (axes[1], "p_best_by_round", "p_best = max over plans of P(win)")):
        _style(ax)
        for i, p in enumerate(shown):
            v = d["players"][p][key]
            ys = [x["rate"] if "rate" in x else x["mean"] for x in v]
            ax.plot(range(1, T + 1), ys, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=LABEL[p])
        ax.set_xticks(range(1, T + 1))
        ax.set_xlabel("decision round k (state before round k)", color=INK2, fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title, color=INK, fontsize=9)
    axes[0].set_ylabel("share of games", color=INK2, fontsize=8)
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"X8 viability over the sampled futures ({d['seeds']['count']} X6 seeds)", color=INK, fontsize=10)
    paths.append(save_figure(fig, figure_path(out, "x8", "viability")))
    # 2. point of no return: stacked bars over lost games
    fig, ax = new_figure(figsize=(9, 3.8))
    _style(ax)
    players = [p for p in ORDER if d["players"][p].get("losses")]
    bottoms = [0] * len(players)
    for k in range(1, T + 2):
        vals = [d["players"][p]["losses"]["pnr_hist"][str(k)] for p in players]
        lab = "lost at commit (k=1)" if k == 1 else (f"k={k}" if k <= T else f"k={k} (viable before last round)")
        ax.bar(range(len(players)), vals, bottom=bottoms, color=BLUES[k - 1], edgecolor=SURFACE, linewidth=2,
               label=lab)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, b in enumerate(bottoms):
        ax.text(i, b + 1, str(b), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels([LABEL[p] for p in players], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, max(bottoms) * 1.12)
    ax.set_ylabel("lost games", color=INK2, fontsize=8)
    ax.set_title("X8 point of no return of lost games (first round from which no sampled future wins)",
                 color=INK, fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    paths.append(save_figure(fig, figure_path(out, "x8", "point_of_no_return")))
    # 3. LTL properties, poll view (ORACLE: true view)
    fig, ax = new_figure(figsize=(10, 3.8))
    _style(ax)
    props = [("ev_lead", "◇Lead (before election)"), ("never_far", "□(margin > −0.05)"),
             ("comeback", "Comeback (share of winners)")]
    w = 0.27
    for j, (key, lab) in enumerate(props):
        xs, ys, lo, hi = [], [], [], []
        for i, p in enumerate(ORDER):
            view = d["players"][p].get("poll") or d["players"][p]["true"]
            r = view.get("winners", {}).get("comeback") if key == "comeback" else view[key]
            if r is None:
                continue
            xs.append(i + (j - 1) * w)
            ys.append(r["rate"])
            lo.append(max(0.0, r["rate"] - r["ci95"][0]))
            hi.append(max(0.0, r["ci95"][1] - r["rate"]))
        ax.bar(xs, ys, w * 0.92, color=SERIES[j], edgecolor=SURFACE, linewidth=1, label=lab)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[p] + (" (true E)" if p == "ORACLE" else "") for p in ORDER], rotation=25, ha="right",
                       fontsize=8)
    ax.set_ylabel("share of traces (95% Wilson CI)", color=INK2, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("X8 LTL properties on the public poll traces", color=INK, fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    paths.append(save_figure(fig, figure_path(out, "x8", "ltl")))
    return paths


def plot_x15(d: dict, out: Path) -> list[Path]:
    paths = []
    # 1. standardised coefficients with bootstrap CIs (three fitted models side by side)
    names = [n for n in ("oracle_loss", "people_pleaser_loss", "oracle_margin", "bayes_margin")
             if d["models"][n].get("fitted")]
    feats = d["models"][names[0]]["features"]
    fig, axes = new_figure(1, len(names), figsize=(3.1 * len(names), 5), sharey=True)
    for ax, n in zip(axes, names):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        m = d["models"][n]
        for i, f in enumerate(feats):
            c = m["coefs"][f]
            col = SERIES[0] if c["ci_excludes_0"] else INK2
            ax.plot(c["ci95"], [i, i], color=col, linewidth=2)
            ax.plot([c["coef"]], [i], "o", color=col, markersize=5)
        ax.axvline(0, color=INK2, linewidth=0.7)
        unit = "log-odds / SD" if m["kind"] == "class" else "E / SD"
        ax.set_title(f"{n}\n({unit}; CV {'AUC' if m['kind'] == 'class' else 'R²'} "
                     f"{m['cv']['logit']['mean']:.2f})", color=INK, fontsize=8)
    axes[0].set_yticks(range(len(feats)))
    axes[0].set_yticklabels(feats, fontsize=7)
    axes[0].invert_yaxis()
    fig.suptitle("X15 standardised ridge coefficients, 95% bootstrap CIs (blue: CI excludes 0)", color=INK, fontsize=10)
    paths.append(save_figure(fig, figure_path(out, "x15", "coefficients")))
    # 2. partial dependence, ORACLE loss, top 3 features (logistic and tree)
    for model in ("oracle_loss", "oracle_margin"):
        m = d["models"][model]
        top = m["top3"]
        fig, axes = new_figure(1, 3, figsize=(10, 3.2), sharey=True)
        for ax, f in zip(axes, top):
            _style(ax)
            pd = m["partial_dependence"][f]
            ax.plot(pd["grid"], pd["logit"], color=SERIES[0], linewidth=2, label="ridge-logistic" if
                    m["kind"] == "class" else "ridge-linear")
            ax.plot(pd["grid"], pd["tree"], color=SERIES[1], linewidth=2, drawstyle="steps-mid", label="depth-3 tree")
            ax.set_xlabel(f, color=INK2, fontsize=8)
        axes[0].set_ylabel("mean predicted P(loss)" if m["kind"] == "class" else "mean predicted E − 0.5",
                           color=INK2, fontsize=8)
        axes[-1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"X15 partial dependence, {model}, top-3 features", color=INK, fontsize=10)
        paths.append(save_figure(fig, figure_path(out, "x15", f"pdp_{model}")))
    # 3. best channel by the largest region's personality
    ch = d["channel_by_personality"]
    fig, ax = new_figure(figsize=(6, 3.4))
    _style(ax)
    pers = [p for p in ("P0", "P1", "P2", "P3") if ch["media_minus_workers_E_by_personality"][p]["n"]]
    ms = [ch["media_minus_workers_E_by_personality"][p] for p in pers]
    ax.bar(range(len(pers)), [m["mean"] for m in ms], 0.6, color=SERIES[0], edgecolor=SURFACE)
    ax.errorbar(range(len(pers)), [m["mean"] for m in ms],
                yerr=[[m["mean"] - m["ci95"][0] for m in ms], [m["ci95"][1] - m["mean"] for m in ms]],
                fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    for i, m in enumerate(ms):
        ax.text(i, 0, f"n={m['n']}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.axhline(0, color=INK2, linewidth=0.7)
    names_p = {"P0": "P0 social", "P1": "P1 media-savvy", "P2": "P2 independent", "P3": "P3 canvassing"}
    ax.set_xticks(range(len(pers)))
    ax.set_xticklabels([names_p[p] for p in pers], fontsize=8)
    ax.set_ylabel("E(pay_media) − E(pay_workers)", color=INK2, fontsize=8)
    ax.set_title("X15 channel gap by the largest region's personality (X0 ₫500)", color=INK, fontsize=9)
    paths.append(save_figure(fig, figure_path(out, "x15", "channel_by_personality")))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="x8,x15")
    parser.add_argument("--reports", default=str(REPORTS))
    args = parser.parse_args()
    rep = Path(args.reports)
    for exp in args.only.split(","):
        path = rep / f"{exp}.json"
        if not path.exists():
            print(f"skip {exp}: {path} missing")
            continue
        {"x8": plot_x8, "x15": plot_x15}[exp](json.loads(path.read_text()), rep / "figures")


if __name__ == "__main__":
    main()
