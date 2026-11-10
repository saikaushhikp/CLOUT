"""Plots for the P4 strategy comparison, from the saved JSON only (no simulation).

Usage: python experiments/plot_strategies.py [--in reports/v2/strategies.json] [--out-dir reports/v2/figures] [--suffix _r12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORACLE_COLOR = "#8a8984"
SUFFIX = [""]  # file-name suffix, e.g. "_r12" (set by --suffix)
LABELS = {"uniform": "Uniform", "top3": "Top-3", "random": "Random", "bayes_even": "Bayes (even spend)",
          "bayes": "Bayes planner", "ORACLE": "ORACLE (true state)"}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(height: float):
    fig, ax = plt.subplots(figsize=(7.5, height), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def win_rates(report: dict, out: Path) -> None:
    names = report["players"] + ["ORACLE"]
    fig, ax = _figure(0.55 * len(names) + 1.2)
    for i, name in enumerate(names):
        s = report["summary"][name]
        lo, hi = s["win_ci95"]
        color = ORACLE_COLOR if name == "ORACLE" else SERIES[i % len(SERIES)]
        ax.plot([max(lo, 0) * 100, hi * 100], [i, i], color=color, linewidth=2, solid_capstyle="round")
        ax.plot([s["win_rate"] * 100], [i], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi * 100 + 1.5, i, f"{s['win_rate'] * 100:.0f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(names)), [LABELS.get(n, n) for n in names], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Win rate (official vote > N/2), 95% Wilson CI", color=MUTED)
    rules = " — R12 sampled queries" if str(report.get("query_rules", "")).startswith("R12") else ""
    ax.set_title(f"Strategy win rates — {report['num_seeds']} seeds, N = {report['num_voters']:,}{rules}", color=INK,
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"strategies_win_rates{SUFFIX[0]}.png", facecolor=SURFACE)
    plt.close(fig)


def paired_differences(report: dict, out: Path, base: str = "uniform") -> None:
    names = [n for n in report["players"] if n != base] + ["ORACLE"]
    data = []
    for name in names:
        vals = [((r["oracle"]["expected"] if name == "ORACLE" else r["runs"][name]["expected"])
                 - r["runs"][base]["expected"]) * 100 for r in report["rows"]]
        data.append(vals)
    fig, ax = _figure(0.6 * len(names) + 1.3)
    parts = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=True,
                       medianprops={"color": INK, "linewidth": 2}, whiskerprops={"color": MUTED},
                       capprops={"color": MUTED}, flierprops={"markersize": 3, "markeredgecolor": MUTED})
    for i, (box, name) in enumerate(zip(parts["boxes"], names)):
        idx = report["players"].index(name) if name in report["players"] else None
        box.set_facecolor(ORACLE_COLOR if idx is None else SERIES[idx % len(SERIES)])
        box.set_edgecolor(SURFACE)
        box.set_alpha(0.85)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(1, len(names) + 1), [LABELS.get(n, n) for n in names], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(f"Expected share minus {LABELS[base]} (percentage points, paired by seed)", color=MUTED)
    ax.set_title("Paired gain in expected vote share", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"strategies_paired_vs_uniform{SUFFIX[0]}.png", facecolor=SURFACE)
    plt.close(fig)


def spend(report: dict, out: Path) -> None:
    names = report["players"]
    cats = [("workers", "Workers"), ("media", "Media"), ("queries", "Queries")]
    fig, ax = _figure(0.55 * len(names) + 1.5)
    for i, name in enumerate(names):
        left = 0.0
        for j, (key, _) in enumerate(cats):
            value = report["summary"][name]["mean_spend"][key]
            if value > 0:
                ax.barh(i, value, left=left, height=0.55, color=SERIES[j], edgecolor=SURFACE, linewidth=2)
            left += value
        ax.text(left + 5, i, f"{left:.0f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(names)), [LABELS.get(n, n) for n in names], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 560)
    ax.set_xlabel("Mean money spent per game (treasury 500)", color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[j]) for j in range(len(cats))]
    ax.legend(handles, [c[1] for c in cats], frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, labelcolor=INK, fontsize=9)
    ax.set_title("Spend breakdown", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"strategies_spend{SUFFIX[0]}.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="path", default="reports/v2/strategies.json")
    parser.add_argument("--out-dir", default="reports/v2/figures")
    parser.add_argument("--suffix", default="", help="appended to every figure file name, e.g. _r12")
    args = parser.parse_args()
    SUFFIX[0] = args.suffix
    report = json.loads(Path(args.path).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    win_rates(report, out)
    if "uniform" in report["players"]:
        paired_differences(report, out)
    spend(report, out)
    print(f"saved figures to {out}")


if __name__ == "__main__":
    main()
