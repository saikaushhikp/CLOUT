"""Figures for the v3 experiment suite, drawn ONLY from the saved summaries ``reports/v3/<id>.json`` (plan §1.4,
§14.4). Each experiment registers one function ``plot_x<k>(data, out_dir) -> list[Path]``; experiments without
a saved JSON (or without a plot function yet) are skipped with a message.

Command:
  python experiments/plot_v3.py [--only x0,x10] [--reports reports/v3]

Phase 0: scaffolding only. The plot functions are added experiment by experiment as each one finishes.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports" / "v3"

EXPERIMENTS = [f"x{k}" for k in range(24)]   # X24 dropped (D8), X25 deferred (D1)
PlotFn = Callable[[dict, Path], list[Path]]
PLOTS: dict[str, PlotFn] = {}


def register(exp_id: str) -> Callable[[PlotFn], PlotFn]:
    def wrap(fn: PlotFn) -> PlotFn:
        PLOTS[exp_id] = fn
        return fn
    return wrap


def figure_path(out_dir: Path, exp_id: str, name: str) -> Path:
    """``reports/v3/figures/<id>_<name>.png`` (§1.4 naming)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{exp_id}_{name}.png"


def new_figure(*args, **kwargs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(*args, **kwargs)


def save_figure(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return path


# --------------------------------------------------------------------------- one function per experiment
# Reference palette (dataviz skill, light mode): fixed categorical order, recessive ink, diverging blue/orange.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#f0efec"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("media_workers", ["#eb6834", NEUTRAL, "#2a78d6"])


X0_RUN_ORDER = ["uniform/none", "uniform/planner", "top3/planner", "informed/planner", "oracle/none", "oracle/planner",
                "informed/free_even", "informed/pay_workers", "informed/front_workers", "informed/pay_media",
                "informed/pay_split"]


@register("x0")
def plot_x0(data: dict, out_dir: Path) -> list[Path]:
    """Win rates (Wilson CIs) per run and treasury; gate pass/fail grid; best paid pattern counts."""
    treasuries = data["treasuries"]
    paths = []
    fig, ax = new_figure(figsize=(10, 4.2))
    _style(ax)
    width = 0.8 / len(treasuries)
    for j, t in enumerate(treasuries):
        summ = data["per_treasury"][t]["summary"]
        xs = [i + (j - (len(treasuries) - 1) / 2) * width for i in range(len(X0_RUN_ORDER))]
        rates = [summ[k]["rate"] for k in X0_RUN_ORDER]
        lo = [max(0.0, summ[k]["rate"] - summ[k]["ci95"][0]) for k in X0_RUN_ORDER]
        hi = [max(0.0, summ[k]["ci95"][1] - summ[k]["rate"]) for k in X0_RUN_ORDER]
        ax.bar(xs, rates, width * 0.9, color=SERIES[j], label=f"treasury ₫{t}", edgecolor=SURFACE, linewidth=1)
        ax.errorbar(xs, rates, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
    for y, txt in ((0.30, "gate 1: uniform/none ≤ 30%"), (0.80, "gate 4a: paid pattern < 80%"),
                   (0.85, "gate 2: oracle/planner ≥ 85%")):
        ax.axhline(y, color=INK2, linewidth=0.7, linestyle=":")
        ax.text(len(X0_RUN_ORDER) - 0.5, y, txt, fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(len(X0_RUN_ORDER)))
    ax.set_xticklabels(X0_RUN_ORDER, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("win rate (95% Wilson CI)", color=INK2)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(f"X0 win rates, N={data['n']}, T={data['rounds']}, {len(data['seeds'])} seeds", color=INK, fontsize=10)
    paths.append(save_figure(fig, figure_path(out_dir, "x0", "win_rates")))

    crit = data["gate_criteria"]
    fig, ax = new_figure(figsize=(7, 2.6))
    ax.set_facecolor(SURFACE)
    for i, c in enumerate(crit):
        for j, t in enumerate(treasuries):
            ok = data["gate_by_treasury"][t][c]
            ax.add_patch(__import__("matplotlib").patches.FancyBboxPatch((j + 0.06, i + 0.08), 0.88, 0.84,
                         boxstyle="round,pad=0,rounding_size=0.08", color=GOOD if ok else CRITICAL, alpha=0.85))
            ax.text(j + 0.5, i + 0.5, "✓ pass" if ok else "✗ fail", ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(0, len(treasuries))
    ax.set_ylim(len(crit), 0)
    ax.set_xticks([j + 0.5 for j in range(len(treasuries))])
    ax.set_xticklabels([f"₫{t}" for t in treasuries], fontsize=8, color=INK2)
    ax.set_yticks([i + 0.5 for i in range(len(crit))])
    ax.set_yticklabels(crit, fontsize=8, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("X0 balance gate (spec §12) per treasury", color=INK, fontsize=10)
    paths.append(save_figure(fig, figure_path(out_dir, "x0", "gate")))

    patterns = ["pay_workers", "front_workers", "pay_media", "pay_split"]
    fig, ax = new_figure(figsize=(6, 3))
    _style(ax)
    left = [0] * len(treasuries)
    for k, p in enumerate(patterns):
        vals = [data["per_treasury"][t]["best_paid_pattern_counts"][p] for t in treasuries]
        ax.barh([f"₫{t}" for t in treasuries], vals, left=left, color=SERIES[k], label=p, edgecolor=SURFACE, linewidth=2)
        for j, v in enumerate(vals):
            if v >= 6:
                ax.text(left[j] + v / 2, j, str(v), ha="center", va="center", color="white", fontsize=8)
        left = [a + b for a, b in zip(left, vals)]
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("seeds where the pattern is the best paid pattern (informed manifesto)", color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.set_title("X0 best paid pattern per seed", color=INK, fontsize=10)
    paths.append(save_figure(fig, figure_path(out_dir, "x0", "best_pattern")))
    return paths


def _grid(data: dict):
    ns, ts = data["ns"], data["rounds"]
    return ns, ts, [f"{n:,}" for n in ns]


@register("x10")
def plot_x10(data: dict, out_dir: Path) -> list[Path]:
    """Gate map over (N, T); channel-dominance map; log–log scaling of the channel effects."""
    import math
    ns, ts, nlab = _grid(data)
    crit = ["1_uniform_no_campaign_le_30pct", "2_oracle_planner_ge_85pct", "3_informed_beats_top3_p_lt_0.01",
            "4a_no_paid_pattern_ge_80pct", "4b_workers_and_media_each_best_sometimes"]
    short = ["1", "2", "3", "4a", "4b"]
    paths = []
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = new_figure(figsize=(6.4, 4))
    passed = [[sum(data["gate_map"][f"{n}|{T}"][c] for c in crit) for n in ns] for T in ts]
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#104281"])
    ax.imshow(passed, cmap=cmap, vmin=0, vmax=5, aspect="auto")
    for i, T in enumerate(ts):
        for j, n in enumerate(ns):
            g = data["gate_map"][f"{n}|{T}"]
            failed = [s for s, c in zip(short, crit) if not g[c]]
            label = "PASS" if not failed else "fail: " + ",".join(failed)
            ax.text(j, i, f"{passed[i][j]}/5\n{label}", ha="center", va="center", fontsize=7,
                    color="white" if passed[i][j] >= 3 else INK)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(nlab, fontsize=8)
    ax.set_yticks(range(len(ts)))
    ax.set_yticklabels([f"T={T}" for T in ts], fontsize=8)
    ax.set_xlabel("N (voters)", color=INK2)
    ax.set_title(f"X10 gate criteria passed (of 5), treasury ₫{data['treasury']}", color=INK, fontsize=10)
    paths.append(save_figure(fig, figure_path(out_dir, "x10", "gate_map")))

    fig, ax = new_figure(figsize=(6.4, 4))
    rates = [[data["channel_dominance"][f"{n}|{T}"]["workers_better_rate"] for n in ns] for T in ts]
    im = ax.imshow(rates, cmap=_diverging(), vmin=0, vmax=1, aspect="auto")
    for i, T in enumerate(ts):
        for j, n in enumerate(ns):
            d = data["channel_dominance"][f"{n}|{T}"]
            ax.text(j, i, f"{rates[i][j]:.0%}\nbest: {d['most_often_best_paid_pattern'].replace('pay_', '')}",
                    ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(nlab, fontsize=8)
    ax.set_yticks(range(len(ts)))
    ax.set_yticklabels([f"T={T}" for T in ts], fontsize=8)
    ax.set_xlabel("N (voters)", color=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("share of seeds: workers-only beats media-only", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("X10 channel dominance (blue = workers, orange = media)", color=INK, fontsize=10)
    paths.append(save_figure(fig, figure_path(out_dir, "x10", "channel_dominance")))

    metrics = [("dshare_per_worker", "Δshare per worker"), ("converts_per_worker", "cascade converts per worker"),
               ("dshare_per_rupee_media", "media Δshare per ₫")]
    fig, axes = new_figure(1, 3, figsize=(12, 3.8))
    for ax, (m, label) in zip(axes, metrics):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for k, T in enumerate(ts):
            pts = [(n, data["cells"][f"{n}|{T}"]["channels"][m]) for n in ns]
            pts = [(n, c) for n, c in pts if c["mean"] is not None and not math.isnan(c["mean"]) and c["mean"] > 0]
            if not pts:
                continue
            xs = [n for n, _ in pts]
            ys = [c["mean"] for _, c in pts]
            lo = [max(c["mean"] - c["ci95"][0], 0) for _, c in pts]
            hi = [max(0.0, c["ci95"][1] - c["mean"]) for _, c in pts]
            sl = data["slopes"][m][str(T)]
            lab = f"T={T}" + (f" (slope {sl['slope']:.2f})" if sl.get("slope") is not None else "")
            ax.errorbar(xs, ys, yerr=[lo, hi], color=SERIES[k], marker="o", markersize=4, linewidth=2, capsize=2,
                        label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (voters)", color=INK2)
        ax.set_title(label, color=INK, fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("X10 channel effects vs N (cell means, 95% t-CI; log–log slope per T)", color=INK, fontsize=10)
    paths.append(save_figure(fig, figure_path(out_dir, "x10", "scaling")))
    return paths


# =========================================================================== X7 and X23 (Phase 1; separate block)
# Fixed categorical order (dataviz reference palette, light): blue, orange, aqua, yellow.
_X_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
_X_INK, _X_MUTED = "#0b0b0b", "#52514e"


def _x_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_X_MUTED)
    ax.tick_params(colors=_X_MUTED, labelsize=8)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


@register("x7")
def plot_x7(data: dict, out_dir: Path) -> list[Path]:
    treasuries, fees = data["grid"]["treasury"], data["grid"]["worker_fee"]
    cells = data["cells"]
    fee_x = [int(float(w)) for w in fees]

    def key(t, w, q):
        return f"T{int(float(t))}_W{int(float(w))}_Q{'any' if q is None else int(float(q))}"

    series = [("ORACLE planner", "ORACLE", None), ("BayesPlanner, query ₫5", "BayesPlanner", "5.00"),
              ("BayesPlanner, query ₫10", "BayesPlanner", "10.00")]
    out = []
    # 1. win rate vs worker fee, one panel per treasury
    fig, axes = new_figure(1, len(treasuries), figsize=(11, 3.6), sharey=True)
    for ax, t in zip(axes, treasuries):
        for (label, player, q), color in zip(series, _X_SERIES):
            win = [cells[player][key(t, w, q)]["win"] for w in fees]
            y = [x["rate"] for x in win]
            err = [[x["rate"] - x["ci95"][0] for x in win], [x["ci95"][1] - x["rate"] for x in win]]
            ax.errorbar(fee_x, y, yerr=err, color=color, marker="o", markersize=5, linewidth=2, capsize=3, label=label)
        ax.set_xscale("log")
        ax.set_xticks(fee_x, [f"₫{x}" for x in fee_x])
        ax.minorticks_off()
        ax.set_title(f"Treasury ₫{int(float(t))}", fontsize=10, color=_X_INK)
        ax.set_xlabel("Worker fee", fontsize=9, color=_X_MUTED)
        _x_axes(ax)
    axes[0].set_ylabel("Win rate (95% Wilson CI)", fontsize=9, color=_X_MUTED)
    axes[0].set_ylim(0, 1.05)
    axes[-1].legend(fontsize=8, frameon=False, loc="lower left")
    out.append(save_figure(fig, figure_path(out_dir, "x7", "win_rates")))

    # 2. spend mix (workers / media / surveys) per worker fee, ORACLE and BayesPlanner (query ₫10)
    fig, axes = new_figure(2, len(treasuries), figsize=(11, 5.6), sharey=True)
    for row, (label, player, q) in enumerate([series[0], series[2]]):
        for ax, t in zip(axes[row], treasuries):
            mixes = [cells[player][key(t, w, q)]["spend"] for w in fees]
            bottom = [0.0] * len(fees)
            xs = range(len(fees))
            for part, color in zip(("workers", "media", "queries"), _X_SERIES):
                vals = [m[part]["mean"] for m in mixes]
                ax.bar(xs, vals, bottom=bottom, color=color, width=0.6, edgecolor="white", linewidth=1.5,
                       label={"queries": "surveys"}.get(part, part))
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xticks(list(xs), [f"₫{x}" for x in fee_x])
            ax.set_title(f"{label}, ₫{int(float(t))}", fontsize=9, color=_X_INK)
            _x_axes(ax)
        axes[row][0].set_ylabel("Mean spend (₫)", fontsize=9, color=_X_MUTED)
    for ax in axes[1]:
        ax.set_xlabel("Worker fee", fontsize=9, color=_X_MUTED)
    axes[0][-1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    out.append(save_figure(fig, figure_path(out_dir, "x7", "spend_mix")))

    # 3. pay_workers − pay_media (expected share) vs worker fee, per treasury
    tests = data["tests"]["A_workers_vs_media"]["tests"]
    fig, ax = new_figure(figsize=(6, 3.8))
    for t, color in zip(treasuries, _X_SERIES):
        rows = [x for x in tests if x["treasury"] == t]
        rows.sort(key=lambda x: float(x["worker_fee"]))
        y = [x["mean"] for x in rows]
        err = [[x["mean"] - x["ci95"][0] for x in rows], [x["ci95"][1] - x["mean"] for x in rows]]
        ax.errorbar(fee_x, y, yerr=err, color=color, marker="o", markersize=5, linewidth=2, capsize=3,
                    label=f"₫{int(float(t))}")
    ax.axhline(0, color=_X_MUTED, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(fee_x, [f"₫{x}" for x in fee_x])
    ax.minorticks_off()
    ax.set_xlabel("Worker fee", fontsize=9, color=_X_MUTED)
    ax.set_ylabel("E[share]: all-workers − all-media", fontsize=9, color=_X_MUTED)
    ax.legend(title="Treasury", fontsize=8, title_fontsize=8, frameon=False)
    _x_axes(ax)
    out.append(save_figure(fig, figure_path(out_dir, "x7", "workers_vs_media")))
    return out


@register("x23")
def plot_x23(data: dict, out_dir: Path) -> list[Path]:
    arms, families = data["arms"], data["families"]
    labels = {"cap1": "cap 1", "cap2": "cap 2", "default": "default", "cap5": "cap 5", "cap1000": "cap ∞",
              "min_degree1": "deg ≥ 1", "min_degree10": "deg ≥ 10", "rule_gamma": "γ", "rule_gamma_p2": "γP²"}
    fam_default = {"attempt_cap": "cap 3", "min_degree": "deg ≥ 3", "conversion_rule": "γP"}
    out = []

    def panels(metric_fn, ylabel, name, runs, ylim=None):
        fig, axes = new_figure(1, len(families), figsize=(11, 3.6), sharey=True,
                               gridspec_kw={"width_ratios": [len(v) for v in families.values()]})
        width = 0.8 / len(runs)
        for ax, (fam, members) in zip(axes, families.items()):
            for j, ((run, rlabel), color) in enumerate(zip(runs, _X_SERIES)):
                xs, ys, lo, hi = [], [], [], []
                for i, arm in enumerate(members):
                    v = metric_fn(arms[arm]["runs"][run])
                    if v is None:
                        continue
                    xs.append(i - 0.4 + width * (j + 0.5))
                    ys.append(v[0])
                    lo.append(v[0] - v[1])
                    hi.append(v[2] - v[0])
                ax.bar(xs, ys, width=width * 0.9, color=color, label=rlabel,
                       yerr=[lo, hi] if any(lo) or any(hi) else None, capsize=2, ecolor=_X_MUTED)
            ax.set_xticks(range(len(members)),
                          [fam_default[fam] if a == "default" else labels[a] for a in members])
            ax.set_title(fam.replace("_", " "), fontsize=10, color=_X_INK)
            _x_axes(ax)
            if ylim:
                ax.set_ylim(*ylim)
        axes[0].set_ylabel(ylabel, fontsize=9, color=_X_MUTED)
        axes[-1].legend(fontsize=8, frameon=False)
        return save_figure(fig, figure_path(out_dir, "x23", name))

    worker_runs = [("oracle/planner", "ORACLE planner"), ("oracle/pay_workers", "pay_workers"),
                   ("oracle/front_workers", "front_workers"), ("oracle/pay_split", "pay_split")]

    def mean_ci(field):
        def fn(r):
            x = r.get(field)
            return None if x is None else (x["mean"], x["ci95"][0], x["ci95"][1])
        return fn

    out.append(panels(mean_ci("reach_per_worker"), "Distinct converts per worker (95% CI)", "reach", worker_runs))
    out.append(panels(lambda r: (r["win"]["rate"], *r["win"]["ci95"]), "Win rate (95% Wilson CI)", "win_rates",
                      [("oracle/planner", "ORACLE planner"), ("oracle/pay_split", "pay_split"),
                       ("oracle/pay_workers", "pay_workers"), ("oracle/pay_media", "pay_media")], ylim=(0, 1.05)))
    out.append(panels(lambda r: None if r["cap_bind_share_pooled"] is None else
                      (r["cap_bind_share_pooled"],) * 3, "Targeted voters at the cap (share)", "cap_binding",
                      worker_runs[:2], ylim=(0, 1.0)))
    return out
# =========================================================================== end X7 / X23 block


# =========================================================================== Phase 3: X6, X12, X13, X14, X19 (separate block)
# Same reference palette as above (fixed categorical order); every figure is drawn from the saved summary JSON only.
_P3_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")
_P3_INK, _P3_MUTED, _P3_GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _p3_axes(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_P3_MUTED)
    ax.tick_params(colors=_P3_MUTED, labelsize=8)
    ax.grid(axis=grid_axis, color=_P3_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _p3_err(stat: dict) -> tuple[float, float, float]:
    m = stat["mean"]
    lo, hi = stat["ci95"]
    return m, m - lo, hi - m


@register("x13")
def plot_x13(data: dict, out_dir: Path) -> list[Path]:
    paths = []
    gaps = sorted(data["gap_by_seed"].values())
    fig, ax = new_figure(figsize=(6.0, 3.4))
    ax.hist(gaps, bins=20, color=_P3_SERIES[0], edgecolor="white")
    m = data["gap_search_minus_oracle"]["mean"]
    ax.axvline(m, color=_P3_INK, linewidth=1.0)
    ax.axvline(0.01, color=_P3_SERIES[1], linewidth=1.0, linestyle="--")
    ax.text(m, ax.get_ylim()[1] * 0.92, f" mean {m:+.3f}", fontsize=8, color=_P3_INK)
    ax.text(0.01, ax.get_ylim()[1] * 0.80, " predicted ≤ 0.01", fontsize=8, color=_P3_SERIES[1])
    ax.set_xlabel("E(search) − E(ORACLE planner), mean of 20 fresh streams", fontsize=9)
    ax.set_ylabel("seeds", fontsize=9)
    ax.set_title(f"X13: open-loop plan search beats the ORACLE on {data['gap_search_minus_oracle']['a_better']}/"
                 f"{data['n_seeds']} seeds", fontsize=10, loc="left")
    ax.set_xlim(min(0.0, min(gaps)), max(gaps) * 1.1)
    _p3_axes(ax)
    paths.append(save_figure(fig, figure_path(out_dir, "x13", "gap")))
    feats = [("media_spend_share", "media share of spend"), ("workers_round1_share", "workers bought in round 1"),
             ("worker_region_hhi", "worker focus (HHI)")]
    counts = [("workers", "workers bought"), ("airings", "media airings")]
    fig, axes = new_figure(1, 2, figsize=(8.0, 3.2), gridspec_kw={"width_ratios": [3, 2]})
    for ax, group in zip(axes, (feats, counts)):
        x = range(len(group))
        for k, (who, color, off) in enumerate((("search", _P3_SERIES[0], -0.18), ("oracle", _P3_SERIES[1], 0.18))):
            vals = [_p3_err(data["features"][f][who]) for f, _ in group]
            ax.bar([i + off for i in x], [v[0] for v in vals], width=0.36, color=color,
                   yerr=[[v[1] for v in vals], [v[2] for v in vals]], error_kw={"elinewidth": 0.8, "ecolor": _P3_MUTED},
                   label="plan search" if who == "search" else "ORACLE planner")
        ax.set_xticks(list(x))
        ax.set_xticklabels([lbl for _, lbl in group], fontsize=8)
        _p3_axes(ax)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title("X13: the better plans are media-first and front-loaded", fontsize=10, loc="left")
    paths.append(save_figure(fig, figure_path(out_dir, "x13", "features")))
    return paths


_P3_X6_ORDER = ["ORACLE", "bayes", "bayes_admm", "bayes_blend", "people_pleaser", "top3", "strong_arm_random",
                "uniform", "random", "strong_arm"]
_P3_X6_LABEL = {"ORACLE": "ORACLE (ref.)", "bayes": "II BayesPlanner (PG)", "bayes_admm": "II + ADMM b",
                "bayes_blend": "II + blend b", "people_pleaser": "I People Pleaser", "top3": "top-3",
                "strong_arm_random": "III Strong-Arm, random b", "uniform": "uniform", "random": "random",
                "strong_arm": "III Strong-Arm, fixed b"}


@register("x6")
def plot_x6(data: dict, out_dir: Path) -> list[Path]:
    paths = []
    players = data["players"]
    order = [p for p in _P3_X6_ORDER if p in players]
    fig, axes = new_figure(1, 2, figsize=(9.0, 3.8), sharey=True)
    y = list(range(len(order)))[::-1]
    for ax, key, title in ((axes[0], "expected", "mean expected share E (95% CI)"),
                           (axes[1], "win", "win rate (95% Wilson CI)")):
        for yi, p in zip(y, order):
            st = players[p][key]
            m = st["mean"] if key == "expected" else st["rate"]
            lo, hi = st["ci95"]
            color = _P3_SERIES[1] if p == "ORACLE" else _P3_SERIES[0]
            ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt="o", color=color, ms=4, elinewidth=1)
        if key == "expected" and "x13_bound" in data:
            b = data["x13_bound"]["E20_search"]["mean"]
            ax.axvline(b, color=_P3_SERIES[2], linestyle="--", linewidth=1)
            ax.text(b, len(order) - 0.6, " X13 bound", fontsize=8, color=_P3_SERIES[2])
        if key == "expected":
            ax.axvline(0.5, color=_P3_MUTED, linewidth=0.6)
        ax.set_title(title, fontsize=9, loc="left")
        _p3_axes(ax, "x")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_P3_X6_LABEL.get(p, p) for p in order], fontsize=8)
    fig.suptitle("X6: strategy tournament, 100 paired seeds", fontsize=10, x=0.02, ha="left")
    paths.append(save_figure(fig, figure_path(out_dir, "x6", "tournament")))
    admm = data.get("admm_residuals") or {}
    if admm.get("mean_trace"):
        fig, ax = new_figure(figsize=(6.0, 3.2))
        tr = admm["mean_trace"]
        ax.semilogy(range(1, len(tr["primal"]) + 1), tr["primal"], color=_P3_SERIES[0], label="primal ‖x − 1⊗z‖")
        ax.semilogy(range(1, len(tr["dual"]) + 1), tr["dual"], color=_P3_SERIES[1], label="dual ρ√R‖z − z_prev‖")
        ax.set_xlabel("ADMM iteration", fontsize=9)
        ax.set_ylabel("residual (mean over solves)", fontsize=9)
        ax.set_title(f"X6: consensus-ADMM residuals ({admm['solves']} manifesto solves)", fontsize=10, loc="left")
        ax.legend(fontsize=8, frameon=False)
        _p3_axes(ax)
        paths.append(save_figure(fig, figure_path(out_dir, "x6", "admm_residuals")))
    return paths


@register("x12")
def plot_x12(data: dict, out_dir: Path) -> list[Path]:
    paths = []
    if "oracle_arm_best_coalition_by_lambda" in data:
        lams = [float(x) for x in data["oracle_arm_best_coalition_by_lambda"]]
        fig, axes = new_figure(1, 2, figsize=(9.0, 3.4))
        ax = axes[0]
        for (key, label, color) in (("search_arm_best_coalition_by_lambda", "X13 plan search", _P3_SERIES[0]),
                                    ("oracle_arm_best_coalition_by_lambda", "ORACLE planner", _P3_SERIES[1])):
            vals = [_p3_err(data[key][str(l)]["E"]) for l in lams]
            ax.errorbar(lams, [v[0] for v in vals], yerr=[[v[1] for v in vals], [v[2] for v in vals]], marker="o",
                        ms=4, color=color, label=label, capsize=2)
        allc = [data["oracle_arm_all_coalitions_by_lambda"][str(l)]["mean"] for l in lams[1:]]
        ax.plot(lams[1:], allc, linestyle="--", color=_P3_SERIES[1], alpha=0.6, label="ORACLE planner, all coalitions")
        ax.set_xlabel("λ  (0 = ORACLE manifesto b₀, 1 = coalition blend b_C)", fontsize=9)
        ax.set_ylabel("mean E (20 fresh streams)", fontsize=9)
        ax.set_title("E along b(λ), best coalition per seed", fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False)
        _p3_axes(ax)
        ax = axes[1]
        sp = data["spectra_and_reach_by_lambda"]
        vals = [_p3_err(sp[str(l)]["lambda_max_coalition_max"]) for l in lams]
        ax.errorbar(lams, [v[0] for v in vals], yerr=[[v[1] for v in vals], [v[2] for v in vals]], marker="o", ms=4,
                    color=_P3_SERIES[2], capsize=2, label="max over coalition regions")
        vals = [_p3_err(sp[str(l)]["lambda_max_all_max"]) for l in lams]
        ax.errorbar(lams, [v[0] for v in vals], yerr=[[v[1] for v in vals], [v[2] for v in vals]], marker="s", ms=4,
                    color=_P3_SERIES[3], capsize=2, label="max over all regions")
        ax.axhline(1.0, color=_P3_MUTED, linewidth=0.8, linestyle=":")
        ax.set_xlabel("λ", fontsize=9)
        ax.set_ylabel("λ_max(A_rr D_q)", fontsize=9)
        ax.set_title("cascade branching factor per region", fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False)
        _p3_axes(ax)
        fig.suptitle("X12: narrowing the manifesto towards a coalition", fontsize=10, x=0.02, ha="left")
        paths.append(save_figure(fig, figure_path(out_dir, "x12", "lambda")))
    if "joint" in data:
        j = data["joint"]
        fig, ax = new_figure(figsize=(5.6, 3.2))
        bars = [("ORACLE planner", j["oracle_planner"]["joint_minus_b0"])]
        if "search_planner" in j:
            bars.append(("X13 plan search", j["search_planner"]["joint_minus_b0"]))
        bars.append(("static E (no campaign)", j["static_E"]["joint_minus_b0"]))
        for i, (label, st) in enumerate(bars):
            m, lo, hi = _p3_err(st)
            ax.bar(i, m, color=_P3_SERIES[i], yerr=[[lo], [hi]], error_kw={"elinewidth": 0.8, "ecolor": _P3_MUTED})
        ax.axhspan(0.005, 0.02, color=_P3_SERIES[3], alpha=0.15, label="predicted +0.005 to +0.02")
        ax.axhline(0, color=_P3_MUTED, linewidth=0.6)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels([b[0] for b in bars], fontsize=8)
        ax.set_ylabel("E(b_joint) − E(b₀)", fontsize=9)
        ax.set_title("X12: campaign-aware manifesto (3 alternations)", fontsize=10, loc="left")
        ax.legend(fontsize=7, frameon=False)
        _p3_axes(ax)
        paths.append(save_figure(fig, figure_path(out_dir, "x12", "joint")))
    return paths


@register("x14")
def plot_x14(data: dict, out_dir: Path) -> list[Path]:
    paths = []
    ns = [k for k in data if k.startswith("N") and isinstance(data[k], dict) and "arms" in data[k]]
    if not ns:
        return paths
    fig, axes = new_figure(1, len(ns), figsize=(4.6 * len(ns), 3.8), sharey=True, squeeze=False)
    for ax, nk in zip(axes[0], ns):
        arms = data[nk]["arms"]
        names = list(arms)
        y = list(range(len(names)))[::-1]
        for yi, a in zip(y, names):
            st = arms[a]["round1"]["bias"]
            m, lo, hi = _p3_err(st)
            ax.errorbar(m, yi, xerr=[[lo], [hi]], fmt="o", ms=4, color=_P3_SERIES[0], elinewidth=1)
            pooled = arms[a]["pooled"]["bias"]["mean"]
            ax.plot(pooled, yi, marker="|", ms=9, color=_P3_SERIES[1])
        ax.axvline(0, color=_P3_MUTED, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"N = {int(nk[1:]):,}", fontsize=9, loc="left")
        ax.set_xlabel("predicted − realised E (round 1 ●, all rounds |)", fontsize=8)
        _p3_axes(ax, "x")
    fig.suptitle("X14: BayesPlanner calibration bias by arm", fontsize=10, x=0.02, ha="left")
    paths.append(save_figure(fig, figure_path(out_dir, "x14", "bias")))
    fig, ax = new_figure(figsize=(6.0, 3.4))
    for i, nk in enumerate(ns):
        d = data[nk]["arms"]["default"]["per_round"]
        rounds = sorted(int(r) for r in d)
        vals = [_p3_err(d[str(r)]["bias"]) for r in rounds]
        ax.errorbar(rounds, [v[0] for v in vals], yerr=[[v[1] for v in vals], [v[2] for v in vals]], marker="o", ms=4,
                    color=_P3_SERIES[i], capsize=2, label=f"default, N = {int(nk[1:]):,}")
    ax.axhline(0, color=_P3_MUTED, linewidth=0.8)
    ax.set_xlabel("decision round", fontsize=9)
    ax.set_ylabel("predicted − realised E", fontsize=9)
    ax.set_title("X14: prediction bias by decision round (default arm)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    _p3_axes(ax)
    paths.append(save_figure(fig, figure_path(out_dir, "x14", "bias_by_round")))
    return paths


@register("x19")
def plot_x19(data: dict, out_dir: Path) -> list[Path]:
    paths = []
    opt = data["optimisers"]
    names = [m for m in ("PG", "PG_lib_step", "MD", "FW", "ADMM") if m in opt]
    fig, axes = new_figure(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    med = [opt[m]["iterations_to_tol_median"] or 0 for m in names]
    reach = [opt[m]["reached_tol"]["rate"] for m in names]
    ax.bar(range(len(names)), med, color=[_P3_SERIES[i] for i in range(len(names))])
    for i, (v, r) in enumerate(zip(med, reach)):
        ax.text(i, v, f"{v:g}\n{r:.0%} reach", ha="center", va="bottom", fontsize=7, color=_P3_INK)
    ax.set_ylim(0, max(med) * 1.3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("median iterations to E* − 1e-4", fontsize=9)
    ax.set_title("optimisers (8 standard starts)", fontsize=9, loc="left")
    _p3_axes(ax)
    ax = axes[1]
    geo = data["geometry"]
    keys = ["mean", "median_projected", "geomedian_tv", "oracle_manifesto"]
    blends = sorted([k for k in geo if k.startswith("blend_")], key=lambda k: (len(k), k))
    keys += blends
    vals = [_p3_err(geo[k]["tv_to_star"]) for k in keys]
    colors = [_P3_SERIES[1] if not k.startswith("blend_") else _P3_SERIES[{2: 0, 3: 2, 4: 3}[len(k) - 6]] for k in keys]
    ax.bar(range(len(keys)), [v[0] for v in vals], yerr=[[v[1] for v in vals], [v[2] for v in vals]], color=colors,
           error_kw={"elinewidth": 0.6, "ecolor": _P3_MUTED})
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("blend_", "B").replace("_projected", "").replace("geomedian_tv", "geo-med")
                        .replace("oracle_manifesto", "ORACLE b") for k in keys], fontsize=7, rotation=60)
    ax.set_ylabel("TV distance to b*", fontsize=9)
    ax.set_title("where b* sits (B = region blend)", fontsize=9, loc="left")
    _p3_axes(ax)
    fig.suptitle("X19: manifesto optimisers and geometry, 100 seeds", fontsize=10, x=0.02, ha="left")
    paths.append(save_figure(fig, figure_path(out_dir, "x19", "optimisers_geometry")))
    return paths


# =========================================================================== end Phase 3 block


# =========================================================================== Phase 2 (X1, X2, X3, X4, X11, X18) and X9 (separate block)
# Reuses _style / SERIES / INK2 / GRID from the top of this file (dataviz reference palette, light mode).
def _p2_ci(stat: dict | None) -> tuple[float, float, float]:
    """(mean, lower error, upper error) from a mean_ci / rate record (missing CI → zero bars)."""
    if not stat:
        return float("nan"), 0.0, 0.0
    m = stat.get("mean", stat.get("rate"))
    ci = stat.get("ci95") or [m, m]
    return m, max(0.0, m - ci[0]), max(0.0, ci[1] - m)


@register("x3")
def plot_x3(data: dict, out_dir: Path) -> list[Path]:
    out = []
    # 1. observed branching vs the weighted operator (binned), with y = x and the threshold lines
    fig, (ax, ax2) = new_figure(1, 2, figsize=(10, 4))
    bins = data["binned_by_lambda_ADq"]
    xs = [((b["lo"] + min(b["hi"], 20)) / 2) for b in bins]
    for key, label, color in (("R_obs", "R_obs (hops 2–5, pooled)", SERIES[0]), ("R_early", "R_early (hop 2 / hop 1)", SERIES[1])):
        pts = [(x, *_p2_ci(b[key])) for x, b in zip(xs, bins) if b.get(key)]
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[[p[2] for p in pts], [p[3] for p in pts]],
                    marker="o", ms=4, lw=1.5, capsize=2, color=color, label=label)
    ext = [c for c in data["extension_cells"].values() if c["k"] == 10]
    ax.plot([c["lambda_ADq"]["mean"] for c in ext], [c["R_obs"]["mean"] for c in ext], "s", ms=4, color=SERIES[2],
            label="extension points, k = 10")
    ax.plot([0.3, 20], [0.3, 20], ls=":", color=INK2, lw=1, label="R = λ")
    ax.axhline(1, color=CRITICAL, lw=0.8)
    ax.axvline(1, color=CRITICAL, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("λ_max(A D_q)  (q = γ·P(b))", color=INK2)
    ax.set_ylabel("observed branching", color=INK2)
    ax.set_title("Cascade growth vs the weighted operator", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    caps = [(x, *_p2_ci(b["cap_bind"])) for x, b in zip(xs, bins) if b.get("cap_bind")]
    ax2.errorbar([p[0] for p in caps], [p[1] for p in caps], yerr=[[p[2] for p in caps], [p[3] for p in caps]],
                 marker="o", ms=4, lw=1.5, capsize=2, color=SERIES[0])
    ax2.axvline(1, color=CRITICAL, lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("λ_max(A D_q)", color=INK2)
    ax2.set_ylabel("share of targeted voters at the cap (3)", color=INK2)
    ax2.set_title("The attempt cap binds above threshold", fontsize=10)
    _style(ax2)
    out.append(save_figure(fig, figure_path(out_dir, "x3", "threshold")))
    # 2. plan sweep: distinct converts, Δ expected share, cross-region share vs θ for each k
    fig, axes = new_figure(1, 3, figsize=(12, 3.6))
    thetas = data["thetas"]
    for j, k in enumerate(data["ks"]):
        cells = [data["cells"][f"theta{i}_k{k}"] for i in range(len(thetas))]
        for ax, key in zip(axes, ("distinct", "delta_expected", "cross_share")):
            m = [_p2_ci(c[key]) for c in cells]
            ax.errorbar(thetas, [x[0] for x in m], yerr=[[x[1] for x in m], [x[2] for x in m]], marker="o", ms=3,
                        lw=1.4, capsize=2, color=SERIES[j], label=f"k = {k}")
    for ax, lab in zip(axes, ("distinct converts", "Δ expected share vs k = 0", "share of converts outside the region")):
        ax.set_xlabel("θ  (0 = far archetype, 1 = region's own mean)", color=INK2)
        ax.set_ylabel(lab, color=INK2)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    out.append(save_figure(fig, figure_path(out_dir, "x3", "sweep")))
    return out


@register("x11")
def plot_x11(data: dict, out_dir: Path) -> list[Path]:
    out = []
    fees = data["fees"]
    fig, ax = new_figure(figsize=(7, 3.8))
    groups = ["overall"] + sorted(data["by_personality"][fees[0]])
    width = 0.38
    for j, fee in enumerate(fees):
        vals = []
        for g in groups:
            if g == "overall":
                vals.append(_p2_ci(data["interaction"][fee]["seed_level"]))
            else:
                p = data["by_personality"][fee][g]
                m = p["interaction"]["mean"]
                lo, hi = p["cluster_boot_ci95"]
                vals.append((m, m - lo, hi - m))
        xs = [i + (j - 0.5) * width for i in range(len(groups))]
        ax.bar(xs, [v[0] for v in vals], width, yerr=[[v[1] for v in vals], [v[2] for v in vals]], capsize=2,
               color=SERIES[j], label=f"worker fee ₫{int(float(fee))}")
    names = {g: data["by_personality"][fees[0]][g]["name"] for g in groups if g != "overall"}
    ax.set_xticks(range(len(groups)), ["all units"] + [f"{g}\n{names[g]}" for g in groups[1:]], fontsize=7)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("interaction Δ(both) − Δ(w) − Δ(m)", color=INK2)
    ax.set_title("Worker × media interaction by personality of the targeted region", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    out.append(save_figure(fig, figure_path(out_dir, "x11", "interaction")))
    fig, ax = new_figure(figsize=(7, 3.8))
    arms = ["none", "media", "workers", "both", "workers_then_media", "media_then_workers"]
    for j, fee in enumerate(fees):
        vals = [_p2_ci(data["arms"][f"{a}@{fee}"]["expected"]) for a in arms]
        xs = [i + (j - 0.5) * width for i in range(len(arms))]
        ax.bar(xs, [v[0] for v in vals], width, yerr=[[v[1] for v in vals], [v[2] for v in vals]], capsize=2,
               color=SERIES[j], label=f"worker fee ₫{int(float(fee))}")
    ax.set_xticks(range(len(arms)), ["none", "media r1", "workers r1", "both r1", "workers r1\nmedia r2",
                                     "media r1\nworkers r2"], fontsize=7)
    ax.set_ylim(0.4, None)
    ax.set_ylabel("expected share (400 units)", color=INK2)
    ax.set_title("Factorial and sequencing arms (matched spend)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    out.append(save_figure(fig, figure_path(out_dir, "x11", "sequencing")))
    return out


@register("x2")
def plot_x2(data: dict, out_dir: Path) -> list[Path]:
    out = []
    fig, ax = new_figure(figsize=(7, 3.8))
    rules = list(data["rules"])
    ks = [k for k in data["ks"] if k > 0]
    for j, rule in enumerate(rules):
        on = [_p2_ci(data["anchor_share"][f"{rule}|k{k}"]["delta_on"]) for k in ks]
        off = [_p2_ci(data["anchor_share"][f"{rule}|k{k}"]["delta_off"]) for k in ks]
        ax.errorbar(ks, [x[0] for x in on], yerr=[[x[1] for x in on], [x[2] for x in on]], marker="o", ms=4, lw=1.4,
                    capsize=2, color=SERIES[j], label=f"{rule}: cascade on")
        ax.errorbar(ks, [x[0] for x in off], yerr=[[x[1] for x in off], [x[2] for x in off]], marker="s", ms=4, lw=1.4,
                    ls="--", capsize=2, color=SERIES[j], label=f"{rule}: cascade off (anchor only)")
    ax.set_yscale("log")
    ax.set_xlabel("workers bought in round 1 (k)", color=INK2)
    ax.set_ylabel("Δ expected share vs k = 0", color=INK2)
    ax.set_title("Worker value: cascade vs pinned anchor", fontsize=10)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    _style(ax)
    out.append(save_figure(fig, figure_path(out_dir, "x2", "anchor")))
    fig, (ax, ax2) = new_figure(1, 2, figsize=(10, 3.8))
    ts = [int(t) for t in data["long_checkpoints"]]
    keys = [("degroot|z0.05|k0", "DeGroot, ζ = 0.05, k = 0"), ("degroot|z0.05|k20", "DeGroot, ζ = 0.05, k = 20"),
            ("degroot|z0.0|k20", "DeGroot, ζ = 0, k = 20"), ("wmsr_F2|z0.05|k20", "W-MSR F = 2, ζ = 0.05, k = 20")]
    for j, (key, label) in enumerate(keys):
        lr = data["long_runs"][key]
        ax.plot([max(t, 0.5) for t in ts], [lr["share"][str(t)]["mean"] for t in ts], marker="o", ms=3, color=SERIES[j],
                label=label)
        ax2.plot([max(t, 0.5) for t in ts], [lr["disagreement"][str(t)]["mean"] for t in ts], marker="o", ms=3,
                 color=SERIES[j], label=label)
        if key in data["fj"]:
            ax.axhline(data["fj"][key]["share_fj"]["mean"], color=SERIES[j], lw=0.8, ls=":")
            ax2.axhline(max(data["fj"][key]["disagreement_fj"]["mean"], 1e-4), color=SERIES[j], lw=0.8, ls=":")
    for a, lab in ((ax, "expected share (dotted: Friedkin–Johnsen limit)"),
                   (ax2, "disagreement (dotted: FJ fixed point; 1e-4 marks 0)")):
        a.set_xscale("log")
        a.set_xlabel("round (log; game horizon T = 5)", color=INK2)
        a.set_ylabel(lab, color=INK2)
        a.axvline(5, color=INK2, lw=0.6)
        _style(a)
    ax2.set_yscale("log")
    ax.legend(fontsize=6, frameon=False)
    out.append(save_figure(fig, figure_path(out_dir, "x2", "long_runs")))
    return out


@register("x4")
def plot_x4(data: dict, out_dir: Path) -> list[Path]:
    out = []
    fig, axes = new_figure(1, 2, figsize=(10, 3.6), sharey=True)
    width = 0.27
    for ax, K in zip(axes, data["ks"]):
        for j, pl in enumerate(data["placements"]):
            vals = [_p2_ci(data["arms"][f"K{K}|{tm}|{pl}"]["delta_vs_none"]) for tm in data["timings"]]
            xs = [i + (j - 1) * width for i in range(len(data["timings"]))]
            ax.bar(xs, [v[0] for v in vals], width, yerr=[[v[1] for v in vals], [v[2] for v in vals]], capsize=2,
                   color=SERIES[j], label=pl)
        ax.set_xticks(range(len(data["timings"])), ["all in round 1", "even", "all in round 5"], fontsize=8)
        ax.set_title(f"K = {K} workers (₫{20 * K})", fontsize=10)
        _style(ax)
    axes[0].set_ylabel("Δ expected share vs no campaign", color=INK2)
    axes[0].legend(fontsize=7, frameon=False, title="placement", title_fontsize=7)
    out.append(save_figure(fig, figure_path(out_dir, "x4", "timing_placement")))
    fig, ax = new_figure(figsize=(7, 3.6))
    arms = [tuple(a) for a in data["pick_arms"]]
    for j, rule in enumerate(data["pick_rules"]):
        vals = [_p2_ci(data["arms"][f"pick|{rule}|K{K}|{tm}|{pl}"]["delta_vs_none"]) for K, tm, pl in arms]
        xs = [i + (j - 1) * width for i in range(len(arms))]
        ax.bar(xs, [v[0] for v in vals], width, yerr=[[v[1] for v in vals], [v[2] for v in vals]], capsize=2,
               color=SERIES[j], label=rule)
    ax.set_xticks(range(len(arms)), [f"K = {K}, {tm}, {pl}" for K, tm, pl in arms], fontsize=8)
    ax.set_ylim(0.08, None)
    ax.set_ylabel("Δ expected share vs no campaign", color=INK2)
    ax.set_title("Pick-rule ablation (outside the game rules)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    out.append(save_figure(fig, figure_path(out_dir, "x4", "pick_rule")))
    return out


@register("x18")
def plot_x18(data: dict, out_dir: Path) -> list[Path]:
    out = []
    fig, (ax, ax2) = new_figure(1, 2, figsize=(10, 3.8))
    ks = [k for k in data["ks"] if k > 0]
    for j, g in enumerate(("rank0", "rank1", "rank2", "rank3")):
        curve = data["curve"][g]
        ax.plot(ks, [curve[str(k)]["delta_mean"] for k in ks], marker="o", ms=3, color=SERIES[j],
                label=f"size rank {g[-1]}" + (" (largest)" if g == "rank0" else ""))
        ax2.plot(ks, [curve[str(k)]["marginal_gain_per_worker"] for k in ks], marker="o", ms=3, color=SERIES[j])
    ax.set_xlabel("workers k in one region (round 1)", color=INK2)
    ax.set_ylabel("Δ expected share vs k = 0", color=INK2)
    ax.legend(fontsize=7, frameon=False)
    ax2.set_yscale("symlog", linthresh=1e-5)
    ax2.axhline(0, color=INK2, lw=0.6)
    ax2.set_xlabel("k", color=INK2)
    ax2.set_ylabel("marginal gain per worker (symlog)", color=INK2)
    ax.set_title("Worker value curve", fontsize=10)
    ax2.set_title("Diminishing returns", fontsize=10)
    _style(ax)
    _style(ax2)
    out.append(save_figure(fig, figure_path(out_dir, "x18", "curve")))
    runs = data.get("celf_runs") or []
    if runs:
        import numpy as np
        fig, ax = new_figure(figsize=(6, 3.6))
        kk = list(range(0, data["celf"]["k_max"] + 1))
        for j, (key, label) in enumerate((("celf_values", "CELF (greedy, full graph)"),
                                          ("top_degree_values", "top degree"), ("random_values", "random (R22)"))):
            v = np.array([[x - r["f0"] for x in r[key]] for r in runs])
            m = v.mean(0)
            se = v.std(0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(m)
            ax.errorbar(kk, m, yerr=1.96 * se, marker="o", ms=3, lw=1.4, capsize=2, color=SERIES[j], label=label)
        ax.set_xlabel("workers k (largest region)", color=INK2)
        ax.set_ylabel("Δ expected share (4-stream mean)", color=INK2)
        ax.set_title(f"CELF reference ({len(runs)} seeds)", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
        out.append(save_figure(fig, figure_path(out_dir, "x18", "celf")))
    return out


@register("x1")
def plot_x1(data: dict, out_dir: Path) -> list[Path]:
    out = []
    fig, (ax, ax2) = new_figure(1, 2, figsize=(10, 4))
    marks = {10: "o", 20: "s", 30: "^"}
    for j, pg in enumerate(data["pg"]):
        for kbar in data["kbars"]:
            c = data["cells"][f"k{kbar}|{pg}"]
            obs = 1 - c["hom_rho_obs_200_400"]["mean"]
            ax.plot(1 - c["rho_rw"]["mean"], obs, marks[kbar], color=SERIES[j], ms=6)
            ax.plot(1 - c["rho_L"]["mean"], obs, marks[kbar], color=SERIES[j], ms=6, mfc="none")
    lim = [0.0, 0.05]
    ax.plot(lim, lim, ls=":", color=INK2, lw=1)
    ax.set_xlabel("predicted decay per round (filled: ā(1 − μ₂), open: ā λ₂(L)/k̄)", color=INK2, fontsize=8)
    ax.set_ylabel("observed decay 1 − ρ_obs (homogeneous a, t 200–400)", color=INK2, fontsize=8)
    ax.set_title("Consensus speed: random-walk gap vs Laplacian", fontsize=10)
    for j, pg in enumerate(data["pg"]):
        ax.plot([], [], "o", color=SERIES[j], label=f"p_g {pg}")
    for kbar, mk in marks.items():
        ax.plot([], [], mk, color=INK2, label=f"k̄ = {kbar}")
    ax.legend(fontsize=6, frameon=False)
    _style(ax)
    width = 0.27
    for j, pg in enumerate(data["pg"]):
        vals = [_p2_ci(data["cells"][f"k{kbar}|{pg}"]["top3/planner"]["win"]) for kbar in data["kbars"]]
        xs = [i + (j - 1) * width for i in range(len(data["kbars"]))]
        ax2.bar(xs, [v[0] for v in vals], width, yerr=[[v[1] for v in vals], [v[2] for v in vals]], capsize=2,
                color=SERIES[j], label=f"p_g {pg}")
    ax2.set_xticks(range(len(data["kbars"])), [f"k̄ = {k}" for k in data["kbars"]])
    ax2.set_ylabel("win rate, nationwide top-3 manifesto + planner", color=INK2, fontsize=8)
    ax2.set_title("Single nationwide manifesto vs cross-region ties", fontsize=10)
    ax2.legend(fontsize=7, frameon=False)
    _style(ax2)
    out.append(save_figure(fig, figure_path(out_dir, "x1", "mixing_and_wins")))
    fig, ax = new_figure(figsize=(7, 3.8))
    for j, pg in enumerate(data["pg"]):
        c = data["cells"][f"k30|{pg}"]
        ts = sorted((int(t) for t in c["het_between_rel_mean"]), key=int)
        ax.plot([max(t, 0.5) for t in ts], [c["het_between_rel_mean"][str(t)] for t in ts], marker="o", ms=3,
                color=SERIES[j], label=f"k̄ = 30, p_g {pg}")
    ax.axhline(0.5, color=CRITICAL, lw=0.8)
    ax.axvline(5, color=INK2, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("round (log; game horizon T = 5)", color=INK2)
    ax.set_ylabel("between-region dispersion / start", color=INK2)
    ax.set_title("Regional means mixing (zealot-free social-only runs)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    out.append(save_figure(fig, figure_path(out_dir, "x1", "regional_mixing")))
    return out


@register("x9")
def plot_x9(data: dict, out_dir: Path) -> list[Path]:
    out = []
    ok = [r for r in data["results"] if not r.get("oom")]
    fig, (ax, ax2) = new_figure(1, 2, figsize=(10, 3.8))
    series = [(d, B) for d in sorted({r["device"] for r in ok}) for B in sorted({r["B"] for r in ok if r["device"] == d})]
    for j, (dev, B) in enumerate(series):
        rs = sorted([r for r in ok if r["device"] == dev and r["B"] == B], key=lambda r: r["n"])
        ax.plot([r["n"] for r in rs], [r["ms_per_world_round"] for r in rs], marker="o", ms=4,
                ls="-" if dev != "cpu" else "--", color=SERIES[j % len(SERIES)], label=f"{dev}, B = {B}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("voters N", color=INK2)
    ax.set_ylabel("ms per world-round (run_batch)", color=INK2)
    ax.set_title("Round cost vs N and batch size", fontsize=10)
    ax.legend(fontsize=6, frameon=False)
    _style(ax)
    for j, dev in enumerate(("gpu", "cpu")):
        for f in ("0.01",):
            key = f"{dev}|f{f}"
            h = data["cascade_hop_scaling"].get(key)
            if not h:
                continue
            rs = sorted([r for r in ok if r["B"] == 1 and (r["device"] == "cpu") == (dev == "cpu") and "cascade_hop" in r],
                        key=lambda r: r["n"])
            ax2.plot([2 * r["edges_undirected"] for r in rs], h["median_ms"], marker="o", ms=4, color=SERIES[j],
                     label=f"{dev}: slope {h['loglog_slope_time_vs_edges']:.2f}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("directed edges 2E", color=INK2)
    ax2.set_ylabel("cascade hop (ms, 1% spreaders)", color=INK2)
    ax2.set_title("Cascade hop vs edges", fontsize=10)
    ax2.legend(fontsize=7, frameon=False)
    _style(ax2)
    out.append(save_figure(fig, figure_path(out_dir, "x9", "scaling")))
    return out


# =========================================================================== end Phase 2 / X9 block


# =========================================================================== Phase 4 / 5 adapters
# X5, X17, X8, X15, X16 and X20–X22 keep their plotting code in their own modules (written in
# parallel with this file); these adapters let one command redraw every figure from the JSON.
# Imports are lazy to avoid a circular import (those modules reuse this file's style helpers).

def _adapter(module: str, fn: str, passes_dir: bool) -> PlotFn:
    def run(data: dict, out_dir: Path) -> list[Path]:
        import importlib
        f = getattr(importlib.import_module(module), fn)
        res = f(data, out_dir) if passes_dir else f(data)
        return list(res) if isinstance(res, (list, tuple)) else []
    return run


for _id, _mod, _fn, _dir in [
    ("x5", "plot_v3_phase4a", "plot_x5", True),
    ("x17", "plot_v3_phase4a", "plot_x17", True),
    ("x8", "plot_v3_x8_x15", "plot_x8", True),
    ("x15", "plot_v3_x8_x15", "plot_x15", True),
    ("x16", "x16_verdicts", "plot", False),
    ("x20", "x20_luck_skill", "plots", False),
    ("x21", "x21_misspec", "plots", False),
    ("x22", "x22_sensitivity", "plots", False),
    ("x_consensus", "x_consensus", "plot", True),
]:
    PLOTS.setdefault(_id, _adapter(_mod, _fn, _dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated experiment ids, e.g. x0,x10")
    parser.add_argument("--reports", default=str(REPORTS))
    args = parser.parse_args()
    reports = Path(args.reports)
    out_dir = reports / "figures"
    wanted = [x.strip().lower() for x in args.only.split(",") if x.strip()] or EXPERIMENTS + ["x_consensus"]
    written: list[Path] = []
    for exp_id in wanted:
        path = reports / f"{exp_id}.json"
        if not path.exists():
            print(f"{exp_id}: no {path.name} yet, skipped")
            continue
        if exp_id not in PLOTS:
            print(f"{exp_id}: no plot function yet, skipped")
            continue
        written += PLOTS[exp_id](json.loads(path.read_text()), out_dir)
    print(f"{len(written)} figure(s) written")


if __name__ == "__main__":
    main()
