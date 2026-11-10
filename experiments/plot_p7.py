"""P7 figures, generated ONLY from the saved JSON reports (no simulation here).

Usage: python experiments/plot_p7.py [--reports reports/v2] [--figs reports/v2/figures] [--only e1,e2,...]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# colour-blind-validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "lines.markersize": 6, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def err(ci, mean):
    return [[mean - ci[0]], [ci[1] - mean]]


def errbar(ax, x, cells, key, color, label, marker="o", ls="-", sub="mean", offset=0.0):
    xs, ys, lo, hi = [], [], [], []
    for xv, c in zip(x, cells):
        v = c[key]
        m = v["rate"] if "rate" in v else v["mean"]
        xs.append(xv + offset)
        ys.append(m)
        lo.append(m - v["ci95"][0])
        hi.append(v["ci95"][1] - m)
    ax.errorbar(xs, ys, yerr=[lo, hi], color=color, marker=marker, ls=ls, label=label, capsize=3, elinewidth=1.2)


def plain_log(ax, ticks, axis="both", fmt="{:g}"):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    for name in (("x", "y") if axis == "both" else (axis,)):
        a = ax.xaxis if name == "x" else ax.yaxis
        a.set_major_locator(FixedLocator(ticks))
        a.set_minor_locator(NullLocator())
        a.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))


def finish(fig, path: Path, title: str | None = None):
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------------------- E1
def plot_e1(rep, figs: Path):
    cells, kbars, pgs = rep["cells"], rep["kbars"], list(rep["pg_ranges"])
    rows = rep["rows"]
    pg_label = {p: p.replace("[", " [") for p in pgs}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) homogeneous-step lecture prediction
    ax = axes[0]
    for j, pg in enumerate(pgs):
        rs = [r for r in rows if r["pg"] == pg]
        ax.scatter([1 - r["rho_rw"] for r in rs], [1 - r["hom_rho_obs"] for r in rs], s=8, color=C[j], alpha=0.25, lw=0)
        for i, k in enumerate(kbars):
            c = cells[f"k{k}|{pg}"]
            ax.scatter(1 - c["rho_rw"]["mean"], 1 - c["hom_rho_obs"]["mean"], color=C[j], marker=MARKERS[i], s=60,
                       edgecolor="white", linewidth=1.2, zorder=3)
    lim = [5e-3, 6e-2]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    plain_log(ax, [0.005, 0.01, 0.02, 0.04])
    ax.set_xlabel("predicted decay per round  ā·(1 − μ₂(D⁻¹A))")
    ax.set_ylabel("observed decay per round  1 − ρ_obs")
    ax.set_title("(a) uniform step ā: lecture prediction")
    # (b) game-horizon between-region decay with heterogeneous a
    ax = axes[1]
    for j, pg in enumerate(pgs):
        rs = [r for r in rows if r["pg"] == pg]
        ax.scatter([1 - r["rho_between_pred_mu234"] for r in rs], [1 - r["rho_between5"] for r in rs], s=8, color=C[j], alpha=0.25, lw=0)
        for i, k in enumerate(kbars):
            c = cells[f"k{k}|{pg}"]
            ax.scatter(1 - c["rho_between_pred_mu234"]["mean"], 1 - c["rho_between5"]["mean"], color=C[j],
                       marker=MARKERS[i], s=60, edgecolor="white", linewidth=1.2, zorder=3)
    lim = [5e-3, 8e-2]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    plain_log(ax, [0.005, 0.01, 0.02, 0.04, 0.08])
    ax.set_xlabel("predicted  ā·(1 − mean μ₂…μ₄)")
    ax.set_ylabel("observed between-region decay, rounds 0–5")
    ax.set_title("(b) game dynamics (a_i = ω·α_i), T = 5")
    # (c) long-run tail: exact operator vs λ2
    ax = axes[2]
    tail = [r for r in rows if r.get("rho_obs") is not None]
    for j, pg in enumerate(pgs):
        rs = [r for r in tail if r["pg"] == pg]
        ax.scatter([1e3 * r["nu2_operator"] for r in rs], [1e3 * (1 - r["rho_obs"]) for r in rs], s=14, color=C[j], alpha=0.6, lw=0,
                   label=pg_label[pg])
    if tail:
        lo = 0.0
        hi = 1e3 * max(max(r["nu2_operator"] for r in tail), max(1 - r["rho_obs"] for r in tail)) * 1.05
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("exact operator  ν₂(G^½ L G^½), G = diag(a/d)  [×10⁻³]")
    ax.set_ylabel("observed tail decay, rounds 3000–4000  [×10⁻³]")
    ax.set_title("(c) long run: slowest voters, not λ₂")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C[j], marker="o", ls="", label=f"p_g {pg_label[pg]}") for j, pg in enumerate(pgs)]
    handles += [Line2D([], [], color=MUTED, marker=MARKERS[i], ls="", label=f"k̄ = {k}") for i, k in enumerate(kbars)]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.06))
    finish(fig, figs / "e1_consensus_theory.png", "E1 · consensus speed: spectral prediction vs observed (dashed: y = x)")

    # disagreement curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    t10 = list(range(11))
    for j, pg in enumerate(pgs):
        for i, k in enumerate(kbars):
            c = cells[f"k{k}|{pg}"]
            if k not in (8, 48):
                continue
            ls = "-" if k == 8 else ":"
            axes[0].plot(t10, c["within_rel_curve"], color=C[j], ls=ls, label=f"{pg_label[pg]}, k̄={k}")
            axes[1].plot(t10, c["between_rel_curve"], color=C[j], ls=ls, label=f"{pg_label[pg]}, k̄={k}")
    am = np.mean([r["a_mean"] for r in rows])
    pred = [np.mean([r["bulk_round1_ratio"] for r in rows]) ** t for t in t10]
    pred_line, = axes[0].plot(t10, pred, color=INK, lw=1, ls="--", label="i.i.d. prediction [mean (1−a)² + a²/d]^t")
    axes[0].set_title("within-region variance W(t)/W(0)")
    axes[1].set_title("between-region variance B(t)/B(0)")
    for ax in axes:
        ax.set_xlabel("round t (social-only, ζ = 0)")
        ax.set_yscale("log")
        ax.axvline(5, color=MUTED, lw=0.8)
        ax.text(5.1, ax.get_ylim()[1] * 0.9 if ax.get_yscale() == "linear" else 0.95, "T = 5", color=MUTED, fontsize=8, va="top")
    axes[0].set_ylabel("relative variance (log)")
    axes[1].legend(loc="lower left", ncol=1, fontsize=8)
    axes[0].legend(handles=[pred_line], loc="lower left", fontsize=8)
    finish(fig, figs / "e1_variance_curves.png", f"E1 · social-only mixing over the game horizon (ā ≈ {am:.3f})")

    # outcomes
    names = [("oracle/planner", "ORACLE manifesto + planner"), ("bayes", "BayesPlanner (engine)"), ("top3", "Top3Player (engine)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for a, (name, title) in enumerate(names):
        ax = axes[a]
        for j, pg in enumerate(pgs):
            errbar(ax, kbars, [cells[f"k{k}|{pg}"] for k in kbars], f"win_{name}", C[j], f"p_g {pg_label[pg]}",
                   marker=MARKERS[j], offset=(j - 1) * 0.8)
        ax.set_title(title)
        ax.set_xlabel("mean degree k̄")
        ax.set_xticks(kbars)
        ax.set_ylim(0.35, 1.02)
    axes[0].set_ylabel("win rate (95% Wilson CI)")
    axes[0].legend(loc="lower right")
    ax = axes[3]
    pats = [("pay_workers", "workers"), ("pay_media", "media"), ("pay_split", "split")]
    short = {"low": "lo", "default": "def", "high": "hi"}
    labels = [f"{k}\n{short.get(pg.split('[')[0], pg)}" for pg in pgs for k in kbars]
    bottoms = np.zeros(len(labels))
    for i, (p, lab) in enumerate(pats):
        vals = np.array([cells[f"k{k}|{pg}"]["best_paid_pattern"][p] / cells[f"k{k}|{pg}"]["n"] for pg in pgs for k in kbars])
        ax.bar(range(len(labels)), vals, bottom=bottoms, color=C[i], label=lab, edgecolor="white", linewidth=1)
        bottoms += vals
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("share of seeds")
    ax.set_title("best fixed paid pattern")
    ax.set_xlabel("k̄ / p_g range (lo, def, hi)")
    ax.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x")
    finish(fig, figs / "e1_outcomes.png", "E1 · topology and campaign outcomes")


# ----------------------------------------------------------------------------------------- E2
def plot_e2(rep, figs: Path):
    zetas, rules = rep["zetas"], list(rep["rules"])
    lab = {"degroot": "DeGroot", "wmsr_F1": "W-MSR F = 1", "wmsr_F2": "W-MSR F = 2"}
    cells = rep["cells"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    x = [100 * z for z in zetas]
    for i, r in enumerate(rules):
        cs = [cells[f"{z}|{r}"] for z in zetas]
        errbar(axes[0], x, cs, "win", C[i], lab[r], marker=MARKERS[i], offset=(i - 1) * 0.6)
        errbar(axes[1], x, cs, "persuasion", C[i], lab[r], marker=MARKERS[i], offset=(i - 1) * 0.6)
        errbar(axes[2], x, cs, "expected_none", C[i], lab[r], marker=MARKERS[i], offset=(i - 1) * 0.6)
        xs = [100 * z for z in zetas[1:]]
        errbar(axes[3], xs, [cells[f"{z}|{r}"] for z in zetas[1:]], "drag_T30", C[i], lab[r], marker=MARKERS[i], offset=(i - 1) * 0.6)
    axes[0].set_title("win rate (ORACLE + planner)"); axes[0].set_ylabel("win rate (95% CI)")
    axes[1].set_title("persuasion  E_T(planner) − E_T(none)"); axes[1].set_ylabel("expected-share points")
    axes[2].set_title("no-campaign share E_T(none)"); axes[2].set_ylabel("expected share")
    axes[3].set_title("drag probe: extreme zealots, T = 30"); axes[3].set_ylabel("TV shift of non-zealot regional mean")
    for ax in axes:
        ax.set_xlabel("zealot fraction ζ (%)")
    axes[0].legend(loc="lower left")
    finish(fig, figs / "e2_zealots.png", "E2 · zealots × resilient consensus (W-MSR)")

    # predicted vs observed drag ratio
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for i, r in enumerate(("wmsr_F1", "wmsr_F2")):
        for j, z in enumerate(zetas[1:]):
            p = rep["paired"][f"{z}|{r}_vs_degroot"]
            for T, mk in ((5, MARKERS[j]), (30, MARKERS[j])):
                key = f"drag_T{T}_ratio"
                face = C[i + 1] if T == 30 else "white"
                ax.errorbar(p["pred_drag_ratio"]["mean"], p[key]["mean"], yerr=err(p[key]["ci95"], p[key]["mean"]),
                            color=C[i + 1], marker=mk, mfc=face, ms=8, capsize=3, ls="")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_xlabel("first-order prediction  Σa·max(0,z−F)/(d−F) ÷ Σa·z/d")
    ax.set_ylabel("observed drag ratio  W-MSR / DeGroot")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C[1], marker="o", ls="", label="F = 1"), Line2D([], [], color=C[2], marker="o", ls="", label="F = 2")]
    h += [Line2D([], [], color=MUTED, marker=MARKERS[j], ls="", label=f"ζ = {z}") for j, z in enumerate(zetas[1:])]
    h += [Line2D([], [], color=MUTED, marker="o", mfc="white", ls="", label="T = 5 (open)"),
          Line2D([], [], color=MUTED, marker="o", ls="", label="T = 30 (filled)")]
    ax.legend(handles=h, loc="upper left", fontsize=8)
    finish(fig, figs / "e2_drag_ratio.png", "E2 · W-MSR trimming of extreme zealots")


# ----------------------------------------------------------------------------------------- E3
def plot_e3(rep, figs: Path):
    rows, taus = rep["rows"], rep["taus"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for a, key in enumerate(("x_rr", "x_full")):
        ax = axes[a]
        for i, tau in enumerate(taus):
            pts = [p for r in rows for p in r["taus"][str(tau)]["probe"]]
            ax.scatter([p[key] for p in pts], [p["reach_per_seed"] for p in pts], s=6, alpha=0.25, color=C[i % 8], lw=0,
                       label=f"τ_c = {tau}" if a == 0 else None)
        b = rep["collapse"][key]["binned"]
        ax.plot([q["x"] for q in b], [q["reach"] for q in b], color=INK, lw=1.5, label="binned mean")
        ax.axvline(1, color=MUTED, ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel("word-of-mouth reach per seed")
        rho_d = rep["by_tau"]["0.03"]["spearman_reach_vs_lambda_rr"]["rho"]
        ax.set_title("(a) vs τ_c·λ_max(A_rr)/ρ  (regional)\n" + f"one curve; at fixed τ_c reach ~ λ_max(A_rr): Spearman {rho_d:.2f}"
                     if key == "x_rr" else "(b) vs τ_c·λ_max(A)/ρ  (whole graph)\nregions of a seed share x: no within-τ collapse")
        ax.set_xlabel("x = τ_c·λ_max/ρ (log)")
    xs = np.logspace(math.log10(0.1), math.log10(12), 100)
    kbar = np.mean([r["mean_degree"] for r in rows]); lf = np.mean([r["lambda_max_full"] for r in rows])
    mf = [(x * 0.5 / lf) * kbar * sum((1 - 0.5 + x * 0.5) ** h for h in range(3)) for x in xs]
    axes[1].plot(xs, mf, color=C[6], ls=":", lw=1.5, label="linear mean field")
    leg = axes[0].legend(loc="upper left", fontsize=7, ncol=2, markerscale=2)
    for h in leg.legend_handles:
        h.set_alpha(1)
    axes[1].legend(loc="upper left", fontsize=8)
    ax = axes[2]
    for i, (key, lab) in enumerate((("x_rr", "regional λ_max(A_rr)"), ("x_full", "whole-graph λ_max(A)"))):
        xs_, ys_, lo, hi = [], [], [], []
        for tau in taus:
            c = rep["by_tau"][str(tau)]
            xs_.append(c[key]["mean"]); ys_.append(c["hop_growth_obs"]["mean"])
        ax.plot(xs_, ys_, color=C[i], marker=MARKERS[i], label=f"observed vs x({lab})")
    grid = np.logspace(math.log10(0.15), math.log10(9), 60)
    ax.plot(grid, 1 + 0.5 * (grid - 1), color=INK, ls="--", lw=1, label="linearised prediction g = 1 + ρ(x − 1)")
    ax.axhline(1, color=MUTED, lw=1); ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.set_xscale("log")
    plain_log(ax, [0.2, 0.5, 1, 2, 5], axis="x")
    ax.set_xlabel("x = τ_c·λ_max/ρ (log)")
    ax.set_ylabel("hop growth  new₃ / new₂")
    ax.set_ylim(0.4, 3.0)
    ax.set_title("(c) supercritical iff growth > 1")
    ax.legend(loc="upper left", fontsize=8)
    finish(fig, figs / "e3_cascade_threshold.png", "E3 · word-of-mouth reach vs the SIR spectral threshold")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cs = [rep["by_tau"][str(t)] for t in taus]
    errbar(axes[0], taus, cs, "pay_workers_beats_media", C[0], "P(pay_workers > pay_media)")
    axes[0].axhline(0.5, color=MUTED, lw=1)
    axes[0].set_ylabel("share of seeds (95% CI)")
    axes[0].set_title("paid workers beat paid media")
    errbar(axes[1], taus, cs, "win_oracle/planner", C[1], "ORACLE + planner", marker="s")
    errbar(axes[1], taus, cs, "win_informed/pay_workers", C[2], "informed + pay_workers", marker="^")
    errbar(axes[1], taus, cs, "win_informed/pay_media", C[3], "informed + pay_media", marker="D")
    axes[1].set_ylabel("win rate (95% CI)")
    axes[1].set_title("campaign outcome")
    axes[1].legend(loc="lower right")
    thr = rep["threshold_tau_rr"]["mean"]
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("transmission τ_c (log)")
        ax.axvline(thr, color=MUTED, ls="--", lw=1)
        ax.axvline(0.03, color=MUTED, ls=":", lw=1)
    tr = axes[0].get_xaxis_transform()
    axes[0].text(thr * 0.97, 0.03, "ρ/λ_max(A_rr) ", color=MUTED, fontsize=8, ha="right", transform=tr)
    axes[0].text(0.031, 0.03, " default 0.03", color=MUTED, fontsize=8, transform=tr)
    finish(fig, figs / "e3_workers_vs_media.png", "E3 · when do workers beat media?")


# ----------------------------------------------------------------------------------------- E5
def plot_e5(rep, figs: Path):
    Bs = rep["budgets"]
    x = [float(b) for b in Bs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    names = [("oracle", "ORACLE + planner"), ("bayes", "BayesPlanner"), ("top3", "Top3Player")]
    for i, (n, lab) in enumerate(names):
        cs = [{"w": rep["by_budget"][b][n]["win"]} for b in Bs]
        errbar(axes[0], x, cs, "w", C[i], lab, marker=MARKERS[i])
    axes[0].set_xscale("log", base=2); axes[0].set_xticks(x); axes[0].set_xticklabels([f"{int(v)}" for v in x])
    axes[0].set_xlabel("starting treasury B"); axes[0].set_ylabel("P(Victory | B), 95% CI")
    axes[0].set_title("win rate vs budget"); axes[0].legend(loc="lower right")
    ax = axes[1]
    cats = [("workers", "extra workers"), ("media", "media"), ("queries", "survey queries")]
    pos, labels = [], []
    for i, (n, lab) in enumerate(names):
        for j, b in enumerate(Bs):
            p = i * 4 + j
            pos.append(p); labels.append(f"{int(float(b))}")
    for k, (c, lab) in enumerate(cats):
        bottom = np.zeros(len(pos))
        for kk in range(k):
            bottom += np.array([rep["by_budget"][b][n]["spend_share"][cats[kk][0]] for n, _ in names for b in Bs])
        vals = np.array([rep["by_budget"][b][n]["spend_share"][c] for n, _ in names for b in Bs])
        ax.bar(pos, vals, bottom=bottom, color=C[k], label=lab, edgecolor="white", linewidth=1)
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=8)
    for i, (n, lab) in enumerate(names):
        ax.text(i * 4 + 1, -0.13, lab, ha="center", va="top", fontsize=9, color=INK, transform=ax.get_xaxis_transform())
    ax.set_ylabel("share of money spent"); ax.set_title("spend mix (by budget B)"); ax.grid(axis="x")
    ax.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.2))
    finish(fig, figs / "e5_budget.png", "E5 · budget")


# ----------------------------------------------------------------------------------------- E7
def plot_e7(rep, figs: Path):
    rows = [r for r in rep["rows"] if "error" not in r]
    devs = sorted({r["device"] for r in rows}, key=lambda d: not d.startswith("cuda"))
    med = lambda xs: float(np.median(xs))  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=False)
    series = [("generation_canonical_s", "generation, canonical CPU draw (R8)"), ("generation_device_s", "generation, GPU draw"),
              ("round_s", "one engine round (apply allocate)"), ("top3", "full game, Top3Player"), ("bayes", "full game, BayesPlanner")]
    for d_i, dev in enumerate(devs[:2]):
        ax = axes[d_i]
        rs = sorted([r for r in rows if r["device"] == dev], key=lambda r: r["num_voters"])
        for i, (k, lab) in enumerate(series):
            xs, ys = [], []
            for r in rs:
                if k in ("top3", "bayes"):
                    if k in r["full_game"]:
                        xs.append(r["num_voters"]); ys.append(r["full_game"][k]["seconds"])
                elif k in r:
                    xs.append(r["num_voters"]); ys.append(med(r[k]))
            if xs:
                ax.plot(xs, ys, color=C[i], marker=MARKERS[i], label=lab)
        ax.axhline(1.0, color=MUTED, lw=1, ls=":")
        ax.text(1.1e4, 1.15, "1 s interactive target", color=MUTED, fontsize=8)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-3, 40)
        ax.set_xlabel("voters N"); ax.set_ylabel("seconds (median of 3 / 5)")
        ax.set_title(f"time on {'GPU (RTX A5000, cuda:2)' if dev.startswith('cuda') else 'CPU (4 threads)'}")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.36, -0.08), fontsize=8)
    ax = axes[2]
    for dev in devs:
        rs = sorted([r for r in rows if r["device"] == dev], key=lambda r: r["num_voters"])
        xs = [r["num_voters"] for r in rs]
        if dev.startswith("cuda"):
            ax.plot(xs, [r["peak_gpu_generation_gib"] for r in rs], color=C[0], marker="o", label="GPU peak, generation")
            ax.plot(xs, [r["peak_gpu_engine_gib"] for r in rs], color=C[6], marker="v", label="GPU peak, engine + games")
            ax.plot(xs, [r["peak_host_rss_gib"] for r in rs], color=C[1], marker="s", label="host RSS, GPU run")
        else:
            ax.plot(xs, [r["peak_host_rss_gib"] for r in rs], color=C[3], marker="D", ls="--", label="host RSS, CPU run")
    ax.axhline(24, color=MUTED, lw=1, ls=":")
    ax.text(1.1e4, 26, "24 GB A5000", color=MUTED, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-3, 60)
    ax.set_xlabel("voters N"); ax.set_ylabel("GiB"); ax.set_title("peak memory")
    ax.legend(fontsize=8, loc="lower right")
    finish(fig, figs / "e7_scale.png", "E7 · scale")


# ----------------------------------------------------------------------------------------- E4 / E6
def plot_e4(curve, suff, figs: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    if curve:
        ks = [c["k"] for c in curve["curve"]]
        cs = [{"w": {"rate": c["win_rate"], "ci95": c["win_ci95"]}} for c in curve["curve"]]
        errbar(axes[0], ks, cs, "w", C[0], f"win rate (n = {curve['num_seeds']})")
        ax2 = axes[0]
        ax2.set_xlabel("forced survey queries k")
        ax2.set_ylabel("win rate (95% CI)")
        ax2.set_title("BayesPlanner forced to ask k queries")
        ax2.set_ylim(0.6, 1.0)
        ax2.legend(loc="lower left")
    if suff:
        for i, pol in enumerate(("gain", "random")):
            s = suff["summary"][pol]
            ks = sorted(int(k) for k in s)
            cs = [{"g": {"rate": s[str(k)]["guaranteed_round1"], "ci95": s[str(k)]["guaranteed_round1_ci"]}} for k in ks]
            errbar(axes[1], ks, cs, "g", C[i + 1], f"{pol} query selection", marker=MARKERS[i + 1], offset=(i - 0.5) * 0.12)
        axes[1].set_xlabel("survey queries k"); axes[1].set_ylabel("share of games (95% CI)")
        axes[1].set_title(f"K_p(GuaranteedVictory) at round 1 (n = {suff['meta']['seeds']})")
        axes[1].set_ylim(0.3, 1.0)
        axes[1].legend(loc="lower right")
    finish(fig, figs / "e4_information.png", "E4 · information sufficiency (consolidated, not rerun)")


def plot_e6(strat, figs: Path):
    s = strat["summary"]
    order = ["random", "uniform", "top3", "bayes_even", "bayes", "ORACLE"]
    labels = {"random": "Random", "uniform": "Uniform", "top3": "Top-3 stat", "bayes_even": "Bayes (even spend)",
              "bayes": "BayesPlanner", "ORACLE": "ORACLE (benchmark)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for i, k in enumerate(order):
        r, ci = s[k]["win_rate"], s[k]["win_ci95"]
        axes[0].barh(i, r, color=C[6] if k == "ORACLE" else C[0], height=0.6)
        axes[0].errorbar(r, i, xerr=err(ci, r), color=INK, capsize=3, lw=1)
        axes[0].text(ci[1] + 0.02, i, f"{100 * r:.1f}%", va="center", fontsize=8, color=INK)
        m = s[k]["mean_expected_share"]
        axes[1].barh(i, m, color=C[6] if k == "ORACLE" else C[2], height=0.6)
        axes[1].text(m + 0.01, i, f"{m:.3f}", va="center", fontsize=8, color=INK)
    for ax in axes:
        ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[k] for k in order]); ax.grid(axis="y")
    axes[0].set_xlim(0, 1.15); axes[0].set_xlabel("win rate (95% Wilson CI)")
    axes[1].axvline(0.5, color=MUTED, lw=1, ls="--"); axes[1].set_xlim(0, 0.7); axes[1].set_xlabel("mean expected share E")
    finish(fig, figs / "e6_strategies.png", f"E6 · strategy ranking (consolidated; n = {strat['num_seeds']} paired seeds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports/v2")
    ap.add_argument("--figs", default="reports/v2/figures")
    ap.add_argument("--only", default="e1,e2,e3,e4,e5,e6,e7")
    args = ap.parse_args()
    rep, figs = Path(args.reports), Path(args.figs)
    figs.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(","))
    jobs = {"e1": ("e1_topology.json", plot_e1), "e2": ("e2_zealots.json", plot_e2), "e3": ("e3_cascades.json", plot_e3),
            "e5": ("e5_budget.json", plot_e5), "e7": ("e7_scale.json", plot_e7)}
    for key, (fname, fn) in jobs.items():
        if key in only:
            d = load(rep / fname)
            if d is None:
                print("missing", fname)
            else:
                fn(d, figs)
    if "e4" in only:
        plot_e4(load(rep / "info_curve_r12.json"), load(rep / "info_sufficiency.json"), figs)
    if "e6" in only:
        plot_e6(load(rep / "strategies_r12.json"), figs)


if __name__ == "__main__":
    main()
