"""Consensus towards the manifesto: how the campaign pulls voters' preferences to b, round by round.

For each seed (X0 seed block, N = 10⁴, T = 5, ₫500, ORACLE manifesto b) and each campaign
(none, media only, workers only, split, ORACLE planner), record after every round t = 0..T:
  * S̄       mean agreement with b, S_i = 1 − TV(v_i, b)
  * won      share of voters with S_i ≥ z (they vote for b with chance ≥ ½)
  * E        expected vote share (mean of σ(β_v(S_i − z)))
  * spread   mean TV distance of voters from their own population mean (disagreement; consensus → 0)
  * S̄ per region
Plus, for the first seed, agreement histograms and a 2-D projection of 1,500 sampled voters at t = 0 and t = T.
The figures are drawn from the saved JSON only (``--plot-only``).

usage: OMP_NUM_THREADS=4 python experiments/x_consensus.py --seeds 50 --first-seed 10000 --device cuda:2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch

import x_common as xc
from balance import POLICIES, oracle_manifesto
from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.electorate import generate_electorate
from dumbocracy.simulate import simulate_world

ARMS = {"none": "no campaign", "pay_media": "media only", "pay_workers": "workers only",
        "pay_split": "split (half/half)", "planner": "ORACLE planner"}
OUT = xc.REPORTS / "x_consensus.json"
BINS = 40


def metrics(v: torch.Tensor, b: torch.Tensor, region_of: torch.Tensor, cfg: GameConfig) -> dict:
    s = 1.0 - 0.5 * (v - b).abs().sum(1)
    z, beta = cfg.voting.threshold, cfg.voting.sensitivity
    centre = v.mean(0, keepdim=True)
    return {"S": float(s.mean()), "won": float((s >= z).float().mean()),
            "E": float(torch.sigmoid(beta * (s - z)).mean()),
            "spread": float((0.5 * (v - centre).abs().sum(1)).mean()),
            "S_region": [float(s[region_of == r].mean()) for r in range(4)]}


def play(e, cfg, b, name):
    states = []

    def capture(ctx):
        states.append(ctx.preferences.clone())
        return POLICIES[name](ctx)

    out = simulate_world(e, cfg, b, capture)
    states.append(out.final_preferences.clone())
    return states, out


def mean_ci(xs: list[float]) -> list[float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = 1.96 * sd / math.sqrt(n) if n > 1 else 0.0
    return [m, m - h, m + h]


def run(args) -> dict:
    device = torch.device(args.device)
    torch.set_num_threads(4)
    base = GameConfig(device=str(device), rounds=5, electorate=ElectorateConfig(num_voters=args.n))
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    per_seed = {a: [] for a in ARMS}
    example = {}
    t0 = time.time()
    for k, seed in enumerate(seeds):
        cfg = dataclasses.replace(base, seed=seed)
        e = generate_electorate(cfg, device=device)
        b = oracle_manifesto(e, cfg).to(device)
        for arm in ARMS:
            states, out = play(e, cfg, b, arm)
            rows = [metrics(v, b, e.region_of, cfg) for v in states]
            per_seed[arm].append({"seed": seed, "rounds": rows, "won_election": bool(out.result.victory)})
            if k == 0 and arm in ("none", "planner"):
                s0 = (1 - 0.5 * (states[0] - b).abs().sum(1)).cpu()
                sT = (1 - 0.5 * (states[-1] - b).abs().sum(1)).cpu()
                edges = torch.linspace(0.3, 1.0, BINS + 1)
                example[arm] = {"hist_edges": edges.tolist(),
                                "hist_t0": torch.histc(s0, BINS, 0.3, 1.0).tolist(),
                                "hist_tT": torch.histc(sT, BINS, 0.3, 1.0).tolist()}
                if arm == "planner":
                    g = torch.Generator().manual_seed(seed)
                    idx = torch.randperm(e.num_voters, generator=g)[:1500]
                    x0, xT = states[0].cpu().double(), states[-1].cpu().double()
                    mu = x0.mean(0)
                    _, _, V = torch.linalg.svd(x0 - mu, full_matrices=False)
                    P = V[:2].T
                    proj = lambda x: ((x - mu) @ P).tolist()
                    at_b = sT >= 0.98   # converts and workers sit (almost) exactly at b
                    example["projection"] = {"region": e.region_of[idx].cpu().tolist(), "t0": proj(x0[idx]),
                                             "tT": proj(xT[idx]), "b": proj(b.cpu().double()[None])[0],
                                             "at_b_tT": at_b[idx].tolist(), "count_at_b_tT": int(at_b.sum()),
                                             "count_at_b_t0": int((s0 >= 0.98).sum())}
        if (k + 1) % 10 == 0:
            print(f"[x_consensus] {k + 1}/{len(seeds)} seeds, {time.time() - t0:.0f}s", flush=True)
    T = base.rounds
    summary = {}
    for arm, rows in per_seed.items():
        summary[arm] = {m: [mean_ci([r["rounds"][t][m] for r in rows]) for t in range(T + 1)]
                        for m in ("S", "won", "E", "spread")}
        summary[arm]["S_region"] = [[mean_ci([r["rounds"][t]["S_region"][g] for r in rows])[0] for g in range(4)]
                                    for t in range(T + 1)]
        summary[arm]["win_rate"] = sum(r["won_election"] for r in rows) / len(rows)
    return {"experiment": "x_consensus", "question": "How far does each campaign pull voters towards b?",
            "seeds": {"first": seeds[0], "count": len(seeds)}, "n": args.n, "rounds": T, "arms": ARMS,
            "threshold_z": base.voting.threshold, "metric_definitions": {
                "S": "mean over voters of 1 − TV(v_i, b)", "won": "share of voters with S_i ≥ z",
                "E": "mean of σ(β_v(S_i − z))", "spread": "mean TV distance of v_i from the population mean"},
            "summary": summary, "example_seed": seeds[0], "example": example, "per_seed": per_seed,
            "provenance": xc.provenance_v3(args, device, time.time() - t0)}


def plot(d: dict, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_v3 import INK2, SERIES, _style, save_figure
    out = []
    T = d["rounds"]
    ts = list(range(T + 1))
    arms = list(d["arms"])
    colors = dict(zip(arms, [SERIES[4], SERIES[1], SERIES[2], SERIES[3], SERIES[0]]))

    # 1. consensus towards b: agreement and share won over, by round
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for arm in arms:
        s = d["summary"][arm]
        for ax, m in zip(axes, ("S", "won", "spread")):
            mean = [x[0] for x in s[m]]
            lo, hi = [x[1] for x in s[m]], [x[2] for x in s[m]]
            ax.plot(ts, mean, marker="o", ms=4, color=colors[arm], label=d["arms"][arm])
            ax.fill_between(ts, lo, hi, color=colors[arm], alpha=0.15, lw=0)
    axes[0].set_title("Mean agreement with the manifesto  S̄ = 1 − TV(v, b)", fontsize=10)
    axes[1].set_title(f"Voters won over (S ≥ z = {d['threshold_z']})", fontsize=10)
    axes[1].axhline(0.5, color=INK2, lw=0.8, ls=":")
    axes[2].set_title("Disagreement (mean TV distance to the population mean)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("round")
        ax.set_xticks(ts)
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Consensus towards the manifesto ({d['seeds']['count']} seeds, N = {d['n']:,}, mean and 95% CI)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.append(save_figure(fig, out_dir / "xc_consensus_rounds.png"))

    # 2. one world: agreement distribution before/after, and voters moving in a 2-D projection
    ex = d["example"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    edges = ex["none"]["hist_edges"]
    mids = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    w = edges[1] - edges[0]
    ax = axes[0]
    ax.bar(mids, ex["planner"]["hist_t0"], width=w, color=SERIES[4], alpha=0.55, label="round 0 (before campaign)")
    ax.stairs(ex["none"]["hist_tT"], edges, color=INK2, lw=1.2, label=f"round {T}, no campaign")
    ax.bar(mids, ex["planner"]["hist_tT"], width=w, color=SERIES[0], alpha=0.65, label=f"round {T}, ORACLE planner")
    ax.stairs(ex["none"]["hist_tT"], edges, color=INK2, lw=1.2)
    ax.axvline(d["threshold_z"], color=SERIES[1], ls="--", lw=1)
    ax.text(d["threshold_z"] + 0.005, ax.get_ylim()[1] * 0.92, "vote threshold z", color=SERIES[1], fontsize=8)
    ax.set_xlabel("agreement with b,  S_i = 1 − TV(v_i, b)")
    ax.set_ylabel("voters")
    ax.set_title(f"Seed {d['example_seed']}: who agrees with the manifesto", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style(ax)
    pr = ex["projection"]
    reg_colors = [SERIES[2], SERIES[3], SERIES[6], SERIES[7]]
    for ax, key, title in ((axes[1], "t0", "round 0"), (axes[2], "tT", f"round {T} (ORACLE planner)")):
        xs = [p[0] for p in pr[key]]
        ys = [p[1] for p in pr[key]]
        ax.scatter(xs, ys, s=4, c=[reg_colors[r] for r in pr["region"]], alpha=0.55, lw=0)
        ax.scatter([pr["b"][0]], [pr["b"][1]], marker="*", s=300, facecolor="none", edgecolor=INK2, lw=1.2,
                   zorder=6)
        if key == "tT":
            n_at_b = pr["count_at_b_tT"]
            ax.annotate(f"manifesto b\n{n_at_b:,} voters converted\nor hired sit here", pr["b"], xytext=(14, 18),
                        textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
        else:
            ax.annotate("manifesto b", pr["b"], xytext=(8, 8), textcoords="offset points", fontsize=8)
        ax.set_title(f"Voter preferences, {title}", fontsize=10)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        _style(ax)
    lims = [min(axes[1].get_xlim()[0], axes[2].get_xlim()[0]), max(axes[1].get_xlim()[1], axes[2].get_xlim()[1])]
    limy = [min(axes[1].get_ylim()[0], axes[2].get_ylim()[0]), max(axes[1].get_ylim()[1], axes[2].get_ylim()[1])]
    for ax in axes[1:]:
        ax.set_xlim(lims)
        ax.set_ylim(limy)
    for r in range(4):
        axes[1].scatter([], [], s=12, color=reg_colors[r], label=f"region {r + 1}")
    axes[1].legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    out.append(save_figure(fig, out_dir / "xc_consensus_example.png"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=10000)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        d = run(args)
        Path(args.out).write_text(json.dumps(d))
    d = json.loads(Path(args.out).read_text())
    for p in plot(d, xc.REPORTS / "figures"):
        print("wrote", p)


if __name__ == "__main__":
    main()
