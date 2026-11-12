"""S5 — How round-1 workers spread influence (slide graphs).

Command (reproduces reports/v3/experimentation/expRes_for_slides/worker_spread*.{json,png,svg} and logs/worker_spread.log):
  OMP_NUM_THREADS=4 python experiments/s5_worker_spread.py --seeds 50 --first-seed 14000 --device cuda:0 \
      2>&1 | tee reports/v3/experimentation/expRes_for_slides/logs/worker_spread.log

Design: N = 10⁴, T = 5, default rules/prices/network/zealots; 50 held-out worlds (seeds 14000–14049), one game per
(world, k) on the world's own random streams (``x_batch.run_batch``, full game dynamics, one batched GPU call per k).
* Manifesto = "Blend my hunches" over all four regions with TRUE hunches: the voter-weighted mean of the four
  regions' true archetype means (``x_playbook.ui_blend_raw`` formula), then the UI's integer-percent rounding
  (``ui_to_integer_simplex`` → ``ui_percent_to_manifesto``) — i.e. exactly what the UI commits.
* Workers: k ∈ {0, 5, 10, 20}, all bought in round 1 (``x_policies.WorkerTiming(k, "front", "proportional")``:
  largest-remainder split by region size), library placement rule R22 (random eligible voter, degree ≥ min_degree,
  not a zealot), no later purchases, no media.

Metrics per round t = 0..T (cumulative, per world, then mean ± t-based 95% CI over worlds):
* touched_t = voters with attempts > 0 after round t. ``attempts`` is the engine's per-voter R27 counter
  (``run_batch``: attempts += eff, eff = min(#spreading friends, cap − attempts), 0 for workers/new workers/
  zealots). attempts > 0 ⇔ the voter got ≥ 1 real conversion draw from a worker or convert (the same counter x3's
  ``targeted``/``cap_bind`` use at the final round). Workers themselves and zealots are never touched.
* converted_t = distinct voters converted in rounds ≤ t (union of the per-round ``converted`` masks; workers are
  not counted, they are reported separately).
* E_t = expected vote share = mean over voters of P(vote) on v after round t (``WorldOutcome.share_trajectory``).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_playbook import ui_percent_to_manifesto, ui_to_integer_simplex  # noqa: E402
from x_policies import WorkerTiming  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import ARCHETYPES  # noqa: E402

KS = (0, 5, 10, 20)
OUT_DIR = xc.ROOT / "reports" / "v3" / "experimentation" / "expRes_for_slides"
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']


def blend_manifesto(e) -> tuple[list[float], list[float]]:
    """Size-weighted blend of the four regions' TRUE archetypes (raw), and the UI-rounded committed vector."""
    sizes = [int(x) for x in e.sizes.tolist()]
    acc = [0.0] * 5
    for r in range(4):
        mean = ARCHETYPES[int(e.archetype_of_region[r])].mean
        for j in range(5):
            acc[j] += mean[j] * sizes[r]
    raw = [x / sum(sizes) for x in acc]
    return raw, ui_percent_to_manifesto(ui_to_integer_simplex(raw))


def play(worlds: list[dict], cfg: GameConfig, k: int) -> list[dict]:
    es = [w["e"] for w in worlds]
    B, n, T = len(es), es[0].num_voters, cfg.rounds
    ms = torch.tensor([w["manifesto"] for w in worlds], dtype=es[0].preferences.dtype, device=es[0].device)
    st: dict = {"touched": [], "conv": [], "touched_r": None, "conv_r": None}

    def observer(t: int, info: dict) -> None:
        game = info["game"]
        if t == 0:
            st["ever"] = torch.zeros_like(info["converted"])
        st["ever"] = st["ever"] | info["converted"]
        touched = info["attempts"] > 0
        assert not (st["ever"] & info["pinned"]).any() and not (touched & info["pinned"]).any()
        st["touched"].append(touched.view(B, n).sum(1))
        st["conv"].append(st["ever"].view(B, n).sum(1))
        if t == T - 1:
            key = game.world_of * 4 + game.region_of
            st["touched_r"] = torch.bincount(key[touched], minlength=4 * B).view(B, 4).cpu()
            st["conv_r"] = torch.bincount(key[st["ever"]], minlength=4 * B).view(B, 4).cpu()
            st["workers_r"] = torch.bincount(key[info["pinned"]], minlength=4 * B).view(B, 4).cpu()
            st["zealot_r"] = torch.bincount(key[game.zealot], minlength=4 * B).view(B, 4).cpu()

    outs = run_batch(es, cfg, ms, WorkerTiming(k, "front", "proportional"), seeds=[w["seed"] for w in worlds],
                     observer=observer)
    touched = torch.stack(st["touched"], 1).cpu().tolist()
    conv = torch.stack(st["conv"], 1).cpu().tolist()
    rows = []
    for b, (w, o) in enumerate(zip(worlds, outs)):
        rows.append({"seed": w["seed"], "k": k, "hires": [list(a.hires_by_region) for a in o.allocations],
                     "touched": [0] + [int(x) for x in touched[b]], "converted": [0] + [int(x) for x in conv[b]],
                     "converted_per_round": o.converted, "E": o.share_trajectory,
                     "expected_final": o.result.expected_share, "victory": bool(o.result.victory),
                     "touched_by_region": st["touched_r"][b].tolist(), "converted_by_region": st["conv_r"][b].tolist(),
                     "workers_by_region": st["workers_r"][b].tolist(), "zealots_by_region": st["zealot_r"][b].tolist()})
    return rows


def summarise(rows: list[dict], worlds: list[dict], T: int) -> dict:
    by_seed = {w["seed"]: w for w in worlds}
    out = {}
    for k in KS:
        rs = [r for r in rows if r["k"] == k]
        cell = {"k": k, "n_worlds": len(rs)}
        for key in ("touched", "converted", "E"):
            scale = 100.0 if key == "E" else 1.0
            cell[key + ("_pct" if key == "E" else "")] = [xc.mean_ci([r[key][t] * scale for r in rs]) for t in range(T + 1)]
        cell["win_rate"] = xc.rate([r["victory"] for r in rs])
        base0 = {r["seed"]: r["E"] for r in rows if r["k"] == 0}
        cell["delta_E_pp_vs_k0_paired"] = [xc.mean_ci([100 * (r["E"][t] - base0[r["seed"]][t]) for r in rs])
                                           for t in range(T + 1)]
        # per region, at round T: largest region ("target") vs the other three, and per region rank by size
        tgt_conv, oth_conv, tgt_tch, oth_tch, by_rank = [], [], [], [], {i: {"conv": [], "touched": [], "workers": []}
                                                                         for i in range(4)}
        for r in rs:
            sizes = by_seed[r["seed"]]["sizes"]
            order = sorted(range(4), key=lambda i: -sizes[i])
            big = order[0]
            others = order[1:]
            tgt_conv.append(r["converted_by_region"][big] / sizes[big])
            tgt_tch.append(r["touched_by_region"][big] / sizes[big])
            oth_conv.append(sum(r["converted_by_region"][i] for i in others) / sum(sizes[i] for i in others))
            oth_tch.append(sum(r["touched_by_region"][i] for i in others) / sum(sizes[i] for i in others))
            for rank, i in enumerate(order):
                by_rank[rank]["conv"].append(r["converted_by_region"][i] / sizes[i])
                by_rank[rank]["touched"].append(r["touched_by_region"][i] / sizes[i])
                by_rank[rank]["workers"].append(r["workers_by_region"][i])
        cell["regions_round_T"] = {
            "largest_region_converted_share": xc.mean_ci(tgt_conv),
            "other_regions_converted_share": xc.mean_ci(oth_conv),
            "largest_region_touched_share": xc.mean_ci(tgt_tch),
            "other_regions_touched_share": xc.mean_ci(oth_tch),
            "by_size_rank": {f"rank{i + 1}": {"converted_share": xc.mean_ci(v["conv"]),
                                              "touched_share": xc.mean_ci(v["touched"]),
                                              "workers_mean": sum(v["workers"]) / len(v["workers"])}
                             for i, v in by_rank.items()}}
        out[f"k{k}"] = cell
    return out


def _band(ax, xs, cells, color, ls, label, scale=1.0):
    m = [c["mean"] * scale for c in cells]
    lo = [c["ci95"][0] * scale if c["n"] > 1 else c["mean"] * scale for c in cells]
    hi = [c["ci95"][1] * scale if c["n"] > 1 else c["mean"] * scale for c in cells]
    ax.plot(xs, m, color=color, ls=ls, lw=2.5, marker="o", ms=5, label=label)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)


def plot(summary: dict, T: int, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16, "legend.fontsize": 14,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "svg.fonttype": "none"})
    xs = list(range(T + 1))
    files = []
    # (a) touched vs converted
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    colors = {5: PALETTE[0], 10: PALETTE[1], 20: PALETTE[2]}
    for k in (5, 10, 20):
        c = summary[f"k{k}"]
        _band(ax, xs, c["touched"], colors[k], "--", None)
        _band(ax, xs, c["converted"], colors[k], "-", None)
    handles = [Line2D([], [], color=colors[k], lw=3, label=f"{k} workers") for k in (5, 10, 20)]
    handles += [Line2D([], [], color="0.3", lw=2.5, ls="--", label="touched"),
                Line2D([], [], color="0.3", lw=2.5, ls="-", label="converted")]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=1)
    ax.set_title("Round-1 workers: reach keeps growing for 5 rounds")
    ax.set_xlabel("Round")
    ax.set_ylabel("Voters (cumulative, of 10,000)")
    ax.set_xticks(xs)
    ax.set_xlim(0, T)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        p = out_dir / f"worker_spread_touched_converted.{ext}"
        fig.savefig(p, dpi=200, facecolor="white")
        files.append(str(p))
    plt.close(fig)
    # (b) expected vote share
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    colors = {0: "0.45", 5: PALETTE[0], 10: PALETTE[1], 20: PALETTE[2]}
    for k in KS:
        _band(ax, xs, summary[f"k{k}"]["E_pct"], colors[k], "-", "no workers" if k == 0 else f"{k} workers")
    ax.axhline(50, color=PALETTE[5], ls="--", lw=1.8)
    ax.text(T, 50, "50% to win ", color=PALETTE[5], ha="right", va="bottom", fontsize=14)
    ax.set_title("Workers lift the expected vote share")
    ax.set_xlabel("Round")
    ax.set_ylabel("Expected vote share (%)")
    ax.set_xticks(xs)
    ax.set_xlim(0, T)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        p = out_dir / f"worker_spread_vote_share.{ext}"
        fig.savefig(p, dpi=200, facecolor="white")
        files.append(str(p))
    plt.close(fig)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=14000)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="worker_spread", help="JSON file stem (smoke tests use another tag)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.n > 10_000:
        raise SystemExit("N must not exceed 10,000")
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    if args.device.startswith("cuda") and str(device) != str(torch.device(args.device)):
        raise SystemExit(f"refusing to fall back from {args.device} to {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    free0 = xc.free_bytes(device)
    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        worlds = []
        for s in seeds:
            e = generate_electorate(dataclasses.replace(base, seed=s), device=device)
            raw, ui = blend_manifesto(e)
            worlds.append({"seed": s, "e": e, "sizes": [int(x) for x in e.sizes.tolist()],
                           "archetypes": [ARCHETYPES[int(a)].id for a in e.archetype_of_region.tolist()],
                           "blend_raw": raw, "manifesto": ui})
        rows = []
        for k in KS:
            t0 = time.perf_counter()
            rs = play(worlds, base, k)
            rows += rs
            m = lambda key, t: sum(r[key][t] for r in rs) / len(rs)  # noqa: E731
            print(f"[s5] k={k:2d}: {time.perf_counter() - t0:.1f}s | touched by round "
                  f"{[round(m('touched', t)) for t in range(base.rounds + 1)]} | converted "
                  f"{[round(m('converted', t)) for t in range(base.rounds + 1)]} | E% "
                  f"{[round(100 * m('E', t), 2) for t in range(base.rounds + 1)]}", flush=True)
        summary = summarise(rows, worlds, base.rounds)
    seconds = time.perf_counter() - start
    files = [] if args.no_plots else plot(summary, base.rounds, out_dir)
    report = {
        "experiment": "S5 worker spread (slides)",
        "question": "How do workers bought only in round 1 (no media, no later buys) spread influence?",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget), "ks": list(KS),
        "prices": {"worker": str(base.workers.worker_cost)},
        "rules": {"attempt_cap": base.workers.attempt_cap, "min_degree": base.workers.min_degree,
                  "conversion_rule": base.workers.conversion_rule},
        "manifesto": ("'Blend my hunches' over all four regions with TRUE hunches: voter-weighted mean of the four "
                      "regions' true archetype means (x_playbook.ui_blend_raw formula, computed from "
                      "Electorate.archetype_of_region and region sizes), then UI integer-percent rounding "
                      "(ui_to_integer_simplex -> ui_percent_to_manifesto) = the vector the UI commits"),
        "workers": "k bought in round 1 only, WorkerTiming(k, 'front', 'proportional') = largest-remainder split by "
                   "region size; library pick rule R22 (uniform among non-zealot, non-worker voters with degree >= "
                   "min_degree in that region); no media; game's own random streams (one game per world and k)",
        "metric_definitions": {
            "touched": "cumulative # voters with engine attempts counter > 0 after round t (>= 1 real conversion "
                       "draw from a worker or convert; attempts += min(#spreading friends, cap - attempts), 0 for "
                       "workers and zealots). Excludes workers and zealots. Round 0 = 0.",
            "converted": "cumulative # distinct voters converted in rounds <= t (workers excluded). Round 0 = 0.",
            "E_pct": "expected vote share in % = 100 x mean over voters of P(vote | v after round t) "
                     "(WorldOutcome.share_trajectory; round 0 = before any action)",
            "regions_round_T": "after round T: converted (touched) voters / region size; 'largest' = the largest "
                               "region of each world (gets the most workers), 'other' = the other three pooled; "
                               "by_size_rank: rank1 = largest region",
            "delta_E_pp_vs_k0_paired": "per world: E_pct(k) - E_pct(k=0), same world (paired), mean ± 95% CI",
            "ci": "t-based 95% CI over the worlds (x_common.mean_ci)"},
        "summary": summary,
        "worlds": [{"seed": w["seed"], "sizes": w["sizes"], "archetypes": w["archetypes"],
                    "blend_raw": [round(x, 6) for x in w["blend_raw"]], "manifesto": [round(x, 6) for x in w["manifesto"]]}
                   for w in worlds],
        "per_game": rows,
        "files": files,
        "provenance": {"command": "OMP_NUM_THREADS=4 python " + " ".join(sys.argv),
                       **xc.provenance_v3(args, device, seconds, free_at_start=free0, monitor=monitor,
                                          device_record=dev_record)},
    }
    xc.save(out_dir / f"{args.tag}.json", report)
    print(f"[s5] done in {seconds:.1f}s; wrote {out_dir / (args.tag + '.json')} and {len(files)} figure files", flush=True)


if __name__ == "__main__":
    main()
