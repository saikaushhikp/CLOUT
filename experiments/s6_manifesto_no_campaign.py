"""S6 — No campaign, five fixed manifestos: how does vote share drift when the party only commits a manifesto?

Command (reproduces reports/v3/experimentation/expRes_for_slides/manifesto_no_campaign.{json,png,svg} and the log):
  OMP_NUM_THREADS=4 python experiments/s6_manifesto_no_campaign.py --seeds 50 --first-seed 14000 --device cuda:0 \
      2>&1 | tee reports/v3/experimentation/expRes_for_slides/logs/manifesto_no_campaign.log
10-round variant (writes manifesto_no_campaign_10r.* instead; same seeds):
  OMP_NUM_THREADS=4 python experiments/s6_manifesto_no_campaign.py --seeds 50 --first-seed 14000 --device cuda:0 \
      --rounds 10 --tag manifesto_no_campaign_10r \
      2>&1 | tee reports/v3/experimentation/expRes_for_slides/logs/manifesto_no_campaign_10r.log

Design: N = 2,500 voters, default 4 regions / rules / zealots (5%), no workers, no media, no surveys; held-out seeds
14000–14049. Five FIXED manifestos per world, committed at round 0 and never changed, built from the TRUE archetype
TEMPLATES of the regions (what a player who guessed every region right would blend), not the regions' realised means:
  * "1 region":  the largest region's archetype template;
  * "2/3/4 regions": the UI's "Blend my hunches" (``x_playbook.ui_blend``: voter-weighted mean of the chosen regions'
    templates, rounded to integer percents) over the 2 / 3 / 4 largest regions;
  * "Balanced": exactly 20% per sector.
"1 region" goes through the same integer-percent rounding (templates are already whole percents, so it is exact).
Dynamics: social-only steps (``r2_drift_no_campaign.social_run`` = the engine's no-spend round: DeGroot step,
zealots frozen). Expected share E = mean vote probability. Rounds 0–R are the game (``--rounds R``, default 5);
10 and 20 (those > R) are context only.
A parity check replays the first 3 worlds through the full batched game engine (``x_batch.run_batch``, T = R, a
policy that never spends) and compares the share trajectories.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from r2_drift_no_campaign import social_run  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_playbook import ui_blend, ui_percent_to_manifesto, ui_regions_by_size, ui_to_integer_simplex  # noqa: E402
from x_policies import Sequenced  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import ARCHETYPES, REGIONS, SECTORS  # noqa: E402
from dumbocracy.strategies.worlds import _union_graph  # noqa: E402

OUT_DIR = xc.REPORTS / "experimentation" / "expRes_for_slides"
NAME = "manifesto_no_campaign"
MANIFESTOS = ("1 region", "2 regions", "3 regions", "4 regions", "Balanced")
GAME_ROUNDS = (0, 1, 2, 3, 4, 5)   # default; overridden by --rounds
EXTRA_ROUNDS = (10, 20)
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']


def manifestos_for(e) -> tuple[list[list[float]], dict]:
    sizes = [int(x) for x in e.sizes.tolist()]
    arche = [ARCHETYPES[int(i)] for i in e.archetype_of_region.tolist()]
    obs = {"regions": [{"id": REGIONS[r], "voters": sizes[r]} for r in range(4)],
           "archetype_library": [{"id": a.id, "mean": list(a.mean)} for a in ARCHETYPES]}
    hunches = {REGIONS[r]: arche[r].id for r in range(4)}          # the TRUE templates
    order = ui_regions_by_size(obs)
    out = [ui_percent_to_manifesto(ui_to_integer_simplex(list(ARCHETYPES[[a.id for a in ARCHETYPES].index(hunches[order[0]])].mean)))]
    for k in (2, 3, 4):
        out.append(ui_blend(obs, hunches, order[:k]))
    out.append([0.2] * 5)
    info = {"sizes": dict(zip(REGIONS, sizes)), "archetypes": {REGIONS[r]: f"{arche[r].id} {arche[r].name}" for r in range(4)},
            "regions_by_size": order}
    return out, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=14000)
    ap.add_argument("--voters", type=int, default=2500)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--tag", default=NAME)
    ap.add_argument("--rounds", type=int, default=5, help="game rounds R: record/plot 0..R, parity at T=R")
    args = ap.parse_args()
    global GAME_ROUNDS, EXTRA_ROUNDS
    R = args.rounds
    assert R >= 1
    GAME_ROUNDS = tuple(range(R + 1))
    EXTRA_ROUNDS = tuple(t for t in EXTRA_ROUNDS if t > R)
    assert args.voters <= 10_000
    t0 = time.time()
    device, dev_rec = xc.select_device(args.device, need_bytes=2 * 2**30)
    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(electorate=ElectorateConfig(num_voters=args.voters))
    print(f"[{args.tag}] device={device} seeds={seeds[0]}..{seeds[-1]} N={args.voters}", flush=True)

    worlds = []
    for s in seeds:
        cfg = dataclasses.replace(base, seed=s)
        e = generate_electorate(cfg, device=device)
        bs, info = manifestos_for(e)
        worlds.append({"seed": s, "e": e, "b": torch.tensor(bs, dtype=e.preferences.dtype, device=device), "info": info})
    B, n = len(worlds), args.voters
    es = [w["e"] for w in worlds]
    graph = _union_graph([e.graph for e in es], n) if B > 1 else es[0].graph
    v0 = torch.cat([e.preferences for e in es])[None]                                    # [1, B·n, K]
    weight = (base.dynamics.social_step * torch.cat([e.susceptibility.social for e in es]))[None]
    frozen = torch.cat([e.susceptibility.zealot.to(torch.bool) for e in es])[None]
    bmat = torch.stack([w["b"] for w in worlds])                                         # [B, M, K]
    beta, z = base.voting.sensitivity, base.voting.threshold
    record = sorted(set(GAME_ROUNDS + EXTRA_ROUNDS))
    E = {}                                                                                # round -> [B, M]

    def on_round(t, v):
        if t in record:
            vb = v[0].view(B, n, 1, -1)
            sim = 1 - 0.5 * (vb - bmat[:, None]).abs().sum(-1)                           # [B, n, M]
            E[t] = torch.sigmoid(beta * (sim - z)).mean(1).double().cpu()

    with torch.no_grad():
        social_run(v0, graph, weight, frozen, max(record), base.dynamics.social_rule, base.dynamics.wmsr_trim, on_round)

    # parity with the full game engine (no-spend policy, T = R) on the first 3 worlds, each manifesto
    par_w = worlds[:3]
    cfg5 = dataclasses.replace(base, rounds=R)
    maxdiff = 0.0
    for j in range(len(MANIFESTOS)):
        outs = run_batch([w["e"] for w in par_w], cfg5, torch.stack([w["b"][j] for w in par_w]),
                         Sequenced(("none",)), seeds=[w["seed"] for w in par_w])
        for i, o in enumerate(outs):
            for t in GAME_ROUNDS:
                maxdiff = max(maxdiff, abs(o.share_trajectory[t] - float(E[t][i, j])))
    parity = {"worlds": [w["seed"] for w in par_w], "rounds": list(GAME_ROUNDS), "max_abs_diff_E": maxdiff,
              "ok": maxdiff < 1e-4}
    print(f"[{args.tag}] parity vs run_batch: max |ΔE| = {maxdiff:.2e} ok={parity['ok']}", flush=True)

    series = {}
    for j, m in enumerate(MANIFESTOS):
        series[m] = {}
        for t in record:
            vals = [100 * float(E[t][b, j]) for b in range(B)]
            st = xc.mean_ci(vals)
            st["above_50pct_worlds"] = sum(v >= 50 for v in vals)
            series[m][str(t)] = st
        line = " ".join(f"r{t}={series[m][str(t)]['mean']:.1f}" for t in record)
        print(f"[{args.tag}] {m:10s} {line}", flush=True)
    per_world = [{"seed": w["seed"], **w["info"],
                  "manifestos": {m: [round(x, 4) for x in w["b"][j].tolist()] for j, m in enumerate(MANIFESTOS)},
                  "E_pct": {m: {str(t): round(100 * float(E[t][b, j]), 3) for t in record} for j, m in enumerate(MANIFESTOS)}}
                 for b, w in enumerate(worlds)]
    best_r5 = {m: sum(1 for b in range(B) if max(range(len(MANIFESTOS)), key=lambda j: float(E[R][b, j])) == j)
               for j, m in enumerate(MANIFESTOS)}
    seconds = time.time() - t0
    report = {
        "experiment": "S6 no campaign, five fixed manifestos",
        "design": {"voters": args.voters, "regions": 4, "seeds": seeds, "campaign": "none (no workers, media, surveys)",
                   "rules": "default GameConfig (zealots 5%, social step 0.15, β=20, z=0.825)",
                   "dynamics": "social-only DeGroot steps, zealots frozen (= engine round with no spend)",
                   "metric": "expected vote share E = mean_i σ(β(S_i − z)), in %; mean over worlds, t-based 95% CI",
                   "game_rounds": list(GAME_ROUNDS), "context_rounds": list(EXTRA_ROUNDS)},
        "manifestos": {
            "archetype_vector_used": "archetype TEMPLATE means (sectors.ARCHETYPES), not the regions' realised mean v(0)",
            "1 region": "template of the largest region (integer percents)",
            "2 regions": "x_playbook.ui_blend over the 2 largest regions (voter-weighted mean of templates → integer %)",
            "3 regions": "ui_blend over the 3 largest regions",
            "4 regions": "ui_blend over all four regions",
            "Balanced": "0.2 each sector", "sectors": list(SECTORS),
            "example_world": per_world[0]},
        "series_pct": series,
        f"best_manifesto_at_round{R}_worlds": best_r5,
        "parity_check_vs_game_engine": parity,
        "per_world": per_world,
        "provenance": xc.provenance_v3(args, device, seconds, device_record=dev_rec,
                                       extra={"command": "OMP_NUM_THREADS=4 python " + " ".join(sys.argv)}),
    }
    path = out_dir / f"{args.tag}.json"
    path.write_text(json.dumps(report, indent=1, default=float))
    print(f"[{args.tag}] saved {path} ({seconds:.1f}s)", flush=True)
    plot(report, out_dir / args.tag)


def plot(report: dict, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    xs = list(GAME_ROUNDS)
    R = xs[-1]
    ends, crossings, lows, highs = [], [], [], []
    for j, m in enumerate(MANIFESTOS):
        s = report["series_pct"][m]
        mu = [s[str(t)]["mean"] for t in xs]
        lo = [s[str(t)]["ci95"][0] for t in xs]
        hi = [s[str(t)]["ci95"][1] for t in xs]
        ax.fill_between(xs, lo, hi, color=PALETTE[j], alpha=0.15, lw=0)
        ax.plot(xs, mu, color=PALETTE[j], lw=2.5, marker="o", ms=5, label=m)
        ends.append((mu[-1], m, PALETTE[j]))
        lows.append(min(lo)); highs.append(max(hi))
        k = next((t for t, v in zip(xs, mu) if v >= 50), None)
        if k is not None:
            crossings.append((k, m, PALETTE[j], mu[xs.index(k)]))
    ax.axhline(50, color="0.3", ls="--", lw=1.3)
    for k, m, c, v in crossings:   # mark the first round each mean line reaches 50%
        ax.plot([k], [v], marker="*", ms=16, color=c, mec="black", mew=0.8, zorder=5)
    if crossings:   # lines occupy the right end near 50%: put the label top-left, clear of the data
        ax.text(-0.1, 50.5, "50% = win", color="0.3", fontsize=13, va="bottom", ha="left")
    else:
        ax.text(R + 0.1, 49.4, "50% = win", color="0.3", fontsize=13, va="top", ha="right")
    ax.set_xlabel("Round")
    ax.set_ylabel("Expected vote share (%)")
    ax.set_xticks(xs)
    if not crossings:
        title = "No campaign: mean share stays below 50%"
    elif len(crossings) == 1:
        title = f"No campaign: {crossings[0][1]} passes 50% at round {crossings[0][0]}"
    else:
        title = f"No campaign: {len(crossings)} manifestos pass 50% by round {max(c[0] for c in crossings)}"
    ax.set_title(title, fontsize=17)
    ax.set_ylim(min(28, int(min(lows)) - 1), max(52, int(max(highs)) + 2))
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.2, R + 0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=14, title="Manifesto",
              title_fontsize=14)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {stem}.png/.svg", flush=True)


if __name__ == "__main__":
    main()
