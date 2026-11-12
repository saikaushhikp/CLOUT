"""S7 — All media vs all workers, same money, in a single-region electorate.

Command (reproduces reports/v3/experimentation/expRes_for_slides/media_vs_workers.{json,png,svg} and the log):
  OMP_NUM_THREADS=4 python experiments/s7_media_vs_workers.py --seeds 50 --first-seed 14000 --device cuda:0 \
      2>&1 | tee reports/v3/experimentation/expRes_for_slides/logs/media_vs_workers.log

Single region without touching library code: the engine always has 4 regions (``sectors.NUM_REGIONS``), but region
sizes come from ``ElectorateConfig``: with ``min_region_fraction = 0`` (no 15% floor) and a vanishing Dirichlet
concentration ``region_dirichlet_a = 1e-3`` the split is one-hot, i.e. all N voters fall in ONE region and the other
three are empty (checked per world; the script aborts otherwise). Which of N/S/E/W holds everyone, and which archetype
and personality it gets, vary by seed (recorded). The SBM sparsity is recomputed by the library for the realised
sizes, so the mean degree stays ≈ 30.

Arms (common random numbers: same world and seed streams), Balanced manifesto (0.2 each), T = 3, no surveys:
  * "No campaign";
  * "All media": air the region in rounds 1, 2, 3. M = 3 × the airing price charged by the engine
    (``dynamics.campaign.media_cost``, m0·4·n_r/N);
  * "All workers": floor(M / worker fee) workers bought in round 1 in that region (``x_policies.Sequenced``,
    placement region:r), nothing afterwards.
Budget is set to M (the default ₫500 cannot pay three airings of a whole-electorate region).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_policies import Sequenced  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.dynamics.campaign import media_cost  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import ARCHETYPES, PERSONALITIES, REGIONS  # noqa: E402

OUT_DIR = xc.REPORTS / "experimentation" / "expRes_for_slides"
NAME = "media_vs_workers"
ARMS = ("No campaign", "All media", "All workers")
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']
COLORS = {"No campaign": "0.45", "All media": PALETTE[0], "All workers": PALETTE[5]}
ROUNDS = 3


def one_region(r: int) -> tuple[bool, bool, bool, bool]:
    return tuple(i == r for i in range(4))  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=14000)
    ap.add_argument("--voters", type=int, default=2500)
    ap.add_argument("--region-dirichlet-a", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--tag", default=NAME)
    args = ap.parse_args()
    assert args.voters <= 10_000
    t0 = time.time()
    device, dev_rec = xc.select_device(args.device, need_bytes=2 * 2**30)
    out_dir = Path(args.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(rounds=ROUNDS, electorate=ElectorateConfig(
        num_voters=args.voters, min_region_fraction=0.0, region_dirichlet_a=args.region_dirichlet_a))
    print(f"[{args.tag}] device={device} seeds={seeds[0]}..{seeds[-1]} N={args.voters}", flush=True)

    worlds = []
    for s in seeds:
        e = generate_electorate(dataclasses.replace(base, seed=s), device=device)
        sizes = [int(x) for x in e.sizes.tolist()]
        r = max(range(4), key=lambda i: sizes[i])
        if sizes[r] != args.voters:
            raise SystemExit(f"seed {s}: region split {sizes} is not single-region")
        price = media_cost(sizes, one_region(r), base.media.base_cost)
        worlds.append({"seed": s, "e": e, "region": r, "sizes": sizes, "price": price,
                       "archetype": ARCHETYPES[int(e.archetype_of_region[r])],
                       "personality": PERSONALITIES[int(e.personality_of_region[r])].id,
                       "mean_degree": float(e.graph.degree.double().mean()),
                       "zealots": int(e.susceptibility.zealot.sum())})
    prices = {w["price"] for w in worlds}
    assert len(prices) == 1, prices
    price = prices.pop()
    fee = base.workers.worker_cost
    M = ROUNDS * price
    k = int(M // fee)
    leftover = M - k * fee
    cfg = dataclasses.replace(base, budget=M)
    print(f"[{args.tag}] airing price ₫{price}, M = ₫{M}, worker fee ₫{fee} → {k} workers, leftover ₫{leftover}",
          flush=True)

    b = torch.full((len(worlds), 5), 0.2, device=device)
    es = [w["e"] for w in worlds]
    pol = {"No campaign": lambda w: Sequenced(("none",)),
           "All media": lambda w: Sequenced(("media",) * ROUNDS, media_regions=one_region(w["region"])),
           "All workers": lambda w: Sequenced(("workers",), workers=k, placement=f"region:{w['region']}")}
    traj, rows = {}, []
    for arm in ARMS:
        outs = run_batch(es, cfg, b, [pol[arm](w) for w in worlds], seeds=seeds)
        traj[arm] = [[100 * x for x in o.share_trajectory] for o in outs]
        for w, o in zip(worlds, outs):
            hires = sum(a.total_hires for a in o.allocations)
            airings = sum(sum(a.media_regions) for a in o.allocations)
            rows.append({"seed": w["seed"], "arm": arm, "spent": str(o.spent), "workers": int(hires),
                         "airings": int(airings), "victory": bool(o.result.victory),
                         "expected_final_pct": 100 * o.result.expected_share,
                         "trajectory_pct": [round(100 * x, 4) for x in o.share_trajectory],
                         "converted_by_round": o.converted})
        spent = sorted({r["spent"] for r in rows if r["arm"] == arm})
        print(f"[{args.tag}] {arm:12s} spent {spent} " +
              " ".join(f"r{t}={sum(x[t] for x in traj[arm]) / len(worlds):.2f}" for t in range(ROUNDS + 1)), flush=True)

    series = {arm: {str(t): xc.mean_ci([x[t] for x in traj[arm]]) for t in range(ROUNDS + 1)} for arm in ARMS}
    fin = {arm: [x[ROUNDS] for x in traj[arm]] for arm in ARMS}
    diff = xc.paired(fin["All workers"], fin["All media"])
    vic = {arm: sum(r["victory"] for r in rows if r["arm"] == arm) for arm in ARMS}
    above50 = {arm: sum(x >= 50 for x in fin[arm]) for arm in ARMS}
    head = {"workers_higher": sum(a > m for a, m in zip(fin["All workers"], fin["All media"])),
            "media_higher": sum(m > a for a, m in zip(fin["All workers"], fin["All media"])),
            "ties": sum(m == a for a, m in zip(fin["All workers"], fin["All media"]))}
    lift = {arm: xc.paired(fin[arm], fin["No campaign"]) for arm in ("All media", "All workers")}
    split_reg = {REGIONS[r]: sum(w["region"] == r for w in worlds) for r in range(4)}
    split_arch = {}
    for w in worlds:
        key = f"{w['archetype'].id} {w['archetype'].name}"
        split_arch[key] = split_arch.get(key, 0) + 1
    split_pers = {}
    for w in worlds:
        split_pers[w["personality"]] = split_pers.get(w["personality"], 0) + 1
    by_arch = {}
    for key in split_arch:
        idx = [i for i, w in enumerate(worlds) if f"{w['archetype'].id} {w['archetype'].name}" == key]
        by_arch[key] = {arm: round(sum(fin[arm][i] for i in idx) / len(idx), 2) for arm in ARMS}
    print(f"[{args.tag}] round-3 workers − media = {diff['mean']:.2f} pp, CI {diff['ci95'][0]:.2f}..{diff['ci95'][1]:.2f}, "
          f"head-to-head {head}, victories {vic}", flush=True)
    print(f"[{args.tag}] single-region split: region {split_reg} archetype {split_arch} personality {split_pers}",
          flush=True)
    seconds = time.time() - t0
    report = {
        "experiment": "S7 all media vs all workers, same money, single-region electorate",
        "design": {"voters": args.voters, "rounds": ROUNDS, "seeds": seeds, "manifesto": [0.2] * 5,
                   "surveys": "none", "budget": str(M),
                   "metric": "expected vote share E (mean vote probability), %; mean over worlds, t-based 95% CI",
                   "rounds_axis": "round 0 = before round 1; round t = after round t"},
        "single_region": {
            "how": "ElectorateConfig(min_region_fraction=0.0, region_dirichlet_a=%g): Dirichlet split is one-hot, all "
                   "voters in one region, other three empty; no library change" % args.region_dirichlet_a,
            "verified_all_worlds_single_region": True,
            "region_counts": split_reg, "archetype_counts": split_arch, "personality_counts": split_pers,
            "mean_degree": xc.mean_ci([w["mean_degree"] for w in worlds]),
            "zealots": xc.mean_ci([w["zealots"] for w in worlds]),
            "round3_mean_pct_by_archetype": by_arch},
        "money": {"airing_price": str(price), "M": str(M), "worker_fee": str(fee), "workers": k,
                  "leftover": str(leftover),
                  "price_source": "dynamics.campaign.media_cost(sizes, region, media.base_cost) = m0·4·n_r/N"},
        "arms": {"No campaign": "nothing bought",
                 "All media": f"air the region in rounds 1..{ROUNDS} (₫{price} each, total ₫{M})",
                 "All workers": f"{k} workers in round 1 in the region (₫{k * fee}), nothing afterwards"},
        "series_pct": series,
        "round3": {"workers_minus_media_pp": diff, "head_to_head_worlds": head,
                   "engine_victories": vic, "E_at_least_50pct_worlds": above50,
                   "lift_over_no_campaign_pp": lift},
        "per_world": [{"seed": w["seed"], "region": REGIONS[w["region"]],
                       "archetype": f"{w['archetype'].id} {w['archetype'].name}", "personality": w["personality"],
                       "E_pct": {arm: [round(x, 3) for x in traj[arm][i]] for arm in ARMS}}
                      for i, w in enumerate(worlds)],
        "rows": rows,
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
    xs = list(range(ROUNDS + 1))
    for arm in ARMS:
        s = report["series_pct"][arm]
        mu = [s[str(t)]["mean"] for t in xs]
        ax.fill_between(xs, [s[str(t)]["ci95"][0] for t in xs], [s[str(t)]["ci95"][1] for t in xs],
                        color=COLORS[arm], alpha=0.15, lw=0)
        ax.plot(xs, mu, color=COLORS[arm], lw=2.5, marker="o", ms=6, label=arm,
                ls=":" if arm == "No campaign" else "-")
    ax.axhline(50, color="0.3", ls="--", lw=1.3)
    m = report["money"]
    d = report["round3"]["workers_minus_media_pp"]
    h = report["round3"]["head_to_head_worlds"]
    fig.suptitle("Same money: workers beat media by round 3", fontsize=17, x=0.5, y=0.97)
    ax.text(3.1, 49.4, "50% = win", color="0.3", fontsize=13, va="top", ha="right")
    ax.set_ylim(20, 52)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=14)
    fig.text(0.695, 0.56, f"Budget \u20ab{float(m['M']):.0f}\n= 3 airings\n= {m['workers']} workers\n\n"
             f"Round 3: workers\n{d['mean']:+.1f} pp vs media\n(95% CI {d['ci95'][0]:.1f} to {d['ci95'][1]:.1f})\n"
             f"ahead in {h['workers_higher']}/{h['workers_higher'] + h['media_higher'] + h['ties']} worlds",
             fontsize=13, va="top", color="0.2")
    ax.set_xlabel("Round")
    ax.set_ylabel("Expected vote share (%)")
    ax.set_xticks(xs)
    ax.grid(alpha=0.3)
    fig.subplots_adjust(left=0.11, right=0.67, top=0.88, bottom=0.14)
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {stem}.png/.svg", flush=True)


if __name__ == "__main__":
    main()
