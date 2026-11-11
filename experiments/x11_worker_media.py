"""X11 — Worker × media interaction (plan ``docs/v2/plans/implementation_plan.md`` §4 X11; new, N2).

Command (reproduces reports/v3/x11.json, reports/v3/raw/x11.jsonl and reports/v3/logs/x11.log):
  OMP_NUM_THREADS=4 python experiments/x11_worker_media.py --seeds 100 --first-seed 11000 --device cuda:2 \
      --out reports/v3/x11.json 2>&1 | tee reports/v3/logs/x11.log

Question: do workers and media reinforce each other (supermodularity)? N = 10⁴, T = 5, ₫500, ORACLE manifesto
(``oracle_manifesto`` on the true world), 100 paired seeds (block 11000). Every seed is played once per TARGET
REGION r (all four regions), so each (seed, r) is one unit and the personality τ_r of the targeted region varies.

Arms per (seed, r), all spending in region r only, nothing else bought:
* 2×2 factorial in round 1, at matched spend: none; workers (k in r); media (one airing of r, cost m_r = ₫50·4·n_r/N);
  both (the airing and the k workers in round 1). k = max(1, round(m_r / fee)) so that the worker spend matches the
  airing as closely as a whole number of workers allows (the realised spends are recorded per row);
* sequencing: media in round 1 then the k workers in round 2 ("media_then_workers"), the reverse
  ("workers_then_media"), and "both" (both in round 1);
* at worker fees ₫20 (the game) and ₫100. "none" and "media" do not depend on the fee and are shared.

Interaction (per unit) I = Δ(both) − Δ(workers) − Δ(media), Δ = expected share minus that of "none" (same world and
streams: common random numbers). Reported with a paired 95% CI over seeds (the four regions of a seed are averaged,
so the units are independent), per fee, and per personality of the targeted region (cluster bootstrap over seeds).
Mechanism probes from an observer: the mean conversion chance γP over region r at the start of round 2, the converts
in r, and the mean P of region r at the end of round 1.
All games run batched on the GPU through ``x_batch.run_batch`` (one call per arm: 100 seeds × 4 regions).
"""
from __future__ import annotations

import argparse
import dataclasses
import random
import sys
import time
from decimal import Decimal
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x_batch import conversion_chance_rows, run_batch  # noqa: E402
from x_policies import Sequenced  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.dynamics.campaign import media_cost  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import PERSONALITIES  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402
from dumbocracy.voting import vote_probability  # noqa: E402

FEES = ("20.00", "100.00")
SCHEDULES = {"none": ("none",), "workers": ("workers",), "media": ("media",), "both": ("both",),
             "media_then_workers": ("media", "workers"), "workers_then_media": ("workers", "media")}
FEE_FREE = ("none", "media")
RAW_PATH = xc.RAW / "x11.jsonl"


def one_region(r: int) -> tuple[bool, bool, bool, bool]:
    return tuple(i == r for i in range(4))  # type: ignore[return-value]


def prepare(seeds, base: GameConfig, device) -> list[dict]:
    out = []
    for s in seeds:
        cfg = dataclasses.replace(base, seed=s)
        e = generate_electorate(cfg, device=device)
        sizes = [int(x) for x in e.sizes.tolist()]
        out.append({"seed": int(s), "e": e, "b": oracle_manifesto(e, cfg), "sizes": sizes,
                    "personality": [PERSONALITIES[int(i)].id for i in e.personality_of_region],
                    "media_cost": [media_cost(sizes, one_region(r), base.media.base_cost) for r in range(4)]})
    return out


def workers_for(cost: Decimal, fee: Decimal) -> int:
    return max(1, int((cost / fee).to_integral_value(rounding="ROUND_HALF_UP")))


def play(units: list[tuple[dict, int]], cfg: GameConfig, arm: str) -> list[dict]:
    """One batched call: ``units`` = [(world, target region)]."""
    B = len(units)
    es = [w["e"] for w, _ in units]
    n = es[0].num_voters
    fee = cfg.workers.worker_cost
    pols = [Sequenced(SCHEDULES[arm], workers=workers_for(w["media_cost"][r], fee), media_regions=one_region(r),
                      placement=f"region:{r}") for w, r in units]
    target = torch.tensor([r for _, r in units], device=es[0].device)
    vcfg = cfg.voting
    st: dict = {}

    def observer(t: int, info: dict) -> None:
        game = info["game"]
        if t == 0:
            st["in_r"] = game.region_of == target[game.world_of]
            st["ever"] = torch.zeros_like(info["converted"])
        st["ever"] |= info["converted"]
        if t == 0:
            v = info["v"]
            q = conversion_chance_rows(game.campaign, v, game.b_rows, vcfg.sensitivity, vcfg.threshold,
                                       cfg.workers.conversion_rule)
            p = vote_probability(1 - 0.5 * (v - game.b_rows).abs().sum(-1), vcfg.sensitivity, vcfg.threshold)
            mask = st["in_r"] & ~game.zealot
            cnt = mask.view(B, n).sum(1).clamp_min(1)
            st["q_r2"] = (torch.where(mask, q, torch.zeros_like(q)).view(B, n).sum(1) / cnt).cpu()
            st["p_r_end1"] = (torch.where(mask, p, torch.zeros_like(p)).view(B, n).sum(1) / cnt).cpu()
        if t == cfg.rounds - 1:
            st["conv_r"] = (st["ever"] & st["in_r"] & ~info["pinned"]).view(B, n).sum(1).cpu()
            st["conv_all"] = (st["ever"] & ~info["pinned"]).view(B, n).sum(1).cpu()

    outs = run_batch(es, cfg, torch.stack([w["b"] for w, _ in units]), pols, seeds=[w["seed"] for w, _ in units],
                     observer=observer)
    rows = []
    for b, ((w, r), o) in enumerate(zip(units, outs)):
        hires = sum(a.total_hires for a in o.allocations)
        airings = sum(sum(a.media_regions) for a in o.allocations)
        rows.append({"seed": w["seed"], "region": r, "personality": w["personality"][r], "arm": arm,
                     "fee": str(fee), "expected": o.result.expected_share, "victory": bool(o.result.victory),
                     "workers": int(hires), "airings": int(airings), "spent": str(o.spent),
                     "spent_workers": str(fee * hires), "media_cost_r": str(w["media_cost"][r]),
                     "q_region_start_round2": float(st["q_r2"][b]), "p_region_end_round1": float(st["p_r_end1"][b]),
                     "converts_region": int(st["conv_r"][b]), "converts_total": int(st["conv_all"][b]),
                     "converted_by_round": o.converted, "trajectory": [round(x, 6) for x in o.share_trajectory]})
    return rows


def cluster_boot(values_by_seed: dict[int, list[float]], reps: int = 4000, seed: int = 0) -> list[float]:
    """95% percentile CI of the mean over units, resampling seeds (clusters) with replacement."""
    rng = random.Random(seed)
    keys = list(values_by_seed)
    means = []
    for _ in range(reps):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        vals = [x for k in pick for x in values_by_seed[k]]
        means.append(sum(vals) / len(vals))
    means.sort()
    return [means[int(0.025 * reps)], means[int(0.975 * reps) - 1]]


def summarise(rows: list[dict], seeds: list[int]) -> dict:
    from scipy.stats import wilcoxon
    idx = {(r["seed"], r["region"], r["arm"], r["fee"]): r for r in rows}

    def get(seed, reg, arm, fee):
        return idx[(seed, reg, arm, FEES[0] if arm in FEE_FREE else fee)]

    out: dict = {"arms": {}, "interaction": {}, "sequencing": {}, "by_personality": {}, "tests": {}}
    for fee in FEES:
        for arm in SCHEDULES:
            rs = [get(s, r, arm, fee) for s in seeds for r in range(4)]
            out["arms"][f"{arm}@{fee}"] = {
                "expected": xc.mean_ci([x["expected"] for x in rs]), "win": xc.rate([x["victory"] for x in rs]),
                "workers": xc.mean_ci([x["workers"] for x in rs]),
                "spent_workers": xc.mean_ci([float(x["spent_workers"]) for x in rs]),
                "spent": xc.mean_ci([float(x["spent"]) for x in rs]),
                "q_region_start_round2": xc.mean_ci([x["q_region_start_round2"] for x in rs]),
                "p_region_end_round1": xc.mean_ci([x["p_region_end_round1"] for x in rs]),
                "converts_region": xc.mean_ci([x["converts_region"] for x in rs]),
                "converts_total": xc.mean_ci([x["converts_total"] for x in rs])}
        out["arms"][f"media@{fee}"]["note"] = "fee-free arm (shared)"
        # per-unit effects
        units = {}
        for s in seeds:
            for r in range(4):
                e0 = get(s, r, "none", fee)["expected"]
                d = {a: get(s, r, a, fee)["expected"] - e0 for a in SCHEDULES if a != "none"}
                d["interaction"] = d["both"] - d["workers"] - d["media"]
                d["q_gain_media"] = get(s, r, "media", fee)["q_region_start_round2"] - \
                    get(s, r, "none", fee)["q_region_start_round2"]
                units[(s, r)] = d
        seed_mean = {s: sum(units[(s, r)]["interaction"] for r in range(4)) / 4 for s in seeds}
        vals = list(seed_mean.values())
        nz = [v for v in vals if v != 0]
        p = float(wilcoxon(nz).pvalue) if len(nz) > 1 else 1.0
        rel = [units[(s, r)]["interaction"] / (units[(s, r)]["workers"] + units[(s, r)]["media"])
               for s in seeds for r in range(4) if abs(units[(s, r)]["workers"] + units[(s, r)]["media"]) > 1e-6]
        out["interaction"][fee] = {
            "seed_level": xc.mean_ci(vals), "wilcoxon_p": p,
            "share_units_positive": sum(units[k]["interaction"] > 0 for k in units) / len(units),
            "delta_workers": xc.mean_ci([sum(units[(s, r)]["workers"] for r in range(4)) / 4 for s in seeds]),
            "delta_media": xc.mean_ci([sum(units[(s, r)]["media"] for r in range(4)) / 4 for s in seeds]),
            "delta_both": xc.mean_ci([sum(units[(s, r)]["both"] for r in range(4)) / 4 for s in seeds]),
            "interaction_over_sum_of_main_effects_median": sorted(rel)[len(rel) // 2] if rel else None,
            "q_gain_from_media_round1": xc.mean_ci([units[k]["q_gain_media"] for k in units]),
            "units": len(units)}
        # sequencing (seed level, paired)
        seq = {}
        for a, b in (("media_then_workers", "workers_then_media"), ("both", "media_then_workers"),
                     ("both", "workers_then_media")):
            da = [sum(get(s, r, a, fee)["expected"] for r in range(4)) / 4 for s in seeds]
            db = [sum(get(s, r, b, fee)["expected"] for r in range(4)) / 4 for s in seeds]
            seq[f"{a} − {b}"] = xc.paired(da, db)
            seq[f"{a} − {b}"]["units_a_better"] = sum(get(s, r, a, fee)["expected"] > get(s, r, b, fee)["expected"]
                                                     for s in seeds for r in range(4))
            seq[f"{a} − {b}"]["mcnemar_wins"] = xc.mcnemar([get(s, r, a, fee)["victory"] for s in seeds for r in range(4)],
                                                           [get(s, r, b, fee)["victory"] for s in seeds for r in range(4)])
        out["sequencing"][fee] = seq
        # personality of the targeted region
        pers = {}
        for pid in sorted({r["personality"] for r in rows}):
            by_seed: dict[int, list[float]] = {}
            for (s, r), d in units.items():
                if get(s, r, "none", fee)["personality"] == pid:
                    by_seed.setdefault(s, []).append(d["interaction"])
            flat = [x for v in by_seed.values() for x in v]
            nzp = [x for x in flat if x != 0]
            pers[pid] = {"name": next(pp.name for pp in PERSONALITIES if pp.id == pid), "units": len(flat),
                         "seeds": len(by_seed), "interaction": xc.mean_ci(flat),
                         "cluster_boot_ci95": cluster_boot(by_seed), "wilcoxon_p": float(wilcoxon(nzp).pvalue) if len(nzp) > 1 else 1.0,
                         "delta_workers": xc.mean_ci([units[k]["workers"] for k in units
                                                      if get(*k, "none", fee)["personality"] == pid]),
                         "delta_media": xc.mean_ci([units[k]["media"] for k in units
                                                    if get(*k, "none", fee)["personality"] == pid])}
        out["by_personality"][fee] = pers
    # test families, Holm-corrected
    fam_i = [("interaction", fee, None, out["interaction"][fee]["wilcoxon_p"]) for fee in FEES] + \
            [("interaction", fee, pid, out["by_personality"][fee][pid]["wilcoxon_p"]) for fee in FEES
             for pid in out["by_personality"][fee]]
    fam_s = [("sequencing", fee, k, v["wilcoxon_p"]) for fee in FEES for k, v in out["sequencing"][fee].items()]
    for name, fam in (("interaction", fam_i), ("sequencing", fam_s)):
        adj = xc.holm([x[3] for x in fam])
        out["tests"][name] = [{"fee": f, "subset": k, "p": p, "holm_p": a} for (_, f, k, p), a in zip(fam, adj)]
    out["test_families"] = {
        "interaction": "Wilcoxon signed-rank of the interaction (≠ 0): overall at each fee (seed-level means, 2 tests) "
                       "+ per personality of the targeted region at each fee (unit level, 8 tests); Holm over the 10",
        "sequencing": "Wilcoxon on seed-level expected-share differences, 3 pairs × 2 fees; Holm over the 6"}
    return out


def parity(seeds: list[int], device: torch.device, base: GameConfig) -> dict:
    def fn(seed, dev):
        cfg = dataclasses.replace(base, device=str(dev))
        w = prepare([seed], cfg, dev)[0]
        r = max(range(4), key=lambda i: w["sizes"][i])
        row = play([(w, r)], cfg, "both")[0]
        return {"expected": row["expected"], "converts_total_share_of_n": row["converts_total"] / cfg.electorate.num_voters,
                "q_region_start_round2": row["q_region_start_round2"]}
    out = xc.parity_check(fn, seeds, device=device, abs_tol=0.01)
    out["cell"] = "arm 'both' at ₫20, largest region"
    out["note"] = "statistical tolerance max(0.01, 3·SE); converts scaled by N"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X11"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--out", default=str(xc.REPORTS / "x11.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    if args.device.startswith("cuda") and str(device) != str(torch.device(args.device)):
        raise SystemExit(f"refusing to fall back from {args.device} to {device} (shared host: only the requested GPU)")
    if device.type == "cuda":
        torch.cuda.set_device(device)   # no stray context on the default GPU
    free0 = xc.free_bytes(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text("")
    record: dict = {}
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        worlds = prepare(seeds, base, device)
        record["setup_seconds"] = time.perf_counter() - start
        print(f"[x11] {len(worlds)} worlds + ORACLE manifestos in {record['setup_seconds']:.0f}s", flush=True)
        units = [(w, r) for w in worlds for r in range(4)]
        cal = xc.calibrate_batch(lambda B: play(units[:B], base, "both"), device, sizes=(8, 16), max_batch=len(units))
        record["calibration"] = cal
        rows: list[dict] = []
        for fee in FEES:
            cfg = dataclasses.replace(base, workers=dataclasses.replace(base.workers, worker_cost=Decimal(fee)))
            for arm in SCHEDULES:
                if arm in FEE_FREE and fee != FEES[0]:
                    continue
                t0 = time.perf_counter()
                out = xc.run_chunks(lambda ch: play(ch, cfg, arm), units, cal["batch"], record=record)
                for r in out:
                    xc.write_row(RAW_PATH, {"exp": "X11", **r})
                rows += out
                print(f"[x11] fee {fee} arm {arm}: {time.perf_counter() - t0:.1f}s; mean E "
                      f"{sum(r['expected'] for r in out) / len(out):.4f}", flush=True)
        summary = summarise(rows, seeds)
        for fee in FEES:
            i = summary["interaction"][fee]
            print(f"[x11] fee {fee}: interaction {i['seed_level']['mean']:+.4f} {i['seed_level']['ci95']}", flush=True)
        par = parity(seeds[:args.parity_seeds], device, base)
        print(f"[x11] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    report = {"experiment": "X11", "question": "Worker × media interaction (supermodularity) at matched spend",
              "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
              "n": args.n, "rounds": base.rounds, "treasury": str(base.budget), "fees": list(FEES),
              "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost),
                         "media_base": str(base.media.base_cost)},
              "arms": {k: list(v) for k, v in SCHEDULES.items()},
              "metric_definitions": {
                  "unit": "(seed, targeted region r); every arm spends in r only",
                  "k": "workers = max(1, round(m_r / fee)), m_r = one airing of r = ₫50·4·n_r/N (matched spend)",
                  "delta": "expected share of the arm − expected share of 'none' (same world, same streams)",
                  "interaction": "Δ(both) − Δ(workers) − Δ(media); > 0 = complementarity (supermodular)",
                  "seed_level": "mean over the 4 regions of a seed, then mean and t-CI over the seeds",
                  "q_region_start_round2": "mean γ·P over non-zealot voters of r at the start of round 2",
                  "p_region_end_round1": "mean P over non-zealot voters of r after round 1",
                  "cluster_boot_ci95": "percentile CI, 4000 bootstrap resamples of seeds"},
              **summary,
              "deviations": ["Matched spend is approximate: k = max(1, round(m_r / fee)) workers against one airing of r "
                             "(m_r ≈ ₫30–110 by region size); at ₫100 this is 1 worker (₫100) for every region. The "
                             "realised spends are reported per arm (arms.*.spent_workers).",
                             "All four regions are targeted in turn per seed (400 units) so that the personality "
                             "breakdown has ~100 units per personality; the overall CI uses seed-level means."],
              "config": base.to_record(),
              "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=record.get("batch_sizes"),
                                             free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                             extra={"calibration": record.get("calibration"),
                                                    "setup_seconds": record.get("setup_seconds")})}
    xc.save(args.out, report)
    print(f"[x11] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
