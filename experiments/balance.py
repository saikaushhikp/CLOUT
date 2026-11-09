"""P2 balance gate (spec §12): does the default economy make decisions matter?

Usage:
  python experiments/balance.py --seeds 200 --n 10000 --device cuda:2 [--set workers.transmission=0.03 ...]

Policies (all use the available workers = free + permanent hires so far, R16; they differ in how the
*paid* budget is spent — every extra worker bought is a one-time hire that keeps working in later rounds):
  none          no campaign at all (free workers idle)
  free_even     free workers split evenly, no spending
  pay_workers   budget paced evenly over rounds, all on extra workers (split by region size)
  pay_media     budget paced evenly, all on media (largest regions first while affordable)
  pay_split     half workers / half media
  planner       ORACLE marginal-value planner (uses true v): ranks worker and media options by
                expected share gain per unit cost
Manifestos: uniform, top3 (free global statistic only — uninformed), informed (knows the region
archetypes σ — the best a perfect surveyor could learn), oracle (optimised on the true v(0)).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from decimal import Decimal

import torch

from dumbocracy.config import GameConfig, ElectorateConfig, resolve_device
from dumbocracy.dynamics.campaign import Allocation, media_cost
from dumbocracy.electorate import generate_electorate
from dumbocracy.rng import dirichlet
from dumbocracy.sectors import PERSONALITIES, archetype_means
from dumbocracy.simulate import RoundContext, simulate_world
from dumbocracy.strategies.manifesto_opt import optimise_manifesto
from dumbocracy.strategies.oracle import (  # noqa: F401  (re-exported for scripts)
    anticipate, informed_manifesto, oracle_manifesto, synthetic_world, top3_manifesto,
)
from dumbocracy.voting import similarity, vote_probability

NO_MEDIA = (False, False, False, False)


# ----------------------------------------------------------------------------- config overrides
def override(config: GameConfig, path: str, raw: str) -> GameConfig:
    head, _, rest = path.partition(".")
    current = getattr(config, head)
    if rest:
        return dataclasses.replace(config, **{head: override(current, rest, raw)})
    if isinstance(current, Decimal):
        value = Decimal(raw)
    elif isinstance(current, bool):
        value = raw.lower() in ("1", "true", "yes")
    elif isinstance(current, int):
        value = int(raw)
    elif isinstance(current, float):
        value = float(raw)
    elif isinstance(current, tuple):
        value = tuple(float(x) for x in raw.split(","))
    else:
        value = raw
    return dataclasses.replace(config, **{head: value})


# ----------------------------------------------------------------------------- helpers
def split_by_weight(total: int, weights: list[float]) -> tuple[int, int, int, int]:
    """Largest-remainder integer split of ``total`` proportional to ``weights``."""
    s = sum(weights)
    if total <= 0 or s <= 0:
        return (0, 0, 0, 0)
    raw = [total * w / s for w in weights]
    base = [int(math.floor(x)) for x in raw]
    order = sorted(range(4), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[: total - sum(base)]:
        base[i] += 1
    return tuple(base)  # type: ignore[return-value]


def round_cap(ctx: RoundContext) -> Decimal:
    return (ctx.budget / (ctx.rounds - ctx.round)).quantize(Decimal("0.01"), rounding="ROUND_FLOOR")


def sizes_of(ctx: RoundContext) -> list[int]:
    return [int(x) for x in ctx.electorate.sizes.tolist()]


def media_within(ctx: RoundContext, cap: Decimal, order: list[int]) -> tuple[bool, bool, bool, bool]:
    chosen = [False] * 4
    for r in order:
        trial = list(chosen)
        trial[r] = True
        if media_cost(sizes_of(ctx), trial, ctx.config.media.base_cost) <= cap:
            chosen = trial
    return tuple(chosen)  # type: ignore[return-value]


# ----------------------------------------------------------------------------- policies
def policy_none(ctx: RoundContext) -> Allocation:
    return Allocation((0, 0, 0, 0), 0, NO_MEDIA)


def policy_free_even(ctx: RoundContext) -> Allocation:
    return Allocation(split_by_weight(ctx.available_workers, [1, 1, 1, 1]), 0, NO_MEDIA)


def _workers_with(ctx: RoundContext, cap: Decimal, weights: list[float]) -> Allocation:
    extra = int(cap // ctx.config.workers.extra_worker_cost)
    return Allocation(split_by_weight(ctx.available_workers + extra, weights), extra, NO_MEDIA)


def policy_pay_workers(ctx: RoundContext) -> Allocation:
    return _workers_with(ctx, round_cap(ctx), [float(x) for x in sizes_of(ctx)])


def policy_pay_media(ctx: RoundContext) -> Allocation:
    order = sorted(range(4), key=lambda r: -sizes_of(ctx)[r])
    media = media_within(ctx, round_cap(ctx), order)
    return Allocation(split_by_weight(ctx.available_workers, [1, 1, 1, 1]), 0, media)


def policy_pay_split(ctx: RoundContext) -> Allocation:
    cap = round_cap(ctx)
    order = sorted(range(4), key=lambda r: -sizes_of(ctx)[r])
    media = media_within(ctx, cap / 2, order)
    spent = media_cost(sizes_of(ctx), media, ctx.config.media.base_cost)
    workers = _workers_with(ctx, cap - spent, [float(x) for x in sizes_of(ctx)])
    return Allocation(workers.workers_by_region, workers.extra_workers, media)


def policy_planner(ctx: RoundContext) -> Allocation:
    """ORACLE: marginal share gain per unit cost from the true state (first-order in intensity)."""
    cfg, e = ctx.config, ctx.electorate
    sim = similarity(ctx.preferences, ctx.manifesto)
    prob = vote_probability(sim, cfg.voting.sensitivity, cfg.voting.threshold)
    gain = cfg.voting.sensitivity * prob * (1 - prob) * (1 - sim)  # d P_i / d c_i
    sus = e.susceptibility
    q = cfg.workers.contacts_per_worker(e.num_voters)
    worker_value, media_value = [], []
    for r in range(4):
        mask = e.region_of == r
        worker_value.append(float(q * cfg.workers.contact_strength * (sus.campaign[mask] * gain[mask]).mean()))
        eta = cfg.media.strength / (1 + cfg.media.decay * ctx.prior_airings[r])
        media_value.append(float(cfg.media.exposure_prob * eta * (sus.media[mask] * gain[mask]).sum()))
    cap = round_cap(ctx)
    cost_w = cfg.workers.extra_worker_cost
    best_worker = max(worker_value)
    ratios = []
    for r in range(4):
        one = [False] * 4
        one[r] = True
        cost = media_cost(sizes_of(ctx), one, cfg.media.base_cost)
        ratios.append((media_value[r] / float(cost), r, cost))
    media = [False] * 4
    remaining = cap
    for ratio, r, cost in sorted(ratios, reverse=True):
        if ratio > best_worker / float(cost_w) and cost <= remaining:
            media[r] = True
            remaining -= cost
    extra = int(remaining // cost_w)
    weights = [max(v, 0.0) ** 2 for v in worker_value]  # concentrate on the best regions
    return Allocation(split_by_weight(ctx.available_workers + extra, weights), extra, tuple(media))


POLICIES = {"none": policy_none, "free_even": policy_free_even, "pay_workers": policy_pay_workers,
            "pay_media": policy_pay_media, "pay_split": policy_pay_split, "planner": policy_planner}
PAID_PATTERNS = ("pay_workers", "pay_media", "pay_split")


# ----------------------------------------------------------------------------- manifestos
def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def run(config: GameConfig, seeds: list[int], device: torch.device, zealot_sweep: list[float], sweep_seeds: int) -> dict:
    rows = []
    start = time.perf_counter()
    for seed in seeds:
        cfg = dataclasses.replace(config, seed=seed)
        e = generate_electorate(cfg, device=device)
        v0 = e.preferences
        b = {
            "uniform": torch.full((5,), 0.2, dtype=v0.dtype, device=device),
            "top3": top3_manifesto(v0),
            "informed": informed_manifesto(e, cfg),
            "oracle": oracle_manifesto(e, cfg),
        }
        row = {"seed": seed, "personality": [PERSONALITIES[int(i)].id for i in e.personality_of_region],
               "sizes": e.sizes.tolist(), "runs": {}}

        def play(manifesto_name: str, policy_name: str) -> None:
            out = simulate_world(e, cfg, b[manifesto_name], POLICIES[policy_name])
            row["runs"][f"{manifesto_name}/{policy_name}"] = {
                "share": out.result.vote_share, "expected": out.result.expected_share,
                "victory": out.result.victory, "spent": str(out.spent),
                "trajectory": [round(x, 4) for x in out.share_trajectory]}

        play("uniform", "none")
        play("uniform", "planner")
        play("top3", "planner")
        play("informed", "planner")
        play("oracle", "none")
        play("oracle", "planner")
        for name in ("free_even",) + PAID_PATTERNS:
            play("informed", name)
        rows.append(row)
    elapsed = time.perf_counter() - start

    def win_rate(key: str) -> dict:
        wins = sum(r["runs"][key]["victory"] for r in rows)
        lo, hi = wilson(wins, len(rows))
        return {"rate": wins / len(rows), "ci95": [lo, hi],
                "mean_expected_share": sum(r["runs"][key]["expected"] for r in rows) / len(rows)}

    keys = sorted(rows[0]["runs"])
    summary = {k: win_rate(k) for k in keys}
    # criterion 3: informed vs uninformed (top3) manifesto, same planner, paired.
    from scipy.stats import wilcoxon
    diffs = [r["runs"]["informed/planner"]["expected"] - r["runs"]["top3/planner"]["expected"] for r in rows]
    p_value = float(wilcoxon(diffs, alternative="greater").pvalue) if any(d != 0 for d in diffs) else 1.0
    # criterion 4: best paid pattern per seed.
    best_counts = {p: 0 for p in PAID_PATTERNS}
    best_by_personality: dict[str, dict[str, int]] = {}
    for r in rows:
        best = max(PAID_PATTERNS, key=lambda p: r["runs"][f"informed/{p}"]["expected"])
        best_counts[best] += 1
        largest = r["personality"][max(range(4), key=lambda i: r["sizes"][i])]
        best_by_personality.setdefault(largest, {p: 0 for p in PAID_PATTERNS})[best] += 1
    max_pattern_rate = max(summary[f"informed/{p}"]["rate"] for p in PAID_PATTERNS)
    # criterion 5b: zealot sweep for the oracle planner on a subset of seeds.
    zealot = {}
    for frac in zealot_sweep:
        wins, shares = 0, []
        for seed in seeds[:sweep_seeds]:
            cfg = dataclasses.replace(config, seed=seed, electorate=dataclasses.replace(config.electorate, zealot_fraction=frac))
            e = generate_electorate(cfg, device=device)
            bo = oracle_manifesto(e, cfg)
            out = simulate_world(e, cfg, bo, policy_planner)
            wins += out.result.victory
            gain = out.share_trajectory[-1] - out.share_trajectory[0]
            shares.append(gain)
        zealot[str(frac)] = {"win_rate": wins / sweep_seeds, "mean_campaign_gain": sum(shares) / len(shares)}
    gate = {
        "1_uniform_no_campaign_le_30pct": summary["uniform/none"]["rate"] <= 0.30,
        "2_oracle_planner_ge_85pct": summary["oracle/planner"]["rate"] >= 0.85,
        "3_informed_beats_top3_p_lt_0.01": p_value < 0.01 and sum(diffs) > 0,
        "4a_no_paid_pattern_ge_80pct": max_pattern_rate < 0.80,
        "4b_workers_and_media_each_best_sometimes": best_counts["pay_workers"] > 0 and best_counts["pay_media"] > 0,
    }
    return {"num_seeds": len(rows), "seconds": elapsed, "gate": gate, "summary": summary,
            "informed_vs_top3": {"mean_diff": sum(diffs) / len(diffs), "wilcoxon_p": p_value},
            "best_paid_pattern_counts": best_counts, "best_pattern_by_largest_region_personality": best_by_personality,
            "zealot_sweep_oracle_planner": zealot, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", action="append", default=[], help="override, e.g. voting.threshold=0.7")
    parser.add_argument("--zealots", type=float, nargs="*", default=[0.0, 0.05, 0.15, 0.30])
    parser.add_argument("--sweep-seeds", type=int, default=40)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = resolve_device(args.device)
    config = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    for item in args.set:
        key, _, raw = item.partition("=")
        config = override(config, key, raw)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    report = run(config, seeds, device, args.zealots, min(args.sweep_seeds, args.seeds))
    report["config"] = config.to_record()
    printable = {k: report[k] for k in ("num_seeds", "seconds", "gate", "informed_vs_top3", "best_paid_pattern_counts",
                                        "best_pattern_by_largest_region_personality", "zealot_sweep_oracle_planner")}
    printable["win_rates"] = {k: round(v["rate"], 3) for k, v in report["summary"].items()}
    printable["mean_expected"] = {k: round(v["mean_expected_share"], 3) for k, v in report["summary"].items()}
    print(json.dumps(printable, indent=1))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
