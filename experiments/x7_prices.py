"""X7 — Budget and price sensitivity (plan ``docs/v2/plans/implementation_plan.md`` §3 X7; sensitivity study only,
the game keeps ₫20 workers / ₫10 queries, D5).

Command (reproduces reports/v3/x7.json, reports/v3/raw/x7.jsonl and reports/v3/logs/x7.log):
  OMP_NUM_THREADS=4 python experiments/x7_prices.py --seeds 50 --first-seed 10000 --device cuda:0 \
      --out reports/v3/x7.json 2>&1 | tee reports/v3/logs/x7.log

Grid: treasury {500, 750, 1000} × worker fee {₫10, 20, 50, 100} × query fee {₫5, 10}; 50 paired seeds per cell
(seed block 10000, §1.5). N = 10⁴, T = 5.

Players
* ORACLE (reference, not a player): oracle manifesto + ``balance.policy_planner`` on the true state, run batched on
  the GPU through ``x_batch.run_batch``. It asks no queries, so its result does not depend on the query fee: it is
  run once per (treasury, worker fee) and reported for both query-fee columns (stated in the JSON).
  Alongside it, the four scripted paid patterns of the X0 gate (pay_workers, front_workers, pay_media, pay_split,
  informed manifesto, exactly as ``balance.py`` criterion 4) give the per-cell "dominant pattern".
* Balanced player = ``BayesPlanner`` playing full games through ``GameEngine`` (observation dicts only), with the
  cell's config built experiment-side (the ``--set`` equivalent: ``x_common.override``) and passed both to the engine
  and as ``BayesPlanner.config`` (its planner prices workers from the config, not from the observation).
  **Fix from the plan:** ``survey_budget`` is set explicitly in every query-fee arm to ``SURVEY_QUESTIONS`` (= 5)
  questions' worth: ₫25 at ₫5, ₫50 at ₫10, so the number of affordable questions is the same across fees. All other
  BayesPlanner parameters are the library defaults (recorded in the JSON).
  GPU workaround (deviation, library fix awaiting owner approval): ``BayesPlanner(device="cuda")`` crashes in
  ``manifesto()``; the subclass from ``x_gpu_benchmarks._gpu_planner_class`` keeps the manifesto worlds on the CPU
  so the per-round rollouts run on the GPU. Games run in 2 spawned processes sharing the GPU (§1.1: at most 2).

Electorates use the canonical CPU draw (R8, ``generation="cpu"``) moved to the GPU, so the ORACLE batch and the
engine games see bit-identical worlds per seed.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
import x_policies  # noqa: E402
from x_batch import run_batch  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.dynamics.campaign import media_cost  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.oracle import informed_manifesto, oracle_manifesto  # noqa: E402

TREASURIES = ("500.00", "750.00", "1000.00")
WORKER_FEES = ("10.00", "20.00", "50.00", "100.00")
QUERY_FEES = ("5.00", "10.00")
SURVEY_QUESTIONS = 5                       # survey_budget = 5 × query fee (₫25 at ₫5, ₫50 at ₫10)
PATTERNS = ("pay_workers", "front_workers", "pay_media", "pay_split")
DEFAULT_CELL = ("500.00", "20.00", "10.00")
RAW_PATH = xc.RAW / "x7.jsonl"


def survey_budget(query_fee: str) -> Decimal:
    return (Decimal(query_fee) * SURVEY_QUESTIONS).quantize(Decimal("0.01"))


def cell_config(base: GameConfig, treasury: str, worker_fee: str, query_fee: str) -> GameConfig:
    return xc.apply_overrides(base, [f"budget={treasury}", f"workers.worker_cost={worker_fee}",
                                     f"survey.query_cost={query_fee}"])


def cell_key(treasury: str, worker_fee: str, query_fee: str | None) -> str:
    return f"T{int(Decimal(treasury))}_W{int(Decimal(worker_fee))}_Q{'any' if query_fee is None else int(Decimal(query_fee))}"


def dominant(workers: float, media: float) -> str:
    """Spend-mix label: "workers" if ≥ 2/3 of campaign spend went to workers, "media" if ≤ 1/3, else "mixed"."""
    total = workers + media
    if total <= 0:
        return "none"
    share = workers / total
    return "workers" if share >= 2 / 3 else "media" if share <= 1 / 3 else "mixed"


# ============================================================================== ORACLE + patterns (GPU batch)
def outcome_spend(o, sizes, cfg: GameConfig) -> dict:
    fee = cfg.workers.worker_cost
    workers = sum(sum(a.hires_by_region) for a in o.allocations) * fee
    media = sum((media_cost(sizes, a.media_regions, cfg.media.base_cost) for a in o.allocations), Decimal("0"))
    return {"workers": float(workers), "media": float(media), "queries": 0.0, "total": float(workers + media),
            "num_workers": int(sum(sum(a.hires_by_region) for a in o.allocations)),
            "media_airings": int(sum(sum(a.media_regions) for a in o.allocations))}


def oracle_rows(base: GameConfig, seeds: list[int], device: torch.device, record: dict) -> list[dict]:
    t0 = time.perf_counter()
    es = [generate_electorate(dataclasses.replace(base, seed=s), device=device) for s in seeds]
    m_oracle = torch.stack([oracle_manifesto(e, dataclasses.replace(base, seed=s)) for e, s in zip(es, seeds)])
    m_informed = torch.stack([informed_manifesto(e, dataclasses.replace(base, seed=s)) for e, s in zip(es, seeds)])
    record["setup_seconds"] = time.perf_counter() - t0
    sizes = [[int(x) for x in e.sizes.tolist()] for e in es]
    cal_cfg = cell_config(base, *DEFAULT_CELL)
    cal = xc.calibrate_batch(lambda B: run_batch(es[:B], cal_cfg, m_oracle[:B], x_policies.policy_planner,
                                                 seeds=seeds[:B]), device, sizes=(2, 4), max_batch=len(seeds))
    record["calibration"] = cal
    batch = cal["batch"]
    rows = []
    idx = list(range(len(seeds)))
    for treasury, wfee in itertools.product(TREASURIES, WORKER_FEES):
        cfg = cell_config(base, treasury, wfee, "10.00")
        arms = [("ORACLE", "oracle", "planner", m_oracle)] + [("pattern", "informed", p, m_informed) for p in PATTERNS]
        for player, mname, pname, ms in arms:
            policy = x_policies.GATE_POLICIES[pname]
            outs = xc.run_chunks(lambda ch: run_batch([es[i] for i in ch], cfg, ms[ch], policy,
                                                      seeds=[seeds[i] for i in ch]), idx, batch, record=record)
            for i, o in zip(idx, outs):
                spend = outcome_spend(o, sizes[i], cfg)
                rows.append({"exp": "X7", "seed": seeds[i], "player": player, "manifesto": mname, "policy": pname,
                             "treasury": treasury, "worker_fee": wfee, "query_fee": None,
                             "cell": cell_key(treasury, wfee, None), "expected": o.result.expected_share,
                             "vote_share": o.result.vote_share, "victory": bool(o.result.victory), "spend": spend,
                             "converted": sum(o.converted), "dominant": dominant(spend["workers"], spend["media"])})
        print(f"[oracle] T={treasury} W={wfee}: ORACLE win "
              f"{sum(r['victory'] for r in rows[-5 * len(seeds):-4 * len(seeds)]) / len(seeds):.2f}", flush=True)
    record["oracle_seconds"] = time.perf_counter() - t0
    return rows


# ============================================================================== BayesPlanner (engine games)
def _planner_class(device: str):
    from dumbocracy.strategies.players import BayesPlanner
    if device == "cpu":
        return BayesPlanner
    from x_gpu_benchmarks import _gpu_planner_class
    return _gpu_planner_class()


def bayes_seed(task: tuple) -> dict:
    """All grid cells for one seed (one electorate, reused). Runs in a spawned process."""
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.strategies.players import play_game
    from strategies import spend_breakdown
    seed, n, device, cells = task
    torch.set_num_threads(4)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.zeros(1, device=dev)               # initialise the CUDA context in a fresh spawned process
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.perf_counter()
    base = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(base, device=dev)
    cls = _planner_class(device)
    rows = []
    for treasury, wfee, qfee in cells:
        cfg = cell_config(base, treasury, wfee, qfee)
        state0 = GameEngine.new_game(cfg, electorate=e, game_id=f"x7_{seed}")
        player = cls(seed=seed, device=device, config=dataclasses.replace(cfg, device="cpu"),
                     survey_budget=survey_budget(qfee))
        g0 = time.perf_counter()
        final, log = play_game(state0, player)
        obs = observation_for(final, "p0")
        res = obs["result"]
        spend = spend_breakdown(obs, cfg)
        rows.append({"exp": "X7", "seed": seed, "player": "BayesPlanner", "manifesto": "posterior", "policy": "rollout",
                     "treasury": treasury, "worker_fee": wfee, "query_fee": qfee,
                     "survey_budget": str(survey_budget(qfee)), "cell": cell_key(treasury, wfee, qfee),
                     "expected": res["expected_share"], "vote_share": res["vote_share"], "victory": bool(res["victory"]),
                     "spend": spend, "num_workers": int(round(spend["workers"] / float(wfee))),
                     "dominant": dominant(spend["workers"], spend["media"]),
                     "rejected": sum(1 for *_, ok, _c in log if not ok), "game_seconds": time.perf_counter() - g0})
    peak = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    return {"seed": seed, "rows": rows, "seconds": time.perf_counter() - start, "peak_bytes": peak}


# ============================================================================== parity (§1.1 point 7)
def parity(seeds: list[int], device: torch.device, n: int) -> dict:
    """CPU↔GPU, 3 seeds, cell (₫750, ₫50, ₫10): (a) ORACLE planner via run_batch, (b) one BayesPlanner game
    (CPU planner + CPU engine vs GPU-rollout planner + GPU engine)."""
    cell = ("750.00", "50.00", "10.00")
    cache: dict = {}

    def oracle_fn(seed, dev):
        cfg = cell_config(GameConfig(seed=seed, device=str(dev), electorate=ElectorateConfig(num_voters=n)), *cell)
        e = generate_electorate(cfg, device=dev)
        o = run_batch([e], cfg, oracle_manifesto(e, cfg)[None], x_policies.policy_planner, seeds=[seed])[0]
        return {"expected": o.result.expected_share, "victory": o.result.victory, "converted": sum(o.converted)}

    def bayes_fn(seed, dev):
        key = (seed, str(dev))
        if key not in cache:
            row = bayes_seed((seed, n, str(dev), [cell]))["rows"][0]
            cache[key] = {"expected": row["expected"], "victory": row["victory"], "workers_spend": row["spend"]["workers"]}
        return cache[key]

    a = xc.parity_check(oracle_fn, seeds, device=device, abs_tol=0.01)
    a["metrics"]["converted"]["note"] = "tolerance in voters"
    b = xc.parity_check(bayes_fn, seeds, device=device, abs_tol=0.02)
    b["metrics"]["workers_spend"]["note"] = "tolerance in ₫"
    return {"cell": list(cell), "passed": bool(a["passed"] and b["passed"]), "oracle_run_batch": a,
            "bayes_planner_game": b}


# ============================================================================== statistics
def summarise(rows: list[dict], seeds: list[int]) -> dict:
    by: dict[tuple, dict[int, dict]] = {}
    for r in rows:
        by.setdefault((r["player"], r["policy"], r["cell"]), {})[r["seed"]] = r

    def cell_summary(rs: list[dict]) -> dict:
        n = len(rs)
        spend = {k: xc.mean_ci([r["spend"][k] for r in rs]) for k in ("workers", "media", "queries", "total")}
        tot_w = sum(r["spend"]["workers"] for r in rs)
        tot_m = sum(r["spend"]["media"] for r in rs)
        tot_q = sum(r["spend"]["queries"] for r in rs)
        tot = tot_w + tot_m + tot_q
        labels = {}
        for r in rs:
            labels[r["dominant"]] = labels.get(r["dominant"], 0) + 1
        out = {"n": n, "win": xc.rate([r["victory"] for r in rs]), "expected": xc.mean_ci([r["expected"] for r in rs]),
               "spend": spend,
               "spend_mix": {"workers": tot_w / tot if tot else 0.0, "media": tot_m / tot if tot else 0.0,
                             "surveys": tot_q / tot if tot else 0.0},
               "dominant_pattern": dominant(tot_w, tot_m), "dominant_counts": labels}
        if "num_queries" in rs[0]["spend"]:
            out["mean_queries"] = sum(r["spend"]["num_queries"] for r in rs) / n
        if "num_workers" in rs[0]["spend"]:
            out["mean_workers"] = sum(r["spend"]["num_workers"] for r in rs) / n
        elif "num_workers" in rs[0]:
            out["mean_workers"] = sum(r["num_workers"] for r in rs) / n
        return out

    cells: dict = {}
    for (player, policy, cell), d in sorted(by.items()):
        name = player if player != "pattern" else f"pattern/{policy}"
        cells.setdefault(name, {})[cell] = cell_summary([d[s] for s in seeds if s in d])

    # best paid pattern per seed and per (treasury, worker fee): balance.py criterion 4 analogue
    patterns: dict = {}
    for t, w in itertools.product(TREASURIES, WORKER_FEES):
        cell = cell_key(t, w, None)
        counts = {p: 0 for p in PATTERNS}
        for s in seeds:
            best = max(PATTERNS, key=lambda p: by[("pattern", p, cell)][s]["expected"])
            counts[best] += 1
        best_rate = max(PATTERNS, key=lambda p: cells[f"pattern/{p}"][cell]["win"]["rate"])
        patterns[cell] = {"best_pattern_counts": counts, "dominant_pattern": max(counts, key=counts.get),
                          "max_pattern_win_rate": cells[f"pattern/{best_rate}"][cell]["win"]["rate"],
                          "max_pattern": best_rate,
                          "gate_4a_no_pattern_ge_80pct": cells[f"pattern/{best_rate}"][cell]["win"]["rate"] < 0.80}

    tests: dict = {}

    def vals(player, policy, cell, key="expected"):
        d = by[(player, policy, cell)]
        return [d[s][key] for s in seeds]

    # family A: workers vs media (pay_workers − pay_media, expected share), per (treasury, worker fee)
    fam = []
    for t, w in itertools.product(TREASURIES, WORKER_FEES):
        c = cell_key(t, w, None)
        res = xc.paired(vals("pattern", "pay_workers", c), vals("pattern", "pay_media", c))
        fam.append({"treasury": t, "worker_fee": w, "a": "pay_workers", "b": "pay_media", **res,
                    "mcnemar": xc.mcnemar(vals("pattern", "pay_workers", c, "victory"),
                                          vals("pattern", "pay_media", c, "victory"))})
    for x, p in zip(fam, xc.holm([x["wilcoxon_p"] for x in fam])):
        x["holm_p"] = p
    tests["A_workers_vs_media"] = {"family": "pay_workers − pay_media expected share, 12 (treasury, worker fee) cells; "
                                             "Wilcoxon, Holm within family", "tests": fam}

    # family B: diminishing returns — (E750 − E500) − (E1000 − E750) per player and fee column (equal ₫250 steps)
    fam = []
    columns = [("ORACLE", "planner", w, None) for w in WORKER_FEES] + \
              [("BayesPlanner", "rollout", w, q) for w in WORKER_FEES for q in QUERY_FEES]
    for player, policy, w, q in columns:
        e = [vals(player, policy, cell_key(t, w, q)) for t in TREASURIES]
        first = [b - a for a, b in zip(e[0], e[1])]
        second = [c - b for b, c in zip(e[1], e[2])]
        fam.append({"player": player, "worker_fee": w, "query_fee": q,
                    "gain_500_750": xc.mean_ci(first), "gain_750_1000": xc.mean_ci(second),
                    "first_minus_second": xc.paired(first, second)})
    for x, p in zip(fam, xc.holm([x["first_minus_second"]["wilcoxon_p"] for x in fam])):
        x["holm_p"] = p
    tests["B_diminishing_returns"] = {"family": "paired (gain 500→750) − (gain 750→1000) in expected share, per player "
                                                "and fee column; Wilcoxon, Holm within family", "tests": fam}

    # family C: query fee ₫5 vs ₫10 for BayesPlanner, per (treasury, worker fee)
    fam = []
    for t, w in itertools.product(TREASURIES, WORKER_FEES):
        a, b = cell_key(t, w, "5.00"), cell_key(t, w, "10.00")
        fam.append({"treasury": t, "worker_fee": w, **xc.paired(vals("BayesPlanner", "rollout", a),
                                                                vals("BayesPlanner", "rollout", b)),
                    "mcnemar": xc.mcnemar(vals("BayesPlanner", "rollout", a, "victory"),
                                          vals("BayesPlanner", "rollout", b, "victory"))})
    for x, p in zip(fam, xc.holm([x["wilcoxon_p"] for x in fam])):
        x["holm_p"] = p
    tests["C_query_fee_bayes"] = {"family": "BayesPlanner query fee ₫5 − ₫10 (survey budget 5 questions in both), "
                                            "12 cells; Wilcoxon, Holm within family", "tests": fam}

    # family D: each worker fee vs the game's ₫20, same treasury (and query fee), per player
    fam = []
    for t in TREASURIES:
        for w in WORKER_FEES:
            if w == "20.00":
                continue
            for player, policy, q in [("ORACLE", "planner", None), ("BayesPlanner", "rollout", "5.00"),
                                      ("BayesPlanner", "rollout", "10.00")]:
                a, b = cell_key(t, w, q), cell_key(t, "20.00", q)
                fam.append({"player": player, "treasury": t, "worker_fee": w, "query_fee": q, "vs_worker_fee": "20.00",
                            **xc.paired(vals(player, policy, a), vals(player, policy, b)),
                            "mcnemar": xc.mcnemar(vals(player, policy, a, "victory"), vals(player, policy, b, "victory"))})
    for x, p in zip(fam, xc.holm([x["wilcoxon_p"] for x in fam])):
        x["holm_p"] = p
    for x, p in zip(fam, xc.holm([x["mcnemar"]["p"] for x in fam])):
        x["mcnemar"]["holm_p"] = p
    tests["D_worker_fee_vs_20"] = {"family": "each worker fee − ₫20 at the same treasury/query fee, per player; "
                                             "Wilcoxon (expected) and McNemar (wins), each Holm within family",
                                   "tests": fam}
    return {"cells": cells, "patterns": patterns, "tests": tests}


# ============================================================================== main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X7"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2, help="processes for the BayesPlanner games (≤ 2 per GPU)")
    parser.add_argument("--budget-hours", type=float, default=3.0, help="reduce seeds if the estimate exceeds this")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--skip-bayes", action="store_true")
    parser.add_argument("--out", default=str(xc.REPORTS / "x7.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    free0 = xc.free_bytes(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text("")                      # this script's own output, regenerated per run
    record: dict = {}
    deviations = ["BayesPlanner(device='cuda') crashes in manifesto() (CPU simplex tensor stacked with GPU tensors); "
                  "experiment-side subclass keeps the manifesto worlds on the CPU, rollouts on the GPU "
                  "(x_gpu_benchmarks._gpu_planner_class). Library fix awaits owner approval.",
                  f"Device {device} used instead of the plan default cuda:2 (cuda:2 runs X0/X10 for another agent)."]
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        print(f"[x7] device {device}, seeds {seeds[0]}..{seeds[-1]}", flush=True)
        rows = oracle_rows(base, seeds, device, record)
        bayes_info: dict = {}
        if not args.skip_bayes:
            cells = list(itertools.product(TREASURIES, WORKER_FEES, QUERY_FEES))
            pilot = bayes_seed((seeds[0], args.n, str(device), cells))
            per_seed = pilot["seconds"]
            estimate = per_seed * (len(seeds) - 1) / max(1, args.workers) * 1.15   # 15% contention allowance
            print(f"[x7] pilot: seed {seeds[0]}, {len(cells)} games in {per_seed:.0f}s "
                  f"({per_seed / len(cells):.2f} s/game); estimated remaining {estimate / 60:.0f} min "
                  f"with {args.workers} processes", flush=True)
            bayes_info = {"pilot_seed": seeds[0], "pilot_seconds": per_seed, "pilot_games": len(cells),
                          "estimated_remaining_seconds": estimate, "processes": args.workers,
                          "seeds_per_cell": len(seeds)}
            bayes_seeds = seeds[1:]
            if estimate > args.budget_hours * 3600:
                # corner cells keep all seeds; inner cells get 25 (recorded deviation)
                raise SystemExit("estimate exceeds the time budget: rerun with the corner/inner reduction (not needed "
                                 "at the measured speed; implement if this triggers)")
            results = [pilot] + xc.run_tasks(bayes_seed, [(s, args.n, str(device), cells) for s in bayes_seeds],
                                             args.workers, label="[x7 bayes]")
            for res in results:
                rows += res["rows"]
            bayes_info["seconds"] = sum(r["seconds"] for r in results)
            bayes_info["child_peak_bytes_max"] = max((r["peak_bytes"] or 0) for r in results)
            bayes_info["rejected_actions"] = sum(r["rejected"] for res in results for r in res["rows"])
        for r in rows:
            xc.write_row(RAW_PATH, r)
        par = parity(seeds[:args.parity_seeds], device, args.n)
        print(f"[x7] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    summary = summarise(rows, seeds) if not args.skip_bayes else {"note": "bayes skipped"}
    from dumbocracy.strategies.players import BayesPlanner
    bp_defaults = {f.name: (str(getattr(BayesPlanner(), f.name)) if isinstance(getattr(BayesPlanner(), f.name), Decimal)
                            else getattr(BayesPlanner(), f.name))
                   for f in dataclasses.fields(BayesPlanner) if f.name not in ("latency", "diagnostics", "config")}
    report = {
        "experiment": "X7", "question": "Budget and price sensitivity (sensitivity only; game prices stay ₫20/₫10, D5)",
        "grid": {"treasury": TREASURIES, "worker_fee": WORKER_FEES, "query_fee": QUERY_FEES},
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
        "n": args.n, "rounds": base.rounds,
        "players": {
            "ORACLE": "oracle manifesto + balance.policy_planner via x_batch.run_batch (true state; reference, not a "
                      "player). No queries, so it is query-fee invariant: cell keys use Q'any'.",
            "pattern/<p>": "X0 scripted paid patterns with the informed manifesto (balance.py criterion 4), per cell",
            "BayesPlanner": "GameEngine games, observation dicts only; config overrides passed to engine and planner"},
        "survey_budget_rule": {"questions": SURVEY_QUESTIONS,
                               "survey_budget_by_query_fee": {q: str(survey_budget(q)) for q in QUERY_FEES},
                               "reason": "same number of affordable questions (5) in both query-fee arms; the library "
                                         "default ₫50 would buy 10 questions at ₫5 and 5 at ₫10"},
        "bayes_planner_parameters": {**bp_defaults, "survey_budget": "per query fee (above)", "device": str(device),
                                     "manifesto_device": "cpu (workaround)"},
        "dominant_rule": "workers if ≥ 2/3 of campaign spend (workers + media) went to workers, media if ≤ 1/3, else mixed",
        "default_cell": cell_key(*DEFAULT_CELL),
        **summary,
        "bayes_run": bayes_info, "oracle_run": record, "deviations": deviations,
        "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, workers=args.workers,
                                       batch_sizes={"oracle_run_batch": record.get("batch_sizes"),
                                                    "bayes_planner_num_worlds": BayesPlanner().num_worlds},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"bayes_child_peak_bytes_max": bayes_info.get("child_peak_bytes_max")}),
    }
    xc.save(args.out, report)
    print(f"[x7] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
