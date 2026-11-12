"""S4 — Do the strategies generalise? Unseen worlds, electorate size N and treasury, on HELD-OUT seeds.

Commands (write under reports/v3/experimentation/):
  # 1. budget-aware playbook (x_playbook_budget.BudgetPlaybook): tune (M media rounds, f round-1 worker share) on
  #    TRAINING seeds 13000–13049 at every tested treasury (s4_tune.json, raw/s4_tune.jsonl, logs/s4_tune.log)
  OMP_NUM_THREADS=4 python experiments/s4_generalisation.py --stage tune --seeds 50 --first-seed 13000 --device cuda:0
  # 2. test every arm on HELD-OUT seeds 14000–14099 (s4.json, raw/s4.jsonl, logs/s4.log)
  OMP_NUM_THREADS=4 python experiments/s4_generalisation.py --stage test --seeds 100 --first-seed 14000 --device cuda:0

Seed roles: 12000–12099 = S1–S3 test worlds, 13000–13099 = playbook training worlds (S2/S3), 14000–14099 = held out
(only S1-holdout ran smart vs bayes there, at the base cell; nothing was tuned on them). The test stage refuses
seeds outside 14000–14099 unless --out-dir is given (smoke runs).

Cells (never more than 10⁴ voters — hard limit of the owner; checked twice):
  base      N = 10⁴, T = 5, ₫500           (question 1, unseen worlds; also the reference point of 2 and 3)
  n1000/n2000/n5000   N ∈ {1000, 2000, 5000}, T = 5, ₫500                  (question 2, electorate size)
  b1000/b2500/b10000/b100000   N = 10⁴, T = 5, ₫ ∈ {1000, 2500, 10000, 100000}   (question 3, treasury)
The treasury is ``GameConfig.budget``: the engine starts the ledger with it and every player reads it from the
observation (``initial_budget`` → ``players.public_config``) — the same path the web UI uses.

World construction and scoring are S2's (= X6): ``GameConfig(seed, rounds, budget, ElectorateConfig(N))`` →
``generate_electorate`` on the device → ``GameEngine.new_game(..., game_id=f"x6_{seed}")`` → ``play_game`` on
observation dicts; E = ``result.expected_share``, win = official result. All arms of a cell play the same world
(paired). ``play_game(strict=False)``: a rejected action is counted (``rejected``) and replaced by a forfeit
instead of aborting the world; a crash of one arm is recorded in ``error`` (none expected).

Arms (every cell): smart (``smart_player()`` = RECOMMENDED), bayes (``BayesPlanner`` defaults), playbook
(``x_playbook.TUNED_PLAYBOOK``), adaptive (``AdaptivePlaybook`` with s3_tune.json → chosen, as in S3), top3; plus
budget_playbook (``BudgetPlaybook`` with s4_tune.json → chosen) when the tune file exists.
Reported per cell × arm: E, win rate, paired Δ vs bayes / playbook / top3 (95 % t-CI, Wilcoxon), McNemar on wins
vs bayes, aligned/divided split (TV of the two largest regions' archetypes ≥ 0.35, the S1/S2 split), and money:
spent on workers / media / survey, total spent, share of the treasury spent, money left at the end, share of the
treasury spent in rounds 1–2, workers by round.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_playbook as s2  # noqa: E402
import x_common as xc  # noqa: E402
import x_playbook as xp  # noqa: E402

OUT = s2.OUT
N_MAX = 10_000    # owner's hard limit: no run with more than 10,000 voters
CELLS = {
    "base": {"n": 10_000, "rounds": 5, "budget": "500"},
    "n1000": {"n": 1_000, "rounds": 5, "budget": "500"},
    "n2000": {"n": 2_000, "rounds": 5, "budget": "500"},
    "n5000": {"n": 5_000, "rounds": 5, "budget": "500"},
    "b1000": {"n": 10_000, "rounds": 5, "budget": "1000"},
    "b2500": {"n": 10_000, "rounds": 5, "budget": "2500"},
    "b10000": {"n": 10_000, "rounds": 5, "budget": "10000"},
    "b100000": {"n": 10_000, "rounds": 5, "budget": "100000"},
}
assert all(c["n"] <= N_MAX for c in CELLS.values())
QUESTIONS = {"unseen_worlds": ["base"], "electorate_size": ["n1000", "n2000", "n5000", "base"],
             "budget": ["base", "b1000", "b2500", "b10000", "b100000"]}
TUNE_CELLS = ("base", "b1000", "b2500", "b10000", "b100000")
TUNE_GRID = {"media_rounds": (2, 5), "worker_frac": (0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)}
HELD_OUT = range(14000, 14100)
TRAIN = range(13000, 13050)


def adaptive_kwargs() -> dict:
    return dict(json.loads((s2.OUT / "s3_tune.json").read_text())["chosen"])   # always the real S3 tuning


def budget_kwargs(tune_file: Path) -> tuple[dict, dict] | None:
    """(per-budget schedule dials, single-rule dials) from the S4 tuning file, or None."""
    if not tune_file.exists():
        return None
    d = json.loads(tune_file.read_text())
    return dict(d["chosen_schedule"]), dict(d["chosen"])


def make_player(kind: str, kwargs: dict, seed: int, device: str):
    if kind == "smart":
        from dumbocracy.strategies.smart import smart_player
        return smart_player(seed=seed, device=device, **kwargs)
    if kind == "budget":
        import x_playbook_budget as xb
        return xb.BudgetPlaybook(seed=seed, **kwargs)
    return s2.make_player(kind, kwargs, seed, device)


def seed_task(task: tuple) -> dict:
    """All arms on one world. ``task = (seed, cell_name, cell, arms, device)``; arms = [(name, kind, kwargs)]."""
    from decimal import Decimal

    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.strategies.players import play_game
    from strategies import spend_breakdown
    seed, cname, cell, arms, device = task
    if int(cell["n"]) > N_MAX:
        raise SystemExit(f"refusing N = {cell['n']} > {N_MAX}")
    torch.set_num_threads(4)
    dev = torch.device(device)
    start = time.perf_counter()
    budget = Decimal(cell["budget"]).quantize(Decimal("0.01"))
    cfg = GameConfig(seed=seed, device=device, rounds=int(cell["rounds"]), budget=budget,
                     electorate=ElectorateConfig(num_voters=int(cell["n"])))
    e = generate_electorate(cfg, device=dev)
    state0 = GameEngine.new_game(cfg, electorate=e, game_id=f"x6_{seed}")
    obs0 = observation_for(state0, "p0")
    assert Decimal(obs0["initial_budget"]) == budget and Decimal(obs0["treasury"]) == budget
    feats = s2.world_features(e, obs0)
    rows = []
    for name, kind, kwargs in arms:
        row = {"seed": seed, "cell": cname, "arm": name, "kind": kind, "budget": float(budget),
               "n": int(cell["n"]), "rounds": int(cell["rounds"])}
        g0 = time.perf_counter()
        try:
            player = make_player(kind, kwargs, seed, device)
            final, log = play_game(state0, player, strict=False)
        except Exception as exc:  # noqa: BLE001  — recorded, the other arms still run
            row.update(error=f"{type(exc).__name__}: {exc}", game_seconds=time.perf_counter() - g0)
            rows.append(row)
            continue
        obs = observation_for(final, "p0")
        res = obs["result"]
        allocs = sorted(obs["allocations"], key=lambda a: a["round"])
        sp = spend_breakdown(obs, cfg)
        round_cost = [float(a["cost"]) for a in allocs]
        row.update({
            "expected": res["expected_share"], "vote_share": res["vote_share"], "victory": bool(res["victory"]),
            "spend": sp, "unspent": float(obs["treasury"]), "spent_frac": sp["total"] / float(budget),
            "early_frac": (sp["queries"] + sum(round_cost[:2])) / float(budget),
            "round_cost": round_cost, "workers_by_round": [sum(int(h) for h in a["hires_by_region"]) for a in allocs],
            "airings_by_round": [sum(bool(m) for m in a["media_regions"]) for a in allocs],
            "rejected": sum(1 for *_, ok, _c in log if not ok),
            "rejections": [str(c) for *_, ok, c in log if not ok][:5],
            "game_seconds": time.perf_counter() - g0, "manifesto": [float(x) for x in obs["manifesto"]],
            "scenario": "divided" if feats["tv_two_largest"] >= xp.DIVIDED_TV else "aligned"})
        rec = getattr(player, "record", None)
        if rec and kind in ("adaptive", "budget"):
            row["branch"] = rec.get("branch")
        if kind in ("playbook", "adaptive", "budget"):
            row["human_executable"] = player.human_executable
        rows.append(row)
    return {"seed": seed, "cell": cname, "rows": rows, "features": feats, "seconds": time.perf_counter() - start}


def run_cell(cname: str, seeds, arms, device: str, workers: int, label: str):
    tasks = [(s, cname, CELLS[cname], arms, device) for s in seeds]
    results = xc.run_tasks(seed_task, tasks, workers, label=f"{label} {cname}")
    return [r for res in results for r in res["rows"]], {res["seed"]: res["features"] for res in results}, \
        sum(res["seconds"] for res in results)


# ============================================================================== statistics
def summarise_cell(rows: list[dict], seeds: list[int], arm_names: list[str]) -> dict:
    by: dict[str, dict[int, dict]] = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    errors = {a: sum(1 for s in seeds if "error" in by[a][s]) for a in arm_names}
    ok = [s for s in seeds if all("error" not in by[a][s] for a in arm_names)]
    scen = {g: [s for s in ok if by[arm_names[0]][s].get("scenario") == g] for g in ("aligned", "divided")}

    def v(a, key="expected", ss=None):
        return [by[a][s][key] for s in (ok if ss is None else ss)]

    out = {"seeds_used": len(ok), "errors": errors, "scenario_counts": {g: len(ss) for g, ss in scen.items()},
           "players": {}}
    for a in arm_names:
        rs = [by[a][s] for s in ok]
        d = {"expected": xc.mean_ci(v(a)), "win": xc.rate(v(a, "victory")),
             "money": {"workers": xc.mean_ci([r["spend"]["workers"] for r in rs]),
                       "media": xc.mean_ci([r["spend"]["media"] for r in rs]),
                       "survey": xc.mean_ci([r["spend"]["queries"] for r in rs]),
                       "total_spent": xc.mean_ci([r["spend"]["total"] for r in rs]),
                       "spent_frac_of_treasury": xc.mean_ci([r["spent_frac"] for r in rs]),
                       "unspent_at_end": xc.mean_ci([r["unspent"] for r in rs]),
                       "worlds_with_unspent_ge_1pct": sum(1 for r in rs if r["unspent"] >= 0.01 * r["budget"]),
                       "spent_by_end_of_round2_frac": xc.mean_ci([r["early_frac"] for r in rs]),
                       "workers_by_round": [sum(r["workers_by_round"][t] for r in rs if t < len(r["workers_by_round"]))
                                            / len(rs) for t in range(max(len(r["workers_by_round"]) for r in rs))],
                       "airings_by_round": [sum(r["airings_by_round"][t] for r in rs if t < len(r["airings_by_round"]))
                                            / len(rs) for t in range(max(len(r["airings_by_round"]) for r in rs))]},
             "rejected_actions": sum(r["rejected"] for r in rs),
             "game_seconds": xc.mean_ci([r["game_seconds"] for r in rs]),
             "human_executable": rs[0].get("human_executable")}
        for ref in ("bayes", "playbook", "top3"):
            if ref in by and ref != a:
                d[f"vs_{ref}"] = xc.paired(v(a), v(ref))
        if "bayes" in by and a != "bayes":
            d["mcnemar_vs_bayes"] = xc.mcnemar(v(a, "victory"), v("bayes", "victory"))
        if any(r.get("branch") for r in rs):
            d["branch_counts"] = {}
            for r in rs:
                d["branch_counts"][str(r.get("branch"))] = d["branch_counts"].get(str(r.get("branch")), 0) + 1
        d["by_scenario"] = {}
        for g, ss in scen.items():
            if not ss:
                continue
            sd = {"n": len(ss), "expected": xc.mean_ci(v(a, ss=ss)), "win": xc.rate(v(a, "victory", ss)),
                  "losses": sum(1 for s in ss if not by[a][s]["victory"])}
            for ref in ("bayes", "playbook"):
                if ref in by and ref != a and len(ss) > 1:
                    sd[f"vs_{ref}"] = xc.paired(v(a, ss=ss), v(ref, ss=ss))
            d["by_scenario"][g] = sd
        out["players"][a] = d
    out["ranking_by_mean_E"] = sorted(arm_names, key=lambda a: -out["players"][a]["expected"]["mean"])
    return out


def fmt_ci(d: dict) -> str:
    return f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


# ============================================================================== stages
def stage_tune(args, log, device) -> dict:
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    assert set(seeds) <= set(TRAIN), "tuning only on training seeds 13000–13049"
    import x_playbook_budget as xb
    fixed = {k: v for k, v in xb.BUDGET_PLAYBOOK.items() if k not in TUNE_GRID}
    grid = [dict(zip(TUNE_GRID, vals)) for vals in itertools.product(*TUNE_GRID.values())]
    arms = [(f"M{g['media_rounds']}f{g['worker_frac']}", "budget", {**fixed, **g}) for g in grid]
    arms.append(("playbook", "playbook", dict(xp.TUNED_PLAYBOOK)))
    cells = [c for c in (args.cells.split(",") if args.cells else TUNE_CELLS)]
    log(f"[s4 tune] {len(arms)} arms × cells {cells} × {len(seeds)} seeds (training)")
    rows, mean = [], {}
    for c in cells:
        r, _, _ = run_cell(c, seeds, arms, str(device), args.workers, "[s4 tune]")
        rows += r
        bad = [x for x in r if "error" in x]
        if bad:
            log(f"[s4 tune] {c}: {len(bad)} errors, e.g. {bad[0]['error']}")
        E = {}
        for x in r:
            if "error" not in x:
                E.setdefault(x["arm"], []).append(x["expected"])
        mean[c] = {a: sum(v) / len(v) for a, v in E.items()}
        best = max(mean[c], key=mean[c].get)
        log(f"[s4 tune] {c}: best {best} E={mean[c][best]:.4f}; playbook {mean[c]['playbook']:.4f}; "
            + ", ".join(f"{a} {mean[c][a]:.4f}" for a, _, _ in arms if a != "playbook"))
    names = [a for a, _, _ in arms if a != "playbook"]
    overall = {a: sum(mean[c][a] for c in cells) / len(cells) for a in names}
    big = [c for c in cells if c != "base"]
    overall_big = {a: sum(mean[c][a] for c in big) / len(big) for a in names}
    chosen_arm = max(overall, key=overall.get)
    chosen = next(kw for a, _, kw in arms if a == chosen_arm)
    log(f"[s4 tune] chosen {chosen_arm} (mean E over {cells} {overall[chosen_arm]:.4f}; playbook "
        f"{sum(mean[c]['playbook'] for c in cells) / len(cells):.4f})")
    raw = OUT / "raw" / f"s4_tune{args.tag}.jsonl"
    raw.write_text("")
    for r in rows:
        xc.write_row(raw, {"exp": "S4-tune", **{k: v for k, v in r.items() if k != "manifesto"}})
    per_cell_pb = {c: {"playbook_E": mean[c]["playbook"], "best_arm": max((a for a in names), key=mean[c].get),
                       "best_E": max(mean[c][a] for a in names), "chosen_E": mean[c][chosen_arm]} for c in cells}
    for c in cells:   # ties (e.g. ₫500, where every M5 arm buys the same 3 workers) go to the single rule
        if mean[c][chosen_arm] >= per_cell_pb[c]["best_E"] - 1e-6:   # GPU float noise ~1e-8
            per_cell_pb[c]["best_arm"] = chosen_arm
    dials = {a: kw for a, _, kw in arms}
    schedule = [{"budget": float(CELLS[c]["budget"]), "arm": per_cell_pb[c]["best_arm"],
                 **{k: dials[per_cell_pb[c]["best_arm"]][k] for k in TUNE_GRID}} for c in cells]
    chosen_schedule = {**fixed, "schedule": schedule}
    log(f"[s4 tune] per-budget schedule: {schedule}")
    return {"grid": TUNE_GRID, "fixed": fixed, "cells": {c: CELLS[c] for c in cells}, "mean_E": mean,
            "mean_E_over_cells": overall, "mean_E_over_cells_except_base": overall_big, "per_cell": per_cell_pb,
            "chosen_arm": chosen_arm, "chosen": chosen, "chosen_schedule": chosen_schedule,
            "selection": {"chosen": "single rule: highest mean E over all tuned cells (equal weight per cell)",
                          "chosen_schedule": "per-budget table: best arm of each tuned treasury; a game uses the "
                                             "row whose budget is nearest on a log scale"},
            "seeds": {"first": seeds[0], "count": len(seeds), "role": "training (13000–13049; disjoint from "
                                                                         "12000–12099 and held-out 14000–14099)"},
            "games": len(rows)}


def stage_test(args, log, device) -> dict:
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    if not args.out_dir:
        assert set(seeds) <= set(HELD_OUT), "test only on held-out seeds 14000–14099"
    small = list(range(args.first_seed, args.first_seed + (args.seeds_other or args.seeds)))
    tune_file = Path(args.tune_file) if args.tune_file else OUT / f"s4_tune{args.tag}.json"
    arms = [("smart", "smart", {}), ("bayes", "bayes", {}), ("playbook", "playbook", dict(xp.TUNED_PLAYBOOK)),
            ("adaptive", "adaptive", adaptive_kwargs()), ("top3", "top3", {})]
    bkw = budget_kwargs(tune_file)
    if bkw is not None:
        arms.append(("budget_playbook", "budget", bkw[0]))
        arms.append(("budget_playbook_single", "budget", bkw[1]))
    if args.arms:
        keep = args.arms.split(",")
        arms = [a for a in arms if a[0] in keep]
    names = [a for a, _, _ in arms]
    cells = args.cells.split(",") if args.cells else list(CELLS)
    log(f"[s4 test] arms {names} × cells {cells}; seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}) for base, "
        f"{len(small)} for the other cells; budget playbook dials {bkw}")
    out, all_rows, feats_all, busy = {}, [], {}, 0.0
    for c in cells:
        ss = seeds if c == "base" else small
        t0 = time.perf_counter()
        rows, feats, b = run_cell(c, ss, arms, str(device), args.workers, "[s4 test]")
        busy += b
        all_rows += rows
        feats_all[c] = feats
        d = summarise_cell(rows, ss, names)
        d["cell"] = CELLS[c]
        d["seeds"] = {"first": ss[0], "count": len(ss)}
        d["wall_seconds"] = time.perf_counter() - t0
        d["mean_friends"] = xc.mean_ci([feats[s]["mean_friends"] for s in ss])
        out[c] = d
        P = d["players"]
        log(f"[s4 test] {c} ({len(ss)} seeds, {d['wall_seconds']:.0f}s, errors {d['errors']}):")
        for a in names:
            p = P[a]
            m = p["money"]
            log(f"   {a:16s} E {p['expected']['mean']:.4f} win {p['win']['rate']:.2f} "
                + (f"Δbayes {fmt_ci(p['vs_bayes'])} " if "vs_bayes" in p else "")
                + (f"Δplaybook {fmt_ci(p['vs_playbook'])} " if "vs_playbook" in p else "")
                + f"spent {m['spent_frac_of_treasury']['mean']:.3f} left ₫{m['unspent_at_end']['mean']:.1f} "
                  f"by-r2 {m['spent_by_end_of_round2_frac']['mean']:.3f} "
                  f"workers/round {[round(x, 1) for x in m['workers_by_round']]} rej {p['rejected_actions']} "
                  f"{p['game_seconds']['mean']:.1f}s/game")
    # reproduction check vs S1-holdout (same worlds, same players, base cell)
    repro = {}
    ref_path = s2.OUT / "raw" / "s1_holdout.jsonl"
    if "base" in cells and ref_path.exists():
        ref = {(r["player"], r["seed"]): r["expected"] for r in xc.read_rows(ref_path)}
        for a in ("smart", "bayes"):
            diffs = [abs(r["expected"] - ref[(a, r["seed"])]) for r in all_rows
                     if r["cell"] == "base" and r["arm"] == a and "error" not in r and (a, r["seed"]) in ref]
            if diffs:
                repro[a] = {"seeds_compared": len(diffs), "max_abs_dE": max(diffs)}
        log(f"[s4 test] reproduction vs raw/s1_holdout.jsonl: {repro}")
    raw = OUT / "raw" / f"s4{args.tag}.jsonl"
    raw.write_text("")
    for r in all_rows:
        xc.write_row(raw, {"exp": "S4", **r})
    for c, f in feats_all.items():
        for s, x in f.items():
            xc.write_row(raw, {"exp": "S4", "cell": c, "seed": s, "arm": "_features", **x})
    questions = {q: [c for c in cs if c in out] for q, cs in QUESTIONS.items()}
    return {"cells": out, "questions": questions, "reproduction_vs_s1_holdout": repro,
            "arms": {a: {"kind": k, "dials": kw or ("RECOMMENDED" if k == "smart" else "defaults")} for a, k, kw in arms},
            "tune_file": tune_file.name if bkw is not None else None,
            "seeds": {"base": {"first": seeds[0], "count": len(seeds)},
                      "other_cells": {"first": small[0], "count": len(small)}, "paired": True,
                      "role": "held out (14000–14099): never used for tuning any strategy"},
            "cpu_busy_seconds": busy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tune", "test"), required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seeds-other", type=int, default=None, help="test: seeds for the non-base cells")
    parser.add_argument("--first-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cells", default="")
    parser.add_argument("--arms", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--tune-file", default=None)
    args = parser.parse_args()
    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir)
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if args.first_seed is None:
        args.first_seed = 13000 if args.stage == "tune" else 14000
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    name = ("s4_tune" if args.stage == "tune" else "s4") + args.tag
    log_path = OUT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    log = s2.Log(log_path)
    free0 = xc.free_bytes(device)
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        report = stage_tune(args, log, device) if args.stage == "tune" else stage_test(args, log, device)
    seconds = time.perf_counter() - start
    report = {"experiment": name.upper(),
              "question": "Do the strategies generalise to unseen worlds, other electorate sizes (N ≤ 10⁴) and other "
                          "treasuries? (+ budget-aware playbook)", "n_max": N_MAX, **report,
              "provenance": xc.provenance_v3(args, device, seconds, workers=args.workers, free_at_start=free0,
                                             monitor=monitor, device_record=dev_record,
                                             extra={"env": {"OMP_NUM_THREADS": "4"}})}
    report["provenance"]["command"] = "OMP_NUM_THREADS=4 " + report["provenance"]["command"]
    xc.save(str(OUT / f"{name}.json"), report)
    log(f"[{name}] done in {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
