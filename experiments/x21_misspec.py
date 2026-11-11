"""X21 — Robustness to model misspecification (plan ``docs/v2/plans/implementation_plan.md`` §7 X21).

Command (reproduces reports/v3/x21.json, reports/v3/raw/x21.jsonl, reports/v3/logs/x21.log and the figures):
  OMP_NUM_THREADS=4 python experiments/x21_misspec.py --seeds 50 --first-seed 14000 --device cuda:2 \
      --out reports/v3/x21.json 2>&1 | tee reports/v3/logs/x21.log

BayesPlanner models the world with the public default parameters (``players.public_config`` = ``GameConfig()``
defaults except N, T and the treasury). Here the ENGINE uses a perturbed config, one true parameter at a time
×0.5 and ×1.5: ζ (zealot fraction), κ (Dirichlet concentration), k̄ (mean degree), ω (social step), β_v (vote
sensitivity), z (vote threshold), p_m (media exposure). 50 paired seeds per arm (block 14000), N = 10⁴, T = 5, ₫500.

Per (arm, seed), on cuda:2:
  bayes          ``BayesPlanner(seed, device)`` — keeps the defaults (misspecified), full ``GameEngine`` game;
  bayes_true     (extra control) ``BayesPlanner(..., config=<true perturbed config>)`` — the same player told the
                 true parameters (its survey answer model ``bayes.build_answer_model()`` takes no config and stays
                 the same), so bayes − bayes_true is the pure cost of misspecification;
  oracle         ``oracle_manifesto`` + ORACLE planner (``balance.policy_planner``) on the TRUE perturbed state, all
                 50 worlds batched with ``x_batch.run_batch`` (canonical streams = the engine's streams).
Primary measure (plan): Δgap = (E_bayes − E_oracle)_arm − (E_bayes − E_oracle)_default, paired by seed.
Secondary: misspecification cost = E_bayes − E_bayes_true (same world, same streams).

Check first (assertion, plan): none of the perturbed values leaks into the observation. For every arm the initial
observation is compared with the default arm's: ``costs`` must be identical; for the dynamics parameters
(ω, β_v, z, p_m) the whole observation must be identical; for the generation parameters (ζ, κ, k̄) only the
realised public census (``regions``, ``inter_edges``) and the free statistic (``free_stats``) may differ; no key
anywhere in the observation (initial or final) may be named like a perturbed config field, except the census
``regions[*].mean_degree`` (the realised mean degree of a region, public by design); and
``public_config(obs)`` must equal the default public config.

z × 1.5 = 1.2375 is outside [0, 1] (ConfigError: P = σ(β_v(S − z)) with S ≤ 1 would give nobody a vote). It is
replaced by two arms on the margin 1 − z: z = 1 − 0.5(1 − z₀) = 0.9125 ("z_margin_x0.5", the ×1.5 direction) and
z = 1 − 1.5(1 − z₀) = 0.7375 ("z_margin_x1.5"); the plan's z × 0.5 = 0.4125 is kept. Recorded as a deviation.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
import x_parity  # noqa: E402
import x_phase5a as p5  # noqa: E402
import x_policies  # noqa: E402
from x_batch import run_batch  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.game import GameEngine  # noqa: E402
from dumbocracy.game.observation import observation_for  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402
from dumbocracy.strategies.players import BayesPlanner, play_game, public_config  # noqa: E402

D = GameConfig()
PARAMS = {  # symbol: (config path, default value, kind)
    "zeta": ("electorate.zealot_fraction", D.electorate.zealot_fraction, "generation"),
    "kappa": ("electorate.dirichlet_concentration", D.electorate.dirichlet_concentration, "generation"),
    "kbar": ("electorate.mean_degree", D.electorate.mean_degree, "generation"),
    "omega": ("dynamics.social_step", D.dynamics.social_step, "dynamics"),
    "beta_v": ("voting.sensitivity", D.voting.sensitivity, "dynamics"),
    "z": ("voting.threshold", D.voting.threshold, "dynamics"),
    "p_m": ("media.exposure_prob", D.media.exposure_prob, "dynamics"),
}
SYMBOL = {"zeta": "ζ", "kappa": "κ", "kbar": "k̄", "omega": "ω", "beta_v": "β_v", "z": "z", "p_m": "p_m"}
FIELD_NAMES = {path.rsplit(".", 1)[1] for path, _, _ in PARAMS.values()}
ALLOWED_GENERATION_DIFF = {"regions", "inter_edges", "free_stats"}


def build_arms() -> dict[str, dict]:
    arms = {"default": {"param": None, "factor": 1.0, "set": []}}
    for name, (path, value, _) in PARAMS.items():
        for f in (0.5, 1.5):
            if name == "z" and f == 1.5:
                continue
            arms[f"{name}_x{f}"] = {"param": name, "factor": f, "value": round(value * f, 10),
                                    "set": [f"{path}={round(value * f, 10)}"]}
    z0 = PARAMS["z"][1]
    for f in (0.5, 1.5):
        v = round(1 - f * (1 - z0), 10)
        arms[f"z_margin_x{f}"] = {"param": "z", "factor": f, "value": v, "set": [f"voting.threshold={v}"],
                                  "note": f"z = 1 − {f}·(1 − z₀) (z × 1.5 is invalid)"}
    return arms


ARMS = build_arms()


def arm_config(base: GameConfig, arm: str, seed: int | None = None) -> GameConfig:
    cfg = xc.apply_overrides(base, ARMS[arm]["set"])
    return dataclasses.replace(cfg, seed=int(seed)) if seed is not None else cfg


# ============================================================================== leak check (plan: check first)
def _keys(obj, path="") -> list[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(f"{path}.{k}" if path else k)
            out += _keys(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for v in obj:
            out += _keys(v, f"{path}[*]")
    return out


def assert_no_config_keys(obs: dict) -> None:
    for p in _keys(obs):
        leaf = p.rsplit(".", 1)[-1]
        if leaf in FIELD_NAMES:
            assert p == "regions[*].mean_degree", f"observation key {p!r} is named like a perturbed config field"


def leak_check(base: GameConfig, seeds: list[int], device: torch.device) -> dict:
    out = {"seeds": seeds, "arms": {}}
    for seed in seeds:
        cfg0 = arm_config(base, "default", seed)
        obs0 = observation_for(GameEngine.new_game(cfg0, electorate=generate_electorate(cfg0, device=device),
                                                   game_id="leak"), "p0")
        assert_no_config_keys(obs0)
        pc0 = public_config(obs0)
        for arm, spec in ARMS.items():
            if arm == "default":
                continue
            cfg = arm_config(base, arm, seed)
            obs = observation_for(GameEngine.new_game(cfg, electorate=generate_electorate(cfg, device=device),
                                                      game_id="leak"), "p0")
            assert obs["costs"] == obs0["costs"], f"{arm}: costs differ"
            diff = sorted(k for k in obs0 if obs0[k] != obs[k])
            kind = PARAMS[spec["param"]][2]
            if kind == "dynamics":
                assert not diff, f"{arm}: dynamics parameter changed the observation: {diff}"
            else:
                assert set(diff) <= ALLOWED_GENERATION_DIFF, f"{arm}: unexpected observation fields differ: {diff}"
            assert_no_config_keys(obs)
            assert public_config(obs) == pc0, f"{arm}: the player's public config changed"
            head, field = PARAMS[spec["param"]][0].split(".")
            assert getattr(getattr(public_config(obs), head), field) == PARAMS[spec["param"]][1] != \
                getattr(getattr(cfg, head), field), f"{arm}: the player's model holds the true value"
            out["arms"].setdefault(arm, {"kind": kind, "differing_fields": set()})["differing_fields"] |= set(diff)
    for a in out["arms"].values():
        a["differing_fields"] = sorted(a["differing_fields"])
    out["passed"] = True
    out["rule"] = ("costs identical; dynamics arms: whole initial observation identical; generation arms: only "
                   f"{sorted(ALLOWED_GENERATION_DIFF)} (realised public census / free statistic) differ; no key named "
                   f"like {sorted(FIELD_NAMES)} except regions[*].mean_degree (realised census); public_config(obs) "
                   "== default public config")
    return out


# ============================================================================== games
def engine_game(cfg: GameConfig, electorate, player) -> dict:
    state0 = GameEngine.new_game(cfg, electorate=electorate, game_id=f"x21_{cfg.seed}")
    g0 = time.perf_counter()
    final, log = play_game(state0, player)
    obs = observation_for(final, "p0")
    assert_no_config_keys(obs)
    res = obs["result"]
    hires = sum(sum(a["hires_by_region"]) for a in obs["allocations"])
    return {"expected": float(res["expected_share"]), "vote_share": float(res["vote_share"]),
            "victory": bool(res["victory"]), "queries": len(obs["queries"]), "workers": int(hires),
            "airings": int(sum(obs["media_airings"].values())), "manifesto": [round(float(x), 5) for x in obs["manifesto"]],
            "game_seconds": time.perf_counter() - g0}


def bayes_task(args) -> dict:
    kind, cfg, electorate, device = args
    player = BayesPlanner(seed=cfg.seed, device=str(device)) if kind == "bayes" else \
        BayesPlanner(seed=cfg.seed, device=str(device), config=cfg)
    return engine_game(cfg, electorate, player)


def oracle_batch(es, cfgs, base_arm: GameConfig, B: int, record: dict) -> list[dict]:
    ms = torch.stack([oracle_manifesto(e, c) for e, c in zip(es, cfgs)])
    idx = list(range(len(es)))
    outs = xc.run_chunks(lambda ch: run_batch([es[i] for i in ch], base_arm, ms[ch], x_policies.policy_planner,
                                              seeds=[cfgs[i].seed for i in ch]), idx, B, record=record)
    return [{"expected": float(o.result.expected_share), "vote_share": float(o.result.vote_share),
             "victory": bool(o.result.victory), "workers": int(sum(sum(a.hires_by_region) for a in o.allocations)),
             "airings": int(sum(sum(a.media_regions) for a in o.allocations))} for o in outs]


# ============================================================================== statistics
def summarise(res: dict[str, dict[str, list[dict]]]) -> dict:
    E = lambda rs: [r["expected"] for r in rs]  # noqa: E731
    W = lambda rs: [r["victory"] for r in rs]  # noqa: E731
    arms = {}
    for arm, runs in res.items():
        a = {k: {"win": xc.rate(W(rs)), "expected": xc.mean_ci(E(rs))} for k, rs in runs.items()}
        a["gap_bayes_minus_oracle"] = xc.paired(E(runs["bayes"]), E(runs["oracle"]))
        a["wins_bayes_vs_oracle"] = xc.mcnemar(W(runs["bayes"]), W(runs["oracle"]))
        if "bayes_true" in runs:
            a["misspec_cost"] = xc.paired(E(runs["bayes"]), E(runs["bayes_true"]))
            a["wins_bayes_vs_bayes_true"] = xc.mcnemar(W(runs["bayes"]), W(runs["bayes_true"]))
        arms[arm] = {**ARMS[arm], **a}
    gap0 = [b - o for b, o in zip(E(res["default"]["bayes"]), E(res["default"]["oracle"]))]
    tests = {"delta_gap": [], "misspec_cost": [], "wins_bayes_vs_oracle": [], "misspec_wins": []}
    for arm, runs in res.items():
        if arm == "default":
            continue
        gap = [b - o for b, o in zip(E(runs["bayes"]), E(runs["oracle"]))]
        tests["delta_gap"].append({"arm": arm, **xc.paired(gap, gap0)})
        tests["misspec_cost"].append({"arm": arm, **arms[arm]["misspec_cost"]})
        tests["misspec_wins"].append({"arm": arm, **arms[arm]["wins_bayes_vs_bayes_true"]})
    for arm in res:
        tests["wins_bayes_vs_oracle"].append({"arm": arm, **arms[arm]["wins_bayes_vs_oracle"]})
    for fam, items in tests.items():
        key = "p" if "wins" in fam else "wilcoxon_p"
        for x, h in zip(items, xc.holm([x[key] for x in items])):
            x["holm_p"] = h
    sens = {}
    for name in PARAMS:
        rows = [t for t in tests["delta_gap"] if ARMS[t["arm"]]["param"] == name]
        cost = [t for t in tests["misspec_cost"] if ARMS[t["arm"]]["param"] == name]
        if not rows:
            continue
        worst = max(rows, key=lambda t: abs(t["mean"]))
        worst_cost = min(cost, key=lambda t: t["mean"])
        sens[name] = {"max_abs_delta_gap": abs(worst["mean"]), "arm": worst["arm"], "delta_gap": worst["mean"],
                      "delta_gap_ci95": worst["ci95"], "holm_p": worst["holm_p"],
                      "max_misspec_cost": -worst_cost["mean"], "misspec_arm": worst_cost["arm"],
                      "misspec_cost_ci95": worst_cost["ci95"], "misspec_holm_p": worst_cost["holm_p"]}
    rank = sorted(sens, key=lambda n: -sens[n]["max_abs_delta_gap"])
    rank_cost = sorted(sens, key=lambda n: -sens[n]["max_misspec_cost"])
    return {"arms": arms, "tests": tests, "sensitivity": sens, "ranking_delta_gap": rank, "ranking_misspec_cost": rank_cost,
            "test_families": {
                "delta_gap": f"{len(tests['delta_gap'])} non-default arms: (E_bayes − E_oracle)_arm − (…)_default, "
                             "Wilcoxon signed-rank, Holm",
                "misspec_cost": f"{len(tests['misspec_cost'])} arms: E_bayes − E_bayes_true, Wilcoxon, Holm",
                "wins_bayes_vs_oracle": f"{len(tests['wins_bayes_vs_oracle'])} arms: exact McNemar bayes vs ORACLE, Holm",
                "misspec_wins": f"{len(tests['misspec_wins'])} arms: exact McNemar bayes vs bayes_true, Holm"}}


# ============================================================================== parity
def parity(base: GameConfig, seeds: list[int], device: torch.device) -> dict:
    """CPU↔GPU on 3 seeds, arm beta_v_x0.5: misspecified BayesPlanner (engine) and ORACLE, canonical CPU draw."""
    def fn(seed, dev):
        cfg = dataclasses.replace(arm_config(base, "beta_v_x0.5", seed), device=str(dev), generation="cpu")
        e = generate_electorate(cfg, device=dev)
        b = engine_game(cfg, e, BayesPlanner(seed=seed, device=str(dev)))
        o = run_batch([e], cfg, oracle_manifesto(e, cfg)[None], x_policies.policy_planner, seeds=[seed])[0]
        return {"E_bayes": b["expected"], "E_oracle": o.result.expected_share, "win_bayes": float(b["victory"])}
    return x_parity.parity_rel(fn, seeds, device=device, abs_tol=0.02)


# ============================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X21"])
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--threads", type=int, default=3, help="Bayes games in flight in this one process")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--no-bayes-true", action="store_true")
    ap.add_argument("--parity-seeds", type=int, default=3)
    ap.add_argument("--leak-seeds", type=int, default=2)
    ap.add_argument("--cap-gib", type=float, default=10.0)
    ap.add_argument("--out", default=str(xc.REPORTS / "x21.json"))
    ap.add_argument("--raw", default=str(xc.RAW / "x21.jsonl"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    device, dev_record = p5.pin_device(args.device, need_bytes=int(args.cap_gib * 2**30))
    free0 = xc.free_bytes(device)
    torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(device=str(device), generation="device", electorate=ElectorateConfig(num_voters=args.n))
    arms = args.arms or list(ARMS)
    raw = Path(args.raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("")
    record: dict = {"arm_seconds": {}}
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        leak = leak_check(base, seeds[:args.leak_seeds], device)
        print(f"[x21] leak check passed ({len(leak['arms'])} arms × {args.leak_seeds} seeds)", flush=True)
        res: dict[str, dict[str, list[dict]]] = {}
        B = None
        for arm in arms:
            t0 = time.perf_counter()
            cfgs = [arm_config(base, arm, s) for s in seeds]
            es = [generate_electorate(c, device=device) for c in cfgs]
            if B is None:
                ms = torch.stack([oracle_manifesto(e, c) for e, c in zip(es[:8], cfgs[:8])])
                cal = p5.capped_batch(lambda b: run_batch(es[:b], cfgs[0], ms[:b], x_policies.policy_planner,
                                                          seeds=seeds[:b]), device, sizes=(4, 8), max_batch=len(seeds),
                                      cap_bytes=int(args.cap_gib * 2**30))
                record["calibration"] = cal
                B = cal["batch"]
            runs = {"oracle": oracle_batch(es, cfgs, arm_config(base, arm), B, record)}
            kinds = ["bayes"] + ([] if (arm == "default" or args.no_bayes_true) else ["bayes_true"])
            tasks = [(k, c, e, device) for k in kinds for c, e in zip(cfgs, es)]
            with ThreadPoolExecutor(args.threads) as pool:
                outs = list(pool.map(bayes_task, tasks))
            for k in kinds:
                runs[k] = outs[kinds.index(k) * len(seeds):(kinds.index(k) + 1) * len(seeds)]
            if arm == "default":
                runs["bayes_true"] = runs["bayes"]   # identical player and world (not stored twice)
            res[arm] = runs
            for k, rs in runs.items():
                if arm == "default" and k == "bayes_true":
                    continue
                for s, r in zip(seeds, rs):
                    xc.write_row(raw, {"exp": "X21", "arm": arm, "set": ARMS[arm]["set"], "seed": s, "player": k, **r})
            record["arm_seconds"][arm] = time.perf_counter() - t0
            b, o = runs["bayes"], runs["oracle"]
            print(f"[x21] {arm}: {record['arm_seconds'][arm]:.0f}s; win bayes {sum(r['victory'] for r in b) / len(b):.2f} "
                  f"oracle {sum(r['victory'] for r in o) / len(o):.2f} bayes_true "
                  f"{sum(r['victory'] for r in runs['bayes_true']) / len(b):.2f}; E bayes "
                  f"{sum(r['expected'] for r in b) / len(b):.3f} oracle {sum(r['expected'] for r in o) / len(o):.3f}",
                  flush=True)
            del es
            torch.cuda.empty_cache()
        games_seconds = time.perf_counter() - start
        par = parity(base, seeds[:args.parity_seeds], device)
        print(f"[x21] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    summ = summarise(res)
    report = {
        "experiment": "X21", "title": "Robustness to model misspecification",
        "question": "BayesPlanner assumes the public default parameters. How badly does it do when the true world differs?",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget),
        "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
        "parameters": {k: {"symbol": SYMBOL[k], "path": v[0], "default": v[1], "kind": v[2]} for k, v in PARAMS.items()},
        "players": {"bayes": "BayesPlanner(seed, device=cuda:2), default public config (misspecified)",
                    "bayes_true": "BayesPlanner(seed, device=cuda:2, config=true perturbed config) (extra control)",
                    "oracle": "oracle_manifesto + balance.policy_planner on the true state (run_batch)"},
        "metric_definitions": {
            "gap_bayes_minus_oracle": "paired E_bayes − E_oracle in the same perturbed world",
            "delta_gap": "(E_bayes − E_oracle)_arm − (E_bayes − E_oracle)_default, paired by seed (primary)",
            "misspec_cost": "E_bayes − E_bayes_true (same world and streams; negative = misspecification hurts)",
            "sensitivity.max_abs_delta_gap": "max over the parameter's arms of |mean delta_gap| (primary ranking)",
            "sensitivity.max_misspec_cost": "max over the parameter's arms of −mean misspec_cost (secondary ranking)"},
        "leak_check": leak, **summ,
        "predicted": {"most_sensitive": ["z", "beta_v"], "least_sensitive": "kappa"},
        "deviations": [
            "z × 1.5 = 1.2375 is invalid (z must lie in [0, 1]); replaced by z = 0.9125 (margin 1 − z halved) and "
            "z = 0.7375 (margin × 1.5). The plan's z × 0.5 = 0.4125 is kept.",
            "Extra control arm bayes_true (BayesPlanner told the true parameters), beyond the plan's ORACLE comparison.",
            "Electorates drawn on the GPU (generation='device'); parity uses the canonical CPU draw.",
            f"BayesPlanner games run {args.threads} at a time in threads of one process (one GPU process at a time)."],
        "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, workers=args.threads,
                                       batch_sizes={"oracle_run_batch_B": record.get("batch_sizes"),
                                                    "bayes_threads": args.threads},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"calibration": record.get("calibration"), "arm_seconds": record["arm_seconds"],
                                              "games_seconds": games_seconds,
                                              "command": "python experiments/x21_misspec.py --seeds %d --first-seed %d "
                                                         "--device %s --out %s" % (len(seeds), seeds[0], args.device,
                                                                                   args.out)}),
    }
    xc.save(args.out, report)
    if not args.no_plots:
        plots(report)
        xc.save(args.out, report)
    print(f"[x21] ranking Δgap {summ['ranking_delta_gap']}; misspec cost {summ['ranking_misspec_cost']}", flush=True)
    c = report["provenance"]["compute"]
    print(f"[x21] wall {seconds:.0f}s, peak {c['peak_bytes'] / 2**30:.2f} GiB, util {c.get('gpu_util_mean')}", flush=True)


def plots(report: dict) -> None:
    figs = []
    arms = [a for a in report["arms"] if a != "default"]
    dg = {t["arm"]: t for t in report["tests"]["delta_gap"]}
    mc = {t["arm"]: t for t in report["tests"]["misspec_cost"]}
    fig, axes = p5.new_figure(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, tests, title in ((axes[0], dg, "Δgap = (E_bayes − E_oracle)_arm − default"),
                             (axes[1], mc, "misspecification cost = E_bayes − E_bayes_true")):
        p5.style(ax)
        ax.grid(axis="x", color=p5.GRID, linewidth=0.6)
        ys = list(range(len(arms)))
        means = [tests[a]["mean"] for a in arms]
        lo = [tests[a]["mean"] - tests[a]["ci95"][0] for a in arms]
        hi = [tests[a]["ci95"][1] - tests[a]["mean"] for a in arms]
        cols = [p5.SERIES[list(PARAMS).index(report["arms"][a]["param"])] for a in arms]
        ax.barh(ys, means, color=cols, height=0.7)
        ax.errorbar(means, ys, xerr=[lo, hi], fmt="none", ecolor=p5.INK2, elinewidth=0.8, capsize=2)
        ax.axvline(0, color=p5.INK2, linewidth=0.8)
        ax.set_title(title, fontsize=9, color=p5.INK)
        ax.set_xlabel("paired mean difference in E (95% t-CI)", fontsize=8, color=p5.INK2)
    axes[0].set_yticks(range(len(arms)))
    axes[0].set_yticklabels([f"{a} ({report['arms'][a]['value']})" for a in arms], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("X21: BayesPlanner under one misspecified parameter (50 paired seeds per arm)", fontsize=10)
    figs.append(p5.save_figure(fig, "x21_sensitivity.png"))

    fig, ax = p5.new_figure(figsize=(11, 3.8))
    p5.style(ax)
    allarms = list(report["arms"])
    for j, k in enumerate(("bayes", "bayes_true", "oracle")):
        xs = [i + (j - 1) * 0.27 for i in range(len(allarms))]
        r = [report["arms"][a][k]["win"] for a in allarms]
        ax.bar(xs, [x["rate"] for x in r], 0.25, color=p5.SERIES[j], label=k)
        ax.errorbar(xs, [x["rate"] for x in r], yerr=[[x["rate"] - x["ci95"][0] for x in r],
                                                     [x["ci95"][1] - x["rate"] for x in r]],
                    fmt="none", ecolor=p5.INK2, elinewidth=0.7, capsize=1.5)
    ax.set_xticks(range(len(allarms)))
    ax.set_xticklabels(allarms, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("win rate (95% Wilson CI)", fontsize=8, color=p5.INK2)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_title("X21: win rates per arm", fontsize=10, color=p5.INK)
    figs.append(p5.save_figure(fig, "x21_win_rates.png"))
    report["figures"] = figs


if __name__ == "__main__":
    main()
