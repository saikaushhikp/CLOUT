"""S2 — Playbook player (strategies.md Method 1) tuned with a small policy search (Method 2), main tournament.

Commands (write under reports/v3/experimentation/):
  # 1. tuning on TRAINING worlds (seeds 13000–13099, disjoint from the X6 test block)
  OMP_NUM_THREADS=4 python experiments/s2_playbook.py --stage tune --seeds 100 --first-seed 13000 --device cuda:0
  # 2. test on the X6 tournament worlds (seeds 12000–12099), paired with reports/v3/x6.json / x13.json
  OMP_NUM_THREADS=4 python experiments/s2_playbook.py --stage test --seeds 100 --first-seed 12000 --device cuda:0

World construction and scoring are X6's (``x6_tournament.seed_task``): ``GameConfig(seed, N = 10⁴, T = 5, ₫500)``,
``generate_electorate`` on the device, ``GameEngine.new_game(..., game_id=f"x6_{seed}")``, full ``play_game`` on
observation dicts only; E = the result's expected vote share, win = the official result. Engine streams depend only
on the seed, so a player's game is identical to what it would have been inside X6 (checked: the re-run top3 and
uniform reproduce raw/x6.jsonl exactly; see ``pairing_check``).

Tuning (stage "tune"): coordinate search, every arm on the same training worlds.
  A  survey × manifesto rule (human: dossier hunches + blend button) at the Method 1 default spending
     (3 workers in the friendly region in round 1, media in all 4 regions in rounds 1 and 2, the rest in round 3);
     comparison arms (NOT human): Bayes-MAP hunches, election-day forecast blend.
  B  spending dials (workers, placement, media regions in rounds 1/2, media order, pace of later rounds) with A's
     best human survey + rule;
  C  A again (top survey × rule pairs) with B's best spending.
  The chosen dials (highest mean E on the training worlds) are written to s2_tune.json → ``chosen``.
Test (stage "test"): baselines uniform / top3 / people_pleaser / bayes (default BayesPlanner, re-run), the untuned
Method 1 playbook, the tuned playbook (headline, human-executable), every manifesto rule with the tuned spending,
and the non-human comparison arms; plus stored references (X6 bayes / ORACLE per seed, X13 search bound per seed).
Hard worlds: "divided" = TV distance between the archetypes of the two largest regions ≥ 0.35 (R1/R4/X15 split),
"fragmented" = inter-region friendship share in the bottom tercile of the 100 worlds.
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
import x_common as xc  # noqa: E402
import x_playbook as xp  # noqa: E402

OUT = xc.ROOT / "reports" / "v3" / "experimentation"   # --out-dir overrides (smoke runs)
X6_RAW = xc.RAW / "x6.jsonl"
X13_PATH = xc.REPORTS / "x13.json"
DEFAULT = dict(survey_regions=2, survey_set="WA+IE", manifesto_rule="coalition", workers=3, worker_place="friendly",
               media1=4, media2=4, media_order="size", later="front")   # Method 1 as written in strategies.md
SPEND_KEYS = ("workers", "workers2", "worker_place", "media1", "media2", "media_order", "later")


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        with self.path.open("a") as f:
            f.write(msg + "\n")


# ============================================================================== one world, many arms
def make_player(kind: str, kwargs: dict, seed: int, device: str):
    from dumbocracy.strategies.players import BayesPlanner, Top3Player, UniformPlayer
    import x_players
    if kind == "uniform":
        return UniformPlayer()
    if kind == "top3":
        return Top3Player()
    if kind == "people_pleaser":
        return x_players.PeoplePleaser(seed=seed, device=device)
    if kind == "bayes":
        return BayesPlanner(seed=seed, device=device)
    if kind == "playbook":
        return xp.PlaybookPlayer(seed=seed, **kwargs)
    if kind == "adaptive":
        return xp.AdaptivePlaybook(seed=seed, **kwargs)
    raise ValueError(kind)


def world_features(e, obs0: dict) -> dict:
    sizes = [int(x) for x in e.sizes.tolist()]
    sigma = [int(x) for x in e.archetype_of_region.tolist()]
    order = sorted(range(4), key=lambda r: -sizes[r])
    ids = [a.id for a in xp.ARCHETYPES]
    return {"sizes": sizes, "sigma": sigma,
            "tv_two_largest": xp.archetype_tv(ids[sigma[order[0]]], ids[sigma[order[1]]]),
            "mean_friends": xp.mean_friends(obs0), "inter_share": xp.inter_share(obs0)}


def seed_task(task: tuple) -> dict:
    """All arms on one electorate. ``task = (seed, cond, arms, device)``; ``cond`` = dict with n, rounds,
    mean_degree, inter_prob_range (None = default); ``arms`` = [(name, kind, kwargs)]."""
    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.strategies.players import play_game
    from strategies import spend_breakdown
    seed, cond, arms, device = task
    torch.set_num_threads(4)
    dev = torch.device(device)
    start = time.perf_counter()
    ekw = {"num_voters": int(cond.get("n", 10_000))}
    if cond.get("mean_degree") is not None:
        ekw["mean_degree"] = float(cond["mean_degree"])
    if cond.get("inter_prob_range") is not None:
        ekw["inter_prob_range"] = tuple(cond["inter_prob_range"])
    cfg = GameConfig(seed=seed, device=device, rounds=int(cond.get("rounds", 5)), electorate=ElectorateConfig(**ekw))
    e = generate_electorate(cfg, device=dev)
    state0 = GameEngine.new_game(cfg, electorate=e, game_id=f"x6_{seed}")
    feats = world_features(e, observation_for(state0, "p0"))
    rows = []
    for name, kind, kwargs in arms:
        player = make_player(kind, kwargs, seed, device)
        g0 = time.perf_counter()
        final, log = play_game(state0, player)
        obs = observation_for(final, "p0")
        res = obs["result"]
        row = {"seed": seed, "cond": cond.get("name", "default"), "arm": name, "kind": kind,
               "expected": res["expected_share"], "vote_share": res["vote_share"], "victory": bool(res["victory"]),
               "spend": spend_breakdown(obs, cfg), "rejected": sum(1 for *_, ok, _c in log if not ok),
               "game_seconds": time.perf_counter() - g0, "manifesto": [float(x) for x in obs["manifesto"]],
               "allocations": [{"round": int(a["round"]), "hires": [int(h) for h in a["hires_by_region"]],
                                "media": [bool(m) for m in a["media_regions"]]}
                               for a in sorted(obs["allocations"], key=lambda a: a["round"])],
               "num_queries": len(obs["queries"])}
        rec = getattr(player, "record", None)
        if rec:
            row["record"] = dict(rec)
        if kind in ("playbook", "adaptive"):
            row["human_executable"] = player.human_executable
        rows.append(row)
    return {"seed": seed, "cond": cond.get("name", "default"), "rows": rows, "features": feats,
            "seconds": time.perf_counter() - start}


def run_arms(seeds, cond, arms, device, workers, label, chunk=None):
    """Every arm on every seed (one task per seed, or per (seed, arm chunk) when ``chunk``)."""
    groups = [arms] if not chunk else [arms[i:i + chunk] for i in range(0, len(arms), chunk)]
    tasks = [(s, cond, g, device) for s in seeds for g in groups]
    results = xc.run_tasks(seed_task, tasks, workers, label=label)
    feats = {r["seed"]: r["features"] for r in results}
    rows = [row for r in results for row in r["rows"]]
    return rows, feats, sum(r["seconds"] for r in results)


def table(rows, seeds) -> dict[str, dict[int, dict]]:
    by: dict[str, dict[int, dict]] = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    return {a: v for a, v in by.items() if all(s in v for s in seeds)}


def mean_E(by, arm, seeds) -> float:
    return sum(by[arm][s]["expected"] for s in seeds) / len(seeds)


# ============================================================================== tuning
def arm_name(kw: dict) -> str:
    return "pb|" + ",".join(f"{k}={kw[k]}" for k in sorted(kw))


def stage_tune(args, log, device) -> dict:
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    assert not set(seeds) & set(range(12000, 12100)), "training seeds must be disjoint from the test block"
    cond = {"name": "default"}
    all_rows, report = [], {"stages": {}}
    surveys = [(k, s) for k in (2, 3, 4) for s in ("AI", "WA+IE")]

    # ---- A: survey × rule (human), at Method-1 spending; + non-human arms
    spend = {k: DEFAULT[k] for k in SPEND_KEYS if k in DEFAULT}
    armsA = []
    for (k, s), rule in itertools.product(surveys, xp.MANIFESTO_RULES):
        kw = {**spend, "survey_regions": k, "survey_set": s, "manifesto_rule": rule}
        armsA.append((arm_name(kw), "playbook", kw))
    for k, rule in itertools.product((1, 2, 4), xp.MANIFESTO_RULES):   # NOT human: Bayes-MAP hunches
        kw = {**spend, "survey_regions": k, "survey_set": "AI", "manifesto_rule": rule, "hunch_source": "bayes_map"}
        armsA.append((arm_name(kw), "playbook", kw))
    log(f"[s2 tune] stage A: {len(armsA)} arms × {len(seeds)} seeds")
    rows, feats, _ = run_arms(seeds, cond, armsA, str(device), args.workers, "[s2 tune A]")
    all_rows += rows
    by = table(rows, seeds)
    human = [a for a, _, kw in armsA if kw.get("hunch_source", "dossier") == "dossier"]
    rankA = sorted(human, key=lambda a: -mean_E(by, a, seeds))
    report["stages"]["A"] = {a: mean_E(by, a, seeds) for a in sorted(by, key=lambda a: -mean_E(by, a, seeds))}
    bestA = next(kw for a, _, kw in armsA if a == rankA[0])
    log(f"[s2 tune] A best human: {rankA[0]} E={mean_E(by, rankA[0], seeds):.4f}")
    # election-day comparison arms (NOT human) on the 6 best human survey × rule pairs
    top_pairs = []
    for a in rankA:
        kw = next(kw for n, _, kw in armsA if n == a)
        key = (kw["survey_regions"], kw["survey_set"], kw["manifesto_rule"])
        if key not in top_pairs:
            top_pairs.append(key)
        if len(top_pairs) == 6:
            break
    armsE = []
    for k, s, rule in top_pairs:
        kw = {**spend, "survey_regions": k, "survey_set": s, "manifesto_rule": rule, "election_day": True}
        armsE.append((arm_name(kw), "playbook", kw))
    rows, _, _ = run_arms(seeds, cond, armsE, str(device), args.workers, "[s2 tune A-eday]")
    all_rows += rows
    by = table(all_rows, seeds)
    report["stages"]["A_election_day"] = {
        a: {"E": mean_E(by, a, seeds),
            "today_twin_E": mean_E(by, arm_name({k2: v for k2, v in kw.items() if k2 != "election_day"}), seeds)}
        for a, _, kw in armsE}

    # ---- B: spending dials with A's best human survey + rule
    base = {k: bestA[k] for k in ("survey_regions", "survey_set", "manifesto_rule")}
    armsB = []
    for W, place, m1, m2, order, later in itertools.product((0, 1, 2, 3, 4, 6, 8), ("friendly", "largest", "size"),
                                                            (2, 3, 4), (2, 3, 4), ("size", "match"), ("front", "even")):
        if W == 0 and place != "friendly":
            continue
        kw = {**base, "workers": W, "worker_place": place, "media1": m1, "media2": m2, "media_order": order,
              "later": later}
        armsB.append((arm_name(kw), "playbook", kw))
    log(f"[s2 tune] stage B: {len(armsB)} arms × {len(seeds)} seeds")
    rows, _, _ = run_arms(seeds, cond, armsB, str(device), args.workers, "[s2 tune B]")
    all_rows += rows
    by = table(all_rows, seeds)
    rankB = sorted([a for a, _, _ in armsB], key=lambda a: -mean_E(by, a, seeds))
    report["stages"]["B_top30"] = {a: mean_E(by, a, seeds) for a in rankB[:30]}
    report["stages"]["B_by_workers"] = {
        str(W): max(mean_E(by, a, seeds) for a, _, kw in armsB if kw["workers"] == W) for W in (0, 1, 2, 3, 4, 6, 8)}
    bestB = next(kw for a, _, kw in armsB if a == rankB[0])
    log(f"[s2 tune] B best: {rankB[0]} E={mean_E(by, rankB[0], seeds):.4f}")

    # ---- C: survey × rule again with B's spending
    spendB = {k: bestB[k] for k in SPEND_KEYS if k in bestB}
    armsC = []
    for (k, s), rule in itertools.product(surveys, xp.MANIFESTO_RULES):
        kw = {**spendB, "survey_regions": k, "survey_set": s, "manifesto_rule": rule}
        armsC.append((arm_name(kw), "playbook", kw))
    for rule, lam in itertools.product(("coalition", "anchor", "divided", "two_largest", "three_largest"),
                                       (0.25, 0.5, 0.75)):   # lean dial (hand arithmetic), 4-region survey only
        kw = {**spendB, "survey_regions": 4, "survey_set": "WA+IE", "manifesto_rule": rule, "lean": lam}
        armsC.append((arm_name(kw), "playbook", kw))
    rows, _, _ = run_arms(seeds, cond, armsC, str(device), args.workers, "[s2 tune C]")
    all_rows += rows
    by = table(all_rows, seeds)
    rankC = sorted([a for a, _, _ in armsC], key=lambda a: -mean_E(by, a, seeds))
    report["stages"]["C"] = {a: mean_E(by, a, seeds) for a in rankC}
    chosen = next(kw for a, _, kw in armsC if a == rankC[0])
    log(f"[s2 tune] C best (chosen): {rankC[0]} E={mean_E(by, rankC[0], seeds):.4f}")
    default_arm = arm_name({**DEFAULT})
    report.update({"chosen": chosen, "chosen_arm": rankC[0], "chosen_train_E": mean_E(by, rankC[0], seeds),
                   "method1_default": DEFAULT,
                   "method1_default_train_E": mean_E(by, default_arm, seeds) if default_arm in by else None,
                   "seeds": {"first": seeds[0], "count": len(seeds), "role": "training (disjoint from 12000–12099)"},
                   "num_arms": len(by), "games": len(all_rows)})
    raw = OUT / "raw" / f"s2_tune{args.tag}.jsonl"
    raw.write_text("")
    for r in all_rows:
        xc.write_row(raw, {"exp": "S2-tune", **{k: v for k, v in r.items() if k not in ("allocations",)}})
    return report


# ============================================================================== test
def load_refs(seeds):
    refs: dict[str, dict[int, dict]] = {}
    for r in xc.read_rows(X6_RAW):
        refs.setdefault(f"x6:{r['player']}", {})[r["seed"]] = r
    x13 = json.loads(X13_PATH.read_text())
    return refs, x13


def summarise(by, seeds, feats, refs, x13, names) -> dict:
    def vals(a, key="expected"):
        return [by[a][s][key] for s in seeds]

    bound = [x13["E20_search_by_seed"][str(s)] for s in seeds]
    bayes_x6 = [refs["x6:bayes"][s]["expected"] for s in seeds]
    divided = [s for s in seeds if feats[s]["tv_two_largest"] >= xp.DIVIDED_TV]
    united = [s for s in seeds if s not in divided]
    shares = sorted(feats[s]["inter_share"] for s in seeds)
    cut = shares[len(shares) // 3]
    fragmented = [s for s in seeds if feats[s]["inter_share"] < cut]
    out = {"players": {}, "splits": {"divided": divided, "united": united, "fragmented_inter_share_lt": cut,
                                     "fragmented": fragmented}}
    for a in names:
        rs = [by[a][s] for s in seeds]
        d = {"expected": xc.mean_ci(vals(a)), "win": xc.rate(vals(a, "victory")),
             "vs_bayes": xc.paired(vals(a), vals("bayes")), "vs_bayes_x6_stored": xc.paired(vals(a), bayes_x6),
             "vs_top3": xc.paired(vals(a), vals("top3")), "gap_to_x13_bound": xc.paired(bound, vals(a)),
             "mcnemar_vs_bayes": xc.mcnemar(vals(a, "victory"), vals("bayes", "victory")),
             "mcnemar_vs_top3": xc.mcnemar(vals(a, "victory"), vals("top3", "victory")),
             "spend": {k: xc.mean_ci([r["spend"].get(k, 0.0) for r in rs]) for k in ("workers", "media", "queries", "total")},
             "mean_queries": sum(r["num_queries"] for r in rs) / len(rs),
             "rejected_actions": sum(r["rejected"] for r in rs),
             "human_executable": rs[0].get("human_executable")}
        for split in ("divided", "united", "fragmented"):
            ss = out["splits"][split]
            v = [by[a][s]["expected"] for s in ss]
            d[split] = {"n": len(ss), "expected": xc.mean_ci(v), "win": xc.rate([by[a][s]["victory"] for s in ss]),
                        "losses": sum(1 for s in ss if not by[a][s]["victory"]),
                        "vs_bayes": xc.paired(v, [by["bayes"][s]["expected"] for s in ss]),
                        "vs_top3": xc.paired(v, [by["top3"][s]["expected"] for s in ss]),
                        "gap_to_x13_bound": xc.paired([x13["E20_search_by_seed"][str(s)] for s in ss], v)}
        out["players"][a] = d
    out["references"] = {
        "x13_search_bound": xc.mean_ci(bound),
        "x6_bayes_stored": xc.mean_ci(bayes_x6), "x6_bayes_stored_win": xc.rate([refs["x6:bayes"][s]["victory"] for s in seeds]),
        "x6_oracle": xc.mean_ci([refs["x6:ORACLE"][s]["expected"] for s in seeds]),
        "bayes_rerun_minus_x6_stored": xc.paired(vals("bayes"), bayes_x6),
        "note": "x13 bound = X13 open-loop plan search with full information, mean E over 20 fresh streams per seed "
                "(players here: one canonical-stream game per seed)"}
    for split in ("divided", "united", "fragmented"):
        ss = out["splits"][split]
        out["references"][f"x13_bound_{split}"] = xc.mean_ci([x13["E20_search_by_seed"][str(s)] for s in ss])
    out["ranking_by_mean_E"] = sorted(names, key=lambda a: -out["players"][a]["expected"]["mean"])
    return out


def stage_test(args, log, device) -> dict:
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    tune = json.loads(Path(args.tune_file or OUT / "s2_tune.json").read_text())
    chosen = tune["chosen"]
    spend = {k: chosen[k] for k in SPEND_KEYS if k in chosen}
    arms = [("uniform", "uniform", {}), ("top3", "top3", {}), ("people_pleaser", "people_pleaser", {}),
            ("playbook", "playbook", dict(chosen)), ("playbook_method1_untuned", "playbook", dict(DEFAULT))]
    for k, s in ((2, "WA+IE"), (3, "WA+IE"), (4, "WA+IE"), (2, "AI"), (4, "AI")):
        for rule in xp.MANIFESTO_RULES:
            kw = {**spend, "survey_regions": k, "survey_set": s, "manifesto_rule": rule}
            if kw != chosen:
                arms.append((f"rule:{rule}|survey={k}x{s}", "playbook", kw))
    for rule in xp.MANIFESTO_RULES:   # NOT human
        arms.append((f"nonhuman_bayesmap:{rule}", "playbook", {**chosen, "survey_regions": 2, "survey_set": "AI",
                                                                "manifesto_rule": rule, "hunch_source": "bayes_map"}))
    arms.append(("nonhuman_election_day", "playbook", {**chosen, "election_day": True}))
    arms.append(("bayes", "bayes", {}))
    log(f"[s2 test] {len(arms)} arms × {len(seeds)} seeds on {device}")
    rows, feats, busy = run_arms(seeds, {"name": "default"}, arms, str(device), args.workers, "[s2 test]")
    by = table(rows, seeds)
    refs, x13 = load_refs(seeds)
    pairing = {p: max(abs(by[p][s]["expected"] - refs[f"x6:{p}"][s]["expected"]) for s in seeds)
               for p in ("uniform", "top3", "people_pleaser")}
    log(f"[s2 test] pairing check vs raw/x6.jsonl (max |ΔE|): {pairing}")
    names = [a for a, _, _ in arms]
    summary = summarise(by, seeds, feats, refs, x13, names)
    raw = OUT / "raw" / f"s2{args.tag}.jsonl"
    raw.write_text("")
    for r in rows:
        xc.write_row(raw, {"exp": "S2", **r})
    for s in seeds:
        xc.write_row(raw, {"exp": "S2", "seed": s, "arm": "_features", **feats[s]})
    return {"pairing_check_max_abs_dE_vs_x6": pairing, "chosen_dials": chosen, "tune_file": "s2_tune.json",
            "arms": {a: {"kind": k, "dials": kw} for a, k, kw in arms}, **summary,
            "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True, "role": "test (X6 block)"},
            "cpu_busy_seconds": busy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tune", "test"), required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tag", default="", help="suffix for smoke runs (output files s2<tag>.*)")
    parser.add_argument("--out-dir", default=None, help="write outputs here instead of reports/v3/experimentation")
    parser.add_argument("--tune-file", default=None, help="test stage: tuning JSON (default <out>/s2_tune.json)")
    args = parser.parse_args()
    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir)
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if args.first_seed is None:
        args.first_seed = 13000 if args.stage == "tune" else 12000
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=2 * 2**30)
    name = ("s2_tune" if args.stage == "tune" else "s2") + args.tag
    log_path = OUT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    log = Log(log_path)
    free0 = xc.free_bytes(device)
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        report = stage_tune(args, log, device) if args.stage == "tune" else stage_test(args, log, device)
    seconds = time.perf_counter() - start
    report = {"experiment": name.upper(), "question": "Playbook player (Method 1) tuned by a small policy search "
              "(Method 2); human-executable rules", "n": 10_000, "rounds": 5, "treasury": "500.00",
              "tv_table": xp.tv_table(), "survey_sets": {k: list(v) for k, v in xp.SURVEY_SETS.items()}, **report,
              "provenance": xc.provenance_v3(args, device, seconds, workers=args.workers, free_at_start=free0,
                                             monitor=monitor, device_record=dev_record)}
    report["provenance"]["command"] = "OMP_NUM_THREADS=4 " + report["provenance"]["command"]
    xc.save(str(OUT / f"{name}.json"), report)
    log(f"[{name}] done in {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
