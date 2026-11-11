"""X5 — Value of information at ₫10 (plan ``docs/v2/plans/implementation_plan.md`` §6 X5).

Command (reproduces reports/v3/x5.json, reports/v3/raw/x5.jsonl and reports/v3/logs/x5.log):
  OMP_NUM_THREADS=4 python experiments/x5_info_value.py --seeds 50 --first-seed 13000 --device cuda:2 \
      --out reports/v3/x5.json 2>&1 | tee reports/v3/logs/x5.log

A BayesPlanner is forced to ask exactly k survey queries, k ∈ {0, 1, 2, 4, 8, 12}, chosen by expected information
gain ("gain", the library rule) or uniformly at random ("random"; same count, same later play), at the game price
₫10 per query and at the comparison price ₫5. Every stop rule is disabled and no in-round queries are asked
(``round_query_gain=None``), so after the survey knowledge grows only from the free statistics and the polls.

**Budget fix (plan X5).** BayesPlanner's default ``survey_budget = ₫50`` would cap it at 5 queries at ₫10 and
silently truncate the 8 and 12 arms. Here ``survey_budget = k × price`` and ``num_queries = k`` per arm (both
recorded per row; the realised query count is checked).

Per game (paired: every arm plays the same 50 electorates, seed block 13000, N = 10⁴, T = 5, ₫500, ₫20 workers):
* outcome: expected share E (deterministic, spec §5), official victory, spending by category;
* posterior accuracy after the survey (decision point = start of round 1): P(true σ), MAP σ fully correct,
  per-region MAP accuracy, entropy over σ (nats);
* manifesto quality: the committed b's static expected share on the anticipated TRUE electorate (no campaign),
  and the same for the ORACLE manifesto (reference);
* per-round epistemic verdicts: ``logic.report.timeline_with_diagnostics`` (16 poll-consistent worlds of 1,600
  voters per round, the library plan set Π_t; worlds sampled and replayed on the GPU, batched).

GPU first (§1.1): engine state (true electorate, drawn by the canonical CPU generator R8 and moved to the GPU),
the planner's posterior worlds and rollouts, and the logic layer's world sampling all run on ``cuda:2``. The
electorate is drawn once per seed and shared by all 22 arms. Games are sequential Python decisions, so the
parallelism is over worlds × candidate plans × voters inside each decision (WorldBatch), not over games.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import time
import warnings
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x_parity import parity_rel  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import REGIONS, SECTORS  # noqa: E402
from dumbocracy.strategies.players import BayesPlanner  # noqa: E402

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

KS = (0, 1, 2, 4, 8, 12)
PRICES = ("10.00", "5.00")
RULES = ("gain", "random")
RAW_PATH = xc.RAW / "x5.jsonl"
GPU_MEMORY_CAP = 10 * 2**30          # lead's instruction: ≤ ~10 GB on the shared cuda:2
TIMELINE_WORLDS = 16
TIMELINE_VOTERS = 1600


# ============================================================================== device (hard-pinned)
def pinned_device(requested: str) -> tuple[torch.device, dict]:
    """Use exactly ``requested`` (never another GPU): refuses to fall back. On CUDA the process is capped at
    ``GPU_MEMORY_CAP`` with ``set_per_process_memory_fraction`` (shared host)."""
    if requested == "cpu":
        return torch.device("cpu"), {"requested": "cpu", "chosen": "cpu"}
    device = torch.device(requested)
    stats = xc.gpu_stats()
    idx = device.index or 0
    free = stats.get(idx, {}).get("free", 0)
    record = {"requested": requested, "chosen": str(device), "pinned": True, "free_bytes_at_select": int(free),
              "memory_cap_bytes": GPU_MEMORY_CAP, "util_at_select": stats.get(idx, {}).get("util")}
    if free < 2 * 2**30:
        raise SystemExit(f"{requested} has only {free / 2**30:.1f} GiB free; refusing to fall back to another GPU")
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_MEMORY_CAP / total), device)
    return device, record


# ============================================================================== players
class RandomQueryPlanner(BayesPlanner):
    """BayesPlanner whose survey queries are uniformly random (pair of sectors, region), keyed by the player seed
    and the query index (same count and same later play as the information-gain player). Ported from the v2
    ``scenario_table.RandomQueryPlanner``."""

    def _maybe_query(self, observation, posterior, survey):
        if not survey or "query" not in observation["legal_actions"]:
            return None
        cost = Decimal(observation["costs"]["query"])
        spent = sum((Decimal(str(q["cost"])) for q in observation.get("queries", [])), Decimal("0.00"))
        if spent + cost > self.survey_budget or Decimal(str(observation["treasury"])) < cost:
            return None
        asked = [q for q in observation.get("queries", []) if q.get("phase") == "SURVEY"]
        if self.num_queries is not None and len(asked) >= self.num_queries:
            return None
        rng = random.Random(f"{self.seed}:{observation['seed']}:{len(asked)}")
        a, b = rng.sample(SECTORS, 2)
        return {"type": "query", "sector_a": a, "sector_b": b, "region": rng.choice(REGIONS)}


def forced_player(k: int, rule: str, price: Decimal, seed: int, device: str) -> BayesPlanner:
    """Exactly k survey queries (budget fix: survey_budget = k × price, num_queries = k), no stop rules, no
    in-round queries."""
    cls = BayesPlanner if rule == "gain" else RandomQueryPlanner
    return cls(seed=seed, num_queries=k, survey_budget=Decimal(k) * price, stop_entropy=None, min_gain_per_coin=None,
               round_query_gain=None, device=device)


def arm_name(price: str, k: int, rule: str) -> str:
    return f"p{int(Decimal(price))}_k{k}_{'none' if k == 0 else rule}"


ARMS: list[tuple[str, int, str]] = [(p, k, r) for p in PRICES for k in KS for r in RULES if not (k == 0 and r == "random")]


# ============================================================================== one seed
def config_for(seed: int, n: int, device: str, price: str) -> GameConfig:
    base = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    return dataclasses.replace(base, survey=dataclasses.replace(base.survey, query_cost=Decimal(price)))


def sigma_index(sigma: list[int]) -> int:
    from dumbocracy.strategies.bayes import ASSIGNMENTS
    hit = (ASSIGNMENTS == torch.tensor(sigma, dtype=ASSIGNMENTS.dtype)).all(1).nonzero().flatten()
    return int(hit[0])


def posterior_metrics(obs: dict, true_sigma: list[int], fraction: float) -> dict:
    """Belief over σ at the start of round 1 (after the survey, the round-1 free statistic and the round-1 poll)."""
    from dumbocracy.logic.knowledge import observation_at
    from dumbocracy.strategies.bayes import posterior_from_observation
    obs1 = observation_at(obs, 1, stage="start")
    post = posterior_from_observation(obs1, sample_fraction=fraction)
    marg = post.region_marginals()
    mapa = [int(x) for x in post.map_assignment().tolist()]
    return {"p_true_sigma": float(post.probs[sigma_index(true_sigma)]), "map_correct": mapa == true_sigma,
            "map_region_accuracy": sum(int(a == b) for a, b in zip(mapa, true_sigma)) / 4,
            "p_true_region_mean": float(sum(marg[r, true_sigma[r]] for r in range(4)) / 4),
            "entropy_sigma": post.entropy(), "inconsistent": bool(post.inconsistent)}


def spend_of(obs: dict, cfg: GameConfig) -> dict:
    worker_unit = cfg.workers.worker_cost
    workers = sum((sum(a["hires_by_region"]) * worker_unit for a in obs["allocations"]), Decimal("0.00"))
    alloc = sum((Decimal(a["cost"]) for a in obs["allocations"]), Decimal("0.00"))
    queries = sum((Decimal(q["cost"]) for q in obs["queries"]), Decimal("0.00"))
    return {"workers": float(workers), "media": float(alloc - workers), "queries": float(queries),
            "total": float(alloc + queries), "num_queries": len(obs["queries"]),
            "survey_queries": sum(1 for q in obs["queries"] if q["phase"] == "SURVEY"),
            "repeat_queries": sum(1 for q in obs["queries"] if q.get("repeat_index", 1) > 1),
            "media_airings": sum(obs["media_airings"].values()), "unspent": float(Decimal(str(obs["treasury"])))}


def play_arm(state0, electorate, cfg: GameConfig, seed: int, price: str, k: int, rule: str, device: torch.device,
             v_t: torch.Tensor, timeline: bool = True) -> dict:
    from dumbocracy.game.observation import observation_for
    from dumbocracy.logic.report import timeline_with_diagnostics
    from dumbocracy.strategies.manifesto_opt import expected_share
    from dumbocracy.strategies.players import play_game
    player = forced_player(k, rule, Decimal(price), seed, str(device))
    g0 = time.perf_counter()
    final, log = play_game(state0, player)
    game_s = time.perf_counter() - g0
    obs = observation_for(final, "p0")
    res = obs["result"]
    true_sigma = [int(x) for x in electorate.archetype_of_region.tolist()]
    b = torch.tensor(obs["manifesto"], dtype=v_t.dtype, device=v_t.device)
    spend = spend_of(obs, cfg)
    if spend["survey_queries"] != k:
        raise RuntimeError(f"seed {seed} arm {arm_name(price, k, rule)}: asked {spend['survey_queries']} != k = {k}")
    row: dict[str, Any] = {
        "exp": "X5", "seed": seed, "arm": arm_name(price, k, rule), "price": price, "k": k,
        "rule": "none" if k == 0 else rule, "num_queries_param": k, "survey_budget_param": str(Decimal(k) * Decimal(price)),
        "expected": float(res["expected_share"]), "vote_share": float(res["vote_share"]), "victory": bool(res["victory"]),
        "spend": spend, "manifesto": [round(float(x), 6) for x in obs["manifesto"]],
        "static_share_anticipated": float(expected_share(v_t, b, cfg.voting.sensitivity, cfg.voting.threshold)),
        "posterior": posterior_metrics(obs, true_sigma, cfg.survey.query_sample_fraction),
        "queries": [{kk: q[kk] for kk in ("sector_a", "sector_b", "region")} for q in obs["queries"]],
        "polls": [round(float(p["estimate"]), 4) for p in obs["polls"]],
        "rejected": sum(1 for *_, ok, _c in log if not ok), "game_seconds": game_s,
        "decision_seconds_mean": sum(x for xs in player.latency.values() for x in xs)
        / max(1, sum(len(xs) for xs in player.latency.values())),
    }
    if timeline:
        t0 = time.perf_counter()
        tl, diag = timeline_with_diagnostics(final, "p0", worlds=TIMELINE_WORLDS, world_voters=TIMELINE_VOTERS, seed=0,
                                             device=device)
        row["timeline"] = [{"round": r["round"], "possible": r["possible_victory"], "guaranteed": r["guaranteed_victory"],
                            "knows_win_exists": r["knows_win_exists"], "inevitable_defeat": r["inevitable_defeat"],
                            "truth_possible": r["truth_victory_possible"],
                            "entropy_sigma": r["knowledge"]["posterior_entropy"],
                            "best_win_fraction": r["best_plan"]["win_fraction"]} for r in tl["rounds"]]
        row["timeline_diag"] = [{kk: d[kk] for kk in ("round", "draws", "accepted", "fallback", "num_policies")}
                                for d in diag["rounds"]]
        row["timeline_seconds"] = time.perf_counter() - t0
    return row


def seed_rows(seed: int, n: int, device: torch.device, arms: list[tuple[str, int, str]], done: set,
              timeline: bool = True) -> tuple[list[dict], dict]:
    from dumbocracy.game import GameEngine
    from dumbocracy.strategies.manifesto_opt import expected_share
    from dumbocracy.strategies.oracle import anticipate, oracle_manifesto
    cfg10 = config_for(seed, n, str(device), PRICES[0])
    electorate = generate_electorate(cfg10, device=device)       # canonical CPU draw (R8), moved to the device
    v_t = anticipate(electorate, cfg10)
    b_or = oracle_manifesto(electorate, cfg10)
    info = {"seed": seed, "sigma": [int(x) for x in electorate.archetype_of_region.tolist()],
            "tau": [int(x) for x in electorate.personality_of_region.tolist()],
            "sizes": [int(x) for x in electorate.sizes.tolist()],
            "oracle_static_share": float(expected_share(v_t, b_or, cfg10.voting.sensitivity, cfg10.voting.threshold))}
    states = {}
    rows = []
    for price, k, rule in arms:
        if (seed, arm_name(price, k, rule)) in done:
            continue
        if price not in states:
            states[price] = GameEngine.new_game(config_for(seed, n, str(device), price), electorate=electorate,
                                                game_id=f"x5_{seed}_{price}")
        row = play_arm(states[price], electorate, config_for(seed, n, str(device), price), seed, price, k, rule,
                       device, v_t, timeline)
        row["oracle_static_share"] = info["oracle_static_share"]
        rows.append(row)
    return rows, info


# ============================================================================== parity
def parity(seeds: list[int], device: torch.device, n: int) -> dict:
    """CPU↔GPU on 3 seeds, arm ₫10 k = 2 gain (engine, planner worlds and logic worlds all on each device)."""
    cache: dict = {}

    def fn(seed: int, dev: torch.device) -> dict:
        key = (seed, str(dev))
        if key not in cache:
            rows, _ = seed_rows(seed, n, dev, [("10.00", 2, "gain")], set())
            r = rows[0]
            cache[key] = {"expected": r["expected"], "victory": float(r["victory"]),
                          "static_share": r["static_share_anticipated"],
                          "p_true_sigma": r["posterior"]["p_true_sigma"],
                          "guaranteed_rounds": float(sum(x["guaranteed"] for x in r["timeline"])),
                          "possible_rounds": float(sum(x["possible"] for x in r["timeline"]))}
        return cache[key]

    return parity_rel(fn, seeds, device=device, abs_tol=0.02)


# ============================================================================== statistics
def _by(rows: list[dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for r in rows:
        out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


def summarise(rows: list[dict], seeds: list[int]) -> dict:
    by = _by(rows)
    arms_out: dict[str, Any] = {}
    for price, k, rule in ARMS:
        name = arm_name(price, k, rule)
        rs = [by[name][s] for s in seeds]
        T = len(rs[0]["timeline"])
        g1 = [r["timeline"][0]["guaranteed"] for r in rs]
        wins = [r["victory"] for r in rs]

        def cond(c: list[bool]) -> dict:
            sel = [w for w, x in zip(wins, c) if x]
            return {"n": len(sel), "p": (sum(sel) / len(sel)) if sel else None,
                    "ci95": xc.wilson(sum(sel), len(sel)) if sel else None}

        arms_out[name] = {
            "price": price, "k": k, "rule": "none" if k == 0 else rule, "n": len(rs),
            "win_rate": xc.rate(wins), "expected": xc.mean_ci([r["expected"] for r in rs]),
            "static_share_anticipated": xc.mean_ci([r["static_share_anticipated"] for r in rs]),
            "query_spend": xc.mean_ci([r["spend"]["queries"] for r in rs]),
            "campaign_spend": xc.mean_ci([r["spend"]["workers"] + r["spend"]["media"] for r in rs]),
            "workers_spend": xc.mean_ci([r["spend"]["workers"] for r in rs]),
            "realised_survey_queries": xc.mean_ci([r["spend"]["survey_queries"] for r in rs]),
            "repeat_queries": xc.mean_ci([r["spend"]["repeat_queries"] for r in rs]),
            "posterior": {
                "p_true_sigma": xc.mean_ci([r["posterior"]["p_true_sigma"] for r in rs]),
                "map_correct": xc.rate([r["posterior"]["map_correct"] for r in rs]),
                "map_region_accuracy": xc.mean_ci([r["posterior"]["map_region_accuracy"] for r in rs]),
                "entropy_sigma": xc.mean_ci([r["posterior"]["entropy_sigma"] for r in rs])},
            "verdicts_by_round": {
                key: [xc.rate([r["timeline"][t][key] for r in rs]) for t in range(T)]
                for key in ("guaranteed", "possible", "knows_win_exists", "inevitable_defeat", "truth_possible")},
            "best_win_fraction_by_round": [xc.mean_ci([r["timeline"][t]["best_win_fraction"] for r in rs])["mean"]
                                           for t in range(T)],
            "P(win | guaranteed at round 1)": cond(g1),
            "P(win | not guaranteed at round 1)": cond([not x for x in g1]),
            "P(win | guaranteed at round T)": cond([r["timeline"][-1]["guaranteed"] for r in rs]),
            "timeline_acceptance": sum(d["accepted"] / d["draws"] for r in rs for d in r["timeline_diag"])
            / sum(len(r["timeline_diag"]) for r in rs),
            "timeline_fallback_rounds": sum(d["fallback"] for r in rs for d in r["timeline_diag"]),
            "game_seconds_mean": sum(r["game_seconds"] for r in rs) / len(rs),
            "timeline_seconds_mean": sum(r["timeline_seconds"] for r in rs) / len(rs),
        }
    # ---- value of information: Δ vs k = 0 (per ₫ of query spend) and marginal Δ between consecutive k
    value: dict[str, Any] = {}
    for price in PRICES:
        for rule in RULES:
            key = f"p{int(Decimal(price))}_{rule}"
            base = arm_name(price, 0, rule)
            seq = []
            prev = base
            for k in KS[1:]:
                name = arm_name(price, k, rule)
                e = [by[name][s]["expected"] for s in seeds]
                e0 = [by[base][s]["expected"] for s in seeds]
                ep = [by[prev][s]["expected"] for s in seeds]
                vs0 = xc.paired(e, e0)
                seq.append({"k": k, "query_spend": k * float(Decimal(price)),
                            "delta_vs_k0": vs0, "delta_vs_k0_per_coin": vs0["mean"] / (k * float(Decimal(price))),
                            "win_delta_vs_k0": (sum(by[name][s]["victory"] for s in seeds)
                                                - sum(by[base][s]["victory"] for s in seeds)) / len(seeds),
                            "marginal_vs_prev_k": {"prev_k": by[prev][seeds[0]]["k"], **xc.paired(e, ep)},
                            "marginal_per_query": xc.paired(e, ep)["mean"] / (k - by[prev][seeds[0]]["k"])})
                prev = name
            means = {k: arms_out[arm_name(price, k, rule)]["expected"]["mean"] for k in KS}
            wins = {k: arms_out[arm_name(price, k, rule)]["win_rate"]["rate"] for k in KS}
            value[key] = {"curve": seq, "best_k_by_expected": max(means, key=means.get),
                          "best_k_by_win_rate": max(wins, key=lambda k: (wins[k], -k)),
                          "mean_expected_by_k": means, "win_rate_by_k": wins}
    return {"arms": arms_out, "value_of_information": value}


def tests(rows: list[dict], seeds: list[int]) -> dict:
    """Paired tests (Wilcoxon on E, exact McNemar on victory), Holm within each family."""
    by = _by(rows)

    def ev(name):
        return [by[name][s]["expected"] for s in seeds]

    def vic(name):
        return [by[name][s]["victory"] for s in seeds]

    fams: dict[str, list[dict]] = {"curve": [], "price_contrast": [], "gain_vs_random": []}
    for price in PRICES:
        for rule in RULES:
            for k in KS[1:]:
                for ref in (0, 2):
                    if k <= ref:
                        continue
                    a, b = arm_name(price, k, rule), arm_name(price, ref, rule)
                    fams["curve"].append({"a": a, "b": b, "E": xc.paired(ev(a), ev(b)), "victory": xc.mcnemar(vic(a), vic(b))})
    for rule in RULES:
        for k in (4, 8, 12):
            d10 = [x - y for x, y in zip(ev(arm_name("10.00", k, rule)), ev(arm_name("10.00", 2, rule)))]
            d5 = [x - y for x, y in zip(ev(arm_name("5.00", k, rule)), ev(arm_name("5.00", 2, rule)))]
            fams["price_contrast"].append({"a": f"(E_k{k} − E_k2)@₫10", "b": f"(E_k{k} − E_k2)@₫5", "rule": rule,
                                           "E": xc.paired(d10, d5)})
    for price in PRICES:
        for k in KS[1:]:
            a, b = arm_name(price, k, "gain"), arm_name(price, k, "random")
            fams["gain_vs_random"].append({"a": a, "b": b, "E": xc.paired(ev(a), ev(b)), "victory": xc.mcnemar(vic(a), vic(b)),
                                           "p_true_sigma": xc.paired([by[a][s]["posterior"]["p_true_sigma"] for s in seeds],
                                                                     [by[b][s]["posterior"]["p_true_sigma"] for s in seeds])})
    for fam, items in fams.items():
        for metric in ("E", "victory"):
            ps = [it[metric]["wilcoxon_p" if metric == "E" else "p"] for it in items if metric in it]
            if not ps:
                continue
            adj = xc.holm(ps)
            j = 0
            for it in items:
                if metric in it:
                    it[metric]["holm_p"] = adj[j]
                    j += 1
    return {"families": {f: {"size": len(v), "tests": v} for f, v in fams.items()},
            "holm_note": "Holm within each family and metric (E: Wilcoxon signed-rank; victory: exact McNemar)"}


# ============================================================================== main
def main() -> None:
    global KS, ARMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X5"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="keep rows already in raw/x5.jsonl and skip them")
    parser.add_argument("--out", default=str(xc.REPORTS / "x5.json"))
    parser.add_argument("--raw", default=str(RAW_PATH))
    parser.add_argument("--ks", default=",".join(map(str, KS)), help="smoke tests only")
    args = parser.parse_args()
    KS = tuple(int(k) for k in args.ks.split(","))
    ARMS = [(p, k, r) for p in PRICES for k in KS for r in RULES if not (k == 0 and r == "random")]
    raw_path = Path(args.raw)
    torch.set_num_threads(4)
    device, dev_record = pinned_device(args.device)
    free0 = xc.free_bytes(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    old = xc.read_rows(raw_path) if args.resume else []
    if not args.resume:
        raw_path.write_text("")
    done = {(r["seed"], r["arm"]) for r in old}
    rows: list[dict] = [r for r in old if r["seed"] in seeds]
    infos: list[dict] = []
    start = time.perf_counter()
    print(f"[x5] device {device} seeds {seeds[0]}..{seeds[-1]} arms {len(ARMS)} (resumed rows: {len(rows)})", flush=True)
    with xc.GpuMonitor(device) as monitor:
        for i, seed in enumerate(seeds):
            t0 = time.perf_counter()
            new, info = seed_rows(seed, args.n, device, ARMS, done)
            infos.append(info)
            for r in new:
                xc.write_row(raw_path, r)
            rows += new
            sub = {r["arm"]: r for r in rows if r["seed"] == seed}
            print(f"[x5] seed {seed} ({i + 1}/{len(seeds)}) {time.perf_counter() - t0:.0f}s, total "
                  f"{(time.perf_counter() - start) / 60:.1f} min; E ₫10 gain k=0/2/12: "
                  + "/".join(f"{sub[arm_name('10.00', k, 'gain')]['expected']:.3f}" for k in (0, 2, 12)), flush=True)
        summary = summarise(rows, seeds)
        test = tests(rows, seeds)
        for key, v in summary["value_of_information"].items():
            print(f"[x5] {key}: mean E by k {json.dumps({k: round(x, 4) for k, x in v['mean_expected_by_k'].items()})}; "
                  f"best k {v['best_k_by_expected']}", flush=True)
        t0 = time.perf_counter()
        par = parity(seeds[:args.parity_seeds], device, args.n)
        par["seconds"] = time.perf_counter() - t0
        print(f"[x5] parity passed: {par['passed']} ({par['seconds']:.0f}s)", flush=True)
    seconds = time.perf_counter() - start
    base = config_for(seeds[0], args.n, str(device), PRICES[0])
    report = {
        "experiment": "X5", "title": "Value of information at ₫10",
        "question": "How many survey queries are worth buying at ₫10 (vs ₫5), chosen by information gain or at random?",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget),
        "prices": {"worker": str(base.workers.worker_cost), "query": list(PRICES), "media_base": str(base.media.base_cost)},
        "ks": list(KS), "rules": list(RULES),
        "arms": {arm_name(p, k, r): {"price": p, "k": k, "rule": "none" if k == 0 else r, "num_queries": k,
                                     "survey_budget": str(Decimal(k) * Decimal(p))} for p, k, r in ARMS},
        "player": "BayesPlanner(num_queries=k, survey_budget=k×price, stop_entropy=None, min_gain_per_coin=None, "
                  "round_query_gain=None, device=cuda:2); rule 'random' = RandomQueryPlanner (uniform pair + region)",
        "metric_definitions": {
            "expected": "deterministic expected vote share E = mean_i P_i at the election (spec §5)",
            "victory": "official sampled vote > N/2",
            "posterior.*": "belief over σ at the start of round 1 (survey answers + round-1 free statistic + round-1 "
                           "poll): p_true_sigma = posterior mass on the true σ; map_correct = MAP σ equals the true σ in "
                           "all 4 regions; entropy_sigma in nats (prior log 360 = 5.886)",
            "static_share_anticipated": "expected share of the committed manifesto on the anticipated TRUE electorate "
                                        "(social dynamics only, no campaign) — manifesto quality",
            "oracle_static_share": "the same for the ORACLE manifesto (true electorate; reference, not a player)",
            "verdicts_by_round": "fraction of games where the property holds at decision point t (16 poll-consistent "
                                 "worlds of 1,600 voters, library plan set Π_t): guaranteed = ∃π∀w win, possible = ∃π∃w, "
                                 "knows_win_exists = ∀w∃π, inevitable_defeat = ∀π∀w ¬win, truth_possible = ∃π winning "
                                 "in the TRUE world",
            "delta_vs_k0": "paired mean E(k) − E(0) with t-CI and Wilcoxon p; per_coin divides by the query spend k×price",
            "marginal_vs_prev_k": "paired mean E(k) − E(previous k on the grid); marginal_per_query divides by the "
                                  "number of extra queries",
            "net loss": "a further question is a net loss when the marginal ΔE is negative (the money buys fewer "
                        "workers/media than the information is worth)"},
        **summary, "tests": test, "seed_info": infos,
        "deviations": [
            "Budget fix applied as planned: survey_budget = k × price and num_queries = k per arm (recorded in arms.*).",
            "No in-round queries (round_query_gain=None) in every arm, as in the v2 info-sufficiency design, so the "
            "survey is the only paid information.",
            "The ₫5 comparison uses both selection rules (11 arms), like ₫10.",
            "Games are played one at a time (sequential decisions); the GPU batches worlds × plans × voters inside "
            "each decision (8 planner worlds, ≤ 20+ candidate plans, 16 logic worlds per round), not games."],
        "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds,
                                       batch_sizes={"planner_num_worlds": 8, "planner_world_voters": 1600,
                                                    "timeline_worlds": TIMELINE_WORLDS,
                                                    "timeline_world_voters": TIMELINE_VOTERS,
                                                    "games_per_batch": 1},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"command": "python experiments/x5_info_value.py --seeds 50 --first-seed "
                                                         "13000 --device cuda:2 --out reports/v3/x5.json"}),
    }
    xc.save(args.out, report)
    print(f"[x5] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
