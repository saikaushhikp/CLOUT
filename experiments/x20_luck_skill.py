"""X20 — Luck vs skill: variance decomposition (plan ``docs/v2/plans/implementation_plan.md`` §7 X20).

Command (reproduces reports/v3/x20.json, reports/v3/raw/x20.jsonl, reports/v3/logs/x20.log and the figures):
  OMP_NUM_THREADS=4 python experiments/x20_luck_skill.py --seeds 50 --first-seed 14000 --salts 20 --device cuda:2 \
      --out reports/v3/x20.json 2>&1 | tee reports/v3/logs/x20.log

Design. 50 worlds (seed block 14000) × 3 fixed policies, all with the ORACLE manifesto (``oracle_manifesto`` on the
true world, as X0's ``oracle/*`` runs): ``planner`` (ORACLE planner), ``pay_split`` ("split"), ``none``.
N = 10⁴, T = 5, ₫500, ₫20 workers. One random source is re-drawn at a time with ``x_streams.SaltedStreams``
(salts 1..20; salt 0 = the canonical draw, shared by all streams, so every (world, stream) cell has 21 draws):

  cascade, media, workers   the in-game streams (``x_batch.run_batch(..., streams=...)``);
  election                  only the ballot draw (the campaign is unchanged, so the final v of the salt-0 game is
                            re-voted with ``run_election`` and the salted ``election`` generator, exactly what
                            ``run_batch`` would do);
  world                     all six generation streams (regions, block_probs, graph, preferences, attributes,
                            zealots) via ``generate_electorate(cfg, seeds=SaltedStreams(...))``: a fresh world; the
                            ORACLE manifesto is recomputed for it; in-game streams stay at salt 0;
  world_graph / world_prefs / world_attrs   (extra) only {block_probs, graph} / {preferences} / {attributes,
                            zealots} re-drawn (region sizes kept), to split the world component.

Metrics per game: E (expected share after the campaign; pre-election), V (realised vote share), win (V > 1/2).

Decomposition (per policy p and metric m; ddof = 1 within cells):
  σ²_s       = mean over worlds of the variance over the 21 draws of stream s (within-stream, "luck from s");
  σ²_total   = variance over all 50 × 21 world-stream draws (every one a fresh world);
  σ²_world   = σ²_total − Σ_{s ∈ cascade, media, workers, election} σ²_s (the world draw; additive model);
  ANOVA      one-way per stream (groups = worlds): σ²_between = (MSB − MSW)/21 — the between-world component
             with only s varying (it still contains the fixed draws of the other in-game streams);
  skill      Var_p(μ_p) (population variance of the 3 policy means over the world-stream draws) and
             skill share = Var_p(μ_p) / [Var_p(μ_p) + mean_p σ²_total(p)]; also for the pair planner vs split.
95% CIs by bootstrap over worlds (2,000 resamples).
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
import x_parity  # noqa: E402
import x_phase5a as p5  # noqa: E402
import x_policies  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_streams import SaltedStreams  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig, SeedStreams  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402
from dumbocracy.voting import run_election  # noqa: E402

POLICIES = ("planner", "pay_split", "none")
GAME_STREAMS = ("cascade", "media", "workers")
LUCK_STREAMS = GAME_STREAMS + ("election",)
WORLD_VARIANTS = {"world": ("regions", "block_probs", "graph", "preferences", "attributes", "zealots"),
                  "world_graph": ("block_probs", "graph"), "world_prefs": ("preferences",),
                  "world_attrs": ("attributes", "zealots")}
STREAMS = LUCK_STREAMS + tuple(WORLD_VARIANTS)
METRICS = ("E", "V", "win")
RAW_PATH = xc.RAW / "x20.jsonl"


def summary(o) -> dict:
    return {"E": float(o.result.expected_share), "V": float(o.result.vote_share), "win": bool(o.result.victory),
            "spent": float(o.spent), "workers": int(sum(sum(a.hires_by_region) for a in o.allocations))}


def play(es, ms, streams, policy, cfg, keep_final=False):
    return run_batch(es, cfg, ms, x_policies.GATE_POLICIES[policy], streams=streams, keep_final=keep_final)


# ============================================================================== decomposition
def components(cube: dict[str, np.ndarray], pol_means_world: np.ndarray | None = None) -> dict:
    """``cube[stream]`` = array [W, K] for one policy and metric. Returns the variance components."""
    out: dict = {}
    for s, a in cube.items():
        W, K = a.shape
        within = a.var(axis=1, ddof=1)
        msw = within.mean()
        msb = K * a.mean(axis=1).var(ddof=1)
        out[s] = {"within_var": float(msw), "within_sd": float(math.sqrt(max(msw, 0.0))),
                  "anova_between_var": float(max(0.0, (msb - msw) / K)), "anova_msb": float(msb)}
    total = float(cube["world"].reshape(-1).var(ddof=1))
    luck = sum(out[s]["within_var"] for s in LUCK_STREAMS)
    out["total_var"] = total
    out["world_var"] = total - luck
    out["sum_luck_streams_var"] = luck
    out["shares"] = {s: (out[s]["within_var"] / total if total > 0 else None) for s in LUCK_STREAMS}
    out["shares"]["world"] = out["world_var"] / total if total > 0 else None
    for s in WORLD_VARIANTS:
        if s != "world":
            out["shares"][s] = out[s]["within_var"] / total if total > 0 else None
    # additivity check: between-world variance seen with only `election` varying should equal
    # σ²_world + σ²_cascade + σ²_media + σ²_workers (one fixed draw of each in-game stream per world)
    pred = out["world_var"] + sum(out[s]["within_var"] for s in GAME_STREAMS)
    out["additivity_check"] = {"anova_between_election_design": out["election"]["anova_between_var"],
                               "world_plus_game_streams": pred}
    return out


def skill(world_draws: dict[str, np.ndarray], policies=POLICIES) -> dict:
    """``world_draws[p]`` = flat array of the world-stream draws of policy p."""
    means = np.array([world_draws[p].mean() for p in policies])
    within = float(np.mean([world_draws[p].var(ddof=1) for p in policies]))
    between = float(means.var(ddof=0))
    return {"policies": list(policies), "policy_means": {p: float(m) for p, m in zip(policies, means)},
            "var_policy_means": between, "mean_within_policy_var": within,
            "skill_share": between / (between + within) if between + within > 0 else None}


def bootstrap(cubes, n_boot: int, rng: np.random.Generator) -> dict:
    """``cubes[p][m][s]`` = [W, K]. CIs for the shares and the skill shares, resampling worlds."""
    W = next(iter(next(iter(cubes.values())).values()))["world"].shape[0]
    acc: dict = {}
    for _ in range(n_boot):
        idx = rng.integers(0, W, W)
        for m in METRICS:
            for p in POLICIES:
                comp = components({s: cubes[p][m][s][idx] for s in STREAMS})
                for s, v in comp["shares"].items():
                    if v is not None:
                        acc.setdefault((m, p, s), []).append(v)
                acc.setdefault((m, p, "world_var"), []).append(comp["world_var"])
            for name, pols in (("all3", POLICIES), ("planner_vs_split", ("planner", "pay_split"))):
                sk = skill({p: cubes[p][m]["world"][idx].reshape(-1) for p in pols}, pols)["skill_share"]
                if sk is not None:
                    acc.setdefault((m, "skill", name), []).append(sk)
    return {"|".join(k): [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


# ============================================================================== parity (§1.1 point 7)
def parity(seeds, device) -> dict:
    """CPU↔GPU on 3 seeds: ORACLE planner on the canonical (CPU-drawn) world, salt 0 and cascade salt 1."""
    def fn(seed, dev):
        cfg = GameConfig(seed=seed, device=str(dev), electorate=ElectorateConfig(num_voters=10_000))
        e = generate_electorate(cfg, device=dev)
        m = oracle_manifesto(e, cfg)[None]
        a = play([e], m, [SaltedStreams(seed)], "planner", cfg)[0]
        b = play([e], m, [SaltedStreams(seed, {"cascade": 1})], "planner", cfg)[0]
        return {"E_salt0": a.result.expected_share, "win_salt0": float(a.result.victory),
                "E_cascade1": b.result.expected_share, "V_cascade1": b.result.vote_share}
    out = x_parity.parity_rel(fn, seeds, device=device, abs_tol=0.01)
    # SaltedStreams with all salts 0 == SeedStreams (bit-identical generators)
    s = seeds[0]
    out["salted_zero_equals_seedstreams"] = all(
        SaltedStreams(s).derive(n, 0, "p0") == SeedStreams(s).derive(n, 0, "p0") for n in ("cascade", "media", "workers"))
    out["passed"] = bool(out["passed"] and out["salted_zero_equals_seedstreams"])
    return out


# ============================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X20"])
    ap.add_argument("--salts", type=int, default=20)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--cap-gib", type=float, default=10.0)
    ap.add_argument("--world-chunk", type=int, default=100)
    ap.add_argument("--parity-seeds", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=str(xc.REPORTS / "x20.json"))
    ap.add_argument("--raw", default=str(xc.RAW / "x20.jsonl"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    global RAW_PATH
    RAW_PATH = Path(args.raw)
    torch.set_num_threads(4)
    device, dev_record = p5.pin_device(args.device, need_bytes=int(args.cap_gib * 2**30))
    free0 = xc.free_bytes(device)
    torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    salts = list(range(1, args.salts + 1))
    K = args.salts + 1
    base = GameConfig(device=str(device), generation="device", electorate=ElectorateConfig(num_voters=args.n))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text("")
    record: dict = {"timings": {}}
    W = len(seeds)
    # data[p][m][s] = [W, K]; column 0 = canonical draw
    data = {p: {m: {s: np.full((W, K), np.nan) for s in STREAMS} for m in METRICS} for p in POLICIES}

    def put(p, w, s, k, row, extra=None):
        for m in METRICS:
            data[p][m][s][w, k] = float(row[m])
        xc.write_row(RAW_PATH, {"exp": "X20", "seed": seeds[w], "policy": p, "stream": s, "salt": k, **row, **(extra or {})})

    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        t0 = time.perf_counter()
        es = [generate_electorate(dataclasses.replace(base, seed=s), device=device) for s in seeds]
        ms = torch.stack([oracle_manifesto(e, dataclasses.replace(base, seed=s)) for e, s in zip(es, seeds)])
        record["timings"]["setup"] = time.perf_counter() - t0
        cal = p5.capped_batch(lambda B: play(es[:B], ms[:B], [SaltedStreams(s) for s in seeds[:B]], "planner", base),
                              device, sizes=(4, 8), max_batch=250, cap_bytes=int(args.cap_gib * 2**30))
        record["calibration"] = cal
        B = cal["batch"]
        print(f"[x20] device {device}; batch B = {B} (cap {args.cap_gib} GiB; {cal})", flush=True)

        # ---- canonical draw (salt 0) per policy; keeps v_T for the election re-draws
        t0 = time.perf_counter()
        idx = list(range(W))
        for p in POLICIES:
            outs = xc.run_chunks(lambda ch: [(i, o) for i, o in zip(ch, play(
                [es[i] for i in ch], ms[ch], [SaltedStreams(seeds[i]) for i in ch], p, base, keep_final=True))],
                idx, B, record=record)
            for i, o in outs:
                row = summary(o)
                for s in STREAMS:
                    for m in METRICS:
                        data[p][m][s][i, 0] = float(row[m])
                xc.write_row(RAW_PATH, {"exp": "X20", "seed": seeds[i], "policy": p, "stream": "canonical", "salt": 0, **row})
                # ---- election stream: re-vote the same final v with salted election generators
                for k in salts:
                    g = SaltedStreams(seeds[i], {"election": k}).generator("election", "p0", device=device)
                    r = run_election(o.final_preferences, ms[i], base.voting.sensitivity, base.voting.threshold, g)
                    put(p, i, "election", k, {"E": r.expected_share, "V": r.vote_share, "win": bool(r.victory),
                                              "spent": row["spent"], "workers": row["workers"]})
                o.final_preferences = None
        record["timings"]["canonical_and_election"] = time.perf_counter() - t0
        print(f"[x20] canonical + election done, {time.perf_counter() - start:.0f}s", flush=True)

        # ---- in-game streams: cascade, media, workers
        for s in GAME_STREAMS:
            t0 = time.perf_counter()
            items = [(i, k) for i in range(W) for k in salts]
            for p in POLICIES:
                outs = xc.run_chunks(lambda ch: list(zip(ch, play(
                    [es[i] for i, _ in ch], ms[[i for i, _ in ch]],
                    [SaltedStreams(seeds[i], {s: k}) for i, k in ch], p, base))), items, B, record=record)
                for (i, k), o in outs:
                    put(p, i, s, k, summary(o))
            record["timings"][s] = time.perf_counter() - t0
            print(f"[x20] stream {s}: {record['timings'][s]:.0f}s", flush=True)

        # ---- world streams (fresh electorates + ORACLE manifestos, in chunks)
        for s, gen_streams in WORLD_VARIANTS.items():
            t0 = time.perf_counter()
            items = [(i, k) for i in range(W) for k in salts]
            for c in range(0, len(items), args.world_chunk):
                chunk = items[c:c + args.world_chunk]
                wes, wms = [], []
                for i, k in chunk:
                    cfg = dataclasses.replace(base, seed=seeds[i])
                    e = generate_electorate(cfg, seeds=SaltedStreams(seeds[i], {g: k for g in gen_streams}), device=device)
                    wes.append(e)
                    wms.append(oracle_manifesto(e, cfg))
                wms_t = torch.stack(wms)
                sizes = [[int(x) for x in e.sizes.tolist()] for e in wes]
                for p in POLICIES:
                    sub = list(range(len(chunk)))
                    outs = xc.run_chunks(lambda ch: list(zip(ch, play(
                        [wes[j] for j in ch], wms_t[ch], [SaltedStreams(seeds[chunk[j][0]]) for j in ch], p, base))),
                        sub, B, record=record)
                    for j, o in outs:
                        i, k = chunk[j]
                        put(p, i, s, k, summary(o), {"sizes": sizes[j]})
                del wes, wms, wms_t
                torch.cuda.empty_cache()
            record["timings"][s] = time.perf_counter() - t0
            print(f"[x20] {s}: {record['timings'][s]:.0f}s", flush=True)
        games_seconds = time.perf_counter() - start
        par = parity(seeds[:args.parity_seeds], device)
        print(f"[x20] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    assert all(not np.isnan(a).any() for p in POLICIES for m in METRICS for a in data[p][m].values())

    # ---- analysis
    results = {m: {p: components({s: data[p][m][s] for s in STREAMS}) for p in POLICIES} for m in METRICS}
    skills = {m: {"all3": skill({p: data[p][m]["world"].reshape(-1) for p in POLICIES}),
                  "planner_vs_split": skill({p: data[p][m]["world"].reshape(-1) for p in ("planner", "pay_split")},
                                            ("planner", "pay_split"))} for m in METRICS}
    ci = bootstrap(data, args.bootstrap, np.random.default_rng(14000))
    ranking = {}
    for m in METRICS:
        ranking[m] = {}
        for p in POLICIES:
            sh = results[m][p]["shares"]
            order = sorted(["world", *LUCK_STREAMS], key=lambda s: -(sh[s] or 0.0))
            ranking[m][p] = order
    pred_order = ["world", "workers", "cascade", "media", "election"]
    elec_sd = {p: results["V"][p]["election"]["within_sd"] for p in POLICIES}
    # paired check of the prediction's strict pairs on V (planner): per-world within-variance comparisons
    pair_tests = []
    for m in ("V", "E"):
        for p in ("planner", "pay_split"):
            for a, b in (("workers", "cascade"), ("cascade", "media"), ("media", "election")):
                va, vb = data[p][m][a].var(axis=1, ddof=1), data[p][m][b].var(axis=1, ddof=1)
                pair_tests.append({"metric": m, "policy": p, "a": a, "b": b,
                                   **xc.paired(np.sqrt(va).tolist(), np.sqrt(vb).tolist())})
    for t, h in zip(pair_tests, xc.holm([t["wilcoxon_p"] for t in pair_tests])):
        t["holm_p"] = h
    canon = {p: {"win": xc.rate([bool(x) for x in data[p]["win"]["cascade"][:, 0]]),
                 "E": xc.mean_ci(data[p]["E"]["cascade"][:, 0].tolist())} for p in POLICIES}

    report = {
        "experiment": "X20", "title": "Luck vs skill: variance decomposition",
        "question": "How much of the outcome is decided by the player's decisions, and how much by randomness?",
        "seeds": {"first": seeds[0], "count": W, "list": seeds, "salts_per_stream": args.salts,
                  "draws_per_cell": K, "paired": True},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget),
        "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
        "policies": {"planner": "ORACLE manifesto + ORACLE planner (balance.policy_planner)",
                     "pay_split": "ORACLE manifesto + split (half media on the largest regions, half workers by size)",
                     "none": "ORACLE manifesto, no spending"},
        "streams": {**{s: [s] for s in LUCK_STREAMS}, **{k: list(v) for k, v in WORLD_VARIANTS.items()}},
        "metric_definitions": {
            "E": "expected vote share after the campaign (mean vote probability), before the ballot draw",
            "V": "realised vote share on election day", "win": "V > 1/2 (victory)",
            "within_var": "mean over worlds of the variance (ddof 1) over the 21 draws of that stream only",
            "total_var": "variance over the 50 × 21 world-stream draws (each a fresh world)",
            "world_var": "total_var − Σ within_var over cascade, media, workers, election (additive model)",
            "shares": "component / total_var (per policy)",
            "anova_between_var": "one-way random-effects ANOVA per stream, groups = worlds: (MSB − MSW)/21",
            "skill_share": "Var_p(policy means, ddof 0) / [that + mean_p total_var(p)], world-stream draws",
            "bootstrap_ci": "2.5/97.5 percentiles over 2,000 resamples of the 50 worlds; key 'metric|policy|component'"},
        "canonical": canon,
        "components": results, "skill": skills, "bootstrap_ci95": ci, "ranking_by_share": ranking,
        "predicted_order": pred_order, "election_noise_sd_V": elec_sd,
        "pair_tests": pair_tests,
        "test_family": "within-world sd of adjacent streams in the predicted order (workers>cascade, cascade>media, "
                       "media>election) for V and E × planner and pay_split: 12 Wilcoxon signed-rank tests on the "
                       "per-world sd, Holm",
        "deviations": ["Electorates drawn on the GPU (generation='device', allowed by plan §1.1 point 3) instead of the "
                       "canonical CPU draw, so the 4,050 worlds take seconds; parity uses the canonical CPU draw.",
                       "Extra (beyond the plan): the world component split into graph / preferences / attributes "
                       "sub-streams, and an E/V/win triple instead of E and win only."],
        "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=record.get("batch_sizes"), free_at_start=free0,
                                       monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"calibration": cal, "timings": record["timings"],
                                              "games_seconds": games_seconds,
                                              "oom_backoffs": record.get("oom_backoffs", []),
                                              "command": "python experiments/x20_luck_skill.py --seeds %d --first-seed %d "
                                                         "--salts %d --device %s --out %s" % (
                                                             W, seeds[0], args.salts, args.device, args.out)}),
    }
    xc.save(args.out, report)
    if not args.no_plots:
        plots(report)
        xc.save(args.out, report)
    for m in METRICS:
        print(f"[x20] {m}: shares planner {results[m]['planner']['shares']}; skill {skills[m]['all3']['skill_share']:.3f}",
              flush=True)
    c = report["provenance"]["compute"]
    print(f"[x20] wall {seconds:.0f}s, peak {c['peak_bytes'] / 2**30:.2f} GiB, util {c.get('gpu_util_mean')}", flush=True)


def plots(report: dict) -> None:
    figs = []
    comps = report["components"]
    cats = ["world", "workers", "cascade", "media", "election"]
    fig, axes = p5.new_figure(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, m in zip(axes, METRICS):
        p5.style(ax)
        for j, p in enumerate(POLICIES):
            vals = [math.sqrt(max(0.0, comps[m][p]["world_var"] if s == "world" else comps[m][p][s]["within_var"]))
                    for s in cats]
            xs = [i + (j - 1) * 0.27 for i in range(len(cats))]
            ax.bar(xs, vals, 0.25, color=p5.SERIES[j], label=p)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, fontsize=8)
        ax.set_title({"E": "expected share E", "V": "vote share V", "win": "win (0/1)"}[m], fontsize=9, color=p5.INK)
        ax.set_ylabel("sd of the component", color=p5.INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("X20: outcome sd attributable to each random source (50 worlds × 21 draws)", fontsize=10)
    figs.append(p5.save_figure(fig, "x20_components.png"))

    fig, ax = p5.new_figure(figsize=(7, 3.4))
    p5.style(ax)
    # stacked bars: luck components + skill on the pooled scale (all 3 policies)
    for m_i, m in enumerate(METRICS):
        sk = report["skill"][m]["all3"]
        tot = sk["var_policy_means"] + sk["mean_within_policy_var"]
        mean_comp = {s: sum((comps[m][p]["world_var"] if s == "world" else comps[m][p][s]["within_var"])
                            for p in POLICIES) / 3 for s in ["world", "workers", "cascade", "media", "election"]}
        left = 0.0
        parts = [("skill (policy)", sk["var_policy_means"])] + [(s, v) for s, v in mean_comp.items()]
        for j, (name, v) in enumerate(parts):
            w = max(0.0, v) / tot
            ax.barh(m_i, w, left=left, color=p5.SERIES[j], label=name if m_i == 0 else None, edgecolor=p5.SURFACE)
            if w > 0.06:
                ax.text(left + w / 2, m_i, f"{w:.2f}", ha="center", va="center", color="white", fontsize=7)
            left += w
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels(["E", "V", "win"])
    ax.set_xlabel("share of pooled variance (3 policies)", color=p5.INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("X20: skill vs luck shares", fontsize=10, color=p5.INK)
    figs.append(p5.save_figure(fig, "x20_shares.png"))
    report["figures"] = figs


if __name__ == "__main__":
    main()
