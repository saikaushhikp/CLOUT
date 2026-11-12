"""Stage 3: build and verify the WIN demo for one seed; writes every deliverable in Demo/win/.

Usage: python Demo/win/search/finalize.py --seed S --search <search.jsonl> --screening <screen.jsonl>
       [--why "..."] (devices: cuda:0, cuda:2, cpu; never GPU 1)

1. Survey questions: the BayesPlanner's information-gain choice (``BayesPlanner._maybe_query``, survey phase) on the
   live API observation (cuda:0), asked twice in sequence; answers recorded.
2. API replays (FastAPI TestClient on ``create_app(device=...)``) on cuda:0, cuda:2 and CPU of
   (a) the campaign script and (b) NO campaign (same questions and manifesto, an empty plan every round).
3. Best possible no-campaign manifesto: ``oracle_manifesto`` and a multi-start PG optimum (``common.
   best_no_campaign_manifesto``, 64 random + 13 structured starts) on the 5-round no-campaign forecast, plus the
   best whole-percent manifesto; both also played with no campaign through the API on the three devices.
4. Robustness on 10 alternative random streams (``SaltedStreams`` salts 1..10 on workers/cascade/media/election):
   the campaign and the no-campaign plays (``x_batch.run_batch``, cuda:0).
5. Comparison plays (cuda:0 and CPU): even 20% manifesto ± campaign, everything spent in round 5, ORACLE greedy planner.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F401,F403

import numpy as np  # noqa: E402
import x_common as xc  # noqa: E402
import x_policies  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_streams import SaltedStreams  # noqa: E402

import replay_win  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.bayes import expected_information_gain  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402
from dumbocracy.strategies.players import BayesPlanner  # noqa: E402

DEVICES = ("cuda:0", "cuda:2", "cpu")
EMPTY = replay_win.EMPTY_ROUND


def trajectory(rep):
    return [rep["expected_share_at_commit"]] + [r["expected_share_after_round"] for r in rep["rounds"]]


def brief(rep):
    return {"result": rep["result"], "expected_share_by_round": trajectory(rep), "manifesto_pct": rep["manifesto_pct"],
            "polls_after_round": [None if r["poll_after_round"] is None else r["poll_after_round"]["estimate"]
                                  for r in rep["rounds"]],
            "spent": f"{Decimal('850.00') - Decimal(rep['rounds'][-1]['treasury_left']):.2f}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--search", required=True)
    ap.add_argument("--screening", required=True)
    ap.add_argument("--why", default="")
    args = ap.parse_args()
    t_start = time.time()
    seed = args.seed
    best = [json.loads(line) for line in open(args.search) if json.loads(line)["seed"] == seed][-1]
    torch.set_num_threads(4)
    dev0, rec0 = xc.select_device("cuda:0", need_bytes=2 * 2**30)
    settings = {"num_voters": N, "seed": seed, "rounds": T, "budget": "850.00"}

    # ---- 1. survey questions (BayesPlanner information gain on the live observation, cuda:0)
    client = replay_win.make_client("cuda:0")
    g = client.post("/api/games", json=settings).json()
    gid, obs = g["game_id"], g["observation"]
    initial_obs = copy.deepcopy(obs)
    bp = BayesPlanner(player_id="p0", seed=0)
    queries, q_info = [], []
    for _ in range(2):
        post = bp._posterior(obs)
        ranked = expected_information_gain(post, bp.model, [r["voters"] for r in obs["regions"]], round_index=0,
                                           sample_fraction=bp._fraction())
        choice = bp._maybe_query(obs, post, survey=True)
        q = (choice["sector_a"], choice["sector_b"], choice["region"]) if choice else ranked[0][1]
        res = client.post(f"/api/games/{gid}/players/p0/query",
                          json=dict(zip(("sector_a", "sector_b", "region"), q))).json()
        obs = res["observation"]
        queries.append({"sector_a": q[0], "sector_b": q[1], "region": q[2]})
        q_info.append({"query": queries[-1], "expected_info_gain_nats": next(gv for gv, t in ranked if t == q),
                       "posterior_entropy_before": post.entropy(),
                       "picked_by": "BayesPlanner._maybe_query" if choice else "expected_information_gain ranked[0]",
                       "answer_pct_b_over_a": res["answer"]["pct_b_over_a"], "moe": res["answer"]["moe"]})
    post = bp._posterior(obs)
    posterior_after = {"posterior_entropy_after": post.entropy(),
                       "posterior_top3": [(float(p), list(a)) for p, a in post.top(3)]}

    # ---- 2. the world, the best no-campaign manifesto
    cfg = cfg_for(seed, dev0)
    e = generate_electorate(cfg, device=dev0)
    v_t = anticipate(e, cfg)
    b_or = oracle_manifesto(e, cfg)
    opct = int_percent(b_or.tolist())
    nc = best_no_campaign_manifesto(e, cfg, v_t=v_t, random_starts=64)
    nocamp_best = {"oracle_manifesto": [round(float(x), 6) for x in b_or.tolist()],
                   "oracle_manifesto_share": share_of(v_t, b_or, cfg), "oracle_manifesto_pct": opct,
                   "pg_multistart": nc,
                   "best_share_any_manifesto": max(nc["continuous_share"], share_of(v_t, b_or, cfg)),
                   "method": ("With zero spending the electorate on election day is the 5-round social-only forecast "
                              "v_T (oracle.anticipate), which does not depend on the manifesto; the best no-campaign "
                              "share is max_b E(v_T, b), found by oracle_manifesto and by multi-start projected "
                              "gradient (optimise_manifesto, 77 starts incl. 64 random) plus whole-percent hill climb.")}

    # ---- 3. scripts and API replays
    alloc = [{"workers": list(h), "media": [bool(x) for x in m]} for h, m in zip(best["hires"], best["media"])]
    script = {"settings": {**settings, "player_id": "p0"}, "queries": queries, "manifesto_pct": best["pct"],
              "allocations": alloc, "no_campaign_allocations": [EMPTY] * T}
    reps, nocs, best_nocs = {}, {}, {}
    for d in DEVICES:
        cl = replay_win.make_client(d)
        t = time.time()
        reps[d] = replay_win.play(script, d, client=cl)
        reps[d]["wall_seconds"] = time.time() - t
        nocs[d] = replay_win.play_no_campaign(script, d, client=cl)
        best_nocs[d] = {"oracle_manifesto_int": brief(replay_win.play_no_campaign(script, d, client=cl, manifesto_pct=opct)),
                        "pg_best_int": brief(replay_win.play_no_campaign(script, d, client=cl, manifesto_pct=nc["integer_pct"]))}
        print(d, "campaign", reps[d]["result"], "| no campaign", nocs[d]["result"], flush=True)
    r0, n0 = reps["cuda:0"]["result"], nocs["cuda:0"]["result"]
    same = (reps["cuda:2"]["result"] == r0 and nocs["cuda:2"]["result"] == n0
            and trajectory(reps["cuda:2"]) == trajectory(reps["cuda:0"]) and reps["cuda:2"]["answers"] == reps["cuda:0"]["answers"])
    camp_shares = [reps[d]["result"]["vote_share"] for d in DEVICES]
    noc_shares = [nocs[d]["result"]["vote_share"] for d in DEVICES]
    script["expected"] = {
        "cuda": {"campaign_votes": r0["votes"], "campaign_vote_share": r0["vote_share"],
                 "campaign_expected_share": r0["expected_share"], "no_campaign_votes": n0["votes"],
                 "no_campaign_vote_share": n0["vote_share"], "no_campaign_expected_share": n0["expected_share"]},
        "cpu": {"campaign_votes": reps["cpu"]["result"]["votes"], "no_campaign_votes": nocs["cpu"]["result"]["votes"]},
        "campaign_min_vote_share": round(min(camp_shares) - 0.01, 2),
        "no_campaign_max_vote_share": round(max(noc_shares) + 0.01, 2),
    }

    # ---- 4. robustness on 10 alternative streams (cuda:0)
    streams = [SaltedStreams(seed, {s: k for s in SALTED}) for k in range(1, 11)]

    def alt(pct, al):
        B = len(streams)
        man = torch.tensor(web_manifesto(pct), dtype=e.preferences.dtype, device=dev0).reshape(1, -1).expand(B, -1)
        outs = run_batch([e] * B, cfg, man, Plan([a["workers"] for a in al], [a["media"] for a in al]), streams=streams,
                         aggregation="exact")
        E = [o.result.expected_share for o in outs]
        V = [o.result.vote_share for o in outs]
        return {"expected_share": E, "vote_share": V, "wins": int(sum(o.result.victory for o in outs)), "n": B,
                "E_min": min(E), "E_mean": float(np.mean(E)), "E_max": max(E), "share_min": min(V), "share_max": max(V)}
    robust = {"streams": "SaltedStreams(seed, salt k on workers/cascade/media/election), k = 1..10 (never searched)",
              "campaign": alt(best["pct"], alloc), "no_campaign_same_manifesto": alt(best["pct"], [EMPTY] * T),
              "no_campaign_best_int_manifesto": alt(nc["integer_pct"], [EMPTY] * T)}
    print("robust campaign", robust["campaign"]["E_min"], robust["campaign"]["wins"], "| no-campaign wins",
          robust["no_campaign_same_manifesto"]["wins"], robust["no_campaign_best_int_manifesto"]["wins"], flush=True)

    # ---- 5. comparison plays (cuda:0, cpu)
    sizes = [r["voters"] for r in initial_obs["regions"]]
    oman = torch.tensor(web_manifesto(opct), dtype=e.preferences.dtype, device=dev0)
    planner_out = run_batch([e], cfg, oman.reshape(1, -1), x_policies.policy_planner, seeds=[seed], aggregation="exact")[0]
    planner_alloc = [{"workers": list(a.hires_by_region), "media": list(a.media_regions)} for a in planner_out.allocations]
    k5 = int((PLAY_BUDGET - media_cost(sizes, ALL, cfg.media.base_cost)) // cfg.workers.worker_cost)
    r_big = max(range(4), key=lambda r: sizes[r])
    late = [EMPTY] * 4 + [{"workers": [k5 if i == r_big else 0 for i in range(4)], "media": [True] * 4}]
    comparisons = {
        "even_20pct_no_campaign": ([20] * 5, [EMPTY] * T),
        "even_20pct_same_campaign": ([20] * 5, alloc),
        "demo_manifesto_all_spent_in_round5": (best["pct"], late),
        "oracle_manifesto_oracle_greedy_planner": (opct, planner_alloc),
    }
    comp = {}
    for name, (pct, al) in comparisons.items():
        comp[name] = {"manifesto_pct": pct, "allocations": al}
        for d in ("cuda:0", "cpu"):
            comp[name][d] = brief(replay_win.play(script, d, manifesto_pct=pct, allocations=al))
        print(name, comp[name]["cuda:0"]["result"], flush=True)

    # ---- 6. deliverables
    def rounds_table(rep):
        return [{k: r[k] for k in ("round", "workers", "media", "cost", "treasury_left", "expected_share_after_round",
                                   "poll_after_round", "converted", "team_by_region")} for r in rep["rounds"]]

    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()

    def api_block(rep):
        return {"engine_device": rep["device"], "result": rep["result"], "answers": rep["answers"],
                "manifesto_pct": rep["manifesto_pct"], "expected_share_at_commit": rep["expected_share_at_commit"],
                "poll_round1_start": rep["poll_round1_start"], "rounds": rounds_table(rep),
                "regions_debrief": rep["debrief"]["regions"], "oracle_debrief": rep["debrief"]["oracle"]}
    ver = {
        "seed": seed, "settings": settings, "script": "Demo/win/script.json",
        "headline": (f"Without campaigning: {n0['vote_share']:.1%} (defeat). With this campaign: {r0['vote_share']:.1%} "
                     f"(victory). CUDA engine; CPU: {nocs['cpu']['result']['vote_share']:.1%} vs "
                     f"{reps['cpu']['result']['vote_share']:.1%}."),
        "campaign_api": {d: {**api_block(rep), "wall_seconds": rep["wall_seconds"]} for d, rep in reps.items()},
        "no_campaign_api": {d: api_block(rep) for d, rep in nocs.items()},
        "cuda0_equals_cuda2": same,
        "best_no_campaign_manifesto": {**nocamp_best, "api_no_campaign_plays": best_nocs,
                                       "strong_condition_holds": bool(nocamp_best["best_share_any_manifesto"] < 0.5 and all(
                                           not v[k]["result"]["victory"] for v in best_nocs.values() for k in v)),
                                       "margin_below_half": 0.5 - nocamp_best["best_share_any_manifesto"]},
        "survey_question_choice": {"method": "BayesPlanner(seed=0)._maybe_query on the live observation (information "
                                             "gain I(sigma;Y), survey phase), cuda:0 engine", "questions": q_info,
                                   **posterior_after},
        "search": {k: best[k] for k in ("search_score", "start", "E_canonical", "E_fresh_mean", "E_fresh_min",
                                        "screen_E_best", "cost", "seconds", "world_evals")},
        "robustness_alt_streams": robust,
        "comparison_plays": comp,
        "provenance": {"git_rev": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                       "git_status_outside_Demo": [x for x in git("status", "--porcelain").splitlines() if "Demo/" not in x],
                       "devices": list(DEVICES), "device_selection_cuda0": rec0,
                       "gpu_name": torch.cuda.get_device_name(dev0), "torch": torch.__version__,
                       "wall_seconds_finalize": time.time() - t_start,
                       "runner": "fastapi.testclient.TestClient(create_app(device=...)), the routes the web game calls",
                       "scripts": "Demo/win/search/{screen,search,finalize,summary}.py"},
    }
    (WIN / "verification.json").write_text(json.dumps(ver, indent=1))
    (WIN / "script.json").write_text(json.dumps(script, indent=1))
    (WIN / "initial_observation.json").write_text(json.dumps(initial_obs, indent=1))
    (WIN / "game_settings.json").write_text(json.dumps({**settings, "player_id": "p0", "why_this_seed": args.why}, indent=1))
    plot(ver)
    print("written", round(time.time() - t_start), "s", flush=True)


def plot(ver) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = ver["campaign_api"]["cuda:0"]
    n = ver["no_campaign_api"]["cuda:0"]
    tr = lambda r: [r["expected_share_at_commit"]] + [x["expected_share_after_round"] for x in r["rounds"]]  # noqa: E731
    bn = ver["best_no_campaign_manifesto"]["api_no_campaign_plays"]["cuda:0"]["pg_best_int"]
    cp = ver["comparison_plays"]
    series = [("With this campaign", tr(c), "#2a78d6", 2.8, "-"),
              ("ORACLE greedy planner + ORACLE manifesto", cp["oracle_manifesto_oracle_greedy_planner"]["cuda:0"]["expected_share_by_round"], "#1baf7a", 1.6, "-"),
              ("Same manifesto, everything spent in round 5", cp["demo_manifesto_all_spent_in_round5"]["cuda:0"]["expected_share_by_round"], "#eda100", 1.6, "-"),
              ("No campaign, best possible manifesto", bn["expected_share_by_round"], "#6250d6", 1.6, "--"),
              ("No campaign (same questions & manifesto)", tr(n), "#eb6834", 2.8, "-")]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    x = list(range(6))
    for name, ys, col, lw, ls in series:
        ax.plot(x, ys, color=col, lw=lw, ls=ls, marker="o", ms=4.5, label=f"{name}: {ys[-1]:.3f}")
    ax.axhline(0.5, color="#888", lw=1, ls=":")
    ax.text(0.05, 0.505, "50% = win line", color="#555", fontsize=8)
    ax.set_xticks(x, ["start\n(commit)", "after R1", "after R2", "after R3", "after R4", "after R5\n(election)"])
    ax.set_ylabel("Expected vote share (true, from the debrief)")
    lo = min(min(s[1]) for s in series)
    ax.set_ylim(max(0.0, lo - 0.04), max(max(s[1]) for s in series) + 0.05)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.set_title(f"Seed {ver['seed']} · 100,000 voters · ₫850 · 5 rounds: campaign vs no campaign (CUDA engine)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(WIN / "share_over_rounds.png")


if __name__ == "__main__":
    main()
