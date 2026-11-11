"""X17 — Learning personalities by probing (plan ``docs/v2/plans/implementation_plan.md`` §6 X17).

Command (reproduces reports/v3/x17.json, reports/v3/raw/x17.jsonl and reports/v3/logs/x17.log):
  OMP_NUM_THREADS=4 python experiments/x17_probe.py --seeds 50 --first-seed 13000 --device cuda:2 \
      --out reports/v3/x17.json 2>&1 | tee reports/v3/logs/x17.log

Surveys say nothing about the personalities τ (R13) and polls are national, so τ can only be learnt from how the
national poll responds to the player's own campaign. Players (observation-only; all ``BayesPlanner`` defaults:
survey by information gain, ₫50 survey cap, rollout planner with 8 worlds of 1,600 voters):

* ``bayes``        plain ``BayesPlanner`` (its planning worlds draw τ uniformly and ignore the polls).
* ``poll_planner`` :class:`PollPlanner`: identical, except that its planning worlds are poll-consistent
                   (``logic.worlds.consistent_worlds``: σ from the posterior, τ uniform, kept iff every poll so
                   far is matched within 2.5 SE + 0.01). Control arm: separates "using the polls" from "probing".
* ``probe``        :class:`ProbePlayer` = PollPlanner, except that in round 1 it spends on ONE region only — either
                   one airing in r, or the even round cap (treasury / 5) on workers in r — choosing among these 8
                   probes the one that maximises the expected split of the τ hypotheses: in 96 posterior worlds with
                   uniform τ, the between-group variance of the predicted round-2 poll (groups = personality of the
                   probed region) divided by (poll SE² + within-group variance). It then reads the round-2 poll
                   (through its poll-consistent planning worlds) and plays the planner.

Measured per game (paired seeds, block 13000, N = 10⁴, T = 5, ₫500, ₫20 workers, ₫10 queries):
* **τ learning** at each decision point t = 1..5: D = 512 candidate worlds (σ from the player's posterior, τ uniform
  over the 24 permutations, fresh micro-state; ``logic.worlds`` keys) are replayed ONCE through the player's whole
  history on the GPU, recording the expected share after every resolved round; the accepted set at t is the worlds
  that match every poll s ≤ t (the ``consistent_worlds`` rule). From the accepted τ values: entropy over the 24
  permutations (plug-in and Miller–Madow), mean per-region marginal entropy, the marginal entropy of the probed
  region, and the mass on the true τ / true τ_r. The same D candidates (same keys) are used for every arm and
  round (common random numbers). Checked against the library ``consistent_worlds`` (identical accepted set).
* **K_p(Guaranteed)** per round: ``logic.report.timeline_with_diagnostics`` (16 worlds of 1,600 voters).
* the final win rate and expected share E.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x_parity import parity_rel  # noqa: E402
from x5_info_value import pinned_device  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig, SeedStreams  # noqa: E402
from dumbocracy.dynamics.campaign import Allocation, media_cost  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.logic.worlds import PERMUTATIONS, consistent_worlds, poll_standard_error  # noqa: E402
from dumbocracy.sectors import NUM_REGIONS  # noqa: E402
from dumbocracy.strategies.planner import plan_round, round_cap, rounds_left, workers_for  # noqa: E402
from dumbocracy.strategies.players import BayesPlanner, _treasury, public_config  # noqa: E402
from dumbocracy.strategies.worlds import WorldBatch, public_sizes, sample_worlds  # noqa: E402

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

ARMS = ("bayes", "poll_planner", "probe")
RAW_PATH = xc.RAW / "x17.jsonl"
TAU_DRAWS = 512
POLL_Z, SLACK = 2.5, 0.01
TIMELINE_WORLDS = 16
LN24 = math.log(24)


# ============================================================================== players
@dataclass
class PollPlanner(BayesPlanner):
    """BayesPlanner whose planning worlds are poll-consistent (``logic.worlds.consistent_worlds``)."""

    poll_z: float = POLL_Z

    def _consistent(self, observation: dict[str, Any], posterior, count: int, tag: str):
        cfg = public_config(observation, self.config)
        seed = self._streams(observation).derive("strategy", "consistent", tag, observation["round"])
        n = min(self.world_voters, int(observation["num_voters"]))
        ws = consistent_worlds(observation, cfg, count, world_voters=n, seed=seed, poll_z=self.poll_z,
                               device=self.device, posterior=posterior)
        return cfg, ws

    def allocation(self, observation: dict[str, Any], posterior=None):
        if not self.planning:
            return super().allocation(observation, posterior)
        posterior = posterior or self._posterior(observation)
        cfg, ws = self._consistent(observation, posterior, self.num_worlds, "plan")
        seed = self._streams(observation).derive("strategy", "rollout", observation["round"])
        a, diag = plan_round(observation, ws.batch, observation["manifesto"], cfg, rollout=self.rollout, seed=seed,
                             current=ws.current, current_workers=ws.workers)
        diag["acceptance"] = ws.acceptance_rate
        diag["fallback"] = ws.fallback
        return a, diag


def probe_candidates(observation: dict[str, Any], cfg: GameConfig) -> list[tuple[str, int, Allocation]]:
    """One airing in r (if affordable) and the even round cap on workers in r, for each region r."""
    sizes = public_sizes(observation)
    treasury = _treasury(observation)
    cap = round_cap(treasury, rounds_left(observation))
    out = []
    for r in range(NUM_REGIONS):
        one = tuple(k == r for k in range(NUM_REGIONS))
        if media_cost(sizes, one, cfg.media.base_cost) <= treasury:
            out.append(("media", r, Allocation((0, 0, 0, 0), one)))  # type: ignore[arg-type]
        hires = workers_for(cap, [1.0 if k == r else 0.0 for k in range(NUM_REGIONS)], cfg)
        if sum(hires) > 0:
            out.append(("workers", r, Allocation(hires, (False,) * NUM_REGIONS)))  # type: ignore[arg-type]
    return out


@dataclass
class ProbePlayer(PollPlanner):
    """Round 1: the single-region probe that best splits the τ hypotheses; later rounds: PollPlanner."""

    probe_worlds: int = 96
    probe_record: dict | None = None

    def choose_probe(self, observation: dict[str, Any], posterior) -> tuple[Allocation, dict]:
        cfg = public_config(observation, self.config)
        streams = self._streams(observation)
        sigma = posterior.sample(self.probe_worlds, streams.generator("strategy", "posterior", "probe", 1))
        tau = PERMUTATIONS[torch.randint(len(PERMUTATIONS), (self.probe_worlds,),
                                         generator=streams.generator("strategy", "probe-tau", 1))]
        n = min(self.world_voters, int(observation["num_voters"]))
        worlds = sample_worlds(observation, sigma, cfg, num_voters=n, seed=streams.derive("strategy", "probe-worlds", 1),
                               device=self.device, personalities=tau)
        batch = WorldBatch(worlds, cfg)
        current, workers = batch.replay_state(observation, streams.derive("strategy", "probe-replay", 1))
        cands = probe_candidates(observation, cfg)
        allocs = [a for *_, a in cands]
        C = len(allocs)
        b = batch.manifesto_tensor(observation["manifesto"])
        airings = torch.zeros((C, NUM_REGIONS), dtype=torch.int64, device=batch.device)
        gen = streams.generator("strategy", "probe-step", 1, device=batch.device)
        v = current.unsqueeze(0).expand(C, -1, -1).contiguous()
        v, _ = batch.step(v, allocs, airings, b, gen, workers)
        shares = batch.shares(v, b).to(torch.float64).cpu()                     # [C, D] predicted round-2 poll
        poll1 = max(observation["polls"], key=lambda p: int(p["round"]))
        se = poll_standard_error(poll1["estimate"], int(poll1.get("sample_size") or 400))
        scores = []
        for c, (kind, r, _a) in enumerate(cands):
            groups = [shares[c][tau[:, r] == p] for p in range(NUM_REGIONS)]
            groups = [g for g in groups if g.numel() > 1]
            means = torch.stack([g.mean() for g in groups])
            between = float(means.var(unbiased=False))
            within = float(sum(g.var() * g.numel() for g in groups) / sum(g.numel() for g in groups))
            scores.append(between / (se * se + within))
        best = max(range(C), key=lambda c: scores[c])
        kind, region, alloc = cands[best]
        record = {"kind": kind, "region": region, "hires": list(alloc.hires_by_region),
                  "media": list(alloc.media_regions), "score": scores[best],
                  "candidates": [{"kind": k, "region": r, "score": s, "pred_share_mean": float(shares[c].mean())}
                                 for c, ((k, r, _a), s) in enumerate(zip(cands, scores))],
                  "poll_se": se, "worlds": self.probe_worlds}
        return alloc, record

    def allocation(self, observation: dict[str, Any], posterior=None):
        if int(observation["round"]) != 1:
            return super().allocation(observation, posterior)
        start = time.perf_counter()
        posterior = posterior or self._posterior(observation)
        alloc, record = self.choose_probe(observation, posterior)
        record["seconds"] = time.perf_counter() - start
        self.probe_record = record
        return alloc, {"best_score": record["score"], "num_candidates": len(record["candidates"]),
                       "seconds": record["seconds"]}


def make_player(name: str, seed: int, device: str) -> BayesPlanner:
    if name == "bayes":
        return BayesPlanner(seed=seed, device=device)
    if name == "poll_planner":
        return PollPlanner(seed=seed, device=device)
    if name == "probe":
        return ProbePlayer(seed=seed, device=device)
    raise ValueError(name)


# ============================================================================== τ learning
def _entropy(counts: torch.Tensor) -> float:
    n = float(counts.sum())
    if n <= 0:
        return float("nan")
    p = counts[counts > 0].to(torch.float64) / n
    return float(-(p * p.log()).sum())


def tau_learning(obs: dict, cfg: GameConfig, true_tau: list[int], probe_region: int, *, draws: int, seed: int,
                 device: torch.device, world_voters: int = 1600) -> dict:
    """Accepted-τ statistics at every decision point t = 1..T from ONE replay of D candidate worlds (module
    docstring). Keys as ``consistent_worlds`` (stream ``worlds``, chunk 0), so the accepted set at t equals
    ``consistent_worlds(observation_at(obs, t), count=D, max_draws=D, seed=seed)``."""
    from dumbocracy.logic.knowledge import observation_at
    from dumbocracy.strategies.bayes import ASSIGNMENTS, posterior_from_observation
    T = int(obs["rounds"])
    obs_T = observation_at(obs, T, stage="decision")
    posterior = posterior_from_observation(obs_T, sample_fraction=cfg.survey.query_sample_fraction)
    cdf = torch.cumsum(posterior.probs, 0)
    streams = SeedStreams(int(seed))
    u = torch.rand(draws, generator=streams.generator("worlds", "logic-sigma", 0), dtype=torch.float64)
    sigma = ASSIGNMENTS[torch.searchsorted(cdf, u * cdf[-1]).clamp_max(cdf.numel() - 1)]
    tau = PERMUTATIONS[torch.randint(len(PERMUTATIONS), (draws,), generator=streams.generator("worlds", "logic-tau", 0))]
    n = min(int(world_voters), int(obs["num_voters"]))
    worlds = sample_worlds(obs_T, sigma, cfg, num_voters=n, seed=streams.derive("worlds", "logic-micro", 0),
                           device=device, personalities=tau)
    batch = WorldBatch(worlds, cfg)
    _v, _w, shares = batch.replay_state(obs_T, streams.derive("worlds", "logic-replay", 0), record_shares=True)
    shares = shares.cpu()                                                     # [D, T+1]
    polls = sorted((p for p in obs["polls"] if int(p["round"]) <= T), key=lambda p: int(p["round"]))   # decision points
    true = torch.tensor(true_tau)
    is_true = (tau == true).all(1)
    out = []
    z_run = torch.zeros(draws, dtype=torch.float64)
    loglik = torch.zeros(draws, dtype=torch.float64)
    perm_all = (tau.unsqueeze(1) == PERMUTATIONS.unsqueeze(0)).all(-1).to(torch.int64).argmax(1)       # [D]
    for p in polls:
        s = int(p["round"])
        se = poll_standard_error(p["estimate"], int(p.get("sample_size") or 400))
        scale = se + SLACK
        z_run = torch.maximum(z_run, (shares[:, s - 1] - float(p["estimate"])).abs() / scale)
        loglik = loglik - 0.5 * ((shares[:, s - 1] - float(p["estimate"])) / se) ** 2
        w = torch.softmax(loglik, 0)                       # soft (Gaussian poll likelihood, no slack) posterior
        soft = torch.zeros(24, dtype=torch.float64).index_add_(0, perm_all, w)
        soft_marg = [torch.zeros(4, dtype=torch.float64).index_add_(0, tau[:, r], w) for r in range(NUM_REGIONS)]
        soft_stats = {"H_soft": float(-(soft[soft > 0] * soft[soft > 0].log()).sum()),
                      "H_soft_probe_region": float(-(soft_marg[probe_region][soft_marg[probe_region] > 0]
                                                     * soft_marg[probe_region][soft_marg[probe_region] > 0].log()).sum()),
                      "p_true_tau_soft": float(w[is_true].sum()),
                      "p_true_tau_probe_region_soft": float(soft_marg[probe_region][true_tau[probe_region]]),
                      "ess_soft": float(1.0 / (w * w).sum())}
        ok = z_run <= POLL_Z
        k = int(ok.sum())
        acc_tau = tau[ok]
        perm_index = (acc_tau.unsqueeze(1) == PERMUTATIONS.unsqueeze(0)).all(-1).to(torch.int64).argmax(1)
        counts = torch.bincount(perm_index, minlength=24)
        h = _entropy(counts)
        m = int((counts > 0).sum())
        marg = [_entropy(torch.bincount(acc_tau[:, r], minlength=4)) for r in range(NUM_REGIONS)]
        p_true_r = [float((acc_tau[:, r] == true_tau[r]).to(torch.float64).mean()) if k else float("nan")
                    for r in range(NUM_REGIONS)]
        out.append({"round": s, "accepted": k, "acceptance": k / draws,
                    "H_perm": h, "H_perm_mm": h + (m - 1) / (2 * k) if k else float("nan"),
                    "H_marginal_mean": sum(marg) / NUM_REGIONS if k else float("nan"),
                    "H_probe_region": marg[probe_region] if k else float("nan"),
                    "p_true_tau": float(is_true[ok].to(torch.float64).mean()) if k else float("nan"),
                    "p_true_tau_r_mean": sum(p_true_r) / NUM_REGIONS if k else float("nan"),
                    "p_true_tau_probe_region": p_true_r[probe_region], **soft_stats,
                    "_ok": ok})
    return {"rounds": out, "draws": draws, "sigma_entropy": posterior.entropy(), "_shares": shares,
            "_poll_rounds": [int(p["round"]) for p in polls]}


def check_against_library(obs: dict, cfg: GameConfig, true_tau: list[int], seed: int, device: torch.device,
                          draws: int) -> dict:
    """The accepted set of :func:`tau_learning` at t = T equals the library ``consistent_worlds``' at t = T."""
    from dumbocracy.logic.knowledge import observation_at
    T = int(obs["rounds"])
    tl = tau_learning(obs, cfg, true_tau, 0, draws=draws, seed=seed, device=device)
    mine = tl["rounds"][-1]["_ok"]
    mine_polls = tl["_shares"][:, [r - 1 for r in tl["_poll_rounds"]]]
    ws = consistent_worlds(observation_at(obs, T, stage="decision"), cfg, draws, world_voters=1600, seed=seed,
                           max_draws=draws, device=device)
    lib = int(ws.accepted)
    # the library returns the accepted worlds (all D when everything is accepted) in draw order
    same_worlds = len(ws) == int(mine.sum())
    diff = float((ws.poll_shares - mine_polls[mine]).abs().max()) if same_worlds and len(ws) else None
    return {"draws": draws, "accepted_mine": int(mine.sum()), "accepted_library": lib,
            "identical_count": int(mine.sum()) == lib, "max_abs_diff_poll_shares": diff,
            "identical_tau": bool(same_worlds and torch.equal(ws.tau.cpu(), tl_tau_accepted(obs, cfg, seed, draws, mine)))}


def tl_tau_accepted(obs: dict, cfg: GameConfig, seed: int, draws: int, ok: torch.Tensor) -> torch.Tensor:
    streams = SeedStreams(int(seed))
    tau = PERMUTATIONS[torch.randint(len(PERMUTATIONS), (draws,), generator=streams.generator("worlds", "logic-tau", 0))]
    return tau[ok]


# ============================================================================== one seed
def seed_rows(seed: int, n: int, device: torch.device, arms, done: set, *, draws: int = TAU_DRAWS,
              timeline: bool = True) -> tuple[list[dict], dict]:
    from dumbocracy.game import GameEngine
    from dumbocracy.game.observation import observation_for
    from dumbocracy.logic.report import timeline_with_diagnostics
    from dumbocracy.strategies.players import play_game
    cfg = GameConfig(seed=seed, device=str(device), electorate=ElectorateConfig(num_voters=n))
    electorate = generate_electorate(cfg, device=device)
    true_tau = [int(x) for x in electorate.personality_of_region.tolist()]
    state0 = GameEngine.new_game(cfg, electorate=electorate, game_id=f"x17_{seed}")
    info = {"seed": seed, "tau": true_tau, "sigma": [int(x) for x in electorate.archetype_of_region.tolist()],
            "sizes": [int(x) for x in electorate.sizes.tolist()]}
    games = {}
    for name in arms:
        player = make_player(name, seed, str(device))
        g0 = time.perf_counter()
        final, log = play_game(state0, player)
        games[name] = (player, final, time.perf_counter() - g0, log)
    probe = games.get("probe", (None,))[0]
    probe_region = probe.probe_record["region"] if probe is not None and probe.probe_record else 0
    info["probe_region"] = probe_region
    pub = public_config(observation_for(state0, "p0"), cfg)
    rows = []
    for name in arms:
        if (seed, name) in done:
            continue
        player, final, game_s, log = games[name]
        obs = observation_for(final, "p0")
        res = obs["result"]
        t0 = time.perf_counter()
        tl_tau = tau_learning(obs, pub, true_tau, probe_region, draws=draws, seed=seed * 7919 + 17, device=device)
        tau_s = time.perf_counter() - t0
        row: dict[str, Any] = {
            "exp": "X17", "seed": seed, "arm": name, "expected": float(res["expected_share"]),
            "vote_share": float(res["vote_share"]), "victory": bool(res["victory"]),
            "probe_region": probe_region, "true_tau": true_tau,
            "probe": getattr(player, "probe_record", None),
            "allocations": [{"round": int(a["round"]), "hires": [int(h) for h in a["hires_by_region"]],
                             "media": [bool(m) for m in a["media_regions"]], "cost": str(a["cost"])}
                            for a in sorted(obs["allocations"], key=lambda a: a["round"])],
            "num_queries": len(obs["queries"]), "polls": [round(float(p["estimate"]), 4) for p in obs["polls"]],
            "planner": [{k: d.get(k) for k in ("round", "best_score", "num_candidates", "seconds", "entropy")}
                        for d in player.diagnostics],
            "tau_learning": [{k: v for k, v in r.items() if k != "_ok"} for r in tl_tau["rounds"]],
            "tau_seconds": tau_s, "game_seconds": game_s, "rejected": sum(1 for *_, ok, _c in log if not ok)}
        if timeline:
            t0 = time.perf_counter()
            tl, diag = timeline_with_diagnostics(final, "p0", worlds=TIMELINE_WORLDS, world_voters=1600, seed=0,
                                                 device=device)
            row["timeline"] = [{"round": r["round"], "guaranteed": r["guaranteed_victory"], "possible": r["possible_victory"],
                                "knows_win_exists": r["knows_win_exists"], "inevitable_defeat": r["inevitable_defeat"],
                                "truth_possible": r["truth_victory_possible"],
                                "best_win_fraction": r["best_plan"]["win_fraction"]} for r in tl["rounds"]]
            row["timeline_diag"] = [{k: d[k] for k in ("round", "draws", "accepted", "fallback")} for d in diag["rounds"]]
            row["timeline_seconds"] = time.perf_counter() - t0
        rows.append(row)
    return rows, info


# ============================================================================== parity
def parity(seeds: list[int], device: torch.device, n: int) -> dict:
    """CPU↔GPU on 3 seeds for the probe arm (engine, planner worlds, probe worlds, τ replay, logic on each device)."""
    cache: dict = {}

    def fn(seed: int, dev: torch.device) -> dict:
        key = (seed, str(dev))
        if key not in cache:
            rows, _ = seed_rows(seed, n, dev, ("probe",), set(), draws=256, timeline=False)
            r = rows[0]
            last = r["tau_learning"][-1]
            cache[key] = {"expected": r["expected"], "victory": float(r["victory"]),
                          "probe_region": float(r["probe_region"]), "H_perm_T": last["H_perm"],
                          "acceptance_T": last["acceptance"]}
        return cache[key]

    return parity_rel(fn, seeds, device=device, abs_tol=0.02)


# ============================================================================== statistics
def summarise(rows: list[dict], seeds: list[int]) -> dict:
    by: dict[str, dict[int, dict]] = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    T = len(rows[0]["tau_learning"])
    metrics = ("H_perm", "H_perm_mm", "H_marginal_mean", "H_probe_region", "p_true_tau", "p_true_tau_r_mean",
               "p_true_tau_probe_region", "acceptance", "H_soft", "H_soft_probe_region", "p_true_tau_soft",
               "p_true_tau_probe_region_soft", "ess_soft")
    arms: dict[str, Any] = {}
    for name in ARMS:
        rs = [by[name][s] for s in seeds]
        arms[name] = {
            "n": len(rs), "win_rate": xc.rate([r["victory"] for r in rs]),
            "expected": xc.mean_ci([r["expected"] for r in rs]),
            "tau_by_round": {m: [xc.mean_ci([r["tau_learning"][t][m] for r in rs]) for t in range(T)] for m in metrics},
            "guaranteed_by_round": [xc.rate([r["timeline"][t]["guaranteed"] for r in rs]) for t in range(T)],
            "possible_by_round": [xc.rate([r["timeline"][t]["possible"] for r in rs]) for t in range(T)],
            "knows_win_exists_by_round": [xc.rate([r["timeline"][t]["knows_win_exists"] for r in rs]) for t in range(T)],
            "truth_possible_by_round": [xc.rate([r["timeline"][t]["truth_possible"] for r in rs]) for t in range(T)],
            "round1_spend": xc.mean_ci([float(Decimal(r["allocations"][0]["cost"])) for r in rs]),
            "num_queries": xc.mean_ci([r["num_queries"] for r in rs]),
            "game_seconds_mean": sum(r["game_seconds"] for r in rs) / len(rs),
        }
    pr = [by["probe"][s]["probe"] for s in seeds]
    probes = {"kind_counts": {k: sum(1 for p in pr if p["kind"] == k) for k in ("media", "workers")},
              "region_counts": [sum(1 for p in pr if p["region"] == r) for r in range(NUM_REGIONS)],
              "probed_region_personality_counts": [sum(1 for s in seeds if by["probe"][s]["true_tau"][by["probe"][s]["probe_region"]] == p)
                                                   for p in range(4)],
              "score": xc.mean_ci([p["score"] for p in pr]),
              "probe_seconds": xc.mean_ci([p["seconds"] for p in pr])}
    # ---- paired tests: probe vs bayes, probe vs poll_planner (and poll_planner vs bayes)
    pairs = [("probe", "bayes"), ("probe", "poll_planner"), ("poll_planner", "bayes")]
    fam_tau, fam_out = [], []
    for a, b in pairs:
        for t in (1, 2, 4):          # decision points 2, 3, 5 (after the probe has been seen in a poll)
            for m in ("H_perm", "H_probe_region", "p_true_tau_probe_region", "H_soft", "H_soft_probe_region"):
                xa = [by[a][s]["tau_learning"][t][m] for s in seeds]
                xb = [by[b][s]["tau_learning"][t][m] for s in seeds]
                keep = [i for i in range(len(seeds)) if not (math.isnan(xa[i]) or math.isnan(xb[i]))]
                fam_tau.append({"a": a, "b": b, "round": t + 1, "metric": m, "n": len(keep),
                                **xc.paired([xa[i] for i in keep], [xb[i] for i in keep])})
        ea = [by[a][s]["expected"] for s in seeds]
        eb = [by[b][s]["expected"] for s in seeds]
        fam_out.append({"a": a, "b": b, "E": xc.paired(ea, eb),
                        "victory": xc.mcnemar([by[a][s]["victory"] for s in seeds], [by[b][s]["victory"] for s in seeds]),
                        "guaranteed_rounds": xc.paired([sum(x["guaranteed"] for x in by[a][s]["timeline"]) for s in seeds],
                                                       [sum(x["guaranteed"] for x in by[b][s]["timeline"]) for s in seeds])})
    for t, adj in zip(fam_tau, xc.holm([t["wilcoxon_p"] for t in fam_tau])):
        t["holm_p"] = adj
    for key, pk in (("E", "wilcoxon_p"), ("victory", "p"), ("guaranteed_rounds", "wilcoxon_p")):
        for t, adj in zip(fam_out, xc.holm([t[key][pk] for t in fam_out])):
            t[key]["holm_p"] = adj
    return {"arms": arms, "probes": probes,
            "tests": {"tau_learning": {"size": len(fam_tau), "holm": "over all listed tests", "tests": fam_tau},
                      "outcome": {"size": len(fam_out), "holm": "per metric over the 3 pairs", "tests": fam_out}}}


# ============================================================================== main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X17"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--draws", type=int, default=TAU_DRAWS)
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=str(xc.REPORTS / "x17.json"))
    parser.add_argument("--raw", default=str(RAW_PATH))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = pinned_device(args.device)
    free0 = xc.free_bytes(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    raw_path = Path(args.raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    old = xc.read_rows(raw_path) if args.resume else []
    if not args.resume:
        raw_path.write_text("")
    done = {(r["seed"], r["arm"]) for r in old}
    rows = [r for r in old if r["seed"] in seeds]
    infos = []
    start = time.perf_counter()
    print(f"[x17] device {device} seeds {seeds[0]}..{seeds[-1]} arms {ARMS} draws {args.draws}", flush=True)
    with xc.GpuMonitor(device) as monitor:
        for i, seed in enumerate(seeds):
            t0 = time.perf_counter()
            if all((seed, a) in done for a in ARMS):
                continue
            new, info = seed_rows(seed, args.n, device, ARMS, done, draws=args.draws)
            infos.append(info)
            for r in new:
                xc.write_row(raw_path, r)
            rows += new
            sub = {r["arm"]: r for r in rows if r["seed"] == seed}
            pr = sub["probe"]["probe"]
            print(f"[x17] seed {seed} ({i + 1}/{len(seeds)}) {time.perf_counter() - t0:.0f}s, total "
                  f"{(time.perf_counter() - start) / 60:.1f} min; probe {pr['kind']} r{pr['region']}; E "
                  + " ".join(f"{a}={sub[a]['expected']:.3f}" for a in ARMS) + "; H_perm t5 "
                  + " ".join(f"{sub[a]['tau_learning'][-1]['H_perm']:.2f}" for a in ARMS), flush=True)
        summary = summarise(rows, seeds)
        # library check of the one-replay τ measurement (seed 0 of the block, probe arm)
        from dumbocracy.game import GameEngine
        from dumbocracy.game.observation import observation_for
        from dumbocracy.strategies.players import play_game
        cfg = GameConfig(seed=seeds[0], device=str(device), electorate=ElectorateConfig(num_voters=args.n))
        e = generate_electorate(cfg, device=device)
        final, _ = play_game(GameEngine.new_game(cfg, electorate=e, game_id="x17_check"), make_player("probe", seeds[0], str(device)))
        obs = observation_for(final, "p0")
        lib_check = check_against_library(obs, public_config(obs, cfg), [int(x) for x in e.personality_of_region.tolist()],
                                          seeds[0] * 7919 + 17, device, args.draws)
        print(f"[x17] library check: {lib_check}", flush=True)
        t0 = time.perf_counter()
        par = parity(seeds[:args.parity_seeds], device, args.n)
        par["seconds"] = time.perf_counter() - t0
        print(f"[x17] parity passed: {par['passed']} ({par['seconds']:.0f}s)", flush=True)
    seconds = time.perf_counter() - start
    for name, a in summary["arms"].items():
        print(f"[x17] {name}: win {a['win_rate']['rate']:.2f} E {a['expected']['mean']:.4f} H_perm by round "
              + " ".join(f"{x['mean']:.3f}" for x in a["tau_by_round"]["H_perm"]), flush=True)
    report = {
        "experiment": "X17", "title": "Learning personalities by probing",
        "question": "Can a deliberate single-region probe in round 1 make τ learnable from the national poll, and is "
                    "the exploration worth its price?",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
        "n": args.n, "rounds": cfg.rounds, "treasury": str(cfg.budget),
        "prices": {"worker": str(cfg.workers.worker_cost), "query": str(cfg.survey.query_cost),
                   "media_base": str(cfg.media.base_cost)},
        "arms": {"bayes": "BayesPlanner (library defaults)",
                 "poll_planner": "BayesPlanner with poll-consistent planning worlds (consistent_worlds, 8 worlds, "
                                 "poll_z 2.5, slack 0.01) — control",
                 "probe": "PollPlanner + round-1 single-region probe (media in r or the even cap on workers in r; "
                          "8 candidates scored on 96 worlds)"},
        "metric_definitions": {
            "tau_by_round[t]": "decision point t+1: statistics of the τ values of the accepted worlds among D candidates "
                               "(σ ~ posterior, τ uniform over 24 permutations) that match every poll s ≤ t+1 within "
                               "2.5·(SE_s + 0.01)",
            "H_perm": "plug-in entropy (nats) of the accepted τ over the 24 permutations (prior ln 24 = 3.178)",
            "H_perm_mm": "Miller–Madow corrected H_perm (+ (m − 1)/2n)",
            "H_soft / *_soft / ess_soft": "secondary, 'what the polls could teach': the D candidates weighted by the "
                                          "Gaussian poll likelihood Π_s N(p̂_s; E_w(s), SE_s²) WITHOUT the 0.01 slack "
                                          "(soft Bayesian posterior over τ; ess = 1/Σw² effective sample size)",
            "H_marginal_mean": "mean over regions of the entropy of the accepted personality of that region (prior ln 4 = 1.386)",
            "H_probe_region": "the same for the region the probe player probed in that seed (all arms use it)",
            "p_true_tau": "fraction of accepted worlds with exactly the true τ (prior 1/24 = 0.042)",
            "p_true_tau_r_mean / p_true_tau_probe_region": "fraction with the true personality of a region (prior 0.25)",
            "guaranteed_by_round": "K_p(Guaranteed) = ∃π∀w win over 16 poll-consistent worlds (logic.report timeline)",
            "expected / victory": "deterministic expected share E at the election; official sampled vote > N/2"},
        **summary, "seed_info": infos, "library_check": lib_check,
        "deviations": [
            "Added a control arm poll_planner (BayesPlanner with poll-consistent planning worlds): the plain BayesPlanner "
            "never conditions its planning worlds on the polls, so without the control the probe arm would change two "
            "things at once (probing and using the polls).",
            "τ learning is measured with D = 512 candidate worlds per game replayed once through the whole history "
            "(acceptance at every t from the recorded shares) instead of calling consistent_worlds at each t; this is "
            "the same rule and the same keys (library_check compares the accepted count at t = T with "
            "consistent_worlds). σ for these worlds is drawn from the posterior at t = T (all queries); σ entropy is "
            "≈ 0 after the survey (seed_info / tau_learning), so this does not affect τ.",
            "The probe score is a discriminability ratio (between-τ_r variance of the predicted round-2 poll over "
            "SE² + within variance), a monotone proxy for the expected information about τ_r from one Gaussian poll.",
            "World micro-states are drawn by the library sampler (sample_worlds) on CPU generators and moved to cuda:2; "
            "replays, rollouts and the probe scoring run on the GPU."],
        "config": cfg.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds,
                                       batch_sizes={"tau_draws_D": args.draws, "probe_worlds": 96, "planner_worlds": 8,
                                                    "timeline_worlds": TIMELINE_WORLDS, "world_voters": 1600,
                                                    "games_per_batch": 1},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"command": "python experiments/x17_probe.py --seeds 50 --first-seed 13000 "
                                                         "--device cuda:2 --out reports/v3/x17.json"}),
    }
    xc.save(args.out, report)
    print(f"[x17] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
