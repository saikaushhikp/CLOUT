"""X16 — Reliability and calibration of epistemic verdicts (plan ``docs/v2/plans/implementation_plan.md`` §6 X16).

Command (reproduces reports/v3/x16.json, reports/v3/raw/x16.jsonl, reports/v3/logs/x16.log and the figures):
  OMP_NUM_THREADS=4 python experiments/x16_verdicts.py --seeds 50 --first-seed 13000 --device cuda:0 \
      --out reports/v3/x16.json 2>&1 | tee reports/v3/logs/x16.log

Games. BayesPlanner and Top-3 (library players) each play the same 50 electorates (seed block 13000, N = 10⁴,
T = 5, ₫500, ₫20 workers, ₫10 queries; canonical CPU draw R8 moved to the GPU). Engine state, the planner's
posterior worlds and rollouts run on the GPU.

Verdicts. For every finished game and every decision point t = 1..5 the epistemic timeline of
``logic/report.timeline_with_diagnostics`` is recomputed for every (worlds W, world_voters n) ∈ {8,16,32,64} ×
{800,1600,4000} and every plan-set size |Π| ∈ {8, 16, all}:
* the sampled worlds W_t are ``logic.worlds.consistent_worlds`` with exactly the timeline's seeds (timeline seed 0);
* Π_t is ``logic.epistemic.policy_set(…, max_policies=k)`` — the approved D2 knob; a smaller Π is a prefix of the
  full Π, and with common random numbers a plan's outcome column does not depend on the other plans, so the
  ``[W, Π]`` outcome matrix is computed once with the full Π and the k = 8 and k = 16 verdicts are read from its
  first k columns. Both facts are checked at run time against the library ``timeline_with_diagnostics(…,
  max_policies=k)`` (``checks.library_identity``: identical rows) and in ``tests/unit/test_policy_set_max_policies.py``.

Truth.
* **Truth row** (the timeline's own ground truth): Π_t evaluated in the TRUE world at the decision point (engine
  replay, same evaluation stream) — ``truth_victory_possible(…, return_shares=True)``; once per game and round, the
  same for every (W, n).
* **Long-run truth** L_t (plan: "∃π in the X13 plan search wins in the true world"): the X13 open-loop plan search
  (cross-entropy method, 256 plans × 30 iterations, elite 10 %, smoothing 0.7, K = 4 replicate streams, then a
  1-move hill climb on K = 8 fixed streams; ``x13_plan_search.Pricing`` / ``neighbours``) run from the TRUE state at
  decision point t (true preferences, the player's workers, airings, remaining treasury and rounds, the player's
  committed manifesto), warm-started with every plan of the full Π_t (as open-loop sequences) and the player's own
  continuation. The search plan and all warm plans are re-evaluated on 20 fresh streams; L_t = 1 iff some plan's
  mean expected share over those streams is > 0.5.

Metrics (per verdict; a "decision point" is (game, t); counts pooled over decision points, CIs by a cluster
bootstrap over games, 2000 resamples):
* false-assert rate (soundness failure) = P(target false | verdict asserted);
* missed-truth rate (completeness failure) = P(verdict not asserted | target true);
  targets: possible / knows_win_exists → "∃π ∈ Π_t wins in the true world" (truth row) and L_t; guaranteed → "the
  guaranteed witness (first plan winning in every sampled world) wins in the true world" (truth row, de re) and L_t;
  inevitable_defeat → "no π ∈ Π_t wins in the true world" and ¬L_t.
* graded: p̂ = fraction of sampled worlds in which the best plan (most wins, ties by mean share) wins
  (``best_plan.win_fraction``); outcome y = the game's official victory (primary), plus "the best plan wins in the
  true world" (truth row) and L_t. Brier score with its Murphy decomposition (bins = the W + 1 possible p̂ values),
  reliability diagram (10 bins), over-confidence index OC = mean (p̂ − y)·sign(p̂ − ½) over p̂ ≠ ½ (> 0 =
  over-confident), and the calibration slope of a logistic recalibration y ~ logit(p̂) (p̂ clipped to
  [1/(2W), 1 − 1/(2W)]; slope < 1 = over-confident).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import io
import json
import multiprocessing as mp
import math
import sys
import time
import warnings
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x_parity import parity_rel  # noqa: E402
from x13_plan_search import Pricing, neighbours  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig, SeedStreams  # noqa: E402
from dumbocracy.dynamics.campaign import Allocation  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.game import GameEngine  # noqa: E402
from dumbocracy.game.observation import observation_for  # noqa: E402
from dumbocracy.logic.epistemic import evaluate, policy_set, properties, truth_victory_possible  # noqa: E402
from dumbocracy.logic.knowledge import decision_rounds, observation_at  # noqa: E402
from dumbocracy.logic.report import decision_states, timeline_with_diagnostics  # noqa: E402
from dumbocracy.electorate.electorate import Electorate  # noqa: E402
from dumbocracy.electorate.sbm import Graph  # noqa: E402
from dumbocracy.logic.epistemic import simulate_plans  # noqa: E402
from dumbocracy.logic.worlds import PERMUTATIONS, _poll_targets, consistent_worlds  # noqa: E402
from dumbocracy.strategies.bayes import ASSIGNMENTS, posterior_from_observation  # noqa: E402
from dumbocracy.sectors import REGIONS  # noqa: E402
from dumbocracy.strategies.planner import allocation_cost  # noqa: E402
from dumbocracy.strategies.players import BayesPlanner, Top3Player, play_game, public_config  # noqa: E402
from dumbocracy.strategies.worlds import BatchWorkers, WorldBatch, public_sizes, sample_worlds  # noqa: E402

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

WORLDS = (8, 16, 32, 64)
VOTERS = (800, 1600, 4000)
PI = (8, 16, None)                  # None = all candidates
PLAYERS = ("bayes", "top3")
PID = "p0"
TL_SEED = 0                         # timeline seed, as the API/X5
RAW_PATH = xc.RAW / "x16.jsonl"
FIG = xc.REPORTS / "figures"
GPU_MEMORY_CAP = 12 * 2**30         # lead's instruction: ≤ ~12 GB on the shared cuda:0
DEFAULT = (16, 1600, None)          # the API default timeline
VERDICTS = ("possible", "guaranteed", "knows_win_exists", "inevitable_defeat")
FRESH = 20


DEVIATIONS = [
    "Device cuda:0 (hard-pinned, lead's instruction; cuda:1 and cuda:2 are used by others), per-process cap 12 GB.",
    "Full grid as planned (4 worlds × 3 world sizes × 3 plan-set sizes, BayesPlanner and Top-3, 50 seeds each); no "
    "reduction was needed.",
    "World generation runs on the CPU (§1.1 point 5 exception): the library draws every synthetic world with CPU "
    "generators and only then moves it to the GPU, and this draw is ≈ 85 % of a timeline's cost. It runs in a spawn "
    "process pool (8 × 1 thread); everything else (history replay, poll filter, plan evaluation, truth row, long-run "
    "search) runs on the GPU. The pooled sampler re-implements consistent_worlds with pre-generated chunks and is "
    "checked against the library (checks.library_identity).",
    "§1.1 point 4 (batch the worlds of all decision points) is not done: WorldBatch.step applies one plan list to all "
    "worlds of a batch, and Π_t differs per round, so joining rounds would need a library change (not approved). The "
    "worlds × plans × voters of each decision point are batched; GPU utilisation is low because the GPU work per "
    "timeline is small (≈ 0.1–0.4 s) next to the CPU world generation.",
    "The |Π| = 8 and 16 verdicts are read from the first k columns of the full-Π outcome matrix (policy_set(…, "
    "max_policies=k) is a prefix and outcome columns are independent of the other plans under common random numbers; "
    "unit test + checks.library_identity against timeline_with_diagnostics(…, max_policies=k)).",
    "Identity checks compare discrete fields exactly and floats within 1e-5: GPU float32 sums are not bit-reproducible "
    "(the library differs from itself by ~1 ulp in some cells, checks.library_identity.library_vs_library_bit_identical).",
    "Long-run truth: X13's own rows cannot be reused (different seeds, they start at round 1 with the ORACLE manifesto), "
    "so the X13 search code (same CEM / hill-climb settings) is run from every decision point of the X16 games with the "
    "player's committed manifesto and state, warm-started with the full Π_t and the player's own continuation; the "
    "final re-evaluation uses 20 fresh WorldBatch streams of the true electorate (not x_batch.run_batch).",
    "Graded outcome: the plan says 'the realised outcome'; the primary y is the game's official victory, and two "
    "further targets are reported (the best plan wins in the true world; L_t).",
    "CPU↔GPU parity uses BayesPlanner games and the library timeline at 16 and 64 worlds of 1,600 voters (3 seeds); "
    "count metrics (rounds with a verdict, 0–5) use abs tolerance 0.25 or 5 % (x_parity.parity_rel).",
    "Per-cell 'sample_seconds' in timing.per_cell are main-process waits on the CPU pool (overlapped with the long-run "
    "search), not CPU cost; the CPU vs GPU time per library timeline is timing.library_timeline_cpu_vs_gpu.",
]

def pi_name(k: int | None) -> str:
    return "all" if k is None else str(k)


def cell_name(W: int, n: int, k: int | None) -> str:
    return f"W{W}_n{n}_pi{pi_name(k)}"


# ============================================================================== device (hard-pinned)
def pinned_device(requested: str) -> tuple[torch.device, dict]:
    """Exactly ``requested`` (never another GPU, no fallback); per-process cap ``GPU_MEMORY_CAP``."""
    if requested == "cpu":
        return torch.device("cpu"), {"requested": "cpu", "chosen": "cpu"}
    device = torch.device(requested)
    stats = xc.gpu_stats()
    idx = device.index or 0
    free = stats.get(idx, {}).get("free", 0)
    record = {"requested": requested, "chosen": str(device), "pinned": True, "free_bytes_at_select": int(free),
              "memory_cap_bytes": GPU_MEMORY_CAP, "util_at_select": stats.get(idx, {}).get("util")}
    if free < 4 * 2**30:
        raise SystemExit(f"{requested} has only {free / 2**30:.1f} GiB free; refusing to fall back to another GPU")
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_MEMORY_CAP / total), device)
    return device, record


# ============================================================================== games
def make_player(name: str, seed: int, device: torch.device):
    if name == "bayes":
        return BayesPlanner(seed=seed, device=str(device))
    if name == "top3":
        return Top3Player()
    raise ValueError(name)


def play_games(seed: int, n: int, device: torch.device, players=PLAYERS) -> dict[str, Any]:
    cfg = GameConfig(seed=seed, device=str(device), electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=device)
    state0 = GameEngine.new_game(cfg, electorate=e, game_id=f"x16_{seed}")
    out = {}
    for name in players:
        t0 = time.perf_counter()
        final, _ = play_game(state0, make_player(name, seed, device))
        out[name] = (final, time.perf_counter() - t0)
    return out


@dataclasses.dataclass
class GameCtx:
    """What ``timeline_with_diagnostics`` derives from a finished game (same seeds, same order)."""
    final: Any
    observation: dict
    config: GameConfig
    world_seed: int
    rounds: list[int]
    obs_t: dict[int, dict]
    eval_seed: dict[int, int]
    truth_states: dict[int, Any]
    plans: dict[int, list]


def game_ctx(final, seed: int = TL_SEED) -> GameCtx:
    observation = observation_for(final, PID)
    config = public_config(observation, replace(final.config, device="cpu"))
    streams = SeedStreams(int(seed))
    rounds = list(decision_rounds(observation))
    obs_t = {t: observation_at(observation, t, stage="decision") for t in rounds}
    return GameCtx(final=final, observation=observation, config=config,
                   world_seed=streams.derive("worlds", "logic-timeline", PID), rounds=rounds, obs_t=obs_t,
                   eval_seed={t: streams.derive("worlds", "logic-eval", PID, t) for t in rounds},
                   truth_states=decision_states(final, PID), plans={t: policy_set(obs_t[t], config) for t in rounds})


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)



# ============================================================================== CPU world generation pool
# ``consistent_worlds`` draws every synthetic world (SBM graph, preferences, susceptibilities) with CPU generators
# and only then moves it to the device (``strategies.worlds.sample_worlds``), and the SBM draw dominates the cost
# (≈ 85 % of a 64-world timeline). That step cannot run on the GPU without changing the library (its CPU streams
# define the worlds), so it is the §1.1 point 5 CPU exception: chunks are generated in a spawn process pool
# (``--cpu-workers`` × 1 thread) while the main process keeps everything else on the GPU (replay of the player's
# history, the poll filter, the plan evaluation, the truth row and the long-run search). ``pooled_consistent_worlds``
# re-implements ``consistent_worlds`` step by step with those pre-generated chunks; it is checked against the
# library (``checks.sampler_identity`` and ``checks.library_identity``).
def _worker_init() -> None:
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")


def sample_chunk(obs_t: dict, config: GameConfig, n: int, seed: int, k: int, size: int) -> bytes:
    """Chunk k of ``consistent_worlds`` (σ, τ and the micro-state of ``size`` worlds), generated on the CPU."""
    posterior = posterior_from_observation(obs_t, sample_fraction=config.survey.query_sample_fraction)
    cdf = torch.cumsum(posterior.probs, 0)
    streams = SeedStreams(int(seed))
    u = torch.rand(size, generator=streams.generator("worlds", "logic-sigma", k), dtype=torch.float64)
    idx = torch.searchsorted(cdf, u * cdf[-1]).clamp_max(cdf.numel() - 1)
    sigma = ASSIGNMENTS[idx]
    tau = PERMUTATIONS[torch.randint(len(PERMUTATIONS), (size,), generator=streams.generator("worlds", "logic-tau", k))]
    worlds = sample_worlds(obs_t, sigma, config, num_voters=n, seed=streams.derive("worlds", "logic-micro", k),
                           device=None, personalities=tau)
    buf = io.BytesIO()
    torch.save({"sigma": sigma, "tau": tau, "worlds": [dataclasses.asdict(w) | {"graph": dataclasses.asdict(w.graph),
                                                                               "susceptibility": w.susceptibility}
                                                       for w in worlds]}, buf)
    return buf.getvalue()


def _unpack(blob: bytes, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[Electorate]]:
    """Rebuild the chunk's worlds and move them exactly as ``sample_worlds(device=…)`` does."""
    d = torch.load(io.BytesIO(blob), weights_only=False)
    worlds = []
    for w in d["worlds"]:
        g = w["graph"]
        sus = w["susceptibility"]
        if device.type != "cpu":
            g = Graph(g["num_nodes"], g["src"].to(device), g["dst"].to(device), g["indptr"].to(device),
                      g["indices"].to(device), g["degree"].to(device))
            regions, prefs = w["region_of"].to(device), w["preferences"].to(device)
            sus = type(sus)(sus.social.to(device), sus.media.to(device), sus.campaign.to(device), sus.zealot.to(device))
        else:
            g = Graph(**g)
            regions, prefs = w["region_of"], w["preferences"]
        worlds.append(Electorate(w["sizes"], regions, g, prefs, sus, w["archetype_of_region"],
                                 w["personality_of_region"], w["p_intra"], w["p_inter"], w["edge_probability"],
                                 w["sparsity"], w["sparsity_capped"]))
    return d["sigma"], d["tau"], worlds


@dataclasses.dataclass
class Sampled:
    """The parts of a ``logic.worlds.WorldSet`` the verdicts need."""
    batch: WorldBatch
    current: torch.Tensor
    workers: BatchWorkers
    sigma: torch.Tensor
    tau: torch.Tensor
    draws: int
    accepted: int
    fallback: bool

    def __len__(self) -> int:
        return self.batch.num_worlds


class ChunkSource:
    """Futures of pre-submitted chunks; chunks not pre-submitted are generated on demand (in the pool)."""

    def __init__(self, pool: cf.Executor | None):
        self.pool = pool
        self.futures: dict[tuple, Any] = {}

    def submit(self, key: tuple, obs_t: dict, config: GameConfig, n: int, seed: int, k: int, size: int) -> None:
        self.futures[key + (k, size)] = self.pool.submit(sample_chunk, obs_t, config, n, seed, k, size)

    def get(self, key: tuple, obs_t: dict, config: GameConfig, n: int, seed: int, k: int, size: int) -> bytes:
        f = self.futures.pop(key + (k, size), None)
        if f is not None:
            return f.result()
        if self.pool is not None:
            return self.pool.submit(sample_chunk, obs_t, config, n, seed, k, size).result()
        return sample_chunk(obs_t, config, n, seed, k, size)

    def drop(self, key: tuple) -> None:
        for kk in [x for x in self.futures if x[:len(key)] == key]:
            self.futures.pop(kk).cancel()


def pooled_consistent_worlds(obs_t: dict, config: GameConfig, count: int, *, world_voters: int, seed: int,
                             device: torch.device, source: ChunkSource, key: tuple, poll_z: float = 2.5,
                             slack: float = 0.01) -> Sampled:
    """``logic.worlds.consistent_worlds`` (same defaults, same streams, same order) with CPU chunks from ``source``."""
    max_draws = max(4 * count, count)
    t = int(obs_t["round"])
    n = min(int(world_voters), int(obs_t["num_voters"]))
    streams = SeedStreams(int(seed))
    poll_rounds, est, scale = _poll_targets(obs_t, slack)
    chunks: list[dict[str, Any]] = []
    accepted = draws = k = 0
    while draws < max_draws and accepted < count:
        size = min(count, max_draws - draws)
        sigma, tau, worlds = _unpack(source.get(key, obs_t, config, n, seed, k, size), device)
        batch = WorldBatch(worlds, config)
        v_t, workers_t, shares = batch.replay_state(obs_t, streams.derive("worlds", "logic-replay", k), record_shares=True)
        at_polls = shares[:, [r - 1 for r in poll_rounds]].cpu() if poll_rounds else torch.zeros((size, 0), dtype=torch.float64)
        z = ((at_polls - est) / scale).abs().amax(1) if poll_rounds else torch.zeros(size, dtype=torch.float64)
        ok = z <= poll_z
        chunks.append({"worlds": worlds, "v": v_t.view(size, batch.n, -1), "w": workers_t, "sigma": sigma, "tau": tau,
                       "z": z, "ok": ok})
        accepted += int(ok.sum())
        draws += size
        k += 1
    source.drop(key)
    picks = [(c, i) for c, ch in enumerate(chunks) for i in range(len(ch["worlds"])) if bool(ch["ok"][i])][:count]
    fallback = not picks
    if fallback:
        allz = [(float(ch["z"][i]), c, i) for c, ch in enumerate(chunks) for i in range(len(ch["worlds"]))]
        picks = [(c, i) for _, c, i in sorted(allz)[:count]]
    batch = WorldBatch([chunks[c]["worlds"][i] for c, i in picks], config)
    current = torch.cat([chunks[c]["v"][i] for c, i in picks])
    n_w = chunks[0]["v"].shape[1]
    workers = BatchWorkers(*(torch.cat([getattr(chunks[c]["w"], name)[:, i * n_w:(i + 1) * n_w] for c, i in picks], dim=1)
                             for name in ("pinned", "spreaders", "attempts")))
    return Sampled(batch=batch, current=current, workers=workers,
                   sigma=torch.stack([chunks[c]["sigma"][i] for c, i in picks]),
                   tau=torch.stack([chunks[c]["tau"][i] for c, i in picks]),
                   draws=draws, accepted=accepted, fallback=fallback)


def submit_game(source: ChunkSource, ctx: GameCtx, tag: tuple, grid: list[tuple[int, int]], speculative: int = 2) -> None:
    """Pre-submit chunks 0..speculative−1 of every (W, n, t) of one game (typical draws: 2 chunks)."""
    for W, n in grid:
        nn = min(int(n), int(ctx.observation["num_voters"]))
        for t in ctx.rounds:
            for k in range(speculative):
                source.submit(tag + (W, n, t), ctx.obs_t[t], ctx.config, nn, ctx.world_seed, k, W)

# ============================================================================== verdict cells
def props_k(values: torch.Tensor, labels: list[str], k: int | None) -> dict:
    kk = values.shape[1] if k is None else min(k, values.shape[1])
    p = properties(values[:, :kk], labels[:kk])
    win = values[:, :kk] > 0.5
    known = win.all(0)
    return {"k": kk, "possible": p["possible_victory"], "guaranteed": p["guaranteed_victory"],
            "knows_win_exists": p["knows_win_exists"], "inevitable_defeat": p["inevitable_defeat"],
            "best": p["best_index"], "p_hat": p["best_plan"]["win_fraction"], "best_mean": p["best_plan"]["mean_share"],
            "witness": int(torch.nonzero(known)[0]) if bool(known.any()) else None,
            "n_guaranteed_plans": int(known.sum())}


@torch.no_grad()
def verdict_cell(ctx: GameCtx, W: int, n: int, device: torch.device, source: ChunkSource | None = None,
                 tag: tuple = ()) -> dict:
    """All decision points of one game at (W, n): worlds sampled once per round, full Π evaluated once.
    ``source`` = pooled CPU chunks (``pooled_consistent_worlds``); ``None`` = the library ``consistent_worlds``."""
    rounds = []
    t_sample = t_eval = 0.0
    for t in ctx.rounds:
        sync(device)
        a = time.perf_counter()
        if source is None:
            ws = consistent_worlds(ctx.obs_t[t], ctx.config, W, world_voters=n, seed=ctx.world_seed, device=device)
            batch, current, workers = ws.batch, ws.current, ws.workers
        else:
            ws = pooled_consistent_worlds(ctx.obs_t[t], ctx.config, W, world_voters=n, seed=ctx.world_seed,
                                          device=device, source=source, key=tag + (W, n, t))
            batch, current, workers = ws.batch, ws.current, ws.workers
        sync(device)
        b = time.perf_counter()
        plans = ctx.plans[t]
        values = simulate_plans(batch, current, plans, ctx.obs_t[t], ctx.config, seed=ctx.eval_seed[t], workers=workers)
        sync(device)
        c = time.perf_counter()
        t_sample += b - a
        t_eval += c - b
        labels = [p.label for p in plans]
        win = values > 0.5
        rounds.append({"round": t, "num_worlds": len(ws), "draws": ws.draws, "accepted": ws.accepted,
                       "fallback": bool(ws.fallback), "num_policies": len(plans),
                       "plan_wins": [int(x) for x in win.sum(0).tolist()],
                       "plan_mean": [round(float(x), 5) for x in values.mean(0).tolist()],
                       "by_pi": {pi_name(k): props_k(values, labels, k) for k in PI}})
    return {"rounds": rounds, "sample_seconds": t_sample, "eval_seconds": t_eval}


@torch.no_grad()
def truth_rows(ctx: GameCtx) -> list[dict]:
    out = []
    for t in ctx.rounds:
        if t not in ctx.truth_states:
            out.append({"round": t, "shares": None})
            continue
        _, shares = truth_victory_possible(ctx.truth_states[t], ctx.plans[t], player_id=PID, observation_t=ctx.obs_t[t],
                                           seed=ctx.eval_seed[t], return_shares=True)
        out.append({"round": t, "shares": [round(float(x), 6) for x in shares.tolist()]})
    return out


# ============================================================================== long-run truth (X13 search from t)
@dataclasses.dataclass
class LRSettings:
    population: int = 256
    iterations: int = 30
    elite_frac: float = 0.10
    smoothing: float = 0.7
    replicates: int = 4
    hc_replicates: int = 8
    hc_max_passes: int = 40
    init_scale: float = 1.0
    init_media: float = 0.25
    chunk: int = 256
    fresh: int = FRESH


class MidEvaluator:
    """C open-loop plans for the remaining L rounds from the TRUE decision-point state, K replicate streams of the
    true electorate in one ``WorldBatch`` (common random numbers across plans), as ``x13_plan_search.Evaluator``."""

    def __init__(self, state, obs_t: dict, K: int, chunk: int):
        e = state.electorate
        self.batch = WorldBatch([e] * K, state.config)
        self.K, self.chunk = K, chunk
        self.v0 = state.preferences.to(self.batch.device).repeat(K, 1)
        w = state.player(PID).workers
        self.w0 = None if w is None else BatchWorkers(*(x.repeat(1, K) for x in (
            BatchWorkers.from_state(w).pinned, BatchWorkers.from_state(w).spreaders, BatchWorkers.from_state(w).attempts)))
        self.air0 = [int(obs_t.get("media_airings", {}).get(r, 0)) for r in REGIONS]
        self.b = self.batch.manifesto_tensor(obs_t["manifesto"])
        self.plans_evaluated = 0

    @torch.no_grad()
    def _run(self, h: np.ndarray, m: np.ndarray, seed: int) -> torch.Tensor:
        batch, dev = self.batch, self.batch.device
        C, L = h.shape[0], h.shape[1]
        v = self.v0.unsqueeze(0).expand(C, -1, -1).contiguous()
        airings = torch.tensor([self.air0] * C, dtype=torch.int64, device=dev)
        workers = self.w0
        streams = SeedStreams(int(seed))
        for s in range(L):
            plans = [Allocation(tuple(int(x) for x in h[c, s]), tuple(bool(x) for x in m[c, s])) for c in range(C)]
            v, workers = batch.step(v, plans, airings, self.b, streams.generator("strategy", "x16-lr", s, device=dev),
                                    workers)
            airings = airings + torch.as_tensor(m[:, s], device=dev).to(torch.int64)
        return batch.shares(v, self.b)                                                   # [C, K]

    def __call__(self, h: np.ndarray, m: np.ndarray, seed: int, reduce: bool = True) -> np.ndarray:
        out, i = [], 0
        while i < h.shape[0]:
            j = min(h.shape[0], i + self.chunk)
            try:
                s = self._run(h[i:j], m[i:j], seed)
                out.append((s.mean(1) if reduce else s).double().cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                if self.chunk <= 1:
                    raise
                torch.cuda.empty_cache()
                self.chunk = max(1, self.chunk // 2)
                print(f"[x16] CUDA out of memory: long-run chunk halved to {self.chunk}", flush=True)
                continue
            i = j
        self.plans_evaluated += int(h.shape[0])
        return np.concatenate(out)


def open_loop(plan, sizes, treasury: Decimal, L: int, config: GameConfig) -> tuple[np.ndarray, np.ndarray]:
    """A Π_t plan as its open-loop allocation sequence (exactly the budget path of ``simulate_plans``)."""
    h = np.zeros((L, 4), dtype=np.int64)
    m = np.zeros((L, 4), dtype=bool)
    budget = treasury
    for s in range(L):
        a = plan.first if s == 0 else plan.allocation(sizes, budget, L - s, config)
        budget -= allocation_cost(a, sizes, config)
        h[s], m[s] = a.hires_by_region, a.media_regions
    return h, m


@torch.no_grad()
def long_run_truth(ctx: GameCtx, t: int, seed: int, settings: LRSettings) -> dict:
    """X13 search from the TRUE state at decision point t (module docstring)."""
    t0 = time.perf_counter()
    state, obs_t = ctx.truth_states[t], ctx.obs_t[t]
    cfg = state.config
    sizes = public_sizes(obs_t)
    treasury = Decimal(str(obs_t["treasury"]))
    L = int(obs_t["rounds"]) - t + 1
    pricing = Pricing(sizes, replace(cfg, budget=treasury))
    warm = [open_loop(p, sizes, treasury, L, ctx.config) for p in ctx.plans[t]]
    own = sorted([a for a in ctx.observation["allocations"] if int(a["round"]) >= t], key=lambda a: int(a["round"]))
    oh = np.array([[int(x) for x in a["hires_by_region"]] for a in own], dtype=np.int64).reshape(-1, 4)
    om = np.array([[bool(x) for x in a["media_regions"]] for a in own], dtype=bool).reshape(-1, 4)
    if len(own) == L:
        warm.append((oh, om))
    for h, m in warm:
        assert pricing.cost(h[None], m[None])[0] <= pricing.budget, "warm plan over budget"
    rng = np.random.default_rng([int(seed), 16, int(t)])
    Kc = pricing.max_workers + 1
    ev = MidEvaluator(state, obs_t, settings.replicates, settings.chunk)
    q = np.exp(-np.arange(Kc) / settings.init_scale)
    q = np.broadcast_to(q / q.sum(), (L, 4, Kc)).copy()
    p = np.full((L, 4), settings.init_media)
    n_elite = max(2, int(round(settings.elite_frac * settings.population)))
    base = SeedStreams(int(seed)).derive("strategy", "x16", "cem", t)
    best_h = best_m = None
    trace = []
    for it in range(settings.iterations):
        P = settings.population
        u = rng.random((P, L, 4, 1))
        h = (u > np.cumsum(q, -1)[None]).sum(-1).astype(np.int64).clip(0, Kc - 1)
        m = rng.random((P, L, 4)) < p[None]
        fixed = list(warm) if it == 0 else []
        if best_h is not None:
            fixed.append((best_h, best_m))
        for i, (wh, wm) in enumerate(fixed[:P]):
            h[i], m[i] = wh, wm
        pricing.repair(h, m, rng)
        scores = ev(h, m, base + it)
        order = np.argsort(-scores)
        elite = order[:n_elite]
        best_h, best_m = h[order[0]].copy(), m[order[0]].copy()
        counts = np.zeros((L, 4, Kc))
        for e_ in elite:
            counts[np.arange(L)[:, None], np.arange(4)[None, :], h[e_]] += 1
        q = settings.smoothing * counts / n_elite + (1 - settings.smoothing) * q
        p = settings.smoothing * m[elite].mean(0) + (1 - settings.smoothing) * p
        trace.append(float(scores[order[0]]))
    ev_hc = MidEvaluator(state, obs_t, settings.hc_replicates, settings.chunk)
    hc_seed = SeedStreams(int(seed)).derive("strategy", "x16", "hc", t)
    cand_h = np.concatenate([h[elite], np.stack([w[0] for w in warm])])
    cand_m = np.concatenate([m[elite], np.stack([w[1] for w in warm])])
    cs = ev_hc(cand_h, cand_m, hc_seed)
    j = int(np.argmax(cs))
    cur_h, cur_m, cur = cand_h[j].copy(), cand_m[j].copy(), float(cs[j])
    passes = 0
    for passes in range(1, settings.hc_max_passes + 1):
        H, M, _ = neighbours(cur_h, cur_m, pricing)
        s = ev_hc(H, M, hc_seed)
        j = int(np.argmax(s))
        if s[j] <= cur + 1e-7:
            break
        cur_h, cur_m, cur = H[j].copy(), M[j].copy(), float(s[j])
    # winner's curse: search plan + every warm plan on `fresh` new streams
    ev_f = MidEvaluator(state, obs_t, settings.fresh, settings.chunk)
    fh = np.concatenate([cur_h[None], np.stack([w[0] for w in warm])])
    fm = np.concatenate([cur_m[None], np.stack([w[1] for w in warm])])
    fresh = ev_f(fh, fm, SeedStreams(int(seed)).derive("strategy", "x16", "fresh", t))
    sources = ["search"] + [f"pi{i}" for i in range(len(ctx.plans[t]))] + (["own"] if len(warm) > len(ctx.plans[t]) else [])
    jb = int(np.argmax(fresh))
    return {"round": t, "L": L, "long_run": bool(fresh.max() > 0.5), "best_fresh": float(fresh.max()),
            "best_source": sources[jb], "search_fresh": float(fresh[0]), "search_hc": cur,
            "pi_fresh_max": float(fresh[1:1 + len(ctx.plans[t])].max()),
            "own_fresh": float(fresh[-1]) if sources[-1] == "own" else None,
            "search_plan": {"hires": cur_h.tolist(), "media": cur_m.astype(int).tolist()},
            "cem_best_trace_last": trace[-1], "hc_passes": passes,
            "evaluations": ev.plans_evaluated + ev_hc.plans_evaluated + ev_f.plans_evaluated,
            "seconds": time.perf_counter() - t0}


# ============================================================================== one seed
def seed_rows(seed: int, n_true: int, device: torch.device, grid: list[tuple[int, int]], lr: LRSettings | None,
              done: set, source: ChunkSource | None) -> list[dict]:
    rows = []
    games = play_games(seed, n_true, device)
    ctxs = {name: game_ctx(final) for name, (final, _) in games.items()}
    if source is not None:
        for name, ctx in ctxs.items():
            submit_game(source, ctx, (seed, name), [(W, n) for W, n in grid if (seed, name, f"W{W}_n{n}") not in done])
    for name, (final, game_s) in games.items():
        ctx = ctxs[name]
        res = ctx.observation["result"]
        if (seed, name, "game") not in done:
            a = time.perf_counter()
            truth = truth_rows(ctx)
            sync(device)
            truth_s = time.perf_counter() - a
            longrun = [long_run_truth(ctx, t, seed, lr) for t in ctx.rounds if t in ctx.truth_states] if lr else []
            rows.append({"exp": "X16", "kind": "game", "seed": seed, "player": name,
                         "expected": float(res["expected_share"]), "vote_share": float(res["vote_share"]),
                         "victory": bool(res["victory"]), "game_seconds": game_s, "truth_seconds": truth_s,
                         "manifesto": [round(float(x), 6) for x in ctx.observation["manifesto"]],
                         "plan_labels": {t: [p.label for p in ctx.plans[t]] for t in ctx.rounds},
                         "truth": truth, "long_run": longrun,
                         "long_run_seconds": sum(x["seconds"] for x in longrun)})
    for name, ctx in ctxs.items():
        for W, n in grid:
            if (seed, name, f"W{W}_n{n}") in done:
                continue
            cell = verdict_cell(ctx, W, n, device, source, (seed, name))
            rows.append({"exp": "X16", "kind": "cell", "seed": seed, "player": name, "worlds": W, "world_voters": n,
                         "arm": f"W{W}_n{n}", **cell})
    return rows


# ============================================================================== checks
def same_records(a: Any, b: Any, tol: float = 1e-5) -> bool:
    """Exact on every discrete field (bools, ints, labels, indices), |Δ| ≤ tol on floats (GPU float32 sums are
    not bit-reproducible run to run: the library differs from itself by ~1 ulp)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_records(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_records(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol
    return type(a) is type(b) and a == b


def library_identity(seed: int, n_true: int, device: torch.device, source: ChunkSource | None,
                     cells=((16, 1600), (64, 800), (8, 4000))) -> dict:
    """(a) the pooled CPU sampler gives exactly the library ``consistent_worlds`` result (identical per-round
    records: draws, acceptance, every plan's win count and mean share, every verdict); (b) the script's per-round
    loop + Π prefixes reproduce ``timeline_with_diagnostics(…, max_policies=k)`` (identical verdicts, p̂, best plan
    and truth row)."""
    games = play_games(seed, n_true, device)
    out, ok, pooled_ok = [], True, True
    for name, (final, _) in games.items():
        ctx = game_ctx(final)
        truth = truth_rows(ctx)
        for W, n in cells:
            cell = verdict_cell(ctx, W, n, device)
            again = verdict_cell(ctx, W, n, device)
            out.append({"player": name, "W": W, "n": n, "pi": "-", "round": "all", "check": "library_vs_library",
                        "identical": same_records(again["rounds"], cell["rounds"]),
                        "bit_identical": again["rounds"] == cell["rounds"]})
            if source is not None:
                submit_game(source, ctx, ("identity", name), [(W, n)])
                pooled = verdict_cell(ctx, W, n, device, source, ("identity", name))
                same_pool = same_records(pooled["rounds"], cell["rounds"])
                pooled_ok &= same_pool
                out.append({"player": name, "W": W, "n": n, "pi": "-", "round": "all", "check": "pooled_sampler",
                            "identical": same_pool, "bit_identical": pooled["rounds"] == cell["rounds"]})
            for k in PI:
                with torch.no_grad():
                    tl, diag = timeline_with_diagnostics(final, PID, worlds=W, world_voters=n, seed=TL_SEED,
                                                         device=device, max_policies=k)
                for r, mine, tr in zip(tl["rounds"], cell["rounds"], truth):
                    pk = mine["by_pi"][pi_name(k)]
                    tp = any(x > 0.5 for x in tr["shares"][:pk["k"]]) if tr["shares"] is not None else False
                    same = (r["possible_victory"] == pk["possible"] and r["guaranteed_victory"] == pk["guaranteed"]
                            and r["knows_win_exists"] == pk["knows_win_exists"]
                            and r["inevitable_defeat"] == pk["inevitable_defeat"]
                            and r["truth_victory_possible"] == tp
                            and abs(r["best_plan"]["win_fraction"] - pk["p_hat"]) == 0.0
                            and r["best_plan"]["label"] == ctx.plans[r["round"]][pk["best"]].label)
                    ok &= same
                    out.append({"player": name, "W": W, "n": n, "pi": pi_name(k), "round": r["round"],
                                "check": "library_timeline", "identical": same})
    lib_lib = [x for x in out if x["check"] == "library_vs_library"]
    return {"passed": bool(ok and pooled_ok), "pooled_sampler_identical": bool(pooled_ok),
            "rule": "discrete fields exact, floats |Δ| ≤ 1e-5 (GPU float32 is not bit-reproducible)",
            "library_vs_library_bit_identical": sum(x["bit_identical"] for x in lib_lib),
            "library_vs_library_cells": len(lib_lib),
            "pooled_bit_identical": sum(x.get("bit_identical", False) for x in out if x["check"] == "pooled_sampler"),
            "library_timeline_identical": bool(ok), "seed": seed, "cells": [list(c) for c in cells],
            "comparisons": len(out), "mismatches": [x for x in out if not x["identical"]]}


def parity_and_timing(seeds: list[int], device: torch.device, n_true: int) -> tuple[dict, dict]:
    """CPU↔GPU parity (3 seeds, BayesPlanner game + library timeline, 16 and 64 worlds of 1,600 voters, all on
    each device) and the wall time per library timeline at 16 and 64 worlds on each device."""
    cache: dict = {}
    timing: dict = {}

    def fn(seed: int, dev: torch.device) -> dict:
        key = (seed, str(dev))
        if key not in cache:
            games = play_games(seed, n_true, dev, players=("bayes",))
            final = games["bayes"][0]
            res = observation_for(final, PID)["result"]
            rec = {"expected": float(res["expected_share"]), "victory": float(res["victory"])}
            for W in (16, 64):
                sync(dev)
                a = time.perf_counter()
                tl, _ = timeline_with_diagnostics(final, PID, worlds=W, world_voters=1600, seed=TL_SEED, device=dev)
                sync(dev)
                timing.setdefault(f"{dev.type}_W{W}", []).append(time.perf_counter() - a)
                rows = tl["rounds"]
                rec[f"W{W}_guaranteed_rounds"] = float(sum(r["guaranteed_victory"] for r in rows))
                rec[f"W{W}_possible_rounds"] = float(sum(r["possible_victory"] for r in rows))
                rec[f"W{W}_p_hat_mean"] = sum(r["best_plan"]["win_fraction"] for r in rows) / len(rows)
                rec[f"W{W}_consistent_worlds_mean"] = sum(r["knowledge"]["consistent_worlds"] for r in rows) / len(rows)
            cache[key] = rec
        return cache[key]

    par = parity_rel(fn, seeds, device=device, abs_tol=0.25)
    tim = {k: {"mean_seconds": sum(v) / len(v), "runs": v} for k, v in timing.items()}
    for W in (16, 64):
        c, g = tim.get(f"cpu_W{W}"), tim.get(f"cuda_W{W}")
        if c and g:
            tim[f"speedup_W{W}"] = c["mean_seconds"] / g["mean_seconds"]
    tim["note"] = ("library timeline_with_diagnostics, 5 decision points, 1,600-voter worlds, full Π; BayesPlanner "
                   "games (CPU run: game, electorate and worlds on CPU; GPU run: all on the GPU)")
    return par, tim


# ============================================================================== records and statistics
def decision_records(rows: list[dict]) -> list[dict]:
    """One record per (game, decision point, W, n, Π) with verdicts, p̂ and every truth target."""
    games = {(r["seed"], r["player"]): r for r in rows if r["kind"] == "game"}
    out = []
    for r in rows:
        if r["kind"] != "cell":
            continue
        g = games[(r["seed"], r["player"])]
        truth = {x["round"]: x["shares"] for x in g["truth"]}
        lr = {x["round"]: x for x in g["long_run"]}
        for rd in r["rounds"]:
            t = rd["round"]
            shares = truth.get(t)
            for pi, pk in rd["by_pi"].items():
                k = pk["k"]
                tp = shares is not None and any(x > 0.5 for x in shares[:k])
                rec = {"seed": r["seed"], "player": r["player"], "W": r["worlds"], "n": r["world_voters"], "pi": pi,
                       "t": t, "k": k, "victory": g["victory"],
                       **{v: bool(pk[v]) for v in VERDICTS}, "p_hat": pk["p_hat"],
                       "truth_possible": bool(tp),
                       "witness_wins": (None if pk["witness"] is None or shares is None
                                        else bool(shares[pk["witness"]] > 0.5)),
                       "best_wins": None if shares is None else bool(shares[pk["best"]] > 0.5),
                       "long_run": bool(lr[t]["long_run"]) if t in lr else None,
                       "fallback": rd["fallback"]}
                out.append(rec)
    return out


def _boot_weights(G: int, B: int, seed: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(G, np.full(G, 1.0 / G), size=B).astype(np.float64)        # [B, G]


class Cluster:
    """Per-game sums for ratio statistics with a cluster bootstrap over games."""

    def __init__(self, games: list[tuple], B: int = 2000):
        self.index = {g: i for i, g in enumerate(games)}
        self.wt = _boot_weights(len(games), B)

    def ratio(self, recs: list[dict], num, den, binary: bool = True) -> dict:
        G = len(self.index)
        a, b = np.zeros(G), np.zeros(G)
        for r in recs:
            i = self.index[(r["seed"], r["player"])]
            d = den(r)
            if d:
                b[i] += d
                a[i] += num(r)
        if b.sum() == 0:
            return {"value": None, "num": 0, "den": 0, "ci95": None}
        bw = self.wt @ b
        bootv = (self.wt @ a)[bw > 0] / bw[bw > 0]
        lo, hi = np.percentile(bootv, [2.5, 97.5])
        out = {"value": float(a.sum() / b.sum()), "num": float(a.sum()) if not binary else int(round(a.sum())),
               "den": int(round(b.sum())), "ci95": [float(lo), float(hi)]}
        if binary:
            out["wilson95_unclustered"] = xc.wilson(int(round(a.sum())), int(round(b.sum())))
        return out

    def diff(self, recs_a: list[dict], recs_b: list[dict], num, den) -> dict:
        """Difference of two ratio statistics (A − B) with a paired cluster bootstrap (same resampled games)."""
        def sums(recs):
            G = len(self.index)
            a, b = np.zeros(G), np.zeros(G)
            for r in recs:
                i = self.index[(r["seed"], r["player"])]
                d = den(r)
                if d:
                    b[i] += d
                    a[i] += num(r)
            return a, b
        a1, b1 = sums(recs_a)
        a2, b2 = sums(recs_b)
        if b1.sum() == 0 or b2.sum() == 0:
            return {"diff": None}
        w1, w2 = self.wt @ b1, self.wt @ b2
        ok = (w1 > 0) & (w2 > 0)
        boot = (self.wt @ a1)[ok] / w1[ok] - (self.wt @ a2)[ok] / w2[ok]
        d = a1.sum() / b1.sum() - a2.sum() / b2.sum()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
        return {"diff": float(d), "ci95": [float(lo), float(hi)], "boot_p_two_sided": max(p, 1.0 / len(boot)),
                "a": float(a1.sum() / b1.sum()), "b": float(a2.sum() / b2.sum())}


# targets (truth row and long-run) for every verdict: function(record) -> bool | None
def targets(v: str, ref: str):
    if ref == "row":
        if v in ("possible", "knows_win_exists"):
            return lambda r: r["truth_possible"]
        if v == "guaranteed":
            return lambda r: r["witness_wins"] if r["guaranteed"] else r["truth_possible"]
        return lambda r: not r["truth_possible"]
    if v == "inevitable_defeat":
        return lambda r: None if r["long_run"] is None else not r["long_run"]
    return lambda r: r["long_run"]


def verdict_reliability(cl: Cluster, recs: list[dict]) -> dict:
    out = {}
    for v in VERDICTS:
        out[v] = {"assert_rate": cl.ratio(recs, lambda r: float(r[v]), lambda r: 1.0)}
        for ref in ("row", "long_run"):
            tgt = targets(v, ref)
            sub = [r for r in recs if tgt(r) is not None]
            out[v][ref] = {
                "false_assert": cl.ratio(sub, lambda r: float(not tgt(r)), lambda r: float(r[v])),
                "missed_truth": cl.ratio(sub, lambda r: float(not r[v]), lambda r: float(bool(tgt(r)))),
                "target_rate": cl.ratio(sub, lambda r: float(bool(tgt(r))), lambda r: 1.0)}
    return out


def _logit(p):
    return np.log(p / (1 - p))


def calibration_slope(p: np.ndarray, y: np.ndarray, W: np.ndarray) -> dict:
    """Logistic recalibration y ~ a + b·logit(p̂) (Newton); b < 1 = over-confident."""
    eps = 0.5 / W
    x = _logit(np.clip(p, eps, 1 - eps))
    X = np.stack([np.ones_like(x), x], 1)
    beta = np.zeros(2)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (y - mu)
        H = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1])}


def graded(cl: Cluster, recs: list[dict], outcome: str, B_slope: int = 400) -> dict:
    sub = [r for r in recs if r[outcome] is not None]
    if not sub:
        return {}
    p = np.array([r["p_hat"] for r in sub])
    y = np.array([float(r[outcome]) for r in sub])
    W = np.array([r["W"] for r in sub], dtype=float)
    brier = cl.ratio(sub, lambda r: (r["p_hat"] - float(r[outcome])) ** 2, lambda r: 1.0, binary=False)
    oc = cl.ratio(sub, lambda r: (r["p_hat"] - float(r[outcome])) * math.copysign(1.0, r["p_hat"] - 0.5),
                  lambda r: float(r["p_hat"] != 0.5), binary=False)
    # Murphy decomposition on the distinct p̂ values
    base = y.mean()
    rel = res = 0.0
    for val in np.unique(p):
        s = p == val
        o = y[s].mean()
        rel += s.sum() * (val - o) ** 2
        res += s.sum() * (o - base) ** 2
    rel, res = rel / len(p), res / len(p)
    bins = np.minimum((p * 10).astype(int), 9)
    diagram = []
    for b in range(10):
        s = bins == b
        if s.any():
            diagram.append({"bin": [b / 10, (b + 1) / 10], "n": int(s.sum()), "mean_p_hat": float(p[s].mean()),
                            "observed": float(y[s].mean()), "wilson95": xc.wilson(int(y[s].sum()), int(s.sum()))})
    slope = calibration_slope(p, y, W)
    games = sorted({(r["seed"], r["player"]) for r in sub})
    gi = {g: i for i, g in enumerate(games)}
    idx = np.array([gi[(r["seed"], r["player"])] for r in sub])
    rng = np.random.default_rng(161)
    bs = []
    for _ in range(B_slope):
        pick = rng.integers(len(games), size=len(games))
        sel = np.concatenate([np.nonzero(idx == g)[0] for g in pick])
        if len(np.unique(y[sel])) < 2:
            continue
        bs.append(calibration_slope(p[sel], y[sel], W[sel])["slope"])
    slope["ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None
    return {"n": len(sub), "brier": brier, "murphy": {"reliability": rel, "resolution": res,
                                                     "uncertainty": float(base * (1 - base))},
            "mean_p_hat": float(p.mean()), "base_rate": float(base), "overconfidence": oc,
            "calibration": slope, "reliability_diagram": diagram,
            "share_p_hat_0_or_1": float(((p == 0) | (p == 1)).mean())}


def summarise(rows: list[dict], grid: list[tuple[int, int]]) -> tuple[dict, list[dict]]:
    recs = decision_records(rows)
    games = sorted({(r["seed"], r["player"]) for r in rows if r["kind"] == "game"})
    cl = Cluster(games)
    cells: dict[str, Any] = {}
    for W, n in grid:
        for k in PI:
            name = cell_name(W, n, k)
            sub_all = [r for r in recs if r["W"] == W and r["n"] == n and r["pi"] == pi_name(k)]
            c = {"worlds": W, "world_voters": n, "pi": pi_name(k), "decision_points": len(sub_all),
                 "mean_plan_set_size": float(np.mean([r["k"] for r in sub_all])) if sub_all else None,
                 "fallback_rate": float(np.mean([r["fallback"] for r in sub_all])) if sub_all else None}
            for who in ("pooled",) + PLAYERS:
                sub = sub_all if who == "pooled" else [r for r in sub_all if r["player"] == who]
                c[who] = {"verdicts": verdict_reliability(cl, sub),
                          "graded": {o: graded(cl, sub, o, B_slope=400 if who == "pooled" else 0)
                                     for o in ("victory", "best_wins", "long_run")}}
            cells[name] = c
            print(f"[x16] summarised {name}", flush=True)
    return {"cells": cells}, recs


def truth_agreement(rows: list[dict]) -> dict:
    """Truth row (∃π ∈ Π_t on one stream) vs long-run truth (X13 search, 20 fresh streams)."""
    pairs = []
    for g in (r for r in rows if r["kind"] == "game"):
        lr = {x["round"]: x for x in g["long_run"]}
        for x in g["truth"]:
            if x["shares"] is None or x["round"] not in lr:
                continue
            pairs.append((any(s > 0.5 for s in x["shares"]), lr[x["round"]]["long_run"], g["player"], x["round"],
                          lr[x["round"]]["best_source"], lr[x["round"]]["search_fresh"] - lr[x["round"]]["pi_fresh_max"]))
    if not pairs:
        return {}
    n = len(pairs)
    return {"decision_points": n,
            "row_true": sum(a for a, *_ in pairs), "long_run_true": sum(b for _, b, *_ in pairs),
            "both": sum(a and b for a, b, *_ in pairs), "row_only": sum(a and not b for a, b, *_ in pairs),
            "long_run_only": sum(b and not a for a, b, *_ in pairs),
            "long_run_rate_by_round": {t: xc.rate([b for a, b, p, tt, *_ in pairs if tt == t]) for t in range(1, 6)},
            "row_rate_by_round": {t: xc.rate([a for a, b, p, tt, *_ in pairs if tt == t]) for t in range(1, 6)},
            "best_source_counts": {s: sum(1 for *_, src, _d in pairs if src == s.split(":")[0])
                                   for s in ("search",)} | {"pi": sum(1 for *_, src, _d in pairs if src.startswith("pi")),
                                                             "own": sum(1 for *_, src, _d in pairs if src == "own")},
            "search_minus_best_pi_fresh": xc.mean_ci([d for *_, d in pairs])}


def tests(recs: list[dict], rows: list[dict], cells: dict) -> dict:
    """Pre-registered contrasts (prediction: guaranteed rarer and more reliable as W grows; p̂ over-confident at 16
    worlds). Holm over the family."""
    games = sorted({(r["seed"], r["player"]) for r in rows if r["kind"] == "game"})
    cl = Cluster(games)
    fam = []
    n0, k0 = 1600, "all"

    def sel(W, who="pooled"):
        return [r for r in recs if r["W"] == W and r["n"] == n0 and r["pi"] == k0 and (who == "pooled" or r["player"] == who)]

    for who in ("pooled",) + PLAYERS:
        a, b = sel(64, who), sel(8, who)
        # rarer: assertion rate, paired by decision point (exact McNemar) + cluster bootstrap
        ka = {(r["seed"], r["player"], r["t"]): r["guaranteed"] for r in a}
        kb = {(r["seed"], r["player"], r["t"]): r["guaranteed"] for r in b}
        keys = sorted(set(ka) & set(kb))
        fam.append({"name": f"guaranteed assert rate W64 − W8 ({who})", "metric": "assert_rate",
                    **cl.diff(a, b, lambda r: float(r["guaranteed"]), lambda r: 1.0),
                    "mcnemar": xc.mcnemar([ka[x] for x in keys], [kb[x] for x in keys])})
        fam.append({"name": f"guaranteed false-assert (truth row, de re) W64 − W8 ({who})", "metric": "false_assert_row",
                    **cl.diff(a, b, lambda r: float(not r["witness_wins"]), lambda r: float(r["guaranteed"]))})
        fam.append({"name": f"guaranteed false-assert (long run) W64 − W8 ({who})", "metric": "false_assert_long_run",
                    **cl.diff([r for r in a if r["long_run"] is not None], [r for r in b if r["long_run"] is not None],
                              lambda r: float(not r["long_run"]), lambda r: float(r["guaranteed"]))})
    for who in ("pooled",):
        for o in ("victory", "best_wins", "long_run"):
            g = cells[cell_name(16, n0, None)][who]["graded"][o]
            oc = g["overconfidence"]
            boot_p = None
            fam.append({"name": f"over-confidence OC at W16 vs y = {o} ({who})", "metric": "overconfidence",
                        "value": oc["value"], "ci95": oc["ci95"], "calibration_slope": g["calibration"],
                        "boot_p_two_sided": boot_p})
    # p-values for OC: cluster bootstrap of the OC statistic sign
    for it in fam:
        if it["metric"] == "overconfidence":
            o = it["name"].split("y = ")[1].split(" ")[0]
            sub = [r for r in recs if r["W"] == 16 and r["n"] == n0 and r["pi"] == k0 and r[o] is not None]
            G = len(cl.index)
            a, b = np.zeros(G), np.zeros(G)
            for r in sub:
                i = cl.index[(r["seed"], r["player"])]
                if r["p_hat"] != 0.5:
                    a[i] += (r["p_hat"] - float(r[o])) * math.copysign(1.0, r["p_hat"] - 0.5)
                    b[i] += 1
            bw = cl.wt @ b
            boot = (cl.wt @ a)[bw > 0] / bw[bw > 0]
            it["boot_p_two_sided"] = max(1.0 / len(boot), float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))))
    ps = [it.get("boot_p_two_sided") if it["metric"] != "assert_rate" else it["mcnemar"]["p"] for it in fam]
    adj = xc.holm([1.0 if p is None else p for p in ps])
    for it, p, h in zip(fam, ps, adj):
        it["p"] = p
        it["holm_p"] = h
    return {"family": "X16 pre-registered contrasts at n = 1600, Π = all: guaranteed W64 vs W8 (assert rate: exact "
                      "McNemar paired by decision point; false-assert: cluster-bootstrap p), over-confidence at W16 "
                      "(cluster-bootstrap p of OC ≠ 0); Holm over all items", "tests": fam}


def world_axis(cells: dict) -> dict:
    """Compact table along W (n = 1600, Π = all), the headline of the prediction."""
    out = {}
    for W in WORLDS:
        c = cells.get(cell_name(W, 1600, None))
        if not c:
            continue
        p = c["pooled"]
        out[str(W)] = {"guaranteed_assert": p["verdicts"]["guaranteed"]["assert_rate"]["value"],
                       "guaranteed_false_assert_row": p["verdicts"]["guaranteed"]["row"]["false_assert"]["value"],
                       "guaranteed_false_assert_long_run": p["verdicts"]["guaranteed"]["long_run"]["false_assert"]["value"],
                       "possible_false_assert_row": p["verdicts"]["possible"]["row"]["false_assert"]["value"],
                       "inevitable_defeat_false_assert_long_run":
                           p["verdicts"]["inevitable_defeat"]["long_run"]["false_assert"]["value"],
                       "brier_victory": p["graded"]["victory"]["brier"]["value"],
                       "overconfidence_victory": p["graded"]["victory"]["overconfidence"]["value"],
                       "calibration_slope_victory": p["graded"]["victory"]["calibration"]["slope"]}
    return out


# ============================================================================== figures (from the saved JSON only)
def plot(report: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    cells = report["cells"]
    written = []
    colors = {800: "#9aa8c7", 1600: "#3b5ba5", 4000: "#0b1f4f"}
    pis = {"8": ":", "16": "--", "all": "-"}
    # 1. guaranteed assert rate and false-assert rate vs W
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for n in VOTERS:
        for pi in ("8", "16", "all"):
            xs, a, f_row, f_lr = [], [], [], []
            for W in WORLDS:
                c = cells.get(f"W{W}_n{n}_pi{pi}")
                if not c:
                    continue
                v = c["pooled"]["verdicts"]["guaranteed"]
                xs.append(W)
                a.append(v["assert_rate"]["value"])
                f_row.append(v["row"]["false_assert"]["value"])
                f_lr.append(v["long_run"]["false_assert"]["value"])
            if not xs:
                continue
            kw = dict(color=colors[n], ls=pis[pi], marker="o", ms=3, label=f"n={n}, |Π|={pi}")
            axes[0].plot(xs, a, **kw)
            axes[1].plot(xs, [np.nan if x is None else x for x in f_row], **kw)
            axes[2].plot(xs, [np.nan if x is None else x for x in f_lr], **kw)
    for ax, title in zip(axes, ("P(guaranteed asserted)", "false-assert vs truth row (witness loses)",
                                "false-assert vs long-run truth")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(WORLDS, [str(w) for w in WORLDS])
        ax.set_xlabel("sampled worlds W")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=1)
    fig.suptitle("X16 — 'guaranteed victory' vs the world sample (BayesPlanner + Top-3 pooled, 50 seeds each)", fontsize=10)
    fig.tight_layout()
    p = FIG / "x16_guaranteed_vs_worlds.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))
    # 2. reliability diagrams (y = official victory) for W ∈ {8,16,64}, n = 1600, Π = all
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, o, title in zip(axes, ("victory", "best_wins", "long_run"),
                            ("y = game won", "y = best plan wins (truth row)", "y = long-run truth L_t")):
        ax.plot([0, 1], [0, 1], color="#999", lw=0.8)
        for W, col in zip((8, 16, 64), ("#e3a23b", "#3b5ba5", "#2a8c5a")):
            g = cells.get(f"W{W}_n1600_piall", {}).get("pooled", {}).get("graded", {}).get(o)
            if not g:
                continue
            d = g["reliability_diagram"]
            ax.plot([x["mean_p_hat"] for x in d], [x["observed"] for x in d], marker="o", color=col,
                    label=f"W={W}: Brier {g['brier']['value']:.3f}, slope {g['calibration']['slope']:.2f}")
            for x in d:
                ax.annotate(str(x["n"]), (x["mean_p_hat"], x["observed"]), fontsize=5, color=col,
                            xytext=(2, -7), textcoords="offset points")
        ax.set_xlabel("p̂ (fraction of sampled worlds the best plan wins)")
        ax.set_ylabel("observed frequency")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("X16 — reliability of p̂ (n = 1,600 voters per world, full Π; numbers = decision points per bin)",
                 fontsize=10)
    fig.tight_layout()
    p = FIG / "x16_reliability.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))
    # 3. heatmap: false-assert / missed-truth per verdict across (W, n) at Π = all
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    for j, v in enumerate(VERDICTS):
        for i, (metric, ref) in enumerate((("false_assert", "row"), ("missed_truth", "row"))):
            M = np.full((len(VOTERS), len(WORLDS)), np.nan)
            for a, n in enumerate(VOTERS):
                for b, W in enumerate(WORLDS):
                    c = cells.get(f"W{W}_n{n}_piall")
                    if c:
                        x = c["pooled"]["verdicts"][v][ref][metric]["value"]
                        M[a, b] = np.nan if x is None else x
            ax = axes[i, j]
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            for a in range(len(VOTERS)):
                for b in range(len(WORLDS)):
                    ax.text(b, a, "–" if np.isnan(M[a, b]) else f"{M[a, b]:.2f}", ha="center", va="center",
                            fontsize=8, color="black" if np.isnan(M[a, b]) or M[a, b] >= 0.6 else "white")
            ax.set_xticks(range(len(WORLDS)), [str(w) for w in WORLDS])
            ax.set_yticks(range(len(VOTERS)), [str(n) for n in VOTERS])
            ax.set_title(f"{v}: {metric.replace('_', '-')}", fontsize=9)
            if j == 0:
                ax.set_ylabel("world voters n")
            if i == 1:
                ax.set_xlabel("worlds W")
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.suptitle("X16 — soundness (false-assert) and completeness (missed-truth) vs the truth row, |Π| = all, "
                 "pooled players; '–' = never asserted / target never true", fontsize=10)
    p = FIG / "x16_verdict_errors.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(str(p))
    return written


# ============================================================================== probe (timing)
def probe(seed: int, device: torch.device, n_true: int, lr: LRSettings, source: ChunkSource) -> None:
    games = play_games(seed, n_true, device)
    probe_grid = [(8, 800), (16, 1600), (64, 1600), (64, 4000)]
    a0 = time.perf_counter()
    for name, (final, gs) in games.items():
        print(f"[probe] {name}: game {gs:.1f}s", flush=True)
        a = time.perf_counter()
        ctx = game_ctx(final)
        if source.pool is not None:
            submit_game(source, ctx, ("probe", name), probe_grid)
        truth_rows(ctx)
        print(f"[probe] {name}: ctx+truth {time.perf_counter() - a:.1f}s; Π sizes "
              f"{[len(ctx.plans[t]) for t in ctx.rounds]}", flush=True)
        for W, n in probe_grid:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            cell = verdict_cell(ctx, W, n, device, source if source.pool is not None else None, ("probe", name))
            pk = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0
            print(f"[probe] {name} W{W} n{n}: sample {cell['sample_seconds']:.1f}s eval {cell['eval_seconds']:.1f}s "
                  f"peak {pk:.0f} MiB; G(all) {[r['by_pi']['all']['guaranteed'] for r in cell['rounds']]} "
                  f"p̂ {[round(r['by_pi']['all']['p_hat'], 2) for r in cell['rounds']]}", flush=True)
        a = time.perf_counter()
        lrs = [long_run_truth(ctx, t, seed, lr) for t in ctx.rounds if t in ctx.truth_states]
        print(f"[probe] {name}: long-run {time.perf_counter() - a:.1f}s "
              f"{[(x['round'], x['long_run'], round(x['best_fresh'], 3), x['best_source'], round(x['seconds'], 1)) for x in lrs]}",
              flush=True)
    print(f"[probe] total {time.perf_counter() - a0:.1f}s", flush=True)


# ============================================================================== main
def main() -> None:
    global WORLDS, VOTERS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X16"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--worlds", default=",".join(map(str, WORLDS)))
    parser.add_argument("--voters", default=",".join(map(str, VOTERS)))
    parser.add_argument("--design", default="full", choices=("full", "ofat"),
                        help="full = every (W, n); ofat = W axis at n = 1600 + n axis at W = 16")
    parser.add_argument("--lr-iterations", type=int, default=30)
    parser.add_argument("--cpu-workers", type=int, default=8, help="spawn processes generating worlds on the CPU (0 = inline)")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--skip-checks", action="store_true", help="smoke tests only (no identity/parity)")
    parser.add_argument("--plot-only", action="store_true", help="redraw the figures from the saved --out JSON")
    parser.add_argument("--summarise-only", action="store_true", help="recompute the summary from raw/x16.jsonl")
    parser.add_argument("--resume", action="store_true", help="keep rows already in raw/x16.jsonl and skip them")
    parser.add_argument("--out", default=str(xc.REPORTS / "x16.json"))
    parser.add_argument("--raw", default=str(RAW_PATH))
    args = parser.parse_args()
    WORLDS = tuple(int(x) for x in args.worlds.split(","))
    VOTERS = tuple(int(x) for x in args.voters.split(","))
    if args.design == "full":
        grid = [(W, n) for W in WORLDS for n in VOTERS]
    else:
        grid = sorted({(W, 1600) for W in WORLDS} | {(16, n) for n in VOTERS})
    if args.plot_only:
        report = json.loads(Path(args.out).read_text())
        report["figures"] = plot(report)
        xc.save(args.out, report)
        return
    torch.set_num_threads(4)
    device, dev_record = pinned_device(args.device)
    lr = LRSettings(iterations=args.lr_iterations)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    pool = (cf.ProcessPoolExecutor(args.cpu_workers, mp_context=mp.get_context("spawn"), initializer=_worker_init)
            if args.cpu_workers > 0 else None)
    source = ChunkSource(pool)
    if args.probe:
        probe(seeds[0], device, args.n, lr, source)
        return
    raw_path = Path(args.raw)
    free0 = xc.free_bytes(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    old = xc.read_rows(raw_path) if (args.resume or args.summarise_only) and raw_path.exists() else []
    if not (args.resume or args.summarise_only):
        raw_path.write_text("")
    done = {(r["seed"], r["player"], "game" if r["kind"] == "game" else r["arm"]) for r in old}
    rows = [r for r in old if r["seed"] in seeds]
    start = time.perf_counter()
    print(f"[x16] device {device} seeds {seeds[0]}..{seeds[-1]} grid {grid} (resumed rows: {len(rows)})", flush=True)
    checks: dict[str, Any] = {}
    with xc.GpuMonitor(device) as monitor:
        if not args.summarise_only:
            for i, seed in enumerate(seeds):
                t0 = time.perf_counter()
                new = seed_rows(seed, args.n, device, grid, lr, done, source)
                for r in new:
                    xc.write_row(raw_path, r)
                rows += new
                g = [r for r in rows if r["seed"] == seed and r["kind"] == "game"]
                print(f"[x16] seed {seed} ({i + 1}/{len(seeds)}) {time.perf_counter() - t0:.0f}s, total "
                      f"{(time.perf_counter() - start) / 60:.1f} min; long-run "
                      + "; ".join(f"{r['player']} {[int(x['long_run']) for x in r['long_run']]} "
                                  f"({r['long_run_seconds']:.0f}s) won {int(r['victory'])}" for r in g), flush=True)
        run_seconds = time.perf_counter() - start
        par: dict | None = None
        timing: dict = {}
        if not args.skip_checks:
            t0 = time.perf_counter()
            checks["library_identity"] = library_identity(seeds[0], args.n, device, source)
            checks["library_identity"]["seconds"] = time.perf_counter() - t0
            print(f"[x16] library identity passed: {checks['library_identity']['passed']} "
                  f"({checks['library_identity']['comparisons']} round comparisons)", flush=True)
            t0 = time.perf_counter()
            par, timing = parity_and_timing(seeds[:args.parity_seeds], device, args.n)
            par["seconds"] = time.perf_counter() - t0
            print(f"[x16] parity passed: {par['passed']} ({par['seconds']:.0f}s); timing "
                  f"{json.dumps({k: (v['mean_seconds'] if isinstance(v, dict) else v) for k, v in timing.items() if k != 'note'})}",
                  flush=True)
    seconds = time.perf_counter() - start
    summary, recs = summarise(rows, grid)
    test = tests(recs, rows, summary["cells"])
    truth_cmp = truth_agreement(rows)
    cells_rows = [r for r in rows if r["kind"] == "cell"]
    game_rows = [r for r in rows if r["kind"] == "game"]
    timing_cells = {}
    for W, n in grid:
        sub = [r for r in cells_rows if r["worlds"] == W and r["world_voters"] == n]
        if sub:
            timing_cells[f"W{W}_n{n}"] = {"sample_seconds_mean": float(np.mean([r["sample_seconds"] for r in sub])),
                                          "eval_seconds_mean": float(np.mean([r["eval_seconds"] for r in sub])),
                                          "acceptance": float(sum(d["accepted"] for r in sub for d in r["rounds"])
                                                              / sum(d["draws"] for r in sub for d in r["rounds"])),
                                          "fallback_rounds": int(sum(d["fallback"] for r in sub for d in r["rounds"]))}
    base = GameConfig(seed=seeds[0], electorate=ElectorateConfig(num_voters=args.n))
    report = {
        "experiment": "X16", "title": "Reliability and calibration of epistemic verdicts",
        "question": "How often are the sampled epistemic verdicts (guaranteed / possible / knows-that / inevitable "
                    "defeat) wrong, and how do they depend on the number of worlds, the world size and |Π|?",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True,
                  "choice": "X16 seed block 13000 (plan §1.5); the long-run truth is computed by running the X13 "
                            "search code from each decision point of these games, not by reusing X13's seeds"},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget),
        "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost),
                   "media_base": str(base.media.base_cost)},
        "players": {"bayes": "BayesPlanner(seed=seed, device=cuda:0) (library defaults)",
                    "top3": "Top3Player() (library)"},
        "grid": {"worlds": list(WORLDS), "world_voters": list(VOTERS), "pi": [pi_name(k) for k in PI],
                 "design": args.design, "cells": [f"W{W}_n{n}" for W, n in grid]},
        "timeline_seed": TL_SEED,
        "long_run_search": dataclasses.asdict(lr),
        "metric_definitions": {
            "decision point": "(game, round t), t = 1..5; knowledge at the moment the round-t allocation was chosen "
                              "(observation_at stage 'decision')",
            "verdicts": "over the sampled worlds W_t and Π_t: possible = ∃π∃w win; guaranteed = ∃π∀w win; "
                        "knows_win_exists = ∀w∃π win; inevitable_defeat = ∀π∀w ¬win; win ⇔ expected share > 0.5",
            "truth row": "Π_t evaluated in the TRUE world at the decision point (engine replay, the timeline's "
                         "evaluation stream) — the timeline's truth_victory_possible",
            "long-run truth L_t": "1 iff the best of {X13-style search plan from the true decision-point state, every "
                                  "plan of the full Π_t, the player's own continuation} has mean expected share > 0.5 "
                                  "over 20 fresh cascade/media streams",
            "targets": "possible, knows_win_exists: ∃π∈Π_t wins (row) / L_t; guaranteed: the witness plan wins in "
                       "the true world (row, de re) / L_t; inevitable_defeat: no π∈Π_t wins (row) / ¬L_t. For "
                       "guaranteed, missed-truth (row) uses '∃π∈Π_t wins'",
            "false_assert": "P(target false | verdict asserted) — soundness failure",
            "missed_truth": "P(verdict not asserted | target true) — completeness failure",
            "assert_rate": "P(verdict asserted) over decision points",
            "p_hat": "best_plan.win_fraction: fraction of the sampled worlds in which the best plan (most wins, ties "
                     "by mean share) wins",
            "graded outcomes": "victory = the game's official victory; best_wins = the best plan wins in the true "
                               "world (truth row); long_run = L_t",
            "brier": "mean (p̂ − y)²; murphy = reliability − resolution + uncertainty on the W+1 distinct p̂ values",
            "overconfidence": "OC = mean (p̂ − y)·sign(p̂ − ½) over p̂ ≠ ½; > 0 = over-confident",
            "calibration": "logistic recalibration y ~ a + b·logit(p̂), p̂ clipped to [1/(2W), 1−1/(2W)]; b < 1 = "
                           "over-confident (pooled cells only: 400 cluster-bootstrap resamples for its CI)",
            "ci95": "2.5/97.5 percentiles of a cluster bootstrap over games (2000 resamples); wilson95_unclustered "
                    "treats decision points as independent (too narrow)"},
        "games": {p: {"win_rate": xc.rate([g["victory"] for g in game_rows if g["player"] == p]),
                      "expected": xc.mean_ci([g["expected"] for g in game_rows if g["player"] == p])} for p in PLAYERS},
        "truth_agreement": truth_cmp,
        "world_axis_n1600_piall": world_axis(summary["cells"]),
        **summary, "tests": test, "checks": checks,
        "timing": {"per_cell": timing_cells, "library_timeline_cpu_vs_gpu": timing,
                   "game_seconds_mean": {p: float(np.mean([g["game_seconds"] for g in game_rows if g["player"] == p]))
                                         for p in PLAYERS},
                   "long_run_seconds_mean_per_game": float(np.mean([g["long_run_seconds"] for g in game_rows])),
                   "run_seconds": run_seconds},
        "deviations": DEVIATIONS,
        "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds,
                                       batch_sizes={"timeline_worlds": list(WORLDS), "timeline_world_voters": list(VOTERS),
                                                    "plans_per_evaluate_C": "full Π_t (≤ 21)",
                                                    "long_run_population_C": lr.population,
                                                    "long_run_replicates_K": lr.replicates,
                                                    "long_run_hc_K": lr.hc_replicates, "long_run_fresh_K": lr.fresh,
                                                    "planner_num_worlds": 8, "planner_world_voters": 1600,
                                                    "games_per_batch": 1},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"command": "python experiments/x16_verdicts.py --seeds 50 --first-seed "
                                                         "13000 --device cuda:0 --out reports/v3/x16.json"}),
    }
    xc.save(args.out, report)
    report["figures"] = plot(json.loads(Path(args.out).read_text()))
    xc.save(args.out, report)
    print(f"[x16] done in {seconds / 60:.1f} min; figures {report['figures']}", flush=True)


if __name__ == "__main__":
    main()
