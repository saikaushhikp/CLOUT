"""X13 — ORACLE optimality: open-loop plan search (plan ``docs/v2/plans/implementation_plan.md`` §5 X13).

Command (reproduces reports/v3/x13.json, reports/v3/raw/x13.jsonl and reports/v3/logs/x13.log):
  OMP_NUM_THREADS=4 python experiments/x13_plan_search.py --seeds 100 --first-seed 12000 --device cuda:0 \
      --out reports/v3/x13.json 2>&1 | tee reports/v3/logs/x13.log

Question: how far is the greedy ORACLE planner (``balance.policy_planner``, closed loop on the true state) from
the best open-loop plan? Full information: the true electorate and the ORACLE manifesto b₀ (``oracle_manifesto``).

Plan space: ``hires[t, r] ∈ ℕ`` and ``media[t, r] ∈ {0, 1}`` for T = 5 rounds × 4 regions, total cost ≤ treasury
(cost in integer cents; media prices per region exactly as ``media_cost``; any plan within the total is feasible
round by round).

Search (all on the GPU, ``strategies.worlds.WorldBatch``):
* the true electorate is replicated K times as one disjoint-union batch (K independent cascade/media
  realisations per plan), and C candidate plans are simulated at once with common random numbers;
* cross-entropy method: 256 plans per iteration, elite 10 %, 30 iterations, smoothing α; hires per (t, r) from
  a categorical over 0..⌊treasury/fee⌋, media per (t, r) Bernoulli; over-budget samples are repaired by
  dropping random airings and random workers; every iteration uses fresh streams; the population of
  iteration 0 contains the executed plans of the ORACLE and of the four scripted X0 patterns (warm start), and
  each later population keeps the best plan so far (elitism);
* then a steepest-ascent 1-move hill climb on a fixed set of K_hc streams: move one worker to another (t, r),
  add one, remove one, toggle one airing; stop when no neighbour improves.

Re-evaluation (winner's curse): the final plan, the ORACLE closed-loop policy and the ORACLE's executed plan
replayed open-loop are all played with ``x_batch.run_batch`` (the engine round, bit-exact to ``simulate_world``
on CPU) on the SAME 20 fresh stream sets (``SaltedStreams(seed, salt s)`` on the workers / cascade / media /
election streams, s = 1..20; never used by the search). Reported: the per-seed gap (search − ORACLE) in mean
expected share E over those 20 streams, and the plan features that explain it.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
import x_policies  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_streams import SaltedStreams  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig, SeedStreams  # noqa: E402
from dumbocracy.dynamics.campaign import Allocation, media_cost  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.simulate import RoundContext  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402
from dumbocracy.strategies.worlds import WorldBatch  # noqa: E402

RAW_PATH = xc.RAW / "x13.jsonl"
STREAMS = ("workers", "cascade", "media", "election")
WARM = ("pay_workers", "front_workers", "pay_media", "pay_split")


# ============================================================================== plans
@dataclasses.dataclass(frozen=True)
class PlanPolicy:
    """Open-loop plan as a ``RoundContext`` policy: round t plays ``hires[t]`` / ``media[t]``. A plan is built
    within the treasury, so it never needs clipping; if it ever did, workers are dropped first (recorded by the
    caller through the outcome's allocations)."""

    hires: tuple
    media: tuple

    def __call__(self, ctx: RoundContext) -> Allocation:
        t = ctx.round
        hires = tuple(int(x) for x in self.hires[t])
        media = tuple(bool(x) for x in self.media[t])
        cost_m = media_cost(x_policies.sizes_of(ctx), media, ctx.config.media.base_cost)
        if cost_m > ctx.budget:
            media, cost_m = (False,) * 4, Decimal("0.00")
        fee = ctx.config.workers.worker_cost
        afford = int((ctx.budget - cost_m) // fee) if fee > 0 else 10**9
        if sum(hires) > afford:
            hires = x_policies.split_by_weight(afford, [float(h) for h in hires])
        return Allocation(hires, media)  # type: ignore[arg-type]


def plan_from_allocations(allocations, T: int) -> tuple[np.ndarray, np.ndarray]:
    h = np.zeros((T, 4), dtype=np.int64)
    m = np.zeros((T, 4), dtype=bool)
    for t, a in enumerate(allocations[:T]):
        h[t] = a.hires_by_region
        m[t] = a.media_regions
    return h, m


def policy_of(h: np.ndarray, m: np.ndarray) -> PlanPolicy:
    return PlanPolicy(tuple(tuple(int(x) for x in row) for row in h), tuple(tuple(bool(x) for x in row) for row in m))


class Pricing:
    """Integer-cent prices for one electorate (exactly ``media_cost`` per region; it is additive over regions)."""

    def __init__(self, sizes: list[int], cfg: GameConfig):
        self.sizes = sizes
        self.region = np.array([int(media_cost(sizes, tuple(i == r for i in range(4)), cfg.media.base_cost) * 100)
                                for r in range(4)], dtype=np.int64)
        self.worker = int(cfg.workers.worker_cost * 100)
        self.budget = int(cfg.budget * 100)
        self.max_workers = self.budget // self.worker if self.worker > 0 else 0

    def cost(self, h: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Cents for plans ``h`` ``[..., T, 4]``, ``m`` ``[..., T, 4]``."""
        return h.sum((-2, -1)) * self.worker + (m * self.region).sum((-2, -1))

    def repair(self, h: np.ndarray, m: np.ndarray, rng: np.random.Generator) -> None:
        """In place: drop random airings while media alone exceeds the treasury, then keep a uniformly random
        subset of the worker units that the rest of the money affords."""
        for p in np.nonzero(self.cost(h, m) > self.budget)[0]:
            media_c = int((m[p] * self.region).sum())
            while media_c > self.budget:
                on = np.argwhere(m[p])
                t, r = on[rng.integers(len(on))]
                m[p, t, r] = False
                media_c -= int(self.region[r])
            allowed = (self.budget - media_c) // self.worker
            total = int(h[p].sum())
            if total > allowed:
                units = np.repeat(np.arange(h[p].size), h[p].reshape(-1))
                keep = rng.choice(total, size=int(allowed), replace=False)
                h[p] = np.bincount(units[keep], minlength=h[p].size).reshape(h[p].shape)


def plan_features(h: np.ndarray, m: np.ndarray, pricing: Pricing) -> dict:
    """Timing and focus features of one plan (per X13 output: "timing, focus")."""
    T = h.shape[0]
    workers_t = h.sum(1).astype(float)
    total_w = float(workers_t.sum())
    media_c = (m * pricing.region).sum(1).astype(float)
    worker_c = workers_t * pricing.worker
    spend_t = media_c + worker_c
    spend = float(spend_t.sum())
    by_region = h.sum(0).astype(float)
    hhi = float(((by_region / total_w) ** 2).sum()) if total_w > 0 else float("nan")
    rounds = np.arange(T, dtype=float)
    return {
        "workers": total_w, "airings": int(m.sum()), "spend": spend / 100.0,
        "media_spend_share": float(media_c.sum() / spend) if spend > 0 else 0.0,
        "workers_round1_share": float(workers_t[0] / total_w) if total_w > 0 else 0.0,
        "worker_round_centroid": float((workers_t * rounds).sum() / total_w) if total_w > 0 else float("nan"),
        "spend_round_centroid": float((spend_t * rounds).sum() / spend) if spend > 0 else float("nan"),
        "worker_region_hhi": hhi, "worker_regions_used": int((by_region > 0).sum()),
        "media_regions_used": int(m.any(0).sum()), "unspent": (pricing.budget - spend) / 100.0,
    }


# ============================================================================== batched evaluator
class Evaluator:
    """C plans × K replicates of the true electorate, one ``WorldBatch`` (common random numbers across plans)."""

    def __init__(self, electorate, cfg: GameConfig, manifesto: torch.Tensor, replicates: int, chunk: int):
        self.cfg = cfg
        self.batch = WorldBatch([electorate] * replicates, cfg)
        self.b = manifesto.to(device=self.batch.device, dtype=self.batch.preferences.dtype)
        self.K = replicates
        self.chunk = chunk
        self.T = cfg.rounds
        self.calls = 0
        self.plans_evaluated = 0

    @torch.no_grad()
    def _run(self, h: np.ndarray, m: np.ndarray, seed: int) -> torch.Tensor:
        batch, dev = self.batch, self.batch.device
        C = h.shape[0]
        v = batch.preferences.unsqueeze(0).expand(C, -1, -1).contiguous()
        airings = torch.zeros((C, 4), dtype=torch.int64, device=dev)
        workers = None
        streams = SeedStreams(int(seed))
        for t in range(self.T):
            plans = [Allocation(tuple(int(x) for x in h[c, t]), tuple(bool(x) for x in m[c, t])) for c in range(C)]
            gen = streams.generator("strategy", "plan", t, device=dev)
            v, workers = batch.step(v, plans, airings, self.b, gen, workers)
            airings = airings + torch.as_tensor(m[:, t], device=dev).to(torch.int64)
        return batch.shares(v, self.b)                                                 # [C, K]

    def __call__(self, h: np.ndarray, m: np.ndarray, seed: int) -> np.ndarray:
        """Mean expected share over the K replicates per plan, ``[C]`` (chunks with OOM back-off). All chunks
        use the same streams ``seed`` (common random numbers over the whole population)."""
        out = []
        i = 0
        while i < h.shape[0]:
            j = min(h.shape[0], i + self.chunk)
            try:
                out.append(self._run(h[i:j], m[i:j], seed).mean(1).double().cpu().numpy())
            except torch.cuda.OutOfMemoryError:
                if self.chunk <= 1:
                    raise
                torch.cuda.empty_cache()
                self.chunk = max(1, self.chunk // 2)
                print(f"[x13] CUDA out of memory: chunk halved to {self.chunk}", flush=True)
                continue
            i = j
        self.calls += 1
        self.plans_evaluated += int(h.shape[0])
        return np.concatenate(out)


# ============================================================================== search
@dataclasses.dataclass
class SearchSettings:
    population: int = 256
    iterations: int = 30
    elite_frac: float = 0.10
    smoothing: float = 0.7
    replicates: int = 4          # K streams per plan in the CEM
    hc_replicates: int = 8       # K_hc streams for the hill climb (fixed)
    hc_max_passes: int = 40
    init_scale: float = 1.0      # initial hires ∝ exp(−k / init_scale)
    init_media: float = 0.25
    chunk: int = 256


def neighbours(h: np.ndarray, m: np.ndarray, pricing: Pricing) -> tuple[np.ndarray, np.ndarray, list]:
    """All affordable 1-move neighbours of one plan: move one worker (t,r)→(t',r'), add one, remove one,
    toggle one airing."""
    T = h.shape[0]
    cells = [(t, r) for t in range(T) for r in range(4)]
    H, M, moves = [], [], []
    for (t, r) in cells:
        if h[t, r] > 0:
            for (t2, r2) in cells:
                if (t2, r2) != (t, r):
                    x = h.copy()
                    x[t, r] -= 1
                    x[t2, r2] += 1
                    H.append(x), M.append(m), moves.append(("move", t, r, t2, r2))
            x = h.copy()
            x[t, r] -= 1
            H.append(x), M.append(m), moves.append(("remove", t, r))
        x = h.copy()
        x[t, r] += 1
        H.append(x), M.append(m), moves.append(("add", t, r))
        y = m.copy()
        y[t, r] = not y[t, r]
        H.append(h), M.append(y), moves.append(("toggle", t, r))
    H, M = np.stack(H), np.stack(M)
    ok = pricing.cost(H, M) <= pricing.budget
    return H[ok], M[ok], [mv for mv, k in zip(moves, ok) if k]


def search_plan(electorate, cfg: GameConfig, manifesto: torch.Tensor, seed: int, warm: list[tuple[np.ndarray, np.ndarray]],
                settings: SearchSettings, *, log=None) -> dict:
    """CEM + 1-move hill climb on the true world. Returns the best plan and the search trace."""
    t0 = time.perf_counter()
    sizes = [int(x) for x in electorate.sizes.tolist()]
    pricing = Pricing(sizes, cfg)
    T, K = cfg.rounds, pricing.max_workers + 1
    rng = np.random.default_rng([int(seed), 13])
    ev = Evaluator(electorate, cfg, manifesto, settings.replicates, settings.chunk)
    q = np.exp(-np.arange(K) / settings.init_scale)
    q = np.broadcast_to(q / q.sum(), (T, 4, K)).copy()                              # hires categorical
    p = np.full((T, 4), settings.init_media)                                        # media Bernoulli
    n_elite = max(2, int(round(settings.elite_frac * settings.population)))
    best_h, best_m = None, None
    trace = []
    stream_base = SeedStreams(int(seed)).derive("strategy", "x13", "cem")
    for it in range(settings.iterations):
        P = settings.population
        u = rng.random((P, T, 4, 1))
        h = (u > np.cumsum(q, -1)[None]).sum(-1).astype(np.int64).clip(0, K - 1)    # inverse-CDF categorical
        m = rng.random((P, T, 4)) < p[None]
        fixed = list(warm) if it == 0 else []
        if best_h is not None:
            fixed.append((best_h, best_m))
        for i, (wh, wm) in enumerate(fixed[:P]):
            h[i], m[i] = wh, wm
        pricing.repair(h, m, rng)
        scores = ev(h, m, stream_base + it)
        order = np.argsort(-scores)
        elite = order[:n_elite]
        # incumbent = this iteration's top plan (the previous incumbent competes, re-scored on the new streams)
        best_h, best_m = h[order[0]].copy(), m[order[0]].copy()
        counts = np.zeros((T, 4, K))
        for e in elite:
            counts[np.arange(T)[:, None], np.arange(4)[None, :], h[e]] += 1
        q = settings.smoothing * counts / n_elite + (1 - settings.smoothing) * q
        p = settings.smoothing * m[elite].mean(0) + (1 - settings.smoothing) * p
        trace.append({"it": it, "best": float(scores[order[0]]), "elite_mean": float(scores[elite].mean()),
                      "pop_mean": float(scores.mean()), "elite_workers": float(h[elite].sum((1, 2)).mean()),
                      "elite_airings": float(m[elite].sum((1, 2)).mean())})
    cem_seconds = time.perf_counter() - t0
    # final CEM pick: last elite + warm plans re-scored on the fixed hill-climb streams
    ev_hc = Evaluator(electorate, cfg, manifesto, settings.hc_replicates, settings.chunk)
    hc_seed = SeedStreams(int(seed)).derive("strategy", "x13", "hc")
    cand_h = np.concatenate([h[elite], np.stack([w[0] for w in warm])])
    cand_m = np.concatenate([m[elite], np.stack([w[1] for w in warm])])
    cand_scores = ev_hc(cand_h, cand_m, hc_seed)
    k = int(np.argmax(cand_scores))
    cur_h, cur_m, cur = cand_h[k].copy(), cand_m[k].copy(), float(cand_scores[k])
    cem_pick = {"from": "elite" if k < len(elite) else f"warm:{k - len(elite)}", "score_hc_streams": cur}
    passes, moves = 0, []
    for passes in range(1, settings.hc_max_passes + 1):
        H, M, mv = neighbours(cur_h, cur_m, pricing)
        s = ev_hc(H, M, hc_seed)
        j = int(np.argmax(s))
        if s[j] <= cur + 1e-7:
            break
        cur_h, cur_m, cur = H[j].copy(), M[j].copy(), float(s[j])
        moves.append(list(mv[j]))
    out = {"hires": cur_h, "media": cur_m, "score_hc_streams": cur, "cem_pick": cem_pick, "hc_passes": passes,
           "hc_moves": moves, "trace": trace, "cem_seconds": cem_seconds, "seconds": time.perf_counter() - t0,
           "evaluations": ev.plans_evaluated + ev_hc.plans_evaluated, "chunk": min(ev.chunk, ev_hc.chunk),
           "pricing_cents": {"region": pricing.region.tolist(), "worker": pricing.worker, "budget": pricing.budget}}
    if log:
        log(f"search seed {seed}: CEM best {trace[-1]['best']:.4f}, hc {cur:.4f} after {passes} passes, "
            f"{out['seconds']:.1f}s")
    return out


# ============================================================================== evaluation on fresh streams
def fresh_streams(seed: int, count: int, offset: int = 1) -> list[SaltedStreams]:
    return [SaltedStreams(int(seed), {s: offset + i for s in STREAMS}) for i in range(count)]


def evaluate_policies(electorate, cfg: GameConfig, manifesto: torch.Tensor, policies: dict, streams: list) -> dict:
    """Each policy on the same ``streams`` (B = len(streams) copies of the electorate, one ``run_batch``)."""
    B = len(streams)
    ms = manifesto.reshape(1, -1).expand(B, -1)
    out = {}
    for name, pol in policies.items():
        outs = run_batch([electorate] * B, cfg, ms, pol, streams=streams)
        out[name] = {"expected": [o.result.expected_share for o in outs], "victory": [bool(o.result.victory) for o in outs],
                     "vote_share": [o.result.vote_share for o in outs], "spent": [float(o.spent) for o in outs],
                     "allocations": [[[list(a.hires_by_region), list(a.media_regions)] for a in o.allocations]
                                     for o in outs],
                     "converted": [sum(o.converted) for o in outs]}
    return out


def warm_plans(electorate, cfg: GameConfig, manifesto: torch.Tensor, seed: int) -> dict:
    """Executed plans of the ORACLE and the X0 scripted patterns on the canonical streams of ``seed``."""
    names = ("planner",) + WARM
    outs = {n: run_batch([electorate], cfg, manifesto.reshape(1, -1), x_policies.GATE_POLICIES[n], seeds=[seed])[0]
            for n in names}
    return {n: plan_from_allocations(o.allocations, cfg.rounds) + (o.result.expected_share,) for n, o in outs.items()}


# ============================================================================== one seed
def seed_task(task: tuple) -> dict:
    seed, n, device, settings_dict, fresh = task
    torch.set_num_threads(4)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.zeros(1, device=dev)
        torch.cuda.reset_peak_memory_stats(dev)
    settings = SearchSettings(**settings_dict)
    start = time.perf_counter()
    cfg = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    e = generate_electorate(cfg, device=dev)
    b0 = oracle_manifesto(e, cfg)
    return run_seed(seed, e, cfg, b0, settings, fresh, start, dev)


def run_seed(seed: int, e, cfg: GameConfig, b0: torch.Tensor, settings: SearchSettings, fresh: int, start: float,
             dev: torch.device, extra: dict | None = None) -> dict:
    sizes = [int(x) for x in e.sizes.tolist()]
    pricing = Pricing(sizes, cfg)
    warm = warm_plans(e, cfg, b0, seed)
    found = search_plan(e, cfg, b0, seed, [(h, m) for h, m, _ in warm.values()], settings)
    streams = fresh_streams(seed, fresh)
    oracle_h, oracle_m, oracle_canon = warm["planner"]
    evals = evaluate_policies(e, cfg, b0, {"search": policy_of(found["hires"], found["media"]),
                                           "oracle": x_policies.policy_planner,
                                           "oracle_plan_open_loop": policy_of(oracle_h, oracle_m)}, streams)
    oracle_feats = [plan_features(*plan_from_allocations([Allocation(tuple(a[0]), tuple(a[1])) for a in alloc], cfg.rounds),
                                  pricing) for alloc in evals["oracle"]["allocations"]]
    mean_feats = {k: float(np.nanmean([f[k] for f in oracle_feats])) for k in oracle_feats[0]}
    row = {"exp": "X13", "seed": seed, "sizes": sizes, "manifesto": [round(float(x), 6) for x in b0.tolist()],
           "search_plan": {"hires": found["hires"].tolist(), "media": found["media"].astype(int).tolist()},
           "search_features": plan_features(found["hires"], found["media"], pricing),
           "oracle_plan_canonical": {"hires": oracle_h.tolist(), "media": oracle_m.astype(int).tolist(),
                                     "expected": oracle_canon},
           "oracle_features_mean20": mean_feats,
           "warm_canonical_expected": {k: v[2] for k, v in warm.items()},
           "search_score_hc_streams": found["score_hc_streams"], "cem_pick": found["cem_pick"],
           "hc_passes": found["hc_passes"], "hc_moves": found["hc_moves"], "cem_trace": found["trace"],
           "evaluations": found["evaluations"], "search_seconds": found["seconds"], "cem_seconds": found["cem_seconds"],
           "fresh": {k: {kk: vv for kk, vv in v.items() if kk != "allocations"} for k, v in evals.items()},
           "oracle_allocations_fresh": evals["oracle"]["allocations"], "chunk": found["chunk"],
           "pricing_cents": found["pricing_cents"]}
    for k in ("search", "oracle", "oracle_plan_open_loop"):
        row[f"E20_{k}"] = float(np.mean(evals[k]["expected"]))
        row[f"win20_{k}"] = float(np.mean(evals[k]["victory"]))
    row["gap_search_minus_oracle"] = row["E20_search"] - row["E20_oracle"]
    row["gap_search_minus_oracle_plan"] = row["E20_search"] - row["E20_oracle_plan_open_loop"]
    row["seconds"] = time.perf_counter() - start
    row["peak_bytes"] = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
    if extra:
        row.update(extra)
    return row


# ============================================================================== parity
def parity(seeds: list[int], device: torch.device, n: int) -> dict:
    """CPU↔GPU on 3 seeds with a reduced search (64 plans × 6 iterations, no hill climb passes beyond 3):
    the re-evaluated search E, the ORACLE E (both over 5 fresh streams) and the WorldBatch score of the ORACLE's
    executed plan. Statistical tolerance 0.01 E (different random numbers on CPU and GPU)."""
    small = dataclasses.asdict(SearchSettings(population=64, iterations=6, hc_max_passes=3, replicates=2,
                                              hc_replicates=2, chunk=64))

    def fn(seed, dev):
        row = seed_task((seed, n, str(dev), small, 5))
        return {"E_search": row["E20_search"], "E_oracle": row["E20_oracle"], "gap": row["gap_search_minus_oracle"]}

    return xc.parity_check(fn, seeds, device=device, abs_tol=0.01)


# ============================================================================== statistics
FEATURES = ("workers", "airings", "spend", "media_spend_share", "workers_round1_share", "worker_round_centroid",
            "spend_round_centroid", "worker_region_hhi", "worker_regions_used", "media_regions_used")


def summarise(rows: list[dict]) -> dict:
    from scipy.stats import wilcoxon
    gaps = [r["gap_search_minus_oracle"] for r in rows]
    gaps_plan = [r["gap_search_minus_oracle_plan"] for r in rows]
    q = np.quantile(gaps, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).tolist()
    out = {
        "n_seeds": len(rows),
        "E20": {k: xc.mean_ci([r[f"E20_{k}"] for r in rows]) for k in ("search", "oracle", "oracle_plan_open_loop")},
        "win20": {k: xc.mean_ci([r[f"win20_{k}"] for r in rows]) for k in ("search", "oracle", "oracle_plan_open_loop")},
        "gap_search_minus_oracle": {**xc.paired([r["E20_search"] for r in rows], [r["E20_oracle"] for r in rows]),
                                    "quantiles_0_5_25_50_75_95_100": q,
                                    "frac_gt_0.01": float(np.mean([g > 0.01 for g in gaps])),
                                    "frac_gt_0.02": float(np.mean([g > 0.02 for g in gaps])),
                                    "frac_lt_0": float(np.mean([g < 0 for g in gaps])),
                                    "wilcoxon_greater_p": float(wilcoxon(gaps, alternative="greater").pvalue)},
        "gap_search_minus_oracle_plan_open_loop": xc.paired([r["E20_search"] for r in rows],
                                                            [r["E20_oracle_plan_open_loop"] for r in rows]),
        "gap_oracle_closed_minus_open_loop": xc.paired([r["E20_oracle"] for r in rows],
                                                       [r["E20_oracle_plan_open_loop"] for r in rows]),
        "win20_search_minus_oracle": xc.paired([r["win20_search"] for r in rows], [r["win20_oracle"] for r in rows]),
        "selection_bias": xc.paired([r["search_score_hc_streams"] for r in rows], [r["E20_search"] for r in rows]),
        "search_seconds": xc.mean_ci([r["search_seconds"] for r in rows]),
        "hc_passes": xc.mean_ci([r["hc_passes"] for r in rows]),
        "cem_pick_from_warm": sum(1 for r in rows if r["cem_pick"]["from"].startswith("warm")),
    }
    # features: search plan vs ORACLE (mean over its 20 executed plans), and which differences explain the gap
    feats, fam = {}, []
    for f in FEATURES:
        s = [r["search_features"][f] for r in rows]
        o = [r["oracle_features_mean20"][f] for r in rows]
        ok = [i for i in range(len(rows)) if not (math.isnan(s[i]) or math.isnan(o[i]))]
        diff = [s[i] - o[i] for i in ok]
        feats[f] = {"search": xc.mean_ci([s[i] for i in ok]), "oracle": xc.mean_ci([o[i] for i in ok]),
                    "search_minus_oracle": xc.paired([s[i] for i in ok], [o[i] for i in ok])}
        if len(set(diff)) > 1:
            sp = xc.spearman(diff, [gaps[i] for i in ok])
            fam.append({"feature": f, "n": len(ok), "spearman_rho_with_gap": sp["rho"], "p": sp["p"]})
    for x, p in zip(fam, xc.holm([x["p"] for x in fam])):
        x["holm_p"] = p
    out["features"] = feats
    out["feature_gap_correlations"] = {"family": "Spearman ρ of (search − ORACLE) feature difference with the "
                                                 "per-seed gap; Holm within the family", "tests": fam}
    # standardised OLS of the gap on the feature differences (descriptive)
    cols = [f for f in ("workers", "media_spend_share", "workers_round1_share", "worker_region_hhi", "spend")
            if all(not math.isnan(r["search_features"][f]) and not math.isnan(r["oracle_features_mean20"][f]) for r in rows)]
    X = np.array([[r["search_features"][f] - r["oracle_features_mean20"][f] for f in cols] for r in rows], dtype=float)
    y = np.array(gaps)
    sd = X.std(0)
    keep = sd > 0
    Xs = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    A = np.column_stack([np.ones(len(y)), Xs])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if y.std() > 0 else float("nan")
    out["gap_ols"] = {"features": [c for c, k in zip(cols, keep) if k], "intercept": float(coef[0]),
                      "std_coefficients": [float(c) for c in coef[1:]], "r2": r2,
                      "note": "gap ~ standardised (search − ORACLE) feature differences; descriptive only"}
    # tail: seeds with gap > 0.01 vs the rest
    tail = [r for r in rows if r["gap_search_minus_oracle"] > 0.01]
    rest = [r for r in rows if r["gap_search_minus_oracle"] <= 0.01]
    out["tail_gt_0.01"] = {"n": len(tail), "seeds": [r["seed"] for r in tail],
                           "search_features_mean": {f: float(np.nanmean([r["search_features"][f] for r in tail]))
                                                    for f in FEATURES} if tail else {},
                           "rest_search_features_mean": {f: float(np.nanmean([r["search_features"][f] for r in rest]))
                                                         for f in FEATURES} if rest else {}}
    out["gap_by_seed"] = {str(r["seed"]): r["gap_search_minus_oracle"] for r in rows}
    out["E20_search_by_seed"] = {str(r["seed"]): r["E20_search"] for r in rows}
    out["E20_oracle_by_seed"] = {str(r["seed"]): r["E20_oracle"] for r in rows}
    return out


# ============================================================================== main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=xc.SEED_BLOCKS["X13"])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=2, help="processes on the GPU (≤ 2)")
    parser.add_argument("--fresh", type=int, default=20, help="fresh stream sets for the re-evaluation")
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--hc-replicates", type=int, default=8)
    parser.add_argument("--budget-hours", type=float, default=2.5)
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--out", default=str(xc.REPORTS / "x13.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    free0 = xc.free_bytes(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    settings = SearchSettings(population=args.population, iterations=args.iterations, replicates=args.replicates,
                              hc_replicates=args.hc_replicates)
    sd = dataclasses.asdict(settings)
    deviations = [f"Device {device} instead of the plan default cuda:2 (cuda:2 is used by the Phase 2 agent)."]
    start = time.perf_counter()
    with xc.GpuMonitor(device) as monitor:
        print(f"[x13] device {device}, seeds {seeds[0]}..{seeds[-1]}, settings {sd}", flush=True)
        pilot = seed_task((seeds[0], args.n, str(device), sd, args.fresh))
        est = pilot["seconds"] * (len(seeds) - 1) / max(1, args.workers) * 1.15
        print(f"[x13] pilot seed {seeds[0]}: {pilot['seconds']:.1f}s (search {pilot['search_seconds']:.1f}s), gap "
              f"{pilot['gap_search_minus_oracle']:+.4f}; estimated remaining {est / 60:.1f} min with {args.workers} "
              f"processes", flush=True)
        if args.pilot_only:
            return
        if est > args.budget_hours * 3600:
            raise SystemExit("estimate exceeds the time budget; reduce --population/--iterations and record it")
        rest = xc.run_tasks(seed_task, [(s, args.n, str(device), sd, args.fresh) for s in seeds[1:]], args.workers,
                            label="[x13]")
        rows = [pilot] + rest
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text("")
        for r in rows:
            xc.write_row(RAW_PATH, r)
        par = parity(seeds[:args.parity_seeds], device, args.n)
        print(f"[x13] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    summary = summarise(rows)
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    report = {
        "experiment": "X13", "question": "ORACLE optimality: open-loop plan search (upper bound for X6)",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget),
        "prices": {"worker": str(base.workers.worker_cost), "media_base": str(base.media.base_cost)},
        "manifesto": "oracle_manifesto (anticipated true electorate), same b₀ for search and ORACLE",
        "search": {**sd, "method": "cross-entropy method (categorical hires per (t, r), Bernoulli media per (t, r)), "
                   "fresh streams per iteration, warm start with the ORACLE and X0-pattern executed plans, elitism; "
                   "then steepest-ascent 1-move hill climb (move / add / remove a worker, toggle an airing) on fixed "
                   "streams", "simulator": "strategies.worlds.WorldBatch (true electorate × K replicates, CRN)"},
        "evaluation": {"fresh_streams": args.fresh, "streams_salted": list(STREAMS),
                       "runner": "x_batch.run_batch (engine round; bit-exact to simulate_world on CPU)",
                       "arms": {"search": "final open-loop plan", "oracle": "balance.policy_planner (closed loop)",
                                "oracle_plan_open_loop": "the ORACLE's executed plan on the canonical streams, "
                                                         "replayed open loop"}},
        **summary,
        "pilot": {"seed": seeds[0], "seconds": pilot["seconds"], "estimated_remaining_seconds": est},
        "deviations": deviations, "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, workers=args.workers,
                                       batch_sizes={"plans_per_call_C": max(r["chunk"] for r in rows),
                                                    "replicates_K": settings.replicates,
                                                    "hill_climb_K": settings.hc_replicates,
                                                    "fresh_eval_B": args.fresh},
                                       free_at_start=free0, monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"child_peak_bytes_max": max((r["peak_bytes"] or 0) for r in rows)}),
    }
    xc.save(args.out, report)
    print(f"[x13] mean gap {summary['gap_search_minus_oracle']['mean']:+.4f} "
          f"{summary['gap_search_minus_oracle']['ci95']}, done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
