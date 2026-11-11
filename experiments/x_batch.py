"""Batched TRUE-world game runner for the v3 experiments (plan §1.1 "GPU first", Phase 0).

``simulate_world`` plays one world at a time, and ``WorldBatch`` applies the SAME allocation to every world
(it was built for the planner's synthetic worlds). The experiments need B true electorates (same N,
different seeds), each with its OWN manifesto and its OWN per-round allocation from a ``RoundContext``
policy that sees that world's true state. :func:`run_batch` does exactly that:

* the B electorates are joined into one disjoint-union graph (``[B·N]`` voters), with per-voter manifesto
  rows, and per-world treasury, airings and worker state;
* the policy decisions are made per world in Python (cheap); every tensor operation of the P8 round —
  cascade counts and hop, media intensities, social step, shares — is ONE batched call over all B worlds;
* every random draw comes from that world's OWN seed streams, keyed exactly as in ``simulate_world``
  (``(stream, round, player)``, then ``("election", player)``), and is consumed in the same amounts.

Round order (identical to ``simulate.simulate_world`` + ``dynamics.campaign.campaign_round`` +
``simulate.resolve_campaign_round``): policy → hires (``workers`` stream) → one IC hop with chance from v at
the start of the round (``cascade`` stream) → media intensity (``media`` stream) → pay → new workers and
converts take v = b → social step with zealots frozen and workers pinned → expected share.

Exactness (tested in ``tests/unit/test_x_common.py``). With ``aggregation="exact"`` the neighbour mean
uses the same ``index_select`` + ``index_add_`` as ``dynamics.social.aggregate_degroot`` (CSR order per
row), and every other op is elementwise or a per-row reduction, so on the CPU a batch of B worlds
reproduces B separate ``simulate_world`` runs bit for bit. On CUDA, ``index_add_`` uses float atomics
(``simulate_world`` itself is then not bit-reproducible run to run); there ``aggregation="csr"`` (the
default on CUDA) sums each row with one deterministic sparse mat-mul, as ``WorldBatch`` does. The two
summation orders differ only in the last float bits; the agreement with ``simulate_world`` on the GPU is
checked statistically (see ``reports/v3/gpu_benchmarks.json`` → ``parity``).

Overrides for later experiments, without library changes:
* cascade off: ``config.workers.attempt_cap = 0`` (X2 arm B);
* conversion rule: ``config.workers.conversion_rule`` (X23);
* worker pick rule: ``pick_fn`` (X4/X18), same signature as ``dynamics.campaign.pick_workers`` plus the world;
* salted streams: ``streams=[SaltedStreams(...), ...]`` (X20);
* per-round observation: ``observer(t, info)`` receives the union tensors after each round.
"""
from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any

import torch

from dumbocracy.config import GameConfig, SeedStreams
from dumbocracy.dynamics.campaign import (
    Allocation, WorkerState, _exposed_intensity, combine_intensities, eligible_workers, media_cost,
    spreader_edge_counts, worker_cost,
)
from dumbocracy.dynamics.social import aggregate
from dumbocracy.electorate import Electorate
from dumbocracy.sectors import NUM_REGIONS
from dumbocracy.simulate import Policy, RoundContext, WorldOutcome
from dumbocracy.strategies.worlds import _union_graph
from dumbocracy.voting import run_election, vote_probability

__all__ = ["BatchGame", "PickFn", "run_batch", "conversion_chance_rows"]

PickFn = Callable[..., tuple[torch.Tensor, ...]]


def conversion_chance_rows(campaign: torch.Tensor, v: torch.Tensor, b_rows: torch.Tensor, sensitivity: float,
                           threshold: float, rule: str = "gamma_p") -> torch.Tensor:
    """``dynamics.campaign.conversion_chance`` with one manifesto row per voter (``b_rows`` ``[M, 5]``)."""
    sim = 1 - 0.5 * (v - b_rows).abs().sum(-1)      # == voting.similarity, op for op
    prob = vote_probability(sim, sensitivity, threshold)
    if rule == "gamma_p":
        return (campaign * prob).clamp(0.0, 1.0)
    if rule == "gamma":
        return (campaign * torch.ones_like(prob)).clamp(0.0, 1.0)
    if rule == "gamma_p2":
        return (campaign * (prob * prob)).clamp(0.0, 1.0)
    raise ValueError(f"unknown conversion rule {rule!r}")


def _csr(indptr: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        return torch.sparse_csr_tensor(indptr, indices, values, (n, n))


class BatchGame:
    """B true electorates of equal size as one disjoint union (static tensors only)."""

    def __init__(self, electorates: Sequence[Electorate], config: GameConfig, manifestos: torch.Tensor | Sequence,
                 *, aggregation: str = "auto"):
        if not electorates:
            raise ValueError("need at least one electorate")
        self.electorates = list(electorates)
        self.config = config
        self.B = len(electorates)
        self.n = electorates[0].num_voters
        if any(e.num_voters != self.n for e in electorates):
            raise ValueError("all electorates must have the same number of voters")
        self.device = electorates[0].device
        if any(e.device != self.device for e in electorates):
            raise ValueError("all electorates must live on the same device")
        self.dtype = electorates[0].preferences.dtype
        if aggregation == "auto":
            aggregation = "csr" if self.device.type == "cuda" else "exact"
        if aggregation not in ("exact", "csr"):
            raise ValueError("aggregation must be 'auto', 'exact' or 'csr'")
        self.aggregation = aggregation
        dev, n, B = self.device, self.n, self.B
        self.graph = electorates[0].graph if B == 1 else _union_graph([e.graph for e in electorates], n)
        self.rows = self.graph.rows()
        self.degree = self.graph.degree
        self.world_of = torch.arange(B * n, device=dev) // n
        self.region_of = torch.cat([e.region_of for e in electorates])
        sus = [e.susceptibility for e in electorates]
        self.social = torch.cat([s.social for s in sus])
        self.media = torch.cat([s.media for s in sus])
        self.campaign = torch.cat([s.campaign for s in sus])
        self.zealot = torch.cat([s.zealot for s in sus]).to(torch.bool)
        self.social_weight = (config.dynamics.social_step * self.social).to(self.dtype)
        self.v0 = torch.cat([e.preferences for e in electorates])
        if isinstance(manifestos, torch.Tensor):
            m = manifestos.to(device=dev, dtype=self.dtype).reshape(B, -1)
        else:
            m = torch.stack([torch.as_tensor(x).to(device=dev, dtype=self.dtype) for x in manifestos])
        if m.shape != (B, 5):
            raise ValueError(f"manifestos must have shape [B={B}, 5], got {list(m.shape)}")
        self.manifestos = m
        self.b_rows = m[self.world_of]                                                   # [Bn, 5]
        self.sizes = [[int(x) for x in e.sizes.tolist()] for e in electorates]
        self._adj = None

    # ------------------------------------------------------------------ batched pieces
    def aggregate(self, v: torch.Tensor) -> torch.Tensor:
        rule = self.config.dynamics.social_rule
        if rule != "degroot":
            return aggregate(v, self.graph, rule, self.config.dynamics.wmsr_trim)
        if self.aggregation == "exact":   # == aggregate_degroot (cached rows)
            total = torch.zeros_like(v).index_add_(-2, self.rows, v.index_select(-2, self.graph.indices))
        else:
            if self._adj is None or self._adj.dtype != v.dtype:
                ones = torch.ones(self.graph.indices.numel(), dtype=v.dtype, device=self.device)
                self._adj = _csr(self.graph.indptr, self.graph.indices, ones, self.graph.num_nodes)
            total = self._adj @ v
        degree = self.degree.to(v.dtype).unsqueeze(-1)
        mean = total / degree.clamp_min(1)
        return torch.where((self.degree > 0).unsqueeze(-1), mean, v)

    def step(self, v: torch.Tensor, agg: torch.Tensor, intensity: torch.Tensor, pinned: torch.Tensor) -> torch.Tensor:
        """``update.step_preferences`` for one player with a per-voter manifesto row (op for op)."""
        a = self.social_weight
        c = intensity.to(v.dtype)
        over = a + c > 1
        scale = torch.where(over, (1 - a) / torch.where(over, c, torch.ones_like(c)), torch.ones_like(c))
        c = c * scale
        pull = c.unsqueeze(-1) * self.b_rows
        new = (1 - a - c).unsqueeze(-1) * v + a.unsqueeze(-1) * agg + pull
        new = new.clamp_min(0)
        new = new / new.sum(-1, keepdim=True).clamp_min(torch.finfo(v.dtype).tiny)
        new = torch.where(pinned.unsqueeze(-1), self.b_rows, new)
        return torch.where(self.zealot.unsqueeze(-1), v, new)

    def world_slice(self, b: int) -> slice:
        return slice(b * self.n, (b + 1) * self.n)


def _as_list(x: Any, B: int, name: str) -> list:
    if isinstance(x, (list, tuple)):
        if len(x) != B:
            raise ValueError(f"{name} must have one entry per world ({B})")
        return list(x)
    return [x] * B


@torch.no_grad()
def run_batch(electorates: Sequence[Electorate], config: GameConfig, manifestos: torch.Tensor | Sequence,
              policies: Policy | Sequence[Policy], *, seeds: Sequence[int] | None = None,
              streams: Sequence[Any] | None = None, player: str = "p0", aggregation: str = "auto",
              pick_fn: PickFn | None = None, observer: Callable[[int, dict], None] | None = None,
              keep_final: bool = False) -> list[WorldOutcome]:
    """Play B true-world games at once; returns one ``WorldOutcome`` per world (as ``simulate_world``).

    ``seeds[b]`` (root seed of world b, i.e. ``config.seed`` of that world's ``simulate_world`` call) or
    ``streams[b]`` (any object with ``generator(stream, *keys, device=)``, e.g. ``SeedStreams`` or
    ``x_streams.SaltedStreams``) gives world b its own random streams. ``policies`` is one policy for all worlds
    or one per world. ``pick_fn(region_of, eligible, hires_by_region, generator, *, world, electorate)``
    replaces ``pick_workers`` (default: the library rule R22). Runs under ``torch.no_grad`` (not
    ``inference_mode``, so the returned tensors can still feed an autograd manifesto optimiser).
    """
    game = BatchGame(electorates, config, manifestos, aggregation=aggregation)
    B, n, dev, dtype = game.B, game.n, game.device, game.dtype
    if streams is None:
        if seeds is None:
            raise ValueError("give seeds or streams")
        streams = [SeedStreams(int(s)) for s in _as_list(list(seeds), B, "seeds")]
    streams = _as_list(list(streams), B, "streams")
    policies = _as_list(policies, B, "policies")
    # each world's RoundContext carries the config of its own simulate_world call (config.seed = its root seed)
    ctx_configs = [dataclasses.replace(config, seed=int(st.root)) if isinstance(getattr(st, "root", None), int)
                   else config for st in streams]
    vcfg, wcfg, mcfg = config.voting, config.workers, config.media

    v = game.v0.clone()
    budgets = [config.budget] * B
    airings = [[0] * NUM_REGIONS for _ in range(B)]
    pinned = torch.zeros(B * n, dtype=torch.bool, device=dev)
    spreaders_prev = torch.zeros_like(pinned)
    attempts = torch.zeros(B * n, dtype=torch.int64, device=dev)
    ids = [tuple(torch.zeros(0, dtype=torch.int64, device=dev) for _ in range(NUM_REGIONS)) for _ in range(B)]
    allocations: list[list[Allocation]] = [[] for _ in range(B)]

    def shares(v_: torch.Tensor) -> list[torch.Tensor]:
        sim = 1 - 0.5 * (v_ - game.b_rows).abs().sum(-1)
        prob = vote_probability(sim, vcfg.sensitivity, vcfg.threshold)
        return [prob[game.world_slice(b)].mean() for b in range(B)]   # == current_share per world

    trajectory = [shares(v)]
    converted_counts, spreader_counts = [], []
    for t in range(config.rounds):
        # ---- policies (per world, Python); RoundContext sees that world's TRUE state
        round_alloc = []
        for b in range(B):
            sl = game.world_slice(b)
            state = WorkerState(pinned[sl], ids[b], spreaders_prev[sl], attempts[sl])
            ctx = RoundContext(t, config.rounds, budgets[b], v[sl], game.electorates[b], game.manifestos[b],
                               tuple(airings[b]), ctx_configs[b], state)
            allocation = policies[b](ctx)
            allocation.validate()
            round_alloc.append(allocation)
        gens = [{name: streams[b].generator(name, t, player, device=dev) for name in ("workers", "cascade", "media")}
                for b in range(B)]
        # ---- 1. hires (R22): per world, from that world's workers stream
        new = torch.zeros_like(pinned)
        if any(a.total_hires for a in round_alloc):
            eligible = eligible_workers(game.degree, game.zealot, pinned, wcfg.min_degree)
            if pick_fn is None:
                # == pick_workers per world, with ONE nonzero + one host sync for all worlds: the eligible voters of
                # (world b, region r) are contiguous and ascending in the union, as pick_workers lists them
                cand = torch.nonzero(eligible).flatten()
                group_counts = torch.bincount(game.world_of[cand] * NUM_REGIONS + game.region_of[cand],
                                              minlength=B * NUM_REGIONS).tolist()
                starts = [0]
                for c in group_counts:
                    starts.append(starts[-1] + c)
            for b, allocation in enumerate(round_alloc):
                if allocation.total_hires == 0:
                    continue
                sl = game.world_slice(b)
                e = game.electorates[b]
                if pick_fn is None:
                    picked = []
                    for r, k in enumerate(allocation.hires_by_region):
                        if k == 0:
                            picked.append(torch.zeros(0, dtype=torch.int64, device=dev))
                            continue
                        g = b * NUM_REGIONS + r
                        if group_counts[g] < k:
                            raise ValueError(f"world {b}: region {r} has only {group_counts[g]} eligible voters for {k} "
                                             "new workers")
                        perm = torch.randperm(group_counts[g], generator=gens[b]["workers"], device=dev)[:k]
                        picked.append(cand[starts[g]:starts[g + 1]][perm] - b * n)
                    picked = tuple(picked)
                else:
                    picked = pick_fn(e.region_of, eligible[sl], allocation.hires_by_region, gens[b]["workers"],
                                     world=b, electorate=e)
                ids[b] = tuple(torch.cat([old, add]) for old, add in zip(ids[b], picked))
                new[torch.cat(picked) + b * n] = True
        blocked = pinned | new | game.zealot
        pinned = pinned | new
        spreaders = spreaders_prev | new
        # ---- 2. one IC hop (R24, R25, R27): chance on v at the start of the round
        chance = conversion_chance_rows(game.campaign, v, game.b_rows, vcfg.sensitivity, vcfg.threshold,
                                        wcfg.conversion_rule)
        counts = spreader_edge_counts(game.graph, spreaders, game.rows)
        remaining = (wcfg.attempt_cap - attempts).clamp_min(0)
        eff = torch.where(blocked, torch.zeros_like(counts), torch.minimum(counts, remaining))
        u = torch.cat([torch.rand(n, generator=gens[b]["cascade"], device=dev, dtype=chance.dtype) for b in range(B)])
        p = 1.0 - torch.pow(1.0 - chance.clamp(0.0, 1.0), eff.to(chance.dtype))
        converted = (u < p) & (eff > 0)
        attempts = attempts + eff
        # ---- 3. media (R28): per-voter η_m(n_r)·β for exposed voters of aired regions
        eta_table = torch.tensor([[mcfg.strength / (1.0 + float(mcfg.decay) * k) for k in airings[b]] for b in range(B)],
                                 dtype=game.media.dtype, device=dev).reshape(-1)
        eta = eta_table[game.world_of * NUM_REGIONS + game.region_of]
        aired = torch.tensor([list(a.media_regions) for a in round_alloc], dtype=torch.bool, device=dev).reshape(-1)
        aired_mask = aired[game.world_of * NUM_REGIONS + game.region_of]
        um = torch.cat([torch.rand(n, generator=gens[b]["media"], device=dev) for b in range(B)])
        exposed = aired_mask & (um < mcfg.exposure_prob)
        intensity = combine_intensities(_exposed_intensity(exposed, eta, game.media), frozen=game.zealot)
        # ---- pay (exact Decimal, per world)
        for b, allocation in enumerate(round_alloc):
            cost = worker_cost(allocation.total_hires, wcfg.worker_cost) + \
                media_cost(game.sizes[b], allocation.media_regions, mcfg.base_cost)
            if cost > budgets[b]:
                raise ValueError(f"policy overspent in world {b}, round {t}: cost {cost} > budget {budgets[b]}")
            budgets[b] -= cost
            airings[b] = [x + int(on) for x, on in zip(airings[b], allocation.media_regions)]
            allocations[b].append(allocation)
        # ---- 4. new workers and converts take v = b; social step with workers pinned, zealots frozen
        v = torch.where((new | converted).unsqueeze(-1), game.b_rows, v)
        v = game.step(v, game.aggregate(v), intensity, pinned)
        converted_counts.append(converted.view(B, n).sum(1))
        spreader_counts.append(spreaders.view(B, n).sum(1))
        trajectory.append(shares(v))
        if observer is not None:
            observer(t, {"v": v, "converted": converted, "spreaders": spreaders, "pinned": pinned,
                         "attempts": attempts, "intensity": intensity, "new": new, "game": game})
        spreaders_prev = converted

    # ---- election (per world, its own election stream) and host transfer once
    traj = torch.stack([torch.stack(row) for row in trajectory], 1).to(torch.float64).cpu().tolist()   # [B, T+1]
    conv = torch.stack(converted_counts, 1).cpu().tolist() if converted_counts else [[] for _ in range(B)]
    spr = torch.stack(spreader_counts, 1).cpu().tolist() if spreader_counts else [[] for _ in range(B)]
    out = []
    for b in range(B):
        sl = game.world_slice(b)
        v_b = v[sl]
        result = run_election(v_b, game.manifestos[b], vcfg.sensitivity, vcfg.threshold,
                              streams[b].generator("election", player, device=dev))
        state = WorkerState(pinned[sl], ids[b], spreaders_prev[sl], attempts[sl])
        out.append(WorldOutcome(result, [float(x) for x in traj[b]], config.budget - budgets[b], allocations[b],
                                v_b.clone() if keep_final else None, [int(x) for x in conv[b]],
                                [int(x) for x in spr[b]], state))
    return out


def outcome_row(o: WorldOutcome) -> dict:
    """JSON-safe summary of one ``WorldOutcome`` (for raw JSONL rows)."""
    return {"share": o.result.vote_share, "expected": o.result.expected_share, "votes": o.result.votes,
            "victory": o.result.victory, "spent": str(o.spent), "trajectory": [round(x, 6) for x in o.share_trajectory],
            "converted": o.converted, "spreaders": o.spreaders,
            "allocations": [{"hires": list(a.hires_by_region), "media": list(a.media_regions)} for a in o.allocations]}
