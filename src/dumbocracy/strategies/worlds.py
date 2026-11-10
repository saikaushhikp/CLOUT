"""Observation-consistent synthetic worlds for player strategies (spec §6.1, §10.1; P4).

A player can never see the hidden electorate. What it *can* do is build plausible worlds from public
facts only:

* region sizes ``n_r`` and the structural census (intra / inter-region edge counts) from the observation;
* an archetype assignment σ supplied by the caller (e.g. a posterior sample from :mod:`.bayes`);
* everything else — graph, Dirichlet preferences, personalities τ, susceptibilities, zealots — is
  freshly drawn from the public prior.

Worlds may be rescaled to a smaller ``num_voters`` for speed: region proportions (and the ≥ 15 % floor)
are kept, and the graph keeps the census' per-region degree structure (edge counts scale with N because
the mean degree is fixed). :func:`replay_history` then re-applies the player's *own* past allocations
and committed manifesto (both in the observation) so each world is advanced to the current round.

:class:`WorldBatch` is the batched simulator used by the planner: B worlds of equal size are joined into
one disjoint-union graph and C candidate plans are simulated at once (leading batch dimension), with
common random numbers across candidates (spec §9 "batched worlds"). It runs the P8 round order (hires →
one IC cascade hop → media → social step with workers pinned) and carries each plan's worker state
(:class:`BatchWorkers`) from round to round. Hires are absolute counts for the TRUE electorate; in a world
rescaled to n < N voters each region's hires are scaled by n/N with stochastic rounding (a worker's cascade
reaches a roughly fixed number of voters, so this keeps the share effect per worker unbiased).
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..config import GameConfig, SeedStreams
from ..dynamics.campaign import Allocation, WorkerState, conversion_chance, ic_from_counts
from ..dynamics.social import aggregate
from ..dynamics.update import step_preferences
from ..electorate.attributes import sample_personalities, sample_susceptibilities
from ..electorate.electorate import Electorate
from ..electorate.preferences import sample_preferences
from ..electorate.regions import minimum_region_size
from ..electorate.regions import region_of as region_index
from ..electorate.sbm import Graph, block_probability_matrix, sample_block_probabilities, sample_sbm
from ..sectors import NUM_REGIONS, REGIONS
from ..voting import similarity, vote_probability

__all__ = ["BatchWorkers", "WorldBatch", "observed_allocations", "public_sizes", "replay_history", "sample_worlds",
           "scaled_sizes", "team_by_region"]


# ------------------------------------------------------------------------------------ public facts

def public_sizes(observation: dict[str, Any]) -> list[int]:
    """Region sizes n_r from the observation's public census (order N, S, E, W)."""
    by_id = {row["id"]: int(row["voters"]) for row in observation["regions"]}
    return [by_id[r] for r in REGIONS]


def team_by_region(observation: dict[str, Any]) -> list[int]:
    """Permanent workers bought so far per region (P8): ``workers.by_region`` if present, else Σ ``hires_by_region``
    of the allocations."""
    workers = observation.get("workers")
    if isinstance(workers, dict) and isinstance(workers.get("by_region"), dict):
        return [int(workers["by_region"].get(r, 0)) for r in REGIONS]
    team = [0] * NUM_REGIONS
    for a in observation.get("allocations") or []:
        team = [x + int(h) for x, h in zip(team, a.get("hires_by_region") or [0] * NUM_REGIONS)]
    return team


def scaled_sizes(sizes: Sequence[int], num_voters: int, min_fraction: float = 0.15) -> list[int]:
    """Rescale ``sizes`` to sum to ``num_voters`` keeping proportions (largest remainder) and the floor."""
    total = sum(sizes)
    if num_voters == total:
        return list(sizes)
    floor = minimum_region_size(num_voters, min_fraction)
    if NUM_REGIONS * floor > num_voters:
        raise ValueError("num_voters too small for the minimum region fraction")
    raw = [num_voters * s / total for s in sizes]
    out = [max(floor, int(math.floor(x))) for x in raw]
    # fix the sum: add to the largest remainders / remove from the largest regions above the floor
    while sum(out) < num_voters:
        i = max(range(NUM_REGIONS), key=lambda k: raw[k] - out[k])
        out[i] += 1
    while sum(out) > num_voters:
        i = max((k for k in range(NUM_REGIONS) if out[k] > floor), key=lambda k: out[k] - raw[k])
        out[i] -= 1
    return out


def _census_block_probabilities(observation: dict[str, Any], sizes: Sequence[int]) -> torch.Tensor:
    """Edge probabilities P [4x4] reproducing the public census edge counts, scaled to ``sizes``.

    With a fixed mean degree the number of edges scales linearly with N, so the expected count in the
    rescaled world is ``e_rs · N'/N``; P = expected count / number of candidate pairs (capped at 1).
    """
    true_sizes = public_sizes(observation)
    factor = sum(sizes) / sum(true_sizes)
    counts = torch.zeros(NUM_REGIONS, NUM_REGIONS, dtype=torch.float64)
    for row in observation["regions"]:
        counts[REGIONS.index(row["id"]), REGIONS.index(row["id"])] = float(row["intra_edges"])
    for row in observation["inter_edges"]:
        a, b = (REGIONS.index(x) for x in row["regions"])
        counts[a, b] = counts[b, a] = float(row["edges"])
    n = torch.tensor([float(s) for s in sizes], dtype=torch.float64)
    pairs = torch.outer(n, n)
    pairs.fill_diagonal_(0.0)
    pairs += torch.diag(n * (n - 1) / 2)
    return (counts * factor / pairs.clamp_min(1.0)).clamp(0.0, 1.0)


# ------------------------------------------------------------------------------------ world sampler

def sample_worlds(observation: dict[str, Any], assignments: torch.Tensor, config: GameConfig, *,
                  num_voters: int | None = None, seed: int, graph: str = "census",
                  device: torch.device | str | None = None,
                  personalities: torch.Tensor | None = None) -> list[Electorate]:
    """One synthetic, observation-consistent world per row of ``assignments`` ([B, 4] archetype indices).

    ``personalities`` (optional ``[B, 4]`` personality indices, one permutation τ per row) fixes τ per
    world instead of drawing it; the generator still consumes the τ draw, so every other micro-state draw
    of world b is identical whichever τ is imposed (P5 world sampler).

    ``graph="census"`` (default) draws an ER-SBM whose block probabilities reproduce the public census
    edge counts; ``graph="prior"`` draws fresh ``p_r``/``p_rs`` from the prior (spec §2.2) instead.
    Personalities τ are uniform permutations, preferences ``Dirichlet(κ μ_σ)``, zealots w.p. ζ.
    Draws use CPU generators keyed ``("strategy", "world", seed, b)``, so the result is deterministic.
    """
    if graph not in ("census", "prior"):
        raise ValueError("graph must be 'census' or 'prior'")
    assignments = torch.as_tensor(assignments, dtype=torch.int64).reshape(-1, NUM_REGIONS).cpu()
    ecfg = config.electorate
    sizes_list = public_sizes(observation)
    if num_voters is not None and num_voters < sum(sizes_list):
        sizes_list = scaled_sizes(sizes_list, int(num_voters), ecfg.min_region_fraction)
    sizes = torch.tensor(sizes_list, dtype=torch.int64)
    target = torch.device(device) if device is not None else torch.device("cpu")
    census_probs = _census_block_probabilities(observation, sizes_list) if graph == "census" else None
    if personalities is not None:
        personalities = torch.as_tensor(personalities, dtype=torch.int64).reshape(-1, NUM_REGIONS).cpu()
        if personalities.shape[0] != assignments.shape[0]:
            raise ValueError("personalities must have one row per assignment")
    streams = SeedStreams(int(seed))
    worlds = []
    for b, sigma in enumerate(assignments):
        gen = streams.generator("strategy", "world", b)
        if census_probs is None:
            p_r, p_rs = sample_block_probabilities(ecfg.intra_prob_range, ecfg.inter_prob_range, gen)
            probs, s, capped = block_probability_matrix(sizes, p_r, p_rs, ecfg.mean_degree)
        else:
            probs, s, capped = census_probs, 1.0, False
            p_r, p_rs = torch.diag(probs).clone(), probs.clone().fill_diagonal_(0.0)
        g = sample_sbm(sizes, probs, gen)
        regions = region_index(sizes)
        prefs = sample_preferences(regions, sigma, ecfg.dirichlet_concentration, gen, config.torch_dtype)
        tau = sample_personalities(gen)
        if personalities is not None:
            tau = personalities[b].clone()
        sus = sample_susceptibilities(regions, tau, ecfg.zealot_fraction, gen, gen, config.torch_dtype)
        if target.type != "cpu":
            g = Graph(g.num_nodes, g.src.to(target), g.dst.to(target), g.indptr.to(target), g.indices.to(target),
                      g.degree.to(target))
            regions, prefs = regions.to(target), prefs.to(target)
            sus = type(sus)(sus.social.to(target), sus.media.to(target), sus.campaign.to(target), sus.zealot.to(target))
        worlds.append(Electorate(sizes, regions, g, prefs, sus, sigma.clone(), tau, p_r, p_rs, probs, s, capped))
    return worlds


# ------------------------------------------------------------------------------- batched simulator

def observed_allocations(observation: dict[str, Any]) -> list[Allocation]:
    """The player's own resolved allocations (oldest first) from the observation."""
    rows = sorted(observation.get("allocations", []), key=lambda a: a["round"])
    return [Allocation(tuple(int(h) for h in a["hires_by_region"]), tuple(bool(m) for m in a["media_regions"]))
            for a in rows]


def _csr(indptr: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
        return torch.sparse_csr_tensor(indptr, indices, values, (n, n))


def _union_graph(graphs: Sequence[Graph], n: int) -> Graph:
    device = graphs[0].device
    src = torch.cat([g.src + i * n for i, g in enumerate(graphs)])
    dst = torch.cat([g.dst + i * n for i, g in enumerate(graphs)])
    indices = torch.cat([g.indices + i * n for i, g in enumerate(graphs)])
    degree = torch.cat([g.degree for g in graphs])
    indptr = torch.zeros(len(graphs) * n + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(degree, 0)
    return Graph(len(graphs) * n, src, dst, indptr, indices, degree)


@dataclass(frozen=True, eq=False)
class BatchWorkers:
    """Worker state of C plans over the batched worlds (P8): ``[C, B·n]`` tensors (see ``WorkerState``)."""

    pinned: torch.Tensor     # bool  [C, Bn]
    spreaders: torch.Tensor  # bool  [C, Bn]
    attempts: torch.Tensor   # int64 [C, Bn]

    @classmethod
    def empty(cls, plans: int, total: int, device: torch.device | str = "cpu") -> "BatchWorkers":
        z = torch.zeros((plans, total), dtype=torch.bool, device=device)
        return cls(z, z.clone(), torch.zeros((plans, total), dtype=torch.int64, device=device))

    @classmethod
    def from_state(cls, state: WorkerState) -> "BatchWorkers":
        """One plan (C = 1) from a single-world engine/simulate :class:`WorkerState`."""
        return cls(state.pinned.unsqueeze(0), state.spreaders.unsqueeze(0), state.attempts.unsqueeze(0))

    @property
    def plans(self) -> int:
        return int(self.pinned.shape[0])

    def expand(self, plans: int) -> "BatchWorkers":
        """Broadcast a single plan's state (C = 1) to ``plans`` copies."""
        if self.plans == plans:
            return self
        if self.plans != 1:
            raise ValueError(f"cannot expand worker state of {self.plans} plans to {plans}")
        return BatchWorkers(*(x.expand(plans, -1).clone() for x in (self.pinned, self.spreaders, self.attempts)))

    def select(self, index: int) -> "BatchWorkers":
        return BatchWorkers(self.pinned[index:index + 1], self.spreaders[index:index + 1], self.attempts[index:index + 1])


class WorldBatch:
    """B equal-sized worlds joined into one disjoint union; simulates C plans at once.

    State tensors have shape ``[C, B·n, 5]`` (preferences) and ``[C, B·n]`` (:class:`BatchWorkers`). All
    random draws are shared across the C candidates (common random numbers), which makes candidate
    comparisons low-variance and deterministic per seed.
    """

    def __init__(self, worlds: Sequence[Electorate], config: GameConfig):
        if not worlds:
            raise ValueError("need at least one world")
        sizes = [int(x) for x in worlds[0].sizes.tolist()]
        if any([int(x) for x in w.sizes.tolist()] != sizes for w in worlds):
            raise ValueError("all worlds must share region sizes")
        self.config = config
        self.worlds = list(worlds)
        self.num_worlds = len(worlds)
        self.n = sum(sizes)
        self.sizes = sizes
        self.device = worlds[0].device
        self.graph = _union_graph([w.graph for w in worlds], self.n)
        self.local_region = worlds[0].region_of.repeat(self.num_worlds)                       # [Bn]
        self.preferences = torch.cat([w.preferences for w in worlds])                         # [Bn, 5]
        self.social = torch.cat([w.susceptibility.social for w in worlds])
        self.media = torch.cat([w.susceptibility.media for w in worlds])
        self.campaign = torch.cat([w.susceptibility.campaign for w in worlds])
        self.zealot = torch.cat([w.susceptibility.zealot for w in worlds]).to(torch.bool)
        self.social_weight = config.dynamics.social_step * self.social
        # worker picks (R22): group = (world, region); voters of a group are contiguous in index order
        total = self.num_worlds * self.n
        world_of = torch.arange(total, device=self.device) // self.n
        self.group = world_of * NUM_REGIONS + self.local_region                              # [Bn], non-decreasing
        if bool((self.group[1:] < self.group[:-1]).any()):
            raise ValueError("worlds must list their voters region by region")
        group_start = torch.zeros(self.num_worlds * NUM_REGIONS, dtype=torch.int64, device=self.device)
        counts = torch.bincount(self.group, minlength=self.num_worlds * NUM_REGIONS)
        group_start[1:] = torch.cumsum(counts, 0)[:-1]
        self.rank_in_group = torch.arange(total, device=self.device) - group_start[self.group]
        self.eligible_base = (self.graph.degree >= config.workers.min_degree) & ~self.zealot
        # hires are counts for the true electorate (N voters); scale to this world size
        self.hire_scale = min(1.0, self.n / max(1, int(config.electorate.num_voters)))

    # ------------------------------------------------------------------ one round, C plans
    def _hire_counts(self, allocations: Sequence[Allocation], generator: torch.Generator) -> torch.Tensor:
        """Hires per (plan, world, region) ``[C, B·4]`` after scaling to the world size (stochastic rounding,
        one uniform per (world, region) shared across plans)."""
        dev = self.device
        hires = torch.tensor([list(a.hires_by_region) for a in allocations], dtype=torch.float64, device=dev)  # [C,4]
        u = torch.rand((self.num_worlds, NUM_REGIONS), generator=generator, device=dev, dtype=torch.float64)
        scaled = hires.unsqueeze(1) * self.hire_scale                                            # [C,1,4]
        base = scaled.floor()
        count = base + (u.unsqueeze(0) < scaled - base).to(torch.float64)                       # [C,B,4]
        return count.reshape(len(allocations), -1).to(torch.int64)

    def _pick(self, hires: torch.Tensor, pinned: torch.Tensor, priority: torch.Tensor) -> torch.Tensor:
        """New workers ``[C, Bn]``: per (plan, world, region) the ``hires`` eligible voters with the lowest
        priority (one uniform per voter, shared across plans) — a uniform pick without replacement (R22).
        A synthetic region with too few eligible voters simply gets fewer workers."""
        eligible = self.eligible_base & ~pinned                                                  # [C, Bn]
        key = self.group.to(torch.float64) * 4.0 + torch.where(eligible, priority, torch.full_like(priority, 2.0))
        order = torch.argsort(key, dim=1)
        quota = hires[:, self.group]                                                             # [C, Bn] by sorted position
        chosen = (self.rank_in_group < quota) & torch.gather(eligible, 1, order)
        new = torch.zeros_like(eligible)
        return new.scatter_(1, order, chosen)

    def step(self, v: torch.Tensor, allocations: Sequence[Allocation], airings: torch.Tensor,
             manifesto: torch.Tensor, generator: torch.Generator,
             workers: BatchWorkers | None = None) -> tuple[torch.Tensor, BatchWorkers]:
        """One P8 round for C plans: hires → one cascade hop → media → social step with workers pinned at b.

        ``v`` is ``[C, Bn, 5]``; ``workers`` the plans' worker state (``None`` = nobody hired yet; a C = 1 state
        is broadcast). Returns ``(v', workers')``. P_j for the conversion chance uses ``v`` (start of round).
        """
        cfg_w, cfg_m = self.config.workers, self.config.media
        dev, dtype = self.device, v.dtype
        C, total = len(allocations), self.num_worlds * self.n
        workers = BatchWorkers.empty(C, total, dev) if workers is None else workers.expand(C)
        b = manifesto.to(device=dev, dtype=dtype)
        # 1. hires (draws are made every round so the stream does not depend on the plans)
        hires = self._hire_counts(allocations, generator)
        priority = torch.rand(total, generator=generator, device=dev, dtype=torch.float64)
        new = self._pick(hires, workers.pinned, priority) if bool((hires > 0).any()) else torch.zeros_like(workers.pinned)
        pinned = workers.pinned | new
        spreaders = workers.spreaders | new
        # 2. one IC hop: chance γ·P on v at the start of the round; workers and zealots are never targets
        chance = conversion_chance(self.campaign, v, b, self.config.voting.sensitivity, self.config.voting.threshold)
        counts = self._spreader_counts(spreaders, dtype)
        converted, attempts = ic_from_counts(counts, chance, workers.attempts, pinned | self.zealot,
                                             cfg_w.attempt_cap, generator)
        # 3. media
        aired = torch.tensor([a.media_regions for a in allocations], dtype=torch.bool, device=dev)  # [C,4]
        exposed = aired[:, self.local_region] & (torch.rand(total, generator=generator, device=dev) < cfg_m.exposure_prob)
        eta = cfg_m.strength / (1.0 + cfg_m.decay * airings.to(dtype))                             # [C,4]
        media_e = torch.where(exposed, (eta[:, self.local_region] * self.media).clamp(0, 1), torch.zeros((), dtype=dtype, device=dev))
        media_e = torch.where(self.zealot, torch.zeros((), dtype=dtype, device=dev), media_e)
        # 4. social step on v with new workers and converts at b; workers pinned, zealots frozen
        v = torch.where((new | converted).unsqueeze(-1), b, v)
        agg = self.aggregate(v)
        v = step_preferences(v, agg, self.social_weight, media_e.unsqueeze(0), b.view(1, -1), self.zealot,
                             pinned=pinned.unsqueeze(0))
        return v, BatchWorkers(pinned, converted, attempts)

    def _spreader_counts(self, spreaders: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``count_j`` = spreader → j edges for ``[C, Bn]`` spreaders (one sparse mat-mul)."""
        if not bool(spreaders.any()):
            return torch.zeros(spreaders.shape, dtype=torch.int64, device=self.device)
        counts = self._adjacency(dtype) @ spreaders.T.to(dtype)                                   # [Bn, C]
        return counts.T.round().to(torch.int64)

    def aggregate(self, v: torch.Tensor) -> torch.Tensor:
        """``Agg(v)`` for ``[C, Bn, 5]``; DeGroot as one sparse mat-mul (much faster than gather/scatter)."""
        if self.config.dynamics.social_rule != "degroot":
            return aggregate(v, self.graph, self.config.dynamics.social_rule, self.config.dynamics.wmsr_trim)
        C, total, k = v.shape
        flat = v.permute(1, 0, 2).reshape(total, C * k)
        summed = (self._adjacency(v.dtype) @ flat).reshape(total, C, k).permute(1, 0, 2)
        degree = self.graph.degree.to(v.dtype).view(1, -1, 1)
        return torch.where(degree > 0, summed / degree.clamp_min(1), v)

    def _adjacency(self, dtype: torch.dtype) -> torch.Tensor:
        if getattr(self, "_adj", None) is None or self._adj.dtype != dtype:
            ones = torch.ones(self.graph.indices.numel(), dtype=dtype, device=self.device)
            self._adj = _csr(self.graph.indptr, self.graph.indices, ones, self.graph.num_nodes)
        return self._adj

    def shares(self, v: torch.Tensor, manifesto: torch.Tensor) -> torch.Tensor:
        """Expected share per (plan, world): ``[C, B]``."""
        vote = self.config.voting
        prob = vote_probability(similarity(v, manifesto), vote.sensitivity, vote.threshold)
        return prob.view(v.shape[0], self.num_worlds, self.n).mean(-1)

    def manifesto_tensor(self, manifesto: Sequence[float] | torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(manifesto, dtype=self.preferences.dtype).to(self.device)

    def replay_state(self, observation: dict[str, Any], seed: int = 0, *,
                     record_shares: bool = False):
        """Advance every world through the player's resolved rounds (their hires and media, P8).

        Returns ``(v_t [B·n, 5], workers (C = 1))``; with ``record_shares`` also ``shares [B, S+1]`` (the
        expected share after s resolved rounds, column 0 = before round 1)."""
        v = self.preferences.unsqueeze(0)
        workers = BatchWorkers.empty(1, self.num_worlds * self.n, self.device)
        if observation.get("manifesto") is None:
            out = (v[0].clone(), workers)
            return (*out, torch.zeros((self.num_worlds, 1), dtype=torch.float64)) if record_shares else out
        b = self.manifesto_tensor(observation["manifesto"])
        airings = torch.zeros((1, NUM_REGIONS), dtype=torch.int64, device=self.device)
        shares = [self.shares(v, b)[0]]
        for t, allocation in enumerate(observed_allocations(observation)):
            gen = SeedStreams(int(seed)).generator("strategy", "replay", t, device=self.device)
            v, workers = self.step(v, [allocation], airings, b, gen, workers)
            airings = airings + torch.tensor([allocation.media_regions], dtype=torch.int64, device=self.device)
            if record_shares:
                shares.append(self.shares(v, b)[0])
        if record_shares:
            return v[0], workers, torch.stack(shares, dim=1).to(torch.float64)
        return v[0], workers

    def replay(self, observation: dict[str, Any], seed: int = 0) -> torch.Tensor:
        """``v_t [B·n, 5]`` after replaying the player's resolved rounds (see :meth:`replay_state`)."""
        return self.replay_state(observation, seed)[0]


def replay_history(world: Electorate, observation: dict[str, Any], config: GameConfig, *, seed: int = 0) -> torch.Tensor:
    """Re-apply the player's own past allocations and committed manifesto to ``world``: v_t ``[n, 5]``.

    This approximates the true current state, which the player cannot see (campaign draws, graph and
    attributes are the world's own, not the real ones). Before commit it returns v(0).
    """
    return WorldBatch([world], config).replay(observation, seed)

