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
common random numbers across candidates (spec §9 "batched worlds").
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import torch

from ..config import GameConfig, SeedStreams
from ..dynamics.campaign import Allocation
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

__all__ = ["WorldBatch", "available_workers", "hired_workers", "observed_allocations", "public_sizes", "replay_history",
           "sample_worlds", "scaled_sizes"]


# ------------------------------------------------------------------------------------ public facts

def public_sizes(observation: dict[str, Any]) -> list[int]:
    """Region sizes n_r from the observation's public census (order N, S, E, W)."""
    by_id = {row["id"]: int(row["voters"]) for row in observation["regions"]}
    return [by_id[r] for r in REGIONS]


def hired_workers(observation: dict[str, Any]) -> int:
    """Permanent hires so far (R16): ``workers.hired`` if present, else Σ ``extra_workers`` of the allocations."""
    workers = observation.get("workers")
    if isinstance(workers, dict) and "hired" in workers:
        return int(workers["hired"])
    return sum(int(a.get("extra_workers", 0)) for a in observation.get("allocations") or [])


def available_workers(observation: dict[str, Any], free_workers: int | None = None) -> int:
    """Workers usable in the next allocation before new hires: free + hired (R16)."""
    if free_workers is None:
        workers = observation.get("workers")
        if isinstance(workers, dict) and "free" in workers:
            free_workers = int(workers["free"])
        else:
            free_workers = int(observation["costs"]["free_workers"])
    return int(free_workers) + hired_workers(observation)


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
    return [Allocation(tuple(int(w) for w in a["workers_by_region"]), int(a["extra_workers"]),
                       tuple(bool(m) for m in a["media_regions"])) for a in rows]


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


class WorldBatch:
    """B equal-sized worlds joined into one disjoint union; simulates C plans at once.

    State tensors have shape ``[C, B·n, 5]``. All random draws are shared across the C candidates
    (common random numbers), which makes candidate comparisons low-variance and deterministic per seed.
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
        self.rows = self.graph.rows()
        self.local_region = worlds[0].region_of.repeat(self.num_worlds)                       # [Bn]
        offsets = [0]
        for s in sizes[:-1]:
            offsets.append(offsets[-1] + s)
        self.offsets = torch.tensor(offsets, dtype=torch.int64, device=self.device)
        self.size_t = torch.tensor(sizes, dtype=torch.float64, device=self.device)
        self.preferences = torch.cat([w.preferences for w in worlds])                         # [Bn, 5]
        self.social = torch.cat([w.susceptibility.social for w in worlds])
        self.media = torch.cat([w.susceptibility.media for w in worlds])
        self.campaign = torch.cat([w.susceptibility.campaign for w in worlds])
        self.zealot = torch.cat([w.susceptibility.zealot for w in worlds]).to(torch.bool)
        self.social_weight = config.dynamics.social_step * self.social
        self.contacts_per_worker = config.workers.contacts_per_worker(self.n)

    # ------------------------------------------------------------------ one round, C plans
    def _intensity(self, allocations: Sequence[Allocation], airings: torch.Tensor,
                   generator: torch.Generator) -> torch.Tensor:
        """Campaign intensity c ``[C, Bn]`` for one round (spec §3), common random numbers over C."""
        cfg_w, cfg_m = self.config.workers, self.config.media
        dev, dtype = self.device, self.preferences.dtype
        C, B, n = len(allocations), self.num_worlds, self.n
        total = B * n
        workers = torch.tensor([a.workers_by_region for a in allocations], dtype=torch.int64, device=dev)  # [C,4]
        q = self.contacts_per_worker
        draws_per_region = workers * q                                                           # [C,4]
        kmax = int(draws_per_region.max()) if C else 0
        counts = torch.zeros(C * total, dtype=dtype, device=dev)
        if kmax > 0:
            u = torch.rand((B, NUM_REGIONS, kmax), generator=generator, device=dev, dtype=torch.float64)
            local = (u * self.size_t.view(1, -1, 1)).floor().to(torch.int64)
            local = torch.minimum(local, (self.size_t.to(torch.int64) - 1).view(1, -1, 1))
            voter = local + self.offsets.view(1, -1, 1) + (torch.arange(B, device=dev) * n).view(-1, 1, 1)  # [B,4,K]
            k = torch.arange(kmax, device=dev)
            valid = k.view(1, 1, 1, -1) < draws_per_region.view(C, 1, NUM_REGIONS, 1)           # [C,1,4,K]
            valid = valid.expand(C, B, NUM_REGIONS, kmax)
            flat = (voter.unsqueeze(0) + (torch.arange(C, device=dev) * total).view(-1, 1, 1, 1))
            counts.index_add_(0, flat[valid], torch.ones(int(valid.sum()), dtype=dtype, device=dev))
        counts = counts.view(C, total)
        per_contact = (cfg_w.contact_strength * self.campaign).clamp(0, 1)
        worker_e = 1.0 - torch.pow(1.0 - per_contact, counts)
        # word of mouth: hop-limited SIR on the union graph (same rules as dynamics.campaign.sir_cascade).
        # One uniform per directed CSR entry (row i, col j) decides whether j transmits to i this hop,
        # so "who is hit" is a sparse mat-vec: hit = X @ infectious with X_ij = [u_ij < τ_c].
        spread = ~self.zealot
        seeds = (counts > 0).T                                                                   # [Bn, C]
        eligible = spread.unsqueeze(1) & ~seeds
        infectious = seeds & spread.unsqueeze(1)
        reached = torch.zeros_like(seeds)
        for _ in range(cfg_w.hops):
            fire = (torch.rand(self.graph.indices.numel(), generator=generator, device=dev) < cfg_w.transmission).to(dtype)
            x = _csr(self.graph.indptr, self.graph.indices, fire, total)
            hit = (x @ infectious.to(dtype)) > 0
            new = hit & eligible
            eligible = eligible & ~new
            reached = reached | new
            survive = torch.rand(total, generator=generator, device=dev) >= cfg_w.recovery
            infectious = (infectious & survive.unsqueeze(1)) | new
        reached = reached.T
        wom_e = torch.where(reached, (cfg_w.word_of_mouth_strength * self.campaign).clamp(0, 1), torch.zeros((), dtype=dtype, device=dev))
        # media
        aired = torch.tensor([a.media_regions for a in allocations], dtype=torch.bool, device=dev)  # [C,4]
        exposed = aired[:, self.local_region] & (torch.rand(total, generator=generator, device=dev) < cfg_m.exposure_prob)
        eta = cfg_m.strength / (1.0 + cfg_m.decay * airings.to(dtype))                             # [C,4]
        media_e = torch.where(exposed, (eta[:, self.local_region] * self.media).clamp(0, 1), torch.zeros((), dtype=dtype, device=dev))
        combined = 1.0 - (1.0 - worker_e) * (1.0 - wom_e) * (1.0 - media_e)
        return torch.where(self.zealot, torch.zeros((), dtype=dtype, device=dev), combined.clamp(0, 1))

    def step(self, v: torch.Tensor, allocations: Sequence[Allocation], airings: torch.Tensor,
             manifesto: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Campaign → synchronous social update (spec §3–§4) for C plans; ``v`` is ``[C, Bn, 5]``."""
        intensity = self._intensity(allocations, airings, generator)
        agg = self.aggregate(v)
        return step_preferences(v, agg, self.social_weight, intensity.unsqueeze(0), manifesto.view(1, -1), self.zealot)

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

    def replay(self, observation: dict[str, Any], seed: int = 0) -> torch.Tensor:
        """Advance every world through the player's resolved rounds: returns v_t ``[B·n, 5]``.

        Each recorded allocation already lists every worker it assigned (free + permanent hires, R16), so
        replaying ``workers_by_region`` reproduces the hires' effect without separate bookkeeping."""
        v = self.preferences
        if observation.get("manifesto") is None:
            return v.clone()
        b = self.manifesto_tensor(observation["manifesto"])
        airings = torch.zeros((1, NUM_REGIONS), dtype=torch.int64, device=self.device)
        v = v.unsqueeze(0)
        for t, allocation in enumerate(observed_allocations(observation)):
            gen = SeedStreams(int(seed)).generator("strategy", "replay", t, device=self.device)
            v = self.step(v, [allocation], airings, b, gen)
            airings = airings + torch.tensor([allocation.media_regions], dtype=torch.int64, device=self.device)
        return v[0]


def replay_history(world: Electorate, observation: dict[str, Any], config: GameConfig, *, seed: int = 0) -> torch.Tensor:
    """Re-apply the player's own past allocations and committed manifesto to ``world``: v_t ``[n, 5]``.

    This approximates the true current state, which the player cannot see (campaign draws, graph and
    attributes are the world's own, not the real ones). Before commit it returns v(0).
    """
    return WorldBatch([world], config).replay(observation, seed)

