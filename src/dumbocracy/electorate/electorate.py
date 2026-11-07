"""The hidden electorate bundle and its generator (spec §2). Only ``census()`` is public."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import GameConfig, SeedStreams, resolve_device
from ..sectors import REGIONS
from .attributes import Susceptibilities, sample_personalities, sample_susceptibilities
from .preferences import sample_assignment, sample_preferences
from .regions import region_of as compute_region_of
from .regions import sample_region_sizes
from .sbm import Graph, block_edge_counts, block_probability_matrix, sample_block_probabilities, sample_sbm


@dataclass(frozen=True, eq=False)
class Electorate:
    """Hidden ground truth of one world. Never serialise this to a player."""

    sizes: torch.Tensor           # int64 [4] (CPU)
    region_of: torch.Tensor       # int64 [N] (device)
    graph: Graph
    preferences: torch.Tensor     # v(0) [N, 5] (device)
    susceptibility: Susceptibilities
    archetype_of_region: torch.Tensor    # σ int64 [4] (CPU)
    personality_of_region: torch.Tensor  # τ int64 [4] (CPU)
    p_intra: torch.Tensor         # p_r [4] float64 (CPU)
    p_inter: torch.Tensor         # p_rs [4,4] float64 (CPU)
    edge_probability: torch.Tensor  # P = s·p [4,4] (CPU)
    sparsity: float               # s
    sparsity_capped: bool

    @property
    def num_voters(self) -> int:
        return int(self.region_of.numel())

    @property
    def device(self) -> torch.device:
        return self.region_of.device

    def census(self) -> dict:
        """Public structural census (spec §6.1): sizes and realised edge counts only."""
        counts = block_edge_counts(self.graph, self.region_of)
        degree = self.graph.degree.to(torch.float64)
        region_mean_degree = torch.zeros(len(REGIONS), dtype=torch.float64, device=degree.device)
        region_mean_degree.index_add_(0, self.region_of, degree)
        region_mean_degree = (region_mean_degree.cpu() / self.sizes.to(torch.float64)).tolist()
        return {
            "num_voters": self.num_voters,
            "regions": [
                {"region": name, "voters": int(self.sizes[r]), "intra_edges": int(counts[r, r]),
                 "mean_degree": round(region_mean_degree[r], 3)}
                for r, name in enumerate(REGIONS)
            ],
            "inter_edges": [
                {"regions": [REGIONS[r], REGIONS[s]], "edges": int(counts[r, s])}
                for r in range(len(REGIONS)) for s in range(r + 1, len(REGIONS))
            ],
            "total_edges": self.graph.num_edges,
        }


def _to(graph: Graph, device: torch.device) -> Graph:
    if graph.device == device:
        return graph
    return Graph(graph.num_nodes, graph.src.to(device), graph.dst.to(device), graph.indptr.to(device),
                 graph.indices.to(device), graph.degree.to(device))


def generate_electorate(config: GameConfig, seeds: SeedStreams | None = None,
                        device: torch.device | str | None = None) -> Electorate:
    """Generate one world from (config, seed).

    With ``config.generation == "cpu"`` (default, R8) every draw uses CPU streams and the result is moved
    to ``device``: the same (num_voters, seed) gives a bit-identical initial electorate on any device.
    With ``"device"`` per-voter/per-edge draws run on the GPU (faster at 1e6, device-specific)."""
    seeds = seeds or config.seeds()
    target = resolve_device(config.device) if device is None else torch.device(device)
    ecfg = config.electorate
    dtype = config.torch_dtype

    sizes = sample_region_sizes(ecfg.num_voters, ecfg.region_dirichlet_a, seeds.generator("regions"),
                                ecfg.min_region_fraction).cpu()
    p_r, p_rs = sample_block_probabilities(ecfg.intra_prob_range, ecfg.inter_prob_range, seeds.generator("block_probs"))
    probabilities, sparsity, capped = block_probability_matrix(sizes, p_r, p_rs, ecfg.mean_degree)
    draw = torch.device("cpu") if config.generation == "cpu" else target  # R8: canonical CPU draws
    graph = _to(sample_sbm(sizes, probabilities, seeds.generator("graph", device=draw)), target)

    region_of = compute_region_of(sizes, draw)
    assignment = sample_assignment(seeds.generator("preferences", "assignment"))
    preferences = sample_preferences(region_of, assignment, ecfg.dirichlet_concentration,
                                     seeds.generator("preferences", "draws", device=draw), dtype).to(target)
    personality = sample_personalities(seeds.generator("attributes", "personality"))
    sus = sample_susceptibilities(region_of, personality, ecfg.zealot_fraction,
                                  seeds.generator("attributes", "draws", device=draw),
                                  seeds.generator("zealots", device=draw), dtype)
    susceptibility = Susceptibilities(sus.social.to(target), sus.media.to(target), sus.campaign.to(target), sus.zealot.to(target))
    region_of = region_of.to(target)
    return Electorate(sizes, region_of, graph, preferences, susceptibility, assignment, personality,
                      p_r.cpu(), p_rs.cpu(), probabilities, sparsity, capped)
