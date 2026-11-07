"""ORACLE / anticipation helpers (spec §10.2, §14). Oracle results are labelled and never mixed with
player results. ``anticipate`` = social-only dynamics for T rounds: the electorate faced on election day
if nobody campaigns; manifestos should be optimised against it (calibration insight, spec §14)."""
from __future__ import annotations

import torch

from ..config import GameConfig
from .manifesto_opt import optimise_manifesto


def top3_manifesto(v: torch.Tensor) -> torch.Tensor:
    top3 = torch.topk(v, 3, dim=-1).indices
    counts = torch.zeros(5, dtype=torch.float64, device=v.device)
    counts.index_add_(0, top3.reshape(-1), torch.ones(top3.numel(), dtype=torch.float64, device=v.device))
    return (counts / counts.sum()).to(v.dtype)


def anticipate(electorate, config: GameConfig, preferences: torch.Tensor | None = None) -> torch.Tensor:
    """Run T rounds of social-only dynamics: the electorate the player will face on election day
    if nobody campaigns. Manifestos must be optimised against this, not against v(0)."""
    from dumbocracy.dynamics.social import aggregate
    from dumbocracy.dynamics.update import step_preferences
    v = electorate.preferences if preferences is None else preferences
    weight = config.dynamics.social_step * electorate.susceptibility.social
    empty = torch.zeros((0, electorate.num_voters), dtype=v.dtype, device=v.device)
    no_b = torch.zeros((0, 5), dtype=v.dtype, device=v.device)
    for _ in range(config.rounds):
        agg = aggregate(v, electorate.graph, config.dynamics.social_rule, config.dynamics.wmsr_trim)
        v = step_preferences(v, agg, weight, empty, no_b, electorate.susceptibility.zealot)
    return v


def oracle_manifesto(electorate, config: GameConfig) -> torch.Tensor:
    v_t = anticipate(electorate, config)
    starts = torch.stack([top3_manifesto(v_t), v_t.mean(0)])
    return optimise_manifesto(v_t, config.voting.sensitivity, config.voting.threshold, starts=torch.cat(
        [starts, torch.full((1, 5), 0.2, dtype=v_t.dtype, device=v_t.device)])).manifesto


def synthetic_world(electorate, config: GameConfig, known_assignment: bool, salt: int):
    """A world the player can build from public facts (region sizes) plus, if ``known_assignment``,
    the true archetype per region; everything else (graph, attributes, voters) is freshly sampled."""
    from dumbocracy.electorate.attributes import Susceptibilities, sample_personalities, sample_susceptibilities
    from dumbocracy.electorate.electorate import Electorate
    from dumbocracy.electorate.preferences import sample_assignment, sample_preferences
    from dumbocracy.electorate.regions import region_of as region_index
    from dumbocracy.electorate.sbm import block_probability_matrix, sample_block_probabilities, sample_sbm
    ecfg = config.electorate
    gen = torch.Generator().manual_seed(config.seed * 7919 + salt)
    sizes = electorate.sizes
    p_r, p_rs = sample_block_probabilities(ecfg.intra_prob_range, ecfg.inter_prob_range, gen)
    probs, s, capped = block_probability_matrix(sizes, p_r, p_rs, ecfg.mean_degree)
    graph = sample_sbm(sizes, probs, gen)
    regions = region_index(sizes)
    sigma = electorate.archetype_of_region if known_assignment else sample_assignment(gen)
    prefs = sample_preferences(regions, sigma, ecfg.dirichlet_concentration, gen, config.torch_dtype)
    tau = sample_personalities(gen)
    sus = sample_susceptibilities(regions, tau, ecfg.zealot_fraction, gen, gen, config.torch_dtype)
    world = Electorate(sizes, regions, graph, prefs, sus, sigma, tau, p_r, p_rs, probs, s, capped)
    return world


def informed_manifesto(electorate, config: GameConfig) -> torch.Tensor:
    """Knows σ (what a perfect surveyor learns): optimise on an anticipated synthetic world."""
    world = synthetic_world(electorate, config, known_assignment=True, salt=1)
    return oracle_manifesto(world, config).to(electorate.device)


# ----------------------------------------------------------------------------- gate
