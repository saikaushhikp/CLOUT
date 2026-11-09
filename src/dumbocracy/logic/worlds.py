"""Sampled information sets I(K_p): worlds consistent with what the player knew (spec §6.5, §10.1; P5).

Kripke reading (lecture "Epistemic logic"): a world is w = (θ = (σ, τ), micro-state). Two worlds are
indistinguishable for the player (w ~_p w') when replaying the player's own actions in them yields the
same knowledge log. The accessibility relation ~_p is an equivalence (S5), so the worlds the player
considers possible at time t form the information set I(K_p,t) ∋ w_true (reflexivity). Knowledge is
truth in every world of the set, possibility truth in some world.

We cannot enumerate I(K_p,t) (the micro-state is continuous), so :func:`consistent_worlds` draws a finite
sample W_t ⊆ I(K_p,t) by rejection:

1. **σ (archetypes)** — drawn from the Bayesian posterior over the 360 assignments given the free
   statistics and sampled-survey answers (:func:`dumbocracy.strategies.bayes.posterior_from_observation`;
   a soft version of the §10.1 ε-filter on query answers). Draws use inverse-CDF on fixed uniforms per
   draw index, so a draw only changes when the posterior moves (common random numbers across rounds).
2. **τ (personalities)** — uniform over the 24 permutations (the public prior; surveys say nothing
   about τ, R13), together with a fresh micro-state (graph reproducing the public census, Dirichlet
   preferences, susceptibilities, zealot flags) from :func:`dumbocracy.strategies.worlds.sample_worlds`
   rescaled to ``world_voters`` voters.
3. **Replay** — the player's committed manifesto and own past allocations are re-applied
   (:class:`~dumbocracy.strategies.worlds.WorldBatch`, batched), recording the world's expected share
   E_w at every poll time (the poll of round s is taken after s−1 resolved rounds).
4. **Poll consistency (hard filter)** — a world is kept iff for every poll observed so far
   ``|E_w(s) − p̂_s| ≤ poll_z · (SE_s + slack)``, where ``SE_s = √(p̃(1−p̃)/n_poll)`` is the poll's standard
   error (p̃ = (n p̂ + 1)/(n + 2), so it is never 0; the contract's ``moe`` is the 95 % half-width
   1.96·SE, so ``poll_z`` is in standard-error units) and ``slack`` (default 0.01) absorbs model error from
   world rescaling and finite simulation. Polls respond to campaigns through the hidden τ, zealots and
   network, so this is how those become partially knowable.

Draws are made in chunks of ``count`` until ``count`` consistent worlds are found or ``max_draws`` is
reached (acceptance rate reported). If no draw is consistent the ``count`` draws closest to the polls
(smallest max z) are used and the set is flagged ``fallback`` (the player's model is misspecified).
All draws are seeded (CPU generators, stream ``worlds``), so the result is deterministic per seed.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import torch

from ..config import GameConfig, SeedStreams
from ..electorate.electorate import Electorate
from ..sectors import NUM_REGIONS
from ..strategies.bayes import ASSIGNMENTS, Posterior, posterior_from_observation
from ..strategies.worlds import WorldBatch, observed_allocations, sample_worlds

__all__ = ["PERMUTATIONS", "WorldSet", "consistent_worlds", "poll_standard_error"]

#: All 24 personality permutations τ (region → personality index), lexicographic.
PERMUTATIONS: torch.Tensor = torch.tensor(list(itertools.permutations(range(NUM_REGIONS))), dtype=torch.int64)


def poll_standard_error(estimate: float, sample_size: int) -> float:
    """``√(p̃(1−p̃)/n)`` with ``p̃ = (n p̂ + 1)/(n + 2)`` (never 0)."""
    n = max(1, int(sample_size))
    p = (float(estimate) * n + 1.0) / (n + 2.0)
    return math.sqrt(p * (1.0 - p) / n)


@dataclass(eq=False)
class WorldSet:
    """A finite sample W_t of the information set at round t (module docstring)."""

    round: int
    worlds: list[Electorate]          # accepted worlds (rescaled, synthetic)
    batch: WorldBatch                 # the accepted worlds joined for batched simulation
    current: torch.Tensor             # v at the decision point, [B·n, 5]
    sigma: torch.Tensor               # [B, 4] archetype index per region
    tau: torch.Tensor                 # [B, 4] personality index per region
    poll_shares: torch.Tensor         # [B, P] world expected share at each observed poll time
    max_z: torch.Tensor               # [B] max over polls of |E_w − p̂| / (SE + slack)
    consistent: torch.Tensor          # [B] bool — False only for fallback worlds
    draws: int                        # candidate worlds simulated
    accepted: int                     # candidates that passed the poll filter
    posterior: Posterior
    fallback: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.worlds)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.draws if self.draws else 0.0


def _record_replay(batch: WorldBatch, observation: dict[str, Any], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the player's history (as :meth:`WorldBatch.replay`) recording E_w after every step.

    Returns ``(v_t [B·n, 5], shares [B, S+1])`` where column s is the share after s resolved rounds.
    """
    b = batch.manifesto_tensor(observation["manifesto"])
    airings = torch.zeros((1, NUM_REGIONS), dtype=torch.int64, device=batch.device)
    v = batch.preferences.unsqueeze(0)
    shares = [batch.shares(v, b)[0]]
    for t, allocation in enumerate(observed_allocations(observation)):
        gen = SeedStreams(int(seed)).generator("strategy", "replay", t, device=batch.device)
        v = batch.step(v, [allocation], airings, b, gen)
        airings = airings + torch.tensor([allocation.media_regions], dtype=torch.int64, device=batch.device)
        shares.append(batch.shares(v, b)[0])
    return v[0], torch.stack(shares, dim=1).to(torch.float64)


def _poll_targets(observation: dict[str, Any], slack: float) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    polls = sorted(observation.get("polls") or [], key=lambda p: int(p["round"]))
    rounds = [int(p["round"]) for p in polls]
    est = torch.tensor([float(p["estimate"]) for p in polls], dtype=torch.float64)
    scale = torch.tensor([poll_standard_error(p["estimate"], int(p.get("sample_size") or 400)) + slack for p in polls],
                         dtype=torch.float64)
    return rounds, est, scale


def consistent_worlds(observation_t: dict[str, Any], config: GameConfig, count: int, *, world_voters: int, seed: int,
                      poll_z: float = 2.5, slack: float = 0.01, max_draws: int | None = None,
                      device: torch.device | str | None = None, posterior: Posterior | None = None) -> WorldSet:
    """Sample up to ``count`` worlds consistent with ``observation_t`` (module docstring).

    ``observation_t`` is the player's observation at the decision point (e.g. from
    :func:`dumbocracy.logic.knowledge.observation_at`); ``config`` carries the public rules;
    ``max_draws`` defaults to ``4·count``. ``posterior`` may be passed to skip recomputing it.
    """
    count = int(count)
    if count < 1:
        raise ValueError("count must be >= 1")
    if observation_t.get("manifesto") is None:
        raise ValueError("observation has no committed manifesto")
    max_draws = int(max_draws) if max_draws is not None else 4 * count
    max_draws = max(max_draws, count)
    t = int(observation_t["round"])
    posterior = posterior or posterior_from_observation(observation_t, sample_fraction=config.survey.query_sample_fraction)
    cdf = torch.cumsum(posterior.probs, 0)
    n = min(int(world_voters), int(observation_t["num_voters"]))
    streams = SeedStreams(int(seed))
    poll_rounds, est, scale = _poll_targets(observation_t, slack)
    if any(r > t or r < 1 for r in poll_rounds):
        raise ValueError("observation_t contains polls after its round")

    chunks: list[dict[str, Any]] = []
    accepted = 0
    draws = 0
    k = 0
    while draws < max_draws and accepted < count:
        size = min(count, max_draws - draws)
        u = torch.rand(size, generator=streams.generator("worlds", "logic-sigma", k), dtype=torch.float64)
        idx = torch.searchsorted(cdf, u * cdf[-1]).clamp_max(cdf.numel() - 1)
        sigma = ASSIGNMENTS[idx]
        tau = PERMUTATIONS[torch.randint(len(PERMUTATIONS), (size,), generator=streams.generator("worlds", "logic-tau", k))]
        worlds = sample_worlds(observation_t, sigma, config, num_voters=n, seed=streams.derive("worlds", "logic-micro", k),
                               device=device, personalities=tau)
        batch = WorldBatch(worlds, config)
        v_t, shares = _record_replay(batch, observation_t, streams.derive("worlds", "logic-replay", k))
        at_polls = shares[:, [r - 1 for r in poll_rounds]].cpu() if poll_rounds else torch.zeros((size, 0), dtype=torch.float64)
        z = ((at_polls - est) / scale).abs().amax(1) if poll_rounds else torch.zeros(size, dtype=torch.float64)
        ok = z <= poll_z
        chunks.append({"worlds": worlds, "v": v_t.view(size, batch.n, -1), "sigma": sigma, "tau": tau,
                       "polls": at_polls, "z": z, "ok": ok})
        accepted += int(ok.sum())
        draws += size
        k += 1

    picks: list[tuple[int, int]] = [(c, i) for c, ch in enumerate(chunks) for i in range(len(ch["worlds"])) if bool(ch["ok"][i])]
    picks = picks[:count]
    fallback = not picks
    if fallback:
        allz = [(float(ch["z"][i]), c, i) for c, ch in enumerate(chunks) for i in range(len(ch["worlds"]))]
        picks = [(c, i) for _, c, i in sorted(allz)[:count]]
    chosen = [chunks[c]["worlds"][i] for c, i in picks]
    batch = WorldBatch(chosen, config)
    current = torch.cat([chunks[c]["v"][i] for c, i in picks])
    return WorldSet(
        round=t, worlds=chosen, batch=batch, current=current,
        sigma=torch.stack([chunks[c]["sigma"][i] for c, i in picks]),
        tau=torch.stack([chunks[c]["tau"][i] for c, i in picks]),
        poll_shares=torch.stack([chunks[c]["polls"][i] for c, i in picks]),
        max_z=torch.stack([chunks[c]["z"][i] for c, i in picks]),
        consistent=torch.stack([chunks[c]["ok"][i] for c, i in picks]),
        draws=draws, accepted=accepted, posterior=posterior, fallback=fallback,
        meta={"poll_z": poll_z, "slack": slack, "world_voters": n, "max_draws": max_draws, "poll_rounds": poll_rounds},
    )
