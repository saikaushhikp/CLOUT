"""Observation-only AI players (spec §8, §10.2; P4).

Every player implements :class:`dumbocracy.players.base.Player`: ``act(observation) -> action dict`` in the
JSON body format accepted by :func:`dumbocracy.game.actions.action_from_dict`. Players receive *only* the
allowlisted contract ``Observation`` dict (web_api_v2.md) — never the engine state — and use the public
rules (a :class:`GameConfig` with default parameters; its seed is never used to regenerate the world).

* :class:`UniformPlayer` — no queries, uniform manifesto, free workers split evenly, no spending.
* :class:`Top3Player`   — no queries, manifesto = normalised free top-3 statistic, spends the treasury evenly
  over the rounds, half media (least-aired first) / half new hires (permanent, R16); all available workers
  split by region size.
* :class:`BayesPlanner` — information-gain sampled-survey queries (:mod:`.bayes`; repeats allowed, stops when the
  expected information per coin is low), manifesto optimised on the anticipated
  electorate of posterior-sampled worlds, then a rollout planner (:mod:`.planner`) every round on
  posterior-sampled worlds replayed to the current round (:mod:`.worlds`).
* :class:`RandomPlayer` — seeded random legal actions.

:func:`play_game` drives a full engine game, handing each player a deep copy of its observation only.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Sequence

import torch

from ..config import GameConfig, SeedStreams
from ..dynamics.campaign import Allocation
from ..sectors import NUM_REGIONS, NUM_SECTORS, REGIONS, SECTORS
from .manifesto_opt import optimise_manifesto, project_simplex
from .oracle import anticipate, top3_manifesto
from .planner import default_allocation, plan_round, rounds_left, split_by_weight
from .worlds import WorldBatch, available_workers, hired_workers, public_sizes, sample_worlds

__all__ = ["BayesPlanner", "RandomPlayer", "Top3Player", "UniformPlayer", "allocate_dict", "play_game",
           "public_config", "simplex_list"]


# ------------------------------------------------------------------------------------- helpers

def public_config(observation: dict[str, Any], base: GameConfig | None = None) -> GameConfig:
    """The public rules for this game: default (or ``base``) parameters with N, T and (R17) the starting
    treasury from the observation."""
    base = base or GameConfig()
    budget = Decimal(str(observation["initial_budget"])) if observation.get("initial_budget") is not None else base.budget
    return replace(base, rounds=int(observation["rounds"]), device="cpu", budget=budget,
                   electorate=replace(base.electorate, num_voters=int(observation["num_voters"])))


def simplex_list(values: Sequence[float] | torch.Tensor) -> list[float]:
    """Float list ≥ 0 summing to 1 within 1e-12 (fixes float error on the largest entry)."""
    x = [max(0.0, float(v)) for v in (values.tolist() if isinstance(values, torch.Tensor) else values)]
    total = math.fsum(x)
    if total <= 0:
        return [1.0 / len(x)] * len(x)
    x = [v / total for v in x]
    i = max(range(len(x)), key=lambda k: x[k])
    x[i] = max(0.0, 1.0 - math.fsum(x[:i] + x[i + 1:]))
    return x


def allocate_dict(a: Allocation) -> dict[str, Any]:
    return {"type": "allocate", "workers_by_region": [int(w) for w in a.workers_by_region],
            "extra_workers": int(a.extra_workers), "media_regions": [bool(m) for m in a.media_regions]}


def _treasury(observation: dict[str, Any]) -> Decimal:
    return Decimal(str(observation["treasury"]))


def _airings(observation: dict[str, Any]) -> list[int]:
    return [int(observation.get("media_airings", {}).get(r, 0)) for r in REGIONS]


def _asked_now(observation: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Queries asked in the current phase/round, in order (R12: repeats allowed, so may contain duplicates)."""
    phase, rnd = observation["phase"], int(observation["round"])
    return [(q["sector_a"], q["sector_b"], q["region"]) for q in observation.get("queries", [])
            if q.get("phase") == phase and int(q.get("round", -1)) == rnd]


def _query_spend(observation: dict[str, Any]) -> Decimal:
    return sum((Decimal(str(q["cost"])) for q in observation.get("queries", [])), Decimal("0.00"))


class _Timed:
    """Mixin recording per-decision latency (seconds) by action type."""

    def _record(self, kind: str, seconds: float) -> None:
        self.latency.setdefault(kind, []).append(seconds)


# --------------------------------------------------------------------------------- baselines

@dataclass
class UniformPlayer(_Timed):
    """No queries; uniform manifesto; available workers (free + any earlier hires) split evenly; never spends."""

    player_id: str = "p0"
    latency: dict[str, list[float]] = field(default_factory=dict)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        if observation["phase"] == "SURVEY":
            action = {"type": "commit", "manifesto": [1.0 / NUM_SECTORS] * NUM_SECTORS}
        else:
            action = allocate_dict(Allocation(split_by_weight(available_workers(observation), [1.0] * NUM_REGIONS)))
        self._record(action["type"], time.perf_counter() - start)
        return action


@dataclass
class Top3Player(_Timed):
    """No queries; manifesto = normalised free top-3 statistic; even split workers/media spending."""

    player_id: str = "p0"
    config: GameConfig | None = None
    latency: dict[str, list[float]] = field(default_factory=dict)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        if observation["phase"] == "SURVEY":
            stat = min(observation["free_stats"], key=lambda e: int(e["round"]))["top3_pct"]
            action = {"type": "commit", "manifesto": simplex_list([stat[s] for s in SECTORS])}
        else:
            cfg = public_config(observation, self.config)
            a = default_allocation(public_sizes(observation), _treasury(observation), rounds_left(observation),
                                   _airings(observation), cfg, hired_workers(observation))   # R16: hires persist
            action = allocate_dict(a)
        self._record(action["type"], time.perf_counter() - start)
        return action


@dataclass
class RandomPlayer(_Timed):
    """Seeded random *legal* actions: a few random queries, a random manifesto, random affordable plans."""

    player_id: str = "p0"
    seed: int = 0
    query_prob: float = 0.3
    max_queries: int = 4
    latency: dict[str, list[float]] = field(default_factory=dict)

    def _rng(self, observation: dict[str, Any]) -> random.Random:
        key = f"{self.seed}:{observation['seed']}:{observation['phase']}:{observation['round']}:{len(observation.get('queries', []))}"
        return random.Random(key)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        action = self._choose(observation)
        self._record(action["type"], time.perf_counter() - start)
        return action

    def _choose(self, observation: dict[str, Any]) -> dict[str, Any]:
        rng = self._rng(observation)
        treasury = _treasury(observation)
        query_cost = Decimal(observation["costs"]["query"])
        if ("query" in observation["legal_actions"] and len(observation.get("queries", [])) < self.max_queries
                and treasury >= query_cost and rng.random() < self.query_prob):
            a, b = rng.sample(SECTORS, 2)  # R12: repeats are legal (a fresh sample each time)
            return {"type": "query", "sector_a": a, "sector_b": b, "region": rng.choice(REGIONS)}
        if observation["phase"] == "SURVEY":
            return {"type": "commit", "manifesto": simplex_list([rng.gammavariate(1.0, 1.0) for _ in SECTORS])}
        if rng.random() < 0.05:
            return {"type": "forfeit"}
        prices = [Decimal(observation["costs"]["media_by_region"][r]) for r in REGIONS]
        budget = treasury * Decimal(str(round(rng.random(), 2)))
        media = [False] * NUM_REGIONS
        for r in rng.sample(range(NUM_REGIONS), NUM_REGIONS):
            if rng.random() < 0.4 and sum(p for p, m in zip(prices, media) if m) + prices[r] <= budget:
                media[r] = True
        spent = sum((p for p, m in zip(prices, media) if m), Decimal(0))
        extra = int(max(Decimal(0), budget - spent) // Decimal(observation["costs"]["extra_worker"]))
        weights = [rng.random() for _ in range(NUM_REGIONS)]
        total = available_workers(observation) + extra   # free + earlier permanent hires + new hires (R16)
        workers = split_by_weight(rng.randint(0, total), weights)
        return allocate_dict(Allocation(workers, extra, tuple(media)))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- Bayes planner

@dataclass
class BayesPlanner(_Timed):
    """Survey by information gain, optimise the manifesto on anticipated posterior worlds, plan by rollouts.

    Queries are sampled surveys (R12): every answer is a fresh 10% sample, so asking the same triple again
    is legal and can be the most informative choice (:func:`.bayes.best_query` does not exclude repeats).

    Survey stopping (all optional, ``None`` disables): ask at most ``num_queries`` survey queries and spend at
    most ``survey_budget`` on queries over the game; stop once the posterior entropy over σ is below
    ``stop_entropy`` nats, or once the best query's expected information gain per unit of money falls below
    ``min_gain_per_coin`` (nats per 1.00; the default 0.01 = 0.05 nats for a 5.00 query).
    Other parameters: ``num_worlds`` (B, posterior-sampled worlds per decision), ``world_voters`` (size of
    each synthetic world), ``round_query_gain`` (in a round, ask while the best expected gain in nats exceeds
    this, at most ``round_queries`` per round; ``None`` disables), ``planning`` (False → even default pacing
    instead of the rollout planner; ablation), ``device`` for the synthetic worlds (CPU is fastest here).
    """

    player_id: str = "p0"
    seed: int = 0
    num_queries: int | None = 10
    survey_budget: Decimal = Decimal("50.00")
    num_worlds: int = 8
    manifesto_worlds: int = 8
    world_voters: int = 1600
    stop_entropy: float | None = 0.05
    min_gain_per_coin: float | None = 0.01
    round_query_gain: float | None = 0.3
    round_queries: int = 1
    rollout: str = "even"
    planning: bool = True
    device: str = "cpu"
    config: GameConfig | None = None
    latency: dict[str, list[float]] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        from .bayes import build_answer_model
        self.survey_budget = Decimal(str(self.survey_budget))
        self.model = build_answer_model()

    # -- belief
    def _posterior(self, observation: dict[str, Any]):
        from .bayes import posterior_from_observation
        return posterior_from_observation(observation, self.model, sample_fraction=self._fraction())

    def _fraction(self) -> float:
        return (self.config or GameConfig()).survey.query_sample_fraction

    def _streams(self, observation: dict[str, Any]) -> SeedStreams:
        return SeedStreams(int(self.seed) * 1_000_003 + int(observation["seed"]))

    def _worlds(self, observation: dict[str, Any], posterior, count: int, tag: str):
        cfg = public_config(observation, self.config)
        streams = self._streams(observation)
        gen = streams.generator("strategy", "posterior", tag, observation["round"])
        assignments = posterior.sample(count, gen)
        world_seed = streams.derive("strategy", "worlds", tag, observation["round"])
        n = min(self.world_voters, int(observation["num_voters"]))
        return cfg, sample_worlds(observation, assignments, cfg, num_voters=n, seed=world_seed, device=self.device)

    def _maybe_query(self, observation: dict[str, Any], posterior, survey: bool) -> dict[str, Any] | None:
        from .bayes import expected_information_gain
        if "query" not in observation["legal_actions"]:
            return None
        cost = Decimal(observation["costs"]["query"])
        if _query_spend(observation) + cost > self.survey_budget or _treasury(observation) < cost:
            return None
        rnd = int(observation["round"])
        if survey:
            n_survey = sum(1 for q in observation.get("queries", []) if q.get("phase") == "SURVEY")
            if self.num_queries is not None and n_survey >= self.num_queries:
                return None
            if self.stop_entropy is not None and posterior.entropy() < self.stop_entropy:
                return None  # σ is already (almost) certain: save the money for the campaign
            threshold = None if self.min_gain_per_coin is None else self.min_gain_per_coin * float(cost)
        else:
            if self.round_query_gain is None or len(_asked_now(observation)) >= self.round_queries:
                return None
            threshold = self.round_query_gain
        ranked = expected_information_gain(posterior, self.model, public_sizes(observation), round_index=rnd,
                                           sample_fraction=self._fraction())
        if not ranked or (threshold is not None and ranked[0][0] < threshold):
            return None  # expected information per coin too low: stop surveying
        a, b, r = ranked[0][1]  # == best_query(...) without exclusions; repeats allowed (R12)
        return {"type": "query", "sector_a": a, "sector_b": b, "region": r}

    # -- decisions
    def manifesto(self, observation: dict[str, Any], posterior=None) -> list[float]:
        """Optimise b on the concatenated *anticipated* electorates of posterior-sampled worlds (spec §14)."""
        posterior = posterior or self._posterior(observation)
        cfg, worlds = self._worlds(observation, posterior, self.manifesto_worlds, "manifesto")
        v_t = torch.cat([anticipate(w, cfg) for w in worlds])
        stat = min(observation["free_stats"], key=lambda e: int(e["round"]))["top3_pct"]
        extra = torch.stack([top3_manifesto(v_t), v_t.mean(0),
                             torch.tensor(simplex_list([stat[s] for s in SECTORS]), dtype=v_t.dtype)])
        starts = project_simplex(torch.cat([extra, torch.full((1, NUM_SECTORS), 0.2, dtype=v_t.dtype)]))
        sol = optimise_manifesto(v_t, cfg.voting.sensitivity, cfg.voting.threshold, starts=starts)
        return simplex_list(sol.manifesto)

    def allocation(self, observation: dict[str, Any], posterior=None) -> tuple[Allocation, dict[str, Any]]:
        """Rollout-planned allocation (or, with ``planning=False``, the even default pacing)."""
        if not self.planning:
            cfg = public_config(observation, self.config)
            a = default_allocation(public_sizes(observation), _treasury(observation), rounds_left(observation),
                                   _airings(observation), cfg, hired_workers(observation))
            return a, {"best_score": float("nan"), "num_candidates": 1, "seconds": 0.0}
        posterior = posterior or self._posterior(observation)
        cfg, worlds = self._worlds(observation, posterior, self.num_worlds, "plan")
        batch = WorldBatch(worlds, cfg)
        seed = self._streams(observation).derive("strategy", "rollout", observation["round"])
        current = batch.replay(observation, seed)
        return plan_round(observation, batch, observation["manifesto"], cfg, rollout=self.rollout, seed=seed,
                          current=current)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        posterior = self._posterior(observation)
        survey = observation["phase"] == "SURVEY"
        action = self._maybe_query(observation, posterior, survey)
        if action is None:
            if survey:
                action = {"type": "commit", "manifesto": self.manifesto(observation, posterior)}
            else:
                allocation, diag = self.allocation(observation, posterior)
                self.diagnostics.append({"round": observation["round"], "best_score": diag["best_score"],
                                         "num_candidates": diag["num_candidates"], "seconds": diag["seconds"],
                                         "entropy": posterior.entropy()})
                action = allocate_dict(allocation)
        self._record(action["type"], time.perf_counter() - start)
        return action


# ----------------------------------------------------------------------------------- game loop

def play_game(state, players: dict[str, Any] | Any, *, max_steps: int = 500, strict: bool = True):
    """Drive a game to DONE. Each player only ever receives a deep copy of its own observation dict.

    Returns ``(final_state, log)`` where ``log`` lists ``(player_id, action_dict, accepted, error_code)``.
    With ``strict`` a rejected action raises ``RuntimeError``.
    """
    from ..game import GameEngine
    from ..game.observation import observation_for
    if not isinstance(players, dict):
        players = {players.player_id: players}
    log = []
    for _ in range(max_steps):
        if state.phase == "DONE":
            return state, log
        progressed = False
        for pid, player in players.items():
            obs = observation_for(state, pid)
            if not any(a in obs["legal_actions"] for a in ("commit", "allocate")):
                continue  # waiting for other players
            action = player.act(copy.deepcopy(obs))
            step = GameEngine.apply(state, pid, action)
            log.append((pid, action, step.accepted, step.error_code))
            if not step.accepted:
                if strict:
                    raise RuntimeError(f"{type(player).__name__} action rejected: {step.error_code} {step.message} {action}")
                fallback = {"type": "commit", "manifesto": [0.2] * NUM_SECTORS} if state.phase == "SURVEY" else {"type": "forfeit"}
                step = GameEngine.apply(state, pid, fallback)
            state = step.state
            progressed = True
            if state.phase == "DONE":
                break
        if not progressed:
            raise RuntimeError("no player can act")
    raise RuntimeError("game did not finish within max_steps")

