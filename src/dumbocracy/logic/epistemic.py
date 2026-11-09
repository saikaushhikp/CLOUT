"""Epistemic properties over sampled worlds × a finite policy set (spec §10.2; P5; web_api_v2.md "Epistemic timeline").

Setting (lecture "Epistemic logic"): W_t is a finite sample of the player's information set at round t
(:mod:`.worlds`), Π_t a finite, declared set of campaign plans for the remaining rounds. The outcome matrix
``V[w, π]`` is the expected vote share on election day when plan π is played from the decision point in
world w (spec I5: logic is evaluated on the deterministic expected share). ``Victory(w, π) ⇔ V[w, π] > 0.5``.

Knowledge and possibility over W (Kripke semantics, ~_p an S5 equivalence):

* ``K(φ)`` — φ holds in every world of W (axis 0 of a ``[W]`` or ``[W, Π]`` Boolean tensor);
* ``M(φ)`` — φ holds in some world; duality ``M φ ≡ ¬K¬φ`` (tested).

Properties (all scoped to the finite W_t and Π_t — claims are "over the sampled consistent worlds"):

* ``possible_victory``    ∃π ∃w Victory  = M ◇Victory (the player cannot rule winning out);
* ``guaranteed_victory``  ∃π ∀w Victory  = ∃π K(Victory under π) — knowing HOW (de re: one specific plan
  is known to win);
* ``knows_win_exists``    ∀w ∃π Victory  = K(∃π Victory) — knowing THAT (de dicto: in every world some plan
  wins, maybe a different one per world). ∃π∀w ⇒ ∀w∃π, not conversely;
* ``inevitable_defeat``   ∀π ∀w ¬Victory = K □¬Victory ≡ ¬possible_victory;
* ``truth_victory_possible`` — ∃π Victory in the TRUE world (ground truth, used for reporting only).

Policy set Π_t (:func:`policy_set`): every allocation of
:func:`dumbocracy.strategies.planner.candidate_allocations` at round t becomes a *stationary plan*: it is
played as is in round t; in every later round s the same pattern is re-applied with its spending re-capped
to the remaining money: per-round money = min(treasury_s, pace · treasury_s / rounds_left_s) where
pace = cost(a) / (treasury_t / rounds_left_t); the plan's media regions are kept while affordable within
that money (in plan order), the rest buys new hires, and all workers are split with the plan's region
weights. Hires are permanent (R16): a plan's later rounds assign free + every hire made so far (before the
decision point and by the plan itself) + that round's new hires. Plus "No spend" (all available workers by
region size, nothing bought).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Any

import torch

from ..config import GameConfig, SeedStreams
from ..dynamics.campaign import Allocation, media_cost
from ..sectors import NUM_REGIONS, REGIONS
from ..strategies.planner import candidate_allocations, round_cap, rounds_left, split_by_weight
from ..strategies.worlds import WorldBatch, hired_workers, public_sizes
from .worlds import WorldSet

__all__ = ["K", "M", "Plan", "evaluate", "guaranteed_victory", "inevitable_defeat", "knows_win_exists",
           "policy_set", "possible_victory", "properties", "simulate_plans", "truth_victory_possible",
           "victory_matrix"]

CENT = Decimal("0.01")
THRESHOLD = 0.5


# ------------------------------------------------------------------------------- modal operators

def _bool(phi: Any) -> torch.Tensor:
    t = torch.as_tensor(phi)
    if t.dtype != torch.bool:
        raise TypeError("phi must be a Boolean tensor (worlds on axis 0)")
    if t.dim() == 0 or t.shape[0] == 0:
        raise ValueError("phi needs a non-empty world axis 0")
    return t


def K(phi: Any) -> torch.Tensor:
    """Knowledge: φ true in **every** world (reduces axis 0). ``[W] → scalar``, ``[W, Π] → [Π]``."""
    return _bool(phi).all(dim=0)


def M(phi: Any) -> torch.Tensor:
    """Possibility: φ true in **some** world (reduces axis 0); ``M φ ≡ ¬K¬φ``."""
    return _bool(phi).any(dim=0)


def victory_matrix(values: torch.Tensor, threshold: float = THRESHOLD) -> torch.Tensor:
    """``V[w, π] > threshold`` (strict: a tie is not a win)."""
    return torch.as_tensor(values) > threshold


def possible_victory(win: torch.Tensor) -> bool:
    """∃π ∃w Victory."""
    return bool(M(win).any())


def guaranteed_victory(win: torch.Tensor) -> bool:
    """∃π ∀w Victory (knowing how, de re)."""
    return bool(K(win).any())


def knows_win_exists(win: torch.Tensor) -> bool:
    """∀w ∃π Victory (knowing that, de dicto) = K(∃π Victory)."""
    return bool(K(_bool(win).any(dim=1)))


def inevitable_defeat(win: torch.Tensor) -> bool:
    """∀π ∀w ¬Victory = K □¬Victory."""
    return bool(K(~_bool(win)).all())


def properties(values: torch.Tensor, labels: Sequence[str] | None = None, threshold: float = THRESHOLD) -> dict[str, Any]:
    """The four knowledge properties of a ``[W, Π]`` share matrix plus witnesses and the best plan.

    ``best`` = the plan winning in the largest fraction of worlds (ties: higher mean share); a
    ``guaranteed_witness`` is a plan winning in every world, a ``defeat_counter_world`` (when no plan is
    guaranteed) a world index where the best plan loses.
    """
    values = torch.as_tensor(values, dtype=torch.float64)
    win = victory_matrix(values, threshold)
    labels = list(labels) if labels is not None else [str(i) for i in range(values.shape[1])]
    win_frac = win.to(torch.float64).mean(0)
    mean = values.mean(0)
    order = sorted(range(values.shape[1]), key=lambda j: (-float(win_frac[j]), -float(mean[j]), j))
    best = order[0]
    known = K(win)
    out = {
        "possible_victory": possible_victory(win),
        "guaranteed_victory": guaranteed_victory(win),
        "knows_win_exists": knows_win_exists(win),
        "inevitable_defeat": inevitable_defeat(win),
        "num_worlds": int(values.shape[0]),
        "num_policies": int(values.shape[1]),
        "best_index": best,
        "best_plan": {"label": labels[best], "win_fraction": float(win_frac[best]), "mean_share": float(mean[best])},
        "guaranteed_witness": labels[int(torch.nonzero(known)[0])] if bool(known.any()) else None,
        "defeat_counter_world": None if bool(known.any()) else int(torch.nonzero(~win[:, best])[0]),
    }
    return out


# ------------------------------------------------------------------------------------ policy set

@dataclass(frozen=True)
class Plan:
    """A stationary plan for the remaining rounds (module docstring)."""

    label: str
    first: Allocation                    # played as is in the decision round
    media: tuple[bool, bool, bool, bool]
    weights: tuple[float, float, float, float]
    pace: Decimal                        # per-round money as a multiple of the even cap

    def allocation(self, sizes: Sequence[int], treasury: Decimal, left: int, config: GameConfig,
                   hired: int = 0) -> Allocation:
        """The plan's allocation for a later round with ``treasury`` money, ``left`` rounds to go and
        ``hired`` permanent hires already made (R16)."""
        if self.pace <= 0:
            return Allocation(split_by_weight(config.workers.free_workers + hired, self.weights))
        money = min(treasury, (round_cap(treasury, left) * self.pace).quantize(CENT, rounding=ROUND_FLOOR))
        media = [False] * NUM_REGIONS
        for r in range(NUM_REGIONS):
            if self.media[r]:
                trial = list(media)
                trial[r] = True
                if media_cost(sizes, trial, config.media.base_cost) <= money:
                    media = trial
        left_money = money - media_cost(sizes, media, config.media.base_cost)
        extra = max(0, int(left_money // config.workers.extra_worker_cost)) if left_money > 0 else 0
        workers = split_by_weight(config.workers.free_workers + hired + extra, self.weights)
        return Allocation(workers, extra, tuple(media))  # type: ignore[arg-type]


def _cost(a: Allocation, sizes: Sequence[int], config: GameConfig) -> Decimal:
    return a.extra_workers * config.workers.extra_worker_cost + media_cost(sizes, a.media_regions, config.media.base_cost)


def _label(a: Allocation, sizes: Sequence[int], pace: Decimal) -> str:
    aired = [REGIONS[r] for r in range(NUM_REGIONS) if a.media_regions[r]]
    workers = list(a.workers_by_region)
    total = sum(workers)
    nonzero = [REGIONS[r] for r in range(NUM_REGIONS) if workers[r] > 0]
    if total and len(nonzero) == 1:
        where = f"workers focused on {nonzero[0]}"
    else:
        where = "workers by region size"
    media = f"media in {'+'.join(aired)}" if aired else "no media"
    spend = "no spending" if pace <= 0 else f"{int((pace * 100).to_integral_value())}% pace"
    return f"{where}, {media}, {spend}"


def policy_set(observation_t: dict[str, Any], config: GameConfig) -> list[Plan]:
    """Π_t: "No spend" + one stationary plan per candidate allocation at the decision point (labels unique)."""
    sizes = public_sizes(observation_t)
    treasury = Decimal(str(observation_t["treasury"]))
    cap = round_cap(treasury, rounds_left(observation_t))
    by_size = tuple(float(s) for s in sizes)
    no_spend = Allocation(split_by_weight(config.workers.free_workers + hired_workers(observation_t), by_size))
    plans = [Plan("No spend (available workers by region size)", no_spend, (False,) * NUM_REGIONS, by_size, Decimal(0))]  # type: ignore[arg-type]
    seen = {no_spend}
    for a in candidate_allocations(observation_t, config):
        if a in seen:
            continue
        seen.add(a)
        cost = _cost(a, sizes, config)
        pace = (cost / cap).quantize(Decimal("0.01")) if cap > 0 else Decimal(0)
        weights = tuple(float(w) for w in a.workers_by_region) if sum(a.workers_by_region) > 0 else by_size
        plans.append(Plan(_label(a, sizes, pace), a, tuple(a.media_regions), weights, pace))  # type: ignore[arg-type]
    labels: dict[str, int] = {}
    out = []
    for p in plans:
        n = labels.get(p.label, 0)
        labels[p.label] = n + 1
        out.append(p if n == 0 else Plan(f"{p.label} #{n + 1}", p.first, p.media, p.weights, p.pace))
    return out


# ----------------------------------------------------------------------------------- simulation

def simulate_plans(batch: WorldBatch, current: torch.Tensor, policies: Sequence[Plan], observation_t: dict[str, Any],
                   config: GameConfig, *, seed: int = 0, trajectory: bool = False):
    """Play every plan from the decision point to the election in every world of ``batch`` (batched, CRN).

    Returns ``V [W, Π]`` (float64 expected shares at the election) and, with ``trajectory``, also the
    share after every remaining round ``[W, Π, rounds_left]``.
    """
    sizes = public_sizes(observation_t)
    treasury = Decimal(str(observation_t["treasury"]))
    left = rounds_left(observation_t)
    t0 = int(observation_t["round"])
    C = len(policies)
    if C == 0:
        raise ValueError("empty policy set")
    b = batch.manifesto_tensor(observation_t["manifesto"])
    airings0 = [int(observation_t.get("media_airings", {}).get(r, 0)) for r in REGIONS]
    airings = torch.tensor([airings0] * C, dtype=torch.int64, device=batch.device)
    budgets = [treasury] * C
    hires = [hired_workers(observation_t)] * C        # R16: permanent hires, accumulated per plan
    v = current.to(batch.device).unsqueeze(0).expand(C, -1, -1).contiguous()
    streams = SeedStreams(int(seed))
    shares_by_round = []
    for step in range(left):
        if step == 0:
            plans = [p.first for p in policies]
        else:
            plans = [p.allocation(sizes, budgets[c], left - step, config, hires[c]) for c, p in enumerate(policies)]
        for c, a in enumerate(plans):
            budgets[c] = budgets[c] - _cost(a, sizes, config)
            hires[c] += a.extra_workers
            if budgets[c] < 0:
                raise ValueError(f"plan {policies[c].label!r} overspends")
        gen = streams.generator("worlds", "logic-eval", t0 + step, device=batch.device)
        v = batch.step(v, plans, airings, b, gen)
        airings = airings + torch.tensor([p.media_regions for p in plans], dtype=torch.int64, device=batch.device)
        if trajectory or step == left - 1:
            shares_by_round.append(batch.shares(v, b).T.to(torch.float64).cpu())       # [W, Π]
    values = shares_by_round[-1]
    if trajectory:
        return values, torch.stack(shares_by_round, dim=2)
    return values


def evaluate(worlds: WorldSet, policies: Sequence[Plan], observation_t: dict[str, Any], config: GameConfig, *,
             seed: int = 0, trajectory: bool = False):
    """Outcome matrix ``V[w, π]`` over the sampled consistent worlds (see :func:`simulate_plans`)."""
    return simulate_plans(worlds.batch, worlds.current, policies, observation_t, config, seed=seed, trajectory=trajectory)


def truth_victory_possible(true_state, policies: Sequence[Plan], *, player_id: str = "p0",
                           observation_t: dict[str, Any] | None = None, seed: int = 0,
                           return_shares: bool = False):
    """∃π ∈ Π: Victory in the TRUE world — ground truth for reporting, never part of the player's knowledge.

    ``true_state`` is the engine :class:`~dumbocracy.game.state.GameState` at the decision point (its hidden
    electorate and current preferences are used); ``observation_t`` (default: the engine's observation of
    that state) supplies the public plan inputs (treasury, airings, round).
    """
    from ..game.observation import observation_for
    obs = observation_t or observation_for(true_state, player_id)
    batch = WorldBatch([true_state.electorate], true_state.config)
    shares = simulate_plans(batch, true_state.preferences, policies, obs, true_state.config, seed=seed)[0]   # [Π]
    possible = bool((shares > THRESHOLD).any())
    return (possible, shares) if return_shares else possible
