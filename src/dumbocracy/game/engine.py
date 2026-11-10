"""Player-keyed game engine: phase machine, information, round resolution, election (spec §6–§9).

``SURVEY → ROUND(1..T) → DONE`` (the ELECTION is resolved automatically after round T).
``GameEngine.apply`` is a pure state transition: rejected actions return the *same* state object.

Round resolution follows the P8 round order (hires → one cascade hop → media → social step with workers
pinned → polls) and uses exactly the per-round generator keying of :func:`dumbocracy.simulate.simulate_world`
(``seeds.generator(name, t, player)`` with 0-based campaign round t for ``workers``/``cascade``/``media``
and ``seeds.generator("election", player)``), so an engine game reproduces ``simulate_world`` exactly.
"""
from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from ..config import GameConfig, resolve_device
from ..dynamics.campaign import Allocation, CampaignOutcome, WorkerState, campaign_round
from ..electorate import Electorate, generate_electorate
from ..information.knowledge import KnowledgeEntry
from ..information.survey import global_top3_pct, query_moe, run_poll, sample_query
from ..sectors import NUM_REGIONS, REGIONS, SECTORS
from ..simulate import resolve_campaign_round
from ..voting import run_election, similarity, vote_probability
from .actions import (
    Action, Allocate, Commit, ErrorCode, Forfeit, Query, Rejection, action_from_dict, allocation_cost, validate,
)
from .ledger import Ledger, money_str
from .state import GameState, PlayerState

ALIGNMENT_BINS = 5
# Vote-probability bin edges for the debrief network colouring (R15): <10% | 10–40% | 40–60% swing | 60–90% | >90%.
ALIGNMENT_EDGES = (0.1, 0.4, 0.6, 0.9)
NO_ALLOCATION = Allocate((0,) * NUM_REGIONS, (False,) * NUM_REGIONS)


@dataclass(frozen=True, eq=False)
class StepResult:
    """Outcome of :meth:`GameEngine.apply`. ``answer``/``round_report`` are JSON-safe dicts or None."""

    accepted: bool
    error_code: str | None
    message: str | None
    state: GameState
    answer: dict[str, Any] | None = None
    round_report: dict[str, Any] | None = None


# ------------------------------------------------------------------------------- hidden helpers

def _manifesto_tensor(state: GameState, player: PlayerState) -> torch.Tensor:
    return torch.tensor(player.manifesto, dtype=state.preferences.dtype, device=state.device)


def _true_share(state: GameState, v: torch.Tensor, manifesto: torch.Tensor) -> float:
    """Expected share E = mean_i P_i (spec §5), identical to ``simulate.current_share``."""
    voting = state.config.voting
    return float(vote_probability(similarity(v, manifesto), voting.sensitivity, voting.threshold).mean())


def alignment_bins(v: torch.Tensor, manifesto: torch.Tensor, sensitivity: float, threshold: float) -> torch.Tensor:
    """Each voter's probability of voting for the manifesto, P_i = σ(β(S_i − z)) (spec §5), binned at
    ALIGNMENT_EDGES into 0..4 (4 = >90% likely yes), uint8 [N] on CPU (R15). Equal similarity fifths were
    uninformative because almost all voters have S ∈ [0.6, 1]; vote-probability bins match the result."""
    prob = vote_probability(similarity(v, manifesto), sensitivity, threshold)
    edges = torch.tensor(ALIGNMENT_EDGES, dtype=prob.dtype, device=prob.device)
    return torch.bucketize(prob, edges, right=True).to(torch.uint8).cpu()


# ------------------------------------------------------------------------- information updates

def _log(player: PlayerState, kind: str, round: int, payload: dict[str, Any]) -> PlayerState:
    return replace(player, knowledge=player.knowledge.append(KnowledgeEntry.make(kind, round, payload)))


def _add_free_stat(state: GameState, round: int) -> GameState:
    """Spec §6.2: append the exact global top-3 statistic to every player's knowledge log."""
    stat = global_top3_pct(state.preferences)
    for player in state.players:
        state = state.with_player(_log(player, "free_stat", round, {"top3_pct": stat}))
    return state


def _add_poll(state: GameState, player: PlayerState, poll_round: int) -> PlayerState:
    """Spec §6.4: free poll keyed ``("poll", round, player)``; round T+1 = pre-election."""
    cfg = state.config
    gen = cfg.seeds().generator("poll", poll_round, player.player_id, device=state.device)
    size = min(cfg.survey.poll_size, state.electorate.num_voters)
    estimate, moe = run_poll(state.preferences, _manifesto_tensor(state, player), cfg.voting.sensitivity,
                             cfg.voting.threshold, size, gen)
    return _log(player, "poll", poll_round, {"estimate": estimate, "moe": moe, "sample_size": size})


def _snapshot(state: GameState, player: PlayerState) -> PlayerState:
    """Append hidden trajectory/alignment snapshots for the current v (debrief + graph reveal)."""
    b = _manifesto_tensor(state, player)
    return replace(player, share_trajectory=player.share_trajectory + (_true_share(state, state.preferences, b),),
                   alignment=player.alignment + (alignment_bins(state.preferences, b, state.config.voting.sensitivity,
                                                                 state.config.voting.threshold),))


def _start_round(state: GameState, round: int) -> GameState:
    """Enter ROUND ``round``: reset readiness/query pools, add free stat, poll per player (§6.2, §6.4)."""
    state = replace(state, phase="ROUND", round=round)
    state = _add_free_stat(state, round)
    for player in state.players:
        player = replace(player, ready=False, pending=None, query_pools={})
        state = state.with_player(_add_poll(state, player, round))
    return state.with_event("round_start", f"Round {round} of {state.config.rounds} begins")


# ------------------------------------------------------------------------------ action effects

def _apply_query(state: GameState, player: PlayerState, action: Query) -> tuple[GameState, dict[str, Any]]:
    """Spec §6.3 (R12): fresh 10% sample of the region; answer + MoE + pooled estimate for this phase/round.

    The sample is drawn from ``seeds.generator("survey", round, player, query_index)`` where
    ``query_index`` counts this player's previous queries in the game, so replays are exact.
    """
    cfg = state.config
    cost = cfg.survey.query_cost
    gen = cfg.seeds().generator("survey", state.round, player.player_id, player.query_count, device=state.device)
    yes, m, size = sample_query(state.preferences, state.electorate.region_of, SECTORS.index(action.sector_a),
                                SECTORS.index(action.sector_b), REGIONS.index(action.region),
                                cfg.survey.query_sample_fraction, gen)
    prev_yes, prev_m, repeats = player.query_pools.get(action.key, (0, 0, 0))
    pooled_yes, pooled_m = prev_yes + yes, prev_m + m
    pct = 100.0 * yes / m
    answer = {"round": state.round, "phase": state.phase, "sector_a": action.sector_a, "sector_b": action.sector_b,
              "region": action.region, "pct_b_over_a": pct, "sample_size": m, "moe": query_moe(yes, m, size),
              "pooled_pct": 100.0 * pooled_yes / pooled_m, "pooled_sample_size": pooled_m,
              "pooled_moe": query_moe(pooled_yes, pooled_m, fpc=False), "repeat_index": repeats + 1,
              "cost": money_str(cost)}
    pools = {**player.query_pools, action.key: (pooled_yes, pooled_m, repeats + 1)}
    player = replace(player, ledger=player.ledger.debit(cost), query_count=player.query_count + 1, query_pools=pools)
    player = _log(player, "query", state.round, answer)
    state = state.with_player(player).with_event(
        "query", f"Survey {action.sector_b} > {action.sector_a} in {action.region}: {pct:.1f}% "
                 f"± {answer['moe']:.1f} (sample {m})", player.player_id)
    return state, answer


def _apply_commit(state: GameState, player: PlayerState, action: Commit) -> GameState:
    manifesto = tuple(float(x) for x in action.manifesto)
    player = _log(replace(player, manifesto=manifesto, ready=True), "commit", 0, {"manifesto": list(manifesto)})
    state = state.with_player(player).with_event("commit", "Manifesto committed", player.player_id)
    if not all(p.ready for p in state.players):
        return state
    state = _start_round(state, 1)
    for p in state.players:
        state = state.with_player(_snapshot(state, p))
    return state


def _apply_allocate(state: GameState, player: PlayerState, action: Allocate, forfeit: bool = False) -> GameState:
    cfg = state.config
    cost = allocation_cost(action, state.sizes, cfg.workers.worker_cost, cfg.media.base_cost)
    record = {"hires_by_region": list(action.hires_by_region), "media_regions": list(action.media_regions),
              "cost": money_str(cost)}
    team = tuple(a + int(h) for a, h in zip(player.team_by_region, action.hires_by_region))
    player = replace(player, ledger=player.ledger.debit(cost), ready=True, pending=action,
                     allocations=player.allocations + (action,), team_by_region=team)   # P8: permanent workers
    player = _log(player, "allocation", state.round, record)
    message = f"Round {state.round} forfeited" if forfeit else f"Round {state.round} plan locked in (cost {money_str(cost)})"
    return state.with_player(player).with_event("forfeit" if forfeit else "allocate", message, player.player_id)


# ---------------------------------------------------------------------------- round resolution

def _campaign(state: GameState, player: PlayerState, exclude: torch.Tensor) -> CampaignOutcome:
    """Steps 1–3 of the P8 round order for one player, keyed exactly like ``simulate_world``.

    P_j is computed on ``state.preferences`` = v at the start of the round (before any hires)."""
    t = state.round - 1  # simulate_world's 0-based round index
    seeds = state.config.seeds()
    gens = {name: seeds.generator(name, t, player.player_id, device=state.device) for name in ("workers", "cascade", "media")}
    action = player.pending or NO_ALLOCATION
    allocation = Allocation(tuple(action.hires_by_region), tuple(action.media_regions))
    workers = player.workers or WorkerState.empty(state.electorate.num_voters, state.device)
    return campaign_round(state.electorate, allocation, workers, state.preferences, _manifesto_tensor(state, player),
                          player.media_airings, state.config.workers, state.config.media, state.config.voting, gens,
                          exclude=exclude)


def _round_report(round: int, action: Allocate, team: Sequence[int], outcome: CampaignOutcome,
                  election_held: bool) -> dict[str, Any]:
    return {"round": round, "cost": money_str(outcome.cost), "hires_by_region": [int(h) for h in action.hires_by_region],
            "team_by_region": [int(x) for x in team], "spreaders": outcome.spreaders, "converted": outcome.converted,
            "media_exposed": outcome.media_exposed, "election_held": election_held}


def _resolve_round(state: GameState) -> tuple[GameState, dict[str, dict[str, Any]]]:
    """P8 round order: every player's hires → cascade hop → media, then one synchronous social step
    (new workers and converts at b, workers pinned, zealots frozen) → next round or election."""
    cfg, e, t = state.config, state.electorate, state.round
    empty = torch.zeros(e.num_voters, dtype=torch.bool, device=state.device)
    pinned = [p.workers.pinned if p.workers is not None else empty for p in state.players]
    outcomes: list[CampaignOutcome] = []
    for i, player in enumerate(state.players):
        others = empty
        for j, mask in enumerate(pinned):
            if j != i:
                others = others | mask
        outcome = _campaign(state, player, others)
        pinned[i] = outcome.state.pinned
        outcomes.append(outcome)
    manifestos = torch.stack([_manifesto_tensor(state, p) for p in state.players])
    v = resolve_campaign_round(e, cfg, state.preferences, manifestos, outcomes)
    state = replace(state, preferences=v)
    final = t >= cfg.rounds
    reports: dict[str, dict[str, Any]] = {}
    for player, outcome, action in zip(state.players, outcomes, (p.pending or NO_ALLOCATION for p in state.players)):
        airings = tuple(a + int(m) for a, m in zip(player.media_airings, action.media_regions))
        report = _round_report(t, action, player.team_by_region, outcome, final)
        reports[player.player_id] = report
        player = _log(replace(player, media_airings=airings, workers=outcome.state), "round_report", t, report)
        state = state.with_player(_snapshot(state, player))
        state = state.with_event("round_resolved", f"Round {t} resolved: {outcome.hires} workers bought, "
                                 f"{outcome.spreaders} spreaders, {outcome.converted} voters converted, "
                                 f"{outcome.media_exposed} saw your media", player.player_id)
    if final:
        return _run_election(state), reports
    return _start_round(state, t + 1), reports


def _run_election(state: GameState) -> GameState:
    """Spec §5/§7: pre-election poll (round T+1), official election per player, then DONE."""
    cfg = state.config
    seeds = cfg.seeds()
    for player in state.players:
        player = _add_poll(state, player, cfg.rounds + 1)
        result = run_election(state.preferences, _manifesto_tensor(state, player), cfg.voting.sensitivity,
                              cfg.voting.threshold, seeds.generator("election", player.player_id, device=state.device))
        player = replace(_log(player, "result", cfg.rounds + 1, public_result(result)), result=result,
                         ready=False, pending=None)
        state = state.with_player(player).with_event(
            "election", f"Election held: {result.votes} of {result.num_voters} votes "
                        f"({'victory' if result.victory else 'defeat'})", player.player_id)
    return replace(state, phase="DONE", round=cfg.rounds)


def public_result(result) -> dict[str, Any] | None:
    """Contract ``result`` object (drops ``expected_sd``)."""
    if result is None:
        return None
    return {"votes": result.votes, "num_voters": result.num_voters, "vote_share": result.vote_share,
            "victory": result.victory, "expected_share": result.expected_share}


# ------------------------------------------------------------------------------------- engine

def _reject(state: GameState, rejection: Rejection) -> StepResult:
    return StepResult(False, rejection.code, rejection.message, state)


class GameEngine:
    """Stateless transition functions over immutable :class:`GameState` snapshots (spec §7–§8)."""

    @staticmethod
    def new_game(config: GameConfig, player_ids: Sequence[str] = ("p0",), *, game_id: str | None = None,
                 electorate: Electorate | None = None, device: torch.device | str | None = None) -> GameState:
        """Generate (or adopt) the electorate and open the SURVEY phase with free stat #0."""
        ids = tuple(player_ids)
        if not ids or len(set(ids)) != len(ids) or not all(isinstance(p, str) and p for p in ids):
            raise ValueError("player_ids must be distinct non-empty strings")
        if electorate is None:
            target = resolve_device(config.device) if device is None else torch.device(device)
            electorate = generate_electorate(config, device=target)
        players = tuple(PlayerState(pid, Ledger(config.budget)) for pid in ids)
        state = GameState(game_id or uuid.uuid4().hex[:12], config, electorate, electorate.census(),
                          electorate.preferences, "SURVEY", 0, ids, players)
        state = _add_free_stat(state, 0)
        return state.with_event("game_created", f"New game: {electorate.num_voters} voters, {config.rounds} rounds")

    @staticmethod
    def apply(state: GameState, player_id: str, action: Action | Mapping[str, Any]) -> StepResult:
        """Validate then apply one action; rejected actions return ``state`` itself, unchanged."""
        if not state.has_player(player_id):
            return _reject(state, Rejection(ErrorCode.UNKNOWN_PLAYER, f"unknown player {player_id!r}"))
        if isinstance(action, Mapping):
            try:
                action = action_from_dict(action)
            except (KeyError, TypeError, ValueError) as exc:
                return _reject(state, Rejection(ErrorCode.INVALID_ACTION, f"malformed action: {exc}"))
        player = state.player(player_id)
        rejection = validate(state, player, action)
        if rejection:
            return _reject(state, rejection)
        new_state, answer, reports = GameEngine._effect(state, player, action)
        new_state = replace(new_state, history=state.history + ((player_id, action),))
        return StepResult(True, None, None, new_state, answer, reports.get(player_id) if reports else None)

    @staticmethod
    def _effect(state: GameState, player: PlayerState, action: Action):
        if isinstance(action, Query):
            new_state, answer = _apply_query(state, player, action)
            return new_state, answer, None
        if isinstance(action, Commit):
            return _apply_commit(state, player, action), None, None
        forfeit = isinstance(action, Forfeit)
        new_state = _apply_allocate(state, player, NO_ALLOCATION if forfeit else action, forfeit)
        if all(p.ready for p in new_state.players):
            new_state, reports = _resolve_round(new_state)
            return new_state, None, reports
        return new_state, None, None

    @staticmethod
    def replay(config: GameConfig, actions: Iterable[tuple[str, Action | Mapping[str, Any]]],
               player_ids: Sequence[str] = ("p0",), *, electorate: Electorate | None = None,
               device: torch.device | str | None = None) -> GameState:
        """Re-apply recorded actions from a fresh game; raises if any action is rejected."""
        state = GameEngine.new_game(config, player_ids, game_id="replay", electorate=electorate, device=device)
        for player_id, action in actions:
            step = GameEngine.apply(state, player_id, action)
            if not step.accepted:
                raise ValueError(f"replay rejected {action!r}: {step.error_code} {step.message}")
            state = step.state
        return state
