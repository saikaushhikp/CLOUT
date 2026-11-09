"""P5/R14: the post-game epistemic timeline on a full engine game (web_api_v2.md "Epistemic timeline (P5)")."""
import json
import time

import pytest

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.game import GameEngine
from dumbocracy.game.observation import observation_for
from dumbocracy.logic.report import decision_states, epistemic_timeline
from dumbocracy.logic.temporal import F, G, Atom, Pred, game_trace, holds, robustness
from dumbocracy.strategies.players import BayesPlanner, play_game
from tests.privacy_scan import assert_no_leaks

N = 2000
ROUND_KEYS = {"round", "treasury", "knowledge", "possible_victory", "guaranteed_victory", "knows_win_exists",
              "inevitable_defeat", "truth_victory_possible", "best_plan"}
BOOLS = ("possible_victory", "guaranteed_victory", "knows_win_exists", "inevitable_defeat", "truth_victory_possible")


@pytest.fixture(scope="module")
def finished():
    cfg = GameConfig(seed=5, device="cpu", electorate=ElectorateConfig(num_voters=N))
    state, _ = play_game(GameEngine.new_game(cfg, game_id="epi"),
                         BayesPlanner(seed=0, num_worlds=4, manifesto_worlds=4, world_voters=1600))
    start = time.perf_counter()
    timeline = epistemic_timeline(state, "p0", seed=0)
    return state, timeline, time.perf_counter() - start


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def test_timeline_matches_the_contract_schema(finished):
    state, tl, seconds = finished
    assert set(tl) == {"scope", "rounds", "actual", "headline"}
    assert set(tl["scope"]) == {"worlds_per_round", "policies_per_round", "world_voters", "note"}
    assert tl["scope"]["worlds_per_round"] == 16 and tl["scope"]["world_voters"] == 1600
    assert isinstance(tl["scope"]["policies_per_round"], int) and tl["scope"]["policies_per_round"] >= 2
    assert "finite" in tl["scope"]["note"] and "sample" in tl["scope"]["note"]
    assert [r["round"] for r in tl["rounds"]] == list(range(1, state.config.rounds + 1))
    for row in tl["rounds"]:
        assert set(row) == ROUND_KEYS
        assert isinstance(row["treasury"], str) and float(row["treasury"]) >= 0
        assert set(row["knowledge"]) == {"queries", "polls", "posterior_entropy", "consistent_worlds"}
        assert isinstance(row["knowledge"]["queries"], int) and row["knowledge"]["polls"] == row["round"]
        assert _is_num(row["knowledge"]["posterior_entropy"]) and 1 <= row["knowledge"]["consistent_worlds"] <= 16
        assert all(isinstance(row[k], bool) for k in BOOLS)
        assert set(row["best_plan"]) == {"label", "win_fraction", "mean_share"}
        assert isinstance(row["best_plan"]["label"], str) and 0 <= row["best_plan"]["win_fraction"] <= 1
        # logical consistency of the four knowledge properties
        assert row["inevitable_defeat"] == (not row["possible_victory"])
        assert (not row["guaranteed_victory"]) or row["knows_win_exists"]
        assert (not row["knows_win_exists"]) or row["possible_victory"]
        assert row["guaranteed_victory"] == (row["best_plan"]["win_fraction"] == 1.0)
    result = observation_for(state, "p0")["result"]
    assert tl["actual"] == {"victory": result["victory"], "vote_share": result["vote_share"],
                            "expected_share": result["expected_share"]}
    assert isinstance(tl["headline"], str) and tl["headline"].endswith("of the vote.")
    json.dumps(tl)                                                         # JSON-safe
    print(f"timeline at N={N}: {seconds:.1f}s")


def test_timeline_is_deterministic_per_seed(finished):
    state, tl, _ = finished
    assert epistemic_timeline(state, "p0", seed=0) == tl


def test_timeline_contains_no_hidden_fields(finished):
    state, tl, _ = finished
    assert_no_leaks(tl, N)
    text = json.dumps(tl).lower()
    for word in ("sigma", "tau", "archetype", "personalit", "assignment", "seed"):
        assert word not in text, word
    e = state.electorate
    for hidden in (e.archetype_of_region.tolist(), e.personality_of_region.tolist()):
        assert json.dumps(hidden) not in json.dumps(tl)


def test_timeline_requires_a_finished_game_and_known_player(finished):
    state, _, _ = finished
    fresh = GameEngine.new_game(state.config, game_id="early", electorate=state.electorate)
    with pytest.raises(ValueError):
        epistemic_timeline(fresh, "p0")
    with pytest.raises(KeyError):
        epistemic_timeline(state, "nobody")


def test_true_decision_states_match_player_observations(finished):
    state, _, _ = finished
    from dumbocracy.logic.knowledge import observation_at
    final = observation_for(state, "p0")
    states = decision_states(state, "p0")
    assert sorted(states) == list(range(1, state.config.rounds + 1))
    for t, s in states.items():
        obs = observation_for(s, "p0")
        rec = observation_at(final, t, stage="decision")
        for key in ("round", "treasury", "polls", "queries", "allocations", "media_airings"):
            assert obs[key] == rec[key], (t, key)


def test_temporal_trace_of_the_game(finished):
    state, tl, _ = finished
    trace = game_trace(observation_for(state, "p0"), tl)
    assert len(trace) == state.config.rounds + 1
    victory = tl["actual"]["victory"]
    assert holds(F(Atom("Victory")), trace) == victory
    lead_always = G(Pred("margin", 0.0))
    rho = robustness(lead_always, trace)
    if rho != 0:
        assert (rho > 0) == holds(lead_always, trace)
    assert [s["Viable"] for s in trace[:-1]] == [r["possible_victory"] for r in tl["rounds"]]
