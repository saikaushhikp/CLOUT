"""P4: observation-only AI players play full engine games (spec §8, §6.6, §10.2)."""
import json
from decimal import Decimal

import pytest

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.game import GameEngine
from dumbocracy.game.observation import observation_for
from dumbocracy.players.base import Player
from dumbocracy.strategies.players import BayesPlanner, RandomPlayer, Top3Player, UniformPlayer, play_game
from tests.privacy_scan import assert_no_leaks

OBSERVATION_KEYS = set(observation_for(GameEngine.new_game(
    GameConfig(seed=1, device="cpu", electorate=ElectorateConfig(num_voters=1000)), game_id="k"), "p0"))


def _players(seed=0):
    return [UniformPlayer(), Top3Player(), RandomPlayer(seed=seed), BayesPlanner(seed=seed, num_worlds=4,
                                                                                  manifesto_worlds=4, world_voters=1000)]


class Spy:
    """Wraps a player and records exactly what it was handed."""

    def __init__(self, inner):
        self.inner = inner
        self.player_id = inner.player_id
        self.seen = []

    def act(self, observation):
        self.seen.append(observation)
        return self.inner.act(observation)


@pytest.fixture(scope="module")
def game2000():
    cfg = GameConfig(seed=77, device="cpu", electorate=ElectorateConfig(num_voters=2000))
    return cfg, GameEngine.new_game(cfg, game_id="p")


@pytest.mark.parametrize("index", range(4))
def test_player_completes_full_game_with_legal_actions(game2000, index):
    cfg, state0 = game2000
    player = _players(seed=3)[index]
    assert isinstance(player, Player)
    final, log = play_game(state0, player)  # strict: any rejected action raises
    assert final.phase == "DONE"
    assert all(accepted for _, _, accepted, _ in log)
    obs = observation_for(final, "p0")
    assert len(obs["allocations"]) == cfg.rounds
    assert obs["result"] is not None
    assert Decimal(obs["treasury"]) >= 0
    if isinstance(player, BayesPlanner):
        survey = [q for q in obs["queries"] if q["phase"] == "SURVEY"]
        assert 1 <= len(survey) <= player.num_queries
        assert sum(Decimal(q["cost"]) for q in obs["queries"]) <= player.survey_budget
        assert Decimal(obs["treasury"]) < Decimal("100")  # it actually campaigns
    if isinstance(player, UniformPlayer):
        assert obs["treasury"] == "500.00" and obs["manifesto"] == [0.2] * 5
    # R16: hires are permanent — every allocation used at most free + earlier hires + new hires
    hired = 0
    for a in obs["allocations"]:
        assert sum(a["workers_by_region"]) <= 10 + hired + a["extra_workers"]
        hired += a["extra_workers"]
    assert obs["workers"] == {"free": 10, "hired": hired, "available": 10 + hired}
    assert len(obs["round_reports"]) == cfg.rounds
    if isinstance(player, Top3Player):
        assert hired > 0
        later = obs["allocations"][-1]
        assert sum(later["workers_by_region"]) == 10 + hired   # earlier hires keep working in the last round


@pytest.mark.parametrize("index", range(4))
def test_players_complete_games_with_a_larger_treasury(index):
    """R17: with a 1000.00 treasury every player still plays only accepted actions."""
    cfg = GameConfig(seed=78, rounds=3, budget=Decimal("1000.00"), device="cpu",
                     electorate=ElectorateConfig(num_voters=1000))
    final, log = play_game(GameEngine.new_game(cfg, game_id="rich"), _players(seed=4)[index])
    assert final.phase == "DONE" and all(accepted for _, _, accepted, _ in log)
    obs = observation_for(final, "p0")
    assert obs["initial_budget"] == "1000.00" and Decimal(obs["treasury"]) >= 0


def test_bayes_planner_forced_queries_may_repeat_and_stay_fast(game2000):
    """R12: with the stop rules disabled it asks exactly k sampled queries, repeating triples when that is
    the most informative choice; every repeat is accepted and charged; decisions stay well under ~1 s."""
    _, state0 = game2000
    player = BayesPlanner(seed=2, num_queries=8, stop_entropy=None, min_gain_per_coin=None, round_query_gain=None,
                          survey_budget=Decimal("500"), num_worlds=4, manifesto_worlds=4, world_voters=1000)
    final, log = play_game(state0, player)
    obs = observation_for(final, "p0")
    survey = [q for q in obs["queries"] if q["phase"] == "SURVEY"]
    triples = [(q["sector_a"], q["sector_b"], q["region"]) for q in survey]
    assert len(survey) == 8 and all(ok for *_, ok, _ in log)
    assert len(set(triples)) < len(triples)                       # at least one repeat
    assert max(q["repeat_index"] for q in survey) >= 2
    assert all(q["cost"] == "5.00" for q in survey)
    query_latency = player.latency["query"]
    assert sum(query_latency) / len(query_latency) < 0.5


def test_bayes_planner_stops_surveying_when_information_per_coin_is_low(game2000):
    _, state0 = game2000
    player = BayesPlanner(seed=2, num_queries=None, stop_entropy=None, min_gain_per_coin=0.01, round_query_gain=None,
                          survey_budget=Decimal("500"), num_worlds=4, manifesto_worlds=4, world_voters=1000)
    final, _ = play_game(state0, player)
    survey = [q for q in observation_for(final, "p0")["queries"] if q["phase"] == "SURVEY"]
    assert 1 <= len(survey) <= 8                                   # stopped by the gain-per-coin rule


def test_players_only_receive_observation_dicts(game2000):
    cfg, state0 = game2000
    for player in _players(seed=1):
        spy = Spy(player)
        play_game(state0, spy)
        assert spy.seen
        for obs in spy.seen:
            assert type(obs) is dict and set(obs) == OBSERVATION_KEYS
            json.dumps(obs)  # plain JSON data only: no tensors, states or electorates
            assert_no_leaks(obs, cfg.electorate.num_voters)


def test_players_are_deterministic(game2000):
    _, state0 = game2000
    for make in (lambda: RandomPlayer(seed=5), lambda: BayesPlanner(seed=5, num_worlds=4, manifesto_worlds=4,
                                                                    world_voters=1000)):
        _, log_a = play_game(state0, make())
        _, log_b = play_game(state0, make())
        assert [a for _, a, _, _ in log_a] == [b for _, b, _, _ in log_b]


@pytest.mark.slow
def test_bayes_planner_beats_uniform_paired():
    """Paired over 10 seeds at N = 3000: BayesPlanner's mean expected share > UniformPlayer's."""
    diffs = []
    for seed in range(300, 310):
        cfg = GameConfig(seed=seed, device="cpu", electorate=ElectorateConfig(num_voters=3000))
        state0 = GameEngine.new_game(cfg, game_id=f"b{seed}")
        shares = {}
        for name, player in (("uniform", UniformPlayer()), ("bayes", BayesPlanner(seed=seed))):
            final, _ = play_game(state0, player)
            shares[name] = final.player("p0").result.expected_share
        diffs.append(shares["bayes"] - shares["uniform"])
    assert sum(diffs) / len(diffs) > 0.05
    assert sum(d > 0 for d in diffs) >= 9
