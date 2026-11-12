"""S1: SmartPlayer (strategies/smart.py) and the opt-in ``rollout="taper"`` / ``region_values`` planner hooks.

With every option off, SmartPlayer must play exactly like BayesPlanner; each option must stay legal (affordable,
valid actions) and only read the observation dict."""
from decimal import Decimal

import pytest

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.game import GameEngine
from dumbocracy.game.observation import observation_for
from dumbocracy.strategies.planner import allocation_cost, plan_round, taper_allocation
from dumbocracy.strategies.players import BayesPlanner, play_game
from dumbocracy.strategies.smart import RECOMMENDED, SmartPlayer, front_media_candidates, smart_player
from dumbocracy.strategies.worlds import public_sizes

SMALL = dict(num_worlds=3, manifesto_worlds=3, world_voters=800)


@pytest.fixture(scope="module")
def game():
    cfg = GameConfig(seed=31, device="cpu", electorate=ElectorateConfig(num_voters=2000))
    return cfg, GameEngine.new_game(cfg, game_id="s")


def _actions(state, player):
    final, log = play_game(state, player)
    assert all(ok for *_, ok, _c in log)
    return [a for _pid, a, _ok, _c in log], observation_for(final, "p0")


def test_neutral_smart_player_equals_bayes_planner(game):
    _, state = game
    a, oa = _actions(state, BayesPlanner(seed=4, **SMALL))
    b, ob = _actions(state, SmartPlayer(seed=4, **SMALL))
    assert a == b
    assert oa["result"]["expected_share"] == ob["result"]["expected_share"]


@pytest.mark.parametrize("opts", [{"front_candidates": True}, {"rollout": "taper"}, {"plan_world_fraction": 0.5},
                                  {"manifesto_method": "admm"}, RECOMMENDED])
def test_each_option_plays_a_legal_game(game, opts):
    _, state = game
    actions, obs = _actions(state, SmartPlayer(seed=4, **{**SMALL, **opts}))
    assert obs["phase"] == "DONE" and obs["result"] is not None
    assert Decimal(obs["treasury"]) >= 0
    assert sum(1 for a in actions if a["type"] == "allocate") == obs["rounds"]


def test_plan_world_voters_scales_with_n():
    p = SmartPlayer(world_voters=1600, plan_world_fraction=0.4, plan_world_cap=20_000)
    assert p.plan_world_voters(2_000) == 1600          # never below world_voters (then min with N in sampling)
    assert p.plan_world_voters(10_000) == 4000
    assert p.plan_world_voters(1_000_000) == 20_000    # capped
    assert SmartPlayer(world_voters=1600).plan_world_voters(10_000) == 1600  # option off


def test_smart_player_factory_and_validation():
    s = smart_player(seed=3)
    assert s.seed == 3 and all(getattr(s, k) == v for k, v in RECOMMENDED.items())
    with pytest.raises(ValueError):
        SmartPlayer(manifesto_method="nope")
    with pytest.raises(ValueError):
        SmartPlayer(plan_world_fraction=0.0)


def test_taper_allocation_front_loads_media_and_spends_last_round():
    cfg = GameConfig()
    sizes = [2500, 2500, 2500, 2500]
    budget = Decimal("400.00")
    a = taper_allocation(sizes, budget, 4, [0, 0, 0, 0], cfg)
    assert a.total_hires == 0 and allocation_cost(a, sizes, cfg) <= budget
    later = taper_allocation(sizes, budget, 2, [1, 1, 1, 1], cfg)
    assert allocation_cost(later, sizes, cfg) <= budget and later.total_hires == 0
    last = taper_allocation(sizes, Decimal("130.00"), 1, [0, 0, 0, 0], cfg)
    assert allocation_cost(last, sizes, cfg) <= Decimal("130.00")
    assert Decimal("130.00") - allocation_cost(last, sizes, cfg) < cfg.workers.worker_cost  # nothing useful left


def test_front_candidates_affordable_and_shaped(game):
    cfg, state = game
    step = GameEngine.apply(state, "p0", {"type": "commit", "manifesto": [0.2] * 5})
    obs = observation_for(step.state, "p0")
    cands = front_media_candidates(obs, cfg, [1.0, 3.0, 2.0, 0.5], [1.0, 1.0, 2.0, 2.0])
    sizes = public_sizes(obs)
    assert cands and len(cands) == len(set(cands))
    for a in cands:
        a.validate()
        assert allocation_cost(a, sizes, cfg) <= Decimal(obs["treasury"])
        assert sum(1 for h in a.hires_by_region if h) <= 1           # workers in one region only
        assert any(a.media_regions)
    hired = {i for a in cands for i, h in enumerate(a.hires_by_region) if h}
    assert hired == {1, 2}                                            # the two best regions by worker value


def test_plan_round_rejects_unknown_rollout(game):
    cfg, state = game
    obs = observation_for(GameEngine.apply(state, "p0", {"type": "commit", "manifesto": [0.2] * 5}).state, "p0")
    with pytest.raises(ValueError):
        plan_round(obs, [], obs["manifesto"], cfg, rollout="bogus")
