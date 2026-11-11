"""Approved library change for X14 (plan §9, D2): optional ``rollout="front"`` in ``strategies/planner.py``.

The default behaviour must stay bit for bit: the scores below were recorded from ``plan_round`` BEFORE the
change (same fixture as ``test_worlds_planner.py``) and are compared exactly (float hex)."""
from decimal import Decimal

import pytest

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.dynamics.campaign import Allocation
from dumbocracy.electorate import generate_electorate
from dumbocracy.game import GameEngine, observation_for
from dumbocracy.strategies.planner import allocation_cost, default_allocation, front_allocation, plan_round
from dumbocracy.strategies.worlds import public_sizes, sample_worlds

import torch

N = 2000
B = (0.3, 0.2, 0.2, 0.15, 0.15)
PLAYED = [Allocation((1, 1, 0, 0), (True, False, False, False)), Allocation((0, 1, 0, 0), (False, True, True, False))]
ASSIGN = torch.tensor([[0, 1, 2, 3], [5, 4, 3, 2], [2, 0, 5, 1]])

# recorded before the change (git d8d9c4e+uncommitted, 2026-11-11), CPU, float hex of plan_round scores
SNAPSHOT = {
    "even": {"best": [[0, 0, 0, 1], [False, False, True, True]], "scores": [
        "0x1.d36c300000000p-2", "0x1.d760560000000p-2", "0x1.daef880000000p-2", "0x1.d1fd6a0000000p-2",
        "0x1.cb8f1e0000000p-2", "0x1.db01460000000p-2", "0x1.daf3aa0000000p-2", "0x1.dff2220000000p-2",
        "0x1.dc9dba0000000p-2", "0x1.d512480000000p-2", "0x1.cdfe220000000p-2", "0x1.ce44360000000p-2",
        "0x1.d1dc960000000p-2", "0x1.d4f2fa0000000p-2", "0x1.d164c00000000p-2", "0x1.d81c1a0000000p-2",
        "0x1.d313720000000p-2", "0x1.d68f860000000p-2", "0x1.d72e100000000p-2", "0x1.cf64d80000000p-2",
        "0x1.cb495a0000000p-2"]},
    "none": {"best": [[0, 0, 0, 1], [False, False, True, True]], "scores": [
        "0x1.bc0a6e0000000p-2", "0x1.bff1be0000000p-2", "0x1.c5a5280000000p-2", "0x1.c609ae0000000p-2",
        "0x1.cb8f1e0000000p-2", "0x1.d1dee80000000p-2", "0x1.c77af20000000p-2", "0x1.d63e460000000p-2",
        "0x1.d2ba9e0000000p-2", "0x1.bec2100000000p-2", "0x1.c19cae0000000p-2", "0x1.c1afc00000000p-2",
        "0x1.c5daea0000000p-2", "0x1.c9efb80000000p-2", "0x1.c51f820000000p-2", "0x1.cbcdc60000000p-2",
        "0x1.c6e8280000000p-2", "0x1.cce4f00000000p-2", "0x1.c1d65e0000000p-2", "0x1.c35aba0000000p-2",
        "0x1.c7bca00000000p-2"]},
}


@pytest.fixture(scope="module")
def setup():
    config = GameConfig(seed=21, rounds=4, device="cpu", electorate=ElectorateConfig(num_voters=N))
    state = GameEngine.new_game(config, electorate=generate_electorate(config), game_id="w")
    actions = [{"type": "commit", "manifesto": list(B)}] + [
        {"type": "allocate", "hires_by_region": list(a.hires_by_region), "media_regions": list(a.media_regions)}
        for a in PLAYED]
    observations = []
    for action in actions:
        step = GameEngine.apply(state, "p0", action)
        assert step.accepted
        state = step.state
        observations.append(observation_for(state, "p0"))
    obs = observations[-1]
    return config, obs, sample_worlds(obs, ASSIGN, config, num_voters=1000, seed=5), observations[0]


@pytest.mark.parametrize("rollout", [None, "even", "none"])
def test_default_rollouts_bit_identical_to_pre_change(setup, rollout):
    config, obs, worlds, _ = setup
    kwargs = {} if rollout is None else {"rollout": rollout}
    a, d = plan_round(obs, worlds, obs["manifesto"], config, seed=7, **kwargs)
    snap = SNAPSHOT[rollout or "even"]
    assert [x.hex() for x in d["scores"]] == snap["scores"]
    assert [list(a.hires_by_region), list(a.media_regions)] == snap["best"]


def test_front_rollout_valid_and_differs(setup):
    config, obs, worlds, obs1 = setup
    # round 3 of 4: one continuation round, where "front" and "even" coincide by construction
    a, d = plan_round(obs, worlds, obs["manifesto"], config, seed=7, rollout="front")
    assert a in d["candidates"] and [x.hex() for x in d["scores"]] == SNAPSHOT["even"]["scores"]
    # round 1 of 4: three continuation rounds, "front" buys the worker money at once
    worlds1 = sample_worlds(obs1, ASSIGN, config, num_voters=1000, seed=5)
    _, d_even = plan_round(obs1, worlds1, obs1["manifesto"], config, seed=7)
    _, d_front = plan_round(obs1, worlds1, obs1["manifesto"], config, seed=7, rollout="front")
    assert d_front["candidates"] == d_even["candidates"] and d_front["scores"] != d_even["scores"]
    with pytest.raises(ValueError):
        plan_round(obs, worlds, obs["manifesto"], config, seed=7, rollout="late")


@pytest.mark.parametrize("budget", ["0.00", "19.99", "137.45", "500.00"])
@pytest.mark.parametrize("left", [1, 2, 4])
def test_front_allocation_affordable_and_front_loaded(setup, budget, left):
    config, obs, _, _ = setup
    sizes = public_sizes(obs)
    budget = Decimal(budget)
    first = front_allocation(sizes, budget, left, [0, 0, 0, 0], config, first=True)
    later = front_allocation(sizes, budget, left, [0, 0, 0, 0], config, first=False)
    even = default_allocation(sizes, budget, left, [0, 0, 0, 0], config)
    for a in (first, later):
        a.validate()
        assert allocation_cost(a, sizes, config) <= budget
    assert later.total_hires == 0
    assert first.total_hires >= even.total_hires
    assert first.media_regions == even.media_regions
