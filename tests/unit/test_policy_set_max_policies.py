"""Approved library change for X16 (plan §9, D2): optional ``max_policies`` in ``logic/epistemic.policy_set``
(threaded through ``logic/report.timeline_with_diagnostics`` / ``epistemic_timeline``).

The default behaviour must stay bit for bit: the plan set, the outcome matrix (float hex) and the SHA-256 of a whole
timeline below were recorded BEFORE the change (git d8d9c4e+uncommitted, 2026-11-11, CPU) and are compared exactly.
"""
import hashlib
import json
from dataclasses import replace
from decimal import Decimal

import pytest

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.game import GameEngine
from dumbocracy.game.observation import observation_for
from dumbocracy.logic.epistemic import evaluate, policy_set
from dumbocracy.logic.report import timeline_with_diagnostics
from dumbocracy.logic.worlds import consistent_worlds
from dumbocracy.strategies.players import Top3Player, play_game, public_config

# [label, first hires, first media, weights, pace] per plan, recorded before the change
PLANS = [
    ["No spend (earlier workers keep spreading)", [0, 0, 0, 0], [False, False, False, False], [158.0, 309.0, 153.0, 180.0], "0"],
    ["1 workers in S, no media, 20% pace", [0, 1, 0, 0], [False, False, False, False], [0.0, 1.0, 0.0, 0.0], "0.20"],
    ["2 workers (0/1/0/1), no media, 40% pace", [0, 1, 0, 1], [False, False, False, False], [0.0, 1.0, 0.0, 1.0], "0.40"],
    ["5 workers (1/2/1/1), no media, 100% pace", [1, 2, 1, 1], [False, False, False, False], [1.0, 2.0, 1.0, 1.0], "1.00"],
    ["10 workers (2/4/2/2), no media, 200% pace", [2, 4, 2, 2], [False, False, False, False], [2.0, 4.0, 2.0, 2.0], "2.00"],
    ["12 workers (2/5/2/3), no media, 240% pace", [2, 5, 2, 3], [False, False, False, False], [2.0, 5.0, 2.0, 3.0], "2.40"],
    ["1 workers in S, media in S, 97% pace", [0, 1, 0, 0], [False, True, False, False], [0.0, 1.0, 0.0, 0.0], "0.97"],
    ["no workers, media in S, 77% pace", [0, 0, 0, 0], [False, True, False, False], [158.0, 309.0, 153.0, 180.0], "0.77"],
    ["no workers, media in E, 38% pace", [0, 0, 0, 0], [False, False, True, False], [158.0, 309.0, 153.0, 180.0], "0.38"],
    ["5 workers in N, no media, 100% pace", [5, 0, 0, 0], [False, False, False, False], [5.0, 0.0, 0.0, 0.0], "1.00"],
    ["3 workers in N, media in N, 100% pace", [3, 0, 0, 0], [True, False, False, False], [3.0, 0.0, 0.0, 0.0], "1.00"],
    ["5 workers in S, no media, 100% pace", [0, 5, 0, 0], [False, False, False, False], [0.0, 5.0, 0.0, 0.0], "1.00"],
    ["5 workers in E, no media, 100% pace", [0, 0, 5, 0], [False, False, False, False], [0.0, 0.0, 5.0, 0.0], "1.00"],
    ["3 workers in E, media in E, 98% pace", [0, 0, 3, 0], [False, False, True, False], [0.0, 0.0, 3.0, 0.0], "0.98"],
    ["5 workers in W, no media, 100% pace", [0, 0, 0, 5], [False, False, False, False], [0.0, 0.0, 0.0, 5.0], "1.00"],
    ["2 workers in W, media in W, 85% pace", [0, 0, 0, 2], [False, False, False, True], [0.0, 0.0, 0.0, 2.0], "0.85"],
]

# evaluate(4 worlds × all plans) as float hex, recorded before the change
VALUES = [
    ["0x1.1501f60000000p-1", "0x1.49f5340000000p-1", "0x1.4e68280000000p-1", "0x1.5294240000000p-1", "0x1.5168a60000000p-1", "0x1.521a400000000p-1", "0x1.4b131c0000000p-1", "0x1.28da140000000p-1", "0x1.2f4c8c0000000p-1", "0x1.5103380000000p-1", "0x1.61d1420000000p-1", "0x1.479fd40000000p-1", "0x1.5376ae0000000p-1", "0x1.56b3280000000p-1", "0x1.5479ae0000000p-1", "0x1.57caca0000000p-1"],
    ["0x1.180efc0000000p-1", "0x1.5a79680000000p-1", "0x1.5928120000000p-1", "0x1.5483060000000p-1", "0x1.5353080000000p-1", "0x1.544b7e0000000p-1", "0x1.5b8f420000000p-1", "0x1.2b4bb00000000p-1", "0x1.343b4a0000000p-1", "0x1.5a329c0000000p-1", "0x1.6d43f80000000p-1", "0x1.5369660000000p-1", "0x1.62a5b80000000p-1", "0x1.6a1d280000000p-1", "0x1.5cd5980000000p-1", "0x1.629fde0000000p-1"],
    ["0x1.1c487a0000000p-1", "0x1.579d6e0000000p-1", "0x1.5cfaa20000000p-1", "0x1.5cf04a0000000p-1", "0x1.5bd5780000000p-1", "0x1.5a44060000000p-1", "0x1.5d1f920000000p-1", "0x1.2f92180000000p-1", "0x1.31eab60000000p-1", "0x1.616e100000000p-1", "0x1.69e4c00000000p-1", "0x1.596e1e0000000p-1", "0x1.6509220000000p-1", "0x1.6a4aa40000000p-1", "0x1.5cda800000000p-1", "0x1.68417a0000000p-1"],
    ["0x1.14b47c0000000p-1", "0x1.585cd80000000p-1", "0x1.5a760a0000000p-1", "0x1.5d34b80000000p-1", "0x1.59c44a0000000p-1", "0x1.59c22e0000000p-1", "0x1.59911e0000000p-1", "0x1.2274500000000p-1", "0x1.3802ca0000000p-1", "0x1.6345700000000p-1", "0x1.65b31a0000000p-1", "0x1.585f5c0000000p-1", "0x1.5a13580000000p-1", "0x1.6aa8ac0000000p-1", "0x1.5e0f0a0000000p-1", "0x1.67c4740000000p-1"],
]

# sha256(json.dumps(timeline, sort_keys=True)) of a Top-3 game (N = 1000, T = 3; 4 worlds of 600 voters, seed 2)
TIMELINE_SHA = "df499737329ae3d7f80d1580abedf7ba7294a8f911fa395385cc6acd874a316d"


@pytest.fixture(scope="module")
def setup():
    cfg = GameConfig(seed=13, device="cpu", budget=Decimal("500.00"), electorate=ElectorateConfig(num_voters=800))
    state = GameEngine.new_game(cfg, game_id="fixture")
    m = state.electorate.preferences.mean(0).double()
    m = m / m.sum()
    state = GameEngine.apply(state, "p0", {"type": "commit", "manifesto": [float(x) for x in m]}).state
    obs = observation_for(state, "p0")
    pcfg = public_config(obs, replace(cfg, device="cpu"))
    return obs, pcfg


def _rows(plans):
    return [[p.label, list(p.first.hires_by_region), list(p.first.media_regions), list(p.weights), str(p.pace)]
            for p in plans]


def test_default_policy_set_and_values_bit_identical_to_pre_change(setup):
    obs, pcfg = setup
    for plans in (policy_set(obs, pcfg), policy_set(obs, pcfg, None)):
        assert _rows(plans) == PLANS
    ws = consistent_worlds(obs, pcfg, 4, world_voters=800, seed=3)
    values = evaluate(ws, policy_set(obs, pcfg), obs, pcfg, seed=5)
    assert [[x.hex() for x in row] for row in values.tolist()] == VALUES


def test_default_timeline_bit_identical_to_pre_change():
    cfg = GameConfig(seed=31, rounds=3, device="cpu", electorate=ElectorateConfig(num_voters=1000))
    final, _ = play_game(GameEngine.new_game(cfg, game_id="t"), Top3Player())
    tl, _ = timeline_with_diagnostics(final, "p0", worlds=4, world_voters=600, seed=2)
    assert hashlib.sha256(json.dumps(tl, sort_keys=True).encode()).hexdigest() == TIMELINE_SHA


def test_max_policies_is_a_prefix_and_columns_match(setup):
    obs, pcfg = setup
    full = policy_set(obs, pcfg)
    assert len(full) > 8
    ws = consistent_worlds(obs, pcfg, 4, world_voters=800, seed=3)
    v_full = evaluate(ws, full, obs, pcfg, seed=5)
    for k in (1, 8, len(full), len(full) + 5):
        sub = policy_set(obs, pcfg, k)
        assert _rows(sub) == _rows(full[:k])
        # common random numbers: a plan's outcome does not depend on the other plans in the set
        assert evaluate(ws, sub, obs, pcfg, seed=5).equal(v_full[:, :len(sub)])
    with pytest.raises(ValueError):
        policy_set(obs, pcfg, 0)
