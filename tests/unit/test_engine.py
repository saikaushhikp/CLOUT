import json
import time
from decimal import ROUND_HALF_UP, Decimal

import pytest
import torch

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.dynamics.campaign import Allocation
from dumbocracy.electorate import generate_electorate
from dumbocracy.game import Allocate, Commit, ErrorCode, Forfeit, GameEngine, Query, observation_for
from dumbocracy.simulate import simulate_world
from tests.privacy_scan import assert_no_leaks

N = 3000
UNIFORM = (0.2, 0.2, 0.2, 0.2, 0.2)


@pytest.fixture(scope="module")
def config():
    return GameConfig(seed=11, rounds=3, device="cpu", electorate=ElectorateConfig(num_voters=N))


@pytest.fixture(scope="module")
def electorate(config):
    return generate_electorate(config)


@pytest.fixture
def fresh(config, electorate):
    return GameEngine.new_game(config, electorate=electorate, game_id="g")


def ok(state, action, player="p0"):
    step = GameEngine.apply(state, player, action)
    assert step.accepted, (step.error_code, step.message)
    return step


def rejected(state, action, code, player="p0"):
    before_obs = json.dumps(observation_for(state, player))
    step = GameEngine.apply(state, player, action)
    assert not step.accepted and step.error_code == code, (step.error_code, step.message)
    assert step.state is state                                       # atomic: same object
    assert json.dumps(observation_for(state, player)) == before_obs  # nothing changed
    return step


def committed(state):
    return ok(state, Commit(UNIFORM)).state


# ------------------------------------------------------------------------------- phase machine

def test_phase_machine_and_illegal_phase_rejections(fresh, config):
    s = fresh
    assert s.phase == "SURVEY" and s.round == 0
    assert observation_for(s, "p0")["legal_actions"] == ["query", "commit"]
    rejected(s, Allocate((1, 0, 0, 0)), ErrorCode.INVALID_PHASE)
    rejected(s, Forfeit(), ErrorCode.INVALID_PHASE)
    s = committed(s)
    assert s.phase == "ROUND" and s.round == 1
    assert observation_for(s, "p0")["legal_actions"] == ["query", "allocate", "forfeit"]
    rejected(s, Commit(UNIFORM), ErrorCode.INVALID_PHASE)
    for t in range(1, config.rounds + 1):
        assert s.phase == "ROUND" and s.round == t
        s = ok(s, Forfeit()).state
    assert s.phase == "DONE" and s.round == config.rounds
    obs = observation_for(s, "p0")
    assert obs["legal_actions"] == [] and obs["result"] is not None
    for action in (Query("Edu", "Env", "N"), Commit(UNIFORM), Allocate((0, 0, 0, 0)), Forfeit()):
        rejected(s, action, ErrorCode.TERMINAL_STATE)


def test_unknown_player_and_malformed_actions(fresh):
    assert GameEngine.apply(fresh, "p9", Forfeit()).error_code == ErrorCode.UNKNOWN_PLAYER
    rejected(fresh, {"type": "dance"}, ErrorCode.INVALID_ACTION)
    rejected(fresh, {"type": "query", "sector_a": "Edu"}, ErrorCode.INVALID_ACTION)


@pytest.mark.parametrize("manifesto", [(0.25,) * 4, (0.5, 0.5, 0.1, -0.1, 0.0), (0.3, 0.2, 0.2, 0.2, 0.2),
                                       (float("nan"), 0.25, 0.25, 0.25, 0.25), ("a", 0.25, 0.25, 0.25, 0.25), 0.2])
def test_invalid_manifesto_rejected(fresh, manifesto):
    rejected(fresh, Commit(manifesto), ErrorCode.INVALID_MANIFESTO)


@pytest.mark.parametrize("action", [Allocate((1, 0, 0)), Allocate((-1, 0, 0, 0)), Allocate((11, 0, 0, 0)),
                                    Allocate((12, 0, 0, 0), 1), Allocate((True, 0, 0, 0)), Allocate((1, 0, 0, 0), -1),
                                    Allocate((1, 0, 0, 0), 0, (1, 0, 0, 0)), Allocate((1.0, 0, 0, 0))])
def test_invalid_allocation_rejected(fresh, action):
    rejected(committed(fresh), action, ErrorCode.INVALID_ALLOCATION)


@pytest.mark.parametrize("query", [Query("Edu", "Edu", "N"), Query("Edu", "Foo", "N"), Query("Edu", "Env", "X")])
def test_invalid_query_rejected(fresh, query):
    rejected(fresh, query, ErrorCode.INVALID_QUERY)


# ------------------------------------------------------------------------------------- costs

def test_exact_decimal_costs_and_media_price_per_region(fresh, electorate):
    sizes = electorate.sizes.tolist()
    price = {r: (Decimal("50.00") * 4 * n / N).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
             for r, n in zip("NSEW", sizes)}
    obs = observation_for(fresh, "p0")
    assert obs["costs"]["media_by_region"] == {r: f"{p:.2f}" for r, p in price.items()}
    assert obs["costs"]["query"] == "5.00" and obs["costs"]["extra_worker"] == "4.00"
    s = ok(fresh, Query("Edu", "Env", "N")).state
    assert observation_for(s, "p0")["treasury"] == "495.00"
    s = committed(s)
    step = ok(s, Allocate((5, 4, 3, 1), 3, (True, False, False, True)))
    expected = Decimal("12.00") + price["N"] + price["W"]
    obs = observation_for(step.state, "p0")
    assert step.round_report["cost"] == f"{expected:.2f}"
    assert obs["allocations"][0]["cost"] == f"{expected:.2f}"
    assert Decimal(obs["treasury"]) == Decimal("495.00") - expected
    assert obs["media_airings"] == {"N": 1, "S": 0, "E": 0, "W": 1}


def test_insufficient_funds(config, electorate):
    s = committed(GameEngine.new_game(config, electorate=electorate))
    rejected(s, Allocate((10, 0, 0, 0), 200), ErrorCode.INSUFFICIENT_FUNDS)  # 800.00 > 500.00
    poor = GameConfig(seed=config.seed, rounds=2, device="cpu", budget=Decimal("7.00"), electorate=config.electorate)
    s = ok(GameEngine.new_game(poor, electorate=electorate), Query("Edu", "Env", "N")).state
    assert "query" not in observation_for(s, "p0")["legal_actions"]
    rejected(s, Query("Edu", "Env", "S"), ErrorCode.INSUFFICIENT_FUNDS)


# ----------------------------------------------------------------------------------- queries

def test_query_answer_equals_brute_force(fresh, electorate):
    step = ok(fresh, Query("Agri", "Industry", "E"))
    v, regions = electorate.preferences.tolist(), electorate.region_of.tolist()
    members = [i for i in range(N) if regions[i] == 2]
    expected = 100.0 * sum(1 for i in members if v[i][3] > v[i][2]) / len(members)
    assert step.answer["pct_b_over_a"] == pytest.approx(expected, abs=1e-12)
    assert step.answer["phase"] == "SURVEY" and step.answer["round"] == 0 and step.answer["cost"] == "5.00"
    assert observation_for(step.state, "p0")["queries"] == [step.answer]


def test_duplicate_query_rejected_same_round_allowed_next(fresh):
    q = Query("Edu", "Welfare", "S")
    s = ok(fresh, q).state
    rejected(s, q, ErrorCode.DUPLICATE_QUERY)
    s = ok(s, Query("Welfare", "Edu", "S")).state  # different key
    s = committed(s)
    s = ok(s, q).state                              # round 1: allowed again
    rejected(s, q, ErrorCode.DUPLICATE_QUERY)
    s = ok(s, Forfeit()).state
    step = ok(s, q)
    assert step.answer["round"] == 2 and step.answer["phase"] == "ROUND"


# ------------------------------------------------------------------------------------- polls

def test_polls_moe_and_reproducibility(config, electorate):
    def play():
        s = committed(GameEngine.new_game(config, electorate=electorate, game_id="g"))
        for _ in range(config.rounds):
            s = ok(s, Allocate((4, 3, 2, 1))).state
        return observation_for(s, "p0")

    first, second = play(), play()
    assert first == second
    polls = first["polls"]
    assert [p["round"] for p in polls] == list(range(1, config.rounds + 2))
    for p in polls:
        assert p["sample_size"] == 400
        assert p["moe"] == pytest.approx(1.96 * (p["estimate"] * (1 - p["estimate"]) / 400) ** 0.5)


# ----------------------------------------------------------------------------------- full game

def test_full_game_to_done(fresh, config):
    s = ok(fresh, Query("Edu", "Env", "W")).state
    s = committed(s)
    reports = []
    for t in range(config.rounds):
        step = ok(s, Allocate((3, 3, 2, 2), 2, (t == 0, False, True, False)))
        reports.append(step.round_report)
        s = step.state
    assert [r["round"] for r in reports] == [1, 2, 3]
    assert [r["election_held"] for r in reports] == [False, False, True]
    obs = observation_for(s, "p0")
    assert obs["phase"] == "DONE" and obs["round"] == config.rounds
    assert [f["round"] for f in obs["free_stats"]] == [0, 1, 2, 3]
    result = obs["result"]
    assert result["num_voters"] == N and result["victory"] == (2 * result["votes"] > N)
    player = s.player("p0")
    assert len(player.share_trajectory) == config.rounds + 1 and len(player.alignment) == config.rounds + 1
    assert all(a.dtype == torch.uint8 and a.shape == (N,) and int(a.max()) <= 4 for a in player.alignment)
    assert player.share_trajectory[-1] == pytest.approx(result["expected_share"])
    v = s.preferences
    assert torch.all(v >= 0) and float((v.sum(1) - 1).abs().max()) < 1e-5
    json.dumps(obs)


def test_engine_reproduces_simulate_world(config, electorate):
    manifesto = torch.tensor([0.3, 0.1, 0.2, 0.25, 0.15])
    plan = [Allocation((5, 3, 4, 0), 2, (True, False, False, False)),
            Allocation((0, 10, 0, 0), 0, (False, True, True, False)),
            Allocation((2, 2, 2, 4), 5, (True, False, False, True))]
    reference = simulate_world(electorate, config, manifesto, lambda ctx: plan[ctx.round])
    s = ok(GameEngine.new_game(config, electorate=electorate), Commit(tuple(manifesto.tolist()))).state
    for a in plan:
        s = ok(s, Allocate(a.workers_by_region, a.extra_workers, a.media_regions)).state
    player = s.player("p0")
    assert player.result == reference.result
    assert torch.equal(s.preferences, reference.final_preferences)
    assert list(player.share_trajectory) == reference.share_trajectory
    assert config.budget - player.ledger.balance == reference.spent


def test_replay_is_deterministic(config, electorate):
    s = ok(GameEngine.new_game(config, electorate=electorate), Query("Edu", "Agri", "N")).state
    s = committed(s)
    for _ in range(config.rounds):
        s = ok(s, Allocate((2, 2, 3, 3), 1, (False, True, False, False))).state
    again = GameEngine.replay(config, s.history, electorate=electorate)
    assert again.player("p0").result == s.player("p0").result
    assert torch.equal(again.preferences, s.preferences)
    assert again.player("p0").knowledge.to_list() == s.player("p0").knowledge.to_list()


# --------------------------------------------------------------------------------- two players

def test_two_player_round_resolves_only_when_both_ready(config, electorate):
    s = GameEngine.new_game(config, ("p0", "p1"), electorate=electorate)
    s = ok(s, Commit(UNIFORM), "p0").state
    assert s.phase == "SURVEY"
    s = ok(s, Query("Edu", "Env", "N"), "p0").state
    s = ok(s, Commit((0.4, 0.1, 0.1, 0.3, 0.1)), "p1").state
    assert s.phase == "ROUND" and s.round == 1
    step = ok(s, Allocate((10, 0, 0, 0)), "p0")
    assert step.round_report is None and step.state.round == 1
    rejected(step.state, Allocate((10, 0, 0, 0)), ErrorCode.INVALID_PHASE, "p0")
    assert observation_for(step.state, "p0")["legal_actions"] == ["query"]
    step = ok(step.state, Forfeit(), "p1")
    assert step.round_report["round"] == 1 and step.state.round == 2
    o0, o1 = observation_for(step.state, "p0"), observation_for(step.state, "p1")
    assert len(o0["queries"]) == 1 and o1["queries"] == []
    assert o0["manifesto"] != o1["manifesto"] and o0["treasury"] == "495.00" and o1["treasury"] == "500.00"
    assert o0["polls"] != o1["polls"]
    assert all("Query" not in e["message"] for e in o1["events"])
    assert o0["free_stats"] == o1["free_stats"]


# ----------------------------------------------------------------------------------- privacy

def test_privacy_of_every_observation_and_action_payload(fresh, config):
    payloads = [observation_for(fresh, "p0")]
    s = fresh
    for action in [Query("Edu", "Env", "N"), Commit(UNIFORM), Query("Agri", "Env", "S"), Allocate((5, 5, 0, 0), 1, (True,) * 4),
                   Forfeit(), Allocate((0, 0, 5, 5))]:
        step = GameEngine.apply(s, "p0", action)
        s = step.state
        payloads += [observation_for(s, "p0"), step.answer, step.round_report, step.message]
    payloads.append(GameEngine.apply(s, "p0", Forfeit()).message)
    assert s.phase == "DONE"
    for payload in payloads:
        assert_no_leaks(payload, N)
        json.dumps(payload)


def test_same_seed_gives_identical_initial_observation_and_census(config):
    a = GameEngine.new_game(config, game_id="same")
    b = GameEngine.new_game(config, game_id="same")
    assert observation_for(a, "p0") == observation_for(b, "p0")
    assert a.electorate.census() == b.electorate.census()


# ------------------------------------------------------------------------------- performance

@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() <= 2, reason="needs cuda:2 (shared server)")
def test_round_resolves_under_one_second_at_10k_on_gpu():
    cfg = GameConfig(seed=3, rounds=3, device="cuda:2", electorate=ElectorateConfig(num_voters=10_000))
    s = committed(GameEngine.new_game(cfg))
    s = ok(s, Allocate((5, 5, 0, 0), 2, (True, False, False, False))).state  # warm-up
    torch.cuda.synchronize("cuda:2")
    start = time.perf_counter()
    s = ok(s, Allocate((3, 3, 3, 3), 2, (False, True, True, False))).state
    torch.cuda.synchronize("cuda:2")
    assert time.perf_counter() - start < 1.0
