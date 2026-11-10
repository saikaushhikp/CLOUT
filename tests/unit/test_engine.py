import json
import math
import time
from decimal import ROUND_HALF_UP, Decimal

import pytest
import torch

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.dynamics.campaign import Allocation
from dumbocracy.electorate import generate_electorate
from dumbocracy.game import Allocate, Commit, ErrorCode, Forfeit, GameEngine, Query, observation_for
from dumbocracy.information import query_moe, query_pct, query_sample_size
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


@pytest.mark.parametrize("action", [Allocate((1, 0, 0)), Allocate((-1, 0, 0, 0)), Allocate((True, 0, 0, 0)),
                                    Allocate((1, 0, 0, 0), (1, 0, 0, 0)), Allocate((1.0, 0, 0, 0)),
                                    Allocate((1, 0, 0, 0), (False,) * 3), Allocate([1, 0, 0, 0]), Allocate(None)])
def test_invalid_allocation_rejected(fresh, action):
    rejected(committed(fresh), action, ErrorCode.INVALID_ALLOCATION)


@pytest.mark.parametrize("body", [{"type": "allocate", "workers_by_region": [1, 0, 0, 0]},
                                  {"type": "allocate", "hires_by_region": [1, 0, 0, 0], "extra_workers": 1}])
def test_old_allocate_bodies_are_malformed(fresh, body):
    rejected(committed(fresh), body, ErrorCode.INVALID_ACTION)


def test_too_few_eligible_voters_rejected_atomically(fresh, config, electorate):
    from dataclasses import replace
    from dumbocracy.game.actions import available_hires
    s = committed(fresh)
    player = s.player("p0")
    available = available_hires(s, player)
    assert all(0 < a <= n for a, n in zip(available, electorate.sizes.tolist()))
    rich = s.with_player(replace(player, ledger=type(player.ledger)(Decimal("1000000.00"))))
    rejected(rich, Allocate((available[0] + 1, 0, 0, 0)), ErrorCode.INVALID_ALLOCATION)
    assert rich.player("p0").team_by_region == (0, 0, 0, 0)
    step = ok(rich, Allocate((0, available[1], 0, 0)))                 # every eligible voter is fine
    assert step.round_report["team_by_region"] == [0, available[1], 0, 0]
    # bought workers are no longer eligible
    rejected(step.state, Allocate((0, 1, 0, 0)), ErrorCode.INVALID_ALLOCATION)
    assert available_hires(step.state, step.state.player("p0"))[1] == 0


@pytest.mark.parametrize("query", [Query("Edu", "Edu", "N"), Query("Edu", "Foo", "N"), Query("Edu", "Env", "X")])
def test_invalid_query_rejected(fresh, query):
    rejected(fresh, query, ErrorCode.INVALID_QUERY)


# ------------------------------------------------------------------------------------- costs

def test_exact_decimal_costs_and_media_price_per_region(fresh, electorate, config):
    q, w = config.survey.query_cost, config.workers.worker_cost
    sizes = electorate.sizes.tolist()
    price = {r: (config.media.base_cost * 4 * n / N).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
             for r, n in zip("NSEW", sizes)}
    obs = observation_for(fresh, "p0")
    assert obs["costs"]["media_by_region"] == {r: f"{p:.2f}" for r, p in price.items()}
    assert obs["costs"]["query"] == f"{q:.2f}" and obs["costs"]["worker"] == f"{w:.2f}"
    assert obs["costs"]["min_degree"] == 3 and obs["costs"]["attempt_cap"] == 3
    assert set(obs["costs"]) == {"query", "worker", "min_degree", "attempt_cap", "media_by_region"}
    s = ok(fresh, Query("Edu", "Env", "N")).state
    assert Decimal(observation_for(s, "p0")["treasury"]) == config.budget - q
    s = committed(s)
    step = ok(s, Allocate((1, 1, 0, 1), (True, False, False, True)))
    expected = 3 * w + price["N"] + price["W"]
    obs = observation_for(step.state, "p0")
    assert step.round_report["cost"] == f"{expected:.2f}"
    assert obs["allocations"][0]["cost"] == f"{expected:.2f}"
    assert Decimal(obs["treasury"]) == config.budget - q - expected
    assert obs["media_airings"] == {"N": 1, "S": 0, "E": 0, "W": 1}


def test_insufficient_funds(config, electorate):
    q, w = config.survey.query_cost, config.workers.worker_cost
    s = committed(GameEngine.new_game(config, electorate=electorate))
    too_many = int(config.budget // w) + 1
    assert too_many * w > config.budget
    rejected(s, Allocate((too_many, 0, 0, 0)), ErrorCode.INSUFFICIENT_FUNDS)
    # enough for exactly one query, not two
    poor = GameConfig(seed=config.seed, rounds=2, device="cpu", budget=(q * Decimal("1.4")).quantize(Decimal("0.01")),
                      electorate=config.electorate)
    s = ok(GameEngine.new_game(poor, electorate=electorate), Query("Edu", "Env", "N")).state
    assert "query" not in observation_for(s, "p0")["legal_actions"]
    rejected(s, Query("Edu", "Env", "S"), ErrorCode.INSUFFICIENT_FUNDS)


# ----------------------------------------------------------------------------------- queries

QUERY_KEYS = {"round", "phase", "sector_a", "sector_b", "region", "pct_b_over_a", "sample_size", "moe", "pooled_pct",
              "pooled_sample_size", "pooled_moe", "repeat_index", "cost"}


def test_query_answer_is_a_seeded_10pct_sample(fresh, electorate, config):
    step = ok(fresh, Query("Agri", "Industry", "E"))
    ans = step.answer
    assert set(ans) == QUERY_KEYS
    n_r = int((electorate.region_of == 2).sum())
    m = query_sample_size(n_r, 0.10)
    assert ans["sample_size"] == m == max(1, math.floor(0.10 * n_r + 0.5))  # half-up rounding
    # same draw from the documented survey key (round 0, player p0, query index 0), brute force on the sample
    gen = config.seeds().generator("survey", 0, "p0", 0, device="cpu")
    members = torch.nonzero(electorate.region_of == 2).reshape(-1)
    sample = members[torch.randperm(n_r, generator=gen)[:m]].tolist()
    v = electorate.preferences.tolist()
    yes = sum(1 for i in sample if v[i][3] > v[i][2])
    assert ans["pct_b_over_a"] == pytest.approx(100.0 * yes / m, abs=1e-12)
    assert ans["moe"] == pytest.approx(query_moe(yes, m, n_r))
    assert ans["pooled_pct"] == ans["pct_b_over_a"] and ans["pooled_sample_size"] == m and ans["repeat_index"] == 1
    assert ans["pooled_moe"] == pytest.approx(query_moe(yes, m, fpc=False))
    assert ans["phase"] == "SURVEY" and ans["round"] == 0 and ans["cost"] == f"{config.survey.query_cost:.2f}"
    exact = query_pct(electorate.preferences, electorate.region_of, 2, 3, 2)
    assert abs(ans["pct_b_over_a"] - exact) < 4 * ans["moe"]
    assert observation_for(step.state, "p0")["queries"] == [step.answer]


def test_repeats_are_charged_pooled_and_reset_next_round(fresh, config):
    q = Query("Edu", "Welfare", "S")
    answers, s = [], fresh
    for _ in range(3):
        step = ok(s, q)
        answers.append(step.answer)
        s = step.state
    assert Decimal(observation_for(s, "p0")["treasury"]) == config.budget - 3 * config.survey.query_cost  # every repeat is charged
    assert [a["repeat_index"] for a in answers] == [1, 2, 3]
    assert len({a["pct_b_over_a"] for a in answers}) > 1            # fresh sample each time
    m = answers[0]["sample_size"]
    yes = [round(a["pct_b_over_a"] * a["sample_size"] / 100) for a in answers]
    last = answers[-1]
    assert last["pooled_sample_size"] == 3 * m
    assert last["pooled_pct"] == pytest.approx(100.0 * sum(yes) / (3 * m))
    assert last["pooled_moe"] == pytest.approx(query_moe(sum(yes), 3 * m, fpc=False))
    assert last["pooled_moe"] < answers[0]["pooled_moe"]
    assert answers[1]["pooled_sample_size"] == 2 * m
    rev = ok(s, Query("Welfare", "Edu", "S"))                        # reversed triple: a different question
    assert rev.answer["repeat_index"] == 1 and rev.answer["pooled_sample_size"] == m
    s = ok(rev.state, Query("Edu", "Welfare", "N")).state            # other region: not pooled either
    assert observation_for(s, "p0")["queries"][-1]["repeat_index"] == 1
    s = committed(s)
    step = ok(s, q)                                                  # round 1: pooling resets
    assert step.answer["round"] == 1 and step.answer["phase"] == "ROUND"
    assert step.answer["repeat_index"] == 1 and step.answer["pooled_sample_size"] == m
    step = ok(step.state, q)
    assert step.answer["repeat_index"] == 2
    s = ok(step.state, Forfeit()).state
    step = ok(s, q)
    assert step.answer["round"] == 2 and step.answer["repeat_index"] == 1
    assert len(observation_for(step.state, "p0")["queries"]) == 8


# ------------------------------------------------------------------------------------- polls

def test_polls_moe_and_reproducibility(config, electorate):
    def play():
        s = committed(GameEngine.new_game(config, electorate=electorate, game_id="g"))
        for _ in range(config.rounds):
            s = ok(s, Allocate((1, 0, 0, 0))).state
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
        step = ok(s, Allocate((2, 1, 0, 0) if t == 0 else (0, 0, 0, 0), (t == 0, False, True, False)))
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
    """P8: identical round order and seed keying — workers persist and keep their cascades in both."""
    manifesto = torch.tensor([0.3, 0.1, 0.2, 0.25, 0.15])
    plan = [Allocation((2, 1, 1, 0), (True, False, False, False)),
            Allocation((0, 0, 0, 0), (False, True, True, False)),
            Allocation((0, 0, 0, 1), (True, False, False, True))]
    seen_team = []

    def policy(ctx):
        seen_team.append(ctx.team_by_region)
        return plan[ctx.round]

    reference = simulate_world(electorate, config, manifesto, policy)
    assert seen_team == [(0, 0, 0, 0), (2, 1, 1, 0), (2, 1, 1, 0)]
    s = ok(GameEngine.new_game(config, electorate=electorate), Commit(tuple(manifesto.tolist()))).state
    reports = []
    for a in plan:
        step = ok(s, Allocate(a.hires_by_region, a.media_regions))
        s = step.state
        reports.append(step.round_report)
    player = s.player("p0")
    assert player.team_by_region == (2, 1, 1, 1)
    assert player.result == reference.result
    assert torch.equal(s.preferences, reference.final_preferences)
    assert list(player.share_trajectory) == reference.share_trajectory
    assert config.budget - player.ledger.balance == reference.spent
    assert [r["converted"] for r in reports] == reference.converted
    assert [r["spreaders"] for r in reports] == reference.spreaders
    assert torch.equal(player.workers.pinned, reference.workers.pinned)
    # round 2 buys nothing but last round's converts still spread
    assert reports[1]["spreaders"] == reports[0]["converted"]


# ------------------------------------------------------------------------------ P8 permanent workers

REPORT_KEYS = {"round", "cost", "hires_by_region", "team_by_region", "spreaders", "converted", "media_exposed",
               "election_held"}


def test_workers_are_permanent_paid_once_and_pinned(fresh, config):
    w = config.workers.worker_cost
    s = committed(fresh)
    obs = observation_for(s, "p0")
    assert obs["schema_version"] == "2.2"
    assert obs["workers"] == {"by_region": {"N": 0, "S": 0, "E": 0, "W": 0}, "total": 0}
    assert obs["initial_budget"] == f"{config.budget:.2f}"
    step = ok(s, Allocate((2, 0, 1, 0)))                         # buy 3 in round 1
    assert set(step.round_report) == REPORT_KEYS
    assert step.round_report["cost"] == f"{3 * w:.2f}" and step.round_report["hires_by_region"] == [2, 0, 1, 0]
    assert step.round_report["team_by_region"] == [2, 0, 1, 0] and step.round_report["spreaders"] == 3
    obs = observation_for(step.state, "p0")
    assert obs["workers"] == {"by_region": {"N": 2, "S": 0, "E": 1, "W": 0}, "total": 3}
    assert Decimal(obs["treasury"]) == config.budget - 3 * w
    assert obs["allocations"][0] == {"round": 1, "hires_by_region": [2, 0, 1, 0], "media_regions": [False] * 4,
                                     "cost": f"{3 * w:.2f}"}
    player = step.state.player("p0")
    b = torch.tensor(UNIFORM, dtype=step.state.preferences.dtype)
    assert int(player.workers.pinned.sum()) == 3
    assert torch.equal(step.state.preferences[player.workers.pinned], b.expand(3, -1))
    first = step.round_report["converted"]
    step = ok(step.state, Allocate((0, 1, 0, 0), (True, True, True, True)))   # round 2: one more, media everywhere
    assert step.round_report["team_by_region"] == [2, 1, 1, 0]
    assert step.round_report["spreaders"] == 1 + first
    player = step.state.player("p0")
    assert int(player.workers.pinned.sum()) == 4
    assert torch.equal(step.state.preferences[player.workers.pinned], b.expand(4, -1))   # media/friends never move them
    z = step.state.electorate.susceptibility.zealot
    assert torch.equal(step.state.preferences[z], step.state.electorate.preferences[z])
    step = ok(step.state, Forfeit())                              # forfeit: no hires, earlier converts spread
    assert step.round_report["cost"] == "0.00" and step.round_report["hires_by_region"] == [0, 0, 0, 0]
    assert step.state.phase == "DONE"


def test_insufficient_funds_for_hires_is_atomic(config, electorate):
    w = config.workers.worker_cost
    budget = (w * Decimal("2.4")).quantize(Decimal("0.01"))                       # affords 2 workers, not 3
    poor = GameConfig(seed=config.seed, rounds=2, device="cpu", budget=budget, electorate=config.electorate)
    s = committed(GameEngine.new_game(poor, electorate=electorate))
    rejected(s, Allocate((3, 0, 0, 0)), ErrorCode.INSUFFICIENT_FUNDS)          # 3w > 2.4w, nothing bought
    assert s.player("p0").team_by_region == (0, 0, 0, 0) and s.player("p0").workers is None
    step = ok(s, Allocate((2, 0, 0, 0)))
    assert observation_for(step.state, "p0")["workers"]["total"] == 2
    rejected(step.state, Allocate((1, 0, 0, 0)), ErrorCode.INSUFFICIENT_FUNDS)   # w > 0.4w left
    assert ok(step.state, Allocate()).accepted


def test_round_reports_grow_and_match_action_reports(fresh, config):
    s = committed(fresh)
    assert observation_for(s, "p0")["round_reports"] == []
    reports = []
    for t in range(1, config.rounds + 1):
        step = ok(s, Allocate((1, 0, 0, 0), (t == 2, False, False, False)) if t != 3 else Forfeit())
        s = step.state
        reports.append(step.round_report)
        obs = observation_for(s, "p0")
        assert obs["round_reports"] == reports
        assert [r["round"] for r in obs["round_reports"]] == list(range(1, t + 1))
        assert all(set(r) == REPORT_KEYS for r in obs["round_reports"])
    assert reports[-1]["election_held"] and not any(r["election_held"] for r in reports[:-1])
    json.dumps(observation_for(s, "p0"))


def test_replay_is_deterministic(config, electorate):
    s = ok(GameEngine.new_game(config, electorate=electorate), Query("Edu", "Agri", "N")).state
    s = ok(s, Query("Edu", "Agri", "N")).state
    s = committed(s)
    for t in range(config.rounds):
        s = ok(s, Query("Env", "Industry", "W")).state
        s = ok(s, Allocate((1, 0, 0, 1) if t == 0 else (0, 0, 0, 0), (False, True, False, False))).state
    again = GameEngine.replay(config, s.history, electorate=electorate)
    assert again.player("p0").result == s.player("p0").result
    assert observation_for(again, "p0")["round_reports"] == observation_for(s, "p0")["round_reports"]
    assert torch.equal(again.preferences, s.preferences)
    assert again.player("p0").knowledge.to_list() == s.player("p0").knowledge.to_list()
    assert len(again.player("p0").knowledge.of_kind("query")) == 2 + config.rounds


# --------------------------------------------------------------------------------- two players

def test_two_player_round_resolves_only_when_both_ready(config, electorate):
    s = GameEngine.new_game(config, ("p0", "p1"), electorate=electorate)
    s = ok(s, Commit(UNIFORM), "p0").state
    assert s.phase == "SURVEY"
    fee = config.workers.worker_cost
    s = ok(s, Query("Edu", "Env", "N"), "p0").state
    s = ok(s, Commit((0.4, 0.1, 0.1, 0.3, 0.1)), "p1").state
    assert s.phase == "ROUND" and s.round == 1
    step = ok(s, Allocate((3, 0, 0, 0)), "p0")
    assert step.round_report is None and step.state.round == 1
    rejected(step.state, Allocate((3, 0, 0, 0)), ErrorCode.INVALID_PHASE, "p0")
    assert observation_for(step.state, "p0")["legal_actions"] == ["query"]
    step = ok(step.state, Allocate((2, 0, 0, 0)), "p1")
    assert step.round_report["round"] == 1 and step.state.round == 2
    w0, w1 = step.state.player("p0").workers, step.state.player("p1").workers
    assert int(w0.pinned.sum()) == 3 and int(w1.pinned.sum()) == 2 and not bool((w0.pinned & w1.pinned).any())
    o0, o1 = observation_for(step.state, "p0"), observation_for(step.state, "p1")
    assert len(o0["queries"]) == 1 and o1["queries"] == []
    assert o0["manifesto"] != o1["manifesto"] and Decimal(o0["treasury"]) == config.budget - config.survey.query_cost - 3 * fee
    assert Decimal(o1["treasury"]) == config.budget - 2 * fee
    assert o0["polls"] != o1["polls"]
    assert all("Query" not in e["message"] for e in o1["events"])
    assert o0["free_stats"] == o1["free_stats"]


# ----------------------------------------------------------------------------------- privacy

def test_privacy_of_every_observation_and_action_payload(fresh, config):
    payloads = [observation_for(fresh, "p0")]
    s = fresh
    for action in [Query("Edu", "Env", "N"), Commit(UNIFORM), Query("Agri", "Env", "S"), Allocate((2, 1, 0, 0), (True,) * 4),
                   Forfeit(), Allocate((0, 0, 1, 0))]:
        step = GameEngine.apply(s, "p0", action)
        s = step.state
        payloads += [observation_for(s, "p0"), step.answer, step.round_report, step.message]
    payloads.append(GameEngine.apply(s, "p0", Forfeit()).message)
    assert s.phase == "DONE"
    for payload in payloads:
        assert_no_leaks(payload, N)
        json.dumps(payload)
    # P8: worker counts only — no node ids, gamma, attempts or per-voter data anywhere
    obs = observation_for(s, "p0")
    assert set(obs["workers"]) == {"by_region", "total"}
    text = json.dumps(payloads[1:] + [obs]).lower()
    for word in ("attempt\"", "pinned", "worker_ids", "node", "gamma", "susceptib"):
        assert word not in text, word


def test_same_seed_gives_identical_initial_observation_and_census(config):
    a = GameEngine.new_game(config, game_id="same")
    b = GameEngine.new_game(config, game_id="same")
    assert observation_for(a, "p0") == observation_for(b, "p0")
    assert a.electorate.census() == b.electorate.census()


# ------------------------------------------------------------------------------- performance

def test_round_resolves_under_one_second_at_10k_on_cpu():
    cfg = GameConfig(seed=3, rounds=3, device="cpu", electorate=ElectorateConfig(num_voters=10_000))
    s = committed(GameEngine.new_game(cfg))
    s = ok(s, Allocate((3, 2, 0, 0), (True, False, False, False))).state
    start = time.perf_counter()
    s = ok(s, Allocate((1, 0, 0, 0), (False, True, False, False))).state
    assert time.perf_counter() - start < 1.0


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() <= 2, reason="needs cuda:2 (shared server)")
def test_round_resolves_under_one_second_at_10k_on_gpu():
    cfg = GameConfig(seed=3, rounds=3, device="cuda:2", electorate=ElectorateConfig(num_voters=10_000))
    s = committed(GameEngine.new_game(cfg))
    s = ok(s, Allocate((3, 2, 0, 0), (True, False, False, False))).state  # warm-up
    torch.cuda.synchronize("cuda:2")
    start = time.perf_counter()
    s = ok(s, Allocate((1, 0, 0, 0), (False, True, False, False))).state
    torch.cuda.synchronize("cuda:2")
    assert time.perf_counter() - start < 1.0


def test_alignment_bins_follow_vote_probability_edges():
    from dumbocracy.game.engine import alignment_bins
    b = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float64)
    # Similarities 1.0, 0.825 (P = 0.5), 0.6 → expect bins 4 (>90%), 2 (swing), 0 (<10%) with β=20, z=0.825.
    v = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2], [0.375, 0.025, 0.2, 0.2, 0.2], [0.6, 0.0, 0.0, 0.2, 0.2]], dtype=torch.float64)
    assert alignment_bins(v, b, 20.0, 0.825).tolist() == [4, 2, 0]
