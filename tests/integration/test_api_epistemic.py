"""``GET /api/games/{g}/players/{p}/epistemic`` (P5, decision R14; web_api_v2.md "Epistemic timeline").

The logic module is replaced by a fake so these tests exercise only the route contract
(404/409/200, per-session caching, single computation under concurrency, device propagation, privacy).
"""
import sys
import threading
import time
import types

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dumbocracy.api.app import create_app  # noqa: E402
from tests.privacy_scan import assert_no_leaks  # noqa: E402

N = 1000
ROUNDS = 3
ROUND_KEYS = {"round", "treasury", "knowledge", "possible_victory", "guaranteed_victory", "knows_win_exists",
              "inevitable_defeat", "truth_victory_possible", "best_plan"}


def fake_timeline(state, player_id, *, worlds=16, world_voters=1600, seed=0, device=None):
    rounds = [{
        "round": t, "treasury": "500.00",
        "knowledge": {"queries": 1, "polls": t, "posterior_entropy": 1.5 - 0.1 * t, "consistent_worlds": worlds},
        "possible_victory": True, "guaranteed_victory": t > 1, "knows_win_exists": t > 1, "inevitable_defeat": False,
        "truth_victory_possible": True,
        "best_plan": {"label": "balanced canvass", "win_fraction": 0.5 + 0.25 * t, "mean_share": 0.52},
    } for t in range(1, state.config.rounds + 1)]
    return {
        "scope": {"worlds_per_round": worlds, "policies_per_round": 12, "world_voters": world_voters,
                  "note": "finite worlds and plans"},
        "rounds": rounds,
        "actual": {"victory": True, "vote_share": 0.53, "expected_share": 0.52},
        "headline": "Victory was possible throughout; you knew a winning plan from round 2.",
    }


@pytest.fixture
def calls(monkeypatch):
    """Install a fake ``dumbocracy.logic.report`` module; returns the list of recorded calls."""
    recorded = []

    def spy(state, player_id, **kwargs):
        recorded.append((state.game_id, player_id, kwargs))
        time.sleep(0.05)   # widen the race window for the concurrency test
        return fake_timeline(state, player_id, **kwargs)

    module = types.ModuleType("dumbocracy.logic.report")
    module.epistemic_timeline = spy
    monkeypatch.setitem(sys.modules, "dumbocracy.logic.report", module)
    return recorded


@pytest.fixture
def client():
    return TestClient(create_app(device="cpu"))


def url(game, route):
    return f"/api/games/{game['game_id']}/players/{game['player_id']}/{route}"


def new_game(client):
    response = client.post("/api/games", json={"num_voters": N, "seed": 7, "rounds": ROUNDS})
    assert response.status_code == 200, response.text
    return response.json()


def finish(client, game):
    assert client.post(url(game, "commit"), json={"manifesto": [0.2] * 5}).json()["accepted"]
    for _ in range(ROUNDS):   # forfeit = skip one round
        phase = client.post(url(game, "forfeit")).json()["observation"]["phase"]
    assert phase == "DONE"


def test_404_unknown_game_or_player(client, calls):
    game = new_game(client)
    assert client.get("/api/games/nope/players/p0/epistemic").status_code == 404
    assert client.get(f"/api/games/{game['game_id']}/players/p9/epistemic").status_code == 404
    assert calls == []


def test_409_before_done(client, calls):
    game = new_game(client)
    assert client.get(url(game, "epistemic")).status_code == 409
    client.post(url(game, "commit"), json={"manifesto": [0.2] * 5})
    assert client.get(url(game, "epistemic")).status_code == 409
    assert calls == []


def test_200_after_done_with_schema_cached_and_private(client, calls):
    game = new_game(client)
    finish(client, game)
    first = client.get(url(game, "epistemic"))
    assert first.status_code == 200, first.text
    body = first.json()
    assert set(body) == {"scope", "rounds", "actual", "headline"}
    assert set(body["scope"]) == {"worlds_per_round", "policies_per_round", "world_voters", "note"}
    assert [r["round"] for r in body["rounds"]] == list(range(1, ROUNDS + 1))
    assert all(set(r) == ROUND_KEYS for r in body["rounds"])
    assert all(set(r["knowledge"]) == {"queries", "polls", "posterior_entropy", "consistent_worlds"} for r in body["rounds"])
    assert all(set(r["best_plan"]) == {"label", "win_fraction", "mean_share"} for r in body["rounds"])
    assert set(body["actual"]) == {"victory", "vote_share", "expected_share"} and isinstance(body["headline"], str)
    assert_no_leaks(body, N)
    second = client.get(url(game, "epistemic")).json()
    assert second == body
    assert len(calls) == 1
    game_id, player_id, kwargs = calls[0]
    assert (game_id, player_id) == (game["game_id"], "p0") and str(kwargs["device"]) == "cpu"
    # the debrief stays independent of the timeline cache
    assert client.get(url(game, "debrief")).status_code == 200 and len(calls) == 1


def test_concurrent_first_requests_compute_once(client, calls):
    game = new_game(client)
    finish(client, game)
    results = []

    def fetch():
        results.append(client.get(url(game, "epistemic")))

    threads = [threading.Thread(target=fetch) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r.status_code for r in results] == [200] * 4
    assert all(r.json() == results[0].json() for r in results)
    assert len(calls) == 1


def test_cache_is_per_game(client, calls):
    a, b = new_game(client), new_game(client)
    finish(client, a)
    finish(client, b)
    client.get(url(a, "epistemic"))
    client.get(url(b, "epistemic"))
    client.get(url(a, "epistemic"))
    assert [c[0] for c in calls] == [a["game_id"], b["game_id"]]


def test_503_when_the_logic_module_is_missing(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "dumbocracy.logic.report", None)   # makes the import raise ImportError
    game = new_game(client)
    finish(client, game)
    response = client.get(url(game, "epistemic"))
    assert response.status_code == 503 and "not available" in response.json()["detail"]
