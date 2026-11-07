import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dumbocracy.api.app import create_app  # noqa: E402
from tests.privacy_scan import assert_no_leaks  # noqa: E402

N = 1000
ROUNDS = 2
ALLOC = {"workers_by_region": [3, 3, 2, 2], "extra_workers": 1, "media_regions": [False, True, False, False]}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(device="cpu"))


def new_game(client, **body):
    response = client.post("/api/games", json={"num_voters": N, "seed": 42, "rounds": ROUNDS, **body})
    assert response.status_code == 200, response.text
    return response.json()


def url(game, route):
    return f"/api/games/{game['game_id']}/players/{game['player_id']}/{route}"


@pytest.fixture(scope="module")
def finished(client):
    """A game played to DONE through the API, with every payload recorded."""
    game = new_game(client)
    payloads = [game]
    payloads.append(client.get(f"/api/games/{game['game_id']}/graph_sample?max_nodes=300").json())
    payloads.append(client.post(url(game, "query"), json={"sector_a": "Edu", "sector_b": "Env", "region": "N"}).json())
    payloads.append(client.post(url(game, "commit"), json={"manifesto": [0.2] * 5}).json())
    payloads.append(client.get(f"/api/games/{game['game_id']}/graph_sample?max_nodes=300").json())
    payloads.append(client.post(url(game, "allocate"), json=ALLOC).json())
    payloads.append(client.post(url(game, "forfeit")).json())
    return game, payloads


def test_health(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "device": "cpu", "active_games": body["active_games"], "max_games": 8}


def test_full_flow_through_the_api(client, finished):
    game, payloads = finished
    created, _, query, commit, _, allocate, forfeit = payloads
    obs = created["observation"]
    assert created["player_id"] == "p0" and obs["schema_version"] == "2.0" and obs["phase"] == "SURVEY"
    assert obs["seed"] == 42 and obs["num_voters"] == N and obs["rounds"] == ROUNDS and obs["treasury"] == "500.00"
    assert query["accepted"] and query["answer"]["cost"] == "5.00" and "round_report" not in query
    assert commit["accepted"] and commit["observation"]["phase"] == "ROUND" and "answer" not in commit
    assert len(commit["observation"]["polls"]) == 1 and len(commit["observation"]["free_stats"]) == 2
    assert allocate["accepted"] and allocate["round_report"]["round"] == 1 and not allocate["round_report"]["election_held"]
    assert forfeit["accepted"] and forfeit["round_report"]["election_held"]
    final = forfeit["observation"]
    assert final["phase"] == "DONE" and final["round"] == ROUNDS and final["result"]["num_voters"] == N
    assert [a["cost"] for a in final["allocations"]] == [allocate["round_report"]["cost"], "0.00"]
    again = client.get(url(game, "observation")).json()
    assert again == final


def test_debrief_after_done(client, finished):
    game, _ = finished
    debrief = client.get(url(game, "debrief")).json()
    assert debrief["result"]["num_voters"] == N
    assert [r["id"] for r in debrief["regions"]] == ["N", "S", "E", "W"]
    assert len({r["archetype"] for r in debrief["regions"]}) == 4
    assert len(debrief["share_trajectory"]) == ROUNDS + 1
    assert debrief["share_trajectory"][-1] == pytest.approx(debrief["result"]["expected_share"])
    oracle = debrief["oracle"]
    assert oracle["note"] == "ORACLE: full-information benchmark" and sum(oracle["manifesto"]) == pytest.approx(1, abs=1e-5)
    assert oracle["expected_share_no_campaign"] >= debrief["your_manifesto_no_campaign_share"] - 1e-6


def test_graph_sample_before_and_after_done(client, finished):
    game, payloads = finished
    before = payloads[1]
    assert before["revealed"] is False and before["alignment"] is None
    assert len(before["nodes"]) == 300
    ids = [n["id"] for n in before["nodes"]]
    assert ids == sorted(set(ids))
    assert all(0 <= a < b < 300 for a, b in before["edges"])
    assert {n["region"] for n in before["nodes"]} == {"N", "S", "E", "W"}
    after = client.get(f"/api/games/{game['game_id']}/graph_sample?max_nodes=300").json()
    assert after["revealed"] is True and after["nodes"] == before["nodes"] and after["edges"] == before["edges"]
    assert len(after["alignment"]) == ROUNDS + 1
    assert all(len(row) == 300 and all(0 <= x <= 4 for x in row) for row in after["alignment"])


def test_privacy_of_payloads_before_done(finished):
    _, payloads = finished
    for payload in payloads:
        assert_no_leaks(payload, N)


def test_debrief_409_before_done_and_rejections_are_200(client):
    game = new_game(client)
    assert client.get(url(game, "debrief")).status_code == 409
    response = client.post(url(game, "allocate"), json=ALLOC)
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is False and body["error_code"] == "INVALID_PHASE" and "round_report" not in body
    bad = client.post(url(game, "commit"), json={"manifesto": [0.5, 0.5, 0.5, 0, 0]}).json()
    assert bad["error_code"] == "INVALID_MANIFESTO" and bad["observation"]["manifesto"] is None
    bad = client.post(url(game, "query"), json={"sector_a": "Edu", "sector_b": "Edu", "region": "N"}).json()
    assert bad["error_code"] == "INVALID_QUERY" and bad["observation"]["treasury"] == "500.00"
    client.post(url(game, "commit"), json={"manifesto": [0.2] * 5})
    bad = client.post(url(game, "allocate"), json={**ALLOC, "workers_by_region": [20, 0, 0, 0]}).json()
    assert bad["error_code"] == "INVALID_ALLOCATION"
    bad = client.post(url(game, "allocate"), json={**ALLOC, "extra_workers": 500}).json()
    assert bad["error_code"] == "INSUFFICIENT_FUNDS"


def test_http_errors(client):
    game = new_game(client)
    assert client.get("/api/games/nope/players/p0/observation").status_code == 404
    assert client.get(f"/api/games/{game['game_id']}/players/p7/observation").status_code == 404
    assert client.post(url(game, "query"), json={"sector_a": "Edu"}).status_code == 422
    assert client.post(url(game, "allocate"), json={"workers_by_region": ["x", 0, 0, 0]}).status_code == 422
    assert client.post(url(game, "commit"), content="not json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/api/games", json={"num_voters": 999}).status_code == 422
    assert client.post("/api/games", json={"num_voters": 1000, "rounds": 11}).status_code == 422
    assert client.get(f"/api/games/{game['game_id']}/graph_sample?max_nodes=0").status_code == 422
    assert client.delete(f"/api/games/{game['game_id']}").json() == {"deleted": True}
    assert client.delete(f"/api/games/{game['game_id']}").status_code == 404
    assert client.get(url(game, "observation")).status_code == 404


def test_random_seed_and_same_seed_reproduces(client):
    random_game = client.post("/api/games", json={"num_voters": N, "rounds": 1}).json()
    assert 0 <= random_game["observation"]["seed"] < 2**31
    a, b = new_game(client), new_game(client)
    strip = lambda obs: {k: v for k, v in obs.items() if k != "game_id"}  # noqa: E731
    assert a["game_id"] != b["game_id"]
    assert strip(a["observation"]) == strip(b["observation"])


def test_oldest_idle_game_is_evicted():
    client = TestClient(create_app(device="cpu", max_games=2))
    first, second = new_game(client), new_game(client)
    client.get(url(first, "observation"))           # first is now the most recently used
    third = new_game(client)
    assert client.get(url(second, "observation")).status_code == 404
    assert client.get(url(first, "observation")).status_code == 200
    assert client.get(url(third, "observation")).status_code == 200
    assert client.get("/api/health").json()["active_games"] == 2
