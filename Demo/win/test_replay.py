"""Replays Demo/win/script.json through the API: the campaign wins, no campaign loses (pytest Demo/win/test_replay.py)."""
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay_win  # noqa: E402

SCRIPT = json.loads((HERE / "script.json").read_text())
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture(scope="module")
def client():
    return replay_win.make_client("cuda:0")


@CUDA
def test_campaign_wins_exactly_on_cuda0(client):
    rec = replay_win.play(SCRIPT, "cuda:0", client=client)
    exp = SCRIPT["expected"]["cuda"]
    assert rec["result"]["victory"]
    assert rec["result"]["votes"] == exp["campaign_votes"]
    assert rec["result"]["expected_share"] == pytest.approx(exp["campaign_expected_share"], abs=1e-6)


@CUDA
def test_no_campaign_loses_exactly_on_cuda0(client):
    rec = replay_win.play_no_campaign(SCRIPT, "cuda:0", client=client)
    exp = SCRIPT["expected"]["cuda"]
    assert not rec["result"]["victory"]
    assert rec["result"]["votes"] == exp["no_campaign_votes"]
    assert rec["rounds"][-1]["treasury_left"] == "830.00"


def test_percent_to_manifesto_matches_web():
    m = replay_win.percent_to_manifesto(SCRIPT["manifesto_pct"])
    assert sum(m) == pytest.approx(1.0, abs=1e-15)
    assert [round(x * 100) for x in m] == SCRIPT["manifesto_pct"]
