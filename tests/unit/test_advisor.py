"""scripts/advisor.py: read-only advice in the web UI's words (no API calls here: the observation is built locally)."""
import importlib.util
from pathlib import Path

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.game import GameEngine
from dumbocracy.game.observation import observation_for
from dumbocracy.strategies.players import BayesPlanner

_spec = importlib.util.spec_from_file_location("advisor", Path(__file__).resolve().parents[2] / "scripts" / "advisor.py")
advisor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(advisor)


def test_integer_percents_sum_to_100():
    for shares in ([0.2] * 5, [0.333, 0.333, 0.334, 0.0, 0.0], [0.26268, 0.14988, 0.17219, 0.18126, 0.23399]):
        pct = advisor.integer_percents(shares)
        assert sum(pct) == 100 and all(abs(p - 100 * s) < 1 for p, s in zip(pct, shares))


def test_describe_uses_ui_words():
    cfg = GameConfig(seed=3, device="cpu", electorate=ElectorateConfig(num_voters=1000))
    obs = observation_for(GameEngine.new_game(cfg, game_id="a"), "p0")
    player = BayesPlanner()
    q = advisor.describe(obs, {"type": "query", "sector_a": "Edu", "sector_b": "Env", "region": "N"}, player, 0.1)
    assert any("Sector A = Education, Sector B = Environment, Region = North" in line for line in q)
    c = advisor.describe(obs, {"type": "commit", "manifesto": [0.2] * 5}, player, 0.1)
    assert any("Commit manifesto" in line for line in c)
