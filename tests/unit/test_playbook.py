"""Playbook players (experiments/x_playbook.py): UI replicas, legality, observation-only play, human-rule manifesto."""
from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import x_playbook as xp  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.game import GameEngine  # noqa: E402
from dumbocracy.game.observation import observation_for  # noqa: E402
from dumbocracy.players.base import Player  # noqa: E402
from dumbocracy.sectors import ARCHETYPES, REGIONS  # noqa: E402
from dumbocracy.strategies.players import play_game  # noqa: E402

LIB = [{"id": a.id, "name": a.name, "mean": list(a.mean)} for a in ARCHETYPES]


# ------------------------------------------------------------------ UI replicas (cases ported from web/tests)
def test_apportion_matches_simplex_test_ts():
    assert xp.ui_apportion(100, [1, 1, 1]) == [34, 33, 33]
    assert sum(xp.ui_apportion(7, [0.1, 0.2, 0.7])) == 7
    assert xp.ui_apportion(10, [0, 0, 0, 0, 0]) == [2, 2, 2, 2, 2]
    assert xp.ui_apportion(0, [3, 4]) == [0, 0]


BLEND_OBS = {
    "regions": [{"id": "N", "voters": 4000}, {"id": "S", "voters": 1000}, {"id": "E", "voters": 3000},
                {"id": "W", "voters": 2000}],
    "archetype_library": [{"id": "A0", "name": "Agrarian", "mean": [0.1, 0.1, 0.6, 0.1, 0.1]},
                          {"id": "A1", "name": "Industrial", "mean": [0.1, 0.1, 0.1, 0.6, 0.1]},
                          {"id": "A2", "name": "Green", "mean": [0.2, 0.1, 0.1, 0.1, 0.5]}],
}


def test_blend_matches_blend_test_tsx():
    hunches = {"N": "A0", "S": "A1", "E": "A2"}
    b = xp.ui_blend_raw(BLEND_OBS, hunches, ["N", "E"])
    lib = BLEND_OBS["archetype_library"]
    expected = [(4000 * lib[0]["mean"][k] + 3000 * lib[2]["mean"][k]) / 7000 for k in range(5)]
    assert b == pytest.approx(expected, abs=1e-12)
    assert sum(b) == pytest.approx(1, abs=1e-12)
    assert xp.ui_blend_raw(BLEND_OBS, hunches, ["N", "S"]) != b
    assert xp.ui_blend_raw(BLEND_OBS, hunches, ["N"]) is None
    assert xp.ui_blend_raw(BLEND_OBS, hunches, ["N", "N"]) is None
    assert xp.ui_blend_raw(BLEND_OBS, hunches, ["N", "W"]) is None
    assert xp.ui_blend_raw(BLEND_OBS, {**hunches, "W": "A1"}, ["N", "S", "E", "W"]) is not None
    assert xp.ui_largest_hunch_regions(BLEND_OBS, hunches, 2) == ["N", "E"]
    assert xp.ui_largest_hunch_regions(BLEND_OBS, hunches, 3) == ["N", "E", "S"]


def test_blend_button_committed_vector_by_hand():
    """Real library: 4000 × Agrarian heartland + 3000 × Green urban, rounded to whole percents by the sliders.
    Raw = [19.29, 15.71, 27.14, 12.14, 25.71] %; floors sum to 98; the two largest remainders (.714, Env and
    Welfare) get +1 → [19, 16, 27, 12, 26] %."""
    obs = {"regions": [{"id": "N", "voters": 4000}, {"id": "S", "voters": 1000}, {"id": "E", "voters": 3000},
                       {"id": "W", "voters": 2000}], "archetype_library": LIB}
    b = xp.ui_blend(obs, {"N": "A0", "E": "A2"}, ["N", "E"])
    assert b == pytest.approx([0.19, 0.16, 0.27, 0.12, 0.26], abs=1e-15)
    assert abs(math.fsum(b) - 1) <= 1e-12


def _answer(truth, a, b, moe, **extra):
    from dumbocracy.sectors import SECTORS
    pct = xp.ui_predict_pct(truth, SECTORS.index(a), SECTORS.index(b))
    out = {"round": 0, "phase": "SURVEY", "sector_a": a, "sector_b": b, "region": "N", "pct_b_over_a": pct,
           "sample_size": 250, "moe": moe, "pooled_pct": pct, "pooled_sample_size": 250, "pooled_moe": moe,
           "repeat_index": 1, "cost": "5.00"}
    out.update(extra)
    return out


def test_fit_archetypes_matches_survey_test_ts():
    lib = [LIB[0], LIB[1], LIB[4]]
    truth = lib[0]["mean"]
    obs = {"archetype_library": lib}
    post = []
    for moe in (1, 3, 6, 12, 25):
        fits = xp.ui_fit_archetypes({**obs, "queries": [_answer(truth, "Env", "Welfare", moe)]}, "N")
        assert fits[0]["id"] == "A0"
        post.append(fits[0]["posterior"])
    assert all(post[k] < post[k - 1] for k in range(1, len(post)))
    precise = _answer(truth, "Edu", "Agri", 1.5)
    mislead = xp.ui_predict_pct(lib[1]["mean"], 2, 3)
    noisy = _answer(truth, "Agri", "Industry", 30, pct_b_over_a=mislead, pooled_pct=mislead)
    assert xp.ui_fit_archetypes({**obs, "queries": [precise, noisy]}, "N")[0]["id"] == "A0"
    sharp = xp.ui_fit_archetypes({**obs, "queries": [_answer(truth, "Edu", "Agri", 30),
                                                      {**noisy, "moe": 1, "pooled_moe": 1}]}, "N")
    assert sharp[0]["id"] == "A1"
    first = _answer(truth, "Edu", "Agri", 5, pct_b_over_a=10, pooled_pct=10)
    second = _answer(truth, "Edu", "Agri", 5, repeat_index=2, pooled_moe=3)
    fits = xp.ui_fit_archetypes({**obs, "queries": [first, second]}, "N")
    alone = xp.ui_fit_archetypes({**obs, "queries": [{**second, "repeat_index": 1}]}, "N")
    assert fits[0]["n"] == 1 and [f["posterior"] for f in fits] == [f["posterior"] for f in alone]


def test_survey_set_identifies_every_archetype():
    """WA+IE: predicted answers of any two archetypes differ by ≥ 30 points (Euclidean over the two questions)."""
    from dumbocracy.sectors import SECTORS
    pts = {a["id"]: [xp.ui_predict_pct(a["mean"], SECTORS.index(x), SECTORS.index(y)) for x, y in xp.SURVEY_SETS["WA+IE"]]
           for a in LIB}
    ids = list(pts)
    assert min(math.dist(pts[i], pts[j]) for i in ids for j in ids if i < j) > 30


def test_rules_pick_ui_regions():
    obs = {"regions": [{"id": "N", "voters": 3500}, {"id": "S", "voters": 1500}, {"id": "E", "voters": 2800},
                       {"id": "W", "voters": 2200}], "archetype_library": LIB}
    h = {"N": "A0", "E": "A1", "W": "A3", "S": "A5"}   # A0 vs A1: TV 0.35 → divided
    assert xp.blend_regions(obs, h, "two_largest") == ["N", "E"]
    assert xp.blend_regions(obs, h, "coalition") == ["N", "E"]          # 6300 / 10000 > half
    assert xp.blend_regions(obs, h, "all") == ["N", "E", "W", "S"]
    assert xp.archetype_tv("A0", "A1") >= xp.DIVIDED_TV
    # anchor = largest + the archetype closest to A0 (A3 Welfare-reliant: 0.25; A5 Balanced: 0.2)
    assert xp.blend_regions(obs, h, "anchor") == ["N", "S"]
    assert xp.blend_regions(obs, h, "divided") == ["N", "S"]
    h2 = {"N": "A0", "E": "A5", "W": "A3", "S": "A1"}
    assert xp.blend_regions(obs, h2, "divided") == ["N", "E"]           # united → coalition


# ------------------------------------------------------------------ full games
class Tracking(dict):
    """Observation dict that records which top-level keys a player reads."""

    def __init__(self, *a, seen=None, **k):
        super().__init__(*a, **k)
        self.seen = seen if seen is not None else set()

    def __getitem__(self, key):
        self.seen.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.seen.add(key)
        return super().get(key, default)


class Spy:
    def __init__(self, inner):
        self.inner = inner
        self.player_id = inner.player_id
        self.seen: set[str] = set()
        self.commit_obs = None
        self.queries = []

    def act(self, observation):
        obs = json.loads(json.dumps(observation))   # plain JSON only: nothing hidden can ride along
        action = self.inner.act(Tracking(obs, seen=self.seen))
        if action["type"] == "commit":
            self.commit_obs = obs
        if action["type"] == "query":
            self.queries.append(action)
        return action


OBS_KEYS = set(observation_for(GameEngine.new_game(GameConfig(seed=1, device="cpu",
                                                              electorate=ElectorateConfig(num_voters=1000)),
                                                   game_id="k"), "p0"))

DIALS = [
    {},
    {"survey_regions": 4, "manifesto_rule": "all", "workers": 4, "worker_place": "size", "media2": 2, "later": "even"},
    {"survey_regions": 3, "survey_set": "AI", "manifesto_rule": "divided", "workers": 8, "media_order": "match"},
    {"survey_regions": 4, "manifesto_rule": "anchor", "workers": 0, "media1": 2, "media2": 3, "worker_place": "largest"},
    {"survey_regions": 4, "manifesto_rule": "coalition", "lean": 0.5, "workers": 20, "workers2": 10},
]


@pytest.fixture(scope="module", params=[(77, 2000, 5), (5, 1000, 3)])
def game(request):
    seed, n, rounds = request.param
    cfg = GameConfig(seed=seed, device="cpu", rounds=rounds, electorate=ElectorateConfig(num_voters=n))
    return cfg, GameEngine.new_game(cfg, game_id="pb")


@pytest.mark.parametrize("dials", DIALS)
def test_playbook_full_game_legal_observation_only(game, dials):
    cfg, state0 = game
    player = xp.PlaybookPlayer(seed=1, **dials)
    assert isinstance(player, Player) and player.human_executable
    spy = Spy(player)
    final, log = play_game(state0, spy)          # strict: any rejected action raises
    assert final.phase == "DONE" and all(ok for *_, ok, _c in log)
    assert spy.seen <= OBS_KEYS
    obs = observation_for(final, "p0")
    assert len(obs["allocations"]) == cfg.rounds
    assert Decimal(obs["treasury"]) >= 0
    spent = sum(Decimal(a["cost"]) for a in obs["allocations"]) + sum(Decimal(q["cost"]) for q in obs["queries"])
    assert spent <= cfg.budget
    # survey = the fixed human list, nothing else
    plan = xp.survey_plan(spy.commit_obs, player.survey_regions, player.survey_set)
    assert [(q["sector_a"], q["sector_b"], q["region"]) for q in spy.queries] == plan
    # the committed manifesto is the UI's "Blend my hunches" over the rule's regions, from the dossiers
    hunches = xp.dossier_hunches(spy.commit_obs)
    assert len(set(hunches.values())) == len(hunches)
    regions = xp.blend_regions(spy.commit_obs, hunches, player.manifesto_rule)
    if player.lean is None:
        assert obs["manifesto"] == xp.ui_blend(spy.commit_obs, hunches, regions)
    assert all(abs(100 * x - round(100 * x)) < 1e-9 for x in obs["manifesto"])   # whole slider percents
    # last round never leaves enough for one more worker or airing unspent
    left = Decimal(obs["treasury"])
    assert left < Decimal(obs["costs"]["worker"])


def test_adaptive_branches_and_play(game):
    cfg, state0 = game
    obs0 = observation_for(state0, "p0")
    p = xp.AdaptivePlaybook(seed=2, sparse_k=18.0, small_n=3000, large_n=5000, branches={"small_or_long": {"workers": 9}, "sparse": {"workers": 1},
                                              "short_large": {"workers": 2}})
    assert p.branch(obs0) == "small_or_long"           # N ≤ 3000 in both fixtures
    spy = Spy(p)
    final, log = play_game(state0, spy)
    assert all(ok for *_, ok, _c in log) and spy.seen <= OBS_KEYS
    obs = observation_for(final, "p0")
    assert sum(obs["allocations"][0]["hires_by_region"]) == 9
    big = {**obs0, "num_voters": 10_000, "rounds": 3}
    assert p.branch(big) == "short_large"
    sparse = {**obs0, "num_voters": 10_000, "rounds": 5,
              "regions": [{**r, "mean_degree": 9.0} for r in obs0["regions"]]}
    assert p.branch(sparse) == "sparse"
    assert p.branch({**obs0, "num_voters": 10_000, "rounds": 5,
                     "regions": [{**r, "mean_degree": 30.0} for r in obs0["regions"]]}) == "base"
    assert xp.AdaptivePlaybook(fragmented_share=0.1).human_executable is False
    tuned = xp.AdaptivePlaybook(**xp.TUNED_PLAYBOOK)
    assert tuned.human_executable
    dense = [{**r, "mean_degree": 30.0} for r in obs0["regions"]]
    assert tuned.branch({**obs0, "regions": dense, "num_voters": 1000, "rounds": 3}) == "small_or_long"
    assert tuned.branch({**obs0, "regions": dense, "num_voters": 2000, "rounds": 3}) == "short_large"
    assert tuned.branch({**obs0, "regions": dense, "num_voters": 2000, "rounds": 5}) == "base"
    assert tuned.branch({**obs0, "regions": [{**r, "mean_degree": 20.0} for r in obs0["regions"]],
                         "num_voters": 10_000, "rounds": 5}) == "sparse"


def test_nonhuman_arms_flagged_and_legal():
    cfg = GameConfig(seed=9, device="cpu", electorate=ElectorateConfig(num_voters=1500))
    state0 = GameEngine.new_game(cfg, game_id="nh")
    for kw in ({"hunch_source": "bayes_map", "survey_set": "AI"}, {"election_day": True}):
        p = xp.PlaybookPlayer(seed=3, **kw)
        assert p.human_executable is False
        final, log = play_game(state0, Spy(p))
        assert all(ok for *_, ok, _c in log)
        assert Decimal(observation_for(final, "p0")["treasury"]) >= 0
