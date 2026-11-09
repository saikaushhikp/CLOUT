"""P5: Kripke-style knowledge over sampled worlds × plans, information sets, observation_at (spec §6.5, §10)."""
from dataclasses import replace
from decimal import Decimal

import pytest
import torch

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.game import GameEngine
from dumbocracy.game.observation import observation_for
from dumbocracy.logic.epistemic import (
    K, M, evaluate, guaranteed_victory, inevitable_defeat, knows_win_exists, policy_set, possible_victory,
    properties, truth_victory_possible,
)
from dumbocracy.logic.knowledge import initial_budget, observation_at
from dumbocracy.logic.worlds import consistent_worlds, poll_standard_error
from dumbocracy.strategies.bayes import ASSIGNMENTS, posterior_from_observation
from dumbocracy.strategies.players import BayesPlanner, play_game, public_config


# ------------------------------------------------------------------------------ modal algebra

def _random_bool(gen, *shape):
    return torch.rand(*shape, generator=gen) < torch.rand(1, generator=gen)


def test_k_m_duality_on_random_matrices():
    gen = torch.Generator().manual_seed(0)
    for _ in range(300):
        phi = _random_bool(gen, int(torch.randint(1, 7, (1,), generator=gen)), int(torch.randint(1, 6, (1,), generator=gen)))
        assert torch.equal(M(phi), ~K(~phi))
        assert torch.equal(K(phi), ~M(~phi))
        assert bool((K(phi) <= M(phi)).all())                 # T axiom on a non-empty set: Kφ ⇒ Mφ
        win = phi
        assert inevitable_defeat(win) == (not possible_victory(win))
        if guaranteed_victory(win):                           # ∃π∀w ⇒ ∀w∃π
            assert knows_win_exists(win)
        if knows_win_exists(win):
            assert possible_victory(win)


def test_knowing_that_does_not_imply_knowing_how():
    # two indistinguishable worlds that need different plans: de dicto but not de re
    win = torch.tensor([[True, False], [False, True]])
    assert knows_win_exists(win) and possible_victory(win)
    assert not guaranteed_victory(win) and not inevitable_defeat(win)
    props = properties(torch.tensor([[0.6, 0.4], [0.45, 0.55]]), ["plan A", "plan B"])
    assert props["knows_win_exists"] and not props["guaranteed_victory"]
    assert props["guaranteed_witness"] is None and props["defeat_counter_world"] is not None
    assert props["best_plan"]["win_fraction"] == 0.5


def test_properties_witness_and_ties():
    props = properties(torch.tensor([[0.51, 0.70], [0.52, 0.49]]), ["a", "b"])
    assert props["guaranteed_victory"] and props["guaranteed_witness"] == "a"
    assert props["best_plan"]["label"] == "a"
    assert not properties(torch.tensor([[0.5]]))["possible_victory"]     # a tie is not a victory
    with pytest.raises(TypeError):
        K(torch.ones(2))


# ------------------------------------------------------------------------- forced fixtures

def _round1_state(*, n, budget, seed, manifesto_fn, zealots=0.05):
    cfg = GameConfig(seed=seed, device="cpu", budget=Decimal(budget),
                     electorate=ElectorateConfig(num_voters=n, zealot_fraction=zealots))
    state = GameEngine.new_game(cfg, game_id="fixture")
    b = manifesto_fn(state.electorate.preferences)
    step = GameEngine.apply(state, "p0", {"type": "commit", "manifesto": b})
    assert step.accepted, step.message
    return cfg, step.state


def _aligned(v):
    m = v.mean(0).double()
    m = m / m.sum()
    return [float(x) for x in m]


def _opposed(v):
    k = int(torch.argmin(v.mean(0)))
    return [1.0 if i == k else 0.0 for i in range(5)]


def _knowledge(state, cfg, count=6, n=800):
    obs = observation_for(state, "p0")
    pcfg = public_config(obs, replace(cfg, device="cpu"))
    ws = consistent_worlds(obs, pcfg, count, world_voters=n, seed=3)
    plans = policy_set(obs, pcfg)
    values = evaluate(ws, plans, obs, pcfg, seed=5)
    return obs, ws, plans, values


def test_huge_budget_and_aligned_electorate_is_guaranteed_victory():
    cfg, state = _round1_state(n=800, budget="20000.00", seed=11, manifesto_fn=_aligned, zealots=0.0)
    obs, ws, plans, values = _knowledge(state, cfg)
    props = properties(values, [p.label for p in plans])
    assert props["guaranteed_victory"] and props["knows_win_exists"] and props["possible_victory"]
    assert not props["inevitable_defeat"]
    assert truth_victory_possible(state, plans, seed=5)


def test_zero_budget_and_opposed_electorate_is_inevitable_defeat():
    cfg, state = _round1_state(n=800, budget="0.00", seed=12, manifesto_fn=_opposed)
    obs, ws, plans, values = _knowledge(state, cfg)
    assert obs["treasury"] == "0.00"
    assert all(p.first.extra_workers == 0 and not any(p.first.media_regions) for p in plans)   # only free workers
    props = properties(values)
    assert props["inevitable_defeat"] and not props["possible_victory"] and not props["guaranteed_victory"]
    assert not truth_victory_possible(state, plans, seed=5)


def test_policy_set_is_affordable_labelled_and_deterministic():
    cfg, state = _round1_state(n=800, budget="500.00", seed=13, manifesto_fn=_aligned)
    obs, ws, plans, values = _knowledge(state, cfg, count=4)
    labels = [p.label for p in plans]
    assert len(set(labels)) == len(labels) and labels[0].startswith("No spend")
    assert values.shape == (len(ws), len(plans)) and torch.all((values >= 0) & (values <= 1))
    again = evaluate(ws, plans, obs, public_config(obs, replace(cfg, device="cpu")), seed=5)
    assert torch.equal(values, again)


# --------------------------------------------------------------------------- information sets

@pytest.fixture(scope="module")
def played():
    """A BayesPlanner game that also asks one query per round; every observation it was handed is kept."""
    cfg = GameConfig(seed=21, device="cpu", electorate=ElectorateConfig(num_voters=1000))
    seen = []
    inner = BayesPlanner(seed=1, num_worlds=4, manifesto_worlds=4, world_voters=1000, round_query_gain=0.0, round_queries=1,
                         num_queries=2, survey_budget=Decimal("100.00"))

    class Spy:
        player_id = "p0"

        def act(self, observation):
            seen.append(observation)
            return inner.act(observation)

    state, _ = play_game(GameEngine.new_game(cfg, game_id="played"), Spy())
    return cfg, state, seen


FIELDS = ("phase", "round", "treasury", "free_stats", "polls", "queries", "allocations", "media_airings", "manifesto",
          "result", "legal_actions", "workers", "round_reports", "initial_budget")


def test_observation_at_matches_engine_observations(played):
    cfg, state, seen = played
    final = observation_for(state, "p0")
    rounds = [o for o in seen if o["phase"] == "ROUND"]
    assert any(q["phase"] == "ROUND" for q in final["queries"])          # round queries exist
    for t in range(1, cfg.rounds + 1):
        in_round = [o for o in rounds if o["round"] == t]
        start, decision = in_round[0], in_round[-1]
        assert len(in_round) >= 2                                        # a query, then the allocation
        for stage, expected in (("start", start), ("decision", decision)):
            got = observation_at(final, t, stage=stage)
            for key in FIELDS:
                assert got[key] == expected[key], (t, stage, key)
    assert initial_budget(final) == cfg.budget
    assert final["workers"]["hired"] > 0                                 # R16 hires are reconstructed above
    with pytest.raises(ValueError):
        observation_at(final, 0)


def test_observation_at_reconstructs_workers_without_the_workers_field(played):
    """Older observations (no ``workers``/``initial_budget``) still give hires from the allocations."""
    _, state, _ = played
    final = observation_for(state, "p0")
    legacy = {k: v for k, v in final.items() if k not in ("workers", "initial_budget", "round_reports")}
    for t in range(1, int(final["rounds"]) + 1):
        hired = sum(a["extra_workers"] for a in final["allocations"] if a["round"] < t)
        assert observation_at(legacy, t)["workers"] == {"free": 10, "hired": hired, "available": 10 + hired}
        assert observation_at(final, t)["round_reports"] == final["round_reports"][: t - 1]


def test_true_world_is_consistent_with_its_polls(played):
    # S5 reflexivity: with the default tolerance the TRUE world passes every poll filter (its expected
    # share trajectory is hidden state, read here only for the test)
    cfg, state, _ = played
    obs = observation_for(state, "p0")
    true_shares = state.player("p0").share_trajectory
    for poll in obs["polls"]:
        r = int(poll["round"])
        z = abs(true_shares[r - 1] - poll["estimate"]) / (poll_standard_error(poll["estimate"], poll["sample_size"]) + 0.01)
        assert z <= 2.5 + 1.5, (r, z)        # 4 SE: well beyond poll noise


def test_generous_tolerance_keeps_true_sigma_in_support_and_accepts_all(played):
    cfg, state, _ = played
    obs_t = observation_at(observation_for(state, "p0"), 3, stage="decision")
    posterior = posterior_from_observation(obs_t)
    true_sigma = state.electorate.archetype_of_region
    index = int(torch.nonzero((ASSIGNMENTS == true_sigma).all(1))[0])
    assert float(posterior.probs[index]) > 0
    pcfg = public_config(obs_t, replace(cfg, device="cpu"))
    ws = consistent_worlds(obs_t, pcfg, 6, world_voters=1000, seed=0, poll_z=1e6)
    assert ws.accepted == ws.draws == 6 and ws.acceptance_rate == 1.0 and not ws.fallback
    assert ws.poll_shares.shape == (6, 3) and ws.current.shape == (6 * 1000, 5)


def test_tight_tolerance_rejects_and_falls_back(played):
    cfg, state, _ = played
    obs_t = observation_at(observation_for(state, "p0"), 4)
    pcfg = public_config(obs_t, replace(cfg, device="cpu"))
    ws = consistent_worlds(obs_t, pcfg, 4, world_voters=1000, seed=0, poll_z=1e-9, slack=0.0, max_draws=8)
    assert ws.draws == 8 and ws.accepted == 0 and ws.fallback and len(ws) == 4
    assert not bool(ws.consistent.any())
    a = consistent_worlds(obs_t, pcfg, 4, world_voters=1000, seed=7)
    b = consistent_worlds(obs_t, pcfg, 4, world_voters=1000, seed=7)
    assert torch.equal(a.current, b.current) and torch.equal(a.tau, b.tau) and torch.equal(a.sigma, b.sigma)


def test_headline_cites_when_truth_was_lost_and_false_hope_rounds():
    from dumbocracy.logic.report import headline
    def row(t, possible, truth):
        return {"round": t, "possible_victory": possible, "guaranteed_victory": False, "knows_win_exists": False,
                "inevitable_defeat": not possible, "truth_victory_possible": truth}
    rows = [row(1, False, False), row(2, False, False), row(3, False, False), row(4, True, False), row(5, False, False)]
    text = headline(rows, {"victory": False, "vote_share": 0.473})
    assert "no plan could win from round 1 on" in text and "chance in round 4" in text
    rows = [row(1, True, True), row(2, False, True), row(3, False, True)]
    text = headline(rows, {"victory": False, "vote_share": 0.49})
    assert "winning plan still existed in rounds 2 and 3" in text
