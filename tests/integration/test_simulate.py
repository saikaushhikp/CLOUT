from decimal import Decimal

import torch

from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.dynamics.campaign import Allocation
from dumbocracy.electorate import generate_electorate
from dumbocracy.simulate import simulate_world


def spender(ctx):
    extra = int((ctx.budget / (ctx.rounds - ctx.round)) // ctx.config.workers.extra_worker_cost)
    total = ctx.config.workers.free_workers + extra
    return Allocation((total, 0, 0, 0), extra, (False, True, False, False) if ctx.budget >= 200 else (False,) * 4)


def setup(seed=3):
    cfg = GameConfig(seed=seed, device="cpu", electorate=ElectorateConfig(num_voters=3000))
    return cfg, generate_electorate(cfg)


def test_simulation_is_reproducible_and_respects_budget_and_simplex():
    cfg, e = setup()
    b = torch.tensor([0.3, 0.1, 0.2, 0.2, 0.2])
    first = simulate_world(e, cfg, b, spender)
    second = simulate_world(e, cfg, b, spender)
    assert first.result == second.result and first.share_trajectory == second.share_trajectory
    assert Decimal("0") < first.spent <= cfg.budget
    v = first.final_preferences
    assert torch.all(v >= 0) and float((v.sum(1) - 1).abs().max()) < 1e-5
    zealots = e.susceptibility.zealot
    assert torch.equal(v[zealots], e.preferences[zealots])
    assert len(first.share_trajectory) == cfg.rounds + 1 and len(first.allocations) == cfg.rounds


def test_campaign_pulls_support_toward_the_manifesto():
    cfg, e = setup(4)
    b = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    idle = simulate_world(e, cfg, b, lambda ctx: Allocation((0, 0, 0, 0), 0, (False,) * 4))
    active = simulate_world(e, cfg, b, spender)
    assert active.result.expected_share > idle.result.expected_share
