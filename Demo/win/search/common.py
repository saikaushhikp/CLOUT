"""Shared helpers for the WIN demo search (Demo/win/search/). Fixed demo settings: N = 100,000, ₫850, T = 5, player p0.

Two survey questions cost ₫20, so the campaign is planned with ₫830 (``PLAY_BUDGET``).
"""
from __future__ import annotations

import math
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WIN = ROOT / "Demo" / "win"
SEARCH = WIN / "search"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(WIN))

import torch  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.dynamics.campaign import Allocation, media_cost  # noqa: E402
from dumbocracy.sectors import ARCHETYPES  # noqa: E402
from dumbocracy.strategies.manifesto_opt import expected_share, optimise_manifesto  # noqa: E402
from dumbocracy.strategies.oracle import anticipate, top3_manifesto  # noqa: E402
from dumbocracy.voting import similarity, vote_probability  # noqa: E402

N, T, BUDGET, PLAY_BUDGET = 100_000, 5, Decimal("850.00"), Decimal("830.00")
ALL = (True, True, True, True)
NONE = (False, False, False, False)
SALTED = ("workers", "cascade", "media", "election")


def cfg_for(seed: int, device, budget: Decimal = PLAY_BUDGET) -> GameConfig:
    return GameConfig(seed=int(seed), rounds=T, budget=budget, device=str(device),
                      electorate=ElectorateConfig(num_voters=N))


def int_percent(b) -> list[int]:
    """Largest-remainder whole percentages summing to 100."""
    raw = [float(x) * 100 / sum(float(y) for y in b) for x in b]
    base = [int(math.floor(x)) for x in raw]
    order = sorted(range(5), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:100 - sum(base)]:
        base[i] += 1
    return base


def web_manifesto(pcts) -> list[float]:
    """web/lib/simplex.ts ``percentToManifesto`` in IEEE doubles (same operation order)."""
    s = 0
    for v in pcts:
        s += v
    out = [v / s for v in pcts]
    acc = 0.0
    for v in out:
        acc += v
    i = out.index(max(out))
    out[i] += 1 - acc
    return out


class Plan:
    """Open-loop plan as a RoundContext policy: hires[t] (4 ints), media[t] (4 bools)."""

    def __init__(self, hires, media):
        self.hires = [tuple(int(x) for x in r) for r in hires]
        self.media = [tuple(bool(x) for x in r) for r in media]

    def __call__(self, ctx):
        return Allocation(self.hires[ctx.round], self.media[ctx.round])


def region_support(e, b, cfg, v=None) -> list[float]:
    v = e.preferences if v is None else v
    prob = vote_probability(similarity(v, b.to(v)), cfg.voting.sensitivity, cfg.voting.threshold)
    sums = torch.zeros(4, dtype=prob.dtype, device=prob.device).index_add_(0, e.region_of, prob)
    return [float(s) / int(n) for s, n in zip(sums.tolist(), e.sizes.tolist())]


def share_of(v, b, cfg) -> float:
    return float(vote_probability(similarity(v, b.to(v)), cfg.voting.sensitivity, cfg.voting.threshold).mean())


def best_no_campaign_manifesto(e, cfg, *, random_starts: int = 64, seed: int = 0, v_t=None) -> dict:
    """Best manifesto with NO campaign: with zero spending the election-day electorate v_T does not depend on the
    manifesto (social dynamics only), so the best achievable no-campaign share is max_b E(v_T, b). Multi-start
    projected gradient (``optimise_manifesto``) from: uniform, mean, 5 pulled vertices, top-3, the 6 archetype
    means, and ``random_starts`` Dirichlet(1) points; then the best whole-percent manifesto near the optimum
    (all ±1-point moves, hill climb). Returns continuous and integer optima and their exact shares."""
    v_t = anticipate(e, cfg) if v_t is None else v_t
    dt, dev = v_t.dtype, v_t.device
    gen = torch.Generator().manual_seed(seed)
    rnd = -torch.log(torch.rand((random_starts, 5), generator=gen, dtype=torch.float64))   # Dirichlet(1) via Exp(1)
    rnd = (rnd / rnd.sum(1, keepdim=True)).float()
    starts = torch.cat([torch.full((1, 5), 0.2), v_t.mean(0, keepdim=True).cpu(), 0.6 * torch.eye(5) + 0.08,
                        top3_manifesto(v_t).reshape(1, -1).cpu(), torch.tensor([a.mean for a in ARCHETYPES]),
                        rnd]).to(device=dev, dtype=dt)
    sol = optimise_manifesto(v_t, cfg.voting.sensitivity, cfg.voting.threshold, starts=starts, iterations=600)
    pct = int_percent(sol.manifesto.tolist())

    def s_of(p):
        return share_of(v_t, torch.tensor(web_manifesto(p), dtype=dt, device=dev), cfg)

    cur = s_of(pct)
    while True:
        best = None
        for i in range(5):
            for j in range(5):
                if i != j and pct[i] > 0:
                    p2 = list(pct)
                    p2[i] -= 1
                    p2[j] += 1
                    s = s_of(p2)
                    if s > cur + 1e-9 and (best is None or s > best[0]):
                        best = (s, p2)
        if best is None:
            break
        cur, pct = best
    return {"continuous_manifesto": [round(float(x), 6) for x in sol.manifesto.tolist()], "continuous_share": sol.share,
            "starts": int(starts.shape[0]), "integer_pct": pct, "integer_share": cur}
