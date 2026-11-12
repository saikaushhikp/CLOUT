"""Budget-aware playbook (S4): the tuned playbook whose round-1 workers and media rounds follow the treasury.

Why: the S2 playbook (``x_playbook.TUNED_PLAYBOOK``) buys a FIXED 3 workers in round 1 and airs media in rounds 1–2;
with more money its "later = even" dial airs media in rounds 3–5 and dumps the rest into workers in ONE region in
the last round. At ₫100,000 that dump asks for more workers than the region has eligible voters, the engine rejects
the allocation and ~₫99,000 is left unspent (S4 smoke/tune). :class:`BudgetPlaybook` keeps the survey, hunches and
manifesto of the tuned playbook and spends by a rule a person can follow with the UI's price list:

  * media in ALL 4 regions (largest first while affordable) in rounds 1..M ("media rounds") and in the last round;
  * round 1 workers = max(3, floor(f × starting treasury / worker fee)), split by region size (largest remainder),
    but beyond the first 3 never more than the money left after reserving one "media in all 4" (P_all, the four
    prices added up) for each later media round (the 3 base workers are bought first, even if that leaves too
    little for every later media round);
  * the last round spends everything left: media first, then workers split by region size (not all in one
    region, so the request never exceeds a region's eligible voters at N = 10⁴, ₫100,000).
``schedule`` (optional) picks (M, f) by starting treasury: the row whose budget is nearest on a log scale.

With M = 2 and f = 0 this spends like the tuned playbook in rounds 1–2 (3 workers + media, media) at ₫500.
Reads only the observation dict (prices, treasury, starting treasury, rounds, region sizes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from dumbocracy.dynamics.campaign import media_cost
from dumbocracy.sectors import NUM_REGIONS, REGIONS
from dumbocracy.strategies.planner import split_by_weight
from dumbocracy.strategies.players import _treasury, public_config
from dumbocracy.strategies.worlds import public_sizes

from x_playbook import TUNED_PLAYBOOK, PlaybookPlayer, ui_regions_by_size

__all__ = ["BudgetPlaybook", "BUDGET_PLAYBOOK"]


@dataclass
class BudgetPlaybook(PlaybookPlayer):
    """Tuned playbook (survey/hunches/manifesto unchanged) + the budget-scaled spending rule (module docstring)."""

    media_rounds: int = 2
    worker_frac: float = 0.0
    min_workers: int = 3
    schedule: list[dict[str, Any]] = field(default_factory=list)   # [{"budget": B, "media_rounds": M, "worker_frac": f}]

    def dials(self) -> dict[str, Any]:
        return {**super().dials(), "media_rounds": self.media_rounds, "worker_frac": self.worker_frac,
                "min_workers": self.min_workers, "schedule": self.schedule}

    def row(self, budget: Decimal) -> tuple[int, float]:
        if not self.schedule:
            return int(self.media_rounds), float(self.worker_frac)
        b = math.log(float(budget))
        r = min(self.schedule, key=lambda x: (abs(math.log(float(x["budget"])) - b), float(x["budget"])))
        return int(r["media_rounds"]), float(r["worker_frac"])

    def allocation(self, observation: dict) -> dict[str, Any]:
        cfg = public_config(observation, self.config)
        sizes = public_sizes(observation)
        weights = [float(s) for s in sizes]
        price = {r: Decimal(observation["costs"]["media_by_region"][r]) for r in REGIONS}
        p_all = sum(price.values(), Decimal("0"))
        fee = Decimal(observation["costs"]["worker"])
        money = _treasury(observation)
        budget = Decimal(str(observation.get("initial_budget") or cfg.budget))
        t, T = int(observation["round"]), int(observation["rounds"])
        M, f = self.row(budget)
        M = min(M, T)
        self.record.update(media_rounds=M, worker_frac=f)
        hires = [0] * NUM_REGIONS
        media = [False] * NUM_REGIONS
        last = t == T
        if t <= M or last:
            for r in ui_regions_by_size(observation):
                if price[r] <= money:
                    media[REGIONS.index(r)] = True
                    money -= price[r]
        if t == 1 and not last and fee > 0:
            want = max(int(self.min_workers), int((Decimal(str(f)) * budget) // fee))
            spare = money - p_all * max(0, M - 1)
            n = max(0, min(want, int(spare // fee))) if spare > 0 else 0
            n = max(n, min(int(self.min_workers), int(money // fee)))   # the 3 base workers come before the reserve
            if n > 0:
                hires = list(split_by_weight(n, weights))
                money -= fee * n
        if last and fee > 0 and money >= fee:
            extra = int(money // fee)
            add = split_by_weight(extra, weights)
            hires = [h + a for h, a in zip(hires, add)]
            money -= fee * extra
        assert media_cost(sizes, media, cfg.media.base_cost) + fee * sum(hires) <= _treasury(observation)
        self.record.setdefault("workers_by_round", {})[t] = sum(hires)
        return {"type": "allocate", "hires_by_region": hires, "media_regions": media}


#: Fixed survey / hunch / manifesto dials of the tuned playbook; S4 tuning writes the spending dials to
#: s4_tune.json → chosen (single rule) and → chosen_schedule (per-budget table).
BUDGET_PLAYBOOK: dict[str, Any] = {k: v for k, v in TUNED_PLAYBOOK.items()
                                   if k in ("survey_regions", "survey_set", "manifesto_rule", "media_order")}
