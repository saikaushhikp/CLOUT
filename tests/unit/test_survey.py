import json
import math
from decimal import Decimal

import pytest
import torch

from dumbocracy.game.ledger import InsufficientFunds, Ledger, money_str
from dumbocracy.information import KnowledgeEntry, KnowledgeLog, global_top3_pct, poll_moe, query_pct, run_poll
from dumbocracy.sectors import SECTORS


def random_simplex(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 5, generator=g)
    return x / x.sum(1, keepdim=True)


def test_top3_sums_to_300_and_matches_brute_force():
    v = random_simplex(500)
    stat = global_top3_pct(v)
    assert list(stat) == list(SECTORS)
    assert sum(stat.values()) == pytest.approx(300.0)
    for k, name in enumerate(SECTORS):
        count = sum(1 for row in v.tolist() if k in sorted(range(5), key=lambda j: (-row[j], j))[:3])
        assert stat[name] == pytest.approx(100.0 * count / 500)


def test_top3_ties_break_by_sector_index():
    v = torch.full((10, 5), 0.2)
    assert global_top3_pct(v) == {"Edu": 100.0, "Welfare": 100.0, "Agri": 100.0, "Industry": 0.0, "Env": 0.0}


def test_query_pct_equals_brute_force_and_rejects_same_sector():
    v = random_simplex(400, 1)
    region_of = torch.arange(400) % 4
    for a, b, r in [(0, 1, 0), (3, 2, 1), (4, 0, 3)]:
        members = [i for i in range(400) if int(region_of[i]) == r]
        expected = 100.0 * sum(1 for i in members if float(v[i, b]) > float(v[i, a])) / len(members)
        assert query_pct(v, region_of, a, b, r) == pytest.approx(expected)
    with pytest.raises(ValueError):
        query_pct(v, region_of, 2, 2, 0)


def test_poll_is_reproducible_and_uses_the_moe_formula():
    v = random_simplex(2000, 2)
    b = torch.full((5,), 0.2)
    first = run_poll(v, b, 20.0, 0.825, 400, torch.Generator().manual_seed(7))
    second = run_poll(v, b, 20.0, 0.825, 400, torch.Generator().manual_seed(7))
    assert first == second
    p, moe = first
    assert moe == pytest.approx(1.96 * math.sqrt(p * (1 - p) / 400))
    assert poll_moe(0.5, 400) == pytest.approx(0.049)


def test_poll_of_perfectly_aligned_voters_is_high_and_clamps_sample_size():
    b = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])
    v = b.repeat(50, 1)
    p, _ = run_poll(v, b, 20.0, 0.825, 400, torch.Generator().manual_seed(1))  # n > N -> all 50
    assert p > 0.8


def test_knowledge_log_is_append_only_and_json_safe():
    log = KnowledgeLog()
    entry = KnowledgeEntry.make("query", 0, {"pct": 12.5, "region": "N", "nested": {"xs": [1, 2]}})
    longer = log.append(entry)
    assert len(log) == 0 and len(longer) == 1
    payload = longer.entries[0].payload
    payload["pct"] = 99
    payload["nested"]["xs"].append(3)
    assert longer.entries[0].payload == {"pct": 12.5, "region": "N", "nested": {"xs": [1, 2]}}
    json.dumps(longer.to_list())
    with pytest.raises(TypeError):
        KnowledgeEntry.make("query", 0, {"t": torch.zeros(2)})
    with pytest.raises(ValueError):
        KnowledgeEntry.make("bogus", 0, {})


def test_ledger_debit_is_exact_and_atomic():
    ledger = Ledger(Decimal("10.00"))
    after = ledger.debit(Decimal("3.335"))   # half-up to 3.34
    assert after.balance == Decimal("6.66") and after.spent == Decimal("3.34")
    with pytest.raises(InsufficientFunds):
        after.debit(Decimal("6.67"))
    assert after.balance == Decimal("6.66") and ledger.balance == Decimal("10.00")
    assert money_str(Decimal("500")) == "500.00"
