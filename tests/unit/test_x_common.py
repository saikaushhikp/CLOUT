"""Phase 0 infrastructure of the v3 experiment suite (plan §2): x_common, x_streams, x_batch, x_policies,
and the default-preservation test of the approved X23 library change (``WorkerConfig.conversion_rule``)."""
from __future__ import annotations

import dataclasses
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import x_common  # noqa: E402
import x_policies  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_streams import SaltedStreams  # noqa: E402

from dumbocracy.config import SEED_STREAMS, ConfigError, ElectorateConfig, GameConfig, SeedStreams, WorkerConfig  # noqa: E402
from dumbocracy.dynamics.campaign import conversion_chance  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.simulate import simulate_world  # noqa: E402
from dumbocracy.strategies.oracle import top3_manifesto  # noqa: E402

N_SMALL = 1500


def _config(seed: int, n: int = N_SMALL, device: str = "cpu", **kw) -> GameConfig:
    return GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n), **kw)


def _same(a, b) -> bool:
    return (a.result == b.result and a.share_trajectory == b.share_trajectory and a.converted == b.converted
            and a.spreaders == b.spreaders and a.allocations == b.allocations and a.spent == b.spent
            and torch.equal(a.final_preferences, b.final_preferences)
            and torch.equal(a.workers.attempts, b.workers.attempts) and torch.equal(a.workers.pinned, b.workers.pinned)
            and all(torch.equal(x, y) for x, y in zip(a.workers.worker_ids, b.workers.worker_ids)))


# ------------------------------------------------------------------------------------- statistics / seeds
def test_holm_matches_hand_example():
    # sorted p: 0.005 (×4 = 0.02), 0.01 (×3 = 0.03), 0.03 (×2 = 0.06), 0.04 (×1 = 0.04 → monotone 0.06)
    assert x_common.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert x_common.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert x_common.holm([]) == []


def test_power_paired_binary_known_value():
    # Connor (1987): p_disc = 0.2, delta = 0.1, alpha 0.05, power 0.8 -> 155 pairs
    assert x_common.power_paired_binary(0.2, 0.1) == 155
    assert x_common.power_paired_binary(0.2, 0.15) < x_common.power_paired_binary(0.2, 0.1)
    with pytest.raises(ValueError):
        x_common.power_paired_binary(0.05, 0.1)


def test_seed_blocks_follow_plan_table():
    assert x_common.seed_block("X0", 3) == [10000, 10001, 10002]
    assert x_common.seed_block("x11", 1) == [11000]
    assert x_common.seed_block("X13", 2) == [12000, 12001]
    assert x_common.seed_block("X16", 1) == [13000]
    assert x_common.seed_block("X20", 1) == [14000]
    assert x_common.seed_block("X9", 1) == [20261107]
    with pytest.raises(ValueError):
        x_common.seed_block("X15", 10)


def test_write_row_and_override(tmp_path):
    path = tmp_path / "raw" / "x.jsonl"
    x_common.write_row(path, {"seed": 1, "spent": Decimal("12.50")})
    x_common.write_row(path, {"seed": 2})
    assert x_common.read_rows(path) == [{"seed": 1, "spent": "12.50"}, {"seed": 2}]
    cfg = GameConfig()
    assert x_common.override(cfg, "workers.conversion_rule=gamma").workers.conversion_rule == "gamma"
    assert x_common.override(cfg, "budget=750.00").budget == Decimal("750.00")
    assert x_common.override(cfg, "workers.attempt_cap", "0").workers.attempt_cap == 0
    assert x_common.override(cfg, "electorate.inter_prob_range=0.1,0.4").electorate.inter_prob_range == (0.1, 0.4)
    with pytest.raises(ConfigError):
        x_common.override(cfg, "workers.conversion_rule=gamma_p3")


# ------------------------------------------------------------------------------------- streams
def test_salted_streams_with_zero_salts_equal_seed_streams():
    plain = SeedStreams(1234)
    for salted in (SaltedStreams(1234), SaltedStreams(1234, {s: 0 for s in SEED_STREAMS})):
        for stream in SEED_STREAMS:
            for keys in ((), (0, "p0"), (3, "p0"), ("assignment",)):
                assert salted.derive(stream, *keys) == plain.derive(stream, *keys)
                a = salted.generator(stream, *keys)
                b = plain.generator(stream, *keys)
                assert torch.equal(torch.rand(8, generator=a), torch.rand(8, generator=b))


def test_salted_streams_change_only_the_salted_stream():
    plain, salted = SeedStreams(7), SaltedStreams(7, {"cascade": 3})
    assert salted.derive("cascade", 0, "p0") != plain.derive("cascade", 0, "p0")
    assert salted.derive("media", 0, "p0") == plain.derive("media", 0, "p0")
    with pytest.raises(ValueError):
        SaltedStreams(7, {"nope": 1})


def test_salted_streams_drive_simulate_world():
    cfg = _config(5)
    e = generate_electorate(cfg)
    b = top3_manifesto(e.preferences)
    ref = simulate_world(e, cfg, b, x_policies.policy_pay_split)
    same = simulate_world(e, cfg, b, x_policies.policy_pay_split, seeds=SaltedStreams(5))
    other = simulate_world(e, cfg, b, x_policies.policy_pay_split, seeds=SaltedStreams(5, {"cascade": 1}))
    assert same.result == ref.result and same.share_trajectory == ref.share_trajectory
    assert other.converted != ref.converted or other.result != ref.result


# ------------------------------------------------------------------------------------- features
def test_features_of_is_deterministic(tmp_path):
    cfg = _config(10000)
    a = x_common.features_of(generate_electorate(cfg), seed=10000, config=cfg, cache=False)
    b = x_common.features_of(generate_electorate(cfg), seed=10000, config=cfg, cache=False)
    assert a == b
    assert len(a["sizes"]) == 4 and sum(a["sizes"]) == N_SMALL
    assert 0.0 <= a["inter_edge_share"] <= 1.0 and a["lambda2_L"] > 0 and 0 < a["rw_gap"] <= 1
    assert a["rw_abs_gap"] <= a["rw_gap"] + 1e-12 and len(a["lambda_max_rr"]) == 4
    assert a["tv_min_pair"] <= a["tv_mean_pair"]
    # cache round trip
    c = x_common.features_of(generate_electorate(cfg), seed=10000, config=cfg, cache_dir=tmp_path)
    assert c == a and len(list(tmp_path.glob("features_10000_*.json"))) == 1
    assert json.loads(next(tmp_path.glob("*.json")).read_text()) == a


# ------------------------------------------------------------------------------------- conversion rule (X23)
def test_conversion_rule_default_is_bit_identical():
    assert WorkerConfig().conversion_rule == "gamma_p"
    g = torch.Generator().manual_seed(0)
    v = torch.distributions.Dirichlet(torch.ones(5)).sample((200,))
    gamma = torch.rand(200, generator=g)
    b = torch.tensor([0.3, 0.2, 0.2, 0.2, 0.1])
    from dumbocracy.voting import similarity, vote_probability
    old = (gamma * vote_probability(similarity(v, b), 20.0, 0.825)).clamp(0.0, 1.0)   # the pre-change formula
    assert torch.equal(conversion_chance(gamma, v, b, 20.0, 0.825), old)
    assert torch.equal(conversion_chance(gamma, v, b, 20.0, 0.825, "gamma_p"), old)
    assert torch.equal(conversion_chance(gamma, v, b, 20.0, 0.825, "gamma"), gamma.clamp(0, 1))
    p = vote_probability(similarity(v, b), 20.0, 0.825)
    assert torch.equal(conversion_chance(gamma, v, b, 20.0, 0.825, "gamma_p2"), (gamma * (p * p)).clamp(0, 1))


def test_conversion_rule_default_reproduces_games_and_rules_differ():
    cfg = _config(11)
    e = generate_electorate(cfg)
    b = top3_manifesto(e.preferences)
    base = simulate_world(e, cfg, b, x_policies.policy_front_workers)
    explicit = simulate_world(e, x_common.override(cfg, "workers.conversion_rule=gamma_p"), b,
                              x_policies.policy_front_workers)
    assert _same_basic(base, explicit)
    loose = simulate_world(e, x_common.override(cfg, "workers.conversion_rule=gamma"), b, x_policies.policy_front_workers)
    assert sum(loose.converted) > sum(base.converted)


def _same_basic(a, b) -> bool:
    return (a.result == b.result and a.share_trajectory == b.share_trajectory and a.converted == b.converted
            and torch.equal(a.final_preferences, b.final_preferences))


# ------------------------------------------------------------------------------------- batched runner
POLICY_NAMES = ["planner", "pay_split", "front_workers", "pay_media", "none"]


def _worlds(seeds, n=N_SMALL, device="cpu", **kw):
    base = _config(seeds[0], n, device, **kw)
    es, ms = [], []
    for s in seeds:
        cfg = dataclasses.replace(base, seed=s)
        e = generate_electorate(cfg, device=device)
        es.append(e)
        ms.append(top3_manifesto(e.preferences))
    return base, es, ms


@pytest.mark.parametrize("policy", POLICY_NAMES)
def test_batch_b1_equals_simulate_world(policy):
    base, es, ms = _worlds([10000])
    ref = simulate_world(es[0], dataclasses.replace(base, seed=10000), ms[0], x_policies.GATE_POLICIES[policy])
    out = run_batch(es, base, torch.stack(ms), x_policies.GATE_POLICIES[policy], seeds=[10000], keep_final=True)
    assert len(out) == 1 and _same(ref, out[0])


def test_batch_of_worlds_equals_per_world_runs_bit_for_bit():
    seeds = [10000, 10001, 10002, 10003]
    base, es, ms = _worlds(seeds)
    policies = [x_policies.policy_planner, x_policies.policy_pay_split, x_policies.WorkerTiming(6, "front", "largest"),
                x_policies.StrongArm()]
    out = run_batch(es, base, torch.stack(ms), policies, seeds=seeds, keep_final=True)
    for e, s, m, pol, o in zip(es, seeds, ms, policies, out):
        ref = simulate_world(e, dataclasses.replace(base, seed=s), m, pol)
        assert _same(ref, o)


def test_batch_overrides_cascade_off_rule_and_salted_streams():
    seeds = [10000, 10001]
    base, es, ms = _worlds(seeds)
    # cascade off (X2 arm B): attempt_cap = 0 -> nobody converts, workers still pinned
    off = x_common.override(base, "workers.attempt_cap=0")
    out = run_batch(es, off, torch.stack(ms), x_policies.policy_front_workers, seeds=seeds)
    assert all(sum(o.converted) == 0 and int(o.workers.pinned.sum()) > 0 for o in out)
    # conversion rule and salted streams match simulate_world too
    rule = x_common.override(base, "workers.conversion_rule=gamma_p2")
    streams = [SaltedStreams(s, {"cascade": 2}) for s in seeds]
    out = run_batch(es, rule, torch.stack(ms), x_policies.policy_front_workers, streams=streams, keep_final=True)
    for e, s, m, o, st in zip(es, seeds, ms, out, streams):
        assert _same(simulate_world(e, dataclasses.replace(rule, seed=s), m, x_policies.policy_front_workers, seeds=st), o)


def test_batch_pick_hook_replaces_worker_rule():
    seeds = [10000, 10001]
    base, es, ms = _worlds(seeds)
    calls = []

    def highest_degree(region_of, eligible, hires, generator, *, world, electorate):
        calls.append(world)
        degree = electorate.graph.degree
        out = []
        for r, k in enumerate(hires):
            cand = torch.nonzero(eligible & (region_of == r)).flatten()
            out.append(cand[torch.argsort(degree[cand], descending=True, stable=True)[:k]])
        return tuple(out)

    out = run_batch(es, base, torch.stack(ms), x_policies.WorkerTiming(4, "front", "largest"), seeds=seeds,
                    pick_fn=highest_degree)
    assert sorted(set(calls)) == [0, 1]
    for e, o in zip(es, out):
        ids = torch.cat(o.workers.worker_ids)
        assert ids.numel() == 4
        assert int(e.graph.degree[ids].min()) >= int(torch.quantile(e.graph.degree.double(), 0.9))


def test_policies_are_affordable_and_pure():
    base, es, ms = _worlds([10000, 10001])
    pols = [x_policies.make_policy("timing", total=8, timing=t, placement=p)
            for t in ("front", "even", "late") for p in ("proportional", "largest", "best_matched", "region:2")]
    pols += [x_policies.RegionFocus(1, c) for c in ("workers", "media", "split")]
    pols += [x_policies.Sequenced(("media", "workers"), workers=5), x_policies.Probe(0, "workers"),
             x_policies.PeoplePleaser(), x_policies.StrongArm()]
    for pol in pols:
        for o in run_batch(es, base, torch.stack(ms), pol, seeds=[10000, 10001]):
            assert Decimal("0") <= o.spent <= base.budget
    front = run_batch(es[:1], base, ms[0][None], x_policies.WorkerTiming(8, "front"), seeds=[10000])[0]
    late = run_batch(es[:1], base, ms[0][None], x_policies.WorkerTiming(8, "late"), seeds=[10000])[0]
    assert front.allocations[0].total_hires == 8 and late.allocations[-1].total_hires == 8
    b = x_policies.people_pleaser_manifesto(es[0], base)
    assert abs(float(b.sum()) - 1) < 1e-5 and x_policies.majority_regions([10, 20, 30, 40]) == [3, 2]
    assert abs(float(x_policies.strong_arm_manifesto("random", seed=3).sum()) - 1) < 1e-6


def test_run_chunks_halves_on_oom():
    seen = []

    def fn(chunk):
        seen.append(len(chunk))
        if len(chunk) > 2:
            raise torch.cuda.OutOfMemoryError("fake")
        return [x * 2 for x in chunk]

    record = {}
    assert x_common.run_chunks(fn, list(range(7)), 8, log=lambda _: None, record=record) == [x * 2 for x in range(7)]
    assert record["final_batch"] == 2 and record["oom_backoffs"] == [4, 2]


# ------------------------------------------------------------------------------------- GPU (cuda:2 when free)
def _gpu() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device, _ = x_common.select_device("cuda:2", need_bytes=1 * 2**30)
    return str(device)


@pytest.mark.gpu
def test_gpu_batch_matches_simulate_world_on_discrete_outcomes():
    device = _gpu()
    seeds = [10000, 10001, 10002]
    base, es, ms = _worlds(seeds, device=device)
    out = run_batch(es, base, torch.stack(ms), x_policies.policy_planner, seeds=seeds)
    for e, s, m, o in zip(es, seeds, ms, out):
        ref = simulate_world(e, dataclasses.replace(base, seed=s), m, x_policies.policy_planner)
        # CUDA index_add_ uses float atomics: floats agree to a few ulp, discrete outcomes exactly (documented)
        assert ref.result.votes == o.result.votes and ref.converted == o.converted and ref.allocations == o.allocations
        assert abs(ref.result.expected_share - o.result.expected_share) < 1e-5


@pytest.mark.gpu
def test_gpu_parity_check_and_features():
    device = _gpu()

    def fn(seed, dev):
        cfg = _config(seed, device=str(dev))
        e = generate_electorate(cfg, device=dev)
        o = run_batch([e], cfg, top3_manifesto(e.preferences)[None], x_policies.policy_pay_split, seeds=[seed])[0]
        return {"expected": o.result.expected_share, "victory": o.result.victory}

    rec = x_common.parity_check(fn, [10000, 10001, 10002], device=device)
    assert rec["passed"], rec["metrics"]
    e = generate_electorate(_config(10000), device=device)
    gpu = x_common.features_of(e, seed=10000, cache=False, spectra="gpu")
    cpu = x_common.features_of(generate_electorate(_config(10000)), seed=10000, cache=False, spectra="scipy")
    assert gpu["spectra_method"].startswith("lobpcg")
    for k in ("lambda2_L", "rw_gap", "rw_abs_gap"):
        assert gpu[k] == pytest.approx(cpu[k], rel=1e-6)
