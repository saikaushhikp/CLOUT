import pytest
import torch

from dumbocracy.dynamics.social import aggregate, aggregate_degroot, aggregate_wmsr
from dumbocracy.dynamics.update import step_preferences
from dumbocracy.electorate import Graph

E = torch.eye(5, dtype=torch.float64)


def toy_graph() -> Graph:
    """Triangle 0-1-2, pendant 3 on 2, isolated 4."""
    return Graph.from_edges(5, torch.tensor([0, 0, 1, 2]), torch.tensor([1, 2, 2, 3]))


def star_graph() -> Graph:
    """Hub 0 with leaves 1, 2, 3."""
    return Graph.from_edges(4, torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]))


def star_v() -> torch.Tensor:
    return torch.stack([E[0], torch.tensor([0.9, 0.1, 0, 0, 0], dtype=torch.float64),
                        torch.tensor([0.8, 0.2, 0, 0, 0], dtype=torch.float64), E[4]])


def random_simplex(*shape, generator, dtype=torch.float64, device="cpu"):
    x = torch.rand(*shape, 5, generator=generator, dtype=dtype, device=device)
    return x / x.sum(-1, keepdim=True)


def random_graph(n, m, seed, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    src, dst = torch.randint(0, n, (m,), generator=gen), torch.randint(0, n, (m,), generator=gen)
    return Graph.from_edges(n, src.to(device), dst.to(device))


# ---------------------------------------------------------------- DeGroot

def test_degroot_hand_computed():
    agg = aggregate_degroot(E.clone(), toy_graph())
    expected = torch.tensor([
        [0, 0.5, 0.5, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [1 / 3, 1 / 3, 0, 1 / 3, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],          # isolated: own v
    ], dtype=torch.float64)
    torch.testing.assert_close(agg, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("rule,trim", [("degroot", 0), ("wmsr", 1), ("wmsr", 2)])
def test_batch_equals_loop(rule, trim):
    graph = random_graph(40, 120, seed=7)
    v = random_simplex(3, 40, generator=torch.Generator().manual_seed(1))
    batched = aggregate(v, graph, rule, trim)
    looped = torch.stack([aggregate(v[b], graph, rule, trim) for b in range(3)])
    torch.testing.assert_close(batched, looped, rtol=0, atol=1e-14)
    nested = aggregate(v.reshape(1, 3, 40, 5), graph, rule, trim)
    torch.testing.assert_close(nested.reshape(3, 40, 5), batched, rtol=0, atol=1e-14)


def test_aggregate_rejects_bad_input():
    with pytest.raises(ValueError):
        aggregate_degroot(torch.zeros(4, 5), toy_graph())
    with pytest.raises(ValueError):
        aggregate(E.clone(), toy_graph(), "majority", 0)
    with pytest.raises(ValueError):
        aggregate_wmsr(E.clone(), toy_graph(), -1)


# ---------------------------------------------------------------- W-MSR

def test_wmsr_trims_outlier():
    agg = aggregate_wmsr(star_v(), star_graph(), 1)
    # Hub: distances to leaves 0.1, 0.2, 1.0 -> drop leaf 3, average leaves 1 and 2.
    torch.testing.assert_close(agg[0], torch.tensor([0.85, 0.15, 0, 0, 0], dtype=torch.float64))
    # Leaves have degree 1 <= F: keep own v.
    torch.testing.assert_close(agg[1:], star_v()[1:], rtol=0, atol=0)


def test_wmsr_trim_two_and_degree_le_f():
    agg = aggregate_wmsr(star_v(), star_graph(), 2)
    torch.testing.assert_close(agg[0], star_v()[1])           # only the nearest leaf survives
    agg = aggregate_wmsr(star_v(), star_graph(), 3)
    torch.testing.assert_close(agg, star_v(), rtol=0, atol=0)  # every degree <= F


def test_wmsr_ties_are_deterministic():
    # Hub 0 sees leaves 1 and 2 at the same distance; F = 1 keeps the lower index.
    graph = Graph.from_edges(3, torch.tensor([0, 0]), torch.tensor([1, 2]))
    v = torch.stack([E[0], E[1], E[2]])
    torch.testing.assert_close(aggregate_wmsr(v, graph, 1)[0], E[1])


def test_wmsr_isolated_and_f0_equals_degroot():
    v = random_simplex(5, generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(aggregate_wmsr(v, toy_graph(), 1)[4], v[4], rtol=0, atol=0)
    graph = random_graph(60, 200, seed=11)
    v = random_simplex(2, 60, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(aggregate_wmsr(v, graph, 0), aggregate_degroot(v, graph), rtol=0, atol=1e-14)


def test_wmsr_matches_bruteforce():
    graph = random_graph(30, 90, seed=5)
    v = random_simplex(30, generator=torch.Generator().manual_seed(6))
    agg = aggregate_wmsr(v, graph, 2)
    for i in range(30):
        nbrs = graph.indices[graph.indptr[i]:graph.indptr[i + 1]].tolist()
        if len(nbrs) <= 2:
            expected = v[i]
        else:
            ranked = sorted(nbrs, key=lambda j: (float(0.5 * (v[j] - v[i]).abs().sum()), j))
            expected = v[ranked[:len(nbrs) - 2]].mean(0)
        torch.testing.assert_close(agg[i], expected, rtol=0, atol=1e-14)


# ---------------------------------------------------------------- update

def test_scaling_rule_hand_computed():
    v = E[0:1]
    agg = E[1:2]
    b = torch.stack([E[2], E[3]])
    new = step_preferences(v, agg, torch.tensor([0.4], dtype=torch.float64),
                           torch.tensor([[0.5], [0.3]], dtype=torch.float64), b)
    # a + c = 1.2 > 1 -> scale = 0.6 / 0.8 = 0.75 -> c = (0.375, 0.225), own weight 0.
    torch.testing.assert_close(new, torch.tensor([[0, 0.4, 0.375, 0.225, 0]], dtype=torch.float64))


def test_unscaled_step_hand_computed():
    new = step_preferences(E[0:1], E[1:2], torch.tensor([0.2], dtype=torch.float64),
                           torch.tensor([[0.3]], dtype=torch.float64), E[2:3])
    torch.testing.assert_close(new, torch.tensor([[0.5, 0.2, 0.3, 0, 0]], dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_simplex_preserved_over_50_steps(dtype):
    n, players = 200, 2
    gen = torch.Generator().manual_seed(9)
    graph = random_graph(n, 800, seed=8)
    v = random_simplex(3, n, generator=gen, dtype=dtype)
    b = random_simplex(players, generator=gen, dtype=dtype)
    frozen = torch.rand(n, generator=gen) < 0.1
    alpha = torch.rand(n, generator=gen, dtype=dtype) * ~frozen
    v0 = v.clone()
    for t in range(50):
        intensity = torch.rand(players, 3, n, generator=gen, dtype=dtype) * ~frozen
        rule = "wmsr" if t % 2 else "degroot"
        v = step_preferences(v, aggregate(v, graph, rule, 1), 0.5 * alpha, intensity, b, frozen)
        assert torch.all(v >= 0)
        assert torch.all((v.sum(-1) - 1).abs() < 1e-5)
    assert torch.equal(v[:, frozen], v0[:, frozen])
    assert not torch.equal(v[:, ~frozen], v0[:, ~frozen])


def test_pure_social_step_and_shared_intensity():
    graph = toy_graph()
    v = random_simplex(2, 5, generator=torch.Generator().manual_seed(2))
    a = torch.full((5,), 0.5, dtype=torch.float64)
    agg = aggregate_degroot(v, graph)
    new = step_preferences(v, agg, a, torch.zeros(0, 2, 5, dtype=torch.float64), torch.zeros(0, 5, dtype=torch.float64))
    torch.testing.assert_close(new, 0.5 * v + 0.5 * agg)
    # [P, N] intensity broadcasts over the batch.
    c = torch.rand(1, 5, generator=torch.Generator().manual_seed(3), dtype=torch.float64) * 0.4
    shared = step_preferences(v, agg, a, c, E[:1])
    full = step_preferences(v, agg, a, c.unsqueeze(1).expand(1, 2, 5), E[:1])
    torch.testing.assert_close(shared, full, rtol=0, atol=0)


def test_step_rejects_bad_input():
    v = E.clone()
    a = torch.full((5,), 0.2, dtype=torch.float64)
    c = torch.full((1, 5), 0.2, dtype=torch.float64)
    b = E[:1]
    step_preferences(v, v, a, c, b)  # valid baseline
    bad = [
        (v, v[:4], a, c, b),                                       # aggregated shape
        (v, v, a[:4], c, b),                                       # social_weight shape
        (v, v, a + 0.9, c, b),                                     # a > 1
        (v, v, a, c[:, :4], b),                                    # intensity N
        (v, v, a, torch.full((2, 5), 0.2, dtype=torch.float64), b),  # P mismatch
        (v, v, a, c + 1.0, b),                                     # intensity > 1
        (v, v, a, -c, b),                                          # intensity < 0
        (v, v, a, c, torch.full((1, 5), 0.3, dtype=torch.float64)),  # manifesto off simplex
        (v, v, a, c, torch.tensor([[1.5, -0.5, 0, 0, 0]], dtype=torch.float64)),
        (v, v, a, c, E[:1, :4]),                                   # manifesto K
    ]
    for args in bad:
        with pytest.raises(ValueError):
            step_preferences(*args)
    with pytest.raises(ValueError):
        step_preferences(v, v, a, c, b, frozen=torch.zeros(4, dtype=torch.bool))


# ---------------------------------------------------------------- GPU parity

@pytest.mark.gpu
def test_gpu_matches_cpu():
    if torch.cuda.device_count() <= 2:
        pytest.skip("needs cuda:2")
    device = "cuda:2"
    n = 2000
    gen = torch.Generator().manual_seed(12)
    graph = random_graph(n, 10_000, seed=13)
    graph_gpu = random_graph(n, 10_000, seed=13, device=device)
    v = random_simplex(2, n, generator=gen)
    b = random_simplex(2, generator=gen)
    a = 0.5 * torch.rand(n, generator=gen, dtype=torch.float64)
    c = torch.rand(2, 2, n, generator=gen, dtype=torch.float64)
    frozen = torch.rand(n, generator=gen) < 0.1
    for rule, trim in (("degroot", 0), ("wmsr", 1), ("wmsr", 3)):
        cpu_agg = aggregate(v, graph, rule, trim)
        gpu_agg = aggregate(v.to(device), graph_gpu, rule, trim)
        torch.testing.assert_close(gpu_agg.cpu(), cpu_agg, rtol=0, atol=1e-12)
        cpu_new = step_preferences(v, cpu_agg, a, c, b, frozen)
        gpu_new = step_preferences(v.to(device), gpu_agg, a.to(device), c.to(device), b.to(device), frozen.to(device))
        torch.testing.assert_close(gpu_new.cpu(), cpu_new, rtol=0, atol=1e-12)
