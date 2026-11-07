import math

import pytest
import torch

from dumbocracy.voting import (
    ElectionResult, expected_share, run_election, sample_ballots, similarity, vote_probability,
)

E = torch.eye(5, dtype=torch.float64)


def test_similarity_extremes_and_broadcast():
    b = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2], dtype=torch.float64)
    assert float(similarity(b.unsqueeze(0), b)[0]) == pytest.approx(1.0)
    assert float(similarity(E[0:1], E[1])[0]) == pytest.approx(0.0)
    v = torch.stack([E[0], b, 0.5 * (E[0] + E[1])])
    torch.testing.assert_close(similarity(v, E[0]), torch.tensor([1.0, 0.1, 0.5], dtype=torch.float64))
    # Batched v [2, 3, 5] with per-world manifestos [2, 5].
    out = similarity(torch.stack([v, v]), torch.stack([E[0], b]))
    torch.testing.assert_close(out[0], similarity(v, E[0]))
    torch.testing.assert_close(out[1], similarity(v, b))
    with pytest.raises(ValueError):
        similarity(v, E[0, :4])


def test_vote_probability_at_threshold():
    s = torch.tensor([0.75, 1.0, 0.0], dtype=torch.float64)
    p = vote_probability(s, 12.0, 0.75)
    assert float(p[0]) == 0.5
    assert float(p[1]) == pytest.approx(1 / (1 + math.exp(-3)))
    assert float(p[2]) == pytest.approx(1 / (1 + math.exp(9)))


def test_ballot_mean_matches_expected_share():
    n = 200_000
    gen = torch.Generator().manual_seed(0)
    prob = torch.rand(n, generator=gen, dtype=torch.float64)
    ballots = sample_ballots(prob, torch.Generator().manual_seed(1))
    assert ballots.dtype == torch.bool and ballots.shape == (n,)
    e = float(expected_share(prob))
    sd = math.sqrt(float((prob * (1 - prob)).sum())) / n
    assert abs(float(ballots.double().mean()) - e) < 5 * sd


def test_expected_share_batched():
    prob = torch.tensor([[0.2, 0.4], [1.0, 0.0]])
    torch.testing.assert_close(expected_share(prob), torch.tensor([0.3, 0.5]))


@pytest.mark.gpu
def test_sample_ballots_rejects_wrong_device_generator():
    if torch.cuda.device_count() <= 2:
        pytest.skip("needs cuda:2 for a device mismatch")
    with pytest.raises(ValueError):
        sample_ballots(torch.rand(3), torch.Generator(device="cuda:2"))


def test_run_election_reproducible():
    gen = torch.Generator().manual_seed(5)
    x = torch.rand(10_001, 5, generator=gen, dtype=torch.float64)
    v = x / x.sum(-1, keepdim=True)
    b = torch.full((5,), 0.2, dtype=torch.float64)
    r1 = run_election(v, b, 12.0, 0.75, torch.Generator().manual_seed(42))
    r2 = run_election(v, b, 12.0, 0.75, torch.Generator().manual_seed(42))
    assert r1 == r2
    assert r1.num_voters == 10_001 and r1.vote_share == r1.votes / 10_001
    assert r1.victory == (2 * r1.votes > 10_001)
    prob = vote_probability(similarity(v, b), 12.0, 0.75)
    assert r1.expected_share == pytest.approx(float(prob.mean()))
    assert r1.expected_sd == pytest.approx(math.sqrt(float((prob * (1 - prob)).sum())) / 10_001)
    public = r1.to_public_dict()
    assert set(public) == {"votes", "num_voters", "vote_share", "victory", "expected_share", "expected_sd"}
    assert all(isinstance(x, (int, float, bool)) for x in public.values())


def test_victory_strict_majority_tie_loses():
    # Half the voters agree perfectly (P = σ(β(1 − z)) → 1 with large β); the other half are disjoint.
    v = torch.cat([E[0].expand(5, 5), E[1].expand(5, 5)])
    tie = run_election(v, E[0], 1e4, 0.5, torch.Generator().manual_seed(0))
    assert tie.votes == 5 and tie.num_voters == 10 and not tie.victory
    v_win = torch.cat([E[0].expand(6, 5), E[1].expand(4, 5)])
    win = run_election(v_win, E[0], 1e4, 0.5, torch.Generator().manual_seed(0))
    assert win.votes == 6 and win.victory


def test_run_election_rejects_bad_shapes():
    with pytest.raises(ValueError):
        run_election(E.unsqueeze(0), E[0], 12.0, 0.75, torch.Generator())
    with pytest.raises(ValueError):
        run_election(E, E, 12.0, 0.75, torch.Generator())
    assert isinstance(run_election(E, E[0], 12.0, 0.75, torch.Generator()), ElectionResult)


@pytest.mark.gpu
def test_gpu_election_reproducible():
    if torch.cuda.device_count() <= 2:
        pytest.skip("needs cuda:2")
    v = E.to("cuda:2")
    r1 = run_election(v, v[0], 12.0, 0.75, torch.Generator(device="cuda:2").manual_seed(3))
    r2 = run_election(v, v[0], 12.0, 0.75, torch.Generator(device="cuda:2").manual_seed(3))
    assert r1 == r2
