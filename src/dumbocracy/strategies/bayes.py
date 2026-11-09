"""Observation-only Bayesian belief over the hidden archetype assignment σ (spec §2.3, §6, §10.1; P4).

This is the epistemic information set I(K_p) of spec §6.5/§10.1 *restricted to σ*: instead of the
hard ε-consistency filter over sampled worlds we keep a soft (Gaussian-likelihood) posterior over all
360 injective maps σ: region → archetype, starting from the uniform prior of §2.3 (U11). P5 (formal
reasoning) reuses it to draw σ for its world sampler.

Personalities τ (§2.4) are **not** identifiable from surveys: free statistics and query answers are
functions of the preference vectors v only, and v(0) ~ Dirichlet(κ μ_σ(r)) does not depend on τ. The
belief over τ therefore stays uniform (1/24 each) and is not represented here; a θ = (σ, τ) posterior
is the product of this posterior with the uniform τ prior.

Answer model (precomputed once by Monte Carlo, :func:`build_answer_model`): for a voter with
v ~ Dir(κ μ_k), κ = 60 (§14),
  * ``query_p[k, a, b] = P(v_b > v_a)`` — the per-voter probability behind a §6.3 query answer;
  * ``top3[k, s] = P(s ∈ top-3 of v)`` — ties broken by sector index exactly as §6.2 /
    :func:`dumbocracy.information.survey.global_top3_pct`.

Likelihood (independent Gaussian approximations, :func:`posterior_from_observation`):
  * a query answer y (percent) about (a, b, r) asked at round t is (R12) the frequency in a fresh
    random sample without replacement of m = ``sample_size`` voters of region r (n_r voters), so
    y ≈ N(100 p, (100 √(p(1−p)/m · (n_r−m)/(n_r−1)) + slack + drift·t)²) with p = query_p[σ(r), a, b]
    (m defaults to round(0.10·n_r) for answers without ``sample_size``). ``slack`` absorbs the region's
    own finite-population deviation from p plus Monte Carlo/model error; ``drift·t`` widens the
    likelihood in later rounds because social dynamics (§4) and campaigns (§3) move v away from v(0).
    Every answer — including each repeat of the same triple — is one independent observation; the
    ``pooled_*`` fields are a summary of those same samples and are deliberately ignored (using both
    would double count).
  * the free top-3 statistic is a size-weighted mixture over regions: mean 100 Σ_r (n_r/N) top3[σ(r), s],
    sampling sd 100 √(Σ_r (n_r/N)² p(1−p)/n_r) plus slack. The five sector components are treated as
    independent although they are dependent (they sum to exactly 300%) — a mild over-counting of the
    evidence that the slack term partly compensates.
  * different answers are treated as conditionally independent given σ (samples of the same region
    overlap and share the region's realised deviation from p, so this slightly over-counts evidence;
    the slack term bounds the effect for a handful of repeats).

A hard consistency cut in the spirit of §10.1 (``hard_z``) gives zero mass to assignments whose
predicted answer is more than ``hard_z`` standard deviations from an observed one; if every
assignment is cut (or the likelihood is non-finite everywhere) the posterior falls back to the prior
and is flagged ``inconsistent``.
"""
from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..information.survey import query_sample_size
from ..rng import dirichlet
from ..sectors import ARCHETYPES, NUM_REGIONS, NUM_SECTORS, REGIONS, SECTORS, archetype_means, region_index, sector_index

DTYPE = torch.float64
NUM_ARCHETYPES = len(ARCHETYPES)

#: All injective maps region → archetype index, int64 [360, 4], lexicographic order (§2.3, U11).
ASSIGNMENTS: torch.Tensor = torch.tensor(list(itertools.permutations(range(NUM_ARCHETYPES), NUM_REGIONS)),
                                         dtype=torch.int64)
NUM_ASSIGNMENTS = int(ASSIGNMENTS.shape[0])

#: Unordered sector pairs (a < b): the reversed query carries the same information (answer = 100 − y).
SECTOR_PAIRS: tuple[tuple[int, int], ...] = tuple(itertools.combinations(range(NUM_SECTORS), 2))

_LOG_2PI = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------------------------- model
@dataclass(frozen=True)
class AnswerModel:
    """Per-archetype answer probabilities under v ~ Dir(κ μ_k) (spec §2.3, §6.2, §6.3)."""

    concentration: float
    query_p: torch.Tensor  # [6,5,5] float64, query_p[k,a,b] = P(v_b > v_a) for v ~ Dir(κ μ_k) (diag 0)
    top3: torch.Tensor     # [6,5] P(sector in voter's top-3) for archetype k


@functools.lru_cache(maxsize=8)
def build_answer_model(concentration: float = 60.0, samples: int = 200_000, seed: int = 0) -> AnswerModel:
    """Monte Carlo answer model (cached per ``(concentration, samples, seed)``).

    Draws ``samples`` voters per archetype with :func:`dumbocracy.rng.dirichlet` on a seeded CPU
    generator; MC standard error ≤ 0.5/√samples ≈ 0.11 pp at the default size.
    """
    if concentration <= 0 or samples <= 0:
        raise ValueError("concentration and samples must be > 0")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    means = archetype_means(dtype=DTYPE)                                     # [6, 5]
    alpha = (float(concentration) * means).unsqueeze(1).expand(NUM_ARCHETYPES, int(samples), NUM_SECTORS)
    v = dirichlet(alpha.contiguous(), generator)                             # [6, S, 5]
    query_p = (v.unsqueeze(-2) > v.unsqueeze(-1)).to(DTYPE).mean(dim=1)     # [k,a,b]: v_b > v_a
    order = torch.sort(v, dim=-1, descending=True, stable=True).indices[..., :3]
    top3 = torch.zeros(NUM_ARCHETYPES, NUM_SECTORS, dtype=DTYPE)
    top3.scatter_add_(1, order.reshape(NUM_ARCHETYPES, -1), torch.ones(NUM_ARCHETYPES, 3 * int(samples), dtype=DTYPE))
    return AnswerModel(float(concentration), query_p, top3 / float(samples))


# ----------------------------------------------------------------------------------------- posterior
def _entropy(probs: torch.Tensor) -> torch.Tensor:
    positive = probs > 0
    return -(torch.where(positive, probs * torch.log(torch.where(positive, probs, torch.ones_like(probs))),
                         torch.zeros_like(probs))).sum(-1)


@dataclass(frozen=True)
class Posterior:
    """Belief over σ (spec §10.1 information set restricted to σ); τ stays uniform (module docstring)."""

    probs: torch.Tensor  # [360] float64, sums to 1 (row i ↔ ASSIGNMENTS[i])
    inconsistent: bool = False

    @classmethod
    def uniform(cls) -> "Posterior":
        return cls(torch.full((NUM_ASSIGNMENTS,), 1.0 / NUM_ASSIGNMENTS, dtype=DTYPE))

    def region_marginals(self) -> torch.Tensor:
        """``[4, 6]``: P(region r has archetype k)."""
        marg = torch.zeros(NUM_REGIONS, NUM_ARCHETYPES, dtype=DTYPE)
        probs = self.probs.to(DTYPE).unsqueeze(0).expand(NUM_REGIONS, -1).contiguous()
        return marg.scatter_add_(1, ASSIGNMENTS.T.contiguous(), probs)

    def map_assignment(self) -> torch.Tensor:
        """Most probable σ as int64 ``[4]`` (lowest index on ties)."""
        return ASSIGNMENTS[int(torch.argmax(self.probs))].clone()

    def entropy(self) -> float:
        """Shannon entropy of σ in nats (log 360 ≈ 5.886 for the prior)."""
        return float(_entropy(self.probs))

    def sample(self, count: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """``[count, 4]`` assignments drawn i.i.d. ∝ ``probs`` (CPU ``generator``)."""
        idx = torch.multinomial(self.probs, int(count), replacement=True, generator=generator)
        return ASSIGNMENTS[idx]

    def top(self, k: int) -> list[tuple[float, tuple[int, int, int, int]]]:
        """The ``k`` most probable assignments as ``(prob, (σ(N), σ(S), σ(E), σ(W)))``."""
        values, idx = torch.topk(self.probs, min(int(k), NUM_ASSIGNMENTS))
        return [(float(p), tuple(int(x) for x in ASSIGNMENTS[i].tolist())) for p, i in zip(values.tolist(), idx.tolist())]

    def to_public_dict(self) -> dict[str, Any]:
        """JSON-safe summary: per-region archetype marginals (keyed by archetype id, names listed),
        entropy (nats), top-5 assignments and the inconsistency flag. Contains only beliefs."""
        marg = self.region_marginals().tolist()
        return {
            "archetypes": [{"id": a.id, "name": a.name} for a in ARCHETYPES],
            "marginals": {REGIONS[r]: {ARCHETYPES[k].id: float(marg[r][k]) for k in range(NUM_ARCHETYPES)}
                          for r in range(NUM_REGIONS)},
            "entropy": self.entropy(),
            "top": [{"prob": p, "assignment": {REGIONS[r]: ARCHETYPES[k].id for r, k in enumerate(a)},
                     "names": {REGIONS[r]: ARCHETYPES[k].name for r, k in enumerate(a)}} for p, a in self.top(5)],
            "inconsistent": bool(self.inconsistent),
        }


def _region_sizes(observation: Mapping[str, Any]) -> torch.Tensor | None:
    regions = observation.get("regions") or []
    if not regions:
        return None
    sizes = torch.zeros(NUM_REGIONS, dtype=DTYPE)
    for row in regions:
        sizes[region_index(row["id"])] = float(row["voters"])
    if torch.any(sizes <= 0):
        raise ValueError("observation regions must give a positive voter count for every region")
    return sizes


def _gauss_logpdf(y: torch.Tensor, mean: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    z = (y - mean) / sd
    return -0.5 * z * z - torch.log(sd) - 0.5 * _LOG_2PI


def _sampling_sd(p: torch.Tensor, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """``100 √(p(1−p)/m · (n−m)/(n−1))`` in percentage points (no FPC when n = 1); broadcasting."""
    fpc = torch.where(n > 1, (n - m) / (n - 1).clamp_min(1.0), torch.ones_like(n))
    return 100.0 * torch.sqrt(p * (1 - p) / m * fpc)


def _query_terms(queries: Sequence[Mapping[str, Any]], model: AnswerModel, sizes: torch.Tensor,
                 slack_pp: float, drift_pp_per_round: float,
                 sample_fraction: float = 0.10) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-likelihood ``[Q, 360]`` and |z| ``[Q, 360]`` of the sampled query answers (§6.3, R12)."""
    a = torch.tensor([sector_index(q["sector_a"]) for q in queries])
    b = torch.tensor([sector_index(q["sector_b"]) for q in queries])
    r = torch.tensor([region_index(q["region"]) for q in queries])
    t = torch.tensor([float(q.get("round", 0)) for q in queries], dtype=DTYPE)
    y = torch.tensor([float(q["pct_b_over_a"]) for q in queries], dtype=DTYPE)
    n = sizes[r]
    m = torch.tensor([float(q["sample_size"]) if q.get("sample_size") else
                      float(query_sample_size(int(sizes[int(rr)]), sample_fraction)) for q, rr in zip(queries, r)],
                     dtype=DTYPE)
    k = ASSIGNMENTS[:, r].T                                                  # [Q, 360] archetype of region r
    p = model.query_p[k, a.unsqueeze(1), b.unsqueeze(1)]                    # [Q, 360]
    sd = (_sampling_sd(p, m.unsqueeze(1), n.unsqueeze(1))
          + slack_pp + drift_pp_per_round * t.unsqueeze(1))
    mean = 100.0 * p
    return _gauss_logpdf(y.unsqueeze(1), mean, sd), (y.unsqueeze(1) - mean).abs() / sd


def _free_stat_terms(stats: Sequence[Mapping[str, Any]], model: AnswerModel, sizes: torch.Tensor,
                     slack_pp: float, drift_pp_per_round: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-likelihood ``[F·5, 360]`` and |z| of free top-3 statistics (§6.2), sectors independent."""
    w = sizes / sizes.sum()                                                  # [4]
    p = model.top3[ASSIGNMENTS]                                              # [360, 4, 5]
    mean = 100.0 * (w[None, :, None] * p).sum(1)                             # [360, 5]
    sampling = 100.0 * torch.sqrt((w[None, :, None] ** 2 * p * (1 - p) / sizes[None, :, None]).sum(1))
    logs, zs = [], []
    for entry in stats:
        y = torch.tensor([float(entry["top3_pct"][s]) for s in SECTORS], dtype=DTYPE)  # [5]
        sd = sampling + slack_pp + drift_pp_per_round * float(entry.get("round", 0))
        logs.append(_gauss_logpdf(y, mean, sd).T)
        zs.append(((y - mean).abs() / sd).T)
    return torch.cat(logs), torch.cat(zs)


def posterior_from_observation(observation: Mapping[str, Any], model: AnswerModel | None = None, *,
                               slack_pp: float = 1.5, drift_pp_per_round: float = 3.0,
                               use_free_stats: str = "first", hard_z: float | None = 8.0,
                               sample_fraction: float = 0.10) -> Posterior:
    """Posterior over σ from a contract ``Observation`` dict (web_api_v2.md; spec §6.2, §6.3, §10.1).

    Reads only public fields: ``regions[].{id, voters}``, ``free_stats[]`` and ``queries[]``.
    ``use_free_stats``: ``"first"`` (the round-0 / earliest entry — later ones reflect campaign-driven
    drift), ``"all"`` (every entry, sd widened by drift·round) or ``"none"``. ``hard_z`` (None
    disables) zeroes assignments whose prediction is more than ``hard_z`` sd from any observation.
    ``sample_fraction`` gives m for query answers that carry no ``sample_size`` (pre-R12 logs).
    """
    if use_free_stats not in ("first", "all", "none"):
        raise ValueError("use_free_stats must be 'first', 'all' or 'none'")
    queries = list(observation.get("queries") or [])
    stats = list(observation.get("free_stats") or [])
    if use_free_stats == "none":
        stats = []
    elif use_free_stats == "first" and stats:
        stats = [min(stats, key=lambda e: int(e.get("round", 0)))]
    if not queries and not stats:
        return Posterior.uniform()
    sizes = _region_sizes(observation)
    if sizes is None:
        raise ValueError("observation needs regions[].voters to interpret queries/free stats")
    model = model or build_answer_model()

    parts = []
    if queries:
        parts.append(_query_terms(queries, model, sizes, slack_pp, drift_pp_per_round, sample_fraction))
    if stats:
        parts.append(_free_stat_terms(stats, model, sizes, slack_pp, drift_pp_per_round))
    log_lik = torch.cat([lp for lp, _ in parts])                             # [M, 360]
    z = torch.cat([zz for _, zz in parts])
    total = log_lik.sum(0)
    total = torch.where(torch.isfinite(total), total, torch.full_like(total, -math.inf))
    if hard_z is not None:
        total = torch.where((z > hard_z).any(0), torch.full_like(total, -math.inf), total)
    if not torch.isfinite(total).any():
        return Posterior(Posterior.uniform().probs, inconsistent=True)
    return Posterior(torch.softmax(total, dim=0))


def posterior_region_mean(posterior: Posterior, model: AnswerModel | None = None) -> torch.Tensor:
    """``[4, 5]``: E[μ_σ(r)] = Σ_k P(σ(r)=k) μ_k — expected preference mean per region (§2.3).

    ``model`` is accepted for API symmetry; E[v] under Dir(κ μ) is μ for any κ.
    """
    del model
    return posterior.region_marginals() @ archetype_means(dtype=DTYPE)


# ------------------------------------------------------------------------------- query selection
def _candidates() -> list[tuple[int, int, int]]:
    return [(a, b, r) for a, b in SECTOR_PAIRS for r in range(NUM_REGIONS)]


def expected_information_gain(posterior: Posterior, model: AnswerModel, region_sizes: Sequence[int], *,
                              round_index: int = 0, slack_pp: float = 1.5, drift_pp_per_round: float = 3.0,
                              grid_points: int = 401, sample_fraction: float = 0.10,
                              ) -> list[tuple[float, tuple[str, str, str]]]:
    """Mutual information I(σ; Y) (nats) for every query (a < b) × region, sorted descending (§6.3, §10.2).

    Y is one *sampled* answer (R12) from m = round(``sample_fraction`` · n_r) voters, with the same
    sampling sd (incl. finite-population correction) as :func:`posterior_from_observation`. Because a
    repeat draws a fresh sample, an already-asked triple can still carry information (and may be best).

    The answer Y to (a, b, r) depends on σ only through σ(r), so I(σ; Y) = I(σ(r); Y) under the same
    Gaussian answer model as :func:`posterior_from_observation`: with w_k = P(σ(r)=k),
    I = Σ_k w_k ∫ N_k(y) [log N_k(y) − log Σ_j w_j N_j(y)] dy, evaluated by a Riemann sum on a
    per-candidate grid spanning the means ± 8 sd (vectorised over all 40 candidates).
    """
    sizes = torch.as_tensor([float(s) for s in region_sizes], dtype=DTYPE)
    cands = _candidates()
    a = torch.tensor([c[0] for c in cands])
    b = torch.tensor([c[1] for c in cands])
    r = torch.tensor([c[2] for c in cands])
    w = posterior.region_marginals()[r]                                      # [C, 6]
    p = model.query_p[:, a, b].T                                             # [C, 6]
    mean = 100.0 * p
    m = torch.tensor([float(query_sample_size(int(sizes[c[2]]), sample_fraction)) for c in cands], dtype=DTYPE)
    sd = (_sampling_sd(p, m.unsqueeze(1), sizes[r].unsqueeze(1))
          + slack_pp + drift_pp_per_round * float(round_index))
    lo = (mean - 8 * sd).min(1, keepdim=True).values
    hi = (mean + 8 * sd).max(1, keepdim=True).values
    u = torch.linspace(0.0, 1.0, grid_points, dtype=DTYPE)
    y = lo + (hi - lo) * u                                                   # [C, G]
    dy = (hi - lo) / (grid_points - 1)                                       # [C, 1]
    log_nk = _gauss_logpdf(y.unsqueeze(1), mean.unsqueeze(2), sd.unsqueeze(2))  # [C, 6, G]
    log_w = torch.log(w.clamp_min(1e-300)).unsqueeze(2)
    log_mix = torch.logsumexp(log_w + log_nk, dim=1, keepdim=True)          # [C, 1, G]
    integrand = torch.exp(log_nk) * (log_nk - log_mix)                      # [C, 6, G]
    mi = (w * (integrand.sum(2) * dy)).sum(1).clamp_min(0.0)                # [C]
    out = [(float(g), (SECTORS[c[0]], SECTORS[c[1]], REGIONS[c[2]])) for g, c in zip(mi.tolist(), cands)]
    out.sort(key=lambda item: -item[0])
    return out


def best_query(posterior: Posterior, model: AnswerModel, region_sizes: Sequence[int], *,
               asked: Iterable[tuple[str, str, str]] = (), round_index: int = 0, slack_pp: float = 1.5,
               drift_pp_per_round: float = 3.0, sample_fraction: float = 0.10) -> tuple[str, str, str]:
    """Highest-information ``(sector_a, sector_b, region)`` (R12: repeats allowed, so by default nothing
    is excluded — asking the same triple again can be optimal).

    ``asked`` (optional) lists triples to exclude; either orientation of a pair is excluded (the reversed
    question carries the same information). Falls back to the overall best if every candidate is excluded.
    """
    seen = set()
    for sa, sb, reg in asked:
        seen.add((sa, sb, reg))
        seen.add((sb, sa, reg))
    ranked = expected_information_gain(posterior, model, region_sizes, round_index=round_index,
                                       slack_pp=slack_pp, drift_pp_per_round=drift_pp_per_round,
                                       sample_fraction=sample_fraction)
    for _, triple in ranked:
        if triple not in seen:
            return triple
    return ranked[0][1]


__all__ = ["ASSIGNMENTS", "AnswerModel", "Posterior", "SECTOR_PAIRS", "best_query", "build_answer_model",
           "expected_information_gain", "posterior_from_observation", "posterior_region_mean"]
