"""Playbook players (strategies.md Method 1 "Playbook", Method 3 "Adaptive playbook", with Method 2 dials).

Every player here implements the library ``Player`` protocol (``act(observation) -> action dict``) and reads ONLY
the allowlisted observation dict, exactly like ``strategies.players``. The headline players are built so that a
person can follow the same rules in the web UI without a computer:

* survey questions are a fixed list: "ask sector pair (a, b) about the k largest regions";
* hunches are what the UI's region dossiers rank first (``ui_fit_archetypes`` = ``web/lib/insight.ts``
  ``fitArchetypes``, ported line by line); a hunch already used by a larger region is skipped (the public rule
  "no two regions share an archetype");
* the manifesto is the UI's "Blend my hunches" button over a chosen set of regions (``ui_blend`` =
  ``blendSelected``/``blendGuesses``, then the slider rounding ``toIntegerSimplex`` and ``percentToManifesto`` of
  ``web/lib/simplex.ts``, ported exactly);
* spending is counts of workers per region and which regions get media, from the region sizes / media prices,
  the hunches, the "Match with each archetype" meter (``ui_similarity``) and the media "airings" counter.

Comparison arms that are NOT human-executable are flagged by ``PlaybookPlayer.human_executable`` = False:
``hunch_source="bayes_map"`` (joint Bayesian posterior over all 360 assignments incl. the free top-3 statistic) and
``election_day=True`` (the blend of the regions' *forecast election-day* preferences, from simulated worlds).

:class:`AdaptivePlaybook` adds branches on what the UI shows: average "Friends per voter" over the four regions
(campaign screen), the electorate size N and the number of rounds T (chosen at setup). The optional
``inter_share`` branch (share of friendships between regions, from ``inter_edges``) is in the observation/API but
is NOT shown in the UI; it is off by default.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

import torch

from dumbocracy.config import GameConfig
from dumbocracy.dynamics.campaign import media_cost
from dumbocracy.sectors import ARCHETYPES, NUM_REGIONS, REGIONS, SECTORS
from dumbocracy.strategies.players import _Timed, _treasury, public_config, simplex_list
from dumbocracy.strategies.worlds import public_sizes

# ============================================================================== UI replicas (web/lib/*.ts)
KAPPA = 60.0                 # web/lib/constants.ts KAPPA
FIT_SLACK_PP = 2.5           # web/lib/insight.ts FIT_SLACK_PP
SIMPLEX_TOTAL = 100          # web/lib/simplex.ts SIMPLEX_TOTAL
BLEND_MIN, BLEND_MAX = 2, 4  # web/lib/constants.ts


def ui_erf(x: float) -> float:
    """Abramowitz–Stegun erf of ``insight.ts`` (|err| < 1.5e-7), same operation order."""
    s = (x > 0) - (x < 0)
    a = abs(x)
    t = 1 / (1 + 0.3275911 * a)
    y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-a * a)
    return s * y


def ui_normal_cdf(z: float) -> float:
    return 0.5 * (1 + ui_erf(z / math.sqrt(2)))


def ui_predict_pct(mean, a: int, b: int, kappa: float = KAPPA) -> float:
    """``predictPctBOverA``: predicted % of voters with v[b] > v[a] under Dirichlet(κ μ) (normal approximation)."""
    ma, mb = mean[a], mean[b]
    variance = (ma * (1 - ma) + mb * (1 - mb) + 2 * ma * mb) / (kappa + 1)
    sd = math.sqrt(max(variance, 1e-12))
    return 100 * ui_normal_cdf((mb - ma) / sd)


def _pool_key(q: dict) -> str:
    return f"{q['phase']}:{q['round']}:{q['region']}:{q['sector_a']}>{q['sector_b']}"


def ui_group_answers(queries: list[dict]) -> list[dict]:
    """``groupAnswers`` (web/lib/survey.ts): pool by (phase, round, triple); the latest repeat carries the pooled
    estimate. Groups in first-seen order (JS ``Map`` insertion order)."""
    groups: dict[str, list[dict]] = {}
    for q in queries:
        groups.setdefault(_pool_key(q), []).append(q)
    out = []
    for key, answers in groups.items():
        latest = answers[0]
        for q in answers:
            if q["repeat_index"] >= latest["repeat_index"]:
                latest = q
        out.append({"key": key, "latest": latest, "pct": latest["pooled_pct"], "moe": latest["pooled_moe"]})
    return out


def ui_fit_archetypes(observation: dict, region: str, slack: float = FIT_SLACK_PP) -> list[dict]:
    """``fitArchetypes`` (web/lib/insight.ts): the region dossier's ranking of the archetype library.

    Returns the library as dicts ``{id, posterior, logLik, rmse, n}`` sorted by posterior (descending, stable:
    ties keep library order; posterior NaN = no evidence counts as 0 for the sort)."""
    lib = observation["archetype_library"]
    groups = ui_group_answers([q for q in observation.get("queries", []) if q["region"] == region])
    evidence = [{"a": SECTORS.index(g["latest"]["sector_a"]), "b": SECTORS.index(g["latest"]["sector_b"]),
                 "pct": g["pct"], "sd": math.sqrt((g["moe"] / 1.96) ** 2 + slack ** 2)} for g in groups]
    fits = []
    for arch in lib:
        if not evidence:
            fits.append({"id": arch["id"], "name": arch["name"], "rmse": float("nan"), "logLik": 0.0,
                         "posterior": float("nan"), "n": 0})
            continue
        ll = wse = w = 0.0
        for e in evidence:
            gap = ui_predict_pct(arch["mean"], e["a"], e["b"]) - e["pct"]
            ll += -0.5 * (gap / e["sd"]) ** 2 - math.log(e["sd"])
            wse += gap ** 2 / e["sd"] ** 2
            w += 1 / e["sd"] ** 2
        fits.append({"id": arch["id"], "name": arch["name"], "rmse": math.sqrt(wse / w), "logLik": ll,
                     "posterior": 0.0, "n": len(evidence)})
    if evidence:
        top = max(f["logLik"] for f in fits)
        z = sum(math.exp(f["logLik"] - top) for f in fits)
        for f in fits:
            f["posterior"] = math.exp(f["logLik"] - top) / z
    key = lambda f: 0.0 if math.isnan(f["posterior"]) else f["posterior"]  # noqa: E731
    return sorted(fits, key=lambda f: -key(f))  # Python sort is stable like Array.prototype.sort


def ui_apportion(amount: float, weights: list[float]) -> list[int]:
    """``apportion`` (web/lib/simplex.ts): largest-remainder integers summing to ``amount``."""
    n = len(weights)
    if n == 0:
        return []
    amt = max(0, _js_round(amount))
    w = [x if (math.isfinite(x) and x > 0) else 0.0 for x in weights]
    sum_w = sum(w)
    basis = w if sum_w > 0 else [1.0] * n
    sum_b = sum_w if sum_w > 0 else n
    raw = [(amt * x) / sum_b for x in basis]
    out = [math.floor(x) for x in raw]
    rest = amt - sum(out)
    order = sorted(({"i": i, "frac": x - math.floor(x), "w": basis[i]} for i, x in enumerate(raw)),
                   key=lambda o: (-o["frac"], -o["w"], o["i"]))
    k = 0
    while rest > 0:
        out[order[k]["i"]] += 1
        rest -= 1
        k = (k + 1) % n
    return out


def _js_round(x: float) -> int:
    """JavaScript ``Math.round`` (halves round towards +∞)."""
    return math.floor(x + 0.5)


def ui_to_integer_simplex(weights: list[float], total: int = SIMPLEX_TOTAL) -> list[int]:
    return ui_apportion(total, weights)


def ui_percent_to_manifesto(values: list[int]) -> list[float]:
    """``percentToManifesto``: integer percents → the vector the UI commits (drift added to the first max)."""
    s = sum(values)
    out = [v / s for v in values]
    drift = 1 - sum(out)
    i_max = out.index(max(out))
    out[i_max] += drift
    return out


def ui_similarity(a, b) -> float:
    """``similarity`` = 1 − ½‖a − b‖₁ (the "Match with each archetype" meter)."""
    return 1 - sum(abs(x - y) for x, y in zip(a, b)) / 2


def ui_blend_raw(observation: dict, hunches: dict[str, str], region_ids: list[str]) -> list[float] | None:
    """``blendSelected`` → ``blendGuesses``: voter-weighted mean of the chosen regions' hunch archetypes
    (2–4 distinct regions, all with a hunch; else None). Accumulates in ``observation["regions"]`` order."""
    chosen = list(dict.fromkeys(region_ids))
    lib = {a["id"]: a for a in observation["archetype_library"]}
    if len(chosen) < BLEND_MIN or len(chosen) > BLEND_MAX:
        return None
    if any(hunches.get(r) not in lib for r in chosen):
        return None
    acc = [0.0] * 5
    w = 0
    for r in observation["regions"]:
        if r["id"] not in chosen:
            continue
        mean = lib[hunches[r["id"]]]["mean"]
        for k, m in enumerate(mean):
            acc[k] += m * r["voters"]
        w += r["voters"]
    if w == 0:
        return None
    return [x / w for x in acc]


def ui_blend(observation: dict, hunches: dict[str, str], region_ids: list[str]) -> list[float] | None:
    """The manifesto the UI commits after "Blend my hunches" → "Apply blend" → "Commit manifesto"."""
    raw = ui_blend_raw(observation, hunches, region_ids)
    if raw is None:
        return None
    return ui_percent_to_manifesto(ui_to_integer_simplex(raw))


def ui_regions_by_size(observation: dict) -> list[str]:
    """Region ids, most voters first (ties: N, S, E, W order — the UI's stable sort of ``obs.regions``)."""
    return [r["id"] for r in sorted(observation["regions"], key=lambda r: -r["voters"])]


def ui_largest_hunch_regions(observation: dict, hunches: dict[str, str], k: int) -> list[str]:
    """``largestHunchRegions``: the "2 largest" / "3 largest" shortcut buttons of the blend picker."""
    lib = {a["id"] for a in observation["archetype_library"]}
    return [r for r in ui_regions_by_size(observation) if hunches.get(r) in lib][:k]


# ============================================================================== playbook rules
#: Fixed survey question sets (sector_a, sector_b): the answer is "% of the region preferring more b than a".
#: "AI" — one question, Agriculture vs Industry: spreads the six archetypes over 0–100% (A0 0, A3 22, A5 50, A2 78,
#:        A4 98, A1 100) but cannot tell Knowledge hub from Industrial belt.
#: "WA+IE" — two questions, Welfare vs Agriculture and Industry vs Environment: the two-question set whose predicted
#:        answers are furthest apart for every pair of archetypes (≥ 30 points), i.e. identifies the region.
SURVEY_SETS: dict[str, tuple[tuple[str, str], ...]] = {
    "AI": (("Agri", "Industry"),),
    "WA+IE": (("Welfare", "Agri"), ("Industry", "Env")),
}

MANIFESTO_RULES = ("two_largest", "three_largest", "all", "coalition", "anchor", "divided")
DIVIDED_TV = 0.35  # R1/R4/X15: the two largest regions "disagree" when their archetypes differ by TV ≥ 0.35


def archetype_tv(a_id: str, b_id: str) -> float:
    ma = next(a.mean for a in ARCHETYPES if a.id == a_id)
    mb = next(a.mean for a in ARCHETYPES if a.id == b_id)
    return 0.5 * sum(abs(x - y) for x, y in zip(ma, mb))


def tv_table() -> dict[str, dict[str, float]]:
    """The 6×6 archetype distance table a human uses for "closest region" and "do the two largest disagree"."""
    return {a.id: {b.id: round(archetype_tv(a.id, b.id), 3) for b in ARCHETYPES} for a in ARCHETYPES}


def survey_plan(observation: dict, survey_regions: int, survey_set: str) -> list[tuple[str, str, str]]:
    """The fixed question list: for each of the ``survey_regions`` largest regions, every pair of the set."""
    pairs = SURVEY_SETS[survey_set]
    return [(a, b, r) for r in ui_regions_by_size(observation)[:survey_regions] for a, b in pairs]


def dossier_hunches(observation: dict) -> dict[str, str]:
    """Hunch per surveyed region = the dossier's top archetype; regions are processed largest first and an
    archetype already given to a larger region is skipped (take the next one in that dossier)."""
    hunches: dict[str, str] = {}
    used: set[str] = set()
    asked = {q["region"] for q in observation.get("queries", [])}
    for r in ui_regions_by_size(observation):
        if r not in asked:
            continue
        for f in ui_fit_archetypes(observation, r):
            if f["id"] not in used:
                hunches[r] = f["id"]
                used.add(f["id"])
                break
    return hunches


def coalition_regions(observation: dict, hunches: dict[str, str]) -> list[str]:
    """Largest regions with a hunch, added until they hold more than half of all voters (at least 2)."""
    sizes = {r["id"]: r["voters"] for r in observation["regions"]}
    total = sum(sizes.values())
    chosen: list[str] = []
    for r in ui_largest_hunch_regions(observation, hunches, 4):
        chosen.append(r)
        if len(chosen) >= 2 and 2 * sum(sizes[x] for x in chosen) > total:
            break
    return chosen


def anchor_regions(observation: dict, hunches: dict[str, str], exclude_second: bool = False) -> list[str]:
    """Largest hunched region + the hunched region whose archetype is closest to it (TV table; ties → larger).
    ``exclude_second``: skip the second-largest region (used when the two largest disagree)."""
    order = ui_largest_hunch_regions(observation, hunches, 4)
    if len(order) < 2:
        return order
    first, rest = order[0], order[1:]
    if exclude_second and len(rest) > 1:
        rest = rest[1:]
    best = min(rest, key=lambda r: (archetype_tv(hunches[first], hunches[r]), rest.index(r)))
    return [first, best]


def blend_regions(observation: dict, hunches: dict[str, str], rule: str) -> list[str]:
    """Which regions to tick in the blend picker under ``rule`` (see MANIFESTO_RULES)."""
    if rule == "two_largest":
        return ui_largest_hunch_regions(observation, hunches, 2)
    if rule == "three_largest":
        return ui_largest_hunch_regions(observation, hunches, 3)
    if rule == "all":
        return ui_largest_hunch_regions(observation, hunches, 4)
    if rule == "coalition":
        return coalition_regions(observation, hunches)
    if rule == "anchor":
        return anchor_regions(observation, hunches)
    if rule == "divided":
        two = ui_largest_hunch_regions(observation, hunches, 2)
        if len(two) == 2 and archetype_tv(hunches[two[0]], hunches[two[1]]) >= DIVIDED_TV:
            return anchor_regions(observation, hunches, exclude_second=True)
        return coalition_regions(observation, hunches)
    raise ValueError(f"unknown manifesto rule {rule!r}")


def mean_friends(observation: dict) -> float:
    """Plain average of the four regions' "Friends per voter" (campaign screen)."""
    return sum(float(r["mean_degree"]) for r in observation["regions"]) / len(observation["regions"])


def inter_share(observation: dict) -> float:
    """Share of friendships that cross regions (census ``inter_edges``; NOT shown in the UI)."""
    inter = sum(int(x["edges"]) for x in observation["inter_edges"])
    intra = sum(int(r["intra_edges"]) for r in observation["regions"])
    return inter / max(1, inter + intra)


# ============================================================================== the player
@dataclass
class PlaybookPlayer(_Timed):
    """Method 1 playbook with Method 2 dials.

    Survey: ``survey_set`` questions about each of the ``survey_regions`` largest regions (SURVEY phase only).
    Hunches: ``hunch_source`` "dossier" (UI ranking, human) or "bayes_map" (comparison arm, NOT human).
    Manifesto: "Blend my hunches" over ``blend_regions(rule)``; ``election_day`` (NOT human) replaces each
    hunch archetype by the forecast election-day mean of that region (simulated worlds, no campaign).
    ``lean`` (optional Method 2 dial; needs hand arithmetic on five numbers, no single UI button): the manifesto
    is λ × (rule's blend) + (1 − λ) × (blend of all hunched regions), rounded to whole percents like the sliders.
    Spending (round t = 1..T):
      * workers: ``workers`` bought in round ``worker_round`` (default 1), placed by ``worker_place``:
        "friendly" = all in the hunched region whose archetype matches the manifesto best (match meter),
        "largest" = all in the largest region, "size" = spread over the four regions by voters;
        ``workers2`` more in round ``worker_round + 1`` (same placement);
      * media in round 1 / 2: the first ``media1`` / ``media2`` regions in ``media_order`` ("size": largest first;
        "match": best hunch match first, unhunched regions after, by size) that the money still covers;
      * rounds ≥ 3: ``later`` "front" = air every region you can afford (fewest airings first, then largest),
        "even" = the same but only up to treasury / rounds left; the last round always spends everything
        (media first, then workers in the friendly region with whatever is left).
    """

    player_id: str = "p0"
    seed: int = 0
    survey_regions: int = 2
    survey_set: str = "WA+IE"
    hunch_source: str = "dossier"
    manifesto_rule: str = "coalition"
    election_day: bool = False
    lean: float | None = None
    workers: int = 3
    workers2: int = 0
    worker_round: int = 1
    worker_place: str = "friendly"
    media1: int = 4
    media2: int = 4
    media_order: str = "size"
    later: str = "front"
    config: GameConfig | None = None
    device: str = "cpu"
    latency: dict[str, list[float]] = field(default_factory=dict)
    record: dict[str, Any] = field(default_factory=dict)

    @property
    def human_executable(self) -> bool:
        return self.hunch_source == "dossier" and not self.election_day

    def dials(self) -> dict[str, Any]:
        keys = ("survey_regions", "survey_set", "hunch_source", "manifesto_rule", "election_day", "workers",
                "workers2", "worker_round", "worker_place", "media1", "media2", "media_order", "later", "lean")
        return {k: getattr(self, k) for k in keys}

    # ------------------------------------------------------------------ beliefs
    def hunches(self, observation: dict) -> dict[str, str]:
        if self.hunch_source == "dossier":
            return dossier_hunches(observation)
        if self.hunch_source == "bayes_map":
            from dumbocracy.strategies.bayes import build_answer_model, posterior_from_observation
            sigma = posterior_from_observation(observation, build_answer_model()).map_assignment().tolist()
            return {REGIONS[r]: ARCHETYPES[int(k)].id for r, k in enumerate(sigma)}
        raise ValueError(f"unknown hunch_source {self.hunch_source!r}")

    # ------------------------------------------------------------------ manifesto
    def manifesto(self, observation: dict) -> list[float]:
        hunches = self.hunches(observation)
        regions = blend_regions(observation, hunches, self.manifesto_rule)
        if len(regions) < BLEND_MIN:  # cannot press the blend button: fall back to the top-3 statistic
            stat = min(observation["free_stats"], key=lambda e: int(e["round"]))["top3_pct"]
            self.record["fallback"] = "top3"
            return simplex_list([stat[s] for s in SECTORS])
        self.record.update(hunches=hunches, blend_regions=regions)
        if self.lean is not None and not self.election_day:
            # Method 2 dial (hand arithmetic, see docstring): λ × blend(rule regions) + (1 − λ) × blend(all hunched)
            every = ui_largest_hunch_regions(observation, hunches, 4)
            a, b = ui_blend_raw(observation, hunches, regions), ui_blend_raw(observation, hunches, every)
            if b is not None:
                lam = float(self.lean)
                return ui_percent_to_manifesto(ui_to_integer_simplex([lam * x + (1 - lam) * y for x, y in zip(a, b)]))
        if not self.election_day:
            return ui_blend(observation, hunches, regions)
        means = self._election_day_means(observation, hunches)
        sizes = {r["id"]: r["voters"] for r in observation["regions"]}
        acc = [0.0] * 5
        for r in regions:
            for k in range(5):
                acc[k] += sizes[r] * means[r][k]
        w = sum(sizes[r] for r in regions)
        return ui_percent_to_manifesto(ui_to_integer_simplex([x / w for x in acc]))

    def _election_day_means(self, observation: dict, hunches: dict[str, str]) -> dict[str, list[float]]:
        """NOT human-executable: forecast each region's mean preference on election day (no campaign) on four
        simulated census-consistent worlds whose σ is the hunches, completed by the Bayes MAP for regions
        without a hunch."""
        from dumbocracy.config import SeedStreams
        from dumbocracy.strategies.bayes import ASSIGNMENTS, build_answer_model, posterior_from_observation
        from dumbocracy.strategies.oracle import anticipate
        from dumbocracy.strategies.worlds import sample_worlds
        ids = [a.id for a in ARCHETYPES]
        post = posterior_from_observation(observation, build_answer_model())
        ok = torch.ones(ASSIGNMENTS.shape[0], dtype=torch.bool)
        for r, a in hunches.items():
            ok &= ASSIGNMENTS[:, REGIONS.index(r)] == ids.index(a)
        probs = torch.where(ok, post.probs, torch.zeros_like(post.probs))
        sigma = ASSIGNMENTS[int(torch.argmax(probs if probs.sum() > 0 else post.probs))]
        cfg = public_config(observation, self.config)
        seed = SeedStreams(int(self.seed) * 1_000_003 + int(observation["seed"])).derive("strategy", "playbook_eday")
        worlds = sample_worlds(observation, sigma.repeat(4, 1), cfg, num_voters=min(1600, int(observation["num_voters"])),
                               seed=seed, device=self.device)
        sums = torch.zeros(NUM_REGIONS, 5, dtype=torch.float64)
        counts = torch.zeros(NUM_REGIONS, dtype=torch.float64)
        for w in worlds:
            v = anticipate(w, cfg).to(torch.float64).cpu()
            reg = w.region_of.cpu()
            sums.index_add_(0, reg, v)
            counts += torch.bincount(reg, minlength=NUM_REGIONS).to(torch.float64)
        means = sums / counts.unsqueeze(1)
        return {REGIONS[r]: means[r].tolist() for r in range(NUM_REGIONS)}

    # ------------------------------------------------------------------ spending
    def _friendly(self, observation: dict, hunches: dict[str, str]) -> str:
        """Region whose hunch archetype has the highest match with the committed manifesto (ties → larger);
        the largest region if no region has a hunch."""
        lib = {a["id"]: a["mean"] for a in observation["archetype_library"]}
        order = ui_regions_by_size(observation)
        cands = [r for r in order if hunches.get(r) in lib]
        if not cands:
            return order[0]
        b = observation["manifesto"]
        return max(cands, key=lambda r: (ui_similarity(b, lib[hunches[r]]), -order.index(r)))

    def _media_rank(self, observation: dict, hunches: dict[str, str]) -> list[str]:
        order = ui_regions_by_size(observation)
        if self.media_order == "size":
            return order
        if self.media_order == "match":
            lib = {a["id"]: a["mean"] for a in observation["archetype_library"]}
            b = observation["manifesto"]
            score = {r: (ui_similarity(b, lib[hunches[r]]) if hunches.get(r) in lib else -1.0) for r in order}
            return sorted(order, key=lambda r: (-score[r], order.index(r)))
        raise ValueError(f"unknown media_order {self.media_order!r}")

    def _hire(self, observation: dict, hunches: dict[str, str], count: int) -> list[int]:
        if count <= 0:
            return [0] * NUM_REGIONS
        if self.worker_place == "size":
            from dumbocracy.strategies.planner import split_by_weight
            return list(split_by_weight(count, [float(s) for s in public_sizes(observation)]))
        if self.worker_place == "friendly":
            target = self._friendly(observation, hunches)
        elif self.worker_place == "largest":
            target = ui_regions_by_size(observation)[0]
        else:
            raise ValueError(f"unknown worker_place {self.worker_place!r}")
        hires = [0] * NUM_REGIONS
        hires[REGIONS.index(target)] = count
        return hires

    def allocation(self, observation: dict) -> dict[str, Any]:
        cfg = public_config(observation, self.config)
        sizes = public_sizes(observation)
        price = {r: Decimal(observation["costs"]["media_by_region"][r]) for r in REGIONS}
        fee = Decimal(observation["costs"]["worker"])
        money = _treasury(observation)
        t, T = int(observation["round"]), int(observation["rounds"])
        left = T - t + 1
        hunches = self.hunches(observation)
        hires = [0] * NUM_REGIONS
        want = self.workers if t == self.worker_round else (self.workers2 if t == self.worker_round + 1 else 0)
        if want > 0:
            n = min(want, int(money // fee)) if fee > 0 else want
            hires = self._hire(observation, hunches, n)
            money -= fee * sum(hires)
        media = [False] * NUM_REGIONS
        airings = {r: int(observation.get("media_airings", {}).get(r, 0)) for r in REGIONS}
        if t <= 2 and t < T:
            count = self.media1 if t == 1 else self.media2
            for r in self._media_rank(observation, hunches):
                if sum(media) >= count:
                    break
                if price[r] <= money:
                    media[REGIONS.index(r)] = True
                    money -= price[r]
        else:
            last = t == T
            cap = money if (last or self.later == "front") else (money / left).quantize(Decimal("0.01"))
            order = sorted(ui_regions_by_size(observation), key=lambda r: airings[r])  # stable: larger first
            spent = Decimal("0")
            for r in order:
                if spent + price[r] <= cap:
                    media[REGIONS.index(r)] = True
                    spent += price[r]
            money -= spent
            if last and fee > 0 and money >= fee:
                extra = int(money // fee)
                friendly = REGIONS.index(self._friendly(observation, hunches))
                hires[friendly] += extra
                money -= fee * extra
        assert media_cost(sizes, media, cfg.media.base_cost) + fee * sum(hires) <= _treasury(observation)
        return {"type": "allocate", "hires_by_region": hires, "media_regions": media}

    # ------------------------------------------------------------------ protocol
    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        if observation["phase"] == "SURVEY":
            plan = survey_plan(observation, self.survey_regions, self.survey_set)
            asked = [q for q in observation.get("queries", []) if q.get("phase") == "SURVEY"]
            cost = Decimal(observation["costs"]["query"])
            if len(asked) < len(plan) and "query" in observation["legal_actions"] and _treasury(observation) >= cost:
                a, b, r = plan[len(asked)]
                action = {"type": "query", "sector_a": a, "sector_b": b, "region": r}
            else:
                action = {"type": "commit", "manifesto": self.manifesto(observation)}
        else:
            action = self.allocation(observation)
        self._record(action["type"], time.perf_counter() - start)
        return action


# ============================================================================== adaptive playbook
#: Tuned headline playbook (s2_tune.json → chosen; training seeds 13000–13099): 1 question (Agriculture vs Industry)
#: in each of the 4 regions, blend ALL four hunches, round 1: 3 workers split by region size + media in all 4
#: regions, round 2: media in all 4 regions (the treasury is then spent).
TUNED_PLAYBOOK: dict[str, Any] = {"survey_regions": 4, "survey_set": "AI", "manifesto_rule": "all", "workers": 3,
                                  "worker_place": "size", "media1": 4, "media2": 4, "media_order": "size",
                                  "later": "even"}
#: Tuned branch table and thresholds (s3_tune.json → chosen; training seeds 13000–13039).
DEFAULT_BRANCHES: dict[str, dict[str, Any]] = {
    "sparse": {"workers": 6, "worker_place": "size", "media1": 4, "media2": 0, "later": "even"},
    "small_or_long": {"workers": 2, "worker_place": "size", "media1": 4, "media2": 4, "later": "front"},
    "short_large": {"workers": 12, "worker_place": "size", "media1": 4, "media2": 2, "later": "front"},
}


@dataclass
class AdaptivePlaybook(PlaybookPlayer):
    """Method 3: the playbook whose spending dials switch on visible conditions.

    Branches, checked in this order (first match wins; otherwise the base dials):
      * "small_or_long": N ≤ ``small_n`` or T ≥ ``long_t``;
      * "sparse": average "Friends per voter" < ``sparse_k``;
      * "short_large": T ≤ ``short_t`` and N ≥ ``large_n``;
      * "fragmented" (only if ``fragmented_share`` is set; NOT human: inter-region friendship share from the
        census, not shown in the UI): inter share < ``fragmented_share``.
    Each branch overrides any of the dials in ``branches[name]``."""

    sparse_k: float = 22.5
    small_n: int = 1500
    long_t: int = 8
    short_t: int = 3
    large_n: int = 1500
    fragmented_share: float | None = None
    branches: dict[str, dict[str, Any]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_BRANCHES.items()})

    def branch(self, observation: dict) -> str:
        n, T = int(observation["num_voters"]), int(observation["rounds"])
        if (n <= self.small_n or T >= self.long_t) and "small_or_long" in self.branches:
            return "small_or_long"
        if mean_friends(observation) < self.sparse_k and "sparse" in self.branches:
            return "sparse"
        if T <= self.short_t and n >= self.large_n and "short_large" in self.branches:
            return "short_large"
        if self.fragmented_share is not None and inter_share(observation) < self.fragmented_share \
                and "fragmented" in self.branches:
            return "fragmented"
        return "base"

    @property
    def human_executable(self) -> bool:
        return super().human_executable and self.fragmented_share is None

    def allocation(self, observation: dict) -> dict[str, Any]:
        name = self.branch(observation)
        self.record.setdefault("branch", name)
        if name == "base":
            return super().allocation(observation)
        inner = replace(self, branches={}, latency={}, record={})
        for k, v in self.branches[name].items():
            setattr(inner, k, v)
        return PlaybookPlayer.allocation(inner, observation)
