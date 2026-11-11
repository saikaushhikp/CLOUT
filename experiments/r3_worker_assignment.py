"""R3 — Worker assignment scenarios: where should a fixed number of party workers go?

Command (writes everything under reports/v3/experimentation/):
  OMP_NUM_THREADS=4 python experiments/r3_worker_assignment.py --seeds 50 --first-seed 11000 --device cuda:0
  python experiments/r3_worker_assignment.py --plot-only        # figures from the saved r3.json only

Owner's question: "Given a certain population spread across the regions and a fixed number of party workers, how do
you assign/use the party workers in different scenarios such as antagonistic preference, aligned preference, ...?"

Design. N = 10⁴, T = 5, ₫500, ₫20 workers, ORACLE manifesto per world, workers only (no media), all K workers bought
in round 1 (X4: early buying dominates). Seeds 11000–11049 (the X4 block), paired across arms.
* Population scenarios (``ElectorateConfig.region_dirichlet_a``; the 15% floor R7 still applies):
  balanced (a = 50), default (a = 2), skewed (a = 0.5).
* Region roles per world, from the round-0 TRUE regional mean P under that world's manifesto b
  (:func:`assign_roles`): aligned = highest mean P; antagonistic = lowest; swing = of the two remaining regions the
  one whose mean P is closest to 0.5; other = the last region. Ties are broken by region index (lower first), so the
  four roles are always four distinct regions.
* Arms, K ∈ {4, 10}: none; all→aligned; all→swing; all→antagonistic; aligned+swing (half each); even over the 4
  regions; proportional to region size. Integer splits by largest remainder (:func:`largest_remainder`; ties go to
  the larger region, then the lower index).
* Per game: E (expected share), official win, converts per region (accumulated from the per-round ``converted``
  mask through ``run_batch``'s observer hook), final regional mean P (from ``keep_final``).
Stats: mean with 95% t-CI; paired Wilcoxon (E) and exact McNemar (wins) for each arm vs none and every pair of
target arms; Holm within each family (scenario × K).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_parity import parity_rel  # noqa: E402
from x_policies import NO_MEDIA, affordable_workers  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig  # noqa: E402
from dumbocracy.dynamics.campaign import Allocation  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.strategies.oracle import oracle_manifesto  # noqa: E402
from dumbocracy.voting import similarity, vote_probability  # noqa: E402

OUT = xc.ROOT / "reports" / "v3" / "experimentation"
RAW_PATH = OUT / "raw" / "r3.jsonl"
JSON_PATH = OUT / "r3.json"
LOG_PATH = OUT / "logs" / "r3.log"
FIG = OUT / "figures"

SCENARIOS = {"balanced": 50.0, "default": 2.0, "skewed": 0.5}
KS = (4, 10)
TARGETS = ("aligned", "swing", "antagonistic", "aligned_swing", "even", "proportional")
ROLES = ("aligned", "swing", "antagonistic", "other")


# ============================================================================== helpers (unit-tested)
def assign_roles(support: list[float]) -> dict[str, int]:
    """Region roles from regional mean P: aligned = max, antagonistic = min, swing = of the remaining two the one
    closest to 0.5, other = the last. Ties by lower region index; the four roles are distinct regions."""
    if len(support) != 4:
        raise ValueError("need 4 regional values")
    order = sorted(range(4), key=lambda r: (-support[r], r))
    aligned, antagonistic = order[0], order[-1]
    rest = sorted(order[1:3], key=lambda r: (abs(support[r] - 0.5), r))
    return {"aligned": aligned, "swing": rest[0], "antagonistic": antagonistic, "other": rest[1]}


def largest_remainder(total: int, weights: list[float], priority: list[int] | None = None) -> tuple[int, ...]:
    """Integer split of ``total`` proportional to ``weights`` (largest remainder). Equal remainders go to the
    region listed first in ``priority`` (default: region index)."""
    s = float(sum(weights))
    if total <= 0 or s <= 0:
        return tuple(0 for _ in weights)
    rank = {r: i for i, r in enumerate(priority if priority is not None else range(len(weights)))}
    raw = [total * w / s for w in weights]
    base = [int(x // 1) for x in raw]
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - base[i]), rank[i]))
    for i in order[: total - sum(base)]:
        base[i] += 1
    return tuple(base)


def target_hires(target: str, K: int, roles: dict[str, int], sizes: list[int]) -> tuple[int, int, int, int]:
    """Workers per region for one arm."""
    by_size = sorted(range(4), key=lambda r: (-sizes[r], r))
    one = lambda r: [1.0 if i == r else 0.0 for i in range(4)]  # noqa: E731
    if target in ("aligned", "swing", "antagonistic"):
        w = one(roles[target])
    elif target == "aligned_swing":
        w = [a + b for a, b in zip(one(roles["aligned"]), one(roles["swing"]))]
        return largest_remainder(K, w, [roles["aligned"], roles["swing"], roles["antagonistic"], roles["other"]])
    elif target == "even":
        w = [1.0] * 4
    elif target == "proportional":
        w = [float(x) for x in sizes]
    else:
        raise ValueError(f"unknown target {target!r}")
    return largest_remainder(K, w, by_size)  # type: ignore[return-value]


@dataclass(frozen=True)
class RoundOneHires:
    """Buy ``hires`` (per region) in round 1 only, no media; clipped to what the treasury affords."""

    hires: tuple[int, int, int, int]

    def __call__(self, ctx) -> Allocation:
        if ctx.round != 0 or sum(self.hires) == 0:
            return Allocation((0, 0, 0, 0), NO_MEDIA)
        if sum(self.hires) > affordable_workers(ctx, ctx.budget):
            raise ValueError("treasury cannot afford the requested workers")
        return Allocation(tuple(int(x) for x in self.hires), NO_MEDIA)  # type: ignore[arg-type]


def regional_mean_p(v: torch.Tensor, b: torch.Tensor, region_of: torch.Tensor, cfg: GameConfig) -> list[float]:
    p = vote_probability(similarity(v, b), cfg.voting.sensitivity, cfg.voting.threshold).to(torch.float64)
    sums = torch.zeros(4, dtype=torch.float64, device=p.device).index_add_(0, region_of, p)
    cnt = torch.bincount(region_of, minlength=4).to(torch.float64)
    return (sums / cnt.clamp_min(1)).tolist()


# ============================================================================== playing
class Log:
    def __init__(self, path: Path | None):
        self.path = path

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        if self.path is not None:
            with open(self.path, "a") as f:
                f.write(msg + "\n")


def make_world(seed: int, a: float, base: GameConfig, device) -> dict:
    cfg = dataclasses.replace(base, seed=seed, electorate=dataclasses.replace(base.electorate, region_dirichlet_a=a))
    e = generate_electorate(cfg, device=device)
    b = oracle_manifesto(e, cfg)
    sizes = [int(x) for x in e.sizes.tolist()]
    p0 = regional_mean_p(e.preferences, b, e.region_of, cfg)
    return {"seed": seed, "e": e, "b": b, "cfg": cfg, "sizes": sizes, "p0": p0, "roles": assign_roles(p0),
            "gini": xc._gini(sizes)}


def play(jobs: list[dict], cfg: GameConfig) -> list[dict]:
    """jobs: {"world", "arm", "K", "target", "hires"}; one run_batch call for all of them."""
    es = [j["world"]["e"] for j in jobs]
    B, n = len(es), es[0].num_voters
    conv = {}

    def observer(t, info):
        game = info["game"]
        idx = game.world_of * 4 + game.region_of
        c = torch.zeros(B * 4, dtype=torch.float64, device=idx.device).index_add_(0, idx, info["converted"].to(torch.float64))
        conv["by_region"] = c if "by_region" not in conv else conv["by_region"] + c

    outs = run_batch(es, cfg, torch.stack([j["world"]["b"] for j in jobs]), [RoundOneHires(j["hires"]) for j in jobs],
                     seeds=[j["world"]["seed"] for j in jobs], observer=observer, keep_final=True)
    by_region = conv["by_region"].view(B, 4).cpu().tolist()
    rows = []
    for k, (j, o) in enumerate(zip(jobs, outs)):
        w = j["world"]
        final_p = regional_mean_p(o.final_preferences, w["b"], w["e"].region_of, cfg)
        hires = [list(a.hires_by_region) for a in o.allocations]
        rows.append({"seed": w["seed"], "arm": j["arm"], "K": j["K"], "target": j["target"],
                     "expected": o.result.expected_share, "victory": bool(o.result.victory),
                     "share": o.result.vote_share, "spent": str(o.spent),
                     "hires_by_round": [sum(h) for h in hires],
                     "hires_by_region": [sum(h[r] for h in hires) for r in range(4)],
                     "converts": int(sum(o.converted)), "converts_by_region": [int(x) for x in by_region[k]],
                     "final_p_by_region": final_p, "trajectory": [round(x, 6) for x in o.share_trajectory]})
    return rows


def arms_for(world: dict) -> list[dict]:
    jobs = [{"world": world, "arm": "none", "K": 0, "target": "none", "hires": (0, 0, 0, 0)}]
    for K in KS:
        for t in TARGETS:
            jobs.append({"world": world, "arm": f"K{K}|{t}", "K": K, "target": t,
                         "hires": target_hires(t, K, world["roles"], world["sizes"])})
    return jobs


# ============================================================================== summary
def by_role(vals: list[float], roles: dict[str, int]) -> dict[str, float]:
    return {role: vals[roles[role]] for role in ROLES}


def summarise(rows: list[dict], worlds: dict[str, list[dict]]) -> dict:
    out: dict = {"scenarios": {}, "tests": {}}
    idx = {(r["scenario"], r["arm"], r["seed"]): r for r in rows}
    for sc, ws in worlds.items():
        seeds = [w["seed"] for w in ws]
        rec: dict = {"region_dirichlet_a": SCENARIOS[sc],
                     "gini": xc.mean_ci([w["gini"] for w in ws]),
                     "largest_share": xc.mean_ci([max(w["sizes"]) / sum(w["sizes"]) for w in ws]),
                     "smallest_share": xc.mean_ci([min(w["sizes"]) / sum(w["sizes"]) for w in ws]),
                     "p0_by_role": {role: xc.mean_ci([w["p0"][w["roles"][role]] for w in ws]) for role in ROLES},
                     "size_share_by_role": {role: xc.mean_ci([w["sizes"][w["roles"][role]] / sum(w["sizes"]) for w in ws])
                                            for role in ROLES},
                     "largest_region_role_counts": {role: sum(1 for w in ws if w["roles"][role] ==
                                                              max(range(4), key=lambda r: (w["sizes"][r], -r)))
                                                    for role in ROLES},
                     "arms": {}}
        base = [idx[(sc, "none", s)] for s in seeds]
        arms = ["none"] + [f"K{K}|{t}" for K in KS for t in TARGETS]
        for arm in arms:
            rs = [idx[(sc, arm, s)] for s in seeds]
            a = {"expected": xc.mean_ci([r["expected"] for r in rs]), "win": xc.rate([r["victory"] for r in rs]),
                 "converts": xc.mean_ci([r["converts"] for r in rs]),
                 "converts_by_role": {role: xc.mean_ci([by_role(r["converts_by_region"], w["roles"])[role]
                                                        for r, w in zip(rs, ws)]) for role in ROLES},
                 "final_p_by_role": {role: xc.mean_ci([by_role(r["final_p_by_region"], w["roles"])[role]
                                                       for r, w in zip(rs, ws)]) for role in ROLES}}
            if arm != "none":
                a["delta_vs_none"] = xc.mean_ci([r["expected"] - b["expected"] for r, b in zip(rs, base)])
                a["win_gain_vs_none"] = xc.mean_ci([int(r["victory"]) - int(b["victory"]) for r, b in zip(rs, base)])
                a["d_support_by_role"] = {role: xc.mean_ci([
                    by_role(r["final_p_by_region"], w["roles"])[role] - by_role(b["final_p_by_region"], w["roles"])[role]
                    for r, b, w in zip(rs, base, ws)]) for role in ROLES}
                # contribution of each role's region to ΔE: (n_r / N)·ΔP_r (sums to ΔE)
                a["d_e_contribution_by_role"] = {role: xc.mean_ci([
                    w["sizes"][w["roles"][role]] / sum(w["sizes"]) *
                    (r["final_p_by_region"][w["roles"][role]] - b["final_p_by_region"][w["roles"][role]])
                    for r, b, w in zip(rs, base, ws)]) for role in ROLES}
                a["converts_outside_target_share"] = _outside_share(rs, ws)
            rec["arms"][arm] = a
        for K in KS:
            ds = {t: rec["arms"][f"K{K}|{t}"]["delta_vs_none"]["mean"] for t in TARGETS}
            rec[f"best_K{K}"] = max(ds, key=ds.get)
            rec[f"worst_K{K}"] = min(ds, key=ds.get)
            rec[f"range_K{K}"] = max(ds.values()) - min(ds.values())
            fam = f"{sc}|K{K}"
            tests = []

            def test(a_arm, b_arm):
                ra = [idx[(sc, a_arm, s)] for s in seeds]
                rb = [idx[(sc, b_arm, s)] for s in seeds]
                tests.append({"a": a_arm, "b": b_arm, **xc.paired([x["expected"] for x in ra], [x["expected"] for x in rb]),
                              "mcnemar": xc.mcnemar([x["victory"] for x in ra], [x["victory"] for x in rb])})
            for t in TARGETS:
                test(f"K{K}|{t}", "none")
            for i, t1 in enumerate(TARGETS):
                for t2 in TARGETS[i + 1:]:
                    test(f"K{K}|{t1}", f"K{K}|{t2}")
            for t, p in zip(tests, xc.holm([t["wilcoxon_p"] for t in tests])):
                t["holm_p"] = p
            for t, p in zip(tests, xc.holm([t["mcnemar"]["p"] for t in tests])):
                t["mcnemar_holm_p"] = p
            out["tests"][fam] = tests
        out["scenarios"][sc] = rec
    out["posthoc_largest_region"] = _posthoc(idx, worlds["default"])
    return out


def _outside_share(rs: list[dict], ws: list[dict]) -> dict:
    """Share of converts outside the regions that received workers (pooled over worlds, and mean of per-world)."""
    inside = outside = 0
    per = []
    for r, w in zip(rs, ws):
        tgt = [i for i in range(4) if r["hires_by_region"][i] > 0]
        cin = sum(r["converts_by_region"][i] for i in tgt)
        cout = sum(r["converts_by_region"]) - cin
        inside, outside = inside + cin, outside + cout
        if cin + cout > 0:
            per.append(cout / (cin + cout))
    return {"pooled": outside / max(1, inside + outside), "per_world": xc.mean_ci(per)}


def _posthoc(idx: dict, ws: list[dict]) -> dict:
    """Default scenario: worlds whose largest region is the aligned region vs the antagonistic region."""
    out = {}
    for grp in ("aligned", "antagonistic", "swing", "other"):
        sub = [w for w in ws if w["roles"][grp] == max(range(4), key=lambda r: (w["sizes"][r], -r))]
        g = {"n_worlds": len(sub), "seeds": [w["seed"] for w in sub]}
        if sub:
            base = [idx[("default", "none", w["seed"])] for w in sub]
            g["none_expected"] = xc.mean_ci([b["expected"] for b in base])
            g["none_win"] = xc.rate([b["victory"] for b in base])
            g["arms"] = {}
            for K in KS:
                for t in TARGETS:
                    rs = [idx[("default", f"K{K}|{t}", w["seed"])] for w in sub]
                    g["arms"][f"K{K}|{t}"] = {"delta_vs_none": xc.mean_ci([r["expected"] - b["expected"]
                                                                            for r, b in zip(rs, base)]),
                                              "win": xc.rate([r["victory"] for r in rs])}
        out[f"largest_is_{grp}"] = g
    return out


# ============================================================================== figures (from JSON only)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
ROLE_COLORS = {"aligned": "#2a78d6", "swing": "#eda100", "antagonistic": "#e34948", "other": "#8a8984"}
LABEL = {"aligned": "all→aligned", "swing": "all→swing", "antagonistic": "all→antagonistic",
         "aligned_swing": "aligned+swing", "even": "even (4 regions)", "proportional": "proportional"}


def plot(report: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "axes.axisbelow": True})
    S = report["scenarios"]
    scs = list(SCENARIOS)
    paths = []

    def err(ci_rec):
        m = ci_rec["mean"]
        return m, [[m - ci_rec["ci95"][0]], [ci_rec["ci95"][1] - m]]

    # (a) ΔE by arm, per scenario, K = 4 and 10
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    width = 0.13
    for ax, K in zip(axes, KS):
        x = np.arange(len(scs))
        for i, t in enumerate(TARGETS):
            for j, sc in enumerate(scs):
                m, e = err(S[sc]["arms"][f"K{K}|{t}"]["delta_vs_none"])
                ax.bar(x[j] + (i - 2.5) * width, m, width * 0.9, color=COLORS[i], yerr=e, capsize=2,
                       error_kw={"elinewidth": 0.8, "ecolor": "#52514e"}, label=LABEL[t] if j == 0 else None)
        ax.set_xticks(x, [f"{sc}\n(a = {SCENARIOS[sc]:g})" for sc in scs])
        ax.set_title(f"K = {K} workers (₫{20 * K}), all bought in round 1")
        ax.axhline(0, color="#52514e", lw=0.8)
    axes[0].set_ylabel("ΔE vs no campaign (95% CI)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    p = FIG / "r3_delta_by_arm.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # (b) per-region support change by arm (where the gain lands)
    fig, axes = plt.subplots(len(scs), 2, figsize=(11, 8), sharey=True)
    for r_i, sc in enumerate(scs):
        for c_i, K in enumerate(KS):
            ax = axes[r_i][c_i]
            x = np.arange(len(TARGETS))
            for k, role in enumerate(ROLES):
                ms = [S[sc]["arms"][f"K{K}|{t}"]["d_support_by_role"][role]["mean"] for t in TARGETS]
                lo = [m - S[sc]["arms"][f"K{K}|{t}"]["d_support_by_role"][role]["ci95"][0] for m, t in zip(ms, TARGETS)]
                hi = [S[sc]["arms"][f"K{K}|{t}"]["d_support_by_role"][role]["ci95"][1] - m for m, t in zip(ms, TARGETS)]
                ax.bar(x + (k - 1.5) * 0.2, ms, 0.18, color=ROLE_COLORS[role], yerr=[lo, hi], capsize=1.5,
                       error_kw={"elinewidth": 0.7, "ecolor": "#52514e"},
                       label=f"{role} region" if (r_i == 0 and c_i == 1) else None)
            ax.set_xticks(x, [LABEL[t].replace(" (4 regions)", "") for t in TARGETS], rotation=20, fontsize=7.5)
            ax.set_title(f"{sc} (a = {SCENARIOS[sc]:g}), K = {K}", fontsize=9)
            ax.axhline(0, color="#52514e", lw=0.8)
            if c_i == 0:
                ax.set_ylabel("Δ regional mean P\nvs no campaign")
    axes[0][1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    p = FIG / "r3_region_support.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # (c) heat map target × (scenario, K)
    cols = [(sc, K) for K in KS for sc in scs]
    mat = np.array([[S[sc]["arms"][f"K{K}|{t}"]["delta_vs_none"]["mean"] for sc, K in cols] for t in TARGETS])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=float(mat.max()))
    ax.set_xticks(range(len(cols)), [f"{sc}\nK = {K}" for sc, K in cols], fontsize=8)
    ax.set_yticks(range(len(TARGETS)), [LABEL[t] for t in TARGETS])
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.6 * mat.max() else "#0b0b0b")
    fig.colorbar(im, ax=ax, label="mean ΔE vs no campaign")
    ax.set_title("ΔE by worker target (rows) × population scenario and K (columns)", fontsize=9)
    fig.tight_layout()
    p = FIG / "r3_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    # (d) post hoc: largest region aligned vs antagonistic (default scenario)
    ph = report["posthoc_largest_region"]
    groups = [g for g in ("largest_is_aligned", "largest_is_antagonistic") if ph[g]["n_worlds"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, K in zip(axes, KS):
        x = np.arange(len(TARGETS))
        for gi, g in enumerate(groups):
            ms, lo, hi = [], [], []
            for t in TARGETS:
                c = ph[g]["arms"][f"K{K}|{t}"]["delta_vs_none"]
                ms.append(c["mean"])
                ok = c["n"] > 1
                lo.append(c["mean"] - c["ci95"][0] if ok else 0)
                hi.append(c["ci95"][1] - c["mean"] if ok else 0)
            ax.bar(x + (gi - 0.5) * 0.38, ms, 0.36, color=["#2a78d6", "#e34948"][gi], yerr=[lo, hi], capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": "#52514e"},
                   label=f"largest region is {g.split('_')[-1]} ({ph[g]['n_worlds']} worlds)")
        ax.set_xticks(x, [LABEL[t].replace(" (4 regions)", "") for t in TARGETS], rotation=20, fontsize=8)
        ax.set_title(f"default scenario, K = {K}")
        ax.axhline(0, color="#52514e", lw=0.8)
    axes[0].set_ylabel("ΔE vs no campaign (95% CI)")
    axes[0].set_ylim(0, 0.19)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout()
    p = FIG / "r3_largest_split.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))
    return paths


# ============================================================================== parity
def parity(seeds: list[int], device: torch.device, base: GameConfig) -> dict:
    def fn(seed, dev):
        cfg = dataclasses.replace(base, device=str(dev))
        w = make_world(seed, SCENARIOS["default"], cfg, dev)
        rows = play([j for j in arms_for(w) if j["arm"] in ("none", "K4|swing", "K10|even")], cfg)
        r = {x["arm"]: x for x in rows}
        return {"expected_none": r["none"]["expected"], "expected_K4_swing": r["K4|swing"]["expected"],
                "expected_K10_even": r["K10|even"]["expected"],
                "converts_share_K10_even": r["K10|even"]["converts"] / cfg.electorate.num_voters}
    out = parity_rel(fn, seeds, device=device)
    out["cell"] = "default scenario: none, K = 4 all→swing, K = 10 even"
    return out


# ============================================================================== main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--first-seed", type=int, default=11000)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=650, help="games per run_batch call")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        for p in plot(json.loads(JSON_PATH.read_text())):
            print("figure", p)
        return
    torch.set_num_threads(4)
    for d in (OUT / "raw", OUT / "logs", FIG):
        d.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("")
    log = Log(LOG_PATH)
    if "cuda:1" == args.device:
        raise SystemExit("GPU 1 is reserved for another group")
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    if args.device.startswith("cuda") and str(device) != str(torch.device(args.device)):
        raise SystemExit(f"refusing to fall back from {args.device} to {device}")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    free0 = xc.free_bytes(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=args.n))
    RAW_PATH.write_text("")
    log(f"[r3] device {device}; seeds {seeds[0]}..{seeds[-1]}; N {args.n}; batch {args.batch}")
    start = time.perf_counter()
    rows: list[dict] = []
    worlds: dict[str, list[dict]] = {}
    batch_rec: dict = {}
    timings = {}
    with xc.GpuMonitor(device) as monitor:
        for sc, a in SCENARIOS.items():
            t0 = time.perf_counter()
            ws = [make_world(s, a, base, device) for s in seeds]
            worlds[sc] = ws
            t1 = time.perf_counter()
            jobs = [j for w in ws for j in arms_for(w)]
            rec: dict = {}
            res = xc.run_chunks(lambda chunk: play(chunk, base), jobs, args.batch, log=log, record=rec)
            batch_rec[sc] = rec
            for j, r in zip(jobs, res):
                w = j["world"]
                r.update(scenario=sc, region_dirichlet_a=a, sizes=w["sizes"], gini=w["gini"], roles=w["roles"],
                         p0_by_region=w["p0"])
                rows.append(r)
                xc.write_row(RAW_PATH, {"exp": "R3", **r})
            timings[sc] = {"worlds_s": t1 - t0, "games_s": time.perf_counter() - t1, "games": len(jobs)}
            mean = lambda arm: sum(r["expected"] for r in res if r["arm"] == arm) / len(ws)  # noqa: E731
            log(f"[r3] {sc}: {len(jobs)} games in {timings[sc]['games_s']:.1f}s (worlds {t1 - t0:.1f}s); "
                f"E none {mean('none'):.4f}; " + "; ".join(f"{t} {mean('K10|' + t):.4f}" for t in TARGETS) + " (K10)")
        summary = summarise(rows, worlds)
        par = parity(seeds[:args.parity_seeds], device, base)
        log(f"[r3] parity passed: {par['passed']}")
    seconds = time.perf_counter() - start
    report = {
        "experiment": "R3", "question": "How to assign a fixed number of party workers across regions in different "
                                        "population / preference scenarios (aligned, swing, antagonistic regions)",
        "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds, "paired": True},
        "n": args.n, "rounds": base.rounds, "treasury": str(base.budget), "worker_cost": str(base.workers.worker_cost),
        "ks": list(KS), "targets": list(TARGETS), "scenarios_a": SCENARIOS,
        "definitions": {
            "roles": "from round-0 TRUE regional mean P under the world's ORACLE manifesto: aligned = max, "
                     "antagonistic = min, swing = of the remaining two the one with |P − 0.5| smallest, other = last; "
                     "ties by lower region index",
            "splits": "largest remainder; equal remainders go to the larger region (aligned before swing for "
                      "aligned_swing)",
            "delta_vs_none": "E(arm) − E(none) in the same world and streams",
            "d_support_by_role": "final regional mean P (arm) − final regional mean P (none), region by role",
            "d_e_contribution_by_role": "(n_r/N)·ΔP_r; the four roles sum to delta_vs_none",
            "converts_by_role": "voters converted by the IC cascade over the game, by region (observer hook)",
            "converts_outside_target_share": "share of converts outside the regions that received workers",
            "posthoc_largest_region": "default scenario split by the role of the largest region"},
        "world_roles": {sc: [{"seed": w["seed"], "sizes": w["sizes"], "gini": w["gini"], "roles": w["roles"],
                              "p0_by_region": w["p0"]} for w in ws] for sc, ws in worlds.items()},
        **summary,
        "test_families": "per scenario × K: 6 arms vs none + 15 pairwise between arms (21), Wilcoxon on E and exact "
                         "McNemar on wins, Holm within the family",
        "deviations": [
            "Parity uses x_parity.parity_rel (x_common.parity_check with tolerance max(0.01, 3·SE, 5% of the mean), "
            "as in X4), because CPU and GPU draw different random numbers.",
            "Per-region converts come from run_batch's observer hook (the per-round converted mask); no library change.",
            "With K = 4 and near-equal regions, 'even' and 'proportional' give the same 1/1/1/1 split in most worlds "
            "(identical games)."],
        "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, batch_sizes=batch_rec, free_at_start=free0,
                                       monitor=monitor, device_record=dev_record, parity=par,
                                       extra={"timings": timings}),
    }
    xc.save(str(JSON_PATH), report)
    for p in plot(report):
        log(f"[r3] figure {p}")
    log(f"[r3] done in {seconds / 60:.1f} min; peak {report['provenance']['compute']['peak_bytes'] / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
