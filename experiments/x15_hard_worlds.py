"""X15 — Which elections are unwinnable? (plan ``docs/v2/plans/implementation_plan.md`` §7 X15).

Command (reproduces reports/v3/x15.json, reports/v3/raw/x15.jsonl and reports/v3/logs/x15.log; CPU, minutes):
  OMP_NUM_THREADS=4 python experiments/x15_hard_worlds.py --out reports/v3/x15.json 2>&1 | tee reports/v3/logs/x15.log

Post-hoc, no new games. It joins, per world (seed):
  * X0  (reports/v3/raw/x0.jsonl, seeds 10000–10099, ₫500/750/1000): ORACLE = oracle manifesto + ORACLE planner;
        the best paid channel (informed manifesto + pay_workers / pay_media / pay_split) at ₫500;
  * X10 default cell (raw/x10.jsonl, N = 10⁴, T = 5, ₫500, seeds 10000–10049): the same canonical worlds and the same
        runs as X0 ₫500 — checked to be identical row by row, so it adds no new world (counted once);
  * X6  (raw/x6.jsonl, seeds 12000–12099, ₫500): ORACLE, BayesPlanner (II), People Pleaser (I), top-3 games;
with ``x_common.features_of(seed)`` (canonical CPU electorate, default config) plus the realised block
probabilities p_g (inter-region) and p_r (intra-region) of the SBM, which ``features_of`` does not return.

Targets (one row per world): ORACLE loss at ₫500 (200 worlds), Bayes loss (100 X6 worlds), and — because both are
rare — the margins E − 0.5 (continuous) and the People Pleaser loss (supplementary binary, X6).
Models (numpy; scikit-learn is not installed in the env and nothing is installed):
  (a) logistic regression on standardised features, a weak ridge (λ = 1 on the coefficients, not the intercept)
      against separation, coefficient 95% CIs from 2,000 bootstrap resamples (stratified by outcome);
  (b) a depth-3 CART (Gini; min leaf 5) — a regression tree (MSE) for the margins.
Evaluated by 5-fold stratified CV AUC (repeated 20× with different splits) plus a label-permutation null (200×);
R² for the margins. Partial dependence (logistic and tree) for the top 3 features. A rule-based "why you lost"
taxonomy with thresholds at the terciles of the 200 worlds. Pre-registered checks (Holm over the family):
H1 the two largest regions' archetype TV distance is higher in ORACLE-loss worlds; H2 p_g is lower in them;
H3 the largest region's personality is associated with the best paid channel (X0, ₫500).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402

from dumbocracy.config import GameConfig  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.sectors import PERSONALITIES  # noqa: E402

RAW_PATH = xc.RAW / "x15.jsonl"
PERS = [p.id for p in PERSONALITIES]
PERS_NAME = {p.id: p.name for p in PERSONALITIES}
# numeric features (standardised); the largest region's personality enters as 3 dummies (P0 = reference)
NUMERIC = ("tv_two_largest", "tv_mean_pair", "p_g_mean", "p_g_two_largest", "inter_edge_share", "lambda2_L",
           "rw_gap", "size_gini", "largest_share", "two_largest_share", "zealot_share", "p_intra_mean")
DUMMIES = ("largest_P1", "largest_P2", "largest_P3")
STRUCT = NUMERIC + DUMMIES
WITH_START = STRUCT + ("start_share",)
LAMBDA = 1.0
B_BOOT = 2000
CV_REPEATS = 20
N_PERM = 200


# ============================================================================== data
def world_features(seed: int) -> dict:
    cfg = GameConfig(seed=seed)
    e = generate_electorate(cfg, device="cpu")
    f = xc.features_of(e, seed=seed)
    sizes = f["sizes"]
    order = sorted(range(4), key=lambda r: -sizes[r])
    p_inter = e.p_inter.to(torch.float64)
    off = [float(p_inter[i, j]) for i in range(4) for j in range(4) if i != j]
    n = sum(sizes)
    return {
        "seed": seed, "sizes": sizes, "tau_ids": f["tau_ids"], "sigma": f["sigma"],
        "tv_two_largest": f["tv_two_largest"], "tv_mean_pair": f["tv_mean_pair"], "tv_min_pair": f["tv_min_pair"],
        "p_g_mean": sum(off) / len(off), "p_g_two_largest": float(p_inter[order[0], order[1]]),
        "p_intra_mean": float(e.p_intra.to(torch.float64).mean()),
        "inter_edge_share": f["inter_edge_share"], "lambda2_L": f["lambda2_L"], "rw_gap": f["rw_gap"],
        "size_gini": f["size_gini"], "largest_share": sizes[order[0]] / n,
        "two_largest_share": (sizes[order[0]] + sizes[order[1]]) / n,
        "zealot_share": sum(f["zealots_by_region"]) / n, "largest_personality": f["largest_region_personality"],
        **{f"largest_{p}": float(f["largest_region_personality"] == p) for p in ("P1", "P2", "P3")},
        "spectra_method": f["spectra_method"],
    }


def load_outcomes() -> tuple[dict, dict]:
    """Per-seed outcomes and the X10 duplication check."""
    worlds: dict[int, dict] = defaultdict(dict)
    x0 = xc.read_rows(xc.RAW / "x0.jsonl")
    for r in x0:
        s, t = r["seed"], r["treasury"]
        key = f"{r['manifesto']}/{r['policy']}"
        w = worlds[s]
        w["source"] = "X0"
        if key == "oracle/planner":
            w[f"oracle_E_{t}"] = r["expected"]
            w[f"oracle_win_{t}"] = bool(r["victory"])
            if t == "500":
                w["start_share"] = r["trajectory"][0]
        if t == "500" and key.startswith("informed/pay_"):
            w[f"channel_{r['policy']}"] = r["expected"]
    x10 = [r for r in xc.read_rows(xc.RAW / "x10.jsonl") if r["n"] == 10_000 and r["rounds"] == 5]
    x0_500 = {(r["seed"], r["manifesto"], r["policy"]): r for r in x0 if r["treasury"] == "500"}
    pairs = [(x0_500[k], r) for r in x10 if (k := (r["seed"], r["manifesto"], r["policy"])) in x0_500]
    same = sum(1 for a, r in pairs if abs(a["expected"] - r["expected"]) <= 1e-5 and a["victory"] == r["victory"])
    dup = {"x10_default_rows": len(x10), "identical_to_x0_500": same,
           "bit_identical": sum(1 for a, r in pairs if a["expected"] == r["expected"]),
           "max_abs_expected_diff": max(abs(a["expected"] - r["expected"]) for a, r in pairs),
           "victory_disagreements": sum(1 for a, r in pairs if a["victory"] != r["victory"]),
           "match_rule": "same seed/manifesto/policy, |ΔE| <= 1e-5 (GPU float atomics) and the same victory",
           "x10_seeds": sorted({r["seed"] for r in x10})[:1] + sorted({r["seed"] for r in x10})[-1:]}
    for r in xc.read_rows(xc.RAW / "x6.jsonl"):
        w = worlds[r["seed"]]
        w["source"] = "X6"
        p = r["player"]
        if p == "ORACLE":
            w["oracle_E_500"], w["oracle_win_500"] = r["expected"], bool(r["victory"])
            w["start_share"] = r["trace"]["expected_by_round"][0]
            w["oracle_media_share"] = None
        elif p in ("bayes", "people_pleaser", "top3", "bayes_admm"):
            w[f"{p}_E"], w[f"{p}_win"] = r["expected"], bool(r["victory"])
    return worlds, dup


# ============================================================================== models (numpy)
def standardise(X: np.ndarray, mu=None, sd=None):
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) if sd is None else sd
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd, mu, sd


def logit_fit(X: np.ndarray, y: np.ndarray, lam: float = LAMBDA, iters: int = 100) -> np.ndarray:
    """Ridge-penalised logistic regression by Newton/IRLS; X standardised, intercept unpenalised."""
    n, p = X.shape
    Z = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(p + 1)
    pen = np.full(p + 1, lam)
    pen[0] = 0.0
    for _ in range(iters):
        eta = np.clip(Z @ w, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        g = Z.T @ (mu - y) + pen * w
        H = (Z * (mu * (1 - mu))[:, None]).T @ Z + np.diag(pen) + 1e-9 * np.eye(p + 1)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def logit_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(w[0] + X @ w[1:], -30, 30)))


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    n, p = X.shape
    Z = np.hstack([np.ones((n, 1)), X])
    pen = np.full(p + 1, lam)
    pen[0] = 0.0
    return np.linalg.solve(Z.T @ Z + np.diag(pen), Z.T @ y)


def ridge_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return w[0] + X @ w[1:]


def tree_fit(X: np.ndarray, y: np.ndarray, depth: int = 3, min_leaf: int = 5, kind: str = "class") -> dict:
    """CART: Gini (class, y ∈ {0,1}) or MSE (reg). Leaf value = mean y (= P(y=1) for class)."""
    def impurity(v):
        if len(v) == 0:
            return 0.0
        if kind == "class":
            p = v.mean()
            return 2 * p * (1 - p)
        return v.var()

    def build(idx, d):
        node = {"n": int(len(idx)), "value": float(y[idx].mean())}
        if d == depth or len(idx) < 2 * min_leaf or impurity(y[idx]) == 0:
            return node
        best = None
        base = impurity(y[idx]) * len(idx)
        for j in range(X.shape[1]):
            xs = X[idx, j]
            order = np.argsort(xs, kind="stable")
            xs_s, ys_s = xs[order], y[idx][order]
            m = len(idx)
            i = np.arange(min_leaf, m - min_leaf + 1)          # left = first i sorted rows
            if len(i) == 0:
                continue
            c1, c2 = np.cumsum(ys_s), np.cumsum(ys_s ** 2)
            sl, sl2 = c1[i - 1], c2[i - 1]
            sr, sr2 = c1[-1] - sl, c2[-1] - sl2
            nl, nr = i.astype(float), (m - i).astype(float)
            if kind == "class":     # n·Gini = 2·s·(1 − s/n)
                score = 2 * sl * (1 - sl / nl) + 2 * sr * (1 - sr / nr)
            else:                   # n·var = Σy² − (Σy)²/n
                score = (sl2 - sl ** 2 / nl) + (sr2 - sr ** 2 / nr)
            valid = xs_s[i - 1] != xs_s[np.minimum(i, m - 1)]
            if not valid.any():
                continue
            score = np.where(valid, score, np.inf)
            a = int(np.argmin(score))
            if best is None or score[a] < best[0] - 1e-12:
                best = (float(score[a]), j, 0.5 * (xs_s[i[a] - 1] + xs_s[i[a]]))
        if best is None or best[0] >= base - 1e-12:
            return node
        _, j, thr = best
        node.update(feature=int(j), threshold=float(thr), gain=float(base - best[0]),
                    left=build(idx[X[idx, j] <= thr], d + 1), right=build(idx[X[idx, j] > thr], d + 1))
        return node

    return build(np.arange(len(y)), 0)


def tree_predict(node: dict, X: np.ndarray) -> np.ndarray:
    out = np.empty(len(X))
    for i, x in enumerate(X):
        n = node
        while "feature" in n:
            n = n["left"] if x[n["feature"]] <= n["threshold"] else n["right"]
        out[i] = n["value"]
    return out


def tree_importance(node: dict, p: int) -> np.ndarray:
    imp = np.zeros(p)
    stack = [node]
    while stack:
        n = stack.pop()
        if "feature" in n:
            imp[n["feature"]] += n["gain"]
            stack += [n["left"], n["right"]]
    return imp / imp.sum() if imp.sum() > 0 else imp


def tree_named(node: dict, names: list[str], mu: np.ndarray, sd: np.ndarray) -> dict:
    """Tree with feature names and thresholds in raw units (fitted on standardised X)."""
    out = {"n": node["n"], "value": node["value"]}
    if "feature" in node:
        j = node["feature"]
        out.update(feature=names[j], threshold=float(node["threshold"] * sd[j] + mu[j]),
                   left=tree_named(node["left"], names, mu, sd), right=tree_named(node["right"], names, mu, sd))
    return out


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def folds(y: np.ndarray, k: int, rng: np.random.Generator, stratify: bool = True) -> list[np.ndarray]:
    idx = np.arange(len(y))
    if stratify:
        out = [[] for _ in range(k)]
        for cls in np.unique(y):
            c = rng.permutation(idx[y == cls])
            for i, j in enumerate(c):
                out[i % k].append(j)
        return [np.array(sorted(f)) for f in out]
    p = rng.permutation(idx)
    return [p[i::k] for i in range(k)]


def cv_scores(X: np.ndarray, y: np.ndarray, kind: str, rng: np.random.Generator, repeats: int = CV_REPEATS) -> dict:
    """Repeated 5-fold CV (standardisation inside the fold). class → AUC; reg → R²."""
    res = {"logit": [], "tree": []}
    for _ in range(repeats):
        oof = {"logit": np.zeros(len(y)), "tree": np.zeros(len(y))}
        for te in folds(y if kind == "class" else np.zeros(len(y)), 5, rng):
            tr = np.setdiff1d(np.arange(len(y)), te)
            Xtr, mu, sd = standardise(X[tr])
            Xte, _, _ = standardise(X[te], mu, sd)
            if kind == "class":
                oof["logit"][te] = logit_predict(logit_fit(Xtr, y[tr]), Xte)
                oof["tree"][te] = tree_predict(tree_fit(Xtr, y[tr], kind="class"), Xte)
            else:
                oof["logit"][te] = ridge_predict(ridge_fit(Xtr, y[tr]), Xte)
                oof["tree"][te] = tree_predict(tree_fit(Xtr, y[tr], kind="reg"), Xte)
        for m in res:
            if kind == "class":
                res[m].append(auc(y, oof[m]))
            else:
                res[m].append(1 - ((y - oof[m]) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    out = {}
    for m, v in res.items():
        v = np.array(v)
        out[m] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "sd": float(v.std()),
                  "repeats": repeats}
    return out


def perm_null(X: np.ndarray, y: np.ndarray, kind: str, observed: dict, rng: np.random.Generator) -> dict:
    """Label-permutation null of the (2×-repeated) CV score; p = (1 + #null ≥ obs) / (1 + N_PERM)."""
    nulls = {"logit": [], "tree": []}
    for _ in range(N_PERM):
        sc = cv_scores(X, rng.permutation(y), kind, rng, repeats=2)
        for m in nulls:
            nulls[m].append(sc[m]["mean"])
    return {m: {"null_mean": float(np.mean(v)), "null_q95": float(np.quantile(v, 0.95)),
                "p": float((1 + sum(x >= observed[m]["mean"] for x in v)) / (1 + N_PERM))} for m, v in nulls.items()}


def bootstrap_coefs(X: np.ndarray, y: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    ws = []
    idx = np.arange(len(y))
    for _ in range(B_BOOT):
        if kind == "class":
            b = np.concatenate([rng.choice(idx[y == c], size=(y == c).sum(), replace=True) for c in (0, 1)])
            Xb, _, _ = standardise(X[b])
            ws.append(logit_fit(Xb, y[b]))
        else:
            b = rng.choice(idx, size=len(idx), replace=True)
            Xb, _, _ = standardise(X[b])
            ws.append(ridge_fit(Xb, y[b]))
    return np.array(ws)


def pdp(predict, X: np.ndarray, j: int, grid: np.ndarray) -> list[float]:
    out = []
    for g in grid:
        Xg = X.copy()
        Xg[:, j] = g
        out.append(float(predict(Xg).mean()))
    return out


def analyse(name: str, rows: list[dict], feats: tuple, target: str, kind: str, rng: np.random.Generator,
            log=print) -> dict:
    X = np.array([[r[f] for f in feats] for r in rows], dtype=float)
    y = np.array([float(r[target]) for r in rows])
    out: dict = {"target": target, "kind": kind, "n": int(len(y)), "features": list(feats)}
    if kind == "class":
        out["events"] = int(y.sum())
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            out["fitted"] = False
            out["reason"] = f"only {int(y.sum())} events: too few for a regression, a tree or a 5-fold CV AUC"
            return out
    Xs, mu, sd = standardise(X)
    w = logit_fit(Xs, y) if kind == "class" else ridge_fit(Xs, y)
    boot = bootstrap_coefs(X, y, kind, rng)
    coefs = {}
    for j, f in enumerate(feats):
        lo, hi = np.quantile(boot[:, j + 1], [0.025, 0.975])
        coefs[f] = {"coef": float(w[j + 1]), "ci95": [float(lo), float(hi)],
                    "sign_agreement": float(np.mean(np.sign(boot[:, j + 1]) == np.sign(w[j + 1]))),
                    "ci_excludes_0": bool(lo > 0 or hi < 0)}
    tree = tree_fit(Xs, y, kind="class" if kind == "class" else "reg")
    imp = tree_importance(tree, len(feats))
    cv = cv_scores(X, y, kind, rng)
    out.update(fitted=True, intercept=float(w[0]), coefs=coefs, model="ridge-logistic λ=1" if kind == "class" else
               "ridge-linear λ=1", standardisation={"mean": dict(zip(feats, mu.tolist())), "sd": dict(zip(feats, sd.tolist()))},
               tree=tree_named(tree, list(feats), mu, sd),
               tree_importance={f: float(v) for f, v in zip(feats, imp)},
               in_sample={"auc_logit": auc(y, logit_predict(w, Xs)), "auc_tree": auc(y, tree_predict(tree, Xs))}
               if kind == "class" else {"r2_logit": float(1 - ((y - ridge_predict(w, Xs)) ** 2).sum() / ((y - y.mean()) ** 2).sum())},
               cv=cv)
    if kind == "class":
        out["cv_permutation_null"] = perm_null(X, y, kind, cv, rng)
    # top 3 features: by |standardised coefficient| (ties to the tree importance), then PDPs on raw grids
    ranked = sorted(feats, key=lambda f: (-abs(coefs[f]["coef"]), -out["tree_importance"][f]))
    out["top3"] = ranked[:3]
    out["top3_tree"] = sorted(feats, key=lambda f: -out["tree_importance"][f])[:3]
    pd = {}
    pred_l = (lambda Z: logit_predict(w, Z)) if kind == "class" else (lambda Z: ridge_predict(w, Z))
    pred_t = lambda Z: tree_predict(tree, Z)  # noqa: E731
    for f in dict.fromkeys(out["top3"] + out["top3_tree"]):
        j = feats.index(f)
        raw = np.unique(X[:, j])
        grid_raw = raw if len(raw) <= 2 else np.quantile(X[:, j], np.linspace(0.02, 0.98, 25))
        grid = (grid_raw - mu[j]) / sd[j]
        pd[f] = {"grid": grid_raw.tolist(), "logit": pdp(pred_l, Xs, j, grid), "tree": pdp(pred_t, Xs, j, grid)}
    out["partial_dependence"] = pd
    log(f"[x15] {name}: n={len(y)}" + (f" events={int(y.sum())}" if kind == "class" else "") +
        f" cv logit {cv['logit']['mean']:.3f} tree {cv['tree']['mean']:.3f}; top3 {out['top3']}")
    return out


# ============================================================================== hypotheses, channel, taxonomy
def mannwhitney(a: list[float], b: list[float], alternative: str) -> dict:
    from scipy.stats import mannwhitneyu
    r = mannwhitneyu(a, b, alternative=alternative)
    return {"U": float(r.statistic), "p": float(r.pvalue), "n_a": len(a), "n_b": len(b),
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "auc_a_gt_b": float(r.statistic / (len(a) * len(b)))}


def channel_analysis(worlds: list[dict]) -> dict:
    from scipy.stats import chi2_contingency, kruskal
    rows = [w for w in worlds if "channel_pay_workers" in w]
    best = {}
    gap = defaultdict(list)
    table = defaultdict(Counter)
    for w in rows:
        ch = {c: w[f"channel_pay_{c}"] for c in ("workers", "media", "split")}
        b = max(ch, key=ch.get)
        best[w["seed"]] = b
        pers = w["largest_personality"]
        table[pers][b] += 1
        gap[pers].append(ch["media"] - ch["workers"])
    chans = ("workers", "media", "split")
    mat = np.array([[table[p][c] for c in chans] for p in PERS if table[p]])
    keep = mat.sum(0) > 0
    chi = chi2_contingency(mat[:, keep])
    n = mat.sum()
    v = math.sqrt(chi.statistic / (n * (min(mat[:, keep].shape) - 1)))
    kw = kruskal(*[gap[p] for p in PERS if gap[p]])
    return {"n_worlds": len(rows), "source": "X0 ₫500, informed manifesto + pay_workers / pay_media / pay_split",
            "best_channel_by_largest_personality": {p: dict(table[p]) for p in PERS},
            "media_minus_workers_E_by_personality": {p: xc.mean_ci(gap[p]) for p in PERS},
            "chi2": {"statistic": float(chi.statistic), "dof": int(chi.dof), "p": float(chi.pvalue), "cramers_v": v},
            "kruskal_media_minus_workers": {"statistic": float(kw.statistic), "p": float(kw.pvalue)},
            "overall_best": dict(Counter(best.values()))}


def taxonomy(worlds: list[dict], losses: dict[str, list[int]]) -> dict:
    """Rule-based 'why you lost' classes; thresholds = terciles over all worlds; first matching class wins."""
    q = lambda f, p: float(np.quantile([w[f] for w in worlds], p))  # noqa: E731
    th = {"start_share_low": q("start_share", 1 / 3), "tv_two_largest_high": q("tv_two_largest", 2 / 3),
          "p_g_low": q("p_g_mean", 1 / 3), "close_call": 0.01}
    rules = [
        ("close_call", "lost within 0.01 of 50% in E: the election draw decided", None),
        ("divided_majority", "the two largest regions want different things (archetype TV distance in the top "
                             "tercile) and the manifesto starts low (start share in the bottom tercile)", None),
        ("low_start", "the best manifesto starts low (start share in the bottom tercile)", None),
        ("divided_regions", "two largest regions far apart (TV in the top tercile)", None),
        ("fragmented_network", "few links between regions (p_g in the bottom tercile)", None),
        ("independent_largest", "the largest region is P2 Independent (low susceptibility to every channel)", None),
        ("other", "none of the above", None),
    ]

    def classify(w: dict, E: float) -> str:
        low = w["start_share"] <= th["start_share_low"]
        far = w["tv_two_largest"] >= th["tv_two_largest_high"]
        if abs(E - 0.5) <= th["close_call"]:
            return "close_call"
        if far and low:
            return "divided_majority"
        if low:
            return "low_start"
        if far:
            return "divided_regions"
        if w["p_g_mean"] <= th["p_g_low"]:
            return "fragmented_network"
        if w["largest_personality"] == "P2":
            return "independent_largest"
        return "other"

    by_seed = {w["seed"]: w for w in worlds}
    out = {"thresholds": th, "classes": {k: d for k, d, _ in rules}, "by_player": {}}
    for who, (seeds, ekey) in losses.items():
        counts = Counter(classify(by_seed[s], by_seed[s][ekey]) for s in seeds)
        # base rate: the same structural classes among this player's WON worlds (close_call excluded there)
        won = [w for w in worlds if ekey in w and w["seed"] not in seeds]
        base = Counter(classify(w, 0.0) for w in won)   # E = 0 → never close_call
        out["by_player"][who] = {"losses": len(seeds), "counts": dict(counts), "won_worlds": len(won),
                                 "won_counts": dict(base),
                                 "examples": {c: [s for s in seeds if classify(by_seed[s], by_seed[s][ekey]) == c][:5]
                                              for c in counts}}
    return out


# ============================================================================== main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(xc.REPORTS / "x15.json"))
    parser.add_argument("--rng-seed", type=int, default=15)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rng = np.random.default_rng(args.rng_seed)
    start = time.perf_counter()
    outcomes, dup = load_outcomes()
    print(f"[x15] X10 default cell: {dup['identical_to_x0_500']}/{dup['x10_default_rows']} rows identical to X0 ₫500",
          flush=True)
    worlds = []
    t_feat = time.perf_counter()
    for s in sorted(outcomes):
        w = {**world_features(s), **outcomes[s]}
        worlds.append(w)
    feat_seconds = time.perf_counter() - t_feat
    print(f"[x15] features for {len(worlds)} worlds in {feat_seconds:.0f}s", flush=True)
    for w in worlds:
        w["oracle_loss"] = float(not w["oracle_win_500"])
        w["oracle_margin"] = w["oracle_E_500"] - 0.5
        if "bayes_E" in w:
            w["bayes_loss"] = float(not w["bayes_win"])
            w["bayes_margin"] = w["bayes_E"] - 0.5
            w["people_pleaser_loss"] = float(not w["people_pleaser_win"])
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text("")
    for w in worlds:
        xc.write_row(RAW_PATH, {"exp": "X15", **w})
    x6w = [w for w in worlds if w["source"] == "X6"]
    models = {}
    specs = [("oracle_loss", worlds, STRUCT, "oracle_loss", "class"),
             ("oracle_loss_with_start", worlds, WITH_START, "oracle_loss", "class"),
             ("bayes_loss", x6w, STRUCT, "bayes_loss", "class"),
             ("oracle_margin", worlds, STRUCT, "oracle_margin", "reg"),
             ("bayes_margin", x6w, STRUCT, "bayes_margin", "reg"),
             ("people_pleaser_loss", x6w, STRUCT, "people_pleaser_loss", "class"),
             ("start_share", worlds, STRUCT, "start_share", "reg")]
    for name, rows, feats, target, kind in specs:
        models[name] = analyse(name, rows, feats, target, kind, rng)
    # pre-registered hypotheses
    lost = [w for w in worlds if w["oracle_loss"]]
    won = [w for w in worlds if not w["oracle_loss"]]
    ch = channel_analysis(worlds)
    fam = [{"name": "H1 tv_two_largest higher in ORACLE-loss worlds",
            **mannwhitney([w["tv_two_largest"] for w in lost], [w["tv_two_largest"] for w in won], "greater")},
           {"name": "H2 p_g_mean lower in ORACLE-loss worlds",
            **mannwhitney([w["p_g_mean"] for w in lost], [w["p_g_mean"] for w in won], "less")},
           {"name": "H3 largest-region personality ~ best paid channel (chi-square, X0 ₫500)", "p": ch["chi2"]["p"],
            "cramers_v": ch["chi2"]["cramers_v"]}]
    for x, hp in zip(fam, xc.holm([x["p"] for x in fam])):
        x["holm_p"] = hp
    # descriptive: the rare losses
    bayes_losses = [{k: w[k] for k in ("seed", "bayes_E", "oracle_E_500", "start_share", "tv_two_largest", "p_g_mean",
                                       "largest_personality", "sizes", "tau_ids")} for w in x6w if w["bayes_loss"]]
    oracle_losses = [{k: w[k] for k in ("seed", "source", "oracle_E_500", "start_share", "tv_two_largest", "p_g_mean",
                                        "largest_personality", "sizes", "tau_ids")} for w in lost]
    loss_sets = {"ORACLE": ([w["seed"] for w in lost], "oracle_E_500"),
                 "BayesPlanner": ([w["seed"] for w in x6w if w["bayes_loss"]], "bayes_E"),
                 "PeoplePleaser": ([w["seed"] for w in x6w if w["people_pleaser_loss"]], "people_pleaser_E")}
    tax = taxonomy(worlds, loss_sets)
    by_t = {t: xc.rate([not w[f"oracle_win_{t}"] for w in worlds if f"oracle_win_{t}" in w and w["source"] == "X0"])
            for t in ("500", "750", "1000")}
    # the archetype distances take few values: losses per value, and a Fisher test for "two largest far apart"
    from scipy.stats import fisher_exact
    by_value = {}
    for f in ("tv_two_largest", "tv_mean_pair"):
        tab = {}
        for v in sorted({round(w[f], 4) for w in worlds}):
            ws = [w for w in worlds if round(w[f], 4) == v]
            x6s = [w for w in ws if w["source"] == "X6"]
            tab[str(v)] = {"worlds": len(ws), "oracle_losses": int(sum(w["oracle_loss"] for w in ws)),
                           "x6_worlds": len(x6s), "people_pleaser_losses": int(sum(w["people_pleaser_loss"] for w in x6s)),
                           "bayes_losses": int(sum(w["bayes_loss"] for w in x6s)),
                           "oracle_E_mean": float(np.mean([w["oracle_E_500"] for w in ws]))}
        by_value[f] = tab
    far = [w for w in worlds if w["tv_two_largest"] >= 0.35 - 1e-9]
    near = [w for w in worlds if w["tv_two_largest"] < 0.35 - 1e-9]
    table = [[int(sum(w["oracle_loss"] for w in far)), len(far) - int(sum(w["oracle_loss"] for w in far))],
             [int(sum(w["oracle_loss"] for w in near)), len(near) - int(sum(w["oracle_loss"] for w in near))]]
    fe = fisher_exact(table, alternative="greater")
    by_value["fisher_tv_two_largest_ge_0.35"] = {"table_[far,near]x[loss,win]": table, "p_one_sided": float(fe.pvalue),
                                                 "loss_rate_far": xc.rate([w["oracle_loss"] for w in far]),
                                                 "loss_rate_near": xc.rate([w["oracle_loss"] for w in near])}
    seconds = time.perf_counter() - start
    report = {
        "loss_by_archetype_distance": by_value,
        "experiment": "X15", "question": "Which elections are unwinnable? (hidden-world features → loss)",
        "sources": {"X0": "reports/v3/raw/x0.jsonl (seeds 10000–10099)", "X6": "reports/v3/raw/x6.jsonl (seeds 12000–12099)",
                    "X10": "reports/v3/raw/x10.jsonl default cell N=1e4, T=5 (seeds 10000–10049)",
                    "features": "x_common.features_of (cached raw/features_<seed>.json) + SBM p_g, p_r"},
        "x10_duplication": dup,
        "n_worlds": len(worlds), "n_worlds_x0": len(worlds) - len(x6w), "n_worlds_x6": len(x6w),
        "event_counts": {"oracle_loss_500": int(sum(w["oracle_loss"] for w in worlds)),
                         "oracle_loss_500_x0": int(sum(w["oracle_loss"] for w in worlds if w["source"] == "X0")),
                         "oracle_loss_500_x6": int(sum(w["oracle_loss"] for w in x6w)),
                         "bayes_loss": int(sum(w["bayes_loss"] for w in x6w)),
                         "people_pleaser_loss": int(sum(w["people_pleaser_loss"] for w in x6w))},
        "oracle_loss_by_treasury_x0": by_t,
        "definitions": {
            "oracle_loss": "ORACLE (oracle manifesto + ORACLE planner) lost at ₫500 (X0 for seeds 100xx, X6 for 120xx)",
            "bayes_loss": "BayesPlanner (II, X6) lost", "people_pleaser_loss": "People Pleaser (I, X6) lost",
            "oracle_margin / bayes_margin": "expected share E − 0.5 of that game",
            "start_share": "true E after the ORACLE manifesto commit, before round 1 (no campaign)",
            "tv_two_largest": "total-variation distance between the archetype means of the two largest regions",
            "tv_mean_pair": "mean pairwise archetype TV over the 4 regions",
            "p_g_mean": "mean realised SBM inter-region probability p_rs (off-diagonal), the network's fragmentation "
                        "(low = fragmented)", "p_g_two_largest": "p_rs between the two largest regions",
            "p_intra_mean": "mean intra-region probability p_r", "inter_edge_share": "census share of inter-region edges",
            "lambda2_L": "algebraic connectivity λ₂ of the Laplacian", "rw_gap": "1 − μ₂ of D⁻¹A",
            "size_gini": "Gini of region sizes", "largest_share / two_largest_share": "share of N",
            "zealot_share": "realised zealots / N", "largest_P1..P3": "personality of the largest region (P0 = reference)",
            "models": "standardised features; (a) ridge-logistic (λ=1, intercept free) with 2000 stratified bootstrap "
                      "CIs; ridge-linear for margins; (b) depth-3 CART (Gini / MSE, min leaf 5)",
            "cv": f"5-fold stratified CV repeated {CV_REPEATS}× (standardisation inside folds): AUC (class) or R² (reg); "
                  f"permutation null with {N_PERM} label shuffles (2 repeats each)",
            "top3": "the three features with the largest |standardised logistic/linear coefficient|",
        },
        "models": models, "hypotheses": {"family": "H1–H3, one-sided Mann–Whitney (H1, H2), chi-square (H3); Holm",
                                         "tests": fam},
        "channel_by_personality": ch,
        "losses": {"oracle": oracle_losses, "bayes": bayes_losses}, "taxonomy": tax,
        "deviations": [
            "X10's default cell (N=1e4, T=5, ₫500, seeds 10000–10049) re-runs the X0 ₫500 worlds: "
            f"{dup['identical_to_x0_500']}/{dup['x10_default_rows']} rows agree within 1e-5 in E with the same result "
            f"(max |ΔE| {dup['max_abs_expected_diff']:.1e}, GPU float atomics), so it adds no new world; counted once.",
            "Bayes loss has only 2 events in 100 X6 worlds: no regression, tree or CV AUC is fitted for it. The two "
            "losses are listed, and the Bayes margin E − 0.5 (continuous) is modelled instead.",
            "ORACLE loss at ₫500 has 11 events in 200 worlds; the logistic model uses a weak ridge (λ=1) and every CV "
            "AUC comes with a permutation null.",
            "Supplementary targets not in the plan: People Pleaser loss (27 events), start share and the margins.",
            "scikit-learn is not installed in the env: logistic, ridge, CART, AUC and partial dependence are numpy "
            "implementations in this script.",
            "CPU only (plan §1.1 point 5 lists the X15 fitting as a CPU exception); features use the SciPy spectra "
            "(N = 1e4 < GPU_SPECTRA_MIN_N).",
        ],
        "provenance": xc.provenance_v3(args, "cpu", seconds, extra={"feature_seconds": feat_seconds,
                                                                     "rng_seed": args.rng_seed,
                                                                     "parity": "not applicable (no simulation; CPU "
                                                                               "fitting is a §1.1 exception)"}),
    }
    xc.save(args.out, report)
    print(json.dumps({"events": report["event_counts"], "hypotheses": [(x["name"], round(x["holm_p"], 4)) for x in fam]}),
          flush=True)
    print(f"[x15] done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
