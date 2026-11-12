"""Stage 2: joint plan + whole-percent manifesto search on one true world (full information).

Usage: python Demo/win/search/search.py --seed S --screening <screen.jsonl> --out <search.jsonl> [--device cuda:0]

Plan = workers per (round, region) and media per (round, region), total cost ≤ ₫830 (₫20 kept for 2 questions);
manifesto = 5 whole percentages. First-improvement local search (random neighbour order, chunks of 12 candidates):
move / add / remove one worker, toggle an airing, move an airing to another round, swap an airing for workers (and
back), move 1–2 manifesto points between sectors. Starts: the best ``--starts`` of {ORACLE planner's executed plan,
7 scripted templates} under the ORACLE manifesto. Objective: mean expected share over the canonical streams + 3
salted stream sets (salts 1001..1003), via ``x_batch.run_batch`` (aggregation="exact", the engine's own summation).
Final plan re-scored on the canonical streams and 10 fresh salted stream sets (salts 1..10, never searched).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F401,F403

import numpy as np  # noqa: E402
from x_batch import run_batch  # noqa: E402
from x_streams import SaltedStreams  # noqa: E402

from dumbocracy.config import SeedStreams  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from screen import templates  # noqa: E402


class World:
    def __init__(self, seed, dev, n_streams=4, chunk_worlds=48):
        self.seed, self.dev = seed, dev
        self.cfg = cfg_for(seed, dev)
        self.e = generate_electorate(self.cfg, device=dev)
        self.sizes = [int(x) for x in self.e.sizes.tolist()]
        self.region_cents = [int(media_cost(self.sizes, tuple(i == r for i in range(4)), self.cfg.media.base_cost) * 100)
                             for r in range(4)]
        self.streams = [SeedStreams(seed)] + [SaltedStreams(seed, {s: 1000 + k for s in SALTED}) for k in range(1, n_streams)]
        self.chunk_worlds = chunk_worlds
        self.evals = 0

    def cost_cents(self, h, m) -> int:
        return int(np.sum(h)) * 2000 + int(sum(self.region_cents[r] for t in range(T) for r in range(4) if m[t][r]))

    def run(self, cands, streams):
        """cands: list of (hires, media, pct). Returns [len(cands), len(streams)] expected shares."""
        items = [(c, s) for c in range(len(cands)) for s in range(len(streams))]
        out = np.zeros((len(cands), len(streams)))
        for i in range(0, len(items), self.chunk_worlds):
            part = items[i:i + self.chunk_worlds]
            pols = [Plan(cands[c][0], cands[c][1]) for c, _ in part]
            ms = torch.tensor([web_manifesto(list(cands[c][2])) for c, _ in part], dtype=self.e.preferences.dtype,
                              device=self.dev)
            res = run_batch([self.e] * len(part), self.cfg, ms, pols, streams=[streams[s] for _, s in part],
                            aggregation="exact")
            for (c, s), o in zip(part, res):
                out[c, s] = o.result.expected_share
        self.evals += len(items)
        return out

    def score(self, cands):
        return self.run(cands, self.streams).mean(1)


def key(c):
    return (tuple(map(tuple, c[0])), tuple(map(tuple, c[1])), tuple(c[2]))


def neighbours(w, cand, rng):
    h, m, pct = cand
    h = [list(r) for r in h]
    m = [list(r) for r in m]
    out = []

    def add(h2, m2, p2):
        if w.cost_cents(h2, m2) <= 83000:
            out.append(([tuple(r) for r in h2], [tuple(r) for r in m2], tuple(p2)))
    cells = [(t, r) for t in range(T) for r in range(4)]
    for t, r in cells:
        if h[t][r] > 0:
            for t2, r2 in cells:
                if (t2, r2) != (t, r):
                    h2 = [list(x) for x in h]; h2[t][r] -= 1; h2[t2][r2] += 1; add(h2, m, pct)  # noqa: E702
            h2 = [list(x) for x in h]; h2[t][r] -= 1; add(h2, m, pct)  # noqa: E702
        h2 = [list(x) for x in h]; h2[t][r] += 1; add(h2, m, pct)  # noqa: E702
        m2 = [list(x) for x in m]; m2[t][r] = not m2[t][r]; add(h, m2, pct)  # noqa: E702
        if m[t][r]:
            for t2 in range(T):
                if t2 != t and not m[t2][r]:
                    m2 = [list(x) for x in m]; m2[t][r] = False; m2[t2][r] = True; add(h, m2, pct)  # noqa: E702
            k = w.region_cents[r] // 2000
            for t2, r2 in cells:
                m2 = [list(x) for x in m]; m2[t][r] = False  # noqa: E702
                for kk in range(1, k + 2):
                    h2 = [list(x) for x in h]; h2[t2][r2] += kk; add(h2, m2, pct)  # noqa: E702
        else:
            k = -(-w.region_cents[r] // 2000)
            for t2, r2 in cells:
                if h[t2][r2] >= 1:
                    m2 = [list(x) for x in m]; m2[t][r] = True  # noqa: E702
                    h2 = [list(x) for x in h]; h2[t2][r2] = max(0, h2[t2][r2] - k); add(h2, m2, pct)  # noqa: E702
    for i in range(5):
        for j in range(5):
            if i != j:
                for d in (1, 2):
                    if pct[i] - d >= 0:
                        p2 = list(pct); p2[i] -= d; p2[j] += d; add(h, m, p2)  # noqa: E702
    rng.shuffle(out)
    return out


def local_search(w, start, start_score, rng, max_evals=4000, chunk=12, log=print):
    cur, cur_s = start, start_score
    seen = {key(cur)}
    used = 0
    while used < max_evals:
        nb = [c for c in neighbours(w, cur, rng) if key(c) not in seen]
        improved = False
        for i in range(0, len(nb), chunk):
            part = nb[i:i + chunk]
            seen.update(key(c) for c in part)
            s = w.score(part)
            used += len(part)
            j = int(np.argmax(s))
            if s[j] > cur_s + 1e-6:
                cur, cur_s, improved = part[j], float(s[j]), True
                log(f"  improve {cur_s:.5f} (evals {used}) pct={list(cur[2])} h={[list(x) for x in cur[0]]} "
                    f"m={[[int(y) for y in x] for x in cur[1]]}")
                break
            if used >= max_evals:
                break
        if not improved:
            break
    return cur, cur_s, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--screening", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-evals", type=int, default=4000)
    ap.add_argument("--starts", type=int, default=3)
    args = ap.parse_args()
    if args.device in ("cuda:1",):
        raise SystemExit("never use GPU 1")
    torch.set_num_threads(4)
    dev = torch.device(args.device)
    t0 = time.time()
    seed = args.seed
    row = [json.loads(line) for line in open(args.screening) if json.loads(line)["seed"] == seed][0]
    w = World(seed, dev)
    pct = tuple(row["manifesto_pct"])
    b = torch.tensor(web_manifesto(list(pct)), dtype=w.e.preferences.dtype, device=dev)
    tps, _, _ = templates(w.e, b, w.cfg)
    starts = [([tuple(a[0]) for a in row["planner_plan"]], [tuple(bool(x) for x in a[1]) for a in row["planner_plan"]],
               pct, "planner")]
    starts += [(p.hires, p.media, pct, name) for name, p in tps.items() if w.cost_cents(p.hires, p.media) <= 83000]
    s = w.score([c[:3] for c in starts])
    order = np.argsort(-s)
    print(f"seed {seed}: starts {[(starts[i][3], round(float(s[i]), 4)) for i in order]}", flush=True)
    rng = random.Random(seed)
    results = []
    for i in order[:args.starts]:
        best, bs, used = local_search(w, starts[i][:3], float(s[i]), rng, max_evals=args.max_evals)
        results.append((bs, best, starts[i][3]))
        print(f" from {starts[i][3]}: {bs:.5f} after {used} evals", flush=True)
    results.sort(key=lambda x: -x[0])
    bs, best, origin = results[0]
    fresh = [SeedStreams(seed)] + [SaltedStreams(seed, {k: s_ for k in SALTED}) for s_ in range(1, 11)]
    ev = w.run([best], fresh)[0]
    out = {"seed": seed, "search_score": bs, "start": origin, "pct": list(best[2]), "hires": [list(x) for x in best[0]],
           "media": [[int(y) for y in x] for x in best[1]], "cost": w.cost_cents(best[0], best[1]) / 100,
           "E_canonical": float(ev[0]), "E_fresh_mean": float(ev[1:].mean()), "E_fresh_min": float(ev[1:].min()),
           "screen_E_best": row["E_best"], "nocamp_best_share": row["nocamp_best"]["continuous_share"],
           "seconds": time.time() - t0, "world_evals": w.evals, "device": str(dev)}
    print(json.dumps(out), flush=True)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
