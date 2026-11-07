"""P1 scale report: electorate generation time, memory and graph diagnostics at several N.

Usage: python experiments/scale.py --n 10000 100000 1000000 --device cuda:2 --out reports/v2/scale_electorate.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time

import torch

from dumbocracy.analysis.robustness import robustness_upper_bound
from dumbocracy.analysis.spectral import algebraic_connectivity, connected_components, region_lambda_max
from dumbocracy.config import ElectorateConfig, GameConfig, resolve_device
from dumbocracy.electorate import generate_electorate


def revision() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="reports/v2/scale_electorate.json")
    args = parser.parse_args()
    device = resolve_device(args.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.zeros(1, device=device)  # context creation outside the timed region
    rows = []
    for n in args.n:
        cfg = GameConfig(seed=args.seed, electorate=ElectorateConfig(num_voters=n))
        times = []
        for _ in range(args.repeats):
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            e = generate_electorate(cfg, device=device)
            if cuda:
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
        census = e.census()
        intra = sum(r["intra_edges"] for r in census["regions"])
        row = {
            "num_voters": n, "device": str(device), "generation_seconds": times,
            "median_seconds": sorted(times)[len(times) // 2],
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
            "edges": e.graph.num_edges, "mean_degree": 2 * e.graph.num_edges / n,
            "intra_edge_fraction": intra / max(e.graph.num_edges, 1),
            "isolated_voters": int((e.graph.degree == 0).sum()),
            "sparsity_factor": e.sparsity, "region_sizes": e.sizes.tolist(),
            "region_lambda_max": region_lambda_max(e.graph, e.region_of),
        }
        if n <= 100_000:
            row["components"] = connected_components(e.graph)
            row["algebraic_connectivity"] = algebraic_connectivity(e.graph)
            bound = robustness_upper_bound(e.graph, e.region_of)
            row["robustness_upper_bound"] = {"r_upper": bound.r_upper, "witness": bound.witness, "min_degree": bound.min_degree}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("num_voters", "median_seconds", "peak_allocated_gib", "edges", "mean_degree", "intra_edge_fraction")}))
    report = {"experiment": "P1 electorate scale", "seed": args.seed, "revision": revision(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
              "gpu": torch.cuda.get_device_name(device) if cuda else None, "rows": rows}
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
