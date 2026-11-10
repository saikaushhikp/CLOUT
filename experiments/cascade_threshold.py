"""IC cascade reach vs the regional spectral radius (P8: R24–R27; replaces the old SIR τ_c threshold probe).

For the default electorate (N = 10 000, seed 1, CPU) prints λ_max(A_rr) per region and, for each region,
the mean per-hop number of voters converted when ``--workers`` random eligible voters of that region are
bought as workers, run for ``--hops`` one-hop-per-round IC steps with chance γ_j·P_j (P on v(0) for the
manifesto = that region's mean preference, frozen across hops; no social step), averaged over ``--trials``
generator seeds. Also prints the branching estimate R0 ≈ k̄_r · E_r[γP] (the IC analogue of τ·λ/ρ).

Usage: python experiments/cascade_threshold.py [--n 10000] [--seed 1] [--trials 20] [--workers 5] [--hops 5]
"""
from __future__ import annotations

import argparse

import torch

from dumbocracy.analysis.spectral import region_lambda_max
from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.dynamics.campaign import conversion_chance, eligible_workers, ic_hop, pick_workers
from dumbocracy.electorate import generate_electorate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--hops", type=int, default=5)
    args = parser.parse_args()

    cfg = GameConfig(seed=args.seed, device="cpu", electorate=ElectorateConfig(num_voters=args.n))
    e = generate_electorate(cfg)
    graph, region_of, zealot = e.graph, e.region_of, e.susceptibility.zealot
    lambdas = region_lambda_max(graph, region_of)
    rows = graph.rows()
    print(f"N={args.n} seed={args.seed} sizes={e.sizes.tolist()} edges={graph.num_edges} "
          f"mean_degree={2 * graph.num_edges / args.n:.2f}")
    print(f"\nper-hop converts ({args.workers} workers in one region, b = region mean, {args.trials} trials)")
    for r, lam in enumerate(lambdas):
        b = e.preferences[region_of == r].mean(0)
        b = b / b.sum()
        chance = conversion_chance(e.susceptibility.campaign, e.preferences, b, cfg.voting.sensitivity, cfg.voting.threshold)
        mask = region_of == r
        r0 = float(graph.degree[mask].double().mean()) * float(chance[mask].mean())
        per_hop = [0.0] * args.hops
        for trial in range(args.trials):
            eligible = eligible_workers(graph.degree, zealot, torch.zeros_like(zealot), cfg.workers.min_degree)
            hires = [0] * 4
            hires[r] = args.workers
            ids = torch.cat(pick_workers(region_of, eligible, hires, torch.Generator().manual_seed(1000 * r + trial)))
            pinned = torch.zeros(args.n, dtype=torch.bool)
            pinned[ids] = True
            spreaders, attempts = pinned.clone(), torch.zeros(args.n, dtype=torch.int64)
            for h in range(args.hops):
                spreaders, attempts = ic_hop(graph, spreaders, chance, attempts, pinned | zealot, cfg.workers.attempt_cap,
                                             torch.Generator().manual_seed(5000 + 100 * trial + h), rows)
                per_hop[h] += int(spreaders.sum()) / args.trials
        print(f"region {r}: lambda_max(A_rr) = {lam:6.2f}  R0 ~ {r0:5.2f}  converts/hop = "
              + " ".join(f"{x:7.1f}" for x in per_hop))


if __name__ == "__main__":
    main()
