"""Word-of-mouth SIR reach vs the regional spectral threshold (spec §3.1; informs τ_c calibration, P2).

For the default electorate (N = 10 000, seed 1, CPU) prints λ_max(A_rr) per region and, for each
region and a grid of transmissions τ, the mean number of secondary voters reached per seed when
250 random voters of that region are seeded (ρ = 0.5, L = 3; zealots cannot spread), averaged over
20 generator seeds. Supercritical when τ·λ_max > ρ.

Usage: python experiments/cascade_threshold.py [--n 10000] [--seed 1] [--trials 20]
"""
from __future__ import annotations

import argparse

import torch

from dumbocracy.analysis.spectral import region_lambda_max
from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.dynamics.campaign import sir_cascade
from dumbocracy.electorate import generate_electorate

TRANSMISSIONS = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seeds-per-trial", type=int, default=250)
    parser.add_argument("--recovery", type=float, default=0.5)
    parser.add_argument("--hops", type=int, default=3)
    args = parser.parse_args()

    electorate = generate_electorate(GameConfig(seed=args.seed, device="cpu",
                                                electorate=ElectorateConfig(num_voters=args.n)))
    graph, region_of = electorate.graph, electorate.region_of
    can_spread = ~electorate.susceptibility.zealot
    lambdas = region_lambda_max(graph, region_of)
    print(f"N={args.n} seed={args.seed} sizes={electorate.sizes.tolist()} edges={graph.num_edges} "
          f"mean_degree={2 * graph.num_edges / args.n:.2f}")
    for r, lam in enumerate(lambdas):
        print(f"region {r}: lambda_max(A_rr) = {lam:.2f}  threshold rho/lambda = {args.recovery / lam:.4f}")

    members = [torch.nonzero(region_of == r).squeeze(1) for r in range(4)]
    print(f"\nmean secondary reach per seed ({args.seeds_per_trial} seeds in one region, rho={args.recovery}, "
          f"L={args.hops}, {args.trials} trials)")
    print("tau     " + "".join(f"  R{r}:reach (tau*lam/rho)" for r in range(4)))
    for tau in TRANSMISSIONS:
        cells = []
        for r in range(4):
            total = 0
            for trial in range(args.trials):
                pick = torch.Generator().manual_seed(1000 * r + trial)
                chosen = members[r][torch.randperm(members[r].numel(), generator=pick)[:args.seeds_per_trial]]
                seeds = torch.zeros(args.n, dtype=torch.bool)
                seeds[chosen] = True
                result = sir_cascade(graph, seeds, can_spread, tau, args.recovery, args.hops,
                                     torch.Generator().manual_seed(5000 + trial))
                total += int(result.reached.sum())
            per_seed = total / (args.trials * args.seeds_per_trial)
            cells.append(f"  {per_seed:9.3f} ({tau * lambdas[r] / args.recovery:5.2f})     ")
        print(f"{tau:<8.3f}" + "".join(cells))


if __name__ == "__main__":
    main()
