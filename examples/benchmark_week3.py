#!/usr/bin/env python3
"""Benchmark harness; run only by the user/reviewer per Week 3 runbook."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dumbocracy.campaign import CampaignAction
from dumbocracy.config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from dumbocracy.engine import GameEngine


def code_revision() -> str:
    try:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
            return "uncommitted"
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def benchmark(nodes: int, seed: int, backend: str, rounds: int) -> dict[str, object]:
    probability = min(1.0, 8.0 / max(1, nodes - 1))
    config = GameConfig(SeedConfig(seed), GraphSpec("erdos_renyi", nodes, probability), rounds, Decimal("1000000.00"), DynamicsConfig(.35), ActionConfig(.001, .001))
    import torch
    device = torch.device(backend) if backend.startswith("cuda") else None
    free_start_mib = total_mib = None
    if device:
        free_start_mib, total_mib = (value / (1024 * 1024) for value in torch.cuda.mem_get_info(device))
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
    engine = GameEngine(backend, revision=code_revision())
    # Warm up one transition on a separate episode so initialization timing is
    # not mixed with the timed round sample.
    warm = engine.initialize(config)
    warmed = engine.step(warm, CampaignAction("no_op"))
    if not warmed.accepted:
        raise RuntimeError("benchmark warm-up no_op was unexpectedly rejected")
    if device:
        torch.cuda.synchronize(device)
    del warm, warmed
    if device:
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    state = engine.initialize(config)
    initialized = time.perf_counter()
    if device:
        torch.cuda.synchronize(device)
    step_durations: list[float] = []
    for _ in range(rounds):
        before = time.perf_counter()
        response = engine.step(state, CampaignAction("no_op"))
        if device:
            torch.cuda.synchronize(device)
        step_durations.append(time.perf_counter() - before)
        state = response.state
    ended = time.perf_counter()
    summary = state.graph.summary()
    result = state.result
    return {
        "backend": backend,
        "GPU/device": torch.cuda.get_device_name(device) if device else "CPU",
        "N": summary.node_count,
        "E": summary.edge_count,
        "p": probability,
        "seed": seed,
        "rounds": rounds,
        "action_sequence": ["no_op"] * rounds,
        "init_seconds": initialized - start,
        "median_step_ms": statistics.median(step_durations) * 1000,
        "step_spread_ms": (max(step_durations) - min(step_durations)) * 1000,
        "episode_seconds": ended - start,
        "peak_GPU_memory_MiB": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device else None,
        "peak_GPU_reserved_memory_MiB": torch.cuda.max_memory_reserved(device) / (1024 * 1024) if device else None,
        "free_GPU_memory_start_MiB": free_start_mib,
        "total_GPU_memory_MiB": total_mib,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "revision": code_revision(),
        "final_vote_share": result.vote_share if result else None,
        "victory": result.victory if result else None,
        "correctness_check": "not run by benchmark; see separate CPU/GPU parity check",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nodes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the benchmark harness; use week3_gpu_episode.py for CPU episode fallback") from exc
    if args.device == "auto":
        backend = "cuda:0" if torch.cuda.is_available() else "cpu"
        if backend == "cpu":
            print("CUDA fallback: no compatible GPU device; selecting CPU", file=sys.stderr)
    else:
        backend = args.device
        if backend.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
    for nodes in args.nodes:
        print(json.dumps(benchmark(nodes, args.seed, backend, args.rounds), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
