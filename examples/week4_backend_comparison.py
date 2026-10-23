#!/usr/bin/env python3
"""Matched CPU/CUDA correctness and timing sample; human-run only."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import platform
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


def sync(backend: str) -> None:
    if backend.startswith("cuda"):
        import torch
        torch.cuda.synchronize(torch.device(backend))


def driver_version() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unavailable"


def code_revision() -> str:
    try:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
            return "uncommitted"
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_episode(config: GameConfig, backend: str, rounds: int) -> dict[str, object]:
    import torch
    device = torch.device(backend) if backend.startswith("cuda") else None
    free_start = total = None
    if device:
        free_start, total = (value / (1024 * 1024) for value in torch.cuda.mem_get_info(device))
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
    engine = GameEngine(backend, revision=code_revision())
    started = time.perf_counter()
    state = engine.initialize(config)
    sync(backend)
    initialized = time.perf_counter()
    initial_graph, initial_voters = state.graph, state.voters
    summary = state.graph_summary
    durations: list[float] = []
    for index in range(rounds):
        action = CampaignAction("broadcast") if index == 0 else CampaignAction("no_op")
        before = time.perf_counter()
        response = engine.step(state, action)
        if not response.accepted:
            raise RuntimeError(f"fixed comparison action rejected: {response.error_code}")
        sync(backend)
        durations.append(time.perf_counter() - before)
        state = response.state
    ended = time.perf_counter()
    result = state.result
    return {
        "backend": backend,
        "device": torch.cuda.get_device_name(device) if device else "CPU",
        "graph_summary": {
            "node_count": summary.node_count,
            "edge_count": summary.edge_count,
            "density": summary.density,
            "mean_degree": summary.mean_degree,
            "component_count": summary.component_count,
        },
        "initial_graph": initial_graph,
        "initial_voters": initial_voters,
        "final_support": tuple(voter.support for voter in state.voters),
        "ledger": (str(state.ledger.initial_budget), str(state.ledger.remaining_budget), str(state.ledger.spent)),
        "result": (result.player_votes, result.electorate_size, result.vote_share, result.victory, result.score) if result else None,
        "init_seconds": initialized - started,
        "median_step_ms": statistics.median(durations) * 1000,
        "step_spread_ms": (max(durations) - min(durations)) * 1000,
        "episode_seconds": ended - started,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device else None,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / (1024 * 1024) if device else None,
        "free_start_MiB": free_start,
        "total_MiB": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, choices=(1000, 10000), default=1000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cuda-device", default="cuda:0")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for a matched CPU/CUDA comparison") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this comparison requires both CPU and CUDA")
    cuda_device = torch.device(args.cuda_device)
    if cuda_device.type != "cuda" or (cuda_device.index is not None and cuda_device.index >= torch.cuda.device_count()):
        raise SystemExit("requested CUDA device is invalid or unavailable")
    p = min(1.0, 8.0 / max(1, args.nodes - 1))
    config = GameConfig(
        SeedConfig(args.seed), GraphSpec("erdos_renyi", args.nodes, p), args.rounds,
        Decimal("100000.00"), DynamicsConfig(.35), ActionConfig(.01, .002),
    )

    # Warm the selected device on a disposable tiny episode, then clear peaks.
    warm_config = GameConfig(SeedConfig(args.seed), GraphSpec("erdos_renyi", 8, 1.0), 1, Decimal("10.00"), DynamicsConfig(.35), ActionConfig(.01, .002))
    warm_engine = GameEngine(str(cuda_device))
    warm_state = warm_engine.initialize(warm_config)
    warm_engine.step(warm_state, CampaignAction("no_op"))
    torch.cuda.synchronize(cuda_device)
    run_episode(warm_config, "cpu", 1)  # disposable CPU warm-up for symmetric setup
    with torch.cuda.device(cuda_device):
        torch.cuda.reset_peak_memory_stats()

    cpu = run_episode(config, "cpu", args.rounds)
    cuda = run_episode(config, str(cuda_device), args.rounds)
    if cpu["initial_graph"] != cuda["initial_graph"] or cpu["initial_voters"] != cuda["initial_voters"]:
        raise RuntimeError("paired CPU/CUDA initialization was not identical")
    max_difference = max(abs(left - right) for left, right in zip(cpu["final_support"], cuda["final_support"]))
    correct = (
        len(cpu["final_support"]) == len(cuda["final_support"])
        and max_difference <= 1e-10
        and cpu["ledger"] == cuda["ledger"]
        and cpu["result"] == cuda["result"]
    )
    # Voter arrays are needed for comparison but are deliberately omitted from output.
    cpu.pop("initial_graph"); cuda.pop("initial_graph")
    cpu.pop("initial_voters"); cuda.pop("initial_voters")
    cpu.pop("final_support"); cuda.pop("final_support")
    record = {
        "schema_version": "1.0",
        "seed": args.seed,
        "nodes": args.nodes,
        "edge_probability": p,
        "rounds": args.rounds,
        "action_sequence": ["broadcast"] + ["no_op"] * (args.rounds - 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver_version": driver_version(),
        "cuda_device": torch.cuda.get_device_name(cuda_device),
        "revision": code_revision(),
        "correctness": {"passed": correct, "identical_initialization": True, "max_final_support_abs_difference": max_difference, "tolerance": 1e-10, "ledger_equal": cpu["ledger"] == cuda["ledger"], "result_equal": cpu["result"] == cuda["result"]},
        "runs": [cpu, cuda],
        "interpretation": "matched same-seed workload; timings are environment-specific and do not alone establish general speedup",
    }
    print(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if correct else 2


if __name__ == "__main__":
    raise SystemExit(main())
