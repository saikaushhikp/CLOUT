#!/usr/bin/env python3
"""Run a scripted episode and optional sparse-network scale study."""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

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


def choose_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested != "auto":
            raise SystemExit("PyTorch is required for the requested CUDA device")
        print("CUDA fallback: PyTorch is not installed; selecting CPU", file=sys.stderr)
        return "cpu"
    if torch.cuda.is_available():
        index = 0 if requested == "auto" else int(requested.split(":", 1)[1])
        return f"cuda:{index}"
    if requested != "auto":
        raise SystemExit("CUDA was requested but the installed driver/device is unavailable")
    print("CUDA fallback: no compatible GPU device; selecting CPU", file=sys.stderr)
    return "cpu"


def episode(nodes: int, seed: int, backend: str, rounds: int, probability: float | None = None) -> dict[str, object]:
    if probability is None:
        probability = min(1.0, 8.0 / max(1, nodes - 1))
    config = GameConfig(
        SeedConfig(seed), GraphSpec("erdos_renyi", nodes, probability), rounds,
        Decimal("100000.00"), DynamicsConfig(.35), ActionConfig(.01, .002),
    )
    if backend.startswith("cuda"):
        import torch
        device = torch.device(backend)
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        gpu_free_mib, gpu_total_mib = (value / (1024 * 1024) for value in torch.cuda.mem_get_info(device))
        torch_version, cuda_version = torch.__version__, torch.version.cuda
    else:
        gpu_free_mib = gpu_total_mib = None
        try:
            import torch
            torch_version, cuda_version = torch.__version__, torch.version.cuda
        except ImportError:
            torch_version, cuda_version = "not-installed", None
    start = time.perf_counter()
    revision = code_revision()
    engine = GameEngine(backend, revision=revision)
    state = engine.initialize(config)
    initialized = time.perf_counter()
    graph_summary = state.graph.summary()
    step_times: list[float] = []
    for round_index in range(rounds):
        action = CampaignAction("broadcast") if round_index == 0 else CampaignAction("no_op")
        before = time.perf_counter()
        response = engine.step(state, action)
        if not response.accepted:
            raise RuntimeError(f"script action rejected: {response.error_code}")
        if backend.startswith("cuda"):
            import torch
            torch.cuda.synchronize(torch.device(backend))
        step_times.append(time.perf_counter() - before)
        state = response.state
    finished = time.perf_counter()
    memory_mib = None
    reserved_memory_mib = None
    device_name = "CPU"
    if backend.startswith("cuda"):
        import torch
        device = torch.device(backend)
        memory_mib = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        reserved_memory_mib = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        device_name = torch.cuda.get_device_name(device)
    result = state.result
    return {
        "backend": backend,
        "device": device_name,
        "N": graph_summary.node_count,
        "E": graph_summary.edge_count,
        "density": graph_summary.density,
        "average_degree": graph_summary.mean_degree,
        "p": probability,
        "seed": seed,
        "rounds": state.round_index,
        "action_sequence": ["broadcast"] + ["no_op"] * (rounds - 1),
        "init_seconds": initialized - start,
        "median_step_ms": sorted(step_times)[len(step_times) // 2] * 1000,
        "episode_seconds": finished - start,
        "peak_GPU_memory_MiB": memory_mib,
        "peak_GPU_reserved_memory_MiB": reserved_memory_mib,
        "free_GPU_memory_start_MiB": gpu_free_mib,
        "total_GPU_memory_MiB": gpu_total_mib,
        "final_vote_share": result.vote_share if result else None,
        "victory": result.victory if result else None,
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda_runtime": cuda_version,
        "revision": revision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--benchmark", action="store_true", help="run 1k/10k/100k sparse ER scale rows")
    args = parser.parse_args()
    if args.nodes < 1 or args.rounds < 1:
        parser.error("nodes and rounds must be positive")
    backend = choose_device(args.device)
    node_values = (1000, 10000, 100000) if args.benchmark else (args.nodes,)
    for nodes in node_values:
        record = episode(nodes, args.seed, backend, args.rounds)
        print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
