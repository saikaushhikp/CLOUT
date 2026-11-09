"""Shared helpers for the P7 experiments (E1–E7): device choice, statistics with 95% CIs, paired tests,
provenance and a small deterministic process pool.

Every experiment script: ``python experiments/e{n}_*.py --seeds S --n N --device D --out reports/v2/e{n}_*.json``.
Results depend only on (seed, cell); the pool returns results in task order, so the output is identical
for any ``--workers``.
"""
from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ----------------------------------------------------------------------------- device
def pick_device(requested: str) -> str:
    """``auto``: cuda:2 if it has >= 4 GB free and < 50% utilisation (shared host), else ``cpu``."""
    if requested != "auto":
        return requested
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
            return "cpu"
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            idx, free, util = (int(x.strip()) for x in line.split(","))
            if idx == 2 and free >= 4096 and util < 50:
                return "cuda:2"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


# ----------------------------------------------------------------------------- statistics
def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate(flags: Sequence[bool]) -> dict:
    k, n = int(sum(bool(f) for f in flags)), len(flags)
    return {"rate": k / n if n else float("nan"), "k": k, "n": n, "ci95": wilson(k, n)}


def mean_ci(xs: Sequence[float]) -> dict:
    """Mean with a t-based 95% CI."""
    from scipy.stats import t
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return {"mean": float("nan"), "ci95": [float("nan")] * 2, "sd": float("nan"), "n": 0}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    half = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": m, "ci95": [m - half, m + half], "sd": sd, "n": n}


def paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired difference a − b: mean with 95% CI and a two-sided Wilcoxon signed-rank p."""
    from scipy.stats import wilcoxon
    d = [float(x) - float(y) for x, y in zip(a, b)]
    out = mean_ci(d)
    nonzero = [x for x in d if x != 0]
    out["wilcoxon_p"] = float(wilcoxon(d).pvalue) if len(nonzero) >= 1 else 1.0
    out["a_better"] = sum(1 for x in d if x > 0)
    return out


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (binomial on discordant pairs)."""
    from scipy.stats import binomtest
    gained = sum(1 for x, y in zip(a, b) if x and not y)
    lost = sum(1 for x, y in zip(a, b) if y and not x)
    p = float(binomtest(gained, gained + lost).pvalue) if gained + lost else 1.0
    return {"a_only": gained, "b_only": lost, "p": p}


def spearman(x: Sequence[float], y: Sequence[float]) -> dict:
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue)}


# ----------------------------------------------------------------------------- provenance
def provenance(args: Any, device: str, seconds: float, workers: int) -> dict:
    import torch
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        rev += "+uncommitted" if dirty else ""
    except Exception:  # noqa: BLE001
        rev = "unknown"
    gpu = None
    if device.startswith("cuda"):
        try:
            gpu = torch.cuda.get_device_name(torch.device(device))
        except Exception:  # noqa: BLE001
            gpu = "cuda"
    return {"git": rev, "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "device": device, "gpu": gpu, "workers": workers, "threads_per_worker": _threads(),
            "seconds": seconds, "args": {k: v for k, v in vars(args).items()},
            "command": " ".join(["python"] + sys.argv)}


def _threads() -> int:
    return int(os.environ.get("P7_THREADS", os.environ.get("OMP_NUM_THREADS", "4")))


# ----------------------------------------------------------------------------- pool
def _init_worker(threads: int) -> None:
    import torch
    os.environ["OMP_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)


def run_tasks(fn: Callable[[Any], Any], tasks: Iterable[Any], workers: int, label: str = "") -> list:
    """Run ``fn`` over ``tasks`` in ``workers`` spawned processes; results in task order with progress."""
    tasks = list(tasks)
    threads = _threads()
    start = time.perf_counter()
    if workers <= 1:
        _init_worker(threads)
        out = []
        for i, t in enumerate(tasks):
            out.append(fn(t))
            if (i + 1) % max(1, len(tasks) // 20) == 0 or i + 1 == len(tasks):
                print(f"{label} [{i + 1}/{len(tasks)}] {time.perf_counter() - start:.0f}s", flush=True)
        return out
    results: list = [None] * len(tasks)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init_worker,
                             initargs=(threads,)) as pool:
        futures = {pool.submit(fn, t): i for i, t in enumerate(tasks)}
        done = 0
        from concurrent.futures import as_completed
        for f in as_completed(futures):
            results[futures[f]] = f.result()
            done += 1
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"{label} [{done}/{len(tasks)}] {time.perf_counter() - start:.0f}s", flush=True)
    return results


def save(path: str, report: dict) -> None:
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=1, default=float))
    print(f"saved {path}", flush=True)
