"""Shared helpers for the v3 experiment suite (plan ``docs/v2/plans/implementation_plan.md`` §1–§2).

* statistics: re-exports ``p7_common`` (``wilson``, ``rate``, ``mean_ci``, ``paired``, ``mcnemar``, ``spearman``)
  and adds :func:`holm` and :func:`power_paired_binary` (§1.3);
* seeds: :func:`seed_block` (§1.5 table);
* data: :func:`write_row` (JSONL), ``save`` (summary JSON), :func:`override` (``--set a.b=v``, moved from
  ``balance.py``, which keeps its own copy);
* compute, GPU first (§1.1): :func:`select_device`, :func:`calibrate_batch` (memory-based batch sizer),
  :func:`run_chunks` (OOM back-off), :class:`GpuMonitor` (``nvidia-smi`` utilisation sampler) and
  :func:`provenance_v3` (device, batch sizes, peak/free memory, wall time, utilisation);
* correctness: :func:`parity_check` (CPU vs GPU with a statistical tolerance, §1.1 point 7);
* :func:`features_of`: the hidden-world features of §2 (X15 and covariates), spectra on the GPU with a
  SciPy CPU reference/fallback, cached to ``reports/v3/raw/features_<seed>.json``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

from p7_common import (  # noqa: F401  (re-exported)
    ROOT, mean_ci, mcnemar, paired, pick_device, provenance, rate, run_tasks, save, spearman, wilson,
)
from dumbocracy.config import GameConfig, cuda_free_memory

REPORTS = ROOT / "reports" / "v3"
RAW = REPORTS / "raw"
DEFAULT_DEVICE = "cuda:2"
# GPUs never chosen automatically on the shared host (GPU 1 is another group's job); override with X_AVOID_GPUS="".
AVOID_GPUS = tuple(int(x) for x in os.environ.get("X_AVOID_GPUS", "1").split(",") if x.strip())


# ============================================================================== statistics (§1.3)
def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (step-down, monotone, capped at 1), in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(pvalues[i])))
        adjusted[i] = running
    return adjusted


def power_paired_binary(p_disc: float, delta: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Pairs needed for a two-sided McNemar test (Connor 1987).

    ``p_disc`` = expected discordant fraction (p10 + p01), ``delta`` = win-rate difference p10 − p01.
    n = [z_{1−α/2} √p_disc + z_{power} √(p_disc − δ²)]² / δ².
    """
    from scipy.stats import norm
    if not 0 < abs(delta) <= p_disc <= 1:
        raise ValueError("need 0 < |delta| <= p_disc <= 1")
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    n = (za * math.sqrt(p_disc) + zb * math.sqrt(p_disc - delta * delta)) ** 2 / (delta * delta)
    return int(math.ceil(n))


# ============================================================================== seeds (§1.5)
SEED_BLOCKS: dict[str, int] = {
    **{x: 10000 for x in ("X0", "X7", "X10", "X23")},
    **{x: 11000 for x in ("X1", "X2", "X3", "X4", "X11", "X18")},
    **{x: 12000 for x in ("X6", "X12", "X13", "X14", "X19")},
    **{x: 13000 for x in ("X5", "X8", "X16", "X17")},
    **{x: 14000 for x in ("X20", "X21", "X22")},
    "X9": 20261107,
}


def seed_block(exp_id: str, n: int) -> list[int]:
    """The first ``n`` seeds of an experiment's block (§1.5). Arms of one experiment share this list (paired).
    X15 is post-hoc (it reuses the seeds of the experiments it reads) and has no block. X9 uses the canonical
    seed: ``seed_block("X9", 1) == [20261107]``."""
    key = exp_id.upper()
    if key == "X15":
        raise ValueError("X15 is post-hoc and reuses the seeds of X0, X6 and X10")
    if key not in SEED_BLOCKS:
        raise ValueError(f"no seed block for {exp_id!r}")
    if n < 0 or n > 1000:
        raise ValueError("n must lie in [0, 1000] (blocks are 1000 apart)")
    return list(range(SEED_BLOCKS[key], SEED_BLOCKS[key] + n))


# ============================================================================== data (§1.4)
def _json_default(x: Any) -> Any:
    if isinstance(x, Decimal):
        return str(x)
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return float(x)


def write_row(path: str | Path, row: dict) -> None:
    """Append one JSON object as a line to ``path`` (raw per-game rows, ``reports/v3/raw/<id>.jsonl``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, default=_json_default) + "\n")


def read_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def override(config: GameConfig, path: str, raw: str | None = None) -> GameConfig:
    """``override(cfg, "workers.worker_cost=25.00")`` or ``override(cfg, "workers.worker_cost", "25.00")``.

    Moved from ``balance.py`` (which keeps its copy). The value is parsed by the type of the current field:
    Decimal, bool, int, float, tuple of floats, else the raw string (e.g. ``workers.conversion_rule=gamma``)."""
    if raw is None:
        path, sep, raw = path.partition("=")
        if not sep:
            raise ValueError(f"override needs 'a.b=value', got {path!r}")
    path, raw = path.strip(), raw.strip()
    head, _, rest = path.partition(".")
    current = getattr(config, head)
    if rest:
        return dataclasses.replace(config, **{head: override(current, rest, raw)})
    if isinstance(current, Decimal):
        value: Any = Decimal(raw)
    elif isinstance(current, bool):
        value = raw.lower() in ("1", "true", "yes")
    elif isinstance(current, int):
        value = int(raw)
    elif isinstance(current, float):
        value = float(raw)
    elif isinstance(current, tuple):
        value = tuple(float(x) for x in raw.split(","))
    else:
        value = raw
    return dataclasses.replace(config, **{head: value})


def apply_overrides(config: GameConfig, items: Iterable[str]) -> GameConfig:
    for item in items:
        config = override(config, item)
    return config


# ============================================================================== GPU first (§1.1)
def gpu_stats() -> dict[int, dict[str, int]]:
    """``{torch index: {"free": bytes, "util": %}}`` via ``nvidia-smi`` matched by UUID (no CUDA contexts)."""
    if not torch.cuda.is_available():
        return {}
    free = cuda_free_memory()
    util = {i: 0 for i in free}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True).stdout
        by_uuid = {u.strip().removeprefix("GPU-"): int(x) for u, x in (line.split(",") for line in out.strip().splitlines())}
        for i in free:
            util[i] = by_uuid.get(str(torch.cuda.get_device_properties(i).uuid), 0)
    except Exception:  # noqa: BLE001
        pass
    return {i: {"free": free[i], "util": util[i]} for i in free}


def select_device(requested: str = DEFAULT_DEVICE, need_bytes: int = 2 * 2**30, *, max_util: int = 50,
                  allow_cpu: bool = False) -> tuple[torch.device, dict]:
    """§1.1 point 1. Use ``requested`` (default cuda:2) if it has ``need_bytes`` free; otherwise the GPU with the
    most free memory that is not busy (util < ``max_util``) and not in ``AVOID_GPUS``. Never silently CPU:
    without a usable GPU this raises, unless ``allow_cpu`` (then the returned record has ``deviation``).

    Returns ``(device, record)``; ``record`` goes into the provenance block."""
    if requested == "cpu":
        return torch.device("cpu"), {"requested": "cpu", "chosen": "cpu", "deviation": "CPU requested explicitly"}
    stats = gpu_stats()
    record: dict[str, Any] = {"requested": requested, "need_bytes": int(need_bytes),
                              "free_bytes": {str(i): s["free"] for i, s in stats.items()}}
    if requested.startswith("cuda") and stats:
        index = torch.device(requested).index or 0
        if index in stats and stats[index]["free"] >= need_bytes:
            record["chosen"] = f"cuda:{index}"
            return torch.device("cuda", index), record
        record["fallback_reason"] = f"{requested} lacks {need_bytes} free bytes"
    usable = {i: s for i, s in stats.items() if i not in AVOID_GPUS and s["free"] >= need_bytes and s["util"] < max_util}
    if usable:
        best = max(usable, key=lambda i: usable[i]["free"])
        record["chosen"] = f"cuda:{best}"
        return torch.device("cuda", best), record
    if allow_cpu:
        record.update(chosen="cpu", deviation="no GPU had room; CPU fallback (record in report.md)")
        return torch.device("cpu"), record
    raise RuntimeError(f"no GPU has {need_bytes} free bytes (stats {stats}); refusing to fall back to CPU silently")


def free_bytes(device: torch.device | str) -> int:
    device = torch.device(device)
    if device.type != "cuda":
        return 0
    return int(torch.cuda.mem_get_info(device)[0])


def calibrate_batch(probe: Callable[[int], Any], device: torch.device | str, *, sizes: Sequence[int] = (1, 2),
                    fraction: float = 0.70, max_batch: int = 1024, min_batch: int = 1) -> dict:
    """Memory-based batch sizer (§1.1 point 3): run ``probe(B)`` at two small B, measure the peak memory
    (``torch.cuda.max_memory_allocated``), fit bytes ≈ fixed + per_item·B, and return the largest B whose
    estimate fits in ``fraction`` of the free memory. On CPU returns ``max_batch`` (memory not modelled)."""
    device = torch.device(device)
    if device.type != "cuda":
        return {"batch": max_batch, "per_item_bytes": None, "fixed_bytes": None, "free_bytes": None}
    peaks = []
    for b in sizes:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        probe(int(b))
        torch.cuda.synchronize(device)
        peaks.append(torch.cuda.max_memory_allocated(device) - base)
    (b0, b1), (p0, p1) = sizes[:2], peaks[:2]
    per_item = max((p1 - p0) / max(b1 - b0, 1), p1 / max(b1, 1) * 0.5, 1.0)
    fixed = max(p0 - per_item * b0, 0.0)
    torch.cuda.empty_cache()
    free = free_bytes(device)
    batch = int((fraction * free - fixed) // per_item)
    return {"batch": max(min_batch, min(max_batch, batch)), "per_item_bytes": int(per_item), "fixed_bytes": int(fixed),
            "free_bytes": int(free), "probe_sizes": list(sizes), "probe_peaks": [int(p) for p in peaks]}


def run_chunks(fn: Callable[[list], list], items: Sequence, batch: int, *, log: Callable[[str], None] = print,
               record: dict | None = None) -> list:
    """Run ``fn`` over ``items`` in chunks of ``batch`` (results concatenated in order). On
    ``torch.cuda.OutOfMemoryError`` the chunk size is halved, the cache emptied and the chunk retried; the
    final batch sizes are logged and stored in ``record["batch_sizes"]``. Never falls back to CPU."""
    items = list(items)
    out: list = []
    i = 0
    used: list[int] = []
    while i < len(items):
        chunk = items[i:i + batch]
        try:
            out.extend(fn(chunk))
        except torch.cuda.OutOfMemoryError:
            if batch <= 1:
                raise
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            log(f"[x_common] CUDA out of memory: batch halved to {batch}")
            if record is not None:
                record.setdefault("oom_backoffs", []).append(batch)
            continue
        used.append(len(chunk))
        i += len(chunk)
    if record is not None:
        record["batch_sizes"] = sorted(set(used))
        record["final_batch"] = batch
    return out


class GpuMonitor:
    """Background ``nvidia-smi`` sampler of the GPU utilisation (every ``interval`` s) for one device.

    ``with GpuMonitor(device) as mon: ...`` then ``mon.summary()`` → mean/max utilisation and samples. Does
    nothing on CPU or when ``nvidia-smi`` is missing."""

    def __init__(self, device: torch.device | str, interval: float = 2.0):
        self.device = torch.device(device)
        self.interval = interval
        self.samples: list[tuple[int, int]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._uuid = None
        if self.device.type == "cuda":
            try:
                self._uuid = "GPU-" + str(torch.cuda.get_device_properties(self.device.index or 0).uuid)
            except Exception:  # noqa: BLE001
                self._uuid = None

    def _sample(self) -> None:
        try:
            out = subprocess.run(["nvidia-smi", f"--id={self._uuid}", "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
            util, mem = (int(x) for x in out.strip().split(","))
            self.samples.append((util, mem))
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuMonitor":
        if self._uuid is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def summary(self) -> dict:
        if not self.samples:
            return {"gpu_util_mean": None, "gpu_util_max": None, "samples": 0}
        utils = [u for u, _ in self.samples]
        return {"gpu_util_mean": sum(utils) / len(utils), "gpu_util_max": max(utils), "samples": len(utils),
                "gpu_mem_used_mib_max": max(m for _, m in self.samples), "interval_s": self.interval}


def provenance_v3(args: Any, device: torch.device | str, seconds: float, *, workers: int = 1,
                  batch_sizes: Any = None, free_at_start: int | None = None, monitor: GpuMonitor | None = None,
                  device_record: dict | None = None, parity: dict | None = None, extra: dict | None = None,
                  grad_mode: str = "no_grad") -> dict:
    """``p7_common.provenance`` plus the §1.1 point 6 compute record."""
    device = torch.device(device)
    out = provenance(args if args is not None else _Empty(), str(device), seconds, workers)
    compute: dict[str, Any] = {"device": str(device), "batch_sizes": batch_sizes, "wall_seconds": seconds,
                               "free_bytes_at_start": free_at_start, "peak_bytes": None,
                               "grad_mode": grad_mode, "dtype": "float32"}
    if device.type == "cuda":
        compute["device_name"] = torch.cuda.get_device_name(device)
        compute["peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
    else:
        compute["device_name"] = "cpu"
    if monitor is not None:
        compute.update(monitor.summary())
    if device_record is not None:
        compute["device_selection"] = device_record
    out["compute"] = compute
    if parity is not None:
        out["parity"] = parity
    if extra:
        out.update(extra)
    return out


class _Empty:
    pass


# ============================================================================== CPU↔GPU parity (§1.1 point 7)
def parity_check(fn: Callable[[int, torch.device], dict], seeds: Sequence[int], *, device: str | torch.device = DEFAULT_DEVICE,
                 reference: str = "cpu", keys: Sequence[str] | None = None, abs_tol: float = 0.01, z: float = 3.0) -> dict:
    """Run ``fn(seed, device) -> {metric: number|bool}`` on the CPU reference and on the GPU for each seed and
    compare metric by metric. CPU and GPU draw different random numbers, so the check is statistical (spec §9):
    a metric passes if |mean_gpu − mean_cpu| ≤ max(abs_tol, z · SE of the paired difference). Booleans are
    compared as rates. Returns a JSON-safe record with ``passed``."""
    device = torch.device(device)
    rows = []
    for seed in seeds:
        rows.append({"seed": int(seed), "cpu": fn(int(seed), torch.device(reference)), "gpu": fn(int(seed), device)})
    keys = list(keys or rows[0]["cpu"].keys())
    metrics, passed = {}, True
    for k in keys:
        a = [float(r["cpu"][k]) for r in rows]
        b = [float(r["gpu"][k]) for r in rows]
        d = [y - x for x, y in zip(a, b)]
        n = len(d)
        mean_d = sum(d) / n
        sd = math.sqrt(sum((x - mean_d) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
        tol = max(abs_tol, z * sd / math.sqrt(n))
        ok = abs(mean_d) <= tol
        passed &= ok
        metrics[k] = {"cpu_mean": sum(a) / n, "gpu_mean": sum(b) / n, "mean_diff": mean_d, "tolerance": tol, "passed": ok}
    return {"passed": bool(passed), "seeds": [int(s) for s in seeds], "reference": reference, "device": str(device),
            "metrics": metrics, "rows": rows}


# ============================================================================== hidden-world features (§2)
def _gini(xs: Sequence[float]) -> float:
    xs = sorted(float(x) for x in xs)
    n, total = len(xs), sum(xs)
    if n == 0 or total == 0:
        return 0.0
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * total)


def _sparse(rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, n: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n, n)).coalesce()


def _lobpcg_top(matrix: torch.Tensor, k: int, seed: int = 0, tol: float = 1e-6) -> torch.Tensor:
    """Top-k eigenvalues of a symmetric sparse matrix (residual tol 1e-6 → eigenvalue error ~ tol², far below
    the 9 reported digits)."""
    n = matrix.shape[0]
    gen = torch.Generator(device=matrix.device).manual_seed(seed)
    x = torch.randn(n, k, dtype=torch.float64, device=matrix.device, generator=gen)
    values, _ = torch.lobpcg(matrix, k=k, X=x, largest=True, niter=1000, tol=tol)
    return values


def spectra_gpu(graph, *, block: int = 6) -> dict:
    """λ₂(L) and the random-walk spectrum on the graph's device with ``torch.lobpcg`` (float64, sparse).

    λ₂(L): top-2 of (c·I − L), c = 2·d_max + 1. Random walk: top-2 of (I + S) and top-1 of (I − S) with
    S = D^{-1/2} A D^{-1/2} (same spectrum as D⁻¹A) over non-isolated nodes. Returns
    ``lambda2_L``, ``mu2``, ``mu_min``."""
    n = graph.num_nodes
    dev = graph.device
    rows, cols = graph.rows(), graph.indices
    deg = graph.degree.to(torch.float64)
    c = 2.0 * float(deg.max()) + 1.0
    diag = torch.arange(n, device=dev)
    shifted = _sparse(torch.cat([rows, diag]), torch.cat([cols, diag]),
                      torch.cat([torch.ones(rows.numel(), dtype=torch.float64, device=dev), c - deg]), n)
    top = _lobpcg_top(shifted, block)
    lam2 = max(c - float(top[1]), 0.0)
    keep = graph.degree > 0
    new_index = torch.cumsum(keep.to(torch.int64), 0) - 1
    m = int(keep.sum())
    inv = deg.clamp_min(1).rsqrt()
    edge_keep = keep[rows] & keep[cols]
    r, s = new_index[rows[edge_keep]], new_index[cols[edge_keep]]
    w = inv[rows[edge_keep]] * inv[cols[edge_keep]]
    diag = torch.arange(m, device=dev)
    ones = torch.ones(m, dtype=torch.float64, device=dev)
    plus = _sparse(torch.cat([r, diag]), torch.cat([s, diag]), torch.cat([w, ones]), m)
    minus = _sparse(torch.cat([r, diag]), torch.cat([s, diag]), torch.cat([-w, ones]), m)
    mu2 = float(_lobpcg_top(plus, block)[1]) - 1.0
    mu_min = 1.0 - float(_lobpcg_top(minus, 2)[0])
    return {"lambda2_L": lam2, "mu2": mu2, "mu_min": mu_min}


def spectra_scipy(graph) -> dict:
    """CPU reference: λ₂(L) from ``analysis/spectral.algebraic_connectivity`` and μ₂, μ_min of
    S = D^{-1/2} A D^{-1/2} with SciPy ``eigsh`` (as ``degroot_spectral_gap``, which returns only the |·| gap)."""
    import numpy as np
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh
    from dumbocracy.analysis.spectral import algebraic_connectivity
    rows = graph.rows().cpu().numpy()
    cols = graph.indices.cpu().numpy()
    adjacency = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(graph.num_nodes, graph.num_nodes))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    keep = degree > 0
    inv = diags(1.0 / np.sqrt(degree[keep]))
    sym = inv @ adjacency[keep][:, keep] @ inv
    if sym.shape[0] <= 400:
        values = np.sort(np.linalg.eigvalsh(sym.toarray()))
        mu2, mu_min = float(values[-2]), float(values[0])
    else:
        mu2 = float(min(eigsh(sym, k=2, which="LA", return_eigenvectors=False, tol=1e-8)))
        mu_min = float(eigsh(sym, k=1, which="SA", return_eigenvectors=False, tol=1e-8)[0])
    return {"lambda2_L": algebraic_connectivity(graph), "mu2": mu2, "mu_min": mu_min}


GPU_SPECTRA_MIN_N = 50_000   # measured in Phase 0 (gpu_benchmarks.json): SciPy is faster below ~5e4 voters


def _config_key(config: GameConfig | None) -> str:
    if config is None:
        return "default"
    record = config.to_record()
    for k in ("seed", "device"):
        record.pop(k, None)
    default = GameConfig().to_record()
    for k in ("seed", "device"):
        default.pop(k, None)
    if record == default:
        return "default"
    return hashlib.blake2b(json.dumps(record, sort_keys=True).encode(), digest_size=4).hexdigest()


def features_of(electorate, *, seed: int | None = None, config: GameConfig | None = None, cache: bool = True,
                cache_dir: str | Path = RAW, spectra: str = "auto", round_digits: int = 9) -> dict:
    """Hidden-world features of one electorate (plan §2; X15 and covariates elsewhere).

    Region sizes and their Gini; σ (archetype per region) and τ (personality per region); TV distance between
    the archetype means of the two largest regions and the min / mean pairwise TV over the four regions; the
    personality of the largest region; realised zealots per region; λ₂(L), the random-walk gap 1 − μ₂(D⁻¹A)
    (and the library's |·| version 1 − max(|μ₂|, |μ_min|)); λ_max(A_rr) per region; the census inter-region edge
    share. ``spectra="gpu"`` uses ``torch.lobpcg`` on the electorate's device (SciPy fallback on failure, recorded);
    ``"scipy"`` forces the CPU reference; ``"auto"`` (default) takes the GPU from ``GPU_SPECTRA_MIN_N`` voters on,
    where it was measured faster (§1.1 point 4), else SciPy. The method used is stored in ``spectra_method``.
    With ``seed`` given and ``cache``, the result is cached as ``features_<seed>.json`` (default config) or ``features_<seed>_<confighash>.json``. Floats are rounded to
    ``round_digits`` significant digits so the cache is stable across runs."""
    from dumbocracy.analysis.spectral import region_lambda_max
    from dumbocracy.sectors import PERSONALITIES, archetype_means

    path = None
    if seed is not None and cache:
        key = _config_key(config) if config is not None else \
            ("default" if electorate.num_voters == GameConfig().electorate.num_voters else f"n{electorate.num_voters}")
        name = f"features_{int(seed)}.json" if key == "default" else f"features_{int(seed)}_{key}.json"
        path = Path(cache_dir) / name
        if path.exists():
            return json.loads(path.read_text())

    sizes = [int(x) for x in electorate.sizes.tolist()]
    sigma = [int(x) for x in electorate.archetype_of_region.tolist()]
    tau = [int(x) for x in electorate.personality_of_region.tolist()]
    means = archetype_means(dtype=torch.float64)[torch.tensor(sigma)]
    tv = 0.5 * (means.unsqueeze(0) - means.unsqueeze(1)).abs().sum(-1)                     # [4, 4]
    pairs = [float(tv[i, j]) for i in range(4) for j in range(i + 1, 4)]
    by_size = sorted(range(4), key=lambda r: -sizes[r])
    largest, second = by_size[0], by_size[1]
    zealot = electorate.susceptibility.zealot.to(torch.bool)
    zealots = [int(x) for x in torch.bincount(electorate.region_of[zealot], minlength=4).tolist()]
    census = electorate.census()
    inter = sum(row["edges"] for row in census["inter_edges"])
    graph = electorate.graph

    if spectra == "auto":
        spectra = "gpu" if graph.device.type == "cuda" and graph.num_nodes >= GPU_SPECTRA_MIN_N else "scipy"
    method = spectra
    spec: dict[str, Any] = {}
    if spectra == "gpu":
        try:
            spec = spectra_gpu(graph)
            method = f"lobpcg@{graph.device}"
        except Exception as exc:  # noqa: BLE001
            method = f"scipy (fallback: {type(exc).__name__})"
    if not spec:
        spec = spectra_scipy(graph)
    spec["rw_gap"] = 1.0 - spec["mu2"]
    spec["rw_abs_gap"] = 1.0 - max(abs(spec["mu2"]), abs(spec["mu_min"]))
    lam_rr = region_lambda_max(graph, electorate.region_of)

    def rnd(x: Any) -> Any:
        if x is None or isinstance(x, (int, bool)):
            return x
        x = float(x)
        return 0.0 if x == 0 else round(x, round_digits - 1 - int(math.floor(math.log10(abs(x)))))

    out = {
        "seed": seed, "num_voters": electorate.num_voters, "sizes": sizes, "size_gini": rnd(_gini(sizes)),
        "sigma": sigma, "tau": tau, "tau_ids": [PERSONALITIES[i].id for i in tau],
        "tv_two_largest": rnd(float(tv[largest, second])), "tv_min_pair": rnd(min(pairs)),
        "tv_mean_pair": rnd(sum(pairs) / len(pairs)), "largest_region": largest,
        "largest_region_personality": PERSONALITIES[tau[largest]].id, "zealots_by_region": zealots,
        "lambda2_L": rnd(spec["lambda2_L"]), "rw_gap": rnd(spec.get("rw_gap")), "rw_abs_gap": rnd(spec["rw_abs_gap"]),
        "mu2": rnd(spec.get("mu2")), "mu_min": rnd(spec.get("mu_min")),
        "lambda_max_rr": [rnd(x) for x in lam_rr], "inter_edge_share": rnd(inter / max(1, census["total_edges"])),
        "total_edges": census["total_edges"], "spectra_method": method,
        "config_key": _config_key(config) if config is not None else None,
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1))
    return out
