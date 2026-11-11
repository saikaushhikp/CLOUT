"""CPU↔GPU parity with metric-appropriate tolerances (plan §1.1 point 7; used by the Phase 2/6 scripts).

``x_common.parity_check`` uses max(abs_tol, z·SE). With 3 seeds the SE is itself very noisy, and a fixed absolute
tolerance is too tight for count-like metrics (Phase 1 finding). :func:`parity_rel` re-evaluates the same rows with
tolerance = max(abs_tol, z·SE, rel·max(|mean_cpu|, |mean_gpu|)) per metric (default rel = 5%: CPU and GPU draw
different random numbers, so only agreement within the run-to-run noise of one world is expected, spec §9).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

import x_common as xc


def parity_rel(fn: Callable[[int, torch.device], dict], seeds: Sequence[int], *, device, abs_tol: float = 0.01,
               rel_tol: float = 0.05, z: float = 3.0) -> dict:
    out = xc.parity_check(fn, seeds, device=device, abs_tol=abs_tol, z=z)
    passed = True
    for m in out["metrics"].values():
        rel = rel_tol * max(abs(m["cpu_mean"]), abs(m["gpu_mean"]))
        m["tolerance_abs_or_se"] = m["tolerance"]
        m["tolerance"] = max(m["tolerance"], rel)
        m["passed"] = abs(m["mean_diff"]) <= m["tolerance"]
        passed &= m["passed"]
    out["passed"] = bool(passed)
    out["rule"] = f"|mean_gpu − mean_cpu| ≤ max({abs_tol}, {z}·SE, {rel_tol:.0%}·|mean|)"
    return out
