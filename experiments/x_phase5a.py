"""Shared helpers for X20, X21 and X22 (plan §7, Phase 5): hard-pinned device, a memory-capped batch sizer and
the figure style.

* :func:`pin_device` returns ``cuda:<k>`` exactly as requested (default cuda:2) and raises if it lacks the memory:
  unlike ``x_common.select_device`` it NEVER falls back to another GPU (shared host; GPU 1 belongs to another
  user, GPU 0 to another of our agents). It also makes that device the process default.
* :func:`capped_batch` wraps ``x_common.calibrate_batch`` and caps the batch at ``cap_bytes`` (default 10 GiB) of
  GPU memory as well as 70% of the free memory, because another of our processes shares cuda:2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402

CAP_BYTES = 10 * 2**30
FIGURES = xc.REPORTS / "figures"
LOGS = xc.REPORTS / "logs"

# figure palette: the same constants as plot_v3.py (dataviz reference palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#f0efec"


def pin_device(requested: str = "cuda:2", need_bytes: int = 2 * 2**30) -> tuple[torch.device, dict]:
    if not requested.startswith("cuda:"):
        raise ValueError("X20-X22 run on an explicit cuda:<k> only (no CPU, no auto selection)")
    index = int(requested.split(":", 1)[1])
    if index == 1:
        raise ValueError("GPU 1 is another user's job; never use it")
    stats = xc.gpu_stats()
    free = stats.get(index, {}).get("free", 0)
    if free < need_bytes:
        raise RuntimeError(f"{requested} has only {free / 2**30:.1f} GiB free (< {need_bytes / 2**30:.1f}); "
                           "refusing to fall back to another GPU")
    torch.cuda.set_device(index)
    record = {"requested": requested, "chosen": requested, "pinned": True, "need_bytes": int(need_bytes),
              "free_bytes": {str(i): s["free"] for i, s in stats.items()},
              "util": {str(i): s["util"] for i, s in stats.items()}}
    return torch.device("cuda", index), record


def capped_batch(probe, device: torch.device, *, sizes=(2, 4), max_batch: int = 1024,
                 cap_bytes: int = CAP_BYTES) -> dict:
    """``calibrate_batch`` with the batch also limited so fixed + per_item·B ≤ ``cap_bytes``."""
    cal = xc.calibrate_batch(probe, device, sizes=sizes, max_batch=max_batch)
    if cal.get("per_item_bytes"):
        by_cap = int((cap_bytes - cal["fixed_bytes"]) // cal["per_item_bytes"])
        cal["batch_uncapped"] = cal["batch"]
        cal["batch"] = max(1, min(cal["batch"], by_cap))
        cal["cap_bytes"] = int(cap_bytes)
    return cal


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def new_figure(*args, **kwargs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(*args, **kwargs)


def save_figure(fig, name: str) -> str:
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(xc.ROOT)}", flush=True)
    return str(path.relative_to(xc.ROOT))
