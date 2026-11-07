"""Validated, immutable game configuration, named seed streams and device selection (spec §9).

Every default marked ⚙ in ``docs/v2/technical_spec_v2.md`` lives here and nowhere else.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
import hashlib
import math
from typing import Literal

import torch

SEED_STREAMS: tuple[str, ...] = (
    "regions", "graph", "block_probs", "preferences", "attributes", "zealots",
    "workers", "cascade", "media", "poll", "election", "strategy", "worlds",
)


class ConfigError(ValueError):
    """Raised for an invalid configuration value."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _finite(value: float, name: str) -> float:
    _check(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"{name} must be a finite number")
    return float(value)


def _prob(value: float, name: str) -> float:
    _check(0.0 <= _finite(value, name) <= 1.0, f"{name} must lie in [0, 1]")
    return float(value)


def _money(value: Decimal, name: str) -> None:
    _check(isinstance(value, Decimal) and value.is_finite() and value >= 0, f"{name} must be a finite non-negative Decimal")


def _range(rng: tuple[float, float], name: str) -> None:
    _check(len(rng) == 2 and 0.0 < _finite(rng[0], name) <= _finite(rng[1], name) <= 1.0, f"{name} must satisfy 0 < low <= high <= 1")


@dataclass(frozen=True, slots=True)
class ElectorateConfig:
    num_voters: int = 10_000
    min_region_fraction: float = 0.15                     # R7: every region >= 15% of N
    region_dirichlet_a: float = 2.0                       # ⚙ §2.1 (split of the remaining voters)
    mean_degree: float = 30.0                             # R6: k̄ = 30 friends per voter (sets s)
    intra_prob_range: tuple[float, float] = (0.5, 0.9)    # R5: p_r ~ U[range]
    inter_prob_range: tuple[float, float] = (0.01, 0.2)   # R5: p_g ~ U[range]
    dirichlet_concentration: float = 60.0                 # κ §2.3 (v2.1 calibrated)
    zealot_fraction: float = 0.05                         # ⚙ ζ §2.4

    def __post_init__(self) -> None:
        _check(isinstance(self.num_voters, int) and not isinstance(self.num_voters, bool) and self.num_voters >= 4, "num_voters must be an integer >= 4")
        _check(0.0 <= _finite(self.min_region_fraction, "min_region_fraction") <= 0.25, "min_region_fraction must lie in [0, 0.25]")
        _check(4 * math.ceil(self.min_region_fraction * self.num_voters) <= self.num_voters, "num_voters too small for the minimum region fraction")
        _check(_finite(self.region_dirichlet_a, "region_dirichlet_a") > 0, "region_dirichlet_a must be > 0")
        _check(_finite(self.mean_degree, "mean_degree") >= 0, "mean_degree must be >= 0")
        _range(self.intra_prob_range, "intra_prob_range")
        _range(self.inter_prob_range, "inter_prob_range")
        _check(_finite(self.dirichlet_concentration, "dirichlet_concentration") > 0, "dirichlet_concentration must be > 0")
        _prob(self.zealot_fraction, "zealot_fraction")


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    social_step: float = 0.15                             # ω §4 (v2.1 calibrated)
    social_rule: Literal["degroot", "wmsr"] = "degroot"   # §4
    wmsr_trim: int = 1                                    # F for W-MSR

    def __post_init__(self) -> None:
        _prob(self.social_step, "social_step")
        _check(self.social_rule in ("degroot", "wmsr"), "social_rule must be 'degroot' or 'wmsr'")
        _check(isinstance(self.wmsr_trim, int) and self.wmsr_trim >= 0, "wmsr_trim must be an integer >= 0")


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    free_workers: int = 10                                # ⚙ W_free §3.1
    extra_worker_cost: Decimal = Decimal("4.00")          # ⚙
    contacts_fraction: float = 0.0025                     # ⚙ q = max(1, round(fraction * N))
    contact_strength: float = 0.50                        # η_w (v2.1 calibrated)
    transmission: float = 0.03                            # τ_c (v2.1): τ_c·λ_max(A_rr)/ρ ≈ 1.1–1.7, just supercritical
    recovery: float = 0.5                                 # ⚙ ρ
    hops: int = 3                                         # ⚙ L
    word_of_mouth_strength: float = 0.15                  # η_wom (v2.1 calibrated)

    def __post_init__(self) -> None:
        _check(isinstance(self.free_workers, int) and self.free_workers >= 0, "free_workers must be an integer >= 0")
        _money(self.extra_worker_cost, "extra_worker_cost")
        _prob(self.contacts_fraction, "contacts_fraction")
        for name in ("contact_strength", "transmission", "recovery", "word_of_mouth_strength"):
            _prob(getattr(self, name), name)
        _check(isinstance(self.hops, int) and self.hops >= 0, "hops must be an integer >= 0")

    def contacts_per_worker(self, num_voters: int) -> int:
        return max(1, round(self.contacts_fraction * num_voters))


@dataclass(frozen=True, slots=True)
class MediaConfig:
    base_cost: Decimal = Decimal("50.00")                 # m0 §3.2 (v2.1 calibrated)
    exposure_prob: float = 0.6                            # ⚙ p_m
    strength: float = 0.25                                # η_m0 (v2.1 calibrated)
    decay: float = 0.5                                    # ⚙ λ

    def __post_init__(self) -> None:
        _money(self.base_cost, "base_cost")
        _prob(self.exposure_prob, "exposure_prob")
        _prob(self.strength, "strength")
        _check(_finite(self.decay, "decay") >= 0, "decay must be >= 0")


@dataclass(frozen=True, slots=True)
class VotingConfig:
    sensitivity: float = 20.0                             # β_v §5 (v2.1 calibrated)
    threshold: float = 0.825                              # z (v2.1 calibrated)

    def __post_init__(self) -> None:
        _check(_finite(self.sensitivity, "sensitivity") > 0, "sensitivity must be > 0")
        _prob(self.threshold, "threshold")


@dataclass(frozen=True, slots=True)
class SurveyConfig:
    query_cost: Decimal = Decimal("5.00")                 # U8
    poll_size: int = 400                                  # ⚙ §6.4
    consistency_tolerance: float = 0.02                   # ⚙ ε §10.1 (fraction, not %)

    def __post_init__(self) -> None:
        _money(self.query_cost, "query_cost")
        _check(isinstance(self.poll_size, int) and self.poll_size >= 1, "poll_size must be an integer >= 1")
        _prob(self.consistency_tolerance, "consistency_tolerance")


@dataclass(frozen=True, slots=True)
class GameConfig:
    seed: int = 20261107
    rounds: int = 5                                       # U7: 1..10
    budget: Decimal = Decimal("500.00")                   # ⚙ U14 (owner: start at 500)
    device: str = "auto"
    dtype: Literal["float32", "float64"] = "float32"
    # R8: "cpu" draws the electorate from CPU streams so (num_voters, seed) recreates the identical
    # initial electorate on every machine/device; "device" draws on the GPU (fast, for batch experiments).
    generation: Literal["cpu", "device"] = "cpu"
    electorate: ElectorateConfig = field(default_factory=ElectorateConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    workers: WorkerConfig = field(default_factory=WorkerConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    voting: VotingConfig = field(default_factory=VotingConfig)
    survey: SurveyConfig = field(default_factory=SurveyConfig)

    def __post_init__(self) -> None:
        _check(isinstance(self.seed, int) and not isinstance(self.seed, bool), "seed must be an integer")
        _check(isinstance(self.rounds, int) and not isinstance(self.rounds, bool) and 1 <= self.rounds <= 10, "rounds must be an integer in [1, 10]")
        _money(self.budget, "budget")
        _check(self.dtype in ("float32", "float64"), "dtype must be 'float32' or 'float64'")
        _check(self.generation in ("cpu", "device"), "generation must be 'cpu' or 'device'")
        _check(isinstance(self.device, str) and (self.device in ("auto", "cpu") or self.device.startswith("cuda")), "device must be 'auto', 'cpu' or 'cuda[:N]'")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float32 if self.dtype == "float32" else torch.float64

    def seeds(self) -> "SeedStreams":
        return SeedStreams(self.seed)

    def to_record(self) -> dict:
        """JSON-safe dict for result provenance (Decimals as strings)."""
        def convert(value):
            if isinstance(value, Decimal):
                return str(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value
        return convert(asdict(self))


@dataclass(frozen=True, slots=True)
class SeedStreams:
    """Root seed -> independent named streams; sub-keys (round, player, ...) give per-use seeds."""

    root: int

    def __post_init__(self) -> None:
        _check(isinstance(self.root, int) and not isinstance(self.root, bool), "root seed must be an integer")

    def derive(self, stream: str, *keys: object) -> int:
        _check(stream in SEED_STREAMS, f"unknown seed stream {stream!r}")
        material = ":".join(["dumbocracy-v2", str(self.root), stream, *map(str, keys)])
        digest = hashlib.blake2b(material.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") & ((1 << 63) - 1)

    def generator(self, stream: str, *keys: object, device: torch.device | str = "cpu") -> torch.Generator:
        gen = torch.Generator(device=torch.device(device))
        gen.manual_seed(self.derive(stream, *keys))
        return gen


def resolve_device(requested: str = "auto") -> torch.device:
    """``auto`` picks the CUDA device with the most free memory, else CPU (VRAM is per device)."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda"):
        _check(torch.cuda.is_available(), "CUDA requested but unavailable")
        device = torch.device(requested)
        index = 0 if device.index is None else device.index
        _check(index < torch.cuda.device_count(), f"CUDA device {index} not present")
        return torch.device("cuda", index)
    _check(requested == "auto", "device must be 'auto', 'cpu' or 'cuda[:N]'")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = cuda_free_memory()
    best = max(free, key=free.get)
    return torch.device("cuda", best)


def cuda_free_memory() -> dict[int, int]:
    """Free bytes per CUDA index without creating contexts on busy devices (shared server).

    Uses ``nvidia-smi`` matched by GPU UUID (NVML and CUDA orderings can differ); falls back to a
    guarded ``torch.cuda.mem_get_info`` probe, treating a device that cannot be probed as full.
    """
    import subprocess

    count = torch.cuda.device_count()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        by_uuid = {}
        for line in out.strip().splitlines():
            uuid, mib = (part.strip() for part in line.split(","))
            by_uuid[uuid.removeprefix("GPU-")] = int(mib) * 2**20
        result = {}
        for index in range(count):
            uuid = str(torch.cuda.get_device_properties(index).uuid)
            result[index] = by_uuid[uuid]
        return result
    except Exception:
        result = {}
        for index in range(count):
            try:
                result[index] = torch.cuda.mem_get_info(index)[0]
            except RuntimeError:
                result[index] = 0
        return result
