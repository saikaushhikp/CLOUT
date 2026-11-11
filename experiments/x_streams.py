"""Salted seed streams for the v3 experiments (plan §2; used by X20 "luck vs skill").

``SaltedStreams(root, salts)`` has the same ``derive(stream, *keys)`` / ``generator(stream, *keys, device=)``
interface as :class:`dumbocracy.config.SeedStreams`. A non-zero salt for one stream re-draws only that stream;
every other stream stays bit-identical. With all salts 0 (or none) it equals ``SeedStreams(root)`` exactly,
so it can be passed to ``simulate_world(..., seeds=...)`` or ``x_batch.run_batch(..., streams=...)`` without
any library change.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

import torch

from dumbocracy.config import SEED_STREAMS, SeedStreams


@dataclass(frozen=True)
class SaltedStreams:
    root: int
    salts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.salts) - set(SEED_STREAMS)
        if unknown:
            raise ValueError(f"unknown seed streams in salts: {sorted(unknown)}")
        object.__setattr__(self, "salts", {k: int(v) for k, v in self.salts.items()})
        object.__setattr__(self, "_base", SeedStreams(int(self.root)))

    def derive(self, stream: str, *keys: object) -> int:
        salt = self.salts.get(stream, 0)
        base = self._base.derive(stream, *keys)  # validates the stream name
        if salt == 0:
            return base
        material = f"dumbocracy-v3-salt:{base}:{stream}:{salt}"
        digest = hashlib.blake2b(material.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") & ((1 << 63) - 1)

    def generator(self, stream: str, *keys: object, device: torch.device | str = "cpu") -> torch.Generator:
        gen = torch.Generator(device=torch.device(device))
        gen.manual_seed(self.derive(stream, *keys))
        return gen

    def with_salt(self, stream: str, salt: int) -> "SaltedStreams":
        return SaltedStreams(self.root, {**self.salts, stream: int(salt)})
