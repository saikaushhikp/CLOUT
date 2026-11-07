"""FastAPI engine server implementing ``docs/v2/web_api_v2.md`` (the binding HTTP/JSON contract).

Run: ``uvicorn dumbocracy.api.app:app --host 127.0.0.1 --port 61008``. Games live in memory
(``max_games`` sessions, the oldest idle one is evicted). Route handlers are synchronous, so
FastAPI runs them in its threadpool; every engine call on a game holds that game's lock.
Device: ``DUMBOCRACY_DEVICE`` (default ``auto`` = freest CUDA device, else CPU), resolved lazily.
"""
from __future__ import annotations

import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
from fastapi import Body, FastAPI, HTTPException, Query as QueryParam
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from ..config import ElectorateConfig, GameConfig, resolve_device
from ..game import Allocate, Commit, Forfeit, GameEngine, GameState, Query, observation_for, public_result
from ..sectors import ARCHETYPES, NUM_REGIONS, PERSONALITIES, REGIONS
from ..strategies.oracle import anticipate, oracle_manifesto
from ..voting import similarity, vote_probability

DEFAULT_MAX_GAMES = 8
MAX_GRAPH_NODES = 10_000
ORACLE_NOTE = "ORACLE: full-information benchmark"


# ------------------------------------------------------------------------------------ bodies

class NewGameBody(BaseModel):
    num_voters: int = Field(10_000, ge=1_000, le=1_000_000)
    seed: int | None = Field(None, ge=0, lt=2**63)
    rounds: int = Field(5, ge=1, le=10)


class QueryBody(BaseModel):
    sector_a: str
    sector_b: str
    region: str


class CommitBody(BaseModel):
    manifesto: list[float]


class AllocateBody(BaseModel):
    model_config = ConfigDict(strict=False)
    workers_by_region: list[StrictInt]
    extra_workers: StrictInt = 0
    media_regions: list[StrictBool] = Field(default_factory=lambda: [False] * NUM_REGIONS)


# ---------------------------------------------------------------------------------- sessions

@dataclass(eq=False)
class Session:
    state: GameState
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.monotonic)
    cache: dict[Any, Any] = field(default_factory=dict)   # graph topology per max_nodes, debrief


class GameStore:
    """In-memory sessions capped at ``max_games``; creating one more evicts the oldest idle game."""

    def __init__(self, max_games: int) -> None:
        self.max_games = max_games
        self._games: dict[str, Session] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._games)

    def add(self, session: Session) -> None:
        with self._lock:
            while len(self._games) >= self.max_games:
                idle = [(s.last_used, gid) for gid, s in self._games.items() if not s.lock.locked()]
                if not idle:
                    raise HTTPException(503, "server busy: all game slots are in use")
                del self._games[min(idle)[1]]
            self._games[session.state.game_id] = session

    def get(self, game_id: str) -> Session:
        with self._lock:
            session = self._games.get(game_id)
        if session is None:
            raise HTTPException(404, f"unknown game {game_id!r}")
        session.last_used = time.monotonic()
        return session

    def delete(self, game_id: str) -> None:
        with self._lock:
            if self._games.pop(game_id, None) is None:
                raise HTTPException(404, f"unknown game {game_id!r}")


# --------------------------------------------------------------------------- reveal helpers

def sample_graph_nodes(state: GameState, max_nodes: int) -> torch.Tensor:
    """Stratified-by-region uniform sample of ≤ max_nodes voter ids (CPU int64, sorted).

    Quotas are proportional to region sizes (largest remainder); draws use the ``strategy`` stream
    keyed ``"graph_sample"``, so the sample is deterministic per game and ``max_nodes``.
    """
    sizes = state.electorate.sizes.to(torch.int64)
    total = int(sizes.sum())
    k = min(max_nodes, total)
    raw = sizes.to(torch.float64) * k / total
    quota = raw.floor().to(torch.int64)
    remainder = k - int(quota.sum())
    order = torch.argsort(-(raw - quota), stable=True)
    quota[order[:remainder]] += 1
    gen = state.config.seeds().generator("strategy", "graph_sample")
    members = torch.argsort(state.electorate.region_of.cpu(), stable=True)
    offsets = torch.cumsum(sizes, 0) - sizes
    picks = [members[int(offsets[r]) + torch.randperm(int(sizes[r]), generator=gen)[: int(quota[r])]]
             for r in range(NUM_REGIONS)]
    return torch.sort(torch.cat(picks)).values


def graph_topology(state: GameState, max_nodes: int) -> dict[str, Any]:
    """Public part of ``GraphSample``: sampled nodes (region, full-graph degree) + induced edges."""
    e = state.electorate
    ids = sample_graph_nodes(state, max_nodes)
    position = torch.full((e.num_voters,), -1, dtype=torch.int64, device=e.device)
    position[ids.to(e.device)] = torch.arange(ids.numel(), device=e.device)
    a, b = position[e.graph.src], position[e.graph.dst]
    keep = (a >= 0) & (b >= 0)
    edges = torch.stack([a[keep], b[keep]], 1).cpu().tolist()
    regions = e.region_of[ids.to(e.device)].cpu().tolist()
    degree = e.graph.degree[ids.to(e.device)].cpu().tolist()
    nodes = [{"id": int(i), "region": REGIONS[r], "degree": int(d)} for i, r, d in zip(ids.tolist(), regions, degree)]
    return {"ids": ids, "nodes": nodes, "edges": edges}


def _region_shares(state: GameState, v: torch.Tensor, b: torch.Tensor) -> list[float]:
    voting = state.config.voting
    prob = vote_probability(similarity(v, b), voting.sensitivity, voting.threshold).to(torch.float64)
    sums = torch.zeros(NUM_REGIONS, dtype=torch.float64, device=prob.device).index_add_(0, state.electorate.region_of, prob)
    return (sums.cpu() / state.electorate.sizes.to(torch.float64)).tolist()


def _share(state: GameState, v: torch.Tensor, b: torch.Tensor) -> float:
    voting = state.config.voting
    return float(vote_probability(similarity(v, b), voting.sensitivity, voting.threshold).mean())


def build_debrief(state: GameState, player_id: str) -> dict[str, Any]:
    """Contract ``Debrief`` (DONE only): hidden truth, trajectory and the labelled ORACLE benchmark."""
    e, cfg = state.electorate, state.config
    player = state.player(player_id)
    b = torch.tensor(player.manifesto, dtype=state.preferences.dtype, device=state.device)
    initial, final = _region_shares(state, e.preferences, b), _region_shares(state, state.preferences, b)
    zealots = torch.bincount(e.region_of[e.susceptibility.zealot.to(torch.bool)], minlength=NUM_REGIONS).cpu().tolist()
    anticipated = anticipate(e, cfg)
    oracle_b = oracle_manifesto(e, cfg)
    regions = [{"id": REGIONS[r], "archetype": ARCHETYPES[int(e.archetype_of_region[r])].name,
                "personality": PERSONALITIES[int(e.personality_of_region[r])].name, "zealots": int(zealots[r]),
                "final_expected_share": final[r], "initial_expected_share": initial[r]} for r in range(NUM_REGIONS)]
    return {
        "result": public_result(player.result),
        "regions": regions,
        "share_trajectory": list(player.share_trajectory),
        "oracle": {"manifesto": [float(x) for x in oracle_b.cpu().tolist()],
                   "expected_share_no_campaign": _share(state, anticipated, oracle_b), "note": ORACLE_NOTE},
        "your_manifesto_no_campaign_share": _share(state, anticipated, b),
    }


# --------------------------------------------------------------------------------------- app

def create_app(device: str | None = None, max_games: int = DEFAULT_MAX_GAMES) -> FastAPI:
    """Build the API. ``device`` defaults to ``$DUMBOCRACY_DEVICE`` or ``auto`` (resolved lazily)."""
    app = FastAPI(title="Dumbocracy v2 engine", version="2.0")
    store = GameStore(max_games)
    requested = device or os.environ.get("DUMBOCRACY_DEVICE", "auto")
    resolved: dict[str, torch.device] = {}
    device_lock = threading.Lock()

    def engine_device() -> torch.device:
        with device_lock:
            if "device" not in resolved:
                resolved["device"] = resolve_device(requested)
            return resolved["device"]

    def session_player(game_id: str, player_id: str) -> Session:
        session = store.get(game_id)
        if not session.state.has_player(player_id):
            raise HTTPException(404, f"unknown player {player_id!r}")
        return session

    def act(game_id: str, player_id: str, action) -> dict[str, Any]:
        session = session_player(game_id, player_id)
        with session.lock:
            step = GameEngine.apply(session.state, player_id, action)
            session.state = step.state
            response: dict[str, Any] = {"accepted": step.accepted, "error_code": step.error_code,
                                        "message": step.message, "observation": observation_for(step.state, player_id)}
        if step.answer is not None:
            response["answer"] = step.answer
        if step.round_report is not None:
            response["round_report"] = step.round_report
        return response

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "device": str(engine_device()), "active_games": len(store), "max_games": store.max_games}

    @app.post("/api/games")
    def new_game(body: NewGameBody | None = Body(None)) -> dict[str, Any]:
        body = body or NewGameBody()
        seed = secrets.randbits(31) if body.seed is None else body.seed
        target = engine_device()
        config = GameConfig(seed=seed, rounds=body.rounds, device=str(target),
                            electorate=ElectorateConfig(num_voters=body.num_voters))
        state = GameEngine.new_game(config, ("p0",), game_id=uuid.uuid4().hex[:12], device=target)
        store.add(Session(state))
        return {"game_id": state.game_id, "player_id": "p0", "observation": observation_for(state, "p0")}

    @app.get("/api/games/{game_id}/players/{player_id}/observation")
    def observation(game_id: str, player_id: str) -> dict[str, Any]:
        session = session_player(game_id, player_id)
        with session.lock:
            return observation_for(session.state, player_id)

    @app.post("/api/games/{game_id}/players/{player_id}/query")
    def query(game_id: str, player_id: str, body: QueryBody) -> dict[str, Any]:
        return act(game_id, player_id, Query(body.sector_a, body.sector_b, body.region))

    @app.post("/api/games/{game_id}/players/{player_id}/commit")
    def commit(game_id: str, player_id: str, body: CommitBody) -> dict[str, Any]:
        return act(game_id, player_id, Commit(tuple(body.manifesto)))

    @app.post("/api/games/{game_id}/players/{player_id}/allocate")
    def allocate(game_id: str, player_id: str, body: AllocateBody) -> dict[str, Any]:
        return act(game_id, player_id, Allocate(tuple(body.workers_by_region), body.extra_workers, tuple(body.media_regions)))

    @app.post("/api/games/{game_id}/players/{player_id}/forfeit")
    def forfeit(game_id: str, player_id: str) -> dict[str, Any]:
        return act(game_id, player_id, Forfeit())

    @app.get("/api/games/{game_id}/graph_sample")
    def graph_sample(game_id: str, max_nodes: int = QueryParam(1500, ge=1, le=MAX_GRAPH_NODES),
                     player_id: str | None = None) -> dict[str, Any]:
        session = store.get(game_id)
        with session.lock:
            state = session.state
            pid = player_id or state.player_ids[0]
            if not state.has_player(pid):
                raise HTTPException(404, f"unknown player {pid!r}")
            topology = session.cache.get(("graph", max_nodes))
            if topology is None:
                topology = session.cache[("graph", max_nodes)] = graph_topology(state, max_nodes)
            revealed = state.phase == "DONE"
            alignment = [snap[topology["ids"]].tolist() for snap in state.player(pid).alignment] if revealed else None
        return {"revealed": revealed, "nodes": topology["nodes"], "edges": topology["edges"], "alignment": alignment}

    @app.get("/api/games/{game_id}/players/{player_id}/debrief")
    def debrief(game_id: str, player_id: str) -> dict[str, Any]:
        session = session_player(game_id, player_id)
        with session.lock:
            if session.state.phase != "DONE":
                raise HTTPException(409, "the debrief is available only when the game is DONE")
            key = ("debrief", player_id)
            if key not in session.cache:
                session.cache[key] = build_debrief(session.state, player_id)
            return session.cache[key]

    @app.delete("/api/games/{game_id}")
    def delete_game(game_id: str) -> dict[str, Any]:
        store.delete(game_id)
        return {"deleted": True}

    app.state.store = store
    return app


app = create_app()
