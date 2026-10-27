"""Optional local-only FastAPI adapter; responses contain public allowlists only."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .campaign import CampaignAction
from .config import ActionConfig, DynamicsConfig, GameConfig, GraphSpec, SeedConfig
from .engine import GameEngine, GameState
from .verification import verify_episode


@dataclass
class Session:
    config: GameConfig
    backend: str
    state: GameState
    engine: GameEngine
    observations: list[dict] = field(default_factory=list)


class CreateGame(BaseModel):
    node_count: int = Field(default=1000, ge=1, le=100000)
    horizon_rounds: int = Field(default=5, ge=1, le=100)
    initial_budget: str = "20.00"
    seed: int = 20261019
    backend: str = "auto"
    edge_probability: float | None = None


class ActionRequest(BaseModel):
    action_id: str
    target_ids: list[str] = Field(default_factory=list)


app = FastAPI(title="Dumbocracy local research beta", version="0.1.0")
SESSIONS: dict[str, Session] = {}
MAX_SESSIONS = 8
WEB_ROOT = Path(__file__).resolve().parents[2] / "web" / "dist"


def _public(session: Session) -> dict:
    return {"observation": session.engine.observe(session.state).to_public_dict(),
            "backend": session.state.backend, "timeline": list(session.observations)}


def _public_result(state: GameState):
    if state.result is None:
        return None
    result = state.result.to_public_dict()
    metadata = result.get("metadata", {})
    if isinstance(metadata, dict):
        result["metadata"] = {key: value for key, value in metadata.items()
                              if not key.endswith("_seed") and key != "root_seed"}
    return result


@app.get("/api/health")
def health():
    return {"status": "ok", "api_version": "1.0", "session_count": len(SESSIONS), "session_limit": MAX_SESSIONS}


@app.post("/api/games")
def create_game(request: CreateGame):
    if len(SESSIONS) >= MAX_SESSIONS:
        raise HTTPException(429, "local session limit reached")
    try:
        budget = Decimal(request.initial_budget)
        probability = request.edge_probability if request.edge_probability is not None else min(1.0, 8 / max(1, request.node_count - 1))
        config = GameConfig(SeedConfig(request.seed), GraphSpec("erdos_renyi", request.node_count, probability),
                            request.horizon_rounds, budget, DynamicsConfig(0.5),
                            ActionConfig(0.05, 0.02))
        backend = request.backend
        if backend == "auto":
            try:
                import torch
                backend = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                backend = "cpu"
        engine = GameEngine(backend)
        state = engine.initialize(config)
    except (ValueError, ArithmeticError) as exc:
        raise HTTPException(422, "invalid game configuration") from exc
    except Exception as exc:
        raise HTTPException(422, "game could not be initialized") from exc
    game_id = uuid4().hex
    session = Session(config, backend, state, engine)
    session.observations.append(engine.observe(state).to_public_dict())
    SESSIONS[game_id] = session
    return {"game_id": game_id, **_public(session)}


def _get(game_id: str) -> Session:
    session = SESSIONS.get(game_id)
    if session is None:
        raise HTTPException(404, "unknown game")
    return session


@app.get("/api/games/{game_id}/observation")
def observation(game_id: str):
    return _public(_get(game_id))


@app.post("/api/games/{game_id}/actions")
def action(game_id: str, request: ActionRequest):
    session = _get(game_id)
    response = session.engine.step(session.state, CampaignAction(request.action_id, tuple(request.target_ids)))
    session.state = response.state
    if response.accepted:
        session.observations.append(response.observation.to_public_dict())
    return {"accepted": response.accepted, "error_code": response.error_code,
            "terminal": response.terminal, **_public(session),
            "result": _public_result(response.state) if response.accepted and response.state.result else None}


@app.post("/api/games/{game_id}/reset")
def reset(game_id: str):
    session = _get(game_id)
    session.state = session.engine.initialize(session.config)
    session.observations = [session.engine.observe(session.state).to_public_dict()]
    return _public(session)


@app.get("/api/games/{game_id}/result")
def result(game_id: str):
    state = _get(game_id).state
    return _public_result(state)


@app.get("/api/games/{game_id}/analysis")
def analysis(game_id: str):
    state = _get(game_id).state
    if not state.terminal:
        return {"evaluation_mode": "trace_only", "status": "PENDING", "message": "Available after the fixed horizon."}
    verification = verify_episode(state, _get(game_id).engine)
    diagnostics = verification["trace"]["diagnostics"]
    return {"evaluation_mode": "trace_only", "scope": "completed episode only; not a future-state guarantee",
            "diagnostics": diagnostics, "verification": verification}


if WEB_ROOT.is_dir():
    app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
