"""Paired-seed baseline evaluation with safe JSONL records and aggregates."""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import importlib.metadata
import json
import math
import platform
import random
import statistics
import subprocess
import time
from typing import Callable, Mapping, Sequence

from .campaign import CampaignAction
from .config import GameConfig, SeedConfig
from .engine import GameEngine, GameState
from .graph import ElectorateGraph
from .logic import CanonicalTrace, evaluate_diagnostics
from .strategies import HistoryEntry, PublicActionSummary, Strategy


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    records: tuple[dict[str, object], ...]
    aggregates: tuple[dict[str, object], ...]

    def to_jsonl(self) -> str:
        return "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in self.records)

    def to_public_dict(self) -> dict[str, object]:
        return {"schema_version": "1.0", "aggregates": list(self.aggregates)}


@dataclass(frozen=True, slots=True)
class ExperimentScenario:
    scenario_id: str
    config: GameConfig
    graph: ElectorateGraph | None = None


def _revision_or_uncommitted(revision: str) -> str:
    return revision or "uncommitted"


def _runtime_versions(backend: str) -> dict[str, str | None]:
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = "not-installed"
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "not-installed"
    cuda_runtime = None
    driver = None
    device_name = "CPU"
    if backend.startswith("cuda"):
        import torch
        cuda_runtime = str(torch.version.cuda)
        device_name = torch.cuda.get_device_name(torch.device(backend))
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError, IndexError):
            driver = "unavailable"
    return {
        "python": platform.python_version(),
        "numpy": numpy_version,
        "torch": torch_version,
        "cuda_runtime": cuda_runtime,
        "driver_version": driver,
        "backend": backend,
        "device": device_name,
    }


def _synchronize(backend: str) -> None:
    if backend.startswith("cuda"):
        import torch
        torch.cuda.synchronize(torch.device(backend))


def _safe_action_id(action: object) -> str:
    action_id = getattr(action, "action_id", None)
    return action_id if action_id in ("targeted_contact", "broadcast", "no_op") else "unknown"


def _safe_target_ids(action: object, legal_actions: tuple[object, ...]) -> list[str] | None:
    if not isinstance(action, CampaignAction) or _safe_action_id(action) != "targeted_contact":
        return None
    proposed = getattr(action, "target_ids", None)
    eligible = {
        target
        for descriptor in legal_actions
        if getattr(descriptor, "action_id", None) == "targeted_contact"
        for target in getattr(descriptor, "eligible_target_ids", ())
        if isinstance(target, str)
    }
    if not isinstance(proposed, tuple) or any(not isinstance(target, str) or target not in eligible for target in proposed):
        return None
    return list(proposed)


def _graph_record(state: GameState) -> dict[str, object]:
    summary = state.graph_summary
    return {
        "graph_kind": state.config.graph.kind,
        "node_count": summary.node_count,
        "edge_count": summary.edge_count,
        "density": summary.density,
        "min_degree": summary.min_degree,
        "max_degree": summary.max_degree,
        "mean_degree": summary.mean_degree,
        "component_count": summary.component_count,
        "edge_probability": state.config.graph.edge_probability,
        "graph_seed": state.config.seeds.graph,
    }


def _base_record(state: GameState, strategy_name: str, strategy_version: str, scenario_id: str, backend: str, revision: str) -> dict[str, object]:
    config = state.config
    return {
        "schema_version": "1.0",
        "scenario_id": scenario_id,
        "strategy": strategy_name,
        "strategy_version": strategy_version,
        "root_seed": config.seeds.root_seed,
        "derived_seeds": {
            "graph": config.seeds.graph,
            "initial_state": config.seeds.initial_state,
            "transition": config.seeds.transition,
            "strategy": config.seeds.strategy,
        },
        "graph": _graph_record(state),
        "game": {
            "horizon_rounds": config.horizon_rounds,
            "initial_budget": str(config.initial_budget),
            "alpha": config.dynamics.alpha,
            "targeted_cost": str(config.actions.targeted_cost),
            "broadcast_cost": str(config.actions.broadcast_cost),
            "delta_target": config.actions.delta_target,
            "delta_broadcast": config.actions.delta_broadcast,
            "support_threshold": config.support_threshold,
        },
        "runtime": {**_runtime_versions(backend), "revision": _revision_or_uncommitted(revision)},
    }


def _episode_record(
    *, config: GameConfig, graph: ElectorateGraph | None, scenario_id: str,
    strategy_factory: Callable[[], Strategy], backend: str, revision: str,
) -> dict[str, object]:
    started = time.perf_counter()
    engine = GameEngine(backend, revision=_revision_or_uncommitted(revision))
    state = engine.initialize(config, graph=graph)
    _synchronize(backend)
    initialized = time.perf_counter()
    policy = strategy_factory()  # no policy instance/RNG/history crosses episodes
    strategy_name = str(getattr(policy, "name", type(policy).__name__))
    strategy_version = str(getattr(policy, "version", "unknown"))
    record = _base_record(state, strategy_name, strategy_version, scenario_id, backend, revision)
    rng = random.Random(config.seeds.strategy)
    history: tuple[HistoryEntry, ...] = ()
    proposals: list[dict[str, object]] = []
    selection_seconds = 0.0
    step_seconds = 0.0
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_code: str | None = None

    while not state.terminal:
        observation = engine.observe(state)
        legal = engine.legal_actions(state)
        before_selection = time.perf_counter()
        try:
            action = policy.choose(observation, history, legal, rng)
        except Exception as exc:
            selection_seconds += time.perf_counter() - before_selection
            failure_stage, failure_type = "strategy_selection", type(exc).__name__
            break
        selection_seconds += time.perf_counter() - before_selection
        proposal_record: dict[str, object] = {
            "round_index": state.round_index,
            "action_id": _safe_action_id(action),
            "target_ids": _safe_target_ids(action, legal),
            "accepted": None,
            "error_code": None,
        }
        proposals.append(proposal_record)
        before_step = time.perf_counter()
        try:
            response = engine.step(state, action)  # engine remains authoritative
            _synchronize(backend)
        except Exception as exc:
            step_seconds += time.perf_counter() - before_step
            proposal_record["error_code"] = "ENGINE_EXCEPTION"
            failure_stage, failure_type = "engine_step", type(exc).__name__
            break
        step_seconds += time.perf_counter() - before_step
        action_id = str(proposal_record["action_id"])
        proposal_record["accepted"] = response.accepted
        proposal_record["error_code"] = response.error_code
        if not response.accepted:
            failure_stage, failure_type, failure_code = "policy_action_rejected", "RejectedProposal", response.error_code
            break
        history += (HistoryEntry(observation, PublicActionSummary(state.round_index, action_id, True, None)),)
        state = response.state

    finished = time.perf_counter()
    record.update({
        "status": "completed" if failure_stage is None and state.terminal else "failed",
        "action_sequence": proposals,
        "accepted_count": sum(item["accepted"] is True for item in proposals),
        "rejected_count": sum(item["accepted"] is False for item in proposals),
        "failed_count": 0 if failure_stage is None and state.terminal else 1,
        "skipped_count": 0,
        "timing_seconds": {
            "initialization": initialized - started,
            "strategy_selection_total": selection_seconds,
            "engine_step_total": step_seconds,
            "episode_total": finished - started,
        },
    })
    if failure_stage is not None or not state.terminal:
        record.update({"failure_stage": failure_stage or "incomplete_episode", "failure_type": failure_type or "IncompleteEpisode"})
        if failure_code is not None:
            record["error_code"] = failure_code
        record["terminal_result"] = None
        record["logic_diagnostics"] = None
        return record

    result = state.result
    record["terminal_result"] = result.to_public_dict() if result else None
    record["safe_trace_summary"] = {
        "state_count": state.config.horizon_rounds + 1,
        "accepted_transitions": state.round_index,
        "audit_event_count": len(state.internal_events),
    }
    try:
        diagnostics = evaluate_diagnostics(CanonicalTrace.from_game_state(state))
        record["logic_diagnostics"] = {name: item.to_public_dict() for name, item in diagnostics.items()}
    except Exception as exc:
        record.update({
            "status": "failed",
            "failed_count": 1,
            "failure_stage": "logic_diagnostics",
            "failure_type": type(exc).__name__,
            "logic_diagnostics": None,
        })
    return record


def aggregate_records(records: tuple[dict[str, object], ...] | list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        key = (str(record.get("strategy", "unknown")), str(record.get("strategy_version", "unknown")))
        grouped.setdefault(key, []).append(record)
    aggregates: list[dict[str, object]] = []
    for (name, version), rows in sorted(grouped.items()):
        completed = [row for row in rows if row.get("status") == "completed"]
        failed_count = sum(int(row.get("status") == "failed") for row in rows)
        skipped_count = sum(int(row.get("status") == "skipped") for row in rows)
        scores = [float(row["terminal_result"]["score"]) for row in completed]  # type: ignore[index]
        victories = [bool(row["terminal_result"]["victory"]) for row in completed]  # type: ignore[index]
        budgets = [float(Decimal(str(row["terminal_result"]["budget_used"]))) for row in completed]  # type: ignore[index]
        rejects = sum(int(row.get("rejected_count", 0)) for row in rows)
        proposals = sum(len(row.get("action_sequence", [])) for row in rows)  # type: ignore[arg-type]
        mean_score = statistics.fmean(scores) if scores else None
        aggregates.append({
            "strategy": name,
            "strategy_version": version,
            "sample_count": len(rows),
            "completed_count": len(completed),
            "failed_episode_count": failed_count,
            "skipped_episode_count": skipped_count,
            "mean_terminal_score": mean_score,
            "terminal_score_population_stddev": math.sqrt(statistics.fmean((value - mean_score) ** 2 for value in scores)) if scores else None,
            "victory_rate_completed": statistics.fmean(victories) if victories else None,
            "mean_budget_used_completed": statistics.fmean(budgets) if budgets else None,
            "rejected_proposal_count": rejects,
            "rejected_proposal_rate": rejects / proposals if proposals else None,
            "denominators": {
                "score_victory_budget": "completed episodes only",
                "rejected_proposal_rate": "all recorded proposals, including failed episodes",
            },
        })
    return tuple(aggregates)


def run_paired_scenarios(
    scenarios: Sequence[ExperimentScenario],
    root_seeds: tuple[int, ...] | list[int],
    strategy_factories: Mapping[str, Callable[[], Strategy]],
    *,
    backend: str = "cpu",
    revision: str = "uncommitted",
) -> ExperimentReport:
    """Run every policy on identical scenario/root-seed pairs and fresh state."""
    if not scenarios:
        raise ValueError("at least one scenario is required")
    if not root_seeds:
        raise ValueError("root_seeds must contain at least one declared seed")
    if not strategy_factories:
        raise ValueError("at least one strategy factory is required")
    if any(not isinstance(name, str) or not name or not callable(factory) for name, factory in strategy_factories.items()):
        raise ValueError("strategy factories need nonempty string keys and callable values")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in root_seeds):
        raise ValueError("root seeds must be integers")
    if len(set(root_seeds)) != len(root_seeds):
        raise ValueError("root seeds must be unique")
    if backend != "cpu" and not backend.startswith("cuda"):
        raise ValueError("backend must be cpu or cuda:N")
    if any(not isinstance(scenario, ExperimentScenario) or not isinstance(scenario.config, GameConfig) for scenario in scenarios):
        raise ValueError("scenarios must contain ExperimentScenario values with validated GameConfig")
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    if any(not isinstance(scenario_id, str) or not scenario_id for scenario_id in scenario_ids):
        raise ValueError("scenario IDs must be nonempty strings")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario IDs must be unique")

    rows: list[dict[str, object]] = []
    for root_seed in root_seeds:
        for scenario in scenarios:
            paired_config = replace(scenario.config, seeds=SeedConfig(root_seed))
            for factory_key, factory in sorted(strategy_factories.items()):
                # A failed setup still produces an explicit row for this pair.
                try:
                    row = _episode_record(
                        config=paired_config,
                        graph=scenario.graph,
                        scenario_id=scenario.scenario_id,
                        strategy_factory=factory,
                        backend=backend,
                        revision=revision,
                    )
                except Exception as exc:
                    try:
                        numpy_version = importlib.metadata.version("numpy")
                    except importlib.metadata.PackageNotFoundError:
                        numpy_version = "not-installed"
                    try:
                        torch_version = importlib.metadata.version("torch")
                    except importlib.metadata.PackageNotFoundError:
                        torch_version = "not-installed"
                    row = {
                        "schema_version": "1.0",
                        "scenario_id": scenario.scenario_id,
                        "strategy": factory_key,
                        "strategy_version": "unknown",
                        "paired_run_id": f"{scenario.scenario_id}:{root_seed}",
                        "root_seed": root_seed,
                        "derived_seeds": {
                            "graph": paired_config.seeds.graph,
                            "initial_state": paired_config.seeds.initial_state,
                            "transition": paired_config.seeds.transition,
                            "strategy": paired_config.seeds.strategy,
                        },
                        "graph": {
                            "graph_kind": paired_config.graph.kind,
                            "node_count": paired_config.graph.node_count,
                            "edge_probability": paired_config.graph.edge_probability,
                            "graph_seed": paired_config.seeds.graph,
                        },
                        "game": {
                            "horizon_rounds": paired_config.horizon_rounds,
                            "initial_budget": str(paired_config.initial_budget),
                            "alpha": paired_config.dynamics.alpha,
                            "targeted_cost": str(paired_config.actions.targeted_cost),
                            "broadcast_cost": str(paired_config.actions.broadcast_cost),
                            "delta_target": paired_config.actions.delta_target,
                            "delta_broadcast": paired_config.actions.delta_broadcast,
                            "support_threshold": paired_config.support_threshold,
                        },
                        "runtime": {
                            "python": platform.python_version(),
                            "numpy": numpy_version,
                            "torch": torch_version,
                            "cuda_runtime": None,
                            "driver_version": "unavailable",
                            "backend": backend,
                            "device": "unavailable",
                            "revision": _revision_or_uncommitted(revision),
                        },
                        "status": "failed",
                        "failed_count": 1,
                        "skipped_count": 0,
                        "failure_stage": "episode_setup",
                        "failure_type": type(exc).__name__,
                        "action_sequence": [],
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "timing_seconds": {
                            "initialization": None,
                            "strategy_selection_total": None,
                            "engine_step_total": None,
                            "episode_total": None,
                        },
                        "terminal_result": None,
                        "logic_diagnostics": None,
                        "safe_trace_summary": None,
                    }
                row["paired_run_id"] = f"{scenario.scenario_id}:{root_seed}"
                rows.append(row)
    immutable_rows = tuple(rows)
    return ExperimentReport(immutable_rows, aggregate_records(immutable_rows))


def run_paired_experiments(
    config: GameConfig,
    root_seeds: tuple[int, ...] | list[int],
    strategy_factories: Mapping[str, Callable[[], Strategy]],
    *,
    graph: ElectorateGraph | None = None,
    scenario_id: str = "default",
    backend: str = "cpu",
    revision: str = "uncommitted",
) -> ExperimentReport:
    """Convenience wrapper for one scenario and a declared root-seed set."""
    return run_paired_scenarios(
        (ExperimentScenario(scenario_id, config, graph),),
        root_seeds,
        strategy_factories,
        backend=backend,
        revision=revision,
    )
