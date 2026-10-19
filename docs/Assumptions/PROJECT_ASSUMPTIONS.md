# Dumbocracy — Persistent Project Assumptions

This file records project-wide assumptions to carry into later `Implim_WeekN.md` plans. If a later decision changes one, update this file and record the decision in the project's decision log.

## Development environment
- All development, execution, and testing are performed on an SSH-accessible server.
- The server has 2 × NVIDIA A5000 GPUs, 48 GB VRAM combined.
- Treat GPU memory as separate per-device memory (typically 24 GB each), not as one automatically pooled 48 GB allocation.
- Keep correctness tests runnable on CPU. GPU tests and benchmarks are supplementary unless a requirement explicitly states otherwise.

## Coding-agent workflow
- Each week has a corresponding implementation plan, e.g. `Implim_Week1.md`, that directs a coding agent to implement the intended deliverables, run tests, diagnose/fix defects, rerun tests, and produce evidence.
- Coding-agent output is not accepted solely on the agent's assertion that it is complete.
- All five group members manually review the relevant code, tests, outputs, and documentation before proceeding to the next week.
- The agent must distinguish accepted requirements from proposals and unresolved decisions; it must not invent team approval.

## GPU/software engineering policy
- Prefer vectorized NumPy/PyTorch operations and sparse graph representations for large-scale work where appropriate.
- Use GPU acceleration when profiling shows a meaningful benefit; CPU may remain preferable for graph construction, orchestration, small workloads, and some graph algorithms.
- Do not introduce multi-GPU distributed execution without a measured use case and explicit handling of per-device memory and communication.
- Record random seeds, configuration, code version, device/backend, and relevant software versions for experiments.
- Benchmark before making performance claims.

## Modeling caution
- Dumbocracy is an election-campaign resource-allocation game over a graph-based electorate, not simply a consensus simulation.
- Consensus theory is a reference for graph dynamics and mathematical validation, not a requirement that voters converge to one opinion.
- Resilient consensus/W-MSR, Erdős–Rényi graphs, ADMM, and proximal methods are optional candidate techniques unless the team explicitly adopts them.
- Technical Specification v1 and the team decision log are the source of truth for model semantics and API contracts.
