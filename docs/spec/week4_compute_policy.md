# Week 4 Compute-Policy Compatibility Note

This note reconciles wording; it does not amend M0 semantics or add a new team-approved decision.

- CPU remains the semantic/correctness reference and required fallback.
- The Week 3 CUDA implementation is accepted and may be preferred for suitable large tensor-heavy transitions. “CUDA available” is not evidence of a speedup.
- Small workloads, orchestration, graph generation, and graph summaries may remain CPU-side. Current immutable `GameState` snapshots move voter vectors across the CPU/CUDA boundary for transitions.
- Any performance claim requires the same config, graph/root seed, initial voter state, action sequence, warm-up policy, synchronization, and timing boundaries on both backends. Report initialization separately, median and spread, correctness, and per-device memory.
- Each discovered A5000 has separate VRAM. The documented server baseline is two devices, but the GPU-visible `mas` run on 2026-10-23 reported three; discover device count at runtime. There is no pooled-memory assumption, automatic distribution, or multi-GPU execution in Week 4.
- Persistent GPU-resident authoritative state is deferred pending matched measurements and an approved ownership/rollback design.

The frozen `docs/spec/technical_spec_v1.md` / DEC-001 statement (“CPU reference; GPU only after profiling”) remains true: CUDA is an optional implementation path, its use is workload-dependent, and a matched profile is required before claiming benefit. The Week 3 plan's GPU-preferred wording is interpreted as a workload selection preference, not a guarantee or model/API semantic change.
