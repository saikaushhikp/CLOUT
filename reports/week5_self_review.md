# Week 5 self-review

**Status: Implementation and automated validation substantially complete;** DEC-012 is approved as a decision to preserve aggregate-only Observation v1; it does not approve richer voter-level profiles. No commit was created. Baseline `HEAD` remains `775633d5bbe6360b423dedf8d0647230fa1e2cfb`.

# Week 5 baseline audit

Date: 2026-10-23 (Asia/Kolkata)  
Baseline: `775633d5bbe6360b423dedf8d0647230fa1e2cfb`  
Branch: `master`  
Working tree at start: only the user-provided `docs/plans/Implim_Week5.md` was untracked.

## Source and environment audit

The Week 5 plan's cited `Proposal.pdf`, `planning_tentative.pdf`, and `logical_next_steps.md` are actually `/data/ai23btech11017/MAS/just_see/Proposal_sept-15.pdf`, `/data/ai23btech11017/MAS/just_see/Doclings_oct-4.pdf`, and `/data/ai23btech11017/MAS/just_see/logical_next_steps_Oct-2.md` respectively. No claim below is based on reading those absent files; traceability uses the Week 5 plan and the committed project contracts (`technical_spec_v1.md`, `data_contracts.md`, `decision_log.md`, Week 3/4 handoffs, and `PROJECT_ASSUMPTIONS.md`).

The authorized `mas` environment reported Python 3.12.15, NumPy 2.5.2, PyTorch 2.7.1+cu118, CUDA available, and three NVIDIA RTX A5000 devices. The host driver reported 535.183.01; per-device total memory was 24,564 MiB. Actual free memory is transient and must be rechecked before each large run. Matplotlib and httpx were installed; FastAPI, Uvicorn, pytest, and Playwright were not installed at audit time. GPU visibility was verified in an elevated GPU-visible command context; a sandbox CPU fallback is not GPU evidence.

## Contract guardrails

- DEC-002–DEC-009 remain frozen; fixed-horizon outcome and strict `score > 0.5`
  win threshold remain authoritative.
- Existing `Observation` remains aggregate-only and schema v1. No Week 5 plan
  approval is treated as approval to reveal voter-level information.
- Model-checker claims are only exact over explicitly enumerated finite worlds
  and their declared action abstraction; capped search is incomplete.
- Browser/API responses must use public observation/result allowlists only.
- CPU is the semantic reference; CUDA execution must name an actual device.

## Plan objective to evidence map

| Objective | Planned evidence (implementation target) | Status at baseline |
| --- | --- | --- |
| Restore interactive campaign game | Browser UI + local API + real `GameEngine` episode | Missing |
| Observation-respecting policies | Uniform and greedy public-model policies, paired episodes | Baselines only |
| Temporal/epistemic reasoning | Trace diagnostics plus bounded finite-world checker | Trace LTL exists; epistemic checker missing |
| MAS/topology validation | ER and ring-lattice experiments; aggregate trajectories | ER only |
| Resource sensitivity | Exact Decimal budget-factor experiment | Missing |
| Information-level experiment | Approved versioned projection or explicit blocked decision | DEC-012 approved aggregate-only v1; richer profiles explicitly not authorized |
| GPU scale/reproducibility | CUDA 1k/10k parity and repeated 100k benchmark with metadata | Week 4 evidence only; must rerun for Week 5 |
| Reviewable results | JSONL/CSV, plots, runbook, reviewer evidence | Week 4 reports exist; Week 5 outputs missing |

## Baseline test provenance

Week 4 reports describe its own tests and benchmarks; those are historical and not represented as Week 5 executions. Week 5 test counts and benchmark data are recorded in `reports/week5_self_review.md` and individual result files.

# Week 5 budget sensitivity

Run: `GreedyPublicModelStrategy`, ER `N=1,000`, `H=5`, alpha 0.5, 20 paired seeds; base budget `20.00`, exact Decimal levels `10.00` (0.5×), `20.00` (1×), and `40.00` (2×). Every level had 20 complete episodes and zero failures.

| Initial budget | Mean score | Population SD | Victory rate | 95% Wilson interval | Mean spent |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10.00 | 0.56610 | 0.03828 | 1.00 | [0.839, 1.000] | 10.00 |
| 20.00 | 0.64050 | 0.03310 | 1.00 | [0.839, 1.000] | 20.00 |
| 40.00 | 0.67920 | 0.03055 | 1.00 | [0.839, 1.000] | 25.00 |

The highest budget is not fully spent because the five-round schedule allows at most one broadcast per round and the policy stops spending when the available action set no longer improves its synthetic expected score estimate. This is a single-policy, single-scale preliminary study; it is not evidence that more budget always improves results outside this configuration.

Raw episodes, action schedules, aggregate score, seed, budget ledger, and timings are in `reports/results/week5/main/all.jsonl`; aggregate data are in `reports/results/week5/main/all_aggregate.json`. Figure: `reports/figures/week5/budget_victory_rate.png` (95% Wilson intervals), `budget_mean_score.png` (±1 standard error), and `budget_used.png`. All three samples happen to have 20/20 wins, so the Wilson intervals remain wide.

# Week 5 information-availability results

**Status: `APPROVED DECISION`.** DEC-007 retains aggregate-only Observation v1. DEC-012 is a proposal, explicitly not implemented but team-approved. The 25/50/75/100% profile labels in the implementation plan do not define fields, sampling, stable voter identity, or privacy effects, so no richer profile was implemented and no information-level comparison was run.

All Week 5 player strategies, experiment records, dashboard/API payloads and plots use `aggregate_v1`. No voter support/susceptibility arrays, hidden edges, or private seed streams are exposed through the browser service. The proposal and needed team decision fields are in `docs/spec/observation_profiles_experiment.md` and `docs/decisions/DEC-012_observation_profiles_proposal.md`.

# Week 5 MAS and graph validation

The frozen M0 transition and engine were not changed. The experiment-only `ring_lattice_graph` returns the existing sparse undirected graph contract and is deterministic for a fixed seed. Unit coverage checks no loops/duplicates, wraparound, degree invariants, repeatability and 1,000-node sparsity. Existing unit tests cover isolates, disconnected graphs, clipping, exact threshold and synchronous updates.

## Fixed-seed sanity schedules

`reports/results/week5/mas_sanity.jsonl` records six real CPU engine episodes with `N=1,000`, ER `p=8/999`, `H=5`, alpha 0.5, seed 20261019. All schedules completed at fixed horizon; there is no convergence requirement. Aggregate support trajectories (initial through round 5), terminal score, votes, budget use, graph metrics and accepted actions are saved in the JSONL.

| Scenario | Aggregate support start → end | Score | Spent |
| --- | ---: | ---: | ---: |
| No campaign + peer dynamics | 0.50426 → 0.50115 | 0.504 | 0.00 |
| One broadcast, then no-op | 0.50426 → 0.50863 | 0.542 | 5.00 |
| Broadcast each round | 0.50426 → 0.54338 | 0.674 | 25.00 |
| Repeated public target `v0` | 0.50426 → 0.50132 | 0.504 | 5.00 |
| Peer-only (zero campaign deltas) | 0.50426 → 0.50115 | 0.504 | 0.00 |
| Broadcast + targeted + peer | 0.50426 → 0.50866 | 0.542 | 6.00 |

For topology comparison, 20 paired seeds used the same node count, horizon, dynamics, action parameters, strategy and budget. ER mean terminal score was 0.64050; ring-lattice was 0.61565. Both had a 1.00 observed victory rate in this sample. ER and ring-lattice have different graph structures by design; the experiment does not compare identical edge sets. See the paired per-episode metrics in `reports/results/week5/main/all.jsonl` and `reports/figures/week5/topology_score.png` and `reports/figures/week5/topology_victory_rate.png`.

All dynamics conclusions are bounded to these fixed-seed scenarios. No claim of consensus, universal stability, or broad topology causality is made.

# Week 5 finite-world model-check results

The checker branches through the real `GameEngine.step`; its claims are exact only over each explicitly supplied finite world set and action abstraction. Targeted actions are limited to the first eight lexicographically ordered eligible singleton IDs plus broadcast/no-op. Horizon is one in these hand-constructed fixtures, total node cap 10,000 across possible/guaranteed search, horizon cap 128, and input-world cap 256. “Knowledge” values use public observation equivalence. These are not claims about the continuous hidden-state space or universal optimality.

| Scenario/proposition | Worlds | Result | Meaning |
| --- | ---: | --- | --- |
| PossibleVictory (high-support fixture) | 1 | TRUE | An enumerated winning sequence exists |
| GuaranteedVictory (high-support fixture) | 1 | TRUE | Shared policy wins every supplied world |
| InevitableDefeat (low-support, zero-budget fixture) | 1 | TRUE | No enumerated legal sequence wins |
| `K_player(GuaranteedVictory)` (high-support fixture) | 1 | TRUE | Holds in the supplied observation class |
| `K_player(InevitableDefeat)` (low-support fixture) | 1 | TRUE | Holds in the supplied observation class |
| Mixed indistinguishable worlds: PossibleVictory | 2 | TRUE | Each world has a winning singleton target |
| Mixed indistinguishable worlds: GuaranteedVictory | 2 | FALSE | No common observation-based target wins both |
| Mixed indistinguishable worlds: InevitableDefeat | 2 | FALSE | A winning sequence exists in each world with world knowledge |

The mixed pair has the same public aggregate (0.695), graph summary, budget, legal action descriptors and public history; hidden high/low support positions are swapped. It demonstrates the distinction between `forall world exists omniscient action` and `exists observation-consistent policy forall world`. The unit test asserts these exact outcomes. A node-limit fixture verifies `INCOMPLETE` is not converted to a proof.

Machine results: `reports/results/week5/model_check.json`. Code and scoped definitions: `src/dumbocracy/epistemic.py` and `docs/spec/epistemic_semantics_v1.md`. Early defeat remains analysis only; the engine still ends at fixed horizon.

# Week 5 GPU and matched-backend results

All measurements below are machine-specific; they are not general speedup claims. Environment:conda environment named `mas`, Python 3.12.15, NumPy 2.5.2, PyTorch 2.7.1+cu118 (CUDA build 11.8), NVIDIA driver 535.183.01. Runtime discovery found three NVIDIA RTX A5000 GPUs, each 24,564 MiB total. At the final inventory query free memory was 10,682 / 10,887 / 24,218 MiB for devices 0/1/2 respectively. Benchmark device was explicitly `cuda:0`; its benchmark-start free memory was 10,682 MiB. VRAM is per device, not pooled.

## Explicit CUDA scale run

Command:

```bash
conda run -n mas python examples/week5_gpu_scale_benchmark.py --device cuda:0 --nodes 1000 10000 100000 --seed 20261019 --rounds 5 --repeats 3 --output reports/results/week5/gpu_scale.jsonl
```

Nine successful rows (three per size) are in `reports/results/week5/gpu_scale.jsonl`. `gpu_scale_prior_runs.jsonl` preserves earlier attempts for audit; use only the clean final file for the table below. Graph probabilities followed `min(1,8/(N-1))`. Each step was synchronized on CUDA. The action schedule was four broadcasts (one per round while budget allowed) then no-op; all runs reached fixed horizon with identical correctness fields per size.

| N | Edges | Components | Median episode seconds (range) | Median step ms (all 15 steps) | Peak allocated / reserved MiB | Vote share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 3,973 | 1 | 0.09278 (0.03884–0.23409) | 13.72 | 0.17 / 2 | 0.63700 |
| 10,000 | 39,714 | 3 | 0.37118 (0.36602–0.44644) | 36.91 | 1.67 / 2 | 0.64460 |
| 100,000 | 399,252 | 33 | 4.00968 (3.89902–4.01525) | 422.51 | 16.76 / 30 | 0.64483 |

The 1k repetitions have pronounced startup/host jitter; all individual repetitions and step times are preserved in JSONL. No dense adjacency was allocated. The observed run fits the available per-device VRAM.

The first benchmark attempt failed because this PyTorch build rejected a device-valued `reset_peak_memory_stats` argument. A minimal CUDA probe showed the no-argument call works after `torch.cuda.set_device(index)`; the benchmark was updated and a 1k smoke passed before the full 9-row run. No driver changes were made.

## Matched CPU/CUDA equivalence

The Week 4 comparison harness was run twice at each size with identical seed, graph, voters, action schedule and horizon. Final saved repetitions are `reports/results/week5/backend_comparison_1000.jsonl` and `reports/results/week5/backend_comparison_10000.jsonl`.

| N | Max support abs difference (repetitions) | Ledger/result equal | Typical episode seconds CPU / CUDA |
| ---: | --- | --- | --- |
| 1,000 | 2.22e-16; 1.11e-16 | yes / yes | ~0.056–0.058 / ~0.082–0.088 |
| 10,000 | 2.22e-16; 2.22e-16 | yes / yes | ~0.585–0.591 / ~0.437 |

All differences are far below the required `1e-10`. On this run CUDA was slower at N=1k and faster at N=10k; these short, host-sensitive samples do not establish a general speedup. The full CUDA policy episode is saved as `reports/results/week5/demo_episode.json` and completed with `backend=cuda:0`.

The test executor's default shell Python was 3.14 and lacked NumPy; all project checks were therefore rerun using `conda run -n mas`. GPU checks ran in an explicitly GPU-visible execution context.

# Week 5 requirements traceability

Status is updated as implementation proceeds. “Decision-gated” means no implementation/result may imply richer disclosure absent an approved record.

| Plan milestone / requirement | Code / evidence target | Verification |
| --- | --- | --- |
| MM5.0 baseline and source mapping | `reports/week5_baseline_audit.md` | Review commit, status, source inventory |
| MM5.1 finite epistemic semantics | `docs/spec/epistemic_semantics_v1.md`, `src/dumbocracy/epistemic.py` | Unit and integration fixtures |
| MM5.1 observation profile gate | `docs/spec/observation_profiles_experiment.md`, proposed DEC-012 | Confirm default schema v1 unchanged; no profile metrics without approval |
| MM5.3 trace/model distinction | `src/dumbocracy/verification.py` | Trace-only and finite-scope labels; known answers |
| MM5.4 uniform and greedy policies | `src/dumbocracy/strategies.py` | Public-input contract and paired-seed experiment |
| MM5.5 topology and budget studies | `src/dumbocracy/graph_experiments.py`, `examples/week5_run_experiments.py` | Invariants, deterministic JSONL, aggregate reports |
| MM5.5 plot generation | `examples/week5_plot_results.py`, `reports/figures/week5/` | Plot from saved records only |
| MM5.6 local browser API | `src/dumbocracy/webapp.py`, `web/`, API spec | Real-engine integration and recursive privacy scan |
| MM5.7 CUDA 100k and parity | `examples/week5_gpu_scale_benchmark.py`, performance report | Explicit CUDA device; parity at 1k/10k |
| MM5.8 end-to-end episode | `examples/week5_run_demo.py` | Deterministic safe artifact; browser walkthrough |
| MM5.9 regression/handoff | `docs/spec/week5_handoff.md`, self-review, test plan | Full suite, M0 validator, compile, diff check |

## Execution status (2026-10-23)

| Milestone | Status | Evidence / limitation |
| --- | --- | --- |
| MM5.0 | Done | This baseline report and experiment manifests |
| MM5.1 | Done with approved aggregate-only gate | DEC-012 approved aggregate-only v1; richer profiles need a separate exact-field decision |
| MM5.2 | Done, bounded | Real `GameEngine.step` branching, finite fixtures, node cap and incomplete status; see epistemic tests/results |
| MM5.3 | Done with scoped optional model data | `verification.py` joins completed-trace LTL and optional finite-world results; absence of worlds is explicit `NOT_SUPPLIED` |
| MM5.4 | Done, preliminary | Uniform and greedy public-input policies; 20 paired seeds, no general optimality claim |
| MM5.5 | Done, preliminary | Ring lattice, six MAS schedules, 20-seed topology/budget/strategy result set and plots |
| MM5.6 | Partial | Actual FastAPI service/UI, route tests, and Uvicorn Unix-socket end-to-end passed. |
| MM5.7 | Done | 1k/10k/100k CUDA; three repeated sizes; CPU/CUDA parity at 1k/10k |
| MM5.8 | Done for preliminary matrix | Actual JSONL/aggregate/plots; information levels explicitly blocked, not fabricated |
| MM5.9 | Partial | Full suite/M0/compile/diff checks run; five-person approval and TCP/browser review are complete |

The v1 information boundary remained unchanged. Reviewer and Week 6 follow-up items are listed in `docs/spec/week5_handoff.md`.

# Week 5 strategy comparison

Run: CPU reference, ER `N=1,000`, `p=8/999`, `H=5`, `alpha=0.5`, base budget `Decimal("20.00")`, 20 paired root seeds (11…211 as listed in the manifest), fresh episode/RNG per policy and seed. Each policy sees only v1 aggregate observations, public legal actions/history, and its episode RNG. There were 20 completed, zero failed/skipped episodes per policy. Per-episode data and exact configuration/provenance are in `reports/results/week5/main/all.jsonl`.

| Policy | Mean terminal score | Population SD | Victory rate (n=20) | 95% Wilson interval | Mean budget used | Score delta per spend vs paired no-op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NoOp | 0.49830 | 0.03715 | 0.45 | [0.258, 0.658] | 0.00 | — |
| FirstLegal | 0.49895 | 0.03734 | 0.45 | [0.258, 0.658] | 5.00 | 0.00013 |
| SeededRandomLegal | 0.54005 | 0.04989 | 0.75 | [0.531, 0.888] | 7.70 | 0.00422 |
| UniformBudget | 0.62100 | 0.03474 | 1.00 | [0.839, 1.000] | 17.00 | 0.00722 |
| GreedyPublicModel | 0.64050 | 0.03310 | 1.00 | [0.839, 1.000] | 20.00 | 0.00711 |

The GreedyPublicModel mean is 0.0195 above UniformBudget on this small matrix, while UniformBudget's paired score improvement per spent unit is slightly higher. This is descriptive evidence only: 20 seeds, one graph scale and one dynamics/action configuration do not establish a general policy ranking or causal superiority. The greedy model prior is intentionally simple and not trained/calibrated from hidden individuals.

Artifacts generated from saved records only:

- `reports/figures/week5/strategy_score.png` (mean with ±1 standard error)
- `reports/figures/week5/strategy_victory_rate.png` (95% Wilson intervals)
- `reports/figures/week5/support_trajectory.png`
- `reports/figures/week5/action_mix.png`
- `reports/results/week5/main/all_aggregate.json`
- `reports/results/week5/main/all_manifest.json`

Information availability is `aggregate_v1` throughout. A network-aware player policy remains unavailable under DEC-007; no oracle policy is mixed into these results.


# Backlog / milestone status

| Item | Status | Evidence / caveat |
| --- | --- | --- |
| W4-B01 finite epistemic checker | Done, bounded | `src/dumbocracy/epistemic.py`, finite-world tests/results |
| W4-B02 possible/guaranteed/inevitable/knowledge | Done within abstraction | H=1 fixtures; first eight public singleton targets; node cap/incomplete semantics |
| W4-B03 strategies | Done preliminarily | Uniform and aggregate-only greedy; 20 paired seeds, one scale/config |
| W4-B04 playable UI/service | Partial | FastAPI UI/API and real engine; tests + UDS live Uvicorn flow; TCP browser unavailable in executor |
| W4-B05 MAS/topology | Done preliminarily | Ring lattice and six named fixed-seed schedules; core engine unchanged |
| W4-B06 budget experiment | Done preliminarily | Exact Decimal 0.5×/1×/2× levels; 20 paired seeds |
| W4-B07 information levels | Blocked by approved DEC-012 scope | Aggregate-only v1 retained; no profile disclosure or fabricated results |
| W4-B08 review evidence/plots | Done preliminarily | JSONL, aggregates, PNG figures, reports and runbook |
| MM5.0 | Done | `week5_baseline_audit.md`, requirements matrix |
| MM5.1 | Done with approved aggregate-only gate | DEC-012 approved; richer profiles still require a new decision defining fields and privacy semantics |
| MM5.2 | Done | Actual engine branching; mixed indistinguishable worlds and node-cap test |
| MM5.3 | Done with scoped optional model data | Shared verification adapter distinguishes trace-only from supplied finite-world checks |
| MM5.4 | Done preliminarily | Policies plus paired comparison; no optimization/quality generalization claim |
| MM5.5 | Done preliminarily | Sanity/topology/budget records and figures |
| MM5.6 | Partial | Server/API tests and Unix-socket smoke pass; TCP/browser review remains |
| MM5.7 | Done | CUDA 1k/10k/100k, three final repeats, matched parity |
| MM5.8 | Done preliminarily | 200 paired result rows and plots; info-level experiment intentionally blocked |
| MM5.9 | Partial | Regression and handoff done; five-person review is outstanding |

## Commands actually run

- `conda run -n mas python tests/validate_week1_artifacts.py` — passed.
- `conda run -n mas python -m unittest discover -s tests -v` — **82 passed,
  0 failures/errors/skips** on 2026-10-27; CUDA tests executed and passed.
- `conda run -n mas python -m unittest tests.unit.test_epistemic tests.unit.test_graph_experiments tests.unit.test_strategies tests.integration.test_web_api -v` — **18 passed** on 2026-10-27.
- `conda run -n mas python -m compileall -q src/dumbocracy tests examples` —
  passed.
- `git diff --check` — passed.
- Public mock client via activated `mas` environment and fixture stdin — emitted
  `{"action_id":"targeted_contact","target_ids":["v0"]}`. An initial
  `conda run` invocation did not forward redirected stdin; the activated-shell
  rerun passed.
- `examples/week5_model_check.py` — three scenarios saved; high-support
  possible/guaranteed TRUE, low-support inevitable TRUE, mixed indistinguishable
  possible TRUE / guaranteed FALSE / inevitable FALSE.
- `examples/week5_mas_sanity.py` — six named fixed-seed episodes completed.
- `examples/week5_run_experiments.py --experiment all ...` — 200 complete
  episodes, 20 paired seeds, zero failed/skipped; exact command/timestamps and
  aggregates in `reports/results/week5/main/all_manifest.json`.
- `examples/week5_plot_results.py ...` — nine PNG figures generated from saved
  result records.
- `examples/week5_gpu_scale_benchmark.py ...` — final 9 rows (three repeats at
  1k/10k/100k), explicit `cuda:0`; 100k completed.
- `examples/week4_backend_comparison.py ...` — two executions each at 1k and
  10k; CPU/CUDA final-support differences at most 2.22e-16, exact ledger and
  result equality.
- `examples/week5_run_demo.py ...` — full five-round GreedyPublicModel episode
  on `cuda:0`, final score 0.637 / victory true.
- `conda run -n mas python -m compileall -q src/dumbocracy tests examples` — passed.
- Frontend `npm test` — **3 passed**; `npm run build` — passed after the external font import was removed.
- `git diff --check` identifies two trailing-space lines in pre-existing modified Week 3/4 handoffs (`docs/spec/week3_handoff.md`, `docs/spec/week4_handoff.md`); these unrelated user changes were left untouched.

## Defects and limitations found

- The final API integration run initially failed because its CSS check expected
  the stale 560px breakpoint while the production UI uses a 640px mobile
  breakpoint. Updated the test to assert the implemented breakpoint and to
  reject external Google Fonts imports; removed the import to preserve the
  documented offline-capable UI requirement. Both requested test commands now
  pass.
- Default system Python was 3.14 without NumPy; switched to the authorized
  `mas` environment. Optional extras were declared and installed with
  `python -m pip install -e '.[web,analysis,test]'`.
- First scale-script attempt used an unsupported PyTorch
  `reset_peak_memory_stats(device)` call. Fixed by setting current CUDA device
  and using the no-argument memory API; smoke and final run passed. Earlier
  attempt records were preserved in `gpu_scale_prior_runs.jsonl`; the final
  clean evidence is `gpu_scale.jsonl`.
- Source Proposal/plan PDFs and `logical_next_steps.md` named by the Week 5
  plan are absent from this checkout; deliverables are grounded in committed
  specifications and the user-provided Week 5 plan.
- DEC-012 is approved as recorded in `docs/decisions/decision_log.md`; no blanket Week 5 acceptance or five-person review completion is inferred from it.

## Result bundle and next work

Reports include strategy, MAS/topology, budget, information-gate, finite-world, performance and UI evidence. JSONL/JSON outputs are under `reports/results/week5/`; figures are under `reports/figures/week5/`. Reviewer launch/API commands and role checklist are in `docs/spec/week5_handoff.md`.

Before calling Week 5 accepted, reviewers must perform the browser/TCP manual walkthrough, inspect the actual Week 5 tree/results on one common revision, and review bounded checker/policy assumptions.
Week 6 should expand seeds and topologies, calibrate the synthetic policy prior, unify temporal/epistemic reporting, and polish UI/accessibility/production hardening.
