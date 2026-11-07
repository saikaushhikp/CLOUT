# Dumbocracy v2 — Complete Revamp: Master Plan

## Context
v1 (Weeks 1–5, HEAD `e3f7760`) is a working but degenerate game. Peer averaging collapses scalar
support toward 0.5, and broadcast is ~80× more cost-effective than targeting. So "broadcast whenever
affordable" wins every seed. Information is aggregate-only, so the epistemic checker has nothing to
decide. The team is revamping the whole repo and game around its own design (`graph_construction.md`,
primary) plus selected ChatGPT ideas, grounded in the lectures (consensus, resilient consensus, SI/SIR,
prox/ADMM, temporal + epistemic logic). Two-week budget. GPU-first in conda env `mas`.

The deliverable of THIS plan, once approved, is a set of repo documents (`docs/v2/`) that I and any
sub-agents implement, test and debug against:
`technical_spec_v2.md` (frozen maths + contracts) + `decisions_v2.md` + phase plans `P0..P7`.

## 1. Decisions captured from the user (binding)
| # | Decision |
|---|---|
| U1 | Restructure freedom: my choice → **archive v1 (git tag `v1-final`), fresh package in place** |
| U2 | Player **designs manifesto b once** (5-sector budget simplex) after surveying |
| U3 | **Single player now**, but engine is **player-keyed** (per-player treasury, b, workers, info log). There is a `Player` interface for human / scripted / future LiteLLM opponent. No opponent logic is built. |
| U4 | Graph: sparse edge existence (much likelier within region). The 10 numbers are **edge weights**: intra `p_r∈[0.5,0.9]`, inter `p_g,rs∈[0.1,0.4]`. Ranges may be retuned. |
| U5 | **Preferences v_i evolve** (weighted consensus on the simplex); campaigns **pull v_i toward b** |
| U6 | Election: each voter **votes w.p. σ(β(S−z))** (seeded Bernoulli); win iff votes > N/2 |
| U7 | Flow: **Survey phase → commit b → T campaign rounds → election**; T=5 default (1–10) |
| U8 | Paid query (S1,S2,Region) → **exact** region % preferring S2 over S1 (current state), cost 5 |
| U9 | **Fixed free workers each round + paid extra workers**; player assigns them to regions |
| U10 | **Single media channel, region-targetable**; cost scales with region size |
| U11 | 6 **hand-designed archetypes**, 4 drawn **without replacement** (one per region) |
| U12 | Hidden voter attrs: social α_i, media β_i, campaign γ_i, **zealots** (stubborn) |
| U13 | Lectures to build in: **consensus + resilient consensus, SI/SIR, prox/ADMM, temporal+epistemic** |
| U14 | **One treasury** for the whole game (surveys + extra workers + media) |
| U15 | Each round: **free noisy poll** (±MoE); paid queries are available throughout |
| U16 | UI: **4-region map + WebGL sampled graph**; **optional** round timer (engine is turn-based) |
| U17 | Scale: **interactive demo N=10k; batched GPU experiments to 1e5 (required), 1e6 (stretch)** |

## 2. Model specification (to be frozen in `technical_spec_v2.md`; defaults marked ⚙ are calibrated in P2)
Sectors `k ∈ {Edu, Welfare, Agri, Industry, Env}`. Simplex Δ⁴ = {x ≥ 0, Σx = 1}.

**2.1 Regions.** R={N,S,E,W}. `n_r = ⌈0.01N⌉ + Multinomial(N − 4⌈0.01N⌉, π)`, π ~ Dir(a·1), a=2 ⚙.

**2.2 Graph (sparse weighted SBM, GPU).** Draw `p_r ~ U[0.5,0.9]` (4) and `p_g,rs ~ U[0.1,0.4]` (6).
- Existence: same region `P = d_in / n_r`. Cross region `P = d_out · (p_g,rs n_s / Σ_{s'≠r} p_g,rs' n_s') / n_s`.
- Defaults ⚙ `d_in = 14`, `d_out = 2` (≈87% of edges intra).
- Sampling per block: `E_rs ~ Binomial(n_r n_s, P)` with uniform endpoints, then dedupe and drop loops. The cost is O(E), not O(N²).
- Weights: `w_ij = p_r` (intra) or `p_g,rs` (inter). The graph is undirected.
- Storage: CSR / edge-index int32 on GPU, float32 weights, with an optional float64 parity mode.

**2.3 Preferences.** Archetypes μ (public library, hidden assignment) — [Edu,Wel,Agri,Ind,Env]:
Agrarian heartland [.15,.20,.40,.10,.15] · Industrial belt [.15,.15,.10,.45,.15] ·
Green urban [.25,.10,.10,.15,.40] · Welfare-reliant [.15,.45,.15,.10,.15] ·
Knowledge hub [.40,.10,.10,.25,.15] · Balanced middle [.20,.20,.20,.20,.20].
The assignment is an injective map σ: R→archetypes, so there are 360 possibilities. `v_i(0) ~ Dir(κ μ_σ(r))`, κ=20 ⚙.

**2.4 Hidden attributes.**
- Each region draws a hidden "personality" without replacement from {social, media-savvy, independent, receptive}. The personality sets Beta parameters for α_i, β_i and γ_i ∈ [0,1] ⚙.
- Zealots: fraction ζ=5% ⚙ (uniform). Zealots have α=β=γ=0: they never update but still influence their neighbours.

**2.5 Campaign exposure in round t (player p).** Each mechanism gives an intensity in [0,1]; they combine as `c_i^p = 1 − Π(1 − e)`.
- **Workers:**
  - Allocation `W_free=10` ⚙ free workers plus extras at `cost_w=4` ⚙ each, split by region by the player.
  - Each worker contacts q=0.25%·N ⚙ uniform voters in its region; a contacted voter gets intensity `η_w γ_i` (η_w=0.30 ⚙).
  - Word of mouth uses an **SIR cascade**: contacted voters are infectious and transmit along edge (i,j) with probability `τ w_ij` per hop. Infected nodes recover after 1 hop with probability ρ. There are L=3 ⚙ hops.
  - A secondary contact gets `η_wom γ_j` (η_wom=0.10 ⚙). Spectral reference: the cascade can spread when τ·λ_max(W_block) > ρ.
- **Media:**
  - Choose a set of regions; cost per region = `m0 · n_r/(N/4)` (m0=20 ⚙).
  - Each voter in an aired region is exposed with probability `p_m=0.6` ⚙ and gets intensity `η_m(n) β_i`.
  - Diminishing returns: `η_m(n) = η_m0/(1+λ n)` (η_m0=0.15, λ=0.5 ⚙), where n is the number of prior airings in that region.

**2.6 Dynamics (per round, synchronous, simplex-preserving).**
- `a_i = α_i ω` (ω=0.5 ⚙) and `c_i = Σ_p c_i^p`, scaled so that `a_i + c_i ≤ 1`.
- `v_i ← (1 − a_i − c_i) v_i + a_i · Agg_i(v) + Σ_p c_i^p b_p`.
- `Agg_i` options:
  - `degroot` (default): weighted neighbour mean `Σ w_ij v_j / Σ w_ij`.
  - `wmsr_F`: drop up to F neighbours farthest in TV distance from v_i, then take the weighted mean (resilient-consensus variant). This is an experiment switch.
- Isolated voters keep their own v (no pull toward 0, fixing the v1 quirk). Every update is a convex combination, so the simplex invariant holds; it is still asserted in tests.

**2.7 Vote.**
- `S_i = 1 − ½‖v_i − b‖₁` and `P_i = σ(β(S_i − z))`, with β=12 and z=0.75 ⚙ (calibrated).
- Official result: seeded Bernoulli per voter; victory iff votes > N/2.
- Also record the expected share `mean P_i` and its standard deviation for analysis.
- Multi-player generalisation (documented, not built): choice among parties + abstention.

**2.8 Information (per player).**
- **Public, common knowledge:** N, region sizes n_r, archetype library (not the assignment), personality library, graph summary per region (nodes, edges, mean degree), all rules and costs.
- **Free:** at start and in each round, the global % of voters with each sector in their top-3.
- **Paid** (5): query (S1,S2,r) → exact % of region r with v[S2] > v[S1] at the current state. Available in all phases.
- **Free poll each round after commit:** sample n_poll=400 ⚙ voters and report the estimated mean P ± 95% MoE.
- Every answer is appended to the player's knowledge log, which defines the information set I(O).

**2.9 Game phases (player-keyed, turn-based, immutable snapshots).**
`SETUP → SURVEY (queries…; COMMIT b) → ROUND t=1..T (queries…; ALLOCATE{workers_by_region, extra_workers, media_regions} ends round → dynamics) → ELECTION → DONE`.
- Treasury B=100 ⚙ is a Decimal ledger.
- Actions are validated purely; rejections are atomic.
- Seeds: separate streams per subsystem (graph, preferences, attributes, campaign, cascade, media, poll, election) derived from the root seed.

**2.10 Balance acceptance (P2 calibration gate — the v1 lesson).** Over ≥200 seeds at N=10k:
1. Uniform b (0.2 each) with no campaign wins ≤ 30%.
2. Oracle b with a planner campaign wins ≥ 85%.
3. Survey-informed b beats uninformed b (paired test, p<0.01).
4. No fixed allocation pattern wins ≥ 80%; workers-only and media-only each win on some seeds.
5. Region personality and zealot fraction measurably change the best allocation.

## 3. Lecture integration map
| Lecture | Where |
|---|---|
| Consensus / Laplacian | `analysis/spectral.py`: weighted L, λ2 (mixing speed), per-region and inter-region conductance; consensus-only runs |
| Resilient consensus | `wmsr_F` aggregation; zealot experiments (do F stubborn nodes block persuasion?); (r,s)-robustness bounds (exact check is coNP-hard, so use min degree + region-union set pairs + heuristic search) |
| SI/SIR | worker word-of-mouth cascade; λ_max(W_block) threshold; eigenvector-centrality analysis of worker placement |
| Prox / projected gradient / ADMM | Manifesto optimiser: maximise E[votes] over Δ⁴ using projected gradient (simplex projection = prox of the indicator). Consensus-ADMM across regions: each region has a local b_r and z = consensus b. Used by the AI player/advisor and the oracle bound |
| Epistemic logic | Worlds = archetype assignments (360) × personality assignments × sampled micro-states consistent with the knowledge log. K_P φ ⇔ φ in all consistent worlds (exact over assignments, sampled over micro-state) |
| Temporal logic / STL | Per-round traces: ◇Lead, □(margin > −δ), Viable_t = ∃ continuation winning (rollouts); STL robustness = final vote margin; CTL-style ∃◇Victory vs ∀◇Victory over the batched worlds |

## 4. Repository restructure (P0)
- Tag `v1-final`. Move the v1 docs and reports to `docs/archive/v1/` and `reports/archive/v1/`. Remove v1 `src/dumbocracy/*`, its tests and examples (they remain in git history).
- Commit `just_see/` (the source material) and fix `.gitignore`. Delete `graph_construction.md` and the ChatGPT HTML only after their content is captured in the spec; keep them under `docs/v2/sources/`.
- New layout (torch is the single code path; CPU = torch-cpu, so parity comes for free):
```
src/dumbocracy/
  config.py        # frozen dataclasses, seed streams, defaults (⚙ table)
  sectors.py       # sector names, archetype + personality libraries
  electorate/      # regions.py, sbm.py (GPU block sampler), preferences.py, attributes.py, electorate.py
  dynamics/        # social.py (degroot, wmsr), campaign.py (workers+SIR, media), update.py
  voting.py        # similarity, P(vote), Bernoulli election, expected share
  information/     # survey.py (free stats, queries, polls), knowledge.py (per-player log, I(O))
  game/            # state.py (player-keyed snapshot), actions.py, ledger.py, engine.py, observation.py
  players/         # base.py (Player protocol), scripted.py, llm_stub.py (LiteLLM interface only)
  strategies/      # baselines.py, bayes.py (posterior over assignments), manifesto_opt.py (PG + ADMM), planner.py (rollouts), oracle.py
  logic/           # worlds.py (consistent-world sampler), epistemic.py, temporal.py (LTL/STL over traces)
  analysis/        # spectral.py, robustness.py, cascade.py
  batch.py         # vectorised B-world simulation (B×N×5 tensors) for strategies/logic/experiments
  api/             # FastAPI app (sessions keyed by game_id, player_id)
web/               # React 19 + Vite (kept toolchain); new UI
tests/{unit,integration,gpu}/  examples/  experiments/  reports/v2/
docs/v2/{technical_spec_v2.md, decisions_v2.md, plans/P0..P7.md, sources/}
```

## 5. Phase plans (each becomes `docs/v2/plans/Pk.md` with tasks, tests, exit criteria)
| Phase | Days | Content | Exit criterion |
|---|---|---|---|
| **P0 Spec & restructure** | D1 | Write spec v2 and decisions from §1–§3; archive v1; skeleton package; pyproject (torch core dependency); CI command list | Spec reviewed by user; `pytest` runs empty suite in `mas` |
| **P1 Electorate (GPU)** | D2–3 | Regions, block-SBM sampler, weights, archetypes/Dirichlet, attributes, zealots; spectral/robustness analysis | Statistical tests: region minimum, intra fraction, degree, Dirichlet means; simplex invariants; 1e5 < 5 s and 1e6 fits on one A5000; CPU and GPU give identical graphs for the same seed |
| **P2 Dynamics, campaign, voting + calibration** | D3–4 | DeGroot/W-MSR aggregation, workers + SIR cascade, media, update, election; **balance study** to tune ⚙ | Hand-computed toy cases; simplex preserved; zealots fixed; seeded reproducibility; **§2.10 gate passes** and the tuned defaults are frozen in the spec |
| **P3 Information & engine** | D4–5 | Free stats, queries, polls, knowledge log; player-keyed state, phases, ledger, actions, observations, replay; CLI game | Leakage tests (no hidden fields in observations); atomic rejection; replay determinism; full CLI game at N=10k in < 1 s/round |
| **P4 Strategies & optimisation** | D6–7 | Baselines; Bayesian posterior over 360×24 assignments from the knowledge log; PG/ADMM manifesto optimiser; rollout campaign planner on batched worlds; oracle bound; LLM player stub interface | Optimiser converges (simplex KKT residual < 1e-4); ADMM equals the centralised PG result; planner beats baselines (paired test) |
| **P5 Logic** | D7–8 | Consistent-world sampler; PossibleVictory, GuaranteedVictory (∃π∈Π ∀worlds), InevitableDefeat, K_P, Viable_t; LTL/STL over traces; information-sufficiency curve (queries vs K_P(Victory)) | Known-answer fixtures (forced win/loss, indistinguishable worlds); report states the sampling scope |
| **P6 API + UI** | D6–9 (parallel) | FastAPI v2 routes; React: survey query builder, simplex sliders for b, region map, sigma.js sampled graph (~1.5k nodes coloured by S_i, per-round snapshots), allocation panel, poll chart with CI band, results + "why" (oracle gap, region breakdown, logic verdicts), optional timer | API integration tests with a recursive privacy scan; Vitest; manual browser walkthrough at 10k |
| **P7 Experiments, report, demo** | D9–10 | Topology (d_in/d_out, weights), zealots and W-MSR, information sufficiency, budget sensitivity, strategy comparison; plots; 5–10 minute demo script | ≥200 seeds per cell with CIs; figures generated from saved JSONL; demo rehearsed |

Critical path: P0 → P1 → P2 (calibration gate) → P3 → {P4, P5, P6} → P7. P6 can start against a mock
observation JSON right after P3's schema is frozen.

## 6. Engineering rules (carried over from v1, which got these right)
- Each subsystem has its own seed stream. Results record config, seed, revision and device.
- Strategies and players see only `Observation` objects. The oracle is labelled and kept separate.
- Validation is pure; commits are atomic; money is Decimal.
- GPU-first, but CPU torch is the reference path. Parity tolerance: 1e-6 for float32 and 1e-12 for float64 mode.
- Per-device VRAM (24 GB A5000) is checked at runtime, with no pooled-memory assumption. Batch dimension B is used for parallel worlds.
- Every phase ends with: a full test run in `mas`, a short handoff note in `docs/v2/plans/Pk.md`, and the user's review.

## 7. Verification (end-to-end)
- `conda run -n mas python -m pytest tests -q` (unit + integration; GPU tests marked).
- `conda run -n mas python examples/play_cli.py --n 10000 --seed 1`: complete a scripted game.
- `conda run -n mas python experiments/balance.py --seeds 200`: §2.10 gate report.
- `conda run -n mas python experiments/scale.py --n 1e5 1e6 --device cuda:0`: timing and memory.
- Launch `uvicorn dumbocracy.api:app` and play in the browser via the SSH tunnel; check the network payloads for hidden-data leaks.

## 8. Immediate next action after approval
Execute P0: write `docs/v2/technical_spec_v2.md`, `decisions_v2.md`, `plans/P0..P7.md` from this plan,
archive v1, scaffold the package. Then stop for the user's spec review before P1 code.
