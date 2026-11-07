# Dumbocracy v2 — Technical Specification

**Status:** v2.1 (P2 calibration frozen 2026-11-07; see §14). Model semantics below are binding for implementation. Parameters marked
⚙ are *defaults* that P2's calibration study may retune; after P2 the tuned values are frozen here
(spec v2.1) and recorded in [decisions_v2.md](decisions_v2.md).

**Sources:** team design [`sources/graph_construction.md`](sources/graph_construction.md) (primary),
ChatGPT ideation [`sources/chatgpt_ideation.txt`](sources/chatgpt_ideation.txt) (secondary),
course lectures in `just_see/lectures/`, user decisions U1–U17 in [decisions_v2.md](decisions_v2.md).

---

## 0. One-paragraph summary

A single human player (the engine is player-keyed so a second, e.g. LLM, player can be added later)
runs an election campaign over an electorate of N voters living in four regions (N/S/E/W) connected by
a sparse weighted stochastic-block social graph. Every voter holds a preferred **budget allocation**
v_i over five sectors (a point on the probability simplex). The player first spends money on
**surveys** to learn what regions want, then **commits a manifesto** b (their own 5-sector budget),
then plays **T campaign rounds** deploying **party workers** (local, word-of-mouth cascades) and
**region-targeted media**. Each round voters' preferences drift toward their neighbours (consensus) and
toward b where campaigns reach them. At the end every voter votes for the player with probability
σ(β(S_i − z)) where S_i is the similarity between v_i and b. The player wins with > N/2 votes.

## 1. Notation and conventions

| Symbol | Meaning |
|---|---|
| N | number of voters (1e3 … 1e6) |
| R = {N, S, E, W} | regions, index r ∈ {0,1,2,3} in this order |
| K = 5 | sectors, index k in order **[Edu, Welfare, Agri, Industry, Env]** |
| Δ | probability simplex {x ∈ ℝ^5 : x ≥ 0, Σx = 1} |
| v_i(t) ∈ Δ | voter i's preferred allocation at round t (hidden) |
| b_p ∈ Δ | manifesto committed by player p (fixed after commit) |
| G = (V, E) | undirected social graph (ER-SBM, §2.2); w_ij = 1 on edges |
| α_i, β_i, γ_i ∈ [0,1] | social, media, campaign susceptibility (hidden) |
| ζ_i ∈ {0,1} | zealot flag (hidden) |
| T | campaign rounds (default 5, allowed 1–10) |

Floating point: model tensors are `float32` by default; a `float64` mode exists for parity tests.
Money is `decimal.Decimal` with 2 decimal places. All randomness comes from named seed streams (§9).

## 2. Electorate generation

### 2.1 Regions
`m = ceil(f·N)` with f = 0.15 (R7); `π ~ Dirichlet(a·1_4)` (a = 2 ⚙);
`n_r = m + Multinomial(N − 4m, π)_r`. Guarantees n_r ≥ 15% of N and Σ n_r = N; the split of the remaining
40% is arbitrary (random). Voters are indexed contiguously by region (region 0 first). Requires 4m ≤ N.

### 2.2 Social graph — Erdős–Rényi stochastic block model with a sparsity factor (U4 as revised by R4/R5)
1. Draw the "10 numbers" — **ER connection probabilities**: `p_r ~ U[0.5, 0.9]` for each region
   (4 numbers) and `p_rs ~ U[0.01, 0.2]` for each unordered region pair (6 numbers). Ranges are config.
2. Global **sparsity factor** s (one scalar per game) keeps the graph sparse while preserving the
   owner's relative likelihoods exactly:
   `P_rr = s · p_r`, `P_rs = s · p_rs`, with
   `s = k̄ · N / ( Σ_r p_r n_r (n_r − 1) + Σ_{r≠s} p_rs n_r n_s )` so the expected mean degree is
   k̄ = 30 (R6; capped so that every P ≤ 1). Within every block edges are independent Bernoulli(P)
   — i.e. standard ER per block. With typical draws ≈ 74% of a voter's neighbours are in-region; region
   λ_max(A_rr) ≈ 15–34, so the SIR critical transmission τ_c* ≈ ρ/λ_max ≈ 0.015–0.033.
3. Sampling per block (r ≤ s): number of edges `M_rs ~ Binomial(pairs_rs, P_rs)`, endpoints drawn
   uniformly, then remove self-loops and duplicates (canonical i<j, sort, unique). O(E) memory/time;
   must run on GPU. (Duplicate removal slightly lowers the realised count; acceptable and tested.)
4. Edges are **unweighted** (w_ij = 1); the probabilities shape *who* is connected, not how strongly.
   `w_ij` is kept in the formulas below as 1 so a weighted variant remains possible.
5. Storage: canonical undirected edge list `(src,dst)` int64→int32 when N < 2^31, plus a directed
   CSR (both directions) for aggregation. Degree `d_i = Σ_j w_ij`.
6. Public census exposes per-region node counts and realised intra/inter edge counts; the drawn
   p_r, p_rs and s are hidden.

### 2.3 Archetypes and preferences (public library, hidden assignment)
| id | name | μ = [Edu, Welfare, Agri, Industry, Env] |
|---|---|---|
| A0 | Agrarian heartland | [0.15, 0.20, 0.40, 0.10, 0.15] |
| A1 | Industrial belt | [0.15, 0.15, 0.10, 0.45, 0.15] |
| A2 | Green urban | [0.25, 0.10, 0.10, 0.15, 0.40] |
| A3 | Welfare-reliant | [0.15, 0.45, 0.15, 0.10, 0.15] |
| A4 | Knowledge hub | [0.40, 0.10, 0.10, 0.25, 0.15] |
| A5 | Balanced middle | [0.20, 0.20, 0.20, 0.20, 0.20] |

Assignment σ: R → {A0..A5} injective, uniform over the 360 possibilities (U11).
`v_i(0) ~ Dirichlet(κ · μ_σ(r))`, κ = 20 ⚙.

### 2.4 Personalities, susceptibilities, zealots
Personality library (public), assigned to regions by a uniform permutation τ: R → P (24 possibilities):

| id | name | α ~ | β ~ | γ ~ |
|---|---|---|---|---|
| P0 | Socially driven | Beta(6,3) | Beta(2,5) | Beta(3,4) |
| P1 | Media-savvy | Beta(3,4) | Beta(6,3) | Beta(2,5) |
| P2 | Independent | Beta(2,6) | Beta(2,6) | Beta(2,6) |
| P3 | Receptive to canvassing | Beta(3,4) | Beta(3,4) | Beta(6,3) |

Zealots: each voter independently is a zealot w.p. ζ = 0.05 ⚙. Zealots have α=β=γ=0 (never update)
but still influence neighbours (resilient-consensus "malicious-but-honest-broadcast" analogue).

The **hidden discrete world type** is θ = (σ, τ) — 360 × 24 = 8640 possibilities; the continuous
remainder (Dirichlet draws, graph, Betas, zealots) is the **micro-state**.

## 3. Campaign mechanisms (per round t, per player p)

Each mechanism produces per-voter *exposure intensities* e ∈ [0,1]. Intensities from all mechanisms of
player p combine as `c_i^p = 1 − Π_e (1 − e)`; zealots always get c_i^p = 0.

### 3.1 Party workers (U9)
- Each round player p has `W_free = 10` ⚙ free workers and may buy extra workers at `cost_w = 4.00` ⚙
  each (that round only). The player assigns integer worker counts `n_w(r)` to regions; unassigned
  free workers are wasted.
- Each worker contacts `q = max(1, round(0.0025·N))` ⚙ voters drawn uniformly (with replacement)
  from its region. Each contact of voter i gives `e = η_w · γ_i`, η_w = 0.30 ⚙ (multiple contacts
  compound via the product rule).
- **Word of mouth — SIR cascade** (lecture: infection models). Infectious set I_0 = contacted
  non-zealots. For hop h = 1..L (L = 3 ⚙): every infectious i transmits to each neighbour j
  independently w.p. `τ_c · w_ij` (τ_c = 0.03 ⚙, calibrated in P2 to sit near the regional SIR threshold). A newly reached non-zealot j (not reached before this
  round) receives `e = η_wom · γ_j` (η_wom = 0.10 ⚙) and joins the infectious set. After each hop each
  infectious node recovers w.p. ρ = 0.5 ⚙. Reference threshold: spreading is supercritical when
  `τ_c · λ_max(A_block) > ρ`, reported by analysis tools.

### 3.2 Media (U10)
- Single channel, region-targetable. The player chooses a subset of regions to air in this round.
  Cost per region `= m0 · n_r / (N/4)` with m0 = 20.00 ⚙ (rounded half-up to 0.01).
- Each voter in an aired region is exposed w.p. `p_m = 0.6` ⚙; exposure gives
  `e = η_m(n) · β_i` with diminishing returns `η_m(n) = η_m0 / (1 + λ n)`, η_m0 = 0.15, λ = 0.5 ⚙,
  n = number of previous airings by this player in that region.

## 4. Dynamics (U5) — once per round, after all players' allocations

```
a_i   = ω · α_i                       (ω = 0.5 ⚙; zealots: a_i = 0)
c_i   = Σ_p c_i^p
scale = 1 if a_i + c_i ≤ 1 else (1 − a_i)/c_i       (applied to every c_i^p)
v_i  ← (1 − a_i − Σ_p c_i^p) v_i + a_i · Agg_i(v) + Σ_p c_i^p · b_p
```
`Agg_i` (config `social_rule`):
- `degroot` (default): `Σ_j w_ij v_j / Σ_j w_ij`; isolated voters use their own v_i.
- `wmsr` with parameter F (resilient consensus, LeBlanc et al. 2013, adapted to the simplex): rank
  neighbours by TV distance `½‖v_j − v_i‖₁`; discard the F farthest (or all if deg ≤ F → own v_i);
  weighted mean of the rest. Experiment switch, not default.

Synchronous: all reads use v(t). Every update is a convex combination of simplex points, so v stays in
Δ; implementations still renormalise to absorb float error and tests assert `|Σv − 1| < 1e-5`, v ≥ 0.

## 5. Voting (U6)

`S_i = 1 − ½ ‖v_i − b‖₁ ∈ [0,1]`, `P_i = σ(β_v (S_i − z))`, β_v = 12 ⚙, z = 0.75 ⚙.
- **Official result:** each voter votes for the player w.p. P_i (election seed stream); votes = count;
  **victory iff votes > N/2** (ties lose). Non-votes are abstentions/others.
- **Expected share** `E = mean_i P_i` and its sd are recorded for analysis and logic.
- Multi-player generalisation (documented, not built): voter i chooses among parties p with
  probabilities `softmax_p(β_v(S_i^p − z))` including an abstain option of score 0.

## 6. Information model (per player)

### 6.1 Public / common knowledge (always visible)
N, T, current phase/round, the player's own treasury, all costs and rules, region sizes n_r, the
archetype library (not σ), the personality library (not τ), and a structural census: intra-region
edge count per region and inter-region edge count per pair (NOT the edges, NOT p_r/p_rs/s).

### 6.2 Free global statistic (graph_construction.md §3①)
At survey start and at the start of every round: for each sector k, the percentage of ALL voters whose
top-3 sectors (by v_i, ties broken by sector index) include k. (Sums to 300%.) Exact, current state.

### 6.3 Paid query (U8)
`Query(s1, s2, r)` with s1 ≠ s2, cost 5.00 ⚙: returns the **exact** percentage of voters in region r
with `v_i[s2] > v_i[s1]` at the current state. Allowed in the survey phase and in every campaign round.
Repeating an identical query in the same round is rejected as `DUPLICATE_QUERY` (no charge).

### 6.4 Free poll (U15)
After commit, at the start of each round and before the election: sample n_poll = 400 ⚙ voters
uniformly without replacement (poll stream); each sampled voter answers "yes" w.p. P_i;
report `p̂` and `MoE = 1.96·sqrt(p̂(1−p̂)/n_poll)`.

### 6.5 Knowledge log and information set
Every observation item (free stats, query answers, polls, own actions) is appended to the player's
**knowledge log** K_p. The information set is `I(K_p) = {worlds w : replaying w with the player's
actions yields the same K_p}`. Because answers are real-valued, the implementation uses a tolerance
(§10.1).

### 6.6 Privacy invariant
No observation, API payload, log or error message may contain voter-level v_i, α/β/γ, zealot flags,
edges, σ, τ, p_r, p_rs, s or seeds. Enforced by allowlist serialisers and recursive leak tests.

## 7. Game flow and actions (U7)

Phases: `SETUP → SURVEY → ROUND(1..T) → ELECTION → DONE`.

| Action | Phase | Cost | Effect |
|---|---|---|---|
| `query(s1,s2,r)` | SURVEY, ROUND | 5.00 | exact answer appended to K_p |
| `commit(b)` | SURVEY | 0 | fixes b_p (validated ∈ Δ, |Σ−1|<1e-6); player ready |
| `allocate(workers_by_region, extra_workers, media_regions)` | ROUND | 4.00·extra + Σ media | player ready for this round |
| `forfeit_round` | ROUND | 0 | = allocate with nothing |

- SURVEY ends when every active player has committed. A ROUND resolves (campaign effects → dynamics
  → round+1 → new free stats/poll) when every active player has allocated. After ROUND T resolves the
  ELECTION runs automatically, then DONE.
- Validation is pure; rejected actions change nothing and cost nothing (stable error codes:
  `INVALID_ACTION, INVALID_PHASE, INVALID_MANIFESTO, INVALID_ALLOCATION, INSUFFICIENT_FUNDS,
  DUPLICATE_QUERY, UNKNOWN_PLAYER, TERMINAL_STATE`).
- Treasury: one pool per player, B = 500.00 ⚙ (U14; owner set 500 on 2026-11-07). Unspent money has no value.
- Optional per-round timer is a UI/API concern: on timeout the client sends `forfeit_round`.

## 8. State, players, observations

- `GameState` is an immutable snapshot (new object per accepted action) holding: config, electorate
  (graph + attributes, on device), current v (tensor), round/phase, per-player `PlayerState`
  {treasury ledger, manifesto, knowledge log, allocation history, media airing counts, ready flag},
  RNG stream counters, public event log, per-round trace.
- `Player` protocol: `act(observation) -> Action`. Implementations: human (API), scripted strategies,
  `LLMPlayer` stub (LiteLLM-shaped interface, not implemented — U3).
- `Observation(player_id)`: JSON-safe allowlisted view built from §6.1–6.5 for that player only.

## 9. Reproducibility and compute

- **Player-chosen (N, seed) recreates the game (R8/R9):** the electorate is drawn from CPU streams by
  default (`generation = "cpu"`) and then moved to the compute device, so the same (N, seed) yields a
  bit-identical initial electorate on any machine with the same torch version (≈0.4 s at 1e5, ≈4 s at
  1e6). `generation = "device"` (GPU draws) is for batch experiments only.
- Root seed → named streams via BLAKE2b: `regions, graph, block_probs, preferences, attributes,
  zealots, workers, cascade, media, poll, election, strategy`. Each stream is a `torch.Generator` on the
  target device; per-round draws use `(stream, round, player)` sub-seeds so replays are exact.
- GPU-first (U17): all electorate/dynamics/campaign/voting ops are torch tensor ops; CPU torch is the
  reference path. Same seed ⇒ identical results on the same device; CPU↔GPU agreement is checked
  statistically (distributions) and, in float64 deterministic mode, numerically (≤ 1e-9) where RNG
  draws are generated on CPU and moved.
- Batched worlds: dynamics, campaign and voting accept a leading batch dimension B (worlds sharing
  N) for strategy rollouts, logic sampling and experiments.
- Targets: interactive N = 10k with < 1 s per round end-to-end; N = 1e5 batched experiments required;
  N = 1e6 single-world stretch on one 24 GB A5000. Device chosen at runtime; VRAM is per device.
- Every result record stores config, root seed, git revision, device, torch/CUDA versions.

## 10. Formal reasoning (temporal + epistemic)

### 10.1 Worlds and information sets
A world w = (θ = (σ, τ), micro-seed). The world sampler draws worlds from the prior conditioned on the
public facts (N, n_r, census) and keeps those **consistent** with K_p: for each query answer a_obs
about (s1,s2,r) at round t, the world's simulated answer must satisfy `|a_w − a_obs| ≤ ε` (ε = 2
percentage points ⚙). Discrete consistency of θ is decided first using precomputed per-archetype
answer distributions (exact enumeration over the 8640 θ), then micro-states are sampled. Results always
state the sample size and that knowledge is "over the sampled consistent worlds".

### 10.2 Properties (evaluated on deterministic expected share E, threshold 0.5)
- `Victory(w, π)`: policy π played in world w ends with E > 0.5 (and, separately, official sampled vote).
- `PossibleVictory(W)`: ∃π ∈ Π, ∃w ∈ W: Victory.  `GuaranteedVictory(W)`: ∃π ∈ Π ∀w ∈ W: Victory.
- `InevitableDefeat(W)`: ∀π ∈ Π ∀w ∈ W: ¬Victory.  `K_p φ` := φ holds for all w ∈ W = sample of I(K_p).
- `Viable_t`: from round-t state, ∃ continuation in Π that wins (rollouts).
- Π is a declared finite policy set (baselines + optimiser-generated plans); claims are scoped to Π.
- Temporal (per trace): `◇Lead` (poll or E > 0.5 at some round), `□(margin > −δ)`, `◇_{≤T} Victory`;
  STL robustness ρ = final margin E − 0.5 (min/max semantics for □/◇).
- **Information sufficiency:** smallest number of queries (under a query-selection policy) after which
  `K_p(GuaranteedVictory)` holds.

## 11. Analysis tools (lectures)
- Spectral: weighted Laplacian L = D − W; λ2(L) (mixing/consensus speed), per-region λ_max(A_rr)
  (cascade threshold), eigenvector centrality.
- Resilience: min degree, F-local zealot density, (r,s)-robustness lower/upper bounds via region-union
  set pairs and greedy search (exact check is coNP-hard — reported as bounds).
- Optimisation: manifesto maximising expected votes over Δ by projected gradient (Euclidean projection
  onto the simplex = prox of the indicator), with smooth surrogate for ½‖·‖₁ (Huber, δ = 1e-3);
  consensus ADMM across regions (local b_r per region, global z = b) must match centralised PG.

## 12. Balance acceptance gate (P2) — over ≥ 200 seeds at N = 10k, default config
1. Uniform manifesto (0.2 each), no campaign: win rate ≤ 30%.
2. Oracle manifesto (optimiser with true v) + planner campaign: win rate ≥ 85%.
3. Survey-informed manifesto beats uninformed manifesto (paired test, p < 0.01).
4. No fixed allocation pattern wins ≥ 80% of seeds; workers-only and media-only are each best on
   some seeds.
5. Region personality and zealot fraction change the best allocation measurably.
If a criterion fails, ⚙ parameters are retuned (and the change logged) — never the criteria silently.

## 13. Out of scope (v2.0)
Opponent logic (interface only), manifesto revision after commit, turnout model beyond abstention,
changing topology, real-world data calibration, production auth/persistence.

## 14. Calibrated defaults (v2.1, frozen after the P2 balance gate)

These supersede every ⚙ default quoted earlier in this document. Evidence: `reports/v2/balance_gate_v2.1.json`
(200 seeds, N = 10k, cuda:2, 256 s).

| Parameter | v2.0 default | **v2.1** |
|---|---|---|
| Dirichlet concentration κ (§2.3) | 20 | **60** |
| Social step ω (§4) | 0.5 | **0.15** |
| Worker contact strength η_w (§3.1) | 0.30 | **0.50** |
| Word-of-mouth transmission τ_c (§3.1) | 0.25 | **0.03** (τ_c·λ_max(A_rr)/ρ ≈ 1.1–1.7, mildly supercritical; L = 3 hops bounds reach) |
| Word-of-mouth strength η_wom | 0.10 | **0.15** |
| Media base cost m0 (§3.2) | 20.00 | **50.00** |
| Media strength η_m0 | 0.15 | **0.25** |
| Vote sensitivity β_v (§5) | 12 | **20** |
| Vote threshold z (§5) | 0.75 | **0.825** |
| Mean degree k̄, region floor, budget | 16 / 1% / 100 | **30 / 15% / 500** (owner decisions R2, R6, R7) |

Unchanged: W_free = 10, extra worker 4.00, q = 0.25%·N, ρ = 0.5, L = 3, p_m = 0.6, λ = 0.5, ζ = 0.05, query 5.00,
poll 400, ε = 0.02.

Gate results (§12): uniform manifesto with no campaign wins 6%; oracle manifesto + oracle planner wins 90%
(95% CI 85–93%); informed (knows σ) beats uninformed (free top-3 statistic) by +0.018 expected share, Wilcoxon
p = 5e−29; best fixed paid pattern wins ≤ 77.5%; the best pattern is workers on 80 seeds, split on 73,
media on 47 and depends on the largest region's personality (media-savvy → media 32/57, canvassing-receptive
→ workers 30/49); raising zealots from 0 to 30% cuts the oracle win rate from 95% to 62.5%.

Calibration insight: social averaging alone raises the expected share of a well-chosen manifesto a lot over
the five rounds (voters converge to regional means), so manifestos must be optimised against the
*anticipated* electorate (social-only dynamics), not against v(0). The oracle and informed baselines do this.
