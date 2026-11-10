# Dumbocracy v2 — Decision Register

Decisions U1–U17 were taken by the project owner on 2026-11-07 during v2 planning. The technical
spec cites them. Changing one requires a new row here and a spec update.

| ID | Topic | Decision |
|---|---|---|
| U1 | Repo | v1 archived (git tag `v1-final`, `docs/archive/v1`, `reports/archive/v1`); fresh package in place on branch `v2-revamp` |
| U2 | Manifesto | Player designs a 5-sector budget manifesto b **once**, after surveying |
| U3 | Players | Single player now; engine is player-keyed; `Player` protocol with a LiteLLM-style stub for a future opponent |
| U4 | Graph | Intra-region pairs much likelier to connect than inter-region pairs (superseded in detail by R4/R5) |
| U5 | Dynamics | Preferences v_i evolve via weighted consensus on the simplex; campaigns pull v_i toward b |
| U6 | Voting | Seeded Bernoulli vote with P = σ(β(S − z)), S = 1 − TV(v,b); win iff votes > N/2 |
| U7 | Flow | Survey → commit b → T campaign rounds (default 5, range 1–10) → election |
| U8 | Paid query | (S1, S2, Region) → exact region % preferring S2 over S1, current state, cost 5 |
| U9 | Workers | Fixed free workers per round + paid extra workers; player assigns them to regions |
| U10 | Media | Single channel, region-targetable, cost scales with region size |
| U11 | Archetypes | 6 hand-designed archetypes; 4 drawn without replacement, one per region |
| U12 | Voter attributes | Social α, media β, campaign γ susceptibilities; zealots |
| U13 | Lectures | Consensus + resilient consensus, SI/SIR, prox/ADMM, temporal + epistemic logic are all in scope |
| U14 | Economy | One treasury per player for the whole game |
| U15 | Round info | Free noisy poll each round; paid queries available throughout |
| U16 | UI | 4-region map + WebGL sampled graph; optional round timer |
| U17 | Scale | Interactive N = 10k; batched GPU experiments to 1e5 (required), 1e6 (stretch) |

## Implementation decisions (made by the implementer under U1's freedom)

| ID | Decision | Rationale |
|---|---|---|
| I1 | torch is the single numeric code path (CPU torch = reference) | Avoids the duplicated NumPy/torch paths of v1; CPU↔GPU parity is checked within one implementation |
| I2 | pytest replaces unittest | Fixtures, markers (`gpu`, `slow`) |
| I3 | Region-personality library (4 personalities, permuted) | Implements ChatGPT "community personalities"; gives the hidden discrete type θ = (σ, τ) |
| I4 | Free global top-3 stat refreshed every round | It is free and exact in the source design; refreshing makes drift visible |
| I5 | Logic evaluated on expected share E | Deterministic per world; the official sampled vote is reported alongside |

## Parameter changes after calibration (P2)
| ID | Change | Evidence |
|---|---|---|
| C1 | κ 20→60, ω 0.5→0.15, η_w 0.30→0.50, η_wom 0.10→0.15, m0 20→50, η_m0 0.15→0.25, β_v 12→20, z 0.75→0.825; τ_c fixed at 0.03 (R10) | spec §14; `reports/v2/balance_gate_v2.1.json` — all five §12 criteria pass |

## Owner review of spec v2.0 (2026-11-07)
| ID | Decision |
|---|---|
| R1 | Archetype + personality libraries (spec §2.3–2.4) approved |
| R2 | Starting treasury B = 500.00 (other costs re-scaled in P2 calibration) |
| R3 | Graph view during play shows structure/regions only; per-voter alignment revealed at DONE |
| R4 | p_r / p_g are ER connection probabilities (not weights); edges unweighted |
| R6 | Mean degree k̄ = 30 friends per voter (max requested) |
| R7 | Region sizes arbitrary with every region ≥ 15% of N; Σ = N |
| R8 | Electorate drawn from CPU seed streams by default (bit-identical across devices), then moved to GPU |
| R9 | The player chooses N (num_voters) and the seed at game creation; (N, seed) recreates the initial electorate |
| R11 | The game is a **web game only** (Next.js + React in `web/`, served on 127.0.0.1:61007); no CLI. Python engine = FastAPI on 127.0.0.1:61008 behind Next rewrites; contract `docs/v2/web_api_v2.md`; launcher `scripts/run_game.sh` |
| R12 | Paid queries are **sampled surveys**: a fresh 10% random sample of the region per query, answer = sample % ± 95% MoE; repeats allowed and pooled within the same round (supersedes U8's exact answers; owner, 2026-11-08) |
| R13 | Information model kept as is after R12 (free exact top-3 stat; 10% sampled queries; distinct archetypes). Archetypes are cheap to learn (≈2 queries); P5's information-sufficiency analysis focuses on what stays hidden: region personalities τ, zealots and the network (owner, 2026-11-08) |
| R14 | P5 logic surfaces in BOTH the web debrief (post-game per-round epistemic timeline, computed once at DONE) and the research layer (scenario table, information-sufficiency curve, reports) (owner, 2026-11-08) |
| R15 | Debrief network colouring bins each voter's vote probability P_i (edges 10/40/60/90%) instead of equal similarity fifths (which showed ~100% "supportive" even in heavy losses) (lead, after the P6 browser walkthrough, 2026-11-08) |
| R16 | Hired party workers are **permanent**: a one-time fee (₫4.00) when hired, then they work every later round like the 10 free workers (owner, 2026-11-09). Balance gate and P7 experiments were measured under the old per-round hires (spec v2.1) and will be re-run after further mechanics changes (owner) |
| R17 | The player chooses the starting treasury at setup: ₫500–₫1000 (default 500) |
| R18 | The sampled network graph is fetched and rendered only in the post-election debrief, not during rounds (saves frontend rendering) |
| R19 | Observation carries `round_reports` and `workers`; the UI lets the player tap any past round to review its poll, orders, spend and outcomes |
| R20 | "Blend my hunches" lets the player pick which regions (2–4) to blend, weighted by voters |
| R21 | **No free workers**: every party worker is bought (one-time fee) and permanent; supersedes the free-worker part of U9 and the R16 reassignment model (owner, 2026-11-09; plan `plans/P8_party_workers_ic.md`) |
| R22 | A bought worker is a uniformly random voter of the region the player chose, with ≥ 3 friends, not a worker or zealot (seeded `workers` stream); it never changes region and is never reassigned |
| R23 | Workers are pinned at the manifesto b for the whole game (pro-party stubborn agents); they still feed b into friends' averages every round |
| R24 | Independent Cascade, **one hop per round**: spreaders in round t = workers bought in t ∪ voters converted in t−1; each tries every friend once, then goes quiet; a worker spreads only in its purchase round |
| R25 | Conversion chance for friend j = γ_j · P_j(vote for b) at round start; a convert is set to v_j = b. (Chance = γ alone let one ₫4 worker convert ~65% of 10k voters, so the manifesto stopped mattering; owner chose γ·P, 2026-11-09) |
| R26 | Converts are ordinary voters again: they drift with friends and media and spread only in the next round |
| R27 | At most 3 conversion attempts per voter per game (failures count); workers and zealots are never targets |
| R28 | Media mechanism unchanged (already of the §5.1 form) |
| R29 | Worker fee **₫50.00** (owner, 2026-11-10, after the P8 price check; ₫100 was the price passing all five balance checks, ₫50 fails 4a/4b: "split workers+media" wins 82%) |
| R30 | Prices fixed by the owner (2026-11-10): worker fee **₫20.00** (supersedes R29's ₫50), survey query **₫10.00** (was ₫5.00, U8). Balance gate to be re-measured in the new experiment suite |
| R10 | Word-of-mouth transmission τ_c calibrated near the SIR threshold ρ/λ_max(A_rr) (provisional 0.03; final value from P2) |
| R5 | ER-SBM with one global sparsity factor s: P_rr = s·p_r, P_rs = s·p_g, s set for mean degree k̄ = 16 ⚙; ranges p_r ∈ [0.5,0.9], p_g ∈ [0.01,0.2] |
