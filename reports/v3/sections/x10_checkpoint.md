## Phase 1 owner checkpoint (D7): how to respond to X10

**For the owner. Nothing below is implemented.** Prices stay at ₫20 / ₫10 (D5). Under D7, Phase 2 waits for this choice.

Every number cites `reports/v3/x10.json` (50 seeds per cell, treasury ₫500) or `reports/v3/x0.json` (100 seeds, N = 10⁴, T = 5).

### What X10 found (the facts the decision rests on)

1. **No (N, T) cell passes the gate** (`gate_map`), and neither does the current default (X0, all treasuries).
   - The default fails criteria 4a and 4b: split wins 88–95% and media-only is never the best pattern (`x0.json` → `gate_by_treasury`).
2. **T is the main driver of failure, not N:**
   - **T = 1 fails criterion 2 at every N.** The ORACLE planner wins only 0.12 (10⁶) to 0.36 (10³) (`cells.<N|1>.summary.oracle/planner`).
   - **T = 10 fails criterion 1 at every N.** Uniform/none wins 0.38–0.44, and every paid pattern wins 0.94–1.00 (`cells.<N|10>.summary`).
   - **T ∈ {3, 5} come closest.** These cells miss a single criterion:
     - (3·10³, 3) and (10⁴, 3) miss criterion 2 (oracle/planner 0.78 and 0.74);
     - (10⁵, 5) misses criterion 4a (the best paid pattern wins exactly 0.80);
     - (10⁶, 5) misses criterion 2 (oracle/planner 0.80).
3. **The value of a worker scales with N in a way that depends on T.** Log–log slope of Δshare per worker vs N (`slopes.dshare_per_worker`):
   - −0.990 [−0.999, −0.982] at T = 1;
   - −0.751 [−0.786, −0.719] at T = 3;
   - −0.237 [−0.275, −0.205] at T = 5;
   - +0.016 [0.008, 0.023] at T = 10.

   The value of media per ₫ is flat in N at every T: all slope CIs contain 0 (`slopes.dshare_per_rupee_media`).
4. **The dominant channel flips across the grid** (`channel_dominance`):
   - **Workers dominate** at N = 10³ for every T (98–100% of seeds), and at T = 10 for every N (100%).
   - **Media dominates** at T = 1 for N ≥ 3·10³, and at T = 3 for N ≥ 10⁴ (0–30% of seeds favour workers).

### Option A — Scale the worker price with N (e.g. ₫20·N/10⁴)

- **What it fixes:** the 1/N decline of the worker's value, which holds only when the cascade is hop-limited (T = 1, slope −0.99).
- **What the numbers say against it:**
  - At T = 10 a worker's value does not depend on N (slope +0.016). There, ₫20·N/10⁴ would cost ₫2,000 at N = 10⁶, more than the whole ₫500 treasury, although a ₫20 worker already gives the same Δshare per worker at 10⁶ (5.38e-3) as at 10³ (4.84e-3) (`cells.<N|10>.channels.dshare_per_worker`).
  - At T = 5 the fitted price exponent would be about 0.24, not 1.
  - A correct formula would need a T-dependent exponent, roughly price ∝ N^{−slope(T)}, fitted from the four slopes above.
  - It also leaves the T-driven failures untouched: criterion 2 at T = 1 and criterion 1 at T = 10.
- **Cost:** a rule change, with a default-preserving test, and X0 rerun before Phase 2 (§3 checkpoint).

### Option B — Restrict the allowed N and/or T in the UI

- **What it fixes:** the structural failures at the ends of the T range.
  - T = 1: criterion 2 fails in 5 of 5 cells.
  - T = 10: criterion 1 fails in 5 of 5 cells.
- **What the numbers support:**
  - Restrict T to the middle of the range. Only T ∈ {3, 5} were measured; T = 2, 4 and 6–9 were not.
  - Optionally restrict N to [3·10³, 10⁵]. The four single-criterion cells all lie in N ∈ [3·10³, 10⁶] with T ∈ {3, 5}.
- **What it does not fix:** even inside that window no cell passes all five criteria. At the default (10⁴, 5), 4a and 4b still fail at ₫20 (X0). A restriction narrows the failure but does not remove it.
- **Cost:** a UI/API rule change (`api/app.py` bounds), with a default-preserving test.

### Option C — Document the limits ("balanced only near N = 10⁴, T = 5")

- **What it fixes:** nothing in the game. The report states the measured regimes:
  - media wins short games on large electorates;
  - workers win long games and small electorates;
  - T = 1 and T = 10 fail the gate structurally.
- **What the numbers say:** the wording "balanced near N = 10⁴, T = 5" would be inaccurate. The default cell fails 4a and 4b, both in X0 and in X10's (10⁴, 5) cell. The accurate claim is: "closest to balanced for T ∈ {3, 5}; no tested (N, T) passes all five §12 criteria at ₫20."
- **Cost:** none. Phase 2 starts directly.

### Notes for the decision

- The options can be combined. For example, B (restrict T to about 3–5) plus C (document the N-dependence), without a price formula.
- X7 (price and budget grid) runs only at N = 10⁴. It cannot test an N-scaled price unless it is extended.
- X11 (worker × media complementarity) remains the planned explanation for split dominating at the default cell.
