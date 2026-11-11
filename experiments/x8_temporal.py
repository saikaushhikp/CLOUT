"""X8 — Temporal properties of campaigns (plan ``docs/v2/plans/implementation_plan.md`` §6 X8).

Command (reproduces reports/v3/x8.json, reports/v3/raw/x8.jsonl and reports/v3/logs/x8.log; run after X6):
  OMP_NUM_THREADS=4 python experiments/x8_temporal.py --device cuda:0 --out reports/v3/x8.json \
      2>&1 | tee reports/v3/logs/x8.log

Post-hoc on the X6 traces (reports/v3/raw/x6.jsonl: 100 seeds 12000–12099 × 9 players + ORACLE = 1,000 games).

**Linear-time part (``logic.temporal``, per game).** The trace has T + 2 = 7 states: s_k = the state after k
campaign rounds (k = 0 is after the manifesto commit, before round 1; k = 1..T after each round), then the
election-day state s_{T+1}. Two views:
  * ``poll`` (the player's view, as ``logic.temporal.game_trace`` and v2): margin_k = poll_{k+1} − 0.5 (the poll
    published at the decision point / before the election), election-day margin = official vote share − 0.5;
  * ``true`` (hidden): margin_k = true expected share E_k − 0.5.
``Lead`` = margin > 0; ``Victory`` = official result (election-day state only). Formulas (positions from 0):
  ◇Lead            F[≤T] Lead           — led at some campaign state before election day;
  □(margin > −δ)   G Pred(margin, −δ)   — never more than δ behind (δ = 0.05; a δ curve is reported too);
  Comeback         ¬Lead U (Lead ∧ ◇Victory);
  STL robustness   ρ of ◇(margin > 0) on the campaign states, of □(margin > −δ), and of Comeback (winners).
The ORACLE has no polls, so it has only the ``true`` view.

**Branching part (CTL-style over batched futures, GPU).** For every game and every decision round k = 1..T the
true state before round k is rebuilt exactly on the GPU (``x_batch.run_batch``: the same electorate, the game's
manifesto, the game's recorded allocations for rounds 1..k−1 on the game's own canonical seed streams). From that
state the future branches over Π × W paths:
  Π (continuation plans, all with the game's remaining treasury; queries are not continued):
     own        the game's own remaining recorded allocations (open loop);
     planner    the ORACLE first-order planner (``balance.policy_planner``) on the true state;
     workers    even pace, workers by region size;   media  even pace, media on the largest regions;
     split      even pace, half media / half workers; allin  the whole treasury on workers now, by size;
     allin_best the whole treasury on workers now, in the best-matched region;   none  spend nothing.
  W  = 8 stream salts: salt 0 = the game's canonical streams (so (own, salt 0) replays the actual game),
       salts 1..7 re-draw every random stream from round k on (workers, cascade, media) and the election.
Victory on a path = the official election of that path. Then, at round k:
  EF Victory   = ∃ path wins (``temporal.exists_eventually``);   AF Victory = ∀ paths win (``forall_eventually``);
  ⟨⟨p⟩⟩◇Victory = ∃π ∀w wins (``strategy_eventually``);          p_best = max_π mean_w Victory.
**Point of no return** (PNR) of a lost game = the first k from which EF Victory is false at k and at every later
decision round (k = 1: lost at the commit; T + 1 = 6: still viable before the last round, lost only then).
EF/AF are over the SAMPLED Π × W tree (64 paths), not over all plans and all randomness: EF false means "no plan
in Π wins on any of 8 sampled futures".

Replay check: (own, salt 0, k = 1) re-plays the whole game; its trajectory is compared with the X6 trace (GPU sums
differ from the engine's in the last float bits, which can flip a few cascade draws). CPU↔GPU parity (§1.1.7):
p_best and the path win rate on 3 seeds × 3 players, CPU vs GPU.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x_common as xc  # noqa: E402
import x_policies as xp  # noqa: E402
from x_batch import run_batch  # noqa: E402

from dumbocracy.config import ElectorateConfig, GameConfig, SeedStreams  # noqa: E402
from dumbocracy.dynamics.campaign import Allocation  # noqa: E402
from dumbocracy.electorate import generate_electorate  # noqa: E402
from dumbocracy.logic import temporal as tl  # noqa: E402

X6_RAW = xc.RAW / "x6.jsonl"
RAW_PATH = xc.RAW / "x8.jsonl"
PLAYERS = ("uniform", "top3", "random", "people_pleaser", "bayes", "bayes_admm", "bayes_blend", "strong_arm",
           "strong_arm_random", "ORACLE")
CONT = ("own", "planner", "workers", "media", "split", "allin", "allin_best", "none")
DELTA = 0.05
DELTAS = (0.0, 0.02, 0.05, 0.10, 0.15)
NUM_SALTS = 8


# ============================================================================== linear-time part
def states_of(row: dict, view: str) -> list[dict]:
    tr = row["trace"]
    T = len(tr["expected_by_round"]) - 1
    if view == "true":
        margins = [float(x) - 0.5 for x in tr["expected_by_round"]]
    else:
        polls = sorted(tr["polls"], key=lambda p: p["round"])
        if len(polls) < T + 1:
            return []
        margins = [float(p["estimate"]) - 0.5 for p in polls[:T + 1]]
    states = [{"k": k, "Lead": m > 0, "margin": m, "Victory": False} for k, m in enumerate(margins)]
    final = float(row["vote_share"]) - 0.5
    states.append({"k": T + 1, "Lead": bool(row["victory"]), "margin": final, "Victory": bool(row["victory"])})
    return states


LEAD = tl.Atom("Lead")
LEAD_Q = tl.Pred("margin", 0.0)


def linear_props(row: dict, view: str) -> dict | None:
    s = states_of(row, view)
    if not s:
        return None
    T = len(s) - 2
    ev_lead = tl.F(LEAD, bound=T)                         # campaign states 0..T only
    never_far = tl.G(tl.Pred("margin", -DELTA))
    comeback = tl.U(tl.Not(LEAD), tl.And(LEAD, tl.F(tl.Atom("Victory"))))
    always_lead = tl.G(LEAD)
    first_lead = next((st["k"] for st in s[:T + 1] if st["Lead"]), None)
    changes = sum(1 for a, b in zip(s[:-1], s[1:]) if a["Lead"] != b["Lead"])
    out = {
        "ev_lead": tl.holds(ev_lead, s), "always_lead": tl.holds(always_lead, s),
        "never_far": tl.holds(never_far, s), "comeback": tl.holds(comeback, s),
        "behind_at_start": not s[0]["Lead"], "led_at_last_poll": s[T]["Lead"],
        "first_lead_k": first_lead, "lead_changes": changes,
        "rho_ev_lead": tl.robustness(tl.F(LEAD_Q, bound=T), s),
        "rho_never_far": tl.robustness(never_far, s),
        "rho_comeback": tl.robustness(tl.U(tl.Not(LEAD_Q), tl.And(LEAD_Q, tl.F(tl.Atom("Victory")))), s),
        "rho_final": s[-1]["margin"],
        "never_far_by_delta": {str(d): tl.holds(tl.G(tl.Pred("margin", -d)), s) for d in DELTAS},
    }
    if math.isinf(out["rho_comeback"]):
        out["rho_comeback"] = None if out["rho_comeback"] < 0 else out["rho_comeback"]
    return out


# ============================================================================== branching part
class ForkStreams:
    """Canonical ``SeedStreams(root)`` for rounds < ``k0`` (0-based); from round ``k0`` on, and for the election,
    salted re-draws when ``salt`` > 0 (salt 0 = the canonical streams throughout)."""

    def __init__(self, root: int, k0: int, salt: int):
        self.root, self.k0, self.salt = int(root), int(k0), int(salt)
        self._base = SeedStreams(self.root)

    def derive(self, stream: str, *keys: object) -> int:
        base = self._base.derive(stream, *keys)
        forked = stream == "election" or (keys and isinstance(keys[0], int) and keys[0] >= self.k0)
        if self.salt == 0 or not forked:
            return base
        digest = hashlib.blake2b(f"dumbocracy-v3-x8-fork:{base}:{self.salt}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") & ((1 << 63) - 1)

    def generator(self, stream: str, *keys: object, device: torch.device | str = "cpu") -> torch.Generator:
        gen = torch.Generator(device=torch.device(device))
        gen.manual_seed(self.derive(stream, *keys))
        return gen


def _sizes(ctx) -> list[int]:
    return [int(x) for x in ctx.electorate.sizes.tolist()]


def allin(ctx) -> Allocation:
    return Allocation(xp.split_by_weight(xp.affordable_workers(ctx, ctx.budget), [float(x) for x in _sizes(ctx)]),
                      xp.NO_MEDIA)


def allin_best(ctx) -> Allocation:
    return Allocation(xp.split_by_weight(xp.affordable_workers(ctx, ctx.budget),
                                         xp._placement_weights(ctx, "best_matched")), xp.NO_MEDIA)


CONT_POLICIES = {"planner": xp.policy_planner, "workers": xp.policy_pay_workers, "media": xp.policy_pay_media,
                 "split": xp.policy_pay_split, "allin": allin, "allin_best": allin_best, "none": xp.policy_none}


class Fork:
    """Rounds < k0: the game's recorded allocations; from k0 on: ``cont`` (``"own"`` = the recorded ones) with the
    treasury reduced by the game's query spending before round k0 (run_batch only charges allocations)."""

    def __init__(self, recorded: list[Allocation], k0: int, cont: str, query_spent: Decimal):
        self.recorded, self.k0, self.cont, self.offset = recorded, k0, cont, query_spent

    def __call__(self, ctx) -> Allocation:
        if ctx.round < self.k0 or self.cont == "own":
            return self.recorded[ctx.round] if ctx.round < len(self.recorded) else Allocation()
        ctx2 = dataclasses.replace(ctx, budget=max(Decimal("0"), ctx.budget - self.offset))
        return CONT_POLICIES[self.cont](ctx2)


def recorded_allocs(row: dict) -> list[Allocation]:
    allocs = sorted(row["trace"]["allocations"], key=lambda a: a["round"])
    return [Allocation(tuple(int(h) for h in a["hires"]), tuple(bool(m) for m in a["media"])) for a in allocs]


def query_spent_before(row: dict, k0: int) -> Decimal:
    """Queries paid before the decision of 0-based round k0 (survey + campaign queries of earlier rounds; engine
    rounds are 1-based in the trace)."""
    total = Decimal("0")
    for q in row["trace"].get("queries", []):
        if q.get("phase") == "SURVEY" or int(q.get("round", 10**9)) < k0 + 1:
            total += Decimal(str(q.get("cost", "0")))
    return total


def paths_for(row: dict, rounds: int) -> list[tuple]:
    """(row id, k0, cont, salt) items of one game."""
    return [(k0, c, s) for k0 in range(rounds) for c in CONT for s in range(NUM_SALTS)]


def run_items(items: list[tuple], electorates: dict, rows: dict, base: GameConfig) -> list[dict]:
    """items: (row_key, k0, cont, salt). One run_batch over all of them."""
    es, ms, pols, streams = [], [], [], []
    for key, k0, cont, salt in items:
        row = rows[key]
        es.append(electorates[row["seed"]])
        ms.append(row["trace"]["manifesto"])
        pols.append(Fork(recorded_allocs(row), k0, cont, query_spent_before(row, k0)))
        streams.append(ForkStreams(row["seed"], k0, salt))
    dev = es[0].preferences.device
    man = torch.tensor(ms, dtype=es[0].preferences.dtype, device=dev)
    outs = run_batch(es, base, man, pols, streams=streams)
    return [{"victory": bool(o.result.victory), "expected": float(o.result.expected_share),
             "traj": o.share_trajectory} for o in outs]


def branching_for_seed(seed: int, seed_rows: list[dict], e, base: GameConfig, batch: int, record: dict) -> list[dict]:
    rows = {r["player"]: r for r in seed_rows}
    T = base.rounds
    items = [(p, k0, c, s) for p in rows for (k0, c, s) in paths_for(rows[p], T)]
    res = xc.run_chunks(lambda ch: run_items(ch, {seed: e}, rows, base), items, batch, record=record)
    out = []
    for p, row in rows.items():
        win = torch.zeros(T, len(CONT), NUM_SALTS, dtype=torch.bool)
        replay = None
        for (pp, k0, c, s), r in zip(items, res):
            if pp != p:
                continue
            win[k0, CONT.index(c), s] = r["victory"]
            if k0 == 0 and c == "own" and s == 0:
                replay = r
        ef = [tl.exists_eventually(win[k].T) for k in range(T)]           # outcomes as [W, Π]
        af = [tl.forall_eventually(win[k].T) for k in range(T)]
        strat = [tl.strategy_eventually(win[k].T) for k in range(T)]
        p_best = [float(win[k].float().mean(1).max()) for k in range(T)]
        p_paths = [float(win[k].float().mean()) for k in range(T)]
        p_by_cont = {c: [float(win[k, i].float().mean()) for k in range(T)] for i, c in enumerate(CONT)}
        best_cont = [CONT[int(win[k].float().mean(1).argmax())] for k in range(T)]
        # point of no return: first k (1-based) from which EF is false at every later decision round
        pnr = None
        if not row["victory"]:
            pnr = T + 1
            for k in range(T - 1, -1, -1):
                if ef[k]:
                    break
                pnr = k + 1
        guar = None
        if row["victory"]:
            for k in range(T):
                if all(strat[k:]):
                    guar = k + 1
                    break
        traj = [float(x) for x in row["trace"]["expected_by_round"]]
        dev = max(abs(a - b) for a, b in zip(replay["traj"], traj))
        out.append({"seed": seed, "player": p, "victory": bool(row["victory"]), "EF": ef, "AF": af,
                    "strategy_EF": strat, "p_best": p_best, "p_paths": p_paths, "p_by_cont": p_by_cont,
                    "best_cont": best_cont, "pnr": pnr, "guarantee_round": guar,
                    "replay": {"max_abs_traj_diff": dev, "victory": replay["victory"],
                               "victory_matches": replay["victory"] == bool(row["victory"]),
                               "expected": replay["expected"], "expected_x6": row["expected"]}})
    return out


# ============================================================================== summary
def summarise(rows: list[dict], T: int) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["player"]].append(r)
    players = {}
    for p in PLAYERS:
        rs = by[p]
        wins = [r for r in rs if r["victory"]]
        losses = [r for r in rs if not r["victory"]]
        d: dict = {"games": len(rs), "wins": len(wins)}
        for view in ("poll", "true"):
            lp = [r["linear"][view] for r in rs if r["linear"].get(view)]
            if not lp:
                continue
            lw = [r["linear"][view] for r in wins if r["linear"].get(view)]
            v = {"ev_lead": xc.rate([x["ev_lead"] for x in lp]), "always_lead": xc.rate([x["always_lead"] for x in lp]),
                 "never_far": xc.rate([x["never_far"] for x in lp]),
                 "never_far_by_delta": {str(dd): xc.rate([x["never_far_by_delta"][str(dd)] for x in lp]) for dd in DELTAS},
                 "rho_ev_lead": xc.mean_ci([x["rho_ev_lead"] for x in lp]),
                 "rho_never_far": xc.mean_ci([x["rho_never_far"] for x in lp]),
                 "lead_changes": xc.mean_ci([x["lead_changes"] for x in lp])}
            if lw:
                v["winners"] = {"n": len(lw), "comeback": xc.rate([x["comeback"] for x in lw]),
                                "behind_at_start": xc.rate([x["behind_at_start"] for x in lw]),
                                "ev_lead_before_election": xc.rate([x["ev_lead"] for x in lw]),
                                "led_at_last_poll": xc.rate([x["led_at_last_poll"] for x in lw]),
                                "first_lead_k_hist": {str(k): sum(1 for x in lw if x["first_lead_k"] == k)
                                                      for k in [*range(T + 1), None]},
                                "rho_comeback": xc.mean_ci([x["rho_comeback"] for x in lw if x["rho_comeback"] is not None])}
            d[view] = v
        br = [r["branching"] for r in rs]
        d["EF_by_round"] = [xc.rate([b["EF"][k] for b in br]) for k in range(T)]
        d["AF_by_round"] = [xc.rate([b["AF"][k] for b in br]) for k in range(T)]
        d["strategy_EF_by_round"] = [xc.rate([b["strategy_EF"][k] for b in br]) for k in range(T)]
        d["p_best_by_round"] = [xc.mean_ci([b["p_best"][k] for b in br]) for k in range(T)]
        d["p_paths_by_round"] = [xc.mean_ci([b["p_paths"][k] for b in br]) for k in range(T)]
        d["p_by_cont_by_round"] = {c: [sum(b["p_by_cont"][c][k] for b in br) / len(br) for k in range(T)] for c in CONT}
        lb = [r["branching"] for r in losses]
        if lb:
            pnrs = [b["pnr"] for b in lb]
            d["losses"] = {"n": len(lb), "pnr_hist": {str(k): pnrs.count(k) for k in range(1, T + 2)},
                           "pnr": xc.mean_ci(pnrs), "pnr_median": sorted(pnrs)[len(pnrs) // 2],
                           "lost_at_commit": xc.rate([k == 1 for k in pnrs]),
                           "viable_at_round": [xc.rate([b["EF"][k] for b in lb]) for k in range(T)]}
        wb = [r["branching"] for r in wins]
        if wb:
            g = [b["guarantee_round"] for b in wb]
            d["wins_branching"] = {"n": len(wb), "guarantee_round_hist": {str(k): g.count(k) for k in [*range(1, T + 1), None]},
                                   "AF_at_round": [xc.rate([b["AF"][k] for b in wb]) for k in range(T)],
                                   "strategy_EF_at_round": [xc.rate([b["strategy_EF"][k] for b in wb]) for k in range(T)]}
        players[p] = d
    rep = [r["branching"]["replay"] for r in rows]
    replay = {"games": len(rep), "victory_matches": sum(x["victory_matches"] for x in rep),
              "max_abs_traj_diff": max(x["max_abs_traj_diff"] for x in rep),
              "mean_abs_traj_diff": sum(x["max_abs_traj_diff"] for x in rep) / len(rep),
              "games_traj_diff_gt_1e-3": sum(1 for x in rep if x["max_abs_traj_diff"] > 1e-3),
              "expected_diff": xc.mean_ci([x["expected"] - x["expected_x6"] for x in rep])}
    # pooled: winners from behind; PNR weak vs strong
    pooled_w = [r["linear"]["poll"] for r in rows if r["victory"] and r["linear"].get("poll")]
    pooled_wt = [r["linear"]["true"] for r in rows if r["victory"]]
    weak = ("uniform", "random", "strong_arm", "strong_arm_random", "people_pleaser", "top3", "bayes_blend")
    pn_weak = [r["branching"]["pnr"] for r in rows if not r["victory"] and r["player"] in weak]
    pooled = {"winners_poll": {"n": len(pooled_w), "comeback": xc.rate([x["comeback"] for x in pooled_w]),
                               "behind_at_start": xc.rate([x["behind_at_start"] for x in pooled_w])},
              "winners_true": {"n": len(pooled_wt), "comeback": xc.rate([x["comeback"] for x in pooled_wt]),
                               "behind_at_start": xc.rate([x["behind_at_start"] for x in pooled_wt])},
              "weak_losses_pnr": {"players": list(weak), "n": len(pn_weak),
                                  "hist": {str(k): pn_weak.count(k) for k in range(1, T + 2)},
                                  "pnr": xc.mean_ci(pn_weak) if pn_weak else None,
                                  "share_pnr_2_or_3": xc.rate([k in (2, 3) for k in pn_weak]) if pn_weak else None,
                                  "share_pnr_1": xc.rate([k == 1 for k in pn_weak]) if pn_weak else None}}
    # tests: knowledge → outcome (does the round-1 EF/strategy verdict predict the result?), Holm over the family
    tests = []
    for name, key, k in (("P(win | strategy_EF at round 1)", "strategy_EF", 0), ("P(win | EF at round 3)", "EF", 2),
                         ("P(win | EF at round 5)", "EF", T - 1)):
        yes = [r["victory"] for r in rows if r["branching"][key][k]]
        no = [r["victory"] for r in rows if not r["branching"][key][k]]
        tests.append({"name": name, "given_true": xc.rate(yes) if yes else None, "given_false": xc.rate(no) if no else None})
    # paired: BayesPlanner vs People Pleaser p_best at rounds 1 and 3
    by_sp = {(r["seed"], r["player"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    fam = []
    for a, b in (("bayes", "people_pleaser"), ("people_pleaser", "strong_arm"), ("bayes", "top3")):
        for k in (0, 2):
            fam.append({"a": a, "b": b, "round": k + 1,
                        "p_best": xc.paired([by_sp[(s, a)]["branching"]["p_best"][k] for s in seeds],
                                            [by_sp[(s, b)]["branching"]["p_best"][k] for s in seeds])})
    for x, hp in zip(fam, xc.holm([x["p_best"]["wilcoxon_p"] for x in fam])):
        x["p_best"]["holm_p"] = hp
    return {"players": players, "replay": replay, "pooled": pooled,
            "tests": {"knowledge_vs_outcome": tests,
                      "P_best_paired": {"family": "p_best (max_π mean_w Victory) paired per seed, rounds 1 and 3, "
                                                  "Wilcoxon, Holm over the 6 tests", "tests": fam}}}


# ============================================================================== parity
def parity(rows_by_seed: dict, seeds: list[int], device: torch.device, base_gpu: GameConfig) -> dict:
    cache: dict = {}

    def fn(seed, dev):
        key = (seed, str(dev))
        if key not in cache:
            base = dataclasses.replace(base_gpu, device=str(dev))
            e = generate_electorate(dataclasses.replace(base, seed=seed), device=dev)
            sub = [r for r in rows_by_seed[seed] if r["player"] in ("bayes", "people_pleaser", "top3")]
            out = branching_for_seed(seed, sub, e, base, 512, {})
            cache[key] = {f"{o['player']}_{m}_k{k + 1}": o[m][k] for o in out for m in ("p_best", "p_paths")
                          for k in (0, 2)}
        return cache[key]

    return xc.parity_check(fn, seeds, device=device, abs_tol=0.1)


# ============================================================================== main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-gb", type=float, default=4.0, help="hard cap on this process's GPU memory")
    parser.add_argument("--seeds", type=int, default=100, help="first n X6 seeds (pilot: fewer)")
    parser.add_argument("--batch", type=int, default=0, help="worlds per run_batch (0 = calibrate)")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--out", default=str(xc.REPORTS / "x8.json"))
    parser.add_argument("--raw", default=str(RAW_PATH))
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":   # hard pin; never auto-select another GPU (shared host)
        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gb * 2**30 / total), device)
    free0 = xc.free_bytes(device)
    x6 = xc.read_rows(X6_RAW)
    seeds_all = sorted({r["seed"] for r in x6})
    seeds = seeds_all[:args.seeds]
    by_seed = defaultdict(list)
    for r in x6:
        if r["seed"] in seeds:
            by_seed[r["seed"]].append(r)
    base = GameConfig(device=str(device), electorate=ElectorateConfig(num_voters=10_000))
    T = base.rounds
    deviations = ["Device cuda:0 (hard-pinned, ≤ 4 GB) instead of the plan default cuda:2 (used by another agent).",
                  "EF/AF Victory are over a sampled tree: 8 continuation plans × 8 stream salts = 64 paths per "
                  "(game, round), not all plans and all randomness."]
    record: dict = {}
    start = time.perf_counter()
    out_rows = []
    with xc.GpuMonitor(device) as monitor:
        if args.batch:
            batch = args.batch
            record["calibration"] = {"batch": batch, "note": "given on the command line"}
        else:
            e0 = generate_electorate(dataclasses.replace(base, seed=seeds[0]), device=device)
            rows0 = {r["player"]: r for r in by_seed[seeds[0]]}
            probe_items = [("bayes", 0, "planner", s) for s in range(64)]
            cal = xc.calibrate_batch(lambda B: run_items(probe_items[:B] * 1, {seeds[0]: e0}, rows0, base), device,
                                     sizes=(8, 32), max_batch=640)
            batch = cal["batch"]
            record["calibration"] = cal
        print(f"[x8] device {device}, seeds {seeds[0]}..{seeds[-1]}, batch {batch}, paths/game "
              f"{T * len(CONT) * NUM_SALTS}", flush=True)
        for i, seed in enumerate(seeds):
            t0 = time.perf_counter()
            e = generate_electorate(dataclasses.replace(base, seed=seed), device=device)
            br = {(o["seed"], o["player"]): o for o in branching_for_seed(seed, by_seed[seed], e, base, batch, record)}
            for r in by_seed[seed]:
                out_rows.append({"exp": "X8", "seed": seed, "player": r["player"], "victory": bool(r["victory"]),
                                 "expected": r["expected"], "vote_share": r["vote_share"],
                                 "linear": {v: linear_props(r, v) for v in ("poll", "true")},
                                 "branching": br[(seed, r["player"])]})
            del e
            batch = record.get("final_batch", batch)     # keep the OOM back-off across seeds
            print(f"[x8] seed {seed} ({i + 1}/{len(seeds)}) {time.perf_counter() - t0:.1f}s", flush=True)
        par = parity(by_seed, seeds[:args.parity_seeds], device, base) if args.parity_seeds else None
        if par is not None:
            print(f"[x8] parity passed: {par['passed']}", flush=True)
    seconds = time.perf_counter() - start
    raw = Path(args.raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("")
    for r in out_rows:
        xc.write_row(raw, r)
    summary = summarise(out_rows, T)
    report = {
        "experiment": "X8", "question": "Temporal properties of campaigns (LTL/STL on the X6 traces; CTL-style EF/AF "
                                        "Victory over batched futures; point of no return)",
        "source": "reports/v3/raw/x6.jsonl", "seeds": {"first": seeds[0], "count": len(seeds), "list": seeds},
        "n": 10_000, "rounds": T, "treasury": str(base.budget),
        "definitions": {
            "states": "s_0 = after the manifesto commit, s_k = after k campaign rounds (k=1..T), s_{T+1} = election day",
            "views": {"poll": "margin_k = poll published at decision point k+1 (k=T: last pre-election poll) − 0.5; "
                              "as logic.temporal.game_trace and v2", "true": "margin_k = true expected share E_k − 0.5"},
            "Lead": "margin > 0 (election day: Victory)", "Victory": "official result, election-day state only",
            "ev_lead": "F[<=T] Lead from s_0 (led at some campaign state before election day)",
            "always_lead": "G Lead", "never_far": f"G (margin > -{DELTA})",
            "comeback": "(not Lead) U (Lead and F Victory)", "behind_at_start": "not Lead at s_0",
            "rho_ev_lead": "STL robustness of F[<=T](margin > 0) = max campaign margin",
            "rho_never_far": f"robustness of G(margin > -{DELTA}) = min margin + {DELTA}",
            "rho_comeback": "robustness of (margin<=0) U (margin>0 and F Victory); winners only",
            "futures": "from the true state before decision round k (rebuilt by replaying rounds 1..k-1 on the game's "
                       "canonical streams), Π × W paths: Π = " + ", ".join(CONT) + f"; W = {NUM_SALTS} salts "
                       "(salt 0 = canonical streams, others re-draw all streams from round k and the election)",
            "EF": "exists path with Victory (temporal.exists_eventually)", "AF": "all paths Victory (forall_eventually)",
            "strategy_EF": "exists π, all w: Victory (strategy_eventually)", "p_best": "max_π mean_w Victory",
            "pnr": "lost games: first decision round k from which EF is false at k and at all later rounds; "
                   "1 = lost at the commit, T+1 = viable before the last round",
            "guarantee_round": "won games: first k from which strategy_EF holds at k and all later rounds",
            "weak_players": "uniform, random, strong_arm, strong_arm_random, people_pleaser, top3, bayes_blend",
        },
        **summary,
        "deviations": deviations, "config": base.to_record(),
        "provenance": xc.provenance_v3(args, device, seconds, batch_sizes={"run_batch_worlds": record.get("batch_sizes"),
                                                                            "calibration": record.get("calibration")},
                                       free_at_start=free0, monitor=monitor, parity=par),
    }
    xc.save(args.out, report)
    print(json.dumps({p: {"EF": [round(x["rate"], 2) for x in v["EF_by_round"]],
                          "pnr": v.get("losses", {}).get("pnr_hist")} for p, v in summary["players"].items()}), flush=True)
    print(f"[x8] done in {seconds / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
