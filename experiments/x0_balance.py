"""X0 — Balance gate at the new prices (plan §3 X0; spec §12 gate; D5 prices ₫20 worker / ₫10 query).

Command (reproduces reports/v3/x0.json and reports/v3/raw/x0.jsonl):
  OMP_NUM_THREADS=4 python experiments/x0_balance.py --seeds 100 --first-seed 10000 --device cuda:2 \
      --treasuries 500 750 1000 --out reports/v3/x0.json

The five §12 gate criteria are computed exactly as ``balance.run`` computes them (same policies, the same
manifesto × policy runs, the same tie-breaks and thresholds), but every game is played through the batched
true-world runner ``x_batch.run_batch`` on the GPU (bit-identical to ``simulate_world`` on the CPU; checked
statistically CPU↔GPU here, §1.1 point 7).

Manifestos (as ``balance.run``): uniform; top3 (global top-3 of v(0)); informed (``oracle_manifesto`` on the
library's ``synthetic_world(..., known_assignment=True, salt=1)``); oracle (``oracle_manifesto`` on the true
world). GPU first: the synthetic world is drawn by the library on the CPU exactly as ``informed_manifesto``
does, then moved to the device, where the manifesto optimiser runs (the pilot measured agreement with the
CPU library path to ~1e-7 per coordinate, and 50× faster at N = 10⁶).

Manifestos do not depend on the treasury, so each seed's electorate and manifestos are computed once and the
runs are replayed at every treasury (paired seeds across treasuries, §1.2).

The module also exports the shared pieces used by X10 (``x10_n_t_balance.py``): ``RUNS``, ``prepare_chunk``,
``play_runs``, ``gate_of`` and ``chunk_plan``.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time
from decimal import Decimal
from pathlib import Path

import torch

import x_common as xc
from balance import PAID_PATTERNS, POLICIES, oracle_manifesto, synthetic_world, top3_manifesto
from dumbocracy.config import ElectorateConfig, GameConfig
from dumbocracy.electorate import generate_electorate
from dumbocracy.electorate.attributes import Susceptibilities
from dumbocracy.electorate.electorate import _to
from dumbocracy.sectors import PERSONALITIES
from dumbocracy.simulate import simulate_world
from x_batch import run_batch

MANIFESTOS = ("uniform", "top3", "informed", "oracle")
# exactly the runs of balance.run (same order)
RUNS: list[tuple[str, str]] = [("uniform", "none"), ("uniform", "planner"), ("top3", "planner"), ("informed", "planner"),
                               ("oracle", "none"), ("oracle", "planner")] + \
                              [("informed", p) for p in ("free_even",) + PAID_PATTERNS]
GATE_KEYS = ["1_uniform_no_campaign_le_30pct", "2_oracle_planner_ge_85pct", "3_informed_beats_top3_p_lt_0.01",
             "4a_no_paid_pattern_ge_80pct", "4b_workers_and_media_each_best_sometimes"]


# ============================================================================== world preparation
def move_electorate(e, device: torch.device):
    s = e.susceptibility
    return dataclasses.replace(e, region_of=e.region_of.to(device), graph=_to(e.graph, device),
                               preferences=e.preferences.to(device),
                               susceptibility=Susceptibilities(s.social.to(device), s.media.to(device),
                                                               s.campaign.to(device), s.zealot.to(device)))


def seed_config(base: GameConfig, seed: int, rounds: int | None = None) -> GameConfig:
    cfg = dataclasses.replace(base, seed=int(seed))
    return dataclasses.replace(cfg, rounds=rounds) if rounds is not None else cfg


@dataclasses.dataclass
class Prepared:
    seed: int
    electorate: object
    manifestos: dict          # rounds -> {name: tensor [5]}
    meta: dict


def prepare_chunk(seeds, base: GameConfig, device: torch.device, rounds_list=(5,)) -> list[Prepared]:
    """Electorates (``base.generation``) and the four manifestos per T for a chunk of seeds, on ``device``."""
    out = []
    for seed in seeds:
        cfg = seed_config(base, seed)
        e = generate_electorate(cfg, device=device)
        synth = move_electorate(synthetic_world(e, cfg, known_assignment=True, salt=1), device)  # library draws (CPU)
        v0 = e.preferences
        uniform = torch.full((5,), 0.2, dtype=v0.dtype, device=device)
        top3 = top3_manifesto(v0)
        mans = {}
        for T in rounds_list:
            cfg_t = seed_config(base, seed, T)
            mans[T] = {"uniform": uniform, "top3": top3, "informed": oracle_manifesto(synth, cfg_t).to(device),
                       "oracle": oracle_manifesto(e, cfg_t)}
        del synth
        meta = {"personality": [PERSONALITIES[int(i)].id for i in e.personality_of_region],
                "sizes": [int(x) for x in e.sizes.tolist()]}
        out.append(Prepared(int(seed), e, mans, meta))
    return out


def summarize_outcome(o, worker_cost: Decimal) -> dict:
    hires = sum(sum(a.hires_by_region) for a in o.allocations)
    airings = sum(sum(a.media_regions) for a in o.allocations)
    spent_workers = worker_cost * hires
    return {"share": o.result.vote_share, "expected": o.result.expected_share, "victory": bool(o.result.victory),
            "spent": str(o.spent), "spent_workers": str(spent_workers), "spent_media": str(o.spent - spent_workers),
            "workers": int(hires), "airings": int(airings), "converted_total": int(sum(o.converted)),
            "converted": o.converted, "trajectory": [round(x, 6) for x in o.share_trajectory]}


def play_runs(prepared: list[Prepared], cfg: GameConfig, rounds: int, runs=RUNS, batch: int | None = None,
              record: dict | None = None) -> dict[int, dict[str, dict]]:
    """All ``runs`` for the prepared worlds at ``cfg`` (treasury, rounds); batched over worlds."""
    cfg = dataclasses.replace(cfg, rounds=rounds)
    results: dict[int, dict[str, dict]] = {p.seed: {} for p in prepared}
    batch = batch or len(prepared)
    for man, pol in runs:
        def fn(chunk):
            outs = run_batch([p.electorate for p in chunk], cfg, torch.stack([p.manifestos[rounds][man] for p in chunk]),
                             POLICIES[pol], seeds=[p.seed for p in chunk])
            return list(zip(chunk, outs))
        local: dict = {}
        for p, o in xc.run_chunks(fn, prepared, batch, record=local):
            results[p.seed][f"{man}/{pol}"] = summarize_outcome(o, cfg.workers.worker_cost)
        if record is not None:   # accumulate over calls (run_chunks overwrites its record per call)
            record["batch_sizes"] = sorted(set(record.get("batch_sizes", [])) | set(local.get("batch_sizes", [])))
            if local.get("oom_backoffs"):
                record.setdefault("oom_backoffs", []).extend(local["oom_backoffs"])
    return results


# ============================================================================== gate (as balance.run)
def gate_of(rows: list[dict]) -> dict:
    """``rows``: one dict per seed with ``runs`` (key -> summary), ``personality``, ``sizes``. Returns the
    balance.run summary (same criteria) plus Wilson CIs, paired effect sizes and discordant counts."""
    from scipy.stats import wilcoxon
    n = len(rows)
    keys = sorted(rows[0]["runs"])
    summary = {}
    for k in keys:
        wins = [r["runs"][k]["victory"] for r in rows]
        exp = [r["runs"][k]["expected"] for r in rows]
        summary[k] = {**xc.rate(wins), "mean_expected_share": sum(exp) / n, "expected_ci95": xc.mean_ci(exp)["ci95"]}
    inf = [r["runs"]["informed/planner"]["expected"] for r in rows]
    top = [r["runs"]["top3/planner"]["expected"] for r in rows]
    diffs = [a - b for a, b in zip(inf, top)]
    p_value = float(wilcoxon(diffs, alternative="greater").pvalue) if any(d != 0 for d in diffs) else 1.0
    best_counts = {p: 0 for p in PAID_PATTERNS}
    best_by_personality: dict[str, dict[str, int]] = {}
    best_per_seed = {}
    for r in rows:
        best = max(PAID_PATTERNS, key=lambda p: r["runs"][f"informed/{p}"]["expected"])
        best_counts[best] += 1
        best_per_seed[str(r["seed"])] = best
        largest = r["personality"][max(range(4), key=lambda i: r["sizes"][i])]
        best_by_personality.setdefault(largest, {p: 0 for p in PAID_PATTERNS})[best] += 1
    max_pattern = max(PAID_PATTERNS, key=lambda p: summary[f"informed/{p}"]["rate"])
    max_pattern_rate = summary[f"informed/{max_pattern}"]["rate"]
    gate = {
        GATE_KEYS[0]: summary["uniform/none"]["rate"] <= 0.30,
        GATE_KEYS[1]: summary["oracle/planner"]["rate"] >= 0.85,
        GATE_KEYS[2]: p_value < 0.01 and sum(diffs) > 0,
        GATE_KEYS[3]: max_pattern_rate < 0.80,
        GATE_KEYS[4]: (best_counts["pay_workers"] + best_counts["front_workers"]) > 0 and best_counts["pay_media"] > 0,
    }
    paired_eff = xc.paired(inf, top)
    paired_eff["wilcoxon_p_one_sided_greater"] = p_value
    return {"num_seeds": n, "gate": gate, "gate_passed_all": all(gate.values()),
            "gate_failed": [k for k, v in gate.items() if not v], "summary": summary,
            "informed_vs_top3": {"mean_diff": sum(diffs) / n, "wilcoxon_p": p_value, "paired": paired_eff,
                                 "mcnemar_wins": xc.mcnemar([r["runs"]["informed/planner"]["victory"] for r in rows],
                                                            [r["runs"]["top3/planner"]["victory"] for r in rows])},
            "max_paid_pattern": {"pattern": max_pattern, "rate": max_pattern_rate,
                                 "ci95": summary[f"informed/{max_pattern}"]["ci95"]},
            "best_paid_pattern_counts": best_counts, "best_pattern_by_largest_region_personality": best_by_personality,
            "best_paid_pattern_per_seed": best_per_seed}


# ============================================================================== batch sizing
def chunk_plan(base: GameConfig, device: torch.device, n_seeds: int, probe_seeds, rounds: int, log=print) -> dict:
    """Seeds per preparation chunk and games per run_batch call, from free memory (§1.1 point 3):
    input bytes per world (electorate on the device) + ``calibrate_batch`` of run_batch (planner)."""
    if device.type != "cuda":
        return {"chunk": n_seeds, "batch": n_seeds}
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(device)
    probe = prepare_chunk(probe_seeds[:2], base, device, (rounds,))
    torch.cuda.synchronize(device)
    input_bytes = max(1, (torch.cuda.memory_allocated(device) - before) // len(probe))
    cfg = dataclasses.replace(base, rounds=rounds)

    def run(b):
        run_batch([p.electorate for p in probe[:b]], cfg, torch.stack([p.manifestos[rounds]["oracle"] for p in probe[:b]]),
                  POLICIES["planner"], seeds=[p.seed for p in probe[:b]])
    cal = xc.calibrate_batch(run, device, sizes=(1, 2))
    del probe
    torch.cuda.empty_cache()
    free = xc.free_bytes(device)
    per_world = input_bytes + cal["per_item_bytes"]
    chunk = int(max(1, min(n_seeds, (0.70 * free - cal["fixed_bytes"]) // per_world)))
    plan = {"chunk": chunk, "batch": chunk, "input_bytes_per_world": int(input_bytes), "calibration": cal,
            "free_bytes": int(free)}
    log(f"[batch] N={base.electorate.num_voters}: input {input_bytes / 2**20:.1f} MiB/world, run "
        f"{cal['per_item_bytes'] / 2**20:.1f} MiB/world, free {free / 2**30:.1f} GiB -> chunk/B = {chunk}")
    return plan


# ============================================================================== parity (§1.1 point 7)
def parity(base: GameConfig, seeds, device: torch.device) -> dict:
    """CPU vs GPU, statistical (spec §9), for informed/planner and informed/pay_split; plus run_batch on the CPU
    vs simulate_world on the CPU (bit-identical expected)."""
    def fn(seed: int, dev: torch.device) -> dict:
        cfg = dataclasses.replace(base, device=str(dev))
        p = prepare_chunk([seed], cfg, dev, (cfg.rounds,))[0]
        out = {}
        for pol in ("planner", "pay_split"):
            o = run_batch([p.electorate], cfg, p.manifestos[cfg.rounds]["informed"].unsqueeze(0), POLICIES[pol],
                          seeds=[seed])[0]
            out[f"{pol}_expected"] = o.result.expected_share
            out[f"{pol}_victory"] = float(o.result.victory)
            out[f"{pol}_converted"] = float(sum(o.converted))
        return out
    rec = xc.parity_check(fn, seeds, device=device, abs_tol=0.01)
    for k, m in rec["metrics"].items():   # converted counts: tolerance in voters (as in gpu_benchmarks.json)
        if k.endswith("_converted") and not m["passed"]:
            m["note"] = "voters"
    cpu = torch.device("cpu")
    identical = 0
    for seed in seeds:
        cfg = dataclasses.replace(base, device="cpu", seed=int(seed))
        p = prepare_chunk([seed], cfg, cpu, (cfg.rounds,))[0]
        m = p.manifestos[cfg.rounds]["informed"]
        a = run_batch([p.electorate], cfg, m.unsqueeze(0), POLICIES["planner"], seeds=[seed])[0]
        b = simulate_world(p.electorate, cfg, m, POLICIES["planner"])
        identical += int(a.share_trajectory == b.share_trajectory and a.result.expected_share == b.result.expected_share
                         and a.result.victory == b.result.victory)
    rec["cpu_run_batch_vs_simulate_world_bit_identical"] = f"{identical}/{len(seeds)}"
    rec["passed"] = bool(rec["passed"] and identical == len(seeds))
    return rec


# ============================================================================== zealot sweep (not a gate criterion)
def zealot_sweep(base: GameConfig, seeds, fracs, device: torch.device, batch: int) -> dict:
    out = {}
    for frac in fracs:
        cfg = dataclasses.replace(base, electorate=dataclasses.replace(base.electorate, zealot_fraction=frac))
        es, ms = [], []
        for s in seeds:
            c = seed_config(cfg, s)
            e = generate_electorate(c, device=device)
            es.append(e)
            ms.append(oracle_manifesto(e, c))
        outs = []
        for i in range(0, len(seeds), batch):
            outs += run_batch(es[i:i + batch], cfg, torch.stack(ms[i:i + batch]), POLICIES["planner"],
                              seeds=list(seeds[i:i + batch]))
        gains = [o.share_trajectory[-1] - o.share_trajectory[0] for o in outs]
        out[str(frac)] = {**xc.rate([o.result.victory for o in outs]), "win_rate": sum(o.result.victory for o in outs) / len(outs),
                          "mean_campaign_gain": sum(gains) / len(gains), "campaign_gain": xc.mean_ci(gains)}
        del es
    return out


# ============================================================================== main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=10000)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--treasuries", nargs="+", default=["500", "750", "1000"])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--zealots", type=float, nargs="*", default=[0.0, 0.05, 0.15, 0.30])
    parser.add_argument("--zealot-seeds", type=int, default=40)
    parser.add_argument("--zealot-treasury", default="500")
    parser.add_argument("--parity-seeds", type=int, default=3)
    parser.add_argument("--set", action="append", default=[], help="config override a.b=v")
    parser.add_argument("--out", default=str(xc.REPORTS / "x0.json"))
    parser.add_argument("--raw", default=str(xc.RAW / "x0.jsonl"))
    args = parser.parse_args()

    start = time.perf_counter()
    device, dev_record = xc.select_device(args.device, need_bytes=4 * 2**30)
    free_at_start = xc.free_bytes(device)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    assert args.first_seed != 10000 or seeds == xc.seed_block("X0", args.seeds)
    base = GameConfig(device=str(device), rounds=args.rounds, electorate=ElectorateConfig(num_voters=args.n))
    base = xc.apply_overrides(base, args.set)
    raw = Path(args.raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("")
    record: dict = {}
    with xc.GpuMonitor(device) as monitor:   # no outer no_grad: the manifesto optimiser needs autograd
        print(f"[x0] device {device}; parity on {args.parity_seeds} seeds", flush=True)
        par = parity(dataclasses.replace(base, budget=Decimal(args.treasuries[0])), seeds[:args.parity_seeds], device)
        print(f"[x0] parity passed={par['passed']} ({par['cpu_run_batch_vs_simulate_world_bit_identical']} bit-identical "
              "CPU run_batch vs simulate_world)", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        plan = chunk_plan(base, device, len(seeds), seeds, args.rounds)
        t0 = time.perf_counter()
        per_treasury_rows: dict[str, list[dict]] = {t: [] for t in args.treasuries}
        for i in range(0, len(seeds), plan["chunk"]):
            chunk = prepare_chunk(seeds[i:i + plan["chunk"]], base, device, (args.rounds,))
            for treasury in args.treasuries:
                cfg = dataclasses.replace(base, budget=Decimal(treasury))
                res = play_runs(chunk, cfg, args.rounds, batch=plan["batch"], record=record)
                for p in chunk:
                    per_treasury_rows[treasury].append({"seed": p.seed, **p.meta, "runs": res[p.seed]})
                    for key, row in res[p.seed].items():
                        man, pol = key.split("/")
                        xc.write_row(raw, {"experiment": "X0", "treasury": treasury, "n": args.n, "rounds": args.rounds,
                                           "seed": p.seed, "manifesto": man, "policy": pol, **row})
            print(f"[x0] seeds {i + len(chunk)}/{len(seeds)} done, {time.perf_counter() - t0:.0f}s", flush=True)
            del chunk
        games_seconds = time.perf_counter() - t0
        gates = {t: gate_of(rows) for t, rows in per_treasury_rows.items()}
        holm_p = xc.holm([gates[t]["informed_vs_top3"]["wilcoxon_p"] for t in args.treasuries])
        for t, p in zip(args.treasuries, holm_p):
            gates[t]["informed_vs_top3"]["wilcoxon_p_holm"] = p
        zealot = {}
        if args.zealots and args.zealot_seeds > 0:
            print(f"[x0] zealot sweep: {args.zealots} x {args.zealot_seeds} seeds at treasury {args.zealot_treasury}",
                  flush=True)
            zcfg = dataclasses.replace(base, budget=Decimal(args.zealot_treasury))
            zealot = zealot_sweep(zcfg, seeds[:args.zealot_seeds], args.zealots, device, plan["batch"])
    seconds = time.perf_counter() - start
    prov = xc.provenance_v3(args, device, seconds, batch_sizes={"prepare_chunk": plan["chunk"],
                                                                "run_batch_B": record.get("batch_sizes")},
                            free_at_start=free_at_start, monitor=monitor, device_record=dev_record, parity=par,
                            extra={"command": "python experiments/x0_balance.py --seeds %d --first-seed %d --device %s "
                                              "--treasuries %s --out %s" % (args.seeds, args.first_seed, args.device,
                                                                            " ".join(args.treasuries), args.out),
                                   "games_seconds": games_seconds, "batch_plan": plan,
                                   "oom_backoffs": record.get("oom_backoffs", [])})
    report = {
        "experiment": "X0", "title": "Balance gate at the new prices",
        "prices": {"worker": str(base.workers.worker_cost), "query": str(base.survey.query_cost)},
        "n": args.n, "rounds": args.rounds, "seeds": seeds, "treasuries": args.treasuries,
        "generation": base.generation, "runs": [f"{m}/{p}" for m, p in RUNS],
        "gate_criteria": GATE_KEYS,
        "gate_by_treasury": {t: g["gate"] for t, g in gates.items()},
        "per_treasury": gates,
        "holm_family": "informed-vs-top3 one-sided Wilcoxon, one test per treasury (3 tests)",
        "zealot_sweep_oracle_planner": zealot,
        "zealot_sweep_note": (f"oracle manifesto + ORACLE planner, treasury ₫{args.zealot_treasury}, first "
                              f"{args.zealot_seeds} seeds; not one of the five gate criteria"),
        "config": base.to_record(), "provenance": prov,
    }
    xc.save(args.out, report)
    for t in args.treasuries:
        g = gates[t]
        print(f"₫{t}: gate {'PASS' if g['gate_passed_all'] else 'FAIL'} {g['gate']} | best {g['best_paid_pattern_counts']} "
              f"| max paid {g['max_paid_pattern']}", flush=True)
    c = prov["compute"]
    print(f"[x0] wall {seconds:.0f}s, peak {c['peak_bytes'] / 2**30 if c['peak_bytes'] else 0:.2f} GiB, "
          f"GPU util mean {c.get('gpu_util_mean')}", flush=True)


if __name__ == "__main__":
    main()
