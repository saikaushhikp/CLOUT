"""E7 — Scale: generation time, engine round latency, full-game time and peak memory at N ∈ {1e4, 1e5, 1e6}
(spec §9 targets: interactive N = 10k with < 1 s per round; N = 1e5 required; N = 1e6 single world stretch on one
24 GB GPU). Extends experiments/scale.py (generation only) with the player-facing engine.

Each (N, device) runs in a fresh subprocess so peak memory is clean: GPU peak = torch.cuda.max_memory_allocated;
host peak = ru_maxrss of that subprocess. Per (N, device):
* canonical generation (generation = "cpu": CPU seed streams, then moved to the device; R8) — repeats;
  plus GPU-side generation (generation = "device") when the device is CUDA;
* GameEngine.new_game, then 5 rounds of a fixed allocation (10 free + 10 extra workers split evenly,
  media in the largest region in rounds 1–3) — per-round GameEngine.apply latency (campaign → cascade → dynamics → stats/poll);
* full games through play_game: Top3Player and BayesPlanner (posterior worlds of 1 600 voters, so its cost is
  mostly independent of N).

Usage: python experiments/e7_scale.py --n 10000 100000 1000000 --device cuda:2 --cpu-too --out reports/v2/e7_scale.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import p7_common as pc


def child(n: int, device: str, repeats: int, seed: int, bayes: bool) -> dict:
    import dataclasses
    import resource

    import torch
    from dumbocracy.config import ElectorateConfig, GameConfig
    from dumbocracy.electorate import generate_electorate
    from dumbocracy.game import GameEngine
    from dumbocracy.strategies.players import BayesPlanner, Top3Player, play_game

    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))
    dev = torch.device(device)
    cuda = dev.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize(dev)

    if cuda:
        torch.zeros(1, device=dev)
    cfg = GameConfig(seed=seed, device=device, electorate=ElectorateConfig(num_voters=n))
    out = {"num_voters": n, "device": device}
    gen = []
    for _ in range(repeats):
        t = time.perf_counter()
        e = generate_electorate(cfg, device=dev)
        sync()
        gen.append(time.perf_counter() - t)
    out["generation_canonical_s"] = gen
    out["edges"] = e.graph.num_edges
    if cuda:
        g2 = []
        cdev = dataclasses.replace(cfg, generation="device")
        for _ in range(repeats):
            t = time.perf_counter()
            generate_electorate(cdev, device=dev)
            sync()
            g2.append(time.perf_counter() - t)
        out["generation_device_s"] = g2
        out["peak_gpu_generation_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
        torch.cuda.reset_peak_memory_stats(dev)
    # engine rounds with a fixed allocation
    t = time.perf_counter()
    state = GameEngine.new_game(cfg, electorate=e, game_id="scale")
    sync()
    out["new_game_s"] = time.perf_counter() - t
    sizes = e.sizes.tolist()
    largest = max(range(4), key=lambda r: sizes[r])
    step = GameEngine.apply(state, "p0", {"type": "commit", "manifesto": [0.2] * 5})
    state = step.state
    rounds = []
    for t_round in range(cfg.rounds):
        media = [r == largest and t_round < 3 for r in range(4)]
        action = {"type": "allocate", "workers_by_region": [5, 5, 5, 5], "extra_workers": 10, "media_regions": media}
        t = time.perf_counter()
        step = GameEngine.apply(state, "p0", action)
        sync()
        rounds.append(time.perf_counter() - t)
        assert step.accepted, step.error_code
        state = step.state
    out["round_s"] = rounds
    out["phase_after"] = state.phase
    games = {}
    players = [("top3", lambda: Top3Player(config=dataclasses.replace(cfg, device="cpu")))]
    if bayes:
        players.append(("bayes", lambda: BayesPlanner(seed=seed, config=dataclasses.replace(cfg, device="cpu"))))
    state0 = GameEngine.new_game(cfg, electorate=e, game_id="scale-game")
    for name, make in players:
        t = time.perf_counter()
        final, log = play_game(state0, make())
        sync()
        games[name] = {"seconds": time.perf_counter() - t, "actions": len(log), "phase": final.phase}
    out["full_game"] = games
    out["peak_gpu_engine_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30 if cuda else None
    out["peak_gpu_gib"] = max(out["peak_gpu_engine_gib"], out["peak_gpu_generation_gib"]) if cuda else None
    out["peak_host_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cpu-too", action="store_true", help="also measure on CPU")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--seeds", type=int, default=1, help="(unused; one canonical seed — CLI uniformity)")
    ap.add_argument("--no-bayes", action="store_true")
    ap.add_argument("--child", default=None)
    ap.add_argument("--out", default="reports/v2/e7_scale.json")
    args = ap.parse_args()
    if args.child:
        n, device = args.child.split("@")
        print("RESULT " + json.dumps(child(int(n), device, args.repeats, args.seed, not args.no_bayes)), flush=True)
        return
    device = pc.pick_device(args.device)
    devices = [device] + (["cpu"] if args.cpu_too and device != "cpu" else [])
    start = time.perf_counter()
    rows = []
    for dev in devices:
        for n in args.n:
            cmd = [sys.executable, __file__, "--child", f"{n}@{dev}", "--repeats", str(args.repeats), "--seed", str(args.seed)]
            if args.no_bayes:
                cmd.append("--no-bayes")
            t = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
            if not line:
                print(proc.stdout[-2000:], proc.stderr[-4000:], flush=True)
                rows.append({"num_voters": n, "device": dev, "error": proc.stderr[-1000:]})
                continue
            row = json.loads(line[0][7:])
            row["wall_s"] = time.perf_counter() - t
            rows.append(row)
            med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
            print(json.dumps({"n": n, "device": dev, "gen": round(med(row["generation_canonical_s"]), 3),
                              "round": round(med(row["round_s"]), 4), "games": row["full_game"],
                              "gpu_gib": row["peak_gpu_gib"], "rss_gib": round(row["peak_host_rss_gib"], 2)}), flush=True)
    report = {"experiment": "E7 scale", "rows": rows,
              "provenance": pc.provenance(args, device, time.perf_counter() - start, 1)}
    pc.save(args.out, report)


if __name__ == "__main__":
    main()
