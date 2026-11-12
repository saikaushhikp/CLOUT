#!/usr/bin/env python
"""Smart-player advisor for a human playing Clout in the web UI.

Reads YOUR observation of a running game from the local engine API (read only:
``GET /api/games/{game_id}/players/{player_id}/observation``), asks the smart player
(``dumbocracy.strategies.smart``, recommended configuration of experiment S1) what it would do next, and prints it
in the words of the web UI's controls. It never sends an action and never sees hidden state: the smart player only
reads the observation dict, exactly like the AI players in the experiments.

Usage (engine started by scripts/run_game.sh on port 61008):
  conda run -n mas python scripts/advisor.py --game-id <id>          # player p0, http://127.0.0.1:61008
  python scripts/advisor.py --game-id <id> --device cuda:0           # faster on a free GPU
  python scripts/advisor.py --game-id <id> --player bayes            # the plain BayesPlanner instead

Finding the game id: the web app keeps it in the browser tab's session storage. In the browser's developer console
on the game page run   JSON.parse(sessionStorage["dumbocracy.v2.session"]).g
(or look at any /api/games/<id>/... request in the Network tab).

Run it again after every action you take: the advice is for the next single step.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SECTOR_UI = {"Edu": "Education", "Welfare": "Welfare", "Agri": "Agriculture", "Industry": "Industry",
             "Env": "Environment"}
REGION_UI = {"N": "North", "S": "South", "E": "East", "W": "West"}


def fetch_observation(api: str, game_id: str, player_id: str) -> dict:
    url = f"{api.rstrip('/')}/api/games/{game_id}/players/{player_id}/observation"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"API error {e.code} for {url}: {e.read().decode()[:300]}") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"cannot reach the engine at {api} ({e.reason}); is scripts/run_game.sh running?") from None


def integer_percents(shares: list[float]) -> list[int]:
    """Whole percentages summing to 100 (largest remainder), as the manifesto sliders take them."""
    raw = [100.0 * x for x in shares]
    out = [math.floor(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda k: -(raw[k] - out[k]))[:100 - sum(out)]:
        out[i] += 1
    return out


def hunches(obs: dict, model) -> list[str]:
    """Most likely archetype per region under the smart player's posterior (what the UI calls a hunch)."""
    from dumbocracy.sectors import ARCHETYPES, REGIONS
    from dumbocracy.strategies.bayes import posterior_from_observation
    post = posterior_from_observation(obs, model)
    marg = post.region_marginals()
    lines = []
    for r, rid in enumerate(REGIONS):
        k = int(marg[r].argmax())
        lines.append(f"{REGION_UI[rid]}: {ARCHETYPES[k].name} ({100 * float(marg[r, k]):.0f}% sure)")
    return lines


def describe(obs: dict, action: dict, player, seconds: float) -> list[str]:
    from dumbocracy.dynamics.campaign import worker_cost
    out = []
    kind = action["type"]
    if kind == "query":
        a, b, r = SECTOR_UI[action["sector_a"]], SECTOR_UI[action["sector_b"]], REGION_UI[action["region"]]
        out.append(f"ASK A SURVEY QUESTION ({obs['costs']['query']}):")
        out.append(f"  Sector A = {a}, Sector B = {b}, Region = {r}, then press Ask.")
        out.append(f"  (\"What share of {r} voters want more {b} than {a}?\")")
        if obs["phase"] == "ROUND":
            out.append("  Then run the advisor again for this round's spending.")
    elif kind == "commit":
        pct = integer_percents(action["manifesto"])
        out.append("SET THE MANIFESTO SLIDERS AND PRESS Commit manifesto:")
        for (sid, label), p in zip(SECTOR_UI.items(), pct):
            out.append(f"  {label:<12} {p:3d}%")
        out.append("  (The smart player would stop surveying now.)")
    elif kind == "allocate":
        hires, media = action["hires_by_region"], action["media_regions"]
        cost = (worker_cost(sum(hires), Decimal(obs["costs"]["worker"]))
                + sum((Decimal(obs["costs"]["media_by_region"][r]) for r, m in zip(REGION_UI, media) if m),
                      Decimal("0.00")))
        out.append(f"ROUND {obs['round']} OF {obs['rounds']}: set these and press Launch round "
                   f"{obs['round']}" + (" (Final push)" if obs["round"] == obs["rounds"] else "") + ":")
        out.append("  Buy party workers: " + (", ".join(f"{h} in {REGION_UI[r]}" for r, h in zip(REGION_UI, hires) if h)
                                              or "none this round"))
        out.append("  Air media in: " + (", ".join(REGION_UI[r] for r, m in zip(REGION_UI, media) if m) or "nowhere"))
        rest = "left unspent (last round)" if obs["round"] == obs["rounds"] else "left for later rounds"
        out.append(f"  Cost {cost:.2f} of your {Decimal(obs['treasury']):.2f}; "
                   f"{Decimal(obs['treasury']) - cost:.2f} {rest}.")
        if player.diagnostics:
            d = player.diagnostics[-1]
            out.append(f"  Planner's own forecast of your election-day share: {100 * d['best_score']:.1f}% "
                       f"(best of {d['num_candidates']} plans it compared; a rough guide, usually a little low).")
    else:
        out.append(f"Suggested action: {action}")
    out.append(f"(thought for {seconds:.1f}s)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game-id", required=True)
    p.add_argument("--player-id", default="p0")
    p.add_argument("--api", default="http://127.0.0.1:61008")
    p.add_argument("--device", default="cpu", help="where the smart player simulates its imagined electorates")
    p.add_argument("--player", choices=("smart", "bayes"), default="smart")
    p.add_argument("--seed", type=int, default=0, help="the advisor's own random seed")
    p.add_argument("--json", action="store_true", help="also print the raw action dict")
    args = p.parse_args()

    import torch
    torch.set_num_threads(4)
    from dumbocracy.strategies.players import BayesPlanner
    from dumbocracy.strategies.smart import smart_player

    obs = fetch_observation(args.api, args.game_id, args.player_id)
    print(f"Game {obs['game_id']} · player {obs['player_id']} · phase {obs['phase']}"
          + (f" round {obs['round']}/{obs['rounds']}" if obs["phase"] == "ROUND" else "")
          + f" · treasury {obs['treasury']}")
    if obs["phase"] == "DONE":
        res = obs.get("result") or {}
        print(f"The game is over (vote share {res.get('vote_share')}, won: {res.get('victory')}). Nothing to advise.")
        return
    legal = obs.get("legal_actions", [])
    if not any(a in legal for a in ("commit", "allocate")):
        print("You have already acted for this stage; wait for the round to resolve, then run the advisor again.")
        return
    player = (smart_player(seed=args.seed, device=args.device) if args.player == "smart"
              else BayesPlanner(seed=args.seed, device=args.device))
    start = time.perf_counter()
    action = player.act(json.loads(json.dumps(obs)))
    seconds = time.perf_counter() - start
    if obs.get("queries") or obs["phase"] == "ROUND":
        print("Best guess of each region's archetype from your answers so far:")
        for line in hunches(obs, player.model):
            print("  " + line)
    print()
    for line in describe(obs, action, player, seconds):
        print(line)
    if args.json:
        print(json.dumps(action))


if __name__ == "__main__":
    main()
