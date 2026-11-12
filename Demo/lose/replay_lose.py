"""Replay the "can never win" demo through the REAL web API and assert the defeat.

    python Demo/lose/replay_lose.py --device cuda:2            # best-effort script + no-campaign play
    python Demo/lose/replay_lose.py --device cpu --play best   # only the scripted best-effort play
    python Demo/lose/replay_lose.py --device cuda:0 --json /tmp/out.json --epistemic

Exactly what the web game does: ``create_app(device=...)`` → FastAPI ``TestClient`` → ``POST /api/games`` with the
settings of ``game_settings.json`` → the actions of ``script.json`` (survey queries, commit, one allocate per
round) as player ``p0`` until the phase is ``DONE`` → ``GET /debrief``. The no-campaign play commits the same
manifesto (no queries) and allocates nothing in every round. Both plays must end in defeat (fewer than half
of the votes); the script exits with status 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from dumbocracy.api.app import create_app  # noqa: E402


def percent_to_manifesto(values: list[int]) -> list[float]:
    """Same as web/lib/simplex.ts ``percentToManifesto``: v / Σv, rounding drift added to the largest entry."""
    total = sum(values)
    out = [v / total for v in values]
    drift = 1 - sum(out)
    out[out.index(max(out))] += drift
    return out


def _post(client: TestClient, path: str, body: dict) -> dict:
    r = client.post(path, json=body)
    r.raise_for_status()
    data = r.json()
    if not data.get("accepted", True):
        raise RuntimeError(f"{path} rejected: {data.get('error_code')} {data.get('message')}")
    return data


def play(client: TestClient, settings: dict, actions: list[dict], *, epistemic: bool = False) -> dict:
    t0 = time.perf_counter()
    created = client.post("/api/games", json={k: settings[k] for k in ("num_voters", "seed", "rounds", "budget")})
    created.raise_for_status()
    created = created.json()
    gid, pid = created["game_id"], created["player_id"]
    assert pid == settings.get("player_id", "p0") == "p0"
    base = f"/api/games/{gid}/players/{pid}"
    obs = created["observation"]
    log = []
    for a in actions:
        if a["type"] == "query":
            d = _post(client, f"{base}/query", {"sector_a": a["sector_a"], "sector_b": a["sector_b"], "region": a["region"]})
            ans = d["answer"]
            log.append({"type": "query", "question": f"{a['sector_b']} over {a['sector_a']} in {a['region']}",
                        "pct_b_over_a": round(ans["pct_b_over_a"], 2), "moe": round(ans["moe"], 2)})
        elif a["type"] == "commit":
            d = _post(client, f"{base}/commit", {"manifesto": percent_to_manifesto(a["manifesto_pct"])})
            log.append({"type": "commit", "manifesto_pct": a["manifesto_pct"]})
        elif a["type"] == "allocate":
            d = _post(client, f"{base}/allocate", {"hires_by_region": a["hires_by_region"], "media_regions": a["media_regions"]})
            rep = d.get("round_report") or {}
            log.append({"type": "allocate", "round": a.get("round"), "cost": rep.get("cost"),
                        "converted": rep.get("converted"), "spreaders": rep.get("spreaders"),
                        "media_exposed": rep.get("media_exposed")})
        else:
            raise ValueError(f"unknown action {a['type']!r}")
        obs = d["observation"]
    if obs["phase"] != "DONE":
        raise RuntimeError(f"game not finished: phase {obs['phase']}")
    debrief = client.get(f"{base}/debrief").json()
    out = {"game_id": gid, "result": obs["result"], "treasury_left": obs["treasury"],
           "polls": [{"round": p["round"], "estimate": round(p["estimate"], 4), "moe": round(p["moe"], 4)} for p in obs["polls"]],
           "true_expected_share_trajectory": [round(x, 6) for x in debrief["share_trajectory"]],
           "debrief_regions": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in debrief["regions"]],
           "oracle_no_campaign_share": round(debrief["oracle"]["expected_share_no_campaign"], 6),
           "actions": log}
    if epistemic:
        ep = client.get(f"{base}/epistemic").json()
        out["epistemic"] = {"headline": ep.get("headline"),
                            "rounds": [{k: r[k] for k in ("round", "possible_victory", "inevitable_defeat",
                                                          "truth_victory_possible", "knows_win_exists")} for r in ep["rounds"]]}
    out["seconds"] = round(time.perf_counter() - t0, 2)
    client.delete(f"/api/games/{gid}")
    return out


def no_campaign_actions(script: dict) -> list[dict]:
    commit = next(a for a in script["actions"] if a["type"] == "commit")
    rounds = script["settings"]["rounds"]
    return [commit] + [{"type": "allocate", "round": t + 1, "hires_by_region": [0, 0, 0, 0],
                        "media_regions": [False] * 4} for t in range(rounds)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:2", help="engine device (cuda:2, cuda:0, cpu, ...; never cuda:1 on this host)")
    ap.add_argument("--play", choices=("best", "none", "both"), default="both")
    ap.add_argument("--script", default=str(HERE / "script.json"))
    ap.add_argument("--json", default=None, help="write the replay results to this file")
    ap.add_argument("--epistemic", action="store_true", help="also fetch the epistemic timeline (slow)")
    args = ap.parse_args()
    script = json.loads(Path(args.script).read_text())
    settings = script["settings"]
    client = TestClient(create_app(device=args.device))
    health = client.get("/api/health").json()
    results = {"device": health["device"], "settings": settings}
    plays = {"best": script["actions"], "none": no_campaign_actions(script)}
    ok = True
    for name in (("best", "none") if args.play == "both" else (args.play,)):
        res = play(client, settings, plays[name], epistemic=args.epistemic)
        r = res["result"]
        short = r["num_voters"] // 2 + 1 - r["votes"]
        res["votes_short_of_majority"] = short
        results["best_effort" if name == "best" else "no_campaign"] = res
        lost = (not r["victory"]) and r["votes"] * 2 <= r["num_voters"]
        ok &= lost
        print(f"[{health['device']}] {('best effort' if name == 'best' else 'no campaign'):12s} votes {r['votes']:,} / "
              f"{r['num_voters']:,} = {100 * r['vote_share']:.2f}% (expected {100 * r['expected_share']:.2f}%), "
              f"{'DEFEAT' if lost else 'VICTORY'}; {short:,} votes short of a majority; true E per round "
              f"{[round(100 * x, 1) for x in res['true_expected_share_trajectory']]}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    if not ok:
        print("ASSERTION FAILED: a play did not lose", file=sys.stderr)
        return 1
    print("OK: every play lost the election")
    return 0


if __name__ == "__main__":
    sys.exit(main())
