"""Replay the WIN demo through the real FastAPI app, exactly as the web game would, and assert the contrast:

* the scripted CAMPAIGN (2 survey questions, the manifesto, the per-round workers/media) WINS;
* the same 2 questions and the same manifesto with NO campaign (an empty plan every round) LOSES.

Usage (repo root, conda env ``mas``):
    OMP_NUM_THREADS=4 python Demo/win/replay_win.py --device cuda:0
    OMP_NUM_THREADS=4 python Demo/win/replay_win.py --device cpu

Settings, questions, manifesto (whole percentages, converted exactly like web/lib/simplex.ts ``percentToManifesto``)
and allocations come from Demo/win/script.json; player id is always "p0". On a CUDA engine the outcome is exact
(the recorded votes are checked); on a CPU engine the random draws differ slightly (thresholds are checked).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

EMPTY_ROUND = {"workers": [0, 0, 0, 0], "media": [False, False, False, False]}


def percent_to_manifesto(values: list[int]) -> list[float]:
    """web/lib/simplex.ts ``percentToManifesto`` (IEEE doubles, same operation order)."""
    total = 0
    for v in values:
        total += v
    out = [v / total for v in values]
    acc = 0.0
    for v in out:
        acc += v
    i = out.index(max(out))
    out[i] += 1 - acc
    return out


def make_client(device: str):
    from fastapi.testclient import TestClient

    from dumbocracy.api.app import create_app
    return TestClient(create_app(device=device))


def play(script: dict[str, Any], device: str, *, client=None, manifesto_pct: list[int] | None = None,
         allocations: list[dict] | None = None) -> dict[str, Any]:
    """Play one game through the API: ``script``'s settings and questions, then ``manifesto_pct`` /
    ``allocations`` (default: the script's own). Returns a JSON-safe record."""
    start = time.perf_counter()
    client = client or make_client(device)
    settings = {k: script["settings"][k] for k in ("num_voters", "seed", "rounds", "budget")}
    created = client.post("/api/games", json=settings)
    created.raise_for_status()
    body = created.json()
    gid, pid = body["game_id"], body["player_id"]
    assert pid == "p0", pid
    record: dict[str, Any] = {"device": client.get("/api/health").json()["device"], "settings": settings,
                              "initial_observation": body["observation"], "answers": [], "rounds": []}

    def post(path: str, payload: dict) -> dict:
        r = client.post(f"/api/games/{gid}/players/{pid}/{path}", json=payload)
        r.raise_for_status()
        out = r.json()
        if not out["accepted"]:
            raise RuntimeError(f"{path} {payload} rejected: {out['error_code']} {out['message']}")
        return out

    for q in script.get("queries", []):
        a = post("query", {"sector_a": q["sector_a"], "sector_b": q["sector_b"], "region": q["region"]})["answer"]
        record["answers"].append({k: a[k] for k in ("sector_a", "sector_b", "region", "pct_b_over_a", "moe",
                                                     "sample_size", "cost")})
    pct = manifesto_pct if manifesto_pct is not None else script["manifesto_pct"]
    manifesto = percent_to_manifesto(pct)
    obs = post("commit", {"manifesto": manifesto})["observation"]
    record["manifesto_pct"], record["manifesto_sent"] = list(pct), manifesto
    plan = allocations if allocations is not None else script["allocations"]
    for t, alloc in enumerate(plan, start=1):
        res = post("allocate", {"hires_by_region": alloc["workers"], "media_regions": alloc["media"]})
        obs = res["observation"]
        rep = res["round_report"]
        polls = {p["round"]: p for p in obs["polls"]}
        nxt = polls.get(t + 1)
        record["rounds"].append({"round": t, "workers": alloc["workers"], "media": alloc["media"], "cost": rep["cost"],
                                 "treasury_left": obs["treasury"], "team_by_region": rep["team_by_region"],
                                 "converted": rep["converted"], "spreaders": rep["spreaders"],
                                 "media_exposed": rep["media_exposed"],
                                 "poll_after_round": None if nxt is None else {"estimate": nxt["estimate"], "moe": nxt["moe"]}})
    assert obs["phase"] == "DONE", obs["phase"]
    debrief = client.get(f"/api/games/{gid}/players/{pid}/debrief").json()
    traj = debrief["share_trajectory"]          # E at commit, then after each round
    for row, e in zip(record["rounds"], traj[1:]):
        row["expected_share_after_round"] = e
    record["expected_share_at_commit"] = traj[0]
    record["poll_round1_start"] = next(p for p in obs["polls"] if p["round"] == 1)["estimate"]
    record["result"] = obs["result"]
    record["debrief"] = {k: debrief[k] for k in ("regions", "share_trajectory", "oracle", "your_manifesto_no_campaign_share")}
    record["seconds"] = time.perf_counter() - start
    return record


def play_no_campaign(script: dict[str, Any], device: str, *, client=None, manifesto_pct=None) -> dict[str, Any]:
    """Same questions and manifesto (or ``manifesto_pct``), then an empty plan every round (= forfeit)."""
    return play(script, device, client=client, manifesto_pct=manifesto_pct,
                allocations=[EMPTY_ROUND] * int(script["settings"]["rounds"]))


def summary(rec: dict[str, Any], label: str) -> str:
    lines = [f"[{label}] engine {rec['device']}  manifesto {rec['manifesto_pct']}  E at commit "
             f"{rec['expected_share_at_commit']:.4f}"]
    for a in rec["answers"]:
        lines.append(f"  survey {a['sector_b']} > {a['sector_a']} in {a['region']}: {a['pct_b_over_a']:.1f}% ± {a['moe']:.1f}")
    for r in rec["rounds"]:
        poll = r["poll_after_round"]
        lines.append(f"  round {r['round']}: workers {r['workers']} media {[int(x) for x in r['media']]} cost {r['cost']} "
                     f"left {r['treasury_left']}  E {r['expected_share_after_round']:.4f}"
                     + (f"  poll {poll['estimate']:.3f}" if poll else ""))
    res = rec["result"]
    lines.append(f"  RESULT {res['votes']} / {res['num_voters']} votes, share {res['vote_share']:.4f}, expected "
                 f"{res['expected_share']:.4f}, {'VICTORY' if res['victory'] else 'DEFEAT'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda:0", help="engine device: cuda:N or cpu (never cuda:1)")
    parser.add_argument("--script", default=str(HERE / "script.json"))
    parser.add_argument("--out", default=None, help="write both replay records as JSON here")
    args = parser.parse_args()
    if args.device == "cuda:1":
        raise SystemExit("GPU 1 is reserved for another job")
    import torch
    torch.set_num_threads(4)
    script = json.loads(Path(args.script).read_text())
    client = make_client(args.device)
    camp = play(script, args.device, client=client)
    none = play_no_campaign(script, args.device, client=client)
    print(summary(none, "NO CAMPAIGN"))
    print(summary(camp, "CAMPAIGN"))
    if args.out:
        Path(args.out).write_text(json.dumps({"campaign": camp, "no_campaign": none}, indent=1))
    rc, rn = camp["result"], none["result"]
    exp = script["expected"]
    assert not rn["victory"] and 2 * rn["votes"] < rn["num_voters"], "no-campaign play did not lose"
    assert rn["vote_share"] <= exp["no_campaign_max_vote_share"], f"no-campaign share {rn['vote_share']:.4f} too high"
    assert rc["victory"], "the campaign did not win"
    assert rc["vote_share"] >= exp["campaign_min_vote_share"], f"campaign share {rc['vote_share']:.4f} too low"
    if camp["device"].startswith("cuda"):
        assert rc["votes"] == exp["cuda"]["campaign_votes"], (rc["votes"], exp["cuda"]["campaign_votes"])
        assert rn["votes"] == exp["cuda"]["no_campaign_votes"], (rn["votes"], exp["cuda"]["no_campaign_votes"])
    print(f"OK: without campaigning {rn['vote_share']:.1%} (defeat); with the campaign {rc['vote_share']:.1%} (victory)")


if __name__ == "__main__":
    main()
