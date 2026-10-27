import asyncio
import re
import unittest
try:
    import httpx
    from dumbocracy.webapp import app, SESSIONS
except ImportError:
    httpx = None
    app = None
    SESSIONS = None


def private_key_scan(value):
    forbidden = {"voters", "support", "susceptibility", "edges", "seed", "root_seed", "graph_seed", "initial_state_seed", "transition_seed", "strategy_seed", "device_graph", "internal_events", "GameState"}
    if isinstance(value, dict):
        return [key for key, child in value.items() if key in forbidden] + sum((private_key_scan(v) for v in value.values()), [])
    if isinstance(value, list):
        return sum((private_key_scan(v) for v in value), [])
    return []


@unittest.skipIf(httpx is None or app is None, "install the optional web and test extras")
class WebApiTests(unittest.TestCase):
    def request(self, coro):
        return asyncio.run(coro)

    async def flow(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            root = await client.get("/")
            self.assertEqual(root.status_code, 200)
            self.assertIn("Dumbocracy", root.text)
            bundle_paths = re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', root.text)
            self.assertEqual(len(bundle_paths), 2, root.text)
            assets = await asyncio.gather(*(client.get(path) for path in bundle_paths))
            self.assertEqual([asset.status_code for asset in assets], [200, 200])
            javascript = next(asset.text for path, asset in zip(bundle_paths, assets) if path.endswith(".js"))
            stylesheet = next(asset.text for path, asset in zip(bundle_paths, assets) if path.endswith(".css"))
            self.assertIn("Support trajectory", javascript)
            self.assertRegex(stylesheet, r"@media\s*\((?:max-width:\s*640px|width\s*<=\s*640px)\)")
            self.assertNotIn("fonts.googleapis.com", stylesheet)
            created = await client.post("/api/games", json={"node_count": 5, "horizon_rounds": 1, "initial_budget": "0.00", "seed": 9, "backend": "cpu"})
            self.assertEqual(created.status_code, 200, created.text)
            payload = created.json(); gid = payload["game_id"]
            self.assertEqual(private_key_scan(payload), [])
            initial_aggregate = payload["observation"]["aggregate_support"]
            self.assertIsNone((await client.get(f"/api/games/{gid}/result")).json())
            rejected = await client.post(f"/api/games/{gid}/actions", json={"action_id": "broadcast"})
            self.assertFalse(rejected.json()["accepted"])
            self.assertEqual(rejected.json()["observation"]["round_index"], 0)
            self.assertEqual(rejected.json()["observation"]["remaining_budget"], "0.00")
            self.assertEqual((await client.get("/api/games/not-a-game/observation")).status_code, 404)
            accepted = await client.post(f"/api/games/{gid}/actions", json={"action_id": "no_op"})
            self.assertTrue(accepted.json()["terminal"])
            result = await client.get(f"/api/games/{gid}/result")
            self.assertEqual(result.status_code, 200)
            self.assertEqual(private_key_scan(result.json()), [])
            analysis = await client.get(f"/api/games/{gid}/analysis")
            self.assertEqual(analysis.json()["evaluation_mode"], "trace_only")
            self.assertEqual(analysis.json()["verification"]["finite_model"]["status"], "NOT_SUPPLIED")
            reset = await client.post(f"/api/games/{gid}/reset")
            self.assertEqual(reset.json()["observation"]["round_index"], 0)
            self.assertEqual(reset.json()["observation"]["aggregate_support"], initial_aggregate)
            self.assertEqual(private_key_scan(reset.json()), [])
            SESSIONS.pop(gid, None)

    def test_create_reject_terminal_reset_analysis_and_privacy(self):
        self.request(self.flow())


if __name__ == "__main__":
    unittest.main()
