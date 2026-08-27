"""Focused API tests for the standalone web editor."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


def _load_server():
    path = Path(__file__).with_name("server.py")
    spec = importlib.util.spec_from_file_location("webconfig_server_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ServerApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _load_server()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.server.CONFIGS_DIR = Path(self.temporary.name)
        self.client = self.server.app.test_client()

    def tearDown(self):
        self.temporary.cleanup()

    def create(self, name="demo"):
        response = self.client.post("/api/configs", json={"name": name})
        self.assertEqual(response.status_code, 201)
        return response.get_json()

    def test_health_and_empty_list(self):
        self.assertEqual(self.client.get("/api/health").get_json(), {"ok": True})
        self.assertEqual(self.client.get("/api/configs").get_json(), [])

    def test_create_get_and_delete(self):
        self.create()
        self.assertEqual(self.client.get("/api/configs").get_json(), ["demo"])
        loaded = self.client.get("/api/configs/demo")
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(loaded.get_json()["summary"]["primitive_count"], 1)
        self.assertEqual(self.client.delete("/api/configs/demo").status_code, 200)
        self.assertEqual(self.client.get("/api/configs/demo").status_code, 404)

    def test_rejects_unsafe_or_duplicate_names(self):
        self.create()
        self.assertEqual(self.client.post("/api/configs", json={"name": "demo"}).status_code, 409)
        self.assertEqual(self.client.post("/api/configs", json={"name": "bad.name"}).status_code, 400)

    def test_validate_and_full_save_round_trip(self):
        created = self.create()
        raw = created["raw"]
        raw["fps"] = 25
        self.assertEqual(self.client.post("/api/validate", json=raw).status_code, 200)
        saved = self.client.put("/api/configs/demo", json=raw)
        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.get_json()["summary"]["fps"], 25)

    def test_structured_edit(self):
        self.create()
        response = self.client.post(
            "/api/configs/demo/edit",
            json={
                "operation": "set_primitive_notes",
                "arguments": {"primitive_name": "main", "notes": "edited"},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["summary"]["primitives"][0]["notes"], "edited")

    def test_invalid_edit_does_not_modify_file(self):
        self.create()
        path = self.server.CONFIGS_DIR / "demo.json"
        before = path.read_bytes()
        response = self.client.post(
            "/api/configs/demo/edit",
            json={"operation": "remove_primitive", "arguments": {"name": "main"}},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(path.read_bytes(), before)

    def test_metadata_matches_registered_transitions(self):
        payload = self.client.get("/api/transition-types").get_json()
        self.assertEqual(set(payload), set(self.server.TRANSITION_TYPES))
        self.assertNotIn("on_info_equals", payload)

    def test_responses_are_strict_json(self):
        safe = self.server._json_safe({"limits": [-math.inf, math.inf]})
        self.assertEqual(safe, {"limits": [None, None]})
        json.dumps(safe, allow_nan=False)
        created = self.create()
        json.dumps(created, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
