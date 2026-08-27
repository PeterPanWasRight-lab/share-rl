"""Focused API tests for the standalone web editor."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
        cls.runtime = sys.modules["console_runtime"]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.server.CONFIGS_DIR = Path(self.temporary.name)
        self.original_profile_path = self.runtime.PROFILE_PATH
        self.runtime.PROFILE_PATH = Path(self.temporary.name) / "runtime_profile.json"
        self.original_example_runner = self.server.example_runner
        self.server.example_runner = self.runtime.ViewerExampleRunner(
            Path(self.temporary.name) / "example-logs"
        )
        self.client = self.server.app.test_client()

    def tearDown(self):
        self.server.example_runner = self.original_example_runner
        self.runtime.PROFILE_PATH = self.original_profile_path
        self.temporary.cleanup()

    def create(self, name="demo"):
        response = self.client.post("/api/configs", json={"name": name})
        self.assertEqual(response.status_code, 201)
        return response.get_json()

    def test_health_and_empty_list(self):
        self.assertEqual(self.client.get("/api/health").get_json(), {"ok": True})
        self.assertEqual(self.client.get("/api/configs").get_json(), [])

    def test_console_pages_and_default_service_status(self):
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertIn(b'id="view-overview"', root.get_data())
        root.close()
        editor = self.client.get("/editor")
        self.assertEqual(editor.status_code, 200)
        editor.close()
        services = self.client.get("/api/console/services").get_json()
        self.assertEqual(services["actor"]["state"], "stopped")
        self.assertEqual(services["learner"]["state"], "stopped")

    def test_console_profile_round_trip_and_command_preview(self):
        profile = self.client.get("/api/console/profile").get_json()
        profile["batch_size"] = 512
        profile["output_root"] = "outputs/web-console-test"
        response = self.client.put("/api/console/profile", json=profile)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.runtime.PROFILE_PATH.exists())
        self.assertEqual(self.client.get("/api/console/profile").get_json()["batch_size"], 512)

        learner = self.client.get("/api/console/services/learner/command").get_json()["argv"]
        actor = self.client.get("/api/console/services/actor/command").get_json()["argv"]
        self.assertTrue(any(value.endswith("learner_server.py") for value in learner))
        self.assertTrue(any(value.endswith("actor_server.py") for value in actor))
        self.assertIn("--batch_size=512", learner)

    def test_console_rejects_unsafe_profile_and_non_json_mutations(self):
        profile = self.client.get("/api/console/profile").get_json()
        profile["output_root"] = "../outside"
        self.assertEqual(self.client.put("/api/console/profile", json=profile).status_code, 400)

        profile = self.client.get("/api/console/profile").get_json()
        profile["shell_command"] = "echo unsafe"
        self.assertEqual(self.client.put("/api/console/profile", json=profile).status_code, 400)
        self.assertEqual(self.client.put("/api/console/profile", data="{}").status_code, 415)
        self.assertEqual(self.client.post("/api/console/services/actor/start").status_code, 415)

    def test_actor_requires_running_learner(self):
        response = self.client.post("/api/console/services/actor/start", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("learner", response.get_json()["error"])

    def test_viewer_example_api_is_allowlisted(self):
        payload = self.client.get("/api/console/examples").get_json()
        self.assertEqual([item["id"] for item in payload["examples"]], ["pick_insert"])
        self.assertEqual(payload["run"]["state"], "stopped")
        self.assertEqual(self.client.post("/api/console/example-run/start").status_code, 415)
        self.create("viewer_config")
        response = self.client.post(
            "/api/console/example-run/start",
            json={"example_id": "../../unsafe", "config_name": "viewer_config", "steps": 10},
        )
        self.assertEqual(response.status_code, 400)

    def test_viewer_example_start_builds_argv_without_shell(self):
        class FakeProcess:
            pid = 99999

            def __init__(self):
                self.exit_code = None

            def poll(self):
                return self.exit_code

            def terminate(self):
                self.exit_code = 0

            def kill(self):
                self.exit_code = -9

            def wait(self, timeout=None):
                self.exit_code = 0
                return self.exit_code

        self.create("viewer_config")
        fake_process = FakeProcess()
        with patch.object(self.runtime.subprocess, "Popen", return_value=fake_process) as popen:
            response = self.client.post(
                "/api/console/example-run/start",
                json={"example_id": "pick_insert", "config_name": "viewer_config", "steps": 42},
            )
            self.assertEqual(response.status_code, 201)
            command = popen.call_args.args[0]
            self.assertIn("--viewer", command)
            self.assertIn("--steps=42", command)
            self.assertTrue(any(value.endswith("demo_pick_insert.py") for value in command))
            self.assertTrue(any(value.startswith("--config=") for value in command))
            self.assertNotIn("shell", popen.call_args.kwargs)
            blocked_service = self.client.post("/api/console/services/learner/start", json={})
            self.assertEqual(blocked_service.status_code, 400)
            self.assertIn("viewer", blocked_service.get_json()["error"])
            stopped = self.client.post("/api/console/example-run/stop", json={})
            self.assertEqual(stopped.status_code, 200)
            self.assertEqual(stopped.get_json()["state"], "exited")

    def test_pick_insert_example_overlays_saved_web_config(self):
        from examples.demo_pick_insert import build_demo_config

        self.create("viewer_config")
        path = self.server.CONFIGS_DIR / "viewer_config.json"
        runtime_config = build_demo_config(config_path=path)
        self.assertEqual(runtime_config.start_primitive, "main")
        self.assertEqual(runtime_config.reset_primitive, "main")
        self.assertEqual(set(runtime_config.primitives), {"main"})
        self.assertEqual(runtime_config.fps, 10)

    def test_exited_service_uptime_stops_increasing(self):
        class CompletedProcess:
            pid = 12345

            @staticmethod
            def poll():
                return 0

        manager = self.runtime.ServiceManager()
        log_path = Path(self.temporary.name) / "actor.log"
        log_handle = log_path.open("ab")
        manager._records["actor"] = {
            "process": CompletedProcess(),
            "log_handle": log_handle,
            "log_path": log_path,
            "started_at": "test",
            "started_at_unix": 100.0,
        }
        with patch.object(self.runtime.time, "time", return_value=150.0):
            first = manager.status("actor")
        with patch.object(self.runtime.time, "time", return_value=250.0):
            second = manager.status("actor")

        self.assertEqual(first["uptime_s"], 50.0)
        self.assertEqual(second["uptime_s"], 50.0)
        self.assertTrue(log_handle.closed)

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

    def test_saved_config_is_written_and_loadable_by_runtime(self):
        from mpnet_adapter import load_flat_mpnet

        created = self.create("runtime_round_trip")
        path = self.server.CONFIGS_DIR / "runtime_round_trip.json"
        before = path.read_bytes()
        raw = created["raw"]
        raw["fps"] = 37
        raw["primitives"]["main"]["notes"] = "loaded by runtime"

        response = self.client.put("/api/configs/runtime_round_trip", json=raw)

        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(path.read_bytes(), before)
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(stored["fps"], 37)
        self.assertEqual(stored["primitives"]["main"]["notes"], "loaded by runtime")
        loaded = load_flat_mpnet(path)
        self.assertEqual(loaded.fps, 37)
        self.assertEqual(loaded.primitives["main"].notes, "loaded by runtime")
        self.assertFalse(isinstance(loaded.primitives["main"].processor, dict))
        frame = loaded.primitives["main"].task_frame
        self.assertTrue(math.isinf(frame.min_pose[0]) and frame.min_pose[0] < 0)
        self.assertTrue(math.isinf(frame.max_pose[0]) and frame.max_pose[0] > 0)

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
