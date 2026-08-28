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
        self.assertFalse(any(value.startswith("--policy.path=") for value in learner))
        self.assertFalse(any(value.startswith("--policy.path=") for value in actor))

    def test_console_assets_endpoint(self):
        assets = {
            "datasets": [{"path": "outputs/demos", "label": "demo", "mtime": 2}],
            "checkpoints": [{"path": "outputs/run/pretrained_model", "label": "checkpoint", "mtime": 3}],
        }
        with patch.object(self.server, "discover_project_assets", return_value=assets):
            response = self.client.get("/api/console/assets")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), assets)

    def test_log_line_limit_and_clear_endpoints(self):
        with patch.object(self.server.service_manager, "log_tail", return_value="service tail") as tail:
            response = self.client.get("/api/console/services/actor/log?lines=777")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["text"], "service tail")
        tail.assert_called_once_with("actor", lines=777)
        self.assertEqual(self.client.get("/api/console/services/actor/log?lines=0").status_code, 400)

        with patch.object(self.server.service_manager, "clear_log") as clear:
            response = self.client.post("/api/console/services/actor/log/clear", json={})
        self.assertEqual(response.status_code, 200)
        clear.assert_called_once_with("actor")

        with patch.object(self.server.example_runner, "log_tail", return_value="viewer tail") as tail:
            response = self.client.get("/api/console/example-run/log?lines=321")
        self.assertEqual(response.get_json()["text"], "viewer tail")
        tail.assert_called_once_with(lines=321)

        with patch.object(self.server.example_runner, "clear_log") as clear:
            response = self.client.post("/api/console/example-run/log/clear", json={})
        self.assertEqual(response.status_code, 200)
        clear.assert_called_once_with()

    def test_actor_timing_metric_endpoint(self):
        timing = {
            "available": True,
            "primitive": "insert",
            "loop_hz": 5.7,
            "loop_ms": 175.4,
            "policy_hz": 190.0,
            "policy_ms": 5.3,
            "timestamp": "2026-08-27 22:22:33",
        }
        with patch.object(self.server, "fetch_actor_timing", return_value=timing):
            response = self.client.get("/api/console/actor-timing")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), timing)

    def test_actor_timing_reads_latest_debug_sample(self):
        original_repo_root = self.runtime.REPO_ROOT
        project = Path(self.temporary.name)
        self.runtime.REPO_ROOT = project
        try:
            profile = dict(self.runtime.DEFAULT_PROFILE)
            log_path = project / profile["output_root"] / "logs" / f"actor_{profile['job_name']}.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text(
                "DEBUG 2026-08-27 22:22:32 ing_utils.py:30 [ACTOR] primitive=insert "
                "loop=189.79ms (5.3hz) policy= 5.29ms (189.0hz)\n"
                "DEBUG 2026-08-27 22:22:33 ing_utils.py:30 [ACTOR] primitive=insert "
                "loop=180.19ms (5.5hz) policy= 5.63ms (177.8hz)\n",
                encoding="utf-8",
            )
            timing = self.runtime.fetch_actor_timing(profile)
            self.assertTrue(timing["available"])
            self.assertEqual(timing["loop_hz"], 5.5)
            self.assertEqual(timing["policy_hz"], 177.8)
            self.assertEqual(timing["timestamp"], "2026-08-27 22:22:33")
        finally:
            self.runtime.REPO_ROOT = original_repo_root

    def test_asset_discovery_and_role_specific_policy_paths(self):
        original_repo_root = self.runtime.REPO_ROOT
        project = Path(self.temporary.name)
        self.runtime.REPO_ROOT = project
        try:
            demo_info = project / "outputs" / "demos" / "pick" / "meta" / "info.json"
            demo_info.parent.mkdir(parents=True)
            demo_info.write_text("{}", encoding="utf-8")
            demo_info.with_name("stats.json").write_text("{}", encoding="utf-8")
            online_info = project / "outputs" / "run" / "pick" / "dataset" / "meta" / "info.json"
            online_info.parent.mkdir(parents=True)
            online_info.write_text("{}", encoding="utf-8")
            online_info.with_name("stats.json").write_text("{}", encoding="utf-8")
            learner_policy = project / "outputs" / "run" / "pick" / "checkpoints" / "001000" / "pretrained_model"
            actor_policy = project / "outputs" / "run" / "pick" / "checkpoints" / "002000" / "pretrained_model"
            for policy in (learner_policy, actor_policy):
                policy.mkdir(parents=True)
                (policy / "config.json").write_text("{}", encoding="utf-8")

            assets = self.runtime.discover_project_assets()
            self.assertEqual([item["path"] for item in assets["datasets"]], ["outputs/demos"])
            self.assertEqual(len(assets["checkpoints"]), 2)

            profile = dict(self.runtime.DEFAULT_PROFILE)
            profile.update({
                "dataset_root": "outputs/demos",
                "learner_checkpoint": "outputs/run/pick/checkpoints/001000/pretrained_model",
                "actor_checkpoint": "outputs/run/pick/checkpoints/002000/pretrained_model",
            })
            learner = self.runtime.build_service_command("learner", profile)
            actor = self.runtime.build_service_command("actor", profile)
            self.assertNotIn("--dataset.type=dataset", learner)
            self.assertIn(f"--policy.path={learner_policy}", learner)
            self.assertIn(f"--policy.path={actor_policy}", actor)
            self.assertIn(f"--dataset.root={project / 'outputs' / 'demos'}", learner)
            self.assertFalse(any(value.startswith("--dataset.root=") for value in actor))
            self.assertIn(f"--output_dir={project / 'outputs' / 'web-console' / 'insertion'}", learner)
            self.assertIn(f"--output_dir={project / 'outputs' / 'web-console' / 'insertion'}", actor)
        finally:
            self.runtime.REPO_ROOT = original_repo_root

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

    def test_primitive_rename_round_trip_updates_all_references(self):
        raw = self.create("rename_demo")["raw"]
        raw["primitives"]["finish"] = json.loads(json.dumps(raw["primitives"]["main"]))
        raw["primitives"]["main"]["is_terminal"] = False
        raw["primitives"]["finish"]["is_terminal"] = True
        raw["transitions"] = [{
            "type": "always",
            "source": "main",
            "target": "finish",
            "additional_reward": 0,
            "reason": None,
        }]
        raw["primitives"]["approach"] = raw["primitives"].pop("main")
        raw["start_primitive"] = "approach"
        raw["reset_primitive"] = "approach"
        raw["transitions"][0]["source"] = "approach"

        response = self.client.put("/api/configs/rename_demo", json=raw)

        self.assertEqual(response.status_code, 200)
        summary = response.get_json()["summary"]
        self.assertEqual(summary["start_primitive"], "approach")
        self.assertEqual(summary["reset_primitive"], "approach")
        self.assertEqual(summary["transitions"][0]["source"], "approach")
        self.assertEqual({item["name"] for item in summary["primitives"]}, {"approach", "finish"})

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
