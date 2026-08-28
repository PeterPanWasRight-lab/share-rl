"""Focused record-loop integration tests for MP-Net debugger hooks."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
import torch

from lerobot.processor import create_transition

import share.scripts.record as record_module
from share.scripts.record import _commit_episode_buffers, record, record_loop
from share.utils.control_utils import _prepare_dataset_root_for_create


class _FakeDataset:
    def __init__(self):
        self.features = {"obs": object()}
        self.frames = []
        self.writer = SimpleNamespace(episode_buffer={"size": 0})
        self.num_episodes = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.num_episodes += 1
        self.writer.episode_buffer["size"] = 0

    def clear_episode_buffer(self):
        self.writer.episode_buffer["size"] = 0


class _FakeMPNet:
    def __init__(self):
        self.action_dim = 1
        self.active_primitive = "pick"
        self.stop_calls = 0
        self.full_reset_requests = 0
        self.config = SimpleNamespace(
            fps=1000,
            type="mock_robot",
            primitives={
                "pick": SimpleNamespace(task_description="pick task"),
            },
        )

    def reset(self):
        return create_transition(
            observation={"obs": torch.tensor([1.0])},
            info={"primitive_target_pose": {"arm": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]}},
        )

    def step(self, action):
        return create_transition(
            observation={"obs": torch.tensor([2.0])},
            action=action,
            reward=1.0,
            done=True,
            info={
                "primitive_target_pose": {"arm": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]},
                "primitive_step": 1,
                "episode_step": 1,
                "transition_from": "pick",
                "transition_to": "pick",
                "transition_reason": "peg_inserted",
            },
        )

    def stop(self):
        self.stop_calls += 1

    def request_full_reset(self):
        self.full_reset_requests += 1


class _FakeDebugger:
    def __init__(self):
        self.calls = []

    def log_reset(self, mp_net, transition):
        self.calls.append(("reset", mp_net.active_primitive, transition))

    def log_step(self, mp_net, transition):
        self.calls.append(("step", mp_net.active_primitive, transition))


def test_record_loop_emits_debugger_reset_and_step_events(caplog):
    dataset = _FakeDataset()
    debugger = _FakeDebugger()
    caplog.set_level(logging.INFO)

    mp_net = _FakeMPNet()

    info = record_loop(
        mp_net=mp_net,
        datasets={"pick": dataset},
        policies={},
        preprocessors={},
        postprocessors={},
        debugger=debugger,
    )

    assert [call[0] for call in debugger.calls] == ["reset", "step"]
    assert len(dataset.frames) == 1
    assert dataset.frames[0]["task"] == "pick task"
    assert float(dataset.frames[0]["next.reward"][0]) == pytest.approx(1.0)
    assert bool(dataset.frames[0]["next.done"][0])
    assert mp_net.stop_calls == 1
    assert mp_net.full_reset_requests == 1
    assert "reason=peg_inserted reward=1.000" in caplog.text
    assert info["transition_from"] == "pick"
    assert info["transition_to"] == "pick"


def test_prepare_dataset_root_archives_zero_frame_stub(tmp_path):
    root = tmp_path / "insert"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"total_frames": 0, "total_episodes": 0})
    )

    should_resume = _prepare_dataset_root_for_create(root)

    assert not should_resume
    assert not root.exists()
    archives = list(tmp_path.glob("insert.incomplete-*"))
    assert len(archives) == 1
    assert (archives[0] / "meta" / "info.json").is_file()


def test_prepare_dataset_root_resumes_existing_recording(tmp_path):
    root = tmp_path / "insert"
    meta = root / "meta"
    (meta / "episodes").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "data" / "episode.parquet").write_bytes(b"data")
    (meta / "tasks.parquet").write_bytes(b"tasks")
    (meta / "info.json").write_text(
        json.dumps({"total_frames": 10, "total_episodes": 1})
    )

    should_resume = _prepare_dataset_root_for_create(root)

    assert should_resume
    assert root.is_dir()


def test_prepare_dataset_root_refuses_malformed_existing_directory(tmp_path):
    root = tmp_path / "insert"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"total_frames": 10, "total_episodes": 1})
    )

    with pytest.raises(FileExistsError, match="not safely resumable"):
        _prepare_dataset_root_for_create(root)


def test_record_closes_mpnet_when_dataset_setup_fails(monkeypatch):
    class FakeMPNet:
        def __init__(self, _env):
            self.closed = False

        def set_step_info(self, _info):
            pass

        def close(self):
            self.closed = True

    fake_mpnet = FakeMPNet(None)
    monkeypatch.setattr(record_module, "ManipulationPrimitiveNet", lambda _env: fake_mpnet)
    monkeypatch.setattr(
        record_module,
        "make_policies_and_datasets",
        lambda _cfg: (_ for _ in ()).throw(RuntimeError("dataset setup failed")),
    )
    cfg = SimpleNamespace(
        resolve_policy_overrides=lambda: None,
        display_data=False,
        display_compressed_images=False,
        env=SimpleNamespace(),
        dataset=None,
        use_policy=False,
        play_sounds=False,
    )

    with pytest.raises(RuntimeError, match="dataset setup failed"):
        record.__wrapped__(cfg)

    assert fake_mpnet.closed


def test_commit_episode_buffers_uses_dataset_that_received_frames():
    insert = _FakeDataset()
    insert.writer.episode_buffer["size"] = 12

    reached_limit = _commit_episode_buffers(
        {"insert": insert},
        rerecord=False,
        num_episodes=1,
        play_sounds=False,
    )

    assert insert.num_episodes == 1
    assert reached_limit


def test_commit_episode_buffers_discards_failed_demonstration():
    insert = _FakeDataset()
    insert.writer.episode_buffer["size"] = 12

    reached_limit = _commit_episode_buffers(
        {"insert": insert},
        rerecord=True,
        num_episodes=1,
        play_sounds=False,
    )

    assert insert.writer.episode_buffer["size"] == 0
    assert insert.num_episodes == 0
    assert not reached_limit


def test_record_main_handles_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(
        record_module,
        "record",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    record_module.main()
