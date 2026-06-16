"""Focused record-loop integration tests for MP-Net debugger hooks."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lerobot.processor import create_transition

from share.scripts.record import record_loop


class _FakeDataset:
    def __init__(self):
        self.features = {"obs": object()}
        self.frames = []
        self.episode_buffer = {"size": 0}

    def add_frame(self, frame):
        self.frames.append(frame)


class _FakeMPNet:
    def __init__(self):
        self.action_dim = 1
        self.active_primitive = "pick"
        self.stop_calls = 0
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
                "transition_reason": None,
            },
        )

    def stop(self):
        self.stop_calls += 1


class _FakeDebugger:
    def __init__(self):
        self.calls = []

    def log_reset(self, mp_net, transition):
        self.calls.append(("reset", mp_net.active_primitive, transition))

    def log_step(self, mp_net, transition):
        self.calls.append(("step", mp_net.active_primitive, transition))


def test_record_loop_emits_debugger_reset_and_step_events():
    dataset = _FakeDataset()
    debugger = _FakeDebugger()

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
    assert mp_net.stop_calls == 1
    assert info["transition_from"] == "pick"
    assert info["transition_to"] == "pick"
