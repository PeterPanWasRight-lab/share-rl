import importlib.util
import json
import socket
import sys
from pathlib import Path

import numpy as np

from share.robots.mujoco.configuration_mujoco import MujocoRobotConfig
from share.robots.mujoco.mujoco_robot import MujocoRobot
from share.teleoperators.delta_keyboard.lerobot_teleoperator_delta_keyboard.config_delta_keyboard import (
    KeyboardAxisBinding,
    KeyboardVelocityTeleopConfig,
)
from share.teleoperators.delta_keyboard.lerobot_teleoperator_delta_keyboard.teleop_delta_keyboard import (
    KeyboardVelocityTeleop,
)
from share.teleoperators.utils import TeleopEvents


SCRIPT_PATH = Path(__file__).parents[3] / "hardEncodedScripts" / "auto_control.py"
SPEC = importlib.util.spec_from_file_location("auto_control", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _state(tip_fixture):
    return {
        "peg_tip_fixture_m": tip_fixture,
        "fixture_rotation_world": np.eye(3).tolist(),
    }


def _planner_config():
    return MODULE.PlannerConfig(
        step_m=0.005,
        clearance_m=0.070,
        align_tolerance_m=0.0005,
        retreat_lateral_error_m=0.0015,
        target_depth_m=0.075,
    )


def test_auto_control_retreats_before_correcting_large_lateral_error():
    plan = MODULE.plan_intervention(_state([0.04, 0.01, 0.0]), _planner_config())

    assert plan.stage == "retreat"
    np.testing.assert_allclose(plan.world_delta_m, [0.005, 0.0, 0.0])


def test_auto_control_aligns_at_clearance_then_inserts():
    align = MODULE.plan_intervention(_state([0.07, 0.004, -0.003]), _planner_config())
    insert = MODULE.plan_intervention(_state([0.07, 0.0001, 0.0]), _planner_config())

    assert align.stage == "align"
    assert np.linalg.norm(align.world_delta_m) == 0.005
    assert align.world_delta_m[1] < 0
    assert align.world_delta_m[2] > 0
    assert insert.stage == "insert"
    assert insert.world_delta_m[0] < 0
    assert np.linalg.norm(insert.world_delta_m) == 0.005


def test_mujoco_publishes_god_view_only_to_existing_local_socket(tmp_path, monkeypatch):
    socket_path = tmp_path / "god-view.sock"
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.bind(str(socket_path))
    receiver.settimeout(2.0)
    monkeypatch.setenv("SHARE_MUJOCO_GOD_VIEW_SOCKET", str(socket_path))
    robot = MujocoRobot(
        MujocoRobotConfig(id="god-view-test", control_dt=0.004, randomize_fixture_xy=0.0)
    )
    robot.connect()
    try:
        robot.send_action({})
        payload = json.loads(receiver.recv(65536))

        assert payload["version"] == 1
        assert payload["robot_id"] == "god-view-test"
        assert payload["episode"] == 1
        assert len(payload["peg_tip_world_m"]) == 3
        assert len(payload["peg_tip_fixture_m"]) == 3
        assert np.asarray(payload["fixture_rotation_world"]).shape == (3, 3)
    finally:
        robot.disconnect()
        receiver.close()


def test_remote_key_tokens_follow_the_normal_keyboard_intervention_path(tmp_path, monkeypatch):
    keyboard_socket_path = tmp_path / "keyboard.sock"
    monkeypatch.setenv("SHARE_KEYBOARD_TELEOP_SOCKET", str(keyboard_socket_path))
    teleop = KeyboardVelocityTeleop(
        KeyboardVelocityTeleopConfig(
            id="remote-keyboard-test",
            x=KeyboardAxisBinding(pos_key="left", neg_key="right", scale=0.1),
        )
    )
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    teleop.connect()
    try:
        sender.sendto(
            json.dumps({"pulse": [{"key": "left", "value": 0.25}]}).encode(),
            str(keyboard_socket_path),
        )

        events = teleop.get_teleop_events()
        action = teleop.get_action()
        next_events = teleop.get_teleop_events()
        next_action = teleop.get_action()

        assert events[TeleopEvents.IS_INTERVENTION]
        assert action["x.vel"] == 0.025
        assert not next_events[TeleopEvents.IS_INTERVENTION]
        assert next_action["x.vel"] == 0.0
    finally:
        sender.close()
        teleop.disconnect()
