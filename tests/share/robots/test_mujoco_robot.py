from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.processor import TransitionKey
from scipy.spatial.transform import Rotation

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.configs.mujoco_insertion import MujocoInsertionEnvConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.robots.mujoco import MujocoRobot, MujocoRobotConfig
from share.robots.mujoco.model import ASSET_ROOT, build_ur5e_2f85_model
from share.teleoperators.delta_keyboard import KeyboardVelocityTeleop
from share.teleoperators import TeleopEvents
from share.teleoperators.delta_keyboard.lerobot_teleoperator_delta_keyboard.teleop_delta_keyboard import (
    keyboard as pynput_keyboard,
)
from share.utils.transformation_utils import get_robot_pose_from_observation


def test_free_viewer_camera_is_not_resolved_as_a_named_camera(monkeypatch):
    import mujoco.viewer

    viewer = SimpleNamespace(cam=SimpleNamespace(type=None, fixedcamid=None), close=lambda: None)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *_args: viewer)

    robot = MujocoRobot(MujocoRobotConfig(id="test-free-camera", viewer=True, viewer_camera="free"))
    robot.connect()
    try:
        assert viewer.cam.type is None
        assert viewer.cam.fixedcamid is None
    finally:
        robot.disconnect()


def test_default_model_uses_vendored_menagerie_and_act_assets():
    model = build_ur5e_2f85_model()

    assert (ASSET_ROOT / "menagerie" / "universal_robots_ur5e" / "ur5e.xml").is_file()
    assert (ASSET_ROOT / "menagerie" / "robotiq_2f85" / "2f85.xml").is_file()
    assert (ASSET_ROOT / "act" / "bimanual_viperx_insertion.xml").is_file()
    assert model.nq == 21
    assert model.nu == 7
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_peg_joint") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tcp") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_fingers_actuator") >= 0
    arm_actuator_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
        ]
    )
    assert np.all(model.actuator_biasprm[arm_actuator_ids, 1] < 0)
    assert np.isfinite(model.actuator_ctrlrange[arm_actuator_ids]).all()


def test_mujoco_robot_observation_action_and_camera_contract():
    robot = MujocoRobot(
        MujocoRobotConfig(id="test-mujoco", control_dt=0.004, randomize_fixture_xy=0.0)
    )
    robot.connect()
    try:
        robot.set_task_frame(
            TaskFrame(
                target=[0.0] * 6,
                policy_mode=[PolicyMode.RELATIVE] * 6,
                control_mode=[ControlMode.POS] * 6,
            )
        )
        before = robot.get_observation()
        for _ in range(20):
            robot.send_action({"z.ee_pos": -0.01})
        after = robot.get_observation()
        image = robot.render_camera("front", width=64, height=48)

        assert set(robot.observation_features) == set(after)
        assert np.isfinite(list(after.values())).all()
        assert after["z.ee_pos"] != pytest.approx(before["z.ee_pos"])
        assert image.shape == (48, 64, 3)
        assert image.dtype == np.uint8
    finally:
        robot.disconnect()


def test_mujoco_cartesian_ik_target_does_not_accumulate_tracking_error():
    robot = MujocoRobot(
        MujocoRobotConfig(
            id="test-ik-target-base",
            control_dt=1.0 / 30.0,
            randomize_fixture_xy=0.0,
        )
    )
    robot.connect()
    try:
        robot.set_task_frame(
            TaskFrame(
                target=[0.0] * 6,
                policy_mode=[PolicyMode.RELATIVE] * 6,
                control_mode=[ControlMode.POS] * 6,
                min_pose=[-2.0, -2.0, 0.05, -3.14, -3.14, -3.14],
                max_pose=[2.0, 2.0, 2.0, 3.14, 3.14, 3.14],
            )
        )
        for _ in range(10):
            robot.send_action({"x.ee_pos": 0.1})
            robot._ik_data.qpos[:] = robot._data.qpos
            robot._ik_data.qpos[robot._joint_qpos_adr] = robot._joint_position_target
            robot._ik_data.qvel[:] = 0.0
            mujoco.mj_forward(robot._model, robot._ik_data)
            target_pose = robot._world_to_task_pose(
                robot._tcp_world_pose(data=robot._ik_data)
            )
            assert target_pose[0] == pytest.approx(
                robot._virtual_target_task[0], abs=1e-4
            )
    finally:
        robot.disconnect()


def test_mujoco_reset_is_deterministic_without_randomization():
    robot = MujocoRobot(
        MujocoRobotConfig(id="test-reset", control_dt=0.004, randomize_fixture_xy=0.0)
    )
    robot.connect()
    try:
        expected = robot.get_observation()
        robot.send_action({"shoulder_pan_joint.pos": 0.2})
        robot.reset_simulation(seed=7)
        actual = robot.get_observation()
        for key in expected:
            assert actual[key] == pytest.approx(expected[key])
    finally:
        robot.disconnect()


def test_mujoco_rotation_observation_matches_ur_rotvec_contract():
    robot = MujocoRobot(MujocoRobotConfig(id="test-rotvec", randomize_fixture_xy=0.0))
    robot.connect()
    try:
        observation = {f"main.{key}": value for key, value in robot.get_observation().items()}
        parsed_pose = np.asarray(get_robot_pose_from_observation(observation, "main"))
        internal_pose = robot._world_to_task_pose(robot._tcp_world_pose())
        rotation_error = (
            Rotation.from_euler("xyz", parsed_pose[3:])
            * Rotation.from_euler("xyz", internal_pose[3:]).inv()
        ).magnitude()

        np.testing.assert_allclose(parsed_pose[:3], internal_pose[:3], atol=1e-8)
        assert rotation_error < 1e-8
    finally:
        robot.disconnect()


def test_mpnet_zero_action_does_not_drop_position_servo_arm():
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(viewer=False, episode_steps=10_000)
    )
    robot = next(iter(net.robot_dict.values()))
    robot.reset_simulation(seed=0)
    initial_pose = robot._tcp_world_pose().copy()
    net.reset(seed=0)
    try:
        action = torch.zeros(net.action_dim)
        action[-1] = 1.0
        for _ in range(60):
            net.step(action)

        final_pose = robot._tcp_world_pose()
        rotation_error = (
            Rotation.from_euler("xyz", final_pose[3:])
            * Rotation.from_euler("xyz", initial_pose[3:]).inv()
        ).magnitude()
        assert np.linalg.norm(final_pose[:3] - initial_pose[:3]) < 1e-3
        assert rotation_error < 2e-3
    finally:
        net.close()


def test_opening_gripper_releases_free_peg():
    robot = MujocoRobot(
        MujocoRobotConfig(id="test-release", control_dt=0.01, randomize_fixture_xy=0.0)
    )
    robot.connect()
    try:
        robot.set_task_frame(
            TaskFrame(
                target=[0.0] * 6,
                policy_mode=[PolicyMode.RELATIVE] * 6,
                control_mode=[ControlMode.POS] * 6,
                min_pose=[-2.0, -2.0, 0.05, -3.14, -3.14, -3.14],
                max_pose=[2.0, 2.0, 2.0, 3.14, 3.14, 3.14],
            )
        )
        peg_body_id = mujoco.mj_name2id(
            robot._model, mujoco.mjtObj.mjOBJ_BODY, "object_peg"
        )
        held_z = float(robot._data.xpos[peg_body_id, 2])
        for _ in range(30):
            robot.send_action({})
        assert float(robot._data.xpos[peg_body_id, 2]) == pytest.approx(held_z, abs=0.01)

        for _ in range(50):
            robot.send_action({"gripper.pos": 0.0})
        released_z = float(robot._data.xpos[peg_body_id, 2])
        assert released_z < held_z - 0.05
        assert robot.get_observation()["gripper.pos"] < 0.1
    finally:
        robot.disconnect()


def test_mpnet_camera_and_gripper_release_chain():
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(
            viewer=False,
            episode_steps=1,
            release_steps=15,
        )
    )
    net.reset(seed=0)
    try:
        assert net.action_dim == 7
        assert net.cameras["wrist"].async_read().shape == (240, 320, 3)

        robot = next(iter(net.robot_dict.values()))
        peg_body_id = mujoco.mj_name2id(
            robot._model, mujoco.mjtObj.mjOBJ_BODY, "object_peg"
        )
        held_z = float(robot._data.xpos[peg_body_id, 2])

        insert_action = torch.zeros(net.action_dim)
        insert_action[-1] = 1.0
        net.step(insert_action)
        assert net.active_primitive == "release"
        assert net.action_dim == 1
        release_pose = robot._tcp_world_pose().copy()

        teleop = next(iter(net.teleop_dict.values()))
        teleop.set_gripper_position(0.0)
        for _ in range(15):
            net.step(torch.zeros(net.action_dim))

        assert net.active_primitive == "done"
        assert robot.get_observation()["gripper.pos"] < 0.1
        assert float(robot._data.xpos[peg_body_id, 2]) < held_z - 0.05
        done_pose = robot._tcp_world_pose()
        rotation_error = (
            Rotation.from_euler("xyz", done_pose[3:])
            * Rotation.from_euler("xyz", release_pose[3:]).inv()
        ).magnitude()
        assert np.linalg.norm(done_pose[:3] - release_pose[:3]) < 1e-3
        assert rotation_error < 2e-3
    finally:
        net.close()


def test_mujoco_keyboard_matches_lerobot_ee_controls():
    env_config = MujocoInsertionEnvConfig(viewer=False, teleop_mode="keyboard")
    teleop = KeyboardVelocityTeleop(next(iter(env_config.teleop.values())))

    teleop.current_pressed = {"left": True}
    assert teleop._axis_value(teleop.config.x) == pytest.approx(0.1)
    teleop.current_pressed = {"right": True}
    assert teleop._axis_value(teleop.config.x) == pytest.approx(-0.1)

    teleop.current_pressed = {"up": True}
    assert teleop._axis_value(teleop.config.y) == pytest.approx(-0.1)
    teleop.current_pressed = {"down": True}
    assert teleop._axis_value(teleop.config.y) == pytest.approx(0.1)

    teleop.current_pressed = {"shift": True}
    assert teleop._axis_value(teleop.config.z) == pytest.approx(-0.1)
    teleop.current_pressed = {"shift_r": True}
    assert teleop._axis_value(teleop.config.z) == pytest.approx(0.1)

    teleop.current_pressed = {"w": True, "a": True}
    assert all(
        teleop._axis_value(getattr(teleop.config, axis)) == 0.0
        for axis in teleop.AXES
    )

    teleop.current_pressed = {"ctrl_r": True}
    assert teleop._gripper_key_pressed(teleop.config.gripper_open_key)
    teleop.current_pressed = {"ctrl_l": True}
    assert teleop._gripper_key_pressed(teleop.config.gripper_close_key)

    assert teleop._normalize_key(pynput_keyboard.Key.ctrl_l) == "ctrl_l"
    assert teleop._normalize_key(pynput_keyboard.Key.ctrl_r) == "ctrl_r"


def test_mujoco_keyboard_intervention_reaches_virtual_target():
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(
            viewer=False,
            teleop_mode="keyboard",
            episode_steps=10_000,
        )
    )
    net.reset(seed=0)
    try:
        robot = next(iter(net.robot_dict.values()))
        teleop = next(iter(net.teleop_dict.values()))
        initial_target = robot._virtual_target_task.copy()

        teleop.current_pressed = {"left": True}
        transition = net.step(torch.zeros(net.action_dim))

        assert transition[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION]
        assert robot._virtual_target_task[0] == pytest.approx(
            initial_target[0] + 0.1 / net.config.fps
        )
        np.testing.assert_allclose(
            robot._virtual_target_task[1:], initial_target[1:], atol=1e-12
        )
    finally:
        net.close()


def test_mujoco_pynput_ctrl_events_open_and_close_gripper():
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(
            viewer=False,
            teleop_mode="keyboard",
            episode_steps=10_000,
        )
    )
    net.reset(seed=0)
    try:
        robot = next(iter(net.robot_dict.values()))
        teleop = next(iter(net.teleop_dict.values()))

        teleop._on_press(pynput_keyboard.Key.ctrl_r)
        for _ in range(15):
            net.step(torch.zeros(net.action_dim))
        assert robot.get_observation()["gripper.pos"] < 0.1

        teleop._on_release(pynput_keyboard.Key.ctrl_r)
        teleop._on_press(pynput_keyboard.Key.ctrl_l)
        for _ in range(15):
            net.step(torch.zeros(net.action_dim))
        assert robot.get_observation()["gripper.pos"] > 0.9
    finally:
        net.close()


def test_mujoco_sac_action_stats_match_keyboard_physical_units():
    env_config = MujocoInsertionEnvConfig(viewer=False)
    action_stats = env_config.primitives["insert"].policy.dataset_stats["action"]

    assert action_stats["min"] == [-0.1, -0.1, -0.1, -0.5, -0.5, -0.5, 0.0]
    assert action_stats["max"] == [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 1.0]
