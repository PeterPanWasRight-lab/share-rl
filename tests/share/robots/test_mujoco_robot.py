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

    viewer = SimpleNamespace(
        cam=SimpleNamespace(type=None, fixedcamid=None),
        viewport=SimpleNamespace(width=1200, height=900),
        close=lambda: None,
        set_texts=lambda texts: setattr(viewer, "texts", texts),
        set_images=lambda images: setattr(viewer, "images", images),
        set_figures=lambda figures: setattr(viewer, "figures", figures),
    )
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *_args: viewer)

    robot = MujocoRobot(MujocoRobotConfig(id="test-free-camera", viewer=True, viewer_camera="free"))
    robot.connect()
    try:
        assert viewer.cam.type is None
        assert viewer.cam.fixedcamid is None
        assert "Wrist F/T" in viewer.texts[2]
        assert "Fx" in viewer.texts[2]
        assert "N" in viewer.texts[3]
        assert len(viewer.images) == 2
        front_viewport, front_image = viewer.images[0]
        wrist_viewport, wrist_image = viewer.images[1]
        assert front_image.shape == (front_viewport.height, front_viewport.width, 3)
        assert wrist_image.shape == (wrist_viewport.height, wrist_viewport.width, 3)
        assert front_viewport.left < wrist_viewport.left
        assert len(viewer.figures) == 2
        assert viewer.figures[0][1].title == "Wrist force (N)"
        assert viewer.figures[1][1].title == "Wrist torque (Nm)"
        assert viewer.figures[0][1].linepnt[0] >= 1
    finally:
        robot.disconnect()


def test_default_model_uses_vendored_menagerie_and_act_assets():
    model = build_ur5e_2f85_model()

    assert (ASSET_ROOT / "menagerie" / "universal_robots_ur5e" / "ur5e.xml").is_file()
    assert (ASSET_ROOT / "menagerie" / "robotiq_2f85" / "2f85.xml").is_file()
    assert (ASSET_ROOT / "act" / "bimanual_viperx_insertion.xml").is_file()
    assert model.nq == 21
    assert model.nu == 7
    assert model.nlight >= 3
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "diagonal-left") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "diagonal-right") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_peg_joint") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tcp") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist") >= 0
    front_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    np.testing.assert_allclose(model.cam_pos[front_camera_id], [0.40, -0.18, 0.72])
    assert model.vis.global_.azimuth == pytest.approx(225.0)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "socket-pin") == -1
    socket_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"socket-{index}")
        for index in range(1, 5)
    ]
    np.testing.assert_allclose(model.geom_pos[socket_ids[:2], 2], [-0.0315, 0.0315])
    np.testing.assert_allclose(model.geom_size[socket_ids[:2], 2], [0.0195, 0.0195])
    np.testing.assert_allclose(model.geom_pos[socket_ids[2:], 1], [0.0315, -0.0315])
    np.testing.assert_allclose(model.geom_size[socket_ids[2:], 2], [0.012, 0.012])
    tool_tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tcp")
    pinch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_pinch")
    np.testing.assert_allclose(model.site_pos[tool_tcp_id], model.site_pos[pinch_id])
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


def test_mujoco_wrench_is_rotated_from_sensor_site_to_world():
    robot = MujocoRobot(
        MujocoRobotConfig(id="test-wrench-frame", control_dt=0.004, randomize_fixture_xy=0.0)
    )
    robot.connect()
    try:
        local_force = np.array([1.0, 2.0, 3.0])
        local_torque = np.array([-0.2, 0.3, 0.4])
        robot._data.sensordata[robot._sensor_slices["tcp_force"]] = local_force
        robot._data.sensordata[robot._sensor_slices["tcp_torque"]] = local_torque
        sensor_to_world = robot._data.site_xmat[robot._force_torque_site_id].reshape(3, 3)

        wrench_world = robot._sensor_vector()

        np.testing.assert_allclose(wrench_world[:3], sensor_to_world @ local_force)
        np.testing.assert_allclose(wrench_world[3:], sensor_to_world @ local_torque)
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


def test_mujoco_domain_randomization_is_seeded_bounded_and_non_accumulating():
    config = MujocoRobotConfig(
        id="test-domain-randomization",
        control_dt=0.004,
        randomize_fixture_xy=0.01,
        randomize_fixture_z=0.001,
        randomize_fixture_yaw_deg=3.0,
        randomize_camera_position_m=0.005,
        randomize_camera_rotation_deg=1.5,
        randomize_camera_fovy_deg=2.0,
        randomize_light_intensity_fraction=0.2,
        randomize_object_color_fraction=0.1,
        randomize_contact_friction_fraction=0.15,
        randomize_peg_mass_fraction=0.15,
    )
    robot = MujocoRobot(config)
    robot.connect()
    try:
        robot.reset_simulation(seed=17)
        first = robot.domain_randomization_state
        first_fixture_pos = robot._model.body_pos[robot._fixture_body_id].copy()
        first_camera_pos = robot._model.cam_pos.copy()
        robot.reset_simulation(seed=23)
        second = robot.domain_randomization_state
        robot.reset_simulation(seed=17)

        assert first != second
        assert robot.domain_randomization_state == first
        np.testing.assert_allclose(robot._model.body_pos[robot._fixture_body_id], first_fixture_pos)
        np.testing.assert_allclose(robot._model.cam_pos, first_camera_pos)
        assert max(abs(value) for value in first["fixture_offset_m"][:2]) <= 0.01
        assert abs(first["fixture_offset_m"][2]) <= 0.001
        assert abs(first["fixture_yaw_deg"]) <= 3.0
        assert np.max(np.abs(first["camera_position_offsets_m"])) <= 0.005
        assert np.max(np.abs(first["camera_rotation_offsets_deg"])) <= 1.5
        assert all(0.8 <= value <= 1.2 for value in first["light_scales"])
        assert 0.85 <= first["peg_mass_scale"] <= 1.15
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


def test_empty_gripper_can_reach_workbench_with_pinch_tcp():
    robot = MujocoRobot(
        MujocoRobotConfig(
            id="test-empty-gripper-workbench",
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
        for _ in range(50):
            robot.send_action({"gripper.pos": 0.0})
        for _ in range(20):
            robot.send_action({"x.ee_pos": 0.3, "gripper.pos": 0.0})
        for _ in range(35):
            robot.send_action({"z.ee_pos": -0.3, "gripper.pos": 0.0})
        for _ in range(20):
            robot.send_action({"gripper.pos": 0.0})

        contact_pairs = set()
        for index in range(robot._data.ncon):
            contact = robot._data.contact[index]
            names = {
                mujoco.mj_id2name(
                    robot._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                ),
                mujoco.mj_id2name(
                    robot._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                ),
            }
            contact_pairs.add(frozenset(names))

        assert robot._tcp_world_pose()[2] == pytest.approx(0.05, abs=0.005)
        assert any(
            "workbench" in pair
            and ("gripper_left_pad1" in pair or "gripper_right_pad1" in pair)
            for pair in contact_pairs
        )
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
        assert rotation_error < 3e-3
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
            release_steps=30,
        )
    )
    net.reset(seed=0)
    try:
        assert net.action_dim == 4
        assert net.cameras["wrist"].async_read().shape == (64, 64, 3)

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
        for _ in range(30):
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

    teleop.current_pressed = {".": True}
    assert teleop._gripper_key_pressed(teleop.config.gripper_open_key)
    teleop.current_pressed = {",": True}
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


def test_mujoco_pynput_period_and_comma_open_and_close_gripper():
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(
            viewer=False,
            teleop_mode="keyboard",
            episode_steps=10_000,
            success_insertion_depth=10.0,
        )
    )
    net.reset(seed=0)
    try:
        robot = next(iter(net.robot_dict.values()))
        teleop = next(iter(net.teleop_dict.values()))

        period_key = SimpleNamespace(char=".")
        comma_key = SimpleNamespace(char=",")

        teleop._on_press(period_key)
        for _ in range(10):
            net.step(torch.zeros(net.action_dim))
        assert robot._last_gripper == pytest.approx(1.0)
        for _ in range(20):
            net.step(torch.zeros(net.action_dim))
            if robot._last_gripper == pytest.approx(0.0):
                break
        assert robot._last_gripper == pytest.approx(0.0)

        teleop._on_release(period_key)
        teleop._on_press(comma_key)
        for _ in range(10):
            net.step(torch.zeros(net.action_dim))
        assert robot._last_gripper == pytest.approx(0.0)
        assert robot.get_observation()["gripper.pos"] < 0.1
        for _ in range(30):
            net.step(torch.zeros(net.action_dim))
        assert robot.get_observation()["gripper.pos"] > 0.9
    finally:
        net.close()


@pytest.mark.parametrize(
    ("key", "event", "reward", "reason", "target"),
    [
        ("/", TeleopEvents.FAILURE, 0.0, "manual_failure", "done"),
        ("enter", TeleopEvents.SUCCESS, 1.0, "manual_success", "release"),
    ],
)
def test_mujoco_keyboard_manual_episode_outcomes(key, event, reward, reason, target):
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(
            viewer=False,
            teleop_mode="keyboard",
            episode_steps=10_000,
        )
    )
    net.reset(seed=0)
    try:
        teleop = next(iter(net.teleop_dict.values()))
        pynput_key = SimpleNamespace(char=key) if len(key) == 1 else key
        teleop._on_press(pynput_key)
        teleop._on_release(pynput_key)

        transition = net.step(torch.zeros(net.action_dim))
        info = transition[TransitionKey.INFO]

        assert info[event.value]
        assert transition[TransitionKey.DONE]
        assert transition[TransitionKey.REWARD] == pytest.approx(reward)
        assert info["transition_reason"] == reason
        assert net.active_primitive == target
    finally:
        net.close()


def test_mujoco_keyboard_escape_requests_process_stop_without_disconnect():
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(
            viewer=False,
            teleop_mode="keyboard",
            episode_steps=10_000,
        )
    )
    net.reset(seed=0)
    try:
        teleop = next(iter(net.teleop_dict.values()))
        teleop._on_press("esc")

        transition = net.step(torch.zeros(net.action_dim))

        assert transition[TransitionKey.INFO][TeleopEvents.STOP_RECORDING.value]
        assert not transition[TransitionKey.DONE]
        assert teleop.config.escape_disconnects is False
    finally:
        net.close()


def test_mujoco_sac_action_stats_match_keyboard_physical_units():
    env_config = MujocoInsertionEnvConfig(viewer=False)
    action_stats = env_config.primitives["insert"].policy.dataset_stats["action"]

    assert action_stats["min"] == [-0.1, -0.1, -0.1, 0.0]
    assert action_stats["max"] == [0.1, 0.1, 0.1, 1.0]


def test_mujoco_success_rewards_and_requests_next_episode_reset():
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(viewer=False, episode_steps=900)
    )
    net.reset(seed=0)
    try:
        robot = next(iter(net.robot_dict.values()))
        assert net.config.episode_steps == 900
        assert net.config.success_lateral_tolerance == pytest.approx(0.002)
        for step in range(220):
            fixture_rotation = robot._data.xmat[robot._fixture_body_id].reshape(3, 3)
            tip_relative = fixture_rotation.T @ (
                robot._data.site_xpos[robot._peg_tip_site_id]
                - robot._data.xpos[robot._fixture_body_id]
            )
            correction_world = fixture_rotation @ np.array(
                [0.0, -tip_relative[1], -tip_relative[2]]
            )
            action = torch.zeros(net.action_dim)
            action[0] = float(np.clip(8.0 * correction_world[0], -0.08, 0.08))
            action[1] = float(np.clip(8.0 * correction_world[1], -0.08, 0.08))
            action[2] = -0.03 if step >= 30 else 0.0
            action[-1] = 1.0
            transition = net.step(action)
            if transition[TransitionKey.DONE]:
                break

        assert transition[TransitionKey.DONE]
        assert not transition[TransitionKey.TRUNCATED]
        assert transition[TransitionKey.REWARD] == pytest.approx(1.0)
        assert transition[TransitionKey.INFO]["transition_reason"] == "peg_inserted"
        observation = robot.get_observation()
        assert observation["insertion.depth"] >= net.config.success_insertion_depth
        assert (
            observation["insertion.lateral_error"]
            <= net.config.success_lateral_tolerance
        )

        net.request_full_reset()
        net.reset(seed=1)
        assert net.active_primitive == "insert"
    finally:
        net.close()


def test_mujoco_misaligned_peg_does_not_succeed_when_commanded_below_table():
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(viewer=False, episode_steps=900)
    )
    net.reset(seed=0)
    try:
        robot = next(iter(net.robot_dict.values()))
        for _ in range(12):
            action = torch.zeros(net.action_dim)
            action[0] = 0.1
            action[-1] = 1.0
            transition = net.step(action)
        for _ in range(95):
            action = torch.zeros(net.action_dim)
            action[2] = -0.1
            action[-1] = 1.0
            transition = net.step(action)

        observation = robot.get_observation()
        assert net.config.primitives["insert"].task_frame["main"].min_pose[2] == pytest.approx(-1.2)
        assert robot._virtual_target_task[2] < 0.05
        assert (
            observation["insertion.lateral_error"]
            > net.config.success_lateral_tolerance
        )
        assert not transition[TransitionKey.DONE]
        assert transition[TransitionKey.INFO]["transition_reason"] is None
    finally:
        net.close()


def test_mujoco_full_reset_clears_stale_open_gripper_command():
    net = ManipulationPrimitiveNet(
        MujocoInsertionEnvConfig(
            viewer=False,
            teleop_mode="keyboard",
            episode_steps=900,
        )
    )
    net.set_step_info({TeleopEvents.IS_INTERVENTION: True})
    net.reset(seed=0)
    try:
        robot = next(iter(net.robot_dict.values()))
        teleop = next(iter(net.teleop_dict.values()))

        teleop.set_gripper_position(0.0)
        for _ in range(20):
            net.step(torch.zeros(net.action_dim))
        assert robot.get_observation()["gripper.pos"] < 0.1

        net.request_full_reset()
        net.reset(seed=1)
        assert teleop.get_action()["gripper.pos"] == pytest.approx(1.0)

        peg_z_before = float(robot._data.xpos[robot._peg_body_id, 2])
        for _ in range(5):
            net.step(torch.zeros(net.action_dim))
        peg_z_after = float(robot._data.xpos[robot._peg_body_id, 2])
        assert robot.get_observation()["gripper.pos"] > 0.7
        assert peg_z_after == pytest.approx(peg_z_before, abs=0.01)
    finally:
        net.close()
