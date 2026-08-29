from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    TaskFrame,
)
from share.robots.HRCrobot import HRCrobot, HRCrobotConfig
from share.utils.transformation_utils import task_pose_to_world_pose


class _FakeHRCController:
    def __init__(self, tcp_pose: list[float] | None = None) -> None:
        self.is_connected = True
        self.tcp_pose = list(tcp_pose or [0.0] * 6)
        self.gripper_position = 0.0
        self.servo_calls: list[list[float]] = []
        self.gripper_calls: list[float] = []

    def get_tcp_pose(self) -> list[float]:
        return list(self.tcp_pose)

    def get_gripper_position(self) -> float:
        return self.gripper_position

    def servo_cartesian(self, pose: list[float]) -> None:
        self.servo_calls.append(list(pose))

    def set_gripper(self, target: float) -> None:
        self.gripper_position = float(target)
        self.gripper_calls.append(float(target))


def _robot(*, min_gripper_interval_s: float = 0.0) -> tuple[HRCrobot, _FakeHRCController]:
    robot = HRCrobot(
        HRCrobotConfig(
            use_gripper=True,
            gripper_min_command_interval_s=min_gripper_interval_s,
        )
    )
    controller = _FakeHRCController()
    robot.controller = controller
    return robot, controller


def test_observation_uses_ur_mujoco_rotation_vector_contract() -> None:
    robot, controller = _robot()
    origin = [0.40, -0.20, 0.10, 0.20, -0.10, 0.30]
    task_pose_rpy = [0.03, -0.02, 0.08, 0.15, 0.05, -0.20]
    robot.set_task_frame(TaskFrame(origin=origin))

    world_pose_rpy = task_pose_to_world_pose(task_pose_rpy, origin)
    controller.tcp_pose = robot._world_rpy_to_vendor_pose(world_pose_rpy)
    observation = robot.get_observation()

    expected_rotvec = Rotation.from_euler("xyz", task_pose_rpy[3:]).as_rotvec()
    actual = np.array(
        [observation[f"{axis}.ee_pos"] for axis in ("x", "y", "z", "rx", "ry", "rz")]
    )
    np.testing.assert_allclose(actual[:3], task_pose_rpy[:3], atol=1e-10)
    np.testing.assert_allclose(actual[3:], expected_rotvec, atol=1e-10)


def test_hrcrobot_and_mujoco_publish_identical_pose_for_same_tcp_transform() -> None:
    pytest.importorskip("mujoco")
    from share.robots.mujoco import MujocoRobot, MujocoRobotConfig

    sim = MujocoRobot(MujocoRobotConfig(use_gripper=False, viewer=False))
    sim.connect()
    try:
        frame = TaskFrame(origin=[0.02, -0.01, 0.03, 0.10, -0.15, 0.20])
        sim.set_task_frame(frame)
        sim_observation = sim.get_observation()

        robot, controller = _robot()
        robot.set_task_frame(frame)
        controller.tcp_pose = robot._world_rpy_to_vendor_pose(
            sim._tcp_world_pose().tolist()
        )
        hrc_observation = robot.get_observation()

        for axis in ("x", "y", "z", "rx", "ry", "rz"):
            assert hrc_observation[f"{axis}.ee_pos"] == pytest.approx(
                sim_observation[f"{axis}.ee_pos"], abs=1e-10
            )
    finally:
        sim.disconnect()


def test_task_pose_action_reaches_controller_as_world_rotation_vector() -> None:
    robot, controller = _robot()
    origin = [0.35, 0.08, 0.12, -0.10, 0.20, 0.25]
    target = [0.02, 0.01, 0.15, 0.12, -0.08, 0.18]
    robot.set_task_frame(TaskFrame(origin=origin, target=[0.0] * 6))

    action = {
        f"{axis}.ee_pos": target[index]
        for index, axis in enumerate(("x", "y", "z", "rx", "ry", "rz"))
    }
    robot.send_action(action)

    expected_world = task_pose_to_world_pose(target, origin)
    expected_vendor = robot._world_rpy_to_vendor_pose(expected_world)
    assert len(controller.servo_calls) == 1
    np.testing.assert_allclose(controller.servo_calls[0], expected_vendor, atol=1e-10)


def test_hrcrobot_rejects_control_modes_not_supported_by_ur_position_sim() -> None:
    robot, _ = _robot()

    with pytest.raises(ValueError, match="task-space control only"):
        robot.set_task_frame(
            TaskFrame(
                space=ControlSpace.JOINT,
                origin=None,
                target=[0.0] * 6,
                joint_names=[f"joint_{index}" for index in range(6)],
            )
        )

    with pytest.raises(ValueError, match="POS control only"):
        robot.set_task_frame(
            TaskFrame(
                control_mode=[ControlMode.VEL] + [ControlMode.POS] * 5,
                policy_mode=[None] * 6,
            )
        )


def test_gripper_uses_same_normalized_rate_limiter_as_mujoco() -> None:
    robot, controller = _robot(min_gripper_interval_s=10.0)

    assert robot.send_action({"gripper.pos": 1.0})["gripper.pos"] == 1.0
    assert robot.send_action({"gripper.pos": 0.0})["gripper.pos"] == 1.0
    assert controller.gripper_calls == [1.0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frequency", 0.0),
        ("gripper_threshold", 1.1),
        ("gripper_min_command_interval_s", -0.1),
    ],
)
def test_hrcrobot_alignment_config_validation(field: str, value: float) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        HRCrobotConfig(**kwargs)
