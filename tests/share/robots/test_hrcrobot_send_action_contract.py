from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    TASK_FRAME_AXIS_NAMES,
    TaskFrame,
)
from share.robots.HRCrobot import HRCrobot, HRCrobotConfig
from share.utils.transformation_utils import (
    euler_xyz_from_rotvec,
    task_pose_to_world_pose,
)


class _FakeHRCController:
    """controller.py 的纯内存替身：记录调用，不碰硬件。"""

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


def _robot(
    *,
    min_gripper_interval_s: float = 0.0,
    use_gripper: bool = True,
) -> tuple[HRCrobot, _FakeHRCController]:
    robot = HRCrobot(
        HRCrobotConfig(
            use_gripper=use_gripper,
            gripper_min_command_interval_s=min_gripper_interval_s,
        )
    )
    controller = _FakeHRCController()
    robot.controller = controller
    return robot, controller


# ============================================================
# set_task_frame 校验与深拷贝
# ============================================================


def test_set_task_frame_accepts_task_space_pos_frame() -> None:
    robot, _ = _robot()

    frame = TaskFrame(origin=[0.40, -0.10, 0.05, 0.0, 0.0, 0.0])
    robot.set_task_frame(frame)

    assert robot.task_frame.origin == frame.origin


def test_set_task_frame_deep_copies_input() -> None:
    """调用方随后修改原始 TaskFrame 不得影响机器人内部状态。"""
    robot, _ = _robot()

    frame = TaskFrame(origin=[0.40, -0.10, 0.05, 0.0, 0.0, 0.0])
    robot.set_task_frame(frame)

    frame.origin[0] = 999.0
    frame.target[2] = 999.0

    assert robot.task_frame.origin[0] == 0.40
    assert robot.task_frame.target[2] == 0.0


def test_set_task_frame_rejects_wrench_control_mode() -> None:
    """非 POS 的任何 ControlMode（含 WRENCH）都应被拒绝。"""
    robot, _ = _robot()

    with pytest.raises(ValueError, match="POS control only"):
        robot.set_task_frame(
            TaskFrame(
                control_mode=[ControlMode.POS] * 5 + [ControlMode.WRENCH],
                policy_mode=[None] * 6,
            )
        )


# ============================================================
# action_features 派生自 task_frame
# ============================================================


def test_action_features_derive_from_task_frame() -> None:
    robot, _ = _robot()

    expected = {f"{axis}.ee_pos": float for axis in TASK_FRAME_AXIS_NAMES}
    expected["gripper.pos"] = float

    assert robot.action_features == expected


def test_action_features_without_gripper() -> None:
    robot, _ = _robot(use_gripper=False)

    assert "gripper.pos" not in robot.action_features
    assert "gripper.pos" not in robot.observation_features


def test_action_features_survive_task_frame_switch() -> None:
    """切换 TaskFrame（仍为 TASK/POS）后动作键保持 ee_pos 形态。"""
    robot, _ = _robot()

    robot.set_task_frame(TaskFrame(origin=[0.30, 0.05, 0.08, 0.1, 0.0, 0.0]))

    assert set(robot.action_features) == {
        f"{axis}.ee_pos" for axis in TASK_FRAME_AXIS_NAMES
    } | {"gripper.pos"}


# ============================================================
# send_action 的 executed_action 契约
# ============================================================


def test_send_action_returns_executed_gripper_target_on_suppression() -> None:
    """被限频抑制时，返回值必须反映硬件实际执行的（旧）目标。"""
    robot, controller = _robot(min_gripper_interval_s=10.0)

    first = robot.send_action({"gripper.pos": 1.0})
    assert first["gripper.pos"] == 1.0
    assert controller.gripper_calls == [1.0]

    second = robot.send_action({"gripper.pos": 0.0})
    # 冷却期内 0.0 被抑制：返回的是仍在执行的 1.0。
    assert second["gripper.pos"] == 1.0
    assert controller.gripper_calls == [1.0]  # 没有新的硬件调用


def test_send_action_does_not_mutate_input_action() -> None:
    robot, _ = _robot(min_gripper_interval_s=10.0)

    action = {"gripper.pos": 0.0}
    returned = robot.send_action({"gripper.pos": 1.0})
    robot.send_action(action)

    # 输入字典不被修改，返回值是新字典。
    assert action["gripper.pos"] == 0.0
    assert returned is not action


def test_send_action_without_gripper_ignores_gripper_channel() -> None:
    robot, controller = _robot(use_gripper=False)

    returned = robot.send_action({"gripper.pos": 1.0})

    assert returned["gripper.pos"] == 1.0
    assert controller.gripper_calls == []


def test_send_action_not_connected_raises() -> None:
    robot, controller = _robot()
    controller.is_connected = False

    from lerobot.utils.errors import DeviceNotConnectedError

    with pytest.raises(DeviceNotConnectedError):
        robot.send_action({"gripper.pos": 1.0})


# ============================================================
# 观测/动作旋转表示契约
#
# 观测 (*).ee_pos 发布 rotation vector；
# 动作 (*.ee_pos) 按 TaskFrame 约定接收 extrinsic XYZ euler。
# 与 MuJoCo 仿真侧一致（SIM_ALIGNMENT.md），此测试在
# HRCrobot 层固化该往返契约。
# ============================================================


def test_observation_rotvec_round_trips_through_action_to_same_vendor_pose() -> None:
    robot, controller = _robot()

    origin = [0.35, 0.08, 0.12, -0.10, 0.20, 0.25]
    task_pose_rpy = [0.02, 0.01, 0.15, 0.12, -0.08, 0.18]
    robot.set_task_frame(TaskFrame(origin=origin, target=[0.0] * 6))

    # 让"硬件"停在 task 系 task_pose_rpy 对应的世界位姿上。
    world_pose = task_pose_to_world_pose(task_pose_rpy, origin)
    controller.tcp_pose = robot._world_rpy_to_vendor_pose(world_pose)

    # 1. 观测：旋转通道应为 rotvec。
    observation = robot.get_observation()
    observed = [observation[f"{axis}.ee_pos"] for axis in TASK_FRAME_AXIS_NAMES]
    expected_rotvec = Rotation.from_euler("xyz", task_pose_rpy[3:], degrees=False).as_rotvec()
    np.testing.assert_allclose(observed[3:], expected_rotvec, atol=1e-10)

    # 2. 正确的动作消费者把观测 rotvec 转回 euler 再下发。
    action = {
        f"{axis}.ee_pos": observed[i]
        for i, axis in enumerate(TASK_FRAME_AXIS_NAMES)
    }
    action["rx.ee_pos"], action["ry.ee_pos"], action["rz.ee_pos"] = (
        euler_xyz_from_rotvec(observed[3:])
    )
    robot.send_action(action)

    # 3. 到达控制器的厂商位姿应与当前位姿一致（"保持不动"）。
    assert len(controller.servo_calls) == 1
    np.testing.assert_allclose(controller.servo_calls[0], controller.tcp_pose, atol=1e-9)
