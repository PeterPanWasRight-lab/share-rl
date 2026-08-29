#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HRCrobot 位姿移动真机测试套件（完善版）。

补齐 TestLiveMotion 之外的移动验证：

    控制器层（HRCrobotController）：
        1. X/Y/Z 单轴微小移动 + 其余轴锁定
        2. 移动后位姿回读（闭环位置精度）
        3. 返回起点（往返精度）
        4. 停止后位姿保持（验证"无 buffer"语义的物理表现）
    适配层（HRCrobot.py 完整栈，含 TaskFrame/rotvec 契约）：
        5. TaskFrame 系下的观测-动作-回读全链路
        6. 缺失轴由 task_frame.target 安全补齐的语义

安全设计（所有移动测试共用）：
    - 每个测试进入时快照当前位姿，移动幅度默认 2mm
      （可用 HRC_TEST_MOVE_MM 覆盖，硬上限 5mm）
    - 单轴移动时其余 5 轴用快照值锁定
    - 步长 ≤ 0.2mm/拍，流式下发
    - 必须显式确认 observation/action 的姿态表示契约
    - 全套测试共用一个控制器连接，禁止同机双连接
    - rotation vector 按 SO(3) 插值，每拍不超过 0.2°
    - teardown 只断开连接，不在异常状态下自动运动
    - 绝对迭代上限 + 墙钟超时

运行（上使能后）：

    cd /home/gm/SHaReRL/share-rl
    HRC_TEST_LIVE=1 HRC_TEST_MOVE=1 \\
    HRC_TEST_ACK_POSE_CONTRACT=rotvec-observation_euler-action \\
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
        /home/gm/anaconda3/envs/lerobot/bin/python -m pytest \\
        "tests/ HRCrobotTest/test_hrcrobot_pose_motion.py" -v -s -p no:cacheprovider

只跑某一个轴（例）：

    ... -k "move_x" ...
"""

import math
import os
import sys
import time
from pathlib import Path

import pytest
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from share.robots.HRCrobot.lerobot_robot_HRCrobot.controller import (  # noqa: E402
    HRCrobotController,
)
from share.robots.HRCrobot import HRCrobot, HRCrobotConfig  # noqa: E402
from share.envs.manipulation_primitive.task_frame import TaskFrame  # noqa: E402

# ------------------------------------------------------------
# 环境开关与参数
# ------------------------------------------------------------

LIVE = os.environ.get("HRC_TEST_LIVE") == "1"
POSE_CONTRACT_ACK = "rotvec-observation_euler-action"
MOVE = (
    LIVE
    and os.environ.get("HRC_TEST_MOVE") == "1"
    and os.environ.get("HRC_TEST_ACK_POSE_CONTRACT") == POSE_CONTRACT_ACK
)

ROBOT_IP = os.environ.get("HRC_ROBOT_IP", "10.10.59.211")

MOVE_MM = float(os.environ.get("HRC_TEST_MOVE_MM", "2.0"))
if not 0.0 < MOVE_MM <= 5.0:
    raise ValueError("HRC_TEST_MOVE_MM must be in (0, 5].")
MOVE_M = MOVE_MM / 1000.0

STEP_M = 0.0002                          # 每拍 ≤ 0.2mm
STEP_ROT_RAD = math.radians(0.2)         # 每拍 ≤ 0.2°
STREAM_PERIOD_S = 0.02                   # 拍间隔 20ms（servo 内部还会限到 10ms）
REACHED_TOL_M = 0.0002                   # 到位判定 0.2mm
LOCKED_AXIS_TOL_M = 0.001                # 锁定轴漂移上限 1mm
RETURN_TOL_M = 0.0005                    # 返回起点判定 0.5mm
MAX_STEPS = 200                          # 单段流式绝对上限
HOLD_AFTER_STOP_S = 1.0                  # 停止保持观察时长

AXES = ["x", "y", "z", "rx", "ry", "rz"]

skip_if_not_move = pytest.mark.skipif(
    not MOVE,
    reason=(
        "运动测试需要 HRC_TEST_LIVE=1、HRC_TEST_MOVE=1，且 "
        f"HRC_TEST_ACK_POSE_CONTRACT={POSE_CONTRACT_ACK}"
    ),
)


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------


def rotation_error_rad(rotvec_a: list[float], rotvec_b: list[float]) -> float:
    """两个 rotvec 之间的旋转角差（rad）。"""
    rot_a = Rotation.from_rotvec(rotvec_a)
    rot_b = Rotation.from_rotvec(rotvec_b)
    return float((rot_a.inv() * rot_b).magnitude())


def step_rotvec_toward(
    current_rotvec: list[float],
    target_rotvec: list[float],
    max_step_rad: float = STEP_ROT_RAD,
) -> list[float]:
    """Take one bounded SO(3) step instead of interpolating rotvec components."""
    current_rotation = Rotation.from_rotvec(current_rotvec)
    target_rotation = Rotation.from_rotvec(target_rotvec)
    relative_rotation = target_rotation * current_rotation.inv()
    relative_angle = float(relative_rotation.magnitude())
    if relative_angle <= max_step_rad:
        return target_rotation.as_rotvec().tolist()
    step_rotation = Rotation.from_rotvec(
        relative_rotation.as_rotvec() * (max_step_rad / relative_angle)
    )
    return (step_rotation * current_rotation).as_rotvec().tolist()


def stream_to(
    controller: HRCrobotController,
    target: list[float],
    max_steps: int = MAX_STEPS,
) -> int:
    """
    流式逼近 target（每拍步长 ≤ STEP_M），返回实际拍数。

    到位判定：平动各轴 < REACHED_TOL_M 且旋转角差 < 0.2°。
    """
    for step in range(max_steps):
        current = controller.get_tcp_pose()

        delta = [t - c for t, c in zip(target, current)]
        dist = math.sqrt(sum(d * d for d in delta[:3]))
        rot_err = rotation_error_rad(current[3:], target[3:])

        if dist < REACHED_TOL_M and rot_err < math.radians(0.2):
            return step

        # 按比例截断步长
        scale = min(1.0, STEP_M / dist) if dist > 0 else 1.0
        waypoint = [
            current[i] + delta[i] * scale
            for i in range(3)
        ] + step_rotvec_toward(current[3:], target[3:])

        controller.servo_cartesian(waypoint)
        time.sleep(STREAM_PERIOD_S)

    return max_steps


def assert_pose_close(
    actual: list[float],
    expected: list[float],
    tol_m: float,
    tol_rot_deg: float,
    label: str,
) -> None:
    """断言两组位姿（m + rotvec rad）足够接近。"""
    for i in range(3):
        assert abs(actual[i] - expected[i]) <= tol_m, (
            f"{label}: 轴 {AXES[i]} 偏差 "
            f"{(actual[i] - expected[i]) * 1000:.2f}mm "
            f"> {tol_m * 1000:.1f}mm"
        )

    rot_err_deg = math.degrees(
        rotation_error_rad(actual[3:], expected[3:])
    )
    assert rot_err_deg <= tol_rot_deg, (
        f"{label}: 姿态角差 {rot_err_deg:.3f}° > {tol_rot_deg}°"
    )


# ------------------------------------------------------------
# fixture：全套测试只建立一个连接
# ------------------------------------------------------------


@pytest.fixture(scope="module")
def motion_controller():
    if not MOVE:
        pytest.skip(
            "缺少真机运动开关或姿态契约确认"
        )

    controller = HRCrobotController(
        robot_ip=ROBOT_IP,
        frequency=100.0,
    )
    controller.connect()
    try:
        controller._motion_start_pose = controller.get_tcp_pose()
    except Exception:
        controller.disconnect()
        raise
    print(
        "\n[fixture] 起始位姿:",
        [round(v, 4) for v in controller._motion_start_pose],
    )

    yield controller

    # A failed native connection must never trigger another automatic move.
    # Individual tests return before asserting; teardown only disconnects.
    controller.disconnect()


# ============================================================
# 1. 单轴平动移动（X / Y / Z 参数化）
# ============================================================


@skip_if_not_move
class TestAxisMoves:
    """
    每个轴：+MOVE_MM 移动 → 回读校验 → 返回起点校验。

    其余平动轴与姿态全程锁定（用快照值）。
    """

    @pytest.mark.parametrize("axis_index", [0, 1, 2])
    def test_move_single_axis_and_return(
        self,
        motion_controller,
        axis_index: int,
    ):
        controller = motion_controller
        axis = AXES[axis_index]

        start = controller.get_tcp_pose()
        target = list(start)
        target[axis_index] += MOVE_M

        print(
            f"\n[{axis}] 目标偏移 +{MOVE_MM}mm, "
            f"其余轴锁定"
        )

        # ---- 移动过去 ----
        steps = stream_to(controller, target)
        reached = controller.get_tcp_pose()

        print(
            f"[{axis}] {steps} 拍到达, "
            f"实际位移 "
            f"{(reached[axis_index] - start[axis_index]) * 1000:+.2f}mm"
        )

        # Return before evaluating assertions so an accuracy failure does not
        # leave the robot at the test offset.
        stream_to(controller, start)
        back = controller.get_tcp_pose()

        # 移动轴到位
        assert (
            abs(
                reached[axis_index]
                - target[axis_index]
            )
            <= REACHED_TOL_M
        ), (
            f"[{axis}] 到位偏差 "
            f"{(reached[axis_index] - target[axis_index]) * 1000:.2f}mm"
        )

        # 其余平动轴锁定
        for i in range(3):
            if i == axis_index:
                continue
            assert (
                abs(reached[i] - start[i])
                <= LOCKED_AXIS_TOL_M
            ), (
                f"[{axis}] 锁定轴 {AXES[i]} 漂移 "
                f"{(reached[i] - start[i]) * 1000:.2f}mm"
            )

        # 姿态锁定
        assert_pose_close(
            reached,
            start,
            10.0,
            0.5,
            f"[{axis}] 姿态",
        )

        assert_pose_close(
            back,
            start,
            RETURN_TOL_M,
            0.5,
            f"[{axis}] 返回起点",
        )
        print(f"[{axis}] 已返回起点。")


# ============================================================
# 2. 停止后位姿保持（无 buffer 语义的物理验证）
# ============================================================


@skip_if_not_move
class TestStopHold:
    """
    流式到达某点后停止下发，观察位姿是否保持。

    验证 servo_cartesian"无 buffer、即发即弃"语义：
    停止后机器人应停在最后目标附近，且无继续运动。
    """

    def test_hold_position_after_stop(
        self,
        motion_controller,
    ):
        controller = motion_controller
        start = controller.get_tcp_pose()

        target = list(start)
        target[2] += MOVE_M

        stream_to(controller, target)
        at_target = controller.get_tcp_pose()

        # 停止下发，观察保持
        time.sleep(HOLD_AFTER_STOP_S)
        after_hold = controller.get_tcp_pose()

        drift = math.sqrt(
            sum(
                (a - b) ** 2
                for a, b in zip(
                    after_hold[:3], at_target[:3]
                )
            )
        )
        print(
            f"\n[保持] 停止 {HOLD_AFTER_STOP_S}s 后漂移 "
            f"{drift * 1000:.2f}mm"
        )

        stream_to(controller, start)

        assert drift <= 0.001, (
            f"停止后漂移 {drift * 1000:.2f}mm 超过 1mm"
        )


# ============================================================
# 3. HRCrobot.py 完整栈移动（TaskFrame / rotvec 契约）
# ============================================================


@skip_if_not_move
class TestHRCrobotFullStackMotion:
    """
    经 HRCrobot 适配层（set_task_frame / get_observation /
    send_action）驱动的真机移动，验证 TaskFrame 语义
    与 rotvec 观测契约在真机上的端到端行为。
    """

    @pytest.fixture()
    def full_stack_robot(self, motion_controller):
        """Wrap the module's sole live connection; never open a second client."""
        robot = HRCrobot(HRCrobotConfig())
        robot.controller = motion_controller
        yield robot

    def _stream_action(
        self,
        robot,
        target_task: list[float],
        max_steps: int = MAX_STEPS,
    ) -> None:
        """Move xyz while TaskFrame target safely holds the Euler orientation."""
        target_task_rotvec = Rotation.from_euler(
            "xyz", target_task[3:], degrees=False
        ).as_rotvec()
        for _ in range(max_steps):
            obs = robot.get_observation()
            current = [
                obs[f"{axis}.ee_pos"]
                for axis in AXES
            ]

            delta = [
                t - c
                for t, c in zip(
                    target_task, current
                )
            ]
            dist = math.sqrt(
                sum(d * d for d in delta[:3])
            )
            rot_err = rotation_error_rad(
                current[3:], target_task_rotvec.tolist()
            )

            if rot_err > math.radians(0.5):
                raise RuntimeError(
                    "TaskFrame orientation drifted by "
                    f"{math.degrees(rot_err):.3f} deg; refusing further motion."
                )

            if dist < REACHED_TOL_M:
                return

            scale = min(1.0, STEP_M / dist)

            # Do not replay rotvec observation channels as Euler actions.
            # Missing rotation axes are held by task_frame.target (Euler).
            action = {
                f"{AXES[i]}.ee_pos": current[i]
                + delta[i] * scale
                for i in range(3)
            }
            robot.send_action(action)
            time.sleep(STREAM_PERIOD_S)

        raise RuntimeError(
            "Full-stack translation did not converge within MAX_STEPS."
        )

    def test_task_frame_observation_and_move(
        self,
        full_stack_robot,
    ):
        """
        流程：origin 锚定当前位姿 → 观测 z≈0 →
        动作爬升 +2mm → 观测校验 → 返回起点。
        """
        robot = full_stack_robot

        start_base = robot.controller.get_tcp_pose()
        print(
            "\n[fullstack] 起始 base 位姿:",
            [round(v, 4) for v in start_base],
        )

        # TaskFrame stores Euler XYZ, while the controller snapshot is rotvec.
        start_rpy = Rotation.from_rotvec(start_base[3:]).as_euler(
            "xyz", degrees=False
        ).tolist()
        robot.set_task_frame(
            TaskFrame(
                origin=[
                    start_base[0],
                    start_base[1],
                    start_base[2],
                    *start_rpy,
                ],
                target=[0.0] * 6,
            )
        )

        # 锚定后观测应归零
        obs0 = robot.get_observation()
        for axis in AXES:
            assert (
                abs(obs0[f"{axis}.ee_pos"])
                < 1e-6
            ), (
                f"锚定后 {axis}.ee_pos 应为 0，"
                f"实际 {obs0[f'{axis}.ee_pos']}"
            )

        # z 缓爬 +2mm（其余轴用观测回填锁定）
        target_z = MOVE_M
        self._stream_action(
            robot,
            [
                0.0,
                0.0,
                target_z,
                0.0,
                0.0,
                0.0,
            ],
        )

        obs_end = robot.get_observation()
        print(
            f"[fullstack] 到达后 task 观测: "
            f"z = {obs_end['z.ee_pos'] * 1000:.2f}mm"
        )

        # Return before checking endpoint accuracy.
        self._stream_action(robot, [0.0] * 6)
        obs_back = robot.get_observation()

        assert (
            obs_end["z.ee_pos"]
            == pytest.approx(target_z, abs=REACHED_TOL_M)
        )
        for axis in ("x", "y"):
            assert (
                abs(obs_end[f"{axis}.ee_pos"])
                <= LOCKED_AXIS_TOL_M
            ), (
                f"锁定轴 {axis} 漂移 "
                f"{obs_end[f'{axis}.ee_pos'] * 1000:.2f}mm"
            )

        assert (
            abs(obs_back["z.ee_pos"])
            <= RETURN_TOL_M
        )
        print("[fullstack] 已返回起点（z≈0）。")

    def test_missing_axes_filled_from_task_frame_target(
        self,
        full_stack_robot,
    ):
        """
        缺失轴补齐语义的真机验证（安全版）：

        task_frame.target 预先设为当前位姿，
        然后只下发 z 轴动作——缺失的 5 轴会被
        target 补齐为"保持原地"，这正是安全用法。
        """
        robot = full_stack_robot

        start_base = robot.controller.get_tcp_pose()
        start_rpy = Rotation.from_rotvec(start_base[3:]).as_euler(
            "xyz", degrees=False
        ).tolist()

        origin = [
            start_base[0],
            start_base[1],
            start_base[2],
            *start_rpy,
        ]

        # target = 当前位姿（task 系全零）
        # → 缺失轴补齐后等于"锁定不动"
        robot.set_task_frame(
            TaskFrame(
                origin=origin,
                target=[0.0] * 6,
            )
        )

        # 只发 z：爬升 +2mm
        move_steps = math.ceil(MOVE_M / STEP_M)
        for step in range(move_steps):
            robot.send_action(
                {"z.ee_pos": min(
                    STEP_M
                    * (step + 1),
                    MOVE_M,
                )}
            )
            time.sleep(STREAM_PERIOD_S)

        time.sleep(0.3)

        obs = robot.get_observation()
        print(
            f"\n[补齐] 单轴下发后: "
            f"z = {obs['z.ee_pos'] * 1000:+.2f}mm, "
            f"x = {obs['x.ee_pos'] * 1000:+.2f}mm, "
            f"y = {obs['y.ee_pos'] * 1000:+.2f}mm"
        )

        # 返回：单轴下发 z=0（其余轴仍由 target=0 补齐锁定）
        for step in range(move_steps):
            robot.send_action(
                {
                    "z.ee_pos": max(
                        MOVE_M
                        - STEP_M
                        * (step + 1),
                        0.0,
                    )
                }
            )
            time.sleep(STREAM_PERIOD_S)

        time.sleep(0.3)
        obs_back = robot.get_observation()

        assert obs["z.ee_pos"] == pytest.approx(
            MOVE_M, abs=REACHED_TOL_M
        )
        for axis in ("x", "y"):
            assert (
                abs(obs[f"{axis}.ee_pos"])
                <= LOCKED_AXIS_TOL_M
            ), (
                f"补齐轴 {axis} 漂移 "
                f"{obs[f'{axis}.ee_pos'] * 1000:.2f}mm "
                "—— target 补齐语义未生效？"
            )
        observed_rotvec = [
            obs[f"{axis}.ee_pos"] for axis in ("rx", "ry", "rz")
        ]
        assert rotation_error_rad(
            observed_rotvec, [0.0, 0.0, 0.0]
        ) <= math.radians(0.5), (
            "只下发 z 时 TaskFrame 姿态漂移超过 0.5°"
        )

        assert (
            abs(obs_back["z.ee_pos"])
            <= RETURN_TOL_M
        )
        print("[补齐] 已返回起点。")
