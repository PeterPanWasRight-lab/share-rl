#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HRCrobotController 测试。

对应实现：
    src/share/robots/HRCrobot/lerobot_robot_HRCrobot/controller.py

安全分级：

    1. 离线测试（始终运行）
       纯函数、单位换算、未连接时的守卫。
       绝不触碰硬件。

    2. 在线安全测试（需要 HRC_TEST_LIVE=1）
       连接真机、读取 TCP 位姿、断开。
       只读不写，不会驱动机器人。

    3. 运动测试（需要 HRC_TEST_LIVE=1 且 HRC_TEST_MOVE=1）
       流式下发一串微小 setpoint。
       默认跳过。上使能后由人工设置
       HRC_TEST_MOVE=1 再运行。

运行方式：

    # 只跑离线测试
    python -m pytest "tests/ HRCrobotTest/test_hrcrobot_controller.py" -v

    # 在线安全测试（连接 + 读位姿）
    HRC_TEST_LIVE=1 python -m pytest "tests/ HRCrobotTest/test_hrcrobot_controller.py" -v

    # 上使能后的运动测试
    HRC_TEST_LIVE=1 HRC_TEST_MOVE=1 \\
        python -m pytest "tests/ HRCrobotTest/test_hrcrobot_controller.py" -v

真机 IP 可用环境变量覆盖：

    HRC_ROBOT_IP=10.10.59.211
"""

import math
import os
import sys
import time
from pathlib import Path

import pytest
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------
# 让测试在没有 editable install 的环境下也能 import share
# ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from share.robots.HRCrobot.lerobot_robot_HRCrobot.controller import (  # noqa: E402
    DEFAULT_HRC_PORT,
    GRIPPER_IO_CLOSE_PORT,
    GRIPPER_IO_OPEN_PORT,
    HRCrobotController,
    VENDOR_SDK_DIR,
    _rotvec_pose_to_sdk_pose,
    _sdk_pose_to_rotvec_pose,
)

# ------------------------------------------------------------
# 环境开关
# ------------------------------------------------------------

LIVE = os.environ.get("HRC_TEST_LIVE") == "1"
MOVE = LIVE and os.environ.get("HRC_TEST_MOVE") == "1"

ROBOT_IP = os.environ.get("HRC_ROBOT_IP", "10.10.59.211")

# 运动测试参数：1 mm 的 Z 向小步，10 mm/min 量级
MOVE_Z_STEP_M = 0.001
MOVE_STEPS = 10
MOVE_STEP_PERIOD_S = 0.2


skip_if_not_live = pytest.mark.skipif(
    not LIVE,
    reason="在线测试需要 HRC_TEST_LIVE=1",
)

skip_if_not_move = pytest.mark.skipif(
    not MOVE,
    reason="运动测试需要 HRC_TEST_LIVE=1 且 HRC_TEST_MOVE=1",
)


# ============================================================
# 1. 离线：vendor SDK 布局
# ============================================================


def test_vendor_sdk_layout_exists():
    """
    vendor 目录应包含 controller.py 依赖的关键文件。
    """

    assert VENDOR_SDK_DIR.is_dir()
    assert (VENDOR_SDK_DIR / "hsrosi").is_dir()
    assert (
        VENDOR_SDK_DIR / "hsrosi" / "controller.py"
    ).is_file()
    assert (
        VENDOR_SDK_DIR / "libpyhstrajproxy.so"
    ).is_file()


# ============================================================
# 2. 离线：单位换算纯函数
# ============================================================


class TestPoseConversions:
    """
    _sdk_pose_to_rotvec_pose / _rotvec_pose_to_sdk_pose
    """

    def test_sdk_to_rotvec_converts_units(self):
        """
        mm -> m，euler degree -> rotation vector rad。
        """

        sdk_pose = [
            100.0,
            200.0,
            300.0,
            0.0,
            0.0,
            90.0,
        ]

        pose = _sdk_pose_to_rotvec_pose(sdk_pose)

        assert len(pose) == 6
        assert pose[0] == pytest.approx(0.1)
        assert pose[1] == pytest.approx(0.2)
        assert pose[2] == pytest.approx(0.3)

        # 纯 Z 90 度旋转的 rotvec
        assert pose[3] == pytest.approx(0.0, abs=1e-9)
        assert pose[4] == pytest.approx(0.0, abs=1e-9)
        assert pose[5] == pytest.approx(math.pi / 2)

    def test_rotvec_to_sdk_converts_units(self):
        """
        m -> mm，rotation vector rad -> euler degree。
        """

        pose = [
            0.1,
            0.2,
            0.3,
            0.0,
            0.0,
            math.pi / 2,
        ]

        sdk_pose = _rotvec_pose_to_sdk_pose(pose)

        assert len(sdk_pose) == 6
        assert sdk_pose[0] == pytest.approx(100.0)
        assert sdk_pose[1] == pytest.approx(200.0)
        assert sdk_pose[2] == pytest.approx(300.0)
        assert sdk_pose[3] == pytest.approx(0.0, abs=1e-6)
        assert sdk_pose[4] == pytest.approx(0.0, abs=1e-6)
        assert sdk_pose[5] == pytest.approx(90.0)

    def test_round_trip_identity(self):
        """
        任意姿态走 sdk -> rotvec -> sdk 应还原。

        注意 euler 会有多解（例如超过 90 度的分支），
        所以比较的是最终旋转矩阵，而不是原始角度。
        """

        from scipy.spatial.transform import Rotation

        sdk_pose = [
            350.0,
            -20.0,
            410.0,
            15.0,
            -30.0,
            75.0,
        ]

        rotvec = _sdk_pose_to_rotvec_pose(sdk_pose)
        back = _rotvec_pose_to_sdk_pose(rotvec)

        assert back[0] == pytest.approx(sdk_pose[0])
        assert back[1] == pytest.approx(sdk_pose[1])
        assert back[2] == pytest.approx(sdk_pose[2])

        rot_original = Rotation.from_euler(
            "xyz",
            [math.radians(v) for v in sdk_pose[3:6]],
        )
        rot_back = Rotation.from_euler(
            "xyz",
            [math.radians(v) for v in back[3:6]],
        )

        magnitude = (
            rot_original.inv() * rot_back
        ).magnitude()

        assert magnitude == pytest.approx(0.0, abs=1e-9)

    def test_identity_pose_round_trip(self):
        """
        零位姿往返恒等。
        """

        sdk_pose = [0.0] * 6

        back = _rotvec_pose_to_sdk_pose(
            _sdk_pose_to_rotvec_pose(sdk_pose)
        )

        for value, expected in zip(back, sdk_pose):
            assert value == pytest.approx(expected)

    def test_wrong_length_raises(self):
        """
        长度不是 6 时应报 ValueError。
        """

        with pytest.raises(ValueError):
            _sdk_pose_to_rotvec_pose([0.0] * 5)

        with pytest.raises(ValueError):
            _rotvec_pose_to_sdk_pose([0.0] * 7)


# ============================================================
# 3. 离线：构造与未连接守卫
# ============================================================


class TestControllerOffline:
    """
    不连接硬件就能验证的行为。
    """

    def make_controller(self) -> HRCrobotController:
        return HRCrobotController(
            robot_ip=ROBOT_IP,
            frequency=100.0,
            gripper_threshold=0.5,
        )

    def test_defaults(self):
        controller = self.make_controller()

        assert controller.robot_ip == ROBOT_IP
        assert controller.hrc_port == DEFAULT_HRC_PORT
        assert controller.is_connected is False
        assert controller.period == pytest.approx(0.01)
        assert controller.get_gripper_position() == 0.0

    def test_not_connected_guards(self):
        """
        未连接时所有硬件接口都必须直接报错，
        绝不能悄悄执行。
        """

        controller = self.make_controller()

        with pytest.raises(RuntimeError):
            controller.get_tcp_pose()

        with pytest.raises(RuntimeError):
            controller.servo_cartesian(
                [0.0] * 6
            )

        with pytest.raises(RuntimeError):
            controller.set_gripper(1.0)

    def test_disconnect_when_not_connected_is_noop(self):
        controller = self.make_controller()

        # 不应抛异常
        controller.disconnect()

        assert controller.is_connected is False

    def test_wrong_pose_length_rejected(self):
        controller = self.make_controller()
        controller._connected = True  # 绕过连接守卫
        controller.client = object()

        with pytest.raises(ValueError):
            controller.servo_cartesian(
                [0.0, 0.0, 0.0, 0.0, 0.0]
            )


# ============================================================
# 4. 在线安全：连接 + 读取（不运动）
# ============================================================


@pytest.fixture(scope="module")
def live_controller():
    """
    连接真机（只读用途）。

    注意：连接时 SDK 会以当前位姿 start cartesian 模式，
    这是模式声明，不会产生运动。
    """

    if not LIVE:
        pytest.skip("需要 HRC_TEST_LIVE=1")

    controller = HRCrobotController(
        robot_ip=ROBOT_IP,
        frequency=100.0,
        gripper_threshold=0.5,
    )

    controller.connect()
    assert controller.is_connected is True

    yield controller

    controller.disconnect()
    assert controller.is_connected is False


@skip_if_not_live
class TestLiveSafe:
    """
    连接真机但绝不运动。
    """

    def test_connect_and_disconnect(self, live_controller):
        assert live_controller.client is not None

    def test_get_tcp_pose_shape_and_units(self, live_controller):
        """
        读位姿：长度 6，单位应是 m（不是 mm）。

        真机 TCP 通常在基座上方几十厘米内，
        如果这里出现 100 ~ 1000 量级，
        说明单位换算漏了。
        """

        pose = live_controller.get_tcp_pose()

        assert len(pose) == 6
        assert all(
            isinstance(v, float) for v in pose
        )

        xyz = pose[:3]
        rotvec = pose[3:]

        for value in xyz:
            assert -3.0 < value < 3.0

        rotvec_norm = math.sqrt(
            sum(v * v for v in rotvec)
        )
        assert rotvec_norm <= math.pi + 1e-9

    def test_get_tcp_pose_is_stable(self, live_controller):
        """
        静止时连续读两次位姿应基本一致。
        """

        pose_a = live_controller.get_tcp_pose()
        time.sleep(0.05)
        pose_b = live_controller.get_tcp_pose()

        for a, b in zip(pose_a, pose_b):
            assert b == pytest.approx(a, abs=5e-3)

    def test_get_tcp_pose_multiple_reads(self, live_controller):
        """
        连续读 10 次不报错、长度稳定。
        """

        for _ in range(10):
            pose = live_controller.get_tcp_pose()
            assert len(pose) == 6


# ============================================================
# 5. 在线运动：上使能后人工开启
# ============================================================


@skip_if_not_move
class TestLiveMotion:
    """
    HRC_TEST_LIVE=1 且 HRC_TEST_MOVE=1 时才会运行。

    内容：沿 Z 轴下发一串微小 setpoint。

    运行前确认：
        - 已上使能
        - Z 正方向行程安全
        - 机器人周围无人
    """

    def test_servo_cartesian_small_z_move(
        self,
        live_controller,
    ):
        start = live_controller.get_tcp_pose()
        print(
            "\n起始位姿 (m + rotvec rad):",
            [round(v, 4) for v in start],
        )

        target = list(start)
        target[2] += MOVE_Z_STEP_M

        for _ in range(MOVE_STEPS):
            live_controller.servo_cartesian(target)
            time.sleep(MOVE_STEP_PERIOD_S)

        end = live_controller.get_tcp_pose()
        print(
            "结束位姿 (m + rotvec rad):",
            [round(v, 4) for v in end],
        )

        # 实际移动量应接近 1 mm，且不能反向大幅运动
        delta_z = end[2] - start[2]
        assert 0.0 < delta_z <= MOVE_Z_STEP_M * 3.0

    def test_servo_period_limiting(self, live_controller):
        """
        servo_cartesian 的周期限频：
        连续下发 N 个 setpoint，总耗时应 >= (N-1) * period。
        """

        pose = live_controller.get_tcp_pose()

        n_calls = 20
        start_time = time.perf_counter()

        for _ in range(n_calls):
            live_controller.servo_cartesian(pose)

        elapsed = time.perf_counter() - start_time

        assert elapsed >= (n_calls - 1) * live_controller.period * 0.9

    def test_servo_cartesian_rotation_rz(self, live_controller):
        """
        主动旋转测试：RZ 转 +5 度，再转回来。

        这个测试能暴露旋转表示假设的错误：
        如果 SDK 的 rx/ry/rz 不是我们假设的
        extrinsic XYZ euler (degree)，转 5 度时
        其他轴会跟着动，或实际转角对不上。

        结束时姿态应与起始一致（转回去了）。
        """

        start = live_controller.get_tcp_pose()
        print(
            "\n起始姿态 (rotvec rad):",
            [round(v, 4) for v in start[3:]],
        )

        # 目标：当前位姿绕 Z 转 5 度
        target = list(start)
        rz_delta = math.radians(5.0)
        rot = Rotation.from_rotvec(start[3:6])
        rotated = rot * Rotation.from_rotvec(
            [0.0, 0.0, rz_delta]
        )
        target[3:6] = rotated.as_rotvec().tolist()

        # 分 10 步慢慢转过去，每步 0.5 度
        for fraction in [i / 10.0 for i in range(1, 11)]:
            step = list(start)
            step_rot = Rotation.from_rotvec(start[3:6])
            step_rot = step_rot * Rotation.from_rotvec(
                [0.0, 0.0, rz_delta * fraction]
            )
            step[3:6] = step_rot.as_rotvec().tolist()

            live_controller.servo_cartesian(step)
            time.sleep(0.15)

        time.sleep(0.5)

        mid = live_controller.get_tcp_pose()
        print(
            "转动后姿态 (rotvec rad):",
            [round(v, 4) for v in mid[3:]],
        )

        # 实际转动量应接近 5 度（0.0873 rad）
        rot_start = Rotation.from_rotvec(start[3:6])
        rot_mid = Rotation.from_rotvec(mid[3:6])
        actual_angle = (rot_start.inv() * rot_mid).magnitude()

        print(f"实际转角: {math.degrees(actual_angle):.2f} 度")

        assert actual_angle == pytest.approx(
            rz_delta, abs=math.radians(1.0)
        )

        # 平动不应被旋转耦合带动超过 2mm
        for axis in range(3):
            assert (
                abs(mid[axis] - start[axis]) <= 0.002
            ), f"轴 {axis} 被旋转耦合带动: {mid[axis] - start[axis]}"

        # 转回原姿态
        for fraction in [i / 10.0 for i in range(1, 11)]:
            step = list(mid)
            step_rot = Rotation.from_rotvec(mid[3:6])
            step_rot = step_rot * Rotation.from_rotvec(
                [0.0, 0.0, -rz_delta * fraction]
            )
            step[3:6] = step_rot.as_rotvec().tolist()

            live_controller.servo_cartesian(step)
            time.sleep(0.15)

        time.sleep(0.5)

        end = live_controller.get_tcp_pose()
        rot_end = Rotation.from_rotvec(end[3:6])
        return_angle = (rot_start.inv() * rot_end).magnitude()

        print(f"转回后残余角: {math.degrees(return_angle):.3f} 度")

        # 转回后姿态应与起始一致（0.5 度容差）
        assert return_angle <= math.radians(0.5)


# ============================================================
# 6. 夹爪（离线状态检查 + 在线 IO）
# ============================================================


def test_gripper_io_port_constants():
    """
    IO 口定义必须与 vendor example 一致：
    DO25=open，DO26=close。
    """

    assert GRIPPER_IO_OPEN_PORT == 25
    assert GRIPPER_IO_CLOSE_PORT == 26


@skip_if_not_live
class TestLiveGripper:

    def test_set_gripper_updates_state(self, live_controller):
        """
        夹爪命令状态会更新（HSC3 链路不可用也应通过）。
        """

        live_controller.set_gripper(1.0)
        assert (
            live_controller.get_gripper_position()
            == 1.0
        )

        live_controller.set_gripper(0.0)
        assert (
            live_controller.get_gripper_position()
            == 0.0
        )

        live_controller.set_gripper(0.3)
        assert (
            live_controller.get_gripper_position()
            == 0.0
        )
