#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HRCrobot 夹爪专项测试。

夹爪链路说明：
    机器人本体位姿走 hsrosi (9095)，
    夹爪是独立链路，通过 HSC3 (23234) 的 IO 控制：

        DO25=1, DO26=0  -> 打开
        DO25=0, DO26=1  -> 闭合

安全分级：

    1. 离线测试（始终运行）
       库查找、未连接守卫、状态机。
       不触碰硬件。

    2. IO 动作测试（需要 HRC_TEST_LIVE=1 且 HRC_TEST_GRIPPER=1）
       连接 HSC3，下发 IO 命令驱动夹爪开/合。
       不动机械臂。

    注意（真机实测发现）：
        set_io() 下发成功，但 get_io() 读回恒为 False
        （已扫描 0-63 号口）。当前 SDK 的 IO 读回
        不可用于验证，因此在线测试以“下发成功 +
        夹爪物理动作观察”为准，读回差异只告警。

运行方式：

    # 只跑离线测试
    python -m pytest "tests/ HRCrobotTest/test_hrcrobot_gripper.py" -v

    # 夹爪动作测试（会真实驱动夹爪，观察物理动作）
    HRC_TEST_LIVE=1 HRC_TEST_GRIPPER=1 \\
        python -m pytest "tests/ HRCrobotTest/test_hrcrobot_gripper.py" -v -s
"""

import logging
import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from share.robots.HRCrobot.lerobot_robot_HRCrobot.controller import (  # noqa: E402
    GRIPPER_IO_CLOSE_PORT,
    GRIPPER_IO_OPEN_PORT,
    HRCrobotController,
    _HSC3IOClient,
)

logger = logging.getLogger(__name__)

LIVE = os.environ.get("HRC_TEST_LIVE") == "1"
GRIPPER = LIVE and os.environ.get("HRC_TEST_GRIPPER") == "1"

ROBOT_IP = os.environ.get("HRC_ROBOT_IP", "10.10.59.211")
HSC3_PORT = int(os.environ.get("HRC_HSC3_PORT", "23234"))

GRIPPER_SWITCH_DELAY_S = float(
    os.environ.get("HRC_GRIPPER_DELAY", "0.5")
)


skip_if_not_gripper = pytest.mark.skipif(
    not GRIPPER,
    reason=(
        "夹爪动作测试需要 HRC_TEST_LIVE=1 且 "
        "HRC_TEST_GRIPPER=1（会真实驱动夹爪）"
    ),
)


# ============================================================
# 1. 离线：HSC3 库查找
# ============================================================


def test_hsc3_library_findable():
    """
    libhsc3.so 必须能被自动定位。
    """

    found = _HSC3IOClient._find_library()

    assert found is not None
    assert found.is_file()


def test_hsc3_library_env_override(tmp_path, monkeypatch):
    """
    HSC3_LIBRARY 环境变量优先。
    """

    fake_lib = tmp_path / "libhsc3.so"
    fake_lib.write_bytes(b"")

    monkeypatch.setenv("HSC3_LIBRARY", str(fake_lib))

    found = _HSC3IOClient._find_library()

    assert found == fake_lib.resolve()


# ============================================================
# 2. 离线：夹爪状态机（无 HSC3 链路）
# ============================================================


class TestGripperStateMachineOffline:
    """
    HSC3 不可用时（_hsc3=None），夹爪命令
    仍应正常更新状态，绝不抛异常。
    """

    def make_controller(self) -> HRCrobotController:
        controller = HRCrobotController(
            robot_ip=ROBOT_IP,
            frequency=100.0,
        )
        controller._connected = True
        controller._hsc3 = None
        return controller

    def test_close_command_state(self):
        controller = self.make_controller()

        controller.set_gripper(1.0)

        assert (
            controller.get_gripper_position() == 1.0
        )

    def test_open_command_state(self):
        controller = self.make_controller()

        controller.set_gripper(1.0)
        controller.set_gripper(0.0)

        assert (
            controller.get_gripper_position() == 0.0
        )

    def test_threshold_boundary(self):
        """
        阈值边界：0.5 本身算 close。
        """

        controller = self.make_controller()

        controller.set_gripper(0.5)
        assert (
            controller.get_gripper_position() == 1.0
        )

        controller.set_gripper(0.49)
        assert (
            controller.get_gripper_position() == 0.0
        )

    def test_custom_threshold(self):
        controller = HRCrobotController(
            robot_ip=ROBOT_IP,
            frequency=100.0,
            gripper_threshold=0.3,
        )
        controller._connected = True
        controller._hsc3 = None

        controller.set_gripper(0.3)
        assert (
            controller.get_gripper_position() == 1.0
        )

        controller.set_gripper(0.29)
        assert (
            controller.get_gripper_position() == 0.0
        )

    def test_not_connected_guard(self):
        controller = HRCrobotController(
            robot_ip=ROBOT_IP,
            frequency=100.0,
        )

        with pytest.raises(RuntimeError):
            controller.set_gripper(1.0)


# ============================================================
# 3. 在线：夹爪真实动作
# ============================================================


@pytest.fixture(scope="module")
def live_gripper_controller():
    """
    完整 controller（含 HSC3 夹爪链路）。
    """

    if not GRIPPER:
        pytest.skip(
            "需要 HRC_TEST_LIVE=1 且 HRC_TEST_GRIPPER=1"
        )

    controller = HRCrobotController(
        robot_ip=ROBOT_IP,
        frequency=100.0,
    )
    controller.connect()

    yield controller

    controller.disconnect()


@skip_if_not_gripper
class TestHSC3IO:
    """
    HSC3 IO 链路连通性。

    注意：当前固件 get_io() 读回恒为 False
    （扫描过 0-63 号口），所以这里只断言
    “下发不抛异常”，读回结果仅打印参考。
    """

    def test_set_io_commands_succeed(self, live_gripper_controller):
        hsc3 = live_gripper_controller._hsc3

        assert hsc3 is not None

        # 下发打开
        hsc3.set_io(GRIPPER_IO_OPEN_PORT, True)
        hsc3.set_io(GRIPPER_IO_CLOSE_PORT, False)

        # 读回（仅参考，不作为断言）
        open_rb = hsc3.get_io(GRIPPER_IO_OPEN_PORT)
        close_rb = hsc3.get_io(GRIPPER_IO_CLOSE_PORT)
        print(
            f"\n下发 open 后读回: DO25={open_rb}, DO26={close_rb} "
            "(读回恒 False 为当前固件已知现象)"
        )

        # 下发闭合
        hsc3.set_io(GRIPPER_IO_OPEN_PORT, False)
        hsc3.set_io(GRIPPER_IO_CLOSE_PORT, True)

        close_rb = hsc3.get_io(GRIPPER_IO_CLOSE_PORT)
        print(f"下发 close 后读回: DO26={close_rb}")

    def test_rapid_io_switching(self, live_gripper_controller):
        """
        快速连续切换不下发崩溃（通信鲁棒性）。
        """

        hsc3 = live_gripper_controller._hsc3

        for i in range(5):
            hsc3.set_io(
                GRIPPER_IO_OPEN_PORT,
                i % 2 == 0,
            )
            hsc3.set_io(
                GRIPPER_IO_CLOSE_PORT,
                i % 2 == 1,
            )

        # 结束恢复打开
        hsc3.set_io(GRIPPER_IO_OPEN_PORT, True)
        hsc3.set_io(GRIPPER_IO_CLOSE_PORT, False)


@skip_if_not_gripper
class TestGripperFullCycle:
    """
    通过 controller.set_gripper() 走完整链路。

    运行时必须人工观察夹爪物理动作：
        close -> open -> close

    如果 set_io 全部成功但夹爪没动，
    问题在接线 / IO 口配置，不在代码。
    """

    def test_full_cycle(
        self,
        live_gripper_controller,
    ):
        controller = live_gripper_controller

        assert controller._hsc3 is not None

        # 1. close
        controller.set_gripper(1.0)
        time.sleep(GRIPPER_SWITCH_DELAY_S)
        assert (
            controller.get_gripper_position() == 1.0
        )
        print("\n[观察点] 夹爪应已闭合 (DO26=1)")

        # 2. open
        controller.set_gripper(0.0)
        time.sleep(GRIPPER_SWITCH_DELAY_S)
        assert (
            controller.get_gripper_position() == 0.0
        )
        print("[观察点] 夹爪应已打开 (DO25=1)")

        # 3. 阈值内的小数值同样视为 open
        controller.set_gripper(0.3)
        time.sleep(GRIPPER_SWITCH_DELAY_S)
        assert (
            controller.get_gripper_position() == 0.0
        )

        # 4. close，最终停在闭合
        controller.set_gripper(0.8)
        time.sleep(GRIPPER_SWITCH_DELAY_S)
        assert (
            controller.get_gripper_position() == 1.0
        )
        print("[观察点] 夹爪应再次闭合，测试结束")

    def test_state_persists_across_reconnect(
        self,
        live_gripper_controller,
    ):
        """
        断开重连后命令状态有意保留（重连不清零），
        这是 controller 的设计行为：_gripper_position
        表示“最后一次下发的命令”，与连接无关。
        """

        controller = live_gripper_controller

        controller.set_gripper(1.0)
        time.sleep(0.2)
        assert (
            controller.get_gripper_position() == 1.0
        )

        controller.disconnect()
        assert controller.is_connected is False
        assert controller._hsc3 is None

        controller.connect()
        assert controller.is_connected is True

        # 重连不重置命令状态
        assert (
            controller.get_gripper_position() == 1.0
        )

        # 恢复打开状态
        controller.set_gripper(0.0)
        time.sleep(GRIPPER_SWITCH_DELAY_S)
