#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HRC 外部控制演示：夹爪开合、六维力读取和 Z 轴小步运动。

依赖：
  - 当前目录中的 ``hsrosi`` 和 ``libpyhstrajproxy.so``
  - HSC3 的 ``libhsc3.so``（可通过 HSC3_LIBRARY 环境变量指定；也会自动
    查找本机 LeRobot 的 ``hs_robot/hsc3py`` 目录）

执行前请确认机械臂周围无人、Z 轴正方向安全，并按实际控制器修改 IP。
"""

import ctypes
import logging
import os
import time
from ctypes import POINTER, byref, c_bool, c_char_p, c_double, c_int32, c_uint16, c_void_p
from pathlib import Path

from hsrosi import HRCController, HRCError


ROBOT_IP = "10.10.59.211"
HRC_PORT = 9095
HSC3_PORT = 23234

# DO26=1、DO25=0：夹爪闭合；DO25=1、DO26=0：夹爪松开。
IO_OPEN_PORT = 25
IO_CLOSE_PORT = 26
FORCE_R_INDEXES = (20, 21, 22, 23, 24, 25)

Z_MOVE_MM = 10.0
Z_STEP_MM = 0.2
STEP_INTERVAL_S = 0.05
GRIPPER_SWITCH_DELAY_S = 0.5


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


class HSC3Robot:
    """仅封装本 demo 所需的 HSC3：R 寄存器和 IO。"""

    def __init__(self, ip: str, port: int, library_path: str | None = None):
        selected_library = self._find_library(library_path)
        if selected_library is None:
            raise FileNotFoundError(
                "未找到 libhsc3.so。请设置 HSC3_LIBRARY，或将库放到本目录/"
                "LeRobot 的 src/lerobot/robots/hs_robot/hsc3py 目录。"
            )

        self._library = self._load_library(selected_library)
        self._configure_signatures()
        self._handle = self._library.Robot_create(ip.encode("utf-8"), port)
        if not self._handle:
            raise RuntimeError("Robot_create failed")

    @staticmethod
    def _find_library(library_path: str | None) -> Path | None:
        """定位 HSC3 库；环境变量优先，随后查找项目和 LeRobot 安装位置。"""
        if library_path:
            candidates = [Path(library_path)]
        elif os.environ.get("HSC3_LIBRARY"):
            candidates = [Path(os.environ["HSC3_LIBRARY"])]
        else:
            home = Path.home()
            candidates = [
                Path(__file__).with_name("libhsc3.so"),
                home / "Project/lerobot/src/lerobot/robots/hs_robot/hsc3py/libhsc3.so",
                home / "Project/lerobot/tests/assembly_monitor/data_collect_module/hs_robot/hsc3py/libhsc3.so",
                home / "Project/XMH/3D_mouse_touch/real_world_control/libhsc3.so",
            ]

        for candidate in candidates:
            if candidate.is_file():
                logging.info("使用 HSC3 库：%s", candidate)
                return candidate.resolve()
        return None

    @staticmethod
    def _load_library(library_path: Path):
        """加载 HSC3 库及其相对路径的 SDK 依赖。

        HSC3 库的 RUNPATH 使用相对路径 ``./lib``，从本项目目录运行时不会
        自动命中 LeRobot 的依赖目录。加载期间短暂切换到库目录，使动态链接器
        正确解析其中的 ``lib/``；加载完成后立即恢复原工作目录。
        """
        original_cwd = os.getcwd()
        try:
            os.chdir(library_path.parent)
            return ctypes.CDLL(str(library_path))
        finally:
            os.chdir(original_cwd)

    def _configure_signatures(self):
        self._library.Robot_create.argtypes = [c_char_p, c_uint16]
        self._library.Robot_create.restype = c_void_p
        self._library.Robot_destroy.argtypes = [c_void_p]
        self._library.Robot_connect.argtypes = [c_void_p]
        self._library.Robot_connect.restype = c_bool
        self._library.Robot_disconnect.argtypes = [c_void_p]
        self._library.Robot_getR.argtypes = [c_void_p, c_int32, POINTER(c_double)]
        self._library.Robot_getR.restype = c_int32
        self._library.Robot_setR.argtypes = [c_void_p, c_int32, c_double]
        self._library.Robot_setR.restype = c_int32
        self._library.Robot_setIO.argtypes = [c_void_p, c_int32, c_bool]
        self._library.Robot_setIO.restype = c_int32
        self._library.Robot_getIO.argtypes = [c_void_p, c_int32, POINTER(c_bool)]
        self._library.Robot_getIO.restype = c_int32

    def connect(self):
        if not self._library.Robot_connect(self._handle):
            raise ConnectionError("HSC3 连接失败")

    def get_r(self, index: int) -> float:
        value = c_double()
        if self._library.Robot_getR(self._handle, index, byref(value)) != 0:
            raise RuntimeError(f"读取 R{index} 失败")
        return value.value

    def set_r(self, index: int, value: float):
        if self._library.Robot_setR(self._handle, index, value) != 0:
            raise RuntimeError(f"写入 R{index} 失败")

    def set_io(self, index: int, value: bool):
        if self._library.Robot_setIO(self._handle, index, value) != 0:
            raise RuntimeError(f"设置 IO{index} 失败")

    def get_io(self, index: int) -> bool:
        value = c_bool()
        if self._library.Robot_getIO(self._handle, index, byref(value)) != 0:
            raise RuntimeError(f"读取 IO{index} 失败")
        return value.value

    def close(self):
        if getattr(self, "_handle", None):
            self._library.Robot_disconnect(self._handle)
            self._library.Robot_destroy(self._handle)
            self._handle = None


def read_force_torque(robot: HSC3Robot) -> list[float]:
    """读取 [Fx, Fy, Fz, Tx, Ty, Tz]，对应 R20 到 R25。"""
    return [robot.get_r(index) for index in FORCE_R_INDEXES]


def set_gripper(robot: HSC3Robot, close: bool):
    """双 IO 互斥控制气动夹爪，并回读控制器中的 DO 状态。"""
    if close:
        robot.set_io(IO_OPEN_PORT, False)
        robot.set_io(IO_CLOSE_PORT, True)
        logging.info("夹爪：闭合")
    else:
        robot.set_io(IO_CLOSE_PORT, False)
        robot.set_io(IO_OPEN_PORT, True)
        logging.info("夹爪：打开")

    logging.info(
        "DO 回读：DO%d=%d，DO%d=%d",
        IO_OPEN_PORT,
        robot.get_io(IO_OPEN_PORT),
        IO_CLOSE_PORT,
        robot.get_io(IO_CLOSE_PORT),
    )


def move_z_small_distance(hrc: HRCController, distance_mm: float, step_mm: float):
    """从当前笛卡尔位姿沿 Z 轴匀速小步移动指定距离。"""
    if step_mm <= 0:
        raise ValueError("step_mm 必须大于 0")

    start_pose = list(hrc.get_cartesian_position())
    target_pose = start_pose.copy()
    direction = 1.0 if distance_mm >= 0 else -1.0
    remaining_distance = abs(distance_mm)

    logging.info("Z 轴移动：%.2f mm，起始 Z=%.3f", distance_mm, start_pose[2])
    while remaining_distance > 1e-9:
        current_step = min(step_mm, remaining_distance)
        target_pose[2] += direction * current_step
        if not hrc.move_to_cartesian_position(target_pose):
            raise RuntimeError(f"Z 轴移动失败，目标位姿：{target_pose}")
        remaining_distance -= current_step
        time.sleep(STEP_INTERVAL_S)

    logging.info("Z 轴移动完成，目标 Z=%.3f", target_pose[2])


def external_io_force_demo():
    """按 example.py 的连接主干执行完整 demo。"""
    print("=== 外部 IO、六维力与 Z 轴运动 Demo ===")
    hsc3_robot = None

    try:
        with HRCController() as hrc:
            hrc.init()
            if not hrc.connect(ROBOT_IP, HRC_PORT, motion_mode="cartesian"):
                raise ConnectionError(f"HRC 连接失败：{ROBOT_IP}:{HRC_PORT}")
            print(f"HRC 已连接：{ROBOT_IP}:{HRC_PORT}")

            hsc3_robot = HSC3Robot(ROBOT_IP, HSC3_PORT)
            hsc3_robot.connect()
            print(f"HSC3 已连接：{ROBOT_IP}:{HSC3_PORT}")

            print("初始六维力 [Fx, Fy, Fz, Tx, Ty, Tz]：", read_force_torque(hsc3_robot))
            set_gripper(hsc3_robot, close=False)
            time.sleep(GRIPPER_SWITCH_DELAY_S)
            set_gripper(hsc3_robot, close=True)
            time.sleep(GRIPPER_SWITCH_DELAY_S)

            move_z_small_distance(hrc, Z_MOVE_MM, Z_STEP_MM)
            print("移动后的六维力 [Fx, Fy, Fz, Tx, Ty, Tz]：", read_force_torque(hsc3_robot))

    except HRCError as error:
        logging.error("HRC 错误：%s", error)
    except Exception as error:
        logging.error("Demo 失败：%s", error)
    finally:
        if hsc3_robot is not None:
            hsc3_robot.close()


def main():
    setup_logging()
    print("HRC 外部控制示例")
    print("=" * 40)
    external_io_force_demo()
    print("\\n示例演示完成！")


if __name__ == "__main__":
    main()
