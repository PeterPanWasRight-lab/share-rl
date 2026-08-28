import ctypes
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Vendor SDK 位置与默认参数
#
# 布局：
#
#   src/share/robots/HRCrobot/
#       lerobot_robot_HRCrobot/controller.py   <- 本文件
#       vendor/HRCrobotSDK/                    <- 厂家 SDK（hsrosi + .so）
# ------------------------------------------------------------

VENDOR_SDK_DIR = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "HRCrobotSDK"
)

# SDK 内部单位约定：
#   position: mm
#   rotation: degree (extrinsic XYZ euler)
# SHaRe adapter 内部约定：
#   position: m
#   rotation: rad (rotation vector)
_MM_TO_M = 0.001

# 与 vendor/HRCrobotSDK/example_externAix.py 保持一致。
DEFAULT_HRC_PORT = 9095
DEFAULT_HSC3_PORT = 23234
GRIPPER_IO_OPEN_PORT = 25
GRIPPER_IO_CLOSE_PORT = 26


def _sdk_pose_to_rotvec_pose(
    sdk_pose: list[float],
) -> list[float]:
    """
    SDK 笛卡尔位姿 (mm + euler degree)
    -> adapter 约定位姿 (m + rotation vector rad)。
    """

    if len(sdk_pose) != 6:
        raise ValueError(
            "SDK cartesian pose must contain 6 values."
        )

    xyz = [v * _MM_TO_M for v in sdk_pose[:3]]
    rpy = [
        math.radians(v)
        for v in sdk_pose[3:6]
    ]

    rotvec = Rotation.from_euler(
        "xyz",
        rpy,
        degrees=False,
    ).as_rotvec()

    return [
        xyz[0],
        xyz[1],
        xyz[2],
        float(rotvec[0]),
        float(rotvec[1]),
        float(rotvec[2]),
    ]


def _rotvec_pose_to_sdk_pose(
    pose: list[float],
) -> list[float]:
    """
    adapter 约定位姿 (m + rotation vector rad)
    -> SDK 笛卡尔位姿 (mm + euler degree)。

    注意：这里把 SDK 的 rx/ry/rz 解释为
    extrinsic XYZ euler (degree)。如果真机标定后发现
    旋转表示不同，只需要改这一个函数和
    _sdk_pose_to_rotvec_pose()。
    """

    if len(pose) != 6:
        raise ValueError(
            "Cartesian pose must contain 6 values."
        )

    xyz = [v / _MM_TO_M for v in pose[:3]]
    rpy_deg = [
        math.degrees(v)
        for v in Rotation.from_rotvec(
            pose[3:6]
        ).as_euler(
            "xyz",
            degrees=False,
        )
    ]

    return [
        xyz[0],
        xyz[1],
        xyz[2],
        rpy_deg[0],
        rpy_deg[1],
        rpy_deg[2],
    ]


class _HSC3IOClient:
    """
    libhsc3.so 的最小 ctypes 封装。

    只用于夹爪 IO 控制（DO25 / DO26），
    参考实现来自 vendor/HRCrobotSDK/example_externAix.py
    中的 HSC3Robot。
    """

    def __init__(
        self,
        ip: str,
        port: int,
    ):
        library_path = self._find_library()

        if library_path is None:
            raise FileNotFoundError(
                "libhsc3.so not found. Set HSC3_LIBRARY "
                f"or put it into {VENDOR_SDK_DIR}."
            )

        self._library = self._load_library(library_path)
        self._configure_signatures()

        self._handle = self._library.Robot_create(
            ip.encode("utf-8"),
            port,
        )

        if not self._handle:
            raise RuntimeError("HSC3 Robot_create failed.")

    @staticmethod
    def _find_library() -> Path | None:
        candidates = []

        if os.environ.get("HSC3_LIBRARY"):
            candidates.append(
                Path(os.environ["HSC3_LIBRARY"])
            )

        candidates.append(
            VENDOR_SDK_DIR / "libhsc3.so"
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        return None

    @staticmethod
    def _load_library(library_path: Path):
        # libhsc3.so 的 RUNPATH 是相对路径 ./lib，
        # 加载期间短暂切换到库目录，保证依赖可解析。
        original_cwd = os.getcwd()

        try:
            os.chdir(library_path.parent)
            return ctypes.CDLL(str(library_path))
        finally:
            os.chdir(original_cwd)

    def _configure_signatures(self):
        self._library.Robot_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint16,
        ]
        self._library.Robot_create.restype = ctypes.c_void_p

        self._library.Robot_destroy.argtypes = [ctypes.c_void_p]

        self._library.Robot_connect.argtypes = [ctypes.c_void_p]
        self._library.Robot_connect.restype = ctypes.c_bool

        self._library.Robot_disconnect.argtypes = [ctypes.c_void_p]

        self._library.Robot_setIO.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_bool,
        ]
        self._library.Robot_setIO.restype = ctypes.c_int32

        self._library.Robot_getIO.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_bool),
        ]
        self._library.Robot_getIO.restype = ctypes.c_int32

    def connect(self):
        if not self._library.Robot_connect(self._handle):
            raise ConnectionError("HSC3 connect failed.")

    def set_io(self, index: int, value: bool):
        if self._library.Robot_setIO(self._handle, index, value) != 0:
            raise RuntimeError(f"HSC3 set IO{index} failed.")

    def get_io(self, index: int) -> bool:
        value = ctypes.c_bool()

        if self._library.Robot_getIO(
            self._handle,
            index,
            ctypes.byref(value),
        ) != 0:
            raise RuntimeError(f"HSC3 get IO{index} failed.")

        return bool(value.value)

    def close(self):
        if getattr(self, "_handle", None):
            self._library.Robot_disconnect(self._handle)
            self._library.Robot_destroy(self._handle)
            self._handle = None


class HRCrobotController:
    """
    HRCrobot 厂家 SDK 的最薄封装层。

    这个类只负责：

        connect()
        disconnect()

        get_tcp_pose()

        servo_cartesian()

        set_gripper()
        get_gripper_position()

    它不需要知道 SHaRe、MP-Net、Primitive 是什么。

    Vendor SDK：
        vendor/HRCrobotSDK/hsrosi/HRCController

    SDK 没有 servoL 类接口，实际下发使用
    move_to_cartesian_position() 流式发送，
    由本类的周期限频保证下发频率。

    机器人本体位姿走 hsrosi；夹爪是独立链路，
    通过 HSC3 的 IO 口控制。HSC3 不可用时夹爪
    退化为仅记录最后命令状态。
    """

    def __init__(
        self,
        robot_ip: str,
        frequency: float,
        gripper_threshold: float = 0.5,
        use_gripper: bool = True,
        hrc_port: int = DEFAULT_HRC_PORT,
        hsc3_port: int = DEFAULT_HSC3_PORT,
    ):
        self.robot_ip = robot_ip

        self.hrc_port = int(hrc_port)
        self.hsc3_port = int(hsc3_port)

        self.use_gripper = bool(use_gripper)

        self.frequency = float(frequency)
        self.period = 1.0 / self.frequency

        self.gripper_threshold = float(gripper_threshold)

        self.client: Any = None
        self._hsc3: Any = None
        self._connected = False

        # servo 周期控制
        self._next_servo_time = None

        # 如果当前夹爪没有位置反馈，
        # 暂时记录最后一次下发状态
        self._gripper_position = 0.0

    # ============================================================
    # Connection
    # ============================================================

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """
        建立与 HRCrobot 的连接（笛卡尔运动模式）。
        """

        if self._connected:
            return

        # vendor SDK 不在 site-packages 里，
        # 动态加入搜索路径后按需 import。
        if str(VENDOR_SDK_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_SDK_DIR))

        try:
            from hsrosi import HRCController  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"Cannot import vendor SDK from {VENDOR_SDK_DIR}. "
                "Check the HRCrobotSDK layout."
            ) from exc

        # pyhstrajproxy 可能依赖同目录 ./lib 下的库，
        # 加载期间短暂切换工作目录。
        original_cwd = os.getcwd()

        try:
            os.chdir(VENDOR_SDK_DIR)

            self.client = HRCController(
                lib_path=str(
                    VENDOR_SDK_DIR / "libpyhstrajproxy.so"
                )
            )
        finally:
            os.chdir(original_cwd)

        self.client.init()

        if not self.client.connect(
            self.robot_ip,
            self.hrc_port,
            motion_mode="cartesian",
        ):
            raise ConnectionError(
                f"Failed to connect HRCrobot at "
                f"{self.robot_ip}:{self.hrc_port}."
            )

        # --------------------------------------------------------
        # 夹爪走独立的 HSC3 IO 链路。
        # 不可用时只记录命令状态，不影响本体运动。
        # --------------------------------------------------------

        self._hsc3 = None

        if self.use_gripper:
            try:
                self._hsc3 = _HSC3IOClient(
                    self.robot_ip,
                    self.hsc3_port,
                )
                self._hsc3.connect()
            except Exception as exc:
                logger.warning(
                    "HSC3 gripper link unavailable "
                    "(gripper commands will only be recorded): %s",
                    exc,
                )
                self._hsc3 = None

        self._connected = True
        self._next_servo_time = time.perf_counter()

    def disconnect(self) -> None:
        """
        断开机器人连接。
        """

        if not self._connected:
            return

        if self._hsc3 is not None:
            try:
                self._hsc3.close()
            except Exception as exc:
                logger.warning(
                    "Error while closing HSC3 link: %s",
                    exc,
                )
            self._hsc3 = None

        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception as exc:
                logger.warning(
                    "Error while disconnecting HRCrobot: %s",
                    exc,
                )
            self.client = None

        self._connected = False
        self._next_servo_time = None

    # ============================================================
    # Robot state
    # ============================================================

    def get_tcp_pose(self) -> list[float]:
        """
        获取当前 TCP 位姿。

        返回 adapter 统一约定的“厂家格式”：

            [
                x,
                y,
                z,
                rotvec_x,
                rotvec_y,
                rotvec_z,
            ]

        xyz:
            meter

        rotation vector:
            rad

        SDK 返回 mm + euler degree，
        单位与旋转表示在这里统一转换。
        """

        if not self._connected:
            raise RuntimeError(
                "HRCrobot is not connected."
            )

        sdk_pose = self.client.get_cartesian_position()

        return _sdk_pose_to_rotvec_pose(
            list(sdk_pose)
        )

    # ============================================================
    # Cartesian servo
    # ============================================================

    def servo_cartesian(
        self,
        pose: Sequence[float],
    ) -> None:
        """
        下发一个 Cartesian servo setpoint。

        pose 是 adapter 统一约定的“厂家格式”：

            [
                x,
                y,
                z,
                rotvec_x,
                rotvec_y,
                rotvec_z,
            ]

        (meter + rad，与 get_tcp_pose() 对称)

        SDK 没有真正的 servoL，这里用
        move_to_cartesian_position() 流式下发，
        并按 self.period 做周期限频。

        这个函数会被 OpenLoopTrajectoryPrimitive 高频调用。
        """

        if not self._connected:
            raise RuntimeError(
                "HRCrobot is not connected."
            )

        pose = list(pose)

        if len(pose) != 6:
            raise ValueError(
                "Cartesian pose must contain 6 values."
            )

        # ========================================================
        # servo 周期限频
        # ========================================================

        now = time.perf_counter()

        if self._next_servo_time is None:
            self._next_servo_time = now

        if now < self._next_servo_time:
            time.sleep(
                self._next_servo_time - now
            )

        # ========================================================
        # 流式下发笛卡尔 setpoint
        # ========================================================

        sdk_pose = _rotvec_pose_to_sdk_pose(pose)

        if not self.client.move_to_cartesian_position(
            sdk_pose
        ):
            raise RuntimeError(
                f"HRCrobot cartesian move failed: {sdk_pose}"
            )

        self._next_servo_time += self.period

        now = time.perf_counter()

        # 如果 Python 循环已经严重掉周期，
        # 不追赶历史 setpoint。
        if (
            now - self._next_servo_time
            > 2.0 * self.period
        ):
            self._next_servo_time = now

    # ============================================================
    # Gripper
    # ============================================================

    def set_gripper(
        self,
        position: float,
    ) -> None:
        """
        第一阶段统一约定：

            position < 0.5
                -> open

            position >= 0.5
                -> close

        通过 HSC3 IO 控制（DO25=open，DO26=close，
        双口互斥，参考 vendor example）。HSC3 不可用时
        仅记录命令状态。
        """

        if not self._connected:
            raise RuntimeError(
                "HRCrobot is not connected."
            )

        position = float(position)

        if position >= self.gripper_threshold:

            if self._hsc3 is not None:
                self._hsc3.set_io(
                    GRIPPER_IO_OPEN_PORT,
                    False,
                )
                self._hsc3.set_io(
                    GRIPPER_IO_CLOSE_PORT,
                    True,
                )

            self._gripper_position = 1.0

        else:

            if self._hsc3 is not None:
                self._hsc3.set_io(
                    GRIPPER_IO_CLOSE_PORT,
                    False,
                )
                self._hsc3.set_io(
                    GRIPPER_IO_OPEN_PORT,
                    True,
                )

            self._gripper_position = 0.0

    def get_gripper_position(self) -> float:
        """
        如果夹爪有真实反馈，以后在这里读取。

        第一阶段先返回最后一次命令状态。
        """

        return float(
            self._gripper_position
        )
