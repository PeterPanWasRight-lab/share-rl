import time
from typing import Sequence


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
    """

    def __init__(
        self,
        robot_ip: str,
        frequency: float,
        gripper_threshold: float = 0.5,
    ):
        self.robot_ip = robot_ip

        self.frequency = float(frequency)
        self.period = 1.0 / self.frequency

        self.gripper_threshold = float(gripper_threshold)

        self.client = None
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
        建立与 HRCrobot 的连接。
        """

        if self._connected:
            return

        # ========================================================
        # TODO 1
        #
        # 在这里替换成 HRCrobot 厂家的真实 SDK。
        #
        # 例如：
        #
        # from hrc_sdk import RobotClient
        #
        # self.client = RobotClient(self.robot_ip)
        # self.client.connect()
        #
        # ========================================================

        raise NotImplementedError(
            "Please implement HRCrobotController.connect() "
            "using the HRCrobot vendor SDK."
        )

        # SDK 连通以后，保留：
        #
        # self._connected = True
        # self._next_servo_time = time.perf_counter()

    def disconnect(self) -> None:
        """
        断开机器人连接。
        """

        if not self._connected:
            return

        # ========================================================
        # TODO 2
        #
        # 换成厂家 SDK。
        #
        # 例如：
        #
        # self.client.disconnect()
        #
        # ========================================================

        self.client = None
        self._connected = False
        self._next_servo_time = None

    # ============================================================
    # Robot state
    # ============================================================

    def get_tcp_pose(self) -> list[float]:
        """
        获取当前 TCP 位姿。

        这一层返回“厂家格式”。

        下面暂时假设 HRCrobot 和 UR servoL 类似：

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

        如果 HRCrobot 厂家返回的是：
            quaternion
            RPY
            rotation matrix

        都没有关系。

        只需要后面在 HRCrobot.py 里统一转换成 SHaRe 的 xyz+rpy。
        """

        if not self._connected:
            raise RuntimeError(
                "HRCrobot is not connected."
            )

        # ========================================================
        # TODO 3
        #
        # 替换成 HRCrobot SDK。
        #
        # 例如：
        #
        # pose = self.client.get_tcp_pose()
        # return list(pose)
        #
        # ========================================================

        raise NotImplementedError(
            "Please implement get_tcp_pose() "
            "using the HRCrobot vendor SDK."
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

        pose 是“厂家接口需要的格式”。

        当前暂定：

            [
                x,
                y,
                z,
                rotvec_x,
                rotvec_y,
                rotvec_z,
            ]

        这个函数会被 OpenLoopTrajectoryPrimitive 高频调用。
        """

        if not self._connected:
            raise RuntimeError(
                "HRCrobot is not connected."
            )

        if len(pose) != 6:
            raise ValueError(
                "Cartesian pose must contain 6 values."
            )

        # ========================================================
        # 简单 servo 周期限频
        # ========================================================

        now = time.perf_counter()

        if self._next_servo_time is None:
            self._next_servo_time = now

        if now < self._next_servo_time:
            time.sleep(
                self._next_servo_time - now
            )

        # ========================================================
        # TODO 4
        #
        # 替换成真正的 servoL 类接口。
        #
        # 例如：
        #
        # self.client.servoL(
        #     list(pose),
        #     self.period,
        # )
        #
        # 或：
        #
        # self.client.servo_cartesian(
        #     list(pose)
        # )
        #
        # ========================================================

        raise NotImplementedError(
            "Please implement servo_cartesian() "
            "using the HRCrobot servo API."
        )

        # TODO 4 实现以后，把下面代码取消注释：
        #
        # self._next_servo_time += self.period
        #
        # now = time.perf_counter()
        #
        # # 如果 Python 循环已经严重掉周期，
        # # 不追赶历史 setpoint。
        # if (
        #     now - self._next_servo_time
        #     > 2.0 * self.period
        # ):
        #     self._next_servo_time = now

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
        """

        if not self._connected:
            raise RuntimeError(
                "HRCrobot is not connected."
            )

        position = float(position)

        if position >= self.gripper_threshold:

            # ====================================================
            # TODO 5
            #
            # 换成 HRCrobot 的夹爪关闭接口：
            #
            # self.client.close_gripper()
            #
            # ====================================================

            self._gripper_position = 1.0

        else:

            # ====================================================
            # TODO 6
            #
            # 换成 HRCrobot 的夹爪打开接口：
            #
            # self.client.open_gripper()
            #
            # ====================================================

            self._gripper_position = 0.0

    def get_gripper_position(self) -> float:
        """
        如果夹爪有真实反馈，以后在这里读取。

        第一阶段先返回最后一次命令状态。
        """

        return float(
            self._gripper_position
        )