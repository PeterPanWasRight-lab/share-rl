import copy
from functools import cached_property
from typing import Any

from scipy.spatial.transform import Rotation

from lerobot.robots import Robot
from lerobot.utils.errors import (
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)

from share.envs.manipulation_primitive.task_frame import (
    TASK_FRAME_AXIS_NAMES,
    TaskFrame,
)

from share.utils.transformation_utils import (
    task_pose_to_world_pose,
    world_pose_to_task_pose,
)

from .config_HRCrobot import HRCrobotConfig
from .controller import HRCrobotController


class HRCrobot(Robot):
    """
    LeRobot / SHaRe adapter for HRCrobot.

    第一阶段仅支持：

        Cartesian position control
        +
        gripper open / close
    """

    config_class = HRCrobotConfig
    name = "HRCrobot"

    def __init__(
        self,
        config: HRCrobotConfig,
    ):
        super().__init__(config)

        self.config = config

        self.controller = HRCrobotController(
            robot_ip=config.robot_ip,
            frequency=config.frequency,
            gripper_threshold=config.gripper_threshold,
        )

        # 当前 SHaRe TaskFrame。
        #
        # 这里只是保存坐标系信息，
        # set_task_frame() 本身绝不能驱动机器人。
        self.task_frame = TaskFrame(
            origin=[0.0] * 6,
            target=[0.0] * 6,
        )

        self._last_gripper_command = None

    # ============================================================
    # Feature definitions
    # ============================================================

    @property
    def _motors_ft(
        self,
    ) -> dict[str, type]:
        """
        SHaRe ManipulationPrimitive 会读取这个字段。

        第一阶段只提供：

            6D Cartesian pose
            gripper state
        """

        features = {
            f"{axis}.ee_pos": float
            for axis in TASK_FRAME_AXIS_NAMES
        }

        if self.config.use_gripper:
            features["gripper.pos"] = float

        return features

    @cached_property
    def observation_features(
        self,
    ) -> dict[str, type]:
        return dict(
            self._motors_ft
        )

    @property
    def action_features(
        self,
    ) -> dict[str, type]:
        """
        当前只支持 task-space position command。
        """

        features = {
            f"{axis}.ee_pos": float
            for axis in TASK_FRAME_AXIS_NAMES
        }

        if self.config.use_gripper:
            features["gripper.pos"] = float

        return features

    # ============================================================
    # Connection
    # ============================================================

    @property
    def is_connected(
        self,
    ) -> bool:
        return self.controller.is_connected

    @property
    def is_calibrated(
        self,
    ) -> bool:
        # 第一阶段不使用 LeRobot 的 motor calibration
        return True

    def connect(
        self,
        calibrate: bool = True,
    ) -> None:

        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected."
            )

        self.controller.connect()

        self.configure()

    def disconnect(
        self,
    ) -> None:

        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        self.controller.disconnect()

    def calibrate(
        self,
    ) -> None:
        return None

    def configure(
        self,
    ) -> None:
        return None

    # ============================================================
    # Task frame
    # ============================================================

    def set_task_frame(
        self,
        new_task_frame: TaskFrame,
    ) -> None:
        """
        非常重要：

        set_task_frame() 只保存 SHaRe 当前定义的 TaskFrame。

        这里：
            不调用 servo
            不调用 move
            不驱动真机

        真正运动只允许发生在 send_action()。
        """

        self.task_frame = copy.deepcopy(
            new_task_frame
        )

    # ============================================================
    # Vendor pose <-> SHaRe pose
    # ============================================================

    @staticmethod
    def _vendor_pose_to_world_rpy(
        vendor_pose: list[float],
    ) -> list[float]:
        """
        当前假设厂家返回：

            [
                x,
                y,
                z,
                rotvec_x,
                rotvec_y,
                rotvec_z,
            ]

        转成 SHaRe：

            [
                x,
                y,
                z,
                roll,
                pitch,
                yaw,
            ]

        SHaRe 使用：
            extrinsic XYZ Euler
            radians
        """

        if len(vendor_pose) != 6:
            raise ValueError(
                "Vendor TCP pose must contain 6 values."
            )

        xyz = vendor_pose[:3]
        rotvec = vendor_pose[3:6]

        rpy = Rotation.from_rotvec(
            rotvec
        ).as_euler(
            "xyz",
            degrees=False,
        )

        return [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(rpy[0]),
            float(rpy[1]),
            float(rpy[2]),
        ]

    @staticmethod
    def _world_rpy_to_vendor_pose(
        world_pose: list[float],
    ) -> list[float]:
        """
        SHaRe xyz+rpy
        ->
        厂家 xyz+rotation-vector
        """

        if len(world_pose) != 6:
            raise ValueError(
                "World pose must contain 6 values."
            )

        xyz = world_pose[:3]
        rpy = world_pose[3:6]

        rotvec = Rotation.from_euler(
            "xyz",
            rpy,
            degrees=False,
        ).as_rotvec()

        return [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(rotvec[0]),
            float(rotvec[1]),
            float(rotvec[2]),
        ]

    # ============================================================
    # Observation
    # ============================================================

    def get_observation(
        self,
    ) -> dict[str, Any]:

        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        # --------------------------------------------------------
        # 1. 从厂家读取 TCP
        # --------------------------------------------------------

        vendor_pose = (
            self.controller.get_tcp_pose()
        )

        # --------------------------------------------------------
        # 2. 厂家表示
        #    ->
        #    SHaRe world/base xyz+rpy
        # --------------------------------------------------------

        world_pose = (
            self._vendor_pose_to_world_rpy(
                vendor_pose
            )
        )

        # --------------------------------------------------------
        # 3. world/base frame
        #    ->
        #    当前 primitive TaskFrame
        # --------------------------------------------------------

        task_pose = world_pose_to_task_pose(
            world_pose,
            self.task_frame.origin,
        )

        observation: dict[str, Any] = {}

        for i, axis in enumerate(
            TASK_FRAME_AXIS_NAMES
        ):
            observation[
                f"{axis}.ee_pos"
            ] = float(
                task_pose[i]
            )

        if self.config.use_gripper:
            observation[
                "gripper.pos"
            ] = (
                self.controller
                .get_gripper_position()
            )

        return observation

    # ============================================================
    # Action
    # ============================================================

    def send_action(
        self,
        action: dict[str, Any],
    ) -> dict[str, Any]:

        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self} is not connected."
            )

        # ========================================================
        # Cartesian servo
        # ========================================================

        has_cartesian_action = any(
            f"{axis}.ee_pos" in action
            for axis in TASK_FRAME_AXIS_NAMES
        )

        if has_cartesian_action:

            # 当前 TaskFrame target
            # 作为默认目标。
            task_pose = list(
                self.task_frame.target
            )

            # 用 action 覆盖收到的 axis
            for i, axis in enumerate(
                TASK_FRAME_AXIS_NAMES
            ):
                key = f"{axis}.ee_pos"

                if key in action:
                    task_pose[i] = float(
                        action[key]
                    )

            # ----------------------------------------------------
            # TaskFrame pose
            # ->
            # robot base/world pose
            # ----------------------------------------------------

            world_pose = task_pose_to_world_pose(
                task_pose,
                self.task_frame.origin,
            )

            # ----------------------------------------------------
            # SHaRe xyz+rpy
            # ->
            # HRCrobot 厂家 pose
            # ----------------------------------------------------

            vendor_pose = (
                self._world_rpy_to_vendor_pose(
                    world_pose
                )
            )

            # ----------------------------------------------------
            # 真正 servo
            # ----------------------------------------------------

            self.controller.servo_cartesian(
                vendor_pose
            )

        # ========================================================
        # Gripper
        # ========================================================

        if (
            self.config.use_gripper
            and "gripper.pos" in action
        ):
            gripper_cmd = float(
                action["gripper.pos"]
            )

            # 夹爪没必要像 servoL 一样重复发送。
            #
            # 只有目标发生变化时才发。
            if (
                self._last_gripper_command is None
                or abs(
                    gripper_cmd
                    - self._last_gripper_command
                ) > 1e-6
            ):
                self.controller.set_gripper(
                    gripper_cmd
                )

                self._last_gripper_command = (
                    gripper_cmd
                )

        return action