from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("HRCrobot")
@dataclass
class HRCrobotConfig(RobotConfig):
    """
    Configuration of HRCrobot.

    第一阶段只保留：
    - 机器人通信地址
    - Cartesian servo 控制频率
    - 夹爪基本参数
    """

    # ============================================================
    # Communication
    # ============================================================

    # 机器人控制器 IP
    robot_ip: str = "10.10.59.211"

    # hsrosi 位姿/运动链路端口
    hrc_port: int = 9095

    # HSC3 夹爪 IO 链路端口
    hsc3_port: int = 23234

    # ============================================================
    # Cartesian servo
    # ============================================================

    # servo_cartesian 流式下发频率 (Hz)。
    # SDK 没有 servoL 接口，controller 层用
    # move_to_cartesian_position 按此周期限频。
    frequency: float = 100.0

    # ============================================================
    # Gripper
    # ============================================================

    use_gripper: bool = True

    # SHaRe/HRCrobot adapter 内部统一约定：
    #
    # 0.0 -> open
    # 1.0 -> close
    #
    gripper_open: float = 0.0
    gripper_close: float = 1.0
    gripper_threshold: float = 0.5
    gripper_min_command_interval_s: float = 0.5

    # ============================================================
    # Cameras
    # ============================================================

    # 第一阶段不用相机
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.frequency <= 0.0:
            raise ValueError("HRCrobotConfig.frequency must be positive.")
        if not 0.0 <= self.gripper_threshold <= 1.0:
            raise ValueError("HRCrobotConfig.gripper_threshold must be in [0, 1].")
        if self.gripper_min_command_interval_s < 0.0:
            raise ValueError(
                "HRCrobotConfig.gripper_min_command_interval_s must be non-negative."
            )
