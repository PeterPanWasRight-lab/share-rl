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

    robot_ip: str = "192.168.1.10"

    # ============================================================
    # Cartesian servo
    # ============================================================

    # 必须改成真机 servoL 类接口的实际控制频率
    frequency: float = 100.0

    # 如果厂家 servoL 接口有类似参数，可以使用
    # 没有的话暂时不用
    servo_lookahead_time: float = 0.1
    servo_gain: float = 300.0

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

    # ============================================================
    # Cameras
    # ============================================================

    # 第一阶段不用相机
    cameras: dict[str, CameraConfig] = field(default_factory=dict)