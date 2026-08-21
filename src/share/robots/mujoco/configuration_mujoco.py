from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("mujoco")
@dataclass
class MujocoRobotConfig(RobotConfig):
    """Configuration for the repository-native MuJoCo UR5e backend."""

    scene_path: str | None = None
    home_keyframe: str = "home"
    timestep: float = 0.002
    control_dt: float = 1.0 / 30.0
    use_gripper: bool = True
    viewer: bool = False
    viewer_camera: str | None = None  # None/"free" keeps the interactive free camera.
    randomize_fixture_xy: float = 0.002
    seed: int = 0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gravity_compensation: bool = True
    position_servo_stiffness_scale: float = 16.0

    ik_damping: float = 0.05
    ik_iterations: int = 20
    ik_max_joint_step: float = 0.05

    def __post_init__(self) -> None:
        if self.timestep <= 0 or self.control_dt < self.timestep:
            raise ValueError("Require 0 < timestep <= control_dt.")
        if self.ik_damping <= 0:
            raise ValueError("ik_damping must be positive.")
        if self.ik_iterations < 1:
            raise ValueError("ik_iterations must be at least one.")
        if self.ik_max_joint_step <= 0:
            raise ValueError("ik_max_joint_step must be positive.")
        if self.randomize_fixture_xy < 0:
            raise ValueError("randomize_fixture_xy must be non-negative.")
        if self.position_servo_stiffness_scale <= 0:
            raise ValueError("position_servo_stiffness_scale must be positive.")
