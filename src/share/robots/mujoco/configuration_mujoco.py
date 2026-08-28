from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("mujoco")
@dataclass
class MujocoRobotConfig(RobotConfig):
    """Configuration for the repository-native MuJoCo UR5e backend."""

    scene_path: str | None = None
    scene_builder: str = "insertion"  # "insertion" (default) or "pick_insert"
    home_keyframe: str = "home"
    timestep: float = 0.002
    control_dt: float = 1.0 / 30.0
    use_gripper: bool = True
    gripper_min_command_interval_s: float = 0.5
    viewer: bool = False
    viewer_camera: str | None = None  # None/"free" keeps the interactive free camera.
    viewer_wrench_overlay: bool = True
    viewer_front_camera_overlay: bool = True
    viewer_wrist_camera_overlay: bool = True
    viewer_wrench_plot: bool = True
    viewer_diagnostics_width: int = 320
    viewer_diagnostics_fps: float = 10.0
    viewer_wrench_history_samples: int = 300
    randomize_fixture_xy: float = 0.002
    randomize_fixture_z: float = 0.0
    randomize_fixture_yaw_deg: float = 0.0
    randomize_camera_position_m: float = 0.0
    randomize_camera_rotation_deg: float = 0.0
    randomize_camera_fovy_deg: float = 0.0
    randomize_light_intensity_fraction: float = 0.0
    randomize_object_color_fraction: float = 0.0
    randomize_contact_friction_fraction: float = 0.0
    randomize_peg_mass_fraction: float = 0.0
    seed: int = 0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gravity_compensation: bool = True
    position_servo_stiffness_scale: float = 16.0

    ik_damping: float = 0.05
    ik_iterations: int = 20
    ik_max_joint_step: float = 0.05

    def __post_init__(self) -> None:
        if self.scene_builder not in {"insertion", "pick_insert"}:
            raise ValueError("scene_builder must be 'insertion' or 'pick_insert'.")
        if self.timestep <= 0 or self.control_dt < self.timestep:
            raise ValueError("Require 0 < timestep <= control_dt.")
        if self.ik_damping <= 0:
            raise ValueError("ik_damping must be positive.")
        if self.ik_iterations < 1:
            raise ValueError("ik_iterations must be at least one.")
        if self.ik_max_joint_step <= 0:
            raise ValueError("ik_max_joint_step must be positive.")
        non_negative_randomization = {
            "randomize_fixture_xy": self.randomize_fixture_xy,
            "randomize_fixture_z": self.randomize_fixture_z,
            "randomize_fixture_yaw_deg": self.randomize_fixture_yaw_deg,
            "randomize_camera_position_m": self.randomize_camera_position_m,
            "randomize_camera_rotation_deg": self.randomize_camera_rotation_deg,
            "randomize_camera_fovy_deg": self.randomize_camera_fovy_deg,
        }
        for name, value in non_negative_randomization.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        fractional_randomization = {
            "randomize_light_intensity_fraction": self.randomize_light_intensity_fraction,
            "randomize_object_color_fraction": self.randomize_object_color_fraction,
            "randomize_contact_friction_fraction": self.randomize_contact_friction_fraction,
            "randomize_peg_mass_fraction": self.randomize_peg_mass_fraction,
        }
        for name, value in fractional_randomization.items():
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
        if self.position_servo_stiffness_scale <= 0:
            raise ValueError("position_servo_stiffness_scale must be positive.")
        if self.gripper_min_command_interval_s < 0:
            raise ValueError("gripper_min_command_interval_s must be non-negative.")
        if self.viewer_diagnostics_width < 160:
            raise ValueError("viewer_diagnostics_width must be at least 160 pixels.")
        if self.viewer_diagnostics_fps <= 0:
            raise ValueError("viewer_diagnostics_fps must be positive.")
        if not 2 <= self.viewer_wrench_history_samples <= 1001:
            raise ValueError("viewer_wrench_history_samples must be between 2 and 1001.")
