import logging
from dataclasses import dataclass, field
from multiprocessing.managers import SharedMemoryManager
from typing import Optional, Sequence

import numpy as np
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from .controllers import ControllerConfig, ForceModeControllerConfig


@RobotConfig.register_subclass("ur_v2")
@dataclass
class URV2Config(RobotConfig):
    """Full drop-in replacement for URConfig with a pluggable controller backend.

    Identical field set to URConfig except ``force_mode_gain_scaling`` and
    ``speed_limits`` are removed — they now belong to ForceModeControllerConfig.
    The ``controller`` field selects the low-level send path and holds its params.

    Usage:
        URV2Config(controller=ForceModeControllerConfig(force_mode_gain_scaling=0.93), ...)
        URV2Config(controller=DirectTorqueControllerConfig(friction_compensation=True), ...)

    From CLI:
        --env.robot.type=ur_v2 --env.robot.controller.type=direct_torque
    """

    robot_ip: str
    model: str = "ur5e"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # gripper
    use_gripper: bool = False
    gripper_frequency: float = 50.0
    gripper_vel: float = 1.0
    gripper_force: float = 1.0
    gripper_soft_real_time: bool = False
    gripper_rt_core: int = 4

    # controller
    frequency: float = 500.0
    payload_mass: Optional[float] = None
    payload_cog: Optional[Sequence[float]] = None
    tcp_offset_pose: Optional[list[float]] = None
    soft_real_time: bool = False
    rt_core: int = 3
    launch_timeout: float = 10.0
    get_max_k: int = 128
    shm_manager: Optional[SharedMemoryManager] = None
    ft_filter_cutoff_hz: Optional[float] = None

    # impedance gains
    kp: list[float] = field(default_factory=lambda: [2500.0, 2500.0, 2500.0, 150.0, 150.0, 150.0])
    kd: list[float] = field(default_factory=lambda: [80.0, 80.0, 80.0, 8.0, 8.0, 8.0])

    # pose bounds
    max_pose_rpy: list[float] = field(default_factory=lambda: [float("inf")] * 6)
    min_pose_rpy: list[float] = field(default_factory=lambda: [-float("inf")] * 6)

    # force limits
    wrench_limits: list[float] = field(default_factory=lambda: [30.0, 30.0, 30.0, 3.0, 3.0, 3.0])

    # compliance / anti-windup
    compliance_adaptive_limit_enable: list[bool] = field(default_factory=lambda: [False] * 6)
    compliance_reference_limit_enable: list[bool] = field(default_factory=lambda: [False] * 6)
    compliance_desired_wrench: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 0.5, 0.5, 0.5])
    compliance_adaptive_limit_theta: Optional[list[float]] = None
    compliance_adaptive_limit_min: list[float] = field(default_factory=lambda: [0.1] * 6)

    # flags
    use_degrees: bool = False
    verbose: bool = False
    mock: bool = False
    debug: bool = False
    debug_axis: int = 0

    # controller backend
    controller: ControllerConfig = field(default_factory=ForceModeControllerConfig)

    def __post_init__(self):
        if len(self.kp) != 6:
            raise ValueError("URV2Config.kp must be a length-6 list.")
        if len(self.kd) != 6:
            raise ValueError("URV2Config.kd must be a length-6 list.")
        if len(self.compliance_adaptive_limit_enable) != 6:
            raise ValueError("URV2Config.compliance_adaptive_limit_enable must be a length-6 list.")
        if len(self.compliance_reference_limit_enable) != 6:
            raise ValueError("URV2Config.compliance_reference_limit_enable must be a length-6 list.")

        if self.compliance_adaptive_limit_theta is None:
            if self.verbose:
                logging.info("=== Compute parameters for exponential contact force limit scaling: ===")

            self.compliance_adaptive_limit_theta = [0.0] * 6
            for i in range(6):
                if not self.compliance_adaptive_limit_enable[i]:
                    continue

                if self.wrench_limits[i] == float("inf"):
                    self.wrench_limits[i] = 2.0 * self.compliance_desired_wrench[i]

                theta = self.compute_theta(
                    self.wrench_limits[i],
                    self.compliance_desired_wrench[i],
                    self.compliance_adaptive_limit_min[i],
                )

                s_star, ds_df_star = self.exp_scale_and_derivative(
                    self.compliance_desired_wrench[i],
                    theta,
                    self.compliance_adaptive_limit_min[i],
                )
                g_prime = self.wrench_limits[i] * ds_df_star

                if self.verbose:
                    logging.info(f" {['X', 'Y', 'Z', 'A', 'B', 'C'][i]}-Axis:")
                    logging.info(f"  Computed θ = {theta:.4f}")
                    logging.info(f"  At f* = {self.compliance_desired_wrench[i]} N:")
                    logging.info(f"    s(f*) = {s_star:.4f}")
                    logging.info(f"    s'(f*) = {ds_df_star:.4f}")
                    logging.info(f"    g'(f*) = F_max * s'(f*) = {g_prime:.4f}")
                    if abs(g_prime) < 1.0:
                        logging.info("  --> Stable fixed point (|g'(f*)| < 1)")
                    else:
                        logging.warning("  --> Unstable: bifurcation/oscillation likely (|g'(f*)| >= 1)")

                if abs(g_prime) >= 1.0:
                    raise ValueError(
                        f"Likely oscillation on {['X', 'Y', 'Z', 'A', 'B', 'C'][i]}-axis contact "
                        f"force limiter, run again with verbose=True and check parameters!"
                    )

                self.compliance_adaptive_limit_theta[i] = theta

    @staticmethod
    def compute_theta(F_max: float, f_star: float, s_min: float) -> float:
        """Compute decay constant θ satisfying f_star = F_max * [s_min + (1-s_min)*exp(-f_star/θ)]."""
        s_star = f_star / F_max
        if not (s_min < s_star < 1.0):
            raise ValueError("Require s_min < f_star/F_max < 1.0")
        ratio = (s_star - s_min) / (1.0 - s_min)
        return -f_star / np.log(ratio)

    @staticmethod
    def exp_scale_and_derivative(f: float, theta: float, s_min: float) -> tuple:
        """Return scale s(f) and derivative s'(f) for exponential-decay-to-floor."""
        exp_term = np.exp(-f / theta)
        s = s_min + (1 - s_min) * exp_term
        ds_df = -(1 - s_min) / theta * exp_term
        return s, ds_df
