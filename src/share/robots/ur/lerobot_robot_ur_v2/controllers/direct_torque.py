from dataclasses import dataclass

import numpy as np

from share.utils.transformation_utils import sixvec_to_homogeneous

from .base import RTDETaskFrameControllerV2
from .config import ControllerConfig


@ControllerConfig.register_subclass("direct_torque")
@dataclass
class DirectTorqueControllerConfig(ControllerConfig):
    """Parameters specific to the directTorque send path.

    friction_compensation enables UR's internal friction model (gravity is
    always compensated). Polyscope 5.25.x introduced finer-grained control
    via viscous_scale and coulomb_scale per joint; these fields are commented
    out until ur_rtde exposes them.
    """
    friction_compensation: bool = True
    # Polyscope >= 5.25.x (requires ur_rtde update):
    # viscous_scale: list[float] | None = None   # UR default: [0.9, 0.9, 0.8, 0.9, 0.9, 0.9]
    # coulomb_scale: list[float] | None = None   # UR default: [0.8, 0.8, 0.7, 0.8, 0.8, 0.8]


class DirectTorqueController(RTDETaskFrameControllerV2):
    """Task-space controller that sends joint torques via directTorque.

    The impedance control law (wrench computation, bounds, compliance) is
    inherited unchanged from RTDETaskFrameController. The computed task-space
    wrench is transformed to joint torques using the Jacobian transpose:

        τ = J(q)^T · F_world

    where F_world is wrench_F rotated from the task frame to the robot base
    frame. This bypasses UR's internal 2kHz admittance layer used by forceMode,
    giving lower-latency, more direct contact response.

    Moment-arm note: the rotation-only transform is exact for forces and exact
    for moments when the task-frame origin coincides with the TCP. If the origin
    is offset from the TCP, the moment at the TCP differs by cross(r, f). For
    the Hormann insertion config the origin is set at the connector pose (~TCP),
    so the approximation holds. An exact correction can be added by caching the
    world-frame TCP position from read_current_state if needed.
    """

    def _send_task_wrench(self, rtde_c, wrench_F: np.ndarray) -> None:
        # 1. Jacobian at current configuration: 6×6, maps q̇ → v_base
        J = np.array(rtde_c.getJacobian()).reshape(6, 6)

        # 2. Rotate wrench from task frame to robot base (world) frame.
        #    self.origin[3:6] is stored as a rotation vector internally.
        T_wt = sixvec_to_homogeneous(self.origin)
        R_wt = T_wt[:3, :3]
        block_R = np.block([[R_wt, np.zeros((3, 3))], [np.zeros((3, 3)), R_wt]])
        wrench_world = block_R @ np.asarray(wrench_F, dtype=np.float64)

        # 3. τ = J^T · F_world
        tau = J.T @ wrench_world
        rtde_c.directTorque(tau.tolist(), self.config.controller.friction_compensation)

    def _enter_task_force_mode(self, _rtde_c) -> None:
        # directTorque is stateless — must be called every timestep, no mode entry.
        self.force_on = True

    def _cleanup_rtde(self, rtde_c, rtde_r) -> None:
        try:
            rtde_c.directTorque([0.0] * 6, False)
        except Exception:
            pass
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        try:
            rtde_c.disconnect()
        except Exception:
            pass
        try:
            rtde_r.disconnect()
        except Exception:
            pass
