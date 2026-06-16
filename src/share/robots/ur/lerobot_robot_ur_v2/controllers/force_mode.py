from dataclasses import dataclass, field

import numpy as np

from .base import RTDETaskFrameControllerV2
from .config import ControllerConfig


@ControllerConfig.register_subclass("force_mode")
@dataclass
class ForceModeControllerConfig(ControllerConfig):
    """Parameters specific to the forceMode send path."""
    force_mode_gain_scaling: float = 1.0
    speed_limits: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 0.5, 0.5, 0.5])


class ForceModeController(RTDETaskFrameControllerV2):
    """Replicates the original RTDETaskFrameController send path via forceMode.

    Reads force_mode_gain_scaling and speed_limits from config.controller
    instead of the now-removed top-level URConfig fields.
    """

    def _send_task_wrench(self, rtde_c, wrench_F: np.ndarray) -> None:
        rtde_c.forceMode(
            self.origin.tolist(),
            [1, 1, 1, 1, 1, 1],
            np.asarray(wrench_F, dtype=np.float64).tolist(),
            2,
            self.config.controller.speed_limits,
        )
        self.force_on = True

    def _enter_task_force_mode(self, rtde_c) -> None:
        rtde_c.forceModeSetGainScaling(self.config.controller.force_mode_gain_scaling)
        self._send_task_wrench(rtde_c, np.zeros(6, dtype=np.float64))

    def _cleanup_rtde(self, rtde_c, rtde_r) -> None:
        try:
            if self.force_on:
                rtde_c.forceModeStop()
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
