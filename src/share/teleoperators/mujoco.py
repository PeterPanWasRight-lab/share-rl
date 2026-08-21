from dataclasses import dataclass
from typing import Any

from lerobot.processor import RobotAction
from lerobot.processor.hil_processor import HasTeleopEvents
from lerobot.teleoperators import TeleopEvents, Teleoperator, TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


@TeleoperatorConfig.register_subclass("mujoco_delta")
@dataclass
class MujocoDeltaTeleopConfig(TeleoperatorConfig):
    """Connected no-op delta device for autonomous/headless MuJoCo actors."""


class MujocoDeltaTeleop(Teleoperator, HasTeleopEvents):
    config_class = MujocoDeltaTeleopConfig
    name = "mujoco_delta"

    def __init__(self, config: MujocoDeltaTeleopConfig):
        super().__init__(config)
        self._is_connected = False
        self._gripper_position = 1.0

    @property
    def action_features(self) -> dict[str, type]:
        features = {f"{axis}.vel": float for axis in ("x", "y", "z", "rx", "ry", "rz")}
        features["gripper.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._is_connected = True

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        self._is_connected = False

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_action(self) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        action = {key: 0.0 for key in self.action_features}
        action["gripper.pos"] = self._gripper_position
        return action

    def set_gripper_position(self, position: float) -> None:
        """Set the scripted teleoperation target used by simulation demos."""
        self._gripper_position = max(0.0, min(1.0, float(position)))

    def get_teleop_events(self) -> dict[str, Any]:
        return {TeleopEvents.IS_INTERVENTION: False}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback


__all__ = ["MujocoDeltaTeleop", "MujocoDeltaTeleopConfig"]
