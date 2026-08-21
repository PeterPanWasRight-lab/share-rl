from dataclasses import dataclass

import numpy as np
from lerobot.cameras import Camera, CameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from share.robots.mujoco.registry import get_robot


@CameraConfig.register_subclass("mujoco")
@dataclass
class MujocoCameraConfig(CameraConfig):
    """Camera backed by a named camera in a connected MuJoCo robot scene."""

    robot_id: str = "mujoco-arm"
    camera_name: str = "front"
    width: int = 320
    height: int = 240
    fps: int = 30


class MujocoCamera(Camera):
    config_class = MujocoCameraConfig

    def __init__(self, config: MujocoCameraConfig):
        super().__init__(config)
        self.config = config
        self._robot = None

    @property
    def is_connected(self) -> bool:
        return self._robot is not None

    def connect(self, warmup: bool = True) -> None:
        del warmup
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._robot = get_robot(self.config.robot_id)
        self.async_read()

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        self._robot = None

    @staticmethod
    def find_cameras() -> list[dict[str, str]]:
        return []

    def read(self) -> np.ndarray:
        return self.async_read()

    def async_read(self, timeout_ms: float | None = None) -> np.ndarray:
        del timeout_ms
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        return self._robot.render_camera(
            camera_name=self.config.camera_name,
            width=self.config.width,
            height=self.config.height,
        )
