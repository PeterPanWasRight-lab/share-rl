from multiprocessing.managers import SharedMemoryManager

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from share.robots.ur.lerobot_robot_ur.controller import TaskFrameCommand
from share.robots.ur.lerobot_robot_ur.ur import UR
from share.envs.manipulation_primitive.task_frame import ControlSpace
from share.grippers.robotiq_controller import RTDERobotiqController

from .config_ur_v2 import URV2Config
from .controllers import (
    DirectTorqueController,
    DirectTorqueControllerConfig,
    ForceModeController,
    ForceModeControllerConfig,
    RTDETaskFrameControllerV2,
)

_CONTROLLER_MAP = {
    ForceModeControllerConfig: ForceModeController,
    DirectTorqueControllerConfig: DirectTorqueController,
}


def _make_controller(config: URV2Config) -> RTDETaskFrameControllerV2:
    cls = _CONTROLLER_MAP.get(type(config.controller))
    if cls is None:
        raise ValueError(
            f"Unknown controller config type: {type(config.controller).__name__}. "
            f"Expected one of: {[t.__name__ for t in _CONTROLLER_MAP]}"
        )
    return cls(config)


class URV2(UR):
    """UR robot with a pluggable low-level controller backend.

    Identical to UR in every way except the controller is selected at
    construction time via config.controller rather than always using
    RTDETaskFrameController (forceMode). Use DirectTorqueControllerConfig
    to bypass UR's internal admittance layer for contact-rich tasks.
    """

    config_class = URV2Config
    name = "ur_v2"

    def __init__(self, config: URV2Config):
        # Call Robot.__init__ directly to avoid UR.__init__ constructing an
        # RTDETaskFrameController (forceMode) that we would immediately discard.
        Robot.__init__(self, config)
        self.config = config
        self.task_frame = TaskFrameCommand(controller_overrides=self._default_controller_overrides())

        self.shm = SharedMemoryManager()
        self.shm.start()
        config.shm_manager = self.shm

        self.controller = _make_controller(config)

        if self.config.use_gripper:
            gripper_range = self._gripper_range_from_calibration()
            min_position, max_position = gripper_range if gripper_range is not None else (None, None)
            self.gripper = RTDERobotiqController(
                hostname=config.robot_ip,
                shm_manager=self.shm,
                frequency=config.gripper_frequency,
                soft_real_time=config.gripper_soft_real_time,
                rt_core=config.gripper_rt_core,
                auto_calibrate=not self.is_calibrated,
                min_position=min_position,
                max_position=max_position,
                verbose=config.verbose,
            )
        else:
            self.gripper = None

        self.cameras = make_cameras_from_configs(config.cameras)

        self.logs = {}
        self.last_robot_action = TaskFrameCommand()
        self._active_control_space: ControlSpace | None = None
