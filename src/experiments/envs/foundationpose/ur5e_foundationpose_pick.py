"""UR5e pick pipeline with explicit FoundationPose and grasp-pose TODO hooks."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import Camera
from lerobot.envs import EnvConfig
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator

from experiments.envs.foundationpose.primitives import FoundationPosePrimitive, RuntimeFrameTargetPrimitiveConfig
from share.cameras.configuration_realsense_depth import RealSenseDepthCameraConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import Always, OnTargetPoseReached, OnTimeLimit
from share.pose_estimation.grasp_obj_spec import GraspObjectSpec
from share.robots.ur import URConfig
from share.teleoperators.spacemouse import SpaceMouseConfig

import logging

logger = logging.getLogger(__name__)


def _shared_processor(gripper_pos: float=0.0) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=30,
        observation=ObservationConfig(
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_ee_pos_to_observation=True,
            add_joint_position_to_observation=False,
        ),
        gripper=GripperConfig(
            enable=False,
            static_pos=gripper_pos,
        ),
    )

@ManipulationPrimitiveConfig.register_subclass("foundation_pose")
@dataclass
class FoundationPosePrimitiveConfig(ManipulationPrimitiveConfig):
    grasp_obj: GraspObjectSpec|str|None = None
    def validate(self, robot_dict, teleop_dict):
        super().validate(robot_dict, teleop_dict)

    def make(
            self,
            robot_dict: dict[str, Robot],
            teleop_dict: dict[str, Teleoperator],
            cameras: dict[str, Camera],
            device: str = "cpu"
    ):
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)  # todo: fix initial_features

        display_cameras = self.processor.image_preprocessing is not None and self.processor.image_preprocessing.display_cameras
        env = FoundationPosePrimitive(task_frame=self.task_frame,
                                      robot_dict=robot_dict,
                                      cameras=cameras,
                                      display_cameras=display_cameras,
                                      grasp_object=self.grasp_obj,
                                      pose_key="object_pose")

        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor


def get_target_prim_cfg(processor: ManipulationPrimitiveProcessorConfig) -> ManipulationPrimitiveConfig:
    return ManipulationPrimitiveConfig(
        notes="Move to a known safe start pose.",
        processor=processor,
        task_frame=TaskFrame(
            target=[-0.23552485078806693, -0.27116002789910776, 0.37228272132740536, 1.9188068639552711, 0.0017689096521515957, -1.6494817075949697],
            #target=[-0.29878662794237504, -0.24038619921648444, 0.47113762731620834, 2.088740637708761, -0.049881005988045235, -1.1461642042972513],
            policy_mode=[None] * 6,
            control_mode=[ControlMode.POS] * 6,
        ),
    )


def get_object_relative_grasp_prim_cfg(
    grasp_pose: list[float],
    processor: ManipulationPrimitiveProcessorConfig,
) -> RuntimeFrameTargetPrimitiveConfig:
    return RuntimeFrameTargetPrimitiveConfig(
        notes="Move to the fixed grasp pose expressed in the estimated object frame.",
        processor=processor,
        task_frame=TaskFrame(
            target=list(grasp_pose),
            policy_mode=[None] * 6,
            control_mode=[ControlMode.POS] * 6,
        ),
        frame_origin_runtime_key="object_pose",
    )



@EnvConfig.register_subclass("ur5e_foundationpose_pick")
@dataclass
class UR5eFoundationPosePickEnvConfig(ManipulationPrimitiveNetConfig):
    """UR5e pipeline: scan pose, FoundationPose, grasp target, close gripper."""

    robot_ip: str = "172.22.22.2"
    fps: int = 30
    offline: bool = False
    start_primitive: str = "move_to_scan_pose"
    reset_primitive: str = "move_to_scan_pose"
    camera_serial_number: str = "352122271533"

    scan_pose: list[float] = field(default_factory=lambda: [-0.429, 0.126, 0.261, 3.112, 0.068, -2.14])
    stretch_pose: list[float] = field(default_factory=lambda: [-0.3879, -0.2751, 0.2326, 1.5888, -0.0644, -1.662])
    target_tolerance: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.10, 0.10, 0.10])
    grasp_pose_in_object_frame: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    closed_gripper_position: float = 1.0
    open_gripper_position: float = 0.0
    gripper_hold_steps: int = 15
    mock_initial_pose: list[float] = field(default_factory=lambda: [0.45, -0.20, 0.35, 3.14, 0.0, 0.0])

    def make(self):
        return super().make()

    def __post_init__(self) -> None:
        move_processor = _shared_processor(self.open_gripper_position)
        close_gripper_processor = _shared_processor(self.closed_gripper_position)
        open_gripper_processor = _shared_processor(self.open_gripper_position)

        self.robot = URConfig(
            robot_ip=self.robot_ip,
            frequency=500,
            soft_real_time=True,
            rt_core=3,
            use_gripper=True,
            use_force_mode=False
        )
        self.teleop = SpaceMouseConfig(action_scale=[0.25, 0.25, 0.20, 0.50, 0.50, 0.50])
        self.cameras = {
            "main": RealSenseDepthCameraConfig(
                serial_number_or_name=self.camera_serial_number,
                use_depth=True
            ),
        }

        self.primitives = {
            "move_to_scan_pose": get_target_prim_cfg(move_processor),
            "estimate_object_pose": FoundationPosePrimitiveConfig(
                # processor=move_processor,
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="estimate object pose",
                grasp_obj="/home/jzilke/ws/share-rl-pe/hoermann_objects/power_connector/object_spec.json"
            ),
            "move_to_grasp_pose": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame, move_processor),
            "close_gripper": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame, close_gripper_processor),
            "open_gripper": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame, open_gripper_processor),
        }
        self.transitions = [
            OnTargetPoseReached(
                source="move_to_scan_pose",
                target="estimate_object_pose",
                tolerance=list(self.target_tolerance),
            ),
            Always(
                source="estimate_object_pose",
                target="move_to_grasp_pose",
                # tolerance=list(self.target_tolerance)
            ),
            OnTargetPoseReached(
                source="move_to_grasp_pose",
                target="close_gripper",
                tolerance=list(self.target_tolerance),
            ),
            OnTimeLimit(
                source="close_gripper",
                target="open_gripper",
                max_steps=int(self.gripper_hold_steps),
            ),
            OnTimeLimit(
                source="open_gripper",
                target="move_to_scan_pose",
                max_steps=int(self.gripper_hold_steps),
            ),
        ]

        super().__post_init__()


__all__ = [
    "UR5eFoundationPosePickEnvConfig",
]
