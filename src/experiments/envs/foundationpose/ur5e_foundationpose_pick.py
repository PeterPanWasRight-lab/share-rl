"""UR5e pick pipeline with explicit FoundationPose and grasp-pose TODO hooks."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import Camera
from lerobot.envs import EnvConfig
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator

from experiments.envs.foundationpose.primitives import FoundationPosePrimitive, RuntimeFrameTargetPrimitiveConfig, \
    RelativeRuntimeFrameTargetPrimitiveConfig
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


def get_target_prim_cfg(target: list[float], processor: ManipulationPrimitiveProcessorConfig) -> ManipulationPrimitiveConfig:
    return ManipulationPrimitiveConfig(
        notes="Move to a known safe start pose.",
        processor=processor,
        task_frame=TaskFrame(
            target=target,
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

def get_object_relative_grasp_prim_cfg_10_cm_up(
    grasp_pose: list[float],
    processor: ManipulationPrimitiveProcessorConfig,
) -> RelativeRuntimeFrameTargetPrimitiveConfig:
    return RelativeRuntimeFrameTargetPrimitiveConfig(
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

    scan_pose: list[float] = field(default_factory=lambda: [-0.17156228687476865, -0.2535763671826385, 0.1845222693563364, 2.593396419946534, -0.02643405134839627, -1.324594623492149])
    stretch_pose: list[float] = field(default_factory=lambda: [-0.3879, -0.2751, 0.2226, 1.5888, -0.0644, -1.662])
    target_tolerance: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.10, 0.10, 0.10])

    path_pose : list[float] = field(default_factory=lambda:[-0.2901358445965176, -0.23288848515893296, 0.3625410451634672, 2.449444249306145, -0.09502183864460467, -1.4990255343351544])

    grasp_pose_in_object_frame: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    grasp_pose_in_object_frame_2: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    closed_gripper_position: float = 1.0
    open_gripper_position: float = 0.0
    gripper_hold_steps: int = 35
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
            "move_to_scan_pose": get_target_prim_cfg(self.scan_pose, move_processor),
            "estimate_object_pose": FoundationPosePrimitiveConfig(
                # processor=move_processor,
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="estimate object pose",
                grasp_obj="/home/jzilke/ws/share-rl-pe/hoermann_objects/black_obj/object_spec.json"
            ),
            "move_to_pregrasp_pose": get_object_relative_grasp_prim_cfg_10_cm_up(self.grasp_pose_in_object_frame, move_processor),
            "move_to_grasp_pose": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame, move_processor),
            "move_to_postgrasp_pose": get_object_relative_grasp_prim_cfg_10_cm_up(self.grasp_pose_in_object_frame, close_gripper_processor),
            "close_gripper": get_object_relative_grasp_prim_cfg_10_cm_up(self.grasp_pose_in_object_frame, close_gripper_processor),

            # "open_gripper": get_object_relative_grasp_prim_cfg_10_cm_up(self.grasp_pose_in_object_frame, open_gripper_processor),

            # "move_to_path": get_target_prim_cfg(self.path_pose, move_processor),

            "move_to_scan_pose_2": get_target_prim_cfg(self.scan_pose, move_processor),
            "estimate_object_pose_2": FoundationPosePrimitiveConfig(
                # processor=move_processor,
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="estimate object pose",
                grasp_obj="/home/jzilke/ws/share-rl-pe/hoermann_objects/black_obj/object_spec.json"
            ),
            "move_to_grasp_pose_2": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame_2, move_processor),
            "close_gripper_2": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame_2,
                                                                close_gripper_processor),
            "open_gripper_2": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame_2, open_gripper_processor),


        }
        self.transitions = [
            OnTargetPoseReached(
                source="move_to_scan_pose",
                target="estimate_object_pose",
                tolerance=list(self.target_tolerance),
            ),
            Always(
                source="estimate_object_pose",
                target="move_to_pregrasp_pose",
                # tolerance=list(self.target_tolerance)
            ),
            OnTargetPoseReached(
                source="move_to_pregrasp_pose",
                target="move_to_grasp_pose",
                tolerance=list(self.target_tolerance),
            ),
            OnTargetPoseReached(
                source="move_to_grasp_pose",
                target="close_gripper",
                tolerance=list(self.target_tolerance),
            ),

            OnTimeLimit(
                source="close_gripper",
                target="move_to_postgrasp_pose",
                max_steps=int(self.gripper_hold_steps),
            ),

            OnTargetPoseReached(
                source="move_to_postgrasp_pose",
                target="move_to_scan_pose",
                tolerance=list(self.target_tolerance),
            ),

            # OnTimeLimit(
            #     source="open_gripper",
            #     target="move_to_scan_pose",
            #     max_steps=int(self.gripper_hold_steps),
            # ),


            OnTargetPoseReached(
                source="move_to_scan_pose_2",
                target="estimate_object_pose_2",
                tolerance=list(self.target_tolerance),
            ),
            Always(
                source="estimate_object_pose_2",
                target="move_to_grasp_pose_2",
                # tolerance=list(self.target_tolerance)
            ),

            # OnTargetPoseReached(
            #     source="move_to_path",
            #     target="move_to_grasp_pose_2",
            #     tolerance=list(self.target_tolerance),
            # ),
            OnTargetPoseReached(
                source="move_to_grasp_pose_2",
                target="close_gripper_2",
                tolerance=list(self.target_tolerance),
            ),
            OnTimeLimit(
                source="close_gripper_2",
                target="open_gripper_2",
                max_steps=int(self.gripper_hold_steps)*3,
            ),
            OnTimeLimit(
                source="open_gripper_2",
                target="move_to_scan_pose",
                max_steps=int(self.gripper_hold_steps),
            ),
        ]

        super().__post_init__()


__all__ = [
    "UR5eFoundationPosePickEnvConfig",
]
