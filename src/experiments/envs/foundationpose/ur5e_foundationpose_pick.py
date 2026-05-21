"""UR5e pick pipeline with explicit FoundationPose and grasp-pose TODO hooks."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.cameras import Camera
from lerobot.envs import EnvConfig
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator

from experiments.envs.foundationpose.primitives import FoundationPosePrimitive, RuntimeFrameTargetPrimitiveConfig, \
    FoundationPosePrimitiveConfig
from share.cameras.configuration_realsense_depth import RealSenseDepthCameraConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig, EventConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, TaskFrame, PolicyMode, ControlSpace
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import Always, OnTargetPoseReached, OnTimeLimit
from pose_estimation import GraspObjectSpec
from share.robots.ur import URConfig
from share.teleoperators import TeleopEvents
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



def get_target_prim_cfg(target: list[float], processor: ManipulationPrimitiveProcessorConfig) -> ManipulationPrimitiveConfig:
    return ManipulationPrimitiveConfig(
        notes="Move to a known safe start pose.",
        processor=processor,
        task_frame=TaskFrame(
            target=target,
            policy_mode=[None] * 6,
            control_mode=[ControlMode.POS] * 6,
            controller_overrides={
                "use_force_mode": False,
            }
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

    scan_pose: list[float] = field(default_factory=lambda: [-0.23552485078806693, -0.27116002789910776, 0.37228272132740536, 1.9188068639552711, 0.0017689096521515957, -1.6494817075949697])
    scan_pose_2: list[float] = field(default_factory=lambda: [-0.31169864008803394, -0.2704242243991801, 0.2899598338831783, 1.8875908709963132, -0.06287159096366723, -1.609617948354109])
    stretch_pose: list[float] = field(default_factory=lambda: [-0.3879, -0.2751, 0.2226, 1.5888, -0.0644, -1.662])
    plug_pose: list[float] = field(default_factory=lambda: [-0.20310489971477347, -0.34529285807579696, 0.1894367040218624, 2.5756071290528713, -0.0023949499862503387, -1.5539570934357796])

    target_tolerance: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.10, 0.10, 0.10])

    path_pose : list[float] = field(default_factory=lambda:[-0.2901358445965176, -0.23288848515893296, 0.3625410451634672, 2.449444249306145, -0.09502183864460467, -1.4990255343351544])

    grasp_pose_in_object_frame: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    grasp_pose_in_object_frame_2: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
            "move_to_scan_pose": get_target_prim_cfg(self.scan_pose, move_processor),
            "estimate_object_pose": FoundationPosePrimitiveConfig(
                # processor=move_processor,
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="estimate object pose",
                grasp_obj="/home/jzilke/ws/share-rl-pe/hoermann_objects/power_connector/object_spec.json"
            ),
            "move_to_grasp_pose": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame, move_processor),
            "close_gripper": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame, close_gripper_processor),
            "move_to_stretch_pose": get_target_prim_cfg(self.stretch_pose, close_gripper_processor),
            "open_gripper": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame, open_gripper_processor),

            # "move_to_path": get_target_prim_cfg(self.path_pose, move_processor),

            "move_to_scan_pose_2": get_target_prim_cfg(self.scan_pose_2, open_gripper_processor),
            "estimate_object_pose_2": FoundationPosePrimitiveConfig(
                # processor=move_processor,
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="estimate object pose",
                grasp_obj="/home/jzilke/ws/share-rl-pe/hoermann_objects/power_connector/object_spec.json"
            ),
            "move_to_grasp_pose_2": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame_2, move_processor),
            "close_gripper_2": get_object_relative_grasp_prim_cfg(self.grasp_pose_in_object_frame_2,
                                                                close_gripper_processor),
            "move_to_plug_pose": get_target_prim_cfg(self.plug_pose, close_gripper_processor),
            "move_to_scan_pose_3": get_target_prim_cfg(self.scan_pose, open_gripper_processor),
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
                target="move_to_stretch_pose",
                max_steps=int(self.gripper_hold_steps),
            ),
            OnTargetPoseReached(
                source="move_to_stretch_pose",
                target="move_to_scan_pose_2",
                tolerance=list(self.target_tolerance),
            ),
            OnTimeLimit(
                source="open_gripper",
                target="move_to_scan_pose_2",
                max_steps=int(self.gripper_hold_steps),
            ),


            OnTargetPoseReached(
                source="move_to_scan_pose_2",
                target="estimate_object_pose_2",
                tolerance=list(self.target_tolerance),
            ),
            Always(
                source="estimate_object_pose_2",
                target="move_to_grasp_pose_2",
            ),
            OnTargetPoseReached(
                source="move_to_grasp_pose_2",
                target="close_gripper_2",
                tolerance=list(self.target_tolerance),
            ),
            OnTimeLimit(
                source="close_gripper_2",
                target="move_to_plug_pose",
                max_steps=int(self.gripper_hold_steps),
            ),
            OnTimeLimit(
                source="move_to_plug_pose",
                target="move_to_scan_pose_3",
                max_steps=int(150),
            ),
            OnTimeLimit(
                source="move_to_scan_pose_3",
                target="move_to_scan_pose",
                max_steps=int(300),
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
