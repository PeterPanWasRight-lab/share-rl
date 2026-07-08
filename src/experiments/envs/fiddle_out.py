"""Experiment MP-Net config for teleop plus an EE-frame fiddle-out trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs import EnvConfig
from pynput import keyboard

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
    OpenLoopTrajectoryPrimitiveConfig,
    OpenLoopTrajectorySpec,
)
from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
)
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import OnSuccess, OnTargetPoseReached
from share.envs.utils import copy_per_robot
from share.robots.ur import URConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig
from share.utils.transformation_utils import compose_delta_pose, task_pose_to_world_pose, world_pose_to_task_pose


def _shared_processor() -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=30.0,
        observation=ObservationConfig(
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_ee_pos_to_observation=False,
            add_joint_position_to_observation=False,
        ),
        gripper=GripperConfig(
            enable=True,
            discretize=True,
            min_pos=0.5
        ),
        events=EventConfig(
            key_mapping={
                TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
                TeleopEvents.STOP_RECORDING: keyboard.Key.down,
            },
            foot_switch_mapping={
                (TeleopEvents.SUCCESS,): {"device": 20, "toggle": False},
            },
        ),
    )


@ManipulationPrimitiveConfig.register_subclass("ee_fiddle_out_circle")
@dataclass
class EEFiddleOutCirclePrimitiveConfig(OpenLoopTrajectoryPrimitiveConfig):
    """Open-loop EE-frame lift with a small XY circle while holding orientation fixed."""

    circle_radius_m: float | dict[str, float] = 0.005
    circle_frequency_hz: float | dict[str, float] = 2.0

    def validate(self, robot_dict, teleop_dict):
        super().validate(robot_dict, teleop_dict)
        robot_names = list(robot_dict)
        self.circle_radius_m = copy_per_robot(self.circle_radius_m, robot_names)
        self.circle_frequency_hz = copy_per_robot(self.circle_frequency_hz, robot_names)

        if self.trajectory.delta is None:
            raise ValueError("ee_fiddle_out_circle requires trajectory.delta.")

        for name in robot_names:
            radius = float(self.circle_radius_m[name])
            frequency_hz = float(self.circle_frequency_hz[name])
            delta = [float(v) for v in self.trajectory.delta[name]]
            frame = self.trajectory.frame[name]

            if radius < 0.0:
                raise ValueError("ee_fiddle_out_circle requires circle_radius_m >= 0.")
            if frequency_hz < 0.0:
                raise ValueError("ee_fiddle_out_circle requires circle_frequency_hz >= 0.")
            if frame != "ee":
                raise ValueError("ee_fiddle_out_circle requires trajectory.frame == 'ee'.")
            if any(abs(delta[axis]) > 1e-9 for axis in (0, 1, 3, 4, 5)):
                raise ValueError("ee_fiddle_out_circle only supports a pure local-Z trajectory.delta.")

    def target_pose_at(
        self,
        alpha: float,
        start_pose: dict[str, list[float]],
        goal_pose: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        alpha = max(0.0, float(alpha))
        pose_by_robot: dict[str, list[float]] = {}
        for name, frame in self.task_frame.items():
            start_world = task_pose_to_world_pose(start_pose[name], frame.origin)
            delta_z = float(self.trajectory.delta[name][2])
            radius = float(self.circle_radius_m[name])
            elapsed_s = alpha * float(self.trajectory.duration_s[name])
            z_alpha = min(1.0, elapsed_s / float(self.trajectory.duration_s[name]))
            theta = 2.0 * math.pi * float(self.circle_frequency_hz[name]) * elapsed_s
            scripted_world = compose_delta_pose(
                start_pose_world=start_world,
                delta=[
                    float(radius * (math.cos(theta) - 1.0)),
                    float(radius * math.sin(theta)),
                    float(z_alpha * delta_z),
                    0.0,
                    0.0,
                    0.0,
                ],
                frame_name="ee",
            )
            pose_by_robot[name] = world_pose_to_task_pose(scripted_world, frame.origin)

        return pose_by_robot


@EnvConfig.register_subclass("demo_ur3e_teleop_fiddle_out")
@dataclass
class DemoUR3eTeleopFiddleOutEnvConfig(ManipulationPrimitiveNetConfig):
    """Teleop in 6-DoF until success, then execute a small EE-frame fiddle-out."""

    fps: int = 30
    start_primitive: str = "reset"
    reset_primitive: str = "reset"

    fiddle_out_height_m: float = 0.02
    fiddle_out_radius_m: float = 0.002
    fiddle_out_circle_frequency_hz: float = 10.0
    fiddle_out_duration_s: float = 1.0
    teleop_wrench_limits: list[float] = field(default_factory=lambda: [25.0, 25.0, 25.0, 5.0, 5.0, 5.0])
    fiddle_out_wrench_limits: list[float] = field(default_factory=lambda: [6.0, 6.0, 6.0, 0.4, 0.4, 0.4])

    def __post_init__(self):
        processor = _shared_processor()
        processor.fps = float(self.fps)

        self.robot = URConfig(
            robot_ip="172.22.22.2",
            frequency=125,
            kp=[3000, 3000, 3000, 200, 200, 200],
            kd=[50, 50, 300, 5, 5, 5],
            soft_real_time=True,
            rt_core=3,
            use_gripper=True,
            wrench_limits=list(self.teleop_wrench_limits),
            compliance_reference_limit_enable=[True] * 6,
            verbose=True
        )
        self.teleop = SpaceMouseConfig()
        self.teleop.action_scale = [0.1, 0.1, 0.1, 1.0, 1.0, 1.0]
        #self.cameras = {
        #    "wrist": RealSenseCameraConfig(serial_number_or_name="352122271533")
        #}

        teleop_primitive = ManipulationPrimitiveConfig(
            task_frame=TaskFrame(
                target=[0.0] * 6,
                space=ControlSpace.TASK,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[PolicyMode.RELATIVE] * 6,
                controller_overrides={"wrench_limits": list(self.teleop_wrench_limits)},
                origin=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            processor=processor,
            notes="6-DoF relative task-space teleop primitive.",
        )
        fiddle_out_primitive = EEFiddleOutCirclePrimitiveConfig(
            task_frame=TaskFrame(
                target=[0.0] * 6,
                space=ControlSpace.TASK,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[None] * 6,
                controller_overrides={"wrench_limits": list(self.fiddle_out_wrench_limits)},
            ),
            trajectory=OpenLoopTrajectorySpec(
                delta=[0.0, 0.0, -float(self.fiddle_out_height_m), 0.0, 0.0, 0.0],
                frame="ee",
                duration_s=float(self.fiddle_out_duration_s),
            ),
            circle_radius_m=float(self.fiddle_out_radius_m),
            circle_frequency_hz=float(self.fiddle_out_circle_frequency_hz),
            processor=processor,
            notes=(
                "Open-loop fiddle-out: ramp up in the current EE frame, keep orientation fixed, "
                "then keep drawing a local XY circle while holding the final Z command until the "
                "primitive times out or the Z-based transition fires."
            ),
        )

        self.primitives = {
            "reset": teleop_primitive,
            "teleop": teleop_primitive,
            "fiddle_out": fiddle_out_primitive,
        }
        self.transitions = [
            OnSuccess(source="reset", target="teleop"),
            OnSuccess(source="teleop", target="reset"),
            OnTargetPoseReached(source="fiddle_out", target="teleop", axes=["z"]),
        ]

        super().__post_init__()


__all__ = [
    "EEFiddleOutCirclePrimitiveConfig",
    "DemoUR3eTeleopFiddleOutEnvConfig",
]
