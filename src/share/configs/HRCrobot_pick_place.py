"""Small-motion HRCrobot pick-and-place MP-Net configuration.

Every Cartesian trajectory is resolved from the TCP pose observed when the
primitive is entered. No absolute robot-base pose is embedded in this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lerobot.envs import EnvConfig

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
    OpenLoopTrajectoryPrimitiveConfig,
    OpenLoopTrajectorySpec,
    PrimitiveEntryContext,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import ControlMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import OnSuccess, OnTimeLimit
from share.robots.HRCrobot import HRCrobotConfig


ROBOT_NAME = "arm"
_HARD_MAX_EXCURSION_M = 0.025
_HARD_MAX_ROTATION_RAD = math.radians(5.0)


class _DeltaTeleopValidationStub:
    """Validation-only descriptor; it is never connected or used at runtime."""

    action_features = {
        "x.vel": float,
        "y.vel": float,
        "z.vel": float,
        "rx.vel": float,
        "ry.vel": float,
        "rz.vel": float,
    }


def _validation_teleop_dict(robot_dict, teleop_dict):
    result = dict(teleop_dict)
    for robot_name in robot_dict:
        result.setdefault(robot_name, _DeltaTeleopValidationStub())
    return result


@dataclass
class _AutonomousHoldPrimitiveConfig(ManipulationPrimitiveConfig):
    """Hold the entry TCP pose while applying an optional gripper command."""

    def validate(self, robot_dict, teleop_dict):
        return super().validate(
            robot_dict, _validation_teleop_dict(robot_dict, teleop_dict)
        )

    def make(self, robot_dict, teleop_dict, cameras, device: str = "cpu"):
        env, env_processor, action_processor = super().make(
            robot_dict, teleop_dict, cameras, device
        )
        env.uses_autonomous_step = True
        return env, env_processor, action_processor

    def on_entry(
        self,
        env: ManipulationPrimitive,
        entry_context: PrimitiveEntryContext | None,
    ) -> None:
        start_pose, _ = self.resolve_targets(entry_context)
        env.set_target_pose(start_pose, info_key=self.target_pose_info_key)


@dataclass
class _AutonomousTrajectoryPrimitiveConfig(OpenLoopTrajectoryPrimitiveConfig):
    """Mark a repository-native open-loop trajectory as autonomous."""

    def validate(self, robot_dict, teleop_dict):
        return super().validate(
            robot_dict, _validation_teleop_dict(robot_dict, teleop_dict)
        )

    def make(self, robot_dict, teleop_dict, cameras, device: str = "cpu"):
        env, env_processor, action_processor = super().make(
            robot_dict, teleop_dict, cameras, device
        )
        env.uses_autonomous_step = True
        return env, env_processor, action_processor


def _fixed_frame() -> TaskFrame:
    return TaskFrame(
        origin=[0.0] * 6,
        target=[0.0] * 6,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )


def _processor(
    *, fps: int, gripper_static_pos: float | None = None
) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=float(fps),
        observation=ObservationConfig(
            add_joint_position_to_observation=False,
            add_joint_velocity_to_observation=False,
            add_ee_pos_to_observation=True,
            add_ee_velocity_to_observation=False,
            add_ee_wrench_to_observation=False,
        ),
        gripper=GripperConfig(enable=False, static_pos=gripper_static_pos),
    )


def _trajectory(
    *, delta: list[float], duration_s: float, processor: ManipulationPrimitiveProcessorConfig
) -> _AutonomousTrajectoryPrimitiveConfig:
    return _AutonomousTrajectoryPrimitiveConfig(
        task_frame=_fixed_frame(),
        trajectory=OpenLoopTrajectorySpec(
            delta=list(delta),
            frame="world",
            duration_s=float(duration_s),
        ),
        processor=processor,
    )


@EnvConfig.register_subclass("hrcrobot_pick_place")
@dataclass
class HRCrobotPickPlaceConfig(ManipulationPrimitiveNetConfig):
    """Five-stage, millimetre-scale pick-and-place experiment for real hardware."""

    robot_ip: str = "10.10.59.211"
    servo_frequency: float = 100.0
    fps: int = 30

    # [dx, dy, dz, droll, dpitch, dyaw], expressed in the world/base frame.
    move_down_delta: list[float] = field(
        default_factory=lambda: [0.0, 0.0, -0.005, 0.0, 0.0, 0.0]
    )
    lift_delta: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.010, 0.0, 0.0, 0.0]
    )
    move_to_place_delta: list[float] = field(
        default_factory=lambda: [0.010, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    move_down_duration_s: float = 3.0
    lift_duration_s: float = 3.0
    move_to_place_duration_s: float = 4.0
    gripper_hold_s: float = 1.0
    closed_gripper_position: float = 1.0
    open_gripper_position: float = 0.0
    gripper_min_command_interval_s: float = 0.5

    start_primitive: str = "move_down"
    reset_primitive: str = "move_down"

    def __post_init__(self) -> None:
        self._validate_safety_envelope()

        move_processor = _processor(fps=self.fps)
        close_processor = _processor(
            fps=self.fps, gripper_static_pos=self.closed_gripper_position
        )
        open_processor = _processor(
            fps=self.fps, gripper_static_pos=self.open_gripper_position
        )

        self.robot = {
            ROBOT_NAME: HRCrobotConfig(
                robot_ip=self.robot_ip,
                frequency=self.servo_frequency,
                use_gripper=True,
                gripper_min_command_interval_s=self.gripper_min_command_interval_s,
            )
        }
        self.teleop = {}
        self.cameras = {}

        self.primitives = {
            "move_down": _trajectory(
                delta=self.move_down_delta,
                duration_s=self.move_down_duration_s,
                processor=move_processor,
            ),
            "close_gripper": _AutonomousHoldPrimitiveConfig(
                task_frame=_fixed_frame(),
                processor=close_processor,
                notes="Hold the measured pick pose and close the gripper.",
            ),
            "lift": _trajectory(
                delta=self.lift_delta,
                duration_s=self.lift_duration_s,
                processor=move_processor,
            ),
            "move_to_place": _trajectory(
                delta=self.move_to_place_delta,
                duration_s=self.move_to_place_duration_s,
                processor=move_processor,
            ),
            "open_gripper": _AutonomousHoldPrimitiveConfig(
                task_frame=_fixed_frame(),
                processor=open_processor,
                notes="Hold the measured place pose and open the gripper.",
            ),
            "done": _AutonomousHoldPrimitiveConfig(
                task_frame=_fixed_frame(),
                processor=open_processor,
                notes="Terminal state; no additional motion is sent.",
                is_terminal=True,
            ),
        }

        hold_steps = max(1, int(round(self.gripper_hold_s * self.fps)))
        self.transitions = [
            OnSuccess(
                source="move_down",
                target="close_gripper",
                success_key="primitive_complete",
                additional_reward=0.0,
            ),
            OnTimeLimit(source="close_gripper", target="lift", max_steps=hold_steps),
            OnSuccess(
                source="lift",
                target="move_to_place",
                success_key="primitive_complete",
                additional_reward=0.0,
            ),
            OnSuccess(
                source="move_to_place",
                target="open_gripper",
                success_key="primitive_complete",
                additional_reward=0.0,
            ),
            OnTimeLimit(source="open_gripper", target="done", max_steps=hold_steps),
        ]
        super().__post_init__()

    @property
    def relative_waypoints(self) -> dict[str, list[float]]:
        """Return cumulative XYZ offsets from the startup TCP for review."""
        pick = self.move_down_delta[:3]
        lifted = [pick[i] + self.lift_delta[i] for i in range(3)]
        place = [lifted[i] + self.move_to_place_delta[i] for i in range(3)]
        return {
            "pick": [float(v) for v in pick],
            "lifted": [float(v) for v in lifted],
            "place": [float(v) for v in place],
        }

    def _validate_safety_envelope(self) -> None:
        if not 1 <= self.fps <= 100:
            raise ValueError("fps must be in [1, 100].")
        if not 1.0 <= self.servo_frequency <= 250.0:
            raise ValueError("servo_frequency must be in [1, 250] Hz.")
        for name, delta in (
            ("move_down_delta", self.move_down_delta),
            ("lift_delta", self.lift_delta),
            ("move_to_place_delta", self.move_to_place_delta),
        ):
            if len(delta) != 6 or not all(math.isfinite(float(v)) for v in delta):
                raise ValueError(f"{name} must contain six finite values.")
            if math.sqrt(sum(float(v) ** 2 for v in delta[3:])) > _HARD_MAX_ROTATION_RAD:
                raise ValueError(f"{name} rotation exceeds the hard 5 degree limit.")

        if self.move_down_delta[2] >= 0.0:
            raise ValueError("move_down_delta must move along world -Z.")
        if self.lift_delta[2] <= 0.0:
            raise ValueError("lift_delta must move along world +Z.")

        cumulative = [0.0, 0.0, 0.0]
        for name, delta in (
            ("pick", self.move_down_delta),
            ("lifted", self.lift_delta),
            ("place", self.move_to_place_delta),
        ):
            cumulative = [cumulative[i] + float(delta[i]) for i in range(3)]
            excursion = math.sqrt(sum(v * v for v in cumulative))
            if excursion > _HARD_MAX_EXCURSION_M:
                raise ValueError(
                    f"{name} waypoint is {excursion * 1000:.1f} mm from startup TCP; "
                    f"the hard limit is {_HARD_MAX_EXCURSION_M * 1000:.1f} mm."
                )

        for name, duration in (
            ("move_down_duration_s", self.move_down_duration_s),
            ("lift_duration_s", self.lift_duration_s),
            ("move_to_place_duration_s", self.move_to_place_duration_s),
        ):
            if float(duration) < 1.0:
                raise ValueError(f"{name} must be at least 1 second for this real-robot demo.")
        if self.gripper_hold_s < self.gripper_min_command_interval_s:
            raise ValueError(
                "gripper_hold_s must not be shorter than gripper_min_command_interval_s."
            )
