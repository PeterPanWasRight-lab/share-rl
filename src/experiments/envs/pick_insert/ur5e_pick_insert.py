"""UR5e MuJoCo pick-and-insert MP-Net config.

Runs the deterministic pick-and-insert state machine on the dedicated
``pick_insert`` MuJoCo scene (peg resting free at pose A, fixture hole at B):

    hole centre   B = (-0.134, 0.492, 0.08)   (vertical through-hole)
    peg pose      A = (-0.25,  0.30,  0.11)   (0.12 m bar standing on the table)
    straight-down TCP orientation = euler (pi, 0, pi/2)

The peg is a 0.12 m bar, so grasping its upper half means the TCP pinch sits
0.05 m above the peg centre; the same offset applies when inserting into the
hole. Waypoints are loaded from poses.json and reference this geometry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.envs import EnvConfig

from share.cameras.mujoco_camera import MujocoCameraConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
)
from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    TaskFrame,
)
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import (
    OnSuccess,
    OnTargetPoseReached,
    OnTimeLimit,
)
from share.robots.mujoco import MujocoRobotConfig
from share.teleoperators.mujoco import MujocoDeltaTeleopConfig
from share.utils.constants import DEFAULT_ROBOT_NAME

# Straight-down TCP orientation, matching the MuJoCo "home" keyframe (xyz euler).
_DOWN_RPY = [3.141592653589793, 0.0, 1.5707963267948966]

# World-frame reference positions taken from the pick_insert MuJoCo scene.
_PEG_POSE_A = (-0.25, 0.30, 0.11)
_HOLE_POSE_B = (-0.134, 0.492, 0.08)


def _pose(x: float, y: float, z: float) -> list[float]:
    """Build a 6D TCP target at (x, y, z) with a straight-down orientation."""
    return [x, y, z, *_DOWN_RPY]


def _processor(gripper_static_pos: float) -> ManipulationPrimitiveProcessorConfig:
    """Shared processor; the gripper position is scripted per primitive."""
    return ManipulationPrimitiveProcessorConfig(
        fps=30.0,
        observation=ObservationConfig(
            add_ee_pos_to_observation=True,
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_joint_position_to_observation=True,
        ),
        gripper=GripperConfig(enable=False, static_pos=gripper_static_pos),
    )


def _move_primitive(
    target: list[float],
    processor: ManipulationPrimitiveProcessorConfig,
    notes: str,
    *,
    terminal: bool = False,
) -> ManipulationPrimitiveConfig:
    """Scripted absolute-pose primitive resolved by the MuJoCo position servo."""
    return ManipulationPrimitiveConfig(
        notes=notes,
        processor=processor,
        is_terminal=terminal,
        task_frame=TaskFrame(
            target=list(target),
            policy_mode=[None] * 6,
            control_mode=[ControlMode.POS] * 6,
        ),
    )


@EnvConfig.register_subclass("ur5e_pick_insert")
@dataclass
class UR5ePickInsertEnvConfig(ManipulationPrimitiveNetConfig):
    """MuJoCo UR5e pick-and-insert: pick peg at A, insert into hole at B."""

    fps: int = 30
    start_primitive: str = "move_above_A"
    reset_primitive: str = "move_above_A"

    # Path to poses.json; when empty, waypoints derive from the reference geometry.
    poses_file: str = ""

    viewer: bool = False
    viewer_camera: str | None = None
    randomize_fixture_xy: float = 0.0

    target_tolerance: float = 0.01
    gripper_hold_steps: int = 30
    settle_hold_steps: int = 30
    release_hold_steps: int = 45
    insert_hold_steps: int = 60

    def __post_init__(self) -> None:
        poses = self._load_poses()

        open_proc = _processor(0.0)    # gripper open
        closed_proc = _processor(1.0)  # gripper closed

        self.robot = MujocoRobotConfig(
            id="mujoco-arm",
            scene_builder="pick_insert",
            control_dt=1.0 / self.fps,
            viewer=self.viewer,
            viewer_camera=self.viewer_camera,
            randomize_fixture_xy=self.randomize_fixture_xy,
        )
        self.teleop = MujocoDeltaTeleopConfig(id="mujoco-noop")
        self.cameras = {
            "front": MujocoCameraConfig(
                robot_id="mujoco-arm",
                camera_name="front",
                width=64,
                height=64,
                fps=self.fps,
            ),
        }

        self.primitives = {
            "move_above_A": _move_primitive(
                poses["move_above_A"], open_proc, "Move above the peg at A."
            ),
            "open_gripper_A": _move_primitive(
                poses["move_above_A"], open_proc, "Open the gripper above the peg."
            ),
            "descend_A": _move_primitive(
                poses["descend_A"], open_proc, "Descend to grasp the peg at A."
            ),
            "close_gripper": _move_primitive(
                poses["descend_A"], closed_proc, "Close the gripper around the peg."
            ),
            "lift_A": _move_primitive(
                poses["lift_A"], closed_proc, "Lift the grasped peg off the table."
            ),
            "move_above_B": _move_primitive(
                poses["move_above_B"], closed_proc, "Move above the hole at B."
            ),
            "settle_B": _move_primitive(
                poses["move_above_B"], closed_proc, "Hold above the hole to let the peg settle."
            ),
            "pre_insert_B": _move_primitive(
                poses["pre_insert_B"], closed_proc, "Align the peg above the hole."
            ),
            "insert_B": _move_primitive(
                poses["insert_B"], closed_proc, "Insert the peg into the hole."
            ),
            "open_gripper_B": _move_primitive(
                poses["insert_B"], open_proc, "Release the peg inside the hole."
            ),
            "retract_B": _move_primitive(
                poses["retract_B"], open_proc, "Retract the empty gripper from B."
            ),
            "done": _move_primitive(
                poses["retract_B"], open_proc, "Terminal hold.", terminal=True
            ),
        }
        self.transitions = [
            OnTargetPoseReached(
                source="move_above_A",
                target="open_gripper_A",
                robot_name=DEFAULT_ROBOT_NAME,
                axes=["x", "y", "z"],
                tolerance=self.target_tolerance,
            ),
            OnTimeLimit(source="open_gripper_A", target="descend_A", max_steps=self.gripper_hold_steps),
            OnTargetPoseReached(
                source="descend_A",
                target="close_gripper",
                robot_name=DEFAULT_ROBOT_NAME,
                axes=["z"],
                tolerance=self.target_tolerance,
            ),
            OnTimeLimit(source="close_gripper", target="lift_A", max_steps=self.gripper_hold_steps),
            OnTargetPoseReached(
                source="lift_A",
                target="move_above_B",
                robot_name=DEFAULT_ROBOT_NAME,
                axes=["x", "y", "z"],
                tolerance=self.target_tolerance,
            ),
            OnTargetPoseReached(
                source="move_above_B",
                target="settle_B",
                robot_name=DEFAULT_ROBOT_NAME,
                axes=["x", "y", "z"],
                tolerance=self.target_tolerance,
            ),
            OnTimeLimit(source="settle_B", target="pre_insert_B", max_steps=self.settle_hold_steps),
            OnTargetPoseReached(
                source="pre_insert_B",
                target="insert_B",
                robot_name=DEFAULT_ROBOT_NAME,
                axes=["z"],
                tolerance=self.target_tolerance,
            ),
            OnTimeLimit(source="insert_B", target="open_gripper_B", max_steps=self.insert_hold_steps),
            OnTimeLimit(source="open_gripper_B", target="retract_B", max_steps=self.release_hold_steps),
            OnTargetPoseReached(
                source="retract_B",
                target="done",
                robot_name=DEFAULT_ROBOT_NAME,
                axes=["x", "y", "z"],
                tolerance=self.target_tolerance,
            ),
            OnSuccess(source="done", target="move_above_A", success_key="primitive_complete"),
        ]

        super().__post_init__()

    def _load_poses(self) -> dict[str, list[float]]:
        """Load waypoints from poses.json, or build them from the reference geometry."""
        if self.poses_file:
            data = json.loads(Path(self.poses_file).read_text(encoding="utf-8"))
            return {
                key: [float(v) for v in value]
                for key, value in data.items()
                if isinstance(value, list) and len(value) == 6
            }

        peg_x, peg_y, peg_z = _PEG_POSE_A
        hole_x, hole_y, hole_z = _HOLE_POSE_B
        grasp_z = peg_z + 0.05   # TCP pinch sits 0.05 m above the peg centre
        above = 0.28             # safe lateral-move height; peg tip (TCP-0.11) clears the hole top (0.10)
        pre = hole_z + 0.11      # just above the hole mouth (TCP at 0.19, peg tip at 0.08)
        insert_z = hole_z + 0.09  # shallow insertion: peg tip reaches ~0.06 (1/3 into the hole)
        # Compensate the peg's ~6 mm tip drift toward +y during descent by biasing
        # the insert targets slightly toward the robot base (-x, -y).
        bias_x, bias_y = -0.001, -0.006
        return {
            "move_above_A": _pose(peg_x, peg_y, above),
            "descend_A": _pose(peg_x, peg_y, grasp_z),
            "lift_A": _pose(peg_x, peg_y, above),
            "move_above_B": _pose(hole_x, hole_y, above),
            "pre_insert_B": _pose(hole_x + bias_x, hole_y + bias_y, pre),
            "insert_B": _pose(hole_x + bias_x, hole_y + bias_y, insert_z),
            "retract_B": _pose(hole_x, hole_y, above),
        }


__all__ = ["UR5ePickInsertEnvConfig"]
