from __future__ import annotations

from dataclasses import dataclass

from lerobot.envs import EnvConfig
from lerobot.policies.sac.configuration_sac import SACConfig

from share.cameras.mujoco_camera import MujocoCameraConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    ImagePreprocessingConfig,
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig,
    OpenLoopTrajectoryPrimitiveConfig,
    OpenLoopTrajectorySpec,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import (
    AllOf,
    OnObservationThreshold,
    OnSuccess,
    OnTimeLimit,
)
from share.robots.mujoco import MujocoRobotConfig
from share.teleoperators.delta_keyboard import KeyboardAxisBinding, KeyboardVelocityTeleopConfig
from share.teleoperators.mujoco import MujocoDeltaTeleopConfig


def _processor() -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=30.0,
        image_preprocessing=ImagePreprocessingConfig(resize_size=(128, 128)),
        gripper=GripperConfig(enable=True),
        observation=ObservationConfig(
            add_joint_position_to_observation=True,
            add_joint_velocity_to_observation=True,
            add_ee_pos_to_observation=True,
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
        ),
    )


def _scripted_frame() -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6,
        origin=[0.0] * 6,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )


def _insertion_frame(min_tcp_z: float) -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6,
        origin=[0.0] * 6,
        policy_mode=[PolicyMode.RELATIVE] * 6,
        control_mode=[ControlMode.POS] * 6,
        min_pose=[-1.2, -1.2, min_tcp_z, -3.14, -3.14, -3.14],
        max_pose=[1.2, 1.2, 1.8, 3.14, 3.14, 3.14],
    )


@EnvConfig.register_subclass("mujoco_ur5e_insertion")
@dataclass
class MujocoInsertionEnvConfig(ManipulationPrimitiveNetConfig):
    """Turnkey post-grasp peg-in-hole MP-Net for offline-to-online SAC."""

    fps: int = 30
    start_primitive: str = "insert"
    reset_primitive: str = "reset"
    viewer: bool = False
    viewer_camera: str | None = None
    episode_steps: int = 900
    min_tcp_z: float = 0.05
    success_insertion_depth: float = 0.07
    success_lateral_tolerance: float = 0.01
    success_axis_alignment: float = 0.98
    release_steps: int = 30
    teleop_mode: str = "none"
    online_steps: int = 20_000
    online_step_before_learning: int = 100
    policy_device: str = "cpu"
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051

    def __post_init__(self) -> None:
        if self.teleop_mode not in {"none", "keyboard"}:
            raise ValueError("teleop_mode must be 'none' or 'keyboard'.")
        if self.min_tcp_z < 0:
            raise ValueError("min_tcp_z must be non-negative.")
        robot_id = "mujoco-arm"
        policy = SACConfig(
            device=self.policy_device,
            storage_device="cpu",
            online_steps=self.online_steps,
            online_buffer_capacity=100_000,
            offline_buffer_capacity=50_000,
            online_step_before_learning=self.online_step_before_learning,
            use_torch_compile=False,
        )
        # SAC acts in normalized [-1, 1] space. These 7D statistics make its
        # postprocessor recover the same physical velocity units produced by
        # keyboard demonstrations before the command reaches the position servo.
        policy.dataset_stats["action"] = {
            "min": [-0.1, -0.1, -0.1, -0.5, -0.5, -0.5, 0.0],
            "max": [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 1.0],
        }
        policy.actor_learner_config.learner_host = self.learner_host
        policy.actor_learner_config.learner_port = self.learner_port
        processor = _processor()

        self.robot = MujocoRobotConfig(
            id=robot_id,
            control_dt=1.0 / self.fps,
            viewer=self.viewer,
            viewer_camera=self.viewer_camera,
        )
        if self.teleop_mode == "keyboard":
            # Match LeRobot's KeyboardEndEffectorTeleop: arrows move in XY,
            # left/right Shift move Z, and right/left Ctrl open/close.
            self.teleop = KeyboardVelocityTeleopConfig(
                id="mujoco-keyboard",
                x=KeyboardAxisBinding(pos_key="left", neg_key="right", scale=0.1),
                y=KeyboardAxisBinding(pos_key="down", neg_key="up", scale=0.1),
                z=KeyboardAxisBinding(pos_key="shift_r", neg_key="shift", scale=0.1),
                rx=KeyboardAxisBinding(enabled=False),
                ry=KeyboardAxisBinding(enabled=False),
                rz=KeyboardAxisBinding(enabled=False),
                gripper_enabled=True,
                gripper_open_key="ctrl_r",
                gripper_close_key="ctrl_l",
                initial_gripper_position=1.0,
            )
        else:
            self.teleop = MujocoDeltaTeleopConfig(id="mujoco-noop")
        self.cameras = {
            "front": MujocoCameraConfig(
                robot_id=robot_id, camera_name="front", width=320, height=240, fps=self.fps
            ),
            "wrist": MujocoCameraConfig(
                robot_id=robot_id, camera_name="wrist", width=320, height=240, fps=self.fps
            ),
        }

        self.primitives = {
            "reset": OpenLoopTrajectoryPrimitiveConfig(
                task_frame=_scripted_frame(),
                trajectory=OpenLoopTrajectorySpec(
                    delta=[0.0] * 6,
                    frame="world",
                    duration_s=0.05,
                ),
                processor=processor,
                notes="Reset MuJoCo physics, then hand control to insertion.",
            ),
            "insert": ManipulationPrimitiveConfig(
                task_frame=_insertion_frame(self.min_tcp_z),
                processor=processor,
                policy=policy,
                notes="Six-axis relative Cartesian insertion with force/torque observations.",
            ),
            "release": MoveDeltaPrimitiveConfig(
                task_frame=_scripted_frame(),
                processor=processor,
                delta=[0.0] * 6,
                delta_frame="world",
                notes="Hold tool pose while opening the physical 2F-85 gripper.",
            ),
            "done": MoveDeltaPrimitiveConfig(
                task_frame=_scripted_frame(),
                processor=processor,
                delta=[0.0] * 6,
                delta_frame="world",
                is_terminal=True,
                notes="Terminal hold at the release pose before reset.",
            ),
        }
        self.transitions = [
            OnSuccess(source="reset", target="insert", success_key="primitive_complete"),
            AllOf(
                source="insert",
                target="release",
                additional_reward=1.0,
                reason="peg_inserted",
                conditions=[
                    OnObservationThreshold(
                        source="insert",
                        target="release",
                        obs_key="main.insertion.depth",
                        threshold=self.success_insertion_depth,
                        operator="ge",
                    ),
                    OnObservationThreshold(
                        source="insert",
                        target="release",
                        obs_key="main.insertion.lateral_error",
                        threshold=self.success_lateral_tolerance,
                        operator="le",
                    ),
                    OnObservationThreshold(
                        source="insert",
                        target="release",
                        obs_key="main.insertion.axis_alignment",
                        threshold=self.success_axis_alignment,
                        operator="ge",
                    ),
                ],
            ),
            OnTimeLimit(source="insert", target="release", max_steps=self.episode_steps, step_key="primitive_step"),
            OnTimeLimit(
                source="release",
                target="done",
                max_steps=self.release_steps,
                step_key="primitive_step",
                reason="gripper_released",
            ),
        ]
        super().__post_init__()


__all__ = ["MujocoInsertionEnvConfig"]
