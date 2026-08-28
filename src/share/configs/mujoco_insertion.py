from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.envs import EnvConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from share.policies.sac_dagger import SACDaggerBCConfig

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
    OnFailure,
    OnObservationThreshold,
    OnSuccess,
    OnTimeLimit,
)
from share.robots.mujoco import MujocoRobotConfig
from share.rl.force_backoff import ForceBackoffConfig
from share.teleoperators.delta_keyboard import (
    KeyboardAxisBinding,
    KeyboardEventBinding,
    KeyboardVelocityTeleopConfig,
)
from share.teleoperators.mujoco import MujocoDeltaTeleopConfig
from share.teleoperators.utils import TeleopEvents


def _processor(
    image_size: int,
    *,
    stack_frames: int = 1,
    gripper_static_pos: float | None = None,
) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=30.0,
        image_preprocessing=ImagePreprocessingConfig(resize_size=(image_size, image_size)),
        # The insertion policy learns XYZ only. Gripper commands are supplied by
        # the primitive state machine instead of occupying a policy action axis.
        gripper=GripperConfig(enable=False, static_pos=gripper_static_pos),
        observation=ObservationConfig(
            add_joint_position_to_observation=True,
            add_joint_velocity_to_observation=True,
            add_ee_pos_to_observation=True,
            stack_frames=stack_frames,
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
        # Learn Cartesian translation only. The fixed POS rotation axes are
        # resolved from the entry pose by MoveDeltaPrimitiveConfig.on_entry().
        policy_mode=[PolicyMode.RELATIVE] * 3 + [None] * 3,
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
    viewer_wrench_overlay: bool = True
    viewer_front_camera_overlay: bool = True
    viewer_wrist_camera_overlay: bool = True
    viewer_wrench_plot: bool = True
    episode_steps: int = 900
    min_tcp_z: float = -1.2
    success_insertion_depth: float = 0.07
    success_lateral_tolerance: float = 0.002
    success_axis_alignment: float = 0.98
    release_steps: int = 30
    gripper_min_command_interval_s: float = 0.5
    teleop_mode: str = "none"
    online_steps: int = 20_000
    policy_pretrained_path: str | None = None
    policy_training_mode: str = "sac"
    policy_update_freq: int = 1
    policy_actor_update_after: int = 0
    policy_sac_bc_loss_weight: float = 0.0
    policy_freeze_shared_encoder_during_sac: bool = False
    policy_actor_lr: float = 3e-4
    policy_frame_stack: int = 1
    policy_vision_encoder_name: str | None = "helper2424/resnet10"
    online_step_before_learning: int = 100
    policy_device: str = "cpu"
    policy_image_size: int = 64
    state_only_policy: bool = False
    domain_randomization: bool = False
    fixture_xy_randomization_m: float = 0.002
    policy_excluded_image_keys: tuple[str, ...] = field(init=False, default=())
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    force_backoff: ForceBackoffConfig = field(
        default_factory=lambda: ForceBackoffConfig(
            enabled=True,
            robot_name="main",
            force_thresholds_n=[20.0, 20.0, 20.0],
            # MuJoCo's wrist sensor reports the reaction with the opposite
            # motion convention required by the backoff command.
            wrench_to_backoff_sign=[-1.0, -1.0, -1.0],
        )
    )

    def __post_init__(self) -> None:
        if self.teleop_mode not in {"none", "keyboard"}:
            raise ValueError("teleop_mode must be 'none' or 'keyboard'.")
        if self.min_tcp_z >= 1.8:
            raise ValueError("min_tcp_z must be below the configured maximum TCP z.")
        if self.gripper_min_command_interval_s < 0:
            raise ValueError("gripper_min_command_interval_s must be non-negative.")
        if self.policy_image_size < 32:
            raise ValueError("policy_image_size must be at least 32 pixels.")
        if self.policy_frame_stack < 1:
            raise ValueError("policy_frame_stack must be at least 1.")
        if self.policy_update_freq < 1:
            raise ValueError("policy_update_freq must be positive.")
        if self.policy_actor_update_after < 0:
            raise ValueError("policy_actor_update_after must be non-negative.")
        if self.policy_sac_bc_loss_weight < 0:
            raise ValueError("policy_sac_bc_loss_weight must be non-negative.")
        if self.policy_actor_lr <= 0:
            raise ValueError("policy_actor_lr must be positive.")
        if self.fixture_xy_randomization_m < 0:
            raise ValueError("fixture_xy_randomization_m must be non-negative.")
        robot_id = "mujoco-arm"
        policy = SACDaggerBCConfig(
            device=self.policy_device,
            storage_device="cpu",
            pretrained_path=self.policy_pretrained_path,
            training_mode=self.policy_training_mode,
            online_steps=self.online_steps,
            online_buffer_capacity=100_000,
            offline_buffer_capacity=50_000,
            online_step_before_learning=self.online_step_before_learning,
            policy_update_freq=self.policy_update_freq,
            actor_update_after=self.policy_actor_update_after,
            sac_bc_loss_weight=self.policy_sac_bc_loss_weight,
            freeze_shared_encoder_during_sac=self.policy_freeze_shared_encoder_during_sac,
            actor_lr=self.policy_actor_lr,
            use_torch_compile=False,
        )
        # The PyTorch ResNet10 port does not normalize internally. Keep the
        # frozen ImageNet backbone on the distribution used by HIL-SERL.
        policy.normalization_mapping[FeatureType.VISUAL] = NormalizationMode.MEAN_STD
        imagenet_stats = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        policy.dataset_stats["observation.images.front"] = imagenet_stats.copy()
        policy.dataset_stats["observation.images.wrist"] = imagenet_stats.copy()
        policy.vision_encoder_name = self.policy_vision_encoder_name
        # SHaRe freezes its pretrained ResNet10 visual representation. If the
        # encoder is explicitly disabled, retain the trainable random CNN path.
        policy.freeze_vision_encoder = self.policy_vision_encoder_name is not None
        if self.policy_vision_encoder_name is not None:
            policy.pretrained_vision_input_size = 128
            policy.proprio_latent_dim = 64
            policy.bc_random_crop_padding = 4
        # SAC acts in normalized [-1, 1] space. These XYZ statistics make its
        # postprocessor recover the same physical velocity units produced by
        # keyboard demonstrations before the command reaches the position servo.
        policy.dataset_stats["action"] = {
            "min": [-0.1, -0.1, -0.1],
            "max": [0.1, 0.1, 0.1],
        }
        policy.actor_learner_config.learner_host = self.learner_host
        policy.actor_learner_config.learner_port = self.learner_port
        closed_gripper_processor = _processor(
            self.policy_image_size,
            stack_frames=self.policy_frame_stack,
            gripper_static_pos=1.0,
        )
        open_gripper_processor = _processor(
            self.policy_image_size,
            stack_frames=self.policy_frame_stack,
            gripper_static_pos=0.0,
        )
        hold_gripper_processor = _processor(
            self.policy_image_size,
            stack_frames=self.policy_frame_stack,
        )

        self.robot = MujocoRobotConfig(
            id=robot_id,
            control_dt=1.0 / self.fps,
            viewer=self.viewer,
            viewer_camera=self.viewer_camera,
            viewer_wrench_overlay=self.viewer_wrench_overlay,
            viewer_front_camera_overlay=self.viewer_front_camera_overlay,
            viewer_wrist_camera_overlay=self.viewer_wrist_camera_overlay,
            viewer_wrench_plot=self.viewer_wrench_plot,
            gripper_min_command_interval_s=self.gripper_min_command_interval_s,
            randomize_fixture_xy=self.fixture_xy_randomization_m,
            randomize_fixture_z=0.001 if self.domain_randomization else 0.0,
            randomize_fixture_yaw_deg=3.0 if self.domain_randomization else 0.0,
            randomize_camera_position_m=0.005 if self.domain_randomization else 0.0,
            randomize_camera_rotation_deg=1.5 if self.domain_randomization else 0.0,
            randomize_camera_fovy_deg=2.0 if self.domain_randomization else 0.0,
            randomize_light_intensity_fraction=0.20 if self.domain_randomization else 0.0,
            randomize_object_color_fraction=0.10 if self.domain_randomization else 0.0,
            randomize_contact_friction_fraction=0.15 if self.domain_randomization else 0.0,
            randomize_peg_mass_fraction=0.15 if self.domain_randomization else 0.0,
        )
        if self.teleop_mode == "keyboard":
            # Arrows move in XY, left/right Shift move Z, and the adjacent
            # comma/period keys close/open the gripper.
            self.teleop = KeyboardVelocityTeleopConfig(
                id="mujoco-keyboard",
                x=KeyboardAxisBinding(pos_key="left", neg_key="right", scale=0.1),
                y=KeyboardAxisBinding(pos_key="down", neg_key="up", scale=0.1),
                z=KeyboardAxisBinding(pos_key="shift_r", neg_key="shift", scale=0.1),
                rx=KeyboardAxisBinding(enabled=False),
                ry=KeyboardAxisBinding(enabled=False),
                rz=KeyboardAxisBinding(enabled=False),
                gripper_enabled=False,
                gripper_open_key=".",
                gripper_close_key=",",
                initial_gripper_position=1.0,
                event_bindings={
                    TeleopEvents.FAILURE.value: KeyboardEventBinding(key="/"),
                    TeleopEvents.SUCCESS.value: KeyboardEventBinding(key="enter"),
                    TeleopEvents.STOP_RECORDING.value: KeyboardEventBinding(key="esc"),
                },
                escape_disconnects=False,
            )
        else:
            self.teleop = MujocoDeltaTeleopConfig(id="mujoco-noop")
        self.cameras = {
            "front": MujocoCameraConfig(
                robot_id=robot_id,
                camera_name="front",
                width=self.policy_image_size,
                height=self.policy_image_size,
                fps=self.fps,
            ),
            "wrist": MujocoCameraConfig(
                robot_id=robot_id,
                camera_name="wrist",
                width=self.policy_image_size,
                height=self.policy_image_size,
                fps=self.fps,
            ),
        }
        if self.state_only_policy:
            self.policy_excluded_image_keys = (
                "observation.images.front",
                "observation.images.wrist",
            )

        self.primitives = {
            "reset": OpenLoopTrajectoryPrimitiveConfig(
                task_frame=_scripted_frame(),
                trajectory=OpenLoopTrajectorySpec(
                    delta=[0.0] * 6,
                    frame="world",
                    duration_s=0.05,
                ),
                processor=closed_gripper_processor,
                notes="Reset MuJoCo physics, then hand control to insertion.",
            ),
            "insert": MoveDeltaPrimitiveConfig(
                task_frame=_insertion_frame(self.min_tcp_z),
                processor=closed_gripper_processor,
                policy=policy,
                delta=[0.0] * 6,
                delta_frame="world",
                notes="XYZ-only relative Cartesian insertion with entry orientation held fixed.",
            ),
            "release": MoveDeltaPrimitiveConfig(
                task_frame=_scripted_frame(),
                processor=open_gripper_processor,
                delta=[0.0] * 6,
                delta_frame="world",
                notes="Hold tool pose while opening the physical 2F-85 gripper.",
            ),
            "done": MoveDeltaPrimitiveConfig(
                task_frame=_scripted_frame(),
                processor=hold_gripper_processor,
                delta=[0.0] * 6,
                delta_frame="world",
                is_terminal=True,
                notes="Terminal hold at the release pose before reset.",
            ),
        }
        self.transitions = [
            OnSuccess(source="reset", target="insert", success_key="primitive_complete"),
            OnFailure(
                source="insert",
                target="done",
                failure_key=TeleopEvents.FAILURE.value,
                reason="manual_failure",
            ),
            OnSuccess(
                source="insert",
                target="release",
                success_key=TeleopEvents.SUCCESS.value,
                reason="manual_success",
            ),
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
            OnTimeLimit(
                source="insert",
                target="done",
                max_steps=self.episode_steps,
                step_key="primitive_step",
                reason="insertion_timeout",
            ),
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
