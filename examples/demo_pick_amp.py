"""Standalone adaptive pick primitive on the MuJoCo ``pick_insert`` scene.

This example intentionally does not import or modify the existing pick-and-insert
state machine.  It defines one trainable AMP with a HIL-SERL-style action surface:

    action = [delta_x, delta_y, delta_z, gripper]

The sparse success signal fires when the free workpiece has been lifted by a
configurable height.  By default an oracle controller exercises the same four
action dimensions, which makes the environment and reward contract testable
before collecting demonstrations or attaching an Actor/Learner.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lerobot.cameras import Camera
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.envs import EnvConfig
from lerobot.processor import TransitionKey
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator

from share.cameras.mujoco_camera import MujocoCameraConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    GripperConfig,
    ImagePreprocessingConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig,
    PrimitiveEntryContext,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import (
    ManipulationPrimitive,
)
from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    PolicyMode,
    TaskFrame,
)
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)
from share.envs.manipulation_primitive_net.transitions import (
    OnObservationThreshold,
    OnTimeLimit,
)
from share.policies.sac_dagger import SACDaggerBCConfig
from share.robots.mujoco import MujocoRobotConfig
from share.teleoperators.mujoco import MujocoDeltaTeleopConfig
from share.utils.constants import DEFAULT_ROBOT_NAME


_PEG_XY = np.asarray([-0.25, 0.30], dtype=np.float64)
_DOWN_RPY = [np.pi, 0.0, np.pi / 2.0]
_MAX_TRANSLATION_SPEED_M_S = 0.10
_PICK_LIFT_OBSERVATION = "pick.object_lift"


@EnvConfig.register_subclass("mujoco_pick_amp")
@dataclass
class PickAMPNetConfig(ManipulationPrimitiveNetConfig):
    """Serializable top-level config for this standalone demo only."""


class PickAMPEnv(ManipulationPrimitive):
    """Demo-local primitive env that publishes sparse pick task state."""

    def __init__(
        self,
        *args: Any,
        success_lift_m: float,
        peg_xy_randomization_m: float,
        **kwargs: Any,
    ) -> None:
        self.success_lift_m = float(success_lift_m)
        self.peg_xy_randomization_m = float(peg_xy_randomization_m)
        self._randomize_peg_on_next_observation = False
        super().__init__(*args, **kwargs)

    def reset(self, *args: Any, **kwargs: Any):
        self._randomize_peg_on_next_observation = True
        return super().reset(*args, **kwargs)

    def reset_runtime_state(self) -> None:
        super().reset_runtime_state()
        self._initial_object_height_m: float | None = None

    def _get_observation(self) -> dict[str, Any]:
        if self._randomize_peg_on_next_observation:
            self._randomize_free_peg_xy()
            self._randomize_peg_on_next_observation = False
        observation = super()._get_observation()
        object_height_m = self._object_height_m()
        if self._initial_object_height_m is None:
            self._initial_object_height_m = object_height_m
        observation["pick.object_height"] = object_height_m
        observation[_PICK_LIFT_OBSERVATION] = (
            object_height_m - self._initial_object_height_m
        )
        return observation

    def _object_height_m(self) -> float:
        robot = self.robot_dict[DEFAULT_ROBOT_NAME]
        # Simulator truth is deliberately local to this demo. The learned policy
        # does not receive it in observation.state; it is used only for reward.
        body_id = int(robot._peg_body_id)
        return float(robot._data.xpos[body_id, 2])

    def _randomize_free_peg_xy(self) -> None:
        if self.peg_xy_randomization_m <= 0.0:
            return
        robot = self.robot_dict[DEFAULT_ROBOT_NAME]
        offset = self.np_random.uniform(
            -self.peg_xy_randomization_m,
            self.peg_xy_randomization_m,
            size=2,
        )
        peg_qpos = robot._data.qpos[
            robot._peg_qpos_adr : robot._peg_qpos_adr + 7
        ].copy()
        peg_qpos[:2] = _PEG_XY + offset
        robot._data.qpos[
            robot._peg_qpos_adr : robot._peg_qpos_adr + 7
        ] = peg_qpos
        robot._data.qvel[robot._peg_dof_adr : robot._peg_dof_adr + 6] = 0.0
        robot._mujoco.mj_forward(robot._model, robot._data)

    def sync_relative_target_to_tcp(self) -> None:
        """Start demo-local relative axes at the current simulated TCP pose."""
        for name, frame in self.task_frame.items():
            robot = self.robot_dict[name]
            current_task_pose = robot._world_to_task_pose(robot._tcp_world_pose())
            if robot._virtual_target_task is None:
                robot._virtual_target_task = current_task_pose.copy()
            for axis, policy_mode in enumerate(frame.policy_mode):
                if policy_mode == PolicyMode.RELATIVE:
                    robot._virtual_target_task[axis] = current_task_pose[axis]


@ManipulationPrimitiveConfig.register_subclass("pick_amp")
@dataclass
class PickAMPConfig(MoveDeltaPrimitiveConfig):
    """Dedicated config so pick-specific runtime state stays in its env."""

    success_lift_m: float = 0.08
    peg_xy_randomization_m: float = 0.025

    def on_entry(
        self,
        env: ManipulationPrimitive,
        entry_context: PrimitiveEntryContext | None,
    ) -> None:
        super().on_entry(env, entry_context)
        if not isinstance(env, PickAMPEnv):
            raise TypeError("PickAMPConfig requires PickAMPEnv")
        # Keep this MuJoCo initialization detail inside the standalone demo.
        env.sync_relative_target_to_tcp()

    def make(
        self,
        robot_dict: dict[str, Robot],
        teleop_dict: dict[str, Teleoperator],
        cameras: dict[str, Camera],
        device: str = "cpu",
    ):
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)
        display_cameras = bool(
            self.processor.image_preprocessing
            and self.processor.image_preprocessing.display_cameras
        )
        env = PickAMPEnv(
            task_frame=self.task_frame,
            robot_dict=robot_dict,
            cameras=cameras,
            display_cameras=display_cameras,
            success_lift_m=self.success_lift_m,
            peg_xy_randomization_m=self.peg_xy_randomization_m,
        )
        return (
            env,
            self.make_env_processor(device),
            self.make_action_processor(robot_dict, teleop_dict, device),
        )


def _pick_processor(image_size: int) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=30.0,
        image_preprocessing=ImagePreprocessingConfig(
            resize_size=(image_size, image_size)
        ),
        observation=ObservationConfig(
            add_joint_position_to_observation=True,
            add_joint_velocity_to_observation=True,
            add_ee_pos_to_observation=True,
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
        ),
        gripper=GripperConfig(
            enable=True,
            discretize=True,
            threshold=0.5,
            min_pos=0.0,
            max_pos=1.0,
        ),
    )


def _pick_policy(device: str) -> SACDaggerBCConfig:
    policy = SACDaggerBCConfig(
        device=device,
        storage_device="cpu",
        training_mode="sac",
        online_steps=20_000,
        online_buffer_capacity=100_000,
        offline_buffer_capacity=50_000,
        online_step_before_learning=100,
        use_torch_compile=False,
    )
    policy.normalization_mapping[FeatureType.VISUAL] = NormalizationMode.MEAN_STD
    imagenet_stats = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
    policy.dataset_stats["observation.images.front"] = imagenet_stats.copy()
    policy.dataset_stats["observation.images.wrist"] = imagenet_stats.copy()
    policy.dataset_stats["action"] = {
        "min": [-_MAX_TRANSLATION_SPEED_M_S] * 3 + [0.0],
        "max": [_MAX_TRANSLATION_SPEED_M_S] * 3 + [1.0],
    }
    policy.vision_encoder_name = "helper2424/resnet10"
    policy.freeze_vision_encoder = True
    policy.pretrained_vision_input_size = 128
    policy.proprio_latent_dim = 64
    policy.bc_random_crop_padding = 4
    return policy


def build_pick_amp_config(
    *,
    viewer: bool = False,
    device: str = "cpu",
    image_size: int = 64,
    success_lift_m: float = 0.08,
    peg_xy_randomization_m: float = 0.025,
    episode_steps: int = 500,
) -> PickAMPNetConfig:
    """Build an isolated one-AMP pick task on the existing MuJoCo scene."""
    processor = _pick_processor(image_size)
    pick = PickAMPConfig(
        task_description="Pick the red workpiece and lift it clear of the table.",
        notes="4D adaptive pick: relative XYZ plus binary gripper command.",
        success_lift_m=success_lift_m,
        peg_xy_randomization_m=peg_xy_randomization_m,
        delta=[0.0] * 6,
        policy=_pick_policy(device),
        processor=processor,
        task_frame=TaskFrame(
            target=[0.0, 0.0, 0.0, *_DOWN_RPY],
            policy_mode=[PolicyMode.RELATIVE] * 3 + [None] * 3,
            control_mode=[ControlMode.POS] * 6,
            min_pose=[-0.45, 0.10, 0.12, -np.pi, -np.pi, -np.pi],
            max_pose=[-0.05, 0.55, 0.45, np.pi, np.pi, np.pi],
        ),
    )
    terminal_frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )
    reset_processor = copy.deepcopy(processor)
    reset_processor.gripper = GripperConfig(enable=False, static_pos=0.0)
    reset = MoveDeltaPrimitiveConfig(
        task_frame=copy.deepcopy(terminal_frame),
        processor=reset_processor,
        delta=[0.0] * 6,
        notes="One-step open-gripper reset before the adaptive pick.",
    )
    success = MoveDeltaPrimitiveConfig(
        task_frame=copy.deepcopy(terminal_frame),
        processor=copy.deepcopy(processor),
        delta=[0.0] * 6,
        is_terminal=True,
        notes="Successful pick; hold the current TCP pose.",
    )
    timeout = MoveDeltaPrimitiveConfig(
        task_frame=copy.deepcopy(terminal_frame),
        processor=copy.deepcopy(processor),
        delta=[0.0] * 6,
        is_terminal=True,
        notes="Pick timed out; hold the current TCP pose.",
    )

    return PickAMPNetConfig(
        fps=30,
        start_primitive="pick",
        reset_primitive="reset",
        robot=MujocoRobotConfig(
            id="mujoco-arm",
            scene_builder="pick_insert",
            control_dt=1.0 / 30.0,
            viewer=viewer,
            randomize_fixture_xy=0.0,
        ),
        teleop=MujocoDeltaTeleopConfig(id="mujoco-noop"),
        cameras={
            "front": MujocoCameraConfig(
                robot_id="mujoco-arm",
                camera_name="front",
                width=image_size,
                height=image_size,
                fps=30,
            ),
            "wrist": MujocoCameraConfig(
                robot_id="mujoco-arm",
                camera_name="wrist",
                width=image_size,
                height=image_size,
                fps=30,
            ),
        },
        primitives={
            "reset": reset,
            "pick": pick,
            "success": success,
            "timeout": timeout,
        },
        transitions=[
            OnTimeLimit(
                source="reset",
                target="pick",
                max_steps=1,
                reason="pick_ready",
            ),
            OnObservationThreshold(
                source="pick",
                target="success",
                obs_key=_PICK_LIFT_OBSERVATION,
                threshold=success_lift_m,
                operator="ge",
                additional_reward=1.0,
                reason="workpiece_lifted",
            ),
            OnTimeLimit(
                source="pick",
                target="timeout",
                max_steps=episode_steps,
                reason="pick_timeout",
            ),
        ],
    )


class OraclePickController:
    """Small state machine used only to validate the AMP action/reward surface."""

    def __init__(self, peg_xy: np.ndarray | None = None) -> None:
        self.phase = "approach"
        self.close_steps = 0
        self.peg_xy = _PEG_XY.copy() if peg_xy is None else np.asarray(peg_xy)

    def action(self, observation: dict[str, Any]) -> torch.Tensor:
        tcp = np.asarray(
            [
                _scalar(observation[f"{DEFAULT_ROBOT_NAME}.x.ee_pos"]),
                _scalar(observation[f"{DEFAULT_ROBOT_NAME}.y.ee_pos"]),
                _scalar(observation[f"{DEFAULT_ROBOT_NAME}.z.ee_pos"]),
            ],
            dtype=np.float64,
        )
        gripper = 0.0
        if self.phase == "approach":
            target = np.asarray([*self.peg_xy, 0.28])
            if np.linalg.norm(target - tcp) < 0.008:
                self.phase = "descend"
        elif self.phase == "descend":
            target = np.asarray([*self.peg_xy, 0.16])
            if np.linalg.norm(target - tcp) < 0.006:
                self.phase = "close"
        elif self.phase == "close":
            target = tcp
            gripper = 1.0
            self.close_steps += 1
            if self.close_steps >= 30:
                self.phase = "lift"
        else:
            target = np.asarray([*self.peg_xy, 0.30])
            gripper = 1.0

        delta = np.clip(
            target - tcp,
            -_MAX_TRANSLATION_SPEED_M_S,
            _MAX_TRANSLATION_SPEED_M_S,
        )
        return torch.tensor([*delta, gripper], dtype=torch.float32)


def _scalar(value: Any) -> float:
    return float(torch.as_tensor(value).reshape(-1)[0].item())


def run_demo(
    *,
    steps: int = 500,
    viewer: bool = False,
    device: str = "cpu",
) -> bool:
    """Run the oracle against the standalone AMP and return whether it picked."""
    net = ManipulationPrimitiveNet(
        build_pick_amp_config(
            viewer=viewer,
            device=device,
            episode_steps=steps,
        )
    )
    transition = net.reset(seed=0)
    robot = net.robot_dict[DEFAULT_ROBOT_NAME]
    controller = OraclePickController(robot._data.xpos[robot._peg_body_id, :2])
    print(f"start -> {net.active_primitive}; action_dim={net.action_dim}")

    try:
        for step in range(steps):
            observation = transition[TransitionKey.OBSERVATION]
            action = controller.action(observation)
            transition = net.step(action)
            info = transition[TransitionKey.INFO]
            if step % 20 == 0 or info.get("transition_reason") is not None:
                lift_m = _scalar(
                    transition[TransitionKey.OBSERVATION][_PICK_LIFT_OBSERVATION]
                )
                print(
                    f"{step:04d} phase={controller.phase:8s} "
                    f"lift={lift_m:+.3f}m gripper={action[-1]:.0f} "
                    f"reason={info.get('transition_reason')}"
                )
            if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
                succeeded = (
                    info.get("transition_reason") == "workpiece_lifted"
                )
                print(f"finished -> {net.active_primitive}; success={succeeded}")
                return succeeded
    finally:
        net.close()
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    raise SystemExit(
        0
        if run_demo(steps=args.steps, viewer=args.viewer, device=args.device)
        else 1
    )
