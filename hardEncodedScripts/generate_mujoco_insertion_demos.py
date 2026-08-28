#!/usr/bin/env python
"""Generate successful scripted MuJoCo insertion demonstrations."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import TransitionKey
from lerobot.utils.constants import ACTION, DONE, REWARD

from share.configs.mujoco_insertion import MujocoInsertionEnvConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)
from share.envs.utils import env_to_dataset_features
from share.utils.env_config_snapshot import save_env_config_snapshot


TASK = "Insert the held peg into the known fixture"
PRIMITIVE = "insert"
REPO_ID = "local/mujoco-hard-encoded-insertion-xyz-insert"


@dataclass(frozen=True)
class TrajectorySpec:
    index: int
    seed: int
    start_offset_fixture_m: tuple[float, float, float]
    align_gain: float
    approach_speed_m_s: float
    insertion_speed_m_s: float
    approach_waypoint_fixture_m: tuple[float, float]
    curve_bulge_fixture_m: tuple[float, float]
    curve_power: float


@dataclass
class TrajectoryRuntime:
    stage: str = "align"
    approach_start_depth_m: float | None = None


def make_trajectory_specs(
    count: int,
    seed: int,
    *,
    trajectory_randomization: bool = False,
) -> list[TrajectorySpec]:
    """Create a deterministic bank of scripted trajectory parameters."""
    rng = np.random.default_rng(seed)
    specs = []
    for index in range(count):
        if trajectory_randomization:
            waypoint_radius = float(rng.uniform(0.0, 0.015))
            waypoint_angle = float(rng.uniform(-np.pi, np.pi))
            bulge_radius = float(rng.uniform(0.0, 0.008))
            bulge_angle = float(rng.uniform(-np.pi, np.pi))
            start_offset = (
                float(rng.uniform(0.0, 0.025)),
                float(rng.uniform(-0.020, 0.020)),
                float(rng.uniform(-0.020, 0.020)),
            )
            align_gain = float(rng.uniform(4.0, 12.0))
            approach_speed = float(rng.uniform(0.050, 0.110))
            insertion_speed = float(rng.uniform(0.025, 0.055))
            waypoint = (
                waypoint_radius * float(np.cos(waypoint_angle)),
                waypoint_radius * float(np.sin(waypoint_angle)),
            )
            bulge = (
                bulge_radius * float(np.cos(bulge_angle)),
                bulge_radius * float(np.sin(bulge_angle)),
            )
            curve_power = float(rng.uniform(0.7, 1.6))
        else:
            start_offset = (
                float(rng.uniform(0.004, 0.014)),
                float(rng.uniform(-0.007, 0.007)),
                float(rng.uniform(-0.007, 0.007)),
            )
            align_gain = float(rng.uniform(7.0, 10.0))
            approach_speed = float(rng.uniform(0.075, 0.095))
            insertion_speed = float(rng.uniform(0.035, 0.050))
            waypoint = (0.0, 0.0)
            bulge = (0.0, 0.0)
            curve_power = 1.0
        specs.append(
            TrajectorySpec(
                index=index,
                seed=seed + 10_000 + index,
                start_offset_fixture_m=start_offset,
                align_gain=align_gain,
                approach_speed_m_s=approach_speed,
                insertion_speed_m_s=insertion_speed,
                approach_waypoint_fixture_m=waypoint,
                curve_bulge_fixture_m=bulge,
                curve_power=curve_power,
            )
        )
    return specs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record deterministic known-fixture MuJoCo insertion demonstrations."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mujoco/hardEncodedDemosXYZGenerated"),
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--trajectory-randomization",
        action="store_true",
        help="Use wider starts and curved/dog-leg Cartesian approaches.",
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="Randomize fixture pose, cameras, lighting, appearance, friction, and peg mass.",
    )
    parser.add_argument(
        "--fixture-xy-randomization-m",
        type=float,
        default=0.010,
        help="Fixture X/Y half-range used with --domain-randomization.",
    )
    return parser


def _fixture_frame(robot: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fixture_position = robot._data.xpos[robot._fixture_body_id].copy()
    fixture_rotation = robot._data.xmat[robot._fixture_body_id].reshape(3, 3).copy()
    tip_relative = fixture_rotation.T @ (
        robot._data.site_xpos[robot._peg_tip_site_id] - fixture_position
    )
    return fixture_position, fixture_rotation, tip_relative


def _action(world_velocity: np.ndarray, action_dim: int) -> torch.Tensor:
    action = torch.zeros(action_dim, dtype=torch.float32)
    action[:3] = torch.as_tensor(np.clip(world_velocity, -0.1, 0.1), dtype=torch.float32)
    # Legacy insertion policies exposed XYZ plus a learned gripper command.
    # The current insertion primitive is XYZ-only and controls the gripper with
    # its state machine, so writing action[-1] would overwrite the Z command.
    if action_dim > 3:
        action[-1] = 1.0
    return action


def _move_to_random_start(
    net: ManipulationPrimitiveNet,
    transition: dict,
    robot: Any,
    spec: TrajectorySpec,
) -> tuple[dict, np.ndarray]:
    """Move to the randomized episode start without recording setup frames."""
    _, fixture_rotation, _ = _fixture_frame(robot)
    reset_tcp = robot._tcp_world_pose().copy()
    target_position = reset_tcp[:3] + fixture_rotation @ np.asarray(
        spec.start_offset_fixture_m, dtype=np.float64
    )

    for _ in range(60):
        position_error = target_position - robot._tcp_world_pose()[:3]
        if np.linalg.norm(position_error) <= 4e-4:
            break
        transition = net.step(_action(6.0 * position_error, net.action_dim))
        if transition.get(TransitionKey.DONE, False) or transition.get(TransitionKey.TRUNCATED, False):
            raise RuntimeError("Episode ended while moving to its randomized start.")
    else:
        raise RuntimeError("Could not reach randomized start pose within 60 control steps.")

    return transition, robot._tcp_world_pose().copy()


def _planned_action(
    net: ManipulationPrimitiveNet,
    robot: Any,
    spec: TrajectorySpec,
    runtime: TrajectoryRuntime,
) -> tuple[torch.Tensor, TrajectoryRuntime]:
    """Plan one exact-fixture Cartesian command and advance the scripted stage."""
    _, fixture_rotation, tip_relative = _fixture_frame(robot)
    depth, lateral_error, _ = robot._insertion_metrics()

    waypoint = np.asarray(spec.approach_waypoint_fixture_m, dtype=np.float64)
    bulge = np.asarray(spec.curve_bulge_fixture_m, dtype=np.float64)
    if runtime.stage == "align" and np.linalg.norm(tip_relative[1:] - waypoint) <= 5e-4:
        runtime.stage = "approach"
        runtime.approach_start_depth_m = float(depth)
    if runtime.stage == "approach" and depth >= -0.004:
        runtime.stage = "insert"

    local_velocity = np.zeros(3, dtype=np.float64)
    desired_lateral = waypoint
    if runtime.stage == "approach":
        start_depth = min(float(runtime.approach_start_depth_m or depth), -0.005)
        progress = float(np.clip((depth - start_depth) / (-0.004 - start_depth), 0.0, 1.0))
        desired_lateral = waypoint * (1.0 - progress) ** spec.curve_power
        desired_lateral += bulge * np.sin(np.pi * progress)
    elif runtime.stage == "insert":
        desired_lateral = np.zeros(2, dtype=np.float64)
    local_velocity[1:] = np.clip(
        spec.align_gain * (desired_lateral - tip_relative[1:]),
        -0.075,
        0.075,
    )
    if runtime.stage == "approach":
        local_velocity[0] = -spec.approach_speed_m_s
    elif runtime.stage == "insert":
        local_velocity[0] = -spec.insertion_speed_m_s

    return _action(fixture_rotation @ local_velocity, net.action_dim), runtime


def _add_dataset_frame(
    dataset: LeRobotDataset,
    observation: dict[str, Any],
    transition: dict,
) -> None:
    dataset_observation = {
        key: value.squeeze().cpu()
        for key, value in observation.items()
        if key in dataset.features
    }
    dataset.add_frame(
        {
            **dataset_observation,
            ACTION: transition[TransitionKey.ACTION].squeeze().cpu(),
            REWARD: np.asarray([transition[TransitionKey.REWARD]], dtype=np.float32),
            DONE: np.asarray([transition.get(TransitionKey.DONE, False)], dtype=bool),
            "rl.is_intervention": np.asarray([True], dtype=bool),
            "task": TASK,
        }
    )


def _open_dataset(
    output_root: Path,
    config: MujocoInsertionEnvConfig,
    *,
    use_videos: bool,
    repo_id: str = REPO_ID,
) -> LeRobotDataset:
    dataset_root = output_root / PRIMITIVE
    if dataset_root.exists():
        return LeRobotDataset.resume(
            repo_id,
            root=dataset_root,
            video_backend="pyav",
            batch_encoding_size=1,
            vcodec="h264",
            image_writer_threads=8,
        )

    primitive = config.primitives[PRIMITIVE]
    if primitive.features is None:
        raise RuntimeError("Primitive features were not inferred before dataset creation.")
    features = env_to_dataset_features(primitive.features)
    if not use_videos:
        for feature in features.values():
            if feature["dtype"] == "video":
                feature["dtype"] = "image"
    return LeRobotDataset.create(
        repo_id,
        fps=config.fps,
        root=dataset_root,
        features=features,
        robot_type=config.type,
        use_videos=use_videos,
        video_backend="pyav",
        batch_encoding_size=1,
        vcodec="h264",
        image_writer_threads=8,
        metadata_buffer_size=10,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"task": TASK, "episodes": []}
    return json.loads(path.read_text())


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def generate(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive.")
    if args.fixture_xy_randomization_m < 0:
        raise ValueError("--fixture-xy-randomization-m must be non-negative.")

    output_root = args.output_root.resolve()
    config = MujocoInsertionEnvConfig(
        viewer=args.viewer,
        teleop_mode="none",
        episode_steps=300,
        domain_randomization=args.domain_randomization,
        fixture_xy_randomization_m=(
            args.fixture_xy_randomization_m if args.domain_randomization else 0.002
        ),
    )
    net = ManipulationPrimitiveNet(config)
    dataset: LeRobotDataset | None = None
    manifest_path = output_root / "trajectory_manifest.json"

    try:
        save_env_config_snapshot(config, output_root)
        dataset = _open_dataset(output_root, config, use_videos=not args.no_video)
        manifest = _load_manifest(manifest_path)
        # Generate a 2x buffer of specs so skipped entries can be replaced.
        specs = make_trajectory_specs(
            args.episodes * 2,
            args.seed,
            trajectory_randomization=args.trajectory_randomization,
        )
        completed = dataset.num_episodes
        if len(manifest["episodes"]) != completed:
            raise RuntimeError(
                "Dataset/manifest episode mismatch: "
                f"dataset={completed}, manifest={len(manifest['episodes'])}."
            )
        if completed > args.episodes:
            raise RuntimeError(
                f"Dataset already contains {completed} episodes, exceeding requested {args.episodes}."
            )

        # saved counts successes; spec_index advances unconditionally (skipped specs
        # are transparent — the while loop keeps running until saved == args.episodes).
        spec_index = completed
        saved = completed
        while saved < args.episodes:
            if spec_index >= len(specs):
                raise RuntimeError("Ran out of trajectory specs; increase the spec buffer multiplier.")
            spec = specs[spec_index]
            spec_index += 1

            episode_succeeded = False
            for attempt in range(args.max_attempts):
                net.request_full_reset()
                transition = net.reset(seed=spec.seed + attempt * 100_000)
                robot = next(iter(net.robot_dict.values()))
                try:
                    transition, initial_tcp = _move_to_random_start(net, transition, robot, spec)
                except RuntimeError as exc:
                    logging.warning("Spec %d setup attempt %d failed: %s", spec_index - 1, attempt + 1, exc)
                    continue

                runtime = TrajectoryRuntime()
                success = False
                for frame_index in range(config.episode_steps):
                    observation = transition[TransitionKey.OBSERVATION]
                    action, runtime = _planned_action(net, robot, spec, runtime)
                    new_transition = net.step(action)
                    _add_dataset_frame(dataset, observation, new_transition)
                    transition = new_transition

                    if new_transition.get(TransitionKey.DONE, False):
                        success = (
                            float(new_transition[TransitionKey.REWARD]) > 0.0
                            and new_transition[TransitionKey.INFO].get("transition_reason") == "peg_inserted"
                        )
                        break
                    if new_transition.get(TransitionKey.TRUNCATED, False):
                        break

                if not success:
                    dataset.clear_episode_buffer()
                    logging.warning(
                        "Spec %d attempt %d did not insert; retrying.",
                        spec_index - 1,
                        attempt + 1,
                    )
                    continue

                # --- success ---
                final_depth, final_lateral_error, final_axis_alignment = robot._insertion_metrics()
                dataset.save_episode()
                manifest["episodes"].append(
                    {
                        **asdict(spec),
                        "attempt": attempt + 1,
                        "frames": frame_index + 1,
                        "initial_tcp_world": initial_tcp.tolist(),
                        "fixture_world": robot._data.xpos[robot._fixture_body_id].tolist(),
                        "domain_randomization": robot.domain_randomization_state,
                        "final_depth_m": final_depth,
                        "final_lateral_error_m": final_lateral_error,
                        "final_axis_alignment": final_axis_alignment,
                        "reward": 1.0,
                        "done": True,
                    }
                )
                _save_manifest(manifest_path, manifest)
                saved += 1
                print(
                    f"[{saved:03d}/{args.episodes:03d}] "
                    f"spec={spec_index - 1} "
                    f"saved frames={frame_index + 1:03d} depth={final_depth:.4f} "
                    f"lateral={final_lateral_error:.5f}"
                )
                episode_succeeded = True
                break

            if not episode_succeeded:
                logging.warning(
                    "Spec %d failed all %d attempts; skipping to next spec.",
                    spec_index - 1,
                    args.max_attempts,
                )
    finally:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
        net.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate(_parser().parse_args())


if __name__ == "__main__":
    main()
