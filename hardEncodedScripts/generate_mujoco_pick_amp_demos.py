#!/usr/bin/env python
"""Generate successful oracle demonstrations for the standalone Pick AMP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import TransitionKey
from lerobot.utils.constants import ACTION, DONE, REWARD

from examples.demo_pick_amp import OraclePickController, build_pick_amp_config
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)
from share.envs.utils import env_to_dataset_features
from share.utils.constants import DEFAULT_ROBOT_NAME


TASK = "Pick the red workpiece and lift it clear of the table"
PRIMITIVE = "pick"
REPO_ID = "local/mujoco-pick-amp-demos"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--peg-xy-randomization-m", type=float, default=0.025)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mujoco/pickAMPDemos100"),
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser


def _open_dataset(
    output_root: Path,
    config: Any,
    *,
    use_videos: bool,
) -> LeRobotDataset:
    dataset_root = output_root / PRIMITIVE
    if dataset_root.exists():
        return LeRobotDataset.resume(
            REPO_ID,
            root=dataset_root,
            video_backend="pyav",
            batch_encoding_size=1,
            vcodec="h264",
            image_writer_threads=8,
        )

    primitive = config.primitives[PRIMITIVE]
    if primitive.features is None:
        raise RuntimeError("Pick features were not inferred before dataset creation.")
    features = env_to_dataset_features(primitive.features)
    if not use_videos:
        for feature in features.values():
            if feature["dtype"] == "video":
                feature["dtype"] = "image"
    return LeRobotDataset.create(
        REPO_ID,
        fps=config.fps,
        root=dataset_root,
        features=features,
        robot_type="mujoco_pick_amp",
        use_videos=use_videos,
        video_backend="pyav",
        batch_encoding_size=1,
        vcodec="h264",
        image_writer_threads=8,
        metadata_buffer_size=10,
    )


def _add_frame(
    dataset: LeRobotDataset,
    observation: dict[str, Any],
    transition: dict[str, Any],
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


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes <= 0 or args.episode_steps <= 0:
        raise ValueError("--episodes and --episode-steps must be positive.")
    if args.peg_xy_randomization_m < 0.0:
        raise ValueError("--peg-xy-randomization-m must be non-negative.")

    output_root = args.output_root.resolve()
    manifest_path = output_root / "pick_manifest.json"
    config = build_pick_amp_config(
        viewer=args.viewer,
        episode_steps=args.episode_steps,
        peg_xy_randomization_m=args.peg_xy_randomization_m,
    )
    net = ManipulationPrimitiveNet(config)
    dataset: LeRobotDataset | None = None
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"task": TASK, "episodes": []}
    )

    try:
        dataset = _open_dataset(output_root, config, use_videos=not args.no_video)
        completed = dataset.num_episodes
        if completed != len(manifest["episodes"]):
            raise RuntimeError(
                "Dataset/manifest episode mismatch: "
                f"dataset={completed}, manifest={len(manifest['episodes'])}."
            )
        if completed > args.episodes:
            raise RuntimeError(
                f"Dataset already has {completed} episodes; requested {args.episodes}."
            )

        for episode_index in range(completed, args.episodes):
            episode_seed = args.seed + episode_index
            net.request_full_reset()
            transition = net.reset(seed=episode_seed)
            robot = net.robot_dict[DEFAULT_ROBOT_NAME]
            initial_peg = robot._data.xpos[robot._peg_body_id].copy()
            controller = OraclePickController(initial_peg[:2])
            success = False

            for step_index in range(args.episode_steps):
                observation = transition[TransitionKey.OBSERVATION]
                transition = net.step(controller.action(observation))
                _add_frame(dataset, observation, transition)
                if transition.get(TransitionKey.DONE, False):
                    success = (
                        float(transition[TransitionKey.REWARD]) > 0.0
                        and transition[TransitionKey.INFO].get("transition_reason")
                        == "workpiece_lifted"
                    )
                    break
                if transition.get(TransitionKey.TRUNCATED, False):
                    break

            if not success:
                dataset.clear_episode_buffer()
                raise RuntimeError(
                    f"Oracle failed at episode {episode_index}, seed={episode_seed}."
                )

            final_peg = robot._data.xpos[robot._peg_body_id].copy()
            dataset.save_episode()
            manifest["episodes"].append(
                {
                    "episode": episode_index,
                    "seed": episode_seed,
                    "frames": step_index + 1,
                    "initial_peg_world": initial_peg.tolist(),
                    "final_peg_world": final_peg.tolist(),
                    "lift_m": float(final_peg[2] - initial_peg[2]),
                    "success": True,
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                f"[{episode_index + 1:03d}/{args.episodes:03d}] "
                f"frames={step_index + 1:03d} lift={final_peg[2] - initial_peg[2]:.3f}m"
            )
    finally:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
        net.close()

    return manifest


def main() -> None:
    generate(_parser().parse_args())


if __name__ == "__main__":
    main()
