#!/usr/bin/env python
"""Project 7D insertion demonstrations to XYZ plus gripper actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, REWARD


TASK = "Insert the held peg into the known fixture"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("outputs/mujoco/hardEncodedDemos/insert"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mujoco/hardEncodedDemosXYZ"),
    )
    parser.add_argument(
        "--alignment-repeat",
        type=int,
        default=1,
        help="Repeat pre-insertion alignment samples this many times.",
    )
    return parser


def project(source: Path, output_root: Path, *, alignment_repeat: int = 1) -> Path:
    if alignment_repeat < 1:
        raise ValueError("alignment_repeat must be at least 1")
    destination = output_root / "insert"
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    table = pq.read_table(source / "data" / "chunk-000" / "file-000.parquet")
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    source_actions = np.asarray(table[ACTION].to_pylist(), dtype=np.float32)
    rewards = np.asarray(table[REWARD].to_pylist(), dtype=np.float32)
    dones = np.asarray(table[DONE].to_pylist(), dtype=bool)
    interventions = np.asarray(table["rl.is_intervention"].to_pylist(), dtype=bool)
    episode_indices = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)

    if source_actions.shape[1] != 7:
        raise ValueError(f"Expected 7D source actions, got {source_actions.shape[1]}D")
    xyz_gripper_actions = np.concatenate(
        (source_actions[:, :3], source_actions[:, -1:]), axis=1
    )

    features = {
        ACTION: {"dtype": "float32", "shape": (4,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (31,), "names": None},
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
        "rl.is_intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/mujoco-hard-encoded-insertion-xyz-insert",
        fps=30,
        root=destination,
        features=features,
        robot_type="mujoco_ur5e_insertion",
        use_videos=False,
        metadata_buffer_size=10,
    )

    current_episode = int(episode_indices[0])
    source_alignment_frames = 0
    written_alignment_frames = 0
    written_frames = 0
    for index in range(len(states)):
        episode_index = int(episode_indices[index])
        if episode_index != current_episode:
            dataset.save_episode()
            current_episode = episode_index
        # The fixture axis is world Z in this scene. Scripted alignment frames
        # therefore have effectively zero Z velocity, while approach/insertion
        # frames have a clearly nonzero Z command.
        is_alignment = abs(float(source_actions[index, 2])) < 0.005
        repeats = alignment_repeat if is_alignment else 1
        source_alignment_frames += int(is_alignment)
        for _ in range(repeats):
            dataset.add_frame(
                {
                    "observation.state": states[index],
                    ACTION: xyz_gripper_actions[index],
                    REWARD: np.asarray([rewards[index]], dtype=np.float32),
                    DONE: np.asarray([dones[index]], dtype=bool),
                    "rl.is_intervention": np.asarray([interventions[index]], dtype=bool),
                    "task": TASK,
                }
            )
            written_frames += 1
            written_alignment_frames += int(is_alignment)
    dataset.save_episode()
    dataset.finalize()
    print(
        f"Alignment frames: source={source_alignment_frames}/{len(states)}; "
        f"projected={written_alignment_frames}/{written_frames} "
        f"({written_alignment_frames / written_frames:.1%})"
    )
    return destination


def main() -> None:
    args = _parser().parse_args()
    destination = project(
        args.source.resolve(),
        args.output_root.resolve(),
        alignment_repeat=args.alignment_repeat,
    )
    print(f"Projected XYZ dataset written to {destination}")


if __name__ == "__main__":
    main()
