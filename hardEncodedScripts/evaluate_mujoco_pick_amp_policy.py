#!/usr/bin/env python
"""Evaluate a Pick AMP checkpoint on fixed randomized MuJoCo episodes."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.processor import TransitionKey

from examples.demo_pick_amp import build_pick_amp_config
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)
from share.rl.runtime import (
    build_adaptive_registry,
    make_policies_for_registry,
    make_policy_processors,
)
from share.utils.constants import DEFAULT_ROBOT_NAME
from share.utils.control_utils import predict_action


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--peg-xy-randomization-m", type=float, default=0.025)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--video-path",
        type=Path,
        help="Record the first evaluated episode as a front+wrist MP4.",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("outputs/mujoco/pickAMP/evaluation.json"),
    )
    return parser


def _video_frame(robot, *, step: int, initial_height_m: float) -> np.ndarray:
    import cv2

    front = robot.render_camera(camera_name="front", width=320, height=320)
    wrist = robot.render_camera(camera_name="wrist", width=320, height=320)
    composite = np.concatenate((front, wrist), axis=1)
    banner = np.zeros((44, composite.shape[1], 3), dtype=np.uint8)
    lift_m = float(robot._data.xpos[robot._peg_body_id, 2] - initial_height_m)
    cv2.putText(
        banner,
        f"Pick AMP | step {step:03d} | lift {lift_m:+.3f} m | front / wrist",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate((banner, composite), axis=0)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _wilson_interval(successes: int, episodes: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / episodes
    denominator = 1.0 + z * z / episodes
    centre = (rate + z * z / (2.0 * episodes)) / denominator
    radius = (
        z
        * np.sqrt(rate * (1.0 - rate) / episodes + z * z / (4.0 * episodes**2))
        / denominator
    )
    return float(centre - radius), float(centre + radius)


def evaluate(args: argparse.Namespace) -> dict:
    if args.episodes <= 0 or args.episode_steps <= 0:
        raise ValueError("--episodes and --episode-steps must be positive.")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = build_pick_amp_config(
        device=device,
        viewer=args.viewer,
        episode_steps=args.episode_steps,
        peg_xy_randomization_m=args.peg_xy_randomization_m,
    )
    net = ManipulationPrimitiveNet(config)
    try:
        policy_config = PreTrainedConfig.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        policy_config.pretrained_path = checkpoint
        policy_config.device = device
        if not args.stochastic:
            policy_config.training_mode = "bc"
        config.primitives["pick"].policy = policy_config
        registry = build_adaptive_registry(config)
        policies = make_policies_for_registry(config, registry, train_mode=False)
        preprocessors, postprocessors = make_policy_processors(policies)
        policy = policies["pick"]

        results = []
        for episode_index in range(args.episodes):
            episode_seed = args.seed + episode_index
            net.request_full_reset()
            transition = net.reset(seed=episode_seed)
            if hasattr(policy, "reset"):
                policy.reset()
            robot = net.robot_dict[DEFAULT_ROBOT_NAME]
            initial_peg = robot._data.xpos[robot._peg_body_id].copy()
            video_frames = [] if args.video_path is not None and episode_index == 0 else None
            if video_frames is not None:
                video_frames.append(
                    _video_frame(robot, step=0, initial_height_m=float(initial_peg[2]))
                )
            reward_sum = 0.0
            reason = "time_limit"
            success = False

            for step_index in range(args.episode_steps):
                observation = transition[TransitionKey.OBSERVATION]
                policy_observation = {
                    key: value
                    for key, value in observation.items()
                    if key in policy.config.input_features
                }
                action = predict_action(
                    observation=policy_observation,
                    policy=policy,
                    device=torch.device(device),
                    preprocessor=preprocessors["pick"],
                    postprocessor=postprocessors["pick"],
                    use_amp=policy.config.use_amp,
                    task="pick",
                    robot_type=None,
                ).squeeze()
                transition = net.step(action)
                if video_frames is not None:
                    video_frames.append(
                        _video_frame(
                            robot,
                            step=step_index + 1,
                            initial_height_m=float(initial_peg[2]),
                        )
                    )
                reward_sum += float(transition[TransitionKey.REWARD])
                if transition.get(TransitionKey.DONE, False):
                    reason = transition[TransitionKey.INFO].get(
                        "transition_reason", "done"
                    )
                    success = reward_sum > 0.0 and reason == "workpiece_lifted"
                    break
                if transition.get(TransitionKey.TRUNCATED, False):
                    reason = "truncated"
                    break

            final_peg = robot._data.xpos[robot._peg_body_id].copy()
            if video_frames is not None:
                video_frames.extend([video_frames[-1].copy() for _ in range(30)])
                _write_video(args.video_path, video_frames, config.fps)
                print(f"video={args.video_path.resolve()}")
            results.append(
                {
                    "episode": episode_index,
                    "seed": episode_seed,
                    "success": success,
                    "reason": reason,
                    "steps": step_index + 1,
                    "reward": reward_sum,
                    "initial_peg_world": initial_peg.tolist(),
                    "final_peg_world": final_peg.tolist(),
                    "lift_m": float(final_peg[2] - initial_peg[2]),
                }
            )
            print(
                f"[{episode_index + 1:02d}/{args.episodes:02d}] "
                f"success={success} reason={reason} steps={step_index + 1}"
            )

        successes = sum(item["success"] for item in results)
        interval = _wilson_interval(successes, args.episodes)
        summary = {
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "episodes": args.episodes,
            "successes": successes,
            "success_rate": successes / args.episodes,
            "success_rate_wilson_95": list(interval),
            "deterministic": not args.stochastic,
            "peg_xy_randomization_m": args.peg_xy_randomization_m,
            "video_path": (
                str(args.video_path.resolve()) if args.video_path is not None else None
            ),
            "results": results,
        }
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        args.result_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"success_rate={successes}/{args.episodes} "
            f"({summary['success_rate']:.1%}), "
            f"Wilson95%=[{interval[0]:.1%}, {interval[1]:.1%}]"
        )
        return summary
    finally:
        net.close()


def main() -> None:
    evaluate(_parser().parse_args())


if __name__ == "__main__":
    main()
