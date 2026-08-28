#!/usr/bin/env python
"""Evaluate a trained insertion policy from unseen randomized starts."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.processor import TransitionKey

from generate_mujoco_insertion_demos import make_trajectory_specs, _move_to_random_start
from generate_mujoco_force_search_demos import (
    _move_tip_to_search_start,
    make_force_search_specs,
)
from share.configs.mujoco_insertion import MujocoInsertionEnvConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)
from share.rl.runtime import build_adaptive_registry, make_policies_for_registry, make_policy_processors
from share.utils.control_utils import predict_action


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/mujoco/hardEncodedRunXYZ/insert/checkpoints/last/pretrained_model"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument(
        "--start-mode",
        choices=("standard", "force-search"),
        default="standard",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--trajectory-randomization", action="store_true")
    parser.add_argument("--fixture-xy-randomization-m", type=float, default=0.010)
    parser.add_argument(
        "--result-path",
        type=Path,
        default=Path("outputs/mujoco/hardEncodedRunXYZ/evaluation.json"),
    )
    return parser


def evaluate(args: argparse.Namespace) -> dict:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if args.fixture_xy_randomization_m < 0:
        raise ValueError("--fixture-xy-randomization-m must be non-negative.")

    # SAC evaluation samples from the actor distribution. Fix every RNG used by
    # the policy/runtime so a shared --seed means shared starts *and* shared
    # exploration noise across checkpoints.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    config = MujocoInsertionEnvConfig(
        viewer=args.viewer,
        teleop_mode="none",
        episode_steps=args.episode_steps,
        policy_device="cuda" if torch.cuda.is_available() else "cpu",
        state_only_policy=False,
        domain_randomization=args.domain_randomization,
        fixture_xy_randomization_m=(
            args.fixture_xy_randomization_m if args.domain_randomization else 0.002
        ),
    )
    net = ManipulationPrimitiveNet(config)
    try:
        policy_config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        policy_config.pretrained_path = checkpoint
        if args.deterministic:
            if not hasattr(policy_config, "training_mode"):
                raise ValueError("--deterministic requires a policy with training_mode support.")
            policy_config.training_mode = "bc"
        config.primitives["insert"].policy = policy_config
        registry = build_adaptive_registry(config)
        policies = make_policies_for_registry(config, registry, train_mode=False)
        preprocessors, postprocessors = make_policy_processors(policies)
        policy = policies["insert"]
        device = torch.device(policy.config.device)

        episode_results = []
        specs = (
            make_force_search_specs(args.episodes, args.seed)
            if args.start_mode == "force-search"
            else make_trajectory_specs(
                args.episodes,
                args.seed,
                trajectory_randomization=args.trajectory_randomization,
            )
        )
        for episode_index, spec in enumerate(specs):
            net.request_full_reset()
            transition = net.reset(seed=spec.seed)
            robot = next(iter(net.robot_dict.values()))
            if args.start_mode == "force-search":
                transition, _ = _move_tip_to_search_start(net, transition, robot, spec)
                initial_tcp = robot._tcp_world_pose().copy()
                start_offset = list(spec.estimated_hole_offset_m)
            else:
                transition, initial_tcp = _move_to_random_start(net, transition, robot, spec)
                start_offset = list(spec.start_offset_fixture_m)
            if hasattr(policy, "reset"):
                policy.reset()

            success = False
            reason = "time_limit"
            reward_sum = 0.0
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
                    device=device,
                    preprocessor=preprocessors["insert"],
                    postprocessor=postprocessors["insert"],
                    use_amp=policy.config.use_amp,
                    task="insert",
                    robot_type=None,
                ).squeeze()
                transition = net.step(action)
                reward_sum += float(transition[TransitionKey.REWARD])

                if transition.get(TransitionKey.DONE, False):
                    reason = transition[TransitionKey.INFO].get("transition_reason", "done")
                    success = reward_sum > 0.0 and reason == "peg_inserted"
                    break
                if transition.get(TransitionKey.TRUNCATED, False):
                    reason = "truncated"
                    break

            depth, lateral_error, axis_alignment = robot._insertion_metrics()
            result = {
                "episode": episode_index,
                "success": success,
                "reason": reason,
                "steps": step_index + 1,
                "reward": reward_sum,
                "start_mode": args.start_mode,
                "start_offset_fixture_m": start_offset,
                "initial_tcp_world": initial_tcp.tolist(),
                "final_depth_m": depth,
                "final_lateral_error_m": lateral_error,
                "final_axis_alignment": axis_alignment,
                "domain_randomization": robot.domain_randomization_state,
            }
            episode_results.append(result)
            print(
                f"[{episode_index + 1:02d}/{args.episodes:02d}] "
                f"success={success} reason={reason} steps={step_index + 1} "
                f"depth={depth:.4f} lateral={lateral_error:.5f}"
            )

        successes = sum(result["success"] for result in episode_results)
        summary = {
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "episodes": args.episodes,
            "successes": successes,
            "success_rate": successes / args.episodes,
            "domain_randomization": args.domain_randomization,
            "fixture_xy_randomization_m": (
                args.fixture_xy_randomization_m if args.domain_randomization else 0.002
            ),
            "trajectory_randomization": args.trajectory_randomization,
            "results": episode_results,
        }
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        args.result_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"success_rate={successes}/{args.episodes} ({summary['success_rate']:.1%})")
        return summary
    finally:
        net.close()


def main() -> None:
    evaluate(_parser().parse_args())


if __name__ == "__main__":
    main()
