from __future__ import annotations

import argparse
import time

import torch
from lerobot.processor import TransitionKey
from lerobot.utils.robot_utils import precise_sleep

from share.configs.mujoco_insertion import MujocoInsertionEnvConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the built-in MuJoCo insertion scene.")
    parser.add_argument("--headless", action="store_true", help="Disable the interactive MuJoCo viewer.")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps; zero runs until Ctrl-C.")
    parser.add_argument("--manual", action="store_true", help="Disable automatic insertion motion.")
    parser.add_argument(
        "--viewer-camera",
        choices=("free", "front", "side", "wrist"),
        default="free",
        help="Select the MuJoCo viewer camera.",
    )
    parser.add_argument(
        "--release-steps",
        type=int,
        default=30,
        help="Number of control steps used to open the gripper before reset.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch the built-in scene and run keyboard-assisted insertion episodes."""
    args = _parse_args()
    config = MujocoInsertionEnvConfig(
        viewer=not args.headless,
        viewer_camera=None if args.viewer_camera == "free" else args.viewer_camera,
        teleop_mode="none" if args.headless else "keyboard",
        release_steps=args.release_steps,
    )
    net = ManipulationPrimitiveNet(config)
    transition = net.reset(seed=0)
    teleop = next(iter(net.teleop_dict.values()))
    set_gripper_position = getattr(teleop, "set_gripper_position")
    set_gripper_position(1.0)
    camera_shapes = {name: camera.async_read().shape for name, camera in net.cameras.items()}
    print(f"MuJoCo cameras: {camera_shapes}")
    print(
        "Automatic cycle: insert -> open gripper -> drop peg -> reset. "
        "Keyboard: arrows move XY, left/right Shift move Z, "
        "right/left Ctrl open/close; Ctrl-C exits."
    )

    try:
        step = 0
        while args.steps <= 0 or step < args.steps:
            started = time.perf_counter()
            action = torch.zeros(net.action_dim, dtype=torch.float32)
            if net.active_primitive == "insert":
                action[-1] = 1.0
            if not args.manual and net.active_primitive == "insert":
                action[2] = -0.05
            if net.active_primitive == "release":
                set_gripper_position(0.0)
            transition = net.step(action)
            step += 1
            info = transition[TransitionKey.INFO]
            robot = next(iter(net.robot_dict.values()))
            gripper_position = robot.get_observation().get("gripper.pos", float("nan"))
            print(
                f"\rprimitive={net.active_primitive:>6} "
                f"step={info.get('primitive_step', 0):04d} "
                f"gripper={gripper_position:.2f} "
                f"reason={info.get('transition_reason')}",
                end="",
                flush=True,
            )
            if net.in_terminal:
                set_gripper_position(1.0)
                transition = net.reset()
            precise_sleep(max(0.0, 1.0 / config.fps - (time.perf_counter() - started)))
    except KeyboardInterrupt:
        print()
    finally:
        net.close()


if __name__ == "__main__":
    main()
