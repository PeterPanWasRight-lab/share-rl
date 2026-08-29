"""Run the small-motion HRCrobot pick-and-place state machine.

The default invocation is a dry run and never connects to hardware. Use
``--execute`` to connect, review the measured TCP waypoints, and type ``RUN``
before the first motion command is sent.
"""

from __future__ import annotations

import argparse
import time

import torch
from lerobot.processor import TransitionKey
from lerobot.utils.robot_utils import precise_sleep

from share.configs.HRCrobot_pick_place import ROBOT_NAME, HRCrobotPickPlaceConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)


def build_config(args: argparse.Namespace) -> HRCrobotPickPlaceConfig:
    return HRCrobotPickPlaceConfig(
        robot_ip=args.robot_ip,
        servo_frequency=args.servo_frequency,
        fps=args.fps,
        move_down_delta=[0.0, 0.0, -args.down_mm / 1000.0, 0.0, 0.0, 0.0],
        lift_delta=[0.0, 0.0, args.lift_mm / 1000.0, 0.0, 0.0, 0.0],
        move_to_place_delta=[
            args.place_x_mm / 1000.0,
            args.place_y_mm / 1000.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        move_down_duration_s=args.move_duration,
        lift_duration_s=args.lift_duration,
        move_to_place_duration_s=args.place_duration,
    )


def print_plan(config: HRCrobotPickPlaceConfig) -> None:
    print("HRCrobot small-motion pick-and-place plan (relative to startup TCP):")
    for name, offset in config.relative_waypoints.items():
        xyz_mm = [round(value * 1000.0, 3) for value in offset]
        print(f"  {name:>6}: XYZ offset {xyz_mm} mm")
    print(
        "  graph : move_down -> close_gripper -> lift -> "
        "move_to_place -> open_gripper -> done"
    )


def _confirmed_waypoints(
    net: ManipulationPrimitiveNet, config: HRCrobotPickPlaceConfig
) -> bool:
    robot = net.robot_dict[ROBOT_NAME]
    tcp_pose = robot.controller.get_tcp_pose()
    start_xyz = [float(value) for value in tcp_pose[:3]]

    print("\nMeasured startup TCP (m + rotation-vector rad):")
    print(" ", [round(float(value), 6) for value in tcp_pose])
    print("Predicted base-frame XYZ waypoints (m):")
    for name, offset in config.relative_waypoints.items():
        waypoint = [start_xyz[i] + offset[i] for i in range(3)]
        print(f"  {name:>6}: {[round(value, 6) for value in waypoint]}")

    print("\nVerify the workspace is clear and keep the hardware emergency stop ready.")
    try:
        return input("Type RUN to send motion commands: ").strip() == "RUN"
    except EOFError:
        return False


def run(config: HRCrobotPickPlaceConfig, *, max_steps: int) -> None:
    net = ManipulationPrimitiveNet(config)
    try:
        if not _confirmed_waypoints(net, config):
            print("Aborted before reset/step; no trajectory command was sent.")
            return

        net.reset()
        print(f"start -> {net.active_primitive}")
        previous = net.active_primitive

        for step in range(max_steps):
            loop_start = time.perf_counter()
            transition = net.step(torch.zeros(net.action_dim, dtype=torch.float32))
            info = transition[TransitionKey.INFO]
            current = net.active_primitive
            if current != previous:
                print(
                    f"step {step:04d}: {info['transition_from']} -> "
                    f"{info['transition_to']} (reason={info['transition_reason']})"
                )
                previous = current
            if net.in_terminal:
                print("Completed: reached terminal primitive 'done'.")
                return
            precise_sleep(
                max(0.0, 1.0 / config.fps - (time.perf_counter() - loop_start))
            )

        print(f"Stopped after {max_steps} steps (active={net.active_primitive}).")
    except KeyboardInterrupt:
        print("Interrupted; disconnecting. Use the hardware emergency stop if required.")
    finally:
        net.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="connect to and run the real robot"
    )
    parser.add_argument("--robot-ip", default="10.10.59.211")
    parser.add_argument("--servo-frequency", type=float, default=100.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--down-mm", type=float, default=5.0)
    parser.add_argument("--lift-mm", type=float, default=10.0)
    parser.add_argument("--place-x-mm", type=float, default=10.0)
    parser.add_argument("--place-y-mm", type=float, default=0.0)
    parser.add_argument("--move-duration", type=float, default=3.0)
    parser.add_argument("--lift-duration", type=float, default=3.0)
    parser.add_argument("--place-duration", type=float, default=4.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    print_plan(config)
    if not args.execute:
        print("\nDry run only: hardware was not connected. Add --execute to continue.")
        return
    run(config, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
