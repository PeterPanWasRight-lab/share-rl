"""Minimal real-robot example: move to a known pose, then lift 10 cm in z."""

from __future__ import annotations

import time

import torch
from lerobot.processor import TransitionKey
from lerobot.teleoperators import keyboard
from lerobot.utils.robot_utils import precise_sleep

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    ManipulationPrimitiveConfig,
    MoveDeltaPrimitiveConfig, GripperConfig, EventConfig, ObservationConfig, ManipulationPrimitiveProcessorConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, TaskFrame, ControlSpace, PolicyMode
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import OnTargetPoseReached, OnSuccess
from share.robots.ur import URConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig


# Set this to a safe pose for your robot in world coordinates:
# [x, y, z, rx, ry, rz]
START_POSE = [-0.5, -0.0, 0.52, -3.0, 0.2, -1.4]
TARGET_UR3 = [-0.3232706776223777, -0.2579981466756967, 0.3477962169855162, 2.2205149081397813, -0.18851901719037523, -1.328941890818592]
TARGET_UR3_2 = [-0.3232706776223777, -0.2579981466756967, 0.3477962169855162, 2.2205149081397813, -0.18851901719037523, -1.328941890818592]

TARGET_TEST = [-0.429, 0.126, 0.261, 3.112, 0.068, -2.14]
TARGET_TEST_2 = [-0.429, 0.026, 0.261, 3.112, 0.068, -2.14]



home_cfg = ManipulationPrimitiveConfig(
    notes="Move to a known safe start pose.",
    task_frame=TaskFrame(
        target=TARGET_UR3,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    ),
)

target_cfg = ManipulationPrimitiveConfig(
    notes="Target",
    task_frame=TaskFrame(
        target=TARGET_UR3_2,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )
)

teleop_primitive = ManipulationPrimitiveConfig(
    task_frame=TaskFrame(
        target=[0.0] * 6,
        space=ControlSpace.TASK,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.RELATIVE] * 6,
        controller_overrides={"wrench_limits": list([25.0, 25.0, 25.0, 1.0, 1.0, 1.0])},
    ),
    processor=ManipulationPrimitiveProcessorConfig(
        fps=30.0,
        observation=ObservationConfig(
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_ee_pos_to_observation=False,
            add_joint_position_to_observation=False,
        ),
        gripper=GripperConfig(enable=False, static_pos=0.0),
        events=EventConfig(
            foot_switch_mapping={
                (TeleopEvents.SUCCESS,): {"device": 4, "toggle": False},
            },
        ),
    ),
    notes="6-DoF relative task-space teleop primitive.",
)


net_cfg = ManipulationPrimitiveNetConfig(
    fps=30,
    start_primitive="teleop",
    reset_primitive="teleop",
    primitives={
        "home": home_cfg,
        "target": target_cfg,
        "teleop": teleop_primitive
    },
    transitions=[
        OnTargetPoseReached(source="home", target="teleop", axes=["x", "y", "z", "rx", "ry", "rz"], tolerance=0.01),
        OnSuccess(source="teleop", target="target"),
        OnTargetPoseReached(source="target", target="home", axes=["x", "y", "z", "rx", "ry", "rz"], tolerance=0.01),
    ],
    robot=URConfig(
        robot_ip="172.22.22.2",
        frequency=500,
        soft_real_time=True,
        rt_core=3,
        use_gripper=True,
    ),
    # This example does not use teleop input, but the current runtime expects a teleop config.
    teleop=SpaceMouseConfig(action_scale=[0.25, 0.25, 0.20, 0.50, 0.50, 0.50]),
)


def run_demo(max_steps: int = 20_000) -> None:
    """Run until the robot reaches START_POSE + [0, 0, 0.10, 0, 0, 0]."""
    net = ManipulationPrimitiveNet(net_cfg)
    transition = net.reset()
    print([float(transition[TransitionKey.OBSERVATION][k]) for k in
           ['main.x.ee_pos', 'main.y.ee_pos', 'main.z.ee_pos', 'main.rx.ee_pos', 'main.ry.ee_pos', 'main.rz.ee_pos']])

    print(f"start -> {net.active_primitive}")

    for _step in range(max_steps):
        loop_t0 = time.perf_counter()
        action = torch.zeros(net.action_dim, dtype=torch.float32)
        transition = net.step(action)
        info = transition[TransitionKey.INFO]
        print()
        print([float(transition[TransitionKey.OBSERVATION][k]) for k in
               ['main.x.ee_pos', 'main.y.ee_pos', 'main.z.ee_pos', 'main.rx.ee_pos', 'main.ry.ee_pos', 'main.rz.ee_pos']])
        print(transition[TransitionKey.OBSERVATION]['observation.state'])
        print(
            f"[{net.active_primitive}] "
            f"primitive_step={info.get('primitive_step', 0):04d} "
            f"reason={info.get('transition_reason')} "
            f"target={info.get('primitive_target_pose')}"
        )

        # if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
        #     print("Reached the final primitive. Call net.reset() to run the sequence again.")
        #     break

        dt = time.perf_counter() - loop_t0
        precise_sleep(1 / net.config.fps - dt)


if __name__ == "__main__":
    run_demo()
