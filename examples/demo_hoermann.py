"""Real UR5e demo that runs the FoundationPose pick MP-Net config."""
from __future__ import annotations
import logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname).1s | %(filename)s | %(message)s", force=True)
logger = logging.getLogger(__name__)


import time

import torch
from lerobot.processor import TransitionKey
from lerobot.utils.robot_utils import precise_sleep

from experiments.envs.foundationpose.ur5e_foundationpose_pick import UR5eFoundationPosePickEnvConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet




net_cfg = UR5eFoundationPosePickEnvConfig(
    robot_ip="172.22.22.2",
    fps=10,
    object_dir="/home/jzilke/ws/share-rl-pe/hoermann_objects/power_connector",
)


def run_demo() -> None:
    """Run the configured FoundationPose pick pipeline on the robot."""
    net = ManipulationPrimitiveNet(net_cfg)
    transition = net.reset()

    logger.info(f"start -> {net.active_primitive}")

    try:
        for _step in range(100_000):
            loop_t0 = time.perf_counter()
            action = torch.zeros(net.action_dim, dtype=torch.float32)
            transition = net.step(action)

            info = transition[TransitionKey.INFO]
            logger.info(
                f"[{net.active_primitive}] "
                f"primitive_step={info.get('primitive_step', 0):04d} "
                f"reason={info.get('transition_reason')} "
                f"progress={info.get('trajectory_progress', 0.0):.2f}"
            )

            if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
                transition = net.reset()

            dt = time.perf_counter() - loop_t0
            precise_sleep(1 / net.config.fps - dt)
    finally:
        net.close()


if __name__ == "__main__":
    run_demo()
