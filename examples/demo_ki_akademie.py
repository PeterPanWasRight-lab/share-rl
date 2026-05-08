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
    grasp_pose_in_object_frame=[0.004803642844073936, -0.010991811991058816, 0.040837245295234664, 3.0572526495102914, -0.013810568599069484, 1.5082028089374737],
    # [-0.0018128696024512607, -0.025629282618058406, 0.019719200104651516, -3.039952134094152, 0.29274915816824665, -0.21380204602051478]
    #
    grasp_pose_in_object_frame_2=[0.006951169031685781, -0.0016493444129640766, 0.02884079326744321, 3.9983788280586186, 0.2345169808077976, -0.2461135711969693],
    # grasp_pose_in_object_frame_2=[0.006951169031685781, -0.0016493444129640766, 0.02884079326744321, 2.1902423384676757, -0.05215371621298348, -0.3048412161442826]
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
