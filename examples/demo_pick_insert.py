"""MuJoCo UR5e demo that runs the pick-and-insert MP-Net state machine.

Runs the deterministic 10-step pick-and-insert graph on the dedicated
``pick_insert`` MuJoCo scene (peg free at A, hole at B). Set ``--viewer`` to
watch the rollout in the MuJoCo viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lerobot.processor import TransitionKey

from experiments.envs.pick_insert.ur5e_pick_insert import UR5ePickInsertEnvConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)

_REPO_ROOT = Path(__file__).parent.parent


def run_demo(steps: int = 2000, *, viewer: bool = False) -> None:
    net_cfg = UR5ePickInsertEnvConfig(
        fps=30,
        poses_file=str(
            _REPO_ROOT / "src" / "experiments" / "envs" / "pick_insert" / "poses.json"
        ),
        viewer=viewer,
    )
    net = ManipulationPrimitiveNet(net_cfg)
    transition = net.reset()
    print(f"start -> {net.active_primitive}")

    #     1. 外层策略/环境步
    # （net.step()）	10 Hz
    # （每步 0.1 秒）	- 策略网络输出一次动作
    # - 状态机评估一次是否发生跳转（判定各种边）
    # - 相机抓取一帧图片（相机通常为 10~30 FPS）
    # 2. 中层控制器子步
    # （Control Substep）	50 ~ 100 Hz
    # （每步 0.01 秒）	- 机械臂轨迹平滑插值
    # - 逆运动学 (IK) 或末端阻抗力控闭环计算
    # 3. 底层 MuJoCo 物理积分
    # （mj_step）	500 ~ 1000 Hz
    # （每步 1~2 毫秒）	- 刚体动力学数值积分（牛顿-欧拉方程）
    # - 碰撞检测、摩擦力、接触力计算
    # ------------------------------------------------------------------
    # 控制频率层级（由外到内）：
    #   1. 外层策略/环境步  (net.step)    10~30 Hz  状态机评估 + 切换判定
    #   2. 中层控制器子步  (Control)     50~100 Hz  轨迹插值 + IK / 力控
    #   3. 底层物理积分    (mj_step)    500~1000 Hz 动力学 + 碰撞 + 摩擦
    # ------------------------------------------------------------------
    try:
        last_primitive = net.active_primitive
        for step in range(steps):
            transition = net.step(torch.zeros(net.action_dim))
            info = transition[TransitionKey.INFO]

            # 仅在发生原语切换时打印，避免每个 step 都刷屏
            current = net.active_primitive
            if current != last_primitive:
                print(
                    f"step {step:04d}: {info['transition_from']} "
                    f"-> {info['transition_to']} "
                    f"(reason={info['transition_reason']})"
                )
                last_primitive = current

            if net.in_terminal:
                print(f"completed at step {step:04d}: reached terminal primitive '{current}'")
                break
        else:
            print(
                f"stopped after {steps} steps without reaching a terminal primitive "
                f"(active={net.active_primitive})"
            )
    finally:
        net.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MuJoCo pick-and-insert demo.")
    parser.add_argument("--steps", type=int, default=2000, help="max steps to run")
    parser.add_argument(
        "--viewer", action="store_true", help="open the MuJoCo viewer"
    )
    args = parser.parse_args()
    run_demo(steps=args.steps, viewer=args.viewer)
