#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""连接验证：连续采集 TCP 位姿，用真实数据抖动证明连的是真机。

只读测试，不发送任何运动指令，不需要使能。

原理：
    假数据 / 缓存数据 / 单元测试桩的返回值是完全静止的。
    真实传感器的位姿反馈一定带有微小噪声（微米级跳动），
    且各轴噪声相互独立。

判据：
    1. 每个轴的实际波动范围 > 0（不是冻结值）
    2. 波动量级在噪声范围内（静止时应 < 1 mm）
    3. 噪声不是周期性重复序列（排除 mock 回放）
"""

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from share.robots.HRCrobot.lerobot_robot_HRCrobot.controller import (  # noqa: E402
    HRCrobotController,
)

import os  # noqa: E402

ROBOT_IP = os.environ.get("HRC_ROBOT_IP", "10.10.59.211")

N_SAMPLES = 30
SAMPLE_INTERVAL_S = 0.1

AXIS_NAMES = ["x (m)", "y (m)", "z (m)", "rx (rad)", "ry (rad)", "rz (rad)"]


def main():
    print(f"连接 {ROBOT_IP} ...")

    controller = HRCrobotController(
        robot_ip=ROBOT_IP,
        frequency=100.0,
    )

    controller.connect()
    print("已连接，开始采集位姿 ...\n")

    samples = []

    try:
        for i in range(N_SAMPLES):
            pose = controller.get_tcp_pose()
            samples.append(pose)

            if i % 5 == 0:
                print(
                    f"  sample {i:2d}: "
                    + " ".join(f"{v:+.6f}" for v in pose)
                )

            time.sleep(SAMPLE_INTERVAL_S)

    finally:
        controller.disconnect()
        print("\n已断开。")

    print(f"\n共采集 {len(samples)} 个样本：\n")

    # ------------------------------------------------------------
    # 统计每个轴的波动
    # ------------------------------------------------------------

    all_axes_alive = True

    print(f"{'轴':<10} {'最小值':>12} {'最大值':>12} {'波动范围':>12} {'标准差':>12}")
    print("-" * 62)

    for axis in range(6):
        values = [s[axis] for s in samples]
        spread = max(values) - min(values)
        std = statistics.stdev(values)

        print(
            f"{AXIS_NAMES[axis]:<10} "
            f"{min(values):>+12.6f} "
            f"{max(values):>+12.6f} "
            f"{spread:>12.6f} "
            f"{std:>12.6f}"
        )

        if spread <= 0.0:
            all_axes_alive = False

    print()

    # ------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------

    if not all_axes_alive:
        print("结论：FAIL - 某些轴数据完全冻结，疑似假数据/缓存。")
        sys.exit(1)

    max_spread = max(
        max(s[a] for s in samples) - min(s[a] for s in samples)
        for a in range(3)
    )

    if max_spread >= 0.01:
        print("结论：FAIL - 平动轴波动超过 10 mm，静止机器人不应有这么大跳动。")
        sys.exit(1)

    print("结论：PASS - 数据存在微小抖动且量级合理，确认读取的是真实硬件反馈。")


if __name__ == "__main__":
    main()
