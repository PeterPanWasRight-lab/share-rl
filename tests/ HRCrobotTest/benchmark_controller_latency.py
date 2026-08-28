#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""controller 各接口阻塞时间基准测试。

测量内容（全部不含机械臂运动）：
    1. connect()            建立两条链路的总耗时
    2. get_tcp_pose()       单次位姿读取耗时（含 SDK 网络请求）
    3. get_gripper_position() 纯内存读取耗时
    4. set_gripper()        HSC3 IO 下发耗时（夹爪会真实开合）
    5. disconnect()         断开耗时

结束状态保证：夹爪 = 关闭。
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

ROBOT_IP = "10.10.59.211"

POSE_SAMPLES = 100
GRIPPER_STATE_SAMPLES = 10000


def fmt_stats(times_ms: list[float]) -> str:
    times_ms = sorted(times_ms)
    n = len(times_ms)
    p = lambda q: times_ms[min(int(n * q), n - 1)]
    return (
        f"min={times_ms[0]:.3f}  "
        f"p50={p(0.50):.3f}  "
        f"p95={p(0.95):.3f}  "
        f"p99={p(0.99):.3f}  "
        f"max={times_ms[-1]:.3f}  "
        f"mean={statistics.mean(times_ms):.3f}"
    )


def bench(name: str, func, n: int) -> list[float]:
    times = []

    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000.0)

    print(f"{name:<28} (n={n:>5})  {fmt_stats(times)}  ms")
    return times


def main():
    controller = HRCrobotController(
        robot_ip=ROBOT_IP,
        frequency=100.0,
    )

    # --------------------------------------------------------
    # 1. connect
    # --------------------------------------------------------

    t0 = time.perf_counter()
    controller.connect()
    connect_ms = (time.perf_counter() - t0) * 1000.0
    print(f"{'connect()':<28}            {connect_ms:>8.1f}  ms  (一次性)")

    # --------------------------------------------------------
    # 2. get_tcp_pose
    # --------------------------------------------------------

    bench("get_tcp_pose()", controller.get_tcp_pose, POSE_SAMPLES)

    # 连续读取的持续吞吐（含循环开销）
    t0 = time.perf_counter()
    for _ in range(POSE_SAMPLES):
        controller.get_tcp_pose()
    wall = (time.perf_counter() - t0) * 1000.0
    print(f"{'  -> 持续吞吐':<26}            {wall / POSE_SAMPLES:>8.3f}  ms/次  (≈{1000.0 / (wall / POSE_SAMPLES):.0f} Hz)")

    # --------------------------------------------------------
    # 3. get_gripper_position（纯内存）
    # --------------------------------------------------------

    bench(
        "get_gripper_position()",
        controller.get_gripper_position,
        GRIPPER_STATE_SAMPLES,
    )

    # --------------------------------------------------------
    # 4. set_gripper（真实 IO 动作：开-合交替，结束停在关闭）
    # --------------------------------------------------------

    def grip(v: float) -> None:
        controller.set_gripper(v)

    print()
    # 先开
    bench("set_gripper(0.0) [开]", lambda: grip(0.0), 1)
    time.sleep(0.5)
    # 中间交替 6 次
    for i in range(3):
        bench(f"set_gripper(1.0) [合] #{i+1}", lambda: grip(1.0), 1)
        time.sleep(0.3)
        bench(f"set_gripper(0.0) [开] #{i+1}", lambda: grip(0.0), 1)
        time.sleep(0.3)
    # 最终停在关闭
    bench("set_gripper(1.0) [最终-合]", lambda: grip(1.0), 1)
    time.sleep(0.5)

    # --------------------------------------------------------
    # 5. is_connected（property）
    # --------------------------------------------------------

    bench("is_connected (property)", lambda: controller.is_connected, GRIPPER_STATE_SAMPLES)

    # --------------------------------------------------------
    # 6. disconnect
    # --------------------------------------------------------

    t0 = time.perf_counter()
    controller.disconnect()
    disconnect_ms = (time.perf_counter() - t0) * 1000.0
    print(f"{'disconnect()':<28}            {disconnect_ms:>8.1f}  ms  (一次性)")

    print("\n结束状态：夹爪 = 关闭，机器人已断开。")


if __name__ == "__main__":
    main()
