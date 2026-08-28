#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""while True 控制循环节拍测试。

验证目标：
    1. 满速 while True 循环 + 每拍 send_action，
       循环周期是否被 servo 限频稳定在 ~10ms。
    2. 循环中 get_observation / send_action 的时间占比。
    3. 总位移严格控制（只走一点点），其余轴锁定不动。

安全设计：
    - 进入时快照当前 TCP 位姿作为 TaskFrame origin，
      旋转设为恒等（z 轴与 base z 对齐，方向可预期）。
    - 每拍下发完整 6D 动作：z 目标每迭代爬 0.1mm
      (≈10mm/s) 直到 +3mm；x/y/rx/ry/rz 全部用当前
      观测值回填（锁定不动）。绝不只发单轴——
      否则缺失轴会被 task_frame.target(全零) 补齐，
      机器人会冲向 base 原点。
    - 到达 +3mm 后继续流式保持 200 拍（不动），
      凑足节拍统计样本，然后退出。
    - 不触碰夹爪，保持当前状态。

运行（先确认 Z 正方向 5mm 行程安全）：

    cd /home/gm/SHaReRL/share-rl
    HRC_TEST_LIVE=1 HRC_TEST_MOVE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
        /home/gm/anaconda3/envs/lerobot/bin/python \\
        -m pytest "tests/ HRCrobotTest/test_while_loop_timing.py" -v -s
"""

import os
import statistics
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from share.robots.HRCrobot import HRCrobot, HRCrobotConfig  # noqa: E402
from share.envs.manipulation_primitive.task_frame import TaskFrame  # noqa: E402

LIVE = os.environ.get("HRC_TEST_LIVE") == "1"
MOVE = LIVE and os.environ.get("HRC_TEST_MOVE") == "1"

AXES = ["x", "y", "z", "rx", "ry", "rz"]

# 运动参数（刻意保守）
TARGET_DZ_M = 0.003        # 总位移 +3mm（沿 base z 向上）
RAMP_STEP_M = 0.0001       # 每迭代爬 0.1mm → ≈10mm/s
HOLD_ITERS = 200           # 到达后保持流式拍摄的拍数
MAX_ITERS = 500            # 绝对上限，防死循环
WALL_TIMEOUT_S = 15.0      # 绝对超时

skip_if_not_move = pytest.mark.skipif(
    not MOVE,
    reason="需要 HRC_TEST_LIVE=1 且 HRC_TEST_MOVE=1（机械臂会移动 3mm）",
)


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(int(len(values) * q), len(values) - 1)]


def print_ms_stats(name: str, values_ms: list[float]) -> None:
    print(
        f"  {name:<22} min={min(values_ms):7.2f}  "
        f"p50={pct(values_ms, 0.5):7.2f}  "
        f"p95={pct(values_ms, 0.95):7.2f}  "
        f"max={max(values_ms):7.2f}  "
        f"mean={statistics.mean(values_ms):7.2f}  ms"
    )


@skip_if_not_move
def test_while_loop_locks_to_servo_period():
    """
    核心断言：

        1. 循环迭代周期中位数 ≈ 10ms（servo 限频生效）
        2. 总位移 ≈ +3mm（z），x/y 漂移 < 1mm
    """
    robot = HRCrobot(HRCrobotConfig())

    try:
        # ------------------------------------------------
        # 连接 + 快照当前位姿
        # ------------------------------------------------

        robot.connect()

        first = robot.get_observation()
        start_base = [first[f"{ax}.ee_pos"] for ax in AXES]
        print("\n起始位姿 (base 系):")
        print(
            "  xyz = "
            + " ".join(f"{v:+.4f}" for v in start_base[:3])
            + "  m"
        )

        # TaskFrame 原点 = 当前位置，旋转恒等（z 轴与 base z 对齐）
        # → z.ee_pos 从 0 开始，方向即 base z 正方向
        robot.set_task_frame(
            TaskFrame(
                origin=[
                    start_base[0],
                    start_base[1],
                    start_base[2],
                    0.0,
                    0.0,
                    0.0,
                ],
                target=[0.0] * 6,
            )
        )

        # ------------------------------------------------
        # while True 循环（用户模式 + 安全封装）
        # ------------------------------------------------

        iter_ms: list[float] = []
        obs_ms: list[float] = []
        send_ms: list[float] = []
        target_z = 0.0
        reached_at = None

        t_start = time.perf_counter()

        while True:
            t_iter = time.perf_counter()

            # ---- 观测 ----
            t0 = time.perf_counter()
            obs = robot.get_observation()
            obs_ms.append((time.perf_counter() - t0) * 1000.0)

            # ---- 退出条件 ----
            now = time.perf_counter()

            reached = obs["z.ee_pos"] >= TARGET_DZ_M - 2e-4
            if reached and reached_at is None:
                reached_at = len(iter_ms)
                print(
                    f"  [到达 +3mm] 迭代 #{len(iter_ms)}, "
                    f"耗时 {(now - t_start):.2f}s"
                )

            # 到达后保持 HOLD_ITERS 拍再退出（凑统计样本）
            if (
                reached
                and reached_at is not None
                and len(iter_ms) - reached_at >= HOLD_ITERS
            ):
                break

            if len(iter_ms) >= MAX_ITERS:
                print("  [警告] 达到最大迭代数上限")
                break

            if now - t_start > WALL_TIMEOUT_S:
                print("  [警告] 达到墙钟超时")
                break

            # ---- 构造动作：z 缓爬，其余轴用当前观测锁定 ----
            if not reached:
                target_z = min(
                    target_z + RAMP_STEP_M,
                    TARGET_DZ_M,
                )
            else:
                target_z = TARGET_DZ_M

            action = {
                f"{ax}.ee_pos": obs[f"{ax}.ee_pos"]
                for ax in AXES
                if ax != "z"
            }
            action["z.ee_pos"] = target_z

            # ---- 下发（内部可能阻塞到 10ms 节拍点）----
            t0 = time.perf_counter()
            robot.send_action(action)
            send_ms.append((time.perf_counter() - t0) * 1000.0)

            iter_ms.append(
                (time.perf_counter() - t_iter) * 1000.0
            )

        wall_s = time.perf_counter() - t_start

        # ------------------------------------------------
        # 报告
        # ------------------------------------------------

        final = robot.get_observation()
        end_base = [
            final[f"{ax}.ee_pos"] + (
                robot.task_frame.origin[i]
                if i < 3
                else 0.0
            )
            for i, ax in enumerate(AXES)
        ]

        print(f"\n循环统计（共 {len(iter_ms)} 迭代, 墙钟 {wall_s:.2f}s）:")
        print_ms_stats("迭代周期", iter_ms)
        print_ms_stats("  get_observation", obs_ms)
        print_ms_stats("  send_action", send_ms)
        print(
            f"  平均下发频率 = "
            f"{len(iter_ms) / wall_s:.1f} Hz (期望 ≈100)"
        )

        dz_base = end_base[2] - start_base[2]
        drift_xy = max(
            abs(end_base[0] - start_base[0]),
            abs(end_base[1] - start_base[1]),
        )
        print(f"\n位移: dz = {dz_base * 1000:+.2f} mm "
              f"(目标 +{TARGET_DZ_M * 1000:.0f} mm), "
              f"xy 漂移 = {drift_xy * 1000:.2f} mm")

        # ------------------------------------------------
        # 断言
        # ------------------------------------------------

        # 1. 循环周期锁定在 servo 周期附近
        median_iter = pct(iter_ms, 0.5)
        assert 9.0 <= median_iter <= 13.0, (
            f"迭代周期中位数 {median_iter:.2f}ms "
            "不在 10ms 附近，限频未生效或循环过慢"
        )

        # 2. 尾部稳定：95% 的迭代不超过 15ms
        p95 = pct(iter_ms, 0.95)
        assert p95 <= 15.0, (
            f"p95 迭代周期 {p95:.2f}ms 超过 15ms"
        )

        # 3. 允许罕见网络离群，但不得普遍超时
        outliers = sum(1 for v in iter_ms if v > 20.0)
        assert outliers <= len(iter_ms) * 0.02, (
            f"超过 20ms 的迭代过多: {outliers}/{len(iter_ms)}"
        )

        # 4. 位移精确且无侧向漂移
        assert 0.002 <= dz_base <= 0.004, (
            f"z 位移 {dz_base * 1000:.2f}mm 偏离目标 3mm"
        )
        assert drift_xy <= 0.001, (
            f"xy 漂移 {drift_xy * 1000:.2f}mm 超过 1mm"
        )

    finally:
        if robot.is_connected:
            robot.disconnect()
