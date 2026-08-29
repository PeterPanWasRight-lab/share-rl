from __future__ import annotations

import pytest

from share.robots.gripper_command_limiter import GripperCommandLimiter


def test_gripper_limiter_suppresses_repeats_and_fast_reversals():
    now = [0.0]
    limiter = GripperCommandLimiter(min_interval_s=0.5, clock=lambda: now[0])

    assert limiter.filter(1.0) == (1.0, True)
    assert limiter.filter(1.0) == (1.0, False)

    now[0] = 0.49
    assert limiter.filter(0.0) == (1.0, False)

    now[0] = 0.5
    assert limiter.filter(0.0) == (0.0, True)


def test_gripper_limiter_clips_targets_and_validates_interval():
    limiter = GripperCommandLimiter(min_interval_s=0.0, clock=lambda: 0.0)
    assert limiter.filter(2.0) == (1.0, True)
    assert limiter.filter(-1.0) == (0.0, True)

    with pytest.raises(ValueError, match="non-negative"):
        GripperCommandLimiter(min_interval_s=-0.1)


def test_gripper_limiter_suppresses_identical_target_even_after_long_interval():
    now = [0.0]
    limiter = GripperCommandLimiter(min_interval_s=0.5, clock=lambda: now[0])

    assert limiter.filter(1.0) == (1.0, True)

    # 目标没变时，无论隔多久都不应重复下发硬件命令。
    now[0] = 100.0
    assert limiter.filter(1.0) == (1.0, False)
    now[0] = 1000.0
    assert limiter.filter(1.0 + 1e-9) == (1.0, False)


def test_gripper_limiter_target_tolerance_bounds():
    now = [0.0]
    limiter = GripperCommandLimiter(
        min_interval_s=1.0,
        target_tolerance=1e-4,
        clock=lambda: now[0],
    )

    assert limiter.filter(0.5) == (0.5, True)

    # 容差内的微小变化视为同一目标（即使冷却期已过）。
    now[0] = 2.0
    assert limiter.filter(0.5 + 5e-5) == (0.5, False)
    assert limiter.filter(0.5 - 5e-5) == (0.5, False)

    # 超出容差且冷却期已过 → 正常变更。
    now[0] = 3.0
    assert limiter.filter(0.6) == (0.6, True)


def test_gripper_limiter_zero_interval_passes_every_distinct_change():
    now = [0.0]
    limiter = GripperCommandLimiter(min_interval_s=0.0, clock=lambda: now[0])

    assert limiter.filter(1.0) == (1.0, True)
    assert limiter.filter(0.0) == (0.0, True)
    assert limiter.filter(1.0) == (1.0, True)
    assert limiter.filter(1.0) == (1.0, False)


def test_gripper_limiter_negative_tolerance_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        GripperCommandLimiter(min_interval_s=0.5, target_tolerance=-1e-4)


def test_gripper_limiter_synchronize_seeds_state_without_hardware_send():
    now = [0.0]
    limiter = GripperCommandLimiter(min_interval_s=0.5, clock=lambda: now[0])

    # 模拟仿真 reset / 外部已建立的目标：直接对齐状态。
    assert limiter.synchronize(1.0) == 1.0

    # 对齐目标 == 当前目标 → 不发。
    assert limiter.filter(1.0) == (1.0, False)

    # 对齐后立即反向命令仍在冷却期内 → 抑制并持有对齐目标。
    now[0] = 0.2
    assert limiter.filter(0.0) == (1.0, False)

    # 冷却期过后恢复正常的变更放行。
    now[0] = 0.5
    assert limiter.filter(0.0) == (0.0, True)


def test_gripper_limiter_synchronize_without_cooldown_allows_immediate_change():
    now = [0.0]
    limiter = GripperCommandLimiter(min_interval_s=0.5, clock=lambda: now[0])

    limiter.synchronize(1.0, start_cooldown=False)

    # start_cooldown=False：对齐不进入冷却，立即变更即刻放行。
    assert limiter.filter(0.0) == (0.0, True)


def test_gripper_limiter_synchronize_clamps_target():
    limiter = GripperCommandLimiter(min_interval_s=0.0, clock=lambda: 0.0)

    assert limiter.synchronize(5.0) == 1.0
    assert limiter.filter(1.0) == (1.0, False)

    limiter.synchronize(-2.0)
    assert limiter.filter(0.0) == (0.0, False)
