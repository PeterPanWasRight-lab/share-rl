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
