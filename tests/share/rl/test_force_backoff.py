from __future__ import annotations

import pytest
import torch

from share.rl.force_backoff import ForceBackoffConfig, ForceBackoffSafetyFilter


def _observation(*, x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return {
        "main.x.ee_wrench": x,
        "main.y.ee_wrench": y,
        "main.z.ee_wrench": z,
    }


def _filter(*, direction_sign: float = 1.0) -> ForceBackoffSafetyFilter:
    return ForceBackoffSafetyFilter(
        ForceBackoffConfig(
            enabled=True,
            force_thresholds_n=[40.0, 40.0, 40.0],
            wrench_to_backoff_sign=[direction_sign] * 3,
            wrench_filter_alpha=1.0,
        ),
        action_frequency_hz=10.0,
    )


def test_force_backoff_overrides_action_that_pushes_against_backoff():
    safety_filter = _filter()
    safety_filter.reset(_observation())

    result = safety_filter.adjust(
        torch.tensor([0.0, 0.0, -0.08, 1.0]),
        _observation(z=60.0),
    )

    assert result.triggered_axes == ("z",)
    assert result.adjusted_action[2].item() == pytest.approx(0.0025)
    assert result.adjusted_action[3].item() == pytest.approx(1.0)


def test_force_backoff_caps_base_retraction_at_point_three_mm_per_cycle():
    safety_filter = _filter()
    safety_filter.reset(_observation())

    result = safety_filter.adjust(torch.zeros(4), _observation(z=100.0))

    assert result.adjusted_action[2].item() == pytest.approx(0.003)


def test_force_backoff_keeps_half_of_bounded_same_direction_action():
    safety_filter = _filter()
    safety_filter.reset(_observation())

    result = safety_filter.adjust(
        torch.tensor([0.0, 0.0, 0.002, 0.0]),
        _observation(z=60.0),
    )

    assert result.adjusted_action[2].item() == pytest.approx(0.0035)


def test_force_backoff_supports_calibrated_opposite_sensor_sign():
    safety_filter = _filter(direction_sign=-1.0)
    safety_filter.reset(_observation(z=5.0))

    result = safety_filter.adjust(torch.zeros(4), _observation(z=65.0))

    assert result.adjusted_action[2].item() == pytest.approx(-0.0025)


def test_force_backoff_does_not_modify_actions_below_threshold():
    safety_filter = _filter()
    safety_filter.reset(_observation(z=5.0))
    action = torch.tensor([0.01, -0.02, 0.03, 1.0])

    result = safety_filter.adjust(action, _observation(z=44.0))

    torch.testing.assert_close(result.adjusted_action, action)
    assert result.triggered_axes == ()
