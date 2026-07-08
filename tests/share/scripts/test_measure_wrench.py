from __future__ import annotations

import pytest

from share.scripts.measure_wrench import (
    WrenchHistory,
    build_wrench_keys,
    read_wrench_sample,
    resolve_monitored_robot_name,
)


def test_resolve_monitored_robot_name_uses_first_robot_in_insertion_order():
    robot_dict = {"main": object(), "secondary": object()}

    assert resolve_monitored_robot_name(robot_dict) == "main"


def test_build_wrench_keys_uses_raw_scalar_observation_names():
    keys = build_wrench_keys("main")

    assert keys == {
        "x": "main.x.ee_wrench",
        "y": "main.y.ee_wrench",
        "z": "main.z.ee_wrench",
        "wx": "main.wx.ee_wrench",
        "wy": "main.wy.ee_wrench",
        "wz": "main.wz.ee_wrench",
    }


def test_read_wrench_sample_uses_transition_observation_scalars():
    keys = build_wrench_keys("main")
    observation = {
        "main.x.ee_wrench": 1.0,
        "main.y.ee_wrench": 2.0,
        "main.z.ee_wrench": 3.0,
        "main.wx.ee_wrench": 4.0,
        "main.wy.ee_wrench": 5.0,
        "main.wz.ee_wrench": 6.0,
    }

    assert read_wrench_sample(observation, keys) == {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "wx": 4.0,
        "wy": 5.0,
        "wz": 6.0,
    }


def test_read_wrench_sample_raises_on_missing_scalar_key():
    keys = build_wrench_keys("main")
    observation = {
        "main.x.ee_wrench": 1.0,
    }

    with pytest.raises(KeyError, match="Missing wrench observation keys"):
        read_wrench_sample(observation, keys)


def test_wrench_history_keeps_only_recent_window():
    history = WrenchHistory(max_samples=3)

    for index in range(5):
        history.append(
            timestamp_s=float(index),
            sample={
                "x": float(index),
                "y": float(index + 1),
                "z": float(index + 2),
                "wx": float(index + 3),
                "wy": float(index + 4),
                "wz": float(index + 5),
            },
        )

    timestamps, values = history.as_arrays()

    assert timestamps.tolist() == [2.0, 3.0, 4.0]
    assert values["x"].tolist() == [2.0, 3.0, 4.0]
    assert values["wz"].tolist() == [7.0, 8.0, 9.0]
