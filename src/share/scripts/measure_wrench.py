import logging
import time
from collections import deque
from dataclasses import asdict
from pprint import pformat
from typing import Any

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.processor import TransitionKey
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from share.configs.measure_wrench import MeasureWrenchConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.teleoperators import TeleopEvents, has_event
from share.utils.logging_utils import log_runtime_frequency

init_logging()

FORCE_AXES = ("x", "y", "z")
TORQUE_AXES = ("rx", "ry", "rz")
AXIS_COLORS = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}


def resolve_monitored_robot_name(robot_dict: dict[str, Any]) -> str:
    """Return the first configured robot name in insertion order."""
    if not robot_dict:
        raise ValueError("No robots configured for wrench monitoring.")
    return next(iter(robot_dict))


def build_wrench_keys(robot_name: str) -> dict[str, str]:
    """Build raw observation keys for the six EE wrench scalars."""
    return {
        axis: f"{robot_name}.{axis}.ee_wrench"
        for axis in (*FORCE_AXES, *TORQUE_AXES)
    }


def to_scalar(value: Any) -> float:
    """Normalize raw observation values to plain floats."""
    if isinstance(value, torch.Tensor):
        return float(value.item()) if value.ndim == 0 else float(value.flatten()[0].item())
    if isinstance(value, np.ndarray):
        return float(value.item()) if value.ndim == 0 else float(value.reshape(-1)[0])
    return float(value)


class WrenchHistory:
    """Fixed-size rolling history for six wrench channels."""

    def __init__(self, max_samples: int):
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        self.max_samples = max_samples
        self.timestamps = deque(maxlen=max_samples)
        self.values = {
            axis: deque(maxlen=max_samples)
            for axis in (*FORCE_AXES, *TORQUE_AXES)
        }

    def append(self, timestamp_s: float, sample: dict[str, float]) -> None:
        self.timestamps.append(float(timestamp_s))
        for axis, series in self.values.items():
            series.append(float(sample[axis]))

    def as_arrays(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        timestamps = np.asarray(self.timestamps, dtype=np.float32)
        values = {
            axis: np.asarray(series, dtype=np.float32)
            for axis, series in self.values.items()
        }
        return timestamps, values


def create_plot(
    robot_name: str,
    autoscale: bool,
    force_ylim: tuple[float, float] | None,
    torque_ylim: tuple[float, float] | None,
):
    """Create the 3x2 live wrench figure and line handles."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex=True)
    fig.canvas.manager.set_window_title(f"Wrench Monitor: {robot_name}")
    fig.suptitle(f"Live EE Wrench for {robot_name}")

    line_handles: dict[str, Any] = {}
    row_axes = zip(FORCE_AXES, TORQUE_AXES)
    for row_index, (force_axis, torque_axis) in enumerate(row_axes):
        force_ax = axes[row_index, 0]
        torque_ax = axes[row_index, 1]

        (force_line,) = force_ax.plot([], [], color=AXIS_COLORS[force_axis], linewidth=1.8)
        (torque_line,) = torque_ax.plot([], [], color=AXIS_COLORS[force_axis], linewidth=1.8)
        line_handles[force_axis] = force_line
        line_handles[torque_axis] = torque_line

        force_ax.set_title(f"Force {force_axis.upper()}")
        force_ax.set_ylabel("Force")
        force_ax.grid(True, alpha=0.3)

        torque_ax.set_title(f"Torque {force_axis.upper()}")
        torque_ax.set_ylabel("Torque")
        torque_ax.grid(True, alpha=0.3)

        if not autoscale and force_ylim is not None:
            force_ax.set_ylim(*force_ylim)
        if not autoscale and torque_ylim is not None:
            torque_ax.set_ylim(*torque_ylim)

    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    fig.tight_layout()
    return fig, axes, line_handles


def update_plot(
    axes,
    line_handles: dict[str, Any],
    history: WrenchHistory,
    autoscale: bool,
    force_ylim: tuple[float, float] | None,
    torque_ylim: tuple[float, float] | None,
) -> None:
    """Refresh all subplot traces from the rolling history."""
    timestamps, values = history.as_arrays()
    if timestamps.size == 0:
        return

    t0 = float(timestamps[0])
    times = timestamps - t0

    for axis, line in line_handles.items():
        line.set_data(times, values[axis])

    for row_index, force_axis in enumerate(FORCE_AXES):
        force_ax = axes[row_index, 0]
        torque_ax = axes[row_index, 1]

        force_ax.set_xlim(float(times[0]), float(times[-1]) if times.size > 1 else max(1.0, float(times[0]) + 1.0))
        torque_ax.set_xlim(float(times[0]), float(times[-1]) if times.size > 1 else max(1.0, float(times[0]) + 1.0))

        if autoscale:
            force_ax.relim()
            force_ax.autoscale_view(scalex=False, scaley=True)
            torque_ax.relim()
            torque_ax.autoscale_view(scalex=False, scaley=True)
        else:
            if force_ylim is not None:
                force_ax.set_ylim(*force_ylim)
            if torque_ylim is not None:
                torque_ax.set_ylim(*torque_ylim)


def read_wrench_sample(observation: dict[str, Any], wrench_keys: dict[str, str]) -> dict[str, float]:
    """Extract the six wrench scalars from a transition observation."""
    missing = [key for key in wrench_keys.values() if key not in observation]
    if missing:
        raise KeyError(f"Missing wrench observation keys: {missing}")
    return {
        axis: to_scalar(observation[key])
        for axis, key in wrench_keys.items()
    }


@parser.wrap()
def measure_wrench(cfg: MeasureWrenchConfig) -> None:
    import matplotlib.pyplot as plt

    logging.info(pformat(asdict(cfg)))

    if cfg.sample_hz <= 0:
        raise ValueError("sample_hz must be positive.")
    if cfg.history_window_s <= 0:
        raise ValueError("history_window_s must be positive.")

    mp_net = ManipulationPrimitiveNet(cfg.env)
    mp_net.set_step_info({TeleopEvents.IS_INTERVENTION: True})
    robot_name = resolve_monitored_robot_name(mp_net.robot_dict)
    wrench_keys = build_wrench_keys(robot_name)

    max_samples = max(2, int(np.ceil(cfg.sample_hz * cfg.history_window_s)))
    history = WrenchHistory(max_samples=max_samples)

    fig, axes, line_handles = create_plot(
        robot_name=robot_name,
        autoscale=cfg.autoscale,
        force_ylim=cfg.force_ylim,
        torque_ylim=cfg.torque_ylim,
    )

    try:
        plt.show(block=False)
        transition = mp_net.reset()
        observation = transition[TransitionKey.OBSERVATION]
        history.append(time.perf_counter(), read_wrench_sample(observation, wrench_keys))

        while plt.fignum_exists(fig.number):
            start_loop_t = time.perf_counter()
            zero_action = torch.zeros(mp_net.action_dim, dtype=torch.float32)
            transition = mp_net.step(zero_action)
            observation = transition[TransitionKey.OBSERVATION]
            info = transition.get(TransitionKey.INFO, {})
            history.append(time.perf_counter(), read_wrench_sample(observation, wrench_keys))

            update_plot(
                axes=axes,
                line_handles=line_handles,
                history=history,
                autoscale=cfg.autoscale,
                force_ylim=cfg.force_ylim,
                torque_ylim=cfg.torque_ylim,
            )
            fig.canvas.draw_idle()
            plt.pause(0.05)

            if has_event(info, TeleopEvents.STOP_RECORDING):
                break

            if (
                transition.get(TransitionKey.DONE, False)
                or transition.get(TransitionKey.TRUNCATED, False)
                or has_event(info, TeleopEvents.RERECORD_EPISODE)
            ):
                transition = mp_net.reset()
                observation = transition[TransitionKey.OBSERVATION]
                history.append(time.perf_counter(), read_wrench_sample(observation, wrench_keys))
                continue

            dt_load = time.perf_counter() - start_loop_t
            precise_sleep(max(0.0, 1 / cfg.sample_hz - dt_load))
            dt_loop = time.perf_counter() - start_loop_t
            log_runtime_frequency(
                prefix="MEASURE_WRENCH",
                primitive=mp_net.active_primitive,
                task=mp_net.active_primitive,
                loop_dt_s=dt_loop,
                work_dt_s=dt_load,
                work_label="step",
            )
    finally:
        plt.close(fig)
        mp_net.close()


if __name__ == "__main__":
    import experiments
    measure_wrench()
