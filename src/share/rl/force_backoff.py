from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


TRANSLATION_AXES = ("x", "y", "z")


@dataclass
class ForceBackoffConfig:
    """Low-rate force-triggered Cartesian backoff for position interfaces."""

    enabled: bool = False
    robot_name: str = "main"
    force_thresholds_n: list[float] = field(
        default_factory=lambda: [20.0, 20.0, 20.0]
    )
    wrench_to_backoff_sign: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )
    exceedance_gain_m: float = 0.0005
    max_backoff_step_m: float = 0.0003
    same_direction_action_scale: float = 0.5
    wrench_filter_alpha: float = 0.25

    def __post_init__(self) -> None:
        if len(self.force_thresholds_n) != 3:
            raise ValueError("force_thresholds_n must contain x, y, and z thresholds.")
        if any(value <= 0 for value in self.force_thresholds_n):
            raise ValueError("force_thresholds_n values must be positive.")
        if len(self.wrench_to_backoff_sign) != 3:
            raise ValueError("wrench_to_backoff_sign must contain x, y, and z signs.")
        if any(value not in (-1.0, 1.0) for value in self.wrench_to_backoff_sign):
            raise ValueError("wrench_to_backoff_sign values must be -1 or +1.")
        if self.exceedance_gain_m <= 0 or self.max_backoff_step_m <= 0:
            raise ValueError("Backoff gain and maximum step must be positive.")
        if not 0.0 <= self.same_direction_action_scale <= 1.0:
            raise ValueError("same_direction_action_scale must be in [0, 1].")
        if not 0.0 < self.wrench_filter_alpha <= 1.0:
            raise ValueError("wrench_filter_alpha must be in (0, 1].")


@dataclass(frozen=True)
class ForceBackoffResult:
    adjusted_action: torch.Tensor
    filtered_force_n: np.ndarray
    triggered_axes: tuple[str, ...]


class ForceBackoffSafetyFilter:
    """Override unsafe translation actions with bounded force-directed backoff."""

    def __init__(self, config: ForceBackoffConfig, *, action_frequency_hz: float):
        if action_frequency_hz <= 0:
            raise ValueError("action_frequency_hz must be positive.")
        self.config = config
        self.action_frequency_hz = float(action_frequency_hz)
        self._wrench_bias: np.ndarray | None = None
        self._filtered_force = np.zeros(3, dtype=np.float64)

    def reset(self, observation: dict[str, Any]) -> None:
        """Capture the unloaded force bias at primitive or episode entry."""
        self._wrench_bias = self._force_from_observation(observation)
        self._filtered_force.fill(0.0)

    def adjust(
        self,
        action: torch.Tensor | np.ndarray,
        observation: dict[str, Any],
    ) -> ForceBackoffResult:
        action_tensor = torch.as_tensor(action)
        adjusted_action = action_tensor.clone()
        if not self.config.enabled:
            return ForceBackoffResult(adjusted_action, self._filtered_force.copy(), ())
        if adjusted_action.numel() < 3:
            raise ValueError("Force backoff requires at least three translation actions.")

        raw_force = self._force_from_observation(observation)
        if self._wrench_bias is None:
            self._wrench_bias = raw_force.copy()
        residual_force = raw_force - self._wrench_bias
        alpha = self.config.wrench_filter_alpha
        self._filtered_force += alpha * (residual_force - self._filtered_force)

        triggered_axes: list[str] = []
        flat_action = adjusted_action.reshape(-1)
        for axis, axis_name in enumerate(TRANSLATION_AXES):
            force = float(self._filtered_force[axis])
            threshold = float(self.config.force_thresholds_n[axis])
            if abs(force) <= threshold:
                continue

            exceedance_ratio = abs(force) / threshold - 1.0
            backoff_step_m = min(
                self.config.exceedance_gain_m * exceedance_ratio,
                self.config.max_backoff_step_m,
            )
            direction = (
                self.config.wrench_to_backoff_sign[axis] * np.sign(force)
            )

            # Policy actions are physical Cartesian velocities (m/s). Apply the
            # Strategy-B rule in per-cycle displacement units so "0.3 mm/frame"
            # keeps the same meaning at 10 Hz and 30 Hz.
            original_velocity = float(flat_action[axis])
            original_step_m = np.clip(
                original_velocity / self.action_frequency_hz,
                -self.config.max_backoff_step_m,
                self.config.max_backoff_step_m,
            )
            adjusted_step_m = direction * backoff_step_m
            if original_step_m * direction > 0.0:
                adjusted_step_m += (
                    self.config.same_direction_action_scale * original_step_m
                )
            flat_action[axis] = adjusted_step_m * self.action_frequency_hz
            triggered_axes.append(axis_name)

        return ForceBackoffResult(
            adjusted_action=adjusted_action,
            filtered_force_n=self._filtered_force.copy(),
            triggered_axes=tuple(triggered_axes),
        )

    def _force_from_observation(self, observation: dict[str, Any]) -> np.ndarray:
        values = []
        for axis in TRANSLATION_AXES:
            key = f"{self.config.robot_name}.{axis}.ee_wrench"
            if key not in observation:
                raise KeyError(
                    f"Force backoff is enabled but observation is missing '{key}'."
                )
            value = observation[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().reshape(-1)[-1].item()
            values.append(float(value))
        force = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(force)):
            raise ValueError("Force backoff received a non-finite wrench sample.")
        return force
