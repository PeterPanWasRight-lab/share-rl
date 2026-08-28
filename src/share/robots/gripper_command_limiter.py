from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GripperCommandLimiter:
    """Suppress repeated targets and rate-limit target changes."""

    min_interval_s: float = 0.5
    target_tolerance: float = 1e-4
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _target: float | None = field(default=None, init=False, repr=False)
    _last_change_time: float = field(default=float("-inf"), init=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_interval_s < 0.0:
            raise ValueError("min_interval_s must be non-negative.")
        if self.target_tolerance < 0.0:
            raise ValueError("target_tolerance must be non-negative.")

    def synchronize(self, target: float, *, start_cooldown: bool = True) -> float:
        """Align with an externally established target, such as simulation reset."""
        self._target = float(max(0.0, min(1.0, target)))
        self._last_change_time = self.clock() if start_cooldown else float("-inf")
        return self._target

    def filter(self, requested_target: float) -> tuple[float, bool]:
        """Return the safe target and whether a new hardware command is needed."""
        requested_target = float(max(0.0, min(1.0, requested_target)))
        now = self.clock()

        if self._target is None:
            self._target = requested_target
            self._last_change_time = now
            return requested_target, True

        if math.isclose(
            requested_target,
            self._target,
            rel_tol=0.0,
            abs_tol=self.target_tolerance,
        ):
            return self._target, False

        if now - self._last_change_time < self.min_interval_s:
            return self._target, False

        self._target = requested_target
        self._last_change_time = now
        return requested_target, True
