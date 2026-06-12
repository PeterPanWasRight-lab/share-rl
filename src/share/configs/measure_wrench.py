from dataclasses import dataclass

from share.configs.rl import MPNetTrainRLServerPipelineConfig


@dataclass(kw_only=True)
class MeasureWrenchConfig(MPNetTrainRLServerPipelineConfig):
    """Top-level config for live end-effector wrench monitoring."""

    sample_hz: float = 20.0
    history_window_s: float = 10.0
    autoscale: bool = True
    force_ylim: tuple[float, float] | None = None
    torque_ylim: tuple[float, float] | None = None
