"""操作原语模块的延迟动态导出接口。"""

from __future__ import annotations

__all__ = [
    "ManipulationPrimitiveConfig",
    "ManipulationPrimitiveProcessorConfig",
    "ManipulationPrimitiveConfig",
    "MoveDeltaPrimitiveConfig",
    "OpenLoopTrajectoryPrimitiveConfig",
    "ManipulationPrimitive",
    "TaskFrame",
    "TASK_FRAME_AXIS_NAMES",
]


def __getattr__(name: str):
    if name in {
        "PrimitiveConfig",
        "ManipulationPrimitiveProcessorConfig",
        "ManipulationPrimitiveConfig",
        "MoveDeltaPrimitiveConfig",
        "OpenLoopTrajectoryPrimitiveConfig",
    }:
        from .config_manipulation_primitive import __dict__ as config_exports

        return config_exports[name]

    if name == "ManipulationPrimitive":
        from .env_manipulation_primitive import ManipulationPrimitive

        return ManipulationPrimitive

    if name in {"TaskFrame", "TASK_FRAME_AXIS_NAMES"}:
        from .task_frame import __dict__ as task_frame_exports

        return task_frame_exports[name]

    raise AttributeError(name)
