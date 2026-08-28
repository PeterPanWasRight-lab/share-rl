"""MP-Net 环境核心包。

采用延迟动态导入 (Lazy import) 保持模块轻量化，避免在 LeRobot 初始化时产生循环引用。
"""

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
    "ManipulationPrimitiveNetConfig",
    "ManipulationPrimitiveNet",
]


def __getattr__(name: str):
    if name in {
        "ManipulationPrimitiveConfig",
        "ManipulationPrimitiveProcessorConfig",
        "ManipulationPrimitiveConfig",
        "MoveDeltaPrimitiveConfig",
        "OpenLoopTrajectoryPrimitiveConfig",
        "ManipulationPrimitive",
        "TaskFrame",
        "TASK_FRAME_AXIS_NAMES",
    }:
        from .manipulation_primitive import __dict__ as primitive_exports

        return primitive_exports[name]

    if name in {"ManipulationPrimitiveNetConfig", "ManipulationPrimitiveNet"}:
        from .manipulation_primitive_net import __dict__ as net_exports

        return net_exports[name]

    raise AttributeError(name)
