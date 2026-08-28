"""MP-Net 状态机网络模块的延迟动态导出接口。"""

from __future__ import annotations

__all__ = ["ManipulationPrimitiveNetConfig", "ManipulationPrimitiveNet"]


def __getattr__(name: str):
    if name == "ManipulationPrimitiveNetConfig":
        from .config_manipulation_primitive_net import ManipulationPrimitiveNetConfig

        return ManipulationPrimitiveNetConfig

    if name == "ManipulationPrimitiveNet":
        from .env_manipulation_primitive_net import ManipulationPrimitiveNet

        return ManipulationPrimitiveNet

    raise AttributeError(name)
