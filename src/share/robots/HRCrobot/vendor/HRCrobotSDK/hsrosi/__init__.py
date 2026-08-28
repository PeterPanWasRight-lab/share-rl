#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRC Controller Python Package

A Python ctypes wrapper for HRC robotic arm controller library.
"""

__version__ = "1.0.1"
__author__ = "BINGXIN"
__email__ = "none"

# 导入主要类和函数
from .controller import HRCController
from .exceptions import (
    HRCError,
    HRCConnectionError,
    HRCInitError,
    HRCLibraryError,
    HRCMovementError,
)
from .utils import (
    find_library,
    validate_joint_angles,
    validate_cartesian_position,
    degrees_to_radians,
    radians_to_degrees,
)

# 定义包的公共API
__all__ = [
    # 主要类
    "HRCController",

    # 异常类
    "HRCError",
    "HRCConnectionError",
    "HRCInitError",
    "HRCLibraryError",
    "HRCMovementError",

    # 工具函数
    "find_library",
    "validate_joint_angles",
    "validate_cartesian_position",
    "degrees_to_radians",
    "radians_to_degrees",

    # 版本信息
    "__version__",
]

# 包级别配置
import logging

# 设置默认日志
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# 可选：设置包级别的默认库路径（跨平台）
import platform

_system = platform.system().lower()

if _system == "windows":
    DEFAULT_LIBRARY_PATHS = [
        "pyhstrajproxy.dll",
        "./pyhstrajproxy.dll",
        "./bin/pyhstrajproxy.dll",
        "./lib/pyhstrajproxy.dll",
    ]
elif _system == "darwin":  # macOS
    DEFAULT_LIBRARY_PATHS = [
        "libpyhstrajproxy.dylib",
        "./libpyhstrajproxy.dylib",
        "/usr/local/lib/libpyhstrajproxy.dylib",
        "/opt/hrc/lib/libpyhstrajproxy.dylib",
    ]
else:  # Linux
    DEFAULT_LIBRARY_PATHS = [
        "pyhstrajproxy.so",
        "./pyhstrajproxy.so",
        "/usr/local/lib/pyhstrajproxy.so",
        "/opt/hrc/lib/pyhstrajproxy.so",
    ]

def get_version():
    """获取包版本"""
    return __version__

def set_log_level(level):
    """设置日志级别"""
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 包初始化时的检查
def _check_environment():
    """检查运行环境"""
    import sys

    current_system = platform.system()
    if current_system not in ["Linux", "Windows", "Darwin"]:
        logger.warning(
            f"HRC Controller has been tested on Linux, Windows, and macOS. "
            f"Current platform: {current_system} may not be fully supported."
        )

    if sys.version_info < (3, 7):
        raise RuntimeError(
            f"HRC Controller requires Python 3.7 or later. "
            f"Current version: {sys.version}"
        )

# 执行环境检查
_check_environment()