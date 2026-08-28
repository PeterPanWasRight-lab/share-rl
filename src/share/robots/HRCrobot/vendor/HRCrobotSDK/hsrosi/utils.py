#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRC Controller 工具函数

提供各种辅助功能和工具函数。
"""

import os
import math
import time
import ctypes
import platform
from typing import List, Optional, Tuple, Union
from pathlib import Path

from .exceptions import HRCValidationError, HRCLibraryError


def find_library(lib_name: str = "pyhstrajproxy",
                 search_paths: List[str] = None) -> Optional[str]:
    """
    查找动态库文件（跨平台支持）

    Args:
        lib_name: 库文件名（不带扩展名）
        search_paths: 额外搜索路径列表

    Returns:
        找到的库文件完整路径，如果未找到返回None
    """
    system = platform.system().lower()

    # 根据平台确定库文件扩展名和前缀
    if system == "windows":
        lib_extensions = [".dll"]
        lib_prefixes = ["", "lib"]
        # Windows默认搜索路径
        default_paths = [
            ".",  # 当前目录
            "./lib",
            "./bin",
            os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"), "HRC"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "HRC"),
            "C:\\Windows\\System32",
            os.path.expanduser("~\\Documents\\HRC"),
        ]
        # 添加PATH环境变量中的路径
        path_env = os.environ.get("PATH", "")
        default_paths.extend(path_env.split(os.pathsep))

    elif system == "darwin":  # macOS
        lib_extensions = [".dylib", ".so"]
        lib_prefixes = ["lib", ""]
        default_paths = [
            ".",
            "./lib",
            "/usr/local/lib",
            "/usr/lib",
            "/opt/local/lib",
            "/opt/hrc/lib",
            os.path.expanduser("~/lib"),
        ]

    else:  # Linux and other Unix-like systems
        lib_extensions = [".so"]
        lib_prefixes = ["lib", ""]
        default_paths = [
            ".",
            "./lib",
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/opt/hrc/lib",
            os.path.expanduser("~/lib"),
        ]
        # 添加LD_LIBRARY_PATH环境变量中的路径
        ld_path_env = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_path_env:
            default_paths.extend(ld_path_env.split(os.pathsep))

    # 合并搜索路径
    all_paths = default_paths.copy()
    if search_paths:
        all_paths.extend(search_paths)

    # 生成所有可能的库名变体
    lib_variants = []

    # 如果库名已经包含扩展名，直接使用
    if any(lib_name.endswith(ext) for ext in lib_extensions):
        lib_variants.append(lib_name)
    else:
        # 生成不同前缀和扩展名的组合
        for prefix in lib_prefixes:
            for extension in lib_extensions:
                if prefix and not lib_name.startswith(prefix):
                    variant = f"{prefix}{lib_name}{extension}"
                else:
                    variant = f"{lib_name}{extension}"
                lib_variants.append(variant)

    # 在每个路径中搜索每个库名变体
    for path in all_paths:
        if not path or not os.path.isdir(path):
            continue

        for variant in lib_variants:
            full_path = os.path.join(path, variant)
            if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                return os.path.abspath(full_path)

    return None


def validate_joint_angles(angles: List[float],
                          min_angles: List[float] = None,
                          max_angles: List[float] = None) -> bool:
    """
    验证关节角度是否有效

    Args:
        angles: 6个关节角度的列表
        min_angles: 最小角度限制（可选）
        max_angles: 最大角度限制（可选）

    Returns:
        验证是否通过

    Raises:
        HRCValidationError: 验证失败时抛出
    """
    if not isinstance(angles, (list, tuple)):
        raise HRCValidationError("joint_angles", angles, "list or tuple of 6 float values")

    if len(angles) != 6:
        raise HRCValidationError("joint_angles", f"length={len(angles)}", "exactly 6 values")

    # 检查每个角度是否为数字
    for i, angle in enumerate(angles):
        if not isinstance(angle, (int, float)):
            raise HRCValidationError(f"joint_angles[{i}]", angle, "numeric value")

        # 检查角度范围（如果提供了限制）
        if min_angles and i < len(min_angles):
            if angle < min_angles[i]:
                raise HRCValidationError(
                    f"joint_angles[{i}]", angle,
                    f">= {min_angles[i]} degrees"
                )

        if max_angles and i < len(max_angles):
            if angle > max_angles[i]:
                raise HRCValidationError(
                    f"joint_angles[{i}]", angle,
                    f"<= {max_angles[i]} degrees"
                )

    return True


def validate_cartesian_position(position: List[float],
                                min_limits: List[float] = None,
                                max_limits: List[float] = None) -> bool:
    """
    验证笛卡尔坐标是否有效

    Args:
        position: 6个笛卡尔坐标的列表 [x, y, z, rx, ry, rz]
        min_limits: 最小值限制（可选）
        max_limits: 最大值限制（可选）

    Returns:
        验证是否通过

    Raises:
        HRCValidationError: 验证失败时抛出
    """
    if not isinstance(position, (list, tuple)):
        raise HRCValidationError("cartesian_position", position, "list or tuple of 6 float values")

    if len(position) != 6:
        raise HRCValidationError("cartesian_position", f"length={len(position)}", "exactly 6 values")

    # 坐标名称
    coord_names = ["x", "y", "z", "rx", "ry", "rz"]

    # 检查每个坐标
    for i, coord in enumerate(position):
        if not isinstance(coord, (int, float)):
            raise HRCValidationError(f"cartesian_position[{i}] ({coord_names[i]})", coord, "numeric value")

        # 检查坐标范围（如果提供了限制）
        if min_limits and i < len(min_limits):
            if coord < min_limits[i]:
                raise HRCValidationError(
                    f"cartesian_position[{i}] ({coord_names[i]})", coord,
                    f">= {min_limits[i]}"
                )

        if max_limits and i < len(max_limits):
            if coord > max_limits[i]:
                raise HRCValidationError(
                    f"cartesian_position[{i}] ({coord_names[i]})", coord,
                    f"<= {max_limits[i]}"
                )

    return True


def degrees_to_radians(degrees: Union[float, List[float]]) -> Union[float, List[float]]:
    """
    角度转弧度

    Args:
        degrees: 角度值或角度列表

    Returns:
        弧度值或弧度列表
    """
    if isinstance(degrees, (list, tuple)):
        return [math.radians(d) for d in degrees]
    return math.radians(degrees)


def radians_to_degrees(radians: Union[float, List[float]]) -> Union[float, List[float]]:
    """
    弧度转角度

    Args:
        radians: 弧度值或弧度列表

    Returns:
        角度值或角度列表
    """
    if isinstance(radians, (list, tuple)):
        return [math.degrees(r) for r in radians]
    return math.degrees(radians)


def normalize_angle(angle: float, min_angle: float = -180.0, max_angle: float = 180.0) -> float:
    """
    标准化角度到指定范围

    Args:
        angle: 输入角度
        min_angle: 最小角度
        max_angle: 最大角度

    Returns:
        标准化后的角度
    """
    range_size = max_angle - min_angle
    while angle < min_angle:
        angle += range_size
    while angle > max_angle:
        angle -= range_size
    return angle


def calculate_joint_distance(angles1: List[float], angles2: List[float]) -> float:
    """
    计算两个关节位置之间的距离（欧几里德距离）

    Args:
        angles1: 第一个关节位置
        angles2: 第二个关节位置

    Returns:
        距离值
    """
    if len(angles1) != 6 or len(angles2) != 6:
        raise HRCValidationError("joint_angles", "length mismatch", "both positions must have 6 angles")

    return math.sqrt(sum((a1 - a2) ** 2 for a1, a2 in zip(angles1, angles2)))


def calculate_cartesian_distance(pos1: List[float], pos2: List[float]) -> float:
    """
    计算两个笛卡尔位置之间的距离（仅考虑x,y,z坐标）

    Args:
        pos1: 第一个笛卡尔位置
        pos2: 第二个笛卡尔位置

    Returns:
        距离值（毫米）
    """
    if len(pos1) < 3 or len(pos2) < 3:
        raise HRCValidationError("cartesian_position", "insufficient coordinates", "at least x,y,z coordinates required")

    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1[:3], pos2[:3])))


def interpolate_joint_path(start_angles: List[float],
                           end_angles: List[float],
                           num_points: int = 10) -> List[List[float]]:
    """
    在两个关节位置之间生成插值路径

    Args:
        start_angles: 起始关节角度
        end_angles: 结束关节角度
        num_points: 插值点数量

    Returns:
        插值路径点列表
    """
    validate_joint_angles(start_angles)
    validate_joint_angles(end_angles)

    if num_points < 2:
        raise HRCValidationError("num_points", num_points, ">= 2")

    path = []
    for i in range(num_points):
        t = i / (num_points - 1)  # 0 到 1 的插值参数
        interpolated = [
            start + t * (end - start)
            for start, end in zip(start_angles, end_angles)
        ]
        path.append(interpolated)

    return path


def wait_for_condition(condition_func, timeout: float = 10.0, poll_interval: float = 0.1) -> bool:
    """
    等待条件满足

    Args:
        condition_func: 条件检查函数，返回bool
        timeout: 超时时间（秒）
        poll_interval: 轮询间隔（秒）

    Returns:
        条件是否在超时前满足
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(poll_interval)
    return False


def get_system_info() -> dict:
    """
    获取系统信息

    Returns:
        包含系统信息的字典
    """
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture(),
    }


def create_safe_position_limits() -> Tuple[List[float], List[float]]:
    """
    创建安全的关节角度限制

    Returns:
        (最小角度列表, 最大角度列表)
    """
    # 典型的六轴机械臂关节限制（度）
    min_angles = [-180.0, -135.0, -180.0, -180.0, -135.0, -360.0]
    max_angles = [180.0, 135.0, 180.0, 180.0, 135.0, 360.0]

    return min_angles, max_angles


def format_position(position: List[float], position_type: str = "joint") -> str:
    """
    格式化位置信息为可读字符串

    Args:
        position: 位置列表
        position_type: 位置类型 ("joint" 或 "cartesian")

    Returns:
        格式化的字符串
    """
    if position_type.lower() == "joint":
        labels = ["J1", "J2", "J3", "J4", "J5", "J6"]
        unit = "°"
    else:
        labels = ["X", "Y", "Z", "RX", "RY", "RZ"]
        unit = "mm" if len(labels[:3]) else "°"

    formatted_parts = []
    for i, (label, value) in enumerate(zip(labels, position)):
        current_unit = unit if i < 3 or position_type.lower() != "cartesian" else "°"
        formatted_parts.append(f"{label}: {value:.2f}{current_unit}")

    return ", ".join(formatted_parts)