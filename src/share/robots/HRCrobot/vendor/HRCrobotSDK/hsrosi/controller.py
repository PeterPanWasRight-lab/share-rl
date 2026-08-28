#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRC 机械臂控制器主模块

提供HRC机械臂的Python控制接口。
"""

import ctypes
import logging
import platform
from typing import List, Optional, Tuple

from .exceptions import (
    HRCError,
    HRCConnectionError,
    HRCInitError,
    HRCLibraryError,
    HRCMovementError,
    HRCPositionError,
    HRCMotionModeError,
    handle_library_error,
)
from .utils import (
    find_library,
    validate_joint_angles,
    validate_cartesian_position,
    create_safe_position_limits,
    format_position,
)


class HRCController:
    """HRC 机械臂控制器的 Python 封装类"""

    def __init__(self, lib_path: str = None, enable_safety_limits: bool = True):
        """
        初始化 HRC 控制器

        Args:
            lib_path: 动态库路径，如果为None则自动查找
            enable_safety_limits: 是否启用安全限制检查
        """
        self._lib = None
        self._connected = False
        self._initialized = False
        self._enable_safety_limits = enable_safety_limits
        self._min_joint_limits, self._max_joint_limits = create_safe_position_limits()
        
        # 运动模式状态：None表示未设置，"joint"表示关节空间，"cartesian"表示笛卡尔空间
        self._motion_mode = None

        # 设置日志
        self._logger = logging.getLogger(__name__)

        # 加载库并设置函数签名
        self._load_library(lib_path)
        self._setup_function_signatures()

        self._logger.info("HRC Controller initialized")

    def _load_library(self, lib_path: str = None):
        """加载动态库（跨平台支持）"""
        if lib_path is None:
            lib_path = find_library("pyhstrajproxy")
            if lib_path is None:
                system = platform.system()
                lib_ext = ".dll" if system == "Windows" else (".dylib" if system == "Darwin" else ".so")
                raise HRCLibraryError(f"pyhstrajproxy{lib_ext}",
                                      Exception(f"Library not found in standard paths for {system}"))

        try:
            system = platform.system().lower()

            if system == "windows":
                # Windows下尝试不同的加载方式
                try:
                    # 首先尝试CDLL（C调用约定）
                    self._lib = ctypes.CDLL(lib_path)
                except OSError:
                    try:
                        # 如果失败，尝试WinDLL（Windows标准调用约定）
                        self._lib = ctypes.WinDLL(lib_path)
                        self._logger.info("Loaded library using WinDLL (stdcall)")
                    except OSError:
                        # 最后尝试直接加载
                        self._lib = ctypes.windll.LoadLibrary(lib_path)
                        self._logger.info("Loaded library using windll.LoadLibrary")
            else:
                # Linux/macOS使用CDLL
                self._lib = ctypes.CDLL(lib_path)

            self._logger.info(f"Successfully loaded library: {lib_path}")

        except OSError as e:
            raise HRCLibraryError(lib_path, e)

    def _setup_function_signatures(self):
        """设置函数签名"""
        try:
            # void HRC_init();
            self._lib.HRC_init.argtypes = []
            self._lib.HRC_init.restype = None

            # bool HRC_connect(const char* ip, int port);
            self._lib.HRC_connect.argtypes = [ctypes.c_char_p, ctypes.c_int]
            self._lib.HRC_connect.restype = ctypes.c_bool

            # bool HRC_is_connect();
            self._lib.HRC_is_connect.argtypes = []
            self._lib.HRC_is_connect.restype = ctypes.c_bool

            # void HRC_disconnect();
            self._lib.HRC_disconnect.argtypes = []
            self._lib.HRC_disconnect.restype = None

            # bool HRC_start_j_pos(double j1, double j2, double j3, double j4, double j5, double j6);
            self._lib.HRC_start_j_pos.argtypes = [ctypes.c_double] * 6
            self._lib.HRC_start_j_pos.restype = ctypes.c_bool

            # bool HRC_move_to_j_pos(double j1, double j2, double j3, double j4, double j5, double j6);
            self._lib.HRC_move_to_j_pos.argtypes = [ctypes.c_double] * 6
            self._lib.HRC_move_to_j_pos.restype = ctypes.c_bool

            # bool HRC_start_d_pos(double j1, double j2, double j3, double j4, double j5, double j6);
            self._lib.HRC_start_d_pos.argtypes = [ctypes.c_double] * 6
            self._lib.HRC_start_d_pos.restype = ctypes.c_bool

            # bool HRC_move_to_d_pos(double j1, double j2, double j3, double j4, double j5, double j6);
            self._lib.HRC_move_to_d_pos.argtypes = [ctypes.c_double] * 6
            self._lib.HRC_move_to_d_pos.restype = ctypes.c_bool

            # double* HRC_get_j_pos();
            self._lib.HRC_get_j_pos.argtypes = []
            self._lib.HRC_get_j_pos.restype = ctypes.POINTER(ctypes.c_double)

            # double* HRC_get_d_pos();
            self._lib.HRC_get_d_pos.argtypes = []
            self._lib.HRC_get_d_pos.restype = ctypes.POINTER(ctypes.c_double)

            # bool HRC_setAbort();
            self._lib.HRC_setAbort.argtypes = []
            self._lib.HRC_setAbort.restype = ctypes.c_bool

        except AttributeError as e:
            raise HRCLibraryError("Function signature setup failed", e)

    @handle_library_error
    def init(self):
        """初始化HRC"""
        try:
            self._lib.HRC_init()
            self._initialized = True
            self._logger.info("HRC initialized successfully")
        except Exception as e:
            raise HRCInitError(f"Failed to initialize HRC: {e}")

    @handle_library_error
    def connect(self, ip: str, port: int, motion_mode: str = "joint") -> bool:
        """
        连接到HRC控制器并自动设置运动模式

        Args:
            ip: IP地址
            port: 端口号
            motion_mode: 运动模式，"joint"表示关节空间，"cartesian"表示笛卡尔空间，默认为"joint"

        Returns:
            连接是否成功

        Raises:
            HRCConnectionError: 连接失败时抛出
            ValueError: 运动模式参数无效时抛出
        """
        if not self._initialized:
            raise HRCInitError("HRC must be initialized before connecting")
        
        # 验证运动模式参数
        if motion_mode not in ["joint", "cartesian"]:
            raise ValueError(f"无效的运动模式: {motion_mode}，必须是'joint'或'cartesian'")

        try:
            ip_bytes = ip.encode('utf-8')
            result = self._lib.HRC_connect(ip_bytes, port)

            if result:
                self._connected = True
                self._logger.info(f"Connected to HRC at {ip}:{port}")
                
                # 根据运动模式自动调用start
                if motion_mode == "joint":
                    # 获取当前关节位置并启动关节空间运动
                    current_pos = self.get_joint_position()
                    self.start_joint_position(current_pos)
                else:  # cartesian
                    # 获取当前笛卡尔位置并启动笛卡尔空间运动
                    current_pos = self.get_cartesian_position()
                    self.start_cartesian_position(current_pos)
                
                self._logger.info(f"Motion mode automatically set to: {motion_mode}")
            else:
                self._logger.error(f"Failed to connect to HRC at {ip}:{port}")
                raise HRCConnectionError(ip, port)

            return result
        except Exception as e:
            if isinstance(e, (HRCConnectionError, ValueError, HRCMotionModeError)):
                raise
            raise HRCConnectionError(ip, port, f"Connection error: {e}")

    @handle_library_error
    def is_connected(self) -> bool:
        """检查是否已连接"""
        if not self._initialized:
            return False

        try:
            connected = self._lib.HRC_is_connect()
            self._connected = connected
            return connected
        except Exception:
            self._connected = False
            return False

    @handle_library_error
    def disconnect(self):
        """断开连接"""
        if self._connected:
            try:
                self._lib.HRC_disconnect()
                self._connected = False
                self._logger.info("Disconnected from HRC")
            except Exception as e:
                self._logger.error(f"Error during disconnection: {e}")
                raise HRCConnectionError(message=f"Disconnection error: {e}")

    def _validate_connection(self):
        """验证连接状态"""
        if not self._initialized:
            raise HRCInitError("HRC not initialized")
        if not self.is_connected():
            raise HRCConnectionError(message="Not connected to HRC controller")

    def _validate_joint_movement(self, joint_angles: List[float]):
        """验证关节运动参数"""
        if self._enable_safety_limits:
            validate_joint_angles(joint_angles, self._min_joint_limits, self._max_joint_limits)
        else:
            validate_joint_angles(joint_angles)

    def _validate_cartesian_movement(self, cartesian_pos: List[float]):
        """验证笛卡尔运动参数"""
        validate_cartesian_position(cartesian_pos)

    @handle_library_error
    def start_joint_position(self, joint_angles: List[float]) -> bool:
        """
        启动关节位置运动 (单位：角度)
        
        此函数将运动模式设置为关节空间模式。一旦设置,该实例只能使用关节空间运动函数(move_to_joint_position),
        不能使用笛卡尔空间运动函数(move_to_cartesian_position)。

        Args:
            joint_angles: 6个关节角度的列表

        Returns:
            操作是否成功

        Raises:
            HRCMovementError: 运动失败时抛出
            HRCMotionModeError: 如果已经设置了不同的运动模式
        """
        self._validate_connection()
        self._validate_joint_movement(joint_angles)
        
        # 检查运动模式
        if self._motion_mode is not None and self._motion_mode != "joint":
            raise HRCMotionModeError(
                f"无法启动关节空间运动: 当前实例已设置为{self._motion_mode}运动模式。"
                "请创建新的控制器实例以使用不同的运动模式。",
                current_mode=self._motion_mode,
                requested_mode="joint"
            )

        try:
            result = not self._lib.HRC_start_j_pos(*joint_angles)
            if result:
                self._motion_mode = "joint"
                self._logger.info(f"Started joint movement: {format_position(joint_angles, 'joint')}")
                self._logger.info("Motion mode set to: joint")
            else:
                raise HRCMovementError("joint start", joint_angles)
            return result
        except Exception as e:
            if isinstance(e, (HRCMovementError, HRCMotionModeError)):
                raise
            raise HRCMovementError("joint start", joint_angles, str(e))

    @handle_library_error
    def move_to_joint_position(self, joint_angles: List[float]) -> bool:
        """
        移动到关节位置 (单位：角度)
        
        必须先调用start_joint_position设置运动模式后才能使用此函数。

        Args:
            joint_angles: 6个关节角度的列表

        Returns:
            操作是否成功

        Raises:
            HRCMovementError: 运动失败时抛出
            HRCMotionModeError: 如果未设置运动模式或运动模式不匹配
        """
        self._validate_connection()
        self._validate_joint_movement(joint_angles)
        
        # 检查运动模式是否已设置且正确
        if self._motion_mode is None:
            raise HRCMotionModeError(
                "无法执行关节空间运动: 请先调用start_joint_position()设置运动模式",
                current_mode=None,
                requested_mode="joint"
            )
        
        if self._motion_mode != "joint":
            raise HRCMotionModeError(
                f"无法执行关节空间运动: 当前运动模式为{self._motion_mode}。"
                "请使用move_to_cartesian_position()或创建新的控制器实例。",
                current_mode=self._motion_mode,
                requested_mode="joint"
            )

        try:
            result = not self._lib.HRC_move_to_j_pos(*joint_angles)
            if result:
                self._logger.info(f"Moved to joint position: {format_position(joint_angles, 'joint')}")
            else:
                raise HRCMovementError("joint move", joint_angles)
            return result
        except Exception as e:
            if isinstance(e, (HRCMovementError, HRCMotionModeError)):
                raise
            raise HRCMovementError("joint move", joint_angles, str(e))

    @handle_library_error
    def start_cartesian_position(self, cartesian_pos: List[float]) -> bool:
        """
        启动笛卡尔位置运动 (单位：mm)
        
        此函数将运动模式设置为笛卡尔空间模式。一旦设置,该实例只能使用笛卡尔空间运动函数(move_to_cartesian_position),
        不能使用关节空间运动函数(move_to_joint_position)。

        Args:
            cartesian_pos: 6个笛卡尔坐标的列表 [x, y, z, rx, ry, rz]

        Returns:
            操作是否成功

        Raises:
            HRCMovementError: 运动失败时抛出
            HRCMotionModeError: 如果已经设置了不同的运动模式
        """
        self._validate_connection()
        self._validate_cartesian_movement(cartesian_pos)
        
        # 检查运动模式
        if self._motion_mode is not None and self._motion_mode != "cartesian":
            raise HRCMotionModeError(
                f"无法启动笛卡尔空间运动: 当前实例已设置为{self._motion_mode}运动模式。"
                "请创建新的控制器实例以使用不同的运动模式。",
                current_mode=self._motion_mode,
                requested_mode="cartesian"
            )

        try:
            result = not self._lib.HRC_start_d_pos(*cartesian_pos)
            if result:
                self._motion_mode = "cartesian"
                self._logger.info(f"Started cartesian movement: {format_position(cartesian_pos, 'cartesian')}")
                self._logger.info("Motion mode set to: cartesian")
            else:
                raise HRCMovementError("cartesian start", cartesian_pos)
            return result
        except Exception as e:
            if isinstance(e, (HRCMovementError, HRCMotionModeError)):
                raise
            raise HRCMovementError("cartesian start", cartesian_pos, str(e))

    @handle_library_error
    def move_to_cartesian_position(self, cartesian_pos: List[float]) -> bool:
        """
        移动到笛卡尔位置 (单位：mm)
        
        必须先调用start_cartesian_position设置运动模式后才能使用此函数。

        Args:
            cartesian_pos: 6个笛卡尔坐标的列表 [x, y, z, rx, ry, rz]

        Returns:
            操作是否成功

        Raises:
            HRCMovementError: 运动失败时抛出
            HRCMotionModeError: 如果未设置运动模式或运动模式不匹配
        """
        self._validate_connection()
        self._validate_cartesian_movement(cartesian_pos)
        
        # 检查运动模式是否已设置且正确
        if self._motion_mode is None:
            raise HRCMotionModeError(
                "无法执行笛卡尔空间运动: 请先调用start_cartesian_position()设置运动模式",
                current_mode=None,
                requested_mode="cartesian"
            )
        
        if self._motion_mode != "cartesian":
            raise HRCMotionModeError(
                f"无法执行笛卡尔空间运动: 当前运动模式为{self._motion_mode}。"
                "请使用move_to_joint_position()或创建新的控制器实例。",
                current_mode=self._motion_mode,
                requested_mode="cartesian"
            )

        try:
            result = not self._lib.HRC_move_to_d_pos(*cartesian_pos)
            if result:
                self._logger.debug(f"Moved to cartesian position: {format_position(cartesian_pos, 'cartesian')}")
            else:
                raise HRCMovementError("cartesian move", cartesian_pos)
            return result
        except Exception as e:
            if isinstance(e, (HRCMovementError, HRCMotionModeError)):
                raise
            raise HRCMovementError("cartesian move", cartesian_pos, str(e))

    @handle_library_error
    def get_joint_position(self) -> List[float]:
        """
        获取当前关节位置 (单位：角度)

        Returns:
            6个关节角度的列表

        Raises:
            HRCPositionError: 获取位置失败时抛出
        """
        self._validate_connection()

        try:
            ptr = self._lib.HRC_get_j_pos()
            if not ptr:
                raise HRCPositionError("joint")

            # 从指针读取6个double值
            result = []
            for i in range(6):
                result.append(ptr[i])

            self._logger.debug(f"Current joint position: {format_position(result, 'joint')}")
            return result
        except Exception as e:
            if isinstance(e, HRCPositionError):
                raise
            raise HRCPositionError("joint")

    @handle_library_error
    def get_cartesian_position(self) -> List[float]:
        """
        获取当前笛卡尔位置 (单位：mm)

        Returns:
            6个笛卡尔坐标的列表 [x, y, z, rx, ry, rz]

        Raises:
            HRCPositionError: 获取位置失败时抛出
        """
        self._validate_connection()

        try:
            ptr = self._lib.HRC_get_d_pos()
            if not ptr:
                raise HRCPositionError("cartesian")

            # 从指针读取6个double值
            result = []
            for i in range(6):
                result.append(ptr[i])

            self._logger.debug(f"Current cartesian position: {format_position(result, 'cartesian')}")
            return result
        except Exception as e:
            if isinstance(e, HRCPositionError):
                raise
            raise HRCPositionError("cartesian")

    @handle_library_error
    def abort(self) -> bool:
        """
        中止当前运动

        Returns:
            操作是否成功
        """
        self._validate_connection()

        try:
            result = not self._lib.HRC_setAbort()
            if result:
                self._logger.info("Movement aborted successfully")
            else:
                self._logger.warning("Failed to abort movement")
            return result
        except Exception as e:
            self._logger.error(f"Error during abort: {e}")
            raise HRCError(f"Abort failed: {e}")

    def set_safety_limits(self, min_limits: List[float], max_limits: List[float]):
        """
        设置安全限制

        Args:
            min_limits: 最小关节角度限制
            max_limits: 最大关节角度限制
        """
        validate_joint_angles(min_limits)
        validate_joint_angles(max_limits)

        self._min_joint_limits = min_limits.copy()
        self._max_joint_limits = max_limits.copy()
        self._logger.info("Safety limits updated")

    def enable_safety_limits(self, enable: bool = True):
        """启用或禁用安全限制检查"""
        self._enable_safety_limits = enable
        status = "enabled" if enable else "disabled"
        self._logger.info(f"Safety limits {status}")

    def get_status(self) -> dict:
        """
        获取控制器状态

        Returns:
            包含状态信息的字典
        """
        return {
            "initialized": self._initialized,
            "connected": self.is_connected(),
            "safety_limits_enabled": self._enable_safety_limits,
            "library_loaded": self._lib is not None,
            "motion_mode": self._motion_mode,
        }

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        try:
            if self.is_connected():
                self.disconnect()
        except Exception as e:
            self._logger.error(f"Error during cleanup: {e}")

    def __repr__(self):
        """字符串表示"""
        status = self.get_status()
        return (f"HRCController(initialized={status['initialized']}, "
                f"connected={status['connected']}, "
                f"safety_limits={status['safety_limits_enabled']}, "
                f"motion_mode={status['motion_mode']})")
