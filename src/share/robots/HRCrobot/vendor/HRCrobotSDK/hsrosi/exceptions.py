#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRC Controller 自定义异常类

定义了HRC控制器可能抛出的各种异常。
"""


class HRCError(Exception):
    """HRC控制器基础异常类"""

    def __init__(self, message: str, error_code: int = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self):
        if self.error_code is not None:
            return f"HRC Error {self.error_code}: {self.message}"
        return f"HRC Error: {self.message}"


class HRCLibraryError(HRCError):
    """动态库加载相关异常"""

    def __init__(self, library_path: str, original_error: Exception = None):
        message = f"Failed to load HRC library: {library_path}"
        if original_error:
            message += f" ({original_error})"
        super().__init__(message)
        self.library_path = library_path
        self.original_error = original_error


class HRCInitError(HRCError):
    """HRC初始化异常"""

    def __init__(self, message: str = "Failed to initialize HRC controller"):
        super().__init__(message)


class HRCConnectionError(HRCError):
    """连接相关异常"""

    def __init__(self, ip: str = None, port: int = None, message: str = None):
        if message is None:
            if ip and port:
                message = f"Failed to connect to HRC controller at {ip}:{port}"
            else:
                message = "Failed to connect to HRC controller"
        super().__init__(message)
        self.ip = ip
        self.port = port


class HRCMovementError(HRCError):
    """运动控制相关异常"""

    def __init__(self, movement_type: str, target_position: list = None, message: str = None):
        if message is None:
            message = f"Failed to execute {movement_type} movement"
            if target_position:
                message += f" to position {target_position}"
        super().__init__(message)
        self.movement_type = movement_type
        self.target_position = target_position


class HRCPositionError(HRCError):
    """位置获取相关异常"""

    def __init__(self, position_type: str):
        message = f"Failed to get {position_type} position from HRC controller"
        super().__init__(message)
        self.position_type = position_type


class HRCValidationError(HRCError):
    """参数验证异常"""

    def __init__(self, parameter_name: str, value, expected_format: str):
        message = f"Invalid {parameter_name}: {value}. Expected: {expected_format}"
        super().__init__(message)
        self.parameter_name = parameter_name
        self.value = value
        self.expected_format = expected_format


class HRCTimeoutError(HRCError):
    """超时异常"""

    def __init__(self, operation: str, timeout_seconds: float):
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        super().__init__(message)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class HRCSafetyError(HRCError):
    """安全相关异常"""

    def __init__(self, message: str, safety_limit_type: str = None):
        super().__init__(message)
        self.safety_limit_type = safety_limit_type


class HRCMotionModeError(HRCError):
    """运动模式相关异常"""

    def __init__(self, message: str, current_mode: str = None, requested_mode: str = None):
        super().__init__(message)
        self.current_mode = current_mode
        self.requested_mode = requested_mode


# 异常映射表，用于从错误代码映射到具体异常
ERROR_CODE_MAP = {
    1: HRCInitError,
    2: HRCConnectionError,
    3: HRCMovementError,
    4: HRCPositionError,
    5: HRCTimeoutError,
    6: HRCSafetyError,
}


def map_error_code(error_code: int, message: str = None) -> HRCError:
    """根据错误代码映射到相应的异常类"""
    exception_class = ERROR_CODE_MAP.get(error_code, HRCError)
    return exception_class(message or f"Error code: {error_code}")


def handle_library_error(func):
    """装饰器：处理库函数调用异常"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            raise HRCLibraryError("Library function call failed", e)
        except Exception as e:
            raise HRCError(f"Unexpected error in {func.__name__}: {e}")
    return wrapper
