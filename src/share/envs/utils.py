import copy
import types
from typing import get_origin, Union, get_args, Any

import numpy as np
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.utils.constants import REWARD, DONE

from share.envs.manipulation_primitive.task_frame import ControlSpace
from share.utils.transformation_utils import get_robot_pose_from_observation, task_pose_to_world_pose, world_pose_to_task_pose

# 增量平移和旋转动作的常见字段名集合（用于检测是否为相对增量型设备/策略）
DELTA_TRANSLATION_ACTION_NAMES = {
    "delta_x",
    "delta_y",
    "delta_z",
    "x.vel",
    "y.vel",
    "z.vel",
}
DELTA_ROTATION_ACTION_NAMES = {
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "rx.vel",
    "ry.vel",
    "rz.vel",
}
DELTA_AUXILIARY_ACTION_NAMES = {
    GRIPPER_KEY,
    f"{GRIPPER_KEY}.pos",
}
DELTA_ACTION_NAMES = (
    DELTA_TRANSLATION_ACTION_NAMES |
    DELTA_ROTATION_ACTION_NAMES |
    DELTA_AUXILIARY_ACTION_NAMES
)


def check_task_frame_robot(robot_dict: dict[str, "Robot"]) -> dict[str, bool]:
    """检查配置中的各个机械臂接口是否支持直接接收和下发 TaskFrame 任务坐标系。"""
    is_task_frame_robot = {}
    for name, r in robot_dict.items():
        is_task_frame_robot[name] = hasattr(r, "set_task_frame")

    return is_task_frame_robot


def get_teleoperator_action_names(teleoperator: "Teleoperator") -> set[str]:
    """获取遥操作输入设备所输出的所有动作特征名称集合。"""
    action_features = getattr(teleoperator, "action_features", {})
    if not isinstance(action_features, dict):
        return set()

    feature_names = action_features.get("names")
    if isinstance(feature_names, dict):
        return {str(name) for name in feature_names}

    return {str(name) for name in action_features}


def check_delta_teleoperator(teleop_dict: dict[str, "Teleoperator"]) -> dict[str, bool]:
    """检查遥操作设备是否为 Delta 相对增量控制模式（如 3D SpaceMouse 鼠标）。"""
    is_delta_teleoperator = {}
    for name, t in teleop_dict.items():
        action_names = get_teleoperator_action_names(t)
        is_delta_teleoperator[name] = (
            bool(action_names) and
            action_names.issubset(DELTA_ACTION_NAMES) and
            bool(action_names & (DELTA_TRANSLATION_ACTION_NAMES | DELTA_ROTATION_ACTION_NAMES))
        )

    return is_delta_teleoperator


def is_union_with_dict(field_type) -> bool:
    """类型反射辅助函数：检查字段类型注解是否为包含 dict 的联合类型 (如 dict | None)。"""
    origin = get_origin(field_type)
    if origin is types.UnionType or origin is Union:
        return any(get_origin(arg) is dict for arg in get_args(field_type))
    return False


def env_to_dataset_features(env_features: dict[str, PolicyFeature]) -> dict:
    """将环境的 PolicyFeature 规范映射转换为 LeRobot Dataset 录制格式的 Schema 字典。"""
    ds_features = {}
    for key, ft in env_features.items():
        new_ft = {"shape": ft.shape}
        if ft.type == FeatureType.VISUAL:
            new_ft["dtype"] = "video"
            new_ft["names"] = ["channels", "height", "width"]
        else:
            new_ft["dtype"] = "float32"
            new_ft["names"] = None
        ds_features[key] = new_ft

    ds_features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}
    ds_features[DONE] = {"dtype": "bool", "shape": (1,), "names": None}
    ds_features["rl.is_intervention"] = {"dtype": "bool", "shape": (1,), "names": None}
    return ds_features


def copy_per_robot(value: Any, robot_names: list[str]) -> dict[str, Any]:
    """将配置对象为每个机械臂深拷贝一份，支持单对象广播或字典匹配。"""
    if isinstance(value, dict):
        return {
            name: copy.deepcopy(value[name] if name in value else next(iter(value.values()), None))
            for name in robot_names
        }
    return {name: copy.deepcopy(value) for name in robot_names}


def any_enabled(value: bool | dict[str, bool]) -> bool:
    """聚合布尔值：若是字典，任意一个机械臂为 True 则返回 True。"""
    if isinstance(value, dict):
        return any(bool(v) for v in value.values())
    return bool(value)


def resolve_entry_start_pose(
    entry_context: "PrimitiveEntryContext | None",
    robot_name: str,
    frame: "TaskFrame",
) -> list[float]:
    """在原语入口期 (on_entry) 计算机械臂在当前原语坐标系下的起始位姿 (6D)。

    核心坐标变换原理：
    上一原语结束时，观测中的位姿是基于【上一原语坐标系】表达的。
    本函数将该位姿先通过 previous_origin 还原为【绝对世界位姿】，
    再通过当前原语的 frame.origin 转换到【当前原语坐标系】下，
    从而确保跨原语切换时位姿无缝衔接且具备几何严格性。

    Args:
        entry_context: 包含上一原语退出时观测和坐标原点的上下文对象。
        robot_name: 机械臂名称。
        frame: 当前原语为该机械臂配置的 TaskFrame。

    Returns:
        在当前 frame 坐标系下的 6D 起始位姿 [x, y, z, roll, pitch, yaw]。
    """
    if frame.space != ControlSpace.TASK:
        return [float(v) for v in frame.target]

    if entry_context is None or not entry_context.observation:
        return [float(v) for v in frame.target]

    # 1. 提取上一时刻在源坐标系下的位姿
    observed_pose = get_robot_pose_from_observation(entry_context.observation, robot_name)
    previous_origin = entry_context.task_frame_origin.get(robot_name)
    
    # 2. 变换到世界坐标系：T_world = T_prev_origin * T_observed
    world_pose = task_pose_to_world_pose(observed_pose, previous_origin)
    
    # 3. 变换到新原语坐标系：T_current = (T_new_origin)^(-1) * T_world
    return world_pose_to_task_pose(world_pose, frame.origin)


def observed_task_frame_origins(
    observation: dict[str, Any],
    fallback: dict[str, list[float] | None],
) -> dict[str, list[float] | None]:
    """从观测字典中提取控制器实际生效的权威任务坐标系原点。

    当控制器刚切换坐标系原点时，使用观测中自带的 origin 最为精确权威；
    若观测中没有此通道，则回退到配置层指定的 fallback 原点。
    """
    axis_names = ["x", "y", "z", "rx", "ry", "rz"]
    origins: dict[str, list[float] | None] = {}
    for name, fallback_origin in fallback.items():
        origin: list[float] = []
        for axis_name in axis_names:
            value = observation.get(f"{name}.{axis_name}.task_frame_origin")
            if value is None:
                break
            origin.append(float(value.item() if hasattr(value, "item") else value))
        origins[name] = origin if len(origin) == 6 else fallback_origin
    return origins


def task_frame_origins(primitive: Any) -> dict[str, list[float] | None]:
    """从原语配置对象中提取各个机械臂的任务坐标系原点列表。"""
    task_frames = getattr(primitive, "task_frame", {})
    if not isinstance(task_frames, dict):
        return {}
    return {
        name: None if frame.origin is None else [float(v) for v in frame.origin]
        for name, frame in task_frames.items()
    }


def axis_to_index(axis: int | str) -> int:
    """将轴名称 ('x', 'y', 'z', 'rx', 'ry', 'rz') 转换为对应的数字索引 (0..5)。"""
    if isinstance(axis, int):
        return axis
    axis_names = ["x", "y", "z", "rx", "ry", "rz"]
    if axis not in axis_names:
        raise ValueError(f"未知的任务坐标系轴名称: '{axis}'。")
    return axis_names.index(axis)


def resolve_value(source: dict[str, Any], key: str) -> Any:
    """从字典中解析可能包含点号路径 (如 'robot.force.z') 的深层嵌套字段。

    Args:
        source: 传入的字典（如 obs 或 info）。
        key: 普通键名或点号路径。

    Returns:
        解析出的最终嵌套值。
    """
    current: Any = source
    if key in source:
        return current[key]

    for piece in key.split("."):
        if piece not in current:
            raise KeyError(f"在数据源字典中未找到键路径: '{key}'。")
        current = current[piece]

    return current


def to_scalar(value: Any) -> float:
    """将标量、单元素 numpy array 或 torch.Tensor 安全转换为 Python float。"""
    if isinstance(value, (int, float)):
        return float(value)

    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"比较操作期望标量值，实际接收到维度为 {arr.shape} 的数组。")
    return float(arr.reshape(-1)[0])


def compare(lhs: float, rhs: float, operator: str) -> bool:
    """根据指定的比较操作符字符串比对两个数值的大小。"""
    if operator == "ge":
        return lhs >= rhs
    if operator == "gt":
        return lhs > rhs
    if operator == "le":
        return lhs <= rhs
    if operator == "lt":
        return lhs < rhs
    if operator == "eq":
        return lhs == rhs
    if operator == "ne":
        return lhs != rhs
    raise ValueError(f"不支持的比较操作符: '{operator}'。")

