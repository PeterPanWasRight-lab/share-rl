from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ControlSpace(IntEnum):
    """任务坐标系的控制空间类型。"""

    JOINT = 0
    """关节空间控制（每个维度对应机械臂的一个电机旋转关节）。"""

    TASK = 1
    """笛卡尔任务空间控制（6 维向量：[x, y, z, roll, pitch, yaw]）。"""


class PolicyMode(IntEnum):
    """策略网络 (Policy) 输出动作对于某个特定控制轴的解释方式。"""

    ABSOLUTE = 0
    """绝对模式：策略输出的值直接作为该轴的目标绝对位姿/数值。"""

    RELATIVE = 1
    """相对增量模式：策略输出的值作为该轴相对于当前位姿的位移增量 (Delta)。"""


class ControlMode(IntEnum):
    """底层控制器在每个控制轴上应用的物理控制模态。"""

    POS = 0
    """位置控制 (Position Control)。"""

    VEL = 1
    """速度控制 (Velocity Control)。"""

    WRENCH = 2
    """力/力矩控制 (Wrench/Force Control)。"""

    FORCE = 2
    """力控制（WRENCH 的别名）。"""


TASK_FRAME_AXIS_NAMES = ["x", "y", "z", "rx", "ry", "rz"]
"""笛卡尔任务空间 6 个标准控制轴的名称列表。"""


@dataclass(slots=True)
class TaskFrame:
    """任务坐标系 (Task Frame) 契约配置类（可序列化）。

    TaskFrame 是 share-rl 中最核心的控制概念。它在策略 (Policy)、动作处理器 (Processors)
    和底层机器人 (Robots) 之间建立了一个明确的轴级控制契约。
    
    对于笛卡尔任务空间，位姿向量统一采用 ``[x, y, z, roll, pitch, yaw]`` 格式：
    - 前 3 维为位置 (x, y, z)，单位为米 (m)。
    - 后 3 维为外旋 XYZ 欧拉角 (roll, pitch, yaw)，单位为弧度 (rad)。
    """

    target: list[float] = field(default_factory=lambda: 6 * [0.0])
    """默认静态目标向量（6 维）。对于不可学轴，控制器将维持此目标值。"""

    space: ControlSpace = ControlSpace.TASK
    """控制空间：ControlSpace.TASK (末端 6D) 或 ControlSpace.JOINT (关节)。"""

    policy_mode: list[PolicyMode | None] = field(default_factory=lambda: 6 * [PolicyMode.RELATIVE])
    """各轴的策略控制模式列表（长度为 6）：
    - PolicyMode.RELATIVE: 该轴由策略输出相对增量。
    - PolicyMode.ABSOLUTE: 该轴由策略输出绝对目标。
    - None: 该轴为【纯脚本/固定轴】，策略不输出该轴，由 target 静态指定。
    """

    control_mode: list[ControlMode] = field(default_factory=lambda: 6 * [ControlMode.POS])
    """各轴的底层物理控制类型（POS 位置 / VEL 速度 / WRENCH 作用力）。"""

    origin: list[float] | None = None
    """任务坐标系在世界坐标系下的原点位姿 [x, y, z, roll, pitch, yaw]（若为 None 则默认世界原点 [0,0,0,0,0,0]）。"""

    min_pose: list[float] | None = None
    """各轴允许的位姿下限（用于安全截断），默认为 -inf。"""

    max_pose: list[float] | None = None
    """各轴允许的位姿上限（用于安全截断），默认为 +inf。"""

    controller_overrides: dict[str, Any] | None = None
    """覆盖底层控制器参数（如刚度 kp、阻尼 kd、力控增益等）。"""

    joint_names: list[str] | None = None
    """关节空间控制时的关节名称列表。"""

    def __post_init__(self) -> None:
        """校验 TaskFrame 维度、默认值及各轴模式组合的合法性。"""
        width = len(self.target)
        if width == 0:
            raise ValueError("target 目标向量必须包含至少一个控制轴。")
        if len(self.policy_mode) != width:
            raise ValueError(f"policy_mode 长度 ({len(self.policy_mode)}) 必须与 target 长度 ({width}) 一致。")
        if len(self.control_mode) != width:
            raise ValueError(f"control_mode 长度 ({len(self.control_mode)}) 必须与 target 长度 ({width}) 一致。")
        if self.min_pose is None:
            self.min_pose = [float("-inf")] * width
        if len(self.min_pose) != width:
            raise ValueError("min_pose 长度必须与 target 长度一致。")
        if self.max_pose is None:
            self.max_pose = [float("inf")] * width
        if len(self.max_pose) != width:
            raise ValueError("max_pose 长度必须与 target 长度一致。")

        if self.space == ControlSpace.TASK:
            if width != len(TASK_FRAME_AXIS_NAMES):
                raise ValueError("笛卡尔空间 (TASK) 控制必须提供 6 维 target 向量 ([x, y, z, rx, ry, rz])。")
            if self.origin is None:
                self.origin = 6 * [0.0]
            if len(self.origin) != 6:
                raise ValueError("origin 坐标原点必须是 6 维向量 ([x, y, z, roll, pitch, yaw])。")
        elif self.space == ControlSpace.JOINT:
            if self.origin is not None:
                raise ValueError("关节空间 (JOINT) 控制下 origin 必须为 None。")
            if self.joint_names is None:
                self.joint_names = [f"joint_{axis}" for axis in range(width)]
            if len(self.joint_names) != width:
                raise ValueError("joint_names 长度必须与 target 长度一致。")
        
        # 模式兼容性检查
        for i in range(width):
            if self.policy_mode[i] is None:
                continue
                
            if self.policy_mode[i] == PolicyMode.RELATIVE and not self.control_mode[i] == ControlMode.POS:
                raise ValueError("相对模式 (RELATIVE) 仅支持位置控制 (POS) 模式。")
            
            if self.space == ControlSpace.JOINT and not self.control_mode[i] == ControlMode.POS:
                raise ValueError("关节空间 (JOINT) 控制目前仅支持位置控制 (POS) 模式。")

    @property
    def learnable_axis_indices(self) -> list[int]:
        """获取所有由策略网络输出控制的可学习轴索引列表 (policy_mode is not None)。"""
        return [i for i, _policy_mode in enumerate(self.policy_mode) if _policy_mode is not None]

    @property
    def is_adaptive(self) -> bool:
        """检查该任务坐标系是否包含至少一个由策略控制的可学习轴。"""
        return len(self.learnable_axis_indices) > 0

    @property
    def policy_action_dim(self) -> int:
        """根据 TaskFrame 配置契约自动推断策略网络动作输出的实际张量维度。

        流形维度推断规则（强化学习旋转表示最佳实践）：
        - 任何可学习的速度/力控轴 (VEL/FORCE) 占用 1 维。
        - 任何可学习的位置平移轴 (x/y/z) 占用 1 维。
        - 任何以相对模式 (RELATIVE) 学习的位置轴占用 1 维。
        - 以绝对模式 (ABSOLUTE) 学习的旋转轴 (rx/ry/rz) 会被映射到连续流形表示上，避免欧拉角奇异性：
            * 1 个绝对旋转轴 -> 映射到 S1 圆流形 (cos, sin) = 2 维
            * 2 个绝对旋转轴 -> 映射到 S2 球面流形 = 3 维
            * 3 个绝对旋转轴 -> 映射到 SO(3) 连续 6D 旋转流形 = 6 维
        """
        dim = 0
        absolute_rotation_axes = 0

        for axis in self.learnable_axis_indices:
            if self.is_absolute_rotation_axis(axis):
                absolute_rotation_axes += 1
            else:
                dim += 1

        if absolute_rotation_axes == 0:
            return dim
        if absolute_rotation_axes == 1:
            return dim + 2
        if absolute_rotation_axes == 2:
            return dim + 3
        if absolute_rotation_axes == 3:
            return dim + 6

        raise ValueError(
            f"推断 policy_action_dim 时绝对旋转轴数量非法。预期为 0..3，实际为 {absolute_rotation_axes}。"
        )

    def is_absolute_rotation_axis(self, axis: int) -> bool:
        """判断某个轴是否为笛卡尔任务空间下的绝对旋转位置控制轴。"""
        return (
                axis >= 3 and
                self.control_mode[axis] == ControlMode.POS and
                self.policy_mode[axis] == PolicyMode.ABSOLUTE and
                self.space == ControlSpace.TASK
        )

    def joint_name_for_axis(self, axis: int) -> str:
        """获取关节空间下指定轴对应的关节名称。"""
        if self.space != ControlSpace.JOINT or self.joint_names is None:
            raise ValueError("joint_name_for_axis 仅在 space == JOINT 时有效。")
        return self.joint_names[axis]

    def action_key_for_axis(self, axis: int) -> str:
        """获取底层动作字典中指定控制轴的标准键名（如 'x.ee_pos', 'z.ee_wrench'）。"""
        if self.space == ControlSpace.JOINT:
            return f"{self.joint_name_for_axis(axis)}.pos"

        axis_name = TASK_FRAME_AXIS_NAMES[axis]
        suffix = {
            ControlMode.POS: "ee_pos",
            ControlMode.VEL: "ee_vel",
            ControlMode.WRENCH: "ee_wrench",
        }[self.control_mode[axis]]
        return f"{axis_name}.{suffix}"

    def pose_observation_key_for_axis(self, axis: int) -> str:
        """获取观测字典中指定位姿轴的观测键名（如 'x.ee_pos'）。"""
        if self.space != ControlSpace.TASK:
            raise ValueError("pose_observation_key_for_axis 仅在 space == TASK 时有效。")
        return f"{TASK_FRAME_AXIS_NAMES[axis]}.ee_pos"

    def policy_action_keys(self) -> list[str]:
        """获取策略输出张量中每个维度对应的语义名称列表。"""
        keys: list[str] = []
        absolute_rot_axes = [axis for axis in self.learnable_axis_indices if self.is_absolute_rotation_axis(axis)]

        for axis in self.learnable_axis_indices:
            if axis in absolute_rot_axes:
                continue
            keys.append(self.action_key_for_axis(axis))

        if len(absolute_rot_axes) == 1:
            axis_name = TASK_FRAME_AXIS_NAMES[absolute_rot_axes[0]]
            keys.extend([f"{axis_name}.pos.cos", f"{axis_name}.pos.sin"])
        elif len(absolute_rot_axes) == 2:
            keys.extend(["rotation.s2.x", "rotation.s2.y", "rotation.s2.z"])
        elif len(absolute_rot_axes) == 3:
            keys.extend([
                "rotation.so3.a1.x",
                "rotation.so3.a1.y",
                "rotation.so3.a1.z",
                "rotation.so3.a2.x",
                "rotation.so3.a2.y",
                "rotation.so3.a2.z",
            ])
        elif len(absolute_rot_axes) > 3:
            raise ValueError(f"绝对旋转轴数量最多为 3，实际为 {len(absolute_rot_axes)}")

        return keys

    def action_feature_keys(self) -> dict[str, type]:
        """获取该任务坐标系所蕴含的底层动作特性字典类型定义。"""
        if self.space == ControlSpace.JOINT:
            return {f"{joint_name}.pos": float for joint_name in self.joint_names}

        feature_keys: dict[str, type] = {}
        for axis in range(len(self.target)):
            feature_keys[self.action_key_for_axis(axis)] = float
        return feature_keys

    def to_dict(self) -> dict:
        """序列化为 JSON 友好的普通字典。"""
        return {
            "space": int(self.space),
            "origin": self.origin,
            "target": self.target,
            "policy_mode": [int(policy_mode) if policy_mode is not None else None for policy_mode in self.policy_mode],
            "control_mode": [int(control_mode) for control_mode in self.control_mode],
            "min_target": self.min_pose,
            "max_target": self.max_pose,
            "controller_overrides": copy.deepcopy(self.controller_overrides),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> TaskFrame:
        """从序列化字典构建 TaskFrame 实例。"""
        min_target = raw.get("min_target", raw.get("min_pose"))
        max_target = raw.get("max_target", raw.get("max_pose"))
        return cls(
            space=ControlSpace(raw["space"]),
            origin=raw.get("origin"),
            target=list(raw["target"]),
            policy_mode=[PolicyMode(item) if item is not None else None for item in raw["policy_mode"]],
            control_mode=[ControlMode(item) for item in raw["control_mode"]],
            min_pose=list(min_target) if min_target is not None else None,
            max_pose=list(max_target) if max_target is not None else None,
            controller_overrides=copy.deepcopy(raw.get("controller_overrides")),
        )

    @property
    def min_target(self) -> list[float] | None:
        """下限别名属性（与 min_pose 等价）。"""
        return self.min_pose

    @min_target.setter
    def min_target(self, value: list[float] | None) -> None:
        self.min_pose = value

    @property
    def max_target(self) -> list[float] | None:
        """上限别名属性（与 max_pose 等价）。"""
        return self.max_pose

    @max_target.setter
    def max_target(self, value: list[float] | None) -> None:
        self.max_pose = value
