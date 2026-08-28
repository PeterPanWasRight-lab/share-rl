import copy
import time
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from draccus import ChoiceRegistry
from pynput import keyboard

from lerobot.cameras import Camera
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import PREFIXES_TO_STRIP, strip_prefix, create_initial_features
from lerobot.envs import EnvConfig
from lerobot.teleoperators import Teleoperator
from lerobot.robots import Robot
from lerobot.processor import DataProcessorPipeline, DeviceProcessorStep
from lerobot.processor.converters import identity_transition
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from share.envs.manipulation_primitive.env_manipulation_primitive import (
    ManipulationPrimitive,
    OpenLoopTrajectoryPrimitive,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, TaskFrame, TASK_FRAME_AXIS_NAMES
from share.envs.utils import (
    axis_to_index,
    check_task_frame_robot,
    check_delta_teleoperator,
    is_union_with_dict,
    resolve_entry_start_pose,
    any_enabled,
    copy_per_robot
)
from share.utils.kinematics import get_kinematics
from share.processor.action import (
    ToNestedActionProcessorStep,
    MatchTeleopToPolicyActionProcessorStep,
    InterventionActionProcessorStep,
    DiscretizeGripperProcessorStep,
    RelativeFrameActionProcessor,
    ToJointActionProcessorStep
)
from share.processor.info import (
    AddKeyboardEventsAsInfoStep,
    AddFootswitchEventsAsInfoStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep
)
from share.processor.observation import (
    JointsToEEObservation,
    RelativeFrameObservationProcessor,
    StateObservationProcessor, ImageObservationProcessor,
)
from share.teleoperators import TeleopEvents
from share.utils.transformation_utils import task_pose_to_world_pose, compose_delta_pose, world_pose_to_task_pose

PRIMITIVE_TARGET_POSE_INFO_KEY = "primitive_target_pose"
PRIMITIVE_COMPLETE_INFO_KEY = "primitive_complete"
TRAJECTORY_PROGRESS_INFO_KEY = "trajectory_progress"


@dataclass(slots=True)
class PrimitiveEntryContext:
    """原语入口期上下文对象（在状态机切换瞬间传递）。

    当状态机从 source 原语切入 target 原语时，MP-Net 会打包这一瞬间的
    处理后观测、坐标原点以及触发原因，并传递给目标原语的 on_entry() 钩子。
    """

    source_primitive: str | None = None
    """上一个刚退出的源原语名称。"""

    target_primitive: str | None = None
    """即将进入的目标原语名称。"""

    observation: dict[str, Any] = field(default_factory=dict)
    """切换瞬间机器人和相机的处理后最新观测字典。"""

    task_frame_origin: dict[str, list[float] | None] = field(default_factory=dict)
    """源原语在退出时实际生效的任务坐标系原点（用于跨原语几何坐标对齐）。"""

    reason: str | None = None
    """导致本次状态转移发生的判定原因（如 'target_pose_reached', 'time_limit' 等）。"""


@dataclass
class ImagePreprocessingConfig:
    """图像预处理配置（在环境处理器内应用）。"""

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    """相机裁剪参数字典：cam_name -> (top, left, height, width)。"""

    resize_size: tuple[int, int] | None = None
    """图像缩放尺寸：(height, width)，如 (64, 64) 或 (128, 128)。"""

    filter_keys: list[str] | None = None
    """仅保留的相机图像键名列表。"""

    display_cameras: bool = False
    """是否弹出 OpenCV 实时图像显示窗口。"""


@dataclass
class KinematicsConfig:
    """逆运动学 (IK) 与正运动学 (FK) 求解器配置。"""

    enable: bool | dict[str, bool] = False
    """是否启用运动学求解器（例如将末端位姿自动解算为关节角度）。"""

    use_virtual_reference: bool | dict[str, bool] = True
    """是否使用虚拟参考点。"""

    urdf_path: str | dict[str, str | None] | None = None
    """机械臂的 URDF 机器人描述模型文件路径。"""

    target_frame_name: str | dict[str, str | None] | None = None
    """末端法兰或工具坐标系的 Link 名称。"""

    end_effector_bounds: dict[str, list[float]] | dict[str, dict[str, list[float]]] | None = None
    """末端工作空间安全包络范围限制。"""

    end_effector_step_sizes: dict[str, float] | dict[str, dict[str, float]] | None = None
    """单步最大允许的末端位移步长（防止产生过大突跃）。"""


@dataclass
class ObservationConfig:
    """观测状态处理器配置（决定哪些传感器信息会被打包进 observation.state）。"""

    add_joint_position_to_observation: bool | dict[str, bool] = True
    """是否在观测中包含机械臂各关节角度 (joint.pos)。"""

    add_joint_velocity_to_observation: bool | dict[str, bool] = False
    """是否在观测中包含各关节角速度 (joint.vel)。"""

    add_current_to_observation: bool | dict[str, bool] = False
    """是否在观测中包含电机驱动电流。"""

    add_ee_pos_to_observation: bool | dict[str, bool] = False
    """是否在观测中包含末端 6D 笛卡尔位姿 (ee_pos)。"""

    add_ee_velocity_to_observation: bool | dict[str, bool] = False
    """是否在观测中包含末端 6D 线速度与角速度 (ee_vel)。"""

    add_ee_wrench_to_observation: bool | dict[str, bool] = False
    """是否在观测中包含六维力/力矩传感数据 (ee_wrench)。"""

    ee_pos_axes: list[str] | dict[str, list[str]] | None = field(
        default_factory=lambda: [f"{ax}.ee_pos" for ax in TASK_FRAME_AXIS_NAMES]
    )
    """要包含的末端位姿轴名称列表。"""

    ee_velocity_axes: list[str] | dict[str, list[str]] | None = field(
        default_factory=lambda: [f"{ax}.ee_vel" for ax in TASK_FRAME_AXIS_NAMES]
    )
    """要包含的末端速度轴名称列表。"""

    ee_wrench_axes: list[str] | dict[str, list[str]] | None = field(
        default_factory=lambda: [f"{ax}.ee_wrench" for ax in TASK_FRAME_AXIS_NAMES]
    )
    """要包含的力传感器轴名称列表。"""

    stack_frames: int | dict[str, int] = 0
    """观测帧堆叠数量（0 表示不堆叠，>0 则将历史多帧堆叠为时序特征）。"""

    relative_ee_pos: bool | dict[str, bool] = False
    """末端位姿是否转换为相对于任务原点的相对位姿。"""


@dataclass
class GripperConfig:
    """夹爪控制与二值化离散化配置。"""

    enable: bool | dict[str, bool] = False
    """当前原语是否启用夹爪通道。"""

    discretize: bool | dict[str, bool] = False
    """是否将策略输出的连续动作离散化二值为张开(0.0)或闭合(1.0)。"""

    threshold: float | dict[str, float] = 0.5
    """二值化离散化的动作阈值（例如 > 0.5 闭合，<= 0.5 张开）。"""

    mode: Literal["state", "pulse"] | dict[str, Literal["state", "pulse"]] = "state"
    """控制模式：'state' (电平状态) 或 'pulse' (脉冲触发)。"""

    max_pos: float | dict[str, float] = 1.0
    """夹爪最大开度。"""

    min_pos: float | dict[str, float] = 0.0
    """夹爪最小开度（完全闭合）。"""

    static_pos: float | dict[str, float | None] | None = None
    """静态固定开度（若设定，该原语将保持固定开度，策略不输出夹爪）。"""

    penalty: float | dict[str, float | None] | None = None
    """不必要夹爪动作的惩罚系数。"""


@dataclass
class EventConfig:
    """遥操作键盘/脚踏开关交互事件映射配置。"""

    key_mapping: dict[TeleopEvents, dict | keyboard.Key] = field(default_factory=lambda: {})
    """键盘按键映射字典。"""

    pulse_events: tuple[TeleopEvents | str, ...] = ()
    """单次触发脉冲事件集合。"""

    foot_switch_mapping: dict[tuple[TeleopEvents], dict] = field(default_factory=lambda: {})
    """脚踏板硬件开关映射字典。"""


@dataclass
class HookConfig:
    """动作与观测处理器耗时打点调试配置。"""

    time_env_processor: bool = False
    """是否记录环境观测处理器的执行耗时。"""

    time_action_processor: bool = False
    """是否记录动作处理器的执行耗时。"""

    log_every: int = 10
    """打点日志打印频率（每 N 步一次）。"""


@dataclass
class ManipulationPrimitiveProcessorConfig:
    """操作原语的顶层处理器配置集合。"""

    # 全局参数
    control_time_s: float = 10.0
    """单次操作的最长控制超时时间（秒）。"""

    fps: float = 10.0
    """控制频率 (Hz)。"""

    image_preprocessing: ImagePreprocessingConfig | None = None
    """相机视觉预处理配置。"""

    events: EventConfig = field(default_factory=EventConfig)
    """遥操交互事件配置。"""

    hooks: HookConfig = field(default_factory=HookConfig)
    """性能调试钩子配置。"""

    # 机械臂专属参数
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    """观测状态配置。"""

    gripper: GripperConfig = field(default_factory=GripperConfig)
    """夹爪控制配置。"""

    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    """运动学与工作空间配置。"""


@dataclass
class OpenLoopTrajectorySpec:
    """开环脚本轨迹规格定义类。"""

    target: list[float] | dict[str, list[float]] | None = None
    """目标绝对位姿 6D 向量（在任务坐标系下）。"""

    delta: list[float] | dict[str, list[float]] | None = None
    """相对位移增量 6D 向量。"""

    frame: Literal["task", "world", "ee"] | dict[str, Literal["task", "world", "ee_current"]] = "task"
    """delta 位移所参考的坐标系（'task' 任务系 / 'world' 世界系 / 'ee' 当前末端工具系）。"""

    duration_s: float | dict[str, float] = 1.0
    """轨迹执行总时长（秒）。"""


@dataclass
class ManipulationPrimitiveConfig(EnvConfig, ChoiceRegistry):
    """单个操作原语节点的标准声明式配置基类。

    声明了该原语的 TaskFrame 契约、观测/动作处理器、绑定的神经网络策略（若有）、
    以及进入该原语时的生命周期钩子 on_entry()。
    """
    task_frame: TaskFrame | dict[str, TaskFrame] = field(default_factory=TaskFrame)
    """任务坐标系契约配置。"""

    processor: ManipulationPrimitiveProcessorConfig = field(default_factory=ManipulationPrimitiveProcessorConfig)
    """数据处理流水线配置。"""

    policy: PreTrainedConfig | str | None = None
    """绑定的预训练或待训练 Policy 策略配置路径（纯脚本原语为 None）。"""

    policy_overwrites: dict = field(default_factory=dict)
    """策略参数覆盖字典。"""

    notes: str | None = None
    """原语说明备注。"""

    is_terminal: bool = False
    """是否为全局终止原语节点（到达该节点后 Episode 结束）。"""

    task_description: str | None = None
    """任务文字描述（用于语言多模态策略）。"""

    target_pose_info_key: str | None = PRIMITIVE_TARGET_POSE_INFO_KEY
    """发布到 info 字典中的目标位姿键名。"""

    def __post_init__(self):
        self._kinematics_solver = {}

        if isinstance(self.policy, str):
            policy_path = self.policy
            self.policy = PreTrainedConfig.from_pretrained(pretrained_name_or_path=policy_path)
            self.policy.pretrained_path = policy_path

    @property
    def gym_kwargs(self) -> dict:
        """传递给 Gym 环境创建的可选参数。"""
        return {}

    @property
    def is_adaptive(self) -> bool:
        """检查该原语是否包含由策略网络控制的可学习动作维度。"""
        task_frames = self.task_frame.values() if isinstance(self.task_frame, dict) else [self.task_frame]
        return any(tf.policy_action_dim > 0 for tf in task_frames)

    @property
    def num_cameras(self) -> int:
        """获取当前原语中配置的视觉相机总数量。"""
        if self.features is None:
            return 0
        else:
            return len([ft for ft in self.features.values() if ft.type == FeatureType.VISUAL])

    def make(
        self,
        robot_dict: dict[str, Robot],
        teleop_dict: dict[str, Teleoperator],
        cameras: dict[str, Camera],
        device: str = "cpu"
    ):
        """工厂方法：根据配置实例化对应的 ManipulationPrimitive 环境实例和动作/观测处理器。

        Args:
            robot_dict: 机械臂句柄字典。
            teleop_dict: 遥操设备句柄字典。
            cameras: 相机句柄字典。
            device: 运算设备 ('cuda' 或 'cpu')。

        Returns:
            元组 ``(env, env_processor, action_processor)``。
        """
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)

        display_cameras = self.processor.image_preprocessing is not None and self.processor.image_preprocessing.display_cameras
        env = ManipulationPrimitive(task_frame=self.task_frame, robot_dict=robot_dict, cameras=cameras, display_cameras=display_cameras)

        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor

    def make_action_processor(self, robot_dict, teleop_dict, device) -> DataProcessorPipeline:
        """Create the action-side processing pipeline.

        Args:
            robot_dict: Connected robot handles keyed by robot name.
            teleop_dict: Connected teleoperators keyed by robot name.
            device: Torch device hint forwarded to processor steps when needed.

        Returns:
            A ``DataProcessorPipeline`` that normalizes teleop/policy actions
            into the nested low-level action dict expected by the primitive env.
        """
        action_pipeline_steps = []

        # events
        if self.processor.events.key_mapping:
            action_pipeline_steps.append(
                AddKeyboardEventsAsInfoStep(
                    mapping=self.processor.events.key_mapping,
                    pulse_events=self.processor.events.pulse_events,
                )
            )

        if self.processor.events.foot_switch_mapping:
            action_pipeline_steps.append(AddFootswitchEventsAsInfoStep(mapping=self.processor.events.foot_switch_mapping))

        try:
            action_pipeline_steps.append(AddTeleopEventsAsInfoStep(teleoperators=teleop_dict))
        except TypeError:
            pass

        action_pipeline_steps.extend([
            AddTeleopActionAsComplimentaryDataStep(teleoperators=teleop_dict),  # this checks events and should come after Add*EventsAsInfoStep's
            ToNestedActionProcessorStep(
                task_frame=self.task_frame,
                gripper_enable=self.processor.gripper.enable,
            ),

            # make teleop action match policy based on task frame (treat delta ee / vel / force the same):
            # teleop Q:
            # policy delta ee / vel / force: FK + differentiate
            # abs ee: FK
            # delta Q: differentiate
            # abs Q. noop
            #
            # teleop delta ee:
            # policy delta ee / vel / force: noop
            # abs ee: integrate
            # delta Q: IK
            # abs Q: integrate + IK
            MatchTeleopToPolicyActionProcessorStep(
                teleoperators=teleop_dict,
                task_frame=self.task_frame,
                kinematics=self._kinematics_solver,
                use_virtual_reference=self.processor.kinematics.use_virtual_reference,
                gripper_enable=self.processor.gripper.enable,
            ),

            # scatter policy / teleop action (depending on is-intervention event) into full task frame action target
            # send feedback to teleoperators if they need it
            InterventionActionProcessorStep(
                teleoperators=teleop_dict,
                task_frame=self.task_frame,
                gripper_enable=self.processor.gripper.enable,
            ),
            DiscretizeGripperProcessorStep(
                min_pos=self.processor.gripper.min_pos,
                max_pos=self.processor.gripper.max_pos,
                threshold=self.processor.gripper.threshold, 
                discretize=self.processor.gripper.discretize,
                static_pos=self.processor.gripper.static_pos
            ),
        ])

        # action in ee frame instead of in world frame
        if any_enabled(self.processor.observation.relative_ee_pos):
            action_pipeline_steps.append(
                RelativeFrameActionProcessor(
                    enable=self.processor.observation.relative_ee_pos
                )
            )

        # todo: fix this
        is_task_frame_robot = check_task_frame_robot(robot_dict)
        if not all(is_task_frame_robot.values()) and False:
            # after this processor, the action must a dictionary of joint names
            # policy_action: delta vel ->

            action_pipeline_steps.append(
                ToJointActionProcessorStep(
                    is_task_frame_robot=is_task_frame_robot,
                    task_frame=self.task_frame,
                    kinematics=self._kinematics_solver,
                    use_virtual_reference=self.processor.kinematics.use_virtual_reference
                )
            )

        # timing hooks
        if self.processor.hooks.time_action_processor:
            from share.utils.control_utils import make_step_timing_hooks
            action_before_hooks, action_after_hooks = make_step_timing_hooks(
                pipeline_steps=action_pipeline_steps,
                label="action",
                log_every=self.processor.hooks.log_every,
                ema_alpha=0.2,
                also_print=False
            )
        else:
            action_before_hooks, action_after_hooks = [], []

        return DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition,
            before_step_hooks=action_before_hooks, after_step_hooks=action_after_hooks
        )

    def make_env_processor(self, device: str = "cpu") -> DataProcessorPipeline:
        """Create the observation-side processing pipeline.

        Args:
            device: Torch device used for tensor conversion and stacked state
                outputs in downstream processor steps.

        Returns:
            A ``DataProcessorPipeline`` that augments raw env observations with
            FK-derived EE poses, relative-frame channels, state tensors, and any
            configured image preprocessing.
        """
        env_pipeline_steps = []

        # obs is dict with keys {robot_name}.{axis/joint}.{pos/vel/ee_pos/ee_vel/ee_wrench} | {OBS_IMAGES}{camera_name}
        # {axis} is in {x,y,z,wx,wy,wz}
        if self._kinematics_solver:
            # for all robots that have a solver, we want to fetch their joints and add {robot_name}.{axis}.ee_pos to the obs
            env_pipeline_steps.append(
                JointsToEEObservation(
                    kinematics=self._kinematics_solver,
                    motor_names={name: frame.joint_names for name, frame in self.task_frame.items()}
                )
            )

        #env_pipeline_steps.append(
        #    GripperPenaltyProcessorStep(
        #        max_gripper_pos=self.processor.gripper.max_pos,
        #        penalty=self.processor.gripper.penalty,
        #    )
        #)

        env_pipeline_steps.extend([
            # builds OBS_STATE based on what we want to have in there
            # if obs has no joint vel and we want it, compute numerically
            # same for ee_vel
            StateObservationProcessor(
                device=device,
                gripper_enable=self.processor.gripper.enable,
                add_joint_position_to_observation=self.processor.observation.add_joint_position_to_observation,
                add_joint_velocity_to_observation=self.processor.observation.add_joint_velocity_to_observation,
                add_current_to_observation=self.processor.observation.add_current_to_observation,
                add_ee_pos_to_observation=self.processor.observation.add_ee_pos_to_observation,
                ee_pos_axes=self.processor.observation.ee_pos_axes,
                add_ee_velocity_to_observation=self.processor.observation.add_ee_velocity_to_observation,
                ee_velocity_axes=self.processor.observation.ee_velocity_axes,
                add_ee_wrench_to_observation=self.processor.observation.add_ee_wrench_to_observation,
                ee_wrench_axes=self.processor.observation.ee_wrench_axes,
                stack_frames=self.processor.observation.stack_frames,
            ),
        ])

        if self.processor.image_preprocessing:
            env_pipeline_steps.append(
                ImageObservationProcessor(
                    crop_params_dict=self.processor.image_preprocessing.crop_params_dict,
                    resize_size=self.processor.image_preprocessing.resize_size,
                    filter_keys=self.processor.image_preprocessing.filter_keys,
                    debug_timing=self.processor.hooks.time_env_processor,
                    log_every=self.processor.hooks.log_every,
                )
            )

        env_pipeline_steps.append(DeviceProcessorStep(device=device))

        # action relative to starting pose
        if any_enabled(self.processor.observation.relative_ee_pos):
            env_pipeline_steps.append(
                RelativeFrameObservationProcessor(
                    enable=self.processor.observation.relative_ee_pos
                )
            )

        # timing hooks
        if self.processor.hooks.time_env_processor:
            from share.utils.control_utils import make_step_timing_hooks
            env_before_hooks, env_after_hooks = make_step_timing_hooks(
                pipeline_steps=env_pipeline_steps,
                label="env",
                log_every=self.processor.hooks.log_every,
                ema_alpha=0.2,
                also_print=False,
            )
        else:
            env_before_hooks, env_after_hooks = [], []

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition,
            before_step_hooks=env_before_hooks, after_step_hooks=env_after_hooks
        )

    def validate(self, robot_dict, teleop_dict):
        """Validate one primitive against the connected robot/teleop setup.

        Args:
            robot_dict: Connected robot handles keyed by robot name.
            teleop_dict: Connected teleoperator handles keyed by robot name.

        This method normalizes per-robot config fields, initializes any needed
        kinematics solvers, and enforces the task-frame compatibility rules used
        throughout the action and observation pipelines.
        """

        is_task_frame_robot = check_task_frame_robot(robot_dict)
        is_delta_teleoperator = check_delta_teleoperator(teleop_dict)

        # go through each per-robot attribute and check if we need to turn scalar configs into configs for each robot
        if not isinstance(self.task_frame, dict):
            self.task_frame = {name: copy.deepcopy(self.task_frame) for name in robot_dict}

        for attr in ["observation", "gripper", "kinematics"]:
            _attr = getattr(self.processor, attr)
            for fn in fields(_attr):
                if is_union_with_dict(fn.type) and not isinstance(getattr(_attr, fn.name), dict):
                    setattr(_attr, fn.name, {name: getattr(_attr, fn.name) for name in robot_dict})
            setattr(self.processor, attr, _attr)

        # checks per robot
        for name, frame in self.task_frame.items():
            if name not in robot_dict:
                raise ValueError(f"Missing robot for task-frame entry '{name}'.")

            # ENV-101: learnable VEL/FORCE axes require delta teleoperator input.
            for axis in frame.learnable_axis_indices:
                if frame.control_mode[axis] in {ControlMode.VEL, ControlMode.WRENCH} and not is_delta_teleoperator[name]:
                    raise ValueError(
                        "Adaptive task-frame axes with VEL/FORCE control require a delta teleoperator. "
                        f"Got robot='{name}', axis={axis}, control_mode={frame.control_mode[axis].name}, "
                        "teleoperator_kind='absolute'."
                    )

            # ENV-102: JOINT-space and joint-only robots must only receive POS axis modes.
            if frame.space == ControlSpace.JOINT:
                # set joint names
                frame.joint_names = [motor for motor in robot_dict[name].bus.motors if not motor.endswith("_shadow")]

                non_pos_axes = [i for i, mode in enumerate(frame.control_mode) if mode != ControlMode.POS]
                if non_pos_axes:
                    raise ValueError(
                        "ControlSpace.JOINT only supports POS axis modes. "
                        f"Got robot='{name}', non_pos_axes={non_pos_axes}."
                    )

            if not is_task_frame_robot[name]:
                non_pos_axes = [i for i, mode in enumerate(frame.control_mode) if mode != ControlMode.POS]
                if non_pos_axes:
                    raise ValueError(
                        "Joint-only robots only support POS axis modes in this pipeline. "
                        f"Got robot='{name}', non_pos_axes={non_pos_axes}."
                    )

            # ENV-102: TASK-space with absolute-joint teleop or joint-only robot requires kinematics.
            requires_kinematics = (
                    frame.space == ControlSpace.TASK and
                    (not is_delta_teleoperator[name] or not is_task_frame_robot[name])
            )
            if requires_kinematics and not self.processor.kinematics.enable[name]:
                raise ValueError(
                    "Kinematics must be enabled for TASK-space control when teleop/robot modalities require FK/IK. "
                    f"Set processor.kinematics.enable['{name}']=True."
                )

            if requires_kinematics and name not in self._kinematics_solver:
                raise ValueError(
                    "Kinematics are required but no solver was initialized. "
                    f"Check kinematics config and robot interface for '{name}' "
                    "(urdf_path/target_frame_name/bus availability)."
                )

        # if gripper.enable but the robot has no GRIPPER_KEY action feature, disable
        for name, robot in robot_dict.items():
            if self.processor.gripper.enable[name]:
                if self.processor.gripper.static_pos[name]:
                    raise ValueError(
                        f"Gripper processing enabled for robot '{name}' but gripper is set to a static position."
                    )

                if not f"{GRIPPER_KEY}.pos" in robot.action_features:
                    raise ValueError(
                        f"Gripper processing enabled for robot '{name}' but no gripper action feature found. "
                        "Expected an action key like '{GRIPPER_KEY}.pos'."
                    )

        # Set up kinematics solver if inverse kinematics is configured
        for name, robot in robot_dict.items():
            if not is_task_frame_robot[name] and self.processor.kinematics.enable[name]:
                self._kinematics_solver[name] = get_kinematics(
                    robot_name=robot.name,
                    urdf_path=self.processor.kinematics.urdf_path[name],
                    target_frame_name=self.processor.kinematics.target_frame_name[name],
                    joint_names=self.task_frame[name].joint_names,
                )

    def infer_features(self, robot_dict, cameras):
        """Infer policy-visible feature specs from the configured pipelines.

        Args:
            robot_dict: Connected robots used to sample the raw observation
                schema exposed by the primitive env.
            cameras: Connected cameras used to sample visual observation shapes.

        The inferred features mirror the observation/action contract after the
        env processor has transformed the raw env outputs.
        """
        # process features with respective pipeline
        # get initial obs features from robot_dict instead
        initial_features = {}
        for cam_key, cam in cameras.items():
            img = cam.async_read(timeout_ms=10_000)
            initial_features[f"{OBS_IMAGES}.{cam_key}"] = PolicyFeature(type=FeatureType.VISUAL, shape=img.shape)

        for name in robot_dict:
            for k, v in robot_dict[name].get_observation().items():
                if isinstance(v, float):
                    shape = (1, )
                elif hasattr(v, "shape"):
                    shape = v.shape
                elif hasattr(v, "__iter__"):
                    shape = (len(v), )
                else:
                    raise ValueError(f"Unknown type for observation {name}.{k}: {type(v)}")
                initial_features[f"{name}.{k}"] = PolicyFeature(type=FeatureType.STATE, shape=shape)

        initial_features = create_initial_features(observation=initial_features)
        env_processor = self.make_env_processor()
        pipeline_features = env_processor.transform_features(initial_features)
        obs_features = pipeline_features[PipelineFeatureType.OBSERVATION]

        action_dim = sum(frame.policy_action_dim for frame in self.task_frame.values())

        count_gripper = [is_tf and bool(enable) for (is_tf, enable) in zip(check_task_frame_robot(robot_dict).values(), self.processor.gripper.enable.values())]
        action_dim += sum(count_gripper)  # add gripper action dim if enabled

        # expose state, action and visual features
        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
            OBS_STATE: pipeline_features[PipelineFeatureType.OBSERVATION][OBS_STATE]
        }
        for key, ft in obs_features.items():
            if ft.type == FeatureType.VISUAL:
                key = strip_prefix(key, PREFIXES_TO_STRIP)
                self.features[f"{OBS_IMAGES}.{key}"] = PolicyFeature(type=FeatureType.VISUAL, shape=ft.shape)

        if not self.features_map:
            self.features_map = {key: key for key in self.features}

    def on_entry(self, env: ManipulationPrimitive, entry_context: PrimitiveEntryContext | None) -> None:
        """【原语入口期生命周期钩子】：当该原语被状态机激活切入瞬间触发调用。

        负责在进入时将计算得到的初始目标位姿写入 env 运行时状态中。

        Args:
            env: 当前原语的底层运行环境实例。
            entry_context: 上一原语退出时传递过来的上下文对象（包含当时位姿和坐标原点）。
        """
        env.set_target_pose(
            {
                name: [float(v) for v in frame.target]
                for name, frame in self.task_frame.items()
            },
            info_key=self.target_pose_info_key,
        )

    def resolve_targets(
        self,
        entry_context: PrimitiveEntryContext | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """在当前原语的任务坐标系下，解析计算起始位姿 (start_pose) 和目标位姿 (target_pose)。

        Args:
            entry_context: 包含上一原语退出时观测与坐标原点的上下文对象。

        Returns:
            元组 ``(start_pose, target_pose)``，均为以机械臂名称为键的 6D 位姿字典。
        """
        start_pose: dict[str, list[float]] = {}
        target_pose: dict[str, list[float]] = {}
        for name, frame in self.task_frame.items():
            start_pose[name] = resolve_entry_start_pose(entry_context, name, frame)
            target_pose[name] = [float(v) for v in frame.target]
        return start_pose, target_pose


ManipulationPrimitiveConfig.register_subclass("primitive", ManipulationPrimitiveConfig)


# =============================================================================
# 原语子类 1: 传感器力矩清零/去皮原语 (ZeroFTPrimitiveConfig)
# =============================================================================

@ManipulationPrimitiveConfig.register_subclass("zero_ft")
@dataclass
class ZeroFTPrimitiveConfig(ManipulationPrimitiveConfig):
    """力传感器清零/去皮纯脚本原语。

    进入该原语后保持当前位姿静止，调用力传感器的 zero_ft() 消除重力/安装预紧力偏置，
    完成后将 primitive_complete 标记为 True 并退出。
    """

    settle_duration_s: float = 0.3
    """清零前的静止等待稳定时间（秒）。"""

    def validate(self, robot_dict, teleop_dict):
        super().validate(robot_dict, teleop_dict)
        if self.policy is not None:
            raise ValueError("zero_ft 是纯脚本原语，不能配置神经网络策略 policy。")
        if self.settle_duration_s < 0.0:
            raise ValueError("zero_ft 的 settle_duration_s 必须 >= 0。")

    def make(
        self,
        robot_dict,
        teleop_dict,
        cameras,
        device: str = "cpu",
    ):
        env, env_processor, action_processor = super().make(robot_dict, teleop_dict, cameras, device)
        env.uses_autonomous_step = True
        return env, env_processor, action_processor

    def on_entry(self, env: ManipulationPrimitive, entry_context: PrimitiveEntryContext | None) -> None:
        start_pose, _target_pose = self.resolve_targets(entry_context)
        env.set_target_pose(start_pose, info_key=self.target_pose_info_key)
        env.apply_task_frames()

        time.sleep(self.settle_duration_s)
        for robot in env.robot_dict.values():
            controller = getattr(robot, "controller", None)
            if controller is None or not hasattr(controller, "zero_ft"):
                raise AttributeError("zero_ft 原语要求机械臂控制器具有 controller.zero_ft() 方法。")
            controller.zero_ft()

        env._primitive_complete = True


# =============================================================================
# 原语子类 2: 相对增量位移原语 (MoveDeltaPrimitiveConfig)
# =============================================================================

@ManipulationPrimitiveConfig.register_subclass("move_delta")
@dataclass
class MoveDeltaPrimitiveConfig(ManipulationPrimitiveConfig):
    """相对增量位移原语：根据进入瞬间的实际位姿叠加指定的 delta 增量计算目标位姿。

    应用场景举例：
    - 抓取后向上抬升 10cm：设置 delta=[0, 0, 0.1, 0, 0, 0]，delta_frame='world'。
    - 沿当前工具法向向前推进 2cm：设置 delta=[0, 0, 0.02, 0, 0, 0]，delta_frame='ee'。
    - 保持当前 X/Y 平面位置，唯独下落到固定 Z 高度：设置 delta=[0,0,0,0,0,0], absolute_axes=['z']。
    """

    delta: list[float] | dict[str, list[float]] = field(default_factory=lambda: [0.0] * 6)
    """要施加的 6D 相对位姿增量 [dx, dy, dz, drx, dry, drz]。"""

    delta_frame: Literal["world", "ee"] | dict[str, Literal["world", "ee_current"]] = "world"
    """delta 增量所参考的坐标系：'world'（世界坐标系）或 'ee'（机械臂当前末端法兰坐标系）。"""

    absolute_axes: list[int | str] | dict[str, list[int | str]] = field(default_factory=list)
    """跳过 delta 增量计算、直接使用配置中固定绝对目标的轴列表（如 ['z']）。"""

    publish_target_info: bool | dict[str, bool] = True
    """是否将计算出的目标位姿发布到 info 字典中供 OnTargetPoseReached 检测。"""

    def validate(self, robot_dict, teleop_dict):
        super().validate(robot_dict, teleop_dict)
        robot_names = list(robot_dict)
        self.delta = copy_per_robot(self.delta, robot_names)
        self.delta_frame = copy_per_robot(self.delta_frame, robot_names)
        self.absolute_axes = copy_per_robot(self.absolute_axes, robot_names)
        self.publish_target_info = copy_per_robot(self.publish_target_info, robot_names)

        for name, frame in self.task_frame.items():
            if frame.space != ControlSpace.TASK:
                raise ValueError(f"move_delta 原语要求 TASK 笛卡尔空间，实际为 '{name}'。")
            if len(self.delta[name]) != 6:
                raise ValueError(f"move_delta 为 '{name}' 配置的 delta 必须是 6 维向量。")
            self.absolute_axes[name] = [axis_to_index(axis) for axis in self.absolute_axes[name]]

    def on_entry(self, env: ManipulationPrimitive, entry_context: PrimitiveEntryContext | None) -> None:
        """在进入该原语时，根据进入位姿 + delta 增量计算新目标点，并写入 env。"""
        _start_pose, target_pose = self.resolve_targets(entry_context)
        env.set_target_pose(
            target_pose,
            info_key=self.target_pose_info_key if any(self.publish_target_info.values()) else None,
        )

    def resolve_targets(
        self,
        entry_context: PrimitiveEntryContext | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """计算入口起点位姿以及叠加上 delta 之后的新目标位姿。"""
        start_pose, target_pose = super().resolve_targets(entry_context)
        for name, frame in self.task_frame.items():
            start_world = task_pose_to_world_pose(start_pose[name], frame.origin)
            target_world = compose_delta_pose(
                start_pose_world=start_world,
                delta=[float(v) for v in self.delta[name]],
                frame_name=self.delta_frame[name],
            )
            resolved_target = world_pose_to_task_pose(target_world, frame.origin)
            target_pose[name] = [float(v) for v in frame.target]
            fixed_pos_axes = [axis for axis in self._fixed_pos_axes(frame) if axis not in self.absolute_axes[name]]
            fixed_rotation_axes = [axis for axis in fixed_pos_axes if axis >= 3]
            for axis in fixed_pos_axes:
                if axis >= 3 and len(fixed_rotation_axes) < 3:
                    target_pose[name][axis] = self._resolve_partial_rotation_axis_target(
                        start_pose_world=start_world,
                        frame=frame,
                        delta=[float(v) for v in self.delta[name]],
                        delta_frame=self.delta_frame[name],
                        axis=axis,
                    )
                    continue
                target_pose[name][axis] = float(resolved_target[axis])

        return start_pose, target_pose

    @staticmethod
    def _fixed_pos_axes(frame: TaskFrame) -> list[int]:
        """找出所有非学习的固定位置控制轴。"""
        return [
            axis
            for axis in range(len(frame.target))
            if frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] is None
        ]

    @staticmethod
    def _resolve_partial_rotation_axis_target(
        *,
        start_pose_world: list[float],
        frame: TaskFrame,
        delta: list[float],
        delta_frame: str,
        axis: int,
    ) -> float:
        """对单独一个旋转轴施加旋转增量。"""
        single_axis_delta = [0.0] * 6
        single_axis_delta[axis] = float(delta[axis])
        axis_target_world = compose_delta_pose(
            start_pose_world=start_pose_world,
            delta=single_axis_delta,
            frame_name=delta_frame,
        )
        axis_target_task = world_pose_to_task_pose(axis_target_world, frame.origin)
        return float(axis_target_task[axis])


# =============================================================================
# 原语子类 3: 开环插值轨迹原语 (OpenLoopTrajectoryPrimitiveConfig)
# =============================================================================

@ManipulationPrimitiveConfig.register_subclass("open_loop_trajectory")
@dataclass
class OpenLoopTrajectoryPrimitiveConfig(ManipulationPrimitiveConfig):
    """开环轨迹插值脚本原语：按照设定时长平滑插值执行从起点到目标的完整几何轨迹。"""

    trajectory: OpenLoopTrajectorySpec = field(default_factory=OpenLoopTrajectorySpec)
    """轨迹规划规格（指定 target 或 delta，以及 duration_s 时长）。"""

    publish_target_info: bool | dict[str, bool] = True
    """是否发布目标位姿到 info。"""

    def validate(self, robot_dict, teleop_dict):
        super().validate(robot_dict, teleop_dict)
        robot_names = list(robot_dict)
        self.publish_target_info = copy_per_robot(self.publish_target_info, robot_names)
        self.trajectory.frame = copy_per_robot(self.trajectory.frame, robot_names)
        self.trajectory.duration_s = copy_per_robot(self.trajectory.duration_s, robot_names)
        if self.trajectory.target is not None:
            self.trajectory.target = copy_per_robot(self.trajectory.target, robot_names)
        if self.trajectory.delta is not None:
            self.trajectory.delta = copy_per_robot(self.trajectory.delta, robot_names)

        if (self.trajectory.target is None) == (self.trajectory.delta is None):
            raise ValueError("open_loop_trajectory 要求 trajectory.target 与 trajectory.delta 二选一且必须指定其中之一。")
        if self.policy is not None:
            raise ValueError("open_loop_trajectory 是纯脚本原语，禁止配置 policy。")

        for name, frame in self.task_frame.items():
            if frame.space != ControlSpace.TASK:
                raise ValueError(f"open_loop_trajectory 原语要求 TASK 笛卡尔空间，实际为 '{name}'。")
            if frame.is_adaptive:
                raise ValueError(f"open_loop_trajectory 原语不能包含学习轴，但在 '{name}' 中发现了学习轴。")
            if any(mode != ControlMode.POS for mode in frame.control_mode):
                raise ValueError(f"open_loop_trajectory 原语目前仅支持 POS 位置控制轴，在 '{name}' 中发现非 POS 轴。")

            if float(self.trajectory.duration_s[name]) <= 0.0:
                raise ValueError(f"open_loop_trajectory 为 '{name}' 设置的 duration_s 必须 > 0。")

            if self.trajectory.target is not None:
                if len(self.trajectory.target[name]) != 6:
                    raise ValueError(f"open_loop_trajectory 为 '{name}' 设置的 target 必须是 6 维向量。")
                if self.trajectory.frame[name] != "task":
                    raise ValueError("指定 trajectory.target 时，trajectory.frame 必须为 'task'。")

            if self.trajectory.delta is not None:
                if len(self.trajectory.delta[name]) != 6:
                    raise ValueError(f"open_loop_trajectory 为 '{name}' 设置的 delta 必须是 6 维向量。")
                if self.trajectory.frame[name] not in {"world", "ee"}:
                    raise ValueError("指定 trajectory.delta 时，trajectory.frame 必须在 {'world', 'ee'} 中。")

    def make(
        self,
        robot_dict: dict[str, Robot],
        teleop_dict: dict[str, Teleoperator],
        cameras: dict[str, Camera],
        device: str = "cpu",
    ):
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)

        display_cameras = self.processor.image_preprocessing is not None and self.processor.image_preprocessing.display_cameras
        env = OpenLoopTrajectoryPrimitive(
            task_frame=self.task_frame,
            robot_dict=robot_dict,
            cameras=cameras,
            open_loop_config=self,
            display_cameras=display_cameras,
        )
        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor

    def resolve_trajectory(
        self,
        entry_context: PrimitiveEntryContext | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """计算开环插值轨迹的起始点 start_pose 与终点 goal_pose。"""
        start_pose, goal_pose = super().resolve_targets(entry_context)
        for name, frame in self.task_frame.items():
            if self.trajectory.target is not None:
                goal_pose[name] = [float(v) for v in self.trajectory.target[name]]
                continue

            start_world = task_pose_to_world_pose(start_pose[name], frame.origin)
            goal_world = compose_delta_pose(
                start_pose_world=start_world,
                delta=[float(v) for v in self.trajectory.delta[name]],
                frame_name=self.trajectory.frame[name],
            )
            goal_pose[name] = world_pose_to_task_pose(goal_world, frame.origin)
        return start_pose, goal_pose

    def resolve_targets(
        self,
        entry_context: PrimitiveEntryContext | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        return self.resolve_trajectory(entry_context)

    def target_pose_at(
        self,
        alpha: float,
        start_pose: dict[str, list[float]],
        goal_pose: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        """根据当前插值进度 alpha (0.0 -> 1.0) 计算线性插值位姿。"""
        alpha = min(1.0, max(0.0, float(alpha)))
        return {
            name: [
                float(start_pose[name][axis] + alpha * (goal_pose[name][axis] - start_pose[name][axis]))
                for axis in range(len(goal_pose[name]))
            ]
            for name in goal_pose
        }

    def trajectory_timing(self, robot_dict: dict[str, Robot]) -> tuple[int, int]:
        """根据设定时长与底层机械臂的控制频率计算子步步数 (substeps)。"""
        control_hz_candidates: list[float] = []
        for robot in robot_dict.values():
            robot_config = getattr(robot, "config", None)
            frequency = getattr(robot_config, "frequency", None)
            if isinstance(frequency, (int, float)) and frequency > 0:
                control_hz_candidates.append(float(frequency))

        control_hz = max(control_hz_candidates, default=float(self.processor.fps if self.processor.fps > 0 else 1.0))
        outer_hz = float(self.processor.fps if self.processor.fps > 0 else control_hz)
        duration_s = max(float(v) for v in self.trajectory.duration_s.values())
        total_substeps = max(1, int(round(duration_s * control_hz)))
        substeps_per_step = max(1, int(round(control_hz / outer_hz)))
        return total_substeps, substeps_per_step

    def on_entry(self, env: ManipulationPrimitive, entry_context: PrimitiveEntryContext | None) -> None:
        """进入该开环轨迹原语时，根据当前起点与终点初始化轨迹生成器。"""
        start_pose, target_pose = self.resolve_trajectory(entry_context)
        info_key = self.target_pose_info_key if any(self.publish_target_info.values()) else None
        if isinstance(env, OpenLoopTrajectoryPrimitive):
            env.configure_trajectory(start_pose=start_pose, target_pose=target_pose, info_key=info_key)
        else:
            env.set_target_pose(target_pose, info_key=info_key)
