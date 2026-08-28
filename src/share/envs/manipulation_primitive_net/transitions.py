import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal

from draccus import ChoiceRegistry

from share.envs.manipulation_primitive.config_manipulation_primitive import PRIMITIVE_TARGET_POSE_INFO_KEY
from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES
from share.envs.utils import to_scalar, resolve_value, compare, axis_to_index
from share.teleoperators import TeleopEvents
from share.utils.constants import DEFAULT_ROBOT_NAME
from share.utils.transformation_utils import get_robot_pose_from_observation, task_pose_to_world_pose
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal

from draccus import ChoiceRegistry

from share.envs.manipulation_primitive.config_manipulation_primitive import PRIMITIVE_TARGET_POSE_INFO_KEY
from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES
from share.envs.utils import to_scalar, resolve_value, compare, axis_to_index
from share.teleoperators import TeleopEvents
from share.utils.constants import DEFAULT_ROBOT_NAME
from share.utils.transformation_utils import get_robot_pose_from_observation, task_pose_to_world_pose

logger = logging.getLogger(__name__)

DEFAULT_TARGET_POSE_AXES_INFO_KEY = "_primitive_target_pose_axes"


@dataclass
class Outcome:
    """转移条件判定的评估结果。

    每个 Transition 边在单步 evaluate() 后都会返回一个 Outcome 对象，
    告诉 MP-Net 状态机当前边是否被触发、是否给予转移奖励、以及触发原因。
    """
    reward: float = 0.0
    """触发该转移时给予的额外奖励值。"""

    terminated: bool = False
    """是否满足成功/事件终止条件并触发状态转移。"""

    truncated: bool = False
    """是否由于步数超时等截断条件触发状态转移。"""

    reason: str | None = None
    """状态转移发生的原因描述字符串（如 'target_pose_reached', 'time_limit'）。"""


@dataclass
class Transition(ChoiceRegistry):
    """状态转移条件的抽象基类。

    定义了状态机网络中一条有向边 (source -> target) 的基础属性和判定接口。
    继承自 draccus.ChoiceRegistry，支持通过字符串别名在配置文件中声明式实例化。
    """
    source: str
    """当前边的起始原语节点名称（源节点）。"""

    target: str
    """满足条件后将要跳转进入的目标原语节点名称（目标节点）。"""

    additional_reward: float = 0.0
    """边触发时附加的即时奖励（默认为 0.0）。"""

    reason: str | None = None
    """自定义转移原因名称。若为 None 则使用子类的默认原因标识。"""

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        """核心判定方法：根据当前最新观测 obs 和环境元数据 info 计算是否触发转移。

        Args:
            obs: 当前原语处理后的最新观测字典（包含末端位姿、图像、力矩等）。
            info: 当前原语的运行元数据字典（包含当前步数 step、目标位姿等）。

        Returns:
            Outcome 对象，标明是否触发转移 (terminated/truncated 为 True)。
        """
        raise NotImplementedError

    def check(self, obs: dict, info: dict) -> bool:
        """辅助方法：简要检查该边是否被触发（返回 True 或 False）。"""
        result = self.evaluate(obs=obs, info=info)
        return result.terminated or result.truncated


# =============================================================================
# 转移条件类型 1: 无条件跳转 (Always)
# =============================================================================

@Transition.register_subclass("always")
@dataclass
class Always(Transition):
    """无条件立即跳转边。

    无论观测和 info 是什么，进入该原语并步进一次后立刻触发跳转到 target。
    常用于单帧原语（例如瞬时下发一次固定指令后立即切入下一阶段）。
    """
    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        return Outcome(
            terminated=True,
            reward=self.additional_reward,
            reason="always" if self.reason is None else self.reason
        )


# =============================================================================
# 转移条件类型 2: 事件/按键信号触发 (OnEvent / OnSuccess / OnFailure)
# =============================================================================

@Transition.register_subclass("on_event")
@dataclass
class OnEvent(Transition):
    """当 ``info`` 字典中指定的布尔事件标志为 True 时触发跳转。

    常用于人机协同、遥操作介入、踩脚踏开关、或键盘按下特定按键触发的状态跳转。
    """

    event_key: Any = ""
    """需要检测的 info 字典中的事件键名。"""

    default_reason: ClassVar[str] = "event"

    def _resolve_event_key(self) -> Any:
        return self.event_key

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        return Outcome(
            terminated=bool(info.get(self._resolve_event_key(), False)),
            reward=self.additional_reward,
            reason=self.default_reason if self.reason is None else self.reason,
        )


@Transition.register_subclass("on_success")
@dataclass
class OnSuccess(OnEvent):
    """预置成功事件转移边：当收到人类遥操标记的成功事件 (TeleopEvents.SUCCESS) 时触发。"""
    success_key: str = TeleopEvents.SUCCESS
    additional_reward: float = 1.0
    default_reason: ClassVar[str] = "success"

    def _resolve_event_key(self) -> Any:
        return self.success_key


@Transition.register_subclass("on_failure")
@dataclass
class OnFailure(OnEvent):
    """预置失败事件转移边：当收到人类遥操标记的失败事件 (TeleopEvents.FAILURE) 时触发。"""
    failure_key: str = TeleopEvents.FAILURE
    additional_reward: float = 0.0
    default_reason: ClassVar[str] = "failure"

    def _resolve_event_key(self) -> Any:
        return self.failure_key


# =============================================================================
# 转移条件类型 3: 观测数值阈值判定 (OnObservationThreshold)
# =============================================================================

@Transition.register_subclass("on_observation_threshold")
@dataclass
class OnObservationThreshold(Transition):
    """当观测字典中的某个数值达到设定阈值时触发跳转。

    常用于力控接触检测（例如力传感器 Fz > 5.0 N 说明触底）、距离接近检测等。
    """
    obs_key: str = ""
    """观测字典中要比对的键名（支持形如 'robot.force.z' 的层级路径）。"""

    threshold: float = 0.0
    """比对的数值阈值。"""

    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"
    """比较运算符：ge(>=), gt(>), le(<=), lt(<), eq(==), ne(!=)。"""

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        value = to_scalar(resolve_value(obs, self.obs_key))
        fired = compare(value, self.threshold, self.operator)
        return Outcome(
            terminated=fired,
            truncated=False,
            reward=self.additional_reward,
            reason="observation_threshold" if self.reason is None else self.reason
        )


# =============================================================================
# 转移条件类型 4: 步数时间上限判定 (OnTimeLimit)
# =============================================================================

@Transition.register_subclass("on_time_limit")
@dataclass
class OnTimeLimit(Transition):
    """当当前原语内部运行步数达到上限时触发跳转 (truncated=True)。

    常用于：
    1. 夹爪开合：等待夹爪闭合动作执行 10 步后自动切到抬升阶段。
    2. 超时保护/容错分支：某策略探索超过 100 步未成功，超时跳转到安全回退节点。
    """
    max_steps: int = 0
    """最大允许运行步数。"""

    step_key: str = "step"
    """info 中记录步数的键名（默认读取 info['step']）。"""

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        current_steps = int(to_scalar(resolve_value(info, self.step_key)))
        fired = current_steps >= self.max_steps
        return Outcome(
            terminated=False,
            truncated=fired,
            reward=self.additional_reward,
            reason="time_limit" if self.reason is None else self.reason
        )


# =============================================================================
# 转移条件类型 5: 神经网络奖励/成功分类器 (RewardClassifierTransition)
# =============================================================================

@Transition.register_subclass("reward_classifier")
@dataclass
class RewardClassifierTransition(Transition):
    """当预训练的视觉/状态奖励分类网络预测当前观测达到成功概率阈值时触发。

    在后台运行 LeRobot 预训练的分类模型，对当前图像或机器人状态进行前向推理，
    当预测成功概率 >= threshold 时触发状态机跳转。
    """

    pretrained_path: str = ""
    """预训练分类器模型的本地权重路径。"""

    metric_key: str | None = None
    """若在 info 中已提前计算好概率，可直接填 key 名称，避免重复前向推理。"""

    threshold: float = 0.9
    """成功概率判定阈值（默认 0.9 即 90% 确信度）。"""

    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"
    """比较运算符（默认 >=）。"""

    additional_reward: float = 1.0
    """触发成功时赋予的奖励。"""

    device: str = "cuda"
    """推理设备 ('cuda' 或 'cpu')。"""

    image_size: int = 128
    """输入图像缩放尺寸。"""

    prob_info_key: str = "reward_classifier_prob"
    """将推理概率写入 info 字典时所用的键名。"""

    _CACHE: ClassVar[dict[tuple[str, str], tuple[Any, Any]]] = {}

    def _load(self) -> tuple[Any, Any]:
        """延迟加载并缓存模型权重及预处理器。"""
        key = (self.pretrained_path, self.device)
        if key not in self._CACHE:
            if not self.pretrained_path:
                raise ValueError("RewardClassifierTransition 需要提供 'pretrained_path'。")
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
            from lerobot.processor import PolicyProcessorPipeline
            from lerobot.processor.converters import batch_to_transition, transition_to_batch

            from share.policies.reward_classifier import StateRewardClassifier

            config = PreTrainedConfig.from_pretrained(self.pretrained_path)
            policy_cls = StateRewardClassifier if config.type == "state_reward_classifier" else Classifier
            policy = policy_cls.from_pretrained(self.pretrained_path)
            policy.config.device = self.device
            policy.to(self.device)
            policy.eval()
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                self.pretrained_path,
                config_filename="classifier_preprocessor.json",
                overrides={"device_processor": {"device": self.device}},
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            )
            self._CACHE[key] = (policy, preprocessor)
        return self._CACHE[key]

    def _success_probability(self, obs: dict[str, Any]) -> float:
        """从当前观测中提取图像和状态，通过分类网络预测成功概率。"""
        import torch

        from lerobot.utils.constants import OBS_IMAGE, OBS_STATE

        from share.policies.reward_classifier import StateRewardClassifier

        policy, preprocessor = self._load()
        image_keys = [key for key in policy.config.input_features if key.startswith(OBS_IMAGE)]
        needs_state = isinstance(policy, StateRewardClassifier)

        batch: dict[str, Any] = {}
        for key in image_keys + ([OBS_STATE] if needs_state else []):
            value = obs[key]
            value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if key in image_keys:
                if value.dtype == torch.uint8:
                    value = value.float() / 255.0
                if value.dim() == 3:
                    value = value.unsqueeze(0)
            elif value.dim() == 1:
                value = value.unsqueeze(0)
            batch[key] = value

        with torch.no_grad():
            batch = preprocessor(batch)
            for key in image_keys:
                if batch[key].shape[-2:] != (self.image_size, self.image_size):
                    batch[key] = torch.nn.functional.interpolate(
                        batch[key],
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
            images = [batch[key] for key in image_keys]
            if needs_state:
                output = policy.predict(images, state=batch[OBS_STATE])
            else:
                output = policy.predict(images)
        return float(output.probabilities.reshape(-1)[0].item())

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        probability = (
            float(to_scalar(resolve_value(info, self.metric_key)))
            if self.metric_key is not None
            else self._success_probability(obs)
        )
        info[self.prob_info_key] = probability
        fired = compare(probability, self.threshold, self.operator)
        return Outcome(
            terminated=fired,
            truncated=False,
            reward=self.additional_reward if fired else 0.0,
            reason="reward_classifier" if fired and self.reason is None else self.reason if fired else None,
        )


# =============================================================================
# 转移条件类型 6: Info 字典值相等判定 (OnInfoEquals)
# =============================================================================

@Transition.register_subclass("on_info_equals")
@dataclass
class OnInfoEquals(Transition):
    """当 ``info`` 中的指定键等于某个特定离散值时触发跳转。

    常用于原语执行完毕后输出了不同的分类结果（例如 info['outcome'] == 'slip' 发生滑落），
    由此路由到不同的故障处理或分支原语。
    """
    key: str = ""
    """info 字典中的键名。"""

    value: Any = None
    """期望匹配的目标值。"""

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        fired = resolve_value(info, self.key) == self.value
        return Outcome(
            terminated=fired,
            reward=self.additional_reward if fired else 0.0,
            reason="info_equals" if fired and self.reason is None else self.reason if fired else None,
        )


# =============================================================================
# 转移条件类型 7: 复合逻辑与组合器 (AllOf)
# =============================================================================

@Transition.register_subclass("all_of")
@dataclass
class AllOf(Transition):
    """逻辑与 (AND) 组合器：只有当列表中所有的子条件在同一帧同时满足时，才触发跳转。

    例如：既要满足“到达目标位姿 (OnTargetPoseReached)”，又要满足“夹爪完全闭合 (OnObservationThreshold)”。
    子条件的 source/target 会被忽略，仅使用它们的 evaluate() 逻辑。
    """
    conditions: list[Transition] = field(default_factory=list)
    """所有需要同时满足的子转移条件列表。"""

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        outcomes = [condition.evaluate(obs=obs, info=info) for condition in self.conditions]
        fired = bool(outcomes) and all(o.terminated or o.truncated for o in outcomes)
        return Outcome(
            terminated=fired,
            reward=self.additional_reward if fired else 0.0,
            reason="all_of" if fired and self.reason is None else self.reason if fired else None,
        )


# =============================================================================
# 转移条件类型 8: 触发时自动记录末端世界位姿 (RecordWorldPoseTransition)
# =============================================================================

@Transition.register_subclass("record_world_pose")
@dataclass
class RecordWorldPoseTransition(Transition):
    """包装另一个 Transition；每当该条件触发时，将机械臂当前世界坐标位姿追加记录到 JSONL 文件中。

    常用于自动化标定、采集接触点/插孔完成位姿分布。
    """

    condition: Transition | None = None
    """被包装的实际转移条件。"""

    output_file: str = ""
    """位姿记录输出的 JSONL 文件路径。"""

    robot_name: str = DEFAULT_ROBOT_NAME
    """机械臂名称。"""

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        outcome = self.condition.evaluate(obs=obs, info=info)
        if outcome.terminated or outcome.truncated:
            world_pose = self._world_pose(obs)
            record = {"pose": world_pose, "time": datetime.now().isoformat(timespec="seconds")}
            with open(self.output_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Recorded world pose {[round(v, 4) for v in world_pose]} to {self.output_file}")
        return outcome

    def _world_pose(self, obs: dict[str, Any]) -> list[float]:
        """将观测中的相对任务位姿结合 task_frame_origin 还原为绝对世界位姿。"""
        task_pose = get_robot_pose_from_observation(obs, self.robot_name)
        origin_keys = [f"{self.robot_name}.{axis}.task_frame_origin" for axis in TASK_FRAME_AXIS_NAMES]
        origin = [float(obs[key]) for key in origin_keys] if all(key in obs for key in origin_keys) else None
        return task_pose_to_world_pose(task_pose, origin)


# =============================================================================
# 转移条件类型 9: 到达目标位姿判定 (OnTargetPoseReached) —— 最常用
# =============================================================================

@Transition.register_subclass("on_target_pose_reached")
@dataclass
class OnTargetPoseReached(Transition):
    """当机械臂末端 (EE) 实际位姿与当前原语的目标位姿 (primitive_target_pose) 误差在容差范围内时触发。

    这是机械臂状态机中使用频率最高的边条件！
    特性：
    1. 轴向过滤：可以指定只检查特定轴（如 axes=['z'] 或 axes=['x', 'y', 'z']）。
    2. 欧拉角循环连续性：对于旋转轴（roll/pitch/yaw），自动使用 atan2(sin(e), cos(e)) 消除 [-pi, pi] 跳变误差。
    3. 支持各轴独立容差或统一标量容差。
    """
    robot_name: str | None = None
    """要检测的机械臂名称（若为 None 则自动检测所有配置的机械臂）。"""

    axes: list[int | str] | None = None
    """需要比对的目标轴列表。支持数字 [0, 1, 2] 或名称 ['x', 'y', 'z']。
    若为 None，则自动从原语的 TaskFrame 配置中推断当前控制的目标轴。"""

    tolerance: float | list[float] = 0.01
    """容许误差范围。若为标量（如 0.01 表示位置 1cm、角度 0.01rad），若为 6 维列表则代表各轴独立容差。"""

    target_key: str = PRIMITIVE_TARGET_POSE_INFO_KEY
    """info 字典中保存目标位姿的键名（默认 'primitive_target_pose'）。"""

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        """检查当前末端位姿是否已收敛到达目标位姿。

        Args:
            obs: 包含当前机器人状态观测字典。
            info: 包含当前原语目标位姿的字典。

        Returns:
            Outcome 对象（若满足容差要求则 terminated=True）。
        """
        targets = resolve_value(info, self.target_key)
        robot_names = [self.robot_name] if self.robot_name is not None else sorted(targets)
        fired = bool(robot_names)
        for robot_name in robot_names:
            current_pose = get_robot_pose_from_observation(obs, robot_name)
            target_pose = [float(v) for v in targets[robot_name]]
            
            axes = self._resolved_axes(info=info, robot_name=robot_name)
            tolerances = self._resolved_tolerances()
            for axis in axes:
                error = current_pose[axis] - target_pose[axis]
                # 旋转轴 (3:roll, 4:pitch, 5:yaw) 做角度环绕处理，避免 -pi 和 +pi 处的突变
                if axis >= 3:
                    error = math.atan2(math.sin(error), math.cos(error))
                if abs(error) > tolerances[axis]:
                    fired = False
                    break
            if not fired:
                break

        return Outcome(
            terminated=fired,
            reward=self.additional_reward if fired else 0.0,
            reason="target_pose_reached" if fired and self.reason is None else self.reason if fired else None,
        )

    def _resolved_axes(self, info: dict[str, Any], robot_name: str) -> list[int]:
        """推断或解析需要检测的轴索引列表。"""
        if self.axes is not None:
            return [axis_to_index(axis) for axis in self.axes]
        inferred_axes = info.get(DEFAULT_TARGET_POSE_AXES_INFO_KEY, {})
        if robot_name in inferred_axes and inferred_axes[robot_name]:
            return [int(axis) for axis in inferred_axes[robot_name]]
        return [0, 1, 2, 3, 4, 5]

    def _resolved_tolerances(self) -> list[float]:
        """将标量或列表容差规范化为长度为 6 的 float 列表。"""
        if isinstance(self.tolerance, (int, float)):
            return [float(self.tolerance)] * 6
        if len(self.tolerance) != 6:
            raise ValueError("OnTargetPoseReached.tolerance 必须是标量或长度为 6 的列表。")
        return [float(v) for v in self.tolerance]
