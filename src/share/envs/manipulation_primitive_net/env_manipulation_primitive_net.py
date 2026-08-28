import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.processor import create_transition, TransitionKey, EnvTransition
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.utils.constants import ACTION
from lerobot.cameras import Camera
from lerobot.teleoperators import Teleoperator
from lerobot.robots import Robot
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import Transition

from share.envs.manipulation_primitive.config_manipulation_primitive import PrimitiveEntryContext
from share.envs.manipulation_primitive.task_frame import ControlMode
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.processor import create_transition, TransitionKey, EnvTransition
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.utils.constants import ACTION
from lerobot.cameras import Camera
from lerobot.teleoperators import Teleoperator
from lerobot.robots import Robot
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import Transition

from share.envs.manipulation_primitive.config_manipulation_primitive import PrimitiveEntryContext
from share.envs.manipulation_primitive.task_frame import ControlMode
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import DEFAULT_TARGET_POSE_AXES_INFO_KEY
from share.envs.utils import observed_task_frame_origins, task_frame_origins
from share.teleoperators import TeleopEvents


class ManipulationPrimitiveNet(gym.Env):
    """操作原语网络 (MP-Net) 的 Gymnasium 环境包装器。

    负责将多个离散的操作原语 (Manipulation Primitive) 节点通过类型化的状态转移边 (Transitions)
    串联为一个统一执行的有向状态机网络。
    """

    # =========================================================================
    # 功能块 1: 初始化与硬件连接
    # =========================================================================

    def __init__(self, config: ManipulationPrimitiveNetConfig):
        """初始化 MP-Net 状态机环境。

        Args:
            config: MP-Net 配置对象，包含所有原语配置 (primitives) 和状态转移边 (transitions)。
        """

        self.config = config

        # 初始化并连接底层硬件设备 (机械臂、遥操作主端、相机)
        self.robot_dict, self.teleop_dict, self.cameras = self.connect()

        self._envs = {}
        self._env_processors = {}
        self._action_processors = {}
        self._transitions: dict[str, list[Transition]] = {}
        self._shared_runtime_values: dict[str, Any] = {}

        # 遍历构建每一个原语对应的独立 Env 实例以及动作/观测处理器
        for name, primitive in self.config.primitives.items():
            env, env_processor, action_processor = primitive.make(
                self.robot_dict,
                self.teleop_dict,
                self.cameras,
                device=getattr(self.config, "device", "cpu")
            )
            self._envs[name] = env
            self._env_processors[name] = env_processor
            self._action_processors[name] = action_processor
            self._transitions[name] = []
            
            # 为各原语挂载跨原语共享黑板 (用于跨原语传递动态路点/目标)
            attach_shared_runtime_values = getattr(env, "attach_shared_runtime_values", None)
            if callable(attach_shared_runtime_values):
                attach_shared_runtime_values(self._shared_runtime_values)

        # 按 source 节点对转移条件边进行分桶索引
        for transition in self.config.transitions:
            self._transitions[transition.source].append(transition)

        # 初始激活原语设定为复位原语
        self._active = self.config.reset_primitive
        self._last_reset_info: dict[str, Any] = {}
        self._pending_entry_context: PrimitiveEntryContext | None = None
        self._episode_step_count = 0      # 整局 Episode 总步数
        self._primitive_step_count = 0    # 当前活动原语内部步数
        self._needs_full_reset = True     # 是否需要完整复位标志
        self._step_info: dict[str, Any] = {}
        self._last_stepped_primitive: str | None = None

    # =========================================================================
    # 功能块 2: 属性与状态查询
    # =========================================================================

    @property
    def active_primitive(self) -> str:
        """获取当前处于激活状态的原语名称。"""
        return self._active

    @property
    def action_dim(self) -> int:
        """获取当前激活原语所期望的策略动作维度。"""
        return self.config.primitives[self._active].features[ACTION].shape[0]

    @property
    def in_terminal(self):
        """检查当前激活原语是否为全局终止节点 (is_terminal=True)。"""
        return self.config.primitives[self._active].is_terminal

    def connect(self) -> tuple[dict[str, "Robot"], dict[str, "Teleoperator"], dict[str, "Camera"]]:
        """连接并初始化该 MP-Net 所需的所有硬件与输入输出设备。

        Returns:
            元组 ``(robot_dict, teleop_dict, cameras)``，以配置名称为键，
            可供所有原语子环境安全共享使用。
        """
        assert self.config.robot is not None, "实体机器人环境必须提供 robot 配置"

        from lerobot.cameras import make_cameras_from_configs
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.robots import make_robot_from_config

        # 1. 实例化并连接机械臂设备
        robot_dict = {}
        for name in self.config.robot:
            robot_dict[name] = make_robot_from_config(self.config.robot[name])
            robot_dict[name].connect()

        # 2. 实例化并连接遥操作主端设备
        teleop_dict = {}
        for name in self.config.teleop:
            teleop_dict[name] = make_teleoperator_from_config(self.config.teleop[name])
            teleop_dict[name].connect()

        # 3. 实例化并连接所有配置的相机
        cameras = make_cameras_from_configs(self.config.cameras)
        for name in cameras:
            cameras[name].connect()

        return robot_dict, teleop_dict, cameras

    # =========================================================================
    # 功能块 3: 外层 Step 与 Reset 公共接口
    # =========================================================================

    def step(self, action: np.ndarray | torch.Tensor) -> EnvTransition:
        """执行当前激活原语的单步动作，并判定是否发生状态转移边跳转。

        Args:
            action: 当前激活原语对应的一维策略动作向量。

        Returns:
            执行后的处理后转换对象 (EnvTransition)。若触发了状态转移，
            返回信息中的元数据会标明下一个目标原语。
        """
        if self._needs_full_reset:
            raise RuntimeError("step() called after MP-Net episode finished; call reset() before stepping again.")

        transition = self._step_env_and_check_transitions(action)
        self._needs_full_reset |= self.in_terminal

        return transition

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> EnvTransition:
        """复位 MP-Net 环境，或在原语转移后执行新原语的初始化入口。

        Args:
            seed: 可选的 Gym 随机种子。
            options: 可选的复位参数字典，会透传给各原语子环境。

        Returns:
            进入激活原语后的处理后复位转换对象 (EnvTransition)。
        """
        super().reset(seed=seed)
        self._primitive_step_count = 0
        if self._needs_full_reset:
            transition = self._full_reset(seed=seed, options=options)
            self._needs_full_reset = False
            self._episode_step_count = 0
            return transition

        return self._enter_active_primitive(seed=seed, options=options, entry_context=self._pending_entry_context)

    def close(self):
        """断开所有硬件设备连接并关闭各原语子环境，释放系统资源。"""
        for camera in self.cameras.values():
            camera.disconnect()
        for robot in self.robot_dict.values():
            robot.disconnect()
        for teleop in self.teleop_dict.values():
            if getattr(teleop, "is_connected", False):
                teleop.disconnect()

        keys = list(self._envs.keys())
        for k in keys:
            self._envs[k].close()
            del self._envs[k]
            del self._action_processors[k]
            del self._env_processors[k]

    def stop(self) -> None:
        """保持最近步进原语所下发的末端位姿，使机器人悬停静止。"""
        primitive_name = self._last_stepped_primitive or self._active
        env = self._envs.get(primitive_name)
        if env is None:
            return
        env.stop()

    def request_full_reset(self) -> None:
        """标记下一次 ``reset()`` 必须启动全新的 MP-Net Episode。"""
        self._needs_full_reset = True

    def set_step_info(self, info: dict[str, Any] | None) -> None:
        """设置每次外层 ``step(action)`` 调用时注入的基础 Info 标志。"""

        self._step_info = {} if info is None else dict(info)

    # =========================================================================
    # 功能块 4: 核心状态机步进与边条件判定引擎 (心脏逻辑)
    # =========================================================================

    def _step_env_and_check_transitions(self, action: torch.Tensor) -> EnvTransition:
        """执行单步原语动作，并评估至多一条出度转移条件边。

        标准执行六步法：
        1. 动作预处理 (Action Processors)
        2. 底层环境单步执行 (env.step)
        3. 动作回填与遥操作干预覆盖
        4. 观测后处理 (Env Processors)
        5. 注入状态机运行时元数据与步数统计 (Info)
        6. 遍历评估 Transition 边，触发状态跳转与 PrimitiveEntryContext 传递

        Args:
            action: 当前原语的一维动作张量。

        Returns:
            经过完整处理、包含状态机转移元数据的 EnvTransition 对象。
        """
        self._episode_step_count += 1
        self._primitive_step_count += 1
        active = self._active
        if active not in self._envs:
            raise KeyError(f"未知的激活原语名称: '{active}'。")
        primitive = self.config.primitives[active]
        self._last_stepped_primitive = active

        # ---------------------------------------------------------------------
        # 步骤 1: 动作预处理
        # ---------------------------------------------------------------------
        info = dict(self._step_info)
        if primitive.policy is None and not getattr(self._envs[active], "uses_autonomous_step", False):
            info[TeleopEvents.IS_INTERVENTION] = True

        action_transition = create_transition(action=action, info=info)
        processed_action_transition = self._action_processors[active](action_transition)

        # ---------------------------------------------------------------------
        # 步骤 2: 底层子环境执行物理单步
        # ---------------------------------------------------------------------
        raw_obs, reward, terminated, truncated, info = self._envs[active].step(
            processed_action_transition[TransitionKey.ACTION]
        )

        # ---------------------------------------------------------------------
        # 步骤 3: 读取 Info 并视情况覆盖动作 (如人机协同干预)
        # ---------------------------------------------------------------------
        complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
        info.update(processed_action_transition[TransitionKey.INFO].copy())

        if info.get(TeleopEvents.IS_INTERVENTION, False) and TELEOP_ACTION_KEY in complementary_data:
            action_to_record = complementary_data[TELEOP_ACTION_KEY]
        else:
            action_to_record = action

        # ---------------------------------------------------------------------
        # 步骤 4: 观测后处理
        # ---------------------------------------------------------------------
        transition = create_transition(
            observation=raw_obs,
            action=action_to_record,
            reward=reward + processed_action_transition[TransitionKey.REWARD],
            done=terminated or processed_action_transition[TransitionKey.DONE],
            truncated=truncated or processed_action_transition[TransitionKey.TRUNCATED],
            info=info,
            complementary_data=complementary_data,
        )
        processed_transition = self._env_processors[active](transition)
        processed_obs = processed_transition[TransitionKey.OBSERVATION]

        # ---------------------------------------------------------------------
        # 步骤 5: 构建并注入状态机元数据与计数器
        # ---------------------------------------------------------------------
        info = processed_transition.get(TransitionKey.INFO, {})
        info["step"] = self._primitive_step_count
        info["primitive_step"] = self._primitive_step_count
        info["episode_step"] = self._episode_step_count
        info["transition_from"] = active
        info["transition_to"] = active
        info["transition_reason"] = None
        info[DEFAULT_TARGET_POSE_AXES_INFO_KEY] = self._default_target_pose_axes(primitive)

        # ---------------------------------------------------------------------
        # 步骤 6: 遍历出度转移条件边 (至多触发一次跳转)
        # ---------------------------------------------------------------------
        for transition in self._transitions[self._active]:
            result = transition.evaluate(obs=processed_obs, info=info)
            if not (result.terminated or result.truncated):
                continue

            # 边条件命中触发：打包上下文并切换状态
            target = transition.target
            self._pending_entry_context = PrimitiveEntryContext(
                source_primitive=active,
                target_primitive=target,
                observation=dict(processed_obs),
                # 优先提取观测中附带的任务坐标系原点；若无则回退到源原语的静态原点
                task_frame_origin=observed_task_frame_origins(
                    processed_obs, task_frame_origins(primitive)
                ),
                reason=result.reason,
            )
            self._primitive_step_count = 0
            self._active = target
            
            # 立即触发新原语的入口初始化
            self._enter_active_primitive(None, None, self._pending_entry_context)

            # 合并奖励与终止标志
            processed_transition[TransitionKey.REWARD] += result.reward
            processed_transition[TransitionKey.DONE] |= result.terminated
            processed_transition[TransitionKey.TRUNCATED] |= result.truncated
            info["transition_to"] = target
            info["transition_reason"] = result.reason
            break

        info.pop(DEFAULT_TARGET_POSE_AXES_INFO_KEY, None)
        processed_transition[TransitionKey.INFO] = info
        return processed_transition

    @staticmethod
    def _default_target_pose_axes(primitive: Any) -> dict[str, list[int]]:
        """计算原语中各 TaskFrame 默认需要对齐的目标控制轴索引列表。"""
        default_axes: dict[str, list[int]] = {}
        for name, frame in primitive.task_frame.items():
            axes = [
                axis
                for axis in range(len(frame.target))
                if frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] is None
            ]
            default_axes[name] = axes
        return default_axes

    # =========================================================================
    # 功能块 5: 回合复位与进入起始原语 (Reset 拓扑遍历)
    # =========================================================================

    def _step_reset_path_until_start(
        self, obs: dict[str, np.ndarray], info: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """沿复位图拓扑自动步进执行，直至到达 ``start_primitive`` 起始节点。

        Args:
            obs: 复位前滚过程中最新的处理后观测。
            info: 复位前滚过程中最新的 info 字典。

        Returns:
            到达 ``start_primitive`` 节点时的最终 ``(obs, info)`` 元组。
        """
        while self._active != self.config.start_primitive:
            start_loop_t = time.perf_counter()

            action = self.sample_action()
            transition = self._step_env_and_check_transitions(action)
            obs = transition[TransitionKey.OBSERVATION]
            info.update(transition[TransitionKey.INFO])  # 保留上一轮 info 以免被空字典覆盖

            if self._pending_entry_context is not None and self._active != self.config.start_primitive:
                entered = self._enter_active_primitive(
                    seed=None,
                    options=None,
                    entry_context=self._pending_entry_context,
                )
                obs = entered[TransitionKey.OBSERVATION]
                info.update(entered[TransitionKey.INFO])

            dt_load = time.perf_counter() - start_loop_t
            precise_sleep(1 / self.config.fps - dt_load)
        return obs, info

    def _full_reset(self, seed: int | None, options: dict[str, Any] | None) -> EnvTransition:
        """从 ``reset_primitive`` 开始执行完整的全局 Episode 复位。

        Args:
            seed: 用于派生各子环境随机种子的可选全局种子。
            options: 转发给各原语子环境的可选复位参数。

        Returns:
            就绪可供策略开始步进的初始原语处理后转换对象 (EnvTransition)。
        """
        self._pending_entry_context = None
        self._active = self.config.reset_primitive
        
        # 清空跨原语共享黑板
        shared_runtime_values = getattr(self, "_shared_runtime_values", None)
        if shared_runtime_values is not None:
            shared_runtime_values.clear()
            
        # 1. 复位底层机械臂仿真或硬件状态
        for robot_index, robot in enumerate(getattr(self, "robot_dict", {}).values()):
            reset_simulation = getattr(robot, "reset_simulation", None)
            if callable(reset_simulation):
                robot_seed = None if seed is None else seed + robot_index
                reset_simulation(seed=robot_seed)
                
        # 2. 复位遥操作设备状态
        for teleop in getattr(self, "teleop_dict", {}).values():
            reset_episode = getattr(teleop, "reset_episode", None)
            if callable(reset_episode):
                reset_episode()
                
        # 3. 复位除当前激活原语外的所有原语处理器及子环境
        for name, primitive in self.config.primitives.items():
            if name == self._active:
                continue
            self._env_processors[name].reset()
            self._action_processors[name].reset()
            env_seed = None if seed is None else seed + sum(ord(c) for c in name)
            self._envs[name].reset(seed=env_seed, options={} if options is None else dict(options))

        # 4. 进入复位起始原语
        transition = self._enter_active_primitive(seed=seed, options=options, entry_context=None)
        
        # 5. 若起始原语不是 start_primitive，自动执行复位路径直到到达 start_primitive
        if self._active != self.config.start_primitive:
            self._step_reset_path_until_start(
                obs=transition[TransitionKey.OBSERVATION],
                info=transition[TransitionKey.INFO],
            )
            transition = self._enter_active_primitive(
                seed=seed,
                options=options,
                entry_context=self._pending_entry_context,
            )
            
        self._episode_step_count = 0
        return transition

    # =========================================================================
    # 功能块 6: 进入活动原语与 on_entry 钩子处理
    # =========================================================================

    def _enter_active_primitive(
        self,
        seed: int | None,
        options: dict[str, Any] | None,
        entry_context: PrimitiveEntryContext | None,
    ) -> EnvTransition:
        """进入当前激活的原语，并调用其入口期生命周期钩子 (on_entry)。

        Args:
            seed: 用于派生当前子环境种子的可选种子。
            options: 转发给当前原语子环境的可选复位参数。
            entry_context: 包含上一原语退出时观测与任务坐标系原点的上下文信息对象。

        Returns:
            新进入原语对应的处理后复位转换对象 (EnvTransition)。
        """
        primitive = self.config.primitives[self._active]
        self._env_processors[self._active].reset()
        self._action_processors[self._active].reset()

        env_seed = None if seed is None else seed + sum(ord(c) for c in self._active)
        raw_obs, raw_info = self._envs[self._active].reset(
            seed=env_seed,
            options={} if options is None else dict(options),
        )
        transition = create_transition(observation=raw_obs, info=raw_info)
        processed_transition = self._env_processors[self._active](transition)
        processed_obs = processed_transition[TransitionKey.OBSERVATION]
        
        # 若未提供入口上下文 (例如首次启动)，则自动构造默认的初次 entry_context
        if entry_context is None:
            last_stepped_primitive = getattr(self, "_last_stepped_primitive", None)
            origin_primitive = self.config.primitives.get(last_stepped_primitive, primitive)
            entry_context = PrimitiveEntryContext(
                source_primitive=last_stepped_primitive,
                target_primitive=self._active,
                observation=dict(processed_obs),
                task_frame_origin=observed_task_frame_origins(
                    processed_obs, task_frame_origins(origin_primitive)
                ),
            )

        self._envs[self._active].reset_runtime_state()

        # 触发原语配置层定义的入口计算钩子 (计算相对目标/生成动态轨迹)
        primitive.on_entry(self._envs[self._active], entry_context)

        if hasattr(self._envs[self._active], "apply_task_frames"):
            self._envs[self._active].apply_task_frames()

        processed_transition[TransitionKey.INFO] = {
            **processed_transition.get(TransitionKey.INFO, {}),
            **getattr(self._envs[self._active], "_get_info", lambda: {})(),
        }
        self._last_reset_info = processed_transition[TransitionKey.INFO]
        self._pending_entry_context = None
        return processed_transition

    # =========================================================================
    # 功能块 7: 动作采样辅助方法
    # =========================================================================

    def sample_action(self, primitive: str | None = None) -> Any:
        """为指定或当前激活的原语随机采样一个合法的均匀分布动作张量。"""
        if primitive is None:
            primitive = self._active

        ft = self.config.primitives[primitive].features[ACTION]
        return 2 * torch.rand(size=ft.shape) - 1
