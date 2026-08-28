from dataclasses import dataclass, field

from lerobot.envs import EnvConfig
from lerobot.cameras import CameraConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.robots import RobotConfig

from share.envs.manipulation_primitive_net.transitions import Transition
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.rl.force_backoff import ForceBackoffConfig
from share.utils.constants import DEFAULT_ROBOT_NAME


@dataclass
class ManipulationPrimitiveNetConfig(EnvConfig):
    """MP-Net 状态机网络的顶层配置类（支持序列化）。

    用于声明操作原语网络中包含的所有节点（primitives）、边（transitions）
    以及全局硬件环境（机械臂、遥操设备、相机、帧率等）。
    """

    # =========================================================================
    # 核心状态机拓扑字段
    # =========================================================================
    start_primitive: str | None = None
    """任务正式开始时激活的第一个原语节点名称（若不填，默认取 primitives 的第一个）。"""

    reset_primitive: str | None = None
    """全局 reset 时机器人首先执行的复位原语节点名称（例如机械臂回到初始安全位姿）。"""

    primitives: dict[str, ManipulationPrimitiveConfig] = field(default_factory=dict)
    """所有原语节点的配置字典。键是原语名称（如 'move_above_A'），值是对应的原语配置对象。"""

    transitions: list[Transition] = field(default_factory=list)
    """状态转移边列表。定义了在什么条件下从源原语 (source) 跳转到目标原语 (target)。"""

    # =========================================================================
    # 硬件与控制参数
    # =========================================================================
    fps: int = 10
    """控制主循环的运行帧率 (Hz)。"""

    robot: RobotConfig | dict[str, RobotConfig] | None = None
    """机械臂硬件配置。可以是单个机械臂配置，也可以是多机械臂字典。"""

    teleop: TeleoperatorConfig | dict[str, TeleoperatorConfig] | None = None
    """遥操作输入设备配置（如 3D 鼠标）。不使用遥操作时可为 None。"""

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    """相机配置字典。例如 {'wrist': CameraConfig(...), 'base': CameraConfig(...)}。"""

    force_backoff: ForceBackoffConfig = field(default_factory=ForceBackoffConfig)
    """受力过大时的安全退让 (Force Backoff) 保护策略配置。"""

    @property
    def gym_kwargs(self) -> dict:
        """传递给 Gym 环境构造函数的额外参数。"""
        return {}

    def make(self):
        """工厂方法：根据当前配置创建对应的 ManipulationPrimitiveNet 环境实例。"""
        from .env_manipulation_primitive_net import ManipulationPrimitiveNet
        return ManipulationPrimitiveNet(self)

    def __post_init__(self):
        """在配置初始化后，自动校验 MP-Net 状态机图结构的合法性（拓扑有效性检查）。

        检查内容包括：
        1. 规范化机械臂与遥操作字典格式（单设备自动包装为以 'default' 为键的字典）。
        2. 校验 start_primitive 和 reset_primitive 是否存在于 primitives 字典中。
        3. 校验所有 transition 的 source 和 target 是否存在。
        4. 【死胡同检测】：非终止节点 (is_terminal=False) 必须至少有一条出度转移边。
        5. 【可达性检测】：所有终止节点 (is_terminal=True) 必须从 start_primitive 能够遍历到达。
        """
        # 1. 规范化机械臂和遥操作字典（统一包装为字典结构）
        self.robot = self.robot if isinstance(self.robot, dict) else {DEFAULT_ROBOT_NAME: self.robot}
        self.teleop = self.teleop if isinstance(self.teleop, dict) else {DEFAULT_ROBOT_NAME: self.teleop}
        for name, robot_cfg in self.robot.items():
            if robot_cfg is not None:
                robot_cfg.cameras = {}

        # 2. 校验必须包含至少一个原语节点
        assert self.primitives, "MP-Net 必须配置至少一个原语 (primitives)"
        primitive_names = list(self.primitives.keys())
        if self.start_primitive is None:
            self.start_primitive = primitive_names[0]
        if self.reset_primitive is None:
            self.reset_primitive = primitive_names[0]
        if self.start_primitive not in primitive_names:
            raise ValueError(f"start_primitive '{self.start_primitive}' 不存在于 primitives 列表中。")

        # 3. 构建出度转移边邻接表，并校验边的合法性
        outgoing_edges: dict[str, set[str]] = {name: set() for name in primitive_names}

        for transition in self.transitions:
            if transition.source not in primitive_names:
                raise ValueError(f"转移条件边的起点 source '{transition.source}' 不存在于 primitives 列表中。")
            if transition.target not in primitive_names:
                raise ValueError(f"转移条件边的终点 target '{transition.target}' 不存在于 primitives 列表中。")

            outgoing_edges[transition.source].add(transition.target)

        # 辅助函数：使用 BFS 遍历计算从指定节点出发能够到达的所有节点集合
        def _reachable_from(start: str) -> set[str]:
            visited = {start}
            frontier = [start]

            while frontier:
                node = frontier.pop()
                for nxt in outgoing_edges[node]:
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    frontier.append(nxt)

            return visited

        # 4. 【死胡同检测】：防止状态机走到某个普通节点后无路可走而卡死
        for primitive_name, primitive_cfg in self.primitives.items():
            is_terminal = bool(getattr(primitive_cfg, "is_terminal", False))
            if not is_terminal and not outgoing_edges[primitive_name]:
                raise ValueError(
                    "检测到非终止的死胡同原语（没有配置任何出度转移条件边）："
                    f"'{primitive_name}'。请将其标记为 is_terminal=True 或为其添加一条 transition。"
                )

        # 5. 【终止节点可达性检测】：防止设置了无法到达的孤立终止节点
        reachable_from_start = _reachable_from(self.start_primitive)
        unreachable_terminals = sorted(
            name
            for name, primitive_cfg in self.primitives.items()
            if bool(getattr(primitive_cfg, "is_terminal", False)) and name not in reachable_from_start
        )
        if unreachable_terminals:
            raise ValueError(
                f"从起始原语 '{self.start_primitive}' 无法到达以下终止原语节点："
                f"{', '.join(unreachable_terminals)}"
            )

    @property
    def terminals(self) -> list[str]:
        """获取状态机中所有被标记为终止节点 (is_terminal=True) 的原语名称列表。"""
        return [k for k, v in self.primitives.items() if v.is_terminal]
