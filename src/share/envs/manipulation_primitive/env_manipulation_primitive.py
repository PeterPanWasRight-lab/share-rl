from typing import Any

import gymnasium
import numpy as np
from gymnasium.core import ObsType

from lerobot.utils.constants import OBS_IMAGES
from lerobot.cameras import Camera
from lerobot.robots import Robot

from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TASK_FRAME_AXIS_NAMES, TaskFrame
from share.envs.utils import check_task_frame_robot
from share.teleoperators import TeleopEvents


class ManipulationPrimitive(gymnasium.Env):
    """单个操作原语 (Manipulation Primitive) 的底层运行时 Gymnasium 环境。

    作为状态机中的“节点环境”，直接持有机械臂、相机句柄和 TaskFrame 控制契约。
    负责将算法/上层的动作下发给真实或仿真机器人，并收集传感器观测。
    """

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Camera],
        display_cameras: bool = False
    ):
        """初始化原语运行时环境。

        Args:
            task_frame: 每个机械臂对应的任务坐标系对象字典（由原语 Config 定义并在入口期更新）。
            robot_dict: 已连接的机械臂对象字典（从 MP-Net 共享传入）。
            cameras: 已连接的相机对象字典（从 MP-Net 共享传入）。
            display_cameras: 调用 render() 时是否弹出 OpenCV 实时视频预览窗口。
        """
        self.robot_dict = robot_dict
        self.task_frame = task_frame
        self.cameras = cameras
        self.display_cameras = display_cameras

        self.current_step = 0
        self._motor_keys: set[str] = set()
        self._is_task_frame_robot: dict[str, bool] = check_task_frame_robot(robot_dict)
        self._shared_runtime_values: dict[str, Any] | None = None
        self.reset_runtime_state()
        self.apply_task_frames()
        for name, robot in self.robot_dict.items():
            self._motor_keys.update([f"{name}.{key}" for key in robot._motors_ft])

    def step(self, action: dict[str, dict[str, float]]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """执行单步动作下发与观测收集（原语级 Step）。

        Args:
            action: 嵌套动作字典。第一层键为机械臂名称（如 'ur5e'），第二层键为控制轴动作（如 'x.ee_pos': 0.01）。

        Returns:
            标准的 Gym 五元组 (obs, reward, terminated, truncated, info)。
        """
        # 1. 确保任务坐标系已下发给底层控制器
        self.apply_task_frames()

        # 2. 将动作指令发送给机械臂
        for name, robot in self.robot_dict.items():
            robot.send_action(action.get(name, {}))

        # 3. 收集最新相机图像与机器人状态
        obs = self._get_observation()

        if self.display_cameras:
            self.render()

        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """重置原语内部计数器和运行时状态。

        注意：单个原语 reset 时默认不会自动让机械臂回零（避免破坏物理连续性），
        只重新下发 TaskFrame 并清空该原语的临时运行状态。
        """
        super().reset(seed=seed, options=options)

        self.apply_task_frames()
        self.current_step = 0
        self.reset_runtime_state()
        obs = self._get_observation()
        return obs, self._get_info()

    def render(self) -> None:
        """使用 OpenCV 实时窗口可视化当前所有相机的画面。"""
        import cv2
        current_observation = self._get_observation()
        if current_observation is not None and "pixels" in current_observation:
            for key, img in current_observation["pixels"].items():
                cv2.imshow(key, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self) -> None:
        """断开所有机械臂连接。"""
        for robot_dict in self.robot_dict.values():
            if robot_dict.is_connected:
                robot_dict.disconnect()

    def stop(self) -> None:
        """紧急/悬停制动：在所有轴上下发零位移指令，使机械臂保持当前位姿静止。"""
        for name, robot in self.robot_dict.items():
            if not getattr(robot, "is_connected", False):
                continue

            frame = self.task_frame.get(name)
            observation = robot.get_observation()
            action: dict[str, float] = {}

            if frame is not None and frame.space == ControlSpace.TASK and self._is_task_frame_robot.get(name, False):
                n = len(frame.target)
                stop_frame = TaskFrame(
                    origin=frame.origin,
                    space=frame.space,
                    target=[0.0] * n,
                    control_mode=[ControlMode.POS] * n,
                    policy_mode=[PolicyMode.RELATIVE] * n,
                )
                robot.set_task_frame(stop_frame)
                for axis_name in TASK_FRAME_AXIS_NAMES:
                    action[f"{axis_name}.ee_pos"] = 0.0
            else:
                joint_names = list(frame.joint_names) if frame is not None and frame.joint_names is not None else [
                    key.removesuffix(".pos")
                    for key in observation
                    if key.endswith(".pos") and ".ee_" not in key and key != "gripper.pos"
                ]
                for axis, joint_name in enumerate(joint_names):
                    key = f"{joint_name}.pos"
                    if key not in observation:
                        continue
                    value = float(observation[key])
                    action[key] = value
                    if frame is not None and axis < len(frame.target):
                        frame.target[axis] = value
                        frame.control_mode[axis] = ControlMode.POS

            if action:
                robot.send_action(action)

    def apply_task_frames(self) -> None:
        """将当前配置的 TaskFrame 下发同步给所有支持任务坐标系的机器人接口。"""
        for name, robot in self.robot_dict.items():
            if self._is_task_frame_robot.get(name, False):
                robot.set_task_frame(self.task_frame[name])

    def reset_runtime_state(self) -> None:
        """清空当前原语的运行时临时变量（目标位姿、完成标志、轨迹进度）。"""
        self._target_pose_info_key: str | None = None
        self._target_pose: dict[str, list[float]] = {}
        self._primitive_complete = False
        self._trajectory_progress = 0.0

    def attach_shared_runtime_values(self, shared_runtime_values: dict[str, Any]) -> None:
        """挂载由 MP-Net 管理的全局跨原语共享黑板字典。"""
        self._shared_runtime_values = shared_runtime_values

    def set_runtime_value(self, key: str, value: Any) -> None:
        """向跨原语黑板中写入动态运行时数据（例如上一原语抓取物体后的估计位置）。"""
        if self._shared_runtime_values is None:
            raise RuntimeError("当前原语未挂载共享黑板字典 (shared_runtime_values)。")
        self._shared_runtime_values[key] = value

    def get_runtime_value(self, key: str, default: Any = None) -> Any:
        """从跨原语黑板中读取先前原语写入的运行时数据。"""
        if self._shared_runtime_values is None:
            return default
        return self._shared_runtime_values.get(key, default)

    def set_target_pose(
        self,
        target_pose: dict[str, list[float]],
        info_key: str | None,
        *,
        update_task_frame: bool = True,
    ) -> None:
        """设置当前原语的目标位姿，并同步更新 TaskFrame。

        Args:
            target_pose: 各机械臂在当前原语坐标系下的 6D 目标位姿字典。
            info_key: 发布到 info 中的键名（如 'primitive_target_pose'）。
            update_task_frame: 是否同时修改 self.task_frame 的静态 target 字段。
        """
        self._target_pose = {name: list(pose) for name, pose in target_pose.items()}
        self._target_pose_info_key = info_key
        if not update_task_frame:
            return
        for name, pose in self._target_pose.items():
            if name not in self.task_frame:
                continue
            for axis in range(min(len(self.task_frame[name].target), len(pose))):
                self.task_frame[name].target[axis] = float(pose[axis])

    def _get_observation(self) -> dict[str, Any]:
        """抓取并聚合所有相机的当前帧图像以及机械臂的传感器状态，构建统一的原始观测字典。"""
        obs_dict = {}

        for cam_key, cam in self.cameras.items():
            obs_dict[f"{OBS_IMAGES}.{cam_key}"] = cam.async_read()

        for name in self.robot_dict:
            robot_dict_obs_dict = self.robot_dict[name].get_observation()
            obs_dict |= {f"{name}.{key}": robot_dict_obs_dict[key] for key in robot_dict_obs_dict}

        return obs_dict

    def _get_info(self) -> dict[str, Any]:
        """构建当前步的运行时元数据字典。"""
        info = {
            TeleopEvents.IS_INTERVENTION: False,
            "primitive_complete": bool(self._primitive_complete),
            "trajectory_progress": float(self._trajectory_progress),
        }
        if self._target_pose_info_key and self._target_pose:
            info[self._target_pose_info_key] = {
                name: list(pose) for name, pose in self._target_pose.items()
            }
        return info


class OpenLoopTrajectoryPrimitive(ManipulationPrimitive):
    """开环脚本轨迹原语环境（ManipulationPrimitive 的子类）。

    用于执行预先规划好的平滑几何轨迹（如直线插值、最小加加速度轨迹等）。
    在该原语下，外部传入的 action 会被忽略，环境每一步按照时间比例 alpha (0.0 -> 1.0)
    自动计算当前插值位姿并下发给机械臂，更新 trajectory_progress 和 primitive_complete 标志。
    """

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Camera],
        open_loop_config: Any,
        display_cameras: bool = False,
    ):
        """初始化开环脚本轨迹原语环境。

        Args:
            task_frame: 各机械臂对应的任务坐标系字典。
            robot_dict: 已连接的机械臂对象字典。
            cameras: 已连接的相机对象字典。
            open_loop_config: 轨迹生成配置对象（提供 trajectory_timing 和 target_pose_at 方法）。
            display_cameras: 是否开启 OpenCV 视频预览窗口。
        """
        self.open_loop_config = open_loop_config
        self._trajectory_substeps = 0
        super().__init__(
            task_frame=task_frame,
            robot_dict=robot_dict,
            cameras=cameras,
            display_cameras=display_cameras,
        )

    def reset_runtime_state(self) -> None:
        """重置开环轨迹的运行时插值状态与进度计时器。"""
        super().reset_runtime_state()
        self._start_pose: dict[str, list[float]] = {}
        self._duration_substeps = 0
        self._substeps_per_step = 1
        self._trajectory_substeps = 0

    def configure_trajectory(
        self,
        start_pose: dict[str, list[float]],
        target_pose: dict[str, list[float]],
        info_key: str | None,
    ) -> None:
        """在原语入口期 (on_entry) 初始化轨迹的起点、终点和插值步数。

        Args:
            start_pose: 各机械臂在当前任务坐标系下的起始位姿。
            target_pose: 各机械臂在当前任务坐标系下的目标终点位姿。
            info_key: 目标位姿发布到 info 字典中的键名。
        """
        self._start_pose = {name: list(pose) for name, pose in start_pose.items()}
        self._duration_substeps, self._substeps_per_step = self.open_loop_config.trajectory_timing(self.robot_dict)
        self.set_target_pose(
            target_pose=target_pose,
            info_key=info_key,
            update_task_frame=False,
        )
        self._set_live_task_frame_pose(start_pose)
        self._primitive_complete = False
        self._trajectory_progress = 0.0
        self._trajectory_substeps = 0

    def step(self, action: dict[str, dict[str, float]]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """执行一个外层 Step，内部按照配置的子步数 (substeps) 逐步插值下发位姿。

        Args:
            action: 外部策略动作（开环脚本轨迹下忽略此参数）。

        Returns:
            标准的 Gym 五元组 (obs, reward, terminated, truncated, info)。
        """
        substeps = max(1, int(self._substeps_per_step))
        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(substeps):
            self._trajectory_substeps += 1
            alpha = self._trajectory_substeps / float(self._duration_substeps)
            # 依据时间比例 alpha (0.0 ~ 1.0) 从轨迹生成器计算当前这一子步的插值目标位姿
            scripted_pose = self.open_loop_config.target_pose_at(
                alpha=alpha,
                start_pose=self._start_pose,
                goal_pose=self._target_pose,
            )
            self._set_live_task_frame_pose(scripted_pose)
            scripted_action = self._action_from_pose(scripted_pose)
            obs, step_reward, terminated, truncated, _info = super().step(scripted_action)
            reward += step_reward
            self._trajectory_progress = min(1.0, alpha)
            self._primitive_complete = alpha >= 1.0
            if terminated or truncated:
                break

        return obs, reward, terminated, truncated, self._get_info()

    def _set_live_task_frame_pose(self, pose_by_robot: dict[str, list[float]]) -> None:
        """更新当前运行时 TaskFrame 中设定的目标位姿。"""
        for name, pose in pose_by_robot.items():
            if name not in self.task_frame:
                continue
            for axis in range(min(len(self.task_frame[name].target), len(pose))):
                self.task_frame[name].target[axis] = float(pose[axis])

    def _action_from_pose(self, pose_by_robot: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        """将 6D 空间插值位姿转换为底层机械臂所期望的底层动作字典格式。"""
        action: dict[str, dict[str, float]] = {}
        for name, pose in pose_by_robot.items():
            frame = self.task_frame[name]
            action[name] = {
                frame.action_key_for_axis(axis): float(pose[axis])
                for axis in range(len(frame.target))
            }
        return action
