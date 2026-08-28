from __future__ import annotations

import time

import torch

from lerobot.utils.robot_utils import precise_sleep

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
    OpenLoopTrajectoryPrimitiveConfig,
    OpenLoopTrajectorySpec,
)

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    TASK_FRAME_AXIS_NAMES,
    TaskFrame,
)

from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)

from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)

from share.envs.manipulation_primitive_net.transitions import (
    OnSuccess,
    OnTimeLimit,
)

from share.robots.HRCrobot import (
    HRCrobotConfig,
)


# ==============================================================
# Basic parameters
# ==============================================================

ROBOT_NAME = "arm"

# MP-Net 外层状态机频率
OUTER_FPS = 30

ROBOT_IP = "192.168.1.10"

# HRCrobot servoL 类接口实际频率
SERVO_FREQUENCY = 100.0


# ==============================================================
# Pose definition
#
# SHaRe:
#
# [x, y, z, roll, pitch, yaw]
#
# xyz: meter
# rpy: rad
# ==============================================================

A_POSE = [
    0.40,
    -0.20,
    0.30,
    3.14159,
    0.0,
    0.0,
]

B_POSE = [
    0.55,
    0.10,
    0.30,
    3.14159,
    0.0,
    0.0,
]


# ==============================================================
# Gripper
# ==============================================================

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.0


# ==============================================================
# Validation helper
#
# 当前 SHaRe validate() 在 TASK-space 情况下会检查
# teleoperator 类型。
#
# 我们实际不使用 teleoperator。
#
# 这里只构造一个假的 delta teleop 描述用于通过 validation。
# ==============================================================

class _DeltaTeleopValidationStub:
    action_features = {
        "x.vel": float,
        "y.vel": float,
        "z.vel": float,
        "rx.vel": float,
        "ry.vel": float,
        "rz.vel": float,
    }


def _validation_teleop_dict(
    robot_dict,
    teleop_dict,
):
    result = dict(teleop_dict)

    for name in robot_dict:
        if name not in result:
            result[name] = (
                _DeltaTeleopValidationStub()
            )

    return result


# ==============================================================
# Autonomous primitives
#
# 当前 main 中 policy=None 默认可能走 intervention 路径。
#
# 因此对我们这种纯脚本 primitive，
# 明确告诉 MP-Net：
#
#     uses_autonomous_step = True
# ==============================================================

class AutonomousHoldPrimitiveConfig(
    ManipulationPrimitiveConfig
):
    """
    用于：
        close_gripper
        open_gripper
        done
    """

    def validate(
        self,
        robot_dict,
        teleop_dict,
    ):
        return super().validate(
            robot_dict,
            _validation_teleop_dict(
                robot_dict,
                teleop_dict,
            ),
        )

    def make(
        self,
        robot_dict,
        teleop_dict,
        cameras,
        device: str = "cpu",
    ):
        (
            env,
            env_processor,
            action_processor,
        ) = super().make(
            robot_dict,
            teleop_dict,
            cameras,
            device,
        )

        env.uses_autonomous_step = True

        return (
            env,
            env_processor,
            action_processor,
        )


class AutonomousOpenLoopTrajectoryPrimitiveConfig(
    OpenLoopTrajectoryPrimitiveConfig
):
    """
    用于：
        move_to_A
        move_to_B
        return_home
    """

    def validate(
        self,
        robot_dict,
        teleop_dict,
    ):
        return super().validate(
            robot_dict,
            _validation_teleop_dict(
                robot_dict,
                teleop_dict,
            ),
        )

    def make(
        self,
        robot_dict,
        teleop_dict,
        cameras,
        device: str = "cpu",
    ):
        (
            env,
            env_processor,
            action_processor,
        ) = super().make(
            robot_dict,
            teleop_dict,
            cameras,
            device,
        )

        env.uses_autonomous_step = True

        return (
            env,
            env_processor,
            action_processor,
        )


# ==============================================================
# TaskFrame helper
# ==============================================================

def make_fixed_task_frame(
    target: list[float],
) -> TaskFrame:
    """
    第一阶段：

        HRCrobot base frame
        ==
        SHaRe task frame

    所以：

        origin = [0, 0, 0, 0, 0, 0]
    """

    return TaskFrame(
        origin=[0.0] * 6,

        target=list(target),

        # 没有任何 RL / policy axis
        policy_mode=[None] * 6,

        # 六个轴全部 Cartesian position control
        control_mode=[
            ControlMode.POS
        ] * 6,
    )


# ==============================================================
# Processor helper
# ==============================================================

def make_processor(
    gripper_static_pos: float | None = None,
) -> ManipulationPrimitiveProcessorConfig:

    return ManipulationPrimitiveProcessorConfig(
        fps=OUTER_FPS,

        observation=ObservationConfig(
            # HRCrobot adapter 第一阶段不提供 joint state
            add_joint_position_to_observation=False,
            add_joint_velocity_to_observation=False,

            # 提供 TCP pose
            add_ee_pos_to_observation=True,

            add_ee_velocity_to_observation=False,
            add_ee_wrench_to_observation=False,
        ),

        gripper=GripperConfig(
            # gripper 不由 policy 控制
            enable=False,

            # close/open primitive 通过它注入固定命令
            static_pos=gripper_static_pos,
        ),
    )


# ==============================================================
# Primitive 1
#
# 当前实际位置 -> A
# ==============================================================

move_to_A = (
    AutonomousOpenLoopTrajectoryPrimitiveConfig(
        notes="Move from current TCP pose to A.",

        task_frame=make_fixed_task_frame(
            A_POSE
        ),

        trajectory=OpenLoopTrajectorySpec(
            target=A_POSE,
            frame="task",

            # 第一阶段慢一点
            duration_s=2.0,
        ),

        processor=make_processor(),
    )
)


# ==============================================================
# Primitive 2
#
# A 点保持位置 + 关闭夹爪
# ==============================================================

close_gripper = (
    AutonomousHoldPrimitiveConfig(
        notes="Close gripper at A.",

        task_frame=make_fixed_task_frame(
            A_POSE
        ),

        processor=make_processor(
            gripper_static_pos=GRIPPER_CLOSE
        ),
    )
)


# ==============================================================
# Primitive 3
#
# A -> B
# ==============================================================

move_to_B = (
    AutonomousOpenLoopTrajectoryPrimitiveConfig(
        notes="Move from A to B.",

        task_frame=make_fixed_task_frame(
            B_POSE
        ),

        trajectory=OpenLoopTrajectorySpec(
            target=B_POSE,
            frame="task",
            duration_s=2.0,
        ),

        processor=make_processor(),
    )
)


# ==============================================================
# Primitive 4
#
# B 点保持位置 + 打开夹爪
# ==============================================================

open_gripper = (
    AutonomousHoldPrimitiveConfig(
        notes="Open gripper at B.",

        task_frame=make_fixed_task_frame(
            B_POSE
        ),

        processor=make_processor(
            gripper_static_pos=GRIPPER_OPEN
        ),
    )
)


# ==============================================================
# Primitive 5
#
# B -> startup home
#
# 这里先给 placeholder。
#
# 真机连接后，
# 我们会读取程序启动瞬间 TCP pose，
# 再覆盖 trajectory.target。
# ==============================================================

return_home = (
    AutonomousOpenLoopTrajectoryPrimitiveConfig(
        notes="Return to startup TCP pose.",

        task_frame=make_fixed_task_frame(
            B_POSE
        ),

        trajectory=OpenLoopTrajectorySpec(
            target=[0.0] * 6,
            frame="task",
            duration_s=2.0,
        ),

        processor=make_processor(),
    )
)


# ==============================================================
# Primitive 6
#
# Terminal sentinel
# ==============================================================

done = AutonomousHoldPrimitiveConfig(
    notes="Task completed.",

    task_frame=make_fixed_task_frame(
        B_POSE
    ),

    processor=make_processor(),

    is_terminal=True,
)


# ==============================================================
# MP-Net configuration
# ==============================================================

net_cfg = ManipulationPrimitiveNetConfig(
    fps=OUTER_FPS,

    start_primitive="move_to_A",
    reset_primitive="move_to_A",

    robot={
        ROBOT_NAME: HRCrobotConfig(
            robot_ip=ROBOT_IP,

            frequency=SERVO_FREQUENCY,

            use_gripper=True,
        ),
    },

    # 不使用 teleoperation
    teleop={},

    # 第一阶段不使用相机
    cameras={},

    primitives={
        "move_to_A": move_to_A,
        "close_gripper": close_gripper,
        "move_to_B": move_to_B,
        "open_gripper": open_gripper,
        "return_home": return_home,
        "done": done,
    },

    transitions=[
        # ------------------------------------------------------
        # 当前位姿 -> A
        # ------------------------------------------------------

        OnSuccess(
            source="move_to_A",
            target="close_gripper",
            success_key="primitive_complete",
            additional_reward=0.0,
        ),

        # ------------------------------------------------------
        # close gripper
        #
        # 30 Hz × 0.5 s = 15 outer steps
        # ------------------------------------------------------

        OnTimeLimit(
            source="close_gripper",
            target="move_to_B",
            max_steps=15,
        ),

        # ------------------------------------------------------
        # A -> B
        # ------------------------------------------------------

        OnSuccess(
            source="move_to_B",
            target="open_gripper",
            success_key="primitive_complete",
            additional_reward=0.0,
        ),

        # ------------------------------------------------------
        # open gripper
        # ------------------------------------------------------

        OnTimeLimit(
            source="open_gripper",
            target="return_home",
            max_steps=15,
        ),

        # ------------------------------------------------------
        # B -> startup home
        # ------------------------------------------------------

        OnSuccess(
            source="return_home",
            target="done",
            success_key="primitive_complete",
            additional_reward=0.0,
        ),
    ],
)


# ==============================================================
# Helper
# ==============================================================

def get_pose_from_robot_observation(
    observation: dict,
) -> list[float]:

    return [
        float(
            observation[
                f"{axis}.ee_pos"
            ]
        )
        for axis in TASK_FRAME_AXIS_NAMES
    ]


# ==============================================================
# Main
# ==============================================================

def main() -> None:

    # ----------------------------------------------------------
    # 1. 创建 MP-Net
    #
    # ManipulationPrimitiveNet 初始化时会 connect HRCrobot。
    # ----------------------------------------------------------

    net = ManipulationPrimitiveNet(
        net_cfg
    )

    try:

        robot = net.robot_dict[
            ROBOT_NAME
        ]

        # ------------------------------------------------------
        # 2. 保存程序启动时真实 TCP pose
        # ------------------------------------------------------

        observation = (
            robot.get_observation()
        )

        start_pose = (
            get_pose_from_robot_observation(
                observation
            )
        )

        print(
            "[HRCrobot] startup pose:",
            start_pose,
        )

        # ------------------------------------------------------
        # 3. 修改 return_home target
        #
        # net 初始化期间 config 已经被 validate，
        # trajectory.target 已经转换成：
        #
        # {
        #     "arm": [...]
        # }
        #
        # ------------------------------------------------------

        return_home_cfg = (
            net.config.primitives[
                "return_home"
            ]
        )

        return_home_cfg.trajectory.target[
            ROBOT_NAME
        ] = list(
            start_pose
        )

        return_home_cfg.task_frame[
            ROBOT_NAME
        ].target = list(
            start_pose
        )

        # ------------------------------------------------------
        # 4. 开始 episode
        # ------------------------------------------------------

        net.reset()

        print(
            "[HRCrobot] start primitive:",
            net.active_primitive,
        )

        # ------------------------------------------------------
        # 5. MP-Net main loop
        # ------------------------------------------------------

        while not net.in_terminal:

            loop_t0 = (
                time.perf_counter()
            )

            # 所有 axis:
            #
            # policy_mode=None
            #
            # 所以 action_dim 应该为 0。
            action = torch.zeros(
                net.action_dim,
                dtype=torch.float32,
            )

            transition = net.step(
                action
            )

            info = transition.get(
                "info",
                {},
            )

            print(
                "[HRCrobot]",
                "primitive:",
                net.active_primitive,
                "| action_dim:",
                net.action_dim,
                "| reason:",
                info.get(
                    "transition_reason"
                ),
            )

            # --------------------------------------------------
            # MP-Net 外层维持 OUTER_FPS
            # --------------------------------------------------

            dt = (
                time.perf_counter()
                - loop_t0
            )

            precise_sleep(
                max(
                    0.0,
                    1.0 / OUTER_FPS - dt,
                )
            )

        print(
            "[HRCrobot] task finished."
        )

    finally:

        net.close()


if __name__ == "__main__":
    main()