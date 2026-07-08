"""Focused tests for non-hardware UR controller helpers."""

from __future__ import annotations

import time
import uuid
from multiprocessing.managers import SharedMemoryManager
from types import SimpleNamespace

import numpy as np
import pytest

from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode
from share.robots.ur.lerobot_robot_ur.config_ur import URConfig
from share.robots.ur.lerobot_robot_ur.controller import Command, RTDETaskFrameController, TaskFrameCommand


def test_task_frame_command_delta_mode_treats_only_relative_axes_as_deltas():
    """Delta mode should be derived from policy_mode only."""
    command = TaskFrameCommand(
        policy_mode=[
            PolicyMode.RELATIVE,
            PolicyMode.ABSOLUTE,
            None,
            PolicyMode.RELATIVE,
            None,
            PolicyMode.ABSOLUTE,
        ]
    )

    assert command.delta_mode == [
        PolicyMode.RELATIVE,
        PolicyMode.ABSOLUTE,
        PolicyMode.ABSOLUTE,
        PolicyMode.RELATIVE,
        PolicyMode.ABSOLUTE,
        PolicyMode.ABSOLUTE,
    ]


def test_task_frame_command_rejects_unknown_controller_override_keys():
    """Unknown override keys should fail before queueing."""
    with pytest.raises(ValueError, match="Unsupported UR task-frame controller overrides"):
        TaskFrameCommand(
            control_mode=[ControlMode.POS] * 6,
            controller_overrides={"mystery_limit": [1.0] * 6},
        ).to_queue_dict()


def test_controller_zero_ft_reuses_last_command_layout_with_new_opcode():
    """`zero_ft()` should reuse the last command payload with a new opcode."""
    queued_items: list[dict[str, np.ndarray]] = []
    controller = object.__new__(RTDETaskFrameController)
    controller._last_cmd = TaskFrameCommand(
        target=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.ABSOLUTE] * 6,
    )
    controller.robot_cmd_queue = SimpleNamespace(put=queued_items.append)

    controller.zero_ft()

    assert len(queued_items) == 1
    queued = queued_items[0]
    assert queued["cmd"] == int(Command.ZERO_FT)
    np.testing.assert_allclose(queued["target"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_apply_pending_commands_reanchors_axes_that_switch_to_relative_pos():
    """Switching from absolute to relative POS should seed the virtual target from the live pose."""
    controller = object.__new__(RTDETaskFrameController)
    controller.origin = np.zeros(6, dtype=np.float64)
    controller.target = np.zeros(6, dtype=np.float64)
    controller.min_pose = np.full(6, -np.inf, dtype=np.float64)
    controller.max_pose = np.full(6, np.inf, dtype=np.float64)
    controller.control_mode = np.array([int(ControlMode.POS)] * 6, dtype=np.int64)
    controller.delta_mode = np.array([int(PolicyMode.ABSOLUTE)] * 6, dtype=np.int64)
    controller.force_on = True
    controller._resolve_compliance_settings = lambda **_kwargs: None
    controller._enter_task_force_mode = lambda _rtde_c: None
    controller._transform_task_pose_between_frames = lambda pose, source_origin, target_origin: pose
    controller._ensure_control_space = lambda space: ControlSpace(int(space))
    controller.read_current_state = lambda _rtde_r: {"ActualTCPPose": np.array([0.4, -0.3, 0.2, 1.1, -0.9, 0.7])}

    message = TaskFrameCommand(
        target=[0.0] * 6,
        origin=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[
            PolicyMode.RELATIVE,
            PolicyMode.ABSOLUTE,
            PolicyMode.ABSOLUTE,
            PolicyMode.RELATIVE,
            PolicyMode.ABSOLUTE,
            PolicyMode.ABSOLUTE,
        ],
    ).to_queue_dict()
    msgs = {key: np.expand_dims(value, axis=0) for key, value in message.items()}
    msgs["space"] = np.asarray([int(ControlSpace.TASK)], dtype=np.int8)

    keep_running, active_space, x_cmd, q_cmd = controller._apply_pending_commands(
        msgs=msgs,
        n_cmd=1,
        rtde_c=SimpleNamespace(),
        rtde_r=SimpleNamespace(getActualQ=lambda: np.zeros(6, dtype=np.float64)),
        active_space=None,
        x_cmd=np.array([9.0, 8.0, 7.0, -6.0, -5.0, -4.0], dtype=np.float64),
        q_cmd=np.zeros(6, dtype=np.float64),
    )

    assert keep_running is True
    assert active_space == ControlSpace.TASK
    np.testing.assert_allclose(x_cmd[[0, 3]], [0.4, 1.1])
    np.testing.assert_allclose(x_cmd[[1, 2, 4, 5]], [8.0, 7.0, -5.0, -4.0])
    np.testing.assert_allclose(q_cmd, np.zeros(6, dtype=np.float64))


def test_controller_can_initialize_with_mock_config():
    """Mock mode should build shared-memory buffers from in-process RTDE state."""
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    controller = None
    try:
        config = URConfig(
            robot_ip=f"mock-init-{uuid.uuid4()}",
            mock=True,
            frequency=50.0,
            shm_manager=shm_manager,
        )
        controller = RTDETaskFrameController(config)
        state = controller.get_robot_state()
        np.testing.assert_allclose(state["ActualTCPPose"], np.zeros(6))
        np.testing.assert_allclose(state["ActualQ"], np.zeros(6))
    finally:
        if controller is not None:
            controller.close()
        shm_manager.shutdown()


def test_mock_rtde_interfaces_support_force_and_joint_commands():
    """The in-process RTDE mock should exercise task and joint controller outputs."""
    controller = object.__new__(RTDETaskFrameController)
    controller.config = SimpleNamespace(
        mock=True,
        robot_ip=f"mock-rtde-{uuid.uuid4()}",
        frequency=100.0,
    )
    rtde_c, rtde_r = controller._connect_rtde_interfaces()

    rtde_c.forceMode([0.0] * 6, [1] * 6, [6.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2, [5.0] * 6)
    time.sleep(0.05)
    assert rtde_r.getActualTCPPose()[0] > 0.0
    assert rtde_r.getActualTCPForce()[0] > 0.0

    rtde_c.directTorque([4.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)
    time.sleep(0.05)
    assert rtde_r.getActualQ()[0] > 0.0
    assert rtde_r.getActualQd()[0] > 0.0
