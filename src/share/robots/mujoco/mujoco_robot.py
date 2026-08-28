from __future__ import annotations

import json
import logging
import os
import socket
from collections import deque
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TASK_FRAME_AXIS_NAMES,
    TaskFrame,
)
from share.robots.gripper_command_limiter import GripperCommandLimiter
from .configuration_mujoco import MujocoRobotConfig
from .model import ASSET_ROOT, build_pick_insert_model, build_ur5e_2f85_model
from .registry import register_robot, unregister_robot

logger = logging.getLogger(__name__)


class _MotorBusView:
    """Narrow compatibility surface used by primitive joint-space validation."""

    def __init__(self, joint_names: list[str]):
        self.motors = {name: None for name in joint_names}


class MujocoRobot(Robot):
    """MuJoCo UR5e backend implementing the LeRobot Robot contract.

    The simulation advances in ``send_action``. Task-space position commands are
    converted to joint targets with damped least-squares IK and sent to the stock
    MuJoCo Menagerie position servos.
    """

    config_class = MujocoRobotConfig
    name = "mujoco"
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(self, config: MujocoRobotConfig):
        super().__init__(config)
        self.config = config
        self.bus = _MotorBusView(self.joint_names)
        self.task_frame = TaskFrame()
        self._is_connected = False
        self._active_control_space: ControlSpace | None = None
        self._rng = np.random.default_rng(config.seed)
        self._model = None
        self._data = None
        self._mujoco = None
        self._viewer = None
        self._renderer_cache: dict[tuple[int, int], Any] = {}
        self._wrench_history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=config.viewer_wrench_history_samples
        )
        self._last_diagnostics_time = float("-inf")
        self._joint_ids: np.ndarray | None = None
        self._joint_qpos_adr: np.ndarray | None = None
        self._joint_dof_adr: np.ndarray | None = None
        self._arm_actuator_ids: np.ndarray | None = None
        self._tcp_site_id = -1
        self._force_torque_site_id = -1
        self._peg_tip_site_id = -1
        self._peg_body_id = -1
        self._fixture_body_id = -1
        self._fixture_nominal_pos: np.ndarray | None = None
        self._domain_randomization_nominal: dict[str, Any] = {}
        self._domain_randomization_state: dict[str, Any] = {}
        self._sensor_slices: dict[str, slice] = {}
        self._gripper_qpos_adr = -1
        self._gripper_actuator_id = -1
        self._peg_qpos_adr = -1
        self._peg_dof_adr = -1
        self._last_gripper = 0.5
        self._gripper_command_limiter = GripperCommandLimiter(
            min_interval_s=config.gripper_min_command_interval_s,
            clock=lambda: float(self._data.time) if self._data is not None else 0.0,
        )
        self._virtual_target_task: np.ndarray | None = None
        self._joint_position_target: np.ndarray | None = None
        self._ik_data = None
        self._god_view_socket_path = os.environ.get(
            "SHARE_MUJOCO_GOD_VIEW_SOCKET",
            "/tmp/share_mujoco_god_view.sock",
        )
        self._god_view_socket: socket.socket | None = None
        self._god_view_sequence = 0
        self._simulation_episode = 0

    @property
    def scene_path(self) -> Path:
        if self.config.scene_path:
            return Path(self.config.scene_path).expanduser().resolve()
        return ASSET_ROOT / "scene.xml"

    def _load_model(self):
        """Build the MuJoCo model from an explicit XML path or a scene builder."""
        if self.config.scene_path is not None:
            if not self.scene_path.exists():
                raise FileNotFoundError(f"MuJoCo scene does not exist: {self.scene_path}")
            return self._mujoco.MjModel.from_xml_path(str(self.scene_path))
        if self.config.scene_builder == "pick_insert":
            return build_pick_insert_model()
        return build_ur5e_2f85_model()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def _motors_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for joint_name in self.joint_names:
            features[f"{joint_name}.pos"] = float
            features[f"{joint_name}.vel"] = float
        for axis_name in TASK_FRAME_AXIS_NAMES:
            features[f"{axis_name}.ee_pos"] = float
            features[f"{axis_name}.ee_vel"] = float
            features[f"{axis_name}.ee_wrench"] = float
            features[f"{axis_name}.task_frame_origin"] = float
        if self.config.use_gripper:
            features[f"{GRIPPER_KEY}.pos"] = float
        features["insertion.depth"] = float
        features["insertion.lateral_error"] = float
        features["insertion.axis_alignment"] = float
        return features

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def action_features(self) -> dict[str, type]:
        features = self.task_frame.action_feature_keys()
        if self.config.use_gripper:
            features[f"{GRIPPER_KEY}.pos"] = float
        return features

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("Install the MuJoCo backend with `pip install -e '.[mujoco]'`.") from exc

        self._mujoco = mujoco
        self._model = self._load_model()
        self._model.opt.timestep = self.config.timestep
        self._data = mujoco.MjData(self._model)
        self._ik_data = mujoco.MjData(self._model)
        self._cache_model_ids()
        self._is_connected = True
        register_robot(self.config.id, self)
        self.reset_simulation(seed=self.config.seed)

        if self.config.viewer:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            if self.config.viewer_camera and self.config.viewer_camera != "free":
                camera_id = mujoco.mj_name2id(
                    self._model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    self.config.viewer_camera,
                )
                if camera_id < 0:
                    raise ValueError(
                        f"Unknown viewer camera {self.config.viewer_camera!r}; "
                        "expected one of 'front' or 'wrist'."
                    )
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._viewer.cam.fixedcamid = camera_id
            self._update_viewer_diagnostics(force=True)
        logger.info("Connected MuJoCo robot %s with scene %s", self.config.id, self.scene_path)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        for renderer in self._renderer_cache.values():
            renderer.close()
        self._renderer_cache.clear()
        if self._god_view_socket is not None:
            self._god_view_socket.close()
            self._god_view_socket = None
        unregister_robot(self.config.id, self)
        self._data = None
        self._ik_data = None
        self._model = None
        self._mujoco = None
        self._is_connected = False

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def reset_simulation(self, seed: int | None = None) -> None:
        """Reset physics and sample one reproducible domain per episode."""
        self._require_connected()
        self._simulation_episode += 1
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        key_id = self._mujoco.mj_name2id(
            self._model, self._mujoco.mjtObj.mjOBJ_KEY, self.config.home_keyframe
        )
        if key_id >= 0:
            self._mujoco.mj_resetDataKeyframe(self._model, self._data, key_id)
        else:
            self._mujoco.mj_resetData(self._model, self._data)
        self._apply_domain_randomization()
        self._mujoco.mj_forward(self._model, self._data)
        self._close_gripper_around_peg()
        self._wrench_history.clear()
        self._last_diagnostics_time = float("-inf")
        self._last_gripper = 1.0
        self._gripper_command_limiter.synchronize(self._last_gripper)
        self._joint_position_target = self._data.ctrl[self._arm_actuator_ids].copy()
        self._virtual_target_task = self._world_to_task_pose(self._tcp_world_pose())

    @property
    def domain_randomization_state(self) -> dict[str, Any]:
        """Return the sampled episode parameters for manifests and diagnostics."""
        return json.loads(json.dumps(self._domain_randomization_state))

    def set_task_frame(self, frame: TaskFrame) -> None:
        resolved_space = ControlSpace(int(frame.space))
        if self._active_control_space is not None and resolved_space != self._active_control_space:
            raise ValueError("MuJoCo robot does not support switching control space while connected")
        if (
            self.is_connected
            and resolved_space == ControlSpace.TASK
            and any(ControlMode(mode) != ControlMode.POS for mode in frame.control_mode)
        ):
            raise ValueError("MuJoCo position-servo backend only supports task-space POS control")
        self.task_frame = replace(frame)
        self._active_control_space = resolved_space
        if self.is_connected and resolved_space == ControlSpace.TASK:
            current_task = self._world_to_task_pose(self._tcp_world_pose())
            if self._virtual_target_task is None:
                self._virtual_target_task = current_task
            for axis, policy_mode in enumerate(frame.policy_mode):
                if policy_mode != PolicyMode.RELATIVE:
                    self._virtual_target_task[axis] = float(frame.target[axis])

    def get_observation(self) -> dict[str, Any]:
        self._require_connected()
        tcp_world = self._tcp_world_pose()
        tcp_task = self._world_to_task_pose(tcp_world)
        tcp_task_rotvec = Rotation.from_euler("xyz", tcp_task[3:]).as_rotvec()
        tcp_observation_pose = np.concatenate((tcp_task[:3], tcp_task_rotvec))
        velocity_world, _ = self._tcp_velocity_and_jacobian()
        velocity_task = self._rotate_spatial_to_task(velocity_world)
        wrench_task = self._rotate_spatial_to_task(self._sensor_vector())

        observation: dict[str, Any] = {}
        for index, joint_name in enumerate(self.joint_names):
            observation[f"{joint_name}.pos"] = float(self._data.qpos[self._joint_qpos_adr[index]])
            observation[f"{joint_name}.vel"] = float(self._data.qvel[self._joint_dof_adr[index]])
        origin = self.task_frame.origin or [0.0] * 6
        for index, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
            # Match the UR RTDE contract: rotational pose channels are a rotation
            # vector, while TaskFrame itself remains xyz + extrinsic XYZ Euler.
            observation[f"{axis_name}.ee_pos"] = float(tcp_observation_pose[index])
            observation[f"{axis_name}.ee_vel"] = float(velocity_task[index])
            observation[f"{axis_name}.ee_wrench"] = float(wrench_task[index])
            observation[f"{axis_name}.task_frame_origin"] = float(origin[index])
        if self.config.use_gripper:
            closure = float(self._data.qpos[self._gripper_qpos_adr] / 0.8)
            observation[f"{GRIPPER_KEY}.pos"] = np.clip(closure, 0.0, 1.0)
        insertion_depth, lateral_error, axis_alignment = self._insertion_metrics()
        observation["insertion.depth"] = insertion_depth
        observation["insertion.lateral_error"] = lateral_error
        observation["insertion.axis_alignment"] = axis_alignment
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self._require_connected()
        executed_action = dict(action)
        if f"{GRIPPER_KEY}.pos" in action:
            self._last_gripper, _ = self._gripper_command_limiter.filter(
                float(action[f"{GRIPPER_KEY}.pos"])
            )
            executed_action[f"{GRIPPER_KEY}.pos"] = self._last_gripper

        control_space = self._space_from_action(action) or self.task_frame.space
        if self._active_control_space is None:
            self._active_control_space = ControlSpace(int(control_space))
        elif ControlSpace(int(control_space)) != self._active_control_space:
            raise ValueError("MuJoCo actions cannot switch between task and joint control")

        substeps = max(1, round(self.config.control_dt / self._model.opt.timestep))
        if self._active_control_space == ControlSpace.TASK:
            self._update_virtual_target(action)
            self._joint_position_target = self._solve_task_position_target()
        else:
            self._update_joint_position_target(action)
        for _ in range(substeps):
            self._data.ctrl[self._arm_actuator_ids] = self._joint_position_target
            if self.config.use_gripper:
                self._data.ctrl[self._gripper_actuator_id] = self._last_gripper * 255.0
            self._mujoco.mj_step(self._model, self._data)

        self._publish_god_view_state()
        if self._viewer is not None:
            self._update_viewer_diagnostics()
            self._viewer.sync()
        return executed_action

    def _publish_god_view_state(self) -> None:
        """Publish read-only simulation truth while a local receiver exists."""
        socket_path = Path(self._god_view_socket_path)
        if not socket_path.exists():
            return
        if self._god_view_socket is None:
            self._god_view_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

        fixture_position = self._data.xpos[self._fixture_body_id].copy()
        fixture_rotation = self._data.xmat[self._fixture_body_id].reshape(3, 3).copy()
        peg_tip_position = self._data.site_xpos[self._peg_tip_site_id].copy()
        tip_fixture = fixture_rotation.T @ (peg_tip_position - fixture_position)
        depth, lateral_error, axis_alignment = self._insertion_metrics()
        self._god_view_sequence += 1
        payload = {
            "version": 1,
            "robot_id": self.config.id,
            "sequence": self._god_view_sequence,
            "episode": self._simulation_episode,
            "simulation_time_s": float(self._data.time),
            "peg_tip_world_m": peg_tip_position.tolist(),
            "peg_tip_fixture_m": tip_fixture.tolist(),
            "fixture_position_world_m": fixture_position.tolist(),
            "fixture_rotation_world": fixture_rotation.tolist(),
            "tcp_world_pose": self._tcp_world_pose().tolist(),
            "wrench_sensor": self._sensor_vector().tolist(),
            "insertion_depth_m": depth,
            "lateral_error_m": lateral_error,
            "axis_alignment": axis_alignment,
        }
        try:
            self._god_view_socket.sendto(
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                str(socket_path),
            )
        except OSError as exc:
            logger.debug("MuJoCo god-view receiver unavailable: %s", exc)

    def render_camera(self, camera_name: str, width: int, height: int) -> np.ndarray:
        self._require_connected()
        key = (width, height)
        renderer = self._renderer_cache.get(key)
        if renderer is None:
            renderer = self._mujoco.Renderer(self._model, height=height, width=width)
            self._renderer_cache[key] = renderer
        renderer.update_scene(self._data, camera=camera_name)
        return np.asarray(renderer.render()).copy()

    def _cache_model_ids(self) -> None:
        joint_ids = [
            self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        if any(joint_id < 0 for joint_id in joint_ids):
            raise ValueError("MuJoCo scene is missing one or more expected UR5e joints")
        self._joint_ids = np.asarray(joint_ids, dtype=int)
        self._joint_qpos_adr = self._model.jnt_qposadr[self._joint_ids]
        self._joint_dof_adr = self._model.jnt_dofadr[self._joint_ids]
        actuator_ids = [
            self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
        ]
        if any(actuator_id < 0 for actuator_id in actuator_ids):
            raise ValueError("MuJoCo scene is missing one or more expected UR5e position actuators")
        self._arm_actuator_ids = np.asarray(actuator_ids, dtype=int)
        if np.any(self._model.actuator_biasprm[self._arm_actuator_ids, 1] >= 0):
            raise ValueError("MuJoCo UR5e arm actuators must be position servos")
        self._configure_position_servos()
        if self.config.gravity_compensation:
            self._enable_robot_gravity_compensation()
        self._tcp_site_id = self._mujoco.mj_name2id(
            self._model, self._mujoco.mjtObj.mjOBJ_SITE, "tool_tcp"
        )
        if self._tcp_site_id < 0:
            raise ValueError("MuJoCo scene is missing the required 'tool_tcp' site")
        self._force_torque_site_id = self._mujoco.mj_name2id(
            self._model, self._mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        if self._force_torque_site_id < 0:
            raise ValueError("MuJoCo scene is missing the force/torque sensor site")
        self._fixture_body_id = self._mujoco.mj_name2id(
            self._model, self._mujoco.mjtObj.mjOBJ_BODY, "fixture"
        )
        self._peg_body_id = self._mujoco.mj_name2id(
            self._model, self._mujoco.mjtObj.mjOBJ_BODY, "object_peg"
        )
        self._peg_tip_site_id = self._mujoco.mj_name2id(
            self._model, self._mujoco.mjtObj.mjOBJ_SITE, "object_tip"
        )
        if min(self._fixture_body_id, self._peg_body_id, self._peg_tip_site_id) < 0:
            raise ValueError("MuJoCo scene is missing insertion fixture or peg references")
        self._fixture_nominal_pos = self._model.body_pos[self._fixture_body_id].copy()
        if self.config.use_gripper:
            gripper_joint_id = self._mujoco.mj_name2id(
                self._model,
                self._mujoco.mjtObj.mjOBJ_JOINT,
                "gripper_right_driver_joint",
            )
            self._gripper_actuator_id = self._mujoco.mj_name2id(
                self._model,
                self._mujoco.mjtObj.mjOBJ_ACTUATOR,
                "gripper_fingers_actuator",
            )
            if gripper_joint_id < 0 or self._gripper_actuator_id < 0:
                raise ValueError("MuJoCo scene is missing the expected Robotiq 2F-85 controls")
            self._gripper_qpos_adr = int(self._model.jnt_qposadr[gripper_joint_id])
        peg_joint_id = self._mujoco.mj_name2id(
            self._model,
            self._mujoco.mjtObj.mjOBJ_JOINT,
            "object_peg_joint",
        )
        if peg_joint_id < 0:
            raise ValueError("MuJoCo scene is missing the ACT peg free joint")
        self._peg_qpos_adr = int(self._model.jnt_qposadr[peg_joint_id])
        self._peg_dof_adr = int(self._model.jnt_dofadr[peg_joint_id])
        for name in ("tcp_force", "tcp_torque"):
            sensor_id = self._mujoco.mj_name2id(
                self._model, self._mujoco.mjtObj.mjOBJ_SENSOR, name
            )
            if sensor_id < 0:
                raise ValueError(f"MuJoCo scene is missing sensor {name!r}")
            address = int(self._model.sensor_adr[sensor_id])
            dimension = int(self._model.sensor_dim[sensor_id])
            self._sensor_slices[name] = slice(address, address + dimension)
        self._cache_domain_randomization_state()

    def _cache_domain_randomization_state(self) -> None:
        camera_ids = np.asarray(
            [
                self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_CAMERA, name)
                for name in ("front", "wrist")
            ],
            dtype=int,
        )
        geom_ids = np.asarray(
            [
                self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in ("socket-1", "socket-2", "socket-3", "socket-4", "object_red_peg")
            ],
            dtype=int,
        )
        if np.any(camera_ids < 0) or np.any(geom_ids < 0):
            raise ValueError("MuJoCo scene is missing domain-randomization camera or geom names")
        self._domain_randomization_nominal = {
            "camera_ids": camera_ids,
            "geom_ids": geom_ids,
            "fixture_pos": self._model.body_pos[self._fixture_body_id].copy(),
            "fixture_quat": self._model.body_quat[self._fixture_body_id].copy(),
            "camera_pos": self._model.cam_pos[camera_ids].copy(),
            "camera_quat": self._model.cam_quat[camera_ids].copy(),
            "camera_fovy": self._model.cam_fovy[camera_ids].copy(),
            "light_diffuse": self._model.light_diffuse.copy(),
            "light_specular": self._model.light_specular.copy(),
            "geom_rgba": self._model.geom_rgba[geom_ids].copy(),
            "geom_friction": self._model.geom_friction[geom_ids].copy(),
            "peg_mass": float(self._model.body_mass[self._peg_body_id]),
            "peg_inertia": self._model.body_inertia[self._peg_body_id].copy(),
        }

    @staticmethod
    def _rotate_wxyz(quaternion: np.ndarray, local_rotvec: np.ndarray) -> np.ndarray:
        nominal = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        rotated = nominal * Rotation.from_rotvec(local_rotvec)
        xyzw = rotated.as_quat()
        return xyzw[[3, 0, 1, 2]]

    def _apply_domain_randomization(self) -> None:
        nominal = self._domain_randomization_nominal
        if not nominal:
            return
        camera_ids = nominal["camera_ids"]
        geom_ids = nominal["geom_ids"]
        self._model.body_pos[self._fixture_body_id] = nominal["fixture_pos"]
        self._model.body_quat[self._fixture_body_id] = nominal["fixture_quat"]
        self._model.cam_pos[camera_ids] = nominal["camera_pos"]
        self._model.cam_quat[camera_ids] = nominal["camera_quat"]
        self._model.cam_fovy[camera_ids] = nominal["camera_fovy"]
        self._model.light_diffuse[:] = nominal["light_diffuse"]
        self._model.light_specular[:] = nominal["light_specular"]
        self._model.geom_rgba[geom_ids] = nominal["geom_rgba"]
        self._model.geom_friction[geom_ids] = nominal["geom_friction"]
        self._model.body_mass[self._peg_body_id] = nominal["peg_mass"]
        self._model.body_inertia[self._peg_body_id] = nominal["peg_inertia"]

        fixture_offset = np.asarray(
            [
                self._rng.uniform(-self.config.randomize_fixture_xy, self.config.randomize_fixture_xy),
                self._rng.uniform(-self.config.randomize_fixture_xy, self.config.randomize_fixture_xy),
                self._rng.uniform(-self.config.randomize_fixture_z, self.config.randomize_fixture_z),
            ]
        )
        fixture_yaw_deg = float(
            self._rng.uniform(-self.config.randomize_fixture_yaw_deg, self.config.randomize_fixture_yaw_deg)
        )
        self._model.body_pos[self._fixture_body_id] += fixture_offset
        self._model.body_quat[self._fixture_body_id] = self._rotate_wxyz(
            nominal["fixture_quat"], np.deg2rad([fixture_yaw_deg, 0.0, 0.0])
        )

        camera_position_offsets = self._rng.uniform(
            -self.config.randomize_camera_position_m,
            self.config.randomize_camera_position_m,
            size=(len(camera_ids), 3),
        )
        camera_rotation_offsets_deg = self._rng.uniform(
            -self.config.randomize_camera_rotation_deg,
            self.config.randomize_camera_rotation_deg,
            size=(len(camera_ids), 3),
        )
        camera_fovy_offsets_deg = self._rng.uniform(
            -self.config.randomize_camera_fovy_deg,
            self.config.randomize_camera_fovy_deg,
            size=len(camera_ids),
        )
        self._model.cam_pos[camera_ids] += camera_position_offsets
        for index, camera_id in enumerate(camera_ids):
            self._model.cam_quat[camera_id] = self._rotate_wxyz(
                nominal["camera_quat"][index], np.deg2rad(camera_rotation_offsets_deg[index])
            )
        self._model.cam_fovy[camera_ids] = np.clip(
            nominal["camera_fovy"] + camera_fovy_offsets_deg, 10.0, 120.0
        )

        light_scales = self._rng.uniform(
            1.0 - self.config.randomize_light_intensity_fraction,
            1.0 + self.config.randomize_light_intensity_fraction,
            size=(self._model.nlight, 1),
        )
        self._model.light_diffuse[:] = np.clip(nominal["light_diffuse"] * light_scales, 0.0, 1.0)
        self._model.light_specular[:] = np.clip(nominal["light_specular"] * light_scales, 0.0, 1.0)

        color_offsets = self._rng.uniform(
            -self.config.randomize_object_color_fraction,
            self.config.randomize_object_color_fraction,
            size=(len(geom_ids), 3),
        )
        self._model.geom_rgba[geom_ids, :3] = np.clip(
            nominal["geom_rgba"][:, :3] + color_offsets, 0.0, 1.0
        )
        friction_scales = self._rng.uniform(
            1.0 - self.config.randomize_contact_friction_fraction,
            1.0 + self.config.randomize_contact_friction_fraction,
            size=(len(geom_ids), 1),
        )
        self._model.geom_friction[geom_ids] = nominal["geom_friction"] * friction_scales
        peg_mass_scale = float(
            self._rng.uniform(
                1.0 - self.config.randomize_peg_mass_fraction,
                1.0 + self.config.randomize_peg_mass_fraction,
            )
        )
        self._model.body_mass[self._peg_body_id] = nominal["peg_mass"] * peg_mass_scale
        self._model.body_inertia[self._peg_body_id] = nominal["peg_inertia"] * peg_mass_scale
        self._domain_randomization_state = {
            "fixture_offset_m": fixture_offset.tolist(),
            "fixture_yaw_deg": fixture_yaw_deg,
            "camera_position_offsets_m": camera_position_offsets.tolist(),
            "camera_rotation_offsets_deg": camera_rotation_offsets_deg.tolist(),
            "camera_fovy_offsets_deg": camera_fovy_offsets_deg.tolist(),
            "light_scales": light_scales[:, 0].tolist(),
            "peg_mass_scale": peg_mass_scale,
            "friction_scales": friction_scales[:, 0].tolist(),
        }

    def _configure_position_servos(self) -> None:
        """Stiffen Menagerie servos while retaining their damping ratio."""
        stiffness_scale = self.config.position_servo_stiffness_scale
        damping_scale = np.sqrt(stiffness_scale)
        self._model.actuator_gainprm[self._arm_actuator_ids, 0] *= stiffness_scale
        self._model.actuator_biasprm[self._arm_actuator_ids, 1] *= stiffness_scale
        self._model.actuator_biasprm[self._arm_actuator_ids, 2] *= damping_scale

    def _enable_robot_gravity_compensation(self) -> None:
        """Approximate gravity feedforward in an industrial position controller."""
        base_id = self._mujoco.mj_name2id(
            self._model, self._mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        if base_id < 0:
            raise ValueError("MuJoCo scene is missing the UR5e base body")
        for body_id in range(base_id, self._model.nbody):
            ancestor = body_id
            while ancestor > 0 and ancestor != base_id:
                ancestor = int(self._model.body_parentid[ancestor])
            if ancestor == base_id:
                self._model.body_gravcomp[body_id] = 1.0

    def _close_gripper_around_peg(self) -> None:
        """Initialize a physical post-grasp state without welding the workpiece."""
        if not self.config.use_gripper:
            return
        arm_target = self._data.qpos[self._joint_qpos_adr].copy()
        peg_pose = self._data.qpos[self._peg_qpos_adr : self._peg_qpos_adr + 7].copy()
        close_steps = max(1, round(0.3 / self._model.opt.timestep))
        settle_steps = max(1, round(0.1 / self._model.opt.timestep))

        for _ in range(close_steps):
            # Reset represents loading the part between the fingers. Hold it only
            # while the real 2F-85 linkage closes; no equality remains afterwards.
            self._data.qpos[self._peg_qpos_adr : self._peg_qpos_adr + 7] = peg_pose
            self._data.qvel[self._peg_dof_adr : self._peg_dof_adr + 6] = 0.0
            self._mujoco.mj_forward(self._model, self._data)
            self._data.ctrl[self._arm_actuator_ids] = arm_target
            self._data.ctrl[self._gripper_actuator_id] = 255.0
            self._mujoco.mj_step(self._model, self._data)

        self._data.qpos[self._peg_qpos_adr : self._peg_qpos_adr + 7] = peg_pose
        self._data.qvel[self._peg_dof_adr : self._peg_dof_adr + 6] = 0.0
        for _ in range(settle_steps):
            self._mujoco.mj_forward(self._model, self._data)
            self._data.ctrl[self._arm_actuator_ids] = arm_target
            self._data.ctrl[self._gripper_actuator_id] = 255.0
            self._mujoco.mj_step(self._model, self._data)

    def _update_virtual_target(self, action: dict[str, Any]) -> None:
        current_task = self._world_to_task_pose(self._tcp_world_pose())
        if self._virtual_target_task is None:
            self._virtual_target_task = current_task.copy()
        for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
            if ControlMode(self.task_frame.control_mode[axis]) != ControlMode.POS:
                continue
            value = float(action.get(f"{axis_name}.ee_pos", self.task_frame.target[axis]))
            if self.task_frame.policy_mode[axis] == PolicyMode.RELATIVE:
                self._virtual_target_task[axis] += value * self.config.control_dt
            else:
                self._virtual_target_task[axis] = value
        self._virtual_target_task[:3] = np.clip(
            self._virtual_target_task[:3],
            np.asarray(self.task_frame.min_pose[:3]),
            np.asarray(self.task_frame.max_pose[:3]),
        )

    def _update_joint_position_target(self, action: dict[str, Any]) -> None:
        if self._joint_position_target is None:
            self._joint_position_target = self._data.qpos[self._joint_qpos_adr].copy()
        for index, joint_name in enumerate(self.joint_names):
            self._joint_position_target[index] = float(
                action.get(
                    f"{joint_name}.pos",
                    action.get(f"joint_{index + 1}.pos", self._joint_position_target[index]),
                )
            )
        self._joint_position_target = np.clip(
            self._joint_position_target,
            self._model.actuator_ctrlrange[self._arm_actuator_ids, 0],
            self._model.actuator_ctrlrange[self._arm_actuator_ids, 1],
        )

    def _solve_task_position_target(self) -> np.ndarray:
        """Resolve the virtual Cartesian target to a position-servo joint target."""
        desired_world = self._task_to_world_pose(self._virtual_target_task)
        self._ik_data.qpos[:] = self._data.qpos
        self._ik_data.qvel[:] = 0.0
        # The IK error is measured from the current simulated state, so its
        # joint correction must use that same state as its base. Applying the
        # correction to the previous servo target would repeatedly integrate
        # tracking error and make the arm overshoot or appear springy.
        base_target = self._data.qpos[self._joint_qpos_adr].copy()
        accumulated_delta = np.zeros(6, dtype=np.float64)

        for _ in range(self.config.ik_iterations):
            self._mujoco.mj_forward(self._model, self._ik_data)
            current_world = self._tcp_world_pose(data=self._ik_data)
            error = desired_world - current_world
            error[3:] = self._rotation_error(desired_world[3:], current_world[3:])
            if np.linalg.norm(error[:3]) < 1e-5 and np.linalg.norm(error[3:]) < 1e-4:
                break

            _, jacobian = self._tcp_velocity_and_jacobian(data=self._ik_data)
            damping_sq = self.config.ik_damping**2
            delta_q = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping_sq * np.eye(6),
                error,
            )
            delta_q = np.clip(
                delta_q,
                -self.config.ik_max_joint_step,
                self.config.ik_max_joint_step,
            )
            accumulated_delta += delta_q
            self._ik_data.qpos[self._joint_qpos_adr] += delta_q
            self._ik_data.qpos[self._joint_qpos_adr] = np.clip(
                self._ik_data.qpos[self._joint_qpos_adr],
                self._model.actuator_ctrlrange[self._arm_actuator_ids, 0],
                self._model.actuator_ctrlrange[self._arm_actuator_ids, 1],
            )

        return np.clip(
            base_target + accumulated_delta,
            self._model.actuator_ctrlrange[self._arm_actuator_ids, 0],
            self._model.actuator_ctrlrange[self._arm_actuator_ids, 1],
        )

    def _tcp_world_pose(self, data=None) -> np.ndarray:
        data = self._data if data is None else data
        position = data.site_xpos[self._tcp_site_id].copy()
        matrix = data.site_xmat[self._tcp_site_id].reshape(3, 3)
        rpy = Rotation.from_matrix(matrix).as_euler("xyz")
        return np.concatenate((position, rpy))

    def _tcp_velocity_and_jacobian(self, data=None) -> tuple[np.ndarray, np.ndarray]:
        data = self._data if data is None else data
        jacobian_pos = np.zeros((3, self._model.nv), dtype=np.float64)
        jacobian_rot = np.zeros((3, self._model.nv), dtype=np.float64)
        self._mujoco.mj_jacSite(self._model, data, jacobian_pos, jacobian_rot, self._tcp_site_id)
        full = np.vstack((jacobian_pos, jacobian_rot))
        velocity = full @ data.qvel
        return velocity, full[:, self._joint_dof_adr]

    def _sensor_vector(self) -> np.ndarray:
        """Return the wrist wrench in world coordinates."""
        force = self._data.sensordata[self._sensor_slices["tcp_force"]]
        torque = self._data.sensordata[self._sensor_slices["tcp_torque"]]
        # MuJoCo force/torque sensors report vectors in their sensor-site frame.
        # Observation and viewer consumers expect a spatial vector in world
        # coordinates before applying their task-frame rotation.
        sensor_to_world = self._data.site_xmat[self._force_torque_site_id].reshape(3, 3)
        return np.concatenate(
            (sensor_to_world @ force, sensor_to_world @ torque)
        ).astype(np.float64, copy=False)

    def _update_viewer_wrench_overlay(self) -> None:
        """Display the current task-frame wrist wrench in the passive viewer."""
        if (
            self._viewer is None
            or not self.config.viewer_wrench_overlay
            or not hasattr(self._viewer, "set_texts")
        ):
            return
        wrench = self._rotate_spatial_to_task(self._sensor_vector())
        labels = "Wrist F/T (task frame)\nFx\nFy\nFz\nTx\nTy\nTz"
        values = (
            "\n"
            f"{wrench[0]:+8.3f} N\n"
            f"{wrench[1]:+8.3f} N\n"
            f"{wrench[2]:+8.3f} N\n"
            f"{wrench[3]:+8.4f} Nm\n"
            f"{wrench[4]:+8.4f} Nm\n"
            f"{wrench[5]:+8.4f} Nm"
        )
        self._viewer.set_texts((None, None, labels, values))

    def _update_viewer_diagnostics(self, *, force: bool = False) -> None:
        """Refresh global/wrist cameras and rolling wrench plots in the viewer."""
        if self._viewer is None:
            return
        wrench = self._rotate_spatial_to_task(self._sensor_vector())
        self._wrench_history.append((float(self._data.time), wrench))
        self._update_viewer_wrench_overlay()

        update_period = 1.0 / self.config.viewer_diagnostics_fps
        if not force and self._data.time - self._last_diagnostics_time < update_period:
            return
        self._last_diagnostics_time = float(self._data.time)
        panel_width = min(
            self.config.viewer_diagnostics_width,
            max(160, int(self._viewer.viewport.width) // 3),
        )
        camera_height = round(panel_width * 3 / 4)
        plot_height = max(100, camera_height // 2)
        margin = 8
        panel_left = max(0, int(self._viewer.viewport.width) - panel_width - margin)
        camera_bottom = max(0, int(self._viewer.viewport.height) - camera_height - margin)

        if hasattr(self._viewer, "set_images"):
            camera_images = []
            if self.config.viewer_front_camera_overlay:
                front_left = max(0, panel_left - panel_width - margin)
                front_image = self.render_camera("front", panel_width, camera_height)
                camera_images.append(
                    (
                        self._mujoco.MjrRect(
                            front_left, camera_bottom, panel_width, camera_height
                        ),
                        front_image,
                    )
                )
            if self.config.viewer_wrist_camera_overlay:
                wrist_image = self.render_camera("wrist", panel_width, camera_height)
                camera_images.append(
                    (
                        self._mujoco.MjrRect(
                            panel_left, camera_bottom, panel_width, camera_height
                        ),
                        wrist_image,
                    )
                )
            if camera_images:
                self._viewer.set_images(camera_images)

        if self.config.viewer_wrench_plot and hasattr(self._viewer, "set_figures"):
            force_bottom = max(margin + plot_height, camera_bottom - plot_height - margin)
            torque_bottom = max(margin, force_bottom - plot_height - margin)
            force_figure = self._make_wrench_figure("Wrist force (N)", ("Fx", "Fy", "Fz"), 0)
            torque_figure = self._make_wrench_figure(
                "Wrist torque (Nm)", ("Tx", "Ty", "Tz"), 3
            )
            self._viewer.set_figures(
                [
                    (
                        self._mujoco.MjrRect(
                            panel_left, force_bottom, panel_width, plot_height
                        ),
                        force_figure,
                    ),
                    (
                        self._mujoco.MjrRect(
                            panel_left, torque_bottom, panel_width, plot_height
                        ),
                        torque_figure,
                    ),
                ]
            )

    def _make_wrench_figure(
        self, title: str, channel_names: tuple[str, str, str], channel_offset: int
    ):
        """Build one three-channel oscilloscope-style MuJoCo figure."""
        figure = self._mujoco.MjvFigure()
        figure.title = title
        figure.xlabel = "simulation time (s)"
        figure.flg_extend = 0
        samples = list(self._wrench_history)
        times = np.asarray([sample[0] for sample in samples], dtype=np.float32)
        values = np.asarray([sample[1] for sample in samples], dtype=np.float32)
        time_span = max(self.config.control_dt, float(times[-1] - times[0]))
        figure.range[0] = [float(times[-1] - time_span), float(times[-1])]
        selected_values = values[:, channel_offset : channel_offset + 3]
        minimum_span = 1.0 if channel_offset == 0 else 0.1
        amplitude = max(minimum_span, float(np.max(np.abs(selected_values))) * 1.1)
        figure.range[1] = [-amplitude, amplitude]
        for line_index, channel_name in enumerate(channel_names):
            figure.linename[line_index] = channel_name.encode("ascii")
            figure.linepnt[line_index] = len(times)
            figure.linedata[line_index, : 2 * len(times)] = np.column_stack(
                (times, selected_values[:, line_index])
            ).ravel()
        return figure

    def _insertion_metrics(self) -> tuple[float, float, float]:
        """Return peg-tip depth, lateral error, and peg/socket axis alignment."""
        fixture_rotation = self._data.xmat[self._fixture_body_id].reshape(3, 3)
        tip_relative = fixture_rotation.T @ (
            self._data.site_xpos[self._peg_tip_site_id]
            - self._data.xpos[self._fixture_body_id]
        )
        socket_entrance = 0.06
        depth = socket_entrance - float(tip_relative[0])
        lateral_error = float(np.linalg.norm(tip_relative[1:]))

        peg_rotation = self._data.xmat[self._peg_body_id].reshape(3, 3)
        axis_alignment = float(
            abs(np.dot(peg_rotation[:, 0], fixture_rotation[:, 0]))
        )
        return depth, lateral_error, axis_alignment

    def _world_to_task_pose(self, world_pose: np.ndarray) -> np.ndarray:
        origin = np.asarray(self.task_frame.origin or [0.0] * 6, dtype=np.float64)
        transform = np.linalg.inv(self._pose_to_transform(origin)) @ self._pose_to_transform(world_pose)
        return self._transform_to_pose(transform)

    def _task_to_world_pose(self, task_pose: np.ndarray) -> np.ndarray:
        origin = np.asarray(self.task_frame.origin or [0.0] * 6, dtype=np.float64)
        transform = self._pose_to_transform(origin) @ self._pose_to_transform(task_pose)
        return self._transform_to_pose(transform)

    def _task_rotation(self) -> np.ndarray:
        origin = np.asarray(self.task_frame.origin or [0.0] * 6, dtype=np.float64)
        return Rotation.from_euler("xyz", origin[3:]).as_matrix()

    def _rotate_spatial_to_task(self, vector: np.ndarray) -> np.ndarray:
        rotation = self._task_rotation().T
        return np.concatenate((rotation @ vector[:3], rotation @ vector[3:]))

    def _rotate_spatial_to_world(self, vector: np.ndarray) -> np.ndarray:
        rotation = self._task_rotation()
        return np.concatenate((rotation @ vector[:3], rotation @ vector[3:]))

    @staticmethod
    def _rotation_error(desired_rpy: np.ndarray, current_rpy: np.ndarray) -> np.ndarray:
        desired = Rotation.from_euler("xyz", desired_rpy)
        current = Rotation.from_euler("xyz", current_rpy)
        return (desired * current.inv()).as_rotvec()

    @staticmethod
    def _pose_to_transform(pose: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
        transform[:3, 3] = pose[:3]
        return transform

    @staticmethod
    def _transform_to_pose(transform: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_euler("xyz"))
        )

    @staticmethod
    def _space_from_action(action: dict[str, Any]) -> ControlSpace | None:
        task = any(".ee_" in key for key in action)
        joint = any(
            key.endswith(".pos") and ".ee_" not in key and key != f"{GRIPPER_KEY}.pos"
            for key in action
        )
        if task and joint:
            raise ValueError("MuJoCo actions cannot mix task-space and joint-space keys")
        if task:
            return ControlSpace.TASK
        if joint:
            return ControlSpace.JOINT
        return None

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
