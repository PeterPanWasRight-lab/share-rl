import json
from pathlib import Path
from dataclasses import dataclass
from PIL import Image

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from lerobot.cameras import Camera
from lerobot.robots import Robot
from lerobot.utils.rotation import Rotation

from share.cameras import RealSenseDepthCamera
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import TaskFrame
import logging

from share.pose_estimation.grasp_obj_spec import GraspObjectSpec
from share.pose_estimation.pose_estimator import PoseEstimator
from share.utils.constants import DEFAULT_ROBOT_NAME

logger = logging.getLogger(__name__)

EE_POSE_KEYS = ("x.ee_pos", "y.ee_pos", "z.ee_pos", "rx.ee_pos", "ry.ee_pos", "rz.ee_pos")


def _transform_from_translation_rotation(
    translation_m: list[float] | np.ndarray,
    rotation_matrix: list[list[float]] | np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_m, dtype=np.float64).reshape(3)
    return transform


def load_camera_to_gripper_transform(calibration_path: str | Path) -> np.ndarray:
    payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    camera_to_gripper = payload["camera_to_gripper"]
    return _transform_from_translation_rotation(
        translation_m=camera_to_gripper["translation_m"],
        rotation_matrix=camera_to_gripper["rotation_matrix"],
    )


def object_pose_camera_to_robot_base(
    object_pose_camera: np.ndarray,
    tcp_pose_base_to_gripper: list[float] | np.ndarray,
    camera_to_gripper_transform: np.ndarray,
) -> np.ndarray:
    base_to_gripper = np.eye(4, dtype=np.float64)
    base_to_gripper[:3, :3] = Rotation.from_rotvec(np.asarray(tcp_pose_base_to_gripper[3:], dtype=np.float64)).as_matrix()
    base_to_gripper[:3, 3] = np.asarray(tcp_pose_base_to_gripper[:3], dtype=np.float64)
    camera_to_object = np.asarray(object_pose_camera, dtype=np.float64).reshape(4, 4)
    return base_to_gripper @ camera_to_gripper_transform @ camera_to_object


def object_pose_camera_to_tcp_frame(
    object_pose_camera: np.ndarray,
    camera_to_gripper_transform: np.ndarray,
) -> np.ndarray:
    camera_to_object = np.asarray(object_pose_camera, dtype=np.float64).reshape(4, 4)
    return camera_to_gripper_transform @ camera_to_object


def transform_to_pose_xyzrpy(transform: np.ndarray) -> list[float]:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = SciRotation.from_matrix(transform[:3, :3])
    translation = transform[:3, 3].tolist()
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        *[float(value) for value in rotation.as_euler("xyz", degrees=False).tolist()],
    ]


class FoundationPosePrimitive(ManipulationPrimitive):
    def __init__(
            self,
            task_frame: dict[str, TaskFrame],
            robot_dict: dict[str, Robot],
            cameras: dict[str, Camera],
            grasp_object: GraspObjectSpec | str | Path,
            display_cameras: bool = False,
            pose_key: str = "pose",
    ):
        super().__init__(task_frame, robot_dict, cameras, display_cameras)
        self.pose_estimator = PoseEstimator()
        self._pose_estimator_initialized = False
        self.camera_to_gripper_transform = load_camera_to_gripper_transform(
            "/home/jzilke/ws/share-rl-pe/hand_eye_calibration_result_ur3e.json"
        )
        if isinstance(grasp_object, GraspObjectSpec):
            self.object_spec = grasp_object
        else:
            self.object_spec = GraspObjectSpec.from_json_file(grasp_object)

        self._pose_estimator_config = {
            "mesh_path": self.object_spec.mesh_path,
            "prompt": self.object_spec.segmentation_prompt,
            "confidence_threshold": self.object_spec.confidence_threshold,
            # "prompt": "black electrical box in the center", # TODO: REMOVE
            # "confidence_threshold": 0.1, # TODO: REMOVE
        }
        self.debug_output_dir = Path("tmp")
        self.pose_key = pose_key

    def step(self, action: dict[str, dict[str, float]]):
        obs, reward, terminated, truncated, info = super().step(action)
        cam = self.cameras.get(DEFAULT_ROBOT_NAME)
        if not isinstance(cam, RealSenseDepthCamera):
            raise TypeError(f"Expected RealSenseDepthCamera, got {type(cam).__name__}")

        if not self._pose_estimator_initialized:
            self.pose_estimator.configure(
                **self._pose_estimator_config,
                camera_intrinsics=cam.get_camera_intrinsics().tolist(),
            )
            self.pose_estimator.restart_tracking()
            self._pose_estimator_initialized = True
        logger.debug("configuration:", self._pose_estimator_config)
        image = obs.get('observation.images.main')
        depth = cam.read_depth(timeout_ms=False, in_meters=True)

        estimation = self.estimate_pose(image, depth)
        pose = estimation.get('pose')
        tcp_pose = [obs[f"{DEFAULT_ROBOT_NAME}.{key}"] for key in EE_POSE_KEYS]
        pose_base = object_pose_camera_to_robot_base(
            object_pose_camera=pose,
            tcp_pose_base_to_gripper=tcp_pose,
            camera_to_gripper_transform=self.camera_to_gripper_transform,
        )
        object_pose_world = transform_to_pose_xyzrpy(pose_base)

        debug_prints = True
        if debug_prints:
            pose_tcp = object_pose_camera_to_tcp_frame(
                object_pose_camera=pose,
                camera_to_gripper_transform=self.camera_to_gripper_transform,
            )

            tcp_world = np.eye(4, dtype=np.float64)
            tcp_world[:3, :3] = Rotation.from_rotvec(np.asarray(tcp_pose[3:], dtype=np.float64)).as_matrix()
            tcp_world[:3, 3] = np.asarray(tcp_pose[:3], dtype=np.float64)
            camera_world = tcp_world @ self.camera_to_gripper_transform

            logger.debug("object pose in world frame:\n%s", pose_base)
            logger.debug("object pose in tcp frame:\n%s", pose_tcp)
            logger.debug("object pose in camera frame:\n%s", pose)
            logger.debug("tcp pose in world frame:\n%s", tcp_world)
            logger.debug("camera pose in world frame:\n%s", camera_world)

            camera_distance_m = float(np.linalg.norm(np.asarray(pose, dtype=np.float64)[:3, 3]))
            tcp_distance_m = float(np.linalg.norm(pose_tcp[:3, 3]))
            base_distance_m = float(np.linalg.norm(pose_base[:3, 3]))
            logger.debug(f"object translation distance in camera frame [m]: {camera_distance_m:.4f}")
            logger.debug(f"object translation distance in tcp frame [m]: {tcp_distance_m:.4f}")
            logger.debug(f"object translation distance in base frame [m]: {base_distance_m:.4f}")

            logger.debug(f"OBJECT POSE IN TCP FRAME: {transform_to_pose_xyzrpy(pose_tcp)}")

        self.set_runtime_value(self.pose_key, object_pose_world)
        self._pose_estimator_initialized = False
        return obs, reward, terminated, truncated, info


    def estimate_pose(self, image, depth):
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        estimation = self.pose_estimator.estimate_pose(image=image, depth=depth)

        if 'mask' in estimation.keys() and estimation.get('mask') is not None:
            Image.fromarray(image).save(self.debug_output_dir / "obs_main.png")
            Image.fromarray(estimation.get('mask')).save(self.debug_output_dir / "mask.png")

        return estimation


@ManipulationPrimitiveConfig.register_subclass("runtime_frame_target")
@dataclass
class RuntimeFrameTargetPrimitiveConfig(ManipulationPrimitiveConfig):
    """Primitive that resolves its task-frame origin from a saved runtime pose."""

    frame_origin_runtime_key: str = "object_pose"

    def on_entry(self, env: ManipulationPrimitive, entry_context) -> None:
        runtime_origin = env.get_runtime_value(self.frame_origin_runtime_key)
        if runtime_origin is None:
            raise RuntimeError(
                f"Missing runtime frame origin '{self.frame_origin_runtime_key}'. "
                "Run the pose-estimation primitive before entering this primitive."
            )

        origin_by_robot = self._origin_by_robot(runtime_origin)
        for name, frame in self.task_frame.items():
            frame.origin = [float(value) for value in origin_by_robot[name]]

        super().on_entry(env, entry_context)

    def _origin_by_robot(self, runtime_origin) -> dict[str, list[float]]:
        if isinstance(runtime_origin, dict):
            if set(self.task_frame).issubset(runtime_origin):
                return {
                    name: [float(value) for value in runtime_origin[name]]
                    for name in self.task_frame
                }
            if {"x", "y", "z", "rx", "ry", "rz"}.issubset(runtime_origin):
                pose = [float(runtime_origin[axis]) for axis in ("x", "y", "z", "rx", "ry", "rz")]
                return {name: list(pose) for name in self.task_frame}

        if isinstance(runtime_origin, (list, tuple, np.ndarray)) and len(runtime_origin) == 6:
            pose = [float(value) for value in runtime_origin]
            return {name: list(pose) for name in self.task_frame}

        raise ValueError(
            f"Runtime frame origin '{self.frame_origin_runtime_key}' must be a 6D pose or per-robot pose dict, "
            f"got {type(runtime_origin).__name__}."
        )

@ManipulationPrimitiveConfig.register_subclass("relruntime_frame_target")
@dataclass
class RelativeRuntimeFrameTargetPrimitiveConfig(ManipulationPrimitiveConfig):
    """Primitive that resolves its task-frame origin from a saved runtime pose."""

    frame_origin_runtime_key: str = "object_pose"

    def on_entry(self, env: ManipulationPrimitive, entry_context) -> None:
        runtime_origin = env.get_runtime_value(self.frame_origin_runtime_key).copy()
        if runtime_origin is None:
            raise RuntimeError(
                f"Missing runtime frame origin '{self.frame_origin_runtime_key}'. "
                "Run the pose-estimation primitive before entering this primitive."
            )
        runtime_origin[2] += 0.02
        origin_by_robot = self._origin_by_robot(runtime_origin)
        for name, frame in self.task_frame.items():
            frame.origin = [float(value) for value in origin_by_robot[name]]

        super().on_entry(env, entry_context)

    def _origin_by_robot(self, runtime_origin) -> dict[str, list[float]]:
        if isinstance(runtime_origin, dict):
            if set(self.task_frame).issubset(runtime_origin):
                return {
                    name: [float(value) for value in runtime_origin[name]]
                    for name in self.task_frame
                }
            if {"x", "y", "z", "rx", "ry", "rz"}.issubset(runtime_origin):
                pose = [float(runtime_origin[axis]) for axis in ("x", "y", "z", "rx", "ry", "rz")]
                return {name: list(pose) for name in self.task_frame}

        if isinstance(runtime_origin, (list, tuple, np.ndarray)) and len(runtime_origin) == 6:
            pose = [float(value) for value in runtime_origin]
            return {name: list(pose) for name in self.task_frame}

        raise ValueError(
            f"Runtime frame origin '{self.frame_origin_runtime_key}' must be a 6D pose or per-robot pose dict, "
            f"got {type(runtime_origin).__name__}."
        )