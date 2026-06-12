import logging
import math
import threading
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
import sys

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.processor import TransitionKey
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

from share.configs.design_constraints import DesignConstraintsConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.teleoperators import TeleopEvents
from share.utils.transformation_utils import (
    get_robot_pose_from_observation,
    RotationIntervalMode,
    task_pose_to_world_pose, unwrap_angle_near_reference, wrap_to_pi,
)
from share.workspace.mpnet import save_mpnet_config

init_logging()

HOTKEY_HELP = (
    "Hotkeys: "
    "[o] set origin from current pose and reset bounds, "
    "[r] reset bounds in current frame, "
    "[p] print current origin/bounds, "
    "[s] save config, "
    "[q] save and quit"
)

AXIS_NAMES = ("x", "y", "z", "rx", "ry", "rz")


class HotkeyController:
    """Simple one-shot hotkey listener for the calibration loop."""

    def __init__(self):
        from pynput import keyboard

        self._counts = {
            name: 0
            for name in ("set_origin", "reset_bounds", "print_status", "save", "quit")
        }
        self._lock = threading.Lock()
        self._mapping = {
            "o": "set_origin",
            "r": "reset_bounds",
            "p": "print_status",
            "s": "save",
            "q": "quit",
        }

        def on_press(key):
            try:
                if key.char is None:
                    return
                event_name = self._mapping.get(key.char.lower())
                if event_name is None:
                    return
                with self._lock:
                    self._counts[event_name] += 1
            except Exception:
                return

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

    def consume(self, event_name: str) -> bool:
        """Return True once for each queued hotkey press."""
        with self._lock:
            count = self._counts.get(event_name, 0)
            if count <= 0:
                return False
            self._counts[event_name] = count - 1
            return True

    def close(self) -> None:
        self._listener.stop()


class WorkspaceConstraintDesigner:
    """Interactive helper to calibrate task-frame origins and workspace bounds.

    Important frame semantics:
    - `get_robot_pose_from_observation(...)` is assumed to return the current EE
      pose already expressed in the active primitive's task frame.
    - Therefore bound tracking should use that observed pose directly.
    - Setting a new origin requires composing the current task-frame pose with
      the old origin to recover the current EE world pose.

    Crucially, recorded calibration bounds are kept separate from the live
    environment task frame to avoid perturbing the running controller.
    """

    def __init__(self, mp_net: ManipulationPrimitiveNet, output_path: Path):
        self.mp_net = mp_net
        self.output_path = Path(output_path)
        self._tracked_primitives: set[str] = set(mp_net.config.primitives.keys())
        self._recorded_bounds: dict[str, dict[str, dict[str, list[float] | None]]] = {}
        self._initialize_recorded_bounds()

        self._last_pose_in_frame_by_robot: dict[tuple[str, str], list[float]] = {}
        self._rot_unwrapped_min_by_robot: dict[tuple[str, str], np.ndarray] = {}
        self._rot_unwrapped_max_by_robot: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def _rotation_interval_modes_for_frame(frame: "TaskFrame") -> list[str]:
        raw = (frame.controller_overrides or {}).get("rotation_interval_modes", ["linear"] * 6)
        if len(raw) != 6:
            raise ValueError("rotation_interval_modes must have length 6")
        return [str(v) for v in raw]

    def _continuous_pose_in_frame(
        self,
        primitive_name: str,
        robot_name: str,
        pose_in_frame: list[float],
        frame: "TaskFrame",
    ) -> list[float]:
        """Return pose in current task frame, unwrapping wrapped rotational axes for calibration.

        Translation is unchanged.
        Rotation is only unwrapped for axes configured as `ccw_arc`.
        """
        pose = [float(v) for v in pose_in_frame]
        key = (primitive_name, robot_name)
        modes = self._rotation_interval_modes_for_frame(frame)
        prev = self._last_pose_in_frame_by_robot.get(key)

        if prev is not None:
            for j, axis in enumerate(range(3, 6)):
                if modes[axis] == "ccw_arc":
                    pose[axis] = unwrap_angle_near_reference(float(pose[axis]), float(prev[axis]))

        self._last_pose_in_frame_by_robot[key] = list(pose)
        return pose

    def _get_primitive(self, primitive_name: str | None = None):
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive
        return primitive_name, self.mp_net.config.primitives[primitive_name]

    def _initialize_recorded_bounds(self) -> None:
        """Initialize the persistent calibration store from the config."""
        for primitive_name, primitive in self.mp_net.config.primitives.items():
            self._recorded_bounds[primitive_name] = {}
            for robot_name, frame in primitive.task_frame.items():
                self._recorded_bounds[primitive_name][robot_name] = {
                    "origin": None if frame.origin is None else [float(v) for v in frame.origin],
                    "min_pose": None if frame.min_pose is None else [float(v) for v in frame.min_pose],
                    "max_pose": None if frame.max_pose is None else [float(v) for v in frame.max_pose],
                    "rotation_interval_modes": list(self._rotation_interval_modes_for_frame(frame)),
                }

    def _record_for_robot(self, primitive_name: str, robot_name: str) -> dict[str, list[float] | None]:
        return self._recorded_bounds[primitive_name][robot_name]

    @staticmethod
    def _ensure_rotation_interval_modes(rec: dict[str, list[float] | None]) -> list[str]:
        modes = rec.get("rotation_interval_modes")
        if modes is None:
            modes = [RotationIntervalMode.LINEAR.to_name()] * 6
            rec["rotation_interval_modes"] = modes
        if len(modes) != 6:
            raise ValueError("rotation_interval_modes must have length 6")
        return [str(v) for v in modes]

    @staticmethod
    def _set_frame_rotation_interval_modes(frame: "TaskFrame", modes: list[str]) -> None:
        controller_overrides = dict(frame.controller_overrides or {})
        controller_overrides["rotation_interval_modes"] = list(modes)
        frame.controller_overrides = controller_overrides

    def _infer_rotation_interval_modes_for_robot(
        self,
        primitive_name: str,
        robot_name: str,
        frame: "TaskFrame",
    ) -> None:
        """Infer per-axis rotation interval modes from recorded wrapped bounds.

        If the wrapped numeric interval spans more than pi, we interpret it as a
        small allowed arc that crosses the +/-pi branch cut and encode it as
        ``ccw_arc``. Otherwise the axis remains a normal linear interval.
        """
        rec = self._record_for_robot(primitive_name, robot_name)
        min_pose = rec.get("min_pose")
        max_pose = rec.get("max_pose")
        if min_pose is None or max_pose is None:
            return

        modes = self._ensure_rotation_interval_modes(rec)
        key = (primitive_name, robot_name)

        for j, axis in enumerate(range(3, 6)):
            lo = float(min_pose[axis])
            hi = float(max_pose[axis])
            if not (math.isfinite(lo) and math.isfinite(hi)):
                continue

            if abs(hi - lo) > math.pi:
                modes[axis] = RotationIntervalMode.CCW_ARC.to_name()
                arc_start = hi
                arc_end = unwrap_angle_near_reference(lo, hi)
                self._rot_unwrapped_min_by_robot[key] = self._rot_unwrapped_min_by_robot.get(
                    key, np.zeros(3, dtype=np.float64)
                )
                self._rot_unwrapped_max_by_robot[key] = self._rot_unwrapped_max_by_robot.get(
                    key, np.zeros(3, dtype=np.float64)
                )
                self._rot_unwrapped_min_by_robot[key][j] = float(min(arc_start, arc_end))
                self._rot_unwrapped_max_by_robot[key][j] = float(max(arc_start, arc_end))
                min_pose[axis] = float(wrap_to_pi(self._rot_unwrapped_min_by_robot[key][j]))
                max_pose[axis] = float(wrap_to_pi(self._rot_unwrapped_max_by_robot[key][j]))
            else:
                modes[axis] = RotationIntervalMode.LINEAR.to_name()
                min_pose[axis] = min(lo, hi)
                max_pose[axis] = max(lo, hi)

        rec["rotation_interval_modes"] = list(modes)
        self._set_frame_rotation_interval_modes(frame, modes)

    def current_pose_in_frame_by_robot(
            self,
            primitive_name: str | None = None,
    ) -> dict[str, list[float]]:
        """Read current EE pose for each robot in the primitive's task frame."""
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive

        env = self.mp_net._envs[primitive_name]
        primitive = self.mp_net.config.primitives[primitive_name]
        observation = env._get_observation()

        poses: dict[str, list[float]] = {}
        for robot_name, frame in primitive.task_frame.items():
            raw_pose = get_robot_pose_from_observation(observation, robot_name)
            poses[robot_name] = self._continuous_pose_in_frame(
                primitive_name=primitive_name,
                robot_name=robot_name,
                pose_in_frame=raw_pose,
                frame=frame,
            )
        return poses

    def current_world_pose_by_robot(
        self,
        primitive_name: str | None = None,
    ) -> dict[str, list[float]]:
        """Recover the current EE world pose for each robot."""
        primitive_name, primitive = self._get_primitive(primitive_name)
        current_in_frame = self.current_pose_in_frame_by_robot(primitive_name)

        return {
            robot_name: task_pose_to_world_pose(current_in_frame[robot_name], frame.origin)
            for robot_name, frame in primitive.task_frame.items()
        }

    def _reset_runtime_state_for_primitive(self, primitive_name: str) -> None:
        """Reset cached runtime state after intentionally changing live frames."""
        env = self.mp_net._envs[primitive_name]
        env_processor = getattr(self.mp_net, "_env_processors", {}).get(primitive_name)
        action_processor = getattr(self.mp_net, "_action_processors", {}).get(primitive_name)

        if env_processor is not None:
            env_processor.reset()
        if action_processor is not None:
            action_processor.reset()

        reset_runtime_state = getattr(env, "reset_runtime_state", None)
        if callable(reset_runtime_state):
            reset_runtime_state()

        apply_task_frames = getattr(env, "apply_task_frames", None)
        if callable(apply_task_frames):
            apply_task_frames()

    def set_origin_from_current_pose(self, primitive_name: str | None = None) -> None:
        """Set each robot's task-frame origin to the current EE world pose.

        Observations are task-frame-relative, so the new origin is obtained by
        composing the observed pose with the current frame origin.
        Bounds are reset in the new frame; for wrapped rotational axes, both
        endpoints are initialized to the current angle (typically zero after
        origin reset).
        """
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive

        primitive = self.mp_net.config.primitives[primitive_name]
        if not primitive.is_adaptive:
            logging.info("[%s] Ignoring origin set request because the primitive is not adaptive.", primitive_name)
            return

        env = self.mp_net._envs[primitive_name]
        current_in_frame = self.current_pose_in_frame_by_robot(primitive_name)

        for robot_name, frame in primitive.task_frame.items():
            rec = self._record_for_robot(primitive_name, robot_name)
            key = (primitive_name, robot_name)
            new_origin_world = task_pose_to_world_pose(current_in_frame[robot_name], frame.origin)
            print("Origin", new_origin_worldo)
            new_origin_world = [float(v) for v in new_origin_world]

            frame.origin = list(new_origin_world)
            env.task_frame[robot_name].origin = list(new_origin_world)

            zero_pose = [0.0] * 6
            frame.target = list(zero_pose)
            #rec["min_pose"] = list(zero_pose)
            #rec["max_pose"] = list(zero_pose)
            rec["rotation_interval_modes"] = [RotationIntervalMode.LINEAR.to_name()] * 6

            env.task_frame[robot_name].target = list(zero_pose)
            env.task_frame[robot_name].min_pose = list(zero_pose)
            env.task_frame[robot_name].max_pose = list(zero_pose)
            self._set_frame_rotation_interval_modes(frame, rec["rotation_interval_modes"])
            self._set_frame_rotation_interval_modes(env.task_frame[robot_name], rec["rotation_interval_modes"])

            self._last_pose_in_frame_by_robot[key] = list(zero_pose)
            self._rot_unwrapped_min_by_robot[key] = np.zeros(3, dtype=np.float64)
            self._rot_unwrapped_max_by_robot[key] = np.zeros(3, dtype=np.float64)

        self._reset_runtime_state_for_primitive(primitive_name)
        self._tracked_primitives.add(primitive_name)
        logging.info("[%s] Set frame origin from current pose and reset bounds.", primitive_name)
        self.log_status(primitive_name)

    def reset_bounds(self, primitive_name: str | None = None) -> None:
        """Reset min/max bounds to the robot's current pose in the current frame."""
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive

        if primitive_name not in self._tracked_primitives:
            logging.info("[%s] Bounds reset ignored because no origin has been set yet.", primitive_name)
            return

        primitive = self.mp_net.config.primitives[primitive_name]
        env = self.mp_net._envs[primitive_name]
        current_in_frame = self.current_pose_in_frame_by_robot(primitive_name)

        for robot_name, frame in primitive.task_frame.items():
            rec = self._record_for_robot(primitive_name, robot_name)
            key = (primitive_name, robot_name)
            pose = [float(v) for v in current_in_frame[robot_name]]
            modes = self._rotation_interval_modes_for_frame(frame)

            rec["min_pose"] = list(pose)
            rec["max_pose"] = list(pose)

            rot = np.asarray(pose[3:6], dtype=np.float64)
            self._rot_unwrapped_min_by_robot[key] = rot.copy()
            self._rot_unwrapped_max_by_robot[key] = rot.copy()

            # Store wrapped arc endpoints for ccw_arc axes.
            for j, axis in enumerate(range(3, 6)):
                if modes[axis] == "ccw_arc":
                    rec["min_pose"][axis] = float(wrap_to_pi(rot[j]))
                    rec["max_pose"][axis] = float(wrap_to_pi(rot[j]))

            #self._infer_rotation_interval_modes_for_robot(primitive_name, robot_name, frame)
            #self._set_frame_rotation_interval_modes(env.task_frame[robot_name], self._rotation_interval_modes_for_frame(frame))

        self._reset_runtime_state_for_primitive(primitive_name)
        logging.info("[%s] Reset workspace bounds in the current frame.", primitive_name)
        self.log_status(primitive_name)

    def update_bounds(self, primitive_name: str | None = None) -> None:
        """Expand per-axis workspace bounds from the current task-frame pose.

        Translation and linear rotational axes use standard numeric min/max.
        Rotational axes configured as `ccw_arc` are tracked in a temporary
        continuous coordinate during calibration, but stored back into the
        config as wrapped arc endpoints.
        """
        if primitive_name is None:
            primitive_name = self.mp_net.active_primitive

        primitive = self.mp_net.config.primitives[primitive_name]
        current_in_frame = self.current_pose_in_frame_by_robot(primitive_name)

        for robot_name, frame in primitive.task_frame.items():
            rec = self._record_for_robot(primitive_name, robot_name)
            key = (primitive_name, robot_name)
            pose = [float(v) for v in current_in_frame[robot_name]]
            modes = self._rotation_interval_modes_for_frame(frame)

            # --- translation always linear ---
            for axis in range(3):
                cur = float(pose[axis])

                if not math.isfinite(float(rec["min_pose"][axis])):
                    rec["min_pose"][axis] = cur
                else:
                    rec["min_pose"][axis] = min(float(rec["min_pose"][axis]), cur)

                if not math.isfinite(float(rec["max_pose"][axis])):
                    rec["max_pose"][axis] = cur
                else:
                    rec["max_pose"][axis] = max(float(rec["max_pose"][axis]), cur)

            # --- rotation ---
            rot = np.asarray(pose[3:6], dtype=np.float64)

            if key not in self._rot_unwrapped_min_by_robot or not np.all(
                    np.isfinite(self._rot_unwrapped_min_by_robot[key])):
                self._rot_unwrapped_min_by_robot[key] = rot.copy()
            if key not in self._rot_unwrapped_max_by_robot or not np.all(
                    np.isfinite(self._rot_unwrapped_max_by_robot[key])):
                self._rot_unwrapped_max_by_robot[key] = rot.copy()

            rot_min = self._rot_unwrapped_min_by_robot[key]
            rot_max = self._rot_unwrapped_max_by_robot[key]

            for j, axis in enumerate(range(3, 6)):
                if modes[axis] == "ccw_arc":
                    rot_min[j] = min(float(rot_min[j]), float(rot[j]))
                    rot_max[j] = max(float(rot_max[j]), float(rot[j]))
                    rec["min_pose"][axis] = float(wrap_to_pi(rot_min[j]))
                    rec["max_pose"][axis] = float(wrap_to_pi(rot_max[j]))
                else:
                    cur = float(pose[axis])

                    if not math.isfinite(float(rec["min_pose"][axis])):
                        rec["min_pose"][axis] = cur
                    else:
                        rec["min_pose"][axis] = min(float(rec["min_pose"][axis]), cur)

                    if not math.isfinite(float(rec["max_pose"][axis])):
                        rec["max_pose"][axis] = cur
                    else:
                        rec["max_pose"][axis] = max(float(rec["max_pose"][axis]), cur)

            self._infer_rotation_interval_modes_for_robot(primitive_name, robot_name, frame)

    def _apply_recorded_bounds_to_config(self) -> None:
        """Copy recorded calibration data into the config before saving."""
        for primitive_name, primitive in self.mp_net.config.primitives.items():
            for robot_name, frame in primitive.task_frame.items():
                rec = self._record_for_robot(primitive_name, robot_name)
                frame.origin = None if rec["origin"] is None else list(rec["origin"])
                frame.min_pose = None if rec["min_pose"] is None else list(rec["min_pose"])
                frame.max_pose = None if rec["max_pose"] is None else list(rec["max_pose"])
                self._set_frame_rotation_interval_modes(frame, self._ensure_rotation_interval_modes(rec))

    def save(self) -> None:
        self._apply_recorded_bounds_to_config()
        save_mpnet_config(self.mp_net.config, self.output_path)
        logging.info("Saved calibrated MP-Net config to %s", self.output_path)

    def log_status(self, primitive_name: str | None = None) -> None:
        """Log recorded origin and bounds for the active primitive."""
        primitive_name, primitive = self._get_primitive(primitive_name)
        summary = {}
        for robot_name in primitive.task_frame:
            rec = self._record_for_robot(primitive_name, robot_name)
            summary[robot_name] = {
                "origin": rec["origin"],
                "min_pose": rec["min_pose"],
                "max_pose": rec["max_pose"],
            }
        logging.info("[%s] %s", primitive_name, pformat(summary))

    def print_live_pose(self, primitive_name: str | None = None) -> None:
        """Print the current observed task-frame pose on one terminal line."""
        primitive_name, _ = self._get_primitive(primitive_name)
        pose_in_frame = self.current_pose_in_frame_by_robot(primitive_name)

        parts = []
        for robot_name, pose in pose_in_frame.items():
            pose_str = ", ".join(
                f"{axis}={float(value):+.4f}"
                for axis, value in zip(AXIS_NAMES, pose, strict=True)
            )
            parts.append(f"{robot_name}: {pose_str}")

        sys.stdout.write(f"\r[{primitive_name}] " + " | ".join(parts) + "   ")
        sys.stdout.flush()


def calibration_loop(
    mp_net: ManipulationPrimitiveNet,
    designer: WorkspaceConstraintDesigner,
    hotkeys: HotkeyController,
    autosave_on_primitive_change: bool = True,
):
    """Run the interactive calibration loop."""
    mp_net.set_step_info({TeleopEvents.IS_INTERVENTION: True})
    transition = mp_net.reset()
    logging.info(HOTKEY_HELP)
    logging.info("Entered primitive '%s'.", mp_net.active_primitive)

    while True:
        start_loop_t = time.perf_counter()

        if hotkeys.consume("quit"):
            designer.save()
            return transition.get(TransitionKey.INFO, {})

        if hotkeys.consume("save"):
            designer.save()

        if hotkeys.consume("print_status"):
            designer.log_status()

        if hotkeys.consume("set_origin"):
            designer.set_origin_from_current_pose()

        #if hotkeys.consume("reset_bounds"):
        #    designer.reset_bounds()

        action = torch.zeros(mp_net.action_dim, dtype=torch.float32)
        previous_primitive = mp_net.active_primitive
        transition = mp_net.step(action)

        #designer.update_bounds(previous_primitive)
        designer.print_live_pose(previous_primitive)

        info = transition.get(TransitionKey.INFO, {})
        next_primitive = info.get("transition_to", previous_primitive)
        primitive_changed = next_primitive != previous_primitive

        if primitive_changed:
            designer.log_status(previous_primitive)
            if autosave_on_primitive_change:
                designer.save()

            transition = mp_net.reset()
            logging.info("Entered primitive '%s'.", mp_net.active_primitive)

        elif getattr(mp_net, "_needs_full_reset", False):
            designer.log_status(previous_primitive)
            if autosave_on_primitive_change:
                designer.save()

            transition = mp_net.reset()
            logging.info("Restarted episode at primitive '%s'.", mp_net.active_primitive)

        dt_load = time.perf_counter() - start_loop_t
        period = 1.0 / float(mp_net.config.fps)
        precise_sleep(max(0.0, period - dt_load))


@parser.wrap()
def design_constraints(cfg: DesignConstraintsConfig):
    logging.info(pformat(asdict(cfg)))

    mp_net = ManipulationPrimitiveNet(cfg.env)
    hotkeys = HotkeyController()
    designer = WorkspaceConstraintDesigner(mp_net=mp_net, output_path=cfg.output_path)

    logging.info(HOTKEY_HELP)

    try:
        log_say("Start calibration", play_sounds=cfg.play_sounds, blocking=False)
        calibration_loop(
            mp_net=mp_net,
            designer=designer,
            hotkeys=hotkeys,
            autosave_on_primitive_change=cfg.autosave_on_primitive_change,
        )
    finally:
        hotkeys.close()
        mp_net.close()
        log_say("Stop calibration", play_sounds=cfg.play_sounds, blocking=True)


if __name__ == "__main__":
    import experiments

    design_constraints()
