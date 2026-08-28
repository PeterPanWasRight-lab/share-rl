#!/usr/bin/env python
"""Inject intermittent god-view keyboard interventions into a running Actor."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SOCKET_PATH = Path("/tmp/share_mujoco_god_view.sock")
DEFAULT_KEYBOARD_SOCKET_PATH = Path("/tmp/share_keyboard_teleop.sock")


@dataclass(frozen=True)
class InterventionPlan:
    stage: str
    world_delta_m: np.ndarray
    tip_fixture_m: np.ndarray


@dataclass(frozen=True)
class PlannerConfig:
    step_m: float
    clearance_m: float
    align_tolerance_m: float
    retreat_lateral_error_m: float
    target_depth_m: float


def plan_intervention(state: dict[str, Any], config: PlannerConfig) -> InterventionPlan:
    """Plan one bounded Cartesian correction from exact MuJoCo geometry."""
    tip_fixture = np.asarray(state["peg_tip_fixture_m"], dtype=np.float64)
    fixture_rotation = np.asarray(state["fixture_rotation_world"], dtype=np.float64)
    lateral_error = float(np.linalg.norm(tip_fixture[1:]))

    inside_clearance = tip_fixture[0] < config.clearance_m
    if inside_clearance and lateral_error > config.retreat_lateral_error_m:
        stage = "retreat"
        local_delta = np.array([config.clearance_m - tip_fixture[0], 0.0, 0.0])
    elif not inside_clearance and lateral_error > config.align_tolerance_m:
        stage = "align"
        local_delta = np.array(
            [config.clearance_m - tip_fixture[0], -tip_fixture[1], -tip_fixture[2]]
        )
    else:
        stage = "insert"
        target_x = 0.06 - config.target_depth_m
        local_delta = np.array(
            [target_x - tip_fixture[0], -tip_fixture[1], -tip_fixture[2]]
        )

    norm = float(np.linalg.norm(local_delta))
    if norm > config.step_m:
        local_delta *= config.step_m / norm
    return InterventionPlan(
        stage=stage,
        world_delta_m=fixture_rotation @ local_delta,
        tip_fixture_m=tip_fixture,
    )


class KeyboardPulseController:
    """Send the same key tokens consumed by MujocoInsertionEnvConfig."""

    def __init__(
        self,
        speed_m_s: float,
        actor_fps: float,
        socket_path: Path,
        dry_run: bool = False,
    ):
        self.speed_m_s = speed_m_s
        self.actor_fps = actor_fps
        self.dry_run = dry_run
        self.socket_path = socket_path.expanduser().resolve()
        self._socket = None
        self._keys = (
            ("left", "right"),
            ("down", "up"),
            ("shift_r", "shift"),
        )
        if not dry_run:
            self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    def _send_pulse(self, pulse: list[dict[str, float | str]]) -> None:
        try:
            self._socket.sendto(
                json.dumps({"pulse": pulse}).encode("utf-8"),
                str(self.socket_path),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Actor keyboard socket not found at {self.socket_path}; "
                "start Actor with --env.teleop_mode=keyboard"
            ) from exc

    def pulse(self, world_delta_m: np.ndarray) -> None:
        if self.dry_run:
            return
        pulse = []
        for axis, delta in enumerate(world_delta_m):
            if abs(float(delta)) < 1e-6:
                continue
            key = self._keys[axis][0 if delta > 0 else 1]
            value = min(1.0, abs(float(delta)) * self.actor_fps / self.speed_m_s)
            pulse.append({"key": key, "value": value})
        self._send_pulse(pulse)

    def release_all(self) -> None:
        if self._socket is None:
            return
        try:
            self._send_pulse([])
        except RuntimeError:
            pass

    def close(self) -> None:
        self.release_all()
        if self._socket is not None:
            self._socket.close()
            self._socket = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-path", type=Path, default=DEFAULT_SOCKET_PATH)
    parser.add_argument(
        "--keyboard-socket-path", type=Path, default=DEFAULT_KEYBOARD_SOCKET_PATH
    )
    parser.add_argument("--intervention-interval-s", type=float, default=0.5)
    parser.add_argument("--step-m", type=float, default=0.005)
    parser.add_argument("--teleop-speed-m-s", type=float, default=0.1)
    parser.add_argument("--actor-fps", type=float, default=30.0)
    parser.add_argument("--clearance-m", type=float, default=0.070)
    parser.add_argument("--align-tolerance-m", type=float, default=0.0005)
    parser.add_argument("--retreat-lateral-error-m", type=float, default=0.0015)
    parser.add_argument("--target-depth-m", type=float, default=0.075)
    parser.add_argument("--success-depth-m", type=float, default=0.070)
    parser.add_argument("--success-lateral-m", type=float, default=0.002)
    parser.add_argument("--success-axis-alignment", type=float, default=0.98)
    parser.add_argument("--episodes", type=int, default=0, help="0 runs until interrupted.")
    parser.add_argument("--state-timeout-s", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="Plan and print without pressing keys.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "intervention_interval_s",
        "step_m",
        "teleop_speed_m_s",
        "actor_fps",
        "state_timeout_s",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.episodes < 0:
        raise ValueError("--episodes cannot be negative")


def _socket_is_active(path: Path) -> bool:
    proc_net_unix = Path("/proc/net/unix")
    if not proc_net_unix.exists():
        return False
    return str(path) in proc_net_unix.read_text(errors="replace")


def _bind_state_socket(path: Path) -> socket.socket:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _socket_is_active(path):
            raise RuntimeError(f"Another autoControl receiver is already using {path}")
        path.unlink()
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.bind(str(path))
    os.chmod(path, 0o600)
    receiver.settimeout(0.2)
    return receiver


def _is_success(state: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        float(state["insertion_depth_m"]) >= args.success_depth_m
        and float(state["lateral_error_m"]) <= args.success_lateral_m
        and float(state["axis_alignment"]) >= args.success_axis_alignment
    )


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    socket_path = args.socket_path.expanduser().resolve()
    receiver = _bind_state_socket(socket_path)
    keyboard_controller = KeyboardPulseController(
        speed_m_s=args.teleop_speed_m_s,
        actor_fps=args.actor_fps,
        socket_path=args.keyboard_socket_path,
        dry_run=args.dry_run,
    )
    planner_config = PlannerConfig(
        step_m=args.step_m,
        clearance_m=args.clearance_m,
        align_tolerance_m=args.align_tolerance_m,
        retreat_lateral_error_m=args.retreat_lateral_error_m,
        target_depth_m=args.target_depth_m,
    )
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    current_episode: int | None = None
    completed_episodes = 0
    episode_complete = False
    next_intervention_time = time.monotonic()
    last_state_time = time.monotonic()
    print(f"[AUTO] Waiting for Actor god-view state on {socket_path}")

    try:
        while not stopping:
            try:
                payload = receiver.recv(65536)
            except socket.timeout:
                if time.monotonic() - last_state_time > args.state_timeout_s:
                    print("[AUTO] No Actor state received; is the MuJoCo Actor running?")
                    last_state_time = time.monotonic()
                continue
            last_state_time = time.monotonic()
            state = json.loads(payload)
            episode = int(state["episode"])
            if episode != current_episode:
                current_episode = episode
                episode_complete = False
                next_intervention_time = time.monotonic()
                print(f"[AUTO] Actor episode={episode}")

            if episode_complete:
                continue
            if _is_success(state, args):
                completed_episodes += 1
                episode_complete = True
                keyboard_controller.release_all()
                print(
                    f"[AUTO] SUCCESS episode={episode} completed={completed_episodes} "
                    f"depth={state['insertion_depth_m']:.4f} "
                    f"lateral={state['lateral_error_m']:.5f}"
                )
                if args.episodes and completed_episodes >= args.episodes:
                    break
                continue

            now = time.monotonic()
            if now < next_intervention_time:
                continue
            plan = plan_intervention(state, planner_config)
            print(
                f"[AUTO] episode={episode} stage={plan.stage:<7} "
                f"tip_fixture={np.round(plan.tip_fixture_m, 5).tolist()} "
                f"delta_world={np.round(plan.world_delta_m, 5).tolist()}"
            )
            keyboard_controller.pulse(plan.world_delta_m)
            next_intervention_time = now + args.intervention_interval_s
    finally:
        keyboard_controller.close()
        receiver.close()
        if socket_path.exists():
            socket_path.unlink()
        print("[AUTO] Stopped; all injected keys released.")


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
