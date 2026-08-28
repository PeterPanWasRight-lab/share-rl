#!/usr/bin/env python
"""Generate force-guided insertion demonstrations with contact recovery."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import TransitionKey

from generate_mujoco_insertion_demos import (
    TASK,
    _action,
    _add_dataset_frame,
    _fixture_frame,
    _open_dataset,
)
from share.configs.mujoco_insertion import MujocoInsertionEnvConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (
    ManipulationPrimitiveNet,
)
from share.utils.env_config_snapshot import save_env_config_snapshot


REPO_ID = "local/mujoco-moment-guided-visual64"
STRATEGY_VERSION = 2


@dataclass(frozen=True)
class ForceSearchSpec:
    index: int
    seed: int
    estimated_hole_offset_m: tuple[float, float]
    spiral_direction: float
    approach_speed_m_s: float
    contact_force_n: float


@dataclass
class ForceSearchState:
    stage: str = "approach_contact"
    stage_steps: int = 0
    search_steps: int = 0
    recovery_count: int = 0
    filtered_reaction_n: float = 0.0
    filtered_torque_nm: float = 0.0
    baseline_force_local: np.ndarray | None = None
    baseline_torque_local: np.ndarray | None = None
    search_center_local: np.ndarray | None = None
    relief_start_axial_m: float | None = None
    surface_axial_m: float | None = None
    search_point_index: int = 0
    moment_target_local: np.ndarray | None = None
    moment_correction_count: int = 0
    filtered_force_residual_local: np.ndarray | None = None
    filtered_torque_residual_local: np.ndarray | None = None


@dataclass(frozen=True)
class ForceSearchTuning:
    control_dt: float
    contact_on_n: float = 5.0
    contact_off_n: float = 2.0
    dangerous_force_n: float = 30.0
    dangerous_torque_nm: float = 8.0
    relief_distance_m: float = 0.003
    moment_relief_distance_m: float = 0.003
    relief_speed_m_s: float = 0.035
    search_min_radius_m: float = 0.0005
    search_radial_spacing_m: float = 0.0015
    search_angle_step_rad: float = np.pi / 4.0
    search_lateral_gain: float = 15.0
    search_max_lateral_speed_m_s: float = 0.040
    search_position_tolerance_m: float = 0.0004
    axial_admittance_m_s_n: float = 0.0012
    max_axial_speed_m_s: float = 0.025
    hole_entry_depth_m: float = 0.002
    insertion_force_n: float = 12.0
    insertion_speed_m_s: float = 0.035
    lateral_admittance_m_s_n: float = 0.0008
    insertion_max_lateral_speed_m_s: float = 0.018
    wrench_filter_alpha: float = 0.25
    moment_min_axial_force_n: float = 3.0
    moment_min_offset_m: float = 0.001
    moment_max_offset_m: float = 0.020
    moment_correction_step_m: float = 0.004
    moment_spiral_blend: float = 0.10
    moment_insertion_gain: float = 0.5


def make_force_search_specs(count: int, seed: int) -> list[ForceSearchSpec]:
    """Sample biased hole estimates that force contact-rich recovery."""
    rng = np.random.default_rng(seed)
    specs: list[ForceSearchSpec] = []
    for index in range(count):
        radius = float(rng.uniform(0.004, 0.008))
        angle = float(rng.uniform(-np.pi, np.pi))
        specs.append(
            ForceSearchSpec(
                index=index,
                seed=seed + 20_000 + index,
                estimated_hole_offset_m=(
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                ),
                # A deterministic direction keeps feed-forward behavior
                # cloning from averaging clockwise/counter-clockwise actions.
                spiral_direction=1.0,
                approach_speed_m_s=float(rng.uniform(0.012, 0.020)),
                contact_force_n=float(rng.uniform(5.0, 8.0)),
            )
        )
    return specs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mujoco/hardEncodedMomentGuidedVisual64Demos"),
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--episode-steps", type=int, default=1300)
    return parser


def _move_tip_to_search_start(
    net: ManipulationPrimitiveNet,
    transition: dict,
    robot: Any,
    spec: ForceSearchSpec,
) -> tuple[dict, np.ndarray]:
    """Move above the biased hole estimate without recording setup frames."""
    target_local = np.array(
        [0.105, *spec.estimated_hole_offset_m], dtype=np.float64
    )
    for _ in range(90):
        _, fixture_rotation, tip_local = _fixture_frame(robot)
        local_error = target_local - tip_local
        if np.linalg.norm(local_error) <= 4e-4:
            break
        world_velocity = fixture_rotation @ np.clip(6.0 * local_error, -0.1, 0.1)
        transition = net.step(_action(world_velocity, net.action_dim))
        if transition.get(TransitionKey.DONE, False) or transition.get(
            TransitionKey.TRUNCATED, False
        ):
            raise RuntimeError("Episode ended while moving to the search start.")
    else:
        raise RuntimeError("Could not reach force-search start pose within 90 steps.")
    return transition, target_local


def _wrench_local(
    observation: dict[str, Any], fixture_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    wrench_world = np.asarray(
        [
            float(observation[f"main.{axis}.ee_wrench"])
            for axis in ("x", "y", "z", "rx", "ry", "rz")
        ],
        dtype=np.float64,
    )
    return fixture_rotation.T @ wrench_world[:3], fixture_rotation.T @ wrench_world[3:]


def _update_wrench_state(
    state: ForceSearchState,
    force_local: np.ndarray,
    torque_local: np.ndarray,
    tuning: ForceSearchTuning,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    if state.baseline_force_local is None or state.baseline_torque_local is None:
        state.baseline_force_local = force_local.copy()
        state.baseline_torque_local = torque_local.copy()
    force_residual = force_local - state.baseline_force_local
    torque_residual = torque_local - state.baseline_torque_local
    alpha = tuning.wrench_filter_alpha
    if state.filtered_force_residual_local is None:
        state.filtered_force_residual_local = np.zeros(3, dtype=np.float64)
        state.filtered_torque_residual_local = np.zeros(3, dtype=np.float64)
    state.filtered_force_residual_local += alpha * (
        force_residual - state.filtered_force_residual_local
    )
    state.filtered_torque_residual_local += alpha * (
        torque_residual - state.filtered_torque_residual_local
    )
    filtered_force = state.filtered_force_residual_local
    filtered_torque = state.filtered_torque_residual_local
    # The MuJoCo wrist sensor reports the environment reaction on the tool;
    # insertion along fixture -X therefore produces a negative local Fx.
    reaction_n = max(0.0, -float(filtered_force[0]))
    torque_nm = float(np.linalg.norm(filtered_torque[1:]))
    state.filtered_reaction_n = reaction_n
    state.filtered_torque_nm = torque_nm
    return (
        state.filtered_reaction_n,
        state.filtered_torque_nm,
        filtered_force.copy(),
        filtered_torque.copy(),
    )


def _estimate_lateral_contact_offset(
    force_local: np.ndarray,
    torque_local: np.ndarray,
    axial_lever_m: float,
    tuning: ForceSearchTuning,
) -> np.ndarray | None:
    """Estimate the lateral contact point from the signed wrist moment."""
    if -float(force_local[0]) < tuning.moment_min_axial_force_n:
        return None
    axial_force = float(force_local[0])
    if abs(axial_force) <= tuning.moment_min_axial_force_n:
        return None
    # tau = r x F. With fixture X as the insertion axis and the known
    # sensor-to-tip axial lever r_x, solve the Y/Z equations explicitly. This
    # removes the large moment contribution caused by transverse force acting
    # through the tool's axial lever arm.
    lateral_offset = np.asarray(
        [
            (axial_lever_m * force_local[1] - torque_local[2]) / axial_force,
            (torque_local[1] + axial_lever_m * force_local[2]) / axial_force,
        ],
        dtype=np.float64,
    )
    magnitude = float(np.linalg.norm(lateral_offset))
    if magnitude < tuning.moment_min_offset_m:
        return None
    if magnitude > tuning.moment_max_offset_m:
        lateral_offset *= tuning.moment_max_offset_m / magnitude
    return lateral_offset


def _latch_moment_target(
    state: ForceSearchState,
    tip_local: np.ndarray,
    force_local: np.ndarray,
    torque_local: np.ndarray,
    axial_lever_m: float,
    tuning: ForceSearchTuning,
) -> bool:
    contact_offset = _estimate_lateral_contact_offset(
        force_local, torque_local, axial_lever_m, tuning
    )
    if contact_offset is None:
        state.moment_target_local = None
        return False
    direction = contact_offset / np.linalg.norm(contact_offset)
    # For the wrist reaction wrench, the estimated line-of-action offset points
    # from the displaced peg center toward the contacted hole edge. Moving in
    # that direction unloads the edge and reduces the peg/hole center error.
    state.moment_target_local = (
        tip_local[1:] + tuning.moment_correction_step_m * direction
    )
    state.moment_correction_count += 1
    return True


def _change_stage(state: ForceSearchState, stage: str) -> None:
    state.stage = stage
    state.stage_steps = 0


def _spiral_target(
    spec: ForceSearchSpec,
    state: ForceSearchState,
    tuning: ForceSearchTuning,
) -> np.ndarray:
    if state.search_center_local is None:
        raise RuntimeError("Spiral search center is not initialized.")
    search_phase = np.arctan2(
        -spec.estimated_hole_offset_m[1],
        -spec.estimated_hole_offset_m[0],
    )
    swept_angle = tuning.search_angle_step_rad * state.search_point_index
    theta = search_phase + spec.spiral_direction * swept_angle
    radius = tuning.search_min_radius_m + (
        tuning.search_radial_spacing_m * swept_angle / (2.0 * np.pi)
    )
    spiral_target = state.search_center_local + radius * np.array(
        [np.cos(theta), np.sin(theta)], dtype=np.float64
    )
    if state.moment_target_local is None:
        return spiral_target
    blend = float(np.clip(tuning.moment_spiral_blend, 0.0, 1.0))
    return (1.0 - blend) * spiral_target + blend * state.moment_target_local


def force_search_action(
    robot: Any,
    observation: dict[str, Any],
    spec: ForceSearchSpec,
    state: ForceSearchState,
    tuning: ForceSearchTuning,
    action_dim: int,
) -> tuple[Any, dict[str, float | str]]:
    """Apply a position-interface admittance approximation of ConnTact search."""
    fixture_position, fixture_rotation, tip_local = _fixture_frame(robot)
    sensor_local = fixture_rotation.T @ (
        robot._data.site_xpos[robot._force_torque_site_id] - fixture_position
    )
    axial_lever_m = float(tip_local[0] - sensor_local[0])
    force_local, torque_local = _wrench_local(observation, fixture_rotation)
    reaction_n, torque_nm, force_residual, torque_residual = _update_wrench_state(
        state, force_local, torque_local, tuning
    )

    local_velocity = np.zeros(3, dtype=np.float64)
    if (
        reaction_n >= tuning.dangerous_force_n
        or torque_nm >= tuning.dangerous_torque_nm
    ) and state.stage != "safety_retract":
        _latch_moment_target(
            state,
            tip_local,
            force_residual,
            torque_residual,
            axial_lever_m,
            tuning,
        )
        state.recovery_count += 1
        state.relief_start_axial_m = float(tip_local[0])
        _change_stage(state, "safety_retract")

    if state.stage == "approach_contact":
        local_velocity[0] = -spec.approach_speed_m_s
        if reaction_n >= tuning.contact_on_n:
            _latch_moment_target(
                state,
                tip_local,
                force_residual,
                torque_residual,
                axial_lever_m,
                tuning,
            )
            state.search_center_local = tip_local[1:].copy()
            state.relief_start_axial_m = float(tip_local[0])
            state.surface_axial_m = float(tip_local[0])
            _change_stage(state, "force_relief")

    elif state.stage == "force_relief":
        local_velocity[0] = tuning.relief_speed_m_s
        relief_distance = (
            float(tip_local[0]) - state.relief_start_axial_m
            if state.relief_start_axial_m is not None
            else 0.0
        )
        required_relief_m = (
            tuning.moment_relief_distance_m
            if state.moment_target_local is not None
            else tuning.relief_distance_m
        )
        if state.stage_steps >= 2 and (
            reaction_n <= tuning.contact_off_n
            or relief_distance >= required_relief_m
        ):
            _change_stage(state, "spiral_search_move")

    elif state.stage == "safety_retract":
        local_velocity[0] = tuning.relief_speed_m_s
        relief_distance = (
            float(tip_local[0]) - state.relief_start_axial_m
            if state.relief_start_axial_m is not None
            else 0.0
        )
        if state.stage_steps >= 3 and (
            reaction_n <= tuning.contact_off_n
            or relief_distance >= 2.0 * tuning.relief_distance_m
        ):
            state.search_point_index += 1
            _change_stage(state, "spiral_search_move")

    elif state.stage == "spiral_search_move":
        if state.search_center_local is None:
            state.search_center_local = tip_local[1:].copy()
        target_lateral = _spiral_target(spec, state, tuning)
        local_velocity[1:] = np.clip(
            tuning.search_lateral_gain * (target_lateral - tip_local[1:]),
            -tuning.search_max_lateral_speed_m_s,
            tuning.search_max_lateral_speed_m_s,
        )
        if state.surface_axial_m is not None:
            axial_target = state.surface_axial_m + tuning.relief_distance_m
            local_velocity[0] = np.clip(
                8.0 * (axial_target - tip_local[0]),
                -tuning.max_axial_speed_m_s,
                tuning.relief_speed_m_s,
            )
        state.search_steps += 1
        if (
            np.linalg.norm(target_lateral - tip_local[1:])
            <= tuning.search_position_tolerance_m
            and abs(local_velocity[0]) <= 0.004
        ):
            state.moment_target_local = None
            _change_stage(state, "spiral_search_probe")

    elif state.stage == "spiral_search_probe":
        target_lateral = _spiral_target(spec, state, tuning)
        local_velocity[0] = -spec.approach_speed_m_s
        local_velocity[1:] = np.clip(
            tuning.search_lateral_gain * (target_lateral - tip_local[1:]),
            -tuning.search_max_lateral_speed_m_s,
            tuning.search_max_lateral_speed_m_s,
        )
        depth, _, _ = robot._insertion_metrics()
        if depth >= tuning.hole_entry_depth_m:
            _change_stage(state, "compliant_insert")
        elif reaction_n >= tuning.contact_on_n and state.stage_steps >= 2:
            _latch_moment_target(
                state,
                tip_local,
                force_residual,
                torque_residual,
                axial_lever_m,
                tuning,
            )
            state.relief_start_axial_m = float(tip_local[0])
            state.search_point_index += 1
            _change_stage(state, "force_relief")

    elif state.stage == "compliant_insert":
        force_error = tuning.insertion_force_n - reaction_n
        local_velocity[0] = np.clip(
            -tuning.axial_admittance_m_s_n * force_error,
            -tuning.insertion_speed_m_s,
            tuning.max_axial_speed_m_s,
        )
        moment_offset = _estimate_lateral_contact_offset(
            force_residual, torque_residual, axial_lever_m, tuning
        )
        moment_velocity = (
            tuning.moment_insertion_gain * moment_offset
            if moment_offset is not None
            else np.zeros(2, dtype=np.float64)
        )
        # Signed moment is primary; transverse force provides damping/unloading.
        local_velocity[1:] = np.clip(
            moment_velocity
            - tuning.lateral_admittance_m_s_n * force_residual[1:],
            -tuning.insertion_max_lateral_speed_m_s,
            tuning.insertion_max_lateral_speed_m_s,
        )

    state.stage_steps += 1
    moment_offset = _estimate_lateral_contact_offset(
        force_residual, torque_residual, axial_lever_m, tuning
    )
    diagnostics: dict[str, float | str] = {
        "stage": state.stage,
        "reaction_n": reaction_n,
        "torque_nm": torque_nm,
        "tip_lateral_m": float(np.linalg.norm(tip_local[1:])),
        "moment_offset_y_m": float(moment_offset[0]) if moment_offset is not None else 0.0,
        "moment_offset_z_m": float(moment_offset[1]) if moment_offset is not None else 0.0,
    }
    return _action(fixture_rotation @ local_velocity, action_dim), diagnostics


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "task": TASK,
            "strategy": "conntact-inspired-moment-guided-force-search",
            "strategy_version": STRATEGY_VERSION,
            "wrench_frame": "fixture",
            "episodes": [],
        }
    return json.loads(path.read_text())


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def generate(args: argparse.Namespace) -> None:
    if args.episodes <= 0 or args.max_attempts <= 0 or args.episode_steps <= 0:
        raise ValueError("episodes, max-attempts, and episode-steps must be positive.")

    output_root = args.output_root.resolve()
    manifest_path = output_root / "force_search_manifest.json"
    manifest = _load_manifest(manifest_path)
    if manifest.get("strategy_version") != STRATEGY_VERSION:
        recommended_root = output_root.parent / "hardEncodedMomentGuidedVisual64Demos"
        raise RuntimeError(
            f"Existing dataset '{output_root}' uses an incompatible wrench/controller "
            f"version. Keep it unchanged and use --output-root={recommended_root} for "
            "moment-guided demonstrations."
        )

    config = MujocoInsertionEnvConfig(
        viewer=args.viewer,
        teleop_mode="none",
        episode_steps=args.episode_steps,
    )
    tuning = ForceSearchTuning(control_dt=1.0 / config.fps)
    net = ManipulationPrimitiveNet(config)
    dataset: LeRobotDataset | None = None

    try:
        save_env_config_snapshot(config, output_root)
        dataset = _open_dataset(
            output_root,
            config,
            use_videos=not args.no_video,
            repo_id=REPO_ID,
        )
        completed = dataset.num_episodes
        if len(manifest["episodes"]) != completed:
            raise RuntimeError(
                "Dataset/manifest episode mismatch: "
                f"dataset={completed}, manifest={len(manifest['episodes'])}."
            )
        # Some contact configurations are deterministically unrecoverable in the
        # simplified gripper model. Keep the requested number of successful
        # demonstrations and record skipped candidate indices in the log instead
        # of aborting (or repeatedly retrying the same candidate on resume).
        candidate_count = max(args.episodes * 10, args.episodes + 100)
        specs = make_force_search_specs(candidate_count, args.seed)
        next_spec_index = max(
            (int(episode["index"]) for episode in manifest["episodes"]),
            default=-1,
        ) + 1
        saved_count = completed

        for spec in specs[next_spec_index:]:
            if saved_count >= args.episodes:
                break
            for attempt in range(args.max_attempts):
                net.request_full_reset()
                transition = net.reset(seed=spec.seed + attempt * 100_000)
                robot = next(iter(net.robot_dict.values()))
                try:
                    transition, target_local = _move_tip_to_search_start(
                        net, transition, robot, spec
                    )
                except RuntimeError as exc:
                    logging.warning(
                        "Episode %d setup attempt %d failed: %s",
                        spec.index,
                        attempt + 1,
                        exc,
                    )
                    continue

                state = ForceSearchState()
                stage_counts: dict[str, int] = {}
                peak_reaction_n = 0.0
                peak_torque_nm = 0.0
                success = False
                for frame_index in range(config.episode_steps):
                    observation = transition[TransitionKey.OBSERVATION]
                    action, diagnostics = force_search_action(
                        robot,
                        observation,
                        spec,
                        state,
                        tuning,
                        net.action_dim,
                    )
                    stage = str(diagnostics["stage"])
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    peak_reaction_n = max(
                        peak_reaction_n, float(diagnostics["reaction_n"])
                    )
                    peak_torque_nm = max(
                        peak_torque_nm, float(diagnostics["torque_nm"])
                    )
                    new_transition = net.step(action)
                    _add_dataset_frame(dataset, observation, new_transition)
                    transition = new_transition

                    if new_transition.get(TransitionKey.DONE, False):
                        success = (
                            float(new_transition[TransitionKey.REWARD]) > 0.0
                            and new_transition[TransitionKey.INFO].get(
                                "transition_reason"
                            )
                            == "peg_inserted"
                        )
                        break
                    if new_transition.get(TransitionKey.TRUNCATED, False):
                        break

                if not success:
                    depth, lateral_error, _ = robot._insertion_metrics()
                    dataset.clear_episode_buffer()
                    logging.warning(
                        "Episode %d attempt %d failed in stage %s "
                        "(depth=%.4f lateral=%.4f peak_force=%.1fN); retrying.",
                        spec.index,
                        attempt + 1,
                        state.stage,
                        depth,
                        lateral_error,
                        peak_reaction_n,
                    )
                    continue

                depth, lateral_error, axis_alignment = robot._insertion_metrics()
                dataset.save_episode()
                manifest["episodes"].append(
                    {
                        **asdict(spec),
                        "attempt": attempt + 1,
                        "frames": frame_index + 1,
                        "search_start_tip_local": target_local.tolist(),
                        "stage_counts": stage_counts,
                        "recovery_count": state.recovery_count,
                        "moment_correction_count": state.moment_correction_count,
                        "moment_spiral_blend": tuning.moment_spiral_blend,
                        "peak_reaction_n": peak_reaction_n,
                        "peak_torque_nm": peak_torque_nm,
                        "final_depth_m": depth,
                        "final_lateral_error_m": lateral_error,
                        "final_axis_alignment": axis_alignment,
                        "reward": 1.0,
                        "done": True,
                    }
                )
                _save_manifest(manifest_path, manifest)
                saved_count += 1
                print(
                    f"[{saved_count:03d}/{args.episodes:03d}] "
                    f"candidate={spec.index:03d} "
                    f"saved frames={frame_index + 1:03d} recoveries={state.recovery_count} "
                    f"peak_force={peak_reaction_n:.1f}N depth={depth:.4f}"
                )
                break
            else:
                logging.warning(
                    "Skipping force-search candidate %d after %d failed attempts.",
                    spec.index,
                    args.max_attempts,
                )
        if saved_count < args.episodes:
            raise RuntimeError(
                f"Generated only {saved_count}/{args.episodes} successful "
                f"trajectories after {candidate_count} candidates."
            )
    finally:
        if dataset is not None:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
            dataset.finalize()
        net.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate(_parser().parse_args())


if __name__ == "__main__":
    main()
