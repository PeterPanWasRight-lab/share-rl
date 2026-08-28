from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[3] / "hardEncodedScripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from generate_mujoco_force_search_demos import (  # noqa: E402
    ForceSearchSpec,
    ForceSearchState,
    ForceSearchTuning,
    _estimate_lateral_contact_offset,
    _spiral_target,
    make_force_search_specs,
)


def test_force_search_specs_require_contact_rich_offsets():
    specs = make_force_search_specs(20, seed=7)

    radii = [np.linalg.norm(spec.estimated_hole_offset_m) for spec in specs]
    assert all(0.004 <= radius <= 0.008 for radius in radii)
    assert all(5.0 <= spec.contact_force_n <= 8.0 for spec in specs)
    assert all(spec.spiral_direction == 1.0 for spec in specs)


def test_spiral_phase_sweeps_toward_true_hole_once_per_turn():
    spec = ForceSearchSpec(
        index=0,
        seed=1,
        estimated_hole_offset_m=(0.007, 0.0),
        spiral_direction=1.0,
        approach_speed_m_s=0.015,
        contact_force_n=6.0,
    )
    state = ForceSearchState(
        search_center_local=np.asarray(spec.estimated_hole_offset_m),
        search_point_index=32,
    )

    target = _spiral_target(spec, state, ForceSearchTuning(control_dt=1 / 30))

    assert np.linalg.norm(target) <= 0.0006


def test_signed_moment_biases_but_does_not_replace_spiral_target():
    spec = ForceSearchSpec(
        index=0,
        seed=1,
        estimated_hole_offset_m=(0.006, 0.0),
        spiral_direction=1.0,
        approach_speed_m_s=0.015,
        contact_force_n=6.0,
    )
    state = ForceSearchState(
        search_center_local=np.asarray(spec.estimated_hole_offset_m),
        search_point_index=8,
    )
    tuning = ForceSearchTuning(control_dt=1 / 30, moment_spiral_blend=0.1)
    spiral_only = _spiral_target(spec, state, tuning)
    state.moment_target_local = np.array([0.0, 0.004])

    blended = _spiral_target(spec, state, tuning)

    np.testing.assert_allclose(
        blended,
        0.9 * spiral_only + 0.1 * state.moment_target_local,
    )


def test_signed_moment_recovers_lateral_contact_offset():
    tuning = ForceSearchTuning(control_dt=1 / 30)
    force = np.array([-10.0, 1.5, -0.8])
    axial_lever_m = 0.18
    expected_offset = np.array([0.006, -0.004])
    contact_arm = np.array([axial_lever_m, *expected_offset])
    torque = np.cross(contact_arm, force)

    actual_offset = _estimate_lateral_contact_offset(
        force, torque, axial_lever_m, tuning
    )

    np.testing.assert_allclose(actual_offset, expected_offset, atol=1e-12)


def test_moment_offset_is_gated_without_axial_contact_and_clipped():
    tuning = ForceSearchTuning(control_dt=1 / 30)
    assert _estimate_lateral_contact_offset(
        np.array([-2.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 1.0]),
        0.18,
        tuning,
    ) is None

    force = np.array([-10.0, 0.0, 0.0])
    axial_lever_m = 0.18
    torque = np.cross(np.array([axial_lever_m, 0.10, 0.0]), force)
    offset = _estimate_lateral_contact_offset(
        force, torque, axial_lever_m, tuning
    )

    assert np.linalg.norm(offset) == pytest.approx(tuning.moment_max_offset_m)
