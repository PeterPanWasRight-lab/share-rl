import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = Path(__file__).parents[3] / "hardEncodedScripts" / "generate_mujoco_insertion_demos.py"
SPEC = importlib.util.spec_from_file_location("generate_mujoco_insertion_demos", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
make_trajectory_specs = MODULE.make_trajectory_specs


def test_hard_encoded_trajectory_bank_is_deterministic_and_bounded():
    specs = make_trajectory_specs(100, seed=20260824)

    assert specs == make_trajectory_specs(100, seed=20260824)
    assert len(specs) == 100
    assert len({spec.seed for spec in specs}) == 100
    for spec in specs:
        axial, lateral_y, lateral_z = spec.start_offset_fixture_m
        assert 0.004 <= axial <= 0.014
        assert -0.007 <= lateral_y <= 0.007
        assert -0.007 <= lateral_z <= 0.007
        assert 7.0 <= spec.align_gain <= 10.0
        assert 0.075 <= spec.approach_speed_m_s <= 0.095
        assert 0.035 <= spec.insertion_speed_m_s <= 0.050
        assert spec.approach_waypoint_fixture_m == (0.0, 0.0)
        assert spec.curve_bulge_fixture_m == (0.0, 0.0)


def test_trajectory_randomization_widens_starts_and_adds_curved_approaches():
    specs = make_trajectory_specs(300, seed=20260827, trajectory_randomization=True)

    assert any(abs(spec.start_offset_fixture_m[1]) > 0.007 for spec in specs)
    assert any(abs(spec.start_offset_fixture_m[2]) > 0.007 for spec in specs)
    assert any(np.linalg.norm(spec.approach_waypoint_fixture_m) > 0.010 for spec in specs)
    assert any(np.linalg.norm(spec.curve_bulge_fixture_m) > 0.005 for spec in specs)
    for spec in specs:
        axial, lateral_y, lateral_z = spec.start_offset_fixture_m
        assert 0.0 <= axial <= 0.025
        assert -0.020 <= lateral_y <= 0.020
        assert -0.020 <= lateral_z <= 0.020
        assert np.linalg.norm(spec.approach_waypoint_fixture_m) <= 0.015 + 1e-12
        assert np.linalg.norm(spec.curve_bulge_fixture_m) <= 0.008 + 1e-12
        assert 0.7 <= spec.curve_power <= 1.6
