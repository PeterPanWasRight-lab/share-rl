import os

os.environ.setdefault("PYNPUT_BACKEND", "dummy")

from lerobot.utils.constants import ACTION, OBS_STATE

from experiments.envs.hormann import UR3eHormannInsertionEnvConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import MoveDeltaPrimitiveConfig
from share.policies.sac_dagger import SACDaggerBCConfig


def test_hormann_insert_stats_do_not_pick_up_reset_gripper_channel():
    cfg = UR3eHormannInsertionEnvConfig()

    insert_primitive = cfg.primitives["insert"]
    reset_primitive = cfg.primitives["reset"]
    dataset_stats = insert_primitive.policy.dataset_stats

    assert isinstance(insert_primitive.policy, SACDaggerBCConfig)
    assert insert_primitive.processor.gripper.enable is False
    assert reset_primitive.processor.gripper.enable is True
    assert len(dataset_stats[OBS_STATE]["min"]) == 12
    assert len(dataset_stats[OBS_STATE]["max"]) == 12
    assert len(dataset_stats[ACTION]["min"]) == 4
    assert len(dataset_stats[ACTION]["max"]) == 4


def test_hormann_config_routes_pull_out_through_zero_ft_before_reset():
    cfg = UR3eHormannInsertionEnvConfig()

    assert "zero-ft" in cfg.primitives
    zero_ft_primitive = cfg.primitives["zero-ft"]
    reset_primitive = cfg.primitives["reset"]
    assert zero_ft_primitive.settle_duration_s == 0.3
    assert isinstance(reset_primitive, MoveDeltaPrimitiveConfig)
    assert reset_primitive.delta == [0.0] * 6

    pull_out_targets = [transition.target for transition in cfg.transitions if transition.source == "pull-out"]
    zero_ft_targets = [transition.target for transition in cfg.transitions if transition.source == "zero-ft"]

    assert "zero-ft" in pull_out_targets
    assert zero_ft_targets == ["reset"]
    assert cfg.transitions[-2].source == "zero-ft"
    assert cfg.transitions[-2].success_key == "primitive_complete"
