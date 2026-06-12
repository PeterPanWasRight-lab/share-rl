import os

os.environ.setdefault("PYNPUT_BACKEND", "dummy")

from lerobot.utils.constants import ACTION, OBS_STATE

from experiments.envs.hormann import UR3eHormannInsertionEnvConfig
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
