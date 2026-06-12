from __future__ import annotations

from lerobot.configs.default import DatasetConfig

from experiments.envs.hormann import UR3eHormannInsertionEnvConfig
from share.configs.rl import MPNetTrainRLServerPipelineConfig


def test_train_config_to_dict_serializes_base_primitive_configs(tmp_path):
    cfg = MPNetTrainRLServerPipelineConfig(
        env=UR3eHormannInsertionEnvConfig(),
        dataset=DatasetConfig(root=str(tmp_path / "dataset"), repo_id="test/repo"),
        policy=None,
        output_dir=tmp_path / "run",
        job_name="serialize",
    )

    encoded = cfg.to_dict()

    assert encoded["env"]["primitives"]["insert"]["type"] == "primitive"
    assert encoded["env"]["primitives"]["reset"]["type"] == "primitive"
