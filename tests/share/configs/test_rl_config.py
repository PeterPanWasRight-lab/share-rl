from __future__ import annotations

from lerobot.configs.default import DatasetConfig

from experiments.envs.hormann import UR3eHormannInsertionEnvConfig
from share.configs.rl import MPNetTrainRLServerPipelineConfig


def test_rl_config_default_output_dir_uses_dataset_root(tmp_path):
    cfg = MPNetTrainRLServerPipelineConfig(
        env=UR3eHormannInsertionEnvConfig(),
        dataset=DatasetConfig(root=str(tmp_path / "dataset"), repo_id="test/repo"),
        policy=None,
        output_dir=None,
        job_name="job",
    )

    cfg.validate()

    assert cfg.output_dir.parent == tmp_path / "dataset" / "run"
    assert cfg.output_dir.name.startswith("learner-")
