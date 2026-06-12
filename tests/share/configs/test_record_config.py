from __future__ import annotations

import draccus

from share.configs.record import RecordConfig


def test_record_config_accepts_learner_style_top_level_keys(tmp_path):
    cfg = draccus.decode(
        RecordConfig,
        {
            "env": {"type": "ur3e_hormann_insertion"},
            "dataset": {
                "root": str(tmp_path / "dataset"),
                "repo_id": "test/repo",
            },
            "job_name": "test-job",
            "save_freq": 500,
            "wandb": {
                "enable": False,
            },
        },
    )

    assert cfg.job_name == "test-job"
    assert cfg.save_freq == 500
    assert cfg.dataset is not None
    assert cfg.dataset.root == str(tmp_path / "dataset")
