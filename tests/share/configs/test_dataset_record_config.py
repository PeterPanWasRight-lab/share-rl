from __future__ import annotations

import draccus

from share.configs.record import DatasetRecordConfig


def test_dataset_record_config_decodes_without_explicit_type(tmp_path):
    cfg = draccus.decode(
        DatasetRecordConfig,
        {
            "root": str(tmp_path / "dataset"),
            "repo_id": "test/repo",
        },
    )

    assert isinstance(cfg, DatasetRecordConfig)
    assert cfg.root == str(tmp_path / "dataset")


def test_optional_dataset_record_config_decodes_without_explicit_type(tmp_path):
    cfg = draccus.decode(
        DatasetRecordConfig | None,
        {
            "root": str(tmp_path / "dataset"),
            "repo_id": "test/repo",
        },
    )

    assert isinstance(cfg, DatasetRecordConfig)
    assert cfg.repo_id == "test/repo"
