from __future__ import annotations

from share.configs.measure_wrench import MeasureWrenchConfig
from share.workspace.mpnet import create_template_mpnet


def test_measure_wrench_config_accepts_measurement_fields():
    cfg = MeasureWrenchConfig(
        env=create_template_mpnet(),
        job_name='measure-wrench',
        sample_hz=25.0,
        history_window_s=5.0,
        autoscale=False,
        force_ylim=(-20.0, 20.0),
        torque_ylim=(-2.0, 2.0),
    )

    assert cfg.job_name == 'measure-wrench'
    assert cfg.sample_hz == 25.0
    assert cfg.history_window_s == 5.0
    assert cfg.autoscale is False
    assert cfg.force_ylim == (-20.0, 20.0)
    assert cfg.torque_ylim == (-2.0, 2.0)
