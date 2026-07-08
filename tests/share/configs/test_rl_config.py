from __future__ import annotations

from lerobot.configs.default import DatasetConfig
from lerobot.policies.sac.configuration_sac import SACConfig

from experiments.envs.hormann import UR3eHormannInsertionEnvConfig
from share.configs.rl import MPNetTrainRLServerPipelineConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import Always
from share.rl.runtime import build_adaptive_registry


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



def _adaptive_task_frame() -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6,
        origin=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None],
    )


def test_rl_config_validate_does_not_inject_top_level_policy(tmp_path):
    env_cfg = ManipulationPrimitiveNetConfig(
        start_primitive="start",
        reset_primitive="start",
        primitives={
            "start": ManipulationPrimitiveConfig(task_frame={"arm": _adaptive_task_frame()}),
            "insert": ManipulationPrimitiveConfig(
                task_frame={"arm": _adaptive_task_frame()},
                policy=SACConfig(device="cpu", storage_device="cpu"),
                is_terminal=True,
            ),
        },
        transitions=[Always(source="start", target="insert")],
        robot=None,
        teleop=None,
        cameras={},
    )
    cfg = MPNetTrainRLServerPipelineConfig(
        env=env_cfg,
        dataset=DatasetConfig(root=str(tmp_path / "dataset"), repo_id="test/repo"),
        policy=None,
        output_dir=None,
        job_name="job",
    )

    cfg.validate()

    assert cfg.policy is None


def test_build_adaptive_registry_skips_adaptive_primitives_without_policy():
    env_cfg = ManipulationPrimitiveNetConfig(
        start_primitive="start",
        reset_primitive="start",
        primitives={
            "start": ManipulationPrimitiveConfig(task_frame={"arm": _adaptive_task_frame()}),
            "insert": ManipulationPrimitiveConfig(
                task_frame={"arm": _adaptive_task_frame()},
                policy=SACConfig(device="cpu", storage_device="cpu"),
            ),
            "cleanup": ManipulationPrimitiveConfig(
                task_frame={"arm": _adaptive_task_frame()},
                is_terminal=True,
            ),
        },
        transitions=[
            Always(source="start", target="insert"),
            Always(source="insert", target="cleanup"),
        ],
        robot=None,
        teleop=None,
        cameras={},
    )

    registry = build_adaptive_registry(env_cfg)

    assert registry.adaptive_ids == ["insert"]
