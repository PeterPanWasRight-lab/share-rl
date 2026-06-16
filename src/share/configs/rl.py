import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from share.configs.mpnet import DatasetRecordConfig, TrainRLServerPipelineConfig
from share.rl.runtime import build_adaptive_registry
from share.workspace.mpnet import ManipulationPrimitiveNetConfig, PreTrainedConfig


@dataclass(kw_only=True)
class MPNetTrainRLServerPipelineConfig(TrainRLServerPipelineConfig):
    """Train config for MP-Net distributed SAC actor/learner servers."""

    env: ManipulationPrimitiveNetConfig
    dataset: DatasetRecordConfig | None = None
    policy: PreTrainedConfig | None = None
    log_freq: int = 10
    num_workers: int = 6
    batch_size: int = 256

    def validate(self) -> None:
        if not self.job_name:
            self.job_name = f"{self.env.type}_sac"

        if self.output_dir is None:
            now = dt.datetime.now()
            if self.dataset is not None and self.dataset.root is not None:
                self.output_dir = Path(self.dataset.root) / "run" / f"learner-{now:%Y-%m-%d-%H-%M-%S}"
            else:
                self.output_dir = Path(f"outputs/train/{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}")
        else:
            self.output_dir = Path(self.output_dir)

        if self.policy is None:
            policies = [p.policy for p in self.env.primitives.values() if p.policy is not None]
            if policies:
                self.policy = policies[0]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _ = build_adaptive_registry(self.env)
