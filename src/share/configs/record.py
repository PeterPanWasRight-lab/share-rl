from dataclasses import dataclass

from share.configs.mpnet import DatasetRecordConfig, TrainRLServerPipelineConfig
from share.configs.rl import MPNetTrainRLServerPipelineConfig
from share.debug.mpnet_debug import MPNetDebugConfig


@dataclass(kw_only=True)
class RecordConfig(MPNetTrainRLServerPipelineConfig):
    debug: MPNetDebugConfig | None = None
    # Whether record should load and run the configured policy.
    use_policy: bool = True
    # Whether to save only intervention/correction steps instead of the full rollout.
    save_only_interventions: bool = False
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
