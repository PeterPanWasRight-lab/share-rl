from dataclasses import dataclass
from lerobot.policies.sac.configuration_sac import SACConfig


@SACConfig.register_subclass("sac_dagger_bc")
@dataclass
class SACDaggerBCConfig(SACConfig):
    """SAC architecture trained with DAgger-style behavior cloning updates."""

    training_mode: str = "bc"
    bc_lr: float | None = None
    bc_loss_type: str = "mse"
    bc_random_crop_padding: int = 0
    pretrained_vision_input_size: int | None = None
    proprio_latent_dim: int | None = None
    actor_update_after: int = 0
    sac_bc_loss_weight: float = 0.0
    freeze_shared_encoder_during_sac: bool = False
    stream_transitions_immediately: bool = False
    random_action_steps: int = 0
    limit_updates_to_online_transitions: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_mode not in {"bc", "sac"}:
            raise ValueError("training_mode must be 'bc' or 'sac'.")
        if self.bc_loss_type not in {"mse", "nll"}:
            raise ValueError("bc_loss_type must be 'mse' or 'nll'.")
        if self.bc_random_crop_padding < 0:
            raise ValueError("bc_random_crop_padding must be non-negative.")
        if self.pretrained_vision_input_size is not None and self.pretrained_vision_input_size < 1:
            raise ValueError("pretrained_vision_input_size must be positive when set.")
        if self.proprio_latent_dim is not None and self.proprio_latent_dim < 1:
            raise ValueError("proprio_latent_dim must be positive when set.")
        if self.actor_update_after < 0:
            raise ValueError("actor_update_after must be non-negative.")
        if self.sac_bc_loss_weight < 0:
            raise ValueError("sac_bc_loss_weight must be non-negative.")
        if self.random_action_steps < 0:
            raise ValueError("random_action_steps must be non-negative.")
