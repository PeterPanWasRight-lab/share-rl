from dataclasses import dataclass

from lerobot.policies.sac.configuration_sac import SACConfig


@SACConfig.register_subclass("sac_dagger_bc")
@dataclass
class SACDaggerBCConfig(SACConfig):
    """SAC architecture trained with DAgger-style behavior cloning updates."""

    bc_lr: float | None = None
