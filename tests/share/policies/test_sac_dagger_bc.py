from __future__ import annotations

from lerobot.policies.factory import get_policy_class, make_policy_config

import share.policies  # noqa: F401
from share.policies.sac_dagger import SACDaggerBCConfig, SACDaggerBCPolicy


def test_sac_dagger_bc_policy_registration():
    cfg = make_policy_config("sac_dagger_bc", device="cpu", storage_device="cpu", bc_lr=1e-4)

    assert isinstance(cfg, SACDaggerBCConfig)
    assert cfg.bc_lr == 1e-4
    assert get_policy_class(cfg.type) is SACDaggerBCPolicy
