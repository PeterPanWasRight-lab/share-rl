from __future__ import annotations

from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE
import pytest
import torch

import share.policies  # noqa: F401
from share.policies.sac_dagger import SACDaggerBCConfig, SACDaggerBCPolicy
from share.policies.sac_dagger.modeling_sac_dagger import _ResizeImageEncoder, _batched_random_crop
from share.scripts.learner_server import _uses_bc_updates, make_optimizers


def test_sac_dagger_bc_policy_registration():
    cfg = make_policy_config("sac_dagger_bc", device="cpu", storage_device="cpu", bc_lr=1e-4)

    assert isinstance(cfg, SACDaggerBCConfig)
    assert cfg.bc_lr == 1e-4
    assert cfg.bc_loss_type == "mse"
    assert get_policy_class(cfg.type) is SACDaggerBCPolicy


def test_sac_dagger_bc_inference_is_deterministic():
    cfg = make_policy_config(
        "sac_dagger_bc",
        device="cpu",
        storage_device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    policy = SACDaggerBCPolicy(cfg).eval()
    observation = {OBS_STATE: torch.tensor([[0.1, -0.2, 0.3]])}

    first = policy.select_action(observation)
    second = policy.select_action(observation)

    torch.testing.assert_close(first, second)


def test_sac_dagger_online_mode_uses_sac_action_sampling(monkeypatch):
    cfg = make_policy_config(
        "sac_dagger_bc",
        device="cpu",
        storage_device="cpu",
        training_mode="sac",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    policy = SACDaggerBCPolicy(cfg).eval()
    expected = torch.tensor([[0.25, -0.5]])
    monkeypatch.setattr(SACPolicy, "select_action", lambda self, batch: expected)

    actual = policy.select_action({OBS_STATE: torch.zeros(1, 3)})

    assert actual is expected
    assert not _uses_bc_updates(policy)


def test_sac_dagger_rejects_unknown_training_mode():
    with pytest.raises(ValueError, match="training_mode"):
        SACDaggerBCConfig(training_mode="unknown")


def test_sac_dagger_rejects_negative_actor_update_after():
    with pytest.raises(ValueError, match="actor_update_after"):
        SACDaggerBCConfig(actor_update_after=-1)


def test_sac_dagger_rejects_negative_bc_anchor_weight():
    with pytest.raises(ValueError, match="sac_bc_loss_weight"):
        SACDaggerBCConfig(sac_bc_loss_weight=-1)


def test_sac_dagger_bc_optimizer_updates_shared_encoder():
    cfg = make_policy_config(
        "sac_dagger_bc",
        device="cpu",
        storage_device="cpu",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        shared_encoder=True,
    )
    policy = SACDaggerBCPolicy(cfg)

    optimizer = make_optimizers(policy)["actor"]
    optimized_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert {id(parameter) for parameter in policy.actor.encoder.parameters()} <= optimized_parameter_ids


def test_sac_dagger_online_actor_optimizer_excludes_shared_encoder():
    cfg = make_policy_config(
        "sac_dagger_bc",
        device="cpu",
        storage_device="cpu",
        training_mode="sac",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        shared_encoder=True,
    )
    policy = SACDaggerBCPolicy(cfg)

    optimizer = make_optimizers(policy)["actor"]
    optimized_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert {id(parameter) for parameter in policy.actor.encoder.parameters()}.isdisjoint(
        optimized_parameter_ids
    )


def test_sac_dagger_bc_shared_visual_encoder_receives_gradients():
    image_key = "observation.images.front"
    cfg = make_policy_config(
        "sac_dagger_bc",
        device="cpu",
        storage_device="cpu",
        input_features={image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        shared_encoder=True,
        freeze_vision_encoder=False,
        image_encoder_hidden_dim=8,
        image_embedding_pooling_dim=2,
        latent_dim=16,
    )
    policy = SACDaggerBCPolicy(cfg).train()

    loss, _ = policy.compute_loss_bc(
        observations={image_key: torch.rand(2, 3, 64, 64)},
        actions=torch.zeros(2, 2),
    )
    loss.backward()

    assert any(parameter.grad is not None for parameter in policy.actor.encoder.image_encoder.parameters())


def test_hilserl_resize_and_random_crop_keep_expected_shape():
    images = torch.rand(3, 3, 64, 64)
    resized = _ResizeImageEncoder(torch.nn.Identity(), 128)(images)
    cropped = _batched_random_crop(images, padding=4)

    assert resized.shape == (3, 3, 128, 128)
    assert cropped.shape == images.shape


def test_frozen_backbone_keeps_trainable_visual_heads():
    image_key = "observation.images.front"
    cfg = make_policy_config(
        "sac_dagger_bc",
        device="cpu",
        storage_device="cpu",
        input_features={image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        freeze_vision_encoder=True,
        image_encoder_hidden_dim=8,
        image_embedding_pooling_dim=2,
        latent_dim=16,
    )
    policy = SACDaggerBCPolicy(cfg).train()
    loss, _ = policy.compute_loss_bc(
        observations={image_key: torch.rand(2, 3, 64, 64)},
        actions=torch.zeros(2, 2),
    )
    loss.backward()

    assert all(parameter.grad is None for parameter in policy.actor.encoder.image_encoder.parameters())
    assert any(
        parameter.grad is not None
        for parameter in policy.actor.encoder.spatial_embeddings.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in policy.actor.encoder.post_encoders.parameters()
    )
