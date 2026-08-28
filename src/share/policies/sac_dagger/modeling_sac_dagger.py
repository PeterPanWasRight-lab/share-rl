from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.policies.sac.modeling_sac import (
    SACObservationEncoder,
    SACPolicy,
    SpatialLearnedEmbeddings,
    TanhMultivariateNormalDiag,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from torch import Tensor

from share.policies.sac_dagger.configuration_sac_dagger import SACDaggerBCConfig


class _ResizeImageEncoder(nn.Module):
    """Resize images before a pretrained encoder, matching HIL-SERL's JAX path."""

    def __init__(self, encoder: nn.Module, size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.size = size

    def forward(self, images: Tensor) -> Tensor:
        if images.shape[-2:] != (self.size, self.size):
            images = F.interpolate(
                images,
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            )
        return self.encoder(images)


def _batched_random_crop(images: Tensor, padding: int) -> Tensor:
    """Apply independent edge-padded random translations without changing shape."""
    if padding == 0:
        return images
    if images.ndim != 4:
        raise ValueError(f"Expected BCHW images, got shape {tuple(images.shape)}")
    _, _, height, width = images.shape
    padded = F.pad(images, (padding, padding, padding, padding), mode="replicate")
    crops = padded.unfold(2, height, 1).unfold(3, width, 1)
    offsets = torch.randint(
        0,
        2 * padding + 1,
        (images.shape[0], 2),
        device=images.device,
    )
    batch_indices = torch.arange(images.shape[0], device=images.device)
    return crops[batch_indices, :, offsets[:, 0], offsets[:, 1]]


class HILSERLObservationEncoder(SACObservationEncoder):
    """LeRobot encoder adjusted to the frozen HIL-SERL ResNet10 contract."""

    def _init_image_layers(self) -> None:
        super()._init_image_layers()
        target_size = self.config.pretrained_vision_input_size
        if not self.has_images or target_size is None:
            return

        self.image_encoder = _ResizeImageEncoder(self.image_encoder, target_size)
        sample_shape = self.config.input_features[self.image_keys[0]].shape
        dummy = torch.zeros(1, *sample_shape)
        with torch.no_grad():
            _, channels, height, width = self.image_encoder(dummy).shape

        # Only the spatial kernels depend on feature-map height and width. The
        # trainable projection heads created by LeRobot already have the right
        # channel-based input dimension.
        self.spatial_embeddings = nn.ModuleDict(
            {
                key.replace(".", "_"): SpatialLearnedEmbeddings(
                    height=height,
                    width=width,
                    channel=channels,
                    num_features=self.config.image_embedding_pooling_dim,
                )
                for key in self.image_keys
            }
        )

    def _init_state_layers(self) -> None:
        latent_dim = self.config.proprio_latent_dim or self.config.latent_dim
        self.has_env = OBS_ENV_STATE in self.config.input_features
        self.has_state = OBS_STATE in self.config.input_features
        if self.has_env:
            dim = self.config.input_features[OBS_ENV_STATE].shape[0]
            self.env_encoder = nn.Sequential(
                nn.Linear(dim, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh()
            )
        if self.has_state:
            dim = self.config.input_features[OBS_STATE].shape[0]
            self.state_encoder = nn.Sequential(
                nn.Linear(dim, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh()
            )

    def _compute_output_dim(self) -> None:
        proprio_dim = self.config.proprio_latent_dim or self.config.latent_dim
        output_dim = len(self.image_keys) * self.config.latent_dim if self.has_images else 0
        output_dim += proprio_dim if self.has_env else 0
        output_dim += proprio_dim if self.has_state else 0
        self._out_dim = output_dim


class SACDaggerBCPolicy(SACPolicy):
    config_class = SACDaggerBCConfig
    name = "sac_dagger_bc"

    def _init_encoders(self) -> None:
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = HILSERLObservationEncoder(self.config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else HILSERLObservationEncoder(self.config)
        )

    @torch.no_grad()
    def update_target_networks(self) -> None:
        """EMA independent critic heads without rounding shared encoder weights."""
        weight = self.config.critic_target_update_weight
        for target_param, source_param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            if target_param is source_param:
                continue
            target_param.data.lerp_(source_param.data, weight)
        if self.config.num_discrete_actions is not None:
            for target_param, source_param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                if target_param is source_param:
                    continue
                target_param.data.lerp_(source_param.data, weight)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the cloned policy mode without SAC exploration noise."""
        if self.config.training_mode == "sac":
            return super().select_action(batch)
        observation_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observation_features = self.actor.encoder.get_cached_image_features(batch)
        actions = self._actor_distribution(batch, observation_features).mode()

        if self.config.num_discrete_actions is not None:
            discrete_values = self.discrete_critic(batch, observation_features)
            discrete_action = torch.argmax(discrete_values, dim=-1, keepdim=True)
            actions = torch.cat([actions, discrete_action], dim=-1)
        return actions

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic", "bc"] = "critic",
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        if model != "bc":
            return super().forward(batch=batch, model=model)

        actions: Tensor = batch[ACTION]
        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor | None = batch.get("observation_feature")
        if self.training and self.config.bc_random_crop_padding and self.actor.encoder.has_images:
            observations = observations.copy()
            for image_key in self.actor.encoder.image_keys:
                observations[image_key] = _batched_random_crop(
                    observations[image_key],
                    self.config.bc_random_crop_padding,
                )
            # Cached features were computed before augmentation.
            observation_features = None
        loss_bc, training_infos = self.compute_loss_bc(
            observations=observations,
            actions=actions,
            observation_features=observation_features,
        )
        return {"loss_bc": loss_bc, "training_infos": training_infos}

    def augment_observations(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply the HIL-SERL random-translation augmentation to every camera."""
        padding = self.config.bc_random_crop_padding
        if not self.training or padding == 0 or not self.actor.encoder.has_images:
            return observations
        augmented = observations.copy()
        for image_key in self.actor.encoder.image_keys:
            augmented[image_key] = _batched_random_crop(
                observations[image_key], padding
            )
        return augmented

    def compute_loss_bc(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        observation_features: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        dist = self._actor_distribution(observations, observation_features)
        policy_actions = dist.mode()

        target_actions = actions
        continuous_dim = policy_actions.shape[-1]
        if target_actions.shape[-1] != continuous_dim:
            target_actions = target_actions[..., :continuous_dim]
        if self.config.policy_kwargs.use_tanh_squash:
            target_actions = torch.clip(target_actions, -1 + 1e-6, 1 - 1e-6)

        log_probs = dist.log_prob(target_actions)
        mse = F.mse_loss(policy_actions, target_actions, reduction="none").sum(-1).mean()
        nll = -log_probs.mean()
        bc_loss = mse if self.config.bc_loss_type == "mse" else nll
        return bc_loss, {"mse": mse, "nll": nll}

    def _actor_distribution(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> TanhMultivariateNormalDiag:
        obs_enc = self.actor.encoder(
            observations,
            cache=observation_features,
            detach=False,
        )
        outputs = self.actor.network(obs_enc)
        means = self.actor.mean_layer(outputs)

        if self.actor.fixed_std is None:
            log_std = self.actor.std_layer(outputs)
            std = torch.exp(log_std)
            std = torch.clamp(std, self.actor.std_min, self.actor.std_max)
        else:
            std = self.actor.fixed_std.expand_as(means)

        return TanhMultivariateNormalDiag(loc=means, scale_diag=std)
