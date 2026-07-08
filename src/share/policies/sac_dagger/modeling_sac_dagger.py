from typing import Literal

import torch
import torch.nn.functional as F
from lerobot.policies.sac.modeling_sac import SACPolicy, TanhMultivariateNormalDiag
from lerobot.utils.constants import ACTION
from torch import Tensor

from share.policies.sac_dagger.configuration_sac_dagger import SACDaggerBCConfig


class SACDaggerBCPolicy(SACPolicy):
    config_class = SACDaggerBCConfig
    name = "sac_dagger_bc"

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
        loss_bc, training_infos = self.compute_loss_bc(
            observations=observations,
            actions=actions,
            observation_features=observation_features,
        )
        return {"loss_bc": loss_bc, "training_infos": training_infos}

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
        bc_loss = -log_probs.mean()
        return bc_loss, {"mse": mse}

    def _actor_distribution(
        self,
        observations: dict[str, Tensor],
        observation_features: Tensor | None = None,
    ) -> TanhMultivariateNormalDiag:
        obs_enc = self.actor.encoder(
            observations,
            cache=observation_features,
            detach=self.actor.encoder_is_shared,
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
