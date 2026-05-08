import torch
import torch.nn as nn
import torch.nn.functional as F

from aitraineree.networks import NetworkType
from aitraineree.networks.bodies import FcNet


class IntrinsicCuriosityModule(NetworkType):
    """Intrinsic Curiosity Module (ICM) from Pathak et al. (2017).

    Consists of three components:
    - Feature encoder: maps observations to a learned feature space
    - Forward model: predicts next-state features given (features, action)
    - Inverse model: predicts action given (features, next_features)

    The intrinsic reward is the L2 prediction error of the forward model.

    Reference:
        "Curiosity-driven Exploration by Self-supervised Prediction"
        Pathak et al. (ICML 2017), https://arxiv.org/abs/1705.05363
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_size: int,
        feature_dim: int = 64,
        hidden_layers: tuple[int, ...] = (128,),
        beta: float = 0.2,
        eta: float = 0.01,
        device=None,
    ):
        """
        Parameters:
            obs_shape: Shape of the observation space.
            action_size: Dimension of the action space.
            feature_dim: Dimension of the learned feature embedding.
            hidden_layers: Hidden layer sizes for sub-networks.
            beta: Weight for forward vs inverse loss (forward_loss * beta + inverse_loss * (1-beta)).
            eta: Scaling factor for intrinsic reward.
            device: Device for tensors.
        """
        super().__init__()

        self.obs_shape = obs_shape
        self.action_size = action_size
        self.feature_dim = feature_dim
        self.beta = beta
        self.eta = eta

        self.feature_encoder = FcNet(
            (obs_shape[0],),
            (feature_dim,),
            hidden_layers=hidden_layers,
            gate=nn.ReLU(),
            gate_out=nn.ReLU(),
            device=device,
        )

        self.forward_model = FcNet(
            (feature_dim + action_size,),
            (feature_dim,),
            hidden_layers=hidden_layers,
            gate=nn.ReLU(),
            gate_out=nn.Identity(),
            device=device,
        )

        self.inverse_model = FcNet(
            (feature_dim * 2,),
            (action_size,),
            hidden_layers=hidden_layers,
            gate=nn.ReLU(),
            gate_out=nn.Identity(),
            device=device,
        )

    def forward(self, obs, next_obs, actions):
        phi = self.feature_encoder(obs)
        phi_next = self.feature_encoder(next_obs)

        predicted_phi_next = self.forward_model(torch.cat([phi, actions], dim=-1))
        predicted_actions = self.inverse_model(torch.cat([phi, phi_next], dim=-1))

        return phi_next, predicted_phi_next, predicted_actions

    def intrinsic_reward(self, obs, next_obs, actions):
        with torch.no_grad():
            phi_next = self.feature_encoder(next_obs)
            phi = self.feature_encoder(obs)
            predicted_phi_next = self.forward_model(torch.cat([phi, actions], dim=-1))
            reward = self.eta * 0.5 * F.mse_loss(predicted_phi_next, phi_next, reduction="none").sum(dim=-1, keepdim=True)
        return reward

    def compute_loss(self, obs, next_obs, actions):
        phi_next, predicted_phi_next, predicted_actions = self.forward(obs, next_obs, actions)

        forward_loss = 0.5 * F.mse_loss(predicted_phi_next, phi_next.detach())
        inverse_loss = F.mse_loss(predicted_actions, actions)

        loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss
        return loss, forward_loss.item(), inverse_loss.item()
