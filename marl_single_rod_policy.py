# marl_single_rod_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SingleRodActorCritic(nn.Module):
    """
    One PPO-style actor-critic for a single rod (2D action: slide, rotate).
    - Input: global observation (obs_dim)
    - Output: 2D action in [-1, 1]^2 (after tanh)
    - Also outputs a scalar value V(s)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, action_dim: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared body
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch_size, obs_dim)
        returns: mean (batch, action_dim), std (batch, action_dim), value (batch,)
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))

        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std).expand_as(mean)

        value = self.critic(x).squeeze(-1)
        return mean, std, value

    def act(self, obs: torch.Tensor):
        """
        obs: (batch_size, obs_dim)
        returns:
            action: (batch_size, action_dim) in [-1, 1]
            log_prob: (batch_size,)
            value: (batch_size,)
        """
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)

        raw_action = dist.rsample()            # reparameterized sample
        action = torch.tanh(raw_action)        # squash to [-1, 1]

        # For PPO, we approximate log_prob ignoring tanh correction
        log_prob = dist.log_prob(raw_action).sum(-1)

        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Used during PPO update.
        obs:     (batch, obs_dim)
        actions: (batch, action_dim) in [-1, 1]
        returns:
            log_probs, entropy, values
        """
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)

        # Map actions back to pre-tanh space approximately via atanh
        # Clamp for numerical stability
        clipped_actions = torch.clamp(actions, -0.999, 0.999)
        raw_actions = 0.5 * torch.log((1 + clipped_actions) / (1 - clipped_actions))

        log_probs = dist.log_prob(raw_actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        return log_probs, entropy, value
