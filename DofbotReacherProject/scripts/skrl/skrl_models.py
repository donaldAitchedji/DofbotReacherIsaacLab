"""Shared skrl model (network) definitions for the Dofbot reacher task, phase 1
(absolute joint-angle targets). Used by both train.py and play.py so the
architecture is guaranteed to match between training and evaluation.

NOTE: skrl 2.x requires Model/mixin base-class constructors to be called with
keyword arguments only (positional args raise a TypeError), hence the
`observation_space=..., action_space=..., device=...` style below.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class Policy(GaussianMixin, Model):
    """Gaussian policy: observation -> distribution over absolute joint-angle
    targets (in the normalized [-1, 1] action space handled by the env).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    """State-value function V(s), used by A2C/PPO for advantage estimation."""

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class Critic(DeterministicMixin, Model):
    """Action-value function Q(s, a), used by SAC (two critics + two targets)."""

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["observations"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


def build_actor_critic_models(observation_space, action_space, device) -> dict:
    """Shared Policy + Value pair, used by both A2C and PPO (identical
    architecture -- only the agent's update rule differs between the two)."""
    return {
        "policy": Policy(observation_space, action_space, device),
        "value": Value(observation_space, action_space, device),
    }


# aliases kept for readability at the call site (train.py / play.py)
build_a2c_models = build_actor_critic_models
build_ppo_models = build_actor_critic_models


def build_sac_models(observation_space, action_space, device) -> dict:
    return {
        "policy": Policy(observation_space, action_space, device),
        "critic_1": Critic(observation_space, action_space, device),
        "critic_2": Critic(observation_space, action_space, device),
        "target_critic_1": Critic(observation_space, action_space, device),
        "target_critic_2": Critic(observation_space, action_space, device),
    }
