from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from transformers import BertForSequenceClassification


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        features_dim,
        device,
        model_name="bert-base-cased",
    ):
        super().__init__()

        self.policy_net = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3).to(device)
        self.value_net = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3).to(device)
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = 3
        self.latent_dim_vf = 3

        # Policy network
        # self.policy_net = nn.Sequential(
        #     nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        # )
        # Value network
        # self.value_net = nn.Sequential(
        #     nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        # )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # print("actor", features['input_ids'].dtype)
        return self.policy_net(**features).logits

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        # print("critic", features)
        return self.value_net(**features).logits


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, self.device)