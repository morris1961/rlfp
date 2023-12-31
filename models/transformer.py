from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from transformers import BertForMultipleChoice

LLM_SIZE = 5

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
        model_name="bert-base-uncased",
    ):
        super().__init__()
        
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = LLM_SIZE
        self.latent_dim_vf = LLM_SIZE

        # base is bert and is shared by actor and critic
        # self.base_network = BertForMultipleChoice.from_pretrained(model_name)

        # Policy network
        self.policy_net = BertForMultipleChoice.from_pretrained(model_name)
        # Value network
        self.value_net = BertForMultipleChoice.from_pretrained(model_name)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # print("actor", features["input_ids"].shape)
        # x = self.base_network(**features).logits
        return self.policy_net(**features).logits

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        # print("critic", features)
        # x = self.base_network(**features).logits
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
        self.mlp_extractor = CustomNetwork()