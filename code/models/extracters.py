import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

LLM_SIZE = 5

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=LLM_SIZE)

    def forward(self, observations):
        # bert need dtype is th.long
        return {k: v.type(dtype=th.long) for k, v in observations.items()}