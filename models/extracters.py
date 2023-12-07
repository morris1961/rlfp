import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=3)

    def forward(self, observations):
        for key, obs in observations.items():
            observations[key] = obs.type(dtype=th.long)
        # print(observations['input_ids'].type(dtype=th.long))
        return observations