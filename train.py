import gymnasium as gym
from gymnasium.envs.registration import register
from envs import ALFWorldEnv
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from models import CustomActorCriticPolicy
from models import CustomExtractor

if __name__ == "__main__":

    env = ALFWorldEnv(2)
    
    model = A2C(
        CustomActorCriticPolicy,
        env,
        verbose=1,
        policy_kwargs={
            "features_extractor_class":CustomExtractor,
        }
    )

    model.learn(
        total_timesteps=1000,
        reset_num_timesteps=False,
        # callback=WandbCallback(
        #     gradient_save_freq=100,
        #     verbose=2,
        # ),
    )

    done = False
    env.seed(0)
    enc_obs, infos = env.reset()
    print(infos['obs'][0])
    while not done:
        action, _state = model.predict(enc_obs, deterministic=True)
        enc_obs, reward, done, infos = env.step(action)
        print(infos['obs'][0])





    
