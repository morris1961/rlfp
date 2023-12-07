import os
import random
import gymnasium as gym
from gymnasium.envs.registration import register
from envs import ALFWorldEnv
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from models import CustomActorCriticPolicy
from models import CustomExtractor

TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place"
}

if __name__ == "__main__":

    env = ALFWorldEnv(10)
    
    model = A2C(
        CustomActorCriticPolicy,
        env,
        verbose=1,
        policy_kwargs={
            "features_extractor_class":CustomExtractor,
        }
    )

    # model.learn(
    #     total_timesteps=1000,
    #     reset_num_timesteps=False,
    #     # callback=WandbCallback(
    #     #     gradient_save_freq=100,
    #     #     verbose=2,
    #     # ),
    # )

    myseed = 0
    done = False
    env.seed(myseed)
    random.seed(myseed)
    obs, infos = env.reset()
    env_name = infos['extra.gamefile'][0].split('/')[-3]
    example = None
    for i in TASK_TYPES:
        task = TASK_TYPES[i]
        if env_name.startswith(task):
            files = [os.path.join("examples", task, x) for x in os.listdir(os.path.join("examples", task)) if x.endswith(".txt")]
            ex = random.sample(files, 1)[0]
            ex_file = open(str(ex), 'r')
            example = ex_file.read()
            break

    prompt = [obs[0] + '\n\nHere is an example, you are Agent:\n' + example]
    env.get_llm_answer(prompt[0])
    enc_obs = env.tokenize(prompt)    
    while not done:
        action, _state = model.predict(enc_obs, deterministic=True)
        enc_obs, reward, done, _, infos = env.step(action[0])
        print('> ' + env.LLMs[action[0]])
        prompt[0] += env.LLMs[action[0]] + '\n' + infos['obs'][0] + '\n'
        env.get_llm_answer(prompt[0])
        print(infos['obs'][0])