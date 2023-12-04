import gymnasium as gym
from gymnasium.envs.registration import register
from envs import ALFWorldEnv
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from models import CustomActorCriticPolicy
from models import CustomExtractor

# register(
#     id='alfworld-v0',
#     entry_point='envs:ALFWorldEnv'
# )

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "algorithm": DQN,
    "policy_network": "MlpPolicy",
    "save_path": "models/sample_model",

    "epoch_num": 10,
    "timesteps_per_epoch": 1e5,
    "eval_episode_num": 10,
}

# def make_env():
#     env = gym.make('alfworld-v0')
#     return env

def train(env, model, config):

    current_best = 0
    best_epoch = 0

    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score = 0
        avg_highest = 0
        for seed in range(config["eval_episode_num"]):
            done = False

            # Set seed using old Gym API
            env.seed(seed)
            obs = env.reset()

            # Interact with env using old Gym API
            i = 0
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                # if info[0]["illegal_move"]:
                #     i += 1
                # else:
                #     i = 0
                # if i > 20:
                #     break

            avg_highest += info[0]['highest']/config["eval_episode_num"]
            avg_score   += info[0]['score']/config["eval_episode_num"]
        
        print("Avg_score:  ", avg_score)
        print("Avg_highest:", avg_highest)
        print(info[0]['matrix'])
        print()
        # wandb.log(
        #     {"avg_highest": avg_highest,
        #      "avg_score": avg_score}
        # )
        

        ### Save best model
        if current_best < avg_score:
            print("Saving Model")
            current_best = avg_score
            save_path = config["save_path"]
            model.save(f"{save_path}/DQN_best")
            best_epoch = epoch
        print(f"best_epoch: {best_epoch}, current_best: {current_best}")
        print("---------------")


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





    
