from envs import ALFWorldEnv
from stable_baselines3 import A2C
from models import CustomActorCriticPolicy
from models import CustomExtractor
import os

EPOCH = 10
RETRY = 30
TOTAL_TIMESTAMPS = 2000
MYSEED = 0
MODEL_PATH = './checkpoints/'

if __name__ == "__main__":

    env = ALFWorldEnv(max_attempt=RETRY, train=True)
    env.seed(MYSEED)
    # print(f"task in this environment: {env.task}")
    
    model = A2C(
        CustomActorCriticPolicy,
        env,
        n_steps=5,
        verbose=1,
        policy_kwargs={
            "features_extractor_class":CustomExtractor,
        }
    )

    for i in range(EPOCH):
        model.learn(
            total_timesteps=TOTAL_TIMESTAMPS,
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        model.save(os.path.join(MODEL_PATH, f'{i}'))

    # inference
    done = False
    enc_obs, infos = env.reset(MYSEED)
    while not done:
        action, _state = model.predict(enc_obs, deterministic=True)
        enc_obs, reward, done, _, infos = env.step(action)