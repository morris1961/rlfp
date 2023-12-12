from envs import ALFWorldEnv
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from models import CustomActorCriticPolicy
from models import CustomExtractor
import os

EPOCH = 10
RETRY = 30
TOTAL_TIMESTAMPS = 2000
MYSEED = 0
MODEL_PATH = '../models/'

if __name__ == "__main__":

    env = ALFWorldEnv(RETRY)
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

    model = A2C.load(os.path.join(MODEL_PATH, 'model'), env=env)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)

    # # inference
    # done = False
    # enc_obs, infos = env.reset(MYSEED)
    # while not done:
    #     action, _state = model.predict(enc_obs, deterministic=True)
    #     enc_obs, reward, done, _, infos = env.step(action)
    #     print(f"observation: {infos['obs'][0]}")