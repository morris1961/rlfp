from envs import ALFWorldEnv
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from models import CustomActorCriticPolicy
from models import CustomExtractor
import os

EPOCH = 10
RETRY = 60
TOTAL_TIMESTAMPS = 2000
MYSEED = 0
MODEL_PATH = './checkpoints/'

if __name__ == "__main__":

    env = ALFWorldEnv(max_attempt=RETRY, train=False)
    env.seed(MYSEED)
    # print(f"task in this environment: {env.task}")

    model = A2C.load(os.path.join(MODEL_PATH, 'model'), env=env)

    his_avg_reward = []
    his_total_reward = []
    num_trials = 0
    num_wins = 0
        
    for _ in range(EPOCH):
        total_reward = 0
        num_steps = 0
        done = False
        enc_obs, infos = env.reset(MYSEED)
        
        while not done:
            action = 0
            # action, _state = model.predict(enc_obs, deterministic=True)
            enc_obs, reward, done, _, infos = env.step(action)
            total_reward += reward
            num_steps += 1
            if 'won' in infos and infos['won'] == True:
                num_wins += 1
                num_trials += 1
            elif done:
                num_trials += 1
        his_total_reward.append(total_reward)
        his_avg_reward.append(total_reward/num_steps)
        # print(num_wins, num_trials)
        # print(total_reward, num_steps)
    print('Avg step rewards', his_avg_reward)
    print('Accuracy', num_wins/num_trials)