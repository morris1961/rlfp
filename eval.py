from envs import ALFWorldEnv
from stable_baselines3 import A2C
import os

ENV_NUM = 12
RETRY = 50
MYSEED = 0
LLM_TYPES = ["llama2", "bardfree", "bard", "bard2", "gemini"]
model_name = f'{LLM_TYPES}LLMs'

if __name__ == "__main__":
    env = ALFWorldEnv(max_attempt=RETRY, llms=LLM_TYPES, train=False)
    env.seed(MYSEED)

    model = A2C.load(os.path.join('checkpoints', model_name), env=env)

    his_avg_reward = []
    his_total_reward = []
    num_trials = 0
    num_wins = 0
        
    for i in range(ENV_NUM):
        total_reward = 0
        num_steps = 0
        done = False
        enc_obs, infos = env.reset(MYSEED)
    
        while not done:
            action, _ = model.predict(enc_obs, deterministic=True)
            enc_obs, reward, done, _, infos = env.step(action)
            total_reward += reward
            num_steps += 1
            if done and reward == 1.0:
                num_wins += 1
                num_trials += 1
            elif done:
                num_trials += 1
        his_total_reward.append(total_reward)
        his_avg_reward.append(total_reward/num_steps)

    print(f'Success: {num_wins}')
    print(f'Fail: {num_trials - num_wins}')
    print('Avg step rewards', his_avg_reward)
    print('Accuracy', num_wins/num_trials)