import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import os
import time
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import random
from transformers import AutoTokenizer
from utils import get_answer, Reward_Compute
from datetime import datetime

FEATURE_DIM = 256
WIN_REWARD = 1
LOSE_PENALTY = -1
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place"
}

# modification
# Add rule 3 for output format

INIT_PROMPT = '''Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You have two choice:
1. Directly output the action in this turn. Output format: Your next action. 
2. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Output format: THOUGHT: Your thoughts.
After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment output \"Nothing happened.\", that means the previous action is invalid and you should try more options; if the environment output \"OK.\", that means you did not do anything to the environment. You have better do action in next step.
Last but not least, if you think too many times, the reward you acquire will be gradually decreased. So, try to react to the environment after you think.

Here is an example:\n
'''
class ALFWorldEnv(gym.Env):

    def __init__(self, max_attempt, llms, train=False) -> None:
        # load config
        self.config = generic.load_config()
        env_type = self.config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['transformer']['model'], use_fast=True)
        self.LLM_model_name = llms
        self.llm_size = len(self.LLM_model_name)

        self.action_space = spaces.Discrete(self.llm_size)
        self.observation_space = spaces.Dict({"input_ids": spaces.Box(low=0, high=self.tokenizer.vocab_size, shape=(self.llm_size, FEATURE_DIM), dtype=int),
                                              "token_type_ids": spaces.Box(low=0, high=1, shape=(self.llm_size, FEATURE_DIM), dtype=int),
                                              "attention_mask": spaces.Box(low=0, high=1, shape=(self.llm_size, FEATURE_DIM), dtype=int),})
        self.env = getattr(environment, env_type)(self.config, train_eval='train')   
        self.env = self.env.init_env(batch_size=1)
        self.LLMs = []
        self.max_attempt = max_attempt
        self.attempt = 0
        self.reward = 0
        self.history = None
        self.task = None
        self.reward_compute = None
        self.infos = None
        
        current_time = datetime.now()
        formatted_time = current_time.strftime("%H_%M_%S")
        self.history_file = f'history/{formatted_time}.txt'
        os.makedirs('history', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def step(self, action):
        print(f"action {self.attempt}: {self.LLM_model_name[action]}, {self.LLMs[action]}")
        for i, a in enumerate(self.LLMs):
            print(f"model: {self.LLM_model_name[i]}, a: {a}")

        self.attempt += 1
        if "THOUGHT:" in self.LLMs[action]:
            obs = ['OK.']
            print(f"observation: {obs[0]}")
            infos = {}
            self.history += self.LLMs[action] + '\n' + obs[0] + '\n'
            infos['obs'] = obs
            infos['won'] = [False]
            reward = self.reward_compute.think_penalty(self.LLMs[action])

            self.get_llm_answer()
            enc_obs = self.tokenize()
            if self.attempt >= self.max_attempt:
                dones = True
                reward = LOSE_PENALTY
            else:
                dones = False
        else:
            obs, _, dones, infos = self.env.step([self.LLMs[action]])
            if obs[0].startswith('You arrive at loc '):
                ob = obs[0][obs[0].find('. ')+2:]
            else:
                ob = obs[0]
            print(f"observation: {ob}")

            self.history += self.LLMs[action] + '\n' + ob + '\n'
            
            self.get_llm_answer()
            enc_obs = self.tokenize()
            infos['obs'] = [ob]
            if self.attempt >= self.max_attempt:
                dones = True
                reward = LOSE_PENALTY
            else:
                dones = dones[0]
                reward = WIN_REWARD if dones else self.reward_compute.obs_reward(ob)
        
        self.reward += reward
        print(f'reward {self.attempt}: {reward:4f}')
        return enc_obs, reward, dones, False, infos

    def reset(self, seed=None):
        self.seed(seed=seed)
        if self.attempt:
            print("****************************************************")
            print(f"last environment average reward: {self.reward / self.attempt:4f}")
            print("****************************************************")
            file = open(self.history_file, 'a')
            file.write(f"game: {self.infos['extra.gamefile'][0].split('/')[-3]}\naverage reward: {self.reward / self.attempt:4f}\n")
            file.close()
            
        obs, infos = self.env.reset()
        self.infos = infos

        self.task = obs[0].split('\n')[-1].split(':')[-1].strip(' ')

        print("====================================================")
        print(f"game: {infos['extra.gamefile'][0].split('/')[-3]}\ntask: {self.task}")
        print("====================================================")

        self.reward_compute = Reward_Compute(obs[0])

        # first time tell LLM what to do
        ex1 = self.get_example(infos)
        self.history = INIT_PROMPT + ex1 + '\nAnd now is your turn:\n' + obs[0] + '\n'
        self.get_llm_answer()

        obs_text = ['\n'.join(obs[0].split('\n')[:-1])]
        enc_obs = self.tokenize()
        infos['obs'] = obs_text
        infos['task'] = self.task
        self.attempt = 0
        self.reward = 0
        return enc_obs, infos
    
    def tokenize(self):
        question = [self.task for _ in range(self.llm_size)]
        choices = self.LLMs
        enc = self.tokenizer(
            question,
            choices,
            padding="max_length",
            max_length=FEATURE_DIM,
            return_tensors='np'
        )

        return {
            'input_ids': enc['input_ids'],
            'token_type_ids': enc['token_type_ids'],
            'attention_mask': enc['attention_mask'],
        }
    
    def get_llm_answer(self):
        self.LLMs = get_answer(self.history, self.LLM_model_name)
        
        for i, llm in enumerate(self.LLMs):
            if llm is None or llm == '':
                self.LLMs[i] = "look"

        for i in range(len(self.LLMs)):
            if self.LLMs[i] is not None:
                self.LLMs[i] = self.LLMs[i].strip(' ')

            # llama2 output strip
            if self.LLMs[i].startswith('Agent: '):
                self.LLMs[i] = self.LLMs[i][self.LLMs[i].find('Agent: ')+7:]

    def get_example(self, infos):
        env_name = infos['extra.gamefile'][0].split('/')[-3]
        example1 = None
        for i in TASK_TYPES:
            task = TASK_TYPES[i]
            if env_name.startswith(task):
                files = [os.path.join("examples", task, x) for x in os.listdir(os.path.join("examples", task)) if x.endswith("_3.txt")]
                ex1 = random.sample(files, 1)[0]
                ex_file1 = open(str(ex1), 'r')
                example1 = ex_file1.read()
                break

        return example1