import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import os
import time
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import random
from transformers import BertTokenizer
from utils import get_answer, Reward_Compute

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

INIT_PROMPT = '''
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You have two choice:
1. Directly output the action in this turn. Output format:\"your next action\n\". 
2. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Output format:\"THOUGHT: your thoughts. Action: your next action\n\". 
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment output \"Nothing happened\", that means the previous action is invalid and you should try more options; if the environment output \"OK\", that means you did not do anything to the environment. You have better do action in next step.

Rule:
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.
3. Your answer do not start with "ACTION:" or "Agent:".\n
Here is an example:\n
'''
class ALFWorldEnv(gym.Env):

    def __init__(self, max_attempt) -> None:
        # load config
        self.config = generic.load_config()
        env_type = self.config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
        self.tokenizer = BertTokenizer.from_pretrained(self.config['transformer']['model'])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict({"input_ids": spaces.Box(low=0, high=self.tokenizer.vocab_size, shape=(self.tokenizer.model_max_length,), dtype=int),
                                              "token_type_ids": spaces.Box(low=0, high=1, shape=(self.tokenizer.model_max_length,), dtype=int),
                                              "attention_mask": spaces.Box(low=0, high=1, shape=(self.tokenizer.model_max_length,), dtype=int),})
        self.env = getattr(environment, env_type)(self.config, train_eval='train')
        self.env = self.env.init_env(batch_size=1)
        self.LLMs = []
        self.max_attempt = max_attempt
        self.attempt = 0
        self.history = None
        self.task = None
        self.reward_compute = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def step(self, action):
        print(f"attempt: {self.attempt}, a: {self.LLMs[action]}")
        self.attempt += 1
        if "THOUGHT:" in self.LLMs[action]:
            obs = ['OK']
            infos = {}
            self.history += self.LLMs[action] + '\n' + obs[0] + '\n'
            infos['obs'] = obs
            
            # can remove this part
            time.sleep(1)
            
            self.get_llm_answer(self.history)
            enc_obs = self.tokenize(obs)
            dones = self.attempt >= self.max_attempt
            return enc_obs, -1, dones, False, infos
        else:
            obs, _, dones, infos = self.env.step([self.LLMs[action]])
            self.history += self.LLMs[action] + '\n' + obs[0] + '\n'
            reward = self.reward_compute.obs_reward(obs[0])
            
            # can remove this part
            time.sleep(1)
            
            self.get_llm_answer(self.history)
            # if len(self.LLMs) == 0:
            #     self.LLMs = np.random.choice(infos['admissible_commands'][0], 3) # get out put from LLMs
            enc_obs = self.tokenize(obs)
            infos['obs'] = obs
            if self.attempt >= self.max_attempt:
                dones = True
            else:
                dones = dones[0]
            return enc_obs, reward, dones, False, infos

    def reset(self, seed=None):
        self.seed(seed=seed)
        obs, infos = self.env.reset()
        self.task = obs[0].split('\n')[-1].split(':')[-1].strip(' ')
        self.reward_compute = Reward_Compute(obs[0])

        # first time tell LLM what to do
        ex1, ex2 = self.get_example(infos)
        self.get_llm_answer(INIT_PROMPT + ex1 + '\n' + ex2 + '\nAnd now is your turn:\n' + obs[0] + '\n')
        self.history = INIT_PROMPT + ex1 + '\n' + ex2 + '\nAnd now is your turn:\n' + obs[0] + '\n'

        # if len(self.LLMs) == 0:
        #     self.LLMs = np.random.choice(infos['admissible_commands'][0], 3) # get out put from LLMs

        obs_text = ['\n'.join(obs[0].split('\n')[:-1])]
        enc_obs = self.tokenize(obs_text)
        infos['obs'] = obs_text
        infos['task'] = self.task
        self.attempt = 0
        return enc_obs, infos
    
    def tokenize(self, obs):
        # can be commented
        print(f"obs: {obs[0]}\ntask: {self.task}\nLLM outputs: {self.LLMs}")
        enc = self.tokenizer(obs[0] + " [SEP] " + self.task + "[SEP]" + self.LLMs[0] + " [SEP] " + self.LLMs[1] + " [SEP] " + self.LLMs[2],
                            padding="max_length",
                            max_length=self.tokenizer.model_max_length,
                            return_tensors='np')
        new_obs = {
            'input_ids':enc['input_ids'],
            'token_type_ids':enc['token_type_ids'],
            'attention_mask':enc['attention_mask'],
        }
        return new_obs
    
    def get_llm_answer(self, prompt):
        self.LLMs = get_answer(prompt)

    def get_example(self, infos):
        env_name = infos['extra.gamefile'][0].split('/')[-3]
        example1 = None
        example2 = None
        for i in TASK_TYPES:
            task = TASK_TYPES[i]
            if env_name.startswith(task):
                files = [os.path.join("examples", task, x) for x in os.listdir(os.path.join("examples", task)) if x.endswith(".txt")]
                ex1, ex2 = random.sample(files, 2)
                ex_file1 = open(str(ex1), 'r')
                example1 = ex_file1.read()
                ex_file2 = open(str(ex2), 'r')
                example2 = ex_file2.read()
                break
        return example1, example2
