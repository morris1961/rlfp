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

INIT_PROMPT = '''Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. You have two choice:
1. Directly output the action in this turn. Output format: Your next action. 
2. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Output format: THOUGHT: Your thoughts.
After each turn, the environment will give you immediate feedback based on which you plan your next few steps. If the environment output \"Nothing happened.\", that means the previous action is invalid and you should try more options; if the environment output \"OK.\", that means you did not do anything to the environment. You have better do action in next step. Last but not least, your output cannot contain \"Agent: \".

Here is an example:\n
'''
# Rule:
# 1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
# 2. Think when necessary, try to act directly more in the process.
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
        # self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def step(self, action):
        print(f"attempt: {self.attempt}, a: {self.LLMs[action]}")
        self.attempt += 1
        if "THOUGHT:" in self.LLMs[action]:
            obs = ['OK.']
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
            if obs[0].startswith('You arrive at loc '):
                ob = obs[0][obs[0].find('. ')+2:]
            else:
                ob = obs[0]

            self.history += self.LLMs[action] + '\n' + ob + '\n'
            reward = self.reward_compute.obs_reward(ob)
            
            # can remove this part
            time.sleep(1)
            
            self.get_llm_answer(self.history)
            # if len(self.LLMs) == 0:
            #     self.LLMs = np.random.choice(infos['admissible_commands'][0], 3) # get out put from LLMs
            enc_obs = self.tokenize([ob])
            infos['obs'] = [ob]
            if self.attempt >= self.max_attempt:
                dones = True
            else:
                dones = dones[0]
            return enc_obs, reward, dones, False, infos

    def reset(self, seed=None):
        self.seed(seed=seed)
        obs, infos = self.env.reset()
        self.task = obs[0].split('\n')[-1].split(':')[-1].strip(' ')
        print(f"reset happened, task : {self.task}")

        if obs[0].startswith('You arrive at loc '):
            ob = obs[0][obs[0].find('. ')+2:]
        else:
            ob = obs[0]

        self.reward_compute = Reward_Compute(ob)

        # first time tell LLM what to do
        ex1 = self.get_example(infos)
        self.get_llm_answer(INIT_PROMPT + ex1 + '\nAnd now is your turn:\n' + ob + '\n')
        self.history = INIT_PROMPT + ex1 + '\nAnd now is your turn:\n' + ob + '\n'

        # if len(self.LLMs) == 0:
        #     self.LLMs = np.random.choice(infos['admissible_commands'][0], 3) # get out put from LLMs

        obs_text = ['\n'.join(ob.split('\n')[:-1])]
        enc_obs = self.tokenize(obs_text)
        infos['obs'] = obs_text
        infos['task'] = self.task
        self.attempt = 0
        return enc_obs, infos
    
    def tokenize(self, obs):
        # can be commented
        # print(f"obs: {obs[0]}\ntask: {self.task}\nLLM outputs: {self.LLMs}")
        if None in self.LLMs:
            print("===========\nNone exists~~\n===========")
            self.reset()

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

        # llama2 output strip
        if self.LLMs[0].startswith('Agent: '):
            self.LLMs[0] = self.LLMs[0][self.LLMs[0].find('Agent: ')+7:]

        for i in range(len(self.LLMs)):
            if self.LLMs[i] is not None:
                self.LLMs[i] = self.LLMs[i].strip(' ')

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