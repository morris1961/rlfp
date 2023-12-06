import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from transformers import BertTokenizer
from utils.reward import Reward_Compute, extract_task

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
        self.reward_compute = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        obs, scores, dones, infos = self.env.step(self.LLMs[action])
        reward = self.reward_compute.obs_reward(obs[0])
        self.LLMs = np.random.choice(infos['admissible_commands'][0], 3) # get out put from LLMs
        enc_obs = self.tokenize(obs)
        infos['obs'] = obs
        self.attempt += 1
        if self.attempt >= self.max_attempt:
            dones = True
        else:
            dones = dones[0]
        return enc_obs, reward, dones, False, infos

    def reset(self, seed=None):
        self.seed(seed=seed)
        obs, infos = self.env.reset()
        task = extract_task(obs[0])
        self.reward_compute = Reward_Compute(task=task)
        self.LLMs = np.random.choice(infos['admissible_commands'][0], 3) # get out put from LLMs
        # enc_obs = self.tokenize(obs)
        infos['obs'] = obs
        return obs, infos
    
    def tokenize(self, obs):
        enc = self.tokenizer(obs[0] + " [SEP] " + self.LLMs[0] + " [SEP] " + self.LLMs[1] + " [SEP] " + self.LLMs[2],
                            padding="max_length",
                            max_length=self.tokenizer.model_max_length,
                            return_tensors='np')
        new_obs = {
            'input_ids':enc['input_ids'],
            'token_type_ids':enc['token_type_ids'],
            'attention_mask':enc['attention_mask'],
        }
        return new_obs