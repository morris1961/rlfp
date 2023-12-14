import spacy
import pandas as pd

#####################################
# need to install en_core_web_md by
# python -m spacy download en_core_web_sm 
# lg, md, sm is large, medium, small

target_tag = ['PNOUN', 'NOUN', 'PROPN']
task = 'examine an alarmclock with the desklamp.'
weight = 0.5
THOUGHT_PENALTY = -0.2

class Reward_Compute:
    def __init__(self, obs):
      task = obs.split('\n')[-1].split(':')[-1]
      self.task_noun_list = self.NounOfSentence(task)
      self.total_think = 0
      self.cont_think = 0
    
    def NounOfSentence(self, sentence, show_all=False):

        noun_list = []
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")

        # Example sentences
        doc = nlp(sentence)

        if show_all == True:
            columns=['text', 'pos', 'tag', 'dep', 'is_alpha', 'is_stop']
            dim = list(map(lambda x: [x.text, x.pos_, x.tag_, x.dep_, x.is_alpha, x.is_stop], doc))
            print(pd.DataFrame(dim, columns=columns))

        noun_idx = []
        for idx, info in enumerate(doc):
            if info.pos_ in target_tag:
                noun_idx.append(idx)
                if info.text not in noun_list:
                    noun_list.append(info.text)
                    # print(info.text, info.pos_)
        
        return noun_list

    def union_score(self, task_list, obs_list):
        inter = list(set(task_list) & set(obs_list))
        raw_score = len(inter) / len(task_list)
        return weight * raw_score

    def obs_reward(self, obs):
        self.cont_think = 0
        task_noun_list = self.task_noun_list
        obs_noun_list = self.NounOfSentence(obs)
        # print(task_noun_list)
        # print(obs_noun_list)
        reward = self.union_score(task_list=task_noun_list, obs_list=obs_noun_list)
        return reward
    
    def think_penalty(self, content):
        self.cont_think += 1
        self.total_think += 1
        reward = THOUGHT_PENALTY * self.cont_think
        think_useful = self.obs_reward(content)
        if(think_useful == 0):
            reward += THOUGHT_PENALTY * self.cont_think
        else:
            reward += think_useful
        return reward

if __name__=='__main__':
    reward_compute = Reward_Compute(task=task)
    print(f'Task: {task}\n')

    print()
    obs = 'You arrive at loc 8. On the desk 1, you see a pen 1, a bowl 1, a alarmclock 2, a pencil 2, a pencil 3, a creditcard 3, a book 1, a alarmclock 3, a keychain 3, and a book 2.'
    reward = reward_compute.obs_reward(obs=obs)
    print(f'Obs: {obs}')
    print(reward)

    print()
    obs = 'You arrive at loc 1. On the sidetable 2, you see a desklamp 1, and an alarmclock 1.'
    reward = reward_compute.obs_reward(obs=obs)
    print(f'Obs: {obs}')
    print(reward)
