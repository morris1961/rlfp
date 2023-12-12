# adapt from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb
from utils.api import llm
import os
import argparse
import yaml
import json
import random
import alfworld
import alfworld.agents.environment
from envs.alfworld_env import TASK_TYPES, INIT_PROMPT
with open('base_config.yaml') as reader:
    config = yaml.safe_load(reader)

RETRY = 50

def check_ds():
    json_dir = os.path.expanduser('~/.cache/alfworld/json_2.1.1/')
    unseen_dir = os.path.join(json_dir, 'valid_unseen')
    test_dir = os.path.join(json_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        with open('test_trials.json', 'r') as f:
            data = json.load(f)
        for _, v in data.items():
            for path in v:
                source = os.path.join(unseen_dir, path)
                destination = os.path.join(test_dir, path)
                if not os.path.exists(destination):
                    os.mkdir(os.path.dirname(destination))
                os.system(f'cp -r {source} {destination}')

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def process_action(action):
    if action is not None and action != '':
        action = action.strip()
        if action.startswith('Agent: '):
            action = action[action.find('Agent: ')+7:]
    elif action is None or action == '':
        action = 'look'
    return action

def alfworld_run(prompt, to_print=True, ob='', env=None, model='bard'):
    init_prompt = prompt + ob + '\n'
    prompt = ''
    if to_print:
        print(ob)
    for i in range(1, RETRY):
        action = llm(init_prompt + prompt, stop=['\n'], model=model)
        action = process_action(action)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if "THOUGHT:" in action:
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
        prompt += f'{action}\n{observation}\n'
        if done:
            return reward
    return 0

def main(model='bard'):
    split = "eval_out_of_distribution"
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)

    stats = []

    for _ in range(12):
        obs, infos = env.reset()
        env_name = infos['extra.gamefile'][0].split('/')[-3]
        ex1 = None
        print('------------')
        print(f'Playing {env_name}\n')
        for i in TASK_TYPES:
            task = TASK_TYPES[i]
            if env_name.startswith(task):
                files = [os.path.join("examples", task, x) for x in os.listdir(os.path.join("examples", task)) if x.endswith("_3.txt")]
                ex1 = random.sample(files, 1)[0]
                ex_file1 = open(str(ex1), 'r')
                ex1 = ex_file1.read()
                prompt = INIT_PROMPT + ex1 + '\nAnd now is your turn:\n' + obs[0] + '\n'
                print(prompt)
                r = alfworld_run(prompt, env=env, model=model)
                break
        status = 'SUCCESS' if r == 1 else 'FAIL'
        stats.append({
            'idx': _,
            'env_name': env_name,
            'status': status
        })        
        print(f'Env {_}: {status}')
        print('------------\n')
    
    # save stats
    if not os.path.exists('logs'):
        os.mkdir('logs')
    with open(f'logs/{model}.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # print stats
    print('------------')
    success = sum([1 for s in stats if s['status'] == 'SUCCESS'])
    print(f'Success: {success}')
    print(f'Fail: {len(stats) - success}')
    print(f'Accuracy: {success / len(stats)}')
    print('------------')


if __name__ == "__main__":
    check_ds()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bard')
    args = parser.parse_args()

    main(args.model)
