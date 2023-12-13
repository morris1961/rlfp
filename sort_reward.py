import sys

TASK_TYPES = [
    "pick_and_place_simple", 
    "look_at_obj_in_light", 
    "pick_clean_then_place_in_recep", 
    "pick_heat_then_place_in_recep", 
    "pick_cool_then_place_in_recep", 
    "pick_two_obj_and_place"
    ]

if __name__ == '__main__':
    f = open(sys.argv[1], 'r')
    lines = f.readlines()

    reward_list = [[]for i in range(6)]
    type_idx = 100
    for idx, line in enumerate(lines):
        if idx % 2 == 0:
            type_name = line.split(' ')[1].split('-')[0]
            type_idx = TASK_TYPES.index(type_name)
        else:
            reward = float(line.split(' ')[-1].split('\n')[0])
            reward_list[type_idx].append(reward)

    print(*reward_list, sep='\n')