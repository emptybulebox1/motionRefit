import random
import torch
import numpy as np
import os

def set_all_seeds(seed=42):
    print("set all seeds", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


#######################################################
def load_hint_texts_from_file(file_path):
    hint_texts = []
    with open(file_path, 'r') as file:
        for line in file:
            hint_texts.append([line.strip()])
    return hint_texts

def load_mask_from_file(file_path):
    mask = []
    with open(file_path, 'r') as file:
        for line in file:
            mask.append([line.strip()])
    return mask

def load_file_names(file_path):
    with open(file_path, 'r') as file:
        file_names = [line.strip() for line in file]
    return file_names

def gen_prog_ind(num_cases=16, sublist_length=4):
    total_range = 0.9
    step = total_range / sublist_length
    ranges = [(i * step, i * step + step / 5) for i in range(sublist_length)]
    
    prog_ind_all = []
    for _ in range(num_cases):
        while True:
            case = [random.uniform(r[0], r[1]) for r in ranges]
            if all(step*0.8 <= case[i+1] - case[i] <= step*1.6 for i in range(len(case) - 1)):
                prog_ind_all.append([case])
                break
    return prog_ind_all