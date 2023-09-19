import os
import numpy as np
import pandas as pd
import random
import socket
import torch

from datetime import datetime
from torch.nn import Sequential
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

def replace_nans(arr, silent=True):
    if not silent:
        print('NaNs in array: ', len(arr[np.isnan(arr)]))
    arr[np.isnan(arr)] = 0
    return arr 

def add_macro_stats(confusion, silent):

    acc = confusion.diagonal().sum()/confusion.sum()        
    precision = confusion.diagonal()/confusion.sum(axis=0)
    recall = confusion.diagonal()/confusion.sum(axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    if np.isnan(acc):
        acc = 0
    precision = replace_nans(precision, silent)
    recall = replace_nans(recall, silent)
    f1 = replace_nans(f1, silent)

    precision = pd.DataFrame(precision, columns=['precision'])
    recall = pd.DataFrame(recall, columns=['recall'])
    f1 = pd.DataFrame(f1, columns=['f1'])

    df_c_m = pd.DataFrame(data=confusion, columns=range(len(confusion)), index=range(len(confusion)))
    df_c_m = pd.concat([df_c_m, precision.T, recall.T, f1.T], axis=0)

    return df_c_m

def pathify_list(lst, runs, model_pt):
    new_lst = []
    for l in lst:
        new_lst.append(os.path.join(runs, l, model_pt))
    return new_lst

def uniquify(path):
    dir = os.path.dirname(path)
    basename = os.path.basename(path)
    filename, extension = os.path.splitext(basename)
    filebase, counter = filename.split('_') 
    counter = int(counter)

    while os.path.exists(path):
        counter += 1
        path = os.path.join(dir, f"{filebase}_{str(counter)}{extension}")

    return path

def listify_keys(dict, key, number, cast_to):
    l = []
    for i in range(number):
        l.append(dict[f"{key}_{i}"])
    l = np.array(l)[(~pd.Series(l).isna()).tolist()].astype(cast_to).tolist()
    if len(l) == 0:
        return None
    return l

def train_val_split(data, val_split, seed):

    train_len = int((1-val_split) * len(data))
    train_data, val_data = random_split(data, [train_len, len(data) - train_len],generator=torch.Generator().manual_seed(seed))
    
    return train_data, val_data

def choose_split(split, train_data, val_data, test_data):
    if split == 'train':
        return train_data
    elif split == 'val':
        return val_data
    elif split == 'test':
        return test_data
    else:
        print("Invalid split, exiting...")
        exit(1)    

def sample_idx(len, frac, seed):

    random.seed(seed)
    frac_size = int(len * frac)
    return random.sample(range(len), frac_size)

def get_summarywriter(out_dir):

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(
        out_dir, current_time + "_" + socket.gethostname()
    )
    return SummaryWriter(log_dir=log_dir)