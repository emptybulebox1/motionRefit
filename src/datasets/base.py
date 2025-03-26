from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle

# from utils import *
# from constants import get_soft_mask, get_soft_mask2, rand_sample_mask


class baseDataset(Dataset):
    def __init__(self, folders, logger, seq_len, use_cfg, cfg_p, scale):
        self.folders = folders
        self.logger = logger
        self.seq_len = seq_len
        self.use_cfg = use_cfg
        self.cfg_p = cfg_p
        self.scale = scale

        self.data = {}
        self.len = 0

    def load_data(self, folder):
        pass
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        pass

    def print_config(self, **kwargs):
        print("In dataset")
        print("Folders: ", self.folders)
        print("Seq len: ", self.seq_len)
        print("Use cfg: ", self.use_cfg)
        print("Cfg p: ", self.cfg_p)
        print("Scale: ", self.scale)
        print("Len: ", self.len)
        