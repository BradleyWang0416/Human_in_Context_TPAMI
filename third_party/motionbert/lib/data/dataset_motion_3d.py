import torch
import numpy as np
import glob
import os
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from third_party.motionbert.lib.data.augmentation import Augmenter3D
from third_party.motionbert.lib.utils.tools import read_pkl
from third_party.motionbert.lib.utils.utils_data import flip_data
    

class MotionDataset(Dataset):
    def __init__(self, args, data_split, dataset): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.dataset = dataset
        self.data_split = data_split
        file_list_all = []
        if dataset == 'h36m':
            data_path = os.path.join(self.data_root, 'H36M-SH', self.data_split)
        elif dataset == 'amass':
            data_path = os.path.join(self.data_root, 'AMASS', self.data_split)
        motion_list = sorted(os.listdir(data_path))
        for i in motion_list:
            file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionDataset3D(MotionDataset):
    def __init__(self, args, data_split, dataset):
        super(MotionDataset3D, self).__init__(args, data_split, dataset)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]           # [pretrain; train; index=66188] 'data/motion3d/MB3D_f243s81/AMASS/train/00066188.pkl'
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]       # [pretrain; train; index=66188] [243,17,3]
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
                if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')    
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)