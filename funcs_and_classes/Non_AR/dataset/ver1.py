import sys
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))

from lib.utils.tools import read_pkl
from lib.utils.utils_non_AR import skel_to_h36m, generate_masked_joints_seq

class MotionDataset(Dataset):
    def __init__(self, args, data_split):   # data_split: 'train' or 'test'
        np.random.seed(0)
        random.seed(0)
        self.data_split = data_split    # 'train' or 'test'
        self.is_train = (True if data_split == 'train' else False)
        tasks_to_exclude = ['2DAR']
        query_list = []
        sample_count = {}
        global_idx_list = {task: [] for task in args.data.datasets if (task in args.tasks and task not in tasks_to_exclude)}
        global_sample_idx = 0
        for task, dataset_folder in args.data.datasets.items():
            if task in tasks_to_exclude:
                continue
            if task not in args.tasks:
                continue
            data_path = os.path.join(args.data.root_path, dataset_folder, data_split)
            file_list = sorted(os.listdir(data_path))
            sample_count[task] = len(file_list)
            for data_file in file_list:
                file_path = os.path.join(data_path, data_file)
                query_list.append({"task": task, "file_path": file_path})
                global_idx_list[task].append(global_sample_idx)
                global_sample_idx += 1
        print(f'{data_split} sample count: {sample_count}')
        self.query_list = query_list
        self.global_idx_list = global_idx_list
        self.task_to_flag = args.task_to_flag

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.query_list)


class MotionDatasetICL(MotionDataset):
    def __init__(self, args, data_split):
        super(MotionDatasetICL, self).__init__(args, data_split)
        self.clip_len = args.data.clip_len
        self.skel_amass_to_h36m = args.amass_to_h36m

        # all task inputs
        self.rootrel_input = args.rootrel_input
        # MC
        self.drop_ratios_MC = args.data.get('drop_ratios_MC')

    def prepare_sample_PE(self, sample_file):
        sample = read_pkl(sample_file)
        motion_input = sample["data_input"]  # (clip_len,17,3)
        motion_target = sample["data_label"]  # (clip_len,17,3)
        if self.rootrel_input:
            motion_input = motion_input - motion_input[..., [0], :]
        return torch.FloatTensor(motion_input), torch.FloatTensor(motion_target)
    
    def prepare_sample_FPE(self, sample_file):
        sample = read_pkl(sample_file)
        motion_input = sample["data_input"]  # (clip_len,17,3)
        motion_target = sample["data_label"]  # (clip_len,17,3)
        if self.rootrel_input:
            motion_input = motion_input - motion_input[..., [0], :]
        return torch.FloatTensor(motion_input), torch.FloatTensor(motion_target)

    def prepare_sample_MP(self, sample_file):
        sample = read_pkl(sample_file)
        motion_input = sample["data_input"]  # (96,18,3)
        motion_input = skel_to_h36m(motion_input, self.skel_amass_to_h36m)  # (96,17,3)
        motion_target = sample["data_label"]  # (96,18,3)
        motion_target = skel_to_h36m(motion_target, self.skel_amass_to_h36m)  # (96,17,3)
        if self.rootrel_input:
            motion_input = motion_input - motion_input[..., [0], :]
        return torch.FloatTensor(motion_input), torch.FloatTensor(motion_target)

    def prepare_sample_MC(self, sample_file):
        sample = read_pkl(sample_file)
        motion_input = sample['data_input']
        motion_target = sample["data_label"]
        if self.is_train:
            motion_input = skel_to_h36m(motion_input, self.skel_amass_to_h36m)
            motion_target = skel_to_h36m(motion_target, self.skel_amass_to_h36m)
        return torch.FloatTensor(motion_input), torch.FloatTensor(motion_target)

    def __getitem__(self, index):
        query_sample_dict = self.query_list[index] 
        task = query_sample_dict['task']
        task_flag = self.task_to_flag[task]
        query_file = query_sample_dict['file_path']

        if task == 'PE':
            query_input, query_target = self.prepare_sample_PE(query_file)      # (32,17,3), (32,17,3)
            query_input[..., -1] = 0.
        if task == 'FPE':
            query_input_, query_target_ = self.prepare_sample_FPE(query_file)     # (16,17,3), (16,17,3)
            mask = torch.zeros_like(query_input_)
            query_input = torch.cat([query_input_, mask], dim=0)     # (32,17,3)
            query_input[..., -1] = 0.
            query_target = torch.cat([query_input_, query_target_], dim=0)     # (32,17,3)
        elif task == 'MP':
            query_input_, query_target_ = self.prepare_sample_MP(query_file)      # (16,17,3), (16,17,3)
            mask = torch.zeros_like(query_input_)
            query_input = torch.cat([query_input_, mask], dim=0)     # (32,17,3)
            query_target = torch.cat([query_input_, query_target_], dim=0)     # (32,17,3)
        elif task == 'MC':
            drop_ratio = random.choice(self.drop_ratios_MC)
            query_input_, query_target = self.prepare_sample_MC(query_file)      # (32,17,3), (32,17,3)
            query_input, masked_joints = generate_masked_joints_seq(query_input_.clone(), drop_ratio)
            if not self.is_train:
                query_input = query_input_  # (32, 17, 3)

        return query_input, query_target, task_flag
               # (32,17,3), (32,17,3), int
