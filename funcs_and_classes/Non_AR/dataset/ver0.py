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
    def __init__(self, args, data_split, prompt_list=None):   # data_split: 'train' or 'test'
        if data_split == 'train':
            assert prompt_list is None
        if data_split == 'test':
            assert prompt_list is not None
        np.random.seed(0)
        random.seed(0)
        tasks = [task for task in args.tasks if task != '2DAR']
        self.data_split = data_split    # 'train' or 'test'
        self.is_train = (True if data_split == 'train' else False)
        query_list = []
        sample_count = {}
        global_idx_list = {task: [] for task in args.data.datasets if task in tasks}
        if self.is_train:
            prompt_list = {task: [] for task in args.data.datasets if task in tasks}
        global_sample_idx = 0
        for task, dataset_folder in args.data.datasets.items():
            if task not in tasks:
                continue
            data_path = os.path.join(args.data.root_path, dataset_folder, data_split)
            file_list = sorted(os.listdir(data_path))
            sample_count[task] = len(file_list)
            for data_file in file_list:
                file_path = os.path.join(data_path, data_file)
                query_list.append({"task": task, "file_path": file_path})
                global_idx_list[task].append(global_sample_idx)
                global_sample_idx += 1
                if self.is_train:
                    prompt_list[task].append(file_path)
        print(f'{data_split} sample count: {sample_count}')

        self.query_list = query_list
        self.global_idx_list = global_idx_list
        self.prompt_list = prompt_list
        self.task_to_flag = args.task_to_flag

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.query_list)


class MotionDatasetICL(MotionDataset):
    def __init__(self, args, data_split, prompt_list=None):
        super(MotionDatasetICL, self).__init__(args, data_split, prompt_list)
        self.clip_len = args.data.clip_len
        self.skel_amass_to_h36m = args.amass_to_h36m

        # all task inputs
        self.rootrel_input = args.rootrel_input
        # PE
        self.rootrel_target_PE = args.rootrel_target_PE
        self.flip_h36m_y_axis = args.flip_h36m_y_axis
        self.scale_h36m_skel = args.get('scale_h36m_skeleton', 1.0)
        # MP
        self.rootrel_target_MP = args.rootrel_target_MP
        # FPE
        self.rootrel_target_FPE = args.rootrel_target_FPE
        self.flip_h36mFPE_y_axis = args.flip_h36mFPE_y_axis
        self.scale_h36mFPE_skel = args.get('scale_h36mFPE_skeleton', 1.0)
        # MC
        self.drop_ratios_MC = args.data.get('drop_ratios_MC')
        self.rootrel_target_MC = args.get('rootrel_target_MC', True)

        self.num_prompts = 1

    def prepare_sample_PE(self, sample_file):
        sample = read_pkl(sample_file)
        motion_input = sample["data_input"]  # (clip_len,17,3)
        motion_target = sample["data_label"]  # (clip_len,17,3)
        if self.rootrel_input:
            motion_input = motion_input - motion_input[..., [0], :]
        if self.rootrel_target_PE:
            motion_target = motion_target - motion_target[..., [0], :]
        if self.flip_h36m_y_axis:
            motion_input[..., 1] = -motion_input[..., 1]
            motion_target[..., 1] = -motion_target[..., 1]
        if self.scale_h36m_skel != 1.0:
            motion_input = motion_input * self.scale_h36m_skel
            motion_target = motion_target * self.scale_h36m_skel
        return torch.FloatTensor(motion_input), torch.FloatTensor(motion_target)
    
    def prepare_sample_FPE(self, sample_file):
        sample = read_pkl(sample_file)
        motion_input = sample["data_input"]  # (clip_len,17,3)
        motion_target = sample["data_label"]  # (clip_len,17,3)
        if self.rootrel_input:
            motion_input = motion_input - motion_input[..., [0], :]
        if self.rootrel_target_FPE:
            motion_target = motion_target - motion_target[..., [0], :]
        if self.flip_h36mFPE_y_axis:
            motion_input[..., 1] = -motion_input[..., 1]
            motion_target[..., 1] = -motion_target[..., 1]
        if self.scale_h36mFPE_skel != 1.0:
            motion_input = motion_input * self.scale_h36m_skel
            motion_target = motion_target * self.scale_h36m_skel
        return torch.FloatTensor(motion_input), torch.FloatTensor(motion_target)

    def prepare_sample_MP(self, sample_file):
        sample = read_pkl(sample_file)
        motion_input = sample["data_input"]  # (96,18,3)
        motion_input = skel_to_h36m(motion_input, self.skel_amass_to_h36m)  # (96,17,3)
        motion_target = sample["data_label"]  # (96,18,3)
        motion_target = skel_to_h36m(motion_target, self.skel_amass_to_h36m)  # (96,17,3)
        if self.rootrel_input:
            motion_input = motion_input - motion_input[..., [0], :]
        if self.rootrel_target_MP:
            motion_target = motion_target - motion_target[..., [0], :]
        return torch.FloatTensor(motion_input), torch.FloatTensor(motion_target)

    def prepare_sample_MC(self, sample_file, is_prompt=False):
        sample = read_pkl(sample_file)
        motion_input = sample['data_input']
        motion_target = sample["data_label"]
        if self.is_train or is_prompt:
            motion_input = skel_to_h36m(motion_input, self.skel_amass_to_h36m)
            motion_target = skel_to_h36m(motion_target, self.skel_amass_to_h36m)
        return torch.FloatTensor(motion_input), torch.FloatTensor(motion_target)

    def __getitem__(self, index):
        query_sample_dict = self.query_list[index] 
        task = query_sample_dict['task']
        task_flag = self.task_to_flag[task]
        query_file = query_sample_dict['file_path']
        all_prompt_sample_dicts = self.prompt_list[task]

        if self.num_prompts == 1:
            prompt_file = random.choice(all_prompt_sample_dicts)

            if task == 'PE':
                query_input, query_target = self.prepare_sample_PE(query_file)
                prompt_input, prompt_target = self.prepare_sample_PE(prompt_file)
            elif task == 'FPE':
                query_input, query_target = self.prepare_sample_FPE(query_file)
                prompt_input, prompt_target = self.prepare_sample_FPE(prompt_file)
            elif task == 'MP':
                query_input, query_target = self.prepare_sample_MP(query_file)
                prompt_input, prompt_target = self.prepare_sample_MP(prompt_file)
            elif task == 'MC':
                query_input_, query_target = self.prepare_sample_MC(query_file)
                prompt_input, prompt_target = self.prepare_sample_MC(prompt_file, is_prompt=True)
                drop_ratio = random.choice(self.drop_ratios_MC)
                query_input, masked_joints = generate_masked_joints_seq(query_input_.clone(), drop_ratio)
                if not self.is_train:
                    query_input = query_input_  # (clip_len, 17, 3)
                    masked_joints = torch.where(query_input.sum(dim=(0,2)) == 0)[0]
                prompt_input[:, masked_joints] = 0.
            return torch.cat([prompt_input, prompt_target], dim=-3), torch.cat([query_input, query_target], dim=-3), torch.tensor(task_flag)
        
        elif self.num_prompts >= 2:
            if task == 'PE':
                query_input, query_target = self.prepare_sample_PE(query_file)
            elif task == 'FPE':
                query_input, query_target = self.prepare_sample_FPE(query_file)
            elif task == 'MP':
                query_input, query_target = self.prepare_sample_MP(query_file)
            elif task == 'MC':
                query_input_, query_target = self.prepare_sample_MC(query_file)
                drop_ratio = random.choice(self.drop_ratios_MC)
                query_input, masked_joints = generate_masked_joints_seq(query_input_.clone(), drop_ratio)
                if not self.is_train:
                    query_input = query_input_  # (clip_len, 17, 3)
                    masked_joints = torch.where(query_input.sum(dim=(0,2)) == 0)[0]

            prompt_files = [random.choice(all_prompt_sample_dicts) for _ in range(self.num_prompts)]

            prompt_inputs = []
            prompt_targets = []
            for prompt_file in prompt_files:
                if task == 'PE':
                    prompt_input, prompt_target = self.prepare_sample_PE(prompt_file)
                elif task == 'FPE':
                    prompt_input, prompt_target = self.prepare_sample_FPE(prompt_file)
                elif task == 'MP':
                    prompt_input, prompt_target = self.prepare_sample_MP(prompt_file)
                elif task == 'MC':
                    prompt_input, prompt_target = self.prepare_sample_MC(prompt_file, is_prompt=True)
                    prompt_input[:, masked_joints] = 0.
                prompt_inputs.append(prompt_input)
                prompt_targets.append(prompt_target)
            prompt_inputs = torch.stack(prompt_inputs)          # (num_prompts, clip_len, 17, 3)
            prompt_targets = torch.stack(prompt_targets)        # (num_prompts, clip_len, 17, 3)


            query_input = query_input.unsqueeze(0).expand(self.num_prompts, -1, -1, -1)         # (num_prompts, clip_len, 17, 3)
            query_target = query_target.unsqueeze(0).expand(self.num_prompts, -1, -1, -1)       # (num_prompts, clip_len, 17, 3)

            return torch.cat([prompt_inputs, prompt_targets], dim=-3), torch.cat([query_input, query_target], dim=-3), torch.tensor(task_flag).unsqueeze(0).expand(self.num_prompts)

