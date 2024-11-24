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
from lib.utils.utils_non_AR import skel_to_h36m, generate_masked_joints_seq, rotate_y, unify_skeletons, vector_angle, get_complementary_idx


class MotionDataset(Dataset):
    def __init__(self, args, data_split, prompt_list=None, TASK=None, DATASET=None):   # data_split: 'train' or 'test'
        if data_split == 'train':
            assert prompt_list is None and TASK is None and DATASET is None
        if data_split == 'test':
            assert prompt_list is not None and TASK is not None and DATASET is not None
        np.random.seed(0)
        random.seed(0)

        self.data_split = data_split    # 'train' or 'test'
        self.is_train = (True if data_split == 'train' else False)
        
        if self.is_train:
            self.datasets = args.datasets   # {'H36M': <PATH_TO_DATASET>, 'PW3D': <PATH_TO_DATASET>, 'AMASS': <PATH_TO_DATASET>}
            self.tasks = args.tasks         # ['PE', 'MP', 'MC', 'FPE']
            self.dataset_task_info = args.dataset_task_info     
            # { 'PE': ['H36M'], 
            #   'MP': ['AMASS'], 
            #   'MC': ['PW3D'], 
            #   'FPE': ['H36M'] 
            # }
        else:
            self.datasets = {DATASET: args.datasets[DATASET]}
            self.tasks = [TASK]

        
        query_list = []
        sample_count = {}
        global_idx_list = {dataset: [] for dataset in self.datasets}
        if 'MC' in self.tasks:
            joint_mask_idx = []
        if 'MIB' in self.tasks:
            frame_mask_idx = []

        if self.is_train:
            prompt_list = {dataset: [] for dataset in self.datasets}
        global_sample_idx = 0
        for dataset, folder in self.datasets.items():
            data_path = os.path.join(args.data_root_path, folder, data_split)       # args.data_root_path = 'data/non_default_ICL'; folder = 'PW3D/TrainShift1_TestShift4_DropRatios46_ClipLen16'
            file_list = sorted(os.listdir(data_path))
            sample_count[dataset] = len(file_list)
            for data_file in file_list:
                file_path = os.path.join(data_path, data_file)
                query_list.append({"dataset": dataset, "file_path": file_path})
                global_idx_list[dataset].append(global_sample_idx)
                if 'MC' in self.tasks:
                    joint_mask_idx.append(random.sample(range(1,17), int(0.5*17)))
                if 'MIB' in self.tasks:
                    frame_mask_idx.append(random.sample(range(1,16), int(0.5*16)))
                global_sample_idx += 1
                if self.is_train:
                    prompt_list[dataset].append(file_path)
        print(f'{data_split} sample count: {sample_count}')

        
        self.query_list = query_list
        self.global_idx_list = global_idx_list
        if 'MC' in self.tasks:
            self.joint_mask_idx = joint_mask_idx
        if 'MIB' in self.tasks:
            self.frame_mask_idx = frame_mask_idx
        self.prompt_list = prompt_list
        self.task_to_flag = args.task_to_flag
        self.dataset_to_flag = args.dataset_to_flag
        self.current_as_history = args.current_as_history
        
        self.num_prompts = 1
        self.aug = args.get('aug', False)
        self.rotate_prob = args.get('rotate_prob', 0.25)
        self.keep_original_joints = args.get('keep_original_joints', False)
        assert self.keep_original_joints ^ (args.data.num_joints == 17)


    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.query_list)


class MotionDatasetICL(MotionDataset):
    def __init__(self, args, data_split, prompt_list=None, TASK=None, DATASET=None):
        super(MotionDatasetICL, self).__init__(args, data_split, prompt_list, TASK, DATASET)
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
    
    def prepare_sample(self, sample_file_path, task, index, dataset, non_h36m_rotate_angle=None):
        sample = read_pkl(sample_file_path)
        if task == 'PE':
            chunk_3d = sample['chunk_3d']
            chunk_3d = torch.FloatTensor(chunk_3d)

            if dataset in ['AMASS', 'PW3D']:

                # direction_vector_previous = (chunk_3d[0, 1, [0,2]] - chunk_3d[0, 5, [0,2]]).clone()     # 在xOz平面上的向量
                # non_h36m_rotate_angle = 220
                chunk_3d = rotate_y(chunk_3d, non_h36m_rotate_angle)
                # direction_vector_now = (chunk_3d[0, 1, [0,2]] - chunk_3d[0, 5, [0,2]]).clone()
                # angle = vector_angle(direction_vector_previous, direction_vector_now)                   # 在xOz平面上的向量

                chunk_2d = chunk_3d.clone()
                chunk_2d[..., -1] = 0
                if not self.keep_original_joints:
                    chunk_3d = skel_to_h36m(chunk_3d, self.skel_amass_to_h36m)
                    chunk_2d = skel_to_h36m(chunk_2d, self.skel_amass_to_h36m)
            elif dataset == 'H36M':
                chunk_2d = sample['chunk_2d']
                chunk_2d = torch.FloatTensor(chunk_2d)
                chunk_3d = chunk_3d * 2
                chunk_2d = chunk_2d * 2
                chunk_3d[..., 1] = -chunk_3d[..., 1]
                chunk_2d[..., 1] = -chunk_2d[..., 1]
            
            if self.keep_original_joints:
                chunk_3d = unify_skeletons(chunk_3d, dataset=dataset, pad='copy')
                chunk_2d = unify_skeletons(chunk_2d, dataset=dataset, pad='copy')

            if self.current_as_history:
                motion_input = chunk_2d[..., :self.clip_len, :, :]
                motion_output = chunk_3d[..., :self.clip_len, :, :]
            else:
                motion_input = chunk_2d[..., self.clip_len:, :, :]
                motion_output = chunk_3d[..., self.clip_len:, :, :]
        elif task == 'FPE':
            chunk_3d = sample['chunk_3d']
            chunk_3d = torch.FloatTensor(chunk_3d)
            
            if dataset in ['AMASS', 'PW3D']:
                chunk_3d = rotate_y(chunk_3d, non_h36m_rotate_angle)

                chunk_2d = chunk_3d.clone()
                chunk_2d[..., -1] = 0
                if not self.keep_original_joints:
                    chunk_3d = skel_to_h36m(chunk_3d, self.skel_amass_to_h36m)
                    chunk_2d = skel_to_h36m(chunk_2d, self.skel_amass_to_h36m)
            elif dataset == 'H36M':
                chunk_2d = sample['chunk_2d']
                chunk_2d = torch.FloatTensor(chunk_2d)
                chunk_3d = chunk_3d * 2
                chunk_2d = chunk_2d * 2
                chunk_3d[..., 1] = -chunk_3d[..., 1]
                chunk_2d[..., 1] = -chunk_2d[..., 1]
            
            if self.keep_original_joints:
                chunk_3d = unify_skeletons(chunk_3d, dataset=dataset, pad='copy')
                chunk_2d = unify_skeletons(chunk_2d, dataset=dataset, pad='copy')

            motion_input = chunk_2d[..., :self.clip_len, :, :]
            motion_output = chunk_3d[..., self.clip_len:, :, :]
        elif task == 'MP':
            chunk_3d = sample['chunk_3d']
            if dataset in ['AMASS', 'PW3D']:
                chunk_3d = rotate_y(chunk_3d, non_h36m_rotate_angle)

                if not self.keep_original_joints:
                    chunk_3d = skel_to_h36m(chunk_3d, self.skel_amass_to_h36m)
            if dataset == 'H36M':
                chunk_3d = chunk_3d * 2
                chunk_3d[..., 1] = -chunk_3d[..., 1]
            
            if self.keep_original_joints:
                chunk_3d = unify_skeletons(chunk_3d, dataset=dataset, pad='copy')

            chunk_3d = torch.FloatTensor(chunk_3d)
            motion_input = chunk_3d[..., :self.clip_len, :, :]
            motion_output = chunk_3d[..., self.clip_len:, :, :]
        elif task == 'MC':
            joint_mask_idx = self.joint_mask_idx[index]
            chunk_3d = sample['chunk_3d']
            if dataset in ['AMASS', 'PW3D']:
                chunk_3d = rotate_y(chunk_3d, non_h36m_rotate_angle)

                if not self.keep_original_joints:
                    chunk_3d = skel_to_h36m(chunk_3d, self.skel_amass_to_h36m)
            if dataset == 'H36M':
                chunk_3d = chunk_3d * 2
                chunk_3d[..., 1] = -chunk_3d[..., 1]
            
            chunk_3d = torch.FloatTensor(chunk_3d)
            masked_chunk_3d = chunk_3d.clone()
            masked_chunk_3d[..., joint_mask_idx, :] = 0

            if self.keep_original_joints:
                chunk_3d = unify_skeletons(chunk_3d, dataset=dataset, pad='copy')
                masked_chunk_3d = unify_skeletons(masked_chunk_3d, dataset=dataset, pad='copy')
            
            if self.current_as_history:
                motion_input = masked_chunk_3d[..., :self.clip_len, :, :]
                motion_output = chunk_3d[..., :self.clip_len, :, :]
            else:
                motion_input = masked_chunk_3d[..., self.clip_len:, :, :]
                motion_output = chunk_3d[..., self.clip_len:, :, :]
        elif task == 'MIB':
            frame_mask_idx = self.frame_mask_idx[index]
            chunk_3d = sample['chunk_3d']
            if dataset in ['AMASS', 'PW3D']:
                chunk_3d = rotate_y(chunk_3d, non_h36m_rotate_angle)

                if not self.keep_original_joints:
                    chunk_3d = skel_to_h36m(chunk_3d, self.skel_amass_to_h36m)
            if dataset == 'H36M':
                chunk_3d = chunk_3d * 2
                chunk_3d[..., 1] = - chunk_3d[..., 1]
            
            if self.keep_original_joints:
                chunk_3d = unify_skeletons(chunk_3d, dataset=dataset, pad='copy')

            chunk_3d = torch.FloatTensor(chunk_3d)
            masked_chunk_3d = chunk_3d.clone()
            # masked_chunk_3d[..., frame_mask_idx, :, :] = 0
            dup_idx = get_complementary_idx(torch.tensor(frame_mask_idx), max_idx=self.clip_len)
            masked_chunk_3d = masked_chunk_3d[dup_idx]
            if self.current_as_history:
                motion_input = masked_chunk_3d[..., :self.clip_len, :, :]
                motion_output = chunk_3d[..., :self.clip_len, :, :]
            else:
                motion_input = masked_chunk_3d[..., self.clip_len:, :, :]
                motion_output = chunk_3d[..., self.clip_len:, :, :]

        # align root joint to the origin
        motion_input = motion_input - motion_input[..., [0], :]
        motion_output = motion_output - motion_output[..., [0], :]

        if (not self.is_train) and task == 'MC':
            return motion_input, motion_output, joint_mask_idx
        if (not self.is_train) and task == 'MIB':
            return motion_input, motion_output, frame_mask_idx
        return motion_input, motion_output

    def __getitem__(self, query_index):
        query_sample_dict = self.query_list[query_index]

        dataset, query_file_path = query_sample_dict['dataset'], query_sample_dict['file_path']

        all_prompt_sample_dicts = self.prompt_list[dataset]

        prompt_file_path = random.choice(all_prompt_sample_dicts)

        
        if self.is_train:
            tasks = self.dataset_task_info.train[dataset]

            rotate_angle = None
            if self.is_train:
                if self.aug:
                    if random.random() < self.rotate_prob:
                        rotate_angle = random.uniform(0, 180)
                

            QUERY_INPUT = []
            QUERY_OUTPUT = []
            PROMPT_INPUT = []
            PROMPT_OUTPUT = []
            INFO = []
            QUERY_INDEX = []
            for task in tasks:
                query_input, query_output = self.prepare_sample(query_file_path, task, query_index, dataset, non_h36m_rotate_angle=rotate_angle)
                prompt_input, prompt_output = self.prepare_sample(prompt_file_path, task, query_index, dataset, non_h36m_rotate_angle=rotate_angle)
                QUERY_INPUT.append(query_input)
                QUERY_OUTPUT.append(query_output)
                PROMPT_INPUT.append(prompt_input)
                PROMPT_OUTPUT.append(prompt_output)
                INFO.append(torch.tensor([self.dataset_to_flag[dataset], self.task_to_flag[task]]))
                QUERY_INDEX.append(torch.tensor(query_index))
            QUERY_INPUT = torch.stack(QUERY_INPUT)
            QUERY_OUTPUT = torch.stack(QUERY_OUTPUT)
            PROMPT_INPUT = torch.stack(PROMPT_INPUT)
            PROMPT_OUTPUT = torch.stack(PROMPT_OUTPUT)
            INFO = torch.stack(INFO)
            QUERY_INDEX = torch.stack(QUERY_INDEX)


            if self.num_prompts == 1:
                return torch.cat([PROMPT_INPUT, PROMPT_OUTPUT], dim=-3), torch.cat([QUERY_INPUT, QUERY_OUTPUT], dim=-3), INFO, QUERY_INDEX
            
        else:
            task = self.tasks[0]
            task_flag = self.task_to_flag[task]
            rotate_angle = None
            
            if task == 'MIB':
                QUERY_INPUT, QUERY_OUTPUT, frame_mask_idx = self.prepare_sample(query_file_path, task, query_index, dataset, non_h36m_rotate_angle=rotate_angle)
                PROMPT_INPUT, PROMPT_OUTPUT, _ = self.prepare_sample(prompt_file_path, task, query_index, dataset, non_h36m_rotate_angle=rotate_angle)
            elif task == 'MC':
                QUERY_INPUT, QUERY_OUTPUT, joint_mask_idx = self.prepare_sample(query_file_path, task, query_index, dataset, non_h36m_rotate_angle=rotate_angle)
                PROMPT_INPUT, PROMPT_OUTPUT, _ = self.prepare_sample(prompt_file_path, task, query_index, dataset, non_h36m_rotate_angle=rotate_angle)
            else:
                QUERY_INPUT, QUERY_OUTPUT = self.prepare_sample(query_file_path, task, query_index, dataset, non_h36m_rotate_angle=rotate_angle)
                PROMPT_INPUT, PROMPT_OUTPUT = self.prepare_sample(prompt_file_path, task, query_index, dataset, non_h36m_rotate_angle=rotate_angle)

            if task == 'MIB':
                return torch.cat([PROMPT_INPUT, PROMPT_OUTPUT], dim=-3), torch.cat([QUERY_INPUT, QUERY_OUTPUT], dim=-3), torch.tensor(task_flag), torch.tensor(frame_mask_idx)
            elif task == 'MC':
                return torch.cat([PROMPT_INPUT, PROMPT_OUTPUT], dim=-3), torch.cat([QUERY_INPUT, QUERY_OUTPUT], dim=-3), torch.tensor(task_flag), torch.tensor(joint_mask_idx)
            else:
                return torch.cat([PROMPT_INPUT, PROMPT_OUTPUT], dim=-3), torch.cat([QUERY_INPUT, QUERY_OUTPUT], dim=-3), torch.tensor(task_flag)

