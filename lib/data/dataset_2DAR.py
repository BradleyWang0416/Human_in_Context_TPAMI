import torch
import numpy as np
import os
import sys
import random
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /home/pami_ICL/Skeleton-in-Context-tpami/lib/data/
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))   # /home/pami_ICL/Skeleton-in-Context-tpami/
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))

from lib.utils.tools import read_pkl
from lib.utils.viz_skel_seq import viz_skel_seq_anim


class ActionRecognitionDataset(Dataset):
    def __init__(self, args, data_split, prompt_list=None):   # data_split: 'train' or 'test'
        if data_split == 'train':
            assert prompt_list is None
        if data_split == 'test':
            assert prompt_list is not None
        np.random.seed(0)
        random.seed(0)
        self.data_split = data_split    # 'train' or 'test'
        self.is_train = (True if data_split == 'train' else False)
        query_list = []
        sample_count = {}

        ########################################################################################
        tasks = ['2DAR']
        ########################################################################################

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


class ActionRecognitionDataset2D(ActionRecognitionDataset):
    def __init__(self, args, data_split, prompt_list=None):
        super(ActionRecognitionDataset2D, self).__init__(args, data_split, prompt_list)
        self.clip_len = args.data.clip_len
        self.clip_len_2DAR = args.data.clip_len_2DAR
        assert self.clip_len_2DAR < self.clip_len * 2
        self.label_type_2DAR = args.get('label_type_2DAR', 'default')
        self.rootrel_input_2DAR = args.get('rootrel_input_2DAR', True)

    def prepare_sample_2DAR(self, sample_file):
        sample = read_pkl(sample_file)
        motion = sample["data_input"]  # (1 or 2, T, 17, 3), 每个样本 T 大小不一定相同
        label = sample["data_label"]  # 一个属于0到59闭区间的整数
        return motion, label
    
    def __getitem__(self, index):
        query_sample_dict = self.query_list[index] 
        task = query_sample_dict['task']
        task_flag = self.task_to_flag[task]     # {'PE': 0, 'MP': 1, 'FPE': 2, 'MC': 3, '2DAR': 4}
        query_file = query_sample_dict['file_path']
        query_motion, query_label = self.prepare_sample_2DAR(query_file)    # query_motion: (1 or 2, T_q, 17, 3), 不同样本 T 不一定相同. query_label: 一个属于0到59闭区间的整数
        M_q, T_q, J, C = query_motion.shape
        # motion[..., :3] 存储2D骨架数据, motion[..., 3] 存储keypoint_score.

        all_prompt_sample_dicts = self.prompt_list[task]
        while True:
            prompt_file = random.choice(all_prompt_sample_dicts)
            prompt_motion, prompt_label = self.prepare_sample_2DAR(prompt_file) # prompt_motion: (1 or 2, T_p, 17, 3), 不同样本 T 不一定相同. prompt_label: 一个属于0到59闭区间的整数
            M_p, T_p = prompt_motion.shape[:2]
            if M_p == M_q:
                break
        
        # align joints to joint #0 of 1st person
        if self.rootrel_input_2DAR:
            query_motion = query_motion - query_motion[0, :, 0, :].reshape(1, T_q, 1, C)
            prompt_motion = prompt_motion - prompt_motion[0, :, 0, :].reshape(1, T_p, 1, C)

        # make a fake zero person
        if M_q == 1:
            query_fake = np.zeros(query_motion.shape)
            query_motion = np.concatenate((query_motion, query_fake), axis=0)     # (2, T_q, 17, 3)
        if M_p == 1:
            prompt_fake = np.zeros(prompt_motion.shape)
            prompt_motion = np.concatenate((prompt_motion, prompt_fake), axis=0)  # (2, T_p, 17, 3)

        # slice subsamples
        query_input = []
        for frame_st in range(0, T_q - self.clip_len_2DAR + 1, 8):
            frame_indices = np.arange(frame_st, frame_st + self.clip_len_2DAR)
            query_input.append(query_motion[:, frame_indices])
        query_input = np.stack(query_input)     # (N_q, 2, clip_len_2DAR, 17, 3)
        N_q = query_input.shape[0]
        prompt_input = []
        for frame_st in range(0, T_p - self.clip_len_2DAR + 1, 8):
            frame_indices = np.arange(frame_st, frame_st + self.clip_len_2DAR)
            prompt_input.append(prompt_motion[:, frame_indices])
        prompt_input = np.stack(prompt_input)     # (N_p, 2, clip_len_2DAR, 17, 3)
        N_p = prompt_input.shape[0]

        # process label
        label_len = self.clip_len * 2 - self.clip_len_2DAR         # (2, label_len, 17, 3) elements available for storing class label
        query_target = get_AR_labels(self.label_type_2DAR)[query_label]         # (17,3)
        prompt_target = get_AR_labels(self.label_type_2DAR)[prompt_label]       # (17,3)
        query_target = query_target.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N_q, 2, label_len, -1, -1)      # (N_q, 2, label_len, 17, 3)
        prompt_target = prompt_target.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(N_p, 2, label_len, -1, -1)    # (N_p, 2, label_len, 17, 3)

        return query_input, query_target, prompt_input, prompt_target, task_flag
        # (N_q, 2, clip_len_2DAR, 17, 3), (N_q, 2, label_len, 17, 3),
        # (N_p, 2, clip_len_2DAR, 17, 3), (N_p, 2, label_len, 17, 3),
        # int


def collate_fn_2DAR (batch):
    # batch: a list of tuples. len=batch_size. list element is tuple: (query_input, query_label, prompt_input, prompt_label, task_flag), where
    #       query_input: (N_q, 2, clip_len_2DAR, 17, 3). 注意每个样本的num_subsample_q可能不同
    #       query_target: (N_q, 2, label_len, 17, 3)
    #       prompt_input: (N_p ,2, clip_len_2DAR, 17, 3). 注意每个样本的num_subsample_p可能不同
    #       prompt_target: (N_p, 2, label_len, 17, 3)
    #       task_flag=4
    qi = []
    pi = []
    qt = []
    pt = []
    query_slice_indices = []
    for b in range(len(batch)):
        query_input, query_target, prompt_input, prompt_target, task_flag = batch[b]
        N_q = query_input.shape[0]
        query_slice_indices.append(np.array([b, b + N_q]))
        qi.append(query_input)
        pi.append(prompt_input)
        qt.append(query_target)
        pt.append(prompt_target)
    qi_batched = np.concatenate(qi)      # (B_q, 2, clip_len_2DAR, 17, 3)
    qi_batched = torch.FloatTensor(qi_batched)
    pi_batched = np.concatenate(pi)      # (B_p, 2, clip_len_2DAR, 17, 3)
    pi_batched = torch.FloatTensor(pi_batched)
    qt_batched = torch.cat(qt)      # (B_q, 2, label_len, 17, 3)
    pt_batched = torch.cat(pt)      # (B_p, 2, label_len, 17, 3)
    query_slice_indices = np.stack(query_slice_indices)     # (batch_size, 2)

    B_q = qi_batched.shape[0]
    B_p = pi_batched.shape[0]
    if B_q > B_p:
        pi_batched = pi_batched.repeat(B_q//B_p, 1, 1, 1, 1)
        pt_batched = pt_batched.repeat(B_q//B_p, 1, 1, 1, 1)
        pi_batched = torch.cat([pi_batched, pi_batched[:B_q%B_p]], dim=0)
        pt_batched = torch.cat([pt_batched, pt_batched[:B_q%B_p]], dim=0)
    else:
        pi_batched = pi_batched[:B_q]
        pt_batched = pt_batched[:B_q]

    return qi_batched, pi_batched, qt_batched, pt_batched, query_slice_indices


def get_AR_labels(ver):
    if ver == 'default':
        labels_0to48 = torch.eye(51)[:49].reshape(49, 17, 3)            # (49,17,3)
        labels_49to59 = []
        x_coors = np.linspace(-1, 1, 3)
        y_coors = np.linspace(-1, 1, 3)
        z_coors = np.linspace(-1, 1, 3)
        for x in x_coors:
            for y in y_coors:
                for z in z_coors:
                    labels_49to59.append(torch.tensor([x, y, z]))
        labels_49to59 = torch.stack(labels_49to59[:11])      # (11, 3)
        labels_49to59 = labels_49to59.unsqueeze(1).expand(-1, 17, -1)       # (11, 17, 3)
        labels = torch.cat([labels_0to48, labels_49to59], dim=0)    # (60,17,3)
    elif ver == 'cube_xyz':
        labels = []
        x_coors = np.linspace(-1, 1, 4)
        y_coors = np.linspace(-1, 1, 4)
        z_coors = np.linspace(-1, 1, 4)
        for x in x_coors:
            for y in y_coors:
                for z in z_coors:
                    labels.append(torch.tensor([x, y, z]))
        labels = torch.stack(labels[:60]).unsqueeze(1).expand(-1, 17, -1)    # (60, 17, 3)
    else:
        raise NotImplementedError
    return labels       # (60,17,3)


def resample(ori_len, target_len, replay=False, randomness=True):
    if replay:
        if ori_len > target_len:
            st = np.random.randint(ori_len-target_len)
            return range(st, st+target_len)  # Random clipping from sequence
        else:
            return np.array(range(target_len)) % ori_len  # Replay padding
    else:
        if randomness:
            even = np.linspace(0, ori_len, num=target_len, endpoint=False)
            if ori_len < target_len:
                low = np.floor(even)
                high = np.ceil(even)
                sel = np.random.randint(2, size=even.shape)
                result = np.sort(sel*low+(1-sel)*high)
            else:
                interval = even[1] - even[0]
                result = np.random.random(even.shape)*interval + even
            result = np.clip(result, a_min=0, a_max=ori_len-1).astype(np.uint32)
        else:
            result = np.linspace(0, ori_len, num=target_len, endpoint=False, dtype=int)
        return result


if __name__ == "__main__":
        for i in range(1000):
            sample_file = f'data/NTU60_2DAR/train/{i:08d}.pkl'
            sample = read_pkl(sample_file)
            motion = sample["data_input"]  # (1 or 2, T, 17, 3), 每个样本 T 大小不一定相同
            seq = motion[0]

            viz_skel_seq_anim(seq, if_print=True, fig_title=sample_file[-12:-4], file_name=sample_file[-12:-4], file_folder='tmp')
