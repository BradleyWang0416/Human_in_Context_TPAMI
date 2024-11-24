import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)


class Dataset_ActionRecognition(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, prompt_data=None, prompt_label=None):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        np.random.seed(0)
        if split == 'train': assert prompt_data is None
        if split == 'test':  assert prompt_data is not None
        
        if split == 'test':
            self.prompt_data = prompt_data      # equal to Feeder(split='train').data
            self.prompt_label = prompt_label    # equal to Feeder(split='train').label

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()
        self.find_indices()

    def find_indices(self):
        self.indices1 = np.where((self.data[..., -1] == 0).all(-1).all(-1).all(-1) == True)[0]
        self.indices2 = np.setdiff1d(np.arange(len(self.label)), self.indices1)
        if self.split == 'test':
            self.prompt_indices1 = np.where((self.prompt_data[..., -1] == 0).all(-1).all(-1).all(-1) == True)[0]
            self.prompt_indices2 = np.setdiff1d(np.arange(len(self.prompt_label)), self.prompt_indices1)

    def load_data(self):
        # data: N C V T M
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def get_one_sample(self, index, is_prompt=False):
        if self.split == 'test' and is_prompt:
            data_numpy = self.prompt_data[index]
            label = self.prompt_label[index]
        else:
            data_numpy = self.data[index]   # (3,300,25,2)
            label = self.label[index]       # 32
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)    # [TRAIN] 57; [TEST] 110
        # reshape Tx(MVC) to CTVM
        data_numpy = valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)    # (3,64,25,2)

        if self.random_rot: # [TRAIN] False; [TEST] False
            data_numpy = random_rot(data_numpy)
        if self.bone:   # False
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:    # False
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        if self.split == 'test' and is_prompt:
            if index in self.prompt_indices1:
                data_numpy = data_numpy[..., [0]]  # (3,64,25,1)
        else:
            if index in self.indices1:
                data_numpy = data_numpy[..., [0]]  # (3,64,25,1)

        if not isinstance(data_numpy, torch.FloatTensor):
            data_numpy = torch.FloatTensor(data_numpy)

        return data_numpy.permute(3,1,2,0), label, index     # (2,64,25,3); int; int

    def __getitem__(self, index):
        query_input, query_label, query_index = self.get_one_sample(index)  # (2,64,25,3) or (2,64,25,3)
        if self.split == 'test':
            if query_input.shape[0] == 1:
                prompt_index = np.random.choice(self.prompt_indices1)
            else:
                prompt_index = np.random.choice(self.prompt_indices2)
        else:
            if query_input.shape[0] == 1:
                prompt_index = np.random.choice(self.indices1)
            else:
                prompt_index = np.random.choice(self.indices2)
        prompt_input, prompt_label, prompt_index = self.get_one_sample(prompt_index, is_prompt=True)   # (2,64,25,3) or (2,64,25,3)
        assert query_input.shape == prompt_input.shape
        return query_input, query_label, prompt_input, prompt_label


def collate_func(batch):
    QUERY_INPUT_1 = []
    QUERY_INPUT_2 = []
    QUERY_LABEL_1 = []
    QUERY_LABEL_2 = []
    PROMPT_INPUT_1 = []
    PROMPT_INPUT_2 = []
    PROMPT_LABEL_1 = []
    PROMPT_LABEL_2 = []
    for b in range(len(batch)):
        query_input, query_label, prompt_input, prompt_label = batch[b]
        # query_input: (2，64,25,3) or (1,64,25,3)
        # prompt_input: (2,64,25,3) or (1,64,25,3)
        M = query_input.shape[0]
        if M == 1:
            QUERY_INPUT_1.append(query_input)
            PROMPT_INPUT_1.append(prompt_input)
            QUERY_LABEL_1.append(query_label)
            PROMPT_LABEL_1.append(prompt_label)
        else:
            QUERY_INPUT_2.append(query_input)
            PROMPT_INPUT_2.append(prompt_input)
            QUERY_LABEL_2.append(query_label)
            PROMPT_LABEL_2.append(prompt_label)

    if len(QUERY_INPUT_1) != 0:
        QUERY_INPUT_1 = torch.stack(QUERY_INPUT_1)     # (B1,1,T,J,C)
        QUERY_LABEL_1 = torch.tensor(QUERY_LABEL_1, dtype=torch.int)     # (B1)
        PROMPT_INPUT_1 = torch.stack(PROMPT_INPUT_1)   # (B1,1,T,J,C)
        PROMPT_LABEL_1 = torch.tensor(PROMPT_LABEL_1, dtype=torch.int)   # (B1)

    if len(QUERY_INPUT_2) != 0:
        QUERY_INPUT_2 = torch.stack(QUERY_INPUT_2)     # (B2,2,T,J,C)
        QUERY_LABEL_2 = torch.tensor(QUERY_LABEL_2, dtype=torch.int)     # (B2)
        PROMPT_INPUT_2 = torch.stack(PROMPT_INPUT_2)   # (B2,2,T,J,C)
        PROMPT_LABEL_2 = torch.tensor(PROMPT_LABEL_2, dtype=torch.int)   # (B2)

    return QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1, \
           QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2     


def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape   # (3,300,25,2)
    begin = 0
    end = valid_frame_num   # [TRAIN] 57; [TEST] 110
    valid_size = end - begin    # [TRAIN] 57; [TEST] 110

    #crop
    if len(p_interval) == 1:
        p = p_interval[0]   # 0.95
        bias = int((1-p) * valid_size/2)    # 2
        data = data_numpy[:, begin+bias:end-bias, :, :] # center_crop   # (3,106,25,2)
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]   # 0.77440675
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)     # constraint cropped_length lower bound as 64
        bias = np.random.randint(0,valid_size-cropped_length+1)
        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data,dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()

    return data


def random_rot(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot
