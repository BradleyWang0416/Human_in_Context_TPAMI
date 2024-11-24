import os
import sys
import pickle
from tqdm import tqdm
import torch
import numpy as np

sys.path.insert(0, os.getcwd())
from lib.utils.tools import read_pkl
from lib.data.datareader_h36m import DataReaderH36M
from lib.utils.utils_non_AR import vector_angle



############################# customize #############################
use_default_setting = False

use_partial_h36m = False
clip_len = 16           # default: 16
sample_stride = 1       # default: 1
train_stride = 16       # default: 16
############################# customize #############################


if use_default_setting:
    use_partial_h36m, clip_len, sample_stride, train_stride, get_context_data = False, 16, 1, 16, False
    source_h36m_path = 'source_data/H36M.pkl'
    root_path = f"data/H36M"
else:
    if not use_partial_h36m:
        source_h36m_path = 'source_data/H36M.pkl'
        root_path = 'data/non_default_ICL/H36M/ActionsAll/'
    else:
        source_h36m_path = 'source_data/H36M_Actions6.pkl'
        root_path = 'data/non_default_ICL/H36M/Actions6/'
    save_folder = f'ClipLen{clip_len}_SampleStride{sample_stride}_TrainStride{train_stride}'
    root_path = root_path + save_folder


def change_range_lim(x, new_high_offset: int, high_lim: int):
    for i in range(len(x)):
        x_list = list(x[i])
        x_low, x_high = x_list[0], x_list[-1]+1
        new_high = x_high + new_high_offset
        new_low = x_low
        if new_high > high_lim:
            new_high = high_lim
            new_low = high_lim - (x_high - x_low + new_high_offset)
        x[i] = range(new_low, new_high)
    return x


class DataReaderH36M_ICL(DataReaderH36M):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root='data/motion3d', dt_file='h36m_cpn_cam_source.pkl'):
        super().__init__(n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence, dt_root, dt_file)

    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        split_id_train, split_id_test = self.get_split_id()

        split_id_train = change_range_lim(split_id_train, new_high_offset=16, high_lim=len(train_labels))
        split_id_test = change_range_lim(split_id_test, new_high_offset=16, high_lim=len(test_labels))

        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, n_frames, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, n_frames, 17, 3)
        # ipdb.set_trace()
        return train_data, test_data, train_labels, test_labels
    
datareader = DataReaderH36M_ICL(n_frames=clip_len, sample_stride=sample_stride, data_stride_train=train_stride, data_stride_test=clip_len, dt_file = source_h36m_path, dt_root='data')

train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()


print(train_data.shape, test_data.shape)
print(f'Training sample count: {len(train_data)}')
print(f'Testing sample count: {len(test_data)}')
assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

train_data[:, :, :, -1] = 0
test_data[:, :, :, -1] = 0




ANGLES = {
    'train': [],
    'test': []
}

def save_clips(subset_name, root_path, train_data, train_labels):
    len_train = len(train_data)
    root_path = os.path.join(root_path, subset_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {
            "chunk_2d": data_input,
            "chunk_3d": data_label
        }
        
        angle = vector_angle(np.array([1.,0.]), data_label[0, 1, [0,2]]-data_label[0, 4, [0,2]])
        ANGLES[subset_name].append(angle)

        
        # with open(os.path.join(root_path, "%08d.pkl" % i), "wb") as myprofile:  
        #     pickle.dump(data_dict, myprofile)

if not os.path.exists(root_path):
    os.makedirs(root_path)
save_clips("train", root_path, train_data, train_labels)
# save_clips("test", root_path, test_data, test_labels)


import matplotlib.pyplot as plt
# plt.hist(ANGLES['train'], bins=100)
plt.bar(range(len(ANGLES['train'])), ANGLES['train'])
plt.show()
