import os
import sys
import pickle
sys.path.insert(0, os.getcwd())
from lib.data.datareader_h36m import DataReaderH36M
from tqdm import tqdm


############################# customize #############################
use_default_setting = True

use_partial_h36m = True
clip_len = 32           # default: 32
clip_split = 16         # default: 16
sample_stride = 1       # default: 1
train_stride = 16       # default: 16
############################# customize #############################

if use_default_setting:
    use_partial_h36m, clip_len, clip_split, sample_stride, train_stride = False, 32, 16, 1, 16
    source_h36m_path = 'source_data/H36M.pkl'
    root_path = f"data/H36M_FPE"
else:
    if not use_partial_h36m:
        source_h36m_path = 'source_data/H36M.pkl'
        root_path = 'data/non_default/H36M_FPE/ActionsAll/'
    else:
        source_h36m_path = 'source_data/H36M_Actions6.pkl'
        root_path = 'data/non_default/H36M_FPE/Actions6/'
    save_folder = f'ClipLen{clip_len}_In{clip_split}Out{clip_split}_SampleStride{sample_stride}_TrainStride{train_stride}'
    root_path = root_path + save_folder


datareader = DataReaderH36M(n_frames=clip_len, sample_stride=sample_stride, data_stride_train=train_stride, data_stride_test=clip_len, dt_file = source_h36m_path, dt_root='data')
train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()

train_data[:, :, :, -1] = 0
test_data[:, :, :, -1] = 0

train_data = train_data[:, :clip_split]
train_labels = train_labels[:, clip_split:]
test_data = test_data[:, :clip_split]
test_labels = test_labels[:, clip_split:]

print(train_data.shape, test_data.shape)
print(f'Training sample count: {len(train_data)}')
print(f'Testing sample count: {len(test_data)}')
assert len(train_data) == len(train_labels)
assert len(test_data) == len(test_labels)

def save_clips(subset_name, root_path, train_data, train_labels):
    len_train = len(train_data)
    save_path = os.path.join(root_path, subset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(len_train)):
        data_input, data_label = train_data[i], train_labels[i]
        data_dict = {
            "data_input": data_input,
            "data_label": data_label
        }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:  
            pickle.dump(data_dict, myprofile)

if not os.path.exists(root_path):
    os.makedirs(root_path)
save_clips("train", root_path, train_data, train_labels)
save_clips("test", root_path, test_data, test_labels)