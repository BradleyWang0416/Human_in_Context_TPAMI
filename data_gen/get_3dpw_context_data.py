import os
import sys
import glob
import pickle
import random
import numpy as np
import pickle as pkl
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import torch
import torch.utils.data as data

from angle2joint import ang2joint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
from lib.utils.utils_non_AR import skel_to_h36m, generate_masked_joints_seq


############################# customize #############################
use_default_setting = False

input_length = 16           # default: 16
train_shift = 1             # default: 1
test_shift = 4              # default: 4
drop_ratios = [0.4,0.6]     # default: [0.4, 0.6]
JOINT_NUMBER = 22               # default: 18
############################# customize #############################

if use_default_setting:
    input_length, train_shift, test_shift, drop_ratios, JOINT_NUMBER = 16, 1, 4, [0.4, 0.6], 18
    save_root_path = "data/3DPW/"
else:
    save_root_path = "data/non_default_ICL/3DPW/"
    save_folder = f'TrainShift{train_shift}_TestShift{test_shift}_DropRatios{int(drop_ratios[0]*10)}{int(drop_ratios[1]*10)}_ClipLen{input_length}_J{JOINT_NUMBER}'
    save_root_path = save_root_path + save_folder

pw3d_anno_dir = 'data/source_data/PW3D/sequenceFiles/'
pw3d_support_dir = 'data/support_data/'
target_length = 16

def main():
    random.seed(0)

# ----------------------------------------------------------------------------- train -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------- train -----------------------------------------------------------------------------
    dataset_train = PW3D(pw3d_anno_dir=pw3d_anno_dir, pw3d_support_dir=pw3d_support_dir, input_length=input_length, target_length=target_length, split_name='train', shift_step=train_shift, JOINT_NUMBER=JOINT_NUMBER)
    print(f'Train sample count: {len(dataset_train)}')

    save_path = os.path.join(save_root_path, 'train')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(dataset_train)):
        chunk = dataset_train[i%len(dataset_train)]       # (16,18,3)
        data_dict_test = {
            "chunk_3d": chunk.data.numpy()
        }
        # train input and label are identical here; inputs will be masked later during data loading when training
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:
            pickle.dump(data_dict_test, myprofile)



# ----------------------------------------------------------------------------- test -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------- test -----------------------------------------------------------------------------
    dataset_test = PW3D(pw3d_anno_dir=pw3d_anno_dir, pw3d_support_dir=pw3d_support_dir, input_length=input_length, target_length=target_length, split_name='test', shift_step=test_shift, JOINT_NUMBER=JOINT_NUMBER)
    print(f'Test sample count: {len(dataset_test)}')
    
    save_path = os.path.join(save_root_path, 'test')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_total = len(dataset_test)
    drop_ratios_all = []
    for drop_ratio in drop_ratios:
        drop_ratios_all += [drop_ratio] * (test_total//len(drop_ratios))
    if len(drop_ratios_all) != test_total:
        drop_ratios_all += [drop_ratios[-1]] * (test_total-len(drop_ratios_all))
    assert len(drop_ratios_all) == test_total

    test_list = list(range(0, test_total))
    random.shuffle(test_list)
    for i, test_idx in enumerate(test_list):
        drop_ratio = drop_ratios_all[i]

        sample_test = dataset_test[test_idx]       # (16,18,3)
        if JOINT_NUMBER == 18:
            sample_test = skel_to_h36m(sample_test, [[2], [0], [3], [6], [1], [4], [7], [5], [8], [8,11], [11], [10], [13], [15], [9], [12], [14]])
            input_test, masked_joints = generate_masked_joints_seq(sample_test.clone(), drop_ratio)
            data_dict_test = {
                "masked_chunk_3d": input_test,
                "chunk_3d": sample_test,
                "masked_joint_idx": masked_joints
            }
        elif JOINT_NUMBER == 22:
            _, masked_joints = generate_masked_joints_seq(sample_test.clone(), drop_ratio)
            data_dict_test = {
                "chunk_3d": sample_test,
                "masked_joint_idx": masked_joints
            }
        with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:
            pickle.dump(data_dict_test, myprofile)


class PW3D(data.Dataset):
    def __init__(self, pw3d_anno_dir, pw3d_support_dir, input_length, target_length, split_name, shift_step, frame_interval=0, paired=True, JOINT_NUMBER=None):
        super(PW3D, self).__init__()

        self.JOINT_NUMBER = JOINT_NUMBER

        self._split_name = split_name
        self._pw3d_anno_dir = pw3d_anno_dir
        self._support_dir = pw3d_support_dir

        self._pw3d_file_names = self._get_pw3d_names()

        self.pw3d_motion_input_length =  input_length
        self.pw3d_motion_target_length =  target_length
        self.frame_interval = frame_interval
        self.true_length = (self.pw3d_motion_input_length+self.pw3d_motion_target_length-1)*self.frame_interval+self.pw3d_motion_input_length+self.pw3d_motion_target_length

        self.motion_dim = 54
        self.shift_step = shift_step

        self._load_skeleton()
        self._collect_all()
        # train:
        #       self.pw3d_seqs: [371x54, 391x54, ...], len=34
        #       elements: T_max=1263, T_min=333
        # test:
        #       self.pw3d_seqs: [1194x54, 1450x54, ...], len=37
        #       elements: T_max=2205, T_min=388

        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._pw3d_file_names)

    def _get_pw3d_names(self):
        seq_names = glob.glob(self._pw3d_anno_dir + self._split_name + '/*')
        return seq_names

    def _load_skeleton(self):
        skeleton_info = np.load(
                os.path.join(self._support_dir, 'body_models', 'smpl_skeleton.npz')
                )
        self.p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()[:, :22]
        parents = skeleton_info['parents']
        self.parent = {}
        for i in range(len(parents)):
            if i > 21:
                break
            self.parent[i] = parents[i]

    def _collect_all(self):
        self.pw3d_seqs = []
        self.data_idx = []
        idx = 0
        sample_rate = int(60 // 25)
        for pw3d_seq_name in tqdm(self._pw3d_file_names):
            pw3d_info = pkl.load(open(pw3d_seq_name, 'rb'), encoding='latin1')
            pw3d_motion_poses = pw3d_info['poses_60Hz']
            for i in range(len(pw3d_motion_poses)):
                N = len(pw3d_motion_poses[i])

                sampled_index = np.arange(0, N, sample_rate)
                motion_poses = pw3d_motion_poses[i][sampled_index]

                T = motion_poses.shape[0]
                motion_poses = motion_poses.reshape(T, -1, 3)
                motion_poses = motion_poses[:, :-2]
                motion_poses = R.from_rotvec(motion_poses.reshape(-1, 3)).as_rotvec()
                motion_poses = motion_poses.reshape(T, 22, 3)
                motion_poses[:, 0] = 0

                p3d0_tmp = self.p3d0.repeat([motion_poses.shape[0], 1, 1])
                motion_poses = ang2joint(p3d0_tmp, torch.tensor(motion_poses).float(), self.parent)

                if self.JOINT_NUMBER == 18:
                    motion_poses = motion_poses.reshape(-1, 22, 3)[:, 4:22].reshape(T, 54)
                elif self.JOINT_NUMBER == 22:
                    motion_poses = motion_poses.reshape(-1, 22, 3).reshape(T, 66)

                self.pw3d_seqs.append(motion_poses)
                valid_frames = np.arange(0, T - self.true_length + 1, self.shift_step)

                self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
                idx += 1

    def __getitem__(self, index):
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame, start_frame + self.true_length, self.frame_interval+1)
        assert len(frame_indexes) == self.pw3d_motion_input_length + self.pw3d_motion_target_length
        motion = self.pw3d_seqs[idx][frame_indexes]

        return motion.reshape(-1, self.JOINT_NUMBER, 3)


if __name__ == '__main__':
    main()
