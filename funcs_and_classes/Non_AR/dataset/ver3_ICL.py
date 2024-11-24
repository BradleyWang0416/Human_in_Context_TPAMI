from cgi import test
from email.quoprimime import body_check
from logging import config
import sys
import os
from tracemalloc import is_tracing
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import pickle
import time

from third_party.motionbert.human_body_prior import data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
sys.path.append(os.path.join(ROOT_DIR, 'third_party'))

from lib.utils.tools import read_pkl
from lib.utils.utils_non_AR import skel_to_h36m, generate_masked_joints_seq, rotate_y, unify_skeletons, vector_angle, get_complementary_idx
from lib.utils.viz_skel_seq import viz_skel_seq_anim
from lib.utils.utils_data import split_clips

from lib.data.datareader_h36m import DataReaderH36M

# from third_party.motionbert.lib.data.dataset_mesh import MotionSMPL
# from third_party.motionbert.lib.data.dataset_motion_3d import MotionDataset3D
from third_party.motionbert.lib.data.datareader_h36m import DataReaderH36M_3D, DataReaderH36M_MESH
from third_party.motionbert.lib.data.datareader_mesh import DataReaderMesh, DataReaderAMASS_MESH
from third_party.motionbert.lib.utils.utils_smpl import SMPL
from third_party.motionbert.lib.utils.utils_data import crop_scale, crop_scale_3d, crop_scale_2d
from third_party.motionbert.human_body_prior.body_model.body_model import BodyModel

from scipy.spatial.transform import Rotation as R
from data_gen.angle2joint import ang2joint


class DataReaderAMASS(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, dt_root ='', dt_file='amass_joints_h36m_60.pkl', read_confidence=True,
                 return_3d=True):
        self.split_id_train = None
        self.split_id_test = None
        self.dt_root = dt_root
        self.dt_dataset = read_pkl(os.path.join(dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        self.return_3d = return_3d
    
    def read_3d(self):
        train_set = self.dt_dataset['train']
        test_set = self.dt_dataset['test']
        pose3d_train = train_set['joint3d']
        pose3d_test = test_set['joint3d']
        pose3d_train = np.vstack(pose3d_train)[::self.sample_stride]
        pose3d_test = np.vstack(pose3d_test)[::self.sample_stride]
        return pose3d_train, pose3d_test
    
    def read_smpl_param(self):
        train_set = self.dt_dataset['train']
        test_set = self.dt_dataset['test']
        
        root_orient_train = np.vstack([train_set['smpl_param'][i]['root_orient'] for i in range(len(train_set['smpl_param']))])   # (5728109, 3)
        root_orient_test = np.vstack([test_set['smpl_param'][i]['root_orient'] for i in range(len(test_set['smpl_param']))])      # (1882446, 3)
        
        pose_body_train = np.vstack([train_set['smpl_param'][i]['pose_body'] for i in range(len(train_set['smpl_param']))])       # (5728109, 63)     
        pose_body_test = np.vstack([test_set['smpl_param'][i]['pose_body'] for i in range(len(test_set['smpl_param']))])          # (1882446, 63)
        
        pose_hand_train = np.vstack([train_set['smpl_param'][i]['pose_hand'] for i in range(len(train_set['smpl_param']))])       # (5728109, 90)
        pose_hand_test = np.vstack([test_set['smpl_param'][i]['pose_hand'] for i in range(len(test_set['smpl_param']))])          # (1882446, 90)
        
        trans_train = np.vstack([train_set['smpl_param'][i]['trans'] for i in range(len(train_set['smpl_param']))])               # (5728109, 3)
        trans_test = np.vstack([test_set['smpl_param'][i]['trans'] for i in range(len(test_set['smpl_param']))])                  # (1882446, 3)
        
        betas_train = np.vstack([train_set['smpl_param'][i]['betas'] for i in range(len(train_set['smpl_param']))])               # (5728109, 16)
        betas_test = np.vstack([test_set['smpl_param'][i]['betas'] for i in range(len(test_set['smpl_param']))])                  # (1882446, 16)
        
        dmpls_train = np.vstack([train_set['smpl_param'][i]['dmpls'] for i in range(len(train_set['smpl_param']))])               # (5728109, 8)
        dmpls_test = np.vstack([test_set['smpl_param'][i]['dmpls'] for i in range(len(test_set['smpl_param']))])                  # (1882446, 8)

        gender_train = np.concatenate([train_set['gender'][i] for i in range(len(train_set['gender']))])                            # (5728109,)
        gender_test = np.concatenate([test_set['gender'][i] for i in range(len(test_set['gender']))])                               # (1882446,)

        return  root_orient_train[::self.sample_stride], root_orient_test[::self.sample_stride], \
                pose_body_train[::self.sample_stride], pose_body_test[::self.sample_stride], \
                pose_hand_train[::self.sample_stride], pose_hand_test[::self.sample_stride], \
                trans_train[::self.sample_stride], trans_test[::self.sample_stride], \
                betas_train[::self.sample_stride], betas_test[::self.sample_stride], \
                dmpls_train[::self.sample_stride], dmpls_test[::self.sample_stride], \
                gender_train[::self.sample_stride], gender_test[::self.sample_stride]

    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['vid_list'][::self.sample_stride]
        vid_list_test = self.dt_dataset['test']['vid_list'][::self.sample_stride]
        self.split_id_train = split_clips(vid_list_train, self.n_frames, self.data_stride_train)
        self.split_id_test = split_clips(vid_list_test, self.n_frames, self.data_stride_test)
        return self.split_id_train, self.split_id_test
    
    def get_sliced_data(self):

        split_id_train, split_id_test = self.get_split_id()

        pose3d_train, pose3d_test = self.read_3d()          # (5728109,17,3); (1882446,17,3)
        train_3d, test_3d = pose3d_train[split_id_train], pose3d_test[split_id_test]
        train_2d, test_2d = train_3d[..., :2], test_3d[..., :2]
        if self.read_confidence:
            train_confidence = np.ones_like(train_2d)[...,0:1]
            test_confidence = np.ones_like(test_2d)[...,0:1]
            train_2d = np.concatenate((train_2d, train_confidence), axis=-1)
            test_2d = np.concatenate((test_2d, test_confidence), axis=-1)
        else:
            train_confidence = np.zeros_like(train_2d)[...,0:1]
            test_confidence = np.zeros_like(test_2d)[...,0:1]
            train_2d = np.concatenate((train_2d, train_confidence), axis=-1)
            test_2d = np.concatenate((test_2d, test_confidence), axis=-1)

        if self.return_3d:
            mean_3d = np.mean(pose3d_train, axis=(0,1))
            std_3d = np.std(pose3d_train, axis=(0,1))
            mean_2d, std_2d = mean_3d.copy(), std_3d.copy()
            mean_2d[-1] = 0
            std_2d[-1] = 0
            num_train = len(train_3d)
            num_test = len(test_3d)
            return train_2d, test_2d, train_3d, test_3d, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d

        else:
            root_orient_train, root_orient_test, pose_body_train, pose_body_test, pose_hand_train, pose_hand_test, trans_train, trans_test, betas_train, betas_test, dmpls_train, dmpls_test, gender_train, gender_test = self.read_smpl_param()
            root_orient_train, root_orient_test, pose_body_train, pose_body_test, pose_hand_train, pose_hand_test, trans_train, trans_test, betas_train, betas_test, dmpls_train, dmpls_test, gender_train, gender_test = \
                root_orient_train[split_id_train], root_orient_test[split_id_test], \
                pose_body_train[split_id_train], pose_body_test[split_id_test], pose_hand_train[split_id_train], pose_hand_test[split_id_test], \
                trans_train[split_id_train], trans_test[split_id_test], betas_train[split_id_train], betas_test[split_id_test], dmpls_train[split_id_train], dmpls_test[split_id_test], \
                gender_train[split_id_train], gender_test[split_id_test]

            train_smpl = {  'root_orient': root_orient_train,
                            'pose_body': pose_body_train,
                            'pose_hand': pose_hand_train,
                            'trans': trans_train,
                            'betas': betas_train,
                            'dmpls': dmpls_train,
                            'gender': gender_train
                            }
            test_smpl = {   'root_orient': root_orient_test,
                            'pose_body': pose_body_test,
                            'pose_hand': pose_hand_test,
                            'trans': trans_test,
                            'betas': betas_test,
                            'dmpls': dmpls_test,
                            'gender': gender_test
                            }
            num_train = len(train_smpl['pose_body'])
            num_test = len(test_smpl['pose_body'])
            mean_3d, std_3d = None, None
            mean_2d, std_2d = None, None
            return train_2d, test_2d, train_smpl, test_smpl, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d


class VideoAugmentor():
    def __init__(self):
        self.aug_dict = torch.load(f'third_party/wham/dataset/body_models/coco_aug_dict.pth')
        self.s_jittering = 1e-2
        self.s_bias = 1e-2
        self.s_peak_mask = 5e-3
        self.s_peak = 0.01

    def __call__(self, chunk_2d):
        # jitter
        chunk_len, num_joint, _ = chunk_2d.shape
        jittering_noise = torch.normal(
                                        mean=torch.zeros_like(chunk_2d[..., :2]),
                                        std=self.aug_dict['jittering'].reshape(1, num_joint, 1).expand(chunk_len, -1, 2)
                                        ) * self.s_jittering
        # bias
        bias_noise = torch.normal(
                                    mean=torch.zeros((num_joint, 2)), std=self.aug_dict['bias'].reshape(num_joint, 1)
                                ).unsqueeze(0) * self.s_bias
        # Low-frequency high-peak noise
        def get_peak_noise_mask():
            peak_noise_mask = torch.rand(chunk_len, num_joint).float() * self.aug_dict['pmask'].squeeze(0)
            peak_noise_mask = peak_noise_mask < self.s_peak_mask
            return peak_noise_mask

        peak_noise_mask = get_peak_noise_mask()
        peak_noise = peak_noise_mask.float().unsqueeze(-1).repeat(1, 1, 2)
        peak_noise = peak_noise * torch.randn(2) * self.aug_dict['peak'].reshape(1, -1, 1) * self.s_peak

        chunk_2d[..., :2] = chunk_2d[..., :2] + jittering_noise + bias_noise + peak_noise

        return chunk_2d


class MotionDatasetICL(Dataset):
    def __init__(self, args, data_split, TASK=None, DATASET_NAME=None):
        """
        dataset_task_info:
            train:
                H36M_MESH: [PE, MP, MC, FPE]
                H36M_3D: [PE, MP, MC, FPE]
                AMASS: [PE, MP, MC, FPE]
                PW3D_MESH: [PE, FPE, MP, MC]
            test:
                PW3D_MESH: [PE, FPE, MP, MC]
        """
        """
        dataset_file:
            H36M_MESH: 'mesh_det_h36m.pkl'
            H36M_3D: 'h36m_sh_conf_cam_source_final.pkl'
            AMASS: 'amass_joints_h36m_60.pkl'
            PW3D_MESH: 'mesh_det_pw3d.pkl'
        """
        np.random.seed(0)
        random.seed(0)
        assert (data_split == 'train') == (TASK is None) == (DATASET_NAME is None)
        assert (data_split == 'test') == (TASK is not None) == (DATASET_NAME is not None)
        self.is_train = data_split == 'train'
        if self.is_train:
            self.task_dict = args.dataset_task_info['train']
            self.datasets = args.dataset_task_info['train'].keys()
        else:
            self.task_dict = {DATASET_NAME: [TASK]}
            self.datasets = [DATASET_NAME]
        self.dataset_file = args.dataset_file
        self.task_to_flag = args.task_to_flag
        
        self.use_global_orient = args.get('use_global_orient', True)
        if self.is_train: print(f'\tDataset attribute [use_global_orient={self.use_global_orient}]')
        self.dumb_task = args.get('dumb_task', None)
        self.use_task_id_as_prompt = args.use_task_id_as_prompt
        self.normalize_2d = args.normalize_2d
        if self.is_train: print(f'\tDataset attribute [normalize_2d={self.normalize_2d}]')
        self.normalize_3d = args.normalize_3d
        if self.is_train: print(f'\tDataset attribute [normalize_3d={self.normalize_3d}]')
        self.aug = args.aug
        self.aug_shuffle_joints = args.aug_shuffle_joints
        self.clip_len = args.clip_len
        self.num_joint = args.num_joint
        self.current_as_history = args.current_as_history
        self.joint_mask_ratio = args.joint_mask_ratio
        self.frame_mask_ratio = args.frame_mask_ratio
        self.dataset_config = args.dataset_config

        query_list = []
        query_dict = {}
        prompt_list = {}
        prompt_dict = {}
        joint_mask_dict = {}
        frame_mask_dict = {}
        self.fps_dict = {'H36M_MESH': 50, 'H36M_3D': 50, 'AMASS': 60, 'PW3D_MESH': 60}
        for dataset_name in self.datasets:
            if self.is_train:
                print(f"\tLoading train data from [{dataset_name}] for task: {self.task_dict[dataset_name]}...", end=' ')
            else:
                print(f"\tLoading test data from [{dataset_name}] for task: {self.task_dict[dataset_name]}...", end=' ')
            st = time.time()
            dt_file = self.dataset_file[dataset_name]
            if dataset_name == 'H36M_MESH':
                datareader = DataReaderH36M_MESH(   n_frames=self.clip_len*2, sample_stride=self.dataset_config[dataset_name]['sample_stride'], 
                                                    data_stride_train=self.dataset_config[dataset_name]['data_stride_train'], data_stride_test=self.dataset_config[dataset_name]['data_stride_test'], 
                                                    dt_root='', dt_file=dt_file, read_confidence=self.dataset_config[dataset_name]['read_confidence'], 
                                                    return_3d=self.dataset_config[dataset_name]['return_3d'], use_global_orient=self.use_global_orient)
            elif dataset_name == 'H36M_3D':
                datareader = DataReaderH36M_3D(     n_frames=self.clip_len*2, sample_stride=self.dataset_config[dataset_name]['sample_stride'], 
                                                    data_stride_train=self.dataset_config[dataset_name]['data_stride_train'], data_stride_test=self.dataset_config[dataset_name]['data_stride_test'], 
                                                    dt_root='', dt_file=dt_file, read_confidence=self.dataset_config[dataset_name]['read_confidence'])
            elif dataset_name == 'AMASS':
                datareader = DataReaderAMASS(       n_frames=self.clip_len*2, sample_stride=self.dataset_config[dataset_name]['sample_stride'], 
                                                    data_stride_train=self.dataset_config[dataset_name]['data_stride_train'], data_stride_test=self.dataset_config[dataset_name]['data_stride_test'], 
                                                    dt_root='', dt_file=dt_file, read_confidence=self.dataset_config[dataset_name]['read_confidence'],
                                                    return_3d=self.dataset_config[dataset_name]['return_3d'])
            elif dataset_name == 'PW3D_MESH':
                datareader = DataReaderMesh(        n_frames=self.clip_len*2, sample_stride=self.dataset_config[dataset_name]['sample_stride'], 
                                                    data_stride_train=self.dataset_config[dataset_name]['data_stride_train'], data_stride_test=self.dataset_config[dataset_name]['data_stride_test'], 
                                                    dt_root='', dt_file=dt_file, res=[1920, 1920], read_confidence=self.dataset_config[dataset_name]['read_confidence'],
                                                    return_3d=self.dataset_config[dataset_name]['return_3d'], use_global_orient=self.use_global_orient)
            elif dataset_name == 'PW3D_MESH_FRONTVIEW':
                datareader = DataReaderMesh(        n_frames=self.clip_len*2, sample_stride=self.dataset_config[dataset_name]['sample_stride'], 
                                                    data_stride_train=self.dataset_config[dataset_name]['data_stride_train'], data_stride_test=self.dataset_config[dataset_name]['data_stride_test'], 
                                                    dt_root='', dt_file=dt_file, res=[1920, 1920], read_confidence=self.dataset_config[dataset_name]['read_confidence'],
                                                    return_3d=False)
            elif dataset_name == 'AMASS_MESH_FRONTVIEW':
                datareader = DataReaderAMASS_MESH(        n_frames=self.clip_len*2, sample_stride=self.dataset_config[dataset_name]['sample_stride'], 
                                                    data_stride_train=self.dataset_config[dataset_name]['data_stride_train'], data_stride_test=self.dataset_config[dataset_name]['data_stride_test'], 
                                                    dt_root='', dt_file=dt_file, read_confidence=self.dataset_config[dataset_name]['read_confidence'],
                                                    return_3d=False)
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")


            if args.use_presave_data:
                presave_folder = os.path.join(args.presave_folder, os.path.basename(__file__), dataset_name, 
                                              f'nframes{args.clip_len*2}_samplestride{self.dataset_config[dataset_name]["sample_stride"]}_'
                                              +f'datastridetrain{self.dataset_config[dataset_name]["data_stride_train"]}_datastridetest{self.dataset_config[dataset_name]["data_stride_test"]}_'
                                              +f'readconfidence{int(self.dataset_config[dataset_name]["read_confidence"])}_'
                                              +f'filename{os.path.basename(dt_file)}'
                                              )
                if hasattr(datareader, 'return_3d'):
                    presave_folder = presave_folder + f'_return3d{int(datareader.return_3d)}'
                if hasattr(datareader, 'use_global_orient'):
                    presave_folder = presave_folder + f'_useglobalorient{int(datareader.use_global_orient)}'

                presave_file = os.path.join(presave_folder, 'sliced_data.pkl')
                datareader_config_file = os.path.join(presave_folder, 'datareader_config.pkl')
                if not os.path.exists(presave_file):
                    os.makedirs(presave_folder)
                    sliced_data = datareader.get_sliced_data()
                    datareader_config = {
                        'datareader_class': datareader.__class__.__name__,
                        'n_frames': datareader.n_frames,
                        'sample_stride': datareader.sample_stride,
                        'data_stride_train': datareader.data_stride_train,
                        'data_stride_test': datareader.data_stride_test,
                        'dt_root': datareader.dt_root,
                        'dt_file': dt_file,
                        'read_confidence': datareader.read_confidence,
                    }
                    if hasattr(datareader, 'return_3d'):
                        datareader_config['return_3d'] = datareader.return_3d
                    if hasattr(datareader, 'use_global_orient'):
                        datareader_config['use_global_orient'] = datareader.use_global_orient
                    with open(presave_file, 'wb') as f:
                        pickle.dump(sliced_data, f, protocol=4)
                    with open(datareader_config_file, 'wb') as f:
                        pickle.dump(datareader_config, f)
                else:
                    with open(datareader_config_file, 'rb') as f:
                        datareader_config = pickle.load(f)
                    assert  datareader_config['datareader_class'] == datareader.__class__.__name__ and \
                            datareader_config['n_frames'] == datareader.n_frames and \
                            datareader_config['sample_stride'] == datareader.sample_stride and \
                            datareader_config['data_stride_train'] == datareader.data_stride_train and \
                            datareader_config['data_stride_test'] == datareader.data_stride_test and \
                            datareader_config['dt_root'] == datareader.dt_root and \
                            datareader_config['dt_file'] == dt_file and \
                            datareader_config['read_confidence'] == datareader.read_confidence
                    if hasattr(datareader, 'return_3d'): assert datareader_config['return_3d'] == datareader.return_3d
                    if hasattr(datareader, 'use_global_orient'): assert datareader_config['use_global_orient'] == datareader.use_global_orient
                    with open(presave_file, 'rb') as f:
                        sliced_data = pickle.load(f)
            else:
                sliced_data = datareader.get_sliced_data()
                
            chunk_2d_train, chunk_2d_test, chunk_3d_train, chunk_3d_test, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d = sliced_data     # (N, T, 17, 3)
            assert (not self.dataset_config[dataset_name]['read_confidence']) \
            == (chunk_2d_train[..., -1] == 0).all() \
            == (chunk_2d_test[..., -1] == 0).all()
            # == (mean_2d[..., -1] == 0).all() \
            # == (std_2d[..., -1] == 0).all()
            
            
            if args.visualize and False:
                if dataset_name == 'H36M_3D':
                    chunk_3d_train_velocity = np.diff(chunk_3d_train, axis=1)
                    chunk_magnitude = np.linalg.norm(chunk_3d_train_velocity, axis=-1)
                    chunk_magnitude_avg = np.mean(chunk_magnitude, axis=(-2,-1))
                    import matplotlib.pyplot as plt

                    fig, axs = plt.subplots(2, 1, figsize=(18, 14))

                    axs[0].bar(range(len(chunk_magnitude_avg)), chunk_magnitude_avg)
                    axs[0].set_xlabel('Chunk ID')
                    axs[0].set_ylabel('Magnitude Average')
                    axs[0].set_title('Chunk Magnitude Average')

                    axs[1].hist(chunk_magnitude_avg, bins=64)
                    axs[1].set_xlabel('Magnitude Average')
                    axs[1].set_ylabel('Frequency')
                    axs[1].set_title('Chunk Magnitude Average Histogram')

                    plt.tight_layout()
                    plt.savefig(f'velocity_statistic_{dataset_name}.png')
                    top_chunk_ids = np.argsort(chunk_magnitude_avg)[-50:][::-1]
                    data_dict = {}
                    for chunk_id in top_chunk_ids:
                        chunk_3d_train_chunk = chunk_3d_train[chunk_id]
                        data_dict[f'{chunk_id}'] = chunk_3d_train_chunk - chunk_3d_train_chunk[..., [0], :]
                    viz_skel_seq_anim(data_dict, subplot_layout=(5,10), lim3d=0.5, fs=0.2,
                                    if_print=1, file_name='top50velo', file_folder='tmp/viz_top50_velo_h36m_3d')
            if args.visualize and False and dataset_name == 'H36M_3D' and not self.is_train:
                chunk_2d_train, chunk_2d_test, chunk_3d_train, chunk_3d_test, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d = datareader.get_sliced_data()
                datareader_pose_estimation = DataReaderH36M(n_frames=args.clip_len*2, sample_stride=args.dataset_config['H36M_3D']['sample_stride'],
                                                            data_stride_train=args.dataset_config['H36M_3D']['data_stride_train'], data_stride_test=args.dataset_config['H36M_3D']['data_stride_test'], 
                                                            dt_root='', dt_file=args.dataset_file['H36M_3D'])
                train_data, test_data, train_labels, test_labels = datareader_pose_estimation.get_sliced_data()
                for i in range(len(test_labels)):
                    viz_skel_seq_anim({
                        '3D 1': chunk_3d_test[i][::4],
                        '3D 2': test_labels[i][::4],
                    }, mode='img', subplot_layout=(2,1), fs=0.5)
                

            num_query = num_train if self.is_train else num_test
            num_prompt = num_train

            if 'MC' in self.task_dict[dataset_name]:
                joint_mask_dict[dataset_name] = [random.sample(range(1,self.num_joint), int(self.joint_mask_ratio*self.num_joint)) for _ in range(num_query)]
            if 'MIB' in self.task_dict[dataset_name]:
                frame_mask_dict[dataset_name] = [random.sample(range(self.clip_len), int(self.frame_mask_ratio*self.clip_len)) for _ in range(num_query)]

            if self.is_train:
                query_dict[dataset_name] = {'2d': chunk_2d_train, '3d': chunk_3d_train, 'mean_2d': mean_2d, 'std_2d': std_2d, 'mean_3d': mean_3d, 'std_3d': std_3d}
            else:
                query_dict[dataset_name] = {'2d': chunk_2d_test, '3d': chunk_3d_test, 'mean_2d': mean_2d, 'std_2d': std_2d, 'mean_3d': mean_3d, 'std_3d': std_3d}
            query_list.extend(zip([dataset_name]*num_query, list(range(num_query))))

            if args.get('fix_prompt', None) == 'largest_velo':
                chunk_3d_train_velocity = np.diff(chunk_3d_train, axis=1)
                chunk_magnitude = np.linalg.norm(chunk_3d_train_velocity, axis=-1)
                chunk_magnitude_avg = np.mean(chunk_magnitude, axis=(-2,-1))
                top_chunk_id = np.argsort(chunk_magnitude_avg)[-1]
                prompt_list[dataset_name] = [top_chunk_id]
            else:
                prompt_list[dataset_name] = list(range(num_prompt))
            prompt_dict[dataset_name] = {'2d': chunk_2d_train, '3d': chunk_3d_train, 'mean_2d': mean_2d, 'std_2d': std_2d, 'mean_3d': mean_3d, 'std_3d': std_3d}
            print(f"costs {time.time()-st:.2f}s... has {num_query} samples")

        self.query_list = query_list
        self.query_dict = query_dict
        self.prompt_list = prompt_list
        self.prompt_dict = prompt_dict
        self.joint_mask_dict = joint_mask_dict
        self.frame_mask_dict = frame_mask_dict

        self.video_augmentor = VideoAugmentor()

        # PW3D; H36M
        self.smpl = SMPL('third_party/motionbert/data/mesh', batch_size=1)

        # AMASS
        self.J_reg_amass_to_h36m = np.load('third_party/motionbert/data/AMASS/J_regressor_h36m_correct.npy')
        self.real2cam = np.array([[1, 0, 0], 
                     [0, 0, 1], 
                     [0, -1, 0]], dtype=np.float64)
        

        if args.visualize and False:
            CHUNKS = [
                    #   [('PW3D_MESH', 3), ('PW3D_MESH_FRONTVIEW', 3)],
                    #   [('PW3D_MESH', 38), ('PW3D_MESH_FRONTVIEW', 38)],
                    #   [('AMASS', 1), ('AMASS_MESH_FRONTVIEW', 1)],
                    #   [('AMASS', 2), ('AMASS_MESH_FRONTVIEW', 2)], 
                    #   [('AMASS', 10), ('AMASS_MESH_FRONTVIEW', 10)], 
                      [('H36M_MESH', 4), ('H36M_3D', 4)]
                      ]
            for i in range(len(CHUNKS)):
                dataset1, _ = CHUNKS[i][0]
                dataset2, id = CHUNKS[i][1]
                chunk_2d_1, chunk3d_1 = self.prepare_chunk(query_dict, dataset1, id)
                _, chunk3d_1_wo_global_orient = self.prepare_chunk(query_dict, dataset1, id, use_global_orient=False)
                chunk_2d_2, chunk3d_2 = self.prepare_chunk(query_dict, dataset2, id)
                viz_dict = {
                    f'{dataset1}-{id}': chunk3d_1 - chunk3d_1[..., [0], :],
                    f'{dataset2}-{id}': chunk3d_2 - chunk3d_2[..., [0], :],
                    f'{dataset1}-{id} w/o global orient': chunk3d_1_wo_global_orient - chunk3d_1_wo_global_orient[..., [0], :],
                    f'{dataset1}-{id} 2D': chunk_2d_1[..., :2] - chunk_2d_1[..., [0], :2],
                    f'{dataset2}-{id} 2D': chunk_2d_2[..., :2] - chunk_2d_2[..., [0], :2],
                }
                viz_skel_seq_anim(viz_dict, subplot_layout=(2,3), fs=0.5, azim=-76, elev=56, fig_title=f'{dataset1}_{id}_cliplen{self.clip_len}',
                                  if_print=0,
                                  file_name=f'{dataset1}_{id}_wGlobalOrient', file_folder='tmp/viz_chunk_single')
            exit(0)

                    

            chunk_id = 0
            viz_dict = {}
            while chunk_id < 100:
                for dataset_name in query_dict:
                    chunk_2d, chunk_3d = self.prepare_chunk(query_dict, dataset_name, chunk_id)
                    viz_dict[f'{dataset_name} 2d'] = chunk_2d[..., :2] - chunk_2d[..., [0], :2]
                for dataset_name in query_dict:
                    chunk_2d, chunk_3d = self.prepare_chunk(query_dict, dataset_name, chunk_id)
                    viz_dict[f'{dataset_name} 3d'] = chunk_3d - chunk_3d[..., [0], :]
                viz_skel_seq_anim(viz_dict, subplot_layout=(2,1), fs=0.5, azim=-76, elev=56, fig_title=f'{chunk_id}', 
                                  if_print=0, file_name=f'ChunkID{chunk_id}', file_folder='tmp/viz_chunk')
                chunk_id += 1
            exit(0)

        if args.visualize == self.__class__.__name__:
            print(f'Do visualizing in {self.__class__.__name__}...')
            for id in ([1,2,10] + list(range(13,999999,130))):
                viz_dict = {}
                for dname in ['H36M_MESH', 'AMASS', 'PW3D_MESH']:
                    chunk_2d, chunk_3d = self.prepare_chunk(dname, id, is_query=True, use_global_orient=True)
                    input, output = self.prepare_motion(chunk_2d, chunk_3d, dname, 'PE', None, None)
                    mean_velocity = np.linalg.norm(np.diff(output*1000, axis=0), axis=-1).mean()
                    viz_dict.update({
                        f'{dname}|3d|velo{mean_velocity:.2f}': output,
                    })
                for dname in ['H36M_MESH', 'AMASS', 'PW3D_MESH']:
                    chunk_2d_woGlobalOrient, chunk_3d_woGlobalOrient = self.prepare_chunk(dname, id, is_query=True, use_global_orient=False)
                    input_woGlobalOrient, output_woGlobalOrient = self.prepare_motion(chunk_2d_woGlobalOrient, chunk_3d_woGlobalOrient, dname, 'PE', None, None)
                    mean_velocity = np.linalg.norm(np.diff(output_woGlobalOrient*1000, axis=0), axis=-1).mean()
                    viz_dict.update({
                        f'{dname}|3d_woGlobalOrient|velo{mean_velocity:.2f}': output_woGlobalOrient,                    
                    })
                viz_skel_seq_anim(viz_dict, subplot_layout=(2,3), fs=0.5, azim=-76, elev=56, fig_title=f'ID{id}', lim3d=0.6,
                                if_print=1, file_name=f'ID_{id:06d}', file_folder='viz_results/compareSMPL_W_WO_GlobalOrient',
                                tight_layout=True)
            exit(0)

    def __len__(self):
        return len(self.query_list)
    
    def prepare_chunk(self, dataset_name, chunk_id=None, is_query=True, use_global_orient=True):
        if hasattr(self, 'use_global_orient'):
            use_global_orient = self.use_global_orient
        if is_query:
            data_dict = self.query_dict
        else:
            data_dict = self.prompt_dict
        chunk_2d = data_dict[dataset_name]['2d'][chunk_id].copy() if chunk_id is not None else data_dict[dataset_name]['2d'].copy()
        chunk_2d = torch.from_numpy(chunk_2d).float()
        if dataset_name in ['H36M_MESH', 'PW3D_MESH'] and isinstance(data_dict[dataset_name]['3d'], dict):
            chunk_smpl_pose = data_dict[dataset_name]['3d']['pose'][chunk_id].copy()
            chunk_smpl_shape = data_dict[dataset_name]['3d']['shape'][chunk_id].copy()
            chunk_smpl_pose = torch.from_numpy(chunk_smpl_pose).float()                             # (T,72)
            chunk_smpl_shape = torch.from_numpy(chunk_smpl_shape).float()
            chunk_smpl = self.smpl(
                betas=chunk_smpl_shape,                                                 # (T,10)
                body_pose=chunk_smpl_pose[:, 3:],                                       # (T,69)
                global_orient=chunk_smpl_pose[:, :3] if use_global_orient else torch.zeros_like(chunk_smpl_pose[:, :3]),                                   # (T,3)
                pose2rot=True
            )
            chunk_verts = chunk_smpl.vertices.detach()
            J_regressor = self.smpl.J_regressor_h36m                                    # [17, 6890]
            J_regressor_batch = J_regressor[None, :].expand(chunk_verts.shape[0], -1, -1).to(chunk_verts.device)
            chunk_3d = torch.einsum('bij,bjk->bik', J_regressor_batch, chunk_verts)
            chunk_3d = chunk_3d - chunk_3d[:, :1, :]                       # chunk_3d: (T,17,3)

            chunk_verts_ = chunk_smpl.vertices.detach()*1000
            chunk_3d_ = torch.einsum('bij,bjk->bik', J_regressor_batch, chunk_verts_)
            chunk_3d_ = chunk_3d_ - chunk_3d_[:, :1, :]                       # chunk_3d: (T,17,3)

        elif dataset_name in ['PW3D_MESH_FRONTVIEW']:
            skeleton_info = np.load('data/support_data/body_models/smpl_skeleton.npz')
            p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()[:, :22]
            parents = skeleton_info['parents']
            parent = {}
            for i in range(len(parents)):
                if i > 21:
                    break
                parent[i] = parents[i]

            chunk_smpl_pose = data_dict[dataset_name]['3d']['pose'][chunk_id].copy()        # (T,72)
            T = chunk_smpl_pose.shape[0]
            chunk_smpl_pose = chunk_smpl_pose.reshape(T, -1, 3)[:, :-2, :]                  # (T,24,3) --> (T,22,3)
            chunk_smpl_pose = R.from_rotvec(chunk_smpl_pose.reshape(-1, 3)).as_rotvec()     # (T*22,3) --> (T*22,3)
            chunk_smpl_pose = chunk_smpl_pose.reshape(T, -1, 3)                             # (T*22,3) --> (T,22,3)
            chunk_smpl_pose[:, 0, :] = 0

            p3d0_tmp = p3d0.repeat([chunk_smpl_pose.shape[0], 1, 1])                        # (1,22,3) --> (T,22,3)
            chunk_3d = ang2joint(p3d0_tmp, torch.tensor(chunk_smpl_pose).float(), parent)

        elif dataset_name in ['AMASS_MESH_FRONTVIEW']:
            skeleton_info = np.load('data/support_data/body_models/smpl_skeleton.npz')
            p3d0 = torch.from_numpy(skeleton_info['p3d0']).float()
            parents = skeleton_info['parents']
            parent = {}
            for i in range(len(parents)):
                parent[i] = parents[i]

            chunk_smpl_pose = data_dict[dataset_name]['3d']['pose'][chunk_id].copy()        # (T,156)
            T = chunk_smpl_pose.shape[0]
            chunk_smpl_pose = R.from_rotvec(chunk_smpl_pose.reshape(-1, 3)).as_rotvec()
            chunk_smpl_pose = chunk_smpl_pose.reshape(T, 52, 3)
            chunk_smpl_pose[:, 0, :] = 0

            p3d0_tmp = p3d0.repeat([chunk_smpl_pose.shape[0], 1, 1])
            chunk_3d = ang2joint(p3d0_tmp, torch.tensor(chunk_smpl_pose).float(), parent)
            chunk_3d = chunk_3d.reshape(-1, 52, 3)[:, :22, :]
            chunk_2d = chunk_3d[..., :2].clone()
            chunk_2d = torch.cat([chunk_2d, torch.zeros_like(chunk_2d[..., :1])], dim=-1)

        elif dataset_name in ['AMASS'] and isinstance(data_dict[dataset_name]['3d'], dict): 
            chunk_smpl_root_orient = data_dict[dataset_name]['3d']['root_orient'][chunk_id].copy()  # (32, 3)
            chunk_smpl_pose_body = data_dict[dataset_name]['3d']['pose_body'][chunk_id].copy()      # (32, 63)
            chunk_smpl_pose_hand = data_dict[dataset_name]['3d']['pose_hand'][chunk_id].copy()      # (32, 90)
            chunk_smpl_trans = data_dict[dataset_name]['3d']['trans'][chunk_id].copy()              # (32, 3)
            chunk_smpl_betas = data_dict[dataset_name]['3d']['betas'][chunk_id].copy()              # (32, 16)
            chunk_smpl_dmpls = data_dict[dataset_name]['3d']['dmpls'][chunk_id].copy()              # (32,8)
            gender = data_dict[dataset_name]['3d']['gender'][chunk_id].copy().mean().astype(int)
            subject_gender = 'male' if gender==0 else 'female'
            bm_fname = os.path.join('third_party/motionbert/data/AMASS/body_models/smplh/{}/model.npz'.format(subject_gender))
            dmpl_fname = os.path.join('third_party/motionbert/data/AMASS/body_models/dmpls/{}/model.npz'.format(subject_gender))
            bodymodel = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname)
            body_trans_root = bodymodel(                
                                            root_orient=torch.from_numpy(chunk_smpl_root_orient).float() if use_global_orient else torch.zeros(chunk_smpl_root_orient.shape),
                                            pose_body=torch.from_numpy(chunk_smpl_pose_body).float(),
                                            pose_hand=torch.from_numpy(chunk_smpl_pose_hand).float(),
                                            trans=torch.from_numpy(chunk_smpl_trans).float(),
                                            betas=torch.from_numpy(chunk_smpl_betas).float(),
                                            dmpls=torch.from_numpy(chunk_smpl_dmpls).float(),
                                        )
            mesh = body_trans_root.v.cpu().numpy()
            chunk_3d = np.dot(self.J_reg_amass_to_h36m , mesh)    # (17,T,3)
            chunk_3d = np.transpose(chunk_3d, (1,0,2))        # (T,17,3)
            # chunk_3d = chunk_3d @ self.real2cam
            chunk_3d = torch.from_numpy(chunk_3d).float()
        else:
            chunk_3d = data_dict[dataset_name]['3d'][chunk_id].copy() if chunk_id is not None else data_dict[dataset_name]['3d'].copy()
            chunk_3d = torch.from_numpy(chunk_3d).float()
        
        if dataset_name == 'PW3D_MESH' and False:
            chunk_2d = chunk_2d[self.clip_len//2:self.clip_len//2+self.clip_len]
            chunk_3d = chunk_3d[self.clip_len//2:self.clip_len//2+self.clip_len]


            chunk_3d_interpolated = []
            chunk_2d_interpolated = []

            for i in range(chunk_3d.shape[0] - 1):
                avg_3d = (chunk_3d[i] + chunk_3d[i + 1]) / 2
                chunk_3d_interpolated.append(chunk_3d[i])
                chunk_3d_interpolated.append(avg_3d)

                avg_2d = (chunk_2d[i] + chunk_2d[i + 1]) / 2
                chunk_2d_interpolated.append(chunk_2d[i])
                chunk_2d_interpolated.append(avg_2d)

            chunk_3d_interpolated.append(chunk_3d[-1])
            chunk_3d_interpolated.append(chunk_3d[-1])
            chunk_3d = torch.stack(chunk_3d_interpolated)

            chunk_2d_interpolated.append(chunk_2d[-1])
            chunk_2d_interpolated.append(chunk_2d[-1])
            chunk_2d = torch.stack(chunk_2d_interpolated)

            assert len(chunk_3d) == self.clip_len*2 and len(chunk_2d) == self.clip_len*2

        if dataset_name == 'H36M_3D':
            chunk_3d = chunk_3d / self.dataset_config['H36M_3D']['scale_3D']

            
        if self.is_train and self.aug:
            if dataset_name == 'AMASS':
                chunk_2d = self.video_augmentor(chunk_2d)
            chunk_2d = crop_scale_2d(chunk_2d.clone(), scale_range=[0.5, 1])
            chunk_3d = crop_scale_3d(chunk_3d.clone(), scale_range=[0.5, 1])        

        if self.normalize_2d:
            mean_2d = data_dict[dataset_name]['mean_2d']
            std_2d = data_dict[dataset_name]['std_2d']
            mean_2d = torch.from_numpy(mean_2d).float()
            std_2d = torch.from_numpy(std_2d).float()      
            chunk_2d[..., :2] = (chunk_2d[..., :2] - mean_2d[:2]) / std_2d[:2]

        if self.normalize_3d:
            mean_3d = data_dict[dataset_name]['mean_3d']
            std_3d = data_dict[dataset_name]['std_3d']
            mean_3d = torch.from_numpy(mean_3d).float()
            std_3d = torch.from_numpy(std_3d).float()
            assert torch.all(std_3d != 0)
            chunk_3d = (chunk_3d - mean_3d) / std_3d

        return chunk_2d, chunk_3d
    
    def prepare_motion(self, chunk_2d, chunk_3d, dataset_name, task, joint_mask, frame_mask):
        # chunk_2d: (T,17,3) or (N,T,17,3); 
        # chunk_3d: (T,17,3) or (N,T,17,3)
        if task == 'PE':
            motion_input = chunk_2d[..., :self.clip_len, :, :] if self.current_as_history else chunk_2d[..., self.clip_len:, :, :]
            motion_output = chunk_3d[..., :self.clip_len, :, :] if self.current_as_history else chunk_3d[..., self.clip_len:, :, :]
        elif task == 'FPE':
            motion_input = chunk_2d[..., :self.clip_len, :, :]
            motion_output = chunk_3d[..., self.clip_len:, :, :]
        elif task == 'MP':
            # motion_input = chunk_3d[..., self.clip_len:, :, :]
            # motion_output = chunk_3d[..., :self.clip_len, :, :]
            motion_input = chunk_3d[..., :self.clip_len, :, :]
            motion_output = chunk_3d[..., self.clip_len:, :, :]
        elif task == 'MC':
            motion_input = chunk_3d[..., :self.clip_len, :, :] if self.current_as_history else chunk_3d[..., self.clip_len:, :, :]
            motion_output = motion_input.clone()
            assert joint_mask is not None
            motion_input[..., joint_mask, :] = 0
        elif task == 'MIB':
            motion_input = chunk_3d[..., :self.clip_len, :, :] if self.current_as_history else chunk_3d[..., self.clip_len:, :, :]
            motion_output = motion_input.clone()
            assert frame_mask is not None
            motion_input[..., frame_mask, :, :] = 0

        elif task == 'COPY':
            motion_input = chunk_3d[..., :self.clip_len, :, :] if self.current_as_history else chunk_3d[..., self.clip_len:, :, :]
            motion_output = motion_input.clone()
        elif task == 'COPY2D':
            motion_input = chunk_2d[..., :self.clip_len, :, :] if self.current_as_history else chunk_2d[..., self.clip_len:, :, :]
            motion_output = motion_input.clone()
        elif task == 'FPEhis':
            motion_input = chunk_2d[..., self.clip_len:, :, :]
            motion_output = chunk_3d[..., :self.clip_len, :, :]
        elif task == 'MPhis':
            motion_input = chunk_3d[..., self.clip_len:, :, :]
            motion_output = chunk_3d[..., :self.clip_len, :, :]
        elif task == 'MP2D':
            motion_input = chunk_2d[..., :self.clip_len, :, :]
            motion_output = chunk_2d[..., self.clip_len:, :, :]
        elif task == 'MC2D':
            motion_input = chunk_2d[..., :self.clip_len, :, :] if self.current_as_history else chunk_2d[..., self.clip_len:, :, :]
            motion_output = motion_input.clone()
            assert joint_mask is not None
            motion_input[..., joint_mask, :] = 0
        elif task == 'MIB2D':
            motion_input = chunk_2d[..., :self.clip_len, :, :] if self.current_as_history else chunk_2d[..., self.clip_len:, :, :]
            motion_output = motion_input.clone()
            assert frame_mask is not None
            motion_input[..., frame_mask, :, :] = 0




        if self.dataset_config[dataset_name]['rootrel_input']:
            motion_input = motion_input - motion_input[..., [0], :]
        if self.dataset_config[dataset_name]['rootrel_target']:
            motion_output = motion_output - motion_output[..., [0], :]

        return motion_input, motion_output
    
    def postprocess(self, motion_seq, dataset_name, task):

        if dataset_name == 'H36M_3D' and task == 'PE':
            if self.dataset_config['H36M_3D']['rootrel_target']:
                motion_seq[..., 0, :] = 0

        if self.normalize_3d:
            mean_3d = self.query_dict[dataset_name]['mean_3d']
            std_3d = self.query_dict[dataset_name]['std_3d']
            mean_3d = torch.from_numpy(mean_3d).float()
            std_3d = torch.from_numpy(std_3d).float()
            motion_seq = motion_seq * std_3d + mean_3d
        
        if dataset_name == 'H36M_3D':
            motion_seq = motion_seq * self.dataset_config['H36M_3D']['scale_3D']

        return motion_seq



    def __getitem__(self, query_index):
        dataset_name, query_chunk_id = self.query_list[query_index]
        prompt_chunk_id = random.choice(self.prompt_list[dataset_name])

        query_chunk_2d, query_chunk_3d = self.prepare_chunk(dataset_name, chunk_id=query_chunk_id, is_query=True)
        prompt_chunk_2d, prompt_chunk_3d = self.prepare_chunk(dataset_name, chunk_id=prompt_chunk_id, is_query=False)

        if self.is_train and self.aug_shuffle_joints:
            if random.random() > 0.5:
                joint_idx = list(range(1, self.num_joint))
                random.shuffle(joint_idx)
                joint_idx = [0] + joint_idx
                query_chunk_2d = query_chunk_2d[..., joint_idx, :]
                prompt_chunk_2d = prompt_chunk_2d[..., joint_idx, :]
                query_chunk_3d = query_chunk_3d[..., joint_idx, :]
                prompt_chunk_3d = prompt_chunk_3d[..., joint_idx, :]


        QUERY_INPUT = []
        QUERY_TARGET = []
        PROMPT_INPUT = []
        PROMPT_TARGET = []
        INFO = []
        for task in self.task_dict[dataset_name]:
            joint_mask, frame_mask = None, None
            if task == 'MC':
                joint_mask = self.joint_mask_dict[dataset_name][query_chunk_id]
            if task == 'MIB':
                frame_mask = self.frame_mask_dict[dataset_name][query_chunk_id]
            
            query_input, query_target = self.prepare_motion(query_chunk_2d.clone(), query_chunk_3d.clone(), dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask)
            prompt_input, prompt_target = self.prepare_motion(prompt_chunk_2d.clone(), prompt_chunk_3d.clone(), dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask)

            # if self.dataset_config[dataset_name]['rootrel_input']:
            #     query_input = query_input - query_input[..., [0], :]
            #     prompt_input = prompt_input - prompt_input[..., [0], :]
            # if self.dataset_config[dataset_name]['rootrel_target']:
            #     query_target = query_target - query_target[..., [0], :]
            #     prompt_target = prompt_target - prompt_target[..., [0], :]
            if self.use_task_id_as_prompt:
                prompt_input = torch.tensor([self.task_to_flag[task]]).unsqueeze(0).unsqueeze(0).expand_as(prompt_input).float() / 10 
                prompt_target = torch.tensor([self.task_to_flag[task]]).unsqueeze(0).unsqueeze(0).expand_as(prompt_target).float() / 10

            QUERY_INPUT.append(query_input)
            QUERY_TARGET.append(query_target)
            PROMPT_INPUT.append(prompt_input)
            PROMPT_TARGET.append(prompt_target)
            INFO.append({
                'dataset': dataset_name,
                'task': task,
                'joint_mask': joint_mask,
                'frame_mask': frame_mask,
                'query_chunk_id': query_chunk_id,
                'prompt_chunk_id': prompt_chunk_id,
                'query_index': query_index
            })
        
        if self.is_train and self.dumb_task:
            for dumb_task in self.dumb_task.split(','):
                joint_mask, frame_mask = None, None
                if 'MC' in dumb_task:
                    joint_mask = self.joint_mask_dict[dataset_name][query_chunk_id]
                if 'MIB' in dumb_task:
                    frame_mask = self.frame_mask_dict[dataset_name][query_chunk_id]
                query_input, query_target = self.prepare_motion(query_chunk_2d.clone(), query_chunk_3d.clone(), dataset_name=dataset_name, task=dumb_task, joint_mask=joint_mask, frame_mask=frame_mask)
                prompt_input, prompt_target = self.prepare_motion(prompt_chunk_2d.clone(), prompt_chunk_3d.clone(), dataset_name=dataset_name, task=dumb_task, joint_mask=joint_mask, frame_mask=frame_mask)

                # if self.dataset_config[dataset_name]['rootrel_input']:
                #     query_input = query_input - query_input[..., [0], :]
                #     prompt_input = prompt_input - prompt_input[..., [0], :]
                # if self.dataset_config[dataset_name]['rootrel_target']:
                #     query_target = query_target - query_target[..., [0], :]
                #     prompt_target = prompt_target - prompt_target[..., [0], :]
                if self.use_task_id_as_prompt:
                    prompt_input = torch.tensor([self.task_to_flag[task]]).unsqueeze(0).unsqueeze(0).expand_as(prompt_input).float() / 10 
                    prompt_target = torch.tensor([self.task_to_flag[task]]).unsqueeze(0).unsqueeze(0).expand_as(prompt_target).float() / 10

                QUERY_INPUT.append(query_input)
                QUERY_TARGET.append(query_target)
                PROMPT_INPUT.append(prompt_input)
                PROMPT_TARGET.append(prompt_target)
                INFO.append({
                    'dataset': dataset_name,
                    'task': task,
                    'joint_mask': joint_mask,
                    'frame_mask': frame_mask,
                    'query_chunk_id': query_chunk_id,
                    'prompt_chunk_id': prompt_chunk_id,
                    'query_index': query_index
                })



        QUERY_INPUT = torch.stack(QUERY_INPUT)
        QUERY_TARGET = torch.stack(QUERY_TARGET)
        PROMPT_INPUT = torch.stack(PROMPT_INPUT)
        PROMPT_TARGET = torch.stack(PROMPT_TARGET)

        return QUERY_INPUT, QUERY_TARGET, PROMPT_INPUT, PROMPT_TARGET, INFO
    
        if self.is_train:
            return QUERY_INPUT, QUERY_TARGET, PROMPT_INPUT, PROMPT_TARGET, INFO
        else:
            QUERY_INPUT = QUERY_INPUT.squeeze(0)
            QUERY_TARGET = QUERY_TARGET.squeeze(0)
            PROMPT_INPUT = PROMPT_INPUT.squeeze(0)
            PROMPT_TARGET = PROMPT_TARGET.squeeze(0)        
            task_flag = self.task_to_flag[INFO[0]['task']]
            if INFO[0]['task'] == 'MC':
                return torch.cat([PROMPT_INPUT, PROMPT_TARGET], dim=-3), torch.cat([QUERY_INPUT, QUERY_TARGET], dim=-3), torch.tensor(task_flag), torch.tensor(joint_mask)
            elif INFO[0]['task'] == 'MIB':
                return torch.cat([PROMPT_INPUT, PROMPT_TARGET], dim=-3), torch.cat([QUERY_INPUT, QUERY_TARGET], dim=-3), torch.tensor(task_flag), torch.tensor(frame_mask)
            else:
                return torch.cat([PROMPT_INPUT, PROMPT_TARGET], dim=-3), torch.cat([QUERY_INPUT, QUERY_TARGET], dim=-3), torch.tensor(task_flag)

def collate_func(batch):
    batch_size = len(batch)
    QUERY_INPUT = []
    QUERY_TARGET = []
    PROMPT_INPUT = []
    PROMPT_TARGET = []
    INFO = []
    for b in range(batch_size):
        query_input, query_target, prompt_input, prompt_target, info = batch[b]
        # query_input/query_target/prompt_input/prompt_target: [<num_tasks>, 16, 17, 3]
        # info: list of dict. list len=<num_tasks>
        QUERY_INPUT.append(query_input)
        QUERY_TARGET.append(query_target)
        PROMPT_INPUT.append(prompt_input)
        PROMPT_TARGET.append(prompt_target)
        INFO = INFO + info
    QUERY_INPUT = torch.concatenate(QUERY_INPUT)
    QUERY_TARGET = torch.concatenate(QUERY_TARGET)
    PROMPT_INPUT = torch.concatenate(PROMPT_INPUT)
    PROMPT_TARGET = torch.concatenate(PROMPT_TARGET)  
    
    return QUERY_INPUT, QUERY_TARGET, PROMPT_INPUT, PROMPT_TARGET, INFO



    # def __init__(self, args, args_Motion3D, args_MotionSMPL, data_split):
        # args_MotionSMPL.clip_len = args_Motion3D.clip_len
        # args_MotionSMPL.data_stride = args_Motion3D.data_stride
        # motion_h36m = MotionDataset3D(args_Motion3D, data_split=data_split, dataset='h36m')
        # motion_amass = MotionDataset3D(args_Motion3D, data_split=data_split, dataset='amass')
        # mesh_h36m = MotionSMPL(args_MotionSMPL, data_split=data_split, dataset="h36m")
        # mesh_pw3d = MotionSMPL(args_MotionSMPL, data_split=data_split, dataset="pw3d")
        
        # for i in range(0, 1000, 13):
        #     h36m_3d = motion_h36m[i]
        #     amass_3d = motion_amass[i]
        #     h36m_mesh = mesh_h36m[i]
        #     pw3d_mesh = mesh_pw3d[i]

        #     data_dict = {
        #         'h36m_3d 2D': h36m_3d[0][..., :-1], 'h36m_3d 3D': h36m_3d[1]/0.298, 'h36m_3d 3D copy': torch.zeros_like(h36m_3d[1]),
        #         'amass_3d 2D': amass_3d[0][..., :-1], 'amass_3d 3D': amass_3d[1]/0.298, 'amass_3d 3D copy': torch.zeros_like(amass_3d[1]),
        #         'h36m_mesh 2D': h36m_mesh[0][..., :-1], 'h36m_mesh 3D': h36m_mesh[1]['kp_3d']/1000, 'h36m_mesh 3D joints': h36m_mesh[1]['joints_3d'],
        #         'pw3d_mesh 2D': pw3d_mesh[0][..., :-1], 'pw3d_mesh 3D': pw3d_mesh[1]['kp_3d']/1000, 'pw3d_mesh 3D joints': pw3d_mesh[1]['joints_3d'],
        #     }
        #     viz_skel_seq_anim(data_dict, subplot_layout=(4,3), fs=0.5, fig_title=f'{i}')
        # exit(0)
        
