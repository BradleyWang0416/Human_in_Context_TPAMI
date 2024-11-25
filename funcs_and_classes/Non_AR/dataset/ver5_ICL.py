from cgi import test
import chunk
from email.quoprimime import body_check
from logging import config
import sys
import os
from tabnanny import check
from tracemalloc import is_tracing
from typing import OrderedDict
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import pickle
import time
import joblib
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

# from third_party.Pose2Mesh.data.COCO import dataset
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
# from third_party.motionbert.lib.data.datareader_h36m import DataReaderH36M_3D, DataReaderH36M_MESH
# from third_party.motionbert.lib.data.datareader_mesh import DataReaderMesh, DataReaderAMASS_MESH
from third_party.motionbert.lib.utils.utils_smpl import SMPL
from third_party.motionbert.lib.utils.utils_data import crop_scale, crop_scale_3d, crop_scale_2d
from third_party.motionbert.human_body_prior.body_model.body_model import BodyModel

from scipy.spatial.transform import Rotation as R
# from data_gen.angle2joint import ang2joint

from data_icl_gen.fps_v3_velo import farthest_point_sampling


class MotionDatasetICL(Dataset):
    def __init__(self, args, data_split, TASK=None, DATASET_NAME=None, SLICED_DATA=None, PROMPT_LOG=None, **kwargs):
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
        
        self.args = args

        self.visualize = args.visualize

        self.use_smpl = True if hasattr(args, 'Mesh') and args.Mesh.enable else False
        
        if data_split=='train': assert (TASK is None) == (DATASET_NAME is None) == (SLICED_DATA is None)
        if data_split == 'test': assert (TASK is not None) == (DATASET_NAME is not None)
        self.is_train = data_split == 'train'
        if self.is_train:
            self.task_dict = args.dataset_task_info['train']
            self.datasets = args.dataset_task_info['train'].keys()
        else:
            self.task_dict = {DATASET_NAME: [TASK]}
            self.datasets = [DATASET_NAME]
        self.dataset_file = args.dataset_file
        self.task_to_flag = args.task_to_flag
        
        self.dumb_task = args.get('dumb_task', None)
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
        self.num_query_dict = {}
        if args.get('fix_prompt', None) == ['TrainRandom_TestNearest', 'TrainSame_TestNearest']:
            if self.is_train:
                self.prompt_log = {}
            else:
                self.prompt_log = PROMPT_LOG
        if self.is_train: self.sliced_data_dict = {}
        for dataset_name in self.datasets:
            if kwargs.get('rank', 0) == 0: print(f"\tLoading {data_split} data from [{dataset_name}] for task: {self.task_dict[dataset_name]}...", end=' ')
            st = time.time()

            dt_file = self.dataset_file[dataset_name]
            n_frames = self.dataset_config[dataset_name].get('clip_len', self.clip_len) * 2
            if dataset_name in ['H36M_MESH', 'PW3D_MESH', 'H36M_MESH_TCMR', 'AMASS', 'COCO']:
                datareader = DataReaderMesh(dataset_name=dataset_name,
                                            split = data_split,
                                            n_frames = n_frames, 
                                            sample_stride = self.dataset_config[dataset_name]['sample_stride'], 
                                            data_stride = self.dataset_config[dataset_name]['data_stride'],
                                            read_confidence = self.dataset_config[dataset_name]['read_confidence'],
                                            dt_root = '', 
                                            dt_file = dt_file,
                                            res = [1920, 1920] if dataset_name == 'PW3D_MESH' else None,
                                            use_global_orient = self.dataset_config[dataset_name]['use_global_orient'],
                                            return_skel3d=True,
                                            return_smpl=self.use_smpl,
                                            **kwargs)
            elif dataset_name == 'H36M_3D':
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")
            
            presave_folder = os.path.join(args.presave_folder, os.path.splitext(os.path.basename(__file__))[0], dataset_name,
                                        f'nframes{datareader.n_frames} - samplestride{datareader.sample_stride} - '
                                        +f'datastridetrain{datareader.data_stride["train"]} - datastridetest{datareader.data_stride["test"]} - '
                                        +f'readconfidence{int(datareader.read_confidence)} - '
                                        +f'useglobalorient{int(datareader.use_global_orient)} - '
                                        +f'returnskel3d_{int(datareader.return_skel3d)} - '
                                        +f'returnsmpl_{int(datareader.return_smpl)} - '
                                        +f'filename_{os.path.splitext(os.path.basename(dt_file))[0]}'
                                        )
            
            if SLICED_DATA is None:   
                if args.use_presave_data:
                    def get_class_attributes(obj):
                        return {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('__') and not callable(getattr(obj, attr)) and not 'split' in attr and not 'dataset' in attr and attr != 'rank'}

                    presave_file = os.path.join(presave_folder, 'sliced_data.pkl')
                    datareader_config_file = os.path.join(presave_folder, 'datareader_config.pkl')
                    if not os.path.exists(presave_file):
                        if kwargs.get('rank', 0) == 0: print("Presaving...", end=' ')
                        os.makedirs(presave_folder, exist_ok=True)
                        sliced_data = datareader.get_all_data()   # this step will change the self.split_id from None to the actual split_ids
                        datareader_config = get_class_attributes(datareader)
                        with open(presave_file, 'wb') as f:
                            pickle.dump(sliced_data, f, protocol=4)
                        with open(datareader_config_file, 'wb') as f:
                            pickle.dump(datareader_config, f)
                    else:
                        if kwargs.get('rank', 0) == 0: print("Loading presaved...", end=' ')
                        with open(datareader_config_file, 'rb') as f:
                            datareader_config = pickle.load(f)
                        assert datareader_config == get_class_attributes(datareader)
                        with open(presave_file, 'rb') as f:
                            sliced_data = pickle.load(f)
                        # if 'train' in sliced_data: assert len(sliced_data['train']['joint2d']) == datareader.get_num_clips('train')
                        # if 'test' in sliced_data: assert len(sliced_data['test']['joint2d']) == datareader.get_num_clips('test')
                else:
                    sliced_data = datareader.get_all_data()  # this step will change the self.split_id from None to the actual split_ids
                
                if self.is_train: self.sliced_data_dict[dataset_name] = sliced_data
            
            else:
                sliced_data = SLICED_DATA

            num_query = sliced_data[data_split]['joint2d'].shape[0]
            self.num_query_dict[dataset_name] = num_query
            query_dict[dataset_name] = sliced_data[data_split]
            query_list.extend(zip([dataset_name]*num_query, list(range(num_query))))

            num_prompt = sliced_data['train']['joint2d'].shape[0]
            prompt_dict[dataset_name] = sliced_data['train']
            if args.get('fix_prompt', None) == 'largest_velo':
                chunk_3d_train_velocity = np.diff(prompt_dict[dataset_name]['joint3d'], axis=1)
                chunk_magnitude = np.linalg.norm(chunk_3d_train_velocity, axis=-1)
                chunk_magnitude_avg = np.mean(chunk_magnitude, axis=(-2,-1))
                top_chunk_id = np.argsort(chunk_magnitude_avg)[-1]
                prompt_list[dataset_name] = [top_chunk_id]
            elif args.get('fix_prompt', None) == 'TrainRandom_TestNearest':
                if self.is_train:
                    prompt_list[dataset_name] = list(range(num_prompt))
                    self.prompt_log = {}
                else:                    
                    def find_nearest_neighbors(test_data, train_data, batch_size=512):
                        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

                        train_idx_list = np.array([], dtype=int)
                        train_dist_list = np.array([])
                        for test_batch in test_loader:
                            test_batch = test_batch.cuda()
                            B_q = test_batch.shape[0]

                            min_dis = np.ones(B_q) * 1000000
                            min_dis_train_idx = np.zeros(B_q, dtype=int)
                            
                            for train_batch_idx, train_batch in enumerate(train_loader):
                                train_batch = train_batch.cuda()
                                B_p = train_batch.shape[0]

                                dis = torch.mean(torch.norm(test_batch[:, None, :, :, :] - train_batch[None, :, :, :, :], dim=-1), dim=(-2,-1))     # [B_q, B_p]
                                min_values, min_indices = torch.min(dis, dim=-1)       # (B_q,), (B_q,)
                                min_values, min_indices = min_values.cpu().data.numpy(), min_indices.cpu().data.numpy()
                                for i in range(B_q):
                                    if min_values[i] < min_dis[i]:
                                        min_dis[i] = min_values[i]
                                        min_dis_train_idx[i] = train_batch_idx * B_p + min_indices[i]
                            train_idx_list = np.append(train_idx_list, min_dis_train_idx)
                            train_dist_list = np.append(train_dist_list, min_dis)
                            
                        return train_idx_list, train_dist_list

                    train_idx_list, train_dist_list = find_nearest_neighbors(sliced_data['test']['joint3d'], sliced_data['train']['joint3d'])
                    
                    prompt_list[dataset_name] = list(zip(train_idx_list, train_dist_list))

                    if args.visualize:
                        if args.visualize.split(',')[0] == self.__class__.__name__:
                            print(f"Do visualizing in {self.__class__.__name__}: __init__ now...")
                            for i in range(len(prompt_list[dataset_name])):
                                q_seq = query_dict[dataset_name]['joint3d'][i]
                                nearest_id, nearest_dist = prompt_list[dataset_name][i]
                                p_seq = prompt_dict[dataset_name]['joint3d'][nearest_id]
                                viz_skel_seq_anim(
                                    {'q': q_seq[::4], 'p': p_seq[::4]}, mode='img', fig_title=f'{dataset_name} qid:{i} pid:{nearest_id}       dist:{nearest_dist:.4f}', 
                                    subplot_layout=(2,1), fs=0.3, tight_layout=True, lim3d=0.5
                                )

            elif args.get('fix_prompt', None) == 'TrainSame_TestNearest':
                if self.is_train:
                    prompt_list[dataset_name] = random.sample(range(num_prompt), num_prompt)
                    assert num_query == num_prompt
                    self.prompt_log[dataset_name] = {q_id: p_id for q_id, p_id in zip(range(num_query), prompt_list[dataset_name])}
                else:                    
                    def find_nearest_neighbors(test_data, train_data, batch_size=512):
                        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

                        train_idx_list = np.array([], dtype=int)
                        train_dist_list = np.array([])
                        for test_batch in test_loader:
                            test_batch = test_batch.cuda()
                            B_q = test_batch.shape[0]

                            min_dis = np.ones(B_q) * 1000000
                            min_dis_train_idx = np.zeros(B_q, dtype=int)
                            
                            for train_batch_idx, train_batch in enumerate(train_loader):
                                train_batch = train_batch.cuda()
                                B_p = train_batch.shape[0]

                                dis = torch.mean(torch.norm(test_batch[:, None, :, :, :] - train_batch[None, :, :, :, :], dim=-1), dim=(-2,-1))     # [B_q, B_p]
                                min_values, min_indices = torch.min(dis, dim=-1)       # (B_q,), (B_q,)
                                min_values, min_indices = min_values.cpu().data.numpy(), min_indices.cpu().data.numpy()
                                for i in range(B_q):
                                    if min_values[i] < min_dis[i]:
                                        min_dis[i] = min_values[i]
                                        min_dis_train_idx[i] = train_batch_idx * B_p + min_indices[i]
                            train_idx_list = np.append(train_idx_list, min_dis_train_idx)
                            train_dist_list = np.append(train_dist_list, min_dis)
                            
                        return train_idx_list, train_dist_list

                    train_idx_list, train_dist_list = find_nearest_neighbors(sliced_data['test']['joint3d'], sliced_data['train']['joint3d'])
                    
                    prompt_list[dataset_name] = list(zip(train_idx_list, train_dist_list))

            elif args.get('fix_prompt', None) == 'same_across_all_epochs':
                prompt_list[dataset_name] = random.sample(range(num_prompt), num_prompt)
            elif args.get('fix_prompt', None) is not None and args.fix_prompt.split(',')[0] == 'FPS_selected':
                if len(args.get('fix_prompt', None).split(',')) == 1:
                    num_key_prompts = sliced_data['train']['joint2d'].shape[0] // 10
                else:
                    num_key_prompts = int(args.get('fix_prompt', None).split(',')[1])

                fps_sorted_indices_file = os.path.join(presave_folder, 'fps_sorted_indices.npy')

                if not os.path.exists(fps_sorted_indices_file):
                    if kwargs.get('rank', 0) == 0:
                        _ = farthest_point_sampling(torch.from_numpy(sliced_data['train']['joint3d']).cuda(kwargs.get('rank', 0)), presave_folder, max_len=num_key_prompts)
                else:
                    fps_sorted_indices = np.load(fps_sorted_indices_file)
                
                
                prompt_list[dataset_name] = fps_sorted_indices[:num_key_prompts]
                num_prompt = len(prompt_list[dataset_name])
            else:
                prompt_list[dataset_name] = list(range(num_prompt))

            if 'MC' in self.task_dict[dataset_name]:
                joint_mask_presave_path = os.path.join(args.presave_folder, 'joint_masks', dataset_name,
                                                f'nframes{self.dataset_config[dataset_name].get("clip_len", self.clip_len) * 2} - samplestride{self.dataset_config[dataset_name]["sample_stride"]} - '
                                                +f'datastridetrain{self.dataset_config[dataset_name]["data_stride"]["train"]} - datastridetest{self.dataset_config[dataset_name]["data_stride"]["test"]} - '
                                                +f'numjoint{self.num_joint} - jointmaskratio{self.joint_mask_ratio} - '
                                                +f'filename_{os.path.splitext(os.path.basename(self.dataset_file[dataset_name]))[0]}', data_split,
                                                )
                joint_mask_presave_file = os.path.join(joint_mask_presave_path, 'joint_masks.pkl')
                joint_mask_config_file = os.path.join(joint_mask_presave_path, 'joint_mask_config.pkl')
                if not os.path.exists(joint_mask_presave_file):
                    if kwargs.get('rank', 0) == 0: print("Presaving joint masks...", end=' ')
                    os.makedirs(joint_mask_presave_path, exist_ok=True)
                    joint_masks = [random.sample(range(1,self.num_joint), int(self.joint_mask_ratio*self.num_joint)) for _ in range(num_query)]
                    joint_masks_config = {
                        'nframes' : self.dataset_config[dataset_name].get('clip_len', self.clip_len) * 2,
                        'samplestride' : self.dataset_config[dataset_name]['sample_stride'],
                        'datastridetrain' : self.dataset_config[dataset_name]['data_stride']["train"],
                        'datastridetest' : self.dataset_config[dataset_name]['data_stride']["test"],
                        'numsample' : num_query,
                        'numjoint' : self.num_joint,
                        'jointmaskratio' : self.joint_mask_ratio,
                        'filename' : os.path.splitext(os.path.basename(self.dataset_file[dataset_name]))[0],
                    }
                    with open(joint_mask_presave_file, 'wb') as f:
                        pickle.dump(joint_masks, f, protocol=4)
                    with open(joint_mask_config_file, 'wb') as f:
                        pickle.dump(joint_masks_config, f)
                else:
                    if kwargs.get('rank', 0) == 0: print("Loading joint masks...", end=' ')
                    with open(joint_mask_config_file, 'rb') as f:
                        joint_masks_config = pickle.load(f)
                    assert joint_masks_config == {
                        'nframes' : self.dataset_config[dataset_name].get('clip_len', self.clip_len) * 2,
                        'samplestride' : self.dataset_config[dataset_name]['sample_stride'],
                        'datastridetrain' : self.dataset_config[dataset_name]['data_stride']["train"],
                        'datastridetest' : self.dataset_config[dataset_name]['data_stride']["test"],
                        'numsample' : num_query,
                        'numjoint' : self.num_joint,
                        'jointmaskratio' : self.joint_mask_ratio,
                        'filename' : os.path.splitext(os.path.basename(self.dataset_file[dataset_name]))[0],
                    }
                    with open(joint_mask_presave_file, 'rb') as f:
                        joint_masks = pickle.load(f)
                joint_mask_dict[dataset_name] = joint_masks
            if 'MIB' in self.task_dict[dataset_name] or 'MeshInBetween' in self.task_dict[dataset_name]:
                frame_mask_presave_path = os.path.join(args.presave_folder, 'frame_masks', dataset_name, 
                                                f'nframes{self.dataset_config[dataset_name].get("clip_len", self.clip_len) * 2} - samplestride{self.dataset_config[dataset_name]["sample_stride"]} - '
                                                +f'datastridetrain{self.dataset_config[dataset_name]["data_stride"]["train"]} - datastridetest{self.dataset_config[dataset_name]["data_stride"]["test"]} - '
                                                +f'framemaskratio{self.frame_mask_ratio} - '
                                                +f'filename_{os.path.splitext(os.path.basename(self.dataset_file[dataset_name]))[0]}', data_split,
                                                )
                frame_mask_presave_file = os.path.join(frame_mask_presave_path, 'frame_masks.pkl')
                frame_mask_config_file = os.path.join(frame_mask_presave_path, 'frame_mask_config.pkl')
                if not os.path.exists(frame_mask_presave_file):
                    if kwargs.get('rank', 0) == 0: print("Presaving frame masks...", end=' ')
                    os.makedirs(frame_mask_presave_path, exist_ok=True)
                    frame_masks = [random.sample(range(self.clip_len), int(self.frame_mask_ratio*self.clip_len)) for _ in range(num_query)]
                    frame_masks_config = {
                        'nframes' : self.dataset_config[dataset_name].get('clip_len', self.clip_len) * 2,
                        'samplestride' : self.dataset_config[dataset_name]['sample_stride'],
                        'datastridetrain' : self.dataset_config[dataset_name]['data_stride']["train"],
                        'datastridetest' : self.dataset_config[dataset_name]['data_stride']["test"],
                        'numsample' : num_query,
                        'framemaskratio' : self.frame_mask_ratio,
                        'filename' : os.path.splitext(os.path.basename(self.dataset_file[dataset_name]))[0],
                    }
                    with open(frame_mask_presave_file, 'wb') as f:
                        pickle.dump(frame_masks, f, protocol=4)
                    with open(frame_mask_config_file, 'wb') as f:
                        pickle.dump(frame_masks_config, f)
                else:
                    if kwargs.get('rank', 0) == 0: print("Loading frame masks...", end=' ')
                    with open(frame_mask_config_file, 'rb') as f:
                        frame_masks_config = pickle.load(f)
                    assert frame_masks_config == {
                        'nframes' : self.dataset_config[dataset_name].get('clip_len', self.clip_len) * 2,
                        'samplestride' : self.dataset_config[dataset_name]['sample_stride'],
                        'datastridetrain' : self.dataset_config[dataset_name]['data_stride']["train"],
                        'datastridetest' : self.dataset_config[dataset_name]['data_stride']["test"],
                        'numsample' : num_query,
                        'framemaskratio' : self.frame_mask_ratio,
                        'filename' : os.path.splitext(os.path.basename(self.dataset_file[dataset_name]))[0],
                    }
                    with open(frame_mask_presave_file, 'rb') as f:
                        frame_masks = pickle.load(f)
                frame_mask_dict[dataset_name] = frame_masks

            if kwargs.get('rank', 0) == 0: print(f"costs {time.time()-st:.2f}s... has {num_query}/{num_prompt} query/prompt samples")

        self.query_list = query_list
        self.query_dict = query_dict
        self.prompt_list = prompt_list
        self.prompt_dict = prompt_dict
        self.joint_mask_dict = joint_mask_dict
        self.frame_mask_dict = frame_mask_dict

        self.video_augmentor = VideoAugmentor()

        # PW3D; H36M
        # self.smpl = SMPL('third_party/motionbert/data/mesh', batch_size=1)

        # AMASS
        # self.J_reg_amass_to_h36m = np.load('third_party/motionbert/data/AMASS/J_regressor_h36m_correct.npy')
        # self.real2cam = np.array([[1, 0, 0], 
        #              [0, 0, 1], 
        #              [0, -1, 0]], dtype=np.float64)
        
    def __len__(self):
        return len(self.query_list)
    
    def prepare_chunk(self, data_dict, dataset_name, chunk_id=None):
        chunk_dict = OrderedDict()

        if chunk_id is None:
            chunk_slice = slice(None)
        elif isinstance(chunk_id, int):
            chunk_slice = slice(chunk_id, chunk_id+1)
        elif isinstance(chunk_id, list) or isinstance(chunk_id, np.ndarray) or isinstance(chunk_id, torch.Tensor):
            chunk_slice = chunk_id



        for mode in data_dict[dataset_name].keys():
            chunk = data_dict[dataset_name][mode][chunk_slice].copy()
            chunk = torch.from_numpy(chunk).float()

            if mode == 'joint2d':
                if self.is_train and self.aug:
                    if dataset_name == 'AMASS':
                        chunk = self.video_augmentor(chunk)
                    chunk = crop_scale(chunk.clone(), scale_range=[0.5, 1])
            elif mode == 'joint3d':
                if dataset_name == 'H36M_3D':
                    chunk = chunk / self.dataset_config['H36M_3D']['scale_3D']
                if self.is_train and self.aug:
                    chunk = crop_scale_3d(chunk.clone(), scale_range=[0.5, 1])
                
            chunk_dict[mode] = chunk
        
        return chunk_dict
    
    def prepare_motion(self, chunk_dict, dataset_name, task, joint_mask, frame_mask):
        # chunk_dict
        #   'joint2d': (1,T,17,3) or (N,T,17,3); 
        #   'joint3d': (1,T,17,3) or (N,T,17,3);
        #   'smpl_pose': (1,T,72) or (N,T,72);
        #   'smpl_shape': (1,T,10) or (N,T,10);

        if hasattr(self.args, 'prepare_motion_function'):
            prepare_motion_func = getattr(self.args, 'prepare_motion_function')
            return prepare_motion_func(chunk_dict, dataset_name, task, joint_mask, frame_mask, self.args)

        input_dict = {}
        target_dict = {}

        if self.current_as_history:
            indices = slice(None, self.clip_len)
        else:
            indices = slice(self.clip_len, None)

        for mode in chunk_dict.keys():
            if task == 'PE':
                input_dict[mode] = chunk_dict[mode][:, indices].clone()
                target_dict[mode] = chunk_dict[mode][:, indices].clone()
            elif task in ['FPE', 'MP']:
                input_dict[mode] = chunk_dict[mode][:, :self.clip_len].clone()
                target_dict[mode] = chunk_dict[mode][:, self.clip_len:].clone()
            elif task == 'MC':
                input_dict[mode] = chunk_dict[mode][:, indices].clone()
                target_dict[mode] = chunk_dict[mode][:, indices].clone()
                if mode == 'joint3d':
                    assert joint_mask is not None
                    input_dict['joint3d'][:, :, joint_mask] = 0
            elif task == 'MIB':
                input_dict[mode] = chunk_dict[mode][:, indices].clone()
                target_dict[mode] = chunk_dict[mode][:, indices].clone()
                if mode == 'joint3d':
                    assert frame_mask is not None
                    input_dict['joint3d'][:, frame_mask] = 0
            elif task == ['COPY3D', 'COPY2D']:
                input_dict[mode] = chunk_dict[mode][:, indices].clone()
                target_dict[mode] = chunk_dict[mode][:, indices].clone()
            elif task in ['FPEhis', 'MPhis', 'MP2D', 'MC2D', 'MIB2D']:
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown task: {task}")

        # input_dict, target_dict = self.preprocess(input_dict, target_dict, dataset_name)

        return input_dict, target_dict

    def preprocess(self, data_dict):
        
        if hasattr(self.args, 'preprocess_function'):
            preprocess_func = getattr(self.args, 'preprocess_function')
            return preprocess_func(data_dict)
        
        data_dict['joint2d'] = data_dict['joint2d'] - data_dict['joint2d'][:, :, 0:1, :]
        if 'smpl_vertex' in data_dict:
            data_dict['smpl_vertex'] = data_dict['smpl_vertex'] - data_dict['joint3d'][:, :, 0:1, :]
        data_dict['joint3d'] = data_dict['joint3d'] - data_dict['joint3d'][:, :, 0:1, :]
        return data_dict
    
    def postprocess(self, motion_seq, dataset_name, task):
        if dataset_name == 'H36M_3D':
            if task == 'PE' and self.dataset_config['H36M_3D']['rootrel_target']:
                motion_seq[..., 0, :] = 0
            motion_seq = motion_seq * self.dataset_config['H36M_3D']['scale_3D']
        else:
            if self.dataset_config[dataset_name]['rootrel_target']:
                motion_seq = motion_seq - motion_seq[..., 0:1, :]
        return motion_seq

    def __getitem__(self, query_index):
        dataset_name, query_chunk_id = self.query_list[query_index]
        if self.args.get('fix_prompt', None) == 'same_across_all_epochs':
            prompt_chunk_id = query_chunk_id % len(self.prompt_list[dataset_name])
        elif self.args.get('fix_prompt', None) == 'TrainRandom_TestNearest':
            if self.is_train:
                prompt_chunk_id = random.choice(self.prompt_list[dataset_name])
                self.prompt_log[query_index] = prompt_chunk_id
            else:
                train_chunk_id, train_chunk_distance = self.prompt_list[dataset_name][query_chunk_id]
                prompt_chunk_id = self.prompt_log[train_chunk_id]
        elif self.args.get('fix_prompt', None) == 'TrainSame_TestNearest':
            if self.is_train:
                prompt_chunk_id = query_chunk_id
            else:
                train_chunk_id, train_chunk_distance = self.prompt_list[dataset_name][query_chunk_id]
                prompt_chunk_id = self.prompt_log[dataset_name][train_chunk_id]
        elif self.args.get('fix_prompt', None) is not None and self.args.fix_prompt.split(',')[0] == 'FPS_selected':
            prompt_chunk_id = self.prompt_list[dataset_name]

        else:
            prompt_chunk_id = random.choice(self.prompt_list[dataset_name])

        query_chunk_dict = self.prepare_chunk(self.query_dict, dataset_name, chunk_id=query_chunk_id)
        prompt_chunk_dict = self.prepare_chunk(self.prompt_dict, dataset_name, chunk_id=prompt_chunk_id)

        if self.is_train and self.aug_shuffle_joints:
            raise NotImplementedError

        QUERY_INPUT_DICT = defaultdict(list)
        QUERY_TARGET_DICT = defaultdict(list)
        PROMPT_INPUT_DICT = defaultdict(list)
        PROMPT_TARGET_DICT = defaultdict(list)
        INFO_DICT = defaultdict(list)

        for task in self.task_dict[dataset_name]:
            joint_mask, frame_mask = None, None
            if task == 'MC':
                joint_mask = self.joint_mask_dict[dataset_name][query_chunk_id]
            if task == 'MIB':
                frame_mask = self.frame_mask_dict[dataset_name][query_chunk_id]

            query_input, query_target = self.prepare_motion(query_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask)
            prompt_input, prompt_target = self.prepare_motion(prompt_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask)

            assert query_input.keys() == query_target.keys() == prompt_input.keys() == prompt_target.keys()
            for mode in query_target.keys():
                QUERY_INPUT_DICT[mode].append(query_input[mode])
                QUERY_TARGET_DICT[mode].append(query_target[mode])
                PROMPT_INPUT_DICT[mode].append(prompt_input[mode])
                PROMPT_TARGET_DICT[mode].append(prompt_target[mode])

            INFO_DICT['dataset'].append(dataset_name)
            INFO_DICT['task'].append(task)
            INFO_DICT['joint_mask'].append(joint_mask)
            INFO_DICT['frame_mask'].append(frame_mask)
            INFO_DICT['query_chunk_id'].append(query_chunk_id)
            INFO_DICT['prompt_chunk_id'].append(prompt_chunk_id)
            INFO_DICT['query_index'].append(query_index)
            INFO_DICT['use_global_orient'].append(int(self.dataset_config[dataset_name]['use_global_orient']))

            if self.visualize == self.__class__.__name__:
                if dataset_name == 'PW3D_MESH' and task == 'MP':
                    print(f"Do visualizing in {self.__class__.__name__}...")
            
        if self.is_train and self.dumb_task:
            for dumb_task in self.dumb_task.split(','):
                raise NotImplementedError

        for mode in QUERY_TARGET_DICT.keys():
            QUERY_INPUT_DICT[mode] = torch.cat(QUERY_INPUT_DICT[mode])
            QUERY_TARGET_DICT[mode] = torch.cat(QUERY_TARGET_DICT[mode])
            PROMPT_INPUT_DICT[mode] = torch.cat(PROMPT_INPUT_DICT[mode])
            PROMPT_TARGET_DICT[mode] = torch.cat(PROMPT_TARGET_DICT[mode])
        
        return QUERY_INPUT_DICT, QUERY_TARGET_DICT, PROMPT_INPUT_DICT, PROMPT_TARGET_DICT, INFO_DICT


def collate_func(batch):    # batch: list, len=batch_size. list element: tuple containing returned values from __getitem__
    batch_size = len(batch)

    QUERY_INPUT_DICT = defaultdict(list)
    QUERY_TARGET_DICT = defaultdict(list)
    PROMPT_INPUT_DICT = defaultdict(list)
    PROMPT_TARGET_DICT = defaultdict(list)
    INFO_DICT = defaultdict(list)
    
    for b in range(batch_size):
        for mode in batch[0][1].keys():
            QUERY_INPUT_DICT[mode].append(batch[b][0][mode])
            QUERY_TARGET_DICT[mode].append(batch[b][1][mode])
            PROMPT_INPUT_DICT[mode].append(batch[b][2][mode])
            PROMPT_TARGET_DICT[mode].append(batch[b][3][mode])
        for info_key in batch[0][4].keys():
            INFO_DICT[info_key] = INFO_DICT[info_key] + batch[b][4][info_key]

    for mode in QUERY_TARGET_DICT.keys():
        QUERY_INPUT_DICT[mode] = torch.cat(QUERY_INPUT_DICT[mode])
        QUERY_TARGET_DICT[mode] = torch.cat(QUERY_TARGET_DICT[mode])
        PROMPT_INPUT_DICT[mode] = torch.cat(PROMPT_INPUT_DICT[mode])
        PROMPT_TARGET_DICT[mode] = torch.cat(PROMPT_TARGET_DICT[mode])

    return QUERY_INPUT_DICT, QUERY_TARGET_DICT, PROMPT_INPUT_DICT, PROMPT_TARGET_DICT, INFO_DICT
        

class DataReaderMesh(object):
    def __init__(self, dataset_name, split, n_frames, sample_stride, data_stride, read_confidence,
                 dt_root='', dt_file='', res=[1920, 1920],
                 use_global_orient=True, return_skel3d=True, return_smpl=True,
                 **kwargs):
        
        self.dt_root = dt_root
        self.dt_file = dt_file

        self.dataset_name = dataset_name

        self.split = split
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride = data_stride
        self.read_confidence = read_confidence
        self.res = res

        self.use_global_orient = use_global_orient
        self.return_skel3d = return_skel3d
        self.return_smpl = return_smpl

        self.split_id = None
        self.dt_dataset = None

        self.rank = kwargs.get('rank', 0)
    
    def get_amass_dataset(self):
        dt_dataset = {}
        for split in ['train', 'test']:
            data_dir = os.path.join(self.dt_file, split)
            all_dicts = []
            for subdir, subdataset_names, _ in os.walk(data_dir):
                for subdataset_name in subdataset_names:
                    for _, _, files in os.walk(os.path.join(subdir, subdataset_name)):
                        for file in files:
                            file_path = os.path.join(subdir, subdataset_name, file)
                            data_dict = read_pkl(file_path)
                            all_dicts.append(data_dict)
            concatenated_dict = {}
            for d in all_dicts:
                for key, value in d.items():
                    if key not in concatenated_dict:
                        concatenated_dict[key] = value
                    else:
                        concatenated_dict[key] = np.concatenate((concatenated_dict[key], value), axis=0)
            dt_dataset[split] = concatenated_dict

            dt_dataset[split]['joint_2d'] = dt_dataset[split]['joint_3d'][..., :2].copy()

        self.dt_dataset = dt_dataset

    def get_num_clips(self, designated_split):
        return len(self.get_split_id(designated_split))

    def get_split_id(self, designated_split=None):

        if self.dt_dataset is None:
            self.load_dataset()

        if designated_split is not None:
            vid_list = self.dt_dataset[designated_split]['source'][::self.sample_stride]
            split_id = split_clips(vid_list, self.n_frames, self.data_stride[designated_split])
            return split_id

        if self.split_id is not None:
            return self.split_id
        vid_list = self.dt_dataset[self.split]['source'][::self.sample_stride]
        self.split_id = split_clips(vid_list, self.n_frames, self.data_stride[self.split])
        return self.split_id

    def read_2d(self, designated_split=None):
        split = designated_split if designated_split is not None else self.split

        joints_2d = self.dt_dataset[split]['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]

        if self.res is not None:
            res_w, res_h = self.res
            offset = [1, res_h / res_w]
            joints_2d = joints_2d / res_w * 2 - offset
        elif 'img_hw' in self.dt_dataset[split].keys():
            res = np.array(self.dt_dataset[split]['img_hw'])[::self.sample_stride].astype(np.float32)
            res_w, res_h = res.max(1)[:, None, None], res.max(1)[:, None, None]
            offset = 1
            joints_2d = joints_2d / res_w * 2 - offset
        elif 'camera_name' in self.dt_dataset[split].keys():
            for idx, camera_name in enumerate(self.dt_dataset[split]['camera_name']):
                if camera_name == '54138969' or camera_name == '60457274':
                    res_w, res_h = 1000, 1002
                elif camera_name == '55011271' or camera_name == '58860488':
                    res_w, res_h = 1000, 1000
                else:
                    assert 0, '%d data item has an invalid camera name' % idx
                joints_2d[idx, :, :] = joints_2d[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        else:
            # raise ValueError('No resolution information provided')
            if designated_split == 'train': 
                if self.rank == 0: print('(No resolution information provided for normalizing 2D joints...)', end=' ')

        if self.read_confidence:
            if 'confidence' in self.dt_dataset[split]:
                dataset_confidence = self.dt_dataset[split]['confidence'][::self.sample_stride].astype(np.float32)
            else:
                dataset_confidence = np.ones_like(joints_2d[..., :1])
            if len(dataset_confidence.shape)==2: 
                dataset_confidence = dataset_confidence[:,:,None]
            joints_2d = np.concatenate((joints_2d, dataset_confidence), axis=-1)  # [N, 17, 3]
        else:
            joints_2d = np.concatenate((joints_2d, np.zeros_like(joints_2d[..., :1])), axis=-1)
        return joints_2d
    
    def read_smpl(self, designated_split=None):
        split = designated_split if designated_split is not None else self.split

        smpl_pose = self.dt_dataset[split]['smpl_pose'][::self.sample_stride, :]
        smpl_shape = self.dt_dataset[split]['smpl_shape'][::self.sample_stride, :]
        
        # smpl_vertex = []
        # batch_size = 512
        # for i in tqdm(range(0, len(smpl_pose), batch_size)):
        #     batch_pose = smpl_pose[i:i+batch_size]
        #     batch_shape = smpl_shape[i:i+batch_size]
        #     motion_smpl = self.smpl(
        #         betas=torch.from_numpy(batch_shape).float(),
        #         body_pose=torch.from_numpy(batch_pose).float()[:, 3:],
        #         global_orient=torch.from_numpy(batch_pose).float()[:, :3] if self.use_global_orient else torch.zeros_like(torch.from_numpy(batch_pose).float()[:, :3]),
        #         pose2rot=True
        #     )
        #     smpl_vertex.append(motion_smpl.vertices.detach().numpy())
        # smpl_vertex = np.concatenate(smpl_vertex, axis=0)  # [N, 6890, 3]

        return smpl_pose, smpl_shape

    def read_3d(self, designated_split=None):
        split = designated_split if designated_split is not None else self.split

        if self.use_global_orient:
            joints_3d = self.dt_dataset[split]['joint_3d'][::self.sample_stride, :, :]  # [N, 17, 3]
        else:
            joints_3d = self.dt_dataset[split]['joint_3d_wo_GlobalOrient'][::self.sample_stride, :, :]  # [N, 17, 3]
        return joints_3d
    
    def get_split_data(self, designated_split=None):

        if self.dt_dataset is None:
            self.load_dataset()

        split_id = self.get_split_id(designated_split)

        joints_2d = self.read_2d(designated_split)  # [N, 17, 3]
        joints_2d = joints_2d[split_id]  # [n, T, 17, 3]
        assert len(split_id) == len(joints_2d)
        assert (not self.read_confidence) == (joints_2d[..., -1] == 0).all()

        data_dict = {
            'joint2d': joints_2d
        }

        if self.return_skel3d:
            joints_3d = self.read_3d(designated_split)  # [N, 17, 3]
            joints_3d = joints_3d[split_id]  # [n, T, 17, 3]
            assert len(joints_2d) == len(joints_3d)
            data_dict['joint3d'] = joints_3d
        if self.return_smpl:
            smpl_pose, smpl_shape = self.read_smpl(designated_split) # [N, 72], [N, 10], [N, 6890, 3]
            smpl_pose, smpl_shape = smpl_pose[split_id], smpl_shape[split_id]    # [n, T, 72], [n, T, 10]
            assert len(joints_2d) == len(smpl_pose)
            data_dict['smpl_pose'] = smpl_pose
            data_dict['smpl_shape'] = smpl_shape
        return data_dict
    
    def load_dataset(self):
        if self.dataset_name == 'AMASS':
            self.get_amass_dataset()
        else:
            if self.dt_file.endswith('.pt'):
                self.dt_dataset = joblib.load(os.path.join(self.dt_root, self.dt_file))
            elif self.dt_file.endswith('.pkl'):
                self.dt_dataset = read_pkl(os.path.join(self.dt_root, self.dt_file))

    def get_all_data(self):
        
        if self.dt_dataset is None:
            self.load_dataset()

        return_dict = {}
        if 'train' in self.dt_dataset:
            train_data_dict = self.get_split_data(designated_split='train')
            return_dict['train'] = train_data_dict
        if 'test' in self.dt_dataset:
            test_data_dict = self.get_split_data(designated_split='test')
            return_dict['test'] = test_data_dict
        return return_dict


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

