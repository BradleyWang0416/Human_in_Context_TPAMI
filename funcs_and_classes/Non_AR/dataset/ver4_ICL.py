from cgi import test
import chunk
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
import joblib

from third_party.Pose2Mesh.data.COCO import dataset
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
from data_gen.angle2joint import ang2joint


class MotionDatasetICL(Dataset):
    def __init__(self, args, data_split, TASK=None, DATASET_NAME=None, SLICED_DATA=None):
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
        
        self.visualize = args.visualize
        
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
        self.use_task_id_as_prompt = args.use_task_id_as_prompt
        self.normalize_2d = args.normalize_2d
        if self.is_train: print(f'\tDataset global attribute [normalize_2d={self.normalize_2d}]')
        self.normalize_3d = args.normalize_3d
        if self.is_train: print(f'\tDataset global attribute [normalize_3d={self.normalize_3d}]')
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
        if self.is_train: self.sliced_data_dict = {}
        for dataset_name in self.datasets:
            print(f"\tLoading {data_split} data from [{dataset_name}] for task: {self.task_dict[dataset_name]}...", end=' ')
            st = time.time()

            if SLICED_DATA is None:
                dt_file = self.dataset_file[dataset_name]
                n_frames = self.dataset_config[dataset_name].get('clip_len', self.clip_len) * 2
                if dataset_name in ['H36M_MESH', 'PW3D_MESH']:
                    datareader = DataReaderMesh(split = data_split,
                                                n_frames = n_frames, 
                                                sample_stride = self.dataset_config[dataset_name]['sample_stride'], 
                                                data_stride = self.dataset_config[dataset_name]['data_stride'],
                                                read_confidence = self.dataset_config[dataset_name]['read_confidence'],
                                                dt_root = '', 
                                                dt_file = dt_file,
                                                res = [1920, 1920] if dataset_name == 'PW3D_MESH' else None,
                                                return_type = self.dataset_config[dataset_name]['return_type'], # 'joint' or 'smpl' or 'joint_x1000' or 'smpl_x1000' or 'all' or 'all_x1000'
                                                use_global_orient = self.dataset_config[dataset_name]['use_global_orient'])
                elif dataset_name == 'H36M_MESH_TCMR':
                    datareader = DataReaderMesh_H36M_TCMR(split = data_split,
                                                            n_frames = n_frames, 
                                                            sample_stride = self.dataset_config[dataset_name]['sample_stride'], 
                                                            data_stride = self.dataset_config[dataset_name]['data_stride'],
                                                            read_confidence = self.dataset_config[dataset_name]['read_confidence'],
                                                            dt_root = '', 
                                                            dt_file = dt_file,
                                                            res = [1920, 1920] if dataset_name == 'PW3D_MESH' else None,
                                                            return_type = self.dataset_config[dataset_name]['return_type'], # 'joint' or 'smpl' or 'joint_x1000' or 'smpl_x1000' or 'all' or 'all_x1000'
                                                            use_global_orient = self.dataset_config[dataset_name]['use_global_orient'])
                
                elif dataset_name == 'H36M_3D':
                    raise NotImplementedError
                elif dataset_name == 'AMASS':
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown dataset name: {dataset_name}")


                if args.use_presave_data:
                    def get_class_attributes(obj):
                        return {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith('__') and not callable(getattr(obj, attr)) and not 'split' in attr and not 'dataset' in attr}

                    presave_folder = os.path.join(args.presave_folder, os.path.splitext(os.path.basename(__file__))[0], dataset_name,
                                                f'nframes{datareader.n_frames} - samplestride{datareader.sample_stride} - '
                                                +f'datastridetrain{datareader.data_stride["train"]} - datastridetest{datareader.data_stride["test"]} - '
                                                +f'readconfidence{int(datareader.read_confidence)} - '
                                                +f'useglobalorient{int(datareader.use_global_orient)} - '
                                                +f'returntype_{datareader.return_type} - '
                                                +f'filename_{os.path.basename(dt_file)}'
                                                )

                    presave_file = os.path.join(presave_folder, 'sliced_data.pkl')
                    datareader_config_file = os.path.join(presave_folder, 'datareader_config.pkl')
                    if not os.path.exists(presave_file):
                        print("Presaving...", end=' ')
                        os.makedirs(presave_folder, exist_ok=True)
                        sliced_data = datareader.get_all_data()   # this step will change the self.split_id from None to the actual split_ids
                        datareader_config = get_class_attributes(datareader)
                        with open(presave_file, 'wb') as f:
                            pickle.dump(sliced_data, f, protocol=4)
                        with open(datareader_config_file, 'wb') as f:
                            pickle.dump(datareader_config, f)
                    else:
                        print("Loading presaved...", end=' ')
                        with open(datareader_config_file, 'rb') as f:
                            datareader_config = pickle.load(f)
                        assert datareader_config == get_class_attributes(datareader)
                        with open(presave_file, 'rb') as f:
                            sliced_data = pickle.load(f)
                else:
                    sliced_data = datareader.get_all_data()  # this step will change the self.split_id from None to the actual split_ids
                
                if self.is_train: self.sliced_data_dict[dataset_name] = sliced_data
            
            else:
                sliced_data = SLICED_DATA

            num_query = sliced_data[data_split]['num_sample']
            query_dict[dataset_name] = sliced_data[data_split]
            query_list.extend(zip([dataset_name]*num_query, list(range(num_query))))

            num_prompt = sliced_data['train']['num_sample']
            prompt_dict[dataset_name] = sliced_data['train']
            if args.get('fix_prompt', None) == 'largest_velo':
                chunk_3d_train_velocity = np.diff(prompt_dict[dataset_name]['joint3d'], axis=1)
                chunk_magnitude = np.linalg.norm(chunk_3d_train_velocity, axis=-1)
                chunk_magnitude_avg = np.mean(chunk_magnitude, axis=(-2,-1))
                top_chunk_id = np.argsort(chunk_magnitude_avg)[-1]
                prompt_list[dataset_name] = [top_chunk_id]
            else:
                prompt_list[dataset_name] = list(range(num_prompt))

            if 'MC' in self.task_dict[dataset_name]:
                joint_mask_dict[dataset_name] = [random.sample(range(1,self.num_joint), int(self.joint_mask_ratio*self.num_joint)) for _ in range(num_query)]
            if 'MIB' in self.task_dict[dataset_name]:
                frame_mask_dict[dataset_name] = [random.sample(range(self.clip_len), int(self.frame_mask_ratio*self.clip_len)) for _ in range(num_query)]

            print(f"costs {time.time()-st:.2f}s... has {num_query}/{num_prompt} query/prompt samples")

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
        
    def __len__(self):
        return len(self.query_list)
    
    def prepare_chunk(self, data_dict, dataset_name, chunk_id=None):
        chunk_dict = {}

        # 2D JOINTS
        chunk_joint2d = data_dict[dataset_name]['joint2d'][chunk_id].copy() if chunk_id is not None else data_dict[dataset_name]['joint2d'].copy()
        chunk_joint2d = torch.from_numpy(chunk_joint2d).float()
        if self.is_train and self.aug:
            if dataset_name == 'AMASS':
                chunk_joint2d = self.video_augmentor(chunk_joint2d)
            chunk_joint2d = crop_scale(chunk_joint2d.clone(), scale_range=[0.5, 1])
        chunk_dict['joint2d'] = chunk_joint2d

        # 3D JOINTS
        chunk_joint3d = data_dict[dataset_name]['joint3d'][chunk_id].copy() if chunk_id is not None else data_dict[dataset_name]['joint3d'].copy()
        chunk_joint3d = torch.from_numpy(chunk_joint3d).float()
        if dataset_name == 'H36M_3D':
            chunk_joint3d = chunk_joint3d / self.dataset_config['H36M_3D']['scale_3D']
        if self.is_train and self.aug:
            chunk_joint3d = crop_scale(chunk_joint3d.clone(), scale_range=[0.5, 1])
        chunk_dict['joint3d'] = chunk_joint3d

        # SMPL
        if 'smpl_pose' in data_dict[dataset_name]:
            chunk_smpl_pose = data_dict[dataset_name]['smpl_pose'][chunk_id].copy() if chunk_id is not None else data_dict[dataset_name]['smpl_pose'].copy()
            chunk_smpl_pose = torch.from_numpy(chunk_smpl_pose).float()
            chunk_dict['smpl_pose'] = chunk_smpl_pose

        if 'smpl_shape' in data_dict[dataset_name]:
            chunk_smpl_shape = data_dict[dataset_name]['smpl_shape'][chunk_id].copy() if chunk_id is not None else data_dict[dataset_name]['smpl_shape'].copy()
            chunk_smpl_shape = torch.from_numpy(chunk_smpl_shape).float()
            chunk_dict['smpl_shape'] = chunk_smpl_shape

        if 'smpl_vertices' in data_dict[dataset_name]:
            chunk_smpl_vertices = data_dict[dataset_name]['smpl_vertices'][chunk_id].copy() if chunk_id is not None else data_dict[dataset_name]['smpl_vertices'].copy()
            chunk_smpl_vertices = torch.from_numpy(chunk_smpl_vertices).float()
            chunk_dict['smpl_vertices'] = chunk_smpl_vertices
        elif self.dataset_config[dataset_name]['return_type'] in ['smpl', 'smpl_x1000', 'all', 'all_x1000']:
            chunk_smpl = self.smpl(
                betas=chunk_smpl_shape,
                body_pose=chunk_smpl_pose[:, 3:],
                global_orient=chunk_smpl_pose[:, :3] if self.dataset_config[dataset_name]['use_global_orient'] else torch.zeros_like(chunk_smpl_pose[:, :3]),
                pose2rot=True
            )
            chunk_dict['smpl_vertices'] = chunk_smpl.vertices.detach()
        return chunk_dict
    
    def prepare_motion(self, chunk_dict, dataset_name, task, joint_mask, frame_mask):
        # chunk_dict
        #   'joint2d': (T,17,3) or (N,T,17,3); 
        #   'joint3d': (T,17,3) or (N,T,17,3)
        #   'smpl_pose': (T,72) or (N,T,72);
        #   'smpl_shape': (T,10) or (N,T,10);
        #   'smpl_vertices': (T,6890,3) or (N,T,6890,3)

        pose_output, shape_output, vertex_output = None, None, None
        if task == 'PE':
            joint_input = chunk_dict['joint2d'][..., :self.clip_len, :, :].clone() if self.current_as_history else chunk_dict['joint2d'][..., self.clip_len:, :, :].clone()

            joint_output = chunk_dict['joint3d'][..., :self.clip_len, :, :].clone() if self.current_as_history else chunk_dict['joint3d'][..., self.clip_len:, :, :].clone()
            if 'smpl_pose' in chunk_dict: pose_output = chunk_dict['smpl_pose'][..., :self.clip_len, :].clone() if self.current_as_history else chunk_dict['smpl_pose'][..., self.clip_len:, :].clone()
            if 'smpl_shape' in chunk_dict: shape_output = chunk_dict['smpl_shape'][..., :self.clip_len, :].clone() if self.current_as_history else chunk_dict['smpl_shape'][..., self.clip_len:, :].clone()
            if 'smpl_vertices' in chunk_dict: vertex_output = chunk_dict['smpl_vertices'][..., :self.clip_len, :, :].clone() if self.current_as_history else chunk_dict['smpl_vertices'][..., self.clip_len:, :, :].clone()
        elif task == 'FPE':
            joint_input = chunk_dict['joint2d'][..., :self.clip_len, :, :].clone()

            joint_output = chunk_dict['joint3d'][..., self.clip_len:, :, :].clone()
            if 'smpl_pose' in chunk_dict: pose_output = chunk_dict['smpl_pose'][..., self.clip_len:, :].clone()
            if 'smpl_shape' in chunk_dict: shape_output = chunk_dict['smpl_shape'][..., self.clip_len:, :].clone()
            if 'smpl_vertices' in chunk_dict: vertex_output = chunk_dict['smpl_vertices'][..., self.clip_len:, :, :].clone()
        elif task == 'MP':
            joint_input = chunk_dict['joint3d'][..., :self.clip_len, :, :].clone()

            joint_output = chunk_dict['joint3d'][..., self.clip_len:, :, :].clone()
            if 'smpl_pose' in chunk_dict: pose_output = chunk_dict['smpl_pose'][..., self.clip_len:, :].clone()
            if 'smpl_shape' in chunk_dict: shape_output = chunk_dict['smpl_shape'][..., self.clip_len:, :].clone()
            if 'smpl_vertices' in chunk_dict: vertex_output = chunk_dict['smpl_vertices'][..., self.clip_len:, :, :].clone()
        elif task == 'MC':
            joint_output = chunk_dict['joint3d'][..., :self.clip_len, :, :].clone() if self.current_as_history else chunk_dict['joint3d'][..., self.clip_len:, :, :].clone()
            if 'smpl_pose' in chunk_dict: pose_output = chunk_dict['smpl_pose'][..., :self.clip_len, :].clone() if self.current_as_history else chunk_dict['smpl_pose'][..., self.clip_len:, :].clone()
            if 'smpl_shape' in chunk_dict: shape_output = chunk_dict['smpl_shape'][..., :self.clip_len, :].clone() if self.current_as_history else chunk_dict['smpl_shape'][..., self.clip_len:, :].clone()
            if 'smpl_vertices' in chunk_dict: vertex_output = chunk_dict['smpl_vertices'][..., :self.clip_len, :, :].clone() if self.current_as_history else chunk_dict['smpl_vertices'][..., self.clip_len:, :, :].clone()

            joint_input = joint_output.clone()
            assert joint_mask is not None
            joint_input[..., joint_mask, :] = 0
        elif task == 'MIB':
            joint_output = chunk_dict['joint3d'][..., :self.clip_len, :, :].clone() if self.current_as_history else chunk_dict['joint3d'][..., self.clip_len:, :, :].clone()
            if 'smpl_pose' in chunk_dict: pose_output = chunk_dict['smpl_pose'][..., :self.clip_len, :].clone() if self.current_as_history else chunk_dict['smpl_pose'][..., self.clip_len:, :].clone()
            if 'smpl_shape' in chunk_dict: shape_output = chunk_dict['smpl_shape'][..., :self.clip_len, :].clone() if self.current_as_history else chunk_dict['smpl_shape'][..., self.clip_len:, :].clone()
            if 'smpl_vertices' in chunk_dict: vertex_output = chunk_dict['smpl_vertices'][..., :self.clip_len, :, :].clone() if self.current_as_history else chunk_dict['smpl_vertices'][..., self.clip_len:, :, :].clone()

            joint_input = joint_output.clone()
            assert frame_mask is not None
            joint_input[..., frame_mask, :, :] = 0

        elif task == 'COPY':
            joint_input = chunk_dict['joint3d'][..., :self.clip_len, :, :] if self.current_as_history else chunk_dict['joint3d'][..., self.clip_len:, :, :]

            joint_output = joint_input.clone()
            if 'smpl_pose' in chunk_dict: pose_output = chunk_dict['smpl_pose'][..., :self.clip_len, :].clone() if self.current_as_history else chunk_dict['smpl_pose'][..., self.clip_len:, :].clone()
            if 'smpl_shape' in chunk_dict: shape_output = chunk_dict['smpl_shape'][..., :self.clip_len, :].clone() if self.current_as_history else chunk_dict['smpl_shape'][..., self.clip_len:, :].clone()
            if 'smpl_vertices' in chunk_dict: vertex_output = chunk_dict['smpl_vertices'][..., :self.clip_len, :, :].clone() if self.current_as_history else chunk_dict['smpl_vertices'][..., self.clip_len:, :, :].clone()
        elif task == 'COPY2D':
            raise NotImplementedError
            joint_input = chunk_dict['joint2d'][..., :self.clip_len, :, :] if self.current_as_history else chunk_dict['joint2d'][..., self.clip_len:, :, :]
            joint_output = joint_input.clone()
        elif task == 'FPEhis':
            raise NotImplementedError
            joint_input = chunk_dict['joint2d'][..., self.clip_len:, :, :]
            joint_output = chunk_dict['joint3d'][..., :self.clip_len, :, :]
        elif task == 'MPhis':
            raise NotImplementedError
            joint_input = chunk_dict['joint3d'][..., self.clip_len:, :, :]
            joint_output = chunk_dict['joint3d'][..., :self.clip_len, :, :]
        elif task == 'MP2D':
            raise NotImplementedError
            joint_input = chunk_dict['joint2d'][..., :self.clip_len, :, :]
            joint_output = chunk_dict['joint2d'][..., self.clip_len:, :, :]
        elif task == 'MC2D':
            raise NotImplementedError
            joint_input = chunk_dict['joint2d'][..., :self.clip_len, :, :] if self.current_as_history else chunk_dict['joint2d'][..., self.clip_len:, :, :]
            joint_output = joint_input.clone()
            assert joint_mask is not None
            joint_input[..., joint_mask, :] = 0
        elif task == 'MIB2D':
            raise NotImplementedError
            joint_input = chunk_dict['joint2d'][..., :self.clip_len, :, :] if self.current_as_history else chunk_dict['joint2d'][..., self.clip_len:, :, :]
            joint_output = joint_input.clone()
            assert frame_mask is not None
            joint_input[..., frame_mask, :, :] = 0
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # preprocess
        if self.dataset_config[dataset_name]['rootrel_input']:
            joint_input = joint_input - joint_input[..., 0:1, :]
        if self.dataset_config[dataset_name]['rootrel_target']:
            if 'smpl_vertices' in chunk_dict:
                vertex_output = vertex_output * 1000
                J_regressor = self.smpl.J_regressor_h36m                                    # [17, 6890]
                J_regressor_batch = J_regressor[None, :].expand(vertex_output.shape[0], -1, -1).to(vertex_output.device)
                motion_3d_reg = torch.matmul(J_regressor_batch, vertex_output)                 # motion_3d: (T,17,3)
                vertex_output = vertex_output - motion_3d_reg[:, 0:1, :]
                motion_3d_reg = motion_3d_reg - motion_3d_reg[:, 0:1, :]
            joint_output = joint_output - joint_output[..., 0:1, :]     # joint_output 和 motion_3d_reg/1000 的最大误差在1e-7左右

        return joint_input, joint_output, pose_output, shape_output, vertex_output
    
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

        query_chunk_dict = self.prepare_chunk(self.query_dict, dataset_name, chunk_id=query_chunk_id)
        prompt_chunk_dict = self.prepare_chunk(self.prompt_dict, dataset_name, chunk_id=prompt_chunk_id)

        if self.is_train and self.aug_shuffle_joints:
            raise NotImplementedError

        QUERY_JOINT_INPUT = []
        QUERY_JOINT_TARGET = []
        PROMPT_JOINT_INPUT = []
        PROMPT_JOINT_TARGET = []
        INFO = []
        if self.dataset_config[dataset_name]['return_type'] in ['smpl', 'smpl_x1000', 'all', 'all_x1000']:
            QUERY_POSE_TARGET = []
            QUERY_SHAPE_TARGET = []
            QUERY_VERTEX_TARGET = []
            PROMPT_POSE_TARGET = []
            PROMPT_SHAPE_TARGET = []
            PROMPT_VERTEX_TARGET = []

        for task in self.task_dict[dataset_name]:
            joint_mask, frame_mask = None, None
            if task == 'MC':
                joint_mask = self.joint_mask_dict[dataset_name][query_chunk_id]
            if task == 'MIB':
                frame_mask = self.frame_mask_dict[dataset_name][query_chunk_id]
            
            query_joint_input, query_joint_target, query_pose_target, query_shape_target, query_vertex_target = self.prepare_motion(query_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask)
            prompt_joint_input, prompt_joint_target, prompt_pose_target, prompt_shape_target, prompt_vertex_target = self.prepare_motion(prompt_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask)


            if self.visualize == self.__class__.__name__:
                if dataset_name == 'PW3D_MESH' and task == 'MP':
                    print(f"Do visualizing in {self.__class__.__name__}...")
                    avg_velocity = torch.norm(query_joint_target[1:]*1000 - query_joint_target[:-1]*1000, dim=-1).mean()
                    if avg_velocity > 20:
                        data_to_viz = {
                            'prompt_joint_input': prompt_joint_input[..., :3],
                            'prompt_joint_target': prompt_joint_target,
                            'filler': torch.zeros_like(prompt_joint_target),
                            'query_joint_input': query_joint_input[..., :3],
                            'query_joint_target': query_joint_target,
                            'query_vertex_target': query_vertex_target,
                        }
                        viz_skel_seq_anim(data_to_viz, subplot_layout=(2,3), fs=1, fig_title=f'{dataset_name}-{task}-{query_index}-velo{avg_velocity}', if_node=True,
                                          lim3d=[0.5,0.5,0.5,0.5,0.5,800], lw=8, tight_layout=True, node_size=2,
                                          if_print=1, file_name=f'{query_index:08d}', file_folder=f'viz_results/wSMPL/{dataset_name}/{task}')
                        print(f"saved to [viz_results/wSMPL/{dataset_name}/{task}/{query_index:08d}]")
            

            
            QUERY_JOINT_INPUT.append(query_joint_input)
            QUERY_JOINT_TARGET.append(query_joint_target)
            PROMPT_JOINT_INPUT.append(prompt_joint_input)
            PROMPT_JOINT_TARGET.append(prompt_joint_target)
            INFO.append({
                'dataset': dataset_name,
                'task': task,
                'joint_mask': joint_mask,
                'frame_mask': frame_mask,
                'query_chunk_id': query_chunk_id,
                'prompt_chunk_id': prompt_chunk_id,
                'query_index': query_index
            })
            if self.dataset_config[dataset_name]['return_type'] in ['smpl', 'smpl_x1000', 'all', 'all_x1000']:
                QUERY_POSE_TARGET.append(query_pose_target)
                QUERY_SHAPE_TARGET.append(query_shape_target)
                QUERY_VERTEX_TARGET.append(query_vertex_target)
                PROMPT_POSE_TARGET.append(prompt_pose_target)
                PROMPT_SHAPE_TARGET.append(prompt_shape_target)
                PROMPT_VERTEX_TARGET.append(prompt_vertex_target)

        if self.is_train and self.dumb_task:
            for dumb_task in self.dumb_task.split(','):
                raise NotImplementedError

        QUERY_JOINT_INPUT = torch.stack(QUERY_JOINT_INPUT)      # [num_tasks, T, 17, 3]
        QUERY_JOINT_TARGET = torch.stack(QUERY_JOINT_TARGET)    # [num_tasks, T, 17, 3]
        PROMPT_JOINT_INPUT = torch.stack(PROMPT_JOINT_INPUT)    # [num_tasks, T, 17, 3]
        PROMPT_JOINT_TARGET = torch.stack(PROMPT_JOINT_TARGET)  # [num_tasks, T, 17, 3]
        if self.dataset_config[dataset_name]['return_type'] in ['smpl', 'smpl_x1000', 'all', 'all_x1000']:
            QUERY_POSE_TARGET = torch.stack(QUERY_POSE_TARGET)          # [num_tasks, T, 72]
            QUERY_SHAPE_TARGET = torch.stack(QUERY_SHAPE_TARGET)        # [num_tasks, T, 10]
            QUERY_VERTEX_TARGET = torch.stack(QUERY_VERTEX_TARGET)      # [num_tasks, T, 6890, 3]
            PROMPT_POSE_TARGET = torch.stack(PROMPT_POSE_TARGET)        # [num_tasks, T, 72]
            PROMPT_SHAPE_TARGET = torch.stack(PROMPT_SHAPE_TARGET)      # [num_tasks, T, 10]
            PROMPT_VERTEX_TARGET = torch.stack(PROMPT_VERTEX_TARGET)    # [num_tasks, T, 6890, 3]
        
        QUERY_DATA_DICT = {
            'joint_input': QUERY_JOINT_INPUT,
            'joint_target': QUERY_JOINT_TARGET,
        }
        PROMPT_DATA_DICT = {
            'joint_input': PROMPT_JOINT_INPUT,
            'joint_target': PROMPT_JOINT_TARGET,
        }
        if self.dataset_config[dataset_name]['return_type'] in ['smpl', 'smpl_x1000', 'all', 'all_x1000']:
            QUERY_DATA_DICT['pose_target'] = QUERY_POSE_TARGET
            QUERY_DATA_DICT['shape_target'] = QUERY_SHAPE_TARGET
            QUERY_DATA_DICT['vertex_target'] = QUERY_VERTEX_TARGET
            PROMPT_DATA_DICT['pose_target'] = PROMPT_POSE_TARGET
            PROMPT_DATA_DICT['shape_target'] = PROMPT_SHAPE_TARGET
            PROMPT_DATA_DICT['vertex_target'] = PROMPT_VERTEX_TARGET
        return QUERY_DATA_DICT, PROMPT_DATA_DICT, INFO
        # return QUERY_JOINT_INPUT, QUERY_JOINT_TARGET, PROMPT_JOINT_INPUT, PROMPT_JOINT_TARGET, INFO


def collate_func(batch):    # batch: list, len=batch_size. list element: tuple containing returned values from __getitem__
    batch_size = len(batch)

    QUERY_DICT = {}
    for key in batch[0][0].keys():
        QUERY_DICT[key] = torch.cat([batch[b][0][key] for b in range(batch_size)])
    
    PROMPT_DICT = {}
    for key in batch[0][1].keys():
        PROMPT_DICT[key] = torch.cat([batch[b][1][key] for b in range(batch_size)])

    INFO = [info for b in range(batch_size) for info in batch[b][2]]

    return QUERY_DICT, PROMPT_DICT, INFO
        

class DataReaderH36M_3D(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = 'h36m_cpn_cam_source.pkl'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.dt_root = dt_root
        self.dt_file = dt_file
        self.dt_dataset = read_pkl(os.path.join(dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        
    def read_2d(self):
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)  
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)  
                if len(train_confidence.shape)==2: # (1559752, 17)
                    train_confidence = train_confidence[:,:,None]
                    test_confidence = test_confidence[:,:,None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:,:,0:1]
                test_confidence = np.ones(testset.shape)[:,:,0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        else:
            trainset = np.concatenate((trainset, np.zeros(trainset.shape)[:,:,0:1]), axis=-1)
            testset = np.concatenate((testset, np.zeros(testset.shape)[:,:,0:1]), axis=-1)
        return trainset, testset

    def read_3d(self):
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)    # [N, 17, 3]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            train_labels[idx, :, :2] = train_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            train_labels[idx, :, 2:] = train_labels[idx, :, 2:] / res_w * 2
            
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_labels[idx, :, :2] = test_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            test_labels[idx, :, 2:] = test_labels[idx, :, 2:] / res_w * 2
            
        return train_labels, test_labels
    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw
    
    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]                          # (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]                           # (566920,)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train) 
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test
    
    def get_hw(self):
#       Only Testset HW is needed for denormalization
        test_hw = self.read_hw()                                     # train_data (1559752, 2) test_data (566920, 2)
        split_id_train, split_id_test = self.get_split_id()
        test_hw = test_hw[split_id_test][:,0,:]                      # (N, 2)
        return test_hw
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        mean_2d = np.mean(train_data, axis=(0, 1)) # (3,)
        std_2d = np.std(train_data, axis=(0, 1))   # (3,)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        mean_3d = np.mean(train_labels, axis=(0, 1)) # (3,)
        std_3d = np.std(train_labels, axis=(0, 1))   # (3,)
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, 27, 17, 3)
        # ipdb.set_trace()
        num_train = train_data.shape[0]
        num_test = test_data.shape[0]
        return train_data, test_data, train_labels, test_labels, num_train, num_test, mean_2d, std_2d, mean_3d, std_3d
    
    def denormalize(self, test_data):
#       data: (N, n_frames, 51) or data: (N, n_frames, 17, 3)        
        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])
        assert len(data) == len(test_hw)
        # denormalize (x,y,z) coordiantes for results
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data # [n_clips, -1, 17, 3]


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


class DataReaderMesh(object):
    def __init__(self, split, n_frames, sample_stride, data_stride, read_confidence,
                 dt_root='data_icl_gen/processed_data/H36M_MESH', dt_file='mesh_det_h36m_EXTENDED.pkl', res=[1920, 1920],
                 return_type='joint', use_global_orient=True):
        
        self.dt_root = dt_root
        self.dt_file = dt_file
        if dt_file.endswith('.pt'):
            self.dt_dataset = joblib.load(os.path.join(dt_root, dt_file))
        elif dt_file.endswith('.pkl'):
            self.dt_dataset = read_pkl(os.path.join(dt_root, dt_file))

        self.split = split
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride = data_stride
        self.read_confidence = read_confidence
        self.res = res

        self.return_type = return_type  # 'joint' or 'smpl' or 'joint_x1000' or 'smpl_x1000' or 'all' or 'all_x1000'
        assert return_type in ['joint', 'smpl', 'joint_x1000', 'smpl_x1000', 'all', 'all_x1000']
        self.use_global_orient = use_global_orient

        self.split_id = None
        # self.split_id_train = None
        # self.split_id_test = None

        '''
        证明 /home/wxs/Skeleton-in-Context-tpami/data_icl_gen/processed_data/PW3D/mesh_det_pw3d.pkl 的数据帧率是30Hz:
        cnt = {}
        for i in range(len(self.dt_dataset['train']['source'])):
            source = self.dt_dataset['train']['source'][i]
            if source not in tttt:
                tttt[source] = 0
            cnt[source] += 1
        得到的 cnt: 
         {   'courtyard_arguing_000': 765, 'courtyard_backpack_000': 1262, 'courtyard_basketball_000': 418, 'courtyard_bodyScannerMotions_000': 1257, 
                'courtyard_box_000': 1041, 'courtyard_capoeira_000': 388, 'courtyard_captureSelfies_000': 677, 'courtyard_dancing_010': 309, 'courtyard_giveDirections_000': 848, 
                'courtyard_golf_000': 604, 'courtyard_goodNews_000': 431, 'courtyard_jacket_000': 1236, 'courtyard_laceShoe_000': 931, 'courtyard_rangeOfMotions_000': 594, 
                'courtyard_relaxOnBench_000': 546, 'courtyard_relaxOnBench_010': 842, 'courtyard_shakeHands_000': 361, 'courtyard_warmWelcome_000': 562, 'outdoors_climbing_000': 1224, 
                'outdoors_climbing_010': 1061, 'outdoors_climbing_020': 874, 'outdoors_freestyle_000': 498, 'outdoors_slalom_000': 328, 'outdoors_slalom_010': 369, 'courtyard_arguing_001': 765, 
                'courtyard_basketball_001': 468, 'courtyard_capoeira_001': 435, 'courtyard_captureSelfies_001': 693, 'courtyard_dancing_011': 331, 'courtyard_giveDirections_001': 848, 
                'courtyard_goodNews_001': 431, 'courtyard_rangeOfMotions_001': 601, 'courtyard_shakeHands_001': 391, 'courtyard_warmWelcome_001': 343
                }
        又发现3DPW源数据中的 ./sequenceFiles/train/courtyard_arguing_00.pkl 文件的 poses2d 对应765帧, 而 poses_60Hz 对应1530帧
        '''

    def get_split_id(self, designated_split=None):
        if designated_split is not None:
            if 'source' in self.dt_dataset[designated_split]:
                vid_list = self.dt_dataset[designated_split]['source'][::self.sample_stride]
            elif 'vid_name' in self.dt_dataset[designated_split]:
                vid_list = self.dt_dataset[designated_split]['vid_name'][::self.sample_stride]
            else:
                raise KeyError


            split_id = split_clips(vid_list, self.n_frames, self.data_stride[designated_split])
            return split_id

        if self.split_id is not None:
            return self.split_id
        if 'source' in self.dt_dataset[self.split]:
            vid_list = self.dt_dataset[self.split]['source'][::self.sample_stride]    
        elif 'vid_name' in self.dt_dataset[self.split]:
            vid_list = self.dt_dataset[self.split]['source'][::self.sample_stride]
        else:
            raise KeyError
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
            raise ValueError('No resolution information provided')

        if self.read_confidence:
            dataset_confidence = self.dt_dataset[split]['confidence'][::self.sample_stride].astype(np.float32)
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

        if any('vertices' in key for key in self.dt_dataset[split].keys()):
            if self.use_global_orient:
                if 'x1000' in self.return_type:
                    if 'smpl_vertices_x1000_w_GlobalOrient' in self.dt_dataset[split].keys():
                        smpl_vertices = self.dt_dataset[split]['smpl_vertices_x1000_w_GlobalOrient'][::self.sample_stride, :, :]
                    else:
                        smpl_vertices = self.dt_dataset[split]['smpl_vertices_w_GlobalOrient'][::self.sample_stride, :, :] * 1000
                else:
                    smpl_vertices = self.dt_dataset[split]['smpl_vertices_w_GlobalOrient'][::self.sample_stride, :, :]  # [N, 6890, 3]
            else:
                if 'x1000' in self.return_type:
                    if 'smpl_vertices_x1000_wo_GlobalOrient' in self.dt_dataset[split].keys():
                        smpl_vertices = self.dt_dataset[split]['smpl_vertices_x1000_wo_GlobalOrient'][::self.sample_stride, :, :]
                    else:
                        smpl_vertices = self.dt_dataset[split]['smpl_vertices_wo_GlobalOrient'][::self.sample_stride, :, :] * 1000
                else:
                    smpl_vertices = self.dt_dataset[split]['smpl_vertices_wo_GlobalOrient'][::self.sample_stride, :, :]  # [N, 6890, 3]
            return smpl_pose, smpl_shape, smpl_vertices

        return smpl_pose, smpl_shape, None

    def read_3d(self, designated_split=None):

        split = designated_split if designated_split is not None else self.split

        if self.use_global_orient:
            if 'x1000' in self.return_type:
                if 'joint_3d_x1000_w_GlobalOrient' in self.dt_dataset[split].keys():
                    joints_3d = self.dt_dataset[split]['joint_3d_x1000_w_GlobalOrient'][::self.sample_stride, :, :]
                else:
                    joints_3d = self.dt_dataset[split]['joint_3d_w_GlobalOrient'][::self.sample_stride, :, :] * 1000
            else:
                joints_3d = self.dt_dataset[split]['joint_3d_w_GlobalOrient'][::self.sample_stride, :, :]  # [N, 17, 3]
        else:
            if 'x1000' in self.return_type:
                if 'joint_3d_x1000_wo_GlobalOrient' in self.dt_dataset[split].keys():
                    joints_3d = self.dt_dataset[split]['joint_3d_x1000_wo_GlobalOrient'][::self.sample_stride, :, :]
                else:
                    joints_3d = self.dt_dataset[split]['joint_3d_wo_GlobalOrient'][::self.sample_stride, :, :] * 1000
            else:
                joints_3d = self.dt_dataset[split]['joint_3d_wo_GlobalOrient'][::self.sample_stride, :, :]  # [N, 17, 3]
        return joints_3d
    
    def get_split_data(self, designated_split=None):

        split_id = self.get_split_id(designated_split)

        joints_2d = self.read_2d(designated_split)  # [N, 17, 3]
        joints_2d = joints_2d[split_id]  # [n, T, 17, 3]
        assert len(split_id) == len(joints_2d)
        assert (not self.read_confidence) == (joints_2d[..., -1] == 0).all()

        num_samples = len(joints_2d)

        data_dict = {
            'joint2d': joints_2d,
            'num_sample': num_samples
        }

        if self.return_type in ['joint', 'joint_x1000', 'all', 'all_x1000']:
            joints_3d = self.read_3d(designated_split)  # [N, 17, 3]
            joints_3d = joints_3d[split_id]  # [n, T, 17, 3]
            assert len(joints_2d) == len(joints_3d)
            data_dict['joint3d'] = joints_3d
        if self.return_type in ['smpl', 'smpl_x1000', 'all', 'all_x1000']:
            smpl_pose, smpl_shape, smpl_vertices = self.read_smpl(designated_split) # [N, 72], [N, 10], [N, 6890, 3]
            smpl_pose, smpl_shape = smpl_pose[split_id], smpl_shape[split_id]    # [n, T, 72], [n, T, 10], [n, 6890, 3]
            if smpl_vertices is not None:
                smpl_vertices = smpl_vertices[split_id]
            assert len(joints_2d) == len(smpl_pose)
            data_dict['smpl_pose'] = smpl_pose
            data_dict['smpl_shape'] = smpl_shape
            if smpl_vertices is not None:
                data_dict['smpl_vertices'] = smpl_vertices
        else:
            raise ValueError('Invalid return_type')
        
        return data_dict
        
    def get_all_data(self):
        train_data_dict = self.get_split_data(designated_split='train')
        test_data_dict = self.get_split_data(designated_split='test')
        return {'train': train_data_dict, 
                'test': test_data_dict}
    


class DataReaderMesh_H36M_TCMR(object):
    def __init__(self, split, n_frames, sample_stride, data_stride, read_confidence,
                 dt_root='data_icl_gen/processed_data/H36M_MESH', dt_file='mesh_det_h36m_EXTENDED.pkl', res=[1920, 1920],
                 return_type='joint', use_global_orient=True):
        
        self.dt_root = dt_root
        self.dt_file = dt_file
        self.dt_dataset = joblib.load(os.path.join(dt_root, dt_file))

        # self.split = split
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride = data_stride
        self.read_confidence = read_confidence
        self.res = res

        # self.return_type = return_type  # 'joint' or 'smpl' or 'joint_x1000' or 'smpl_x1000' or 'all' or 'all_x1000'
        # assert return_type in ['joint', 'smpl', 'joint_x1000', 'smpl_x1000', 'all', 'all_x1000']
        self.use_global_orient = use_global_orient

        self.split_id = None

    def get_split_id(self):
        if self.split_id is not None:
            return self.split_id
        vid_list = self.dt_dataset['vid_name'][::self.sample_stride]
        self.split_id = split_clips(vid_list, self.n_frames, self.data_stride)
        return self.split_id

    def read_2d(self):

        joints_2d = self.dt_dataset['joints2D'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]

        # if self.res is not None:
        #     res_w, res_h = self.res
        #     offset = [1, res_h / res_w]
        #     joints_2d = joints_2d / res_w * 2 - offset
        # elif 'img_hw' in self.dt_dataset[split].keys():
        #     res = np.array(self.dt_dataset[split]['img_hw'])[::self.sample_stride].astype(np.float32)
        #     res_w, res_h = res.max(1)[:, None, None], res.max(1)[:, None, None]
        #     offset = 1
        #     joints_2d = joints_2d / res_w * 2 - offset
        # elif 'camera_name' in self.dt_dataset[split].keys():
        #     for idx, camera_name in enumerate(self.dt_dataset[split]['camera_name']):
        #         if camera_name == '54138969' or camera_name == '60457274':
        #             res_w, res_h = 1000, 1002
        #         elif camera_name == '55011271' or camera_name == '58860488':
        #             res_w, res_h = 1000, 1000
        #         else:
        #             assert 0, '%d data item has an invalid camera name' % idx
        #         joints_2d[idx, :, :] = joints_2d[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        # else:
        #     raise ValueError('No resolution information provided')

        if self.read_confidence:
            dataset_confidence = self.dt_dataset['confidence'][::self.sample_stride].astype(np.float32)
            if len(dataset_confidence.shape)==2: 
                dataset_confidence = dataset_confidence[:,:,None]
            joints_2d = np.concatenate((joints_2d, dataset_confidence), axis=-1)  # [N, 17, 3]
        else:
            joints_2d = np.concatenate((joints_2d, np.zeros_like(joints_2d[..., :1])), axis=-1)
        return joints_2d
    
    def read_smpl(self):

        smpl_pose = self.dt_dataset['pose'][::self.sample_stride, :]
        smpl_shape = self.dt_dataset['shape'][::self.sample_stride, :]

        return smpl_pose, smpl_shape, None

    def read_3d(self):
        joints_3d = self.dt_dataset['joints3D'][::self.sample_stride, :, :]  # [N, 17, 3]
        return joints_3d
    
    def get_all_data(self):
        joints_2d = self.read_2d()  # [N, 17, 3]
        assert (not self.read_confidence) == (joints_2d[..., -1] == 0).all()

        num_samples = len(joints_2d)

        data_dict = {
            'joint2d': joints_2d,
            'num_sample': num_samples
        }

        joints_3d = self.read_3d()  # [N, 17, 3]
        data_dict['joint3d'] = joints_3d
        
        smpl_pose, smpl_shape, smpl_vertices = self.read_smpl() # [N, 72], [N, 10], [N, 6890, 3]
        data_dict['smpl_pose'] = smpl_pose
        data_dict['smpl_shape'] = smpl_shape
        if smpl_vertices is not None:
            data_dict['smpl_vertices'] = smpl_vertices
        
        return {'train': data_dict}


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

