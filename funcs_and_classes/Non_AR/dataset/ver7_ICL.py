from cgi import test
import chunk
from cmd import PROMPT
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

from funcs_and_classes.Non_AR.dataset.ver5_ICL import MotionDatasetICL as MotionDatasetICL_VER5

CONNECTIONS_H36M_J17_JOINT = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
                              7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}

class MotionDatasetICL(MotionDatasetICL_VER5):
    def __init__(self, args, data_split, TASK=None, DATASET_NAME=None, SLICED_DATA=None, PROMPT_LOG=None, **kwargs):
        super().__init__(args, data_split, TASK, DATASET_NAME, SLICED_DATA, PROMPT_LOG, **kwargs)

        for dataset_name in self.query_dict.keys():
            self.query_dict[dataset_name]['smpl_pose'] = self.query_dict[dataset_name]['smpl_pose'].reshape(self.query_dict[dataset_name]['smpl_pose'].shape[0], -1, 24, 3)
        for dataset_name in self.prompt_dict.keys():
            self.prompt_dict[dataset_name]['smpl_pose'] = self.prompt_dict[dataset_name]['smpl_pose'].reshape(self.prompt_dict[dataset_name]['smpl_pose'].shape[0], -1, 24, 3)

        self.num_mesh_joint = 24
        self.mesh_joint_mask_ratio = args.mesh_joint_mask_ratio
        self.mesh_joint_mask_dict = {}
        for dataset_name in self.datasets:
            n_frames = self.dataset_config[dataset_name].get('clip_len', self.clip_len) * 2
            if 'MeshCompletion' in self.task_dict[dataset_name]:
                if kwargs.get('rank', 0) == 0: print(f"\tPreparing {data_split} [mesh joint masks] from [{dataset_name}] for task: [MeshCompletion]...", end=' ')
                mesh_joint_mask_presave_path = os.path.join(args.presave_folder, 'mesh_joint_masks', dataset_name,
                                                f'nframes{self.dataset_config[dataset_name].get("clip_len", self.clip_len) * 2} - samplestride{self.dataset_config[dataset_name]["sample_stride"]} - '
                                                +f'datastridetrain{self.dataset_config[dataset_name]["data_stride"]["train"]} - datastridetest{self.dataset_config[dataset_name]["data_stride"]["test"]} - '
                                                +f'nummeshjoint{self.num_mesh_joint} - meshjointmaskratio{self.mesh_joint_mask_ratio} - '
                                                +f'filename_{os.path.splitext(os.path.basename(self.dataset_file[dataset_name]))[0]}', data_split,
                                                )
                mesh_joint_mask_presave_file = os.path.join(mesh_joint_mask_presave_path, 'mesh_joint_masks.pkl')
                mesh_joint_mask_config_file = os.path.join(mesh_joint_mask_presave_path, 'mesh_joint_mask_config.pkl')
                if not os.path.exists(mesh_joint_mask_presave_file):
                    if kwargs.get('rank', 0) == 0: print("Presaving mesh joint masks...")
                    os.makedirs(mesh_joint_mask_presave_path, exist_ok=True)
                    mesh_joint_masks = [random.sample(range(1,self.num_mesh_joint), int(self.mesh_joint_mask_ratio*self.num_mesh_joint)) for _ in range(self.num_query_dict[dataset_name])]
                    mesh_joint_masks_config = {
                        'nframes' : n_frames,
                        'samplestride' : self.dataset_config[dataset_name]['sample_stride'],
                        'datastridetrain' : self.dataset_config[dataset_name]['data_stride']["train"],
                        'datastridetest' : self.dataset_config[dataset_name]['data_stride']["test"],
                        'numsample' : self.num_query_dict[dataset_name],
                        'nummeshjoint' : self.num_mesh_joint,
                        'meshjointmaskratio' : self.mesh_joint_mask_ratio,
                        'filename' : os.path.splitext(os.path.basename(self.dataset_file[dataset_name]))[0],
                    }
                    with open(mesh_joint_mask_presave_file, 'wb') as f:
                        pickle.dump(mesh_joint_masks, f, protocol=4)
                    with open(mesh_joint_mask_config_file, 'wb') as f:
                        pickle.dump(mesh_joint_masks_config, f)
                else:
                    if kwargs.get('rank', 0) == 0: print("Loading mesh joint masks...")
                    with open(mesh_joint_mask_config_file, 'rb') as f:
                        mesh_joint_masks_config = pickle.load(f)
                    assert mesh_joint_masks_config == {
                        'nframes' : n_frames,
                        'samplestride' : self.dataset_config[dataset_name]['sample_stride'],
                        'datastridetrain' : self.dataset_config[dataset_name]['data_stride']["train"],
                        'datastridetest' : self.dataset_config[dataset_name]['data_stride']["test"],
                        'numsample' : self.num_query_dict[dataset_name],
                        'nummeshjoint' : self.num_mesh_joint,
                        'meshjointmaskratio' : self.mesh_joint_mask_ratio,
                        'filename' : os.path.splitext(os.path.basename(self.dataset_file[dataset_name]))[0],
                    }
                    with open(mesh_joint_mask_presave_file, 'rb') as f:
                        mesh_joint_masks = pickle.load(f)
                
                # mesh_joint_masks = np.array(mesh_joint_masks)   # (N,9)
                # mesh_joint_masks = np.concatenate((mesh_joint_masks*3,mesh_joint_masks*3+1,mesh_joint_masks*3+2), axis=-1)   # (N,27)
                self.mesh_joint_mask_dict[dataset_name] = mesh_joint_masks

    
    def postprocess(self, motion_seq, dataset_name, task):
        if dataset_name == 'H36M_3D':
            if task == 'PE' and self.dataset_config['H36M_3D']['rootrel_target']:
                motion_seq[..., 0, :] = 0
            motion_seq = motion_seq * self.dataset_config['H36M_3D']['scale_3D']
        else:
            if self.dataset_config[dataset_name]['rootrel_target']:
                motion_seq = motion_seq - motion_seq[..., 0:1, :]
        return motion_seq

    def preprocess(self, data_dict):
        data_dict['smpl_vertex'] = data_dict['smpl_vertex'] - data_dict['joint'][:, :, 0:1, :]
        data_dict['joint'] = data_dict['joint'] - data_dict['joint'][:, :, 0:1, :]
        return data_dict

    def prepare_motion(self, chunk_dict, dataset_name, task, joint_mask, frame_mask, mesh_joint_mask):
        # chunk_dict
        #   'joint2d': (1,T,17,3) or (N,T,17,3); 
        #   'joint3d': (1,T,17,3) or (N,T,17,3);
        #   'smpl_pose': (1,T,24,3) or (N,T,24,3);
        #   'smpl_shape': (1,T,10) or (N,T,10);
        prepare_motion_ver = self.args.get('prepare_motion_ver', 0)
        # if not hasattr(self, 'check_prepare_motion_ver'):
        #     setattr(self, 'check_prepare_motion_ver', prepare_motion_ver)
        #     print(f'\tprepare_motion_ver={prepare_motion_ver}')
        prepare_motion_func = globals().get(f'prepare_motion_ver{prepare_motion_ver}', None)
        return prepare_motion_func(self, chunk_dict, dataset_name, task, joint_mask, frame_mask, mesh_joint_mask)
        


    def __getitem__(self, query_index):
        
        if self.args.data_efficient:
            return self.getitem_dataefficient(query_index)

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
        else:
            prompt_chunk_id = random.choice(self.prompt_list[dataset_name])


        # Checking randomness
        if self.is_train and query_index == 0:
            print(f"\t[check randomness] query_id: {query_chunk_id}, prompt_id: {prompt_chunk_id}")


        query_chunk_dict = self.prepare_chunk(self.query_dict, dataset_name, chunk_id=query_chunk_id)
        prompt_chunk_dict = self.prepare_chunk(self.prompt_dict, dataset_name, chunk_id=prompt_chunk_id)

        if self.is_train and self.aug_shuffle_joints:
            raise NotImplementedError

        QUERY_INPUT_TENSOR = []
        PROMPT_INPUT_TENSOR = []
        QUERY_TARGET_TENSOR = []
        PROMPT_TARGET_TENSOR = []
        QUERY_TARGET_DICT = defaultdict(list)
        PROMPT_TARGET_DICT = defaultdict(list)
        INFO_DICT = defaultdict(list)

        for task in self.task_dict[dataset_name]:
            joint_mask, frame_mask, mesh_joint_mask = None, None, None
            if task == 'MC':
                joint_mask = self.joint_mask_dict[dataset_name][query_chunk_id]
            if task in ['MIB', 'MeshInBetween']:
                frame_mask = self.frame_mask_dict[dataset_name][query_chunk_id]
            if task == 'MeshCompletion':
                mesh_joint_mask = self.mesh_joint_mask_dict[dataset_name][query_chunk_id]

            query_input_tensor, query_target_tensor, query_target, input_mask = self.prepare_motion(query_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask, mesh_joint_mask=mesh_joint_mask)
            prompt_input_tensor, prompt_target_tensor, prompt_target, _ = self.prepare_motion(prompt_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask, mesh_joint_mask=mesh_joint_mask)

            QUERY_INPUT_TENSOR.append(query_input_tensor)
            PROMPT_INPUT_TENSOR.append(prompt_input_tensor)
            QUERY_TARGET_TENSOR.append(query_target_tensor)
            PROMPT_TARGET_TENSOR.append(prompt_target_tensor)
            for mode in query_target.keys():
                QUERY_TARGET_DICT[mode].append(query_target[mode])
            for mode in prompt_target.keys():
                PROMPT_TARGET_DICT[mode].append(prompt_target[mode])

            INFO_DICT['dataset'].append(dataset_name)
            INFO_DICT['task'].append(task)
            INFO_DICT['joint_mask'].append(joint_mask)
            INFO_DICT['frame_mask'].append(frame_mask)
            INFO_DICT['query_chunk_id'].append(query_chunk_id)
            INFO_DICT['prompt_chunk_id'].append(prompt_chunk_id)
            INFO_DICT['query_index'].append(query_index)
            INFO_DICT['use_global_orient'].append(int(self.dataset_config[dataset_name]['use_global_orient']))

            INFO_DICT['input_mask'].append(input_mask)

            if self.visualize == self.__class__.__name__:
                if dataset_name == 'PW3D_MESH' and task == 'MP':
                    print(f"Do visualizing in {self.__class__.__name__}...")
            
        if self.is_train and self.dumb_task:
            for dumb_task in self.dumb_task.split(','):
                raise NotImplementedError

        QUERY_INPUT_TENSOR = torch.cat(QUERY_INPUT_TENSOR)              # [n_task, 16, 24, 3]
        PROMPT_INPUT_TENSOR = torch.cat(PROMPT_INPUT_TENSOR)
        QUERY_TARGET_TENSOR = torch.cat(QUERY_TARGET_TENSOR)              # [n_task, 16, 24, 3]
        PROMPT_TARGET_TENSOR = torch.cat(PROMPT_TARGET_TENSOR)
        for mode in QUERY_TARGET_DICT.keys():
            QUERY_TARGET_DICT[mode] = torch.cat(QUERY_TARGET_DICT[mode])
        for mode in PROMPT_TARGET_DICT.keys():
            PROMPT_TARGET_DICT[mode] = torch.cat(PROMPT_TARGET_DICT[mode])
        
        return QUERY_INPUT_TENSOR, PROMPT_INPUT_TENSOR, QUERY_TARGET_TENSOR, PROMPT_TARGET_TENSOR, QUERY_TARGET_DICT, PROMPT_TARGET_DICT, INFO_DICT
    

    def getitem_dataefficient(self, query_index):        
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
        else:
            prompt_chunk_id = random.choice(self.prompt_list[dataset_name])


        # Checking randomness
        if self.is_train and query_index == 0:
            print(f"\t[check randomness] query_id: {query_chunk_id}, prompt_id: {prompt_chunk_id}")


        query_chunk_dict = self.prepare_chunk(self.query_dict, dataset_name, chunk_id=query_chunk_id)
        prompt_chunk_dict = self.prepare_chunk(self.prompt_dict, dataset_name, chunk_id=prompt_chunk_id)

        if self.is_train and self.aug_shuffle_joints:
            raise NotImplementedError

        QUERY_INPUT_TENSOR = []
        PROMPT_INPUT_TENSOR = []
        # QUERY_TARGET_TENSOR = []
        PROMPT_TARGET_TENSOR = []
        QUERY_TARGET_DICT = defaultdict(list)
        # PROMPT_TARGET_DICT = defaultdict(list)
        INFO_DICT = defaultdict(list)

        for task in self.task_dict[dataset_name]:
            joint_mask, frame_mask, mesh_joint_mask = None, None, None
            if task == 'MC':
                joint_mask = self.joint_mask_dict[dataset_name][query_chunk_id]
            if task in ['MIB', 'MeshInBetween']:
                frame_mask = self.frame_mask_dict[dataset_name][query_chunk_id]
            if task == 'MeshCompletion':
                mesh_joint_mask = self.mesh_joint_mask_dict[dataset_name][query_chunk_id]

            query_input_tensor, query_target_tensor, query_target, input_mask = self.prepare_motion(query_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask, mesh_joint_mask=mesh_joint_mask)
            prompt_input_tensor, prompt_target_tensor, prompt_target, _ = self.prepare_motion(prompt_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask, mesh_joint_mask=mesh_joint_mask)

            QUERY_INPUT_TENSOR.append(query_input_tensor)
            PROMPT_INPUT_TENSOR.append(prompt_input_tensor)
            # QUERY_TARGET_TENSOR.append(query_target_tensor)
            PROMPT_TARGET_TENSOR.append(prompt_target_tensor)
            for mode in query_target.keys():
                QUERY_TARGET_DICT[mode].append(query_target[mode])
            # for mode in prompt_target.keys():
            #     PROMPT_TARGET_DICT[mode].append(prompt_target[mode])

            INFO_DICT['dataset'].append(dataset_name)
            INFO_DICT['task'].append(task)
            INFO_DICT['joint_mask'].append(joint_mask)
            INFO_DICT['frame_mask'].append(frame_mask)
            INFO_DICT['query_chunk_id'].append(query_chunk_id)
            INFO_DICT['prompt_chunk_id'].append(prompt_chunk_id)
            INFO_DICT['query_index'].append(query_index)
            INFO_DICT['use_global_orient'].append(int(self.dataset_config[dataset_name]['use_global_orient']))

            INFO_DICT['input_mask'].append(input_mask)
            temporal_mask = torch.ones(query_input_tensor.shape[:2])
            if dataset_name == 'COCO':
                temporal_mask[:, 1:] = 0.
            INFO_DICT['input_temporal_mask'].append(temporal_mask)

            if self.visualize == self.__class__.__name__:
                if dataset_name == 'PW3D_MESH' and task == 'MP':
                    print(f"Do visualizing in {self.__class__.__name__}...")
            
        if self.is_train and self.dumb_task:
            for dumb_task in self.dumb_task.split(','):
                raise NotImplementedError

        QUERY_INPUT_TENSOR = torch.cat(QUERY_INPUT_TENSOR)              # [n_task, 16, 24, 3]
        PROMPT_INPUT_TENSOR = torch.cat(PROMPT_INPUT_TENSOR)
        # QUERY_TARGET_TENSOR = torch.cat(QUERY_TARGET_TENSOR)              # [n_task, 16, 24, 3]
        PROMPT_TARGET_TENSOR = torch.cat(PROMPT_TARGET_TENSOR)
        for mode in QUERY_TARGET_DICT.keys():
            QUERY_TARGET_DICT[mode] = torch.cat(QUERY_TARGET_DICT[mode])
        # for mode in PROMPT_TARGET_DICT.keys():
        #     PROMPT_TARGET_DICT[mode] = torch.cat(PROMPT_TARGET_DICT[mode])
        
        return QUERY_INPUT_TENSOR, PROMPT_INPUT_TENSOR, None, PROMPT_TARGET_TENSOR, QUERY_TARGET_DICT, None, INFO_DICT
    



def collate_func(batch):
    batch_size = len(batch)
    use_query_target_tensor = (batch[0][2] is not None)
    use_prompt_target_dict = (batch[0][5] is not None)

    QUERY_INPUT_TENSOR = []
    PROMPT_INPUT_TENSOR = []
    PROMPT_TARGET_TENSOR = []
    QUERY_TARGET_DICT = defaultdict(list)
    INFO_DICT = defaultdict(list)

    QUERY_TARGET_TENSOR = [] if use_query_target_tensor else None
    PROMPT_TARGET_DICT = defaultdict(list) if use_prompt_target_dict else None
    
    for b in range(batch_size):
        QUERY_INPUT_TENSOR.append(batch[b][0])
        PROMPT_INPUT_TENSOR.append(batch[b][1])
        PROMPT_TARGET_TENSOR.append(batch[b][3])
        for mode in batch[0][4].keys():
            QUERY_TARGET_DICT[mode].append(batch[b][4][mode])
        for info_key in batch[0][6].keys():
            INFO_DICT[info_key] = INFO_DICT[info_key] + batch[b][6][info_key]

        if use_query_target_tensor:
            QUERY_TARGET_TENSOR.append(batch[b][2])
        if use_prompt_target_dict:
            for mode in batch[0][5].keys():
                PROMPT_TARGET_DICT[mode].append(batch[b][5][mode])

    QUERY_INPUT_TENSOR = torch.cat(QUERY_INPUT_TENSOR)
    PROMPT_INPUT_TENSOR = torch.cat(PROMPT_INPUT_TENSOR)
    PROMPT_TARGET_TENSOR = torch.cat(PROMPT_TARGET_TENSOR)
    for mode in QUERY_TARGET_DICT.keys():
        QUERY_TARGET_DICT[mode] = torch.cat(QUERY_TARGET_DICT[mode])
    
    if use_query_target_tensor:
        QUERY_TARGET_TENSOR = torch.cat(QUERY_TARGET_TENSOR)
    if use_prompt_target_dict:
        for mode in PROMPT_TARGET_DICT.keys():
            PROMPT_TARGET_DICT[mode] = torch.cat(PROMPT_TARGET_DICT[mode])

    return QUERY_INPUT_TENSOR, PROMPT_INPUT_TENSOR, QUERY_TARGET_TENSOR, PROMPT_TARGET_TENSOR, QUERY_TARGET_DICT, PROMPT_TARGET_DICT, INFO_DICT


def init_spatial_adj(adj):
    adj = torch.zeros(adj.shape)
    for i in range(adj.shape[0]):
        if i in CONNECTIONS_H36M_J17_JOINT:
            connected_nodes = CONNECTIONS_H36M_J17_JOINT[i]
            for j in connected_nodes:
                adj[i, j] = 1
    return adj


def prepare_motion_ver0(args, chunk_dict, dataset_name, task, joint_mask, frame_mask, mesh_joint_mask):
    # chunk_dict
    #   'joint2d': (1,T,17,3) or (N,T,17,3); 
    #   'joint3d': (1,T,17,3) or (N,T,17,3);
    #   'smpl_pose': (1,T,72) or (N,T,72);
    #   'smpl_shape': (1,T,10) or (N,T,10);
    target_dict = {}
    if args.current_as_history:
        indices = slice(None, args.clip_len)
    else:
        indices = slice(args.clip_len, None)

    N = chunk_dict['smpl_shape'].shape[0]
    input_tensor = torch.zeros(N,args.clip_len,24,3)
    input_mask = torch.ones(N,24)
    target_tensor = torch.zeros(N,args.clip_len,24,3)
    target_mask = torch.ones(N,24)

    # adj = torch.zeros(24,24)
    
    if task == 'PE':
        input_tensor[:, :, :args.num_joint, :] = chunk_dict['joint2d'][:, indices].clone()
        input_tensor[:, :, :args.num_joint, :] = input_tensor[:, :, :args.num_joint, :] - input_tensor[:, :, 0:1, :]
        input_mask[:, args.num_joint:] = 0

        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()

        target_tensor[:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, indices].clone()
        target_mask[:, args.num_joint:] = 0

        # adj = init_spatial_adj(adj)

    elif task == 'MeshRecover':
        input_tensor[:, :, :args.num_joint, :] = chunk_dict['joint2d'][:, indices].clone()
        input_tensor[:, :, :args.num_joint, :] = input_tensor[:, :, :args.num_joint, :] - input_tensor[:, :, 0:1, :]
        input_mask[:, args.num_joint:] = 0
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
    
        target_tensor = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, args.clip_len, 24, 3)

        # adj = init_spatial_adj(adj)

    elif task == 'FPE':
        input_tensor[:, :, :args.num_joint, :] = chunk_dict['joint2d'][:, :args.clip_len].clone()
        input_tensor[:, :, :args.num_joint, :] = input_tensor[:, :, :args.num_joint, :] - input_tensor[:, :, 0:1, :]
        input_mask[:, args.num_joint:] = 0
        target_dict['joint'] = chunk_dict['joint3d'][:, args.clip_len:].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, args.clip_len:].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, args.clip_len:].clone()

        target_tensor[:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, args.clip_len:].clone()
        target_mask[:, args.num_joint:] = 0

        # adj = init_spatial_adj(adj)

    elif task == 'FutureMeshRecover':
        input_tensor[:, :, :args.num_joint, :] = chunk_dict['joint2d'][:, :args.clip_len].clone()
        input_tensor[:, :, :args.num_joint, :] = input_tensor[:, :, :args.num_joint, :] - input_tensor[:, :, 0:1, :]
        input_mask[:, args.num_joint:] = 0
        target_dict['joint'] = chunk_dict['joint3d'][:, args.clip_len:].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, args.clip_len:].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, args.clip_len:].clone()

        target_tensor = chunk_dict['smpl_pose'][:, args.clip_len:].clone().reshape(N, args.clip_len, 24, 3)

        # adj = init_spatial_adj(adj)

    elif task == 'MP':
        input_tensor[:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, :args.clip_len].clone()
        input_tensor[:, :, :args.num_joint, :] = input_tensor[:, :, :args.num_joint, :] - input_tensor[:, :, 0:1, :]
        input_mask[:, args.num_joint:] = 0
        target_dict['joint'] = chunk_dict['joint3d'][:, args.clip_len:].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, args.clip_len:].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, args.clip_len:].clone()

        target_tensor[:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, args.clip_len:].clone()
        target_mask[:, args.num_joint:] = 0

    elif task == 'MeshPred':
        input_tensor = chunk_dict['smpl_pose'][:, :args.clip_len].clone().reshape(N, args.clip_len, 24, 3)
        target_dict['joint'] = chunk_dict['joint3d'][:, args.clip_len:].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, args.clip_len:].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, args.clip_len:].clone()

        target_tensor = chunk_dict['smpl_pose'][:, args.clip_len:].clone().reshape(N, args.clip_len, 24, 3)

    elif task == 'MC':
        input_tensor[:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, indices].clone()
        input_tensor[:, :, :args.num_joint, :] = input_tensor[:, :, :args.num_joint, :] - input_tensor[:, :, 0:1, :]
        input_mask[:, args.num_joint:] = 0
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        assert joint_mask is not None
        input_tensor[:, :, joint_mask] = 0

        target_tensor[:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, indices].clone()
        target_mask[:, args.num_joint:] = 0

    elif task == 'MeshCompletion':
        input_tensor = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, args.clip_len, 24, 3)
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        assert mesh_joint_mask is not None
        input_tensor[:, :, mesh_joint_mask] = 0

        target_tensor = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, args.clip_len, 24, 3)

    elif task == 'MIB':
        input_tensor[:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, indices].clone()
        input_tensor[:, :, :args.num_joint, :] = input_tensor[:, :, :args.num_joint, :] - input_tensor[:, :, 0:1, :]
        input_mask[:, args.num_joint:] = 0
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        assert frame_mask is not None
        input_tensor[:, frame_mask] = 0

        target_tensor[:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, indices].clone()
        target_mask[:, args.num_joint:] = 0

    elif task == 'MeshInBetween':
        input_tensor = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, args.clip_len, 24, 3)
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, -1, 72)
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        assert frame_mask is not None
        input_tensor[:, frame_mask] = 0

        target_tensor = chunk_dict['smpl_pose'][:, indices].clone().reshape(N, args.clip_len, 24, 3)

    elif task == 'COPY3D':
        raise NotImplementedError
    elif task == 'COPY2D':
        raise NotImplementedError("target_dict['joint'] has to be 3D joints for rootrel-ing smpl_vertex")
    elif task in ['FPEhis', 'MPhis', 'MP2D', 'MC2D', 'MIB2D']:
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown task: {task}")

    return input_tensor, target_tensor, target_dict, input_mask


def prepare_motion_ver1(args, chunk_dict, dataset_name, task, joint_mask, frame_mask, mesh_joint_mask):
    # chunk_dict
    #   'joint2d': (1,T,17,3) or (N,T,17,3); 
    #   'joint3d': (1,T,17,3) or (N,T,17,3);
    #   'smpl_pose': (1,T,72) or (N,T,72);
    #   'smpl_shape': (1,T,10) or (N,T,10);
    input_dict = OrderedDict({'joint': None, 'smpl_pose': None, 'smpl_shape': None})
    target_dict = OrderedDict({'joint': None, 'smpl_pose': None, 'smpl_shape': None})
    if args.current_as_history:
        indices = slice(None, args.clip_len)
    else:
        indices = slice(args.clip_len, None)

    if task in ['PE', 'MeshRecover']:
        input_dict['joint'] = chunk_dict['joint2d'][:, indices].clone()
        input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
    elif task in ['FPE', 'FutureMeshRecover']:
        input_dict['joint'] = chunk_dict['joint2d'][:, :args.clip_len].clone()
        input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, :args.clip_len].clone()
        input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, :args.clip_len].clone()
        target_dict['joint'] = chunk_dict['joint3d'][:, args.clip_len:].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, args.clip_len:].clone()
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, args.clip_len:].clone()
    elif task in ['MP', 'MeshPred']:
        input_dict['joint'] = chunk_dict['joint3d'][:, :args.clip_len].clone()
        input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, :args.clip_len].clone()
        input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, :args.clip_len].clone()
        target_dict['joint'] = chunk_dict['joint3d'][:, args.clip_len:].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, args.clip_len:].clone()
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, args.clip_len:].clone()
    elif task == 'MC':
        input_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        assert joint_mask is not None
        input_dict['joint'][:, :, joint_mask] = 0
    elif task == 'MeshCompletion':
        input_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        assert mesh_joint_mask is not None
        input_dict['smpl_pose'][:, :, mesh_joint_mask] = 0
    elif task in ['MIB', 'MeshInBetween']:
        input_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        assert frame_mask is not None
        input_dict['joint'][:, frame_mask] = 0
        input_dict['smpl_pose'][:, frame_mask] = 0
    elif task == 'COPY3D':
        input_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
    elif task == 'COPY2D':
        raise NotImplementedError("target_dict['joint'] has to be 3D joints for rootrel-ing smpl_vertex")
        input_dict['joint'] = chunk_dict['joint2d'][:, indices].clone()
        input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        target_dict['joint'] = chunk_dict['joint2d'][:, indices].clone()
        target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
        target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
    elif task in ['FPEhis', 'MPhis', 'MP2D', 'MC2D', 'MIB2D']:
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown task: {task}")
    return input_dict, target_dict

