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
from data_gen.angle2joint import ang2joint

from funcs_and_classes.Non_AR.dataset.ver7_ICL import MotionDatasetICL as MotionDatasetICL_VER7

CONNECTIONS_H36M_J17_JOINT = {10: [9], 9: [8, 10], 8: [7, 9], 14: [15, 8], 15: [16, 14], 11: [12, 8], 12: [13, 11],
                              7: [0, 8], 0: [1, 7], 1: [2, 0], 2: [3, 1], 4: [5, 0], 5: [6, 4], 16: [15], 13: [12], 3: [2], 6: [5]}

class MotionDatasetICL(MotionDatasetICL_VER7):
    def __init__(self, args, data_split, TASK=None, DATASET_NAME=None, SLICED_DATA=None, PROMPT_LOG=None, **kwargs):
        super().__init__(args, data_split, TASK, DATASET_NAME, SLICED_DATA, PROMPT_LOG, **kwargs)

        for dataset_name in self.query_dict.keys():
            assert self.query_dict[dataset_name]['smpl_pose'].shape[-2:] == (24, 3)
        for dataset_name in self.prompt_dict.keys():
            assert self.prompt_dict[dataset_name]['smpl_pose'].shape[-2:] == (24, 3)
        for dataset_name in self.datasets:
            if 'MeshCompletion' in self.task_dict[dataset_name]:
                assert np.array(self.mesh_joint_mask_dict[dataset_name]).shape[-1] == int(self.mesh_joint_mask_ratio*self.num_mesh_joint)

    def prepare_motion(self, **kwargs):
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
        return prepare_motion_func(self, **kwargs)


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
        else:
            prompt_chunk_id = random.choice(self.prompt_list[dataset_name])

        # Checking randomness
        if self.is_train and query_index == 0:
            print(f"\t[check randomness] query_id: {query_chunk_id}, prompt_id: {prompt_chunk_id}")

        query_chunk_dict = self.prepare_chunk(self.query_dict, dataset_name, chunk_id=query_chunk_id)
        prompt_chunk_dict = self.prepare_chunk(self.prompt_dict, dataset_name, chunk_id=prompt_chunk_id)

        if self.is_train and self.aug_shuffle_joints:
            raise NotImplementedError

        QUERY_SAMPLE_DICT_JOINT = defaultdict(list)
        PROMPT_SAMPLE_DICT_JOINT = defaultdict(list)
        INFO_DICT_JOINT = defaultdict(list)
        QUERY_SAMPLE_DICT_MESH = defaultdict(list)
        PROMPT_SAMPLE_DICT_MESH = defaultdict(list)
        INFO_DICT_MESH = defaultdict(list)

        for task in self.task_dict[dataset_name]:
            if not self.is_train: assert len(self.task_dict[dataset_name]) == 1
            joint_mask, frame_mask, mesh_joint_mask = None, None, None
            if task == 'MC':
                joint_mask = self.joint_mask_dict[dataset_name][query_chunk_id]
            if task in ['MIB', 'MeshInBetween']:
                frame_mask = self.frame_mask_dict[dataset_name][query_chunk_id]
            if task == 'MeshCompletion':
                mesh_joint_mask = self.mesh_joint_mask_dict[dataset_name][query_chunk_id]

            query_sample_dict = self.prepare_motion(chunk_dict=query_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask, mesh_joint_mask=mesh_joint_mask, if_query=True)
            prompt_sample_dict = self.prepare_motion(chunk_dict=prompt_chunk_dict, dataset_name=dataset_name, task=task, joint_mask=joint_mask, frame_mask=frame_mask, mesh_joint_mask=mesh_joint_mask, if_query=False)

            
            if task in ['PE', 'FPE', 'MP', 'MC', 'MIB']:
                for mode in query_sample_dict.keys():
                    QUERY_SAMPLE_DICT_JOINT[mode].append(query_sample_dict[mode])
                for mode in prompt_sample_dict.keys():
                    PROMPT_SAMPLE_DICT_JOINT[mode].append(prompt_sample_dict[mode])
                INFO_DICT_JOINT['dataset'].append(dataset_name)
                INFO_DICT_JOINT['task'].append(task)
                INFO_DICT_JOINT['frame_mask'].append(frame_mask)
                INFO_DICT_JOINT['joint_mask'].append(joint_mask)
                INFO_DICT_JOINT['mesh_joint_mask'].append(mesh_joint_mask)
                INFO_DICT_JOINT['query_chunk_id'].append(query_chunk_id)
                INFO_DICT_JOINT['prompt_chunk_id'].append(prompt_chunk_id)
                INFO_DICT_JOINT['query_index'].append(query_index)
                INFO_DICT_JOINT['use_global_orient'].append(int(self.dataset_config[dataset_name]['use_global_orient']))

            elif task in ['MeshRecover', 'FutureMeshRecover', 'MeshPred', 'MeshCompletion', 'MeshInBetween']:
                for mode in query_sample_dict.keys():
                    QUERY_SAMPLE_DICT_MESH[mode].append(query_sample_dict[mode])
                for mode in prompt_sample_dict.keys():
                    PROMPT_SAMPLE_DICT_MESH[mode].append(prompt_sample_dict[mode])
                INFO_DICT_MESH['dataset'].append(dataset_name)
                INFO_DICT_MESH['task'].append(task)
                INFO_DICT_MESH['frame_mask'].append(frame_mask)
                INFO_DICT_MESH['joint_mask'].append(joint_mask)
                INFO_DICT_MESH['mesh_joint_mask'].append(mesh_joint_mask)
                INFO_DICT_MESH['query_chunk_id'].append(query_chunk_id)
                INFO_DICT_MESH['prompt_chunk_id'].append(prompt_chunk_id)
                INFO_DICT_MESH['query_index'].append(query_index)
                INFO_DICT_MESH['use_global_orient'].append(int(self.dataset_config[dataset_name]['use_global_orient']))

            
        for mode in QUERY_SAMPLE_DICT_JOINT.keys():
            QUERY_SAMPLE_DICT_JOINT[mode] = torch.cat(QUERY_SAMPLE_DICT_JOINT[mode])
        for mode in PROMPT_SAMPLE_DICT_JOINT.keys():
            PROMPT_SAMPLE_DICT_JOINT[mode] = torch.cat(PROMPT_SAMPLE_DICT_JOINT[mode])
        for mode in QUERY_SAMPLE_DICT_MESH.keys():
            QUERY_SAMPLE_DICT_MESH[mode] = torch.cat(QUERY_SAMPLE_DICT_MESH[mode])
        for mode in PROMPT_SAMPLE_DICT_MESH.keys():
            PROMPT_SAMPLE_DICT_MESH[mode] = torch.cat(PROMPT_SAMPLE_DICT_MESH[mode])
        
        if self.is_train:
            return QUERY_SAMPLE_DICT_JOINT, PROMPT_SAMPLE_DICT_JOINT, INFO_DICT_JOINT, QUERY_SAMPLE_DICT_MESH, PROMPT_SAMPLE_DICT_MESH, INFO_DICT_MESH
        else:
            if task in ['PE', 'FPE', 'MP', 'MC', 'MIB']:
                return QUERY_SAMPLE_DICT_JOINT, PROMPT_SAMPLE_DICT_JOINT, INFO_DICT_JOINT, 'joint'
            elif task in ['MeshRecover', 'FutureMeshRecover', 'MeshPred', 'MeshCompletion', 'MeshInBetween']:
                return QUERY_SAMPLE_DICT_MESH, PROMPT_SAMPLE_DICT_MESH, INFO_DICT_MESH, 'mesh'


def collate_func_train(batch):
    batch_size = len(batch)
    QUERY_SAMPLE_DICT_JOINT = defaultdict(list)
    PROMPT_SAMPLE_DICT_JOINT = defaultdict(list)
    INFO_DICT_JOINT = defaultdict(list)
    QUERY_SAMPLE_DICT_MESH = defaultdict(list)
    PROMPT_SAMPLE_DICT_MESH = defaultdict(list)
    INFO_DICT_MESH = defaultdict(list)
    
    for b in range(batch_size):
        for mode in batch[0][0].keys():
            QUERY_SAMPLE_DICT_JOINT[mode].append(batch[b][0][mode])
        for mode in batch[0][1].keys():
            PROMPT_SAMPLE_DICT_JOINT[mode].append(batch[b][1][mode])
        for info_key in batch[0][2].keys():
            INFO_DICT_JOINT[info_key] = INFO_DICT_JOINT[info_key] + batch[b][2][info_key]
        for mode in batch[0][3].keys():
            QUERY_SAMPLE_DICT_MESH[mode].append(batch[b][3][mode])
        for mode in batch[0][4].keys():
            PROMPT_SAMPLE_DICT_MESH[mode].append(batch[b][4][mode])
        for info_key in batch[0][5].keys():
            INFO_DICT_MESH[info_key] = INFO_DICT_MESH[info_key] + batch[b][5][info_key]

    for mode in QUERY_SAMPLE_DICT_JOINT.keys():
        QUERY_SAMPLE_DICT_JOINT[mode] = torch.cat(QUERY_SAMPLE_DICT_JOINT[mode])
    for mode in QUERY_SAMPLE_DICT_MESH.keys():
        QUERY_SAMPLE_DICT_MESH[mode] = torch.cat(QUERY_SAMPLE_DICT_MESH[mode])
    
    for mode in PROMPT_SAMPLE_DICT_JOINT.keys():
        PROMPT_SAMPLE_DICT_JOINT[mode] = torch.cat(PROMPT_SAMPLE_DICT_JOINT[mode])
    for mode in PROMPT_SAMPLE_DICT_MESH.keys():
        PROMPT_SAMPLE_DICT_MESH[mode] = torch.cat(PROMPT_SAMPLE_DICT_MESH[mode])

    return QUERY_SAMPLE_DICT_JOINT, PROMPT_SAMPLE_DICT_JOINT, INFO_DICT_JOINT, QUERY_SAMPLE_DICT_MESH, PROMPT_SAMPLE_DICT_MESH, INFO_DICT_MESH

def collate_func_test(batch):
    batch_size = len(batch)
    QUERY_SAMPLE_DICT = defaultdict(list)
    PROMPT_SAMPLE_DICT = defaultdict(list)
    INFO_DICT = defaultdict(list)
    
    for b in range(batch_size):
        for mode in batch[0][0].keys():
            QUERY_SAMPLE_DICT[mode].append(batch[b][0][mode])
        for mode in batch[0][1].keys():
            PROMPT_SAMPLE_DICT[mode].append(batch[b][1][mode])
        for info_key in batch[0][2].keys():
            INFO_DICT[info_key] = INFO_DICT[info_key] + batch[b][2][info_key]

    for mode in QUERY_SAMPLE_DICT.keys():
        QUERY_SAMPLE_DICT[mode] = torch.cat(QUERY_SAMPLE_DICT[mode])    
    for mode in PROMPT_SAMPLE_DICT.keys():
        PROMPT_SAMPLE_DICT[mode] = torch.cat(PROMPT_SAMPLE_DICT[mode])

    return QUERY_SAMPLE_DICT, PROMPT_SAMPLE_DICT, INFO_DICT, batch[0][3]


def prepare_motion_ver0(args, chunk_dict, dataset_name, task, joint_mask, frame_mask, mesh_joint_mask, if_query):
    # chunk_dict
    #   'joint2d': (1,T,17,3) or (N,T,17,3); 
    #   'joint3d': (1,T,17,3) or (N,T,17,3);
    #   'smpl_pose': (1,T,72) or (N,T,72);
    #   'smpl_shape': (1,T,10) or (N,T,10);
    if task in ['FPE', 'MP', 'FutureMeshRecover', 'MeshPred']:
        input_frames = slice(None, args.clip_len)
        target_frames = slice(args.clip_len, None)
    else:
        if args.current_as_history:
            input_frames = slice(None, args.clip_len)
            target_frames = slice(None, args.clip_len)
        else:
            input_frames = slice(args.clip_len, None)
            target_frames = slice(args.clip_len, None)
    
    N = chunk_dict['smpl_shape'].shape[0]


    # INPUT
    sample_dict = {
        'input_tensor': torch.zeros(N,args.clip_len,24,3),
        'input_mask': torch.ones(N,24),
        }
    
    if task in ['PE', 'MeshRecover', 'FPE', 'FutureMeshRecover']:
        sample_dict['input_tensor'][:, :, :args.num_joint, :] = chunk_dict['joint2d'][:, input_frames] - chunk_dict['joint2d'][:, input_frames, 0:1, :]
        sample_dict['input_mask'][:, args.num_joint:] = 0
    elif task in ['MP', 'MC', 'MIB']:
        sample_dict['input_tensor'][:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, input_frames] - chunk_dict['joint3d'][:, input_frames, 0:1, :]
        sample_dict['input_mask'][:, args.num_joint:] = 0
    elif task in ['MeshPred', 'MeshCompletion', 'MeshInBetween']:
        sample_dict['input_tensor'] = chunk_dict['smpl_pose'][:, input_frames]
    else:
        raise NotImplementedError
    
    if task == 'MC':
        assert joint_mask is not None
        sample_dict['input_tensor'] = sample_dict['input_tensor'].clone()
        sample_dict['input_tensor'][:, :, joint_mask] = 0
    elif task == 'MeshCompletion':
        assert mesh_joint_mask is not None
        sample_dict['input_tensor'] = sample_dict['input_tensor'].clone()
        sample_dict['input_tensor'][:, :, mesh_joint_mask] = 0
    elif task in ['MIB', 'MeshInBetween']:
        assert frame_mask is not None
        sample_dict['input_tensor'] = sample_dict['input_tensor'].clone()
        sample_dict['input_tensor'][:, frame_mask] = 0
    
    # TARGET
    if if_query:    # only query
        if (not args.is_train) or args.args.train_simultaneously or task in ['PE', 'FPE', 'MP', 'MC', 'MIB']:        
            sample_dict.update({
                'joint3d': chunk_dict['joint3d'][:, target_frames] - chunk_dict['joint3d'][:, target_frames, 0:1, :],
            })
        if (not args.is_train) or args.args.train_simultaneously or task in ['MeshRecover', 'FutureMeshRecover', 'MeshPred', 'MeshCompletion', 'MeshInBetween']:
            sample_dict.update({
                'smpl_pose': chunk_dict['smpl_pose'][:, target_frames],
                'smpl_shape': chunk_dict['smpl_shape'][:, target_frames],
            })
            
    else:   # only prompt
        sample_dict.update({
            'target_tensor': torch.zeros(N,args.clip_len,24,3),
            'target_mask': torch.ones(N,24),
        })
        if task in ['PE', 'FPE', 'MP', 'MC', 'MIB']:
            sample_dict['target_tensor'][:, :, :args.num_joint, :] = chunk_dict['joint3d'][:, target_frames] - chunk_dict['joint3d'][:, target_frames, 0:1, :]
            sample_dict['target_mask'][:, args.num_joint:] = 0
        elif task in ['MeshRecover', 'FutureMeshRecover', 'MeshPred', 'MeshCompletion', 'MeshInBetween']:
            sample_dict['target_tensor'] = chunk_dict['smpl_pose'][:, target_frames]
        else:
            raise NotImplementedError



    return sample_dict

