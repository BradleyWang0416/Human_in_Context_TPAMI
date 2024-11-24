from cgi import test
import chunk
from email.quoprimime import body_check
from logging import config
import sys
import os
from tabnanny import check
from tracemalloc import is_tracing
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import pickle
import time
import joblib
from collections import defaultdict
from tqdm import tqdm
import copy

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

from funcs_and_classes.Non_AR.dataset.ver5_ICL import MotionDatasetICL as MotionDatasetICL_VER5


class MotionDatasetICL(MotionDatasetICL_VER5):
    def __init__(self, args, data_split, TASK=None, DATASET_NAME=None, SLICED_DATA=None):
        super().__init__(args, data_split, TASK, DATASET_NAME, SLICED_DATA)

    def __getitem__(self, query_index):
        dataset_name, query_chunk_id = self.query_list[query_index]
        prompt_chunk_id = random.choice(self.prompt_list[dataset_name])

        if not self.is_train:
            joint_mask = self.joint_mask_dict[dataset_name][query_chunk_id]
            frame_mask = self.frame_mask_dict[dataset_name][query_chunk_id]
            joint_mask = torch.from_numpy(joint_mask)
            frame_mask = torch.from_numpy(frame_mask)

        query_chunk_dict = self.prepare_chunk(self.query_dict, dataset_name, chunk_id=query_chunk_id)
        query_chunk_dict = {k: v.squeeze(0) for k, v in query_chunk_dict.items()}
        # 'joint2d': [32, 17, 3]
        # 'joint3d': [32, 17, 3]
        # 'smpl_pose': [32, 72]
        # 'smpl_shape': [32, 10]
        prompt_chunk_dict = self.prepare_chunk(self.prompt_dict, dataset_name, chunk_id=prompt_chunk_id)
        prompt_chunk_dict = {k: v.squeeze(0) for k, v in prompt_chunk_dict.items()}
        # 'joint2d': [32, 17, 3]
        # 'joint3d': [32, 17, 3]
        # 'smpl_pose': [32, 72]
        # 'smpl_shape': [32, 10]

        info_dict = {
            'dataset': dataset_name,
            'query_index': query_index,
            'query_chunk_id': query_chunk_id,
            'prompt_chunk_id': prompt_chunk_id,
            'use_global_orient': int(self.dataset_config[dataset_name]['use_global_orient']),
        }
        if not self.is_train:
            info_dict['joint_mask'] = joint_mask
            info_dict['frame_mask'] = frame_mask

        query_target_dict = copy.deepcopy(query_chunk_dict)
        prompt_target_dict = copy.deepcopy(prompt_chunk_dict)

        return query_chunk_dict, prompt_chunk_dict, query_target_dict, prompt_target_dict, info_dict
