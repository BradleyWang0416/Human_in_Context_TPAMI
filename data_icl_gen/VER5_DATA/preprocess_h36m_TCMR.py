import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import joblib

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
sys.path.append(os.path.join(ROOT_DIR, 'third_party'))

from lib.utils.tools import read_pkl
from lib.utils.viz_skel_seq import viz_skel_seq_anim
from third_party.motionbert.lib.utils.utils_smpl import SMPL
from third_party.TCMR.lib.data_utils._img_utils import transfrom_keypoints, normalize_2d_kp


SPIN_J49_TO_H36M_J17 = [39,28,29,30,27,26,25,41,37,43,38,34,35,36,33,32,31]


DATA_ALL = {}

for split in ['train', 'test']:

    data_all = joblib.load(f'data_icl_gen/processed_data/TCMR/data/h36m_{split}_25fps_db.pt')

    for id in tqdm(range(len(data_all['joints2D']))):
        kp_2d = data_all['joints2D'][id, :, :2]
        bbox = data_all['bbox'][id]
        kp_2d, trans = transfrom_keypoints(
            kp_2d=kp_2d,
            center_x=bbox[0],
            center_y=bbox[1],
            width=bbox[2],
            height=bbox[3],
            patch_width=224,
            patch_height=224,
            do_augment=False,
        )
        kp_2d = normalize_2d_kp(kp_2d, 224)
        data_all['joints2D'][id, :, :2] = kp_2d

    data_all['source'] = data_all['vid_name']
    data_all['joint_3d'] = data_all['joints3D'][:, SPIN_J49_TO_H36M_J17, :]
    data_all['joint_2d'] = data_all['joints2D'][:, SPIN_J49_TO_H36M_J17, :]
    data_all['smpl_pose'] = data_all['pose']
    data_all['smpl_shape'] = data_all['shape']

    del data_all['vid_name']
    del data_all['joints3D']
    del data_all['joints2D']
    del data_all['pose']
    del data_all['shape']
    del data_all['features']

    DATA_ALL[split] = data_all

joblib.dump(DATA_ALL, 'data_icl_gen/VER5_DATA/H36M_TCMR/h36m_25fps_db_RENAMED.pt')