import torch
import numpy as np
import os
from os import path as osp
import copy
import pickle
import pandas as pd
from tqdm import tqdm

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'third_party'))
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
from third_party.motionbert.human_body_prior.body_model.body_model import BodyModel
from third_party.motionbert.lib.utils.utils_smpl import SMPL
from lib.utils.viz_skel_seq import viz_skel_seq_anim
from lib.utils.utils_data import split_clips




df = pd.read_csv('data_icl_gen/processed_data/AMASS/fps.csv', sep=',',header=None)
fname_list = list(df[0][1:])
length_list = list(df[3][1:])

all_motions = 'data_icl_gen/processed_data/AMASS/all_motions_fps60.pkl'
motion_data = pickle.load(open(all_motions, 'rb'))  # list of dicts. len=10860. every list element is a dict with keys: ['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']

dataset_id_dict = {'train': dict(), 'test': dict()}
cnt_check = 0
for j in tqdm(range(len(fname_list))):
    source = fname_list[j].split('AMASS_')[-1].rstrip()
    dataset_name = source.split('_')[0]
    if dataset_name in ['ACCAD','MPI','CMU','Eyes','KIT','EKUT','TotalCapture','TCD']:
        if dataset_name not in dataset_id_dict['train']:
            dataset_id_dict['train'][dataset_name] = []
        dataset_id_dict['train'][dataset_name].append(j)
        cnt_check += 1
    elif dataset_name in ['BioMotionLab']:
        if dataset_name not in dataset_id_dict['test']:
            dataset_id_dict['test'][dataset_name] = []
        dataset_id_dict['test'][dataset_name].append(j)
        cnt_check += 1
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))
assert cnt_check == len(fname_list)
print('Dataset info: ')
print('\tTrain')
print('\t\t', {k: len(v) for k,v in dataset_id_dict['train'].items()})
print('\tTest')
print('\t\t', {k: len(v) for k,v in dataset_id_dict['test'].items()})

smpl = SMPL('third_party/motionbert/data/mesh', batch_size=1)
max_len = 2916

for split in dataset_id_dict:
    print(split)
    for dataset_name in dataset_id_dict[split]:
        
        joint_3d_w_GlobalOrient = []
        joint_3d_wo_GlobalOrient = []
        smpl_pose = []
        smpl_shape = []
        source = []

        id_list = dataset_id_dict[split][dataset_name]
        print(f'{dataset_name}: {len(id_list)}')
        for i in tqdm(id_list):
            bdata = motion_data[i]

            time_length = len(bdata['trans'])
            assert time_length == int(length_list[i])
            num_slice = time_length // max_len

            source_name = fname_list[i].split('AMASS_')[-1].rstrip()
            
            for sid in range(num_slice+1):
                start = sid*max_len
                end = min((sid+1)*max_len, time_length)
                T = end - start

                motion_smpl_shape = bdata['betas'][None, :10].repeat(T, 0)  # (10,)->(T,10)
                motion_smpl_pose = bdata['poses'][start:end, :72]                   # (T,72)

                for use_global_orient in [True, False]:
                    motion_smpl = smpl(
                        betas=torch.from_numpy(motion_smpl_shape).float(),        # [16,]-->[T,16]
                        body_pose=torch.from_numpy(motion_smpl_pose[:, 3:]).float(),    # [T,69]
                        global_orient=torch.from_numpy(motion_smpl_pose[:, :3]).float() if use_global_orient else torch.zeros_like(torch.from_numpy(motion_smpl_pose[:, :3]).float()),  # [T,3]
                        pose2rot=True
                    )

                    motion_verts = motion_smpl.vertices.detach()      # (T,6890,3)
                    J_regressor = smpl.J_regressor_h36m     # (17,6890)
                    J_regressor_batch = J_regressor[None, :].expand(T, -1, -1).to(motion_verts.device)  # (T,17,6890)
                    motion_3d_reg = torch.matmul(J_regressor_batch, motion_verts)
                    if use_global_orient:
                        joints3d_w_GlobalOrient = motion_3d_reg.cpu().numpy()
                    else:
                        joints3d_wo_GlobalOrient = motion_3d_reg.cpu().numpy()

                
                
                joint_3d_w_GlobalOrient.append(joints3d_w_GlobalOrient)
                joint_3d_wo_GlobalOrient.append(joints3d_wo_GlobalOrient)
                smpl_pose.append(motion_smpl_pose.astype(np.float32))
                smpl_shape.append(motion_smpl_shape.astype(np.float32))

                assert sid <= 999
                source = source + [os.path.splitext(source_name)[0]+f'_{sid:03d}'] * T


        joint_3d_w_GlobalOrient = np.concatenate(joint_3d_w_GlobalOrient)
        joint_3d_wo_GlobalOrient = np.concatenate(joint_3d_wo_GlobalOrient)
        smpl_pose = np.concatenate(smpl_pose)
        smpl_shape = np.concatenate(smpl_shape)

        all_dict = {
                'joint_3d': joint_3d_w_GlobalOrient,
                'joint_3d_wo_GlobalOrient': joint_3d_wo_GlobalOrient,
                'smpl_pose': smpl_pose,
                'smpl_shape': smpl_shape,
                'source': source
        }
        if not os.path.exists(f'data_icl_gen/VER5_DATA/AMASS/{split}/{dataset_name}'):
            os.makedirs(f'data_icl_gen/VER5_DATA/AMASS/{split}/{dataset_name}')
        save_path = f'data_icl_gen/VER5_DATA/AMASS/{split}/{dataset_name}/amass_fps60_JOINTS_SMPL.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(all_dict, f, protocol=4)
