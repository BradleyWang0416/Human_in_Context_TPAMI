import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
sys.path.append(os.path.join(ROOT_DIR, 'third_party'))

from lib.utils.tools import read_pkl
from lib.utils.viz_skel_seq import viz_skel_seq_anim
from third_party.motionbert.lib.utils.utils_smpl import SMPL

data = read_pkl('data_icl_gen/processed_data/H36M_MESH/mesh_det_h36m.pkl')
"""
    train:
        'joint_2d': (312188,17,2)-ndarray, dtype=float64
        'confidence': (312188, 17, 1)-ndarray, dtype=float32
        'joint_cam': (312188, 17, 3)-ndarray, dtype=float64
        'smpl_pose': (312188, 72)-ndarray, dtype=float64
        'smpl_shape': (312188, 10)-ndarray, dtype=float64
        'camera_name': ['54138969', '54138969', ...]
        'action': ['Direction', 'Direction', ...]
        'source': ['s_01_act_02_subact_01_ca_01', 's_01_act_02_subact_01_ca_01', ...]
    test:
        'joint_2d': (102280,17,2)-ndarray, dtype=float64
        'confidence': (102280, 17, 1)-ndarray, dtype=float32
        'joint_cam': (102280, 17, 3)-ndarray, dtype=float64
        'smpl_pose': (102280, 72)-ndarray, dtype=float64
        'smpl_shape': (102280, 10)-ndarray, dtype=float64
        'camera_name': 
        'action': 
        'source':
"""
smpl = SMPL('third_party/motionbert/data/mesh', batch_size=1)

class SMPL_Dataset(Dataset):
    def __init__(self, data, split):
        self.d = data[split]
    def __len__(self):
        return len(self.d['smpl_shape'])
    def __getitem__(self, index):
        smpl_pose = self.d['smpl_pose'][index]
        smpl_shape = self.d['smpl_shape'][index]
        smpl_pose = torch.from_numpy(smpl_pose).float()
        smpl_shape = torch.from_numpy(smpl_shape).float()
        return smpl_pose, smpl_shape
    
for split in ['test', 'train']:

    dataset = SMPL_Dataset(data, split)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    cnt = 0
    for _, (motion_smpl_pose, motion_smpl_shape) in enumerate(tqdm(dataloader)):
        bs = motion_smpl_shape.shape[0]
        # motion_smpl_pose: [256,72]
        # motion_smpl_shape: [256,10]
        for use_global_orient in [True, False]:
            motion_smpl = smpl(
                betas=motion_smpl_shape,
                body_pose=motion_smpl_pose[:, 3:],
                global_orient=motion_smpl_pose[:, :3] if use_global_orient else torch.zeros_like(motion_smpl_pose[:, :3]),
                pose2rot=True
            )
            for scale in [1., 1000.]:
                motion_verts = motion_smpl.vertices.detach()*scale                         # [1,6890,3]
                J_regressor = smpl.J_regressor_h36m                                    # [17, 6890]
                J_regressor_batch = J_regressor[None, :].expand(motion_verts.shape[0], -1, -1).to(motion_verts.device)  # [1, 17, 6890]
                motion_3d_reg = torch.matmul(J_regressor_batch, motion_verts)                 # motion_3d: (1,17,3)  
                motion_verts = motion_verts - motion_3d_reg[:, 0:1, :]
                motion_3d_reg = motion_3d_reg - motion_3d_reg[:, 0:1, :]                       # motion_3d: (1,17,3)    

                if use_global_orient:
                    if scale == 1.:
                        vertices_w_GlobalOrient = motion_verts.cpu().numpy()
                        joints3d_w_GlobalOrient = motion_3d_reg.cpu().numpy()
                    elif scale == 1000.:
                        vertices_x1000_w_GlobalOrient = motion_verts.cpu().numpy()
                        joints3d_x1000_w_GlobalOrient = motion_3d_reg.cpu().numpy()
                else:
                    if scale == 1.:
                        vertices_wo_GlobalOrient = motion_verts.cpu().numpy()
                        joints3d_wo_GlobalOrient = motion_3d_reg.cpu().numpy()
                    elif scale == 1000.:
                        vertices_x1000_wo_GlobalOrient = motion_verts.cpu().numpy()
                        joints3d_x1000_wo_GlobalOrient = motion_3d_reg.cpu().numpy()
        
        for j in range(bs):
            sample_dict = {
                'joint_2d': data[split]['joint_2d'][cnt],
                'confidence': data[split]['confidence'][cnt],
                'joint_cam': data[split]['joint_cam'][cnt],
                'smpl_pose': data[split]['smpl_pose'][cnt],
                'smpl_shape': data[split]['smpl_shape'][cnt],
                'camera_name': data[split]['camera_name'][cnt],
                'action': data[split]['action'][cnt],
                'source': data[split]['source'][cnt],
                # new key value pairs
                'smpl_vertices_w_GlobalOrient': vertices_w_GlobalOrient[j],
                'smpl_vertices_wo_GlobalOrient': vertices_wo_GlobalOrient[j],
                'joint_3d_w_GlobalOrient': joints3d_w_GlobalOrient[j],
                'joint_3d_wo_GlobalOrient': joints3d_wo_GlobalOrient[j],
                'smpl_vertices_x1000_w_GlobalOrient': vertices_x1000_w_GlobalOrient[j],
                'smpl_vertices_x1000_wo_GlobalOrient': vertices_x1000_wo_GlobalOrient[j],
                'joint_3d_x1000_w_GlobalOrient': joints3d_x1000_w_GlobalOrient[j],
                'joint_3d_x1000_wo_GlobalOrient': joints3d_x1000_wo_GlobalOrient[j],
            }

            cnt += 1

            viz_skel_seq_anim(
                data={
                    'joint_3d_wo_GlobalOrient': sample_dict['joint_3d_wo_GlobalOrient'][None, ...],
                    'joint_3d_w_GlobalOrient': sample_dict['joint_3d_w_GlobalOrient'][None, ...],
                    'smpl_vertices_wo_GlobalOrient': sample_dict['smpl_vertices_wo_GlobalOrient'][None, ...],
                    'smpl_vertices_w_GlobalOrient': sample_dict['smpl_vertices_w_GlobalOrient'][None, ...],
                }, subplot_layout=(2,2), if_node=True, fs=0.5, tight_layout=True
            )
            exit(0)

            # TODO
            # save sample_dict into a list



    print(f'{split} done')

# save
# save_path = 'data_icl_gen/processed_data/H36M_MESH/mesh_det_h36m_EXTENDED.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump(data, f, protocol=4)