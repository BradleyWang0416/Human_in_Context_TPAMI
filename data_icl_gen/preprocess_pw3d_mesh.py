import os
import numpy as np
import torch
from tqdm import tqdm
import pickle

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
sys.path.append(os.path.join(ROOT_DIR, 'third_party'))

from lib.utils.tools import read_pkl
from third_party.motionbert.lib.utils.utils_smpl import SMPL



save_vertices = False




dataset = read_pkl('data_icl_gen/processed_data/PW3D/mesh_det_pw3d.pkl')
"""
    train:
        'joint_2d': (22732,17,2)-ndarray, dtype=float32
        'confidence': (22732, 17, 1)-ndarray, dtype=float32
        'joint_cam': (22732, 17, 3)-ndarray, dtype=float64
        'smpl_pose': (22732, 72)-ndarray, dtype=float64
        'smpl_shape': (22732, 10)-ndarray, dtype=float64
        'img_hw': [(1920, 1920), (1920, 1920), ...]
        'image_id': [array([17139.]), array([17140.]), ...]
        'source': ['courtyard_arguing_000', 'courtyard_arguing_000', ...]
    test:
        'joint_2d': (35515,17,2)-ndarray, dtype=float64
        'confidence': (35515, 17, 1)-ndarray, dtype=float64
        'joint_cam': (35515, 17, 3)-ndarray, dtype=float64
        'smpl_pose': (35515, 72)-ndarray, dtype=float64
        'smpl_shape': (35515, 10)-ndarray, dtype=float64
        'img_hw': 
        'image_id': 
        'source':
"""
smpl = SMPL('third_party/motionbert/data/mesh', batch_size=1)

for split in ['test', 'train']:
    if save_vertices:
        vertices_w_GlobalOrient = []
        vertices_wo_GlobalOrient = []
    joints3d_w_GlobalOrient = []
    joints3d_wo_GlobalOrient = []
    if save_vertices:
        vertices_x1000_w_GlobalOrient = []
        vertices_x1000_wo_GlobalOrient = []
    joints3d_x1000_w_GlobalOrient = []
    joints3d_x1000_wo_GlobalOrient = []
    for i in tqdm(range(len(dataset[split]['smpl_shape']))):
        # if i==100: break
        motion_smpl_pose = dataset[split]['smpl_pose'][i:i+1]   # (1, 72)
        motion_smpl_shape = dataset[split]['smpl_shape'][i:i+1] # (1, 10)
        motion_smpl_pose = torch.from_numpy(motion_smpl_pose).float()
        motion_smpl_shape = torch.from_numpy(motion_smpl_shape).float()
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
                if save_vertices:
                    motion_verts = motion_verts - motion_3d_reg[:, 0:1, :]
                    motion_3d_reg = motion_3d_reg - motion_3d_reg[:, 0:1, :]                       # motion_3d: (1,17,3)    

                if use_global_orient:
                    if scale == 1.:
                        if save_vertices:
                            vertices_w_GlobalOrient.append(motion_verts.cpu().numpy())
                        joints3d_w_GlobalOrient.append(motion_3d_reg.cpu().numpy())
                    elif scale == 1000.:
                        if save_vertices:
                            vertices_x1000_w_GlobalOrient.append(motion_verts.cpu().numpy())
                        joints3d_x1000_w_GlobalOrient.append(motion_3d_reg.cpu().numpy())
                else:
                    if scale == 1.:
                        if save_vertices:
                            vertices_wo_GlobalOrient.append(motion_verts.cpu().numpy())
                        joints3d_wo_GlobalOrient.append(motion_3d_reg.cpu().numpy())
                    elif scale == 1000.:
                        if save_vertices:
                            vertices_x1000_wo_GlobalOrient.append(motion_verts.cpu().numpy())
                        joints3d_x1000_wo_GlobalOrient.append(motion_3d_reg.cpu().numpy())

    if save_vertices:
        vertices_w_GlobalOrient = np.concatenate(vertices_w_GlobalOrient)
        vertices_wo_GlobalOrient = np.concatenate(vertices_wo_GlobalOrient)
    joints3d_w_GlobalOrient = np.concatenate(joints3d_w_GlobalOrient)
    joints3d_wo_GlobalOrient = np.concatenate(joints3d_wo_GlobalOrient)
    if save_vertices:
        vertices_x1000_w_GlobalOrient = np.concatenate(vertices_x1000_w_GlobalOrient)
        vertices_x1000_wo_GlobalOrient = np.concatenate(vertices_x1000_wo_GlobalOrient)
    joints3d_x1000_w_GlobalOrient = np.concatenate(joints3d_x1000_w_GlobalOrient)
    joints3d_x1000_wo_GlobalOrient = np.concatenate(joints3d_x1000_wo_GlobalOrient)

    if save_vertices: assert len(vertices_w_GlobalOrient) == len(dataset[split]['smpl_shape'])
    if save_vertices:
        dataset[split].update({'smpl_vertices_w_GlobalOrient': vertices_w_GlobalOrient})
        dataset[split].update({'smpl_vertices_wo_GlobalOrient': vertices_wo_GlobalOrient})
    dataset[split].update({'joint_3d_w_GlobalOrient': joints3d_w_GlobalOrient})
    dataset[split].update({'joint_3d_wo_GlobalOrient': joints3d_wo_GlobalOrient})
    if save_vertices:
        dataset[split].update({'smpl_vertices_x1000_w_GlobalOrient': vertices_x1000_w_GlobalOrient})
        dataset[split].update({'smpl_vertices_x1000_wo_GlobalOrient': vertices_x1000_wo_GlobalOrient})
    dataset[split].update({'joint_3d_x1000_w_GlobalOrient': joints3d_x1000_w_GlobalOrient})
    dataset[split].update({'joint_3d_x1000_wo_GlobalOrient': joints3d_x1000_wo_GlobalOrient})

    print(f'{split} done')

# save
if not save_vertices:
    save_path = 'data_icl_gen/processed_data/PW3D/mesh_det_pw3d_EXTENDED_woVertex.pkl'
else:
    save_path = 'data_icl_gen/processed_data/PW3D/mesh_det_pw3d_EXTENDED.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(dataset, f)