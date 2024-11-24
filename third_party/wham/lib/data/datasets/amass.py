from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import joblib

from lib.utils import transforms

from configs import constants as _C

from ..utils.augmentor import *
from .._dataset import BaseDataset
from ...models import build_body_model
from ...utils import data_utils as d_utils
from ...utils.kp_utils import root_centering

from ...utils.viz_skel_seq import viz_skel_seq_anim


def compute_contact_label(feet, thr=1e-2, alpha=5):
    vel = torch.zeros_like(feet[..., 0])
    label = torch.zeros_like(feet[..., 0])
    
    vel[1:-1] = (feet[2:] - feet[:-2]).norm(dim=-1) / 2.0
    vel[0] = vel[1].clone()
    vel[-1] = vel[-2].clone()
    
    label = 1 / (1 + torch.exp(alpha * (thr ** -1) * (vel - thr)))
    return label


class AMASSDataset(BaseDataset):
    def __init__(self, cfg):
        label_pth = _C.PATHS.AMASS_LABEL                                        # label_pth = 'dataset/parsed_data/amass.pth'
        super(AMASSDataset, self).__init__(cfg, training=True)

        self.supervise_pose = cfg.TRAIN.STAGE == 'stage1'
        self.labels = joblib.load(label_pth)
        self.SequenceAugmentor = SequenceAugmentor(cfg.DATASET.SEQLEN + 1)

        # Load augmentators
        self.VideoAugmentor = VideoAugmentor(cfg)
        self.SMPLAugmentor = SMPLAugmentor(cfg)
        self.d_img_feature = _C.IMG_FEAT_DIM[cfg.MODEL.BACKBONE]
        
        self.n_frames = int(cfg.DATASET.SEQLEN * self.SequenceAugmentor.l_factor) + 1
        self.smpl = build_body_model('cpu', self.n_frames)
        self.prepare_video_batch()
        
        # Naive assumption of image intrinsics
        self.img_w, self.img_h = 1000, 1000
        self.get_naive_intrinsics((self.img_w, self.img_h))
        
        self.CameraAugmentor = CameraAugmentor(cfg.DATASET.SEQLEN + 1, self.img_w, self.img_h, self.focal_length)
        
        
    @property
    def __name__(self, ):
        return 'AMASS'
    
    def get_input(self, target, get_h36m_skel=False):
        gt_kp3d = target['kp3d']
        inpt_kp3d = self.VideoAugmentor(gt_kp3d[:, :self.n_joints, :-1].clone())        # [82,17,3]
        kp2d = perspective_projection(inpt_kp3d, self.cam_intrinsics)                   # [82,17,2]
        mask = self.VideoAugmentor.get_mask()                                           # [82,17]
        kp2d, bbox = self.keypoints_normalizer(kp2d, target['res'], self.cam_intrinsics, 224, 224)      # [82,37]. 37=17*2+3
        
        target['bbox'] = bbox[1:]
        target['kp2d'] = kp2d
        target['mask'] = mask[1:]
        target['features'] = torch.zeros((self.SMPLAugmentor.n_frames, self.d_img_feature)).float()

        if get_h36m_skel:
            gt_kp3d_h36m = target['kp3d_h36m']
            inpt_kp3d_h36m = self.VideoAugmentor(gt_kp3d_h36m[:, :, :-1].clone())        # [82,17,3]
            kp2d_h36m = perspective_projection(inpt_kp3d_h36m, self.cam_intrinsics)                   # [82,17,2]
            kp2d_h36m = perspective_projection(gt_kp3d_h36m[:,:,:-1], self.cam_intrinsics)                   # [82,17,2]
            kp2d_h36m, bbox = self.keypoints_normalizer(kp2d_h36m, target['res'], self.cam_intrinsics, 224, 224)      # [82,37]. 37=17*2+3
            target['kp2d_h36m'] = kp2d_h36m
            


        return target
    
    def get_groundtruth(self, target, get_h36m_skel=False):
        # GT 1. Joints
        gt_kp3d = target['kp3d']                                                        # [82,31,3]
        gt_kp2d = perspective_projection(gt_kp3d, self.cam_intrinsics)                  # [82,31,2]
        target['kp3d'] = torch.cat((gt_kp3d, torch.ones_like(gt_kp3d[..., :1]) * float(self.supervise_pose)), dim=-1)
        target['full_kp2d'] = torch.cat((gt_kp2d, torch.ones_like(gt_kp2d[..., :1]) * float(self.supervise_pose)), dim=-1)[1:]
        target['weak_kp2d'] = torch.zeros_like(target['full_kp2d'])
        target['init_kp3d'] = root_centering(gt_kp3d[:1, :self.n_joints].clone()).reshape(1, -1)
        target['verts'] = torch.zeros((self.SMPLAugmentor.n_frames, 6890, 3)).float()

        if get_h36m_skel:
            gt_kp3d_h36m = target['kp3d_h36m']                                                        # [82,31,3]
            gt_kp2d_h36m = perspective_projection(gt_kp3d_h36m, self.cam_intrinsics)                  # [82,31,2]
            target['kp3d_h36m'] = torch.cat((gt_kp3d_h36m, torch.ones_like(gt_kp3d_h36m[..., :1]) * float(self.supervise_pose)), dim=-1)
            target['full_kp2d_h36m'] = torch.cat((gt_kp2d_h36m, torch.ones_like(gt_kp2d_h36m[..., :1]) * float(self.supervise_pose)), dim=-1)[1:]

        
        # GT 2. Root pose
        vel_world = (target['transl'][1:] - target['transl'][:-1])
        pose_root = target['pose_root'].clone()
        vel_root = (pose_root[:-1].transpose(-1, -2) @ vel_world.unsqueeze(-1)).squeeze(-1)
        target['vel_root'] = vel_root.clone()
        target['pose_root'] = transforms.matrix_to_rotation_6d(pose_root)
        target['init_root'] = target['pose_root'][:1].clone()
        
        # GT 3. Foot contact
        contact = compute_contact_label(target['feet'])
        if 'tread' in target['vid']:
            target['contact'] = torch.ones_like(contact) * (-1)
        else:
            target['contact'] = contact
        
        return target
    
    def forward_smpl(self, target, get_h36m_skel=False):
        output = self.smpl.get_output(
            body_pose=torch.cat((target['init_pose'][:, 1:], target['pose'][1:, 1:])),
            global_orient=torch.cat((target['init_pose'][:, :1], target['pose'][1:, :1])),
            betas=target['betas'],
            pose2rot=False,
            get_h36m_skel=get_h36m_skel)
        
        target['transl'] = target['transl'] - output.offset
        target['transl'] = target['transl'] - target['transl'][0]
        target['kp3d'] = output.joints                                                      # tensor [82,31,3]
        if get_h36m_skel:
            target['kp3d_h36m'] = output.joints_h36m
            viz_skel_seq_anim({
                'kp3d wham': target['kp3d'],
                'kp3d h36m': target['kp3d_h36m']
            },subplot_layout=(1,2), if_node=True)
        target['feet'] = output.feet[1:] + target['transl'][1:].unsqueeze(-2)
        
        return target
    
    def augment_data(self, target, get_h36m_skel=False):
        # Augmentation 1. SMPL params augmentation
        target = self.SMPLAugmentor(target)
        
        # Augmentation 2. Sequence speed augmentation
        target = self.SequenceAugmentor(target)                    # [82,24,3,3]. 这一步会改变时间帧数，从122变成了82

        # Get world-coordinate SMPL
        target = self.forward_smpl(target, get_h36m_skel=get_h36m_skel)                         # target.keys() = dict_keys(['res', 'cam_intrinsics', 'has_full_screen', 'has_smpl', 'has_traj', 'has_verts', 'vid', 'pose', 'transl', 'betas', 'init_pose', 'kp3d', 'feet'])
        
        # Augmentation 3. Virtual camera generation
        target = self.CameraAugmentor(target)                      # target.keys() = dict_keys(['res', 'cam_intrinsics', 'has_full_screen', 'has_smpl', 'has_traj', 'has_verts', 'vid', 'pose', 'transl', 'betas', 'init_pose', 'kp3d', 'feet',
                                                                   #                            'R', 'T', 'transl_cam', 'pose_root', 'cam_angvel'])
        return target
    
    def load_amass(self, index, target):
        start_index, end_index = self.video_indices[index]                      # self.video_indices[29983] = [4420237, 4420358]
        
        # Load AMASS labels
        pose = torch.from_numpy(self.labels['pose'][start_index:end_index+1].copy())            # [122,72]
        pose = transforms.axis_angle_to_matrix(pose.reshape(-1, 24, 3))                         # [122,24,3,3]
        transl = torch.from_numpy(self.labels['transl'][start_index:end_index+1].copy())
        betas = torch.from_numpy(self.labels['betas'][start_index:end_index+1].copy())
        
        # Stack GT
        target.update({'vid': self.labels['vid'][start_index], 
                  'pose': pose, 
                  'transl': transl, 
                  'betas': betas})

        return target

    def get_single_sequence(self, index):
        target = {'res': torch.tensor([self.img_w, self.img_h]).float(),
                  'cam_intrinsics': self.cam_intrinsics.clone(),                # [Train stage 1; index=29983] tensor of shape [1,3,3]
                  'has_full_screen': torch.tensor(True),
                  'has_smpl': torch.tensor(self.supervise_pose),                # [Train srage 1; index=29983] True
                  'has_traj': torch.tensor(True),
                  'has_verts': torch.tensor(False),}
        
        get_h36m_skel = True
        viz_h36m = get_h36m_skel

        target = self.load_amass(index, target)
        target = self.augment_data(target, get_h36m_skel=get_h36m_skel)
        target = self.get_groundtruth(target, get_h36m_skel=get_h36m_skel)
        target = self.get_input(target, get_h36m_skel=get_h36m_skel)
        
        target = d_utils.prepare_keypoints_data(target)
        target = d_utils.prepare_smpl_data(target)
        
        if viz_h36m:
            data_viz = {
                'kp3d wham': target['kp3d'][:, :, :3] - target['kp3d'][:, [0], :3],
                'kp3d h36m': target['kp3d_h36m'][1:, :, :3] - target['kp3d_h36m'][1:, [0], :3],
                'kp2d': target['kp2d'][:, :34].reshape(-1, 17, 2),
                'kp2d h36m': target['kp2d_h36m'][:, :34].reshape(-1, 17, 2),
                'full 2d wham': target['full_kp2d'][:, :, :2] - target['full_kp2d'][:, [0], :2],
                'full 2d h36m': target['full_kp2d_h36m'][:, :, :2] - target['full_kp2d_h36m'][:, [0], :2],
            }
            viz_skel_seq_anim(data_viz, subplot_layout=(1,6), fs=0.6, lim3d=1, if_node=1)

        return target
        # target: {
        #           'kp3d':       [81,31,4],
        #           'full_kp2d':  [81,31,3],
        #           'weak_kp2d':  [81,31,1],
        #           'init_kp3d':  [1,51],
        #           'kp2d':       [81,37],        37=17*2+3. 在self.get_input中由31变成了17
        #           'init_kp2d':  [1,37],
        #          }
    

def perspective_projection(points, cam_intrinsics, rotation=None, translation=None):
    K = cam_intrinsics
    if rotation is not None:
        points = torch.matmul(rotation, points.transpose(1, 2)).transpose(1, 2)
    if translation is not None:
        points = points + translation.unsqueeze(1)
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points.float())
    return projected_points[:, :, :-1]