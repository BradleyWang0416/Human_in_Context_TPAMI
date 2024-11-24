import torch
import torch.nn as nn
from collections import OrderedDict

from third_party.motionbert.lib.utils.utils_mesh import batch_rodrigues
from third_party.motionbert.lib.model.loss import *


class BASE_CLASS(nn.Module):
    def __init__(self):
        super(BASE_CLASS, self).__init__()

    @staticmethod
    def preprocess(data_dict):
        if 'smpl_vertex' in data_dict:  # indicate it's the target dict
            data_dict['smpl_vertex'] = data_dict['smpl_vertex'] - data_dict['joint'][:, :, 0:1, :]
        else: # indicate it's the input dict
            del data_dict['smpl_pose']
            del data_dict['smpl_shape']
        data_dict['joint'] = data_dict['joint'] - data_dict['joint'][:, :, 0:1, :]
        return data_dict
    
    @staticmethod
    def prepare_motion(chunk_dict, dataset_name, task, joint_mask, frame_mask, dataset_args):
        input_dict = OrderedDict({'joint': None, 'smpl_pose': None, 'smpl_shape': None})
        target_dict = OrderedDict({'joint': None, 'smpl_pose': None, 'smpl_shape': None})
        if dataset_args.current_as_history:
            indices = slice(None, dataset_args.clip_len)
        else:
            indices = slice(dataset_args.clip_len, None)

        if task in ['PE', 'MeshRecover']:
            input_dict['joint'] = chunk_dict['joint2d'][:, indices].clone()
            input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
            input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
            target_dict['joint'] = chunk_dict['joint3d'][:, indices].clone()
            target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, indices].clone()
            target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, indices].clone()
        elif task in ['FPE', 'FutureMeshRecover']:
            input_dict['joint'] = chunk_dict['joint2d'][:, :dataset_args.clip_len].clone()
            input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, :dataset_args.clip_len].clone()
            input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, :dataset_args.clip_len].clone()
            target_dict['joint'] = chunk_dict['joint3d'][:, dataset_args.clip_len:].clone()
            target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, dataset_args.clip_len:].clone()
            target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, dataset_args.clip_len:].clone()
        elif task in ['MP', 'MeshPred']:
            input_dict['joint'] = chunk_dict['joint3d'][:, :dataset_args.clip_len].clone()
            input_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, :dataset_args.clip_len].clone()
            input_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, :dataset_args.clip_len].clone()
            target_dict['joint'] = chunk_dict['joint3d'][:, dataset_args.clip_len:].clone()
            target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, dataset_args.clip_len:].clone()
            target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, dataset_args.clip_len:].clone()
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
            assert joint_mask is not None
            input_dict['joint'][:, :, joint_mask] = 0
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
    

class DICT_LOSS(nn.Module):
    def __init__(self, losses_type={}, device='cuda'):
        super(DICT_LOSS, self).__init__()
        # loss_type:
        #       {'mesh': 'L1',
        #       'joint3d': 'MPJPE',
        #       'joint2d': 'MSE'}
        self.device = device
                
        self.losses_type = losses_type
        if losses_type['mesh'] == 'MSE': 
            self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
            self.criterion_regr = nn.MSELoss().to(self.device)
        elif losses_type['mesh'] == 'L1': 
            self.criterion_keypoints = nn.L1Loss(reduction='none').to(self.device)
            self.criterion_regr = nn.L1Loss().to(self.device)
        
    def forward(self, output_dict, target_dict):
        joint3d_output = output_dict['joint3d']
        joint3d_target = target_dict['joint3d']
        joint2d_output = output_dict['joint2d']
        joint2d_target = target_dict['joint2d']
        shape_output = output_dict['smpl_shape']
        shape_target = target_dict['smpl_shape']
        pose_output = output_dict['smpl_pose']
        pose_target = target_dict['smpl_pose']


        joint_loss_dict = self.calculate_joint_loss(joint_output, joint_target)

        joint_loss_dict['loss_2d_pos'] = 0
        raise NotImplementedError
    
    def calculate_joint_loss(self, joint_output, joint_gt):
        loss_dict = {}
        loss_dict['loss_3d_pos'] = loss_mpjpe(joint_output, joint_gt)
        loss_dict['loss_3d_scale'] = n_mpjpe(joint_output, joint_gt)
        loss_dict['loss_3d_velocity'] = loss_velocity(joint_output, joint_gt)
        loss_dict['loss_lv'] = loss_limb_var(joint_output)
        loss_dict['loss_lg'] = loss_limb_gt(joint_output, joint_gt)
        loss_dict['loss_a'] = loss_angle(joint_output, joint_gt)
        loss_dict['loss_av'] = loss_angle_velocity(joint_output, joint_gt)
        return loss_dict
        

    def calcualte_smpl_loss(self, smpl_output, data_gt):
        # to reduce time dimension
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        data_3d_theta = reduce(data_gt['theta'])

        preds = smpl_output[-1]
        pred_theta = preds['theta']
        theta_size = pred_theta.shape[:2]
        pred_theta = reduce(pred_theta)
        preds_local = preds['kp_3d'] - preds['kp_3d'][:, :, 0:1,:]  # (N, T, 17, 3)
        gt_local = data_gt['kp_3d'] - data_gt['kp_3d'][:, :, 0:1,:]
        real_shape, pred_shape = data_3d_theta[:, 72:], pred_theta[:, 72:]
        real_pose, pred_pose = data_3d_theta[:, :72], pred_theta[:, :72]
        loss_dict = {}
        loss_dict['loss_3d_pos'] = loss_mpjpe(preds_local, gt_local)
        loss_dict['loss_3d_scale'] = n_mpjpe(preds_local, gt_local)
        loss_dict['loss_3d_velocity'] = loss_velocity(preds_local, gt_local)
        loss_dict['loss_lv'] = loss_limb_var(preds_local)
        loss_dict['loss_lg'] = loss_limb_gt(preds_local, gt_local)
        loss_dict['loss_a'] = loss_angle(preds_local, gt_local)
        loss_dict['loss_av'] = loss_angle_velocity(preds_local, gt_local)
        
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_norm = torch.norm(pred_theta, dim=-1).mean()
            loss_dict['loss_shape'] = loss_shape 
            loss_dict['loss_pose'] = loss_pose 
            loss_dict['loss_norm'] = loss_norm 
        return loss_dict
        
    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas
