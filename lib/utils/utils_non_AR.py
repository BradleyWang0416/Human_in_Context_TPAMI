import random
import numpy as np
import torch
import math

def generate_masked_joints_seq(seq, drop_ratio):
    '''
    Function: random drop joints
    seq: (F,J,3)
    return: (F,J,3)
    '''
    _, F, _ = seq.shape
    index_range = range(1, F-1)  # range[1,F)
    index_drop = random.sample(index_range, int(drop_ratio * F))  # [drop_ratio * F]
    seq[:, index_drop, :] = 0.  # (F,J,3)
    return seq, index_drop

def skel_to_h36m(x, joints_to_h36m):
    # Input: Tx18x3
    # Output: Tx17x3
    shape = list(x.shape)
    shape[-2] = 17
    if isinstance(x, np.ndarray):
        y = np.zeros(shape)
    elif isinstance(x, torch.Tensor):
        y = torch.zeros(shape)
    for i, j in enumerate(joints_to_h36m):
        y[..., i, :] = x[..., j, :].mean(-2)
    return y

def unify_skeletons(x, dataset, pad='zero', mode='unify'):
    if mode == 'unify':
        shape = list(x.shape)
        shape[-2] = 23

        if isinstance(x, np.ndarray):
            tmp = np.zeros(shape)
        elif isinstance(x, torch.Tensor):
            tmp = torch.zeros(shape)

        if dataset == 'H36M':
            tmp[..., [0, 1,2,3,4, 5,6,7,8, 9,10,11,12,13,14, 15,16,17,18, 19,20,21,22], :] \
            = x[..., [0, 1,2,3,3, 4,5,6,6, 0,7, 7, 8, 9, 10, 14,15,16,16, 11,12,13,13], :]

            # tmp[..., [0, 1,2,3, 5,6,7, 11,12,13,14, 15,16,17, 19,20,21], :] \
            # = x[..., [0, 1,2,3, 4,5,6, 7,8,9,10,    14,15,16, 11,12,13], :]
            # if pad == 'copy':
            #     tmp[..., [4,8,9,10,18,22], :] = \
            #     tmp[..., [3,7,11,11,17,21], :]
            
        elif dataset in ['AMASS', 'PW3D']:
            # tmp[..., [0, 1,2,3,4,  5,6,7,8,  9,10,11,12,13, 15,16,17,18, 19,20,21,22], :] \
            # = x[..., [0, 1,4,7,10, 2,5,8,11, 3,6,9,12,15,   13,16,18,20, 14,17,19,21], :]
            # if pad == 'copy':
            #     tmp[..., [14], :] = \
            #     tmp[..., [13], :]
            tmp[..., [0, 1,2,3,4,  5,6,7,8,  9,10,11,12,13,14, 15,16,17,18, 19,20,21,22], :] \
            = x[..., [0, 1,4,7,10, 2,5,8,11, 3,6,9,12,15,15,   13,16,18,20, 14,17,19,21], :]
        return tmp
    elif mode == 'reverse':
        if dataset == 'H36M':
            return x[..., [0, 1,2,3, 5,6,7, 11,12,13,14, 19,20,21, 15,16,17], :]
        elif dataset in ['AMASS', 'PW3D']:
            return x[..., [0, 1,5,9,2,6,10,3,7,11,4,8,12,15,19,14,16,20,17,21,18,22], :]


def get_complementary_idx(idx_list, max_idx):
    sorted_idx = torch.sort(idx_list)[0]
    full_idx = torch.arange(max_idx+1)
    complementary_idx = full_idx[~torch.isin(full_idx, sorted_idx)]
    cnt = complementary_idx[1:] - complementary_idx[:-1]
    final_idx = []
    for c in range(len(cnt)):
        final_idx = final_idx + [complementary_idx[c].item()] * cnt[c].item()
    assert len(final_idx) == max_idx
    return final_idx


def rotate_y(tensor, angle=None):
    if angle is None: return tensor

    # Rotation matrix around the y-axis
    angle = - math.radians(angle)
    if isinstance(tensor, np.ndarray):
        rotation_matrix = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
        ])
        rotated_tensor = np.einsum('rc, tjc->tjr', rotation_matrix, tensor)

    elif isinstance(tensor, torch.Tensor):
        rotation_matrix = torch.tensor([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
        ])
        rotated_tensor = torch.einsum('rc, tjc->tjr', rotation_matrix, tensor)

    return rotated_tensor


def vector_angle(v1, v2):
    # 计算点积
    dot_product = np.dot(v1, v2)
    # 计算模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_v1 * norm_v2)
    # 使用np.clip防止浮点数精度问题
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # 计算弧度夹角
    angle_radians = np.arccos(cos_theta)
    
    # 计算叉积（二维情况下是标量）
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    
    # 如果叉积为负，说明角度在180度到360度之间
    if cross_product < 0:
        angle_radians = 2 * np.pi - angle_radians

    # 转换为角度
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


def invAAS(seq=10):
    A_inv = torch.zeros(seq, seq)
    for i in range(seq):
        A_inv[i, i] = i + 1
    C = torch.tril(torch.ones(seq, seq))
    C_inv = torch.inverse(C)
    Cinv_Ainv = torch.einsum('ik,kj->ij', C_inv, A_inv)
    return Cinv_Ainv


def AAS(seq=10):
    A = torch.zeros(seq, seq)
    for i in range(seq):
        A[i, i] = 1 / (i + 1)
    C = torch.tril(torch.ones(seq, seq))
    A_C = torch.einsum('ik,kj->ij', A, C)
    return A_C