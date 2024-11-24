import os
import sys
import yaml
from easydict import EasyDict as edict
import numpy as np
import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def viz_img_seq(data, if_node=False, node_size=5,
                lim3d=1, lw=2, if_rot=False, fs=1, azim=30, elev=20, show_axis=True,
                legend=True,
                save=False, fig_title=None, file_folder=None, file_name=None):

    if isinstance(data, np.ndarray):
        frame_n = data.shape[0]

        if if_rot:
            tmp = np.zeros_like(data)
            tmp[..., 0] = data[..., 2]
            tmp[..., 1] = data[..., 0]
            tmp[..., 2] = - data[..., 1]
            data = tmp

        data = {"unnamed key": data}
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

        if if_rot:
            tmp = np.zeros_like(data)
            tmp[..., 0] = data[..., 2]
            tmp[..., 1] = data[..., 0]
            tmp[..., 2] = - data[..., 1]
            data = tmp

        frame_n = data.shape[0]
        data = {"unnamed key": data}
    elif isinstance(data, dict):
        if isinstance(data[list(data.keys())[0]], torch.Tensor):
            data = {key: value.detach().cpu().numpy() for key, value in data.items()}
        frame_n = data[list(data.keys())[0]].shape[0]

        if if_rot:
            for key, value in data.items():
                tmp = np.zeros_like(value)
                tmp[..., 0] = value[..., 2]
                tmp[..., 1] = value[..., 0]
                tmp[..., 2] = - value[..., 1]
                data[key] = tmp

    else:
        raise ValueError('Unsupported input type. (Supported types: np.ndarray, torch.Tensor, dict)')
    
    n_column = frame_n
    n_row = len(data.keys())

    figsize=(n_column*fs, n_row*fs)

    fig, axes = plt.subplots(n_row, n_column, figsize=figsize, subplot_kw=dict(projection='3d'))

    color_dict = {
        'colorful': {"lcolor": '#9b59b6', "rcolor": '#2ecc71'},
        'gray': {"lcolor": '#8e8e8e', "rcolor": '#383838'}
    }

    for row, key in enumerate(data):
        skel_seq = data[key]
        for col, frame in enumerate(range(frame_n)):
            skel_pose = skel_seq[frame]     # (17,3)
            ax = axes[row, col] if n_row != 1 else axes[col]
            ax.view_init(azim=azim, elev=elev)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            ax.set_xlim3d([-1*lim3d, 1*lim3d])
            ax.set_ylim3d([-1*lim3d, 1*lim3d])
            ax.set_zlim3d([-1*lim3d, 1*lim3d])

            if not show_axis:
                ax.axis('off')
                
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            if legend:
                if col == 0: ax.text2D(0, 1.005, key, transform = ax.transAxes)

            color_style = 'gray' if (skel_pose==0).all() else 'colorful'
            colors_LR = color_dict[color_style]

            ax = viz_pose(ax, skel_pose, colors_LR, lw, if_node, node_size)
    
    if legend:
        fig.suptitle(f'{fig_title}')

    plt.tight_layout()
    
    fig.subplots_adjust(hspace=0, wspace=0)

    if not save:
        plt.show()
    else:
        if not os.path.isdir(file_folder):
            os.makedirs(file_folder)
        plt.savefig(f"{file_folder}/{file_name}.png", transparent=True)
        plt.close()



def viz_pose(ax, skel_pose, colors_LR, lw, if_node, node_size):
    if skel_pose.shape[0] == 17:
        connect = [(0,1), (1,2), (2,3),       (0,4), (4,5), (5,6),      (0,7), (7,8), (8,9), (9,10),
                (8,14), (14,15), (15,16),      (8,11), (11,12), (12,13), ]
        LR = [True, True, True,      False, False, False,        True, True, True, True,
            True, True, True,      False, False, False]
    elif skel_pose.shape[0] == 8:
        connect = [(0,1),                       (0,2)           ,      (0,3), (3,4), (4,5),
                        (4,7),                           (4,6), ]
        LR = [True,                               False,                  True, True, True,
                True,                           False]
    elif skel_pose.shape[0] == 2:
        connect = [(0,1)]
        LR = [True]
    LR = [not sign for sign in LR]

    # Start and endpoints of our representation
    I = np.array([touple[0] for touple in connect])
    J = np.array([touple[1] for touple in connect])
    LR = np.array(LR)

    for bone_idx in np.arange(len(I)):
        x = np.array([skel_pose[I[bone_idx], 0], skel_pose[J[bone_idx], 0]])
        y = np.array([skel_pose[I[bone_idx], 1], skel_pose[J[bone_idx], 1]])
        z = np.array([skel_pose[I[bone_idx], 2], skel_pose[J[bone_idx], 2]])
        ax.plot(x, y, z, lw=lw, linestyle='-', c=colors_LR["lcolor"] if LR[bone_idx] else colors_LR["rcolor"])

    if if_node:
        ax.scatter(skel_pose[:, 0], skel_pose[:, 1], skel_pose[:, 2], s=node_size, facecolors='white', edgecolors='black')
    return ax
        

def viz_img_seq_2D(data, save=False, fig_title=None, file_folder=None, file_name=None, lim3d=1, legend=True):

    if isinstance(data, np.ndarray):
        frame_n = data.shape[0]
        data = {"unnamed key": data}
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        frame_n = data.shape[0]
        data = {"unnamed key": data}
    elif isinstance(data, dict):
        if isinstance(data[list(data.keys())[0]], torch.Tensor):
            data = {key: value.detach().cpu().numpy() for key, value in data.items()}
        frame_n = data[list(data.keys())[0]].shape[0]
    else:
        raise ValueError('Unsupported input type. (Supported types: np.ndarray, torch.Tensor, dict)')
    
    n_column = frame_n
    n_row = len(data.keys())

    if not save:
        figsize=(n_column, n_row)
    else:
        figsize=(n_column*3, n_row*3)

    fig, axes = plt.subplots(n_row, n_column, figsize=figsize)

    color_dict = {
        'colorful': {"lcolor": '#9b59b6', "rcolor": '#2ecc71'},
        'gray': {"lcolor": '#8e8e8e', "rcolor": '#383838'}
    }

    for row, key in enumerate(data):
        skel_seq = data[key]
        for col, frame in enumerate(range(frame_n)):
            skel_pose = skel_seq[frame]     # (17,3)
            ax = axes[row, col] if n_row != 1 else axes[col]
            # ax.view_init(azim=30, elev=20)
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            ax.set_xlim([-1*lim3d, 1*lim3d])
            ax.set_ylim([-1*lim3d, 1*lim3d])
            # ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_zticks([])
            if legend:
                if col == 0: ax.text(0, 1.005, key, transform = ax.transAxes)
            
            color_style = 'gray' if (skel_pose==0).all() else 'colorful'
            colors_LR = color_dict[color_style]

            ax = viz_pose_2D(ax, skel_pose, colors_LR)
    
    if legend:
        fig.suptitle(f'{fig_title}')

    plt.tight_layout()

    if not save:
        plt.show()
    else:
        if not os.path.isdir(file_folder):
            os.makedirs(file_folder)
        plt.savefig(f"{file_folder}/{file_name}.png")
        plt.close()



def viz_pose_2D(ax, skel_pose, colors_LR):
    if skel_pose.shape[0] == 17:
        connect = [(0,1), (1,2), (2,3),       (0,4), (4,5), (5,6),      (0,7), (7,8), (8,9), (9,10),
                (8,14), (14,15), (15,16),      (8,11), (11,12), (12,13), ]
        LR = [True, True, True,      False, False, False,        True, True, True, True,
            True, True, True,      False, False, False]
    elif skel_pose.shape[0] == 8:
        connect = [(0,1),                       (0,2)           ,      (0,3), (3,4), (4,5),
                        (4,7),                           (4,6), ]
        LR = [True,                               False,                  True, True, True,
                True,                           False]
    elif skel_pose.shape[0] == 2:
        connect = [(0,1)]
        LR = [True]
    LR = [not sign for sign in LR]

    # Start and endpoints of our representation
    I = np.array([touple[0] for touple in connect])
    J = np.array([touple[1] for touple in connect])
    LR = np.array(LR)

    for bone_idx in np.arange(len(I)):
        x = np.array([skel_pose[I[bone_idx], 0], skel_pose[J[bone_idx], 0]])
        y = np.array([skel_pose[I[bone_idx], 1], skel_pose[J[bone_idx], 1]])
        ax.plot(x, y, lw=4, linestyle='-', c=colors_LR["lcolor"] if LR[bone_idx] else colors_LR["rcolor"])

    return ax

def skel_to_h36m(x, joints_to_h36m):
    # Input: 2xTx25x3 or Tx18x3
    # Output: 2xTx17x3 or Tx18x3
    shape = list(x.shape)
    shape[-2] = 17
    y = np.zeros(shape)
    for i, j in enumerate(joints_to_h36m):  # x是整数, j是列表
        y[..., i, :] = x[..., j, :].mean(-2)
    return y

if __name__ == '__main__':
    # args = get_config('/home/HPL1/Human-in-context/ckpt/exp034/T16.yaml')
    # if args.use_partial_data:
    #     args.data = args.partial_data
    # else:
    #     args.data = args.full_data
    # args.tasks = ['PE', '2D-AR', 'MP', 'MIB']
    # dataset = MotionDataset3D(args, data_split='train')

    # dataset_mp = Subset(dataset, dataset.global_idx_list['MP'])
    # dataset_pe = Subset(dataset, dataset.global_idx_list['PE'])
    # dataset_mib = Subset(dataset, dataset.global_idx_list['MIB'])
    # dataset_ar = Subset(dataset, dataset.global_idx_list['2D-AR'])
    
    # idx = 0
    # prompt, query, task_flag = dataset_mp[idx]
    # query_input = query[:args.data.clip_len]
    # query_target = query[args.data.clip_len:]

    # data_viz = {
    #     'input': query_input, 
    #     'target': query_target
    # }
    data_viz = np.load("/home/HPL1/Human-in-context/data/3DPW_MC/ActionsAll_Ratios35_Frames16_TrainShift1_TestShift4_Interval0_MC/train/00000000.pkl", allow_pickle=True)
    data_viz = np.load("/home/HPL1/Human-in-context/data/AMASS/ActionsAll_ManuInterpV1_Frames16_History10_Future10_TestShift200_TrainShift100_MP/train/00000000.pkl", allow_pickle=True)
    data_viz["data_input"] = data_viz["data_input"] * 2
    # data_viz["data_input"][:, [0,4,5,9,3,14],:] = 0
    data_viz["data_label"] = data_viz["data_label"] * 2
    # data_viz["data_label"][...,1] = - data_viz["data_label"][...,1]
    # viz_img_seq_2D(data_viz, save=True, file_name="FPE_in", file_folder="/home/HPL1/Human-in-context/viz")

    joints_to_h36m=[[2], [0], [3], [6], [1], [4], [7], [5], [8], [8,11], [11], [10], [13], [15], [9], [12], [14]]
    data_viz["data_input"] = skel_to_h36m(data_viz["data_input"], joints_to_h36m)
    data_viz["data_label"] = skel_to_h36m(data_viz["data_label"], joints_to_h36m)
    viz_img_seq(data_viz, save=True, file_name="MP_noaixs", file_folder="/home/HPL1/Human-in-context/viz", zoomin=True)

    










