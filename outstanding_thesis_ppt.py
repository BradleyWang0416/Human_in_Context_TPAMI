# %%
import os
import shutil
import numpy as np
import argparse
import errno
import tensorboardX
from time import time
import random
import prettytable
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from itertools import cycle
from collections import Counter
import importlib

from lib.utils.tools import *
from lib.utils.learning import *
from lib.data.datareader_h36m import DataReaderH36M
from lib.data.dataset_2DAR import ActionRecognitionDataset2D, get_AR_labels, collate_fn_2DAR
from lib.model.loss import *
from lib.utils.viz_skel_seq import viz_skel_seq_anim
from lib.utils.viz_img_seq import viz_img_seq

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def import_class(class_name):
    mod_str, _sep, class_str = class_name.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def import_function(func_name=None):
    """
    动态导入指定的函数。
    
    参数:
    - func_name: 一个字符串，表示函数的全限定名，如 "mymodule.my_function"
    
    返回:
    - 导入的函数对象
    """    
    # 分割模块名和函数名
    module_name, func_name = func_name.rsplit('.', 1)
    
    # 动态导入模块
    module = importlib.import_module(module_name)
    
    # 获取函数对象
    func = getattr(module, func_name)
    
    return func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='ckpt/default', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('-v', '--visualize', action='store_true', help='whether to activate visualization')
    # opts = parser.parse_args()
    opts, _ = parser.parse_known_args()       # 在ipynb中要用这行
    return opts


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, eval_dict):
    print('\tSaving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'eval_dict' : eval_dict
    }, chk_path)

# %%
opts = parse_args()
set_random_seed(opts.seed)
args = get_config(opts.config)

# %%
assert 'bin' not in opts.checkpoint
if args.use_partial_data:
    args.data = args.partial_data
else:
    args.data = args.full_data

# Import specified classes and functions
## dataset AR
dataset_action_recognition_VER = args.func_ver.get('dataset_action_recognition', 1)
dataset_action_recognition = import_class(class_name=f'funcs_and_classes.AR.dataset_AR.ver{dataset_action_recognition_VER}.Dataset_ActionRecognition')
## evaluate AR
evaluate_action_recognition_VER = args.func_ver.get('evaluate_action_recognition', 2)
evaluate_action_recognition = import_function(func_name=f'funcs_and_classes.AR.eval_AR.ver{evaluate_action_recognition_VER}.evaluate_action_recognition')
## train epoch AR
train_epoch_action_recognition_VER = args.func_ver.get('train_epoch_action_recognition', 2)
train_epoch_action_recognition = import_function(func_name=f'funcs_and_classes.AR.train_epoch.ver{train_epoch_action_recognition_VER}.train_epoch')
## dataset non-AR
dataset_VER = args.func_ver.get('dataset_non_AR', 1)
dataset = import_class(class_name=f'funcs_and_classes.Non_AR.dataset.ver{dataset_VER}.MotionDataset3D')
## evaluate non-AR
evaluate_VER = args.func_ver.get('evaluate_non_AR', 1)
evaluate_future_pose_estimation = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate_future_pose_estimation')
evaluate_motion_completion = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate_motion_completion')
evaluate_motion_prediction = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate_motion_prediction')
evaluate_pose_estimation = import_function(func_name=f'funcs_and_classes.Non_AR.eval_funcs.ver{evaluate_VER}.evaluate_pose_estimation')
## train epoch non-AR
train_epoch_VER = args.func_ver.get('train_epoch_non_AR', 1)
train_epoch = import_function(func_name=f'funcs_and_classes.Non_AR.train_epoch.ver{train_epoch_VER}.train_epoch')

# %%
train_dataset = dataset(args, data_split='train')

# %%
# i = 7200
# data = train_dataset[i][1]
# data = data[:256][::5]
# viz_skel_seq_anim(data, if_print=False, file_name=f"{i:08d}", file_folder="tmp", lim3d=0.3, lw=4, if_rot=True, fs=1, azim=-107, elev=8, interval=75)

# %%
i=8113; frames=np.arange(60, 60+128, 1); azim=-66; elev=10; lim3d=0.25; lw=8; fs=1; interval=30; node_size=100; print('frames: ', len(frames)); print("source: '?'"); print("file: '?'")

# %%
# i=5000; frames=np.arange(0, 128, 1); azim=-107; elev=8; lim3d=0.2; lw=8; fs=1; interval=30; node_size=100; print('frames: ', len(frames)); print("source: 's_06_act_07_cam_02'"); print("file: 'S6_Posing_2.55011271_000576.jpg'")

# %%
# i=4950; frames=np.arange(0, 128, 10); azim=-107; elev=8; lim3d=0.5; lw=15; print(len(frames))

# %%
# i=7200; frames=np.arange(0, 128, 10); azim=-40; elev=8; lim3d=0.4; lw=15; print(len(frames))

# %%
data = train_dataset[i][1]
data = data[frames]
# data = data - data[:, [0], :]
data[..., 1] = data[..., 1] + 0.15

# %%
# 尺度
data_bodypart = torch.zeros(len(data), 8, 3)
data_bodypart[:, 0, :] = data[:, 0, :]
data_bodypart[:, 1, :] = data[:, [1,2,3], :].mean(1)
data_bodypart[:, 2, :] = data[:, [4,5,6], :].mean(1)
data_bodypart[:, 3, :] = data[:, [7], :].mean(1)
data_bodypart[:, 4, :] = data[:, [8], :].mean(1)
data_bodypart[:, 5, :] = data[:, [9,10], :].mean(1)
data_bodypart[:, 6, :] = data[:, [11,12,13], :].mean(1)
data_bodypart[:, 7, :] = data[:, [14,15,16], :].mean(1)

data_bipart = torch.zeros(len(data), 2, 3)
data_bipart[:, 0, :] = data[:, [0,1,2,3,4,5,6], :].mean(1)
data_bipart[:, 1, :] = data[:, [7,8,9,10,11,12,13,14,15,16], :].mean(1)


viz_skel_seq_anim({'3D':data[::40], '2D':data[::40,:, :2], '2Ds': {'s1':data[::40,:, :2], 's2':data[::40,:, :2]/2, 's3':data[::40,:, :2]/4}}, mode='img', subplot_layout=(2,2), if_print=0, if_node=1, fig_title=f'{i:08d}', file_name=f"{i:08d}", file_folder="viz_results/outstanding_thesis_ppt", lim3d=lim3d, lim2d=lim3d, lw=2, if_rot=True, fs=0.3, azim=azim, elev=elev, node_size=node_size, interval=interval)
# viz_skel_seq_anim(data, if_print=0, if_node=1, fig_title=f'{i:08d}', file_name=f"{i:08d}", file_folder="viz_results/outstanding_thesis_ppt", lim3d=lim3d, lim2d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval)
# viz_skel_seq_anim(data_bodypart, if_print=1, if_node=1, file_name=f"{i:08d}_bodypart", file_folder="viz_results/outstanding_thesis_ppt", lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval)
# viz_skel_seq_anim(data_bipart, if_print=1, if_node=1, file_name=f"{i:08d}_bipart", file_folder="viz_results/outstanding_thesis_ppt", lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval)
#%%
data_consistent_with_image = data.clone()
data_consistent_with_image[..., 0] = data_consistent_with_image[..., 0] - 0.2
data_consistent_with_image[..., 1] = data_consistent_with_image[..., 1] + 0.005
# viz_skel_seq_anim(data_consistent_with_image, if_print=1, if_node=0, file_name=f"{i:08d}_consistent_with_image", file_folder="viz_results/outstanding_thesis_ppt", lim3d=0.28, lw=lw, if_rot=True, fs=fs, azim=0, elev=0, node_size=node_size, interval=40)
# viz_img_seq(data[::20], save=1, fig_title=f"{i:08d}", file_name=f"{i:08d}", file_folder="viz_results/outstanding_thesis_ppt/black_and_white", azim=0, elev=0, lim3d=0.3, fs=10, lw=10, if_rot=1, show_axis=0, if_node=0)


# %%
# 尺度。骨架尺度变化2D示意图
data_2D = data[[-1], :, :].repeat(2, 1, 1)
data_2D[..., 2] = 0

data_2D[:, 0, 1] = 0
data_2D[:, 1, 1] = 0.02
data_2D[:, 2, 1] = 0.13
data_2D[:, 7:, 1] = data_2D[:, 7:, 1] + 0.06
data_2D[:, 14, 1] = -0.18
data_2D[:, 16, 1] = -0.07
data_2D[:, [4,5,6,11,12,13], 1] = data_2D[:, [1,2,3,14,15,16], 1]

data_2D[:, [0,7,8,9,10], 0] = 0
data_2D[:, 3, 0] = 0.065
data_2D[:, 2, 0] = 0.06
data_2D[:, 1, 0] = 0.04
data_2D[:, 14, 0], data_2D[:, 15, 0], data_2D[:, 16, 0] = 0.03, 0.08, 0.1
data_2D[:, [4,5,6,11,12,13], 0] = - data_2D[:, [1,2,3,14,15,16], 0]

data_bodypart_2D = np.zeros((len(data_2D), 8, 3))
data_bodypart_2D[:, 0, :] = data_2D[:, 0, :]
data_bodypart_2D[:, 1, :] = data_2D[:, [1,2,3], :].mean(1)
data_bodypart_2D[:, 2, :] = data_2D[:, [4,5,6], :].mean(1)
data_bodypart_2D[:, 3, :] = data_2D[:, [7], :].mean(1)
data_bodypart_2D[:, 4, :] = data_2D[:, [8], :].mean(1)
data_bodypart_2D[:, 5, :] = data_2D[:, [9,10], :].mean(1)
data_bodypart_2D[:, 6, :] = data_2D[:, [11,12,13], :].mean(1)
data_bodypart_2D[:, 7, :] = data_2D[:, [14,15,16], :].mean(1)

data_bipart_2D = np.zeros((len(data_2D), 2, 3))
data_bipart_2D[:, 0, :] = data_2D[:, [0,1,2,3,4,5,6], :].mean(1)
data_bipart_2D[:, 1, :] = data_2D[:, [7,8,9,10,11,12,13,14,15,16], :].mean(1)

# viz_skel_seq_anim(data_2D, if_print=1, if_node=1, file_name=f"{i:08d}_2D", file_folder="viz_results/outstanding_thesis_ppt", lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=0, elev=0, node_size=node_size, interval=interval, show_axis=False)
# viz_skel_seq_anim(data_bodypart_2D, if_print=1, if_node=1, file_name=f"{i:08d}_bodypart_2D", file_folder="viz_results/outstanding_thesis_ppt", lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=0, elev=0, node_size=node_size, interval=interval, show_axis=False)
# viz_skel_seq_anim(data_bipart_2D, if_print=1, if_node=1, file_name=f"{i:08d}_bipart_2D", file_folder="viz_results/outstanding_thesis_ppt", lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=0, elev=0, node_size=node_size, interval=interval, show_axis=False)

#%%
# viz_img_seq(data_2D, save=1, fig_title=f"{i:08d}", file_name=f"{i:08d}", file_folder="viz_results/outstanding_thesis_ppt/black_and_white", azim=0, elev=0, lim3d=lim3d, fs=10, lw=lw, if_rot=1, show_axis=0, if_node=1, node_size=300)

#%%
# 时间
special_node = 13
# viz_skel_seq_anim(data, if_print=1, if_node=1, file_name=f"{i:08d}_SpecialNode", file_folder="viz_results/outstanding_thesis_ppt", special_node=special_node, lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval)
# viz_skel_seq_anim(data, if_print=1, if_node=1, file_name=f"{i:08d}_SpecialNode_axisoff", file_folder="viz_results/outstanding_thesis_ppt", special_node=special_node, lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval, show_axis=False)
# viz_skel_seq_anim(data[::15], if_print=1, if_node=1, file_name=f"{i:08d}_SpecialNode_axisoff_skip15", file_folder="viz_results/outstanding_thesis_ppt", special_node=special_node, lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval, show_axis=False)

data_specialnode = data[:, [special_node], :].repeat(1, data.shape[1], 1)
# viz_skel_seq_anim(data_specialnode, if_print=1, if_node=1, file_name=f"{i:08d}_SpecialNode_NodeOnly", file_folder="viz_results/outstanding_thesis_ppt", special_node=np.arange(data.shape[1]), lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval, show_axis=True)


#%%
# 空间
special_node = np.array([4,5,6,13])
special_edge = [(4,13),(5,13),(6,13)]
# viz_skel_seq_anim(data, if_print=1, if_node=1, file_name=f"{i:08d}_SpecialEdge", file_folder="viz_results/outstanding_thesis_ppt", special_node=special_node, special_edge=special_edge, lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval)
# viz_skel_seq_anim(data, if_print=1, if_node=1, file_name=f"{i:08d}_SpecialEdge_axisoff", file_folder="viz_results/outstanding_thesis_ppt", special_node=special_node, special_edge=special_edge, lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval, show_axis=False)
# viz_skel_seq_anim(data[::15], if_print=1, if_node=1, file_name=f"{i:08d}_SpecialEdge_axisoff_skip15", file_folder="viz_results/outstanding_thesis_ppt", special_node=special_node, special_edge=special_edge, lim3d=lim3d, lw=lw, if_rot=True, fs=fs, azim=azim, elev=elev, node_size=node_size, interval=interval, show_axis=False)



#%%
# 四维全局
data_global = {
    'bi part': data_bipart[::15],
    'body part': data_bodypart[::15],
    'original': data[::15],
}
# viz_img_seq(data_global, save=1, file_name=f"{i:08d}_global", file_folder="viz_results/outstanding_thesis_ppt", lim3d=lim3d, lw=15, if_rot=True, fs=10, azim=azim, elev=elev, show_axis=False)


#%%
'''
# 通道。打印x, y, z分别随时间变化GIF
for axis_idx, axis_name in zip([0,2,1], ['x','y','z']):
    fig = plt.figure(figsize=(6.4, 4))
    ax = plt.axes()
    ax.set_xlabel('frame')
    ax.set_ylabel(axis_name)
    ax.set_xlim(0,len(data)) 
    ax.set_ylim(-0.5,0.5) 

    t = np.arange(len(data))

    def animate(i):
        return plt.plot(t[i], -data[i, 13, axis_idx], 'ro')

    my_animation=animation.FuncAnimation(
        fig,
        animate,
        frames=np.arange(len(data)),
        interval=interval,
        blit=True
    )
    # plt.show()
    # plt.axis('off')
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    my_animation.save(f"viz_results/outstanding_thesis_ppt/{i:08d}_{axis_name}.gif", writer='pillow', savefig_kwargs={"transparent": False})
'''