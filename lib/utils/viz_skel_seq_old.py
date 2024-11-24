#%%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#%%
from lib.utils.utils_AR import get_targets

#%%
def viz_skel_seq_anim(data, if_ntu=False, if_node=False, if_origin=False, if_target=False, 
                      lim3d=1, lw=2, if_rot=False, fs=1, azim=40, elev=40, node_size=5,
                      show_axis=True, transparent_background=False,
                      if_print=False, fig_title="unname", file_name=None, file_folder=None, 
                      file_type='gif', mp4_fps=10, interval=75,
                      special_node=None, special_edge=None):
    # data中的元素形状必须是: (t,17,3)

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
    
    # "#9b59b6"紫色, "#2ecc71"绿色
    colors_list = [
        {"lcolor": "#9b59b6", "rcolor": "#2ecc71"},    # 紫+绿
        {"lcolor": np.array([197, 90, 17])/255, "rcolor": np.array([244, 177, 131])/255},    # PE 橙色
        {"lcolor": np.array([0, 176, 240])/255, "rcolor": np.array([157, 195, 230])/255},     # AR 蓝色
        {"lcolor": np.array([146, 208, 80])/255, "rcolor": np.array([197, 224, 180])/255},      # MP 绿色
        {"lcolor": np.array([112, 48, 160])/255, "rcolor": np.array([192, 151, 224])/255},     # 备用 紫色
        {"lcolor": np.array([255, 0, 0])/255, "rcolor": np.array([255, 129, 129])/255},     # 备用 红色
        {"lcolor": np.array([255, 192, 0])/255, "rcolor": np.array([255, 230, 153])/255},     # 备用 黄色
        {"lcolor": 'brown', "rcolor": 'brown'},     
        {"lcolor": 'green', "rcolor": 'green'},   
        {"lcolor": 'blue', "rcolor": 'blue'},     
        {"lcolor": 'yellow', "rcolor": 'yellow'},  
        {"lcolor": 'gray', "rcolor": 'black'},
        {"lcolor": np.array([0.7, 0.7, 0.7]), "rcolor": np.array([0.4, 0.4, 0.4])},
    ]

    fig = plt.figure(figsize=(10*fs, 10*fs))
    ax = plt.axes(projection='3d')
    ax.view_init(azim=azim, elev=elev)
    # ax.set_xlim3d([-1, 1.5])
    # ax.set_ylim3d([-1, 1.5])
    # ax.set_zlim3d([0.0, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ttl = ax.text2D(0, 1.005, '', transform = ax.transAxes)      # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.)

    # vals = np.zeros((17, 3))  # or joints_to_consider
    vals = {}
    for key in data:
        vals[key] = np.zeros((data[key].shape[1],3))
    plots = []
    plots = create_pose(ax, plots, vals, colors_list, update=False, if_node=if_node, if_origin=if_origin, lw=lw, if_target=if_target, if_ntu=if_ntu, node_size=node_size, special_node=special_node, special_edge=special_edge)

    line_anim = animation.FuncAnimation(fig, update, frame_n, fargs=(data, plots, fig, ax, fig_title, ttl, colors_list, if_node, if_origin, lim3d, lw, if_target, if_ntu, node_size, special_node, special_edge), interval=interval, blit=False)

    if not show_axis:
        plt.axis('off')

    # if show z-axis
    # ax.w_zaxis.line.set_lw(0.)
    # ax.set_zticks([])
    # ax.w_xaxis.line.set_lw(0.)
    # ax.set_xticks([])
    # ax.w_yaxis.line.set_lw(0.)
    # ax.set_yticks([])

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    plt.tight_layout()


    if not if_print:
        plt.show()
    else:
        if not os.path.isdir(file_folder):
            os.makedirs(file_folder)

        if file_type == 'gif':
            if not transparent_background:
                line_anim.save(f"{file_folder}/{file_name}.gif", writer='pillow')
            else:
                line_anim.save(f"{file_folder}/{file_name}.gif", writer='pillow', savefig_kwargs={"transparent": True})
        elif file_type == 'mp4':
            plt.rcParams['animation.ffmpeg_path'] = r"/home/HPL1/ffmpeg/ffmpeg-6.0-amd64-static/ffmpeg"
            line_anim.save(f"{file_folder}/{file_name}.mp4", writer=animation.FFMpegWriter(fps=mp4_fps, metadata=dict(artist='Me')))
        plt.close()


def create_pose(ax, plots, vals, colors_list, update=False, if_node=False, if_origin=False, lw=2, if_target=False, if_ntu=False, node_size=5, special_node=None, special_edge=None):

    for vals_per_task_idx, task in enumerate(vals):

        vals_per_task = vals[task]
        colors_per_task = colors_list[vals_per_task_idx]


        if not if_ntu:
            if vals_per_task.shape[0] == 17:
                connect = [(0,1), (1,2), (2,3),       (0,4), (4,5), (5,6),      (0,7), (7,8), (8,9), (9,10),
                        (8,14), (14,15), (15,16),      (8,11), (11,12), (12,13), ]
                ColorFlag = [True, True, True,      False, False, False,        True, True, True, True,
                    True, True, True,      False, False, False]
            elif vals_per_task.shape[0] == 8:
                connect = [(0,1),                       (0,2)           ,      (0,3), (3,4), (4,5),
                        (4,7),                           (4,6), ]
                ColorFlag = [True,                               False,                  True, True, True,
                        True,                           False]
            elif vals_per_task.shape[0] == 2:
                connect = [(0,1)]
                ColorFlag = [True]
        else:
            connect = [(0,1), (1,20), (20,2), (2,3), 
                    (20,4), (4,5), (5,6), (6,7), (7,21), (7,22), 
                    (20,8), (8,9), (9,10), (10,11), (11,23), (11,24), 
                    (0,16), (16,17), (17,18), (18,19), 
                    (0,12), (12,13), (13,14), (14,15)]
            ColorFlag = [True, True, True, True,
                False, False, False, False, False, False,
                True, True, True, True, True, True,
                False, False, False, False,
                True, True, True, True]
                  
        
        SpecialEdgeFlag = [False] * len(connect)
        ls = ['-'] * len(connect)
        lws = [lw] * len(connect)
        colors = []
        for bone_idx in range(len(connect)):
            if not ColorFlag[bone_idx]:
                colors.append(colors_per_task["lcolor"])
            else:
                colors.append(colors_per_task["rcolor"])

        if special_edge is not None:
            connect = connect + special_edge
            SpecialEdgeFlag = SpecialEdgeFlag + [True] * len(special_edge)
            ls = ls + ['--'] * len(special_edge)
            lws = lws + [lw/2] * len(special_edge)
            colors = colors + ['red'] * len(special_edge)


        # Start and endpoints of our representation
        I = np.array([touple[0] for touple in connect])
        J = np.array([touple[1] for touple in connect])

        
        for bone_idx in np.arange(len(I)):
            x = np.array([vals_per_task[I[bone_idx], 0], vals_per_task[J[bone_idx], 0]])
            y = np.array([vals_per_task[I[bone_idx], 1], vals_per_task[J[bone_idx], 1]])
            z = np.array([vals_per_task[I[bone_idx], 2], vals_per_task[J[bone_idx], 2]])
            if not update:
                plots.append(ax.plot(x, y, z, lw=lws[bone_idx], linestyle=ls[bone_idx], c=colors[bone_idx], label=task if bone_idx==0 else ''))
            elif update:
                plots[vals_per_task_idx*(len(I)+1) + bone_idx][0].set_xdata(x)
                plots[vals_per_task_idx*(len(I)+1) + bone_idx][0].set_ydata(y)
                plots[vals_per_task_idx*(len(I)+1) + bone_idx][0].set_3d_properties(z)
                plots[vals_per_task_idx*(len(I)+1) + bone_idx][0].set_color(colors[bone_idx])

    
        node_colors = np.array(['black' for _ in range(len(vals_per_task))])
        sizes = np.array([node_size for _ in range(len(vals_per_task))])
        if special_node is not None:
            node_colors[special_node] = 'red'
            sizes[special_node] = node_size * 5
        vals_per_task = np.concatenate((vals_per_task, np.zeros((1,3))), axis=0)      # (17,3)||(1,3)->(18,3)
        node_colors = np.append(node_colors, 'black')
        sizes = np.append(sizes, node_size)

        if if_node and if_origin:
            if not update:
                plots.append(ax.scatter(vals_per_task[:, 0], vals_per_task[:, 1], vals_per_task[:, 2], c=node_colors, s=sizes))
            elif update:
                plots[vals_per_task_idx*(len(I)+1) + len(I)]._offsets3d = (vals_per_task[:, 0], vals_per_task[:, 1], vals_per_task[:, 2])
        elif if_node:
            if not update:
                plots.append(ax.scatter(vals_per_task[:-1, 0], vals_per_task[:-1, 1], vals_per_task[:-1, 2], c=node_colors[:-1], s=sizes[:-1]))
            elif update:
                plots[vals_per_task_idx*(len(I)+1) + len(I)]._offsets3d = (vals_per_task[:-1, 0], vals_per_task[:-1, 1], vals_per_task[:-1, 2])
        elif if_origin:
            if not update:
                plots.append(ax.scatter(vals_per_task[-1:, 0], vals_per_task[-1:, 1], vals_per_task[-1:, 2], c=node_colors[-1], s=sizes[-1]))
            elif update:
                plots[vals_per_task_idx*(len(I)+1) + len(I)]._offsets3d = (vals_per_task[-1:, 0], vals_per_task[-1:, 1], vals_per_task[-1:, 2])
        else:
            if not update:
                plots.append(ax.scatter(vals_per_task[-1:, 0], vals_per_task[-1:, 1], vals_per_task[-1:, 2], c=node_colors[-1], s=0.5))
            elif update:
                plots[vals_per_task_idx*(len(I)+1) + len(I)]._offsets3d = (vals_per_task[-1:, 0], vals_per_task[-1:, 1], vals_per_task[-1:, 2])

    return plots


def update(num, data, plots, fig, ax, fig_title, ttl, colors_list, if_node, if_origin, lim3d=1, lw=2, if_target=False, if_ntu=False, node_size=5, special_node=None, special_edge=None):
    # vals = data[num]
    vals = {}
    for key in data:
        vals[key] = data[key][num]

    plots = create_pose(ax, plots, vals, colors_list, update=True, if_node=if_node, if_origin=if_origin, lw=lw, if_target=if_target, if_ntu=if_ntu, node_size=node_size, special_node=special_node, special_edge=special_edge)
    # r = 0.4
    # xroot, zroot, yroot = vals[0, 0], vals[0, 1], vals[0, 2]
    # ax.set_xlim3d([-r + xroot, r + xroot])
    # ax.set_ylim3d([-r + yroot, r + yroot])
    # ax.set_zlim3d([-r + zroot, r + zroot])
    ax.set_xlim3d([-1*lim3d, 1*lim3d])
    ax.set_ylim3d([-1*lim3d, 1*lim3d])
    ax.set_zlim3d([-1*lim3d, 1*lim3d])


    if if_target:
        targets = get_targets().data.cpu().numpy()
        ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='gray', s=5)
        # if not update:
        #     plots.append(ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='gray', s=5))
        # else:
        #     plots[-1]._offsets3d(targets[:, 0], targets[:, 1], targets[:, 2])


    # ax.set_title(fig_title+f" | frame {num+1}")
    ttl.set_text(fig_title+f" | frame {num+1}")

    ax.legend(loc='upper right')

    return plots
#%%

if __name__ == "__main__":
    targets = []
    x_coors = np.linspace(-1, 1, 4)
    y_coors = np.linspace(-1, 1, 4)
    z_coors = np.linspace(-1, 1, 4)
    for x in x_coors:
        for y in y_coors:
            for z in z_coors:
                targets.append(np.array([x, y, z]))
    targets = np.stack(targets[:60])   # (60, 3)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    c = ['gray' for _ in range(60)]
    c[21] = 'red'
    c[22] = 'green'
    c[25] = 'blue'
    c[37] = 'orange'
    c[42] = 'purple'
    ax.scatter(targets[:,0], targets[:,1], targets[:,2], c=c)
    ax.view_init(azim=30, elev=20)
    plt.show()
    