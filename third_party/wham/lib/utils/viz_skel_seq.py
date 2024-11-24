#%%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

#%%
# from lib.utils.utils_AR import get_targets

#%%
def viz_skel_seq_anim(data, mode='animation', if_node=False, if_origin=False, if_target=False, 
                      lim3d=1, lim2d=1, lw=2, if_rot=False, fs=1, azim=40, elev=40, node_size=5,
                      show_axis=True, transparent_background=False,
                      if_print=False, fig_title="unname", file_name=None, file_folder=None, 
                      subplot_layout=(1,1), file_type='gif', mp4_fps=10, interval=75,
                      special_node=None, special_edge=None):
    
    # data中的元素形状必须是: (T,edge_ed,3)
    flag_2D = []
    if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        if isinstance(data, torch.Tensor): data = data.detach().cpu().numpy()
        if if_rot: 
            data = rotate_pose(data)
        frame_n = data.shape[0]
        if data.shape[-1] == 3: 
            flag_2D.append(False)
        else:
            flag_2D.append(True)
        init_pose = {"unnamed key": np.zeros(data.shape[1:])}
        data = {"unnamed key": data}
        
    elif isinstance(data, dict):
        init_pose = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                if isinstance(value, torch.Tensor): data[key] = value.detach().cpu().numpy()
                if if_rot:
                    data[key] = rotate_pose(data[key])
                frame_n = value.shape[0]
                if data[key].shape[-1] == 3:
                    flag_2D.append(False)
                else:
                    flag_2D.append(True)
                init_pose[key] = np.zeros(data[key].shape[1:])

            elif isinstance(value, dict):
                init_pose[key] = {}
                for v_key, v_value in value.items():
                    if isinstance(v_value, torch.Tensor):
                        data[key][v_key] = v_value.detach().cpu().numpy()
                    if if_rot:
                        data[key][v_key] = rotate_pose(data[key][v_key])
                    frame_n = v_value.shape[0]
                    init_pose[key][v_key] = np.zeros(data[key][v_key].shape[1:])

                if list(data[key].values())[0].shape[-1] == 3: 
                    flag_2D.append(False)
                else:
                    flag_2D.append(True)

            else:
                raise ValueError(f'Unsupported input type. Got {type(value)}')
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

    n_row, n_col = subplot_layout
    if mode == 'img':
        fig, axes = plt.subplots(n_row, frame_n*n_col, squeeze=False, figsize=(frame_n*n_col*10*fs, n_row*10*fs), subplot_kw=dict(projection='3d'))
        axes = axes.reshape(n_row, n_col, frame_n)        
    else:
        fig, axes = plt.subplots(n_row, n_col, squeeze=False, figsize=(n_col*10*fs, n_row*10*fs), subplot_kw=dict(projection='3d'))

    plots = {}
    ttls = []
    for idx, (key, value) in enumerate(init_pose.items()):

        if mode == 'img':
            for frame in range(frame_n):
                if flag_2D[idx] == True:
                    axes[idx//n_col, idx%n_col, frame].remove()
                    axes[idx//n_col, idx%n_col, frame] = fig.add_subplot(n_row, frame_n*n_col, (frame_n*idx)+frame + 1)
                ax = axes[idx//n_col, idx%n_col, frame]

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                # ax.xaxis.set_ticklabels([])
                # ax.yaxis.set_ticklabels([])
                if ax.name == '3d':
                    ax.set_xlim3d([-1*lim3d, 1*lim3d])
                    ax.set_ylim3d([-1*lim3d, 1*lim3d])
                    ax.set_zlim3d([-1*lim3d, 1*lim3d])
                    ax.set_zlabel('Z')
                    if isinstance(azim, int):
                        ax.view_init(azim=azim, elev=elev)
                    if isinstance(azim, list):
                        ax.view_init(azim=azim[idx], elev=elev[idx])
                    # ax.zaxis.set_ticklabels([])    
                    ttls.append(ax.text2D(0, 1.005, key+f'-{frame}', transform = ax.transAxes))      # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.)
                else:
                    ax.set_xlim([-1*lim2d, 1*lim2d])
                    ax.set_ylim([-1*lim2d, 1*lim2d])
                    ttls.append(ax.text(0, 1.005, key+f'-{frame}', transform = ax.transAxes))      # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.)                   

                if not isinstance(value, dict):
                    colors = colors_list[idx]
                    plots[key] = []
                    plots[key] = create_pose(ax, plots[key], data[key][frame], colors, label=key, update=False, if_node=if_node, if_origin=if_origin, lw=lw, node_size=node_size, special_node=special_node, special_edge=special_edge)
                else:
                    plots[key] = {}
                    for v_idx, (v_key, v_value) in enumerate(init_pose[key].items()):
                        colors = colors_list[v_idx]
                        plots[key][v_key] = []
                        plots[key][v_key] = create_pose(ax, plots[key][v_key], data[key][v_key][frame], colors, label=v_key, update=False, if_node=if_node, if_origin=if_origin, lw=lw, node_size=node_size, special_node=special_node, special_edge=special_edge)
                
                    ax.legend(loc='upper right')

                if if_target:
                    assert ax.name == '3d'
                    targets = get_targets().data.cpu().numpy()
                    ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='gray', s=5)
                

        else:
            if flag_2D[idx] == True:
                axes[idx//n_col, idx%n_col].remove()
                axes[idx//n_col, idx%n_col] = fig.add_subplot(n_row, n_col, idx + 1)
            
            ax = axes[idx//n_col, idx%n_col]

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            # ax.xaxis.set_ticklabels([])
            # ax.yaxis.set_ticklabels([])
            if ax.name == '3d':
                ax.set_zlabel('Z')
                if isinstance(azim, int):
                    ax.view_init(azim=azim, elev=elev)
                if isinstance(azim, list):
                    ax.view_init(azim=azim[idx], elev=elev[idx])
                # ax.zaxis.set_ticklabels([])    
                ttls.append(ax.text2D(0, 1.005, '', transform = ax.transAxes))      # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.)
            else:
                ttls.append(ax.text(0, 1.005, '', transform = ax.transAxes))      # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.)

            if not isinstance(value, dict):
                colors = colors_list[idx]
                plots[key] = []
                plots[key] = create_pose(ax, plots[key], value, colors, label=key, update=False, if_node=if_node, if_origin=if_origin, lw=lw, node_size=node_size, special_node=special_node, special_edge=special_edge)
            else:
                plots[key] = {}
                for v_idx, (v_key, v_value) in enumerate(init_pose[key].items()):
                    colors = colors_list[v_idx]
                    plots[key][v_key] = []
                    plots[key][v_key] = create_pose(ax, plots[key][v_key], v_value, colors, label=v_key, update=False, if_node=if_node, if_origin=if_origin, lw=lw, node_size=node_size, special_node=special_node, special_edge=special_edge)


    fig.suptitle(f'{fig_title}')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0, wspace=0)
    if not show_axis:
        plt.axis('off')

    if mode != 'img':
        line_anim = animation.FuncAnimation(fig, update, frame_n, fargs=(data, plots, fig, axes, fig_title, ttls, colors_list, if_node, if_origin, lim3d, lim2d, lw, if_target, node_size, special_node, special_edge), interval=interval, blit=False)

    if not if_print:
        plt.show()
    else:
        if not os.path.isdir(file_folder):
            os.makedirs(file_folder)

        if mode == 'img':
            plt.savefig(f"{file_folder}/{file_name}.png", transparent=True if transparent_background else False)
        else:
            if file_type == 'gif':
                if not transparent_background:
                    line_anim.save(f"{file_folder}/{file_name}.gif", writer='pillow')
                else:
                    line_anim.save(f"{file_folder}/{file_name}.gif", writer='pillow', savefig_kwargs={"transparent": True})
            elif file_type == 'mp4':
                plt.rcParams['animation.ffmpeg_path'] = r"/home/HPL1/ffmpeg/ffmpeg-6.0-amd64-static/ffmpeg"
                line_anim.save(f"{file_folder}/{file_name}.mp4", writer=animation.FFMpegWriter(fps=mp4_fps, metadata=dict(artist='Me')))
        plt.close()


def create_pose(ax, plot, pose, colors, label=None, update=False, if_node=False, if_origin=False, lw=2, node_size=5, special_node=None, special_edge=None):

    if pose.shape[0] == 17:
        connect = [(0,1), (1,2), (2,3),       (0,4), (4,5), (5,6),      (0,7), (7,8), (8,9), (9,10),
                (8,14), (14,15), (15,16),      (8,11), (11,12), (12,13), ]
        LeftRightFlag = [True, True, True,      False, False, False,        True, True, True, True,
            True, True, True,      False, False, False]
        edge_color = None
    elif pose.shape[0] == 8:
        connect = [(0,1),                       (0,2)           ,      (0,3), (3,4), (4,5),
                (4,7),                           (4,6), ]
        LeftRightFlag = [True,                               False,                  True, True, True,
                True,                           False]
        edge_color = None
    elif pose.shape[0] == 2:
        connect = [(0,1)]
        LeftRightFlag = [True]
        edge_color = None
    elif pose.shape[0] == 25:
        connect = [(0,1), (1,20), (20,2), (2,3), 
                (20,4), (4,5), (5,6), (6,7), (7,21), (7,22), 
                (20,8), (8,9), (9,10), (10,11), (11,23), (11,24), 
                (0,16), (16,17), (17,18), (18,19), 
                (0,12), (12,13), (13,14), (14,15)]
        LeftRightFlag = [True, True, True, True,
            False, False, False, False, False, False,
            True, True, True, True, True, True,
            False, False, False, False,
            True, True, True, True]
        edge_color = None
    elif pose.shape[0] == 18:
        connect = [(2,0), (0,3), (3,6), 
                   (2,1), (1,4), (4,7), 
                   (2,5), (5,8), (8,11),
                   (8,10), (10,13), (13,15), (15,17),
                   (8,9), (9,12), (12,14), (14,16)]
        LeftRightFlag = [True, True, True, 
                         False, False, False, 
                         False, False, False, 
                         True, True, True, True, 
                         False, False, False, False]
        edge_color = None
    elif pose.shape[0] == 22:
        connect = [(0,1), (1,4), (4,7), (7,10), 
                   (0,2), (2,5), (5,8), (8,11), 
                   (0,3), (3,6), (6,9), (9,12), (12,15), 
                   (12,14), (14,17), (17,19), (19,21), 
                   (12,13), (13,16), (16,18), (18,20)]
        LeftRightFlag = [True, True, True, True,
                            False, False, False, False,
                            False, False, False, False, False,
                            True, True, True, True,
                            False, False, False, False]
        edge_color = None
    elif pose.shape[0] == 23:
        connect = [(0,1), (1,2), (2,3), (3,4), 
                   (0,5), (5,6), (6,7), (7,8), 
                   (0,9), (9,10), (10,11), (11,12), (12,13), (13,14), 
                   (12,15), (15,16), (16,17), (17,18), 
                   (12,19), (19,20), (20,21), (21,22)]
        LeftRightFlag = [True, True, True, True,
                            False, False, False, False,
                            False, False, False, False, False, False,
                            True, True, True, True,
                            False, False, False, False]
        edge_color = ['green', 'green', 'green', 'green', 
                      'green', 'green', 'green', 'green', 
                      'blue', 'red', 'yellow', 'orange', 'brown', 'brown', 
                      'brown', 'brown', 'brown', 'brown', 
                      'brown', 'brown', 'brown', 'brown']
        # edge_color = ['green', 'green', 'green', 'green', 
        #               'orange', 'orange', 'orange', 'orange', 
        #               'black', 'black', 'black', 'black', 'black', 'black', 
        #               'green', 'green', 'green', 'green', 
        #               'orange', 'orange', 'orange', 'orange']

    else:
        connect = []
        LeftRightFlag = []
        edge_color = None
        # raise ValueError('Unsupported joint number')
                  
        
    SpecialEdgeFlag = [False] * len(connect)
    ls = ['-'] * len(connect)
    lws = [lw] * len(connect)
    color = []
    for bone_idx in range(len(connect)):
        if edge_color is None:
            if not LeftRightFlag[bone_idx]:
                color.append(colors["lcolor"])
            else:
                color.append(colors["rcolor"])
        else:
            color = edge_color

    if special_edge is not None:
        connect = connect + special_edge
        SpecialEdgeFlag = SpecialEdgeFlag + [True] * len(special_edge)
        ls = ls + ['--'] * len(special_edge)
        lws = lws + [lw/2] * len(special_edge)
        color = color + ['red'] * len(special_edge)

    edge_st = [touple[0] for touple in connect]
    edge_ed = [touple[1] for touple in connect]

    
    for bone_idx in np.arange(len(edge_st)):
        x = np.array([pose[edge_st[bone_idx], 0], pose[edge_ed[bone_idx], 0]])
        y = np.array([pose[edge_st[bone_idx], 1], pose[edge_ed[bone_idx], 1]])
        if ax.name == '3d':
            z = np.array([pose[edge_st[bone_idx], 2], pose[edge_ed[bone_idx], 2]])
        if not update:
            if ax.name == '3d':
                plot.append(ax.plot(x, y, z, lw=lws[bone_idx], linestyle=ls[bone_idx], c=color[bone_idx], label=label if bone_idx==0 else ''))
            else:
                plot.append(ax.plot(x, y, lw=lws[bone_idx], linestyle=ls[bone_idx], c=color[bone_idx], label=label if bone_idx==0 else ''))
        else:
            plot[bone_idx][0].set_xdata(x)
            plot[bone_idx][0].set_ydata(y)
            if ax.name == '3d':
                plot[bone_idx][0].set_3d_properties(z)
            plot[bone_idx][0].set_color(color[bone_idx])

    
    if if_node == False and if_origin == False:
        return plot
    else:
        node_colors = np.array(['black' for _ in range(len(pose))])
        sizes = np.array([node_size for _ in range(len(pose))])
        if if_origin:
            node_colors = np.append(node_colors, 'black')
            sizes = np.append(sizes, node_size)

            pose = np.concatenate((pose, np.zeros((1,pose.shape[-1]))), axis=0)      # (17,3)||(1,3)->(18,3)

        if special_node is not None:
            node_colors[special_node] = 'red'
            sizes[special_node] = node_size * 5
    
        if ax.name == '3d':
            if not update:
                plot.append(ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=node_colors, s=sizes))
            else:
                plot[-1]._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        else:
            if not update:
                plot.append(ax.scatter(pose[:, 0], pose[:, 1], c=node_colors, s=sizes))
            else:
                plot[-1].set_offsets(pose[:, :2])
            
        return plot


def update(num, data, plots, fig, axes, fig_title, ttls, colors_list, if_node, if_origin, lim3d=1, lim2d=1, lw=2, if_target=False, node_size=5, special_node=None, special_edge=None):

    for idx, key in enumerate(plots):
        n_row, n_col = axes.shape
        ax = axes[idx//n_col, idx%n_col]

        if ax.name == '3d':
            ax.set_xlim3d([-1*lim3d, 1*lim3d])
            ax.set_ylim3d([-1*lim3d, 1*lim3d])
            ax.set_zlim3d([-1*lim3d, 1*lim3d])
        else:
            ax.set_xlim([-1*lim2d, 1*lim2d])
            ax.set_ylim([-1*lim2d, 1*lim2d])

        if not isinstance(plots[key], dict):
            colors = colors_list[idx]
            pose = data[key][num]
            plots[key] = create_pose(ax, plots[key], pose, colors, label=key, update=True, if_node=if_node, if_origin=if_origin, lw=lw, node_size=node_size, special_node=special_node, special_edge=special_edge)
        else:
            for v_idx, v_key in enumerate(plots[key]):
                colors = colors_list[v_idx]
                pose = data[key][v_key][num]
                plots[key][v_key] = create_pose(ax, plots[key][v_key], pose, colors, label=v_key, update=True, if_node=if_node, if_origin=if_origin, lw=lw, node_size=node_size, special_node=special_node, special_edge=special_edge)

            ax.legend(loc='upper right')

        if if_target:
            assert ax.name == '3d'
            targets = get_targets().data.cpu().numpy()
            ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='gray', s=5)
            # if not update:
            #     plots.append(ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], c='gray', s=5))
            # else:
            #     plots[-1]._offsets3d(targets[:, 0], targets[:, 1], targets[:, 2])

        # ax.set_title(fig_title+f" | frame {num+1}")
        ttls[idx].set_text(key+f" | frame {num+1}")

    return plots


def rotate_pose(data):
    tmp = np.zeros_like(data)
    if data.shape[-1] == 3:
        tmp[..., 0] = data[..., 2]
        tmp[..., 1] = data[..., 0]
        tmp[..., 2] = - data[..., 1]
    elif data.shape[-1] == 2:
        tmp[..., 0] = data[..., 0]
        tmp[..., 1] = - data[..., 1]
    return tmp


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
    