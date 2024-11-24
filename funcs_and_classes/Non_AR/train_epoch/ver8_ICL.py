from multiprocessing.util import info
import os
from click import prompt
from matplotlib.pyplot import subplot
from pydantic import NoneIsAllowedError
import torch
import numpy as np
from time import time
from collections import OrderedDict, defaultdict
import inspect
import pickle
import copy

from lib.utils.viz_skel_seq import viz_skel_seq_anim
from lib.utils.tools import read_pkl
from lib.utils.utils_non_AR import rotate_y, vector_angle, unify_skeletons
from third_party.motionbert.lib.utils.utils_mesh import compute_error
from third_party.motionbert.lib.utils.utils_smpl import SMPL


def get_target_smpl(args, data_dict, SMPL_MODEL, info_dict):
    target_smpl = {}
    B, T = data_dict['smpl_shape'].shape[:2]
    global_orient_mask = torch.tensor(info_dict['use_global_orient'])
    assert len(global_orient_mask) == B
    global_orient_mask = global_orient_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, T, 3).reshape(B*T, 3)
    shape = data_dict['smpl_shape'].reshape(B*T, 10)
    pose = data_dict['smpl_pose'].reshape(B*T, 72)
    with torch.no_grad():
        motion_smpl = SMPL_MODEL(
            betas=shape,
            body_pose=pose[:, 3:],
            global_orient=pose[:, :3] * global_orient_mask.to(pose.device),
            pose2rot=True
        )
    smpl_vertex = motion_smpl.vertices.detach().reshape(B, T, -1, 3)
    if args.vertex_x1000: 
        smpl_vertex = smpl_vertex * 1000.0
    J_regressor = SMPL_MODEL.J_regressor_h36m   # [17,6890]
    smpl_kp3d = torch.einsum('jv,btvc->btjc', J_regressor.to(smpl_vertex.device), smpl_vertex)
    smpl_vertex = smpl_vertex - smpl_kp3d[:, :, 0:1, :]
    smpl_kp3d = smpl_kp3d - smpl_kp3d[:, :, 0:1, :]
    target_smpl['verts'] = smpl_vertex
    target_smpl['kp_3d'] = smpl_kp3d
    target_smpl['theta'] = torch.cat([pose, shape], dim=-1).reshape(B, T, -1)
    return target_smpl

# 将字典中的所有张量移动到指定设备上的函数
def move_dict_to_device(d, device):
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = value.to(device)
    return d

def train_epoch(args, model, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False, rank=None):
    # torch.autograd.set_detect_anomaly(True)

    model.train()
    st = time()

    SMPL_MODEL = SMPL('third_party/motionbert/data/mesh', batch_size=1)
    if torch.cuda.is_available():
        SMPL_MODEL = SMPL_MODEL.cuda()
    SMPL_MODEL.eval()
    if rank is not None:
        SMPL_MODEL = SMPL_MODEL.to(rank)

    NUM_SAMPLE = 0
    INFO_DICT_EPOCH = defaultdict(list)

    for idx, (QUERY_SAMPLE_DICT_JOINT, PROMPT_SAMPLE_DICT_JOINT, INFO_DICT_JOINT, QUERY_SAMPLE_DICT_MESH, PROMPT_SAMPLE_DICT_MESH, INFO_DICT_MESH) in enumerate(train_loader['non_AR']):

        for (task_mode, QUERY_SAMPLE_DICT, PROMPT_SAMPLE_DICT, INFO_DICT) in [
                                                                             ('mesh', QUERY_SAMPLE_DICT_MESH, PROMPT_SAMPLE_DICT_MESH, INFO_DICT_MESH),
                                                                             ('joint', QUERY_SAMPLE_DICT_JOINT, PROMPT_SAMPLE_DICT_JOINT, INFO_DICT_JOINT)
                                                                             ]:
            for info_key in INFO_DICT.keys():
                INFO_DICT_EPOCH[info_key].extend(INFO_DICT[info_key])

            num_sample = len(INFO_DICT['query_index'])   # N
            NUM_SAMPLE += num_sample
            defined_batch_size = args.batch_size    # B
            
            total_batch = (num_sample + defined_batch_size - 1) // defined_batch_size
            assert total_batch > 0
            for batch_id in range(total_batch):
                if (batch_id+1) * defined_batch_size > num_sample:
                    slices = slice(batch_id * defined_batch_size, None)
                else:
                    slices = slice(batch_id * defined_batch_size, (batch_id+1)*defined_batch_size)

                query_sample_dict = {}
                prompt_sample_dict = {}
                info_dict = {}
                for mode in QUERY_SAMPLE_DICT.keys():
                    query_sample_dict[mode] = QUERY_SAMPLE_DICT[mode][slices]
                for mode in PROMPT_SAMPLE_DICT.keys():
                    prompt_sample_dict[mode] = PROMPT_SAMPLE_DICT[mode][slices]
                for info_key in INFO_DICT.keys():
                    info_dict[info_key] = INFO_DICT[info_key][slices]
                    
                batch_size = len(info_dict['query_index'])    # <= defined_batch_size

                if torch.cuda.is_available():
                    query_sample_dict = {k: v.cuda() for k, v in query_sample_dict.items()}
                    prompt_sample_dict = {k: v.cuda() for k, v in prompt_sample_dict.items()}
                
                if rank is not None:
                    if rank == 0 and idx == 0: print(f'\tDDP is being applied.')
                    device = torch.device(f'cuda:{rank}')
                    query_sample_dict = move_dict_to_device(query_sample_dict, device)
                    prompt_sample_dict = move_dict_to_device(prompt_sample_dict, device)
                
                if args.train_simultaneously:
                    output_joint, output_smpl = model(query_sample_dict, prompt_sample_dict,
                                                    epoch, vertex_x1000=args.vertex_x1000, deactivate_prompt_branch=args.deactivate_prompt_branch)
                else:
                    if task_mode == 'joint':
                        output_joint, _ = model(query_sample_dict, prompt_sample_dict, 
                                                epoch, vertex_x1000=args.vertex_x1000, deactivate_prompt_branch=args.deactivate_prompt_branch)
                    elif task_mode == 'mesh':
                        _, output_smpl = model(query_sample_dict, prompt_sample_dict, 
                                            epoch, vertex_x1000=args.vertex_x1000, deactivate_prompt_branch=args.deactivate_prompt_branch)
                # output_joint: [B, T, 17, 3]
                # output_smpl: 1 element list.
                #   output_smpl[0]:
                #       'theta': [B, T, 82]
                #       'verts': [B, T, 6890, 3]
                #       'kp_3d': [B, T, 17, 3]
                    
                if args.train_simultaneously or task_mode == 'joint':
                    target_joint = query_sample_dict['joint3d'].clone()
                if args.train_simultaneously or task_mode == 'mesh':
                    target_smpl = get_target_smpl(args, query_sample_dict, SMPL_MODEL, info_dict)

                # Optimize
                optimizer.zero_grad()
                loss_total = 0

                # Joint loss
                if args.train_simultaneously or task_mode == 'joint':
                    for loss_name, loss_dict in losses['joint'].items():
                        loss = loss_dict['loss_function'](output_joint, target_joint)
                        weight = loss_dict['loss_weight']
                        loss_dict['loss_logger'].update(loss.item(), batch_size)
                        if weight != 0:
                            loss_total += loss * weight
                    losses['joint_total'].update(loss_total.item(), batch_size)

                # Mesh loss
                if args.train_simultaneously or task_mode == 'mesh':
                    mpjpe, mpve = compute_error(output_smpl, target_smpl)  # both zero-dim, one-element tensor
                    mesh_criteria = losses['mesh_criterion']
                    losses_mesh = mesh_criteria(output_smpl, target_smpl)
                    for loss_name, loss_dict in losses['mesh'].items():
                        loss = losses_mesh[loss_name]
                        weight = loss_dict['loss_weight']
                        loss_dict['loss_logger'].update(loss.item(), batch_size)
                        if weight != 0:
                            loss_total += loss * weight
                    losses['mesh_total'].update(loss_total.item(), batch_size)
                
                
                loss_total.backward()
                optimizer.step()



        dataset_cnt = {dataset_name: 0 for dataset_name in args.dataset_task_info['train']}
        for sample_id in range(num_sample):
            dataset_name = INFO_DICT['dataset'][sample_id]
            dataset_cnt[dataset_name] += 1

        if idx in [len(train_loader['non_AR']) * i // 3 for i in range(1, 3)]:
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples; batch_size (total/sub): {num_sample}/{batch_size}")
        

        if if_debug:
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples; batch_size (total/sub): {num_sample}/{batch_size}")
            if idx > 1: break

    if not if_debug:
        try:
            INFO_DICT_ALL = {}
            # for dataset_name, task, query_index, \
            #     prompt_chunk_id, query_chunk_id, joint_mask, frame_mask in zip(INFO_DICT_EPOCH['dataset'], INFO_DICT_EPOCH['task'], INFO_DICT_EPOCH['query_index'], 
            #                                                                     INFO_DICT_EPOCH['prompt_chunk_id'], INFO_DICT_EPOCH['query_chunk_id'], INFO_DICT_EPOCH['joint_mask'], INFO_DICT_EPOCH['frame_mask'], ):
            #     INFO_DICT_ALL[(dataset_name, task, query_index)] = {'prompt_chunk_id': prompt_chunk_id, 'query_chunk_id': query_chunk_id, 'joint_mask': joint_mask, 'frame_mask': frame_mask}
            for i, (dataset_name, task, query_index) in enumerate(zip(INFO_DICT_EPOCH['dataset'], INFO_DICT_EPOCH['task'], INFO_DICT_EPOCH['query_index'])):
                INFO_DICT_ALL[(dataset_name, task, query_index)] = {key: value[i] for key, value in INFO_DICT_EPOCH.items() if key not in ['dataset', 'task', 'query_index']}
            assert len(INFO_DICT_ALL) == NUM_SAMPLE
            
            if not os.path.exists(os.path.join(args.checkpoint, 'epoch_info_dict')):
                ROOT_DIR_HARDDISK = os.path.dirname(os.path.abspath(__file__)).split('funcs_and_classes')[0].replace('wxs', 'wxs/wxs')
                os.makedirs(os.path.join(ROOT_DIR_HARDDISK, args.checkpoint, 'epoch_info_dict'), exist_ok=True)
                os.symlink(os.path.join(ROOT_DIR_HARDDISK, args.checkpoint, 'epoch_info_dict'), os.path.join(args.checkpoint, 'epoch_info_dict'))
            
            save_path = os.path.join(args.checkpoint, 'epoch_info_dict', f'info_dict_ep{epoch:03d}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(INFO_DICT_ALL, f)
            latest_path = os.path.join(args.checkpoint, 'epoch_info_dict', 'info_dict_latest.pkl')
            if os.path.exists(latest_path):
                os.remove(latest_path)
            with open(latest_path, 'wb') as f:
                pickle.dump(INFO_DICT_ALL, f)
        except:
            print('\n\tEpoch info dict failed to be created and saved.\n')

    if not args.reverse_query_prompt:
        return
    else:
        raise NotImplementedError
    

def train_classifier_epoch(args, model, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False, classifier_type='task'):
    raise NotImplementedError
