import chunk
import copy
from multiprocessing.util import info
import os
from click import prompt
from matplotlib.pyplot import subplot
import torch
import numpy as np
from time import time
from collections import OrderedDict
import inspect

from lib.utils.viz_skel_seq import viz_skel_seq_anim
from lib.utils.tools import read_pkl
from lib.utils.utils_non_AR import rotate_y, vector_angle, unify_skeletons
from third_party.motionbert.lib.utils.utils_mesh import compute_error
from third_party.motionbert.lib.utils.utils_smpl import SMPL


def compute_smpl_vertex(data_dict, SMPL_MODEL):
    B, T = data_dict['smpl_shape'].shape[:2]
    shape = data_dict['smpl_shape'].reshape(B*T, 10)
    pose = data_dict['smpl_pose'].reshape(B*T, 72)
    motion_smpl = SMPL_MODEL(
        betas=shape,
        body_pose=pose[:, 3:],
        global_orient=pose[:, :3],
        pose2rot=True
    )
    return motion_smpl.vertices.detach().reshape(B, T, -1, 3)


def train_epoch(args, model, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False):
    # torch.autograd.set_detect_anomaly(True)

    model.train()
    st = time()

    use_smpl = True if hasattr(args, 'Mesh') and args.Mesh.enable else False
    if use_smpl:
        SMPL_MODEL = SMPL('third_party/motionbert/data/mesh', batch_size=1)
        if torch.cuda.is_available():
            SMPL_MODEL = SMPL_MODEL.cuda()

    for idx, (QUERY_CHUNK_DICT, PROMPT_CHUNK_DICT, QUERY_TARGET_DICT, PROMPT_TARGET_DICT, INFO_DICT) in enumerate(train_loader['non_AR']):

        if args.shuffle_batch:
            raise NotImplementedError

        num_sample = len(INFO_DICT['query_index'])   # N
        defined_batch_size = args.batch_size    # B
        assert num_sample >= defined_batch_size
        
        total_batch = (num_sample + defined_batch_size - 1) // defined_batch_size
        for batch_id in range(total_batch):
            if (batch_id+1) * defined_batch_size > num_sample:
                slices = slice(batch_id * defined_batch_size, None)
            else:
                slices = slice(batch_id * defined_batch_size, (batch_id+1) * defined_batch_size)

            query_chunk_dict = OrderedDict({mode: chunk[slices] for mode, chunk in QUERY_CHUNK_DICT.items()})
            prompt_chunk_dict = OrderedDict({mode: chunk[slices] for mode, chunk in PROMPT_CHUNK_DICT.items()})
            query_target_dict = OrderedDict({mode: chunk[slices] for mode, chunk in QUERY_TARGET_DICT.items()})
            prompt_target_dict = OrderedDict({mode: chunk[slices] for mode, chunk in PROMPT_TARGET_DICT.items()})
            info_dict = {info_key: info[slices] for info_key, info in INFO_DICT.items()}
                
            batch_size = len(info_dict['query_index'])    # <= batch_size

            if torch.cuda.is_available():
                query_chunk_dict = OrderedDict({mode: chunk.cuda() for mode, chunk in query_chunk_dict.items()})
                prompt_chunk_dict = OrderedDict({mode: chunk.cuda() for mode, chunk in prompt_chunk_dict.items()})
                query_target_dict = OrderedDict({mode: chunk.cuda() for mode, chunk in query_target_dict.items()})
                prompt_target_dict = OrderedDict({mode: chunk.cuda() for mode, chunk in prompt_target_dict.items()})
            
            if use_smpl:
                query_target_vertex = compute_smpl_vertex(query_target_dict, SMPL_MODEL)
                prompt_target_vertex = compute_smpl_vertex(prompt_target_dict, SMPL_MODEL)
                query_target_vertex = query_target_vertex - query_target_dict['joint3d'][..., 0:1, :]
                prompt_target_vertex = prompt_target_vertex - prompt_target_dict['joint3d'][..., 0:1, :]

            query_chunk_dict = train_loader['non_AR'].dataset.preprocess(query_chunk_dict)
            prompt_chunk_dict = train_loader['non_AR'].dataset.preprocess(prompt_chunk_dict)
            query_target_dict = train_loader['non_AR'].dataset.preprocess(query_target_dict)
            prompt_target_dict = train_loader['non_AR'].dataset.preprocess(prompt_target_dict)


            query_chunk_recon_tensor, prompt_chunk_recon_tensor \
                = model(query_chunk_dict, prompt_chunk_dict, 
                        info_dict, epoch, vertex_x1000=args.vertex_x1000)
            # output_joint: [B, T, 17, 3]
            # output_smpl: 1 element list.
            #   output_smpl[0]:
            #       'theta': [B, T, 82]
            #       'verts': [B, T, 6890, 3]
            #       'kp_3d': [B, T, 17, 3]


            # Optimize
            optimizer.zero_grad()
            loss_total = 0

            loss_total = torch.nn.functional.mse_loss(query_chunk_recon_tensor, model.module.convert_dict_to_tensor(query_target_dict)) *0.5 \
                       + torch.nn.functional.mse_loss(prompt_chunk_recon_tensor, model.module.convert_dict_to_tensor(prompt_target_dict)) *0.5 
            
            # Joint loss
            while False:
                loss_joint_total = 0
                for loss_name, loss_dict in losses['joint'].items():
                    loss = loss_dict['loss_function'](output_joint, target_joint)
                    weight = loss_dict['loss_weight']
                    loss_dict['loss_logger'].update(loss.item(), batch_size)
                    if weight != 0:
                        loss_joint_total += loss * weight
                losses['joint_total'].update(loss_joint_total.item(), batch_size)

                loss_total = loss_joint_total

                if use_smpl:
                    # Mesh loss
                    loss_mesh_total = 0
                    mesh_criteria = losses['mesh_criterion']
                    losses_mesh = mesh_criteria(output_smpl, target_smpl)
                    for loss_name, loss_dict in losses['mesh'].items():
                        loss = losses_mesh[loss_name]
                        weight = loss_dict['loss_weight']
                        loss_dict['loss_logger'].update(loss.item(), batch_size)
                        if weight != 0:
                            loss_mesh_total += loss * weight
                    losses['mesh_total'].update(loss_mesh_total.item(), batch_size)
                    
                    loss_total += loss_mesh_total
                
                    mpjpe, mpve = compute_error(output_smpl, target_smpl)  # both zero-dim, one-element tensor

                losses['all_total'].update(loss_total.item(), batch_size)
            losses['joint_total'].update(loss_total.item(), batch_size)
            losses['mesh_total'].update(loss_total.item(), batch_size)
            losses['all_total'].update(loss_total.item(), batch_size)
            

            loss_total.backward()
            optimizer.step()


            if args.get('reverse_query_prompt_per_iter', False):
                raise NotImplementedError

        dataset_cnt = {dataset_name: 0 for dataset_name in args.dataset_task_info['train']}
        for sample_id in range(num_sample):
            dataset_name = INFO_DICT['dataset'][sample_id]
            dataset_cnt[dataset_name] += 1

        if idx in [len(train_loader['non_AR']) * i // 3 for i in range(1, 3)]:
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples; batch_size (total/sub): {num_sample}/{batch_size}")
        

        if if_debug:
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples; batch_size (total/sub): {num_sample}/{batch_size}")
            if idx > 1: break

    if not args.reverse_query_prompt:
        return
    else:
        raise NotImplementedError
    

def train_classifier_epoch(args, model, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False, classifier_type='task'):
    raise NotImplementedError
