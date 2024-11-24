from multiprocessing.util import info
import os
from click import prompt
from matplotlib.pyplot import subplot
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


def compute_smpl_vertex(args, data_dict, SMPL_MODEL, info_dict):
    B, T = data_dict['smpl_shape'].shape[:2]
    global_orient_mask = torch.tensor(info_dict['use_global_orient'])
    assert len(global_orient_mask) == B
    global_orient_mask = global_orient_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, T, 3).reshape(B*T, 3)
    shape = data_dict['smpl_shape'].reshape(B*T, 10)
    pose = data_dict['smpl_pose'].reshape(B*T, 72)
    motion_smpl = SMPL_MODEL(
        betas=shape,
        body_pose=pose[:, 3:],
        global_orient=pose[:, :3] * global_orient_mask.to(pose.device),
        pose2rot=True
    )
    return motion_smpl.vertices.detach().reshape(B, T, -1, 3)


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

    use_smpl = True if hasattr(args, 'Mesh') and args.Mesh.enable else False
    if use_smpl:
        SMPL_MODEL = SMPL('third_party/motionbert/data/mesh', batch_size=1)
        if torch.cuda.is_available():
            if rank is None:
                SMPL_MODEL = SMPL_MODEL.cuda()
            else:
                SMPL_MODEL = SMPL_MODEL.cuda(rank)

    NUM_SAMPLE = 0
    INFO_DICT_EPOCH = defaultdict(list)

    for idx, (QUERY_INPUT_TENSOR, PROMPT_INPUT_TENSOR, QUERY_TARGET_TENSOR, PROMPT_TARGET_TENSOR, QUERY_TARGET_DICT, PROMPT_TARGET_DICT, INFO_DICT) in enumerate(train_loader['non_AR']):

        if args.shuffle_batch:
            raise NotImplementedError
        
        for info_key in INFO_DICT.keys():
            INFO_DICT_EPOCH[info_key].extend(INFO_DICT[info_key])


        num_sample = len(INFO_DICT['query_index'])   # N
        NUM_SAMPLE += num_sample
        defined_batch_size = args.batch_size    # B
        
        total_batch = (num_sample + defined_batch_size - 1) // defined_batch_size
        assert total_batch > 0
        if idx == 0 and rank ==0: print(f'\tTotal sub-batch: {total_batch}')
        for batch_id in range(total_batch):
            if (batch_id+1) * defined_batch_size > num_sample:
                slices = slice(batch_id * defined_batch_size, None)
            else:
                slices = slice(batch_id * defined_batch_size, (batch_id+1)*defined_batch_size)

            query_input_tensor = QUERY_INPUT_TENSOR[slices]
            prompt_input_tensor = PROMPT_INPUT_TENSOR[slices]
            query_target_tensor = QUERY_TARGET_TENSOR[slices] if QUERY_TARGET_TENSOR is not None else None
            prompt_target_tensor = PROMPT_TARGET_TENSOR[slices]

            query_target_dict = OrderedDict()
            prompt_target_dict = OrderedDict() if PROMPT_TARGET_DICT is not None else None
            for mode in QUERY_TARGET_DICT.keys():
                query_target_dict[mode] = QUERY_TARGET_DICT[mode][slices]
            if PROMPT_TARGET_DICT is not None:
                for mode in PROMPT_TARGET_DICT.keys():
                    prompt_target_dict[mode] = PROMPT_TARGET_DICT[mode][slices]
                
            info_dict = {}
            for info_key in INFO_DICT.keys():
                info_dict[info_key] = INFO_DICT[info_key][slices]
            input_mask = torch.cat(info_dict['input_mask']) if 'input_mask' in info_dict else None     # [B, 24]
            input_temporal_mask = torch.cat(info_dict['input_temporal_mask']) if 'input_temporal_mask' in info_dict else None     # [B, 24]

            batch_size = len(info_dict['query_index'])    # <= batch_size

            if torch.cuda.is_available():
                if rank is None:
                    query_input_tensor = query_input_tensor.cuda()
                    prompt_input_tensor = prompt_input_tensor.cuda()
                    if query_target_tensor is not None:
                        query_target_tensor = query_target_tensor.cuda()
                    prompt_target_tensor = prompt_target_tensor.cuda()
                    query_target_dict = {k: v.cuda() for k, v in query_target_dict.items()}
                    if prompt_target_dict is not None:
                        prompt_target_dict = {k: v.cuda() for k, v in prompt_target_dict.items()}
                    if input_mask is not None:
                        input_mask = input_mask.cuda()
                        input_temporal_mask = input_temporal_mask.cuda()
                else:
                    if rank == 0 and idx == 0 and batch_id == 0: print(f'\tDDP is being applied.')
                    query_input_tensor = query_input_tensor.cuda(rank)
                    prompt_input_tensor = prompt_input_tensor.cuda(rank)
                    if query_target_tensor is not None:
                        query_target_tensor = query_target_tensor.cuda(rank)
                    prompt_target_tensor = prompt_target_tensor.cuda(rank)
                    query_target_dict = {k: v.cuda(rank) for k, v in query_target_dict.items()}
                    if prompt_target_dict is not None:
                        prompt_target_dict = {k: v.cuda(rank) for k, v in prompt_target_dict.items()}
                    if input_mask is not None:
                        input_mask = input_mask.cuda(rank)
                        input_temporal_mask = input_temporal_mask.cuda()
            
            if use_smpl:
                query_target_vertex = compute_smpl_vertex(args, query_target_dict, SMPL_MODEL, info_dict)
                query_target_dict['smpl_vertex'] = query_target_vertex
                if prompt_target_dict is not None:
                    prompt_target_vertex = compute_smpl_vertex(args, prompt_target_dict, SMPL_MODEL, info_dict)
                    prompt_target_dict['smpl_vertex'] = prompt_target_vertex

            query_target_dict = train_loader['non_AR'].dataset.preprocess(query_target_dict)
            if prompt_target_dict is not None:
                prompt_target_dict = train_loader['non_AR'].dataset.preprocess(prompt_target_dict)
            # preprocessing input already done in perpare_motion

            output_dict = model(
                                query_input_tensor, prompt_input_tensor, {'spatial': input_mask, 'temporal': input_temporal_mask},
                                query_target_tensor, prompt_target_tensor,
                                query_target_dict, prompt_target_dict, 
                                info_dict, epoch, vertex_x1000=args.vertex_x1000, deactivate_prompt_branch=args.deactivate_prompt_branch,
                                return_context=args.get('use_context', None)
                                )
            # output_joint: [B, T, 17, 3]
            # output_smpl: 1 element list.
            #   output_smpl[0]:
            #       'theta': [B, T, 82]
            #       'verts': [B, T, 6890, 3]
            #       'kp_3d': [B, T, 17, 3]


            # Optimize
            optimizer.zero_grad()
            loss_total = 0


            # Joint loss
            loss_joint_total = 0
            for k in output_dict.keys():
                output_joint, output_smpl, target_joint, target_smpl = output_dict[k][:4]
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



            if args.get('use_context', None) == 'post_attach_context_head':
                # context_output: [b, num_class]
                context_output = output_dict['query'][-1]
                context_label = torch.tensor([args.task_to_flag[task] for task in info_dict['task']]).to(context_output.device)
                context_loss = torch.nn.functional.cross_entropy(context_output, context_label)
                loss_total += context_loss



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
            if rank == 0: print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples; batch_size (total/sub): {num_sample}/{batch_size}")
        

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
