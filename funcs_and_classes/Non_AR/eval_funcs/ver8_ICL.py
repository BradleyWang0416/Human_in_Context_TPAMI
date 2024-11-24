#%%
from math import e
import os
from unittest import result
import numpy as np
import prettytable
import torch
import time
from sklearn.metrics import confusion_matrix
import inspect
from collections import defaultdict

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from third_party.motionbert.lib.utils.utils_smpl import SMPL
from third_party.motionbert.lib.utils.utils_mesh import compute_error
from third_party.motionbert.lib.utils.utils_mesh import evaluate_mesh

from lib.utils.viz_skel_seq import viz_skel_seq_anim

from funcs_and_classes.Non_AR.train_epoch.ver8_ICL import get_target_smpl


def evaluate(args, TEST_LOADER, h36m_datareader, model, dataset, eval_task, epoch=None, if_viz=False, if_debug=False):
    print(f'\tEvaluating [{eval_task}] on [{dataset}]...', end=' ')
    model.eval()

    st = time.time()
    num_samples = 0

    if eval_task in ['PE', 'FPE', 'MP', 'MC', 'MIB']:
        TASK_MODE = 'joint'
    elif eval_task in ['MeshRecover', 'FutureMeshRecover', 'MeshPred', 'MeshCompletion', 'MeshInBetween']:
        TASK_MODE = 'mesh'


    if eval_task == 'PE' and dataset == 'H36M_3D':
        results_all = []
    elif eval_task in ['MIB', 'MeshInBetween']:
        mpjpes_joint = np.zeros((int(args.frame_mask_ratio*args.clip_len), args.num_joint))
    elif eval_task == 'MC':
        mpjpes_joint = np.zeros((args.clip_len, int(args.joint_mask_ratio*args.num_joint)))
    # elif eval_task == 'MeshCompletion':
    #     mpjpes_joint = np.zeros((args.clip_len, int(args.mesh_joint_mask_ratio*24)))
    else:
        mpjpes_joint = np.zeros((args.clip_len, args.num_joint))


    SMPL_MODEL = SMPL('third_party/motionbert/data/mesh', batch_size=1)
    if torch.cuda.is_available():
        SMPL_MODEL = SMPL_MODEL.cuda()
    SMPL_MODEL.eval()

    mpjpes_mesh = AverageMeter()
    mpves_mesh = AverageMeter()
    results_mesh = defaultdict(list)

    with torch.no_grad():
        test_loader = TEST_LOADER[(dataset, eval_task)]
        
        for idx, (query_sample_dict, prompt_sample_dict, info_dict, task_mode) in enumerate(test_loader):
            batch_size = len(info_dict['query_index'])
            num_samples += batch_size
            assert info_dict['task'] == [eval_task]*batch_size
            assert task_mode == TASK_MODE

            if torch.cuda.is_available():
                query_sample_dict = {k: v.cuda() for k, v in query_sample_dict.items()}
                prompt_sample_dict = {k: v.cuda() for k, v in prompt_sample_dict.items()}
            
            output_joint, output_smpl = model(query_sample_dict, prompt_sample_dict, 
                                              epoch, vertex_x1000=args.vertex_x1000, deactivate_prompt_branch=args.deactivate_prompt_branch)
            # output_joint: [B, T, 17, 3]
            # output_smpl: 1 element list.
            #   output_smpl[0]:
            #       'theta': [B, T, 82]
            #       'verts': [B, T, 6890, 3]
            #       'kp_3d': [B, T, 17, 3]
            #             
            if dataset == 'H36M_3D':
                if eval_task == 'PE' and args.dataset_config['H36M_3D']['rootrel_target']:
                    output_joint[..., 0, :] = 0
                output_joint = output_joint * args.dataset_config['H36M_3D']['scale_3D']
            else:
                if args.dataset_config[dataset]['rootrel_target']:
                    output_joint = output_joint - output_joint[..., 0:1, :]                    

            # if task_mode == 'joint':
            target_joint = query_sample_dict['joint3d'].clone()
            # elif task_mode == 'mesh':
            target_smpl = get_target_smpl(args, query_sample_dict, SMPL_MODEL, info_dict)

            # Evaluate mesh
            mpjpe_mesh, mpve_mesh = compute_error(output_smpl, target_smpl)
            mpjpes_mesh.update(mpjpe_mesh, batch_size)
            mpves_mesh.update(mpve_mesh, batch_size)
            for keys in output_smpl[0].keys():
                output_smpl[0][keys] = output_smpl[0][keys].detach().cpu().numpy()
                target_smpl[keys] = target_smpl[keys].detach().cpu().numpy()
            results_mesh['kp_3d'].append(output_smpl[0]['kp_3d'])
            results_mesh['verts'].append(output_smpl[0]['verts'])
            results_mesh['kp_3d_gt'].append(target_smpl['kp_3d'])
            results_mesh['verts_gt'].append(target_smpl['verts'])
            

            # Evaluate joint
            if eval_task == 'PE' and dataset == 'H36M_3D':
                results_all.append(output_joint.cpu().numpy())
                continue
            elif eval_task in ['MIB', 'MeshInBetween']:
                frame_mask = torch.tensor(info_dict['frame_mask']).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, output_joint.shape[-2], output_joint.shape[-1]).to(output_joint.device)
                output_joint_to_eval = torch.gather(output_joint, 1, frame_mask)
                target_joint_to_eval = torch.gather(target_joint, 1, frame_mask)
            elif eval_task == 'MC':
                joint_mask = torch.tensor(info_dict['joint_mask']).unsqueeze(1).unsqueeze(-1).expand(-1, output_joint.shape[1], -1, output_joint.shape[-1]).to(output_joint.device)
                output_joint_to_eval = torch.gather(output_joint, 2, joint_mask)
                target_joint_to_eval = torch.gather(target_joint, 2, joint_mask)
            # elif eval_task == 'MeshCompletion':
            #     mesh_joint_mask = torch.tensor(info_dict['mesh_joint_mask']).unsqueeze(1).unsqueeze(-1).expand(-1, output_joint.shape[1], -1, output_joint.shape[-1]).to(output_joint.device)
            #     output_joint_to_eval = torch.gather(output_joint, 2, mesh_joint_mask)
            #     target_joint_to_eval = torch.gather(target_joint, 2, mesh_joint_mask)
            else:
                output_joint_to_eval = output_joint.clone()
                target_joint_to_eval = target_joint.clone()

            mpjpe_joint = torch.norm(output_joint_to_eval*1000 - target_joint_to_eval*1000, dim=-1).sum(0)   # [B,T,J,3]-norm->[B,T,J]-sum->[T,J]. <T for MIB. <J for MC
            mpjpes_joint += mpjpe_joint.cpu().numpy()

            if if_debug:
                if idx >1:
                    break

    if eval_task == 'PE' and dataset == 'H36M_3D':
        mpjpes_joint_avg, mpjpes_joint, p_mpjpes_joint_avg, p_mpjpes_joint, action_names = evaluate_pose_estimation(args, h36m_datareader, results_all)
        header_full = action_names
        results_detail = mpjpes_joint
        header_detail = action_names
    else:
        mpjpes_joint = mpjpes_joint / num_samples       # (T,J). <T for MIB. <J for MC
        if eval_task == 'MC':
            mpjpes_joint = mpjpes_joint.mean()[None]          # (1,)
            results_detail = list(np.round(mpjpes_joint,2))
            header_full = [args.joint_mask_ratio]
            header_detail = [args.joint_mask_ratio]
        elif eval_task in ['MIB', 'MeshInBetween']:
            mpjpes_joint = mpjpes_joint.mean()[None]          # (1,)
            results_detail = list(np.round(mpjpes_joint,2))
            header_full = [args.frame_mask_ratio]
            header_detail = [args.frame_mask_ratio]
        else:
            mpjpes_joint = mpjpes_joint.mean(-1)         # (T,)
            frame_to_show = np.linspace(0, args.clip_len-1, 4).astype(int)
            results_detail = list(np.round(mpjpes_joint[frame_to_show],2))
            header_full = list(np.arange(len(mpjpes_joint)))
            header_detail = list(frame_to_show)

        mpjpes_joint_avg = np.mean(mpjpes_joint)        # scalar


    for term in results_mesh.keys():
        results_mesh[term] = np.concatenate(results_mesh[term])
    error_dict = evaluate_mesh(results_mesh)

    mpve_mesh_avg = error_dict['mpve'].astype(float)
    results_mesh_detail = [v.astype(float) for v in error_dict.values()]
    if not args.vertex_x1000:
        mpve_mesh_avg = mpve_mesh_avg * 1000
        results_mesh_detail = [v * 1000 for v in results_mesh_detail]
    header_mesh_detail = list(error_dict.keys())
    
    # evaluation table (will be printed ONLY when <args.evaluate> is activated, will NOT be printed during training)
    summary_table = prettytable.PrettyTable()
    field_names = [dataset] + [eval_task] + ['Avg'] + header_full
    result_row = [''] + ['MPJPE'] + list(mpjpes_joint_avg[None]) + list(mpjpes_joint)

    field_names = field_names + list(error_dict.keys())
    result_row = result_row + list(error_dict.values())
    summary_table.field_names = field_names
    summary_table.add_row(result_row)
    summary_table.float_format = ".2"

    print(f"costs {time.time()-st:.2f}s")

    return mpjpes_joint_avg, results_detail, header_detail, mpve_mesh_avg, results_mesh_detail, header_mesh_detail, summary_table


def evaluate_pose_estimation(args, datareader, results_all):

    _, split_id_test = datareader.get_split_id()
    for i, split_range in enumerate(split_id_test):
        if args.current_as_history:
            min_id = min(split_range)
            split_id_test[i] = range(min_id, min_id + args.clip_len)
        else:
            max_id = max(split_range)
            split_id_test[i] = range(max_id+1 - args.clip_len, max_id+1)
    datareader.split_id_test = split_id_test
    

    results_all = np.concatenate(results_all)       # (N,16,17,3)
    results_all = datareader.denormalize(results_all)

    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]               # (N,16,17,3)

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:,None,None]     # (96,1,1)
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:,0:1,:]
        gt = gt - gt[:,0:1,:]

        err1 = mpjpe(pred, gt)
        err2 = p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)

    final_result = []
    final_result_procrustes = []
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    
    return e1, final_result, e2, final_result_procrustes, action_names