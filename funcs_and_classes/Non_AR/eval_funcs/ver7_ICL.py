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

from funcs_and_classes.Non_AR.train_epoch.ver5_ICL import compute_smpl_vertex


def evaluate(args, TEST_LOADER, h36m_datareader, model, dataset, eval_task, epoch=None, if_viz=False, if_debug=False):
    print(f'\tEvaluating [{eval_task}] on [{dataset}]...', end=' ')
    model.eval()

    st = time.time()
    num_samples = 0


    if eval_task == 'MIB':
        mpjpes_joint = np.zeros((int(args.frame_mask_ratio*args.clip_len), args.num_joint))
    elif eval_task == 'MC':
        mpjpes_joint = np.zeros((args.clip_len, int(args.joint_mask_ratio*args.num_joint)))
    elif eval_task == 'PE' and dataset == 'H36M_3D':
        results_all = []
    else:
        mpjpes_joint = np.zeros((args.clip_len, args.num_joint))


    use_smpl = True if hasattr(args, 'Mesh') and args.Mesh.enable else False
    if use_smpl:
        SMPL_MODEL = SMPL('third_party/motionbert/data/mesh', batch_size=1)
        if torch.cuda.is_available():
            SMPL_MODEL = SMPL_MODEL.cuda()

        mpjpes_mesh = AverageMeter()
        mpves_mesh = AverageMeter()
        results_mesh = defaultdict(list)

    with torch.no_grad():
        test_loader = TEST_LOADER[(dataset, eval_task)]
        
        for idx, (query_input_tensor, prompt_input_tensor, query_target_tensor, prompt_target_tensor, query_target_dict, prompt_target_dict, info_dict) in enumerate(test_loader):
            batch_size = len(info_dict['query_index'])
            num_samples += batch_size
            assert info_dict['task'] == [eval_task]*batch_size

            input_mask = torch.cat(info_dict['input_mask']) if 'input_mask' in info_dict else None     # [B, 24] 

            if torch.cuda.is_available():
                query_input_tensor = query_input_tensor.cuda()
                prompt_input_tensor = prompt_input_tensor.cuda()
                if query_target_tensor is not None: query_target_tensor = query_target_tensor.cuda()
                prompt_target_tensor = prompt_target_tensor.cuda()
                query_target_dict = {k: v.cuda() for k, v in query_target_dict.items()}
                if prompt_target_dict is not None: prompt_target_dict = {k: v.cuda() for k, v in prompt_target_dict.items()}

                if input_mask is not None: input_mask = input_mask.cuda()
            
            if use_smpl:
                query_target_vertex = compute_smpl_vertex(args, query_target_dict, SMPL_MODEL, info_dict)
                query_target_dict['smpl_vertex'] = query_target_vertex
                if prompt_target_dict is not None: 
                    prompt_target_vertex = compute_smpl_vertex(args, prompt_target_dict, SMPL_MODEL, info_dict)
                    prompt_target_dict['smpl_vertex'] = prompt_target_vertex
            
            query_target_dict = test_loader.dataset.preprocess(query_target_dict)
            if prompt_target_dict is not None: 
                prompt_target_dict = test_loader.dataset.preprocess(prompt_target_dict)

            output_joint, output_smpl, target_joint, target_smpl = \
                model(query_input_tensor, prompt_input_tensor, input_mask,
                      query_target_tensor, prompt_target_tensor,
                      query_target_dict, prompt_target_dict, info_dict, epoch, vertex_x1000=args.vertex_x1000, deactivate_prompt_branch=args.deactivate_prompt_branch)
            # output_joint: [B, T, 17, 3]
            # output_smpl: 1 element list.
            #   output_smpl[0]:
            #       'theta': [B, T, 82]
            #       'verts': [B, T, 6890, 3]
            #       'kp_3d': [B, T, 17, 3]

            output_joint = test_loader.dataset.postprocess(output_joint, dataset_name=dataset, task=eval_task)


            if if_viz:
                if if_viz.split(',')[0] != inspect.currentframe().f_code.co_name:
                    return 1,2,3,4,5,6,7
                if len(if_viz.split(',')) == 2 and if_viz.split(',')[1] != dataset and if_viz.split(',')[1] != eval_task:
                    return 1,2,3,4,5,6,7
                if len(if_viz.split(',')) == 3 and if_viz.split(',')[1] != dataset and if_viz.split(',')[2] != eval_task:
                    return 1,2,3,4,5,6,7
                for i in range(0, batch_size, 130):
                    query_index = info_dict['query_index'][i]
                    # if (dataset, eval_task, query_index) not in [
                    #                                             ('AMASS','MP',5120),
                    #                                             ('AMASS','MP',6274),
                    #                                             ('AMASS','MIB',1284),
                    #                                             ('AMASS','MIB',5120),
                    #                                             ('AMASS','MIB',11264),
                    #                                             ('AMASS','MC',1284),
                    #                                             ('AMASS','MC',5120),
                    #                                             ('AMASS','MC',11264),
                    #                                             ]: continue
                    if eval_task in ['PE', 'FPE']: 
                        assert (query_input_dict['joint'][..., -1] == 0).all()
                        query_input_dict['joint'][i] = query_input_dict['joint'][i, ..., :2]
                    velo_avg = torch.norm(output_joint[i, 1:] - output_joint[i, :-1], dim=-1).mean().item()
                    if velo_avg < 0.01: continue
                    input_joint_viz = query_input_dict['joint'][i].cpu().numpy()
                    if (input_joint_viz[..., 0, :] != 0).any(): print('WARNING: joint #0 isn\'t at origin')
                    output_joint_viz = output_joint[i].cpu().numpy()
                    target_joint_viz = target_joint[i].cpu().numpy()
                    output_vertex_viz = output_smpl[0]['verts'][i].cpu().numpy()
                    target_vertex_viz = target_smpl['verts'][i].cpu().numpy()
                    output_kp3d_viz = output_smpl[0]['kp_3d'][i].cpu().numpy()
                    target_kp3d_viz = target_smpl['kp_3d'][i].cpu().numpy()
                    data_viz = {
                        'query_input': input_joint_viz[::2],
                        'output_joint': output_joint_viz[::2],
                        'target_joint': target_joint_viz[::2],
                        'output_vertex': output_vertex_viz[::2],
                        # 'filler': np.zeros_like(input_joint_viz)[::2],
                        'target_vertex': target_vertex_viz[::2],
                        # 'output_kp3d': output_kp3d_viz[::2],
                        # 'target_kp3d': target_kp3d_viz[::2],
                    }
                    viz_skel_seq_anim(data_viz, mode='img', subplot_layout=(5,1), tight_layout=True, fig_title=f"{dataset}_{eval_task}_{idx}_{i}_queryindex{info_dict['query_index'][i]}", 
                                      if_node=True, lim3d=[0.5,0.5,0.5,0.7,0.7], azim=-140, elev=15,
                                      file_name=f'{dataset}_{eval_task}_queryindex{query_index}',
                                      file_folder='viz_results/exp050106',
                                    #   if_print=True, fs=0.5, lw=5, node_size=2,
                                      if_print=0, fs=0.2, lw=2, node_size=2,
                                      )
                    # exit(0)


            # Evaluate mesh
            if use_smpl:
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
            elif eval_task == 'MIB':
                frame_mask = torch.tensor(info_dict['frame_mask']).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, output_joint.shape[-2], output_joint.shape[-1]).to(output_joint.device)
                output_joint_to_eval = torch.gather(output_joint, 1, frame_mask)
                target_joint_to_eval = torch.gather(target_joint, 1, frame_mask)
            elif eval_task == 'MC':
                joint_mask = torch.tensor(info_dict['joint_mask']).unsqueeze(1).unsqueeze(-1).expand(-1, output_joint.shape[1], -1, output_joint.shape[-1]).to(output_joint.device)
                output_joint_to_eval = torch.gather(output_joint, 2, joint_mask)
                target_joint_to_eval = torch.gather(target_joint, 2, joint_mask)
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
        elif eval_task == 'MIB':
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

    if use_smpl:
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
    if use_smpl:
        field_names = field_names + list(error_dict.keys())
        result_row = result_row + list(error_dict.values())
    summary_table.field_names = field_names
    summary_table.add_row(result_row)
    summary_table.float_format = ".2"

    print(f"costs {time.time()-st:.2f}s")

    if use_smpl:
        return mpjpes_joint_avg, results_detail, header_detail, mpve_mesh_avg, results_mesh_detail, header_mesh_detail, summary_table
    return mpjpes_joint_avg, results_detail, header_detail, summary_table


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