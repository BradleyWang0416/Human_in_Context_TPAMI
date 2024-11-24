#%%
import os
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


def evaluate(args, test_loader, h36m_datareader, model, dataset, eval_task, epoch=None, if_viz=False, if_debug=False):

    use_smpl = (hasattr(args, 'Mesh') and args.Mesh.enable)


    if eval_task == 'PE':
        if dataset in ['H36M_3D']:
            e1, _, summary_table, header, results, smpl_e1 = evaluate_pose_estimation(args, model, test_loader, h36m_datareader, dataset, epoch, if_viz, if_debug, use_smpl)
        else:
            e1, summary_table, header, results, smpl_e1 = evaluate_pose_estimation_excl_H36M(args, test_loader, model, dataset, epoch, if_viz, if_debug, use_smpl)
    elif eval_task == 'FPE':
        e1, summary_table, header, results, smpl_e1 = evaluate_future_pose_estimation(args, test_loader, model, dataset, epoch, if_viz, if_debug, use_smpl)
    elif eval_task == 'MP':
        e1, summary_table, header, results, smpl_e1 = evaluate_motion_prediction(args, test_loader, model, dataset, epoch, if_viz, if_debug, use_smpl)
    elif eval_task == 'MC':
        e1, summary_table, header, results, smpl_e1 = evaluate_motion_completion(args, test_loader, model, dataset, epoch, if_viz, if_debug, use_smpl)
    elif eval_task == 'MIB':
        e1, summary_table, header, results, smpl_e1 = evaluate_motion_in_between(args, test_loader, model, dataset, epoch, if_viz, if_debug, use_smpl)
    return e1, summary_table, header, results, smpl_e1.astype(float)



def evaluate_future_pose_estimation(args, test_loader, model, dataset, epoch=None, if_viz=False, if_debug=False, use_smpl=True):
    st = time.time()
    print(f'\tEvaluating [Future Pose Estimation] on [{dataset}]...', end=' ')
    model.eval()
    num_samples = 0
    frame_to_eval = np.linspace(0, args.clip_len-1, 4).astype(int)
    if dataset in ['PW3D_MESH', 'AMASS']:
        fps = 60 
    elif dataset in ['H36M_3D', 'H36M_MESH']:
        fps = 50
    mpjpe = np.zeros(len(frame_to_eval))

    
    if use_smpl: 
        assert all(args.dataset_config[dataset_name]['return_type'] in ['smpl', 'smpl_x1000', 'all', 'all_x1000'] for dataset_name in args.dataset_task_info['train'].keys())
        assert 'Mesh' in args.func_ver.get('model_name', 'M00_SiC_dynamicTUP')
        losses = AverageMeter()
        losses_dict = {'loss_3d_pos': AverageMeter(), 
                    'loss_3d_scale': AverageMeter(), 
                    'loss_3d_velocity': AverageMeter(),
                    'loss_lv': AverageMeter(), 
                    'loss_lg': AverageMeter(), 
                    'loss_a': AverageMeter(), 
                    'loss_av': AverageMeter(), 
                    'loss_pose': AverageMeter(), 
                    'loss_shape': AverageMeter(),
                    'loss_norm': AverageMeter(),
        }
        mpjpes_mesh = AverageMeter()
        mpves_mesh = AverageMeter()
        results = defaultdict(list)
        smpl = SMPL('third_party/motionbert/data/mesh', batch_size=1).cuda()
        J_regressor = smpl.J_regressor_h36m



    with torch.no_grad():
        for idx, (QUERY_DICT, PROMPT_DICT, INFO) in enumerate(test_loader[(dataset, 'FPE')]):
            assert (torch.tensor([args.task_to_flag[info['task']] for info in INFO]) == args.task_to_flag['FPE']).all()
            QUERY_INPUT = QUERY_DICT['joint_input']         # [N, T, 17, 3]
            QUERY_TARGET = QUERY_DICT['joint_target']       # [N, T, 17, 3]
            PROMPT_INPUT = PROMPT_DICT['joint_input']       # [N, T, 17, 3]
            PROMPT_TARGET = PROMPT_DICT['joint_target']     # [N, T, 17, 3]

            if use_smpl:
                QUERY_POSE_TARGET = QUERY_DICT['pose_target']           # [N, T, 72]
                QUERY_SHAPE_TARGET = QUERY_DICT['shape_target']         # [N, T, 10]
                QUERY_VERTEX_TARGET = QUERY_DICT['vertex_target']       # [N, T, 6890, 3]
                PROMPT_POSE_TARGET = PROMPT_DICT['pose_target']         # [N, T, 72]
                PROMPT_SHAPE_TARGET = PROMPT_DICT['shape_target']       # [N, T, 10]
                PROMPT_VERTEX_TARGET = PROMPT_DICT['vertex_target']
            
            prompt_batch = torch.cat([PROMPT_INPUT, PROMPT_TARGET], dim=-3)
            query_batch = torch.cat([QUERY_INPUT, QUERY_TARGET], dim=-3)
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()  # (B, clip_len*2, 17, 3)
                query_batch = query_batch.cuda()    # (B, clip_len*2, 17, 3)
                if use_smpl:
                    QUERY_POSE_TARGET = QUERY_POSE_TARGET.cuda()
                    QUERY_SHAPE_TARGET = QUERY_SHAPE_TARGET.cuda()
                    QUERY_VERTEX_TARGET = QUERY_VERTEX_TARGET.cuda()
                    PROMPT_POSE_TARGET = PROMPT_POSE_TARGET.cuda()
                    PROMPT_SHAPE_TARGET = PROMPT_SHAPE_TARGET.cuda()
                    PROMPT_VERTEX_TARGET = PROMPT_VERTEX_TARGET.cuda()

            batch_size = len(prompt_batch)
            num_samples += batch_size

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M01_SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'support_data', 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild, target = model(prompt_batch, pseudo_query_batch, query_target, epoch)
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M17_MixSTE_v0_maxlen_Mesh']:
                target = query_batch[:, args.data.clip_len:, :, :].clone()
                rebuild, rebuild_smpl = model(prompt_batch, query_batch, epoch)
                # SMPL
                target_theta = torch.cat([QUERY_POSE_TARGET, QUERY_SHAPE_TARGET], dim=-1).clone()   # [B, T, 82]
                target_verts = QUERY_VERTEX_TARGET.clone()      # [B, T, 6890, 3]
                J_regressor = model.module.mesh_head.J_regressor if model.__class__.__name__ == 'DataParallel' else model.mesh_head.J_regressor     # [17, 6890]
                target_kp3d = torch.einsum('jv,btvc->btjc', J_regressor.to(target_verts.device), target_verts.clone())  # [B, T, 17, 3]
                # target_kp3d/1000 和 target_part 的差的绝对值, 平均9.31x10^(-6), 最大0.0001, 最小0.
                target_smpl = {
                    'theta': target_theta,  # [B, T, 82]
                    'verts': target_verts,  # [B, T, 6890, 3]
                    'kp_3d': target_kp3d    # [B, T, 17, 3]
                }
                

                if if_viz:
                    if if_viz.split(',')[0] != inspect.currentframe().f_code.co_name:
                        return 1,2,3,4
                    if len(if_viz.split(',')) > 1:
                        if if_viz.split(',')[1] != dataset:
                            return 1,2,3,4
                    print(f'Do visualing in [evaluate_future_pose_estimation] on [{dataset}]')
                    for b in range(batch_size):
                        data_to_viz = {
                            '3D joints': target[b],
                            'vertices': target_smpl['verts'][b],
                            '3D joints from vertices': target_smpl['kp_3d'][b]
                        }
                        viz_skel_seq_anim(data_to_viz, subplot_layout=(1,3), if_node=True,
                                          fs=0.5, lim3d=[0.5,500,500])



            else:# args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M00_SiC_dynamicTUP', 'M03_SiC_ChnlCat', 'M04_SiC_dynamicTUP_ChnlCat', 'M05_SiC_CrossAttn', 'M06_MixSTE_v0', 'M07_MixSTE_v1', 'M08_MixSTE_v2', 'M09_MixSTE_v0_1', 'M10_MixSTE_v0_2', 'M11_MixSTE_v0_res', 'M12_MixSTE_v0_avg']:
                target = query_batch[:, args.data.clip_len:, :, :]
                rebuild = model(prompt_batch, query_batch, epoch)
            

            if use_smpl:
                loss_dict = args.Mesh.criterion(rebuild_smpl, target_smpl)     # dict_keys(['loss_3d_pos', 'loss_3d_scale', 'loss_3d_velocity', 'loss_lv', 'loss_lg', 'loss_a', 'loss_av', 'loss_shape', 'loss_pose', 'loss_norm'])
                loss = args.Mesh.lambda_3d      * loss_dict['loss_3d_pos']      + \
                                  args.Mesh.lambda_3dv     * loss_dict['loss_3d_velocity'] + \
                                  args.Mesh.lambda_pose    * loss_dict['loss_pose']        + \
                                  args.Mesh.lambda_shape   * loss_dict['loss_shape']       + \
                                  args.Mesh.lambda_norm    * loss_dict['loss_norm']        + \
                                  args.Mesh.lambda_scale   * loss_dict['loss_3d_scale']    + \
                                  args.Mesh.lambda_lv      * loss_dict['loss_lv']          + \
                                  args.Mesh.lambda_lg      * loss_dict['loss_lg']          + \
                                  args.Mesh.lambda_a       * loss_dict['loss_a']           + \
                                  args.Mesh.lambda_av      * loss_dict['loss_av']
                                  # default lambda: 3d: 0.5
                                  #                 3dv: 10
                                  #                 pose: 1000
                                  #                 shape: 1
                                  #                 norm: 20
                losses.update(loss.item(), batch_size)
                loss_str = ''
                for k, v in loss_dict.items():
                    losses_dict[k].update(v.item(), batch_size)
                    loss_str += '{0} {loss.val:.3f} ({loss.avg:.3f})\t'.format(k, loss=losses_dict[k])
                mpjpe_mesh, mpve_mesh = compute_error(rebuild_smpl, target_smpl)
                mpjpes_mesh.update(mpjpe_mesh, batch_size)
                mpves_mesh.update(mpve_mesh, batch_size)

                for keys in rebuild_smpl[0].keys():
                    rebuild_smpl[0][keys] = rebuild_smpl[0][keys].detach().cpu().numpy()
                    target_smpl[keys] = target_smpl[keys].detach().cpu().numpy()
                results['kp_3d'].append(rebuild_smpl[0]['kp_3d'])
                results['verts'].append(rebuild_smpl[0]['verts'])
                results['kp_3d_gt'].append(target_smpl['kp_3d'])
                results['verts_gt'].append(target_smpl['verts'])



            pred = rebuild[:, frame_to_eval, :, :]     # (B,T,17,3)
            gt = target[:, frame_to_eval, :, :]        # (B,T,17,3)


            
            pred = test_loader[(dataset, 'FPE')].dataset.postprocess(pred, dataset_name=dataset, task='FPE')
            gt = test_loader[(dataset, 'FPE')].dataset.postprocess(gt, dataset_name=dataset, task='FPE')


            mpjpe_ = torch.sum(torch.mean(torch.norm(pred*1000 - gt*1000, dim=3), dim=2), dim=0)
            mpjpe += mpjpe_.cpu().data.numpy()
            if if_debug:
                if idx >1:
                    break
    
    
    if use_smpl:
        for term in results.keys():
            results[term] = np.concatenate(results[term])
        error_dict = evaluate_mesh(results)
        err_str = ''
        for err_key, err_val in error_dict.items():
            # { 'mpve': 303.39117, 
            #   'mpjpe': 281.52002, 
            #   'pa_mpjpe': 139.54958, 
            #   'mpjpe_17j': 244.57515, 
            #   'pa_mpjpe_17j': 131.43068 }
            err_str += '{}: {:.2f}mm \t'.format(err_key, err_val)
        # print(f'=======================> {dataset} validation done: ', loss_str)
        # print(f'=======================> {dataset} validation done: ', err_str)


    mpjpe = mpjpe / num_samples     # (T,)
    mpjpe_avg = np.mean(mpjpe)
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = [f'FPE | {dataset}'] + ['Avg'] + [f'{i}' for i in frame_to_eval]
    summary_table.add_row(['MPJPE'] + [mpjpe_avg] + list(mpjpe))
    summary_table.float_format = ".2"
    print(f"costs {time.time()-st:.2f}s")



    
    return mpjpe_avg, summary_table, [f'{i}' for i in frame_to_eval], list(mpjpe), error_dict['mpve']


def evaluate_motion_completion(args, test_loader, model, dataset, epoch=None, if_viz=False, if_debug=False, use_smpl=True):
    st = time.time()
    print(f'\tEvaluating [Motion Completion] on [{dataset}]...', end=' ')
    model.eval()
    mpjpe_per_ratio = {args.joint_mask_ratio:0}
    count_per_ratio = {args.joint_mask_ratio:0}
    num_samples = 0

    results = defaultdict(list)

    with torch.no_grad():
        for idx, (QUERY_DICT, PROMPT_DICT, INFO) in enumerate(test_loader[(dataset, 'MC')]):
            assert (torch.tensor([args.task_to_flag[info['task']] for info in INFO]) == args.task_to_flag['MC']).all()
            QUERY_INPUT = QUERY_DICT['joint_input']         # [N, T, 17, 3]
            QUERY_TARGET = QUERY_DICT['joint_target']       # [N, T, 17, 3]
            PROMPT_INPUT = PROMPT_DICT['joint_input']       # [N, T, 17, 3]
            PROMPT_TARGET = PROMPT_DICT['joint_target']     # [N, T, 17, 3]

            if use_smpl:
                QUERY_POSE_TARGET = QUERY_DICT['pose_target']           # [N, T, 72]
                QUERY_SHAPE_TARGET = QUERY_DICT['shape_target']         # [N, T, 10]
                QUERY_VERTEX_TARGET = QUERY_DICT['vertex_target']       # [N, T, 6890, 3]
                PROMPT_POSE_TARGET = PROMPT_DICT['pose_target']         # [N, T, 72]
                PROMPT_SHAPE_TARGET = PROMPT_DICT['shape_target']       # [N, T, 10]
                PROMPT_VERTEX_TARGET = PROMPT_DICT['vertex_target']
            prompt_batch = torch.cat([PROMPT_INPUT, PROMPT_TARGET], dim=-3)
            query_batch = torch.cat([QUERY_INPUT, QUERY_TARGET], dim=-3)
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()
                query_batch = query_batch.cuda()
                if use_smpl:
                    QUERY_POSE_TARGET = QUERY_POSE_TARGET.cuda()
                    QUERY_SHAPE_TARGET = QUERY_SHAPE_TARGET.cuda()
                    QUERY_VERTEX_TARGET = QUERY_VERTEX_TARGET.cuda()
                    PROMPT_POSE_TARGET = PROMPT_POSE_TARGET.cuda()
                    PROMPT_SHAPE_TARGET = PROMPT_SHAPE_TARGET.cuda()
                    PROMPT_VERTEX_TARGET = PROMPT_VERTEX_TARGET.cuda()
            batch_size = len(prompt_batch)
            num_samples += batch_size

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M01_SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'support_data', 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild, target = model(prompt_batch, pseudo_query_batch, query_target, epoch)
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M17_MixSTE_v0_maxlen_Mesh']:
                target = query_batch[:, args.data.clip_len:, :, :].clone()
                rebuild, rebuild_smpl = model(prompt_batch, query_batch, epoch)

                # SMPL
                target_theta = torch.cat([QUERY_POSE_TARGET, QUERY_SHAPE_TARGET], dim=-1).clone()   # [B, T, 82]
                target_verts = QUERY_VERTEX_TARGET.clone()      # [B, T, 6890, 3]
                J_regressor = model.module.mesh_head.J_regressor if model.__class__.__name__ == 'DataParallel' else model.mesh_head.J_regressor     # [17, 6890]
                target_kp3d = torch.einsum('jv,btvc->btjc', J_regressor.to(target_verts.device), target_verts.clone())  # [B, T, 17, 3]
                # target_kp3d/1000 和 target_part 的差的绝对值, 平均9.31x10^(-6), 最大0.0001, 最小0.
                target_smpl = {
                    'theta': target_theta,  # [B, T, 82]
                    'verts': target_verts,  # [B, T, 6890, 3]
                    'kp_3d': target_kp3d    # [B, T, 17, 3]
                }

                for keys in rebuild_smpl[0].keys():
                    rebuild_smpl[0][keys] = rebuild_smpl[0][keys].detach().cpu().numpy()
                    target_smpl[keys] = target_smpl[keys].detach().cpu().numpy()
                results['kp_3d'].append(rebuild_smpl[0]['kp_3d'])
                results['verts'].append(rebuild_smpl[0]['verts'])
                results['kp_3d_gt'].append(target_smpl['kp_3d'])
                results['verts_gt'].append(target_smpl['verts'])

            else:# args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M00_SiC_dynamicTUP', 'M03_SiC_ChnlCat', 'M04_SiC_dynamicTUP_ChnlCat', 'M05_SiC_CrossAttn', 'M06_MixSTE_v0', 'M07_MixSTE_v1', 'M08_MixSTE_v2', 'M09_MixSTE_v0_1', 'M10_MixSTE_v0_2', 'M11_MixSTE_v0_res', 'M12_MixSTE_v0_avg']:
                target = query_batch[:, args.data.clip_len:, :, :]
                rebuild = model(prompt_batch, query_batch, epoch)

            for i in range(batch_size):
                pred = rebuild[i]        # (clip_len, 17, 3)
                gt = target[i]           # (clip_len, 17, 3)
                joint_mask_idx = INFO[i]['joint_mask']


                if if_viz:
                    if if_viz.split(',')[0] != inspect.currentframe().f_code.co_name:
                        return 1,2,3,4
                    if len(if_viz.split(',')) > 1:
                        if if_viz.split(',')[1] != dataset:
                            return 1,2,3,4
                    if dataset == 'PW3D_MESH':
                        H36M_chunks_2d, H36M_chunks_3d = test_loader[('H36M_3D', 'MC')].dataset.prepare_chunk(dataset_name='H36M_3D', chunk_id=None, is_query=True)      # [96596, 32, 17, 3]; [96596, 32, 17, 3]
                        H36M_inputs, H36M_targets = test_loader[('H36M_3D', 'MC')].dataset.prepare_motion(H36M_chunks_2d, H36M_chunks_3d, dataset_name='H36M_3D', task='MC', joint_mask=joint_mask_idx, frame_mask=None)    # [96596, 16, 17, 3]; [96596, 16, 17, 3]
                        AMASS_chunks_2d, AMASS_chunks_3d = test_loader[('AMASS', 'MC')].dataset.prepare_chunk(dataset_name='AMASS', chunk_id=None, is_query=True)      # [96596, 32, 17, 3]; [96596, 32, 17, 3]
                        AMASS_inputs, AMASS_targets = test_loader[('AMASS', 'MC')].dataset.prepare_motion(AMASS_chunks_2d, AMASS_chunks_3d, dataset_name='AMASS', task='MC', joint_mask=joint_mask_idx, frame_mask=None)    # [96596, 16, 17, 3]; [96596, 16, 17, 3]
                        all_targets = torch.cat([H36M_targets, AMASS_targets], dim=0)
                        dist = torch.norm(QUERY_TARGET[i:i+1] - all_targets, dim=-1).mean(-1).mean(-1)
                        min_dist, min_idx = torch.topk(dist, 10, largest=False)
                        closest_indices = min_idx.cpu().numpy()
                        closest_dists = min_dist.cpu().numpy()
                        target_alt = all_targets[closest_indices[0]:closest_indices[0]+1]
                        input_alt = target_alt.clone()
                        input_alt[..., joint_mask_idx, :] = 0.
                        prompt_alt = torch.cat([input_alt, target_alt], dim=-3)
                        rebuild_alt = model(prompt_alt, query_batch[i:i+1], epoch)
                        pred = rebuild_alt.squeeze(0)


                pred = test_loader[(dataset, 'MC')].dataset.postprocess(pred, dataset_name=dataset, task='MC')
                gt = test_loader[(dataset, 'MC')].dataset.postprocess(gt, dataset_name=dataset, task='MC')

                if if_viz:
                    if if_viz.split(',')[0] != inspect.currentframe().f_code.co_name:
                        return 1,2,3,4
                    if len(if_viz.split(',')) > 1:
                        if if_viz.split(',')[1] != dataset:
                            return 1,2,3,4
                    print(f'!!! Do visualing now in [{inspect.currentframe().f_code.co_name}]')


                    if i % 130 != 0: continue
                    H36M_chunks_2d, H36M_chunks_3d = test_loader[('H36M_3D', 'MC')].dataset.prepare_chunk(dataset_name='H36M_3D', chunk_id=None, is_query=True)      # [96596, 32, 17, 3]; [96596, 32, 17, 3]
                    H36M_inputs, H36M_targets = test_loader[('H36M_3D', 'MC')].dataset.prepare_motion(H36M_chunks_2d, H36M_chunks_3d, dataset_name='H36M_3D', task='MC', joint_mask=joint_mask_idx, frame_mask=None)    # [96596, 16, 17, 3]; [96596, 16, 17, 3]
                    AMASS_chunks_2d, AMASS_chunks_3d = test_loader[('AMASS', 'MC')].dataset.prepare_chunk(dataset_name='AMASS', chunk_id=None, is_query=True)      # [96596, 32, 17, 3]; [96596, 32, 17, 3]
                    AMASS_inputs, AMASS_targets = test_loader[('AMASS', 'MC')].dataset.prepare_motion(AMASS_chunks_2d, AMASS_chunks_3d, dataset_name='AMASS', task='MC', joint_mask=joint_mask_idx, frame_mask=None)    # [96596, 16, 17, 3]; [96596, 16, 17, 3]
                    all_targets = torch.cat([H36M_targets, AMASS_targets], dim=0)
                    dist = torch.norm(QUERY_TARGET[i:i+1] - all_targets, dim=-1).mean(-1).mean(-1)
                    min_dist, min_idx = torch.topk(dist, 10, largest=False)
                    closest_indices = min_idx.cpu().numpy()
                    closest_dists = min_dist.cpu().numpy()

                    # data_to_viz = {
                    #     '3DPW input': QUERY_INPUT[i][::2],
                    #     '3DPW': QUERY_TARGET[i][::2],
                    #     'H36M min1': all_targets[closest_indices[0]][::2],
                    #     'H36M min2': all_targets[closest_indices[1]][::2],
                    #     'H36M min3': all_targets[closest_indices[2]][::2],
                    # }
                    # viz_skel_seq_anim(data_to_viz, mode='img', subplot_layout=(3,2), fs=0.2, lim3d=0.5, fig_title=f"{INFO[i]['query_index']}-dist{np.round(closest_dists[:3], 2)}-idx{closest_indices[:3]}",
                    #                   )
                    
                    target_alt = all_targets[closest_indices[0]:closest_indices[0]+1]
                    input_alt = target_alt.clone()
                    input_alt[..., joint_mask_idx, :] = 0.
                    prompt_alt = torch.cat([input_alt, target_alt], dim=-3)
                    rebuild_alt = model(prompt_alt, query_batch[i:i+1], epoch)

                    data_to_viz = {
                        'prompt input': PROMPT_INPUT[i][::2], 'alt prompt input': input_alt.squeeze(0)[::2],
                        'prompt target': PROMPT_TARGET[i][::2], 'alt prompt target': target_alt.squeeze(0)[::2],
                        'pred': pred[::2], 'alt pred': rebuild_alt.squeeze(0)[::2],
                        'query input': QUERY_INPUT[i][::2], 'query target': QUERY_TARGET[i][::2],
                    }
                    viz_skel_seq_anim(data_to_viz, mode='img', subplot_layout=(4,2), fs=0.2, lim3d=0.5,
                                      )
                    continue


                    prompt_batch_h36m = torch.cat([PROMPT_INPUT_h36m, PROMPT_TARGET_h36m], dim=-3)
                    if args.normalize_3d:
                        mean_3d_h36m = test_loader[('H36M_3D', 'MC')].dataset.query_dict['H36M_3D']['mean_3d']
                        std_3d_h36m = test_loader[('H36M_3D', 'MC')].dataset.query_dict['H36M_3D']['std_3d']
                        mean_3d_h36m = torch.from_numpy(mean_3d_h36m).float().cuda()
                        std_3d_h36m = torch.from_numpy(std_3d_h36m).float().cuda()
                        query_batch_norm_as_h36m = query_batch[i:i+1].clone() * std_3d + mean_3d
                        query_batch_norm_as_h36m = (query_batch_norm_as_h36m - mean_3d_h36m) / std_3d_h36m
                    rebuild_h36m_prompted = model(prompt_batch_h36m, query_batch_norm_as_h36m, epoch)
                    rebuild_h36m_prompted = rebuild_h36m_prompted.squeeze(0)
                    if args.normalize_3d:
                        # rebuild_h36m_prompted = rebuild_h36m_prompted * std_3d + mean_3d
                        rebuild_h36m_prompted = rebuild_h36m_prompted * std_3d_h36m + mean_3d_h36m


                    data_viz = {
                        'input': query_batch[i, :args.data.clip_len, :, :][::2] * std_3d + mean_3d,
                        'pred': pred[::2],
                        'pred (prompted by h36m)': rebuild_h36m_prompted[::2],
                        'gt': gt[::2],
                    }
                    viz_skel_seq_anim(data_viz, mode='img', subplot_layout=(4,1), fs=0.3, lim3d=0.5, fig_title=f'{dataset}-{idx}-{i}-mask{joint_mask_idx}',
                                        azim=90, elev=90)

                
                pred_ = pred[:, joint_mask_idx]
                gt_ = gt[:, joint_mask_idx]

                masked_frame_num = len(joint_mask_idx)
                assert masked_frame_num == int(args.joint_mask_ratio*args.num_joint)
                mpjpe_ = torch.mean(torch.norm(pred_*1000 - gt_*1000, dim=2))

                for ratio in count_per_ratio:
                    if masked_frame_num == int(ratio * args.num_joint):
                        count_per_ratio[ratio] += 1
                        mpjpe_per_ratio[ratio] += mpjpe_.cpu().data.numpy()
            if if_debug:
                if idx >1:
                    break

    
    for term in results.keys():
        results[term] = np.concatenate(results[term])
    error_dict = evaluate_mesh(results)
    
    assert sum([cnt for ratio, cnt in count_per_ratio.items()]) == num_samples
    for ratio in count_per_ratio:
        num_samples = count_per_ratio[ratio]
        mpjpe_per_ratio[ratio] = mpjpe_per_ratio[ratio] / num_samples
    mpjpe_avg = np.mean(np.array([err for ratio, err in mpjpe_per_ratio.items()]))

    summary_table = prettytable.PrettyTable()
    summary_table.field_names = [f'MC | {dataset}'] + ['Avg'] + [ratio for ratio, err in mpjpe_per_ratio.items()]
    summary_table.add_row(['MPJPE'] + [mpjpe_avg] + [err for ratio, err in mpjpe_per_ratio.items()])
    summary_table.float_format = ".2"
    print(f"costs {time.time()-st:.2f}s")
    return mpjpe_avg, summary_table, [ratio for ratio, err in mpjpe_per_ratio.items()], [err for ratio, err in mpjpe_per_ratio.items()], error_dict['mpve']


def evaluate_motion_in_between(args, test_loader, model, dataset, epoch=None, if_viz=False, if_debug=False, use_smpl=True):
    st = time.time()
    print(f'\tEvaluating [Motion In Between] on [{dataset}]...', end=' ')
    model.eval()
    mpjpe_per_ratio = {args.frame_mask_ratio:0}
    count_per_ratio = {args.frame_mask_ratio:0}
    num_samples = 0

    results = defaultdict(list)

    with torch.no_grad():
        for idx, (QUERY_DICT, PROMPT_DICT, INFO) in enumerate(test_loader[(dataset, 'MIB')]):
            assert (torch.tensor([args.task_to_flag[info['task']] for info in INFO]) == args.task_to_flag['MIB']).all()
            QUERY_INPUT = QUERY_DICT['joint_input']         # [N, T, 17, 3]
            QUERY_TARGET = QUERY_DICT['joint_target']       # [N, T, 17, 3]
            PROMPT_INPUT = PROMPT_DICT['joint_input']       # [N, T, 17, 3]
            PROMPT_TARGET = PROMPT_DICT['joint_target']     # [N, T, 17, 3]

            if use_smpl:
                QUERY_POSE_TARGET = QUERY_DICT['pose_target']           # [N, T, 72]
                QUERY_SHAPE_TARGET = QUERY_DICT['shape_target']         # [N, T, 10]
                QUERY_VERTEX_TARGET = QUERY_DICT['vertex_target']       # [N, T, 6890, 3]
                PROMPT_POSE_TARGET = PROMPT_DICT['pose_target']         # [N, T, 72]
                PROMPT_SHAPE_TARGET = PROMPT_DICT['shape_target']       # [N, T, 10]
                PROMPT_VERTEX_TARGET = PROMPT_DICT['vertex_target']
            prompt_batch = torch.cat([PROMPT_INPUT, PROMPT_TARGET], dim=-3)
            query_batch = torch.cat([QUERY_INPUT, QUERY_TARGET], dim=-3)
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()
                query_batch = query_batch.cuda()
                if use_smpl:
                    QUERY_POSE_TARGET = QUERY_POSE_TARGET.cuda()
                    QUERY_SHAPE_TARGET = QUERY_SHAPE_TARGET.cuda()
                    QUERY_VERTEX_TARGET = QUERY_VERTEX_TARGET.cuda()
                    PROMPT_POSE_TARGET = PROMPT_POSE_TARGET.cuda()
                    PROMPT_SHAPE_TARGET = PROMPT_SHAPE_TARGET.cuda()
                    PROMPT_VERTEX_TARGET = PROMPT_VERTEX_TARGET.cuda()
            batch_size = len(prompt_batch)
            num_samples += batch_size

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M01_SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'support_data', 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild, target = model(prompt_batch, pseudo_query_batch, query_target, epoch)
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M17_MixSTE_v0_maxlen_Mesh']:
                target = query_batch[:, args.data.clip_len:, :, :].clone()
                rebuild, rebuild_smpl = model(prompt_batch, query_batch, epoch)

                # SMPL
                target_theta = torch.cat([QUERY_POSE_TARGET, QUERY_SHAPE_TARGET], dim=-1).clone()   # [B, T, 82]
                target_verts = QUERY_VERTEX_TARGET.clone()      # [B, T, 6890, 3]
                J_regressor = model.module.mesh_head.J_regressor if model.__class__.__name__ == 'DataParallel' else model.mesh_head.J_regressor     # [17, 6890]
                target_kp3d = torch.einsum('jv,btvc->btjc', J_regressor.to(target_verts.device), target_verts.clone())  # [B, T, 17, 3]
                # target_kp3d/1000 和 target_part 的差的绝对值, 平均9.31x10^(-6), 最大0.0001, 最小0.
                target_smpl = {
                    'theta': target_theta,  # [B, T, 82]
                    'verts': target_verts,  # [B, T, 6890, 3]
                    'kp_3d': target_kp3d    # [B, T, 17, 3]
                }

                for keys in rebuild_smpl[0].keys():
                    rebuild_smpl[0][keys] = rebuild_smpl[0][keys].detach().cpu().numpy()
                    target_smpl[keys] = target_smpl[keys].detach().cpu().numpy()
                results['kp_3d'].append(rebuild_smpl[0]['kp_3d'])
                results['verts'].append(rebuild_smpl[0]['verts'])
                results['kp_3d_gt'].append(target_smpl['kp_3d'])
                results['verts_gt'].append(target_smpl['verts'])

            else:# args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M00_SiC_dynamicTUP', 'M03_SiC_ChnlCat', 'M04_SiC_dynamicTUP_ChnlCat', 'M05_SiC_CrossAttn', 'M06_MixSTE_v0', 'M07_MixSTE_v1', 'M08_MixSTE_v2', 'M09_MixSTE_v0_1', 'M10_MixSTE_v0_2', 'M11_MixSTE_v0_res', 'M12_MixSTE_v0_avg']:
                target = query_batch[:, args.data.clip_len:, :, :]
                rebuild = model(prompt_batch, query_batch, epoch)

            for i in range(batch_size):
                pred = rebuild[i]        # (clip_len, 17, 3)
                gt = target[i]           # (clip_len, 17, 3)
                frame_mask_idx = INFO[i]['frame_mask']



                pred = test_loader[(dataset, 'MIB')].dataset.postprocess(pred, dataset_name=dataset, task='MIB')
                gt = test_loader[(dataset, 'MIB')].dataset.postprocess(gt, dataset_name=dataset, task='MIB')


                pred_ = pred[frame_mask_idx]
                gt_ = gt[frame_mask_idx]

                masked_frame_num = len(frame_mask_idx)
                assert masked_frame_num == int(args.frame_mask_ratio*args.clip_len)
                mpjpe_ = torch.mean(torch.norm(pred_*1000 - gt_*1000, dim=2))


                if if_viz:
                    if if_viz.split(',')[0] != inspect.currentframe().f_code.co_name:
                        return 1,2,3,4
                    if len(if_viz.split(',')) > 1:
                        if if_viz.split(',')[1] != dataset:
                            return 1,2,3,4
                    print(f'!!! Do visualing now in [{inspect.currentframe().f_code.co_name}]')
                    if dataset == 'H36M_3D':
                        data_viz = {
                            'input': query_batch[i, :args.data.clip_len, :, :],
                            'pred': pred,
                            'gt': gt,
                        }
                        viz_skel_seq_anim(data_viz, mode='img', subplot_layout=(3,1), fs=0.15, lim3d=0.4, fig_title=f'{dataset}-{idx}-{i}-mask{frame_mask_idx}-{mpjpe_}',
                                          azim=90, elev=90)


                for ratio in count_per_ratio:
                    if masked_frame_num == int(ratio * args.clip_len):
                        count_per_ratio[ratio] += 1
                        mpjpe_per_ratio[ratio] += mpjpe_.cpu().data.numpy()


            if if_debug:
                if idx >1:
                    break

    for term in results.keys():
        results[term] = np.concatenate(results[term])
    error_dict = evaluate_mesh(results)
    
    assert sum([cnt for ratio, cnt in count_per_ratio.items()]) == num_samples
    for ratio in count_per_ratio:
        num_samples = count_per_ratio[ratio]
        mpjpe_per_ratio[ratio] = mpjpe_per_ratio[ratio] / num_samples
    mpjpe_avg = np.mean(np.array([err for ratio, err in mpjpe_per_ratio.items()]))

    summary_table = prettytable.PrettyTable()
    summary_table.field_names = [f'MIB | {dataset}'] + ['Avg'] + [ratio for ratio, err in mpjpe_per_ratio.items()]
    summary_table.add_row(['MPJPE'] + [mpjpe_avg] + [err for ratio, err in mpjpe_per_ratio.items()])
    summary_table.float_format = ".2"
    print(f"costs {time.time()-st:.2f}s")
    return mpjpe_avg, summary_table, [ratio for ratio, err in mpjpe_per_ratio.items()], [err for ratio, err in mpjpe_per_ratio.items()], error_dict['mpve']
    

def evaluate_motion_prediction(args, test_loader, model, dataset, epoch=None, if_viz=False, if_debug=False, use_smpl=True):
    st = time.time()
    print(f'\tEvaluating [Motion Prediction] on [{dataset}]...', end=' ')
    model.eval()
    num_samples = 0
    frame_to_eval = np.linspace(0, args.clip_len-1, 4).astype(int)
    if dataset in ['PW3D_MESH', 'AMASS']:
        fps = 60 
    elif dataset in ['H36M_3D', 'H36M_MESH']:
        fps = 50
    mpjpe = np.zeros(len(frame_to_eval))

    results = defaultdict(list)

    with torch.no_grad():
        for idx, (QUERY_DICT, PROMPT_DICT, INFO) in enumerate(test_loader[(dataset, 'MP')]):
            assert (torch.tensor([args.task_to_flag[info['task']] for info in INFO]) == args.task_to_flag['MP']).all()
            QUERY_INPUT = QUERY_DICT['joint_input']         # [N, T, 17, 3]
            QUERY_TARGET = QUERY_DICT['joint_target']       # [N, T, 17, 3]
            PROMPT_INPUT = PROMPT_DICT['joint_input']       # [N, T, 17, 3]
            PROMPT_TARGET = PROMPT_DICT['joint_target']     # [N, T, 17, 3]

            if use_smpl:
                QUERY_POSE_TARGET = QUERY_DICT['pose_target']           # [N, T, 72]
                QUERY_SHAPE_TARGET = QUERY_DICT['shape_target']         # [N, T, 10]
                QUERY_VERTEX_TARGET = QUERY_DICT['vertex_target']       # [N, T, 6890, 3]
                PROMPT_POSE_TARGET = PROMPT_DICT['pose_target']         # [N, T, 72]
                PROMPT_SHAPE_TARGET = PROMPT_DICT['shape_target']       # [N, T, 10]
                PROMPT_VERTEX_TARGET = PROMPT_DICT['vertex_target']
            prompt_batch = torch.cat([PROMPT_INPUT, PROMPT_TARGET], dim=-3)
            query_batch = torch.cat([QUERY_INPUT, QUERY_TARGET], dim=-3)
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()
                query_batch = query_batch.cuda()
                if use_smpl:
                    QUERY_POSE_TARGET = QUERY_POSE_TARGET.cuda()
                    QUERY_SHAPE_TARGET = QUERY_SHAPE_TARGET.cuda()
                    QUERY_VERTEX_TARGET = QUERY_VERTEX_TARGET.cuda()
                    PROMPT_POSE_TARGET = PROMPT_POSE_TARGET.cuda()
                    PROMPT_SHAPE_TARGET = PROMPT_SHAPE_TARGET.cuda()
                    PROMPT_VERTEX_TARGET = PROMPT_VERTEX_TARGET.cuda()
            batch_size = len(prompt_batch)
            num_samples += batch_size

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M01_SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'support_data', 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild, target = model(prompt_batch, pseudo_query_batch, query_target, epoch)
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M17_MixSTE_v0_maxlen_Mesh']:
                target = query_batch[:, args.data.clip_len:, :, :].clone()
                rebuild, rebuild_smpl = model(prompt_batch, query_batch, epoch)

                # SMPL
                target_theta = torch.cat([QUERY_POSE_TARGET, QUERY_SHAPE_TARGET], dim=-1).clone()   # [B, T, 82]
                target_verts = QUERY_VERTEX_TARGET.clone()      # [B, T, 6890, 3]
                J_regressor = model.module.mesh_head.J_regressor if model.__class__.__name__ == 'DataParallel' else model.mesh_head.J_regressor     # [17, 6890]
                target_kp3d = torch.einsum('jv,btvc->btjc', J_regressor.to(target_verts.device), target_verts.clone())  # [B, T, 17, 3]
                # target_kp3d/1000 和 target_part 的差的绝对值, 平均9.31x10^(-6), 最大0.0001, 最小0.
                target_smpl = {
                    'theta': target_theta,  # [B, T, 82]
                    'verts': target_verts,  # [B, T, 6890, 3]
                    'kp_3d': target_kp3d    # [B, T, 17, 3]
                }

                for keys in rebuild_smpl[0].keys():
                    rebuild_smpl[0][keys] = rebuild_smpl[0][keys].detach().cpu().numpy()
                    target_smpl[keys] = target_smpl[keys].detach().cpu().numpy()
                results['kp_3d'].append(rebuild_smpl[0]['kp_3d'])
                results['verts'].append(rebuild_smpl[0]['verts'])
                results['kp_3d_gt'].append(target_smpl['kp_3d'])
                results['verts_gt'].append(target_smpl['verts'])


            else:# args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M00_SiC_dynamicTUP', 'M03_SiC_ChnlCat', 'M04_SiC_dynamicTUP_ChnlCat', 'M05_SiC_CrossAttn', 'M06_MixSTE_v0', 'M07_MixSTE_v1', 'M08_MixSTE_v2', 'M09_MixSTE_v0_1', 'M10_MixSTE_v0_2', 'M11_MixSTE_v0_res', 'M12_MixSTE_v0_avg']:
                target = query_batch[:, args.data.clip_len:, :, :]
                rebuild = model(prompt_batch, query_batch, epoch)

            pred = rebuild.clone()     # (B,T,17,3)
            gt = target.clone()        # (B,T,17,3)


            pred = test_loader[(dataset, 'MP')].dataset.postprocess(pred, dataset_name=dataset, task='MP')
            gt = test_loader[(dataset, 'MP')].dataset.postprocess(gt, dataset_name=dataset, task='MP')

            
            if if_viz:
                if if_viz.split(',')[0] != inspect.currentframe().f_code.co_name:
                    return 1, 2, 3, 4
                if len(if_viz.split(',')) > 1:
                    if if_viz.split(',')[1] != dataset:
                        return 1, 2, 3, 4
                
                print(f'!!! Do visualing now in [{inspect.currentframe().f_code.co_name}]')
                if dataset in ['H36M_3D', 'PW3D_MESH']:
                    ii = 0
                    while ii < batch_size:
                        errr = torch.mean(torch.norm(pred[ii, frame_to_eval]*1000 - gt[ii, frame_to_eval]*1000, dim=2), dim=1).cpu().numpy()
                        errr = np.round(errr, 1)
                        velocity = np.mean(np.linalg.norm(pred[ii, 1:, :, :].cpu().numpy()*1000 - pred[ii, :-1, :, :].cpu().numpy()*1000, axis=-1))
                        if velocity < 5:
                            ii += 1
                            continue
                        data_viz = {
                            'input normed': query_batch[ii, :args.data.clip_len][::2],
                            'input denormed': query_batch[ii, :args.data.clip_len][::2] * std_3d + mean_3d,
                            'pred': pred[ii][::2],
                            'gt': gt[ii][::2],
                        }
                        viz_skel_seq_anim(data_viz, mode='img', subplot_layout=(4,1), fs=0.3, lim3d=0.4, fig_title=f'{dataset}_{idx}_{ii}-err{errr}-{frame_to_eval}-velo{velocity:.4f}',
                                          azim=78, elev=71)

                        ii += 1


            mpjpe_ = torch.sum(torch.mean(torch.norm(pred[:, frame_to_eval, :, :]*1000 - gt[:, frame_to_eval, :, :]*1000, dim=3), dim=2), dim=0)
            mpjpe += mpjpe_.cpu().data.numpy()
            if if_debug:
                if idx >1:
                    break

    for term in results.keys():
        results[term] = np.concatenate(results[term])
    error_dict = evaluate_mesh(results)
    
    mpjpe = mpjpe / num_samples     # (T,)
    mpjpe_avg = np.mean(mpjpe)
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = [f'MP | {dataset}'] + ['Avg'] + [f'{i}' for i in frame_to_eval]
    summary_table.add_row(['MPJPE'] + [mpjpe_avg] + list(mpjpe))
    summary_table.float_format = ".2"
    print(f"costs {time.time()-st:.2f}s")
    return mpjpe_avg, summary_table, [f'{i}' for i in frame_to_eval], list(mpjpe), error_dict['mpve']


def evaluate_pose_estimation(args, model_pos, test_loader, datareader, dataset, epoch=None, if_viz=False, if_debug=False, use_smpl=True):


    _, split_id_test = datareader.get_split_id()
    for i, split_range in enumerate(split_id_test):
        if args.current_as_history:
            min_id = min(split_range)
            split_id_test[i] = range(min_id, min_id + args.clip_len)
        else:
            max_id = max(split_range)
            split_id_test[i] = range(max_id+1 - args.clip_len, max_id+1)
    datareader.split_id_test = split_id_test

    st = time.time()
    print(f'\tEvaluating [3D Pose Estimation] on [{dataset}]...', end=' ')
    results_all = []
    model_pos.eval()

    results = defaultdict(list)

    with torch.no_grad():
        for idx, (QUERY_DICT, PROMPT_DICT, INFO) in enumerate(test_loader[(dataset, 'PE')]):
            assert (torch.tensor([args.task_to_flag[info['task']] for info in INFO]) == args.task_to_flag['PE']).all()
            QUERY_INPUT = QUERY_DICT['joint_input']         # [N, T, 17, 3]
            QUERY_TARGET = QUERY_DICT['joint_target']       # [N, T, 17, 3]
            PROMPT_INPUT = PROMPT_DICT['joint_input']       # [N, T, 17, 3]
            PROMPT_TARGET = PROMPT_DICT['joint_target']     # [N, T, 17, 3]

            if use_smpl:
                QUERY_POSE_TARGET = QUERY_DICT['pose_target']           # [N, T, 72]
                QUERY_SHAPE_TARGET = QUERY_DICT['shape_target']         # [N, T, 10]
                QUERY_VERTEX_TARGET = QUERY_DICT['vertex_target']       # [N, T, 6890, 3]
                PROMPT_POSE_TARGET = PROMPT_DICT['pose_target']         # [N, T, 72]
                PROMPT_SHAPE_TARGET = PROMPT_DICT['shape_target']       # [N, T, 10]
                PROMPT_VERTEX_TARGET = PROMPT_DICT['vertex_target']
            prompt_batch = torch.cat([PROMPT_INPUT, PROMPT_TARGET], dim=-3)
            query_batch = torch.cat([QUERY_INPUT, QUERY_TARGET], dim=-3)
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()
                query_batch = query_batch.cuda()
                if use_smpl:
                    QUERY_POSE_TARGET = QUERY_POSE_TARGET.cuda()
                    QUERY_SHAPE_TARGET = QUERY_SHAPE_TARGET.cuda()
                    QUERY_VERTEX_TARGET = QUERY_VERTEX_TARGET.cuda()
                    PROMPT_POSE_TARGET = PROMPT_POSE_TARGET.cuda()
                    PROMPT_SHAPE_TARGET = PROMPT_SHAPE_TARGET.cuda()
                    PROMPT_VERTEX_TARGET = PROMPT_VERTEX_TARGET.cuda()
            batch_size = len(prompt_batch)

            if if_debug:
                rebuild_part = query_batch[:, args.data.clip_len:]
                if args.normalize_3d:
                    mean_3d = test_loader[(dataset, 'PE')].dataset.query_dict[dataset]['mean_3d']
                    std_3d = test_loader[(dataset, 'PE')].dataset.query_dict[dataset]['std_3d']
                    mean_3d = torch.from_numpy(mean_3d).float().cuda()
                    std_3d = torch.from_numpy(std_3d).float().cuda()
                    rebuild_part = rebuild_part * std_3d + mean_3d
                if dataset == 'H36M_3D':
                    rebuild_part = rebuild_part * args.dataset_config['H36M_3D']['scale_3D']
                # if args.dataset_config[dataset]['rootrel_target']:
                #     rebuild_part[..., 0, :] = 0
                results_all.append(rebuild_part.cpu().numpy())
                continue

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M01_SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'support_data', 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild_part, target_part = model_pos(prompt_batch, pseudo_query_batch, query_target, epoch)
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M17_MixSTE_v0_maxlen_Mesh']:
                rebuild_part, rebuild_smpl = model_pos(prompt_batch, query_batch, epoch)

                # SMPL
                target_theta = torch.cat([QUERY_POSE_TARGET, QUERY_SHAPE_TARGET], dim=-1).clone()   # [B, T, 82]
                target_verts = QUERY_VERTEX_TARGET.clone()      # [B, T, 6890, 3]
                J_regressor = model_pos.module.mesh_head.J_regressor if model_pos.__class__.__name__ == 'DataParallel' else model_pos.mesh_head.J_regressor     # [17, 6890]
                target_kp3d = torch.einsum('jv,btvc->btjc', J_regressor.to(target_verts.device), target_verts.clone())  # [B, T, 17, 3]
                # target_kp3d/1000 和 target_part 的差的绝对值, 平均9.31x10^(-6), 最大0.0001, 最小0.
                target_smpl = {
                    'theta': target_theta,  # [B, T, 82]
                    'verts': target_verts,  # [B, T, 6890, 3]
                    'kp_3d': target_kp3d    # [B, T, 17, 3]
                }

                for keys in rebuild_smpl[0].keys():
                    rebuild_smpl[0][keys] = rebuild_smpl[0][keys].detach().cpu().numpy()
                    target_smpl[keys] = target_smpl[keys].detach().cpu().numpy()
                results['kp_3d'].append(rebuild_smpl[0]['kp_3d'])
                results['verts'].append(rebuild_smpl[0]['verts'])
                results['kp_3d_gt'].append(target_smpl['kp_3d'])
                results['verts_gt'].append(target_smpl['verts'])


            else:# args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M00_SiC_dynamicTUP', 'M03_SiC_ChnlCat', 'M04_SiC_dynamicTUP_ChnlCat', 'M05_SiC_CrossAttn', 'M06_MixSTE_v0', 'M07_MixSTE_v1', 'M08_MixSTE_v2', 'M09_MixSTE_v0_1', 'M10_MixSTE_v0_2', 'M11_MixSTE_v0_res', 'M12_MixSTE_v0_avg']:
                # target_part = query_batch[:, args.data.clip_len:, :, :]
                rebuild_part = model_pos(prompt_batch, query_batch, epoch)


            rebuild_part = test_loader[(dataset, 'PE')].dataset.postprocess(rebuild_part, dataset_name=dataset, task='PE')


            results_all.append(rebuild_part.cpu().numpy())

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

    summary_table = prettytable.PrettyTable()
    summary_table.field_names = [f'PE | {dataset}'] + ['Avg'] + action_names
    summary_table.add_row(['MPJPE'] + [e1] + final_result)
    summary_table.add_row(['P-MPJPE'] + [e2] + final_result_procrustes)
    summary_table.float_format = ".2"
    print(f"costs {time.time()-st:.2f}s")

    for term in results.keys():
        results[term] = np.concatenate(results[term])
    error_dict = evaluate_mesh(results)

    return e1, e2, summary_table, action_names, final_result, error_dict['mpve']


def evaluate_pose_estimation_excl_H36M(args, test_loader, model, dataset, epoch=None, if_viz=False, if_debug=False, use_smpl=True):
    st = time.time()
    print(f'\tEvaluating [3D Pose Estimation] on [{dataset}]...', end=' ')
    model.eval()
    num_samples = 0
    frame_to_eval = np.linspace(0, args.clip_len-1, 2).astype(int)
    if dataset in ['PW3D_MESH', 'AMASS']:
        fps = 60 
    elif dataset in ['H36M_3D', 'H36M_MESH']:
        fps = 50
    mpjpe = np.zeros(len(frame_to_eval))

    results = defaultdict(list)

    with torch.no_grad():
        for idx, (QUERY_DICT, PROMPT_DICT, INFO) in enumerate(test_loader[(dataset, 'PE')]):
            assert (torch.tensor([args.task_to_flag[info['task']] for info in INFO]) == args.task_to_flag['PE']).all()
            QUERY_INPUT = QUERY_DICT['joint_input']         # [N, T, 17, 3]
            QUERY_TARGET = QUERY_DICT['joint_target']       # [N, T, 17, 3]
            PROMPT_INPUT = PROMPT_DICT['joint_input']       # [N, T, 17, 3]
            PROMPT_TARGET = PROMPT_DICT['joint_target']     # [N, T, 17, 3]

            if use_smpl:
                QUERY_POSE_TARGET = QUERY_DICT['pose_target']           # [N, T, 72]
                QUERY_SHAPE_TARGET = QUERY_DICT['shape_target']         # [N, T, 10]
                QUERY_VERTEX_TARGET = QUERY_DICT['vertex_target']       # [N, T, 6890, 3]
                PROMPT_POSE_TARGET = PROMPT_DICT['pose_target']         # [N, T, 72]
                PROMPT_SHAPE_TARGET = PROMPT_DICT['shape_target']       # [N, T, 10]
                PROMPT_VERTEX_TARGET = PROMPT_DICT['vertex_target']
            prompt_batch = torch.cat([PROMPT_INPUT, PROMPT_TARGET], dim=-3)
            query_batch = torch.cat([QUERY_INPUT, QUERY_TARGET], dim=-3)
            if torch.cuda.is_available():
                prompt_batch = prompt_batch.cuda()
                query_batch = query_batch.cuda()
                if use_smpl:
                    QUERY_POSE_TARGET = QUERY_POSE_TARGET.cuda()
                    QUERY_SHAPE_TARGET = QUERY_SHAPE_TARGET.cuda()
                    QUERY_VERTEX_TARGET = QUERY_VERTEX_TARGET.cuda()
                    PROMPT_POSE_TARGET = PROMPT_POSE_TARGET.cuda()
                    PROMPT_SHAPE_TARGET = PROMPT_SHAPE_TARGET.cuda()
                    PROMPT_VERTEX_TARGET = PROMPT_VERTEX_TARGET.cuda()
            batch_size = len(prompt_batch)
            num_samples += batch_size

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M01_SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'support_data', 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild, target = model(prompt_batch, pseudo_query_batch, query_target, epoch)
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M17_MixSTE_v0_maxlen_Mesh']:
                target = query_batch[:, args.data.clip_len:, :, :].clone()
                rebuild, rebuild_smpl = model(prompt_batch, query_batch, epoch)

                # SMPL
                target_theta = torch.cat([QUERY_POSE_TARGET, QUERY_SHAPE_TARGET], dim=-1).clone()   # [B, T, 82]
                target_verts = QUERY_VERTEX_TARGET.clone()      # [B, T, 6890, 3]
                J_regressor = model.module.mesh_head.J_regressor if model.__class__.__name__ == 'DataParallel' else model.mesh_head.J_regressor     # [17, 6890]
                target_kp3d = torch.einsum('jv,btvc->btjc', J_regressor.to(target_verts.device), target_verts.clone())  # [B, T, 17, 3]
                # target_kp3d/1000 和 target_part 的差的绝对值, 平均9.31x10^(-6), 最大0.0001, 最小0.
                target_smpl = {
                    'theta': target_theta,  # [B, T, 82]
                    'verts': target_verts,  # [B, T, 6890, 3]
                    'kp_3d': target_kp3d    # [B, T, 17, 3]
                }

                for keys in rebuild_smpl[0].keys():
                    rebuild_smpl[0][keys] = rebuild_smpl[0][keys].detach().cpu().numpy()
                    target_smpl[keys] = target_smpl[keys].detach().cpu().numpy()
                results['kp_3d'].append(rebuild_smpl[0]['kp_3d'])
                results['verts'].append(rebuild_smpl[0]['verts'])
                results['kp_3d_gt'].append(target_smpl['kp_3d'])
                results['verts_gt'].append(target_smpl['verts'])

            else:# args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M00_SiC_dynamicTUP', 'M03_SiC_ChnlCat', 'M04_SiC_dynamicTUP_ChnlCat', 'M05_SiC_CrossAttn', 'M06_MixSTE_v0', 'M07_MixSTE_v1', 'M08_MixSTE_v2', 'M09_MixSTE_v0_1', 'M10_MixSTE_v0_2', 'M11_MixSTE_v0_res', 'M12_MixSTE_v0_avg']:
                target = query_batch[:, args.data.clip_len:, :, :]
                rebuild = model(prompt_batch, query_batch, epoch)


            pred = rebuild[:, frame_to_eval, :, :].clone()     # (B,T,17,3)
            gt = target[:, frame_to_eval, :, :].clone()        # (B,T,17,3)



            pred = test_loader[(dataset, 'PE')].dataset.postprocess(pred, dataset_name=dataset, task='PE')
            gt = test_loader[(dataset, 'PE')].dataset.postprocess(gt, dataset_name=dataset, task='PE')



            mpjpe_ = torch.sum(torch.mean(torch.norm(pred*1000 - gt*1000, dim=3), dim=2), dim=0)
            mpjpe += mpjpe_.cpu().data.numpy()
            if if_debug:
                if idx >1:
                    break

    
    for term in results.keys():
        results[term] = np.concatenate(results[term])
    error_dict = evaluate_mesh(results)


    mpjpe = mpjpe / num_samples     # (T,)
    mpjpe_avg = np.mean(mpjpe)
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = [f'PE | {dataset}'] + ['Avg'] + [f'{i}' for i in frame_to_eval]
    summary_table.add_row(['MPJPE'] + [mpjpe_avg] + list(mpjpe))
    summary_table.float_format = ".2"

    print(f"costs {time.time()-st:.2f}s")
    return mpjpe_avg, summary_table, [f'{i}' for i in frame_to_eval], list(mpjpe), error_dict['mpve']


def evaluate_classifier(args, TEST_LOADER, model, epoch=None, if_viz=False, if_debug=False, classifier_type='task'):
    model.eval()
    num_samples = 0
    loss_value = []
    score_frag = []
    score_frag_dict = {(dataset, eval_task): [] for (dataset, eval_task) in TEST_LOADER.keys()}
    label_list = []
    label_dict = {(dataset, eval_task): [] for (dataset, eval_task) in TEST_LOADER.keys()}
    pred_list = []
    step = 0
    with torch.no_grad():
        for (dataset, eval_task) in TEST_LOADER.keys():     
            st = time.time()
            print(f'\tEvaluating [{dataset}-{eval_task}]...', end=' ')                                                     
            test_loader = TEST_LOADER[(dataset, eval_task)]
            for idx, batch in enumerate(test_loader):
                if len(batch) == 3:
                    prompt_batch, query_batch, tasks_label = batch
                elif len(batch) == 4:
                    prompt_batch, query_batch, tasks_label, _ = batch
                assert (tasks_label == args.task_to_flag[eval_task]).all()

                if classifier_type == 'task':
                    class_label = tasks_label
                elif classifier_type == 'dataset':
                    datasets_label = torch.tensor(args.dataset_to_flag[dataset]).unsqueeze(0).expand(tasks_label.size(0))
                    class_label = datasets_label
                elif classifier_type == 'task_dataset':
                    datasets_label = torch.tensor(args.dataset_to_flag[dataset]).unsqueeze(0).expand(tasks_label.size(0))
                    class_label = tasks_label * 3 + datasets_label
                
                label_list.append(class_label.data.cpu().numpy())
                label_dict[(dataset, eval_task)].append(class_label.data.cpu().numpy())
                if torch.cuda.is_available():
                    prompt_batch = prompt_batch.cuda()
                    query_batch = query_batch.cuda()
                batch_size = len(prompt_batch)
                num_samples += batch_size

                # Model forward
                output = model(Prompt=query_batch,  Query=prompt_batch, epoch=epoch)
                label = class_label.long().cuda()
                loss = nn.CrossEntropyLoss()(output, label)
                score_frag.append(output.data.cpu().numpy())
                score_frag_dict[(dataset, eval_task)].append(output.data.cpu().numpy())
                loss_value.append(loss.data.item())

                _, predict_label = torch.max(output.data, 1)
                pred_list.append(predict_label.data.cpu().numpy())
                step += 1

                if if_debug:
                    if idx >1:
                        break
            print(f"costs {time.time()-st:.2f}s")
    
    score = np.concatenate(score_frag)
    loss = np.mean(loss_value)
    accuracy = top_k(score, 1, np.concatenate(label_list))
    print('Accuracy: ', accuracy)

    for (dataset, eval_task) in TEST_LOADER.keys():
        score = np.concatenate(score_frag_dict[(dataset, eval_task)])
        label = np.concatenate(label_dict[(dataset, eval_task)])
        accuracy = top_k(score, 1, label)
        print(f'Accuracy on {(dataset, eval_task)}: ', accuracy)


    label_list = np.concatenate(label_list)
    pred_list = np.concatenate(pred_list)
    confusion = confusion_matrix(label_list, pred_list)
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = list_diag / list_raw_sum
    each_acc = np.round(each_acc, 2)
    print('Each acc', each_acc)

    return


def top_k(score, top_k, label):
    rank = score.argsort()
    hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(label)]
    return sum(hit_top_k) * 1.0 / len(hit_top_k)