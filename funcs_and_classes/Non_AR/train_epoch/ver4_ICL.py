import os
import torch
import numpy as np
from time import time
from lib.utils.viz_skel_seq import viz_skel_seq_anim
from lib.utils.tools import read_pkl
from lib.utils.utils_non_AR import rotate_y, vector_angle, unify_skeletons
from third_party.motionbert.lib.utils.utils_mesh import compute_error

def train_epoch(args, model_pos, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False):
    # torch.autograd.set_detect_anomaly(True)

    model_pos.train()
    st = time()

    use_smpl = (hasattr(args, 'Mesh') and args.Mesh.enable)
    if use_smpl: 
        assert all(args.dataset_config[dataset_name]['return_type'] in ['smpl', 'smpl_x1000', 'all', 'all_x1000'] for dataset_name in args.dataset_task_info['train'].keys())
        assert 'Mesh' in args.func_ver.get('model_name', 'M00_SiC_dynamicTUP')

    for idx, (QUERY_DICT, PROMPT_DICT, INFO) in enumerate(train_loader['non_AR']):
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
            PROMPT_VERTEX_TARGET = PROMPT_DICT['vertex_target']     # [N, T, 6890, 3]

        if args.shuffle_batch:
            raise NotImplementedError

        num_sample = QUERY_INPUT.shape[0]   # N
        batch_size = args.batch_size    # B
        
        total_batch = (num_sample + batch_size - 1) // batch_size
        for batch_id in range(total_batch):
            if (batch_id+1) * batch_size > num_sample:
                query_input = QUERY_INPUT[batch_id * batch_size:]       # [<=B, T, 17, 3]
                query_target = QUERY_TARGET[batch_id * batch_size:]     # [<=B, T, 17, 3]
                prompt_input = PROMPT_INPUT[batch_id * batch_size:]     # [<=B, T, 17, 3]
                prompt_target = PROMPT_TARGET[batch_id * batch_size:]   # [<=B, T, 17, 3]
                info = INFO[batch_id * batch_size:]                     # len<=B
                if use_smpl:
                    query_pose_target = QUERY_POSE_TARGET[batch_id * batch_size:]
                    query_shape_target = QUERY_SHAPE_TARGET[batch_id * batch_size:]
                    query_vertex_target = QUERY_VERTEX_TARGET[batch_id * batch_size:]
                    prompt_pose_target = PROMPT_POSE_TARGET[batch_id * batch_size:]
                    prompt_shape = PROMPT_SHAPE_TARGET[batch_id * batch_size:]
                    prompt_vertex_target = PROMPT_VERTEX_TARGET[batch_id * batch_size:]
            else:
                query_input = QUERY_INPUT[batch_id * batch_size:(batch_id+1)*batch_size]        # [B, T, 17, 3]
                query_target = QUERY_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]      # [B, T, 17, 3]
                prompt_input = PROMPT_INPUT[batch_id * batch_size:(batch_id+1)*batch_size]      # [B, T, 17, 3]
                prompt_target = PROMPT_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]    # [B, T, 17, 3]
                info = INFO[batch_id * batch_size:(batch_id+1)*batch_size]                      # len=B
                if use_smpl:
                    query_pose_target = QUERY_POSE_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                    query_shape_target = QUERY_SHAPE_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                    query_vertex_target = QUERY_VERTEX_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                    prompt_pose_target = PROMPT_POSE_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                    prompt_shape = PROMPT_SHAPE_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                    prompt_vertex_target = PROMPT_VERTEX_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]

            B = query_input.shape[0]    # <= batch_size

            if torch.cuda.is_available():
                query_input = query_input.cuda()        # [B, T, 17, 3]
                query_target = query_target.cuda()      # [B, T, 17, 3]
                prompt_input = prompt_input.cuda()      # [B, T, 17, 3]
                prompt_target = prompt_target.cuda()    # [B, T, 17, 3]
                if use_smpl:
                    query_pose_target = query_pose_target.cuda()        # [B, T, 72]
                    query_shape_target = query_shape_target.cuda()      # [B, T, 10]
                    query_vertex_target = query_vertex_target.cuda()    # [B, T, 6890, 3]
                    prompt_pose_target = prompt_pose_target.cuda()      # [B, T, 72]
                    prompt_shape = prompt_shape.cuda()                  # [B, T, 10]
                    prompt_vertex_target = prompt_vertex_target.cuda()  # [B, T, 6890, 3]

            prompt_batch = torch.cat([prompt_input, prompt_target], dim=-3)    # [B, F, 17, 3]. In most cases, F=2T
            query_batch = torch.cat([query_input, query_target], dim=-3)       # [B, F, 17, 3]. In most cases, F=2T

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M01_SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'support_data', 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild_part = model_pos(prompt_batch, pseudo_query_batch, query_target, epoch)    # [B, F, 17, 3]
                target_part = query_target.clone()
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M06_MixSTE_v0_maxlen_mask_recon', 'M06_MixSTE_v0_maxlen_mask_recon_v2']:
                rebuild_part, recon_out = model_pos(prompt_batch, query_batch, epoch)
                target_part = query_target.clone()
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M06_MixSTE_v0_maxlen_sequential', 'M06_MixSTE_v0_maxlen_sequential_mask']:
                rebuild_part = model_pos(prompt_batch, query_batch, epoch)    # [B, T, 17, 3]
                target_part = torch.cat([prompt_batch, query_batch], dim=-3).clone()
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M06_MixSTE_v0_maxlen_sequential_mask_v2']:
                rebuild_part = model_pos(prompt_batch, query_batch, epoch)    # [B, T, 17, 3]
                target_part = torch.cat([prompt_batch, query_batch], dim=-3).repeat(1, 1, 4, 1).clone()
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M17_MixSTE_v0_maxlen_Mesh']:
                rebuild_part, rebuild_smpl = model_pos(prompt_batch, query_batch, epoch)    
                # rebuild_part: [B, T, 17, 3]
                # rebuild_smpl: 1 element list.
                #   rebuild_smpl[0]:
                #       'theta': [B, T, 82]
                #       'verts': [B, T, 6890, 3]
                #       'kp_3d': [B, T, 17, 3]

                target_part = query_target.clone()  # [B, T, 17, 3]
                target_theta = torch.cat([query_pose_target, query_shape_target], dim=-1).clone()   # [B, T, 82]
                target_verts = query_vertex_target.clone()      # [B, T, 6890, 3]
                J_regressor = model_pos.module.mesh_head.J_regressor if model_pos.__class__.__name__ == 'DataParallel' else model_pos.mesh_head.J_regressor     # [17, 6890]
                target_kp3d = torch.einsum('jv,btvc->btjc', J_regressor.to(target_verts.device), target_verts.clone())  # [B, T, 17, 3]
                # target_kp3d/1000 和 target_part 的差的绝对值, 平均9.31x10^(-6), 最大0.0001, 最小0.
                target_smpl = {
                    'theta': target_theta,  # [B, T, 82]
                    'verts': target_verts,  # [B, T, 6890, 3]
                    'kp_3d': target_kp3d    # [B, T, 17, 3]
                }
            else:
                rebuild_part = model_pos(prompt_batch, query_batch, epoch)    # [B, T, 17, 3]
                target_part = query_target.clone()

            if use_smpl:
                loss_mesh_dict = args.Mesh.criterion(rebuild_smpl, target_smpl)     # dict_keys(['loss_3d_pos', 'loss_3d_scale', 'loss_3d_velocity', 'loss_lv', 'loss_lg', 'loss_a', 'loss_av', 'loss_shape', 'loss_pose', 'loss_norm'])
                loss_mesh_train = args.Mesh.lambda_3d      * loss_mesh_dict['loss_3d_pos']      + \
                                  args.Mesh.lambda_3dv     * loss_mesh_dict['loss_3d_velocity'] + \
                                  args.Mesh.lambda_pose    * loss_mesh_dict['loss_pose']        + \
                                  args.Mesh.lambda_shape   * loss_mesh_dict['loss_shape']       + \
                                  args.Mesh.lambda_norm    * loss_mesh_dict['loss_norm']        + \
                                  args.Mesh.lambda_scale   * loss_mesh_dict['loss_3d_scale']    + \
                                  args.Mesh.lambda_lv      * loss_mesh_dict['loss_lv']          + \
                                  args.Mesh.lambda_lg      * loss_mesh_dict['loss_lg']          + \
                                  args.Mesh.lambda_a       * loss_mesh_dict['loss_a']           + \
                                  args.Mesh.lambda_av      * loss_mesh_dict['loss_av']
                                  # default lambda: 3d: 0.5
                                  #                 3dv: 10
                                  #                 pose: 1000
                                  #                 shape: 1
                                  #                 norm: 20
                loss_str = '\t'
                for k, v in loss_mesh_dict.items():
                    args.Mesh.losses_dict[k].update(v.item(), B)
                    loss_str += '{0} {loss.val:.3f} ({loss.avg:.3f})\t'.format(k, loss=args.Mesh.losses_dict[k])
                mpjpe, mpve = compute_error(rebuild_smpl, target_smpl)  # both zero-dim, one-element tensor

                loss_total = loss_mesh_train
            else:
                loss_total = 0


            # Optimize
            optimizer.zero_grad()
            # loss_total = 0
            for loss_name, loss_dict in losses.items():
                if loss_name == 'total':
                    continue
                if loss_name == 'limb_var':
                    loss = loss_dict['loss_function'](rebuild_part)
                else:
                    loss = loss_dict['loss_function'](rebuild_part, target_part)
                    if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M06_MixSTE_v0_maxlen_mask_recon', 'M06_MixSTE_v0_maxlen_mask_recon_v2']:
                        loss += loss_dict['loss_function'](recon_out, query_input)
                loss_dict['loss_logger'].update(loss.item(), B)
                weight = loss_dict['loss_weight']
                if weight != 0:
                    loss_total += loss * weight
            losses['total']['loss_logger'].update(loss_total.item(), B)
            # with torch.autograd.detect_anomaly():
            loss_total.backward()
            optimizer.step()


            if args.get('reverse_query_prompt_per_iter', False):
                raise NotImplementedError

        dataset_cnt = {dataset_name: 0 for dataset_name in args.dataset_task_info['train']}
        for sample_id in range(num_sample):
            dataset_name = INFO[sample_id]['dataset']
            dataset_cnt[dataset_name] += 1

        if idx in [len(train_loader['non_AR']) * i // 3 for i in range(1, 3)]:
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples")
            print(loss_str)
        

        if if_debug:
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples")
            if idx > 1: break

    if not args.reverse_query_prompt:
        return
    else:
        raise NotImplementedError
    

def train_classifier_epoch(args, model_pos, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False, classifier_type='task'):
    from lib.utils.learning import AverageMeter
    raise NotImplementedError
