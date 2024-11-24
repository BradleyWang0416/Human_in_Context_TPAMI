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


def train_epoch(args, model, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False):
    # torch.autograd.set_detect_anomaly(True)

    model.train()
    st = time()

    use_smpl = True if hasattr(args, 'Mesh') and args.Mesh.enable else False
    if use_smpl:
        SMPL_MODEL = SMPL('third_party/motionbert/data/mesh', batch_size=1)
        if torch.cuda.is_available():
            SMPL_MODEL = SMPL_MODEL.cuda()

    NUM_SAMPLE = 0
    INFO_DICT_EPOCH = defaultdict(list)

    for idx, (QUERY_INPUT_DICT, QUERY_TARGET_DICT, PROMPT_INPUT_DICT, PROMPT_TARGET_DICT, INFO_DICT) in enumerate(train_loader['non_AR']):

        if args.shuffle_batch:
            raise NotImplementedError
        
        for info_key in INFO_DICT.keys():
            INFO_DICT_EPOCH[info_key].extend(INFO_DICT[info_key])


        num_sample = len(INFO_DICT['query_index'])   # N
        NUM_SAMPLE += num_sample
        defined_batch_size = args.batch_size    # B
        
        total_batch = (num_sample + defined_batch_size - 1) // defined_batch_size
        assert total_batch > 0
        for batch_id in range(total_batch):
            query_input_dict = OrderedDict()
            query_target_dict = OrderedDict()
            prompt_input_dict = OrderedDict()
            prompt_target_dict = OrderedDict()
            info_dict = {}
            for mode in QUERY_INPUT_DICT.keys():
                if (batch_id+1) * defined_batch_size > num_sample:
                    query_input_dict[mode] = QUERY_INPUT_DICT[mode][batch_id * defined_batch_size:]
                    query_target_dict[mode] = QUERY_TARGET_DICT[mode][batch_id * defined_batch_size:]
                    prompt_input_dict[mode] = PROMPT_INPUT_DICT[mode][batch_id * defined_batch_size:]
                    prompt_target_dict[mode] = PROMPT_TARGET_DICT[mode][batch_id * defined_batch_size:]
                else:
                    query_input_dict[mode] = QUERY_INPUT_DICT[mode][batch_id * defined_batch_size:(batch_id+1)*defined_batch_size]
                    query_target_dict[mode] = QUERY_TARGET_DICT[mode][batch_id * defined_batch_size:(batch_id+1)*defined_batch_size]
                    prompt_input_dict[mode] = PROMPT_INPUT_DICT[mode][batch_id * defined_batch_size:(batch_id+1)*defined_batch_size]
                    prompt_target_dict[mode] = PROMPT_TARGET_DICT[mode][batch_id * defined_batch_size:(batch_id+1)*defined_batch_size]
                
            for info_key in INFO_DICT.keys():
                if (batch_id+1) * defined_batch_size > num_sample:
                    info_dict[info_key] = INFO_DICT[info_key][batch_id * defined_batch_size:]
                else:
                    info_dict[info_key] = INFO_DICT[info_key][batch_id * defined_batch_size:(batch_id+1)*defined_batch_size]

            batch_size = len(info_dict['query_index'])    # <= batch_size

            if torch.cuda.is_available():
                query_input_dict = {k: v.cuda() for k, v in query_input_dict.items()}
                query_target_dict = {k: v.cuda() for k, v in query_target_dict.items()}
                prompt_input_dict = {k: v.cuda() for k, v in prompt_input_dict.items()}
                prompt_target_dict = {k: v.cuda() for k, v in prompt_target_dict.items()}
            
            if use_smpl:
                query_target_vertex = compute_smpl_vertex(args, query_target_dict, SMPL_MODEL, info_dict)
                prompt_target_vertex = compute_smpl_vertex(args, prompt_target_dict, SMPL_MODEL, info_dict)
                query_target_dict['smpl_vertex'] = query_target_vertex
                prompt_target_dict['smpl_vertex'] = prompt_target_vertex

            query_input_dict = train_loader['non_AR'].dataset.preprocess(query_input_dict)
            prompt_input_dict = train_loader['non_AR'].dataset.preprocess(prompt_input_dict)
            query_target_dict = train_loader['non_AR'].dataset.preprocess(query_target_dict)
            prompt_target_dict = train_loader['non_AR'].dataset.preprocess(prompt_target_dict)



            if if_viz.split(',')[0] == inspect.currentframe().f_code.co_name:
                print('Do visualizing now in train_epoch...')
                dataset_check = OrderedDict({'AMASS': False, 'H36M_MESH_TCMR': False, 'PW3D_MESH': False})
                data_to_viz = OrderedDict({
                                           'AMASS | prompt input': None, 
                                           'AMASS | prompt target': None, 
                                           'H36M_MESH_TCMR | prompt input': None, 
                                           'H36M_MESH_TCMR | prompt target': None, 
                                           'PW3D_MESH | prompt input': None,
                                           'PW3D_MESH | prompt target': None,

                                           'AMASS | query input': None, 
                                           'AMASS | query target': None, 
                                           'H36M_MESH_TCMR | query input': None, 
                                           'H36M_MESH_TCMR | query target': None, 
                                           'PW3D_MESH | query input': None,
                                           'PW3D_MESH | query target': None,

                                           'AMASS | filler': None, 
                                           'AMASS | query target vertex': None, 
                                           'H36M_MESH_TCMR | filler': None, 
                                           'H36M_MESH_TCMR | query target vertex': None, 
                                           'PW3D_MESH | filler': None,
                                           'PW3D_MESH | query target vertex': None,
                                           })
                for i in range(0, batch_size):
                    dataset_name = info_dict['dataset'][i]
                    task = info_dict['task'][i]

                    pi_joint = prompt_input_dict['joint'][i, ..., :2].cpu().numpy()    if task in ['PE', 'FPE'] else prompt_input_dict['joint'][i].cpu().numpy()
                    pt_joint = prompt_target_dict['joint'][i].cpu().numpy()
                    qi_joint = query_input_dict['joint'][i, ..., :2].cpu().numpy()    if task in ['PE', 'FPE'] else query_input_dict['joint'][i].cpu().numpy()
                    qt_joint = query_target_dict['joint'][i].cpu().numpy()
                    qt_vertex = query_target_dict['smpl_vertex'][i].cpu().numpy()

                    # chunk_dict_all = train_loader['non_AR'].dataset.prepare_chunk(
                    #     train_loader['non_AR'].dataset.query_dict,
                    #     dataset_name,
                    # )
                    # smpl_pose_all, smpl_shape_all = chunk_dict_all['smpl_pose'], chunk_dict_all['smpl_shape']
                    # smpl_pose_mean = smpl_pose_all.reshape(-1, 72).mean(0, keepdim=True).cuda()
                    # smpl_shape_mean = smpl_shape_all.reshape(-1, 10).mean(0, keepdim=True).cuda()
                    # motion_mean = SMPL_MODEL(
                    #     betas=smpl_shape_mean,
                    #     body_pose=smpl_pose_mean[:, 3:],
                    #     global_orient=smpl_pose_mean[:, :3],
                    #     pose2rot=True
                    # )
                    # init_mesh = motion_mean.vertices.detach().reshape(1, -1, 3).expand(16, -1, -1)
                    
                    init_mesh = model.module.smpl_head.get_mesh_from_init_pose_shape_only(
                        # init_pose=torch.zeros(144), 
                        init_shape=torch.zeros(10),
                    )[0]['verts'].expand(qt_vertex.shape[0],-1,-1) / 1000
                    
                    data_to_viz[f'{dataset_name} | prompt input'] = pi_joint
                    data_to_viz[f'{dataset_name} | prompt target'] = pt_joint
                    data_to_viz[f'{dataset_name} | query input'] = qi_joint
                    data_to_viz[f'{dataset_name} | query target'] = qt_joint
                    data_to_viz[f'{dataset_name} | filler'] = init_mesh
                    query_target_dict_masked = copy.deepcopy(query_target_dict)
                    query_target_dict_masked['smpl_pose'][..., 3:] = query_target_dict_masked['smpl_pose'][..., 3:] * (torch.randn(batch_size, 1, 23, 1) > 0.6).expand(-1,16,-1,3).reshape(-1,16,69).cuda()
                    data_to_viz[f'{dataset_name} | filler'] = compute_smpl_vertex(args, query_target_dict_masked, SMPL_MODEL, info_dict)[i].cpu().numpy()
                    data_to_viz[f'{dataset_name} | query target vertex'] = qt_vertex

                    dataset_check[dataset_name] = True
                    if all(dataset_check.values()):
                        viz_skel_seq_anim(data_to_viz, subplot_layout=(3,6), fs=0.3, if_node=True, lim3d=1, tight_layout=True,
                                          azim=-110, elev=40,
                                          fig_title=f' ')
                        
                        dataset_check = OrderedDict({'AMASS': False, 'H36M_MESH_TCMR': False, 'PW3D_MESH': False})


            output_dict = model(query_input_dict, prompt_input_dict,
                                query_target_dict, prompt_target_dict, info_dict, epoch, vertex_x1000=args.vertex_x1000, deactivate_prompt_branch=args.deactivate_prompt_branch)
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
                output_joint, output_smpl, target_joint, target_smpl = output_dict[k]
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
