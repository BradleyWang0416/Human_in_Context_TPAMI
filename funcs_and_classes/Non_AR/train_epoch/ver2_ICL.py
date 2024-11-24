import os
import torch
import numpy as np
from time import time
from lib.utils.viz_skel_seq import viz_skel_seq_anim
from lib.utils.tools import read_pkl
from lib.utils.utils_non_AR import rotate_y, vector_angle, unify_skeletons

def train_epoch(args, model_pos, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False):

    # torch.autograd.set_detect_anomaly(True)

    model_pos.train()
    st = time()

    if args.get('multiple_prompts', False):
        assert hasattr(train_loader['non_AR'].dataset, 'num_prompts')
        num_prompts_min, num_prompts_max = args.get('num_prompts_lim', [4, 10])
        num_prompts_scheduler = (np.linspace(num_prompts_min**2, num_prompts_max**2, args.epochs-120)**0.5).astype(np.int)
        num_prompts = num_prompts_scheduler[epoch-120]
        train_loader['non_AR'].dataset.num_prompts = num_prompts


    for idx, (PROMPT_BATCH, QUERY_BATCH, INFO, INDEX) in enumerate(train_loader['non_AR']):

        B, N_TASKS, F, J, C = PROMPT_BATCH.shape
        PROMPT_BATCH = PROMPT_BATCH.reshape(N_TASKS, B, F, J, C)                            # (B, N_TASKS, F, J, C) --reshape--> (N_TASKS,B,F,J,C)
        QUERY_BATCH = QUERY_BATCH.reshape(N_TASKS, B, F, J, C)                              # (B, N_TASKS, F, J, C) --reshape--> (N_TASKS,B,F,J,C)
        INFO = INFO.reshape(N_TASKS, B, 2)                                                  # (B, N_TASKS, 2) --reshape--> (N_TASKS,B,2)
        INDEX = INDEX.reshape(N_TASKS, B)                                                   # (B, N_TASKS) --reshape--> (N_TASKS,B)

        dataset_cnt = {dataset: 0 for dataset in args.datasets}
        for b in range(B):
            dataset = args.flag_to_dataset[f'{INFO[0, b, 0]}']
            dataset_cnt[dataset] += 1


        for n in range(N_TASKS):

            if torch.cuda.is_available():
                prompt_batch = PROMPT_BATCH[n].cuda()       # [128, 32, 17, 3]
                query_batch = QUERY_BATCH[n].cuda()         # [128, 32, 17, 3]
                batch_info  = INFO[n]                       # [128, 2]
                index = INDEX[n]                            # [128]

            batch_size = prompt_batch.shape[0]

            target_part = query_batch[..., args.data.clip_len:, :, :]

            if if_viz:
                # AMASS: idx0-i20. H36M: idx0-i4. PW3D: idx0-i200
                data_dict = {}
                example_h36m = query_batch[4, 16:, :, :]
                example_amass = query_batch[20, 16:, :, :]
                example_3dpw = query_batch[200, 16:, :, :]
                for (dataset, example) in zip(['H36M', 'AMASS', 'PW3D'], [example_h36m, example_amass, example_3dpw]):
                    example = example.cpu().numpy()
                    angle = vector_angle(np.array([1., 0.]), example[0, 1, [0,2]] - example[0, 5, [0,2]] ) 
                    # example = rotate_y(example ,  -angle)
                    data_dict.update({dataset: example})
                viz_skel_seq_anim(data={'H36M': data_dict['H36M']}, subplot_layout=(1,1), fs=1, azim=90, elev=-90, if_node=True, node_size=80, lw=5, lim3d=0.75,
                                  if_print=1, file_name=f'example_skel_h36m_frontview', file_folder='viz_results/example_skeletons', interval=100)
                exit(0)
                data_dict = {}
                dataset_exists = []
                for i in range(batch_size):
                    dataset, task = args.flag_to_dataset[f'{batch_info[i, 0]}'], args.flag_to_task[f'{batch_info[i, 1]}']
                    # if dataset not in dataset_exists:
                    if dataset == 'PW3D' and task == 'PE':
                        dataset_exists.append(dataset)
                        query_sample = query_batch[i, 16:, :, :]
                        angle = vector_angle(np.array([1., 0.]), query_sample[0, 1, [0,2]].cpu().numpy()-query_sample[0, 5, [0,2]].cpu().numpy())
                        data_dict[f'{dataset} | angle: {angle:.2f} | {index[i]} | iter{idx}-idx{i}'] = query_sample
                        # query_sample = rotate_y(query_sample.cpu().numpy(), -angle)
                        # angle = vector_angle(np.array([1., 0.]), query_sample[0, 1, [0,2]]-query_sample[0, 5, [0,2]])
                        # data_dict[f'{dataset} | angle: {angle:.2f} removed | iter{idx}-idx{i}'] = query_sample
                    if len(dataset_exists) == 6:
                        for key in data_dict:
                            if 'AMASS' in key:
                                i_ = int(key.split('idx')[-1])
                                query_index = index[i_]
                                query_file = train_loader['non_AR'].dataset.query_list[query_index]['file_path']
                                sample = read_pkl(query_file)
                                chunk_3d = sample['chunk_3d'][16:, :, :]
                                # data_dict[f'original amass sample | iter{idx}-idx{i_}'] = chunk_3d
                                rotated_chunk_3d = rotate_y(torch.FloatTensor(chunk_3d),45)
                                # data_dict[f'original amass sample rotated | iter{idx}-idx{i_}'] = rotated_chunk_3d
                                rotated_chunk_2d = rotated_chunk_3d.clone()
                                rotated_chunk_2d[..., -1] = 0
                                # data_dict[f'original amass sample rotated 2d | iter{idx}-idx{i_}'] = rotated_chunk_2d
                                break
                        for key in data_dict:        
                            if 'H36M' in key:
                                i_ = int(key.split('idx')[-1])
                                # data_dict[f'H36M 2D | iter{idx}-idx{i_}'] = query_batch[i_, 0, :16]
                                break

                        # viz_skel_seq_anim(data=data_dict, subplot_layout=(2,3), fs=0.6, azim=-90, elev=90, if_node=True, lim3d=0.75,
                        #                   if_print=0, file_name=f'3dpw', file_folder='viz_results/example_skeletons')
                        # exit(0)
                        dataset_exists = []
                        data_dict = {}
                continue
            
            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M01_SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'support_data', 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild_part = model_pos(prompt_batch, pseudo_query_batch, query_target, epoch)    # (N,T,17,3)
            else:# args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M00_SiC_dynamicTUP', 'M03_SiC_ChnlCat', 'M04_SiC_dynamicTUP_ChnlCat', 'M05_SiC_CrossAttn', 'M06_MixSTE_v0', 'M07_MixSTE_v1', 'M08_MixSTE_v2', 'M09_MixSTE_v0_1', 'M10_MixSTE_v0_2', 'M11_MixSTE_v0_res', 'M12_MixSTE_v0_avg']:
                rebuild_part = model_pos(prompt_batch, query_batch, epoch)    # (N,T,17,3)
            # else:
            #     raise ValueError('Undefined backbone type.')

            # Optimize
            optimizer.zero_grad()
            loss_total = 0
            for loss_name, loss_dict in losses.items():
                if loss_name == 'total':
                    continue
                if loss_name == 'limb_var':
                    loss = loss_dict['loss_function'](rebuild_part)
                else:
                    loss = loss_dict['loss_function'](rebuild_part, target_part)
                loss_dict['loss_logger'].update(loss.item(), batch_size)
                weight = loss_dict['loss_weight']
                if weight != 0:
                    loss_total += loss * weight
            losses['total']['loss_logger'].update(loss_total.item(), batch_size)


            # with torch.autograd.detect_anomaly():
            loss_total.backward()

            optimizer.step()

        if idx in [len(train_loader['non_AR']) * i // 3 for i in range(1, 3)]:
            # task_cnt = {task: 0 for task in args.tasks}
            # for i in range(batch_size):
            #     task_cnt[args.flag_to_task[f'{task_flag[i]}']] += 1
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples")
        

        if if_debug:
            if idx > 2: break