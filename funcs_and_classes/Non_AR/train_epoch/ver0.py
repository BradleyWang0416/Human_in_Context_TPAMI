import os
import torch
import numpy as np
from time import time
from lib.utils.viz_skel_seq import viz_skel_seq_anim


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


    for idx, (PROMPT_BATCH, QUERY_BATCH, TASK_FLAG) in enumerate(train_loader['non_AR']):

        if hasattr(train_loader['non_AR'].dataset, 'num_prompts') and train_loader['non_AR'].dataset.num_prompts >= 2:
            assert len(PROMPT_BATCH.shape) == 5
            # assert PROMPT_BATCH.shape[0] == args.batch_size
            assert PROMPT_BATCH.shape[1] == train_loader['non_AR'].dataset.num_prompts
            B, N, F, J, C = PROMPT_BATCH.shape
            PROMPT_BATCH = PROMPT_BATCH.reshape(N, B, F, J, C)                          # (N*B,F,J,C) --reshape--> (B,N,F,J,C)
            QUERY_BATCH = QUERY_BATCH.reshape(N, B, F, J, C)                            # (N*B,F,J,C) --reshape--> (B,N,F,J,C)
            TASK_FLAG = TASK_FLAG.reshape(N, B)
        else:
            PROMPT_BATCH = PROMPT_BATCH.unsqueeze(0)
            QUERY_BATCH = QUERY_BATCH.unsqueeze(0)
            TASK_FLAG = TASK_FLAG.unsqueeze(0)
            N = 1


        for n in range(N):

            if torch.cuda.is_available():
                prompt_batch = PROMPT_BATCH[n].cuda()
                query_batch = QUERY_BATCH[n].cuda()
                task_flag = TASK_FLAG[n]
            batch_size = len(prompt_batch)

            target_part = query_batch[:, args.data.clip_len:]

            if if_viz:
                task_exists = []
                data_dict = {}
                for i in range(batch_size):
                    task = args.flag_to_task[f'{task_flag[i]}']
                    if task not in task_exists:
                        task_exists.append(task)
                        data_dict[f'{task} | iter{idx}-idx{i} qi'] = query_batch[i, :16]
                        data_dict[f'{task} | iter{idx}-idx{i} qo'] = query_batch[i, 16:]
                    if len(task_exists) == 4:
                        viz_skel_seq_anim(data=data_dict, subplot_layout=(4,2), fs=0.5)
                        task_exists = []
                        data_dict = {}
                            
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
                task_cnt = {task: 0 for task in args.tasks}
                for i in range(batch_size):
                    task_cnt[args.flag_to_task[f'{task_flag[i]}']] += 1
                print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {task_cnt} samples")
        
        if if_debug:
            if idx > 2: break
            