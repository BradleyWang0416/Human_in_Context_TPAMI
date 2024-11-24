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

    for idx, (QUERY_INPUT, QUERY_TARGET, PROMPT_INPUT, PROMPT_TARGET, INFO) in enumerate(train_loader['non_AR']):


        if args.shuffle_batch:
            indices = torch.randperm(QUERY_INPUT.size(0))
            QUERY_INPUT = QUERY_INPUT[indices]
            QUERY_TARGET = QUERY_TARGET[indices]
            PROMPT_INPUT = PROMPT_INPUT[indices]
            PROMPT_TARGET = PROMPT_TARGET[indices]
            INFO = [INFO[i] for i in indices]



        batch_size = args.batch_size
        num_sample = QUERY_INPUT.shape[0]
        dataset_cnt = {dataset_name: 0 for dataset_name in args.dataset_task_info['train']}
        for sample_id in range(num_sample):
            dataset_name = INFO[sample_id]['dataset']
            dataset_cnt[dataset_name] += 1

        total_batch = (num_sample + batch_size - 1) // batch_size
        for batch_id in range(total_batch):
            if (batch_id+1) * batch_size > num_sample:
                query_input = QUERY_INPUT[batch_id * batch_size:]
                query_target = QUERY_TARGET[batch_id * batch_size:]
                prompt_input = PROMPT_INPUT[batch_id * batch_size:]
                prompt_target = PROMPT_TARGET[batch_id * batch_size:]
                info = INFO[batch_id * batch_size:]
            else:
                query_input = QUERY_INPUT[batch_id * batch_size:(batch_id+1)*batch_size]
                query_target = QUERY_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                prompt_input = PROMPT_INPUT[batch_id * batch_size:(batch_id+1)*batch_size]
                prompt_target = PROMPT_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                info = INFO[batch_id * batch_size:(batch_id+1)*batch_size]

            if torch.cuda.is_available():
                query_input = query_input.cuda()
                query_target = query_target.cuda()
                prompt_input = prompt_input.cuda()
                prompt_target = prompt_target.cuda()


            if if_viz:
                data_dict = {}
                dataset_exists = {  
                                    'PW3D_MESH': None,
                                    # 'H36M_MESH': None,
                                    'H36M_3D': None,
                                    'AMASS': None
                                    }
                for i in range(len(query_target)):
                    dataset_name, task = info[i]['dataset'], info[i]['task']
                    if task == 'PE':
                        if dataset_exists[dataset_name] is None:
                            dataset_exists[dataset_name] = True
                            bone_length = torch.norm(query_target[i, 0, 2,:] - query_target[i, 0, 3, :]).item()
                            data_dict[f"{dataset_name} | bone_len |{info[i]['query_index']} | 3D"] = query_target[i]
                            data_dict[f'{dataset_name} | {task} | iter{idx}-idx{i} | 2D'] = query_input[i, ..., :2]
                            # data_dict[f'{dataset_name} | {task} | iter{idx}-idx{i} | 3D'] = query_target[i] - query_target[i, ..., [0], :]
                            # data_dict[f'{dataset_name} | {task} | iter{idx}-idx{i} | 2D'] = query_input[i, ..., :2] - query_input[i, ..., [0], :2]
                    if all(dataset_exists.values()):
                        data_dict = dict(sorted(data_dict.items(), key=lambda x: x[0]))
                        viz_skel_seq_anim(data=data_dict, subplot_layout=(2,4), fs=0.5, if_node=True, azim=-90, elev=90, 
                                          if_print=0, file_name=f'bs{args.batch_size}_{idx}_{batch_id}_{i}', file_folder=f'tmp/tmp_viz_aug')
                        print(f"Viz saved at tmp/tmp_viz_aug/bs{args.batch_size}_{idx}_{batch_id}_{i}")
                        dataset_exists = {
                                        'PW3D_MESH': None, 
                                          'H36M_3D': None, 
                                          'AMASS': None
                                        }
                        data_dict = {}
                continue
            
        

            prompt_batch = torch.cat([prompt_input, prompt_target], dim=-3)    # (N,T,17,3)
            query_batch = torch.cat([query_input, query_target], dim=-3)       # (N,T,17,3)



            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M01_SiC_staticTUP':
                avg_pose = torch.from_numpy(np.load(os.path.join(args.data.root_path, 'support_data', 'avg_pose.npy'))).float().cuda()
                avg_pose = avg_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
                query_input = query_batch[:, :args.data.clip_len]
                query_target = query_batch[:, args.data.clip_len:]
                pseudo_query_batch = torch.cat([query_input, avg_pose], dim=1)
                rebuild_part = model_pos(prompt_batch, pseudo_query_batch, query_target, epoch)    # (N,T,17,3)
                target_part = query_target.clone()
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M06_MixSTE_v0_maxlen_mask_recon', 'M06_MixSTE_v0_maxlen_mask_recon_v2']:
                rebuild_part, recon_out = model_pos(prompt_batch, query_batch, epoch)
                target_part = query_target.clone()
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M06_MixSTE_v0_maxlen_sequential', 'M06_MixSTE_v0_maxlen_sequential_mask']:
                rebuild_part = model_pos(prompt_batch, query_batch, epoch)    # (N,T,17,3)
                target_part = torch.cat([prompt_batch, query_batch], dim=-3).clone()
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M06_MixSTE_v0_maxlen_sequential_mask_v2']:
                rebuild_part = model_pos(prompt_batch, query_batch, epoch)    # (N,T,17,3)
                target_part = torch.cat([prompt_batch, query_batch], dim=-3).repeat(1, 1, 4, 1).clone()
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M17_MixSTE_v0_maxlen_Mesh']:
                rebuild_part, rebuild_smpl = model_pos(prompt_batch, query_batch, epoch)    # (N,T,17,3)
                target_part = torch.cat([prompt_batch, query_batch], dim=-3).repeat(1, 1, 4, 1).clone()
            elif args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M18_MixSTE_v0_maxlen_ClassifyTask']:
                rebuild_part, rebuild_context = model_pos(prompt_batch, query_batch, epoch)    # (N,T,17,3)
                target_part = query_target.clone()
                target_context = torch.tensor([args.task_to_flag(info[i]['task']) for i in range(len(info))])
            else:# args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M00_SiC_dynamicTUP', 'M03_SiC_ChnlCat', 'M04_SiC_dynamicTUP_ChnlCat', 'M05_SiC_CrossAttn', 'M06_MixSTE_v0', 'M07_MixSTE_v1', 'M08_MixSTE_v2', 'M09_MixSTE_v0_1', 'M10_MixSTE_v0_2', 'M11_MixSTE_v0_res', 'M12_MixSTE_v0_avg']:
                rebuild_part = model_pos(prompt_batch, query_batch, epoch)    # (N,T,17,3)
                target_part = query_target.clone()

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
                    if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') in ['M06_MixSTE_v0_maxlen_mask_recon', 'M06_MixSTE_v0_maxlen_mask_recon_v2']:
                        loss += loss_dict['loss_function'](recon_out, query_input)
                loss_dict['loss_logger'].update(loss.item(), batch_size)
                weight = loss_dict['loss_weight']
                if weight != 0:
                    loss_total += loss * weight
            losses['total']['loss_logger'].update(loss_total.item(), batch_size)
            # with torch.autograd.detect_anomaly():
            loss_total.backward()
            optimizer.step()


            if args.get('reverse_query_prompt_per_iter', False):
                rebuild_prompt = model_pos(query_batch, prompt_batch, epoch)
                optimizer.zero_grad()
                loss_total = 0
                for loss_name, loss_dict in losses.items():
                    if loss_name == 'total':
                        continue
                    if loss_name == 'limb_var':
                        loss = loss_dict['loss_function'](rebuild_prompt)
                    else:
                        loss = loss_dict['loss_function'](rebuild_prompt, prompt_target)
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
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples")
            if idx > 1: break


    if not args.reverse_query_prompt:
        return
    

    for idx, (PROMPT_INPUT, PROMPT_TARGET, QUERY_INPUT, QUERY_TARGET, INFO) in enumerate(train_loader['non_AR']):

        batch_size = args.batch_size
        num_sample = QUERY_INPUT.shape[0]
        dataset_cnt = {dataset_name: 0 for dataset_name in args.dataset_task_info['train']}
        for sample_id in range(num_sample):
            dataset_name = INFO[sample_id]['dataset']
            dataset_cnt[dataset_name] += 1

        total_batch = (num_sample + batch_size - 1) // batch_size
        for batch_id in range(total_batch):
            if (batch_id+1) * batch_size > num_sample:
                query_input = QUERY_INPUT[batch_id * batch_size:]
                query_target = QUERY_TARGET[batch_id * batch_size:]
                prompt_input = PROMPT_INPUT[batch_id * batch_size:]
                prompt_target = PROMPT_TARGET[batch_id * batch_size:]
                info = INFO[batch_id * batch_size:]
            else:
                query_input = QUERY_INPUT[batch_id * batch_size:(batch_id+1)*batch_size]
                query_target = QUERY_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                prompt_input = PROMPT_INPUT[batch_id * batch_size:(batch_id+1)*batch_size]
                prompt_target = PROMPT_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                info = INFO[batch_id * batch_size:(batch_id+1)*batch_size]

            if torch.cuda.is_available():
                query_input = query_input.cuda()
                query_target = query_target.cuda()
                prompt_input = prompt_input.cuda()
                prompt_target = prompt_target.cuda()


            prompt_batch = torch.cat([prompt_input, prompt_target], dim=-3)    # (N,T,17,3)
            query_batch = torch.cat([query_input, query_target], dim=-3)       # (N,T,17,3)

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
       

            # Optimize
            optimizer.zero_grad()
            loss_total = 0
            for loss_name, loss_dict in losses.items():
                if loss_name == 'total':
                    continue
                if loss_name == 'limb_var':
                    loss = loss_dict['loss_function'](rebuild_part)
                else:
                    loss = loss_dict['loss_function'](rebuild_part, query_target)
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
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples")
            if idx > 1: break



def train_classifier_epoch(args, model_pos, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False, classifier_type='task'):
    from lib.utils.learning import AverageMeter

    # torch.autograd.set_detect_anomaly(True)

    model_pos.train()
    st = time()

    acc_meter = AverageMeter()

    for idx, (QUERY_INPUT, QUERY_TARGET, PROMPT_INPUT, PROMPT_TARGET, INFO) in enumerate(train_loader['non_AR']):

        batch_size = args.batch_size
        num_sample = QUERY_INPUT.shape[0]
        dataset_cnt = {dataset_name: 0 for dataset_name in args.dataset_task_info['train']}
        for sample_id in range(num_sample):
            dataset_name = INFO[sample_id]['dataset']
            dataset_cnt[dataset_name] += 1

        total_batch = (num_sample + batch_size - 1) // batch_size
        for batch_id in range(total_batch):
            if (batch_id+1) * batch_size > num_sample:
                query_input = QUERY_INPUT[batch_id * batch_size:]
                query_target = QUERY_TARGET[batch_id * batch_size:]
                prompt_input = PROMPT_INPUT[batch_id * batch_size:]
                prompt_target = PROMPT_TARGET[batch_id * batch_size:]
                info = INFO[batch_id * batch_size:]
            else:
                query_input = QUERY_INPUT[batch_id * batch_size:(batch_id+1)*batch_size]
                query_target = QUERY_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                prompt_input = PROMPT_INPUT[batch_id * batch_size:(batch_id+1)*batch_size]
                prompt_target = PROMPT_TARGET[batch_id * batch_size:(batch_id+1)*batch_size]
                info = INFO[batch_id * batch_size:(batch_id+1)*batch_size]

            if torch.cuda.is_available():
                query_input = query_input.cuda()
                query_target = query_target.cuda()
                prompt_input = prompt_input.cuda()
                prompt_target = prompt_target.cuda()

            prompt_batch = torch.cat([prompt_input, prompt_target], dim=-3)    # (N,T,17,3)
            query_batch = torch.cat([query_input, query_target], dim=-3)       # (N,T,17,3)

            # Model forward
            rebuild_part = model_pos(Prompt=query_batch, Query=prompt_batch, epoch=epoch)    # (N,T,17,3)
            if classifier_type == 'task':
                class_label = torch.tensor([args.task_to_flag[info[i]['task']] for i in range(len(info))]).long().cuda()
            elif classifier_type == 'dataset':
                class_label = torch.tensor([args.dataset_to_flag[info[i]['dataset']] for i in range(len(info))]).long().cuda()
            elif classifier_type == 'task_dataset':
                task_label = torch.tensor([args.task_to_flag[info[i]['task']] for i in range(len(info))]).long().cuda()
                dataset_label = torch.tensor([args.dataset_to_flag[info[i]['dataset']] for i in range(len(info))]).long().cuda()
                class_label = task_label * 3 + dataset_label


            # Optimize
            optimizer.zero_grad()
            loss_total = 0
            for loss_name, loss_dict in losses.items():
                if loss_name == 'total':
                    continue
                elif loss_name == 'mpjpe':
                    loss = loss_dict['loss_function'](rebuild_part, class_label)
                else:
                    raise ValueError(f"Training classifier cannot have: {loss_name}")
                loss_dict['loss_logger'].update(loss.item(), batch_size)
                weight = loss_dict['loss_weight']
                if weight != 0:
                    loss_total += loss * weight
            losses['total']['loss_logger'].update(loss_total.item(), batch_size)


            # with torch.autograd.detect_anomaly():
            loss_total.backward()

            optimizer.step()


            value, predict_label = torch.max(rebuild_part.data, 1)
            acc = torch.mean((predict_label == class_label.data).float())
            acc_meter.update(acc.data.item(), batch_size)



        if idx in [len(train_loader['non_AR']) * i // 3 for i in range(1, 3)]:
            # task_cnt = {task: 0 for task in args.tasks}
            # for i in range(batch_size):
            #     task_cnt[args.flag_to_task[f'{task_flag[i]}']] += 1
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples")
        

        if if_debug:
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {dataset_cnt} samples")
            if idx > 1: break
    
    print(f'\nAverage training accuracy: {acc_meter.avg*100}')
