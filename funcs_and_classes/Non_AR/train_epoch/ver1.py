import os
import torch
import numpy as np
from time import time


def train_epoch(args, model_pos, train_loader, losses, optimizer, epoch=None, if_viz=False, if_debug=False):
    model_pos.train()
    st = time()
    for idx, (query_input, query_target, task_flag) in enumerate(train_loader['non_AR']):
        if torch.cuda.is_available():
            query_input = query_input.cuda()
            query_target = query_target.cuda()
        batch_size = len(query_input)

        # Model forward
        if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M02_SiC_query_only':
            query_pred = model_pos(query_input, epoch)    # (N,T,17,3)
        else:
            raise ValueError('Undefined backbone type.')

        # Optimize
        optimizer.zero_grad()
        loss_total = 0
        for loss_name, loss_dict in losses.items():
            if loss_name == 'total':
                continue
            if loss_name == 'limb_var':
                loss = loss_dict['loss_function'](query_pred)
            else:
                loss = loss_dict['loss_function'](query_pred, query_target)
            loss_dict['loss_logger'].update(loss.item(), batch_size)
            weight = loss_dict['loss_weight']
            if weight != 0:
                loss_total += loss * weight
        losses['total']['loss_logger'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()

        if idx in [len(train_loader['non_AR']) * i // 3 for i in range(1, 3)]:
            task_cnt = {task: 0 for task in args.tasks}
            for i in range(batch_size):
                task_cnt[args.flag_to_task[f'{task_flag[i]}']] += 1
            print(f"\tIter: {idx}/{len(train_loader['non_AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {task_cnt} samples")

        if if_debug:
            if idx > 2: break