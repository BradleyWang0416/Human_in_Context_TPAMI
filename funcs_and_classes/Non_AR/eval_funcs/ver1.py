#%%
import os
import numpy as np
import prettytable
import torch

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *


def evaluate_future_pose_estimation(args, test_loader, model, epoch=None, if_debug=False):
    print('\tEvaluating Future Pose Estimation...')
    model.eval()
    num_samples = 0
    mpjpe = np.zeros(16)

    with torch.no_grad():
        for idx, (query_input, query_target, task_flag) in enumerate(test_loader):
            assert (task_flag == args.task_to_flag['FPE']).all()
            if torch.cuda.is_available():
                query_input = query_input.cuda()
                query_target = query_target.cuda()
            batch_size = len(query_input)
            num_samples += batch_size

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M02_SiC_query_only':
                query_pred = model(query_input, epoch)    # (B,32,17,3)
            else:
                raise ValueError('Undefined backbone type.')

            pred = query_pred[:, 16:, :, :]     # (B,16,17,3)
            gt = query_target[:, 16:, :, :]     # (B,16,17,3)
            mpjpe_ = torch.sum(torch.mean(torch.norm(pred*1000 - gt*1000, dim=3), dim=2), dim=0)
            mpjpe += mpjpe_.cpu().data.numpy()
            if if_debug:
                if idx >2:
                    break
        mpjpe = mpjpe / num_samples     # (T,)
        mpjpe_avg = np.mean(mpjpe)
        summary_table = prettytable.PrettyTable()
        summary_table.field_names = ['FPE'] + ['Avg'] + [f'{(i + 1) * 20}' for i in range(16)]
        summary_table.add_row(['MPJPE'] + [mpjpe_avg] + list(mpjpe))
        summary_table.float_format = ".2"
        return mpjpe_avg, summary_table


def evaluate_motion_completion(args, test_loader, model, epoch=None, if_debug=False):
    print('\tEvaluating Motion Completion...')
    model.eval()
    mpjpe_per_ratio = {ratio: 0 for ratio in args.data.drop_ratios_MC}
    count_per_ratio = {ratio: 0 for ratio in args.data.drop_ratios_MC}
    num_samples = 0
    with torch.no_grad():
        for idx, (query_input, query_target, task_flag) in enumerate(test_loader):
            assert (task_flag == args.task_to_flag['MC']).all()
            if torch.cuda.is_available():
                query_input = query_input.cuda()
                query_target = query_target.cuda()
            batch_size = len(query_input)
            num_samples += batch_size

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M02_SiC_query_only':
                query_pred = model(query_input, epoch)    # (B,32,17,3)
            else:
                raise ValueError('Undefined backbone type.')

            for i in range(batch_size):
                pred_one_sample = query_pred[i]        # (32, 17, 3)
                gt_one_sample = query_target[i]           # (32, 17, 3)
                masked_frame_idx = torch.all(query_input[i,:,1:].sum(dim=(0,2), keepdim=True) == 0, dim=0).squeeze(-1)
                masked_frame_idx = torch.cat([torch.tensor([False]).cuda(), masked_frame_idx])
                pred_ = pred_one_sample[:, masked_frame_idx]
                gt_ = gt_one_sample[:, masked_frame_idx]
                masked_frame_num = pred_.shape[1]
                assert masked_frame_num in [int(args.data.num_joints * ratio) for ratio in args.data.drop_ratios_MC]
                mpjpe_ = torch.mean(torch.norm(pred_*1000 - gt_*1000, dim=2))

                for ratio in count_per_ratio:
                    if masked_frame_num == int(ratio * args.data.num_joints):
                        count_per_ratio[ratio] += 1
                        mpjpe_per_ratio[ratio] += mpjpe_.cpu().data.numpy()
            
            if if_debug:
                if idx >2:
                    break

        assert sum([cnt for ratio, cnt in count_per_ratio.items()]) == num_samples
        for ratio in count_per_ratio:
            num_samples = count_per_ratio[ratio]
            mpjpe_per_ratio[ratio] = mpjpe_per_ratio[ratio] / num_samples
        mpjpe_avg = np.mean(np.array([err for ratio, err in mpjpe_per_ratio.items()]))

        summary_table = prettytable.PrettyTable()
        summary_table.field_names = ['MC'] + ['Avg'] + [ratio for ratio in args.data.drop_ratios_MC]
        summary_table.add_row(['MPJPE'] + [mpjpe_avg] + [err for ratio, err in mpjpe_per_ratio.items()])
        summary_table.float_format = ".2"
        return mpjpe_avg, summary_table


def evaluate_motion_prediction(args, test_loader, model, epoch=None, if_debug=False):
    print('\tEvaluating Motion Prediction...')
    model.eval()
    num_samples = 0
    mpjpe = np.zeros(16)

    with torch.no_grad():
        for idx, (query_input, query_target, task_flag) in enumerate(test_loader):
            assert (task_flag == args.task_to_flag['MP']).all()
            if torch.cuda.is_available():
                query_input = query_input.cuda()
                query_target = query_target.cuda()
            batch_size = len(query_input)
            num_samples += batch_size

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M02_SiC_query_only':
                query_pred = model(query_input, epoch)    # (B,32,17,3)
            else:
                raise ValueError('Undefined backbone type.')

            pred = query_pred[:, 16:, :, :].clone()     # (B,16,17,3)
            gt = query_target[:, 16:, :, :].clone()        # (B,16,17,3)

            mpjpe_ = torch.sum(torch.mean(torch.norm(pred*1000 - gt*1000, dim=3), dim=2), dim=0)
            mpjpe += mpjpe_.cpu().data.numpy()

            if if_debug:
                if idx >2:
                    break

        mpjpe = mpjpe / num_samples     # (T,)
        mpjpe_avg = np.mean(mpjpe)
        summary_table = prettytable.PrettyTable()
        summary_table.field_names = ['MP'] + ['Avg'] + [f'{(i + 1) * 40}' for i in range(16)]
        summary_table.add_row(['MPJPE'] + [mpjpe_avg] + list(mpjpe))
        summary_table.float_format = ".2"
        return mpjpe_avg, summary_table


def evaluate_pose_estimation(args, model, test_loader, datareader, epoch=None, if_debug=False):
    print('\tEvaluating 3D Pose Estimation...')
    results_all = []
    model.eval()

    with torch.no_grad():
        for idx, (query_input, query_target, task_flag) in enumerate(test_loader):
            assert (task_flag == args.task_to_flag['PE']).all()
            if torch.cuda.is_available():
                query_input = query_input.cuda()
                query_target = query_target.cuda()

            if if_debug:
                results_all.append(query_target.cpu().numpy())
                continue

            # Model forward
            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') == 'M02_SiC_query_only':
                query_pred = model(query_input, epoch)    # (B,32,17,3)
            else:
                raise ValueError('Undefined backbone type.')

            results_all.append(query_pred.cpu().numpy())

    results_all = np.concatenate(results_all)
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
    gt_clips = gts[split_id_test]
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
    summary_table.field_names = ['PE'] + ['Avg'] + action_names
    summary_table.add_row(['MPJPE'] + [e1] + final_result)
    summary_table.add_row(['P-MPJPE'] + [e2] + final_result_procrustes)
    summary_table.float_format = ".2"

    return e1, e2, summary_table

