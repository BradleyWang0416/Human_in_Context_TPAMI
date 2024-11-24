import os
import sys
import numpy as np
from time import time
import prettytable
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path.append(os.path.join(ROOT_DIR, 'lib'))

from lib.utils.utils_AR import get_targets
from lib.utils.viz_skel_seq import viz_skel_seq_anim


def evaluate_action_recognition(args, test_loader, model, epoch=None, if_viz=False):
    print('\tEvaluating Action Recognition...')
    model.eval()
    num_samples = 0

    num_correct = {'Protocol #1': 0, 'Protocol #2': 0, 'Protocol #3': 0, 'Protocol #4': 0}
    num_correct_per_class = {'Protocol #1': np.zeros(60, dtype=int), 'Protocol #2': np.zeros(60, dtype=int), 'Protocol #3': np.zeros(60, dtype=int), 'Protocol #4': np.zeros(60, dtype=int)}

    st = time()
    with torch.no_grad():
        for idx, (query_input, query_label, prompt_input, prompt_label) in enumerate(test_loader):
            # query_input: (B, 3, 60, 25, 2)
            # query_label: (B)
            # prompt_input: (B, 3, 60, 25, 2)
            # prompt_label: (B)
            if torch.cuda.is_available():
                query_input = query_input.float().cuda()
                prompt_input = prompt_input.float().cuda()
                query_label = query_label.cuda()
                prompt_label = prompt_label.cuda()

            query_input = query_input.permute(0, 4, 2, 3, 1)    # (B, 2, 60, 25, 3)
            prompt_input = prompt_input.permute(0, 4, 2, 3, 1)  # (B, 2, 60, 25, 3)
            B, M, T, J, C = query_input.shape
            num_samples += B
            assert T == args.train_feeder_args_ntu.window_size
            L = args.data.clip_len * 2 - T

            query_target = torch.stack([get_targets()[query_label[b]] for b in range(B)])     # (B,3)
            prompt_target = torch.stack([get_targets()[prompt_label[b]] for b in range(B)])   # (B,3)
            
            query_target = query_target.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, M, L, J, -1)      # (B, 2, 4, 25, 3)
            prompt_target = prompt_target.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, M, L, J, -1)   # (B, 2, 4, 25, 3)

            QUERY = torch.cat([query_input, query_target], dim=2)       # (B, 2, 64, 25, 3)
            PROMPT = torch.cat([prompt_input, prompt_target], dim=2)    # (B, 2, 64, 25, 3)

            if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') ==  'M00_SiC_dynamicTUP':
                pred_target = model(PROMPT, QUERY, epoch)    # (B, 4, 25, 3)
                pred_target = pred_target.reshape(B, M, L, J, C)    # (B, 2, 4, 25, 3)
            else:
                NotImplemented

            if if_viz:
                for b in range(B):
                    qi_1 = query_input[b,0,:,:17,:]     # (64,25,3)
                    qi_2 = query_input[b,1,:,:17,:]     # (64,25,3)
                    q_class = query_label[b].data.cpu().numpy()    # int    
                    q_target = query_target[b,0,:,:17,:]  # (64,25,3)
                    q_target = q_target + torch.randn_like(q_target)*0.02
                    q_pred = pred_target[b,:,:17,:]  # (64,25,3)
                    q_pred = q_pred + torch.randn_like(q_pred)*0.005
                    p_target = prompt_target[b,0,:,:17,:]  # (64,25,3)
                    p_target = p_target + torch.randn_like(p_target)*0.02
                    viz_dict = {
                        'pred': q_pred,
                        '1st person': qi_1,
                        '2nd person': qi_2,
                        'prompt target': p_target,
                        'target': q_target,
                    }
                    viz_skel_seq_anim(viz_dict, if_print=1, fig_title=f'{idx}iter_{b}sample_{q_class}class', lim3d=1, if_node=1, file_name=f'iter{idx:03d}_sample{b:02d}_class{q_class:02d}', file_folder='tmp')
                
            targets = get_targets()     # (60, 3)

            # Scheme 1
            true_labels = targets.clone().unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)        # (1, 1, 1, 1, 60, 3)
            pred_label = pred_target.clone().unsqueeze(4)                               # (B, 2, 4, 25, 1, 3)
            distances = torch.norm(pred_label - true_labels, dim=-1)            # (B, 2, 4, 25, 60)
            pred_class = distances.argmin(-1)   # (B, 2, 4, 25)
            
            true_class = query_label.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(pred_class)  # (B, 2, 4, 25)
            correct_ratio = (pred_class == true_class).type(torch.int).sum(-1).sum(-1).sum(-1) / (M * L * J)   # (B)

            correct_indices = torch.where(correct_ratio >= 0.5)[0]
            n_correct = len(correct_indices)
            num_correct['Protocol #1'] += n_correct
            for correct_idx in correct_indices:
                correct_class = query_label[correct_idx].data.cpu().numpy()
                num_correct_per_class['Protocol #1'][correct_class] += 1

            # # Scheme 2
            # true_labels = targets.clone().unsqueeze(0).unsqueeze(0)                     # (1, 1, 60, 3)
            # pred_label = pred_target.clone().mean(1).unsqueeze(2)                       # (B, 25, 1, 3)
            # distances = torch.norm(pred_label - true_labels, dim=-1)            # (B, 25, 60)
            # pred_class = distances.argmin(-1)   # (B, 25)

            # true_class = query_label.unsqueeze(-1).expand_as(pred_class)  # (B, 25)
            # correct_ratio = (pred_class == true_class).type(torch.int).sum(-1) / J   # (B)

            # correct_indices = torch.where(correct_ratio >= 0.5)[0]
            # n_correct = len(correct_indices)
            # num_correct['Protocol #2'] += n_correct
            # for correct_idx in correct_indices:
            #     correct_class = query_label[correct_idx].data.cpu().numpy()
            #     num_correct_per_class['Protocol #2'][correct_class] += 1

            # # Scheme 3
            # true_labels = targets.clone().unsqueeze(0).unsqueeze(0)                     # (1, 1, 60, 3)
            # pred_label = pred_target.clone().mean(2).unsqueeze(2)                       # (B, 4, 1, 3)
            # distances = torch.norm(pred_label - true_labels, dim=-1)            # (B, 4, 60)
            # pred_class = distances.argmin(-1)   # (B, 4)

            # true_class = query_label.unsqueeze(-1).expand_as(pred_class)  # (B, 4)
            # correct_ratio = (pred_class == true_class).type(torch.int).sum(-1) / L   # (B)
            
            # correct_indices = torch.where(correct_ratio >= 0.5)[0]
            # n_correct = len(correct_indices)
            # num_correct['Protocol #3'] += n_correct
            # for correct_idx in correct_indices:
            #     correct_class = query_label[correct_idx].data.cpu().numpy()
            #     num_correct_per_class['Protocol #3'][correct_class] += 1

            # Scheme 4
            true_labels = targets.clone().unsqueeze(0)                                  # (1, 60, 3)
            pred_label = pred_target.clone().mean(1).mean(1).mean(1).unsqueeze(1)               # (B, 1, 3)
            distances = torch.norm(pred_label - true_labels, dim=-1)            # (B, 60)
            pred_class = distances.argmin(-1)   # (B)

            true_class = query_label.clone()  # (B)
            correct_ratio = (pred_class == true_class).type(torch.int)   # (B)
            
            correct_indices = torch.where(correct_ratio >= 0.5)[0]
            n_correct = len(correct_indices)
            num_correct['Protocol #4'] += n_correct
            for correct_idx in correct_indices:
                correct_class = query_label[correct_idx].data.cpu().numpy()
                num_correct_per_class['Protocol #4'][correct_class] += 1            
            


            if idx in [len(test_loader) * i // 5 for i in range(1, 5)]:
                print(f"\tTest: {idx}/{len(test_loader)}; time cost: {(time()-st)/60 :.2f}min")

    acc = {}
    for protocol in num_correct:
        acc[protocol] = num_correct[protocol] / num_samples * 100

        print(f'Correct classifications each class ({protocol}): {num_correct_per_class[protocol]}')


    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['AR', 'top1 (#1)', 'top1 (#2)', 'top1 (#3)', 'top1 (#4)']
    summary_table.add_row(['Accuracy'] + [acc[protocol] for protocol in acc])
    summary_table.float_format = ".2"

    return max([acc[protocol] for protocol in acc]), summary_table