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


def calculate_acc(pred):
    # pred: (B,M,L,25,3) or (B,L,25,3) or (B,M,25,3) or (B,M,L,3) or (B,25,3) or (B,L,3) or (B,M,3) or (B,3)
    # label: (B)
    pred_label = pred.unsqueeze(-2)     # pred_label: (B,M,L,25,1,3) or (B,L,25,1,3) or (B,M,25,1,3) or (B,M,L,1,3) or (B,25,1,3) or (B,L,1,3) or (B,M,1,3) or (B,1,3)
    labels = get_targets()  # (60,3)
    while len(labels.shape) != len(pred_label.shape):
        labels = labels.unsqueeze(0)
    distances = torch.norm(pred_label - labels, dim=-1)     # (B,M,L,25,60) or (B,L,25,60) or (B,M,25,60) or (B,M,L,60) or (B,25,60) or (B,L,60) or (B,M,60) or (B,60)
    return distances


def eval_batch(args, query_input, query_label, prompt_input, prompt_label, model, epoch):
    B, M, T, J, C = query_input.shape
    assert T == args.train_feeder_args_ntu.window_size
    L = args.data.clip_len * 2 - T

    query_label = torch.stack([get_targets()[query_label[b]] for b in range(B)])     # (B,3)
    prompt_label = torch.stack([get_targets()[prompt_label[b]] for b in range(B)])   # (B,3)
    query_target = query_label.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, M, L, J, -1)      # (B, 1, L, 25, 3) or (B, 2, L, 25, 3)
    prompt_target = prompt_label.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, M, L, J, -1)   # (B, 1, L, 25, 3) or (B, 2, L, 25, 3)

    QUERY = torch.cat([query_input, query_target], dim=2)       # (B, 1, F, 25, 3) or (B, 2, F, 25, 3)
    PROMPT = torch.cat([prompt_input, prompt_target], dim=2)    # (B, 1, F, 25, 3) or (B, 2, F, 25, 3)
    if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') ==  'M00_SiC_dynamicTUP':
        output = model(PROMPT, QUERY, epoch)                # [CASE 1] (B, 1, L, 25, 3), if: (1) M=1, or (2) M=2 & merge_double_action is one of ['before_pre_logits', 'before_head', 'after_output']
                                                            # [CASE 2] (B, 2, L, 25, 3), if: M=2 & merge_double_action is 'no'
    else:
        NotImplemented
    return output


def evaluate_action_recognition(args, test_loader, model, epoch=None, if_viz=False):
    print('\tEvaluating Action Recognition...')
    model.eval()
    num_samples = 0

    num_correct = np.zeros((8, 2, 8, 2))

    st = time()
    with torch.no_grad():
        for idx, (QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1,
                  QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2,
                  subdata_num_list_1, subdata_num_list_2) in enumerate(test_loader):
            # tensor: (N1,1,T,J,C); tensor: (N1); tensor: (N1,1,T,J,C); tensor: (N1); 
            # tensor: (N2,2,T,J,C); tensor: (N2); tensor: (N2,2,T,J,C); tensor: (N2);
            # list: B1x<int>; list: B2x<int>
            B1 = len(subdata_num_list_1)
            B2 = len(subdata_num_list_2)
            batch_size = B1 + B2
            num_samples += batch_size

            if B1 != 0:
                subdata_indices_1 = np.array([0] + subdata_num_list_1).cumsum()     # len = B1 + 1
                if torch.cuda.is_available():
                    QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1 = QUERY_INPUT_1.cuda(), QUERY_LABEL_1.cuda(), PROMPT_INPUT_1.cuda(), PROMPT_LABEL_1.cuda()
                OUTPUT1 = eval_batch(args, QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1, model, epoch)  # (N1,1,L,25,3)


            if B2 != 0:
                subdata_indices_2 = np.array([0] + subdata_num_list_2).cumsum()     # len = B1 + 1
                if torch.cuda.is_available():
                    QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2 = QUERY_INPUT_2.cuda(), QUERY_LABEL_2.cuda(), PROMPT_INPUT_2.cuda(), PROMPT_LABEL_2.cuda()
                OUTPUT2 = eval_batch(args, QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2, model, epoch)  # [CASE 1] (N2,1,L,25,3), if merge_double_action is one of ['before_pre_logits', 'before_head', 'after_output']
                                                                                                                        # [CASE 2] (N2,2,L,25,3), if merge_double_action is 'no'            
            
                

            if B1 != 0 and B2 != 0:
                # for o1, output1 in enumerate([OUTPUT1]):
                for o1, output1 in enumerate([OUTPUT1, OUTPUT1.mean(1), OUTPUT1.mean(2), OUTPUT1.mean(3), OUTPUT1.mean(dim=(1,2)), OUTPUT1.mean(dim=(1,3)), OUTPUT1.mean(dim=(2,3)), OUTPUT1.mean(dim=(1,2,3))]):
                                            # (N1,1,L,25,3), (N1,L,25,3), (N1,1,25,3), (N1,1,L,3), (N1,25,3), (N1,L,3), (N1,1,3), (N1,3)
                    DISTANCES = calculate_acc(output1)      # (N1,1,L,25,60) 或 (N1,L,25,60) 或 (N1,1,25,60) 或 (N1,1,L,60) 或 (N1,25,60) 或 (N1,L,60) 或 (N1,1,60) 或 (N1,60)
                    # 例如 (N1,1,L,25,60) 情况下, DISTANCES_{n,m,l,j,60} 代表 output1_{n,m,l,j} 分别到60个真实回归点的距离.
                    dims = list(range(1, DISTANCES.dim() - 1))
                    DISTANCES_avg = DISTANCES.mean(dim=dims) if DISTANCES.dim() >= 3 else DISTANCES.clone()     # (N1,60)

                    for d1, distances in enumerate([DISTANCES, DISTANCES_avg]):      # (N1,1,L,25,60) 或 (N1,L,25,60) 或 (N1,1,25,60) 或 (N1,1,L,60) 或 (N1,25,60) 或 (N1,L,60) 或 (N1,1,60) 或 (N1,60); (N1,60)
                        pred_class = distances.argmin(-1)   # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)

                        true_class = QUERY_LABEL_1.clone()  # (N1)
                        while len(true_class.shape) != len(pred_class.shape):
                            true_class = true_class.unsqueeze(-1)
                        true_class = true_class.expand_as(pred_class)   # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)
                        
                        correct_cnt = (pred_class == true_class).type(torch.float)    # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)

                        num_correct_1 = 0
                        for b1 in range(B1):
                            subdata_st = subdata_indices_1[b1]
                            subdata_ed = subdata_indices_1[b1 + 1]
                            correct_cnt_ = correct_cnt[subdata_st:subdata_ed]    # (S,1,L,25) 或 (S,L,25) 或 (S,1,25) 或 (S,1,L) 或 (S,25) 或 (S,L) 或 (S,1) 或 (S)
                            correct_ratio = correct_cnt_.mean()     # 0 <= correct_ratio < 1
                            if correct_ratio >= 0.5:
                                num_correct_1 += 1

                        # for o2, output2 in enumerate([OUTPUT2]):
                        for o2, output2 in enumerate([OUTPUT2, OUTPUT2.mean(1), OUTPUT2.mean(2), OUTPUT2.mean(3), OUTPUT2.mean(dim=(1,2)), OUTPUT2.mean(dim=(1,3)), OUTPUT2.mean(dim=(2,3)), OUTPUT2.mean(dim=(1,2,3))]):
                                                    # (N1,1,L,25,3), (N1,L,25,3), (N1,1,25,3), (N1,1,L,3), (N1,25,3), (N1,L,3), (N1,1,3), (N1,3)
                            DISTANCES = calculate_acc(output2)      # (N1,1,L,25,60) 或 (N1,L,25,60) 或 (N1,1,25,60) 或 (N1,1,L,60) 或 (N1,25,60) 或 (N1,L,60) 或 (N1,1,60) 或 (N1,60)
                            # 例如 (N1,1,L,25,60) 情况下, DISTANCES_{n,m,l,j,60} 代表 output1_{n,m,l,j} 分别到60个真实回归点的距离.
                            dims = list(range(1, DISTANCES.dim() - 1))
                            DISTANCES_avg = DISTANCES.mean(dim=dims) if DISTANCES.dim() >= 3 else DISTANCES.clone()     # (N1,60)

                            for d2, distances in enumerate([DISTANCES, DISTANCES_avg]):      # (N1,1,L,25,60) 或 (N1,L,25,60) 或 (N1,1,25,60) 或 (N1,1,L,60) 或 (N1,25,60) 或 (N1,L,60) 或 (N1,1,60) 或 (N1,60); (N1,60)
                                pred_class = distances.argmin(-1)   # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)

                                true_class = QUERY_LABEL_2.clone()  # (N1)
                                while len(true_class.shape) != len(pred_class.shape):
                                    true_class = true_class.unsqueeze(-1)
                                true_class = true_class.expand_as(pred_class)   # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)
                                
                                correct_cnt = (pred_class == true_class).type(torch.float)    # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)

                                num_correct_2 = 0
                                for b2 in range(B2):
                                    subdata_st = subdata_indices_2[b2]
                                    subdata_ed = subdata_indices_2[b2 + 1]
                                    correct_cnt_ = correct_cnt[subdata_st:subdata_ed]    # (S,1,L,25) 或 (S,L,25) 或 (S,1,25) 或 (S,1,L) 或 (S,25) 或 (S,L) 或 (S,1) 或 (S)
                                    correct_ratio = correct_cnt_.mean()     # 0 <= correct_ratio < 1
                                    if correct_ratio >= 0.5:
                                        num_correct_2 += 1      


                                num_correct[o1, d1, o2, d2] = num_correct[o1, d1, o2, d2] + num_correct_1 + num_correct_2


            elif B1 != 0:
                # for o1, output1 in enumerate([OUTPUT1]):
                for o1, output1 in enumerate([OUTPUT1, OUTPUT1.mean(1), OUTPUT1.mean(2), OUTPUT1.mean(3), OUTPUT1.mean(dim=(1,2)), OUTPUT1.mean(dim=(1,3)), OUTPUT1.mean(dim=(2,3)), OUTPUT1.mean(dim=(1,2,3))]):
                                            # (N1,1,L,25,3), (N1,L,25,3), (N1,1,25,3), (N1,1,L,3), (N1,25,3), (N1,L,3), (N1,1,3), (N1,3)
                    DISTANCES = calculate_acc(output1)      # (N1,1,L,25,60) 或 (N1,L,25,60) 或 (N1,1,25,60) 或 (N1,1,L,60) 或 (N1,25,60) 或 (N1,L,60) 或 (N1,1,60) 或 (N1,60)
                    # 例如 (N1,1,L,25,60) 情况下, DISTANCES_{n,m,l,j,60} 代表 output1_{n,m,l,j} 分别到60个真实回归点的距离.
                    dims = list(range(1, DISTANCES.dim() - 1))
                    DISTANCES_avg = DISTANCES.mean(dim=dims) if DISTANCES.dim() >= 3 else DISTANCES.clone()     # (N1,60)

                    for d1, distances in enumerate([DISTANCES, DISTANCES_avg]):      # (N1,1,L,25,60) 或 (N1,L,25,60) 或 (N1,1,25,60) 或 (N1,1,L,60) 或 (N1,25,60) 或 (N1,L,60) 或 (N1,1,60) 或 (N1,60); (N1,60)
                        pred_class = distances.argmin(-1)   # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)

                        true_class = QUERY_LABEL_1.clone()  # (N1)
                        while len(true_class.shape) != len(pred_class.shape):
                            true_class = true_class.unsqueeze(-1)
                        true_class = true_class.expand_as(pred_class)   # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)
                        
                        correct_cnt = (pred_class == true_class).type(torch.float)    # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)

                        num_correct_1 = 0
                        for b1 in range(B1):
                            subdata_st = subdata_indices_1[b1]
                            subdata_ed = subdata_indices_1[b1 + 1]
                            correct_cnt_ = correct_cnt[subdata_st:subdata_ed]    # (S,1,L,25) 或 (S,L,25) 或 (S,1,25) 或 (S,1,L) 或 (S,25) 或 (S,L) 或 (S,1) 或 (S)
                            correct_ratio = correct_cnt_.mean()     # 0 <= correct_ratio < 1
                            if correct_ratio >= 0.5:
                                num_correct_1 += 1


                        num_correct[o1, d1, :, :] = num_correct[o1, d1, :, :] + num_correct_1


            elif B2 != 0:
                # for o2, output2 in enumerate([OUTPUT2]):
                for o2, output2 in enumerate([OUTPUT2, OUTPUT2.mean(1), OUTPUT2.mean(2), OUTPUT2.mean(3), OUTPUT2.mean(dim=(1,2)), OUTPUT2.mean(dim=(1,3)), OUTPUT2.mean(dim=(2,3)), OUTPUT2.mean(dim=(1,2,3))]):
                                            # (N1,1,L,25,3), (N1,L,25,3), (N1,1,25,3), (N1,1,L,3), (N1,25,3), (N1,L,3), (N1,1,3), (N1,3)
                    DISTANCES = calculate_acc(output2)      # (N1,1,L,25,60) 或 (N1,L,25,60) 或 (N1,1,25,60) 或 (N1,1,L,60) 或 (N1,25,60) 或 (N1,L,60) 或 (N1,1,60) 或 (N1,60)
                    # 例如 (N1,1,L,25,60) 情况下, DISTANCES_{n,m,l,j,60} 代表 output1_{n,m,l,j} 分别到60个真实回归点的距离.
                    dims = list(range(1, DISTANCES.dim() - 1))
                    DISTANCES_avg = DISTANCES.mean(dim=dims) if DISTANCES.dim() >= 3 else DISTANCES.clone()     # (N1,60)

                    for d2, distances in enumerate([DISTANCES, DISTANCES_avg]):      # (N1,1,L,25,60) 或 (N1,L,25,60) 或 (N1,1,25,60) 或 (N1,1,L,60) 或 (N1,25,60) 或 (N1,L,60) 或 (N1,1,60) 或 (N1,60); (N1,60)
                        pred_class = distances.argmin(-1)   # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)

                        true_class = QUERY_LABEL_2.clone()  # (N1)
                        while len(true_class.shape) != len(pred_class.shape):
                            true_class = true_class.unsqueeze(-1)
                        true_class = true_class.expand_as(pred_class)   # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)
                        
                        correct_cnt = (pred_class == true_class).type(torch.float)    # (N1,1,L,25) 或 (N1,L,25) 或 (N1,1,25) 或 (N1,1,L) 或 (N1,25) 或 (N1,L) 或 (N1,1) 或 (N1)

                        num_correct_2 = 0
                        for b2 in range(B2):
                            subdata_st = subdata_indices_2[b2]
                            subdata_ed = subdata_indices_2[b2 + 1]
                            correct_cnt_ = correct_cnt[subdata_st:subdata_ed]    # (S,1,L,25) 或 (S,L,25) 或 (S,1,25) 或 (S,1,L) 或 (S,25) 或 (S,L) 或 (S,1) 或 (S)
                            correct_ratio = correct_cnt_.mean()     # 0 <= correct_ratio < 1
                            if correct_ratio >= 0.5:
                                num_correct_2 += 1      


                        num_correct[:, :, o2, d2] = num_correct[:, :, o2, d2] + num_correct_2


            if idx in [len(test_loader) * i // 3 for i in range(1, 3)]:
                print(f"\tTest: {idx}/{len(test_loader)}; time cost: {(time()-st)/60 :.2f}min")

    max_acc = num_correct.max() / num_samples * 100
    min_acc = num_correct.min() / num_samples * 100
    where_max_acc = np.where(num_correct == num_correct.max())  # tuple
    where_min_acc = np.where(num_correct == num_correct.min())  # tuple
    # where_min_acc = 'Null'
    
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['AR'] + ['max acc'] + ['min acc'] + ['max protocol'] + ['min protocol']
    summary_table.add_row(['Accuracy'] + [max_acc] + [min_acc] + [where_max_acc] + [where_min_acc])
    summary_table.float_format = ".2"

    return max_acc, summary_table