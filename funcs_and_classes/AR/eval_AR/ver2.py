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


def calculate_acc(pred, label):
    # pred: (B,M,L,25,3) or (B,L,25,3) or (B,M,25,3) or (B,M,L,3) or (B,25,3) or (B,L,3) or (B,M,3) or (B,3)
    # label: (B)
    pred_label = pred.unsqueeze(-2)     # pred_label: (B,M,L,25,1,3) or (B,L,25,1,3) or (B,M,25,1,3) or (B,M,L,1,3) or (B,25,1,3) or (B,L,1,3) or (B,M,1,3) or (B,1,3)
    labels = get_targets()  # (60,3)
    while len(labels.shape) != len(pred_label.shape):
        labels = labels.unsqueeze(0)
    distances = torch.norm(pred_label - labels, dim=-1)            # (B, ..., 60)
    pred_class = distances.argmin(-1)   # (B, ...)

    true_class = label.clone()
    while len(true_class.shape) != len(pred_class.shape):
        true_class = true_class.unsqueeze(-1)
    true_class = true_class.expand_as(pred_class)   # (B,M,L,25) or (B,L,25) or (B,M,25) or (B,M,L) or (B,25) or (B,L) or (B,M) or (B)


    if true_class.dim() > 1:
        non_batch_dim = list(range(1, true_class.dim()))  # [1,2,3] or [1,2] or [1,2] or [1,2] or [1] or [1] or [1]
        correct_ratio = (pred_class == true_class).type(torch.int).sum(dim=non_batch_dim) / pred_class[0].numel()   # (B)
    else:
        correct_ratio = (pred_class == true_class).type(torch.int)

    return correct_ratio


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

    num_correct = np.zeros((8, 8))

    st = time()
    with torch.no_grad():
        for idx, (QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1,
                  QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2) in enumerate(test_loader):
            # (B1,1,T,J,C); tensor: (B1); tensor: (B1,1,T,J,C); tensor: (B1); 
            # (B2,2,T,J,C); tensor: (B2); tensor: (B2,2,T,J,C); tensor: (B2);
            if torch.cuda.is_available():
                QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1 = QUERY_INPUT_1.cuda(), QUERY_LABEL_1.cuda(), PROMPT_INPUT_1.cuda(), PROMPT_LABEL_1.cuda()
                QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2 = QUERY_INPUT_2.cuda(), QUERY_LABEL_2.cuda(), PROMPT_INPUT_2.cuda(), PROMPT_LABEL_2.cuda()
        
            B1 = QUERY_INPUT_1.shape[0]
            B2 = QUERY_INPUT_2.shape[0]
            batch_size = B1 + B2
            num_samples += batch_size
            
            output1 = eval_batch(args, QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1, model, epoch)  # (B1,1,L,25,3)
            output2 = eval_batch(args, QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2, model, epoch)  # [CASE 1] (B2,1,L,25,3), if merge_double_action is one of ['before_pre_logits', 'before_head', 'after_output']
                                                                                                                    # [CASE 2] (B2,2,L,25,3), if merge_double_action is 'no'
            
            if if_viz:
                for b in range(len(output1)):
                    o1 = output1[b,0,:,:17,:]
                    viz_dict = {
                        'output1': o1 + torch.randn_like(o1) * 0.01,
                        'query target': get_targets()[QUERY_LABEL_1[b]].unsqueeze(0).unsqueeze(0).expand_as(o1) + torch.randn_like(o1) * 0.01,
                        'prompt target': get_targets()[PROMPT_LABEL_1[b]].unsqueeze(0).unsqueeze(0).expand_as(o1) + torch.randn_like(o1)* 0.01
                    } 
                    viz_skel_seq_anim(viz_dict, if_print=1, if_target=1, fig_title=f'iter{idx} sample{b} class{QUERY_LABEL_1[b]} pclass{PROMPT_LABEL_1[b]}',
                                      file_name=f'iter{idx:04d}_sample{b:02d}_class{QUERY_LABEL_1[b]:02d}_pclass{PROMPT_LABEL_1[b]:02d}', file_folder='tmp/eval_AR')
            

            output1_to_eval_list = [output1, output1.mean(1), output1.mean(2), output1.mean(3), output1.mean(dim=(1,2)), output1.mean(dim=(1,3)), output1.mean(dim=(2,3)), output1.mean(dim=(1,2,3))]
            output2_to_eval_list = [output2, output2.mean(1), output2.mean(2), output2.mean(3), output2.mean(dim=(1,2)), output2.mean(dim=(1,3)), output2.mean(dim=(2,3)), output2.mean(dim=(1,2,3))]


            for i, output1_to_eval in enumerate(output1_to_eval_list):
                correct_ratio1 = calculate_acc(output1_to_eval, QUERY_LABEL_1)   # (B1)
                correct_indices1 = torch.where(correct_ratio1 >= 0.5)[0]
                n_correct1 = len(correct_indices1)

                for j, output2_to_eval in enumerate(output2_to_eval_list):
                    correct_ratio2 = calculate_acc(output2_to_eval, QUERY_LABEL_2)   # (B2)
                    correct_indices2 = torch.where(correct_ratio2 >= 0.5)[0]
                    n_correct2 = len(correct_indices2)

                    num_correct[i, j] = num_correct[i, j] + n_correct1 + n_correct2



            if idx in [len(test_loader) * i // 3 for i in range(1, 3)]:
                print(f"\tTest: {idx}/{len(test_loader)}; time cost: {(time()-st)/60 :.2f}min")


    acc = num_correct.reshape(-1) / num_samples * 100   # (64)

    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['AR'] + ['max acc'] + [f'#{p}' for p in range(1,65)]
    summary_table.add_row(['Accuracy'] + [f'{acc.max()} (#{acc.argmax()+1})'] + acc.tolist())
    summary_table.float_format = ".2"

    return acc.max(), summary_table