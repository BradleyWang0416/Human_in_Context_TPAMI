from time import time
import torch
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
from lib.utils.utils_AR import get_targets
from lib.utils.viz_skel_seq import viz_skel_seq_anim


def train_batch(args, query_input, query_label, prompt_input, prompt_label, model, epoch):
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
    ground_truth = query_label.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(output)
    return output, ground_truth


def train_epoch(args, model_pos, train_loader, losses, optimizer, epoch=None, if_viz=False):
    '''
    train_loader is a dict: {'non_AR': <non-AR loader>, 'AR': <AR loader>}
    '''
    model_pos.train()
    st = time()
    for idx, (QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1,
              QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2) in enumerate(train_loader['AR']):
        if len(QUERY_INPUT_1) == 0: 
            print(f'\t\tbatch {idx} skipped')
            continue
        if len(QUERY_INPUT_2) == 0:
            print(f'\t\tbatch {idx} skipped')
            continue
        # (B1,1,T,J,C); (B1); (B1,1,T,J,C); (B1); 
        # (B2,2,T,J,C); (B2); (B2,2,T,J,C); (B2); 
        if torch.cuda.is_available():
            QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1 = QUERY_INPUT_1.cuda(), QUERY_LABEL_1.cuda(), PROMPT_INPUT_1.cuda(), PROMPT_LABEL_1.cuda()
            QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2 = QUERY_INPUT_2.cuda(), QUERY_LABEL_2.cuda(), PROMPT_INPUT_2.cuda(), PROMPT_LABEL_2.cuda()
        
        B1 = QUERY_INPUT_1.shape[0]
        B2 = QUERY_INPUT_2.shape[0]
        batch_size = B1 + B2
        
        output1, gt1 = train_batch(args, QUERY_INPUT_1, QUERY_LABEL_1, PROMPT_INPUT_1, PROMPT_LABEL_1, model_pos, epoch)
        output2, gt2 = train_batch(args, QUERY_INPUT_2, QUERY_LABEL_2, PROMPT_INPUT_2, PROMPT_LABEL_2, model_pos, epoch)

        optimizer.zero_grad()
        loss = B1/batch_size * losses['mpjpe']['loss_function'](output1, gt1) + B2/batch_size * losses['mpjpe']['loss_function'](output2, gt2)
        loss.backward()
        optimizer.step()


        if idx in [len(train_loader['AR']) * i // 5 for i in range(1, 5)]:
            print(f"\tIter: {idx}/{len(train_loader['AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {B1}/{B2} single/double-actor samples")


            if if_viz:
                for b in range(batch_size):
                    pred_label_1 = rebuild_part_AR[b, 0, :, :17, :]      # (L,17,3)
                    pred_label_1 = pred_label_1 + torch.randn_like(pred_label_1)*0.005
                    pred_label_2 = rebuild_part_AR[b, 1, :, :17, :]      # (L,17,3)
                    pred_label_2 = pred_label_2 + torch.randn_like(pred_label_2)*0.005
                    true_label = target_part_AR[b, 0, :, :17, :]       # (L,17,3)
                    true_label = true_label + torch.randn_like(true_label)*0.05
                    p_label = prompt_target[b,0,:,:17,:]       # (L,17,3)
                    p_label = p_label + torch.randn_like(p_label)*0.05
                    viz_dict = {
                        'pred output 1st person': pred_label_1,
                        'true target': true_label,
                        'prompt target': p_label,
                        'pred output 2nd person': pred_label_2,
                    }
                    viz_skel_seq_anim(viz_dict, if_print=1, fig_title=f'epoch{epoch} | iter {idx} | sample{b} | true class {query_label[b]} | prompt class {prompt_label[b]}', lim3d=1, if_node=0, file_name=f'ep{epoch:03d}_iter{idx:04d}_sample{b:02d}_class{query_label[b]:02d}_Pclass{prompt_label[b]:02d}', file_folder='tmp/in_train')

    return

