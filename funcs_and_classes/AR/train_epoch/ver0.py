from time import time
import torch
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
sys.path.append(os.path.join(ROOT_DIR, 'lib'))
from lib.utils.utils_AR import get_targets
from lib.utils.viz_skel_seq import viz_skel_seq_anim


def train_epoch(args, model_pos, train_loader, losses, optimizer, epoch=None, if_viz=False):
    '''
    train_loader is a dict: {'non_AR': <non-AR loader>, 'AR': <AR loader>}
    '''
    model_pos.train()
    st = time()
    for idx, (query_input, query_label, prompt_input, prompt_label) in enumerate(train_loader['AR']):

        ####################################### Training on 2DAR #######################################
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
        assert T == args.train_feeder_args_ntu.window_size
        L = args.data.clip_len * 2 - T

        query_target = torch.stack([get_targets()[query_label[b]] for b in range(B)])     # (B,3)
        prompt_target = torch.stack([get_targets()[prompt_label[b]] for b in range(B)])   # (B,3)
        
        query_target = query_target.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, M, L, J, -1)      # (B, 2, 4, 25, 3)
        prompt_target = prompt_target.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, M, L, J, -1)   # (B, 2, 4, 25, 3)

        QUERY = torch.cat([query_input, query_target], dim=2)       # (B, 2, 64, 25, 3)
        PROMPT = torch.cat([prompt_input, prompt_target], dim=2)    # (B, 2, 64, 25, 3)

        if args.func_ver.get('model_name', 'M00_SiC_dynamicTUP') ==  'M00_SiC_dynamicTUP':
            rebuild_part_AR = model_pos(PROMPT, QUERY, epoch)    # (B*2, 4, 25, 3)
            rebuild_part_AR = rebuild_part_AR.reshape(B, M, L, J, C)    # (B, 2, 4, 25, 3)
        else:
            NotImplemented
        
        target_part_AR = query_target.clone()     # (B, 2, 4, 25, 3)
        ####################################### Training on 2DAR #######################################

        # Optimize
        optimizer.zero_grad()

        loss = losses['mpjpe']['loss_function'](rebuild_part_AR, target_part_AR)
        loss.backward()
        optimizer.step()


        # for loss_name, loss_dict in losses.items():
        #     if loss_name != 'mpjpe':
        #         continue
        #     weight = loss_dict['loss_weight']
        #     if weight == 0: continue
        #     loss = loss_dict['loss_function'](rebuild_part_AR, target_part_AR)
        #     loss_dict['loss_logger'].update(loss.item(), B)
        #     if weight != 0:
        #         loss_total += loss * weight
        # losses['total']['loss_logger'].update(loss_total.item(), B)
        # loss_total.backward()
        # optimizer.step()

        if idx in [len(train_loader['AR']) * i // 5 for i in range(1, 5)]:
            print(f"\tIter: {idx}/{len(train_loader['AR'])}; time cost: {(time()-st)/60 :.2f}min; current batch has {B} samples")


            if if_viz:
                for b in range(B):
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

