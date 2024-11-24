#%%
import os
import numpy as np
import prettytable
import torch
import random

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.utils.viz_skel_seq import viz_skel_seq_anim

#%%
def evaluate_generalization(args, test_loader, model, epoch=None, if_viz=False, if_debug=False):

    if if_debug:
        return

    sub_batch_size = 256

    model.eval()
    loader_pe = test_loader['PE']
    loader_mp = test_loader['MP']
    loader_mc = test_loader['MC']
    loader_fpe = test_loader['FPE']

    with torch.no_grad():
        for idx, (data_pe, data_mp, data_mc, data_fpe) in enumerate(zip(loader_pe, loader_mp, loader_mc, loader_fpe)):
            PROMPT_BATCH_pe, QUERY_BATCH_pe, TASK = data_pe
            PROMPT_BATCH_mp, QUERY_BATCH_mp, TASK = data_mp
            PROMPT_BATCH_mc, QUERY_BATCH_mc, TASK = data_mc
            PROMPT_BATCH_fpe, QUERY_BATCH_fpe, TASK = data_fpe

            sub_batch_num = (QUERY_BATCH_pe.shape[0] + sub_batch_size - 1) // sub_batch_size

            for sub_idx in range(sub_batch_num):
                if (sub_idx+1)*sub_batch_size > QUERY_BATCH_pe.shape[0]:
                    prompt_batch_pe = PROMPT_BATCH_pe[  sub_idx*sub_batch_size  :  ]
                    query_batch_pe = QUERY_BATCH_pe[  sub_idx*sub_batch_size  :  ]
                    prompt_batch_mp = PROMPT_BATCH_mp[  sub_idx*sub_batch_size  :  ]
                    query_batch_mp = QUERY_BATCH_mp[  sub_idx*sub_batch_size  :  ]
                    prompt_batch_mc = PROMPT_BATCH_mc[  sub_idx*sub_batch_size  :  ]
                    query_batch_mc = QUERY_BATCH_mc[  sub_idx*sub_batch_size  :  ]
                    prompt_batch_fpe = PROMPT_BATCH_fpe[  sub_idx*sub_batch_size  :  ]
                    query_batch_fpe = QUERY_BATCH_fpe[  sub_idx*sub_batch_size  :  ]
                else:
                    prompt_batch_pe = PROMPT_BATCH_pe[  sub_idx*sub_batch_size  :  (sub_idx+1)*sub_batch_size  ]
                    query_batch_pe = QUERY_BATCH_pe[  sub_idx*sub_batch_size  :  (sub_idx+1)*sub_batch_size  ]
                    prompt_batch_mp = PROMPT_BATCH_mp[  sub_idx*sub_batch_size  :  (sub_idx+1)*sub_batch_size  ]
                    query_batch_mp = QUERY_BATCH_mp[  sub_idx*sub_batch_size  :  (sub_idx+1)*sub_batch_size  ]
                    prompt_batch_mc = PROMPT_BATCH_mc[  sub_idx*sub_batch_size  :  (sub_idx+1)*sub_batch_size  ]
                    query_batch_mc = QUERY_BATCH_mc[  sub_idx*sub_batch_size  :  (sub_idx+1)*sub_batch_size  ]
                    prompt_batch_fpe = PROMPT_BATCH_fpe[  sub_idx*sub_batch_size  :  (sub_idx+1)*sub_batch_size  ]
                    query_batch_fpe = QUERY_BATCH_fpe[  sub_idx*sub_batch_size  :  (sub_idx+1)*sub_batch_size  ]

                if torch.cuda.is_available():
                    prompt_batch_pe = prompt_batch_pe.cuda() 
                    query_batch_pe = query_batch_pe.cuda()
                    prompt_batch_mp = prompt_batch_mp.cuda()
                    query_batch_mp = query_batch_mp.cuda()
                    prompt_batch_mc = prompt_batch_mc.cuda()
                    query_batch_mc = query_batch_mc.cuda()
                    prompt_batch_fpe = prompt_batch_fpe.cuda()
                    query_batch_fpe = query_batch_fpe.cuda()
                

                pred_batch = model(prompt_batch_pe, query_batch_pe)
                pred_batch_shuffle_prompt = model(torch.cat([prompt_batch_pe[2:], prompt_batch_pe[:2]], dim=0), query_batch_pe)
                pred_batch1 = model(prompt_batch_mp, query_batch_pe)

                # for i in range(len(query_batch_pe)):
                #     query_input_one_sample = prompt_batch_mc[i, :args.data.clip_len]    # (clip_len, 17, 3)
                #     masked_frame_idx = torch.all(query_input_one_sample[:,1:].sum(dim=(0,2), keepdim=True) == 0, dim=0).squeeze(-1)
                #     masked_frame_idx = torch.cat([torch.tensor([False]).cuda(), masked_frame_idx])
                #     query_batch_pe[i, :16, masked_frame_idx] = 0
                # pred_batch2 = model(prompt_batch_mc, query_batch_pe)

                # mask_frame = random.sample(range(16), int(16*0.5))
                # prompt_batch_pe[:, mask_frame, :, :] = 0
                # query_batch_pe[:, mask_frame, :, :] = 0

                # pred_batch2 = model(prompt_batch_pe, query_batch_pe)

                # pred_batch3 = model(prompt_batch_mc, query_batch_pe)

                for i in range(len(query_batch_pe)):
                    # if not (idx==0 and i==2): continue
                    # velo = torch.mean(torch.norm(query_batch_pe[i, 1:16]*1000 - query_batch_pe[i, 0:15]*1000, dim=-1))
                    # if velo < 40: continue
                    data_dict = {
                        'input': query_batch_pe[i, :16][[0,2,4,6,8,10,12,14,15]],
                        'gt': query_batch_pe[i, 16:][[0,2,4,6,8,10,12,14,15]],
                        'pred_batch' : pred_batch[i][[0,2,4,6,8,10,12,14,15]],
                        'pred_batch_shuffle_prompt': pred_batch_shuffle_prompt[i][[0,2,4,6,8,10,12,14,15]],
                        # 'pred_batch1' : pred_batch1[i][[0,2,4,6,8,10,12,14,15]],
                        # 'prompt input': prompt_batch_mp[i, :16][[0,2,4,6,8,10,12,14,15]],
                        # 'prompt target': prompt_batch_mp[i, 16:][[0,2,4,6,8,10,12,14,15]],
                        'prompt input': prompt_batch_mc[i, :16][[0,2,4,6,8,10,12,14,15]],
                        'prompt target': prompt_batch_mc[i, 16:][[0,2,4,6,8,10,12,14,15]],
                        # 'pred_batch2' : pred_batch2[i][[0,2,4,6,8,10,12,14,15]],
                        # 'pred_batch3' : pred_batch3[i]
                    }
                    if (idx, i) in [(0,2), (0,3)] or True:
                        # viz_skel_seq_anim(data_dict, subplot_layout=(6,1), mode='img', fig_title=f'iter{idx}-idx{i}', fs=0.5, lim3d=0.5, lw=5,
                        #                 if_print=True, file_name=f'H36M_PE_TEST_iter{idx:04d}_idx{i:04d}_BS{sub_batch_size:04d}', file_folder='PAMI_EVAL_GENERALIZATION')
                        viz_skel_seq_anim(data_dict, subplot_layout=(6,1), mode='img', fig_title=f'iter{idx}-idx{i}', fs=0.25, lim3d=0.5, lw=2,
                                          if_print=False, file_name=f'H36M_PE_TEST_iter{idx:04d}_idx{i:04d}_BS{sub_batch_size:04d}', file_folder='PAMI_EVAL_GENERALIZATION')