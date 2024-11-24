## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lib.model.M06_MixSTE_v0 import Skeleton_in_Context as Skeleton_in_Context_v0
    

class Skeleton_in_Context(Skeleton_in_Context_v0):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2, qkv_bias=True, qk_scale=None, 
                 drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=None, is_train=True, 
                 prompt_enabled=False, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None, 
                 prompt_gt_as_condition=False, use_text=True, fuse_prompt_query='add'):
        super().__init__(num_frame, num_joints, in_chans, embed_dim_ratio, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, 
                         drop_rate, attn_drop_rate, drop_path_rate, norm_layer, is_train, 
                         prompt_enabled, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                         prompt_gt_as_condition, use_text, fuse_prompt_query)

    def ST_foward_i(self, x, i):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        steblock = self.STEblocks[i+1]
        tteblock = self.TTEblocks[i+1]
        
        x = rearrange(x, 'b f n cw -> (b f) n cw')
        x = steblock(x)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        x = tteblock(x)
        x = self.Temporal_norm(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        return x

    def forward(self, Prompt, Query, epoch=None):                                                               # query_input: (B,243,17,3)
        B, F, J, C = Query.shape
        F1 = F // 2
        query_input = Query[:, :F1, :, :]
        prompt_input = Prompt[:, :F1, :, :]
        prompt_groundtruth_output = Prompt[:, F1:, :, :]

        b, f, n, c = query_input.shape


        # PROMPT branch
        prompt = self.prompt_encode_forward(prompt_input, prompt_groundtruth_output)                            # prompt: (B,243,17,512)


        prompt = self.prompt_ste(prompt)                                                                        # prompt: (B,243,17,512)


        prompt = self.prompt_tte(prompt)                                                                        # prompt: (B,243,17,512)

        PROMPTS = []
        for i in range(self.depth - 1):
            prompt = self.prompt_st.forward_i(prompt, i)
            if i > ((self.depth - 1) // 2):
                PROMPTS.append(prompt)


        # QUERY branch
        query = self.encode_forward(query_input, prompt_groundtruth_output)                                     # query: (B,243,17,512)


        query = self.STE_forward(query)                                                                         # query: (B,243,17,512)


        query = self.TTE_foward(query)                                                                          # query: (B,243,17,512)

        mid_feats = []
        for i in range(self.depth - 1):
            if i <= ((self.depth - 1) // 2):
                mid_feats.append(query)
                query = self.ST_foward_i(query, i)
            elif i > ((self.depth - 1) // 2):
                query = self.ST_foward_i(query, i)
                query += mid_feats[-1]
                query += PROMPTS[-1]
                mid_feats.pop()
                PROMPTS.pop()

        # MERGE
        query = self.head(query)
        query = query.view(b, f, n, -1)

        return query