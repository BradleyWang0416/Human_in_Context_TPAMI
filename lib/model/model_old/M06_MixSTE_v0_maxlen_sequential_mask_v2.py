## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import clip
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., changedim=False, currentdim=0, depth=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., comb=False, vis=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 

        self.proj_drop = nn.Dropout(proj_drop)
        self.comb = comb
        self.vis = vis

    def forward(self, x, vis=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.comb==True:
            attn = (q.transpose(-2, -1) @ k) * self.scale
        elif self.comb==False:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        if self.comb==True:
            x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
            x = rearrange(x, 'B H N C -> B N (H C)')
        elif self.comb==False:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., attention=Attention, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, comb=False, changedim=False, currentdim=0, depth=0, vis=False):
        super().__init__()

        self.changedim = changedim
        self.currentdim = currentdim
        self.depth = depth
        if self.changedim:
            assert self.depth>0

        self.norm1 = norm_layer(dim)
        self.attn = attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, comb=comb, vis=vis)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        if self.changedim and self.currentdim < self.depth//2:
            self.reduction = nn.Conv1d(dim, dim//2, kernel_size=1)
        elif self.changedim and depth > self.currentdim > self.depth//2:
            self.improve = nn.Conv1d(dim, dim*2, kernel_size=1)
        self.vis = vis

    def forward(self, x, vis=False):
        x = x + self.drop_path(self.attn(self.norm1(x), vis=vis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if self.changedim and self.currentdim < self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.reduction(x)
            x = rearrange(x, 'b c t -> b t c')
        elif self.changedim and self.depth > self.currentdim > self.depth//2:
            x = rearrange(x, 'b t c -> b c t')
            x = self.improve(x)
            x = rearrange(x, 'b c t -> b t c')
        return x


class STE(nn.Module):
    def __init__(self, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.ste_block = Block( dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Spatial_norm = norm_layer(embed_dim_ratio)
    def forward(self, x):
        b, f, n, c = x.shape
        x = x.reshape(b*f, n, c)
        x = self.pos_drop(x)
        x = self.ste_block(x)
        x = self.Spatial_norm(x)
        x = x.reshape(b, f, n, c)
        return x


class TTE(nn.Module):
    def __init__(self, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.tte_block = Block( dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=False, currentdim=0, depth=depth)
         
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Temporal_norm = norm_layer(embed_dim_ratio)
    def forward(self, x, temporal_pos_embed):
        b, f, n, c  = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b*n, f, c)                                # x: (B*17,243,512)
        x += temporal_pos_embed.unsqueeze(0)
        x = self.pos_drop(x)
        x = self.tte_block(x)                                                                  # x: (B*17,243,512)
        x = self.Temporal_norm(x)
        return x.reshape(b,n,f,c).permute(0,2,1,3)


class ST(nn.Module):
    def __init__(self, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(1, depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(1, depth)])
        self.block_depth = depth
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim_ratio)
    def forward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(0, self.block_depth - 1):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            x = steblock(x)
            x = self.Spatial_norm(x)

            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        return x
    def forward_i(self, x, i):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        steblock = self.STEblocks[i]
        tteblock = self.TTEblocks[i]
        
        x = rearrange(x, 'b f n cw -> (b f) n cw')
        x = steblock(x)
        x = self.Spatial_norm(x)

        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
        x = tteblock(x)
        x = self.Temporal_norm(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        return x


class Branch_temporal(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2, 
                 qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=None, is_train=True, 
                 prompt_enabled=False, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None, 
                 prompt_gt_as_condition=False, use_text=True, fuse_prompt_query='add', max_clip_len=243):
        super().__init__()

        self.prompt_enabled = prompt_enabled
        self.depth = depth
        self.use_text = use_text
        self.fuse_prompt_query = fuse_prompt_query

        self.STE = STE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.TTE = TTE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.ST = ST(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)

        self.spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(num_joints, embed_dim_ratio))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(num_frame*4, embed_dim_ratio))
    
    def forward_encode(self, feat):                    # query_input: (B,243,17,3)
        feat = self.spatial_patch_to_embedding(feat)
        feat += self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        return feat
    def forward_ste(self, feat):
        feat = self.STE(feat)
        return feat
    def forward_tte(self, feat):
        feat = self.TTE(feat, self.temporal_pos_embed)
        return feat
    def forward_st(self, feat):
        for i in range(self.depth - 1):
            feat = self.ST.forward_i(feat, i)
        return feat
    def forward_st_i(self, feat, i):
        feat = self.ST.forward_i(feat, i)
        return feat
    def forward(self, x):
        x = self.forward_encode(x)
        x = self.forward_ste(x)
        x = self.forward_tte(x)
        x = self.forward_st(x)
        return x


class Branch_spatial(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2, 
                 qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=None, is_train=True, 
                 prompt_enabled=False, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None, 
                 prompt_gt_as_condition=False, use_text=True, fuse_prompt_query='add', max_clip_len=243):
        super().__init__()

        self.prompt_enabled = prompt_enabled
        self.depth = depth
        self.use_text = use_text
        self.fuse_prompt_query = fuse_prompt_query

        self.STE = STE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.TTE = TTE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.ST = ST(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )

        self.spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(num_joints*4, embed_dim_ratio))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(num_frame, embed_dim_ratio))    

    def forward_encode(self, feat):                    # query_input: (B,243,17,3)
        feat = self.spatial_patch_to_embedding(feat)
        feat += self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        return feat
    def forward_ste(self, feat):
        feat = self.STE(feat)
        return feat
    def forward_tte(self, feat):
        feat = self.TTE(feat, self.temporal_pos_embed)
        return feat
    def forward_st(self, feat):
        for i in range(self.depth - 1):
            feat = self.ST.forward_i(feat, i)
        return feat
    def forward_st_i(self, feat, i):
        feat = self.ST.forward_i(feat, i)
        return feat
    def forward(self, x):
        x = self.forward_encode(x)
        x = self.forward_ste(x)
        x = self.forward_tte(x)
        x = self.forward_st(x)
        return x
        

class Branch_channel(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2, 
                 qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=None, is_train=True, 
                 prompt_enabled=False, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None, 
                 prompt_gt_as_condition=False, use_text=True, fuse_prompt_query='add', max_clip_len=243):
        super().__init__()

        self.prompt_enabled = prompt_enabled
        self.depth = depth
        self.use_text = use_text
        self.fuse_prompt_query = fuse_prompt_query

        self.STE = STE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.TTE = TTE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.ST = ST(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )

        self.spatial_patch_to_embedding = nn.Linear(in_chans*4, embed_dim_ratio)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(num_joints, embed_dim_ratio))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(num_frame, embed_dim_ratio))    

    def forward_encode(self, feat):                    # query_input: (B,243,17,3)
        feat = self.spatial_patch_to_embedding(feat)
        feat += self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        return feat
    def forward_ste(self, feat):
        feat = self.STE(feat)
        return feat
    def forward_tte(self, feat):
        feat = self.TTE(feat, self.temporal_pos_embed)
        return feat
    def forward_st(self, feat):
        for i in range(self.depth - 1):
            feat = self.ST.forward_i(feat, i)
        return feat
    def forward_st_i(self, feat, i):
        feat = self.ST.forward_i(feat, i)
        return feat
    def forward(self, x):
        x = self.forward_encode(x)
        x = self.forward_ste(x)
        x = self.forward_tte(x)
        x = self.forward_st(x)
        return x


class Skeleton_in_Context(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2, 
                 qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=None, is_train=True, 
                 prompt_enabled=False, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None, 
                 prompt_gt_as_condition=False, use_text=True, fuse_prompt_query='add', max_clip_len=243):
        super().__init__()

        self.branch_temporal = Branch_temporal(num_frame, num_joints, in_chans, embed_dim_ratio, depth, num_heads, mlp_ratio, 
                    qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, is_train, 
                    prompt_enabled, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                    prompt_gt_as_condition, use_text, fuse_prompt_query, max_clip_len)
        self.branch_spatial = Branch_spatial(num_frame, num_joints, in_chans, embed_dim_ratio, depth, num_heads, mlp_ratio, 
                    qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, is_train, 
                    prompt_enabled, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                    prompt_gt_as_condition, use_text, fuse_prompt_query, max_clip_len)
        self.branch_channel = Branch_channel(num_frame, num_joints, in_chans, embed_dim_ratio, depth, num_heads, mlp_ratio, 
                    qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, is_train, 
                    prompt_enabled, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, 
                    prompt_gt_as_condition, use_text, fuse_prompt_query, max_clip_len)      

        self.global_embed = nn.Parameter(torch.zeros(4, num_frame, num_joints, 3))

        self.fuse = nn.Linear(embed_dim_ratio*3, 3)
        self.fuse.bias.data.fill_(0.5)
        self.fuse.weight.data.fill_(0)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )


    def forward(self, Prompt, Query, epoch=None):
        B, TT, J, C = Query.shape
        T = TT // 2
        PROMPT_INPUT = Prompt[:, :T, :, :]
        PROMPT_TARGET = Prompt[:, T:, :, :]
        QUERY_INPUT = Query[:, :T, :, :]
        QUERY_TARGET = Query[:, T:, :, :]

        if self.training:
            mask_pi = torch.rand(B,T,J,C, device=Query.device) > 0.25
            mask_pt = torch.rand(B,T,J,C, device=Query.device) > 0.25
            mask_qi = torch.rand(B,T,J,C, device=Query.device) > 0.25
            mask_qt = torch.rand(B,T,J,C, device=Query.device) > 0.25
        else:
            mask_pi = torch.ones(B,T,J,C, device=Query.device)
            mask_pt = torch.ones(B,T,J,C, device=Query.device)
            mask_qi = torch.ones(B,T,J,C, device=Query.device)
            mask_qt = torch.zeros(B,T,J,C, device=Query.device)
        prompt_input = PROMPT_INPUT * mask_pi
        prompt_target = PROMPT_TARGET * mask_pt
        query_input = QUERY_INPUT * mask_qi
        query_target = QUERY_TARGET * mask_qt

        feat_temporal = torch.cat([prompt_input, prompt_target, query_input, query_target], dim=-3)
        # (B,4T,J,C)
        feat_temporal += self.global_embed.reshape(4*T, J, C).unsqueeze(0)   
        # (4,T,J,C)->(4T,J,C)->(1,4T,J,C)
        
        feat_spatial = torch.cat([prompt_input, prompt_target, query_input, query_target], dim=-2)
        # (B,T,4J,C)
        feat_spatial += self.global_embed.permute(1,0,2,3).reshape(T, 4*J, C).unsqueeze(0)
        # (4,T,J,C)->(T,4,J,C)->(T,4J,C)->(1,T,4J,C)

        feat_channel = torch.cat([prompt_input, prompt_target, query_input, query_target], dim=-1)
        # (B,T,J,4C)
        feat_channel += self.global_embed.permute(1,2,0,3).reshape(T, J, 4*C).unsqueeze(0)
        # (4,T,J,C)->(T,J,4,C)->(T,J,4C)->(1,T,J,4C)
        
        feat_temporal = self.branch_temporal.forward(feat_temporal)  # (B,4T,J,512)
        feat_spatial = self.branch_spatial.forward(feat_spatial)     # (B,T,4J,512)
        feat_channel = self.branch_channel.forward(feat_channel)     # (B,T,J,512)

        feat_temporal = feat_temporal.repeat(1, 1, 4, 1)    # (B,4T,4J,512)
        feat_spatial = feat_spatial.repeat(1, 4, 1, 1)     # (B,4T,4J,512)
        feat_channel = feat_channel.repeat(1, 4, 4, 1)     # (B,4T,4J,512)

        alpha = torch.cat([feat_temporal, feat_spatial, feat_channel], dim=-1) # (B,4T,4J,1536)
        alpha = self.fuse(alpha)        # (B,4T,4J,3）
        alpha = alpha.softmax(dim=-1)   # (B,4T,4J,3)
        feat = feat_temporal * alpha[:,:,:,0:1] + feat_spatial * alpha[:,:,:,1:2] + feat_channel * alpha[:,:,:,2:3]
        # (B,4T,4J,512)

        feat = self.head(feat)  # (B,4T,4J,3)

        if self.training:
            return feat
        else:
            return feat.reshape(B, 4, T, 4, J, 3).mean(1).mean(-3)



        # feat_temporal = self.branch_temporal.forward_ste(feat_temporal)
        # feat_spatial = self.branch_spatial.forward_ste(feat_spatial)   
        # feat_channel = self.branch_channel.forward_ste(feat_channel)   

        # feat_temporal = self.branch_temporal.forward_tte(feat_temporal)
        # feat_spatial = self.branch_spatial.forward_tte(feat_spatial)
        # feat_channel = self.branch_channel.forward_tte(feat_channel)

        # for i in range(self.depth - 1):
        #     feat_temporal = self.branch_temporal.forward_st_i(feat_temporal, i)
        #     feat_spatial = self.branch_spatial.forward_st_i(feat_spatial, i)
        #     feat_channel = self.branch_channel.forward_st_i(feat_channel, i)

