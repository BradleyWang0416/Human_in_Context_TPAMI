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
import copy

import time
import json

from math import sqrt

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from third_party.motionbert.lib.model.model_mesh import SMPLRegressor
from lib.model.third_party.MotionAGFormer.MotionAGFormer import MotionAGFormer

from .BASE import BASE_CLASS


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


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
    def __init__(self, num_frame, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.tte_block = Block( dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=False, currentdim=0, depth=depth)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(num_frame, embed_dim_ratio))     
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Temporal_norm = norm_layer(embed_dim_ratio)
    def forward(self, x):
        b, f, n, c  = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b*n, f, c)                                # x: (B*17,243,512)
        x += self.temporal_pos_embed.unsqueeze(0)
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


class JointRegressor(nn.Module):
    def __init__(self, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.):
        super(JointRegressor, self).__init__()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(num_joints * dim_rep, hidden_dim)
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.fc2 = nn.Linear(num_joints * dim_rep, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.head_joints = nn.Linear(hidden_dim, num_joints * 3)
        nn.init.xavier_uniform_(self.head_joints.weight, gain=0.01)

    def forward(self, feat):
        N, T, J, C = feat.shape
        NT = N * T
        feat = feat.reshape(N, T, -1)

        feat_pose = feat.reshape(NT, -1)  # (N*T, J*C)

        feat_pose = self.dropout(feat_pose)
        feat_pose = self.fc1(feat_pose)
        feat_pose = self.bn1(feat_pose)
        feat_pose = self.relu1(feat_pose)  # (NT, hidden_dim)

        feat_shape = feat.permute(0, 2, 1)  # (N, T, J*C) -> (N, J*C, T)
        feat_shape = self.pool2(feat_shape).reshape(N, -1)  # (N, J*C)

        feat_shape = self.dropout(feat_shape)
        feat_shape = self.fc2(feat_shape)
        feat_shape = self.bn2(feat_shape)
        feat_shape = self.relu2(feat_shape)  # (N, hidden_dim)

        pred_joints = self.head_joints(feat_pose)  # (NT, num_joints * 3)
        pred_joints = pred_joints.view(N, T, J, 3)  # (N, T, 17, 3)

        return pred_joints
    

class Skeleton_in_Context(BASE_CLASS):
    def __init__(self, args, num_frame=9, num_joints=17, in_chans=3, hidden_dim=32, depth=4, num_heads=8, mlp_ratio=2, 
                 qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=None):
        
        with open('lib/model/third_party/MotionAGFormer/model_config_base.json', 'r') as f:
            model_config = json.load(f)
        act_mapper = {'gelu': nn.GELU, 'relu': nn.ReLU}
        model_config['act_layer'] = act_mapper[model_config['act_layer']]
        # model_config['dim_in'] = in_chans + 3
        model_config['n_frames'] = num_frame * 2
        model_config['num_joints'] = 1
        model_config.update({'connections': {0: [0]}})

        hidden_dim = model_config['dim_feat']
        dim_rep = model_config['dim_rep']

        super().__init__()

        self.spatial_pos_embed = nn.ParameterDict({'query': nn.Parameter(torch.zeros(184)),
                                                'prompt': nn.Parameter(torch.zeros(184))})

        self.spatial_patch_to_embedding = nn.ModuleDict({'query': nn.Linear(184, hidden_dim),
                                                         'prompt': nn.Linear(184, hidden_dim)})


        self.MotionAGFormer = nn.ModuleDict({'query': MotionAGFormer(**model_config),
                                             'prompt': MotionAGFormer(**model_config)})

        self.head = nn.Sequential(
            nn.LayerNorm(dim_rep),
            nn.Linear(dim_rep , 184),
        )

    def encode_joint(self, x, key):     # x: [B,F,184]
        x += self.spatial_pos_embed[key].unsqueeze(0).unsqueeze(0)
        x = self.spatial_patch_to_embedding[key](x)   # [B,F,512]
        return x.unsqueeze(2)
       
    def convert_dict_to_tensor(self, data_dict, batch_size, num_frame):
        if not hasattr(self, 'size_info'):
            size_info = OrderedDict({k: v.shape[2:] for k, v in data_dict.items()})
            setattr(self, 'size_info', size_info)
        data_tensor = torch.cat([v.reshape(batch_size, num_frame, -1) for v in data_dict.values()], dim=-1)
        return data_tensor

    def convert_tensor_to_dict(self, data_tensor, batch_size, num_frame):
        """Converts a tensor back to a dictionary format, preserving original structure."""
        data_dict = {}
        start = 0

        # Iterate over the keys in size_info to maintain the original order
        for key, size in self.size_info.items():
            # Calculate the size for the current key
            current_size = size.numel()  # Get the last dimension size for the key
            # Reshape the tensor slice back to the original shape
            data_dict[key] = data_tensor[:, :, start:start + current_size].reshape(batch_size, num_frame, *size)
            start += current_size  # Move the start index for the next key
        assert start == data_tensor.size(-1), "The tensor was not completely consumed"

        return data_dict
        
    @staticmethod
    def prepare_motion(chunk_dict, dataset_name, task, joint_mask, frame_mask, dataset_args):
        return BASE_CLASS.prepare_motion(chunk_dict, dataset_name, task, joint_mask, frame_mask, dataset_args)

    @staticmethod
    def preprocess(chunk_dict):
        chunk_dict['joint2d'] = chunk_dict['joint2d'][..., :2]
        chunk_dict['joint2d'] = chunk_dict['joint2d'] - chunk_dict['joint2d'][..., 0:1, :]
        chunk_dict['joint3d'] = chunk_dict['joint3d'] - chunk_dict['joint3d'][..., 0:1, :]
        return chunk_dict

    def generate_eval_mask(self, task, chunk_dict):
        B, F = chunk_dict['joint3d'].shape[:2]
        T = F // 2
        if task == 'PE':
            chunk_dict['joint2d'][:, F:] = 0
            chunk_dict['joint3d'] = chunk_dict['joint3d'] * 0
            chunk_dict['smpl_pose'] = chunk_dict['smpl_pose'] * 0
            chunk_dict['smpl_shape'] = chunk_dict['smpl_shape'] * 0



    

    def forward(self, query_chunk_dict,
                      prompt_chunk_dict, info_dict, task, epoch=None, vertex_x1000=False, deactivate_prompt_branch=False):
        B, T = query_chunk_dict['joint3d'].shape[:2]
        query_chunk = self.convert_dict_to_tensor(query_chunk_dict, B, T)       # [B,F,184]
        prompt_chunk = self.convert_dict_to_tensor(prompt_chunk_dict, B, T)     # [B,F,184]
        C = query_chunk.shape[-1]

        
        if self.training:
            mask = torch.rand_like(B, T, C, device=query_chunk.device) > 0.5
            # mask_T = torch.rand_like(1, T, 1, device=query_chunk.device) > 0.5
            query_chunk = query_chunk * mask
            prompt_chunk = prompt_chunk * mask

        

        # PROMPT BRANCH
        prompt = self.encode_joint(prompt_chunk, 'prompt')      # [B,F,1,512]
        query = self.encode_joint(query_chunk, 'query')         # [B,F,1,512]
        feat = prompt + query

        for layer_p, layer_q in zip(self.MotionAGFormer['prompt'].layers, self.MotionAGFormer['query'].layers):
            feat_p = layer_p(feat)
            feat_q = layer_q(feat)
            feat = feat_q + feat_p
        
        query_recon = self.MotionAGFormer['query'].norm(feat_q)
        query_recon = self.MotionAGFormer['query'].rep_logit(query_recon)
        query_recon = self.head(query_recon)      # [B,F,1,184]
        query_recon_tensor = query_recon.squeeze(2)    # [B,F,184]
        query_recon_dict = self.convert_tensor_to_dict(query_recon_tensor, B, T)
        # 'joint2d': [B,32,17,3]
        # 'joint3d': [B,32,17,3]
        # 'smpl_pose': [B,32,72]
        # 'smpl_shape': [B,32,10]

        prompt_recon = self.MotionAGFormer['prompt'].norm(feat_p)
        prompt_recon = self.MotionAGFormer['prompt'].rep_logit(prompt_recon)
        prompt_recon = self.head(prompt_recon)      # [B,F,1,184]
        prompt_recon_tensor = prompt_recon.squeeze(2)    # [B,F,184]
        prompt_recon_dict = self.convert_tensor_to_dict(prompt_recon_tensor, B, T)

        # query_recon_dict['theta'] = torch.cat([query_recon_dict.pop('smpl_pose'),
        #                                        query_recon_dict.pop('smpl_shape')], dim=-1)
        # prompt_recon_dict['theta'] = torch.cat([prompt_recon_dict.pop('smpl_pose'),
        #                                        prompt_recon_dict.pop('smpl_shape')], dim=-1)
        # query_recon_dict['kp_3d'] = query_recon_dict.pop('joint3d')
        # prompt_recon_dict['kp_3d'] = prompt_recon_dict.pop('joint3d')        
        # query_recon_dict['kp_2d'] = query_recon_dict.pop('joint2d')
        # prompt_recon_dict['kp_2d'] = prompt_recon_dict.pop('joint2d')        
        return query_recon_tensor, prompt_recon_tensor
