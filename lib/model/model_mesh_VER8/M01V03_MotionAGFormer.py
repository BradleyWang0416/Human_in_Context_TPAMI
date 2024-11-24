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


class Skeleton_in_Context(nn.Module):
    def __init__(self, args, num_frame=9, num_joints=17, in_chans=3, hidden_dim=32, depth=4, num_heads=8, mlp_ratio=2, 
                 qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=None):
        
        with open('lib/model/third_party/MotionAGFormer/model_config_base.json', 'r') as f:
            model_config = json.load(f)
        act_mapper = {'gelu': nn.GELU, 'relu': nn.ReLU}
        model_config['act_layer'] = act_mapper[model_config['act_layer']]
        # model_config['dim_in'] = in_chans + 3
        model_config['n_frames'] = num_frame

        model_config['num_joints'] = 24
        num_joints = 24
        # model_config.update({'connections': {0: [0]}})

        hidden_dim = model_config['dim_feat']
        dim_rep = model_config['dim_rep']

        super().__init__()

        self.spatial_pos_embed = nn.ParameterDict({'query': nn.Parameter(torch.zeros(num_joints, hidden_dim)),
                                                'prompt': nn.Parameter(torch.zeros(num_joints, hidden_dim))})

        self.spatial_patch_to_embedding = nn.ModuleDict({'query': nn.Linear(in_chans+3, hidden_dim),
                                                         'prompt': nn.Linear(in_chans+3, hidden_dim)})

        self.MotionAGFormer = nn.ModuleDict({'query': MotionAGFormer(**model_config),
                                             'prompt': MotionAGFormer(**model_config)})

        self.smpl_head = SMPLRegressor(data_root='third_party/motionbert/data/mesh', dim_rep=dim_rep, num_joints=num_joints, hidden_dim=1024, dropout_ratio=0.1)
        
        # self.joint_head = JointRegressor(dim_rep=dim_rep, num_joints=num_joints, hidden_dim=1024, dropout_ratio=0.1)
        self.V2J_Regressor = nn.Parameter(self.smpl_head.smpl.J_regressor_h36m.clone())    # [17,6890]

        self.fully_connected_graph = args.fully_connected_graph

        self.joint_head = nn.Sequential(
            nn.LayerNorm(dim_rep),
            nn.Linear(dim_rep, 3),
        )

    # def joint_head(self, x):
    #     return torch.einsum('jv,btvc->btjc', self.V2J_Regressor.to(x.device), x)  # [B,T,6890,3] --> [B,T,17,3]

    def encode_joint(self, x_in, x_out, key):
        x = torch.cat([x_in, x_out], dim=-1)
        x = self.spatial_patch_to_embedding[key](x)   # [B,T,17,512]
        x += self.spatial_pos_embed[key].unsqueeze(0).unsqueeze(0)
        return x

    def forward(self, query_sample_dict, prompt_sample_dict,
                      epoch=None, vertex_x1000=False, deactivate_prompt_branch=False):
        # 如果使用多显卡, 则info_dict中的value的长度不等于query_sample_dict的长度, 因为info_dict不会被分配到各显卡上
        
        query_input = query_sample_dict['input_tensor']         # [B, 16, 24, 3]
        prompt_input = prompt_sample_dict['input_tensor']       # [B, 16, 24, 3]
        prompt_target = prompt_sample_dict['target_tensor']     # [B, 16, 24, 3]
        query_input_mask = query_sample_dict['input_mask']      # [B, 24]
        prompt_input_mask = prompt_sample_dict['input_mask']    # [B, 24]
        prompt_target_mask = prompt_sample_dict['target_mask']  # [B, 24]
        
        B, T, J, C = query_input.shape

        if self.fully_connected_graph:
            batched_spatial_adj = torch.ones(B,T,J,J).to(query_input.device)
        else:    
            batched_spatial_adj = None

        # PROMPT BRANCH
        prompt = self.encode_joint(prompt_input, prompt_target, 'prompt')       # [B,T,24,3] || [B,T,24,3] --> [B,T,24,6] --> [B,T,24,512]
        PROMPTS = (prompt,)

        for layer in self.MotionAGFormer['prompt'].layers:
            prompt = layer(prompt, batched_spatial_adj=batched_spatial_adj)    # TODO: design spatial adj
            PROMPTS = PROMPTS + (prompt,)


        # QUERY BRANCH
        query = self.encode_joint(query_input, prompt_target, 'query')     # [B,T,24,3] || [B,T,24,3] --> [B,T,24,6] --> [B,T,24,512]
        query += PROMPTS[0]

        for i, layer in enumerate(self.MotionAGFormer['query'].layers):
            query = layer(query, batched_spatial_adj=batched_spatial_adj)
            query += PROMPTS[1+i]
        
        query = self.MotionAGFormer['query'].norm(query)
        query = self.MotionAGFormer['query'].rep_logit(query)
        # [B,T,24,512]

        # MERGE
        output_smpl = self.smpl_head(query, vertex_x1000=vertex_x1000)
        for s in output_smpl:
            s['theta'] = s['theta'].reshape(B, T, -1)
            s['verts'] = s['verts'].reshape(B, T, -1, 3)
            s['kp_3d'] = s['kp_3d'].reshape(B, T, -1, 3)
        # [
        #   {
        #       'theta': [B,T,82],
        #       'verts': [B,T,6890,3],
        #       'kp_3d': [B,T,17,3]
        #   }
        # ]

        # output_joint = self.joint_head(output_smpl[-1]['verts'])    # [B,T,17,3]
        # if vertex_x1000:
        #     output_joint = output_joint / 1000
        output_joint = self.joint_head(query)    # [B,T,24,3]

        return output_joint[:, :, :17, :], output_smpl
