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

        self.spatial_pos_embed = nn.ParameterDict({
            'query': nn.Parameter(torch.zeros(num_joints, hidden_dim)),
            'prompt_input': nn.Parameter(torch.zeros(num_joints, hidden_dim)),
            'prompt_target': nn.Parameter(torch.zeros(num_joints, hidden_dim)),
            })

        self.spatial_patch_to_embedding = nn.ModuleDict({
            'query': nn.Linear(in_chans, hidden_dim),
            'prompt_input': nn.Linear(in_chans, hidden_dim),
            'prompt_target': nn.Linear(in_chans, hidden_dim),
            })

        self.MotionAGFormer = nn.ModuleDict({
            'query': MotionAGFormer(**model_config),
            'prompt_input': MotionAGFormer(**model_config),
            'prompt_target': MotionAGFormer(**model_config),
            })

        self.smpl_head = SMPLRegressor(data_root='third_party/motionbert/data/mesh', dim_rep=dim_rep, num_joints=num_joints, hidden_dim=1024, dropout_ratio=0.1)
        
        # self.joint_head = JointRegressor(dim_rep=dim_rep, num_joints=num_joints, hidden_dim=1024, dropout_ratio=0.1)
        self.V2J_Regressor = nn.Parameter(self.smpl_head.smpl.J_regressor_h36m.clone())    # [17,6890]

        self.fully_connected_graph = args.fully_connected_graph

    def joint_head(self, x):
        return torch.einsum('jv,btvc->btjc', self.V2J_Regressor.to(x.device), x)  # [B,T,6890,3] --> [B,T,17,3]

    def encode_joint(self, x, key):
        x = self.spatial_patch_to_embedding[key](x)   # [B,T,17,512]
        x += self.spatial_pos_embed[key].unsqueeze(0).unsqueeze(0)
        return x
    
    @staticmethod
    def get_target_joint_and_smpl(target_dict, vertex_x1000):
        target_joint = target_dict['joint'].clone()
        target_smpl = {}
        target_smpl['theta'] = torch.cat([target_dict['smpl_pose'], target_dict['smpl_shape']], dim=-1).clone()
        if vertex_x1000:
            target_smpl['verts'] = target_dict['smpl_vertex'].clone() * 1000
            target_smpl['kp_3d'] = target_dict['joint'].clone() * 1000
        else:
            target_smpl['verts'] = target_dict['smpl_vertex'].clone()
            target_smpl['kp_3d'] = target_dict['joint'].clone()
        return target_joint, target_smpl

    def forward(self, query_input_tensor, prompt_input_tensor, input_mask,
                      query_target_tensor, prompt_target_tensor,
                      query_target_dict, prompt_target_dict, info_dict, epoch=None, vertex_x1000=False, deactivate_prompt_branch=False):
        
        
        B, T, J, C = query_input_tensor.shape
        query_input = query_input_tensor        # [B,T,24,3]
        prompt_input = prompt_input_tensor      # [B,T,24,3]
        prompt_target = prompt_target_tensor    # [B,T,24,3]

        if self.fully_connected_graph:
            batched_spatial_adj = torch.ones(B,T,J,J).to(query_input.device)
        else:
            batched_spatial_adj = None


        # PROMPT INPUT BRANCH
        prompt_input = self.encode_joint(prompt_input, 'prompt_input')       # [B,T,24,3] || [B,T,24,3] --> [B,T,24,6] --> [B,T,24,512]
        PROMPTS_INPUT = (prompt_input,)
        for layer in self.MotionAGFormer['prompt_input'].layers:
            prompt_input = layer(prompt_input, batched_spatial_adj=batched_spatial_adj)    # TODO: design spatial adj
            PROMPTS_INPUT = PROMPTS_INPUT + (prompt_input,)

        # PROMPT TARGET BRANCH
        prompt_target = self.encode_joint(prompt_target, 'prompt_target')       # [B,T,24,3] || [B,T,24,3] --> [B,T,24,6] --> [B,T,24,512]
        prompt_target += PROMPTS_INPUT[0]
        PROMPTS_TARGET = (prompt_target,)
        for i, layer in enumerate(self.MotionAGFormer['prompt_target'].layers):
            prompt_target = layer(prompt_target, batched_spatial_adj=batched_spatial_adj)    # TODO: design spatial adj
            prompt_target += PROMPTS_INPUT[i+1]
            PROMPTS_TARGET = PROMPTS_TARGET + (prompt_target,)



        # QUERY BRANCH
        query = self.encode_joint(query_input, 'query')     # [B,T,24,3] || [B,T,24,3] --> [B,T,24,6] --> [B,T,24,512]
        query += PROMPTS_TARGET[0]

        for i, layer in enumerate(self.MotionAGFormer['query'].layers):
            query = layer(query, batched_spatial_adj=batched_spatial_adj)
            query += PROMPTS_TARGET[i+1]
        
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

        output_joint = self.joint_head(output_smpl[-1]['verts'])    # [B,T,17,3]
        if vertex_x1000:
            output_joint = output_joint / 1000

        target_joint, target_smpl = self.get_target_joint_and_smpl(query_target_dict, vertex_x1000)
        if self.training:
            return {'query': (output_joint, output_smpl, target_joint, target_smpl)}
        else:
            return output_joint, output_smpl, target_joint, target_smpl
