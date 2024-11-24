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

from .context_head import ActionHeadClassification


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

        if args.get('use_context', None) == 'post_attach_context_head':
            print('[Use context:? Yes] [How? Post-attach context head')
            self.context_head = ActionHeadClassification(dropout_ratio=args.Context.class_head.dropout_ratio, 
                                                         dim_in=args.Context.class_head.dim_in,
                                                         dim_rep=args.Context.class_head.dim_rep, 
                                                         num_classes=args.Context.class_head.num_class, 
                                                         num_joints=args.Context.class_head.num_joints, 
                                                         hidden_dim=args.Context.class_head.hidden_dim)
        
        if args.get('apply_attnmask', False):
            self.apply_attnmask = True


    def joint_head(self, x):
        return torch.einsum('jv,btvc->btjc', self.V2J_Regressor.to(x.device), x)  # [B,T,6890,3] --> [B,T,17,3]

    def encode_joint(self, x_in, x_out, key):
        x = torch.cat([x_in, x_out], dim=-1)
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
                      query_target_dict, prompt_target_dict, info_dict, epoch=None, vertex_x1000=False, deactivate_prompt_branch=False,
                      return_context=False):
        
        
        B, T, J, C = query_input_tensor.shape
        query_input = query_input_tensor        # [B,T,24,3]
        prompt_input = prompt_input_tensor      # [B,T,24,3]
        prompt_target = prompt_target_tensor    # [B,T,24,3]

        if self.fully_connected_graph:
            batched_spatial_adj = torch.ones(B,T,J,J).to(query_input.device)
            if self.apply_attnmask:
                temporal_mask = input_mask['temporal']    # (B,T)
                adj_eye = torch.eye(J)
                adj_allone = torch.ones(J,J)
                adj_set = torch.stack([adj_eye, adj_allone], dim=0).to(query_input.device).unsqueeze(0).expand(B,-1,-1,-1) # (2,J,J) -> (B,2,J,J)
                batched_spatial_adj = adj_set.gather(1, temporal_mask.long().unsqueeze(-1).unsqueeze(-1).expand(-1,-1,J,J)) # (B,2,J,J) (B,T,J,J) --> (B,T,J,J)

                spatial_mask = input_mask['spatial']    # (B,J)
                spatial_adj_mask = torch.einsum('bik,bkj->bij', spatial_mask.unsqueeze(-1), spatial_mask.unsqueeze(-2))    # (B,J,J)
                for j in range(J): spatial_adj_mask[:, j, j] = 1

                batched_spatial_adj = batched_spatial_adj * spatial_adj_mask.unsqueeze(1).expand(-1,T,-1,-1).to(query_input.device)     # (B,T,J,J)

                # 找到COCO数据集的batch_id: np.where(np.array(info_dict['dataset'])=='COCO')[0]
                # 找到MeshCompletion任务的batch_id: np.where(np.array(info_dict['task']) == 'MeshCompletion')[0]
                # 找到MeshCompletion / MeshPred / MeshInBetween任务的batch_id: 
                #   np.concatenate([ np.where(np.array(info_dict['task']) == 'MeshCompletion')[0],
                #                    np.where(np.array(info_dict['task']) == 'MeshPred')[0],
                #                    np.where(np.array(info_dict['task']) == 'MeshInBetween')[0]
                #                 ])

                ## Run this if you want to double-check if batched_spatial_adj is correctly constructed
                # for b in range(B):
                #     if (info_dict['dataset'][b] == 'COCO') and (info_dict['task'][b] in ['PE', 'FPE', 'MC', 'MP', 'MIB']):  # 是COCO数据集; joint任务
                #         assert (batched_spatial_adj[b, 1:] == torch.eye(J).unsqueeze(0).to(query_input.device)).all()
                #         spatial_mask_ = input_mask['spatial'][b]    # (J,)
                #         spatial_adj_mask_ = torch.einsum('ik,kj->ij', spatial_mask_.unsqueeze(-1), spatial_mask_.unsqueeze(-2))    # (J,J)
                #         for j in range(J): spatial_adj_mask_[j, j] = 1
                #         assert (batched_spatial_adj[b, 0] == spatial_adj_mask_.to(query_input.device)).all()
                #     elif (info_dict['dataset'][b] == 'COCO'):   # 是COCO数据集; mesh任务
                #         assert (batched_spatial_adj[b, 0] == 1).all()
                #         assert (batched_spatial_adj[b, 1:] == torch.eye(J).unsqueeze(0).to(query_input.device)).all()
                #     elif (info_dict['task'][b] in ['PE', 'FPE', 'MC', 'MP', 'MIB']): # 不是COCO数据集; joint任务
                #         spatial_mask_ = input_mask['spatial'][b]    # (J,)
                #         spatial_adj_mask_ = torch.einsum('ik,kj->ij', spatial_mask_.unsqueeze(-1), spatial_mask_.unsqueeze(-2))    # (J,J)
                #         for j in range(J): spatial_adj_mask_[j, j] = 1
                #         assert (batched_spatial_adj[b] == spatial_adj_mask_.unsqueeze(0).to(query_input.device)).all()
                #     else:   # 不是COCO数据集; mesh任务
                #         assert (batched_spatial_adj[b] == 1).all()
                    
        else:    
            batched_spatial_adj = None

        # PROMPT BRANCH
        prompt = self.encode_joint(prompt_input, prompt_target, 'prompt')       # [B,T,24,3] || [B,T,24,3] --> [B,T,24,6] --> [B,T,24,512]
        PROMPTS = (prompt,)

        for layer in self.MotionAGFormer['prompt'].layers:
            prompt = layer(prompt, batched_spatial_adj=batched_spatial_adj, input_mask=input_mask if self.apply_attnmask else None)    # TODO: design spatial adj
            PROMPTS = PROMPTS + (prompt,)


        # QUERY BRANCH
        query = self.encode_joint(query_input, prompt_target, 'query')     # [B,T,24,3] || [B,T,24,3] --> [B,T,24,6] --> [B,T,24,512]
        query += PROMPTS[0]

        for i, layer in enumerate(self.MotionAGFormer['query'].layers):
            query = layer(query, batched_spatial_adj=batched_spatial_adj, input_mask=input_mask if self.apply_attnmask else None)
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

        output_joint = self.joint_head(output_smpl[-1]['verts'])    # [B,T,17,3]
        if vertex_x1000:
            output_joint = output_joint / 1000


        target_joint, target_smpl = self.get_target_joint_and_smpl(query_target_dict, vertex_x1000)

        if self.apply_attnmask:
            output_joint = output_joint * input_mask['temporal'].unsqueeze(-1).unsqueeze(-1) + target_joint.clone() * (1 - input_mask['temporal']).unsqueeze(-1).unsqueeze(-1)
            for s in output_smpl:
                s['theta'] = s['theta'] * input_mask['temporal'].unsqueeze(-1) + target_smpl['theta'].clone() * (1 - input_mask['temporal']).unsqueeze(-1)
                s['verts'] = s['verts'] * input_mask['temporal'].unsqueeze(-1).unsqueeze(-1) + target_smpl['verts'].clone() * (1 - input_mask['temporal']).unsqueeze(-1).unsqueeze(-1)
                s['kp_3d'] = s['kp_3d'] * input_mask['temporal'].unsqueeze(-1).unsqueeze(-1) + target_smpl['kp_3d'].clone() * (1 - input_mask['temporal']).unsqueeze(-1).unsqueeze(-1)


        if return_context:
            context_input = torch.cat([output_joint, query_input_tensor], dim=-2)
            # context_input: [b,16,41,3]
            context_output = self.context_head(context_input)
            if self.training:
                return {'query': (output_joint, output_smpl, target_joint, target_smpl, context_output)}
            else:
                return output_joint, output_smpl, target_joint, target_smpl, context_output


        if self.training:
            return {'query': (output_joint, output_smpl, target_joint, target_smpl)}
        else:
            return output_joint, output_smpl, target_joint, target_smpl

    def context_head_forward(self, context_input):
        return self.context_head(context_input)