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

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def make_zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        B, T, D = h.shape
        emb = emb.view(B, T, D)
        # B, 1, 2D
        emb_out = self.emb_layers(emb)[:,0:1,:]
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h

class TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb=None):                                                                  # x: (bs*17,243,512)
        """                                                                                         # xf: (bs,77,512)
        x: B, T, D                                                                                  # emb: (bs*243,17,512)
        xf: B, N, L
        """
        B, T, D = x.shape                                                                           # B=bs*17
        N = xf.shape[1]                                                                             # N=77
        H = self.num_head                                                                           # H=8
        query = self.query(self.norm(x)).unsqueeze(2)                                               # query: (bs*17, 243, 1, 512)
        key = self.key(self.text_norm(xf)).unsqueeze(1)                                             # key: (bs, 1, 77, 512)
        key = key.repeat(int(B/key.shape[0]), 1, 1, 1)                                              # key: (bs*17, 1, 77, 512)  <--repeat-- (bs, 1, 77, 512)
        query = query.view(B, T, H, -1)                                                             # query: (bs*17, 243, num_head, 64)
        key = key.view(B, N, H, -1)                                                                 # key: (bs*17, 77, num_head, 64)

        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)                 # attention: (bs*17, 243, 77, num_head).  (8,243,64)*(8,64,77)->(8,243,77)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.text_norm(xf)).unsqueeze(1)
        value = value.repeat(int(B/value.shape[0]), 1, 1, 1)   
        value = value.view(B, N, H, -1)                                                             # value: (bs*17, 77, num_head, 64)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)                         # y: (bs*17,243,512).  (8,243,77)*(8,77,64)->(8,243,64)
        if emb is not None:
            y = x + self.proj_out(y, emb)
            return y
        else:
            y = x + y
            return y


class TextPromptCrossAttention(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(text_latent_dim, text_latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, xf, x):                                                                       # x: (bs*17,243,512)
        """                                                                                         # xf: (bs,77,512)
        x: B, T, D                                                                                  # emb: (bs*243,17,512)
        xf: B, N, L
        """
        B, T, D = x.shape                                                                           # B=bs*17
        N = xf.shape[1]                                                                             # N=77
        H = self.num_head                                                                           # H=8

        query = self.query(self.text_norm(xf))                                                      # query: (bs,77,512)
        query = query.repeat(int(B/query.shape[0]), 1, 1)                                           # query: (bs*17,77,512)

        key = self.key(self.norm(x))                                                                # key: (bs*17,243,512)
        value = self.value(self.norm(x))                                                            # value: (bs*17,243,512)

        query = query.reshape(B,N,H,-1)                                                             # query: (bs*17,77,num_head,64)
        key = key.reshape(B,T,H,-1)                                                                 # key: (bs*17,243,num_head,64)
        value = value.reshape(B,T,H,-1)                                                             # value: (bs*17,243,num_head,64)

        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)                 # attention: (bs*17, 77, 243, num_head).  (8,77,64)*(8,243,64)->(8,77,243)
        weight = self.dropout(F.softmax(attention, dim=2))
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, N, D)                         # y: (bs*17,243,512).  (8,77,243)*(8,243,64)->(8,77,64)
        return y
    

class  MixSTE2(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, is_train=True, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio
        out_dim = 3
        self.is_train=is_train

        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans + 3, embed_dim_ratio)

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))


        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))


        self.pos_drop = nn.Dropout(p=drop_rate)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
            nn.GELU(),
            nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
        )


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth

        self.STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)


        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )

        self.temporal_cross_attn = TemporalCrossAttention(512, 512, num_heads, drop_rate, 512)
        self.text_pre_proj = nn.Identity()
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=drop_rate,
            activation="gelu")
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=4)
        self.text_ln = nn.LayerNorm(512)
        self.text_proj = nn.Sequential(
            nn.Linear(512, 512)
        )

        

        self.clip_text, _ = clip.load('ViT-B/32', "cpu")
        set_requires_grad(self.clip_text, False)

        self.remain_len = 4


        ctx_vectors_subject = torch.empty((7-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_subject, std=0.02)
        self.ctx_subject = nn.Parameter(ctx_vectors_subject)

        ctx_vectors_verb = torch.empty((12-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_verb, std=0.02)
        self.ctx_verb = nn.Parameter(ctx_vectors_verb)

        ctx_vectors_speed = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_speed, std=0.02)
        self.ctx_speed = nn.Parameter(ctx_vectors_speed)

        ctx_vectors_head = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_head, std=0.02)
        self.ctx_head = nn.Parameter(ctx_vectors_head)
        
        ctx_vectors_body = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_body, std=0.02)
        self.ctx_body = nn.Parameter(ctx_vectors_body)
        
        ctx_vectors_arm = torch.empty((14-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_arm, std=0.02)
        self.ctx_arm = nn.Parameter(ctx_vectors_arm)

        ctx_vectors_leg = torch.empty((14-self.remain_len), 512, dtype=self.clip_text.dtype)
        nn.init.normal_(ctx_vectors_leg, std=0.02)
        self.ctx_leg = nn.Parameter(ctx_vectors_leg)

                                                                                    # x_2d: (B,243,17,2). x_3d: (B,243,17,3). 
    def STE_forward(self, x_2d, x_3d, t, xf_proj):                                  # t=tensor([ 32, 232, 920, 825], device='cuda:0')
                                                                                    # xf_proj: (B,512)
        if self.is_train:
            x = torch.cat((x_2d, x_3d), dim=-1)                                     # x: (B,243,17,5) <--- x_2d: (B,243,17,2) || x_3d: (B,243,17,3)
            b, f, n, c = x.shape
            x = rearrange(x, 'b f n c  -> (b f) n c', )                             # x: (B*243, 17, 5)
            x = self.Spatial_patch_to_embedding(x)                                  # x: (B*243, 17, 512)
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, :]
            xf_proj = xf_proj.view(xf_proj.shape[0], 1, 1, xf_proj.shape[1])
            time_embed = time_embed + xf_proj
            time_embed = time_embed.repeat(1, f, n, 1)
            time_embed = rearrange(time_embed, 'b f n c  -> (b f) n c', )
            x += time_embed                                                         # x: (B*243, 17, 512)
        else:
            x_2d = x_2d[:,None].repeat(1,x_3d.shape[1],1,1,1)
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, h, f, n, c = x.shape
            x = rearrange(x, 'b h f n c  -> (b h f) n c', )
            x = self.Spatial_patch_to_embedding(x)
            x += self.Spatial_pos_embed
            time_embed = self.time_mlp(t)[:, None, None, None, :]
            xf_proj = xf_proj.view(xf_proj.shape[0], 1, 1, 1, xf_proj.shape[1])
            time_embed = time_embed + xf_proj
            time_embed = time_embed.repeat(1, h, f, n, 1)
            time_embed = rearrange(time_embed, 'b h f n c  -> (b h f) n c', )
            x += time_embed

        x = self.pos_drop(x)                                                        # x: (B*243, 17, 512)

        blk = self.STEblocks[0]                                                     # self.STEblocks[0]: 从self.STEblocks (包含8个attention block) 中取出第一个block
        x = blk(x)                                                                  # x: (B*243, 17, 512)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)                           # x: (B*17, 243, 512)
        return x, time_embed                                                        # time_embed: (B*243, 17, 512)

    def TTE_foward(self, x):                                                        # x: (B*17, 243, 512)
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]                                                     # self.TTEblocks[0]: 从self.TTEblocks (包含8个attention block) 中取出第一个block
        x = blk(x)

        x = self.Temporal_norm(x)                                                   # x: (B*17,243,512)
        return x

    def ST_foward(self, x, control_x=None):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

            if control_x is not None:
                x = x + control_x[i-1].permute(0,2,3,1)
        
        return x
    
    def encode_text(self, text, pre_text_tensor):                                                                                           # text: (B,77). 代表动作类别词汇, 如"WalkDog"
        with torch.no_grad():                                                                                                               # pre_text_tensor: (B,6,77). 6代表"A person", "speed", "head", "body", "arm", "leg"六个词汇
            x = self.clip_text.token_embedding(text).type(self.clip_text.dtype)                                                             # x: (B,77,512)
            pre_text_tensor = self.clip_text.token_embedding(pre_text_tensor).type(self.clip_text.dtype)                                    # pre_text_tensor: (B,6,77,512)

        learnable_prompt_subject = self.ctx_subject                                                                                         # learnable_prompt_subject: (3,512)
        learnable_prompt_subject = learnable_prompt_subject.view(1, self.ctx_subject.shape[0], self.ctx_subject.shape[1])                   # learnable_prompt_subject: (1,3,512)
        learnable_prompt_subject = learnable_prompt_subject.repeat(x.shape[0], 1, 1)                                                        # learnable_prompt_subject: (B,3,512)
        learnable_prompt_subject = torch.cat((learnable_prompt_subject, pre_text_tensor[:, 0, :self.remain_len, :]), dim=1)                 # learnable_prompt_subject: (B,7,512). 取出pre_text_tensor中的"A person"对应的embedding然后只保留77维的前4维 ([B,6,77,512]->[B,77,512]->[B,4,512]), 然后和learnable_prompt_subject拼接起来

        learnable_prompt_verb = self.ctx_verb                                                                                               # learnable_prompt_verb: (8,512)
        learnable_prompt_verb = learnable_prompt_verb.view(1, self.ctx_verb.shape[0], self.ctx_verb.shape[1])
        learnable_prompt_verb = learnable_prompt_verb.repeat(x.shape[0], 1, 1)
        learnable_prompt_verb = torch.cat((learnable_prompt_verb, x[:, :self.remain_len, :]), dim=1)                                        # learnable_prompt_verb: (B,12,512). 12由8和4拼接而来

        learnable_prompt_speed = self.ctx_speed                                                                                             # learnable_prompt_speed: (6,512)
        learnable_prompt_speed = learnable_prompt_speed.view(1, self.ctx_speed.shape[0], self.ctx_speed.shape[1])
        learnable_prompt_speed = learnable_prompt_speed.repeat(x.shape[0], 1, 1)
        learnable_prompt_speed = torch.cat((learnable_prompt_speed, pre_text_tensor[:, 1, :self.remain_len, :]), dim=1)                     # learnable_prompt_speed: (B,10,512). 10=6||4

        learnable_prompt_head = self.ctx_head
        learnable_prompt_head = learnable_prompt_head.view(1, self.ctx_head.shape[0], self.ctx_head.shape[1])
        learnable_prompt_head = learnable_prompt_head.repeat(x.shape[0], 1, 1)
        learnable_prompt_head = torch.cat((learnable_prompt_head, pre_text_tensor[:, 2, :self.remain_len, :]), dim=1)                       # learnable_prompt_head: (B,10,512)

        learnable_prompt_body = self.ctx_body
        learnable_prompt_body = learnable_prompt_body.view(1, self.ctx_body.shape[0], self.ctx_body.shape[1])
        learnable_prompt_body = learnable_prompt_body.repeat(x.shape[0], 1, 1)
        learnable_prompt_body = torch.cat((learnable_prompt_body, pre_text_tensor[:, 3, :self.remain_len, :]), dim=1)                       # learnable_prompt_body: (B,10,512)

        learnable_prompt_arm = self.ctx_arm
        learnable_prompt_arm = learnable_prompt_arm.view(1, self.ctx_arm.shape[0], self.ctx_arm.shape[1])
        learnable_prompt_arm = learnable_prompt_arm.repeat(x.shape[0], 1, 1)
        learnable_prompt_arm = torch.cat((learnable_prompt_arm, pre_text_tensor[:, 4, :self.remain_len, :]), dim=1)                         # learnable_prompt_arm: (B,14,512)

        learnable_prompt_leg = self.ctx_leg
        learnable_prompt_leg = learnable_prompt_leg.view(1, self.ctx_leg.shape[0], self.ctx_leg.shape[1])
        learnable_prompt_leg = learnable_prompt_leg.repeat(x.shape[0], 1, 1)
        learnable_prompt_leg = torch.cat((learnable_prompt_leg, pre_text_tensor[:, 5, :self.remain_len, :]), dim=1)                         # learnable_prompt_leg: (B,14,512)

        x = torch.cat((learnable_prompt_subject, learnable_prompt_verb, learnable_prompt_speed, learnable_prompt_head, learnable_prompt_body, learnable_prompt_arm, learnable_prompt_leg), dim=1)
                                                                                                                                            # x: (B,77,512)
        with torch.no_grad():
            x = x + self.clip_text.positional_embedding.type(self.clip_text.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND                                                                                            # x: (77,B,512)
            x = self.clip_text.transformer(x)
            x = self.clip_text.ln_final(x).type(self.clip_text.dtype)

        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)                                                                                                       # xf_out: (77,B,512)
        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])                                                # xf_proj: (B,512)
        # B, T, D
        xf_out = xf_out.permute(1, 0, 2)                                                                                                    # xf_out: (B,77,512)
        return xf_proj, xf_out

    def forward(self, x_2d, x_3d, t, text, pre_text_tensor, return_rep=False):                  # text: (B,77). 代表动作类别词汇, 如"WalkDog"
        if self.is_train:                                                                       # pre_text_tensor: (B,6,77). 6代表"A person", "speed", "head", "body", "arm", "leg"六个词汇
            b, f, n, c = x_2d.shape                                                             # b=batch_size, f=243, n=17, c=2
        else:
            b, h, f, n, c = x_3d.shape
        
        xf_proj, xf_out = self.encode_text(text, pre_text_tensor)                               # xf_proj: (B,512)
                                                                                                # xf_out: (B,77,512). 即paper中的"fine-grained part-aware prompt P"

        x, time_embed = self.STE_forward(x_2d, x_3d, t, xf_proj)                                # x: (B*17, 243, 512). time_embed: (B*243, 17, 512)
                                                                                                    # STE_forward 中只用到了self.STEblocks中的第一个block, 相当于paper中的第1步: spatial transformer

        x = self.temporal_cross_attn(x, xf_out, time_embed)                                     # x: (B*17, 243, 512). 以x为Query, xf_out为Key和Value, 作cross attention,  相当于paper中的第2和3步: temporal cross attention
        x = self.TTE_foward(x)                                                                  # x: (B*17, 243, 512)
                                                                                                # TTE_foward 中只用到了self.TTEblocks中的第一个block, 相当于paper中的第4步: temporal transformer

        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)                                                                   # 在这里面用了其余的7个self.STEblocks和7个self.TTEblocks作spatial-temporal attention, 相当于paper中的第5步: spatial-temporal transformer

        if return_rep:
            return x
    
        x = self.head(x)



        if self.is_train:
            x = x.view(b, f, n, -1)
        else:
            x = x.view(b, h, f, n, -1)

        return x                                                                                # x: (B,243,17,3)


class  MixSTE2_PromptEnabled(MixSTE2):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2, 
                 qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=None, is_train=True, 
                 prompt_enabled=False, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None, prompt_gt_as_condition=False, use_text=True, fuse_prompt_query='add'):
        super().__init__(num_frame, num_joints, in_chans, embed_dim_ratio, depth, num_heads, mlp_ratio, 
                         qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer,
                         is_train, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        self.prompt_enabled = prompt_enabled
        print(f'[MODEL INFO] prompt_enabled: {prompt_enabled}')
        if prompt_enabled == 'v1':
            self.prompt_spatial_patch_to_embedding = nn.Linear(in_chans+3, embed_dim_ratio)
            self.prompt_spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
            self.prompt_time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(embed_dim_ratio),
                nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
                nn.GELU(),
                nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
            )
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
            self.prompt_encode_block = Block(
                            dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)

            self.Query_Prompt_temporal_cross_attn = TemporalCrossAttention(512, 512, num_heads, drop_rate, 512)
        
        if prompt_enabled == 'v2':
            self.get_mixste_rep = MixSTE2(num_frame, num_joints, in_chans, embed_dim_ratio, depth, num_heads, mlp_ratio, 
                         qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, is_train=True)
            ckpt = torch.load('checkpoint/pretrained_h36m/best_epoch_20_10.bin', map_location=lambda storage, loc: storage)
            mixste_ckpt = ckpt['model_pos']
            new_mixste_ckpt = {}
            for param_name, value in mixste_ckpt.items():
                if 'pose_estimator' in param_name:
                    new_mixste_ckpt[param_name.replace('module.pose_estimator.','')] = value
            self.get_mixste_rep.load_state_dict(new_mixste_ckpt, strict=True)
            for param in self.get_mixste_rep.parameters():
                param.requires_grad = False

            self.QueryPrompt_temporal_cross_attn = TemporalCrossAttention(512, 512, num_heads, drop_rate, 512)
        
        if prompt_enabled == 'v3':
            self.get_mixste_rep = MixSTE2(num_frame, num_joints, in_chans, embed_dim_ratio, depth, num_heads, mlp_ratio, 
                         qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, is_train=True)
            ckpt = torch.load('checkpoint/pretrained_h36m/best_epoch_20_10.bin', map_location=lambda storage, loc: storage)
            mixste_ckpt = ckpt['model_pos']
            new_mixste_ckpt = {}
            for param_name, value in mixste_ckpt.items():
                if 'pose_estimator' in param_name:
                    new_mixste_ckpt[param_name.replace('module.pose_estimator.','')] = value
            self.get_mixste_rep.load_state_dict(new_mixste_ckpt, strict=True)
            for param in self.get_mixste_rep.parameters():
                param.requires_grad = False

            self.QueryPrompt_temporal_cross_attn = TemporalCrossAttention(512, 512, num_heads, drop_rate, 512)
        
        if prompt_enabled == 'v4':
            prompt_ctx_vectors_subject = torch.empty((7-self.remain_len), 512, dtype=self.clip_text.dtype)
            nn.init.normal_(prompt_ctx_vectors_subject, std=0.02)
            self.prompt_ctx_subject = nn.Parameter(prompt_ctx_vectors_subject)

            prompt_ctx_vectors_verb = torch.empty((12-self.remain_len), 512, dtype=self.clip_text.dtype)
            nn.init.normal_(prompt_ctx_vectors_verb, std=0.02)
            self.prompt_ctx_verb = nn.Parameter(prompt_ctx_vectors_verb)

            prompt_ctx_vectors_speed = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
            nn.init.normal_(prompt_ctx_vectors_speed, std=0.02)
            self.prompt_ctx_speed = nn.Parameter(prompt_ctx_vectors_speed)

            prompt_ctx_vectors_head = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
            nn.init.normal_(prompt_ctx_vectors_head, std=0.02)
            self.prompt_ctx_head = nn.Parameter(prompt_ctx_vectors_head)
            
            prompt_ctx_vectors_body = torch.empty((10-self.remain_len), 512, dtype=self.clip_text.dtype)
            nn.init.normal_(prompt_ctx_vectors_body, std=0.02)
            self.prompt_ctx_body = nn.Parameter(prompt_ctx_vectors_body)
            
            prompt_ctx_vectors_arm = torch.empty((14-self.remain_len), 512, dtype=self.clip_text.dtype)
            nn.init.normal_(prompt_ctx_vectors_arm, std=0.02)
            self.prompt_ctx_arm = nn.Parameter(prompt_ctx_vectors_arm)

            prompt_ctx_vectors_leg = torch.empty((14-self.remain_len), 512, dtype=self.clip_text.dtype)
            nn.init.normal_(prompt_ctx_vectors_leg, std=0.02)
            self.prompt_ctx_leg = nn.Parameter(prompt_ctx_vectors_leg)

            self.prompt_spatial_patch_to_embedding = nn.Linear(in_chans+3, embed_dim_ratio)
            self.prompt_spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
            self.prompt_time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(embed_dim_ratio),
                nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
                nn.GELU(),
                nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
            )

            self.TextPrompt_temporal_cross_attn = TextPromptCrossAttention(512, 512, num_heads, drop_rate, 512)

        if prompt_enabled in ['v5', 'v6']:
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.control_STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

            self.control_TTEblocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
                for i in range(depth)])

            self.zero_conv_blocks = nn.ModuleList([
                make_zero_module(nn.Conv2d(embed_dim_ratio, embed_dim_ratio, kernel_size=1))
                for _ in range(depth+1)
               ])
            
            state_dict = {p_name: p_value for p_name, p_value in self.STEblocks.named_parameters()}
            self.control_STEblocks.load_state_dict(state_dict)
            state_dict = {p_name: p_value for p_name, p_value in self.TTEblocks.named_parameters()}
            self.control_TTEblocks.load_state_dict(state_dict)

            self.prompt_gt_as_condition = prompt_gt_as_condition

        if prompt_enabled in ['v7']:
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.control_STEblocks = nn.ModuleList([
            # Block: Attention Block
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

            self.control_TTEblocks = nn.ModuleList([
                Block(
                    dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, comb=False, changedim=False, currentdim=i+1, depth=depth)
                for i in range(depth)])

            self.zero_conv_blocks = nn.ModuleList([
                make_zero_module(nn.Conv2d(embed_dim_ratio, embed_dim_ratio, kernel_size=1))
                for _ in range(depth+1)
               ])
            
            state_dict = {p_name: p_value for p_name, p_value in self.STEblocks.named_parameters()}
            self.control_STEblocks.load_state_dict(state_dict)
            state_dict = {p_name: p_value for p_name, p_value in self.TTEblocks.named_parameters()}
            self.control_TTEblocks.load_state_dict(state_dict)

            self.prompt_gt_as_condition = prompt_gt_as_condition

            self.control_spatial_patch_to_embedding = nn.Linear(in_chans + 3, embed_dim_ratio)
            self.control_time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(embed_dim_ratio),
                nn.Linear(embed_dim_ratio, embed_dim_ratio*2),
                nn.GELU(),
                nn.Linear(embed_dim_ratio*2, embed_dim_ratio),
            )

    
    def control_STE_forward(self, x_2d, x_3d, t, xf_proj):                                  # t=tensor([ 32, 232, 920, 825], device='cuda:0')
                                                                                    # xf_proj: (B,512)
        if self.is_train:
            x = torch.cat((x_2d, x_3d), dim=-1)                                     # x: (B,243,17,5) <--- x_2d: (B,243,17,2) || x_3d: (B,243,17,3)
            b, f, n, c = x.shape
            x = rearrange(x, 'b f n c  -> (b f) n c', )                             # x: (B*243, 17, 5)
            x = self.control_spatial_patch_to_embedding(x)                                  # x: (B*243, 17, 512)
            x += self.Spatial_pos_embed
            time_embed = self.control_time_mlp(t)[:, None, None, :]
            xf_proj = xf_proj.view(xf_proj.shape[0], 1, 1, xf_proj.shape[1])
            time_embed = time_embed + xf_proj
            time_embed = time_embed.repeat(1, f, n, 1)
            time_embed = rearrange(time_embed, 'b f n c  -> (b f) n c', )
            x += time_embed                                                         # x: (B*243, 17, 512)
        else:
            x_2d = x_2d[:,None].repeat(1,x_3d.shape[1],1,1,1)
            x = torch.cat((x_2d, x_3d), dim=-1)
            b, h, f, n, c = x.shape
            x = rearrange(x, 'b h f n c  -> (b h f) n c', )
            x = self.control_spatial_patch_to_embedding(x)
            x += self.Spatial_pos_embed
            time_embed = self.control_time_mlp(t)[:, None, None, None, :]
            xf_proj = xf_proj.view(xf_proj.shape[0], 1, 1, 1, xf_proj.shape[1])
            time_embed = time_embed + xf_proj
            time_embed = time_embed.repeat(1, h, f, n, 1)
            time_embed = rearrange(time_embed, 'b h f n c  -> (b h f) n c', )
            x += time_embed

        x = self.pos_drop(x)                                                        # x: (B*243, 17, 512)

        blk = self.control_STEblocks[0]                                                     # self.STEblocks[0]: 从self.STEblocks (包含8个attention block) 中取出第一个block
        x = blk(x)                                                                  # x: (B*243, 17, 512)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)                           # x: (B*17, 243, 512)
        return x, time_embed                                                        # time_embed: (B*243, 17, 512)
    
    def control_TTE_foward(self, x):                                                        # x: (B*17, 243, 512)
        assert len(x.shape) == 3, "shape is equal to 3"
        b, f, _  = x.shape
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        blk = self.control_TTEblocks[0]                                                     # self.TTEblocks[0]: 从self.TTEblocks (包含8个attention block) 中取出第一个block
        x = blk(x)

        x = self.Temporal_norm(x)                                                   # x: (B*17,243,512)
        return x

    def control_ST_foward(self, x, return_mid_output=True):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape

        if return_mid_output:
            mid_output = ()


        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            steblock = self.control_STEblocks[i]
            tteblock = self.control_TTEblocks[i]
            
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)

            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

            if return_mid_output:
                mid_output = mid_output + (x.permute(0,3,1,2),)
        
        if return_mid_output:
            return x, mid_output
        return x

    
    def encode_text_w_prompt(self, text, pre_text_tensor):                                                                      # text: (B,77). 代表动作类别词汇, 如"WalkDog"
                                                                                                                                            # prompt_rep: (B*17, 243, 512)
        with torch.no_grad():                                                                                                               # pre_text_tensor: (B,6,77). 6代表"A person", "speed", "head", "body", "arm", "leg"六个词汇
            x = self.clip_text.token_embedding(text).type(self.clip_text.dtype)                                                             # x: (B,77,512)
            pre_text_tensor = self.clip_text.token_embedding(pre_text_tensor).type(self.clip_text.dtype)                                    # pre_text_tensor: (B,6,77,512)

        learnable_prompt_subject = self.prompt_ctx_subject                                                                                         # learnable_prompt_subject: (3,512)
        learnable_prompt_subject = learnable_prompt_subject.view(1, self.prompt_ctx_subject.shape[0], self.prompt_ctx_subject.shape[1])                   # learnable_prompt_subject: (1,3,512)
        learnable_prompt_subject = learnable_prompt_subject.repeat(x.shape[0], 1, 1)                                                        # learnable_prompt_subject: (B,3,512)
        learnable_prompt_subject = torch.cat((learnable_prompt_subject, pre_text_tensor[:, 0, :self.remain_len, :]), dim=1)                 # learnable_prompt_subject: (B,7,512). 取出pre_text_tensor中的"A person"对应的embedding然后只保留77维的前4维 ([B,6,77,512]->[B,77,512]->[B,4,512]), 然后和learnable_prompt_subject拼接起来

        learnable_prompt_verb = self.prompt_ctx_verb                                                                                               # learnable_prompt_verb: (8,512)
        learnable_prompt_verb = learnable_prompt_verb.view(1, self.prompt_ctx_verb.shape[0], self.prompt_ctx_verb.shape[1])
        learnable_prompt_verb = learnable_prompt_verb.repeat(x.shape[0], 1, 1)
        learnable_prompt_verb = torch.cat((learnable_prompt_verb, x[:, :self.remain_len, :]), dim=1)                                        # learnable_prompt_verb: (B,12,512). 12由8和4拼接而来

        learnable_prompt_speed = self.prompt_ctx_speed                                                                                             # learnable_prompt_speed: (6,512)
        learnable_prompt_speed = learnable_prompt_speed.view(1, self.prompt_ctx_speed.shape[0], self.prompt_ctx_speed.shape[1])
        learnable_prompt_speed = learnable_prompt_speed.repeat(x.shape[0], 1, 1)
        learnable_prompt_speed = torch.cat((learnable_prompt_speed, pre_text_tensor[:, 1, :self.remain_len, :]), dim=1)                     # learnable_prompt_speed: (B,10,512). 10=6||4

        learnable_prompt_head = self.prompt_ctx_head
        learnable_prompt_head = learnable_prompt_head.view(1, self.prompt_ctx_head.shape[0], self.prompt_ctx_head.shape[1])
        learnable_prompt_head = learnable_prompt_head.repeat(x.shape[0], 1, 1)
        learnable_prompt_head = torch.cat((learnable_prompt_head, pre_text_tensor[:, 2, :self.remain_len, :]), dim=1)                       # learnable_prompt_head: (B,10,512)

        learnable_prompt_body = self.prompt_ctx_body
        learnable_prompt_body = learnable_prompt_body.view(1, self.prompt_ctx_body.shape[0], self.prompt_ctx_body.shape[1])
        learnable_prompt_body = learnable_prompt_body.repeat(x.shape[0], 1, 1)
        learnable_prompt_body = torch.cat((learnable_prompt_body, pre_text_tensor[:, 3, :self.remain_len, :]), dim=1)                       # learnable_prompt_body: (B,10,512)

        learnable_prompt_arm = self.prompt_ctx_arm
        learnable_prompt_arm = learnable_prompt_arm.view(1, self.prompt_ctx_arm.shape[0], self.prompt_ctx_arm.shape[1])
        learnable_prompt_arm = learnable_prompt_arm.repeat(x.shape[0], 1, 1)
        learnable_prompt_arm = torch.cat((learnable_prompt_arm, pre_text_tensor[:, 4, :self.remain_len, :]), dim=1)                         # learnable_prompt_arm: (B,14,512)

        learnable_prompt_leg = self.prompt_ctx_leg
        learnable_prompt_leg = learnable_prompt_leg.view(1, self.prompt_ctx_leg.shape[0], self.prompt_ctx_leg.shape[1])
        learnable_prompt_leg = learnable_prompt_leg.repeat(x.shape[0], 1, 1)
        learnable_prompt_leg = torch.cat((learnable_prompt_leg, pre_text_tensor[:, 5, :self.remain_len, :]), dim=1)                         # learnable_prompt_leg: (B,14,512)

        x = torch.cat((learnable_prompt_subject, learnable_prompt_verb, learnable_prompt_speed, learnable_prompt_head, learnable_prompt_body, learnable_prompt_arm, learnable_prompt_leg), dim=1)
                                                                                                                                            # x: (B,77,512)
        with torch.no_grad():
            x = x + self.clip_text.positional_embedding.type(self.clip_text.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND                                                                                            # x: (77,B,512)
            x = self.clip_text.transformer(x)
            x = self.clip_text.ln_final(x).type(self.clip_text.dtype)

        x = self.text_pre_proj(x)
        xf_out = self.textTransEncoder(x)
        xf_out = self.text_ln(xf_out)                                                                                                       # xf_out: (77,B,512)
        
        return xf_out

    
    def forward(self, x_2d, x_3d, t, text, pre_text_tensor, PROMPT):
        if not self.prompt_enabled:
            return super(MixSTE2_PromptEnabled, self).forward(x_2d, x_3d, t, text, pre_text_tensor)
        
        assert (len(self.prompt_enabled)>0) == (PROMPT is not None)

        if self.prompt_enabled == 'v1':
            return self.forward_v1(x_2d, x_3d, t, text, pre_text_tensor, PROMPT)
        
        if self.prompt_enabled == 'v2':
            return self.forward_v2(x_2d, x_3d, t, text, pre_text_tensor, PROMPT)
        
        if self.prompt_enabled == 'v3':
            return self.forward_v3(x_2d, x_3d, t, text, pre_text_tensor, PROMPT)
        
        if self.prompt_enabled == 'v4':
            return self.forward_v4(x_2d, x_3d, t, text, pre_text_tensor, PROMPT)
        
        if self.prompt_enabled == 'v5':
            return self.forward_v5(x_2d, x_3d, t, text, pre_text_tensor, PROMPT)
        if self.prompt_enabled == 'v6':
            return self.forward_v6(x_2d, x_3d, t, text, pre_text_tensor, PROMPT)
    

    def forward_v2(self, x_2d, x_3d, t, text, pre_text_tensor, PROMPT):
        if self.is_train:                                                                       
            b, f, n, c = x_2d.shape                                                           
        else:
            b, h, f, n, c = x_3d.shape

        xf_proj, xf_out = self.encode_text(text, pre_text_tensor)                          
        x, time_embed = self.STE_forward(x_2d, x_3d, t, xf_proj)                                # x: (B*17, 243, 512). time_embed: (B*243, 17, 512) 

        x = self.temporal_cross_attn(x, xf_out, time_embed)                                     # x: (B*17, 243, 512). 以x为Query, xf_out为Key和Value, 作cross attention,  相当于paper中的第2和3步: temporal cross attention
        x = self.TTE_foward(x)                                                                  # x: (B*17, 243, 512)
                                                                                                # TTE_foward 中只用到了self.TTEblocks中的第一个block, 相当于paper中的第4步: temporal transformer


        prompt_in, prompt_out = PROMPT
        prompt_rep = self.get_mixste_rep(prompt_in, prompt_out, t, text, pre_text_tensor, return_rep=True)                          # (B,243,17,512)
        prompt_rep = prompt_rep.permute(0,2,1,3).reshape(b*n,f,-1)                                                                  # (B*17,243,512)
        x = self.QueryPrompt_temporal_cross_attn(x, prompt_rep, time_embed)






        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)                                                                   # 在这里面用了其余的7个self.STEblocks和7个self.TTEblocks作spatial-temporal attention, 相当于paper中的第5步: spatial-temporal transformer

        x = self.head(x)

        if self.is_train:
            x = x.view(b, f, n, -1)
        else:
            x = x.view(b, h, f, n, -1)

        return x                                                                                # x: (B,243,17,3)



    def forward_v3(self, x_2d, x_3d, t, text, pre_text_tensor, PROMPT):
        if self.is_train:                                                                       
            b, f, n, c = x_2d.shape                                                           
        else:
            b, h, f, n, c = x_3d.shape

        xf_proj, xf_out = self.encode_text(text, pre_text_tensor)                          
        x, time_embed = self.STE_forward(x_2d, x_3d, t, xf_proj)                                # x: (B*17, 243, 512). time_embed: (B*243, 17, 512) 

        x = self.temporal_cross_attn(x, xf_out, time_embed)                                     # x: (B*17, 243, 512). 以x为Query, xf_out为Key和Value, 作cross attention,  相当于paper中的第2和3步: temporal cross attention
        x = self.TTE_foward(x)                                                                  # x: (B*17, 243, 512)
                                                                                                # TTE_foward 中只用到了self.TTEblocks中的第一个block, 相当于paper中的第4步: temporal transformer


        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)                                                                   # 在这里面用了其余的7个self.STEblocks和7个self.TTEblocks作spatial-temporal attention, 相当于paper中的第5步: spatial-temporal transformer



        prompt_in, prompt_out = PROMPT
        prompt_rep = self.get_mixste_rep(prompt_in, prompt_out, t, text, pre_text_tensor, return_rep=True)                          # (B,243,17,512)
        prompt_rep = prompt_rep.permute(0,2,1,3).reshape(b*n,f,-1)                                                                  # (B*17,243,512)
        x = x.permute(0,2,1,3).reshape(b*n,f,-1)
        x = self.QueryPrompt_temporal_cross_attn(x, prompt_rep, time_embed)
        x = x.reshape(b,n,f,-1).permute(0,2,1,3)



        x = self.head(x)

        if self.is_train:
            x = x.view(b, f, n, -1)
        else:
            x = x.view(b, h, f, n, -1)

        return x                               
    

    def forward_v4(self, x_2d, x_3d, t, text, pre_text_tensor, PROMPT):                         # text: (B,77). 代表动作类别词汇, 如"WalkDog"
        if self.is_train:                                                                       # pre_text_tensor: (B,6,77). 6代表"A person", "speed", "head", "body", "arm", "leg"六个词汇
            b, f, n, c = x_2d.shape                                                             # b=batch_size, f=243, n=17, c=2
        else:
            b, h, f, n, c = x_3d.shape


        xf_out = self.encode_text_w_prompt(text, pre_text_tensor)                               # xf_out: (77,B,512). 即paper中的"fine-grained part-aware prompt P"


        prompt_in, prompt_out = PROMPT
        prompt = torch.cat([prompt_in, prompt_out], dim=-1)
        b, f, n, c = prompt.shape
        prompt = rearrange(prompt, 'b f n c  -> (b f) n c', )                                   # prompt: (B*243, 17, 3)
        prompt = self.prompt_spatial_patch_to_embedding(prompt)                                 # prompt: (B*243, 17, 512)
        prompt += self.prompt_spatial_pos_embed
        prompt_time_embed = self.prompt_time_mlp(t)[:, None, None, :]
        prompt_time_embed = prompt_time_embed.repeat(1, f, n, 1)
        prompt_time_embed = rearrange(prompt_time_embed, 'b f n c  -> (b f) n c', )
        prompt += prompt_time_embed                                                             # prompt: (B*243, 17, 512)
        prompt = self.pos_drop(prompt)                                                          # prompt: (B*243, 17, 512)
        prompt = rearrange(prompt, '(b f) n cw -> (b n) f cw', f=f)                             # prompt: (B*17, 243, 512)


        xf_out = self.TextPrompt_temporal_cross_attn(xf_out.permute(1,0,2), prompt)             # xf_out: (B*17,77,512)


        xf_proj = self.text_proj(xf_out[text.argmax(dim=-1), torch.arange(xf_out.shape[1])])                                                # xf_proj: (B,512)
        xf_out = xf_out.permute(1, 0, 2)                                                                                                    # xf_out: (B,77,512)
        
        
        x, time_embed = self.STE_forward(x_2d, x_3d, t, xf_proj)                                # x: (B*17, 243, 512). time_embed: (B*243, 17, 512)
                                                                                                # STE_forward 中只用到了self.STEblocks中的第一个block, 相当于paper中的第1步: spatial transformer


        x = self.temporal_cross_attn(x, xf_out, time_embed)                                     # x: (B*17, 243, 512). 以x为Query, xf_out为Key和Value, 作cross attention,  相当于paper中的第2和3步: temporal cross attention
        x = self.TTE_foward(x)                                                                  # x: (B*17, 243, 512)
                                                                                                # TTE_foward 中只用到了self.TTEblocks中的第一个block, 相当于paper中的第4步: temporal transformer

        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)                                                                   # 在这里面用了其余的7个self.STEblocks和7个self.TTEblocks作spatial-temporal attention, 相当于paper中的第5步: spatial-temporal transformer

        x = self.head(x)

        if self.is_train:
            x = x.view(b, f, n, -1)
        else:
            x = x.view(b, h, f, n, -1)

        return x                                                                                # x: (B,243,17,3)


    def forward_v1(self, x_2d, x_3d, t, text, pre_text_tensor, PROMPT):                 # text: (B,77). 代表动作类别词汇, 如"WalkDog"
        if self.is_train:                                                                       # pre_text_tensor: (B,6,77). 6代表"A person", "speed", "head", "body", "arm", "leg"六个词汇
            b, f, n, c = x_2d.shape                                                             # b=batch_size, f=243, n=17, c=2
        else:
            b, h, f, n, c = x_3d.shape
        
        xf_proj, xf_out = self.encode_text(text, pre_text_tensor)                               # xf_proj: (B,512)
                                                                                                # xf_out: (B,77,512). 即paper中的"fine-grained part-aware prompt P"

        x, time_embed = self.STE_forward(x_2d, x_3d, t, xf_proj)                                # x: (B*17, 243, 512). time_embed: (B*243, 17, 512)
                                                                                                # STE_forward 中只用到了self.STEblocks中的第一个block, 相当于paper中的第1步: spatial transformer


        assert (len(self.prompt_enabled)>0) == (PROMPT is not None)
        if PROMPT is not None and self.prompt_enabled:
            prompt_in, prompt_out = PROMPT
            prompt = torch.cat([prompt_in, prompt_out], dim=-1)
            b, f, n, c = prompt.shape
            prompt = rearrange(prompt, 'b f n c  -> (b f) n c', )                               # prompt: (B*243, 17, 3)
            prompt = self.prompt_spatial_patch_to_embedding(prompt)                             # prompt: (B*243, 17, 512)
            prompt += self.prompt_spatial_pos_embed
            prompt_time_embed = self.prompt_time_mlp(t)[:, None, None, :]
            prompt_xf_proj = xf_proj.view(xf_proj.shape[0], 1, 1, xf_proj.shape[1])
            prompt_time_embed = prompt_time_embed + prompt_xf_proj
            prompt_time_embed = prompt_time_embed.repeat(1, f, n, 1)
            prompt_time_embed = rearrange(prompt_time_embed, 'b f n c  -> (b f) n c', )
            prompt += prompt_time_embed                                                         # prompt: (B*243, 17, 512)
            prompt = self.pos_drop(prompt)                                                      # prompt: (B*243, 17, 512)
            prompt = self.prompt_encode_block(prompt)                                           # prompt: (B*243, 17, 512)
            prompt = self.Spatial_norm(prompt)
            prompt = rearrange(prompt, '(b f) n cw -> (b n) f cw', f=f)                         # prompt: (B*17, 243, 512)
        
            x = self.Query_Prompt_temporal_cross_attn(x, prompt, time_embed) 



        x = self.temporal_cross_attn(x, xf_out, time_embed)                                     # x: (B*17, 243, 512). 以x为Query, xf_out为Key和Value, 作cross attention,  相当于paper中的第2和3步: temporal cross attention
        x = self.TTE_foward(x)                                                                  # x: (B*17, 243, 512)
                                                                                                # TTE_foward 中只用到了self.TTEblocks中的第一个block, 相当于paper中的第4步: temporal transformer

        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        x = self.ST_foward(x)                                                                   # 在这里面用了其余的7个self.STEblocks和7个self.TTEblocks作spatial-temporal attention, 相当于paper中的第5步: spatial-temporal transformer

        x = self.head(x)

        if self.is_train:
            x = x.view(b, f, n, -1)
        else:
            x = x.view(b, h, f, n, -1)

        return x                                                                                # x: (B,243,17,3)


    def forward_v5(self, query_input, query_diffused_output, t, text, pre_text_tensor, PROMPT):                  # text: (B,77). 代表动作类别词汇, 如"WalkDog"
        text_proj, text_out = self.encode_text(text, pre_text_tensor)  
        prompt_input, prompt_diffused_output, prompt_groundtruth_output, prompt_noises = PROMPT
        # init_eps_t = prompt_noises

        ### prompt control branch ###
        if self.is_train:                                                                       
            b, f, n, c = prompt_input.shape                                                        
        else:
            b, h, f, n, c = prompt_diffused_output.shape
        
        if self.prompt_gt_as_condition:
            prompt, time_embed = self.STE_forward(prompt_input, prompt_groundtruth_output, t, text_proj)       # prompt: (B*17,243,512) [train] or (B*num_proposal*17,243,512) [test]
        else:
            prompt, time_embed = self.STE_forward(prompt_input, prompt_diffused_output, t, text_proj)       # prompt: (B*17,243,512) [train] or (B*num_proposal*17,243,512) [test]
        prompt = self.temporal_cross_attn(prompt, text_out, time_embed)                                 # prompt: (B*17,243,512) [train] or (B*num_proposal*17,243,512) [test]
        hidden_dim = prompt.shape[-1]
        control_outputs = (prompt.reshape(-1,n,f,hidden_dim).permute(0,3,2,1),)
        
        prompt = self.control_TTE_foward(prompt)                                                           # prompt: (B*17,243,512)
        
        control_outputs = control_outputs + (prompt.reshape(-1,n,f,hidden_dim).permute(0,3,2,1),)

        prompt = rearrange(prompt, '(b n) f cw -> b f n cw', n=n)
        prompt, mid_outputs = self.control_ST_foward(prompt, return_mid_output=True)
        
        control_outputs = control_outputs + mid_outputs

        ### obtain control residual
        zero_control_outputs = ()
        for control_output, zero_conv_block in zip(control_outputs, self.zero_conv_blocks):
            control_output = zero_conv_block(control_output)
            zero_control_outputs = zero_control_outputs + (control_output,)

        ### query branch ###
        if self.is_train:                                                                       
            b, f, n, c = query_input.shape                                                        
        else:
            b, h, f, n, c = query_diffused_output.shape
        query, time_embed = self.STE_forward(query_input, query_diffused_output, t, text_proj)
        query = self.temporal_cross_attn(query, text_out, time_embed)

        query = query + zero_control_outputs[0].permute(0,3,2,1).reshape(-1,f,hidden_dim)

        query = self.TTE_foward(query)

        query = query + zero_control_outputs[1].permute(0,3,2,1).reshape(-1,f,hidden_dim)

        query = rearrange(query, '(b n) f cw -> b f n cw', n=n)
        query = self.ST_foward(query, zero_control_outputs[2:])

        query = self.head(query)

        if self.is_train:
            query = query.view(b, f, n, -1)
        else:
            query = query.view(b, h, f, n, -1)

        return query


    def forward_v6(self, query_input, query_diffused_output, t, text, pre_text_tensor, PROMPT):                  # text: (B,77). 代表动作类别词汇, 如"WalkDog"
        text_proj, text_out = self.encode_text(text, pre_text_tensor)  
        prompt_input, prompt_diffused_output, prompt_groundtruth_output, prompt_noises = PROMPT

        pred_query_output_guesses = []
        for i in range(query_input.shape[0]):
            query_noise_guess = prompt_noises[i]
            query_diffused_sample = query_diffused_output[i]
            t_ = t[i]
            pred_query_output_guess = (query_diffused_sample - self.sqrt_one_minus_alphas_cumprod[t_] * query_noise_guess) / self.sqrt_alphas_cumprod[t_]
            pred_query_output_guesses.append(pred_query_output_guess)
        pred_query_output_guesses = torch.stack(pred_query_output_guesses).to(query_input.device)


        ### prompt control branch ###
        if self.is_train:                                                                       
            b, f, n, c = prompt_input.shape                                                        
        else:
            b, h, f, n, c = prompt_diffused_output.shape
        prompt, time_embed = self.STE_forward(prompt_input, prompt_diffused_output, t, text_proj)       # prompt: (B*17,243,512)
        prompt = self.temporal_cross_attn(prompt, text_out, time_embed)                                 # prompt: (B*17,243,512)
        
        control_outputs = (prompt.reshape(b,n,f,-1).permute(0,3,2,1),)
        
        prompt = self.control_TTE_foward(prompt)                                                           # prompt: (B*17,243,512)
        
        control_outputs = control_outputs + (prompt.reshape(b,n,f,-1).permute(0,3,2,1),)

        prompt = rearrange(prompt, '(b n) f cw -> b f n cw', n=n)
        prompt = self.control_ST_foward(prompt, return_mid_output=True)
        
        control_outputs = control_outputs + prompt

        ### obtain control residual
        zero_control_outputs = ()
        for control_output, zero_conv_block in zip(control_outputs, self.zero_conv_blocks):
            control_output = zero_conv_block(control_output)
            zero_control_outputs = zero_control_outputs + (control_output,)

        ### query branch ###
        if self.is_train:                                                                       
            b, f, n, c = query_input.shape                                                        
        else:
            b, h, f, n, c = query_diffused_output.shape
        query, time_embed = self.STE_forward(query_input, query_diffused_output, t, text_proj)
        query = self.temporal_cross_attn(query, text_out, time_embed)

        query = query + zero_control_outputs[0].permute(0,3,2,1).reshape(b*n,f,-1)

        query = self.TTE_foward(query)

        query = query + zero_control_outputs[1].permute(0,3,2,1).reshape(b*n,f,-1)

        query = rearrange(query, '(b n) f cw -> b f n cw', n=n)
        query = self.ST_foward(query, zero_control_outputs[2:])

        query = self.head(query)

        if self.is_train:
            query = query.view(b, f, n, -1)
        else:
            query = query.view(b, h, f, n, -1)

        return query + pred_query_output_guesses                            


    def forward_v7(self, query_input, query_diffused_output, t, text, pre_text_tensor, PROMPT):                  # text: (B,77). 代表动作类别词汇, 如"WalkDog"
        text_proj, text_out = self.encode_text(text, pre_text_tensor)  
        prompt_input, prompt_diffused_output, prompt_groundtruth_output, prompt_noises = PROMPT
        # init_eps_t = prompt_noises

        ### [PROMPT] control branch ####################################################################################
        if self.is_train:                                                                       
            b, f, n, c = prompt_input.shape                                                        
        else:
            b, h, f, n, c = prompt_diffused_output.shape
        
        if self.prompt_gt_as_condition:
            prompt, time_embed = self.control_STE_forward(prompt_input, prompt_groundtruth_output, t, text_proj)       # prompt: (B*17,243,512)
        else:
            prompt, time_embed = self.control_STE_forward(prompt_input, prompt_diffused_output, t, text_proj)       # prompt: (B*17,243,512)
        prompt = self.temporal_cross_attn(prompt, text_out, time_embed)                                 # prompt: (B*17,243,512)
        
        control_outputs = (prompt.reshape(b,n,f,-1).permute(0,3,2,1),)
        
        prompt = self.control_TTE_foward(prompt)                                                           # prompt: (B*17,243,512)
        
        control_outputs = control_outputs + (prompt.reshape(b,n,f,-1).permute(0,3,2,1),)

        prompt = rearrange(prompt, '(b n) f cw -> b f n cw', n=n)
        prompt = self.control_ST_foward(prompt, return_mid_output=True)
        
        control_outputs = control_outputs + prompt

        ### obtain control residual
        zero_control_outputs = ()
        for control_output, zero_conv_block in zip(control_outputs, self.zero_conv_blocks):
            control_output = zero_conv_block(control_output)
            zero_control_outputs = zero_control_outputs + (control_output,)

        ### [QUERY] branch #################################################################################
        if self.is_train:                                                                       
            b, f, n, c = query_input.shape                                                        
        else:
            b, h, f, n, c = query_diffused_output.shape
        query, time_embed = self.STE_forward(query_input, query_diffused_output, t, text_proj)
        query = self.temporal_cross_attn(query, text_out, time_embed)

        query = query + zero_control_outputs[0].permute(0,3,2,1).reshape(b*n,f,-1)

        query = self.TTE_foward(query)

        query = query + zero_control_outputs[1].permute(0,3,2,1).reshape(b*n,f,-1)

        query = rearrange(query, '(b n) f cw -> b f n cw', n=n)
        query = self.ST_foward(query, zero_control_outputs[2:])

        query = self.head(query)

        if self.is_train:
            query = query.view(b, f, n, -1)
        else:
            query = query.view(b, h, f, n, -1)

        return query


class HumanMAC_StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h
    

class HumanMAC_TemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)                                                # latent_dim=512
        self.query = nn.Linear(latent_dim, latent_dim, bias=False)                          # self.query = nn.Linear(512, 512, bias=False)
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)                            # self.key = nn.Linear(512, 512, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)                          # self.value = nn.Linear(512, 512, bias=False)
        self.dropout = nn.Dropout(dropout)                                                  # dropout=0.2
        # self.proj_out = HumanMAC_StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, cross_x, emb=None):
        """
        x: B, T, D
        """
        b, T, N, C = x.shape
        x = x.permute(0,2,1,3).reshape(b*N,T,C)
        cross_x = cross_x.permute(0,2,1,3).reshape(b*N,T,C)
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(cross_x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(cross_x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        # y = x + self.proj_out(y, emb)
        y = x + y
        return y.reshape(b,N,T,C).permute(0,2,1,3)


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
    def __init__(self, max_clip_len, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.tte_block = Block( dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer, comb=False, changedim=False, currentdim=0, depth=depth)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(max_clip_len, embed_dim_ratio))     
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.Temporal_norm = norm_layer(embed_dim_ratio)
    def forward(self, x):
        b, f, n, c  = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b*n, f, c)                                # x: (B*17,243,512)
        x += self.temporal_pos_embed[:f, :].unsqueeze(0)
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


class Skeleton_in_Context(MixSTE2):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2, 
                 qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=None, is_train=True, 
                 prompt_enabled=False, sqrt_alphas_cumprod=None, sqrt_one_minus_alphas_cumprod=None, 
                 prompt_gt_as_condition=False, use_text=True, fuse_prompt_query='add', max_clip_len=243):
        super().__init__(num_frame, num_joints, in_chans, embed_dim_ratio, depth, num_heads, mlp_ratio, 
                         qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer,
                         is_train, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        self.prompt_enabled = prompt_enabled
        self.depth = depth
        self.use_text = use_text
        self.fuse_prompt_query = fuse_prompt_query

        self.spatial_pos_embed = nn.Parameter(torch.zeros(num_joints, embed_dim_ratio))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(max_clip_len, embed_dim_ratio))

        self.prompt_ste = STE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.prompt_tte = TTE(max_clip_len, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.prompt_st = ST(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.prompt_Spatial_patch_to_embedding = nn.Linear(in_chans + 3, embed_dim_ratio)
        self.prompt_spatial_pos_embed = nn.Parameter(torch.zeros(num_joints, embed_dim_ratio))
        self.prompt_temporal_cross_attn = TemporalCrossAttention(512, 512, num_heads, drop_rate, 512)

        
        if fuse_prompt_query == 'crossattn':
            self.fuse_query_prompt = nn.ModuleList()
            for i in range(depth + 3):
                self.fuse_query_prompt.append(
                    HumanMAC_TemporalCrossAttention(embed_dim_ratio, num_heads, drop_rate, embed_dim_ratio)
                )

        
        self.GLOBAL_prompt_ste = STE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.GLOBAL_prompt_tte = TTE(max_clip_len, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.GLOBAL_prompt_st = ST(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.GLOBAL_query_ste = STE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.GLOBAL_query_tte = TTE(max_clip_len, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.GLOBAL_query_st = ST(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)



    def prompt_encode_forward(self, x_in, x_out):
        x = torch.cat((x_in, x_out), dim=-1)
        x = self.prompt_Spatial_patch_to_embedding(x)
        x += self.prompt_spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        return x
    
    def encode_forward(self, x_in, x_out):
        x = torch.cat((x_in, x_out), dim=-1)
        x = self.Spatial_patch_to_embedding(x)
        x += self.spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        return x
    
    def STE_forward(self, x):
        b, f, n, c = x.shape
        x = x.reshape(b*f, n, c)
        x = self.pos_drop(x)
        blk = self.STEblocks[0]
        x = blk(x)
        x = self.Spatial_norm(x)
        x = x.reshape(b, f, n, c)
        return x
    
    def TTE_foward(self, x):
        b, f, n, c  = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b*n, f, c)                                # x: (B*17,243,512)
        x += self.temporal_pos_embed[:f].unsqueeze(0)
        x = self.pos_drop(x)
        blk = self.TTEblocks[0]                                                     # self.TTEblocks[0]: 从self.TTEblocks (包含8个attention block) 中取出第一个block
        x = blk(x)                                                                  # x: (B*17,243,512)
        x = self.Temporal_norm(x)
        return x.reshape(b,n,f,c).permute(0,2,1,3)
    
    def ST_foward(self, x, residual_x1=None, residual_x2=None, residual_x3=None):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = rearrange(x, 'b f n cw -> (b f) n cw')
            x = steblock(x)
            x = self.Spatial_norm(x)

            x = rearrange(x, '(b f) n cw -> (b n) f cw', f=f)
            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f cw -> b f n cw', n=n)

            if residual_x1 is not None:
                if self.fuse_prompt_query == 'add':
                    x += residual_x1[i-1] + residual_x2[i-1] + residual_x3[i-1]
                elif self.fuse_prompt_query == 'crossattn':
                    x = self.fuse_query_prompt[i+3](x, residual_x1[i-1])

                
        return x
    
    def temporal_cross_attention_forward(self, x, text_out, time_embed=None):
        b, f, n, c = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b*n, f, c)                                # x: (B*17,243,512)
        time_embed = time_embed.reshape(b*f,n,c)
        x = self.temporal_cross_attn(x, text_out, time_embed)                       # x: (B*17,243,512)
        return x.reshape(b,n,f,c).permute(0,2,1,3)

    def prompt_temporal_cross_attention_forward(self, x, text_out, time_embed=None):
        b, f, n, c = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b*n, f, c)                                # x: (B*17,243,512)
        time_embed = time_embed.reshape(b*f,n,c)
        x = self.prompt_temporal_cross_attn(x, text_out, time_embed)                       # x: (B*17,243,512)
        return x.reshape(b,n,f,c).permute(0,2,1,3)
    
    def forward(self, Prompt, Query, epoch=None):                    # query_input: (B,243,17,3)
        B, F, J, C = Query.shape
        F1 = F // 2
        query_input = Query[:, :F1, :, :]
        prompt_input = Prompt[:, :F1, :, :]
        prompt_groundtruth_output = Prompt[:, F1:, :, :]

        if self.use_text:
            text_proj, text_out = self.encode_text(text, pre_text_tensor)                                           # text_proj: (B,512). text_out: (B,77,512)


        b, f, n, c = query_input.shape


        # PROMPT branch
        prompt = self.prompt_encode_forward(prompt_input, prompt_groundtruth_output)                               # prompt: (B,243,17,512)
        PROMPTS = (prompt,)
        prompt_tmean = prompt.mean(-3, keepdim=True).clone()
        PROMPTS_TMEAN = (prompt_tmean, )

        prompt = self.prompt_ste(prompt)                                                                       # prompt: (B,243,17,512)
        PROMPTS = PROMPTS + (prompt,)
        prompt_tmean = self.GLOBAL_prompt_ste(prompt_tmean)
        PROMPTS_TMEAN = PROMPTS_TMEAN + (prompt_tmean, )

        if self.use_text:
            prompt = self.prompt_temporal_cross_attention_forward(prompt, text_out)                        # prompt: (B,243,17,512)
            PROMPTS = PROMPTS + (prompt,)
        else:
            PROMPTS = PROMPTS + (None,)
            PROMPTS_TMEAN = PROMPTS_TMEAN + (None,)

        prompt = self.prompt_tte(prompt)                                                                        # prompt: (B,243,17,512)
        PROMPTS = PROMPTS + (prompt,)
        prompt_tmean = self.GLOBAL_prompt_tte(prompt_tmean)
        PROMPTS_TMEAN = PROMPTS_TMEAN + (prompt_tmean, )

        for i in range(self.depth - 1):
            prompt = self.prompt_st.forward_i(prompt, i)
            PROMPTS = PROMPTS + (prompt,)
            prompt_tmean = self.GLOBAL_prompt_st.forward_i(prompt_tmean, i)
            PROMPTS_TMEAN = PROMPTS_TMEAN + (prompt_tmean, )


        # QUERY_BRANCH
        query = self.encode_forward(query_input, prompt_groundtruth_output)                                         # query: (B,243,17,512)




        query_tmean = query.mean(-3, keepdim=True).clone().detach()
        QUERY_TMEAN = (query_tmean, )
        query_tmean = self.GLOBAL_query_ste(query_tmean)
        QUERY_TMEAN = QUERY_TMEAN + (query_tmean, )
        QUERY_TMEAN = QUERY_TMEAN + (None,)
        query_tmean = self.GLOBAL_query_tte(query_tmean)
        QUERY_TMEAN = QUERY_TMEAN + (query_tmean, )
        for i in range(self.depth - 1):
            query_tmean = self.GLOBAL_query_st.forward_i(query_tmean, i)
            QUERY_TMEAN = QUERY_TMEAN + (query_tmean, )





        if self.fuse_prompt_query == 'add':
            query += PROMPTS[0] + PROMPTS_TMEAN[0] + QUERY_TMEAN[0]
        elif self.fuse_prompt_query == 'crossattn':
            query = self.fuse_query_prompt[0](query, PROMPTS[0])


        query = self.STE_forward(query)                                                                         # query: (B,243,17,512)
        if self.fuse_prompt_query == 'add':
            query += PROMPTS[1] + PROMPTS_TMEAN[1] + QUERY_TMEAN[1]
        elif self.fuse_prompt_query == 'crossattn':
            query = self.fuse_query_prompt[1](query, PROMPTS[1])

        if self.use_text:
            query = self.temporal_cross_attention_forward(query, text_out)                          # query: (B,243,17,512)
            if self.fuse_prompt_query == 'add':
                query += PROMPTS[2]
            elif self.fuse_prompt_query == 'crossattn':
                query = self.fuse_query_prompt[2](query, PROMPTS[2])

        query = self.TTE_foward(query)                                                                          # query: (B,243,17,512)
        if self.fuse_prompt_query == 'add':
            query += PROMPTS[3] + PROMPTS_TMEAN[3] + QUERY_TMEAN[3]
        elif self.fuse_prompt_query == 'crossattn':
            query = self.fuse_query_prompt[3](query, PROMPTS[3])

        query = self.ST_foward(query, PROMPTS[4:], PROMPTS_TMEAN[4:], QUERY_TMEAN[4:])


        # MERGE
        query = self.head(query)

        query = query.view(b, f, n, -1)

        return query