from functools import partial
from einops import rearrange
import torch
import torch.nn as nn

from timm.models.layers import DropPath
from typing import Optional
from collections import defaultdict
from transformers.utils import ModelOutput, logging

from lib.model.TTT_Linear_official import Block as TTT_Block, TTTConfig

logger = logging.get_logger(__name__)

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


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value


class TTTCache:
    """
    TTTCache is a data structure that holds the last hidden states and gradients for the TTT layer.

    Arguments:
        model: TTTModel
        batch_size: int

    Attributes:
        seqlen_offset: int
        mini_batch_size: int
        params_dict: Dict[str, Dict[int, torch.Tensor]]  *_states, *_grad -> # layer_idx -> [batch_size, ...]
        conv_states_dic: Dict[str, Dict[int, torch.Tensor]]  *_states -> # layer_idx -> [batch_size, ...]

    """

    def __init__(self, model, batch_size, config, device):
        config = config
        self.seqlen_offset = 0
        self.mini_batch_size = config.mini_batch_size

        self.ttt_params_dict = defaultdict(dict)
        if "linear" in config.ttt_layer_type:
            self.ttt_param_names = ["W1", "b1"]
        elif "mlp" in config.ttt_layer_type:
            self.ttt_param_names = ["W1", "b1", "W2", "b2"]
        else:
            raise ValueError(f"TTT Layer Type {config.ttt_layer_type} not supported yet")

        self.conv_states_dic = defaultdict(dict)
        logger.info(f"Creating cache of size: {batch_size}")
        for layer_idx in range(config.num_hidden_layers):
            for name in self.ttt_param_names:
                weight = getattr(model.layers[layer_idx].seq_modeling_block, name)
                tiled_weight = torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(device)
                self.ttt_params_dict[f"{name}_states"][layer_idx] = tiled_weight
                # for decoding, we need to store the gradients as well
                self.ttt_params_dict[f"{name}_grad"][layer_idx] = torch.zeros_like(tiled_weight)

            if config.pre_conv:
                self.conv_states_dic["pre_conv"][layer_idx] = torch.zeros(
                    batch_size,
                    config.hidden_size,
                    config.conv_kernel,
                    device=device,
                )
            if config.share_qk:
                self.conv_states_dic["ttt_conv_q"][layer_idx] = torch.zeros(
                    batch_size,
                    config.hidden_size,
                    config.conv_kernel,
                    device=device,
                )
                self.conv_states_dic["ttt_conv_k"][layer_idx] = torch.zeros(
                    batch_size,
                    config.hidden_size,
                    config.conv_kernel,
                    device=device,
                )

    def update(self, py_tree, layer_idx, seq_len):
        if seq_len % self.mini_batch_size == 0:
            # copy last mini-batch states, clear gradients
            for name in self.ttt_param_names:
                self.ttt_params_dict[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                self.ttt_params_dict[f"{name}_grad"][layer_idx].zero_()
        elif seq_len < self.mini_batch_size:
            if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.mini_batch_size != 0:
                raise ValueError("fractional update not supported yet.")
            if (seq_len + self.seqlen_offset) % self.mini_batch_size == 0:
                # copy last mini-batch states, clear gradients
                for name in self.ttt_param_names:
                    self.ttt_params_dict[f"{name}_states"][layer_idx].copy_(py_tree[f"{name}_states"])
                    self.ttt_params_dict[f"{name}_grad"][layer_idx].zero_()
            else:
                # copy gradients for the next update
                for name in self.ttt_param_names:
                    self.ttt_params_dict[f"{name}_grad"][layer_idx].copy_(py_tree[f"{name}_grad"])
        else:
            raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")

    def ttt_params_to_dict(self, layer_idx):
        return {name: self.ttt_params_dict[name][layer_idx] for name in self.ttt_params_dict}
    

class Skeleton_in_Context(nn.Module):
    def __init__(self, num_frame=16, num_joints=17, in_chans=3, embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2, 
                 qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.query_spatial_patch_to_embedding = nn.Linear(in_chans + 3, embed_dim_ratio)
        self.query_spatial_pos_embed = nn.Parameter(torch.zeros(num_joints, embed_dim_ratio))

        self.query_ste = STE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.query_tte = TTE(num_frame, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)

        self.prompt_spatial_patch_to_embedding = nn.Linear(in_chans + 3, embed_dim_ratio)
        self.prompt_spatial_pos_embed = nn.Parameter(torch.zeros(num_joints, embed_dim_ratio))

        self.prompt_ste = STE(embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        self.prompt_tte = TTE(num_frame, embed_dim_ratio, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, depth, drop_path_rate, norm_layer)
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio),
            nn.Linear(embed_dim_ratio, 3),
        )


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.query_STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.prompt_STEblocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.Spatial_norm = norm_layer(embed_dim_ratio)


        TTT_config = TTTConfig(num_hidden_layers=depth, 
                               use_cache=True,                                 # default: False
                               hidden_size=embed_dim_ratio,
                               pre_conv=False,
                               ttt_layer_type='linear',
                               rms_norm_eps=1e-6,
                               intermediate_size=embed_dim_ratio * 2,           # default: 5504
                               hidden_act='silu',
                               pretraining_tp=1,
                               conv_kernel=4,                                   # default: 4
                               num_attention_heads=num_heads,                   # default: 32
                               mini_batch_size=4,
                               share_qk=False,                                  # default: False. Set True if in Mamba-style
                               use_gate=False,                                  # default: False. Set True if in Mamba-style
                               rope_theta=10000.0,
                               ttt_base_lr=1.0,
                               scan_checkpoint_group_size=0
                               )
        
        self.config = TTT_config
        
        self.prompt_TTT = nn.ModuleDict({
            'layers': nn.ModuleList([TTT_Block(TTT_config, layer_idx) for layer_idx in range(TTT_config.num_hidden_layers)])
        })
        self.query_TTT = nn.ModuleDict({
            'layers': nn.ModuleList([TTT_Block(TTT_config, layer_idx) for layer_idx in range(TTT_config.num_hidden_layers)])
        })

        # self.norm = RMSNorm(TTT_config.hidden_size, eps=TTT_config.rms_norm_eps)


    def prompt_encode_forward(self, x_in, x_out):
        x = torch.cat((x_in, x_out), dim=-1)
        x = self.prompt_spatial_patch_to_embedding(x)
        x += self.prompt_spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        return x
    
    def query_encode_forward(self, x_in, x_out):
        x = torch.cat((x_in, x_out), dim=-1)
        x = self.query_spatial_patch_to_embedding(x)
        x += self.query_spatial_pos_embed.unsqueeze(0).unsqueeze(0)
        return x
    
    def forward(self, Prompt, Query, epoch=None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                cache_params: Optional[TTTCache] = None,
                use_cache: Optional[bool] = None,
                ):
        b, F, j, c = Query.shape
        f = F // 2
        query_input = Query[:, :f, :, :]
        prompt_input = Prompt[:, :f, :, :]
        prompt_groundtruth_output = Prompt[:, f:, :, :]


        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if cache_params is None and use_cache:
            cache_params = {'prompt': TTTCache(model=self.prompt_TTT, batch_size=b*j, device=query_input.device, config=self.config),
                            'query': TTTCache(model=self.query_TTT, batch_size=b*j, device=query_input.device, config=self.config)}
        seqlen_offset = 0
        if cache_params is not None:
            assert cache_params['prompt'].seqlen_offset == cache_params['query'].seqlen_offset
            seqlen_offset = cache_params['prompt'].seqlen_offset
        position_ids = torch.arange(seqlen_offset, seqlen_offset + f, dtype=torch.long, device=query_input.device).unsqueeze(0)
        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)


        # PROMPT branch ###############################################################################
        prompt = self.prompt_encode_forward(prompt_input, prompt_groundtruth_output)
        PROMPTS = (prompt,)

        prompt = self.prompt_ste(prompt)
        PROMPTS = PROMPTS + (prompt,)

        prompt = self.prompt_tte(prompt)
        PROMPTS = PROMPTS + (prompt,)


        # PROMPT TTT ###############################################################################
        prompt_hidden_states = prompt
        for i, ttt_layer in enumerate(self.prompt_TTT['layers']):
            prompt_hidden_states = prompt_hidden_states.permute(0,2,1,3).reshape(b*j,f,-1)
            prompt_hidden_states = ttt_layer(
                prompt_hidden_states,                                                              # must be in the shape of (B,L,C)
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_params=cache_params['prompt'])
            prompt_hidden_states = prompt_hidden_states.reshape(b,j,f,-1).permute(0,2,1,3)
            if i == 1: break

            prompt_hidden_states = prompt_hidden_states.reshape(b*f,j,-1)
            prompt_hidden_states = self.prompt_STEblocks[i](prompt_hidden_states)
            prompt_hidden_states = self.Spatial_norm(prompt_hidden_states)
            prompt_hidden_states = prompt_hidden_states.reshape(b,f,j,-1)

            PROMPTS = PROMPTS + (prompt_hidden_states,)

        if use_cache:
            cache_params['prompt'].seqlen_offset += prompt_hidden_states.shape[1]


        # QUERY branch ###############################################################################
        query = self.query_encode_forward(query_input, prompt_groundtruth_output)
        # query = query + PROMPTS[0]
        
        query = self.query_ste(query)
        # query = query + PROMPTS[1]
            
        query = self.query_tte(query)
        # query = query + PROMPTS[2]
        


        # QUERY TTT ###############################################################################
        query_hidden_states = query
        for i, ttt_layer in enumerate(self.query_TTT['layers']):
            # query_hidden_states = query_hidden_states.permute(0,2,1,3).reshape(b*j,f,-1)
            # query_hidden_states = ttt_layer(
            #     query_hidden_states,                                                              # must be in the shape of (B,L,C)
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            #     cache_params=cache_params['query'])
            # query_hidden_states = query_hidden_states.reshape(b,j,f,-1).permute(0,2,1,3)

            query_hidden_states = query_hidden_states.reshape(b*f,j,-1)
            query_hidden_states = self.query_STEblocks[i](query_hidden_states)
            query_hidden_states = self.Spatial_norm(query_hidden_states)
            query_hidden_states = query_hidden_states.reshape(b,f,j,-1)

            # query = query + PROMPTS[i+3]

        if use_cache:
            cache_params['query'].seqlen_offset += query_hidden_states.shape[1]



        # POST-PROCESS ###############################################################################
        query = self.head(query)

        query = query.view(b, f, j, -1)

        return query