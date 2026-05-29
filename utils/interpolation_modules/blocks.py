import torch
from torch import nn
from functools import partial
from utils.interpolation_modules.attention import SelfAttention, CrossAttention, TemporalAttention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    def __init__(self, dim=768, mlp_ratio=4., act_layer=nn.GELU, proj_drop_rate=0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(proj_drop_rate)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class VAEBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0., act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-5), use_xformers=True, is_encoder=True, add_attn_encoder=False,
                 add_attn_decoder=False, add_attn_type="temporal_attn"):
        super().__init__()
        self.is_encoder = is_encoder
        self.add_attn_encoder = add_attn_encoder
        self.add_attn_decoder = add_attn_decoder
        self.add_attn_type = add_attn_type
        self.norm_self_attn = norm_layer(dim)
        self.self_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, use_xformers)
        if (add_attn_encoder and is_encoder) or (add_attn_decoder and not is_encoder):
            if add_attn_type == "cross_attn":
                self.norm_cross_attn = norm_layer(dim)
                self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, use_xformers)
            elif add_attn_type == "temporal_attn":
                self.norm_temp_attn = norm_layer(dim)
                self.temp_attn = TemporalAttention(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, use_xformers)
            else:
                raise f"No implementation of {add_attn_type}"
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_ratio, act_layer, proj_drop_rate)

    def forward(self, query_tokens, mid_tokens, cond_tokens, ph=None, pw=None):
        b, n, d = mid_tokens.shape
        x = torch.cat((mid_tokens, query_tokens), dim=1)
        x = x + self.self_attn(self.norm_self_attn(x))
        if self.is_encoder:
            y = x[:, n:, :]
        else:
            y = x[:, :n, :]
        if (self.add_attn_encoder and self.is_encoder) or (self.add_attn_decoder and not self.is_encoder):
            if self.add_attn_type == "cross_attn":
                y = y + self.cross_attn(self.norm_cross_attn(y), y=cond_tokens)
            elif self.add_attn_type == "temporal_attn":
                tokens0, tokens1 = cond_tokens.chunk(2, dim=1)
                y = y + self.temp_attn(self.norm_temp_attn(y), x0=tokens0, x1=tokens1, ph=ph, pw=pw)
        y = y + self.mlp(self.norm_mlp(y))
        return y


class DiTBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0., act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-5), use_xformers=True):
        super().__init__()
        self.norm_self_attn = norm_layer(dim)
        self.self_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, use_xformers)
        self.norm_temp_attn = norm_layer(dim)
        self.temp_attn = TemporalAttention(dim, num_heads, qkv_bias, attn_drop_rate, proj_drop_rate, use_xformers)
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_ratio, act_layer, proj_drop_rate)

    def forward(self, query_tokens, tokens0, tokens1, ph, pw, modulations):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulations.chunk(6, dim=1)
        query_tokens = query_tokens + gate_msa.unsqueeze(1) * self.self_attn(modulate(self.norm_self_attn(query_tokens),
                                                                                      shift_msa, scale_msa))
        query_tokens = query_tokens + self.temp_attn(self.norm_temp_attn(query_tokens), x0=tokens0, x1=tokens1,
                                                     ph=ph, pw=pw)
        query_tokens = query_tokens + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_mlp(query_tokens),
                                                                                shift_mlp, scale_mlp))
        return query_tokens
