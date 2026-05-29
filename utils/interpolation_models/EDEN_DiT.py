from torch import nn
from functools import partial
from utils.interpolation_modules import DiTBlock
from utils.interpolation_utils import TimestepEmbedder, get_pos_embedding, preprocess_cond


class DiT(nn.Module):
    def __init__(self, latent_dim=16, dim=768, num_heads=12, mlp_ratio=4.0, depth=12, qkv_bias=False, attn_drop_rate=0.,
                 proj_drop_rate=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_xformers=True):
        super().__init__()
        self.dim = dim
        self.proj_in = nn.Linear(latent_dim, dim)
        self.proj_cond = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.norm_cond = norm_layer(dim)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, proj_drop_rate, act_layer, norm_layer,
                     use_xformers) for _ in range(depth)
        ])
        self.norm_out = norm_layer(dim)
        self.proj_out = nn.Linear(dim, 2 * latent_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.difference_embedder = nn.Linear(1, dim)
        self.denoise_timestep_embedder = TimestepEmbedder(dim)
        self.ph, self.pw = None, None
        self.init_weights()

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        w = self.proj_cond.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj_cond.bias, 0)
        nn.init.normal_(self.difference_embedder.weight, std=0.02)
        nn.init.normal_(self.denoise_timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.denoise_timestep_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation and final layer
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def patch_cond(self, x):
        x, _ = preprocess_cond(x)
        x = self.proj_cond(x).flatten(2).transpose(1, 2)
        cond_pos_embedding = get_pos_embedding(self.ph, self.pw, 1, self.dim).to(x.device)
        x = self.norm_cond(x + cond_pos_embedding)
        return x

    def forward(self, query_latents, denoise_timestep, cond_frames, difference):
        self.ph, self.pw = cond_frames.shape[-2] // 16, cond_frames.shape[-1] // 16
        tokens_0, tokens_1 = self.patch_cond(cond_frames).chunk(2, dim=0)
        denoise_timestep_embedding = self.denoise_timestep_embedder(denoise_timestep)
        difference_embedding = self.difference_embedder(difference)
        condition_embedding = denoise_timestep_embedding + difference_embedding
        modulations = self.adaLN_modulation(condition_embedding)
        pos_embedding = get_pos_embedding(self.ph, self.pw, 2, self.dim).to(query_latents.device)
        query_embedding = self.proj_in(query_latents) + pos_embedding
        for blk in self.blocks:
            query_embedding = blk(query_embedding, tokens_0, tokens_1, self.ph, self.pw, modulations)
        query_latents = self.proj_out(self.norm_out(query_embedding))
        query_latents, _ = query_latents.chunk(2, dim=-1)
        return query_latents

