import torch
from torch import nn
from functools import partial
from utils.interpolation_modules import VAEBlock
from utils.interpolation_utils import DiagonalGaussianDistribution, get_pos_embedding, preprocess_cond
from einops import rearrange


class VAE(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, patch_size=16, hidden_dim=768, num_heads=12, mlp_ratio=4., latent_dim=16,
                 encoder_depth=4, decoder_depth=4, qkv_bias=False, attn_drop_rate=0., proj_drop_rate=0.,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), use_xformers=True,
                 add_attn_encoder=True, add_attn_decoder=True, add_attn_type="temporal_attn"):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.patchify = nn.Conv2d(in_dim, hidden_dim, patch_size, stride=patch_size)
        self.encoder_blocks = nn.ModuleList([
            VAEBlock(hidden_dim, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, proj_drop_rate, act_layer, norm_layer,
                     use_xformers, is_encoder=True, add_attn_encoder=add_attn_encoder, add_attn_type=add_attn_type)
            for _ in range(encoder_depth)
        ])
        self.proj_latent = nn.Linear(hidden_dim, 2 * latent_dim)
        self.proj_token = nn.Linear(latent_dim, hidden_dim)
        self.decoder_blocks = nn.ModuleList([
            VAEBlock(hidden_dim, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, proj_drop_rate, act_layer, norm_layer,
                     use_xformers, is_encoder=False, add_attn_decoder=add_attn_decoder, add_attn_type=add_attn_type)
            for _ in range(decoder_depth)
        ])
        final_dim = patch_size * patch_size * out_dim
        self.unpatchify = nn.Sequential(norm_layer(hidden_dim),
                                        nn.Linear(hidden_dim, final_dim),
                                        act_layer(),
                                        nn.Linear(final_dim, final_dim))
        self.pos_embedding, self.query_pos_embedding = None, None
        self.rh, self.rw, self.ph, self.pw = None, None, None, None
        self.stats = None
        self.get_query = nn.AvgPool2d(kernel_size=2, stride=2)
        self.norm_cond = norm_layer(hidden_dim)
        self.init_weights()

    def init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        w = self.patchify.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patchify.bias, 0)

    def preprocess_mid(self, x):
        return (x - self.stats[0]) / self.stats[1]

    def postprocess(self, x):
        return x * self.stats[1] + self.stats[0]

    def pixel_shuffle(self, x):
        x = self.unpatchify(x)
        x = rearrange(x, "b (ph pw) (p1 p2 c) -> b c (ph p1) (pw p2)",
                      ph=self.ph, pw=self.pw, p1=self.patch_size, p2=self.patch_size, c=self.out_dim)
        return x

    def patch_embed(self, x):
        self.ph, self.pw = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.patchify(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def encode(self, frames):
        frame_0, frame_1, frame_t = frames.chunk(3, dim=0)
        cond_frames, self.stats = preprocess_cond(torch.cat((frame_0, frame_1), dim=0))
        mid_frames = self.preprocess_mid(frame_t)
        frames = torch.cat((cond_frames, mid_frames), dim=0)
        tokens = self.patch_embed(frames)
        self.pos_embedding = get_pos_embedding(self.ph, self.pw, 1, self.hidden_dim).to(frames.device)
        self.query_pos_embedding = get_pos_embedding(self.ph, self.pw, 2, self.hidden_dim).to(frames.device)
        tokens_0, tokens_1, mid_tokens = tokens.chunk(3, dim=0)
        tokens_0, tokens_1 = tokens_0 + self.pos_embedding, tokens_1 + self.pos_embedding
        cond_tokens = self.norm_cond(torch.cat((tokens_0, tokens_1), dim=1))
        mid_frames = rearrange(mid_tokens, "b (ph pw) d -> b d ph pw", ph=self.ph, pw=self.pw)
        query_frames = self.get_query(mid_frames)
        query_tokens = query_frames.flatten(2).transpose(1, 2) + self.query_pos_embedding
        mid_tokens = mid_tokens + self.pos_embedding
        for blk in self.encoder_blocks:
            query_tokens = blk(query_tokens, mid_tokens, cond_tokens,self.ph, self.pw)
        moments = self.proj_latent(query_tokens)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, cond_tokens

    def decode(self, query_latents, cond_tokens):
        query_tokens = self.proj_token(query_latents)
        query_frames = rearrange(query_tokens, "b (ph pw) d -> b d ph pw", ph=self.ph // 2, pw=self.pw // 2)
        recon_frames = nn.functional.interpolate(query_frames, scale_factor=2., mode="bicubic", align_corners=False)
        recon_tokens = recon_frames.flatten(2).transpose(1, 2) + self.pos_embedding
        query_tokens = query_tokens + self.query_pos_embedding
        for blk in self.decoder_blocks:
            recon_tokens = blk(query_tokens, recon_tokens, cond_tokens, self.ph, self.pw)
        recon_frames = self.postprocess(self.pixel_shuffle(recon_tokens))
        return recon_frames

    def forward(self, x):
        posterior, cond_tokens = self.encode(x)
        latent = posterior.sample()
        x_recon = self.decode(latent, cond_tokens)
        return x_recon, posterior
