from utils.interpolation_utils.embedding import TimestepEmbedder, get_pos_embedding
from utils.interpolation_utils.klperceptual import KLLPIPSWithDiscriminator
from utils.interpolation_utils.distributions import DiagonalGaussianDistribution
from utils.interpolation_utils.cal_metrics import CalMetrics
import torch


class InputPadder:
    def __init__(self, img_size, divisor=32):
        self.ht, self.wd = img_size
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, x):
        return torch.nn.functional.pad(x, self._pad, mode="replicate")

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def preprocess_cond(x, eps=1e-8):
    x_flat = x.flatten(1)
    x_mean, x_std = torch.mean(x_flat, dim=-1), torch.std(x_flat, dim=-1) + eps
    while len(x_mean.shape) < len(x.shape):
        x_mean, x_std = x_mean.unsqueeze(-1), x_std.unsqueeze(-1)
    x_norm = (x - x_mean) / x_std
    x_mean_0, x_mean_1 = x_mean.chunk(2, dim=0)
    x_std_0, x_std_1 = x_std.chunk(2, dim=0)
    stats = ((x_mean_0 + x_mean_1) / 2, (x_std_0 + x_std_1) / 2)
    return x_norm, stats


def preprocess_frames(frames):
    frames = frames / 255.
    frame_0, frame_1, gt = frames[:, 0, ...], frames[:, 1, ...], frames[:, 2, ...]
    frames = torch.cat((frame_0, frame_1, gt), dim=0)
    img_size = [frames.shape[2], frames.shape[3]]
    padder = InputPadder(img_size)
    return frames, padder, frame_0, frame_1, gt


def one_iter_for_vae(model, frames, is_train=True):
    frames, padder, _, _, gt = preprocess_frames(frames)
    if not is_train:
        with torch.no_grad():
            recon, posterior = model(padder.pad(frames))
    else:
        recon, posterior = model(padder.pad(frames))
    recon = padder.unpad(recon.clamp(0., 1.))
    return recon, gt, posterior


def one_iter_for_dit(model, vae, frames, transport, sample_fn, vae_mean, vae_scaler, cos_sim_mean, cos_sim_std, is_train=True):
    frames, padder, frame_0, frame_1, gt = preprocess_frames(frames)
    cond_frames = torch.cat((frame_0, frame_1), dim=0)
    difference = ((torch.mean(torch.cosine_similarity(frame_0, frame_1),
                              dim=[1, 2]) - cos_sim_mean) / cos_sim_std).unsqueeze(1).to(frames.device)
    denoise_args = {"cond_frames": padder.pad(cond_frames), "difference": difference}
    with torch.no_grad():
        posterior, cond_tokens = vae.module.encode(padder.pad(frames))
        latent = (posterior.sample() - vae_mean).mul_(vae_scaler)
    if is_train:
        loss_dict = transport.training_losses(model, latent, **denoise_args)
        return loss_dict, latent, cond_tokens, denoise_args
    else:
        with torch.no_grad():
            noise = torch.randn_like(latent).to(frames.device)
            samples = sample_fn(noise, model.module.forward, **denoise_args)[-1]
            generated = vae.module.decode(samples / vae_scaler + vae_mean, cond_tokens)
            generated = padder.unpad(generated.clamp(0., 1.))
        return generated
