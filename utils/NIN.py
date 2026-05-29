import torch
from utils.super_res_utils.models import QuickSRNetSmall
from utils.interpolation_models.EDEN import EDEN
from utils.interpolation_transport import create_transport, Sampler
from utils.interpolation_utils import InputPadder
from types import SimpleNamespace

class NIN_model(torch.nn.Module):
    def __init__(self, interp_config:dict, scaling_factor:int):
        super().__init__()

        self.SR_module = QuickSRNetSmall(scaling_factor=scaling_factor)
        model_interp_name = 'EDEN'
        self.interp_config = SimpleNamespace(**interp_config)
        self.interp_module = EDEN(**interp_config['model_args'])
        transport = create_transport("Linear", "velocity")
        sampler = Sampler(transport)
        self.sample_fn = sampler.sample_ode(sampling_method="euler", num_steps=2, atol=1e-6, rtol=1e-3)

    def run_interpolation(self, frame0, frame1):
        n, c, h, w = frame0.shape
        
        device = frame0.device
        image_size = [h, w]
        padder = InputPadder(image_size)
        difference = ((torch.mean(torch.cosine_similarity(frame0, frame1),
                                dim=[1, 2]) - self.interp_config.cos_sim_mean) / self.interp_config.cos_sim_std).unsqueeze(1).to(device)
        cond_frames = padder.pad(torch.cat((frame0, frame1), dim=0))
        new_h, new_w = cond_frames.shape[2:]
        noise = torch.randn([n, new_h // 32 * new_w // 32, self.interp_config.model_args['latent_dim']]).to(device)
        denoise_kwargs = {"cond_frames": cond_frames, "difference": difference}
        samples = self.sample_fn(noise, self.interp_module.denoise, **denoise_kwargs)[-1]
        denoise_latents = samples / self.interp_config.vae_scaler + self.interp_config.vae_shift
        generated_frame = self.interp_module.decode(denoise_latents)
        generated_frame = padder.unpad(generated_frame.clamp(0., 1.))
        return generated_frame
    
    def run_super_resolution(self, inputs_lr):
        images_sr = []
        for count, img_lr in enumerate(inputs_lr):
            img_lr = img_lr.squeeze(0)
            sr_img = self.SR_module(img_lr)
            images_sr.append(sr_img)

        return images_sr

    def forward(self, frame_t0, frame_t2,):
        frame_t1 = self.run_interpolation(frame_t0, frame_t2)
        SR_frames = self.run_super_resolution([frame_t0, frame_t1, frame_t2])

        return SR_frames



