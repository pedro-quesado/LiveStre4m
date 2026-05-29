import torch
import types
import gc

from dust3r.renderers.gaussian_renderer_P import GaussianRenderer
import numpy as np

import torch

def render_static(render_params, image_size):
    scaling_mode = 'precomp_5e-4_0.3_4'
    scaling_mode = {'type':scaling_mode.split('_')[0], 'min_scaling': eval(scaling_mode.split('_')[1]), 'max_scaling': eval(scaling_mode.split('_')[2]), 'shift': eval(scaling_mode.split('_')[3])}
    
    latent, output_fxfycxcy, output_c2ws = render_params
    (H_org, W_org) = image_size
    H = H_org
    W = W_org
    new_latent = {}
    if scaling_mode['type'] == 'precomp': 
        scaling_factor = latent['scaling_factor']
        x = torch.clip(latent['pre_scaling'] - scaling_mode['shift'], max=np.log(0.3))
        new_latent['scaling'] = torch.exp(x).clamp(min=scaling_mode['min_scaling'], max=scaling_mode['max_scaling']) / scaling_factor
        skip = ['pre_scaling', 'scaling', 'scaling_factor']
    else:
        skip = ['pre_scaling', 'scaling_factor']
    for key in latent.keys():
        if key not in skip:
            new_latent[key] = latent[key]
    gs_render = GaussianRenderer(H_org, W_org, gs_kwargs={'type':scaling_mode['type'], 'min_scaling': scaling_mode['min_scaling'], 'max_scaling': scaling_mode['max_scaling'], 'scaling_factor': scaling_factor})
    results = gs_render(new_latent, output_fxfycxcy.reshape(1,-1,4), output_c2ws.reshape(1,-1,4,4))
    images = results['image']
    images = images.reshape(-1,3,H,W).permute(0,2,3,1)
    return {'images': images}
