
from utils.static_model import AsymmetricMASt3R_stream3r, AsymmetricMASt3R_stream3r_optimized 
from utils.NIN import NIN_model

from dust3r.utils.geometry import normalize_pointcloud, xy_grid, inv, matrix_to_quaternion
from mast3r.losses_P import rotation_6d_to_matrix, transpose_to_landscape_render
from lpips import LPIPS
from mast3r.metrics import compute_pose_error, compute_lpips, compute_psnr, compute_ssim
from dust3r.renderers.gaussian_renderer_P import GaussianRenderer
import numpy as np
from dust3r.datasets.CustomDataset import CustomDataset
from dust3r.datasets import get_data_loader  
import torch 

import cv2
import numpy as np
import torch
import os
import os.path as osp
import glob
from typing import List, Tuple, Optional
from dust3r.datasets.utils.transforms import ImgNorm
import torchvision.transforms as transforms

import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import functional as TF


class EfficientImageLoader_v3:
    """
    Image loader supporting non-square / variable aspect ratio frames.

    Usage:
        loader = EfficientImageLoader_v3(
            resolution=(H, W),
            original_shape=(orig_H, orig_W),
            num_views=3,
            device=device,
            dtype=torch.float32,
            transform=ImgNorm,
            file_globs=["*.png", "*.jpg"],
            select_strategy="first"
        )
        views = loader.load_views_from_dir(DATA_DIR)
    """

    def __init__(
        self,
        resolution: (int, int),
        original_shape= None,
        num_views: int = 1,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        transform: Optional[transforms.Compose] = None,
        file_globs: Optional[List[str]] = None,
        select_strategy: str = "first",
    ):
        # resolution = output H, W you want (target)
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self.resolution = tuple(resolution)  # (H_out, W_out)
        self.num_views = int(num_views)
        self.device = device
        self.dtype = dtype

        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.file_globs = file_globs or ["*.png", "*.jpg", "*.jpeg", "*.JPG"]
        self.select_strategy = select_strategy

        # If original_shape known (H_orig, W_orig), we can compute intrinsics scale
        if original_shape is not None:
            self.orig_shape = tuple(original_shape)
        else:
            self.orig_shape = None

        # Precompute identity pose template
        self._pose_template = (
            torch.eye(4, dtype=self.dtype, device=self.device).unsqueeze(0)
        )

        self._dir = None
        self._files = []

    def set_data_dir(self, data_dir: str):
        self._dir = data_dir
        files = []
        for patt in self.file_globs:
            files.extend(glob.glob(osp.join(data_dir, patt)))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(
                f"No image files in {data_dir} matching {self.file_globs}"
            )
        self._files = files

    def _select_paths(self) -> List[str]:
        if len(self._files) < self.num_views:
            raise ValueError(
                f"Need at least {self.num_views} images in {self._dir}, got {len(self._files)}"
            )
        if self.select_strategy == "first":
            return self._files[: self.num_views]
        if self.select_strategy == "center":
            mid = len(self._files) // 2
            start = max(0, mid - self.num_views // 2)
            return self._files[start : start + self.num_views]
        # evenly spaced
        idxs = [
            int(i * len(self._files) / self.num_views) for i in range(self.num_views)
        ]
        return [self._files[i] for i in idxs]

    def _read_and_resize_tensor(self, path: str) -> (torch.Tensor, (int, int)):
        """
        Returns:
            img: (C, H_out, W_out) float in [0,1], after cropping/resizing
            (h_orig, w_orig): original shape of the read image
        """
        img = read_image(path)  # (C, H_orig, W_orig) uint8
        img = img.float() / 255.0
        C, H0, W0 = img.shape

        # Save original shape for intrinsics adjustment
        orig = (H0, W0)

        # Crop or pad to maintain aspect ratio, or simply center-crop the longer side
        # We'll center-crop the larger dimension so we keep full coverage
        target_H, target_W = self.resolution
        ratio_out = target_W / target_H
        ratio_orig = W0 / H0

        if ratio_orig > ratio_out:
            # orig is wider; crop width
            new_w = int(ratio_out * H0)
            x0 = (W0 - new_w) // 2
            img = img[:, :, x0 : x0 + new_w]
        elif ratio_orig < ratio_out:
            # orig is taller; crop height
            new_h = int(W0 / ratio_out)
            y0 = (H0 - new_h) // 2
            img = img[:, y0 : y0 + new_h, :]

        # Now img is aspect-matched. Resize to (target_H, target_W)
        img = img.unsqueeze(0)  # (1, C, H_crop, W_crop)
        if (img.shape[2], img.shape[3]) != self.resolution:
            img = F.interpolate(
                img, size=self.resolution, mode="bilinear", align_corners=False
            )
        img = img.squeeze(0)

        return img, orig

    def load_views(self) -> List[dict]:
        if not self._files:
            raise RuntimeError("Call set_data_dir first")
        paths = self._select_paths()
        views = []
        for p in paths:
            img, (H0, W0) = self._read_and_resize_tensor(p)

            # apply transform
            try:
                img_t = self.transform(img)
            except Exception:
                from torchvision.transforms.functional import to_pil_image

                img_t = self.transform(to_pil_image(img))

            if not torch.is_tensor(img_t):
                img_t = torch.as_tensor(img_t)
            img_t = img_t.to(device=self.device, dtype=self.dtype)

            
            # If original shape known or captured, do:
            if self.orig_shape is not None:
                H_orig, W_orig = self.orig_shape
            else:
                H_orig, W_orig = H0, W0

            scale_x = 1# self.resolution[1] / W_orig
            scale_y = self.resolution[1]/self.resolution[0] # self.resolution[0] / H_orig
            # you might choose a symmetric scale = mean or use x-scale and y-scale separately
            fx = 1.0 * scale_x
            fy = 1.0 * scale_y
            cx = 0.5  # principal point in normalized coords
            cy = 0.5

            fxfycxcy = torch.tensor(
                [fx, fy, cx, cy], dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            fxfycxcy_unorm = torch.tensor(
                [fx * self.resolution[0], fy * self.resolution[1], cx* self.resolution[1], cy* self.resolution[0]],
                dtype=self.dtype,
                device=self.device,
            ).unsqueeze(0)

            pose = self._pose_template.clone()

            view = {
                "img": (img_t.unsqueeze(0) * 2) - 1,
                "camera_pose": pose,
                "fxfycxcy": fxfycxcy,
                "true_shape": torch.tensor(
                    [[self.resolution[0], self.resolution[1]]], dtype=self.dtype, device=self.device
                ),
                "fxfycxcy_unorm": fxfycxcy_unorm,
            }
            views.append(view)

        return views

    def load_views_from_dir(self, data_dir: str) -> List[dict]:
        self.set_data_dir(data_dir)
        return self.load_views()


class Stream3r(torch.nn.Module):
    def __init__(self, static_args:dict, scaling_factor, interp_args:dict, training:bool):
        super().__init__()
        
        # static NVS modules
        if training:
            self.static_module = AsymmetricMASt3R_stream3r(**static_args)
        else:
            self.static_module = AsymmetricMASt3R_stream3r_optimized(**static_args)

        # NIN module
        self.NIN_module = NIN_model(scaling_factor=scaling_factor, interp_config=interp_args)

        self.prev_output = None
    
    def render(self, render_params, image_size):
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

    def generate_novel_view(self, view1, view2, render_cams, cam_embed=None):
  
        model_dtype = next(self.parameters()).dtype
        aux_view_num = len(view2)
        if cam_embed is None:
            pred1, pred2, _ = self.static_module(view1, view2, enabled=False, dtype=model_dtype, aux_view_num=aux_view_num)
        else:
            pred1, pred2, _ = self.static_module.forward_gs_use_cam_embed(view1, view2, cam_embed=cam_embed, enabled=False, dtype=model_dtype, aux_view_num=aux_view_num)

        render_landscape = transpose_to_landscape_render(self.render)
        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d']
        B, H, W, _ = pr_pts1.shape
        sh_dim = pred1['feature'].shape[-1]
        feature1 = pred1['feature'].reshape(B,H,W,sh_dim)
        feature2 = pred2['feature'].reshape(B,-1,W,sh_dim)
        feature = torch.cat((feature1, feature2), dim=1).float()
        opacity1 = pred1['opacity'].reshape(B,H,W,1)
        opacity2 = pred2['opacity'].reshape(B,-1,W,1)
        opacity = torch.cat((opacity1, opacity2), dim=1).float()
        
        scaling1 = pred1['scaling'].reshape(B,H,W,3)
        scaling2 = pred2['scaling'].reshape(B,-1,W,3)
        scaling = torch.cat((scaling1, scaling2), dim=1).float()
        rotation1 = pred1['rotation'].reshape(B,H,W,4)
        rotation2 = pred2['rotation'].reshape(B,-1,W,4)
        rotation = torch.cat((rotation1, rotation2), dim=1).float()

        output_fxfycxcy = torch.stack([gt['fxfycxcy'] for gt in render_cams], dim=1)
        shape = torch.stack([gt['true_shape'] for gt in render_cams], dim=1)
        pr_pts2_split = torch.split(pr_pts2, 1, dim=0)  # list of 2 tensors
        xyz = torch.cat([pr_pts1] + list(pr_pts2_split), dim=1)  # → [1, 768, 256, 3]
        
        output_c2ws = torch.stack([gt['camera_pose'] for gt in render_cams], dim=1)
        camera_pose = torch.stack([gt1_['camera_pose'] for gt1_ in view1], dim=1)
        
        with torch.cuda.amp.autocast(enabled=False):
            cam_pose32 = camera_pose.float()
            in_camera1 = inv(cam_pose32)

        output_c2ws = torch.einsum('bnjk,bnkl->bnjl', in_camera1.repeat(1,output_c2ws.shape[1],1,1), output_c2ws)
        output_c2ws[..., :3, 3:] = output_c2ws[..., :3, 3:]

        B = output_c2ws.shape[0]
    
        latent = {'xyz': xyz.reshape(B, -1, 3), 'feature': feature.reshape(B, -1, sh_dim),
                 'opacity': opacity.reshape(B, -1, 1), 'pre_scaling': scaling.reshape(B, -1, 3).clone(),
                 'rotation': rotation.reshape(B, -1, 4), 'scaling_factor': 1}
        ret = render_landscape([latent, output_fxfycxcy.reshape(-1,4), output_c2ws.reshape(-1,4,4)], shape.reshape(-1,2))
    
        return ret['images'].permute(0,3,1,2) #  [n_output_views, 3, H, W]]
    
    def forward_offline(self, view1s, view2s, render_cams, cam_embed=None):
        '''
        params
        - view1s (list of dict) composed of views from cam00 accross time domain
        - view2s (list of list of dict) composed of a list of auxiliar views [cam001, ...] accross time domain
        - render_cams (list of dict) with desired camera parameters 
        ''' 
        # BECAUSE ITS OFFLINE YOU GET THE ENTIRE VIDEO UPFRONT
        finals = []
   


        for t in range(len(view1s)):
           
            out_t = self.generate_novel_view(view1s[t], view2s[t], render_cams, cam_embed)
            
            if t > 1:
                f = self.NIN_module(out_previous, out_t) 
                # f is a list with 3 tensor (t0, t1, t2) but after first iteration we already
                # have t_previous, so we just need t_now, t_after
                # each tensor has [n_output_views, 3, H, W]
                finals.append(f[1])
                finals.append(f[2])

            elif t == 1:
                f = self.NIN_module(out_previous, out_t) 
                # f is a list with 3 tensor (t0, t1, t2) 
                # each tensor has [n_output_views, 3, H, W]
                finals.append(f[0])
                finals.append(f[1])
                finals.append(f[2])
            
            
            out_previous = out_t
        out_previous = None
        return finals

    def forward_once(self, view1, view2, render_cam, cam_embed=None):
        
        out = self.generate_novel_view(view1, view2, render_cam, cam_embed=cam_embed)
        return out

    def forward_online(self, view1: dict, view2: list, render_cams: list):
        """
        Online forward pass: processes a single timestep using current and previous frames.

        Args:
            view1:       dict for current frame from cam00
            view2:       list of dicts for current auxiliary views [cam01, ...]
            render_cams: list of dicts with render camera parameters

        Returns:
            None on first call; list of novel-view images (torch.Tensor) on subsequent calls.
        """
        # Step 1: Generate output from current views
        current_output = self.generate_novel_view(view1, view2, render_cams)

        # Step 2: If previous output exists, feed to NIN
        if self.prev_output is not None:
            fused_output = self.NIN_module(self.prev_output, current_output)
        else:
            fused_output = None  # First frame, nothing to fuse

        # Step 3: Update memory for next step
        self.prev_output = current_output

        return fused_output



