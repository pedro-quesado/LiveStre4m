
import os
import glob
import time
import yaml
import argparse
from math import inf

import numpy as np
import imageio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cuda.matmul.allow_tf32 = True  

# Third-party metrics
from lpips import LPIPS

import mast3r.utils.path_to_dust3r
from mast3r.metrics import compute_psnr, compute_lpips
from mast3r.losses_P import transpose_to_landscape_render
from dust3r.utils.geometry import inv

from utils.STREAME3R import Stream3r, EfficientImageLoader_v3
from utils.static_model import AsymmetricMASt3R_stream3r
from utils.training_utils import render_static

# ==========================================
# HARDCODED CONFIGURATIONS & CONSTANTS
# ==========================================
SCALING = 2
DTYPE = torch.bfloat16

# ==========================================
# RESOLUTION & MULTIPLIER REGISTRY
# ==========================================
RESOLUTION_CONFIGS = {
    (320, 256): 1.5,
    (576, 384): 1.9,
    (640, 384): 1.9,
    (704, 512): 1.9,
    (768, 576): 2.1,
    (512, 384): 1.9,
    (256, 192): 1.0
}

# Fallback definitions for variables seen in the snippets to prevent NameErrors
# You can replace these with your actual lists if iterating over multiple scenes
NEAR_LIST = [[0, 10, 20], [0, 10, 20], [0, 10, 20], [0, 10, 20], [0, 10, 20]] 
CENTRAL_VIEW_LIST = [10, 10, 10, 10, 10]
scene_id = 4 # Hardcoded to match your snippet (e.g., scene_id == 4)

static_args = dict(
    wpose=True, pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', 
    head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), 
    conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, 
    dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, 
    desc_conf_mode=('exp', 0, inf)
)

# ==========================================
# CAMERA CONFIGURATION REGISTRY
# ==========================================
# -1 dynamically refers to the last available camera view (N_VIEWS - 1)
CAMERA_CONFIGS = {
    "Neural_3D_Video": {
        "cut_roasted_beef": {
            "near": [5, 6, 15, 16],
            "mid": [3, 6, 13, 16],
            "far": [1, 9, 10, -1],
            "output": [0]
        },
        "sear_steak": {
            "near": [5, 6, 15, 16],
            "mid": [4, 7, 14, 17],
            "far": [1, 10, 11, -1],
            "output": [0]
        }
    },
    "MeetRoom": {
        "discussion": {
            "near": [0, 1, 8, 12],
            "output": [5] 
        },
        "trimming": {
            "near": [0, 1, 8, 12],
            "output": [5]
        },
        "vrheadset": {
            "near": [0, 1, 8, 12],
            "output": [5]
        }
    }
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def generate_novel_view_static_optim(pred1, pred2, view1, view2, render_cams, embed):
    render_landscape = transpose_to_landscape_render(render_static)
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

    return ret['images'].permute(0,3,1,2) #  [n_output_views, 3, H, W]

def get_views_static_no_cam_gt_rectangle(time_step, in_1, in_2s, scene, n_views, out_imgs, res, device, dtype):
    loader = EfficientImageLoader_v3(resolution=res, num_views=n_views, device=device, dtype=dtype)
    views = loader.load_views_from_dir(str(scene + f'/FRAMES/t{time_step}')) 

    view1 = [views[in_1]]
    view2 = [views[in_2] for in_2 in in_2s]
    render_gt = [views[gt_view] for gt_view in out_imgs]

    return render_gt, view1, view2

def get_views_STREAM3R(initial_t, final_t, in_1, in_2s, scene, n_views, out_imgs, res, device, dtype, scale=2):
    # Get ground truth images at high definition 
    hd_GT = []
    view1s = []
    view2s = []
    render_gts = []

    res_HD = (res[0]*scale, res[1]*scale)
    counter = 0
    for t in tqdm(range(initial_t, initial_t+final_t), desc="Loading Dataset Frames"):
        render, _, _ = get_views_static_no_cam_gt_rectangle(t, in_1, in_2s, scene, n_views, out_imgs, res_HD, device, dtype)
        hd_GT.append(render)

        if counter % 2 == 0:
            render_gt, v1, v2 = get_views_static_no_cam_gt_rectangle(t, in_1, in_2s, scene, n_views, out_imgs, res, device, dtype)
            view1s.append(v1)
            view2s.append(v2)
            render_gts.append(render_gt)
        counter += 1
    
    return hd_GT, view1s, view2s, render_gts


def main():
    parser = argparse.ArgumentParser(description="LiveStre4m Inference Script")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset scene (e.g., /projects/prjs1677/Neural_3D_Video/vrheadset)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to LiveStre4m_C_e5.pth weights')
    parser.add_argument('--resolution', type=int, nargs=2, default=[640, 384], help='Target Resolution (W H). Must be one of the predefined pairs.')
    parser.add_argument('--no_cam_optim', action='store_true', help='Skip test-time camera optimization and go directly to rendering')
    parser.add_argument('--online', action='store_true', default=False, help='Run inference in online mode')
    
    parser.add_argument('--dist', type=str, default='near', choices=['near', 'mid', 'far'], help='Camera distance mode')
    parser.add_argument('--num_inputs', type=int, default=2, choices=[2, 4], help='Number of input views (2 or 4)')
    parser.add_argument('--max_frames', type=int, default=300, help='Limit frames') 
    args = parser.parse_args()

    # VALIDATE RESOLUTION & SET MULTIPLIER
    res_tuple = tuple(args.resolution)
    if res_tuple not in RESOLUTION_CONFIGS:
        valid_res = [f"{w} {h}" for w, h in RESOLUTION_CONFIGS.keys()]
        raise ValueError(
            f"CRITICAL: Resolution {res_tuple} is not supported.\n"
            f"Please choose from one of the following (W H): {', '.join(valid_res)}"
        )
    
    multiplier = RESOLUTION_CONFIGS[res_tuple]
    print(f"--- Loaded Resolution: {res_tuple[0]}x{res_tuple[1]} | Multiplier: {multiplier} ---")

    t0_path = os.path.join(args.dataset, 'FRAMES', 't0')
    # Check for common image extensions
    image_files = (
        glob.glob(os.path.join(t0_path, '*.png')) + 
        glob.glob(os.path.join(t0_path, '*.jpg')) + 
        glob.glob(os.path.join(t0_path, '*.jpeg')) +
        glob.glob(os.path.join(t0_path, '*.JPG'))
    )
    N_VIEWS = len(image_files)
    if N_VIEWS == 0:
        raise ValueError(f"CRITICAL ERROR: No images found in {t0_path}. Please check your dataset path.")
    print(f"--> Dynamically detected N_VIEWS = {N_VIEWS} from {t0_path}")

    path_parts = os.path.normpath(args.dataset).split(os.sep)
    scene_name = path_parts[-1]
    dataset_name = path_parts[-2]

    # 1. Validate the dictionary mappings
    if dataset_name not in CAMERA_CONFIGS:
        raise ValueError(f"CRITICAL: Dataset '{dataset_name}' not found in CAMERA_CONFIGS.")
    if scene_name not in CAMERA_CONFIGS[dataset_name]:
        raise ValueError(f"CRITICAL: Scene '{scene_name}' not found for dataset '{dataset_name}'.")
    if args.dist not in CAMERA_CONFIGS[dataset_name][scene_name]:
        raise ValueError(f"CRITICAL: Distance '{args.dist}' not available for '{scene_name}'.")

    # 2. Fetch the raw lists
    cam_list = CAMERA_CONFIGS[dataset_name][scene_name][args.dist]
    out_imgs_raw = CAMERA_CONFIGS[dataset_name][scene_name]["output"]

    # 3. Apply custom logic based on dataset and number of inputs
    if args.num_inputs == 2:
        if dataset_name == "Neural_3D_Video":
            # Rule: Use idx 1 and 2 of the list
            in_1_raw = cam_list[1]
            in_2s_raw = [cam_list[2]]
        elif dataset_name == "MeetRoom":
            # Rule: Use idx 0 and -1 of the list
            in_1_raw = cam_list[0]
            in_2s_raw = [cam_list[-1]]
            
    elif args.num_inputs == 4:
        # Standard rule for 4 inputs: use all 4 cameras in the list
        in_1_raw = cam_list[0]
        in_2s_raw = cam_list[1:]

    # 4. Safely handle the '-1' logic based on dynamically calculated N_VIEWS
    in_1 = N_VIEWS - 1 if in_1_raw == -1 else in_1_raw
    in_2s = [N_VIEWS - 1 if x == -1 else x for x in in_2s_raw]
    out_imgs = [N_VIEWS - 1 if x == -1 else x for x in out_imgs_raw]

    print(f"--- Configuration Loaded: {dataset_name}/{scene_name} ---")
    print(f"Mode: {args.dist}, Inputs: {args.num_inputs}")
    print(f"Assigned Cameras -> in_1: {in_1}, in_2s: {in_2s}, out_imgs: {out_imgs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_fn = LPIPS(net="vgg").to(device)

    # Load interp configs
    with open("configs/interpolation/eval_eden.yaml", "r") as f:
        update_args = yaml.unsafe_load(f)

    static_args['img_size'] = args.resolution

    # ==========================================
    # PART 1 & 2: Loading Models and Data
    # ==========================================
    print(f"Loading weights from {args.ckpt_path}...")
    data = torch.load(args.ckpt_path, map_location=device)

    print("Initializing CamPred...")
    CamPred = AsymmetricMASt3R_stream3r(**static_args)

    unused = {
        'cam_cond_embed_point', 'cam_cond_embed_point_pre', 'cam_cond_encoder_point',
        'cnn_fusion', 'cnn_proj', 'cnn_wobn', 'dec_blocks_point', 'dec_blocks_point_cross',
        'dec_norm', 'dec_norm_point', 'decoder_embed_fxfycxcy', 'decoder_embed_point',
        'decoder_embed_stage2', 'downstream_head1', 'downstream_head2', 'downstream_head4',
        'enc_blocks_stage2', 'enc_inject_stage3', 'enc_norm_stage2', 'inject_stage3',
        'mask_generator', 'patch_embed_fine'
    }
    for prefix in unused:
        if hasattr(CamPred, prefix):
            delattr(CamPred, prefix)

    CamPred.load_state_dict(data['CamPred'], strict=True)
    CamPred = CamPred.to(device).to(DTYPE)
    for p in CamPred.parameters():
        p.requires_grad = False
    CamPred.eval()

    print("Initializing LiveStre4m...")
    LiveStre4m = Stream3r(
        static_args=static_args,
        scaling_factor=SCALING,
        interp_args=update_args,
        training=False
    )
    LiveStre4m.load_state_dict(data['model_state'], strict=True)
    LiveStre4m.to(device).to(DTYPE)
    for p in LiveStre4m.parameters():
        p.requires_grad = False
    LiveStre4m.eval()

    print('Loading scene:', args.dataset, 'at resolution', args.resolution)
    
    hd_GT, view1s, view2s, render_gts = get_views_STREAM3R(
        initial_t=0, 
        final_t=args.max_frames, 
        scene=args.dataset, 
        n_views=N_VIEWS,
        in_1=in_1,           
        in_2s=in_2s,         
        out_imgs=out_imgs,   
        res=(args.resolution[1], args.resolution[0]),
        device=device,
        dtype=DTYPE
    )

    # ==========================================
    # PART 3: Camera Embeddings and Optimization
    # ==========================================
    bottom = torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float32).view(1, 1, 4)

    print("Extracting Camera Embeddings via CamPred...")
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=DTYPE):
        cams, focals, cam_embeds = CamPred.forward_cams_embed_focal_adapt(view1s[0], view2s[0])

    cams = cams.unsqueeze(0)
    n = cams.shape[1]
    E = torch.cat([cams, bottom.view(1, 1, 1, 4).expand(1, n, 1, 4)], dim=-2)

    # Map focal lengths and pose to all timesteps
    for t in range(len(view1s)):
        for view_idx, _ in enumerate(view1s[t] + view2s[t]):
            pose = E[:, view_idx, :, :]
            fx = focals[0, view_idx, 0] * args.resolution[0] * multiplier
            fy = focals[0, view_idx, 1] * args.resolution[1] * multiplier

            new_fxfycxcy = view1s[t][0]['fxfycxcy_unorm'].clone() if view_idx == 0 else view2s[t][view_idx - 1]['fxfycxcy_unorm'].clone() 
            new_fxfycxcy[0, 0] = fx
            new_fxfycxcy[0, 1] = fy

            if view_idx == 0:
                view1s[t][0]['camera_pose'] = pose
                view1s[t][0]['fxfycxcy_unorm'] = new_fxfycxcy
            else:
                view2s[t][view_idx - 1]['camera_pose'] = pose
                view2s[t][view_idx - 1]['fxfycxcy_unorm'] = new_fxfycxcy
        
        for view_idx in range(len(render_gts[t])):
            render_gts[t][view_idx]['camera_pose'] = view1s[t][0]['camera_pose'].clone()
            render_gts[t][view_idx]['fxfycxcy_unorm'] = view1s[t][0]['fxfycxcy_unorm'].clone()

    cam_embed_v1 = cam_embeds[0] 
    cam_embed_v2 = cam_embeds[1]

    if not args.no_cam_optim:
        print("--- Running Camera Refinement ---")
        base_pose = render_gts[0][0]['camera_pose'][:, :3, :].to(torch.float32).detach()
        pose_delta = nn.Parameter(torch.zeros_like(base_pose))
        
        optimizer = torch.optim.Adam([pose_delta], lr=0.005, betas=(0.85, 0.99))
        torch.autograd.set_detect_anomaly(True)
        
        optim_iter = 700
        warmup_iters = 50
        alphas = [0.8, 0.5]
        noise_mod = 0.003
        
        iterations_bar = tqdm(range(optim_iter))
        
        for iteration in iterations_bar:
            if iteration < warmup_iters:
                lr_scale = float(iteration + 1) / warmup_iters
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.005 * lr_scale
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.005

            optimizer.zero_grad()
            
            refined_pose_full = torch.cat([base_pose + pose_delta, bottom.expand(base_pose.shape[0], 1, 4)], dim=1)
            render_gts[0][0]['camera_pose'] = refined_pose_full

            view1s_detached = [{k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in view.items()} for view in view1s[0]]
            view2s_detached = [{k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in view.items()} for view in view2s[0]]
            
            if iteration == 0:
                with torch.cuda.amp.autocast(dtype=DTYPE):
                    model_dtype = next(LiveStre4m.static_module.parameters()).dtype
                    aux_view_num = len(view2s_detached)
                    pred1, pred2, _ = LiveStre4m.static_module.forward_gs_use_cam_embed(
                        view1s_detached, view2s_detached, cam_embed=[cam_embed_v1, cam_embed_v2], 
                        enabled=False, dtype=model_dtype, aux_view_num=aux_view_num
                    )
                    
            with torch.cuda.amp.autocast(dtype=DTYPE):
                outs = generate_novel_view_static_optim(
                    pred1, pred2,
                    view1s_detached, view2s_detached, render_gts[0],
                    embed=[cam_embed_v1, cam_embed_v2]
                )
                
            with torch.cuda.amp.autocast(enabled=False):
                GT = render_gts[0][0]['img']
                GT = (GT + 1) / 2
                mse_loss = ((outs - GT) ** 2).mean()
                lpips_val = lpips_fn(outs, render_gts[0][0]['img']).mean()
                total_loss = alphas[0] * mse_loss + alphas[1] * lpips_val

            total_loss.backward()
            iterations_bar.set_description(f'loss={total_loss.item():.4f}, grad={float(pose_delta.grad.norm()):.4f}')
            
            optimizer.step()
            
            if iteration % 100 == 0 and iteration > 0:
                with torch.no_grad():
                    pose_delta += torch.randn_like(pose_delta) * noise_mod * (1.0 - iteration / optim_iter)

    # ==========================================
    # PART 4: Inference, Cropping, and Metrics
    # ==========================================
    rendered_frames = []

    print(f"Starting {'ONLINE' if args.online else 'OFFLINE'} inference...")
    
    if not args.online:
        start_time_livestre4m = time.perf_counter()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=DTYPE):
            outs = LiveStre4m.forward_offline(view1s, view2s, render_gts[0], cam_embed=[cam_embed_v1, cam_embed_v2])
        end_time_full = time.perf_counter()

        big_pred = torch.stack(outs, dim=0)  
        try:
            N, V, C, H, W = big_pred.shape
        except ValueError:
            big_pred = big_pred.unsqueeze(1)
            N, V, C, H, W = big_pred.shape
        
        big_pred_batched = big_pred.view(N * V, C, H, W)
        pred_cpu = big_pred_batched.float().detach().cpu()

        # Stack Ground Truth
        gt_batches = []
        for gt_time in hd_GT:
            images_gt = torch.stack([gt['img'] for gt in gt_time], dim=1).squeeze(0)
            images_gt = images_gt / 2 + 0.5 
            gt_batches.append(images_gt)
            
        big_gt = torch.stack(gt_batches, dim=0)  
        try:
            big_gt_batched = big_gt.view(N * V, C, H, W).to(device)
        except RuntimeError:
            print("Padding frames to match GT shape...")
            big_gt_batched = big_gt.view((N + 1) * V, C, H, W).to(device)
            last_frame = pred_cpu[-1:].clone()
            pred_cpu = torch.cat([pred_cpu, last_frame], dim=0)

        gt_cpu = big_gt_batched.float().detach().cpu()

        # Apply Dataset-Specific Cropping
        dataset_name_lower = args.dataset.lower()
        if any(kw in dataset_name_lower for kw in ["flame_steak", "roasted_beef", "sear_steak", "cut", "cut_roasted_beef", "sear"]):
            print("Applying N3DV Dataset Crop...")
            gt_cpu_crop = gt_cpu[:, :, 5:-5, 28:-28]
            pred_cpu_crop = pred_cpu[:, :, 5:-5, 28:-28]
        else:
            print("Applying MeetRoom Dataset Crop...")
            gt_cpu_crop = gt_cpu[:, :, 24:-24, :]
            pred_cpu_crop = pred_cpu[:, :, 24:-24, :]

        # Calculate Metrics
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=False):
            psnr = compute_psnr(gt_cpu_crop, pred_cpu_crop).mean()
            # If you want LPIPS over full sequence, you can compute it here but it's memory heavy
            # lpips_metric = compute_lpips(gt_cpu_crop.to(device), pred_cpu_crop.to(device)).mean() 

        print("\n" + "="*30)
        print("INFERENCE COMPLETE")
        print("="*30)
        print(f'PSNR: {psnr.item():.4f}')
        print(f'Full time: {end_time_full - start_time_livestre4m:.2f} seconds') 
        print(f'Time per frame: {(end_time_full - start_time_livestre4m) / pred_cpu.shape[0]:.4f} seconds') 
        
        # Prepare video frames
        for t in range(pred_cpu_crop.shape[0]):
            pred_clamped = torch.clamp(pred_cpu_crop[t], 0, 1)
            frame_np = pred_clamped.permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
            rendered_frames.append(frame_np)

    else:
        # TODO: Implement online inference frame-by-frame
        print("Online inference is currently not published.")
        pass

    # Save Video
    if len(rendered_frames) > 0:
        scene_name = os.path.basename(os.path.normpath(args.dataset))
        mode_str = "online" if args.online else "offline"
        output_video_path = f"output_{scene_name}_{mode_str}.mp4"
        
        print(f"\nSaving video to {output_video_path} at 30 FPS...")
        imageio.mimwrite(output_video_path, rendered_frames, fps=30, macro_block_size=1)
        print("Done!")

if __name__ == "__main__":
    main()