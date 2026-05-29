import torch
import torch.nn.functional as F
from dust3r.renderers.gaussian_utils import GaussianModel
import math
from gsplat.rendering import rasterization
import nerfview
from typing import Tuple
import imageio
import time
import math
import viser
device='cuda'
def render_image(pc, K, RT, height, width, bg_color=(0.0, 0.0, 0.0), scaling_modifier=1.0,debug=False):
    screenspace_points = (torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0)
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=K.device)
    if screenspace_points.requires_grad:
        screenspace_points.retain_grad()
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    colors_precomp = pc.get_features
    
    K[:1] = K[:1] * width
    K[1:2] = K[1:2] * height
    if colors_precomp.shape[1] == 3:
        sh_degree = None
    else:
        sh_degree = int(math.sqrt(colors_precomp.shape[1])) - 1
    render_colors, render_alphas, meta = rasterization(means = means3D, quats= rotations, scales = scales, opacities = opacity.squeeze(), colors = colors_precomp, sh_degree=sh_degree, viewmats = RT[None].inverse(), Ks=K[None][:,:3,:3], width=width, height=height, near_plane=0.00001, render_mode="RGB+D", radius_clip=0.1, camera_model="pinhole")#, rasterize_mode="antialiased")
    render_depths = render_colors.permute(0, 3, 1, 2).squeeze()[3:4, :, :]
    render_colors = render_colors.permute(0, 3, 1, 2).squeeze()[:3, :, :]
    render_alphas = render_alphas.permute(0, 3, 1, 2)[0]
    return {
        "image": render_colors,
        "alpha": render_alphas,
        "depth": render_depths,
    }

def viewer_render_fn(pc, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
    width, height = img_wh
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)
    viewmat = c2w.inverse()
    means = pc.get_xyz
    opacities = pc.get_opacity
    scales = pc.get_scaling
    quats = pc.get_rotation
    colors = pc.get_features
    if colors.shape[1] == 3:
        sh_degree = None
    else:
        sh_degree = int(math.sqrt(colors.shape[1])) - 1
    render_colors, render_alphas, meta = rasterization(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities.squeeze(),  # [N]
        colors,  # [N, S, 3]
        viewmat[None],  # [1, 4, 4]
        K[None],  # [1, 3, 3]
        width,
        height,
        sh_degree=sh_degree,
        render_mode="RGB",
        # this is to speedup large-scale rendering by skipping far-away Gaussians.
        radius_clip=3,
    )
    render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
    return render_rgbs

class GaussianRenderer:
    def __init__(self,
                 height=512,
                 width=512,
                 sh_degree=0,
                 bg_color=(0., 0., 0.),
                 scaling_modifier=1.0,
                 gs_kwargs=dict(),
                 ):
        self.height = height 
        self.width = width
        self.sh_degree = sh_degree
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier
        self.gs_kwargs = gs_kwargs

    def __call__(self, gs_params, Ks, RTs):
        Ks = Ks.to(torch.float32)
        RTs = RTs.to(torch.float32)
        device = RTs.device
        b, v = RTs.shape[:2]
        patchs = None
        colors_list = []
        depths_list = []
        alphas_list = []
        xyz = gs_params['xyz']
        feature = gs_params['feature']
        opacity = gs_params['opacity']
        scaling = gs_params['scaling']
        rotation = gs_params['rotation']
        #print()
        #print('from dust3r/renders/gaussian_renderer')
        #print('the info in the gs_params vector, right before rendering it ')
        #for k in gs_params.keys():
        #    print(k, gs_params[k].shape)
        #print('self.sh_degree',self.sh_degree)
        #print('feature and [0]', feature.shape, feature[0].shape)
        #print()
        scaling_kwargs = self.gs_kwargs
        for i in range(b):
            pc = GaussianModel(sh_degree=self.sh_degree, xyz=xyz[i], feature=feature[i], opacity=opacity[i],
                                    scaling=scaling[i], rotation=rotation[i], scaling_kwargs=scaling_kwargs)
            for j in range(v):
                K_ij = Ks[i, j]
                #print()
                #print('Ks, i, j, b, v: ' , Ks, i, j, b, v)
                #print('K_ij: ', K_ij,  K_ij[0], K_ij[1], K_ij[2], K_ij[3])
                #print()
                fx, fy, cx, cy = K_ij[0], K_ij[1], K_ij[2], K_ij[3]
                new_K_ij = torch.eye(4).to(K_ij)
                new_K_ij[0][0], new_K_ij[1][1], new_K_ij[0][2], new_K_ij[1][2], new_K_ij[2][2] = fx, fy, cx, cy, 1
                render_results = render_image(pc, new_K_ij, RTs[i, j], self.height, self.width)
                colors = render_results["image"]
                depths = render_results["depth"]
                alphas = render_results["alpha"]
                colors_list.append(colors)
                depths_list.append(depths)
                alphas_list.append(alphas)
        colors = torch.stack(colors_list, dim=0)
        depths = torch.stack(depths_list, dim=0)
        alphas = torch.stack(alphas_list, dim=0)
        ret = {'image': colors, 'alpha': alphas, 'depth': depths}
        return ret

     
class GaussianRenderer_P:
    def __init__(self,
                 height=512,
                 width=512,
                 sh_degree=0,
                 bg_color=(0., 0., 0.),
                 scaling_modifier=1.0,
                 gs_kwargs=dict(),
                 interactive: bool = False,
                 port: int = 8080,
                 ):
        self.height = height 
        self.width = width
        self.sh_degree = sh_degree
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier
        self.gs_kwargs = gs_kwargs
        self.interactive = interactive    # store flag
        self.port = port                  # store port

    def __call__(self, gs_params, Ks, RTs):
        Ks = Ks.to(torch.float32)
        RTs = RTs.to(torch.float32)
        device = RTs.device
        b, v = RTs.shape[:2]
        patchs = None
        colors_list = []
        depths_list = []
        alphas_list = []
        xyz = gs_params['xyz']
        feature = gs_params['feature']
        opacity = gs_params['opacity']
        scaling = gs_params['scaling']
        rotation = gs_params['rotation']
        #print()
        #print('from dust3r/renders/gaussian_renderer')
        #print('the info in the gs_params vector, right before rendering it ')
        #for k in gs_params.keys():
        #    print(k, gs_params[k].shape)
        #print('self.sh_degree',self.sh_degree)
        #print('feature and [0]', feature.shape, feature[0].shape)
        #print()
        scaling_kwargs = self.gs_kwargs
        for i in range(b):
            pc = GaussianModel(sh_degree=self.sh_degree, xyz=xyz[i], feature=feature[i], opacity=opacity[i],
                                    scaling=scaling[i], rotation=rotation[i], scaling_kwargs=scaling_kwargs)
            for j in range(v):
                K_ij = Ks[i, j]
                #print()
                #print('Ks, i, j, b, v: ' , Ks, i, j, b, v)
                #print('K_ij: ', K_ij,  K_ij[0], K_ij[1], K_ij[2], K_ij[3])
                #print()
                fx, fy, cx, cy = K_ij[0], K_ij[1], K_ij[2], K_ij[3]
                new_K_ij = torch.eye(4).to(K_ij)
                new_K_ij[0][0], new_K_ij[1][1], new_K_ij[0][2], new_K_ij[1][2], new_K_ij[2][2] = fx, fy, cx, cy, 1
                render_results = render_image(pc, new_K_ij, RTs[i, j], self.height, self.width)
                colors = render_results["image"]
                depths = render_results["depth"]
                alphas = render_results["alpha"]
                colors_list.append(colors)
                depths_list.append(depths)
                alphas_list.append(alphas)
        colors = torch.stack(colors_list, dim=0)
        depths = torch.stack(depths_list, dim=0)
        alphas = torch.stack(alphas_list, dim=0)
        ret = {'image': colors, 'alpha': alphas, 'depth': depths}

        # if interactive mode, launch viewer and block
        if self.interactive:
            import torch.nn.functional as F
            
            # capture local copies for closure
            means = gs_params['xyz']
            quats = F.normalize(gs_params['rotation'], p=2, dim=-1)
            scales = gs_params['scaling']
            opacities = gs_params['opacity'].squeeze(-1)
            colors_sh = gs_params['feature']
            sh_degree = self.sh_degree

            @torch.no_grad()
            def _live_render(camera_state, img_wh):
                return viewer_render_fn(pc,camera_state, img_wh)

            # start server & viewer
            server = viser.ViserServer(port=self.port, verbose=False)
            _ = nerfview.Viewer(
                server=server,
                render_fn=_live_render,
                mode="rendering"
            )
            print(f"Interactive viewer running on port {self.port}")
            import time; time.sleep(1e6)
        return ret