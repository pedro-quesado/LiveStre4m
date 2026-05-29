<p align="center">
  <h1 align="center">LiveStre4m: Feed-Forward Live Streaming of Novel Views from Unposed Multi-View Video</h1>
  <p align="center"> <strong>3DMV@CVPRW 2026</strong></p>
  <h3 align="center"><a href="https://arxiv.org/abs/2604.06740">Paper</a> | <a href="https://pedro-quesado.github.io/LiveStre4m/">Project Page</a></h3>
</p>

<p align="center"> <img src="assets/intro.png" width="95%"> </p>
<p align="left"><i>Figure 1. Illustration of the proposed LiveStre4m method, a feed-forward model for live-streaming novel viewpoint video from two or more low-resolution input streams.</i></p>

## Introduction

LiveStre4m is a feed-forward model for real-time novel view synthesis (NVS) that enables live streaming from as few as two unposed, multi-view inputs by predicting camera parameters directly from RGB video streams. By combining a multi-view vision transformer for 3D scene reconstruction with a diffusion-transformer interpolation module, the system achieves an average reconstruction time of 0.07s per frame at 1024x768 resolution, outperforming optimization-based dynamic scene representation methods in runtime.

<p align="center"> <img src="assets/results.png" width="85%"> </p>
<p align="left"><i>Figure 2. Qualitative results produced by LiveStre4m, synthesizing the target viewpoint using only two neighboring input views, without requiring optimization or ground-truth camera parameters.</i></p>

---

## 🛠️ Environment Setup

This project requires **CUDA 12.1**. To install all dependencies, clone the repository and build the Conda environment:

```bash
cd LiveStre4m
conda env create -f environment.yml
conda activate LiveStre4m
```

---

## Data Preparation

LiveStre4m infers camera geometry on the fly, meaning **no COLMAP data or pre-computed camera poses are required** only time-synchronized video frames.

To evaluate on benchmarks like **Neural3DVideo (N3DV)** or **MeetRoom**, please refer to their original repositories to obtain the raw multi-view video datasets. Once you have the raw videos inside a scene folder, run our fast extraction script:

```bash
python prepare_dataset.py --dataset <dataset_name>/<scene_name> --max_frames 300
```

This will extract and automatically organize the frames into the required directory structure:

```text
<dataset_name>/<scene_name>/
    └── FRAMES/
        ├── t0/                 # Timestep 0 (First frame of the video)
        │   ├── cam_00.png      # Camera views (zero-padded, alphabetically sorted)
        │   ├── cam_01.png      
        │   └── ...
        ├── t1/                 # Timestep 1
        │   ├── cam_00.png      
        │   └── ...
        └── ...
```

---

## 📦 Model Weights

Download the pre-trained LiveStre4m checkpoint (5GB) and place it in the `checkpoints/` directory before running inference.

[📥 Download LiveStre4m.pth](https://huggingface.co/Pedro-Quesado/LiveStre4m/tree/main/checkpoints)

Alternatively, you can download it directly via the command line:

```bash
# Install the Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Download the model directly into the checkpoints/ folder
huggingface-cli download Pedro-Quesado/LiveStre4m checkpoints/LiveStre4m.pth --local-dir .
```

## 🚀 Inference

Run the novel view synthesis pipeline using the following command:

```bash
python INFERENCE.py \
    --dataset <dataset_name>/<scene_name> \
    --ckpt_path checkpoints/LiveStre4m.pth \
    --resolution 640 384
```

**Optional Flags:**
* `--no_cam_optim`: Skip test-time camera optimization and go directly to rendering (useful for quick previews).
* `--optim_iters 250`: Change the number of camera refinement iterations (default is 1000).

---

## Results

### Dynamic Scene Reconstruction 
Quantitative comparison against optimization methods on the Neural3DVideo dataset on a single A100 GPU.

| Category | Method | Runtime (s) ↓ | PSNR ↑ | Camera Free | #Views |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Video Optimization** | K-planes | 48.00 | 32.17 | ❌ | >19 |
| | 4DGS | 7.80 | 32.70 | ❌ | ≥19 |
| | Spacetime-GS | 48.00 | 33.71 | ❌ | ≥19 |
| **Frame Optimization** | StreamRF | 15.00 | 32.09 | ❌ | >19 |
| | 3DGStream | 16.93 | 32.75 | ❌ | ≥19 |
| | IGS | 2.67 | 33.89 | ❌ | ≥19 |
| **Feed-forward** | **LiveStre4m (Ours)** | **0.14** | 20.64 | ✅ | **2** |

*(Note: MeetRoom results follow a similar trend, where LiveStre4m operates in 0.10s using only 2 views.)*

### Feed-Forward Scene Reconstruction
Comparison of feed-forward, camera-free scene reconstruction methods on a single H100 GPU.

| Dataset | Method | Runtime (s) ↓ | PSNR ↑ | Resolution | Pose Free | #Views |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Neural3DVideo** | FLARE | 0.249 | 21.45 | 512x384 | ✅ | 2 |
| | **LiveStre4m (Ours)** | **0.062** | **22.44** | 512x384 | ✅ | 2 |
| | **LiveStre4m (Ours)** | 0.074 | 21.11 | **1024x768** | ✅ | 2 |
| **MeetRoom** | FLARE | 0.243 | 16.65 | 512x384 | ✅ | 2 |
| | **LiveStre4m (Ours)** | **0.062** | **19.32** | 512x384 | ✅ | 2 |
| | **LiveStre4m (Ours)** | 0.074 | 18.65 | **1024x768** | ✅ | 2 |

## Acknowledgement

We adapted some code from some awesome repositories including [FLARE](https://github.com/ant-research/FLARE), [EDEN](https://github.com/bbldcver/EDEN), [Dust3R](https://github.com/naver/dust3r), and [MASt3R](https://github.com/naver/mast3r). Thanks for making the code bases publicly available.

## Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{quesado2026livestre4mfeedforwardlivestreaming,
      title={LiveStre4m: Feed-Forward Live Streaming of Novel Views from Unposed Multi-View Video}, 
      author={Pedro Quesado and Erkut Akdag and Yasaman Kashefbahrami and Willem Menu and Egor Bondarev},
      year={2026},
      eprint={2604.06740},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2604.06740](https://arxiv.org/abs/2604.06740)}, 
}
```
