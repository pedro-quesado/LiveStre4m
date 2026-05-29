import os
import glob
import cv2
import argparse
import subprocess
import shutil
import concurrent.futures
from tqdm import tqdm

def get_video_info(video_path):
    """Instantly grab frame count using OpenCV properties."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def extract_single_camera(args):
    """Runs FFmpeg as a subprocess for a single video."""
    cam_idx, video_path, max_frames, temp_dir = args
    
    # Create a dedicated temp folder for this camera
    cam_temp_dir = os.path.join(temp_dir, f"cam_{cam_idx:02d}")
    os.makedirs(cam_temp_dir, exist_ok=True)

    output_pattern = os.path.join(cam_temp_dir, "%05d.png")
    
    cmd = [
        "ffmpeg", 
        "-y", "-hide_banner", "-loglevel", "error", # Run silently
        "-i", video_path, 
        "-vframes", str(max_frames),                # Limit to exact sync frames
        output_pattern
    ]
    
    subprocess.run(cmd, check=True)
    return cam_idx, cam_temp_dir

def extract_frames_fast(dataset_path, max_frames_limit=None):
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(dataset_path, ext)))
    
    video_paths = sorted(video_paths)
    num_cameras = len(video_paths)
    
    if num_cameras == 0:
        print(f"Error: No video files found in {dataset_path}")
        return

    print(f"Found {num_cameras} video files.")

    # Find lowest frame count for perfect sync
    frame_counts = [get_video_info(vp) for vp in video_paths]
    total_sync_frames = min(frame_counts)
    
    if max_frames_limit and max_frames_limit < total_sync_frames:
        total_sync_frames = max_frames_limit
        
    print(f"Targeting {total_sync_frames} synchronized frames...")

    # Setup Directories
    frames_dir = os.path.join(dataset_path, 'FRAMES')
    temp_dir = os.path.join(dataset_path, 'TEMP_EXTRACTION')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Extract all cameras IN PARALLEL using FFmpeg
    print("Extracting frames via FFmpeg in parallel ...")
    tasks = [(i, video_paths[i], total_sync_frames, temp_dir) for i in range(num_cameras)]
    
    cam_temp_dirs = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(extract_single_camera, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_cameras, desc="Cameras Processed"):
            cam_idx, cam_temp_dir = future.result()
            cam_temp_dirs[cam_idx] = cam_temp_dir

    # Reorganize into the FRAMES/t{X} structure
    print("Reorganizing files into timestep folders (t0, t1, ...)")
    
    # Pre-create all timestep directories
    for t in range(total_sync_frames):
        os.makedirs(os.path.join(frames_dir, f"t{t}"), exist_ok=True)

    for cam_idx, cam_dir in cam_temp_dirs.items():
        # FFmpeg is 1-indexed (00001.png), we need 0-indexed timesteps (t0)
        extracted_frames = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
        
        for t, frame_path in enumerate(extracted_frames):
            if t >= total_sync_frames:
                break
                
            target_path = os.path.join(frames_dir, f"t{t}", f"cam_{cam_idx:02d}.png")
            shutil.move(frame_path, target_path)

    # Cleanup Temp Directory
    shutil.rmtree(temp_dir)
    print(f"\nSuccess! Frames extracted and organized in {frames_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parallel extraction of synced frames.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()
    
    extract_frames_fast(args.dataset, args.max_frames)