# scripts/preprocess_flow.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

"""
Usage:
python scripts/preprocess_flow.py --videos_dir data/videos --out_dir data/flows --fps 15 --clip_len 60 --resize 112 112 --flow_type farneback --grid 1
"""

def compute_flow_for_video(video_path, out_dir, fps=15, clip_len=60, resize=(112,112),
                           flow_type='farneback', grid=112):
    """
    Read video, extract frames at desired fps, compute optical flow between consecutive frames.
    Save array shapes: (num_clips, clip_len, H, W, 3) -> channels: [flow_x, flow_y, magnitude]
    If grid < H: will compute average flow per grid cell and upsample (nearest) to HxW for compatibility.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # get original fps
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(round(orig_fps / fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
        idx += 1
    cap.release()
    if len(frames) < 2:
        print(f"Video {video_path} too short after resampling.")
        return

    H, W = resize
    # compute dense optical flow between consecutive frames
    flows = []
    for i in range(len(frames) - 1):
        prev = frames[i]
        nxt = frames[i+1]
        if flow_type == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev, nxt,
                                                None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            # flow shape (H, W, 2)
        else:
            raise NotImplementedError("Only farneback implemented in this script.")
        mag = np.linalg.norm(flow, axis=2, keepdims=True)
        flow3 = np.concatenate([flow, mag], axis=2)  # (H, W, 3)
        flows.append(flow3.astype(np.float32))
    flows = np.stack(flows, axis=0)  # (T-1, H, W, 3)

    # segment into clips of clip_len frames (we need clip_len frames -> clip_len-1 flows)
    clip_frames = clip_len
    flows_per_clip = clip_frames - 1
    clips = []
    for start in range(0, flows.shape[0] - flows_per_clip + 1, flows_per_clip):
        clip = flows[start:start + flows_per_clip]  # shape (flows_per_clip, H, W, 3)
        clips.append(clip)
    clips = np.array(clips)  # (num_clips, flows_per_clip, H, W, 3)

    # If grid < H -> downsample to grid x grid then upsample (nearest) for compatibility
    if grid != H:
        out_clips = []
        for c in clips:
            # c shape (flows_per_clip, H, W, 3)
            small = cv2.resize(c.reshape(-1, H, W, 3).transpose(0,3,1,2).reshape(-1,3,H,W)[0].transpose(1,2,0), (grid,grid)) # simpler: do per-frame loop
            # do robust per frame:
            smalls = []
            for frame_flow in c:
                # frame_flow (H,W,3) -> average per grid cell
                cell_H = H // grid
                cell_W = W // grid
                small_flow = np.zeros((grid, grid, 3), dtype=np.float32)
                for i in range(grid):
                    for j in range(grid):
                        hs = i * cell_H
                        ws = j * cell_W
                        he = hs + cell_H
                        we = ws + cell_W
                        patch = frame_flow[hs:he, ws:we, :]
                        small_flow[i,j,:] = patch.reshape(-1,3).mean(axis=0)
                # upsample to H,W using nearest
                up = cv2.resize(small_flow, (W, H), interpolation=cv2.INTER_NEAREST)
                smalls.append(up)
            out_clips.append(np.stack(smalls, axis=0))
        clips = np.array(out_clips)

    # Save npy per clip
    for idx, clip in enumerate(clips):
        fname = f"{video_name}_clip{idx:03d}.npy"
        path = os.path.join(out_dir, fname)
        # Save shape: (clip_len-1, H, W, 3)
        np.save(path, clip)
    print(f"Saved {len(clips)} flow clips for {video_name} to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--clip_len', type=int, default=60)
    parser.add_argument('--resize', nargs=2, type=int, default=[112,112])
    parser.add_argument('--flow_type', default='farneback')
    parser.add_argument('--grid', type=int, default=112, help='grid resolution for sparse flow (1 for global average)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for root, _, files in os.walk(args.videos_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, f)
                subject = os.path.basename(root)  # expects /videos/<subject>/<video>.mp4
                out_sub = os.path.join(args.out_dir, subject)
                os.makedirs(out_sub, exist_ok=True)
                compute_flow_for_video(video_path, out_sub,
                                       fps=args.fps, clip_len=args.clip_len,
                                       resize=tuple(args.resize),
                                       flow_type=args.flow_type, grid=args.grid)

if __name__ == '__main__':
    main()
