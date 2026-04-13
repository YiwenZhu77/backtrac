#!/usr/bin/env python3
"""Generate backward trace video with dBz background and bubble classification."""

import argparse
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor

from backtrac.config import load_config
from backtrac.classify import load_gamera_dbz
from backtrac.viz import render_frame, make_video


def main():
    parser = argparse.ArgumentParser(description='BackTrac: generate video')
    parser.add_argument('config', help='YAML config file')
    parser.add_argument('trajectories', help='Trajectory .npy file')
    parser.add_argument('classification', help='Classification .npz file (from analyze.py)')
    parser.add_argument('-o', '--output', default='video.mp4', help='Output mp4 path')
    parser.add_argument('--fps', type=int, default=12, help='Frames per second')
    parser.add_argument('--cores', type=int, default=16, help='Render cores')
    args = parser.parse_args()

    cfg = load_config(args.config)
    traj = np.load(args.trajectories)
    cls = np.load(args.classification)
    bubble_mask = cls['bubble']
    n_steps = traj.shape[0]

    frame_dir = args.output.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    # Pre-load all dBz slices
    print('Loading GAMERA dBz...')
    t0 = time.time()
    dbz_cache = {}
    for t in range(n_steps):
        gam_step = cfg.start_chunk + 360 - t
        if gam_step < 360:
            break
        try:
            _, dbz, xxi, yyi = load_gamera_dbz(cfg.gamera_dir, gam_step)
            dbz_cache[t] = (dbz, xxi, yyi)
        except Exception:
            pass
    print(f'  Loaded {len(dbz_cache)} slices in {time.time()-t0:.0f}s')

    # Render frames
    def _render(frame):
        if frame not in dbz_cache:
            return
        render_frame(frame, traj, bubble_mask, dbz_cache[frame],
                     frame_dir, cfg.bubble.dbz_threshold)

    frames = list(range(len(dbz_cache)))
    print(f'Rendering {len(frames)} frames on {args.cores} cores...')
    with ProcessPoolExecutor(max_workers=args.cores) as pool:
        list(pool.map(_render, frames))

    # Encode video
    print('Encoding video...')
    make_video(frame_dir, args.output, args.fps)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
