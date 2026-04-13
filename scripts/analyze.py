#!/usr/bin/env python3
"""Classify backward trace results: bubble vs convection vs pre-existing."""

import argparse
import numpy as np
import time

from backtrac.config import load_config
from backtrac.classify import BubbleClassifier, load_gamera_dbz


def main():
    parser = argparse.ArgumentParser(description='BackTrac: bubble classification')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('trajectories', help='Trajectory .npy file')
    parser.add_argument('--meta', help='Metadata .npy file (for energy breakdown)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    traj = np.load(args.trajectories)
    n_steps, n_particles, _ = traj.shape
    print(f'Trajectories: {n_steps} steps, {n_particles} particles')

    meta = np.load(args.meta) if args.meta else None

    # Load dBz interpolators for each timestep
    print('Loading GAMERA dBz...')
    t0 = time.time()
    dbz_interps = []
    for t in range(n_steps):
        gam_step = cfg.start_chunk + 360 - t  # Map to GAMERA step
        try:
            interp, _, _, _ = load_gamera_dbz(cfg.gamera_dir, gam_step)
            dbz_interps.append(interp)
        except Exception:
            dbz_interps.append(None)
    print(f'  Loaded {sum(1 for x in dbz_interps if x is not None)} dBz slices in {time.time()-t0:.0f}s')

    # Classify
    classifier = BubbleClassifier(cfg.bubble)
    is_bubble = classifier.classify(traj, dbz_interps)

    # Pre-existing: alive at end, r < 8 RE
    t_end = min(n_steps - 1, cfg.start_chunk)
    alive_end = np.isfinite(traj[t_end, :, 0])
    r_end = np.sqrt(traj[t_end, :, 0] ** 2 + traj[t_end, :, 1] ** 2)
    pre_existing = alive_end & (r_end < 8)

    convection = ~is_bubble & ~pre_existing & np.isfinite(traj[0, :, 0])
    n_total = np.sum(np.isfinite(traj[0, :, 0]))

    print(f'\n=== Results (N={n_total}) ===')
    print(f'  Bubble:       {is_bubble.sum():4d} ({is_bubble.sum()/n_total*100:.1f}%)')
    print(f'  Pre-existing: {pre_existing.sum():4d} ({pre_existing.sum()/n_total*100:.1f}%)')
    print(f'  Convection:   {convection.sum():4d} ({convection.sum()/n_total*100:.1f}%)')

    if meta is not None:
        K = meta['K']
        print(f'\nBy energy:')
        for Klo, Khi in [(10, 30), (30, 60), (60, 120), (120, 200)]:
            m = (K >= Klo) & (K < Khi)
            t = m.sum()
            if t == 0:
                continue
            print(f'  {Klo:3d}-{Khi:3d} keV: bubble {(is_bubble&m).sum()/t*100:.0f}%, '
                  f'pre {(pre_existing&m).sum()/t*100:.0f}%, '
                  f'conv {(convection&m).sum()/t*100:.0f}% (N={t})')

    # Save classification
    out = args.trajectories.replace('.npy', '_class.npz')
    np.savez(out, bubble=is_bubble, pre_existing=pre_existing, convection=convection)
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
