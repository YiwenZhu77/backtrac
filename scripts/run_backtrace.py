#!/usr/bin/env python3
"""Run backward trace using V_eff drift."""

import argparse
import numpy as np
import time

from backtrac.config import load_config, RunConfig
from backtrac.data import MageRCMData
from backtrac.particles import init_from_h5part, init_uniform
from backtrac.integrator import run_backtrace


def main():
    parser = argparse.ArgumentParser(description='BackTrac: V_eff backward trace')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--h5part', help='CHIMP h5part file for particle init')
    parser.add_argument('--indices', help='Numpy file with particle subset indices')
    parser.add_argument('--uniform', action='store_true', help='Use uniform seeding')
    parser.add_argument('-o', '--output', default='trajectories.npy', help='Output file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f'Config: {cfg.n_particles} particles, {cfg.n_chunks} chunks, {cfg.n_cores} cores')

    print('Loading RCM data...')
    t0 = time.time()
    data = MageRCMData(cfg)
    data.load_all()
    print(f'  Loaded in {time.time() - t0:.1f}s')

    if args.h5part:
        indices = np.load(args.indices) if args.indices else None
        particles, metadata = init_from_h5part(args.h5part, data, cfg, indices)
    elif args.uniform:
        particles, metadata = init_uniform(data, cfg, cfg.n_particles)
    else:
        parser.error('Specify --h5part or --uniform for particle initialization')

    print(f'Initialized {len(particles)} particles')

    t0 = time.time()
    traj = run_backtrace(particles, cfg)
    print(f'Done in {time.time() - t0:.1f}s')

    np.save(args.output, traj)
    np.save(args.output.replace('.npy', '_meta.npy'),
            np.array(metadata, dtype=[('idx', 'i4'), ('K', 'f8')]))
    print(f'Saved {args.output} ({traj.shape})')


if __name__ == '__main__':
    main()
