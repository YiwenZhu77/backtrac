#!/usr/bin/env python3
"""Convert MAGE msphere.rcm.h5 (+ msphere.mhdrcm.h5) to BackTrac flat HDF5 format."""

import argparse
import h5py
import numpy as np
import time
import os


def main():
    parser = argparse.ArgumentParser(description='Convert MAGE RCM output to BackTrac format')
    parser.add_argument('rcm_h5', help='Path to msphere.rcm.h5')
    parser.add_argument('--mhdrcm', help='Path to msphere.mhdrcm.h5 (for bVol). '
                        'If not given, looks in same directory as rcm_h5.')
    parser.add_argument('--steps', default='360-600',
                        help='Step range, e.g. 360-600 (default: 360-600)')
    parser.add_argument('-o', '--output', required=True, help='Output HDF5 path')
    args = parser.parse_args()

    step_start, step_end = map(int, args.steps.split('-'))
    steps = list(range(step_start, step_end + 1))
    n_chunks = len(steps)

    # Find mhdrcm file
    if args.mhdrcm:
        mhdrcm_path = args.mhdrcm
    else:
        mhdrcm_path = os.path.join(os.path.dirname(args.rcm_h5), 'msphere.mhdrcm.h5')

    has_mhdrcm = os.path.exists(mhdrcm_path)

    print(f'Converting {args.rcm_h5}')
    print(f'  Steps {step_start}-{step_end} ({n_chunks} chunks)')
    if has_mhdrcm:
        print(f'  mhdrcm: {mhdrcm_path}')
    else:
        print(f'  mhdrcm not found — PVG will use VM only (no bVol)')

    t0 = time.time()
    f_rcm = h5py.File(args.rcm_h5, 'r')
    f_mhd = h5py.File(mhdrcm_path, 'r') if has_mhdrcm else None

    s0 = f_rcm[f'Step#{step_start}']
    j_max, i_max = s0['rcmxmin'].shape
    chunk_size = j_max * i_max
    total_size = n_chunks * chunk_size

    # Get alamc (energy channels) — same for all timesteps
    alamc = s0['alamc'][:]
    kcsize = len(alamc)

    print(f'  Grid: {j_max} x {i_max}, {kcsize} energy channels')

    # Static arrays
    I_1d = np.tile(np.arange(1, i_max + 1), j_max)
    J_1d = np.repeat(np.arange(1, j_max + 1), i_max)
    I_all = np.tile(I_1d, n_chunks)
    J_all = np.tile(J_1d, n_chunks)

    # Time-varying fields
    fields = {k: np.zeros(total_size) for k in
              ['XMIN', 'YMIN', 'V', 'VM', 'COLAT', 'ALOCT', 'BMIN', 'PVG']}

    rcm_to_field = {
        'XMIN': 'rcmxmin', 'YMIN': 'rcmymin', 'V': 'rcmv',
        'VM': 'rcmvm', 'COLAT': 'colat', 'ALOCT': 'aloct',
        'BMIN': 'rcmbmin',
    }

    # Constants for pressure computation
    RE_m = 6.37e6
    ev = 1.6022e-19
    nt = 1e-9
    pressure_factor = (2.0 / 3.0) * ev / RE_m * nt

    for ci, step in enumerate(steps):
        s = f_rcm[f'Step#{step}']
        idx = ci * chunk_size

        # Standard fields
        for out_key, rcm_key in rcm_to_field.items():
            fields[out_key][idx:idx + chunk_size] = s[rcm_key][:].flatten()

        # Compute PVG = P * bVol^(5/3)
        vm = s['rcmvm'][:]
        eeta = s['rcmeeta'][:]  # (kcsize, j_max, i_max)

        # Total pressure: P = sum_k pf * |alamc_k| * eeta_k * vm^2.5
        vm_safe = np.where(vm > 0, vm, 0)
        P = np.zeros((j_max, i_max))
        for k in range(kcsize):
            P += pressure_factor * abs(alamc[k]) * eeta[k] * vm_safe ** 2.5

        # bVol from mhdrcm (359x180 → pad to 361x180)
        if f_mhd is not None and f'Step#{step}' in f_mhd:
            bvol = f_mhd[f'Step#{step}']['bVol'][:]
            bvol_pad = np.zeros((j_max, i_max))
            bvol_pad[1:bvol.shape[0]+1, :] = bvol
            bvol_pad[0, :] = bvol[0, :]
            bvol_pad[-1, :] = bvol[-1, :]
            bvol_safe = np.where(bvol_pad > 0, bvol_pad, 0)
            PVG = P * bvol_safe ** (5.0 / 3.0)
        else:
            # Fallback: PVG ~ P / vm^2.5 * (1/vm)^(5/3) ... approximate
            PVG = P  # just store pressure if no bVol

        fields['PVG'][idx:idx + chunk_size] = PVG.flatten()

        if ci % 50 == 0:
            print(f'  Step {step} ({ci}/{n_chunks})')

    f_rcm.close()
    if f_mhd is not None:
        f_mhd.close()

    # Write output
    print(f'Writing {args.output}...')
    f_out = h5py.File(args.output, 'w')
    f_out.create_dataset('I', data=I_all)
    f_out.create_dataset('J', data=J_all)
    for k, v in fields.items():
        f_out.create_dataset(k, data=v)

    # Also store alamc for energy channel reference
    f_out.create_dataset('alamc', data=alamc)

    f_out.attrs['source'] = args.rcm_h5
    f_out.attrs['step_range'] = args.steps
    f_out.attrs['i_max'] = i_max
    f_out.attrs['j_max'] = j_max
    f_out.attrs['n_chunks'] = n_chunks
    f_out.attrs['kcsize'] = kcsize
    f_out.attrs['colat_unit'] = 'radians'
    f_out.close()

    print(f'Done in {time.time() - t0:.1f}s')
    print(f'  {args.output}: {n_chunks} chunks, {j_max}x{i_max} grid, {kcsize} channels')


if __name__ == '__main__':
    main()
