#!/usr/bin/env python3
"""Convert MAGE msphere.rcm.h5 to BackTrac flat HDF5 format."""

import argparse
import h5py
import numpy as np
import time


def main():
    parser = argparse.ArgumentParser(description='Convert MAGE RCM output to BackTrac format')
    parser.add_argument('rcm_h5', help='Path to msphere.rcm.h5')
    parser.add_argument('--steps', default='360-600',
                        help='Step range, e.g. 360-600 (default: 360-600)')
    parser.add_argument('-o', '--output', required=True, help='Output HDF5 path')
    args = parser.parse_args()

    step_start, step_end = map(int, args.steps.split('-'))
    steps = list(range(step_start, step_end + 1))
    n_chunks = len(steps)

    print(f'Converting {args.rcm_h5}')
    print(f'  Steps {step_start}-{step_end} ({n_chunks} chunks)')

    t0 = time.time()
    f_in = h5py.File(args.rcm_h5, 'r')

    s0 = f_in[f'Step#{step_start}']
    j_max, i_max = s0['rcmxmin'].shape
    chunk_size = j_max * i_max
    total_size = n_chunks * chunk_size

    print(f'  Grid: {j_max} x {i_max} = {chunk_size} per chunk')
    print(f'  Total: {total_size} values per field')

    # Static arrays
    I_1d = np.tile(np.arange(1, i_max + 1), j_max)
    J_1d = np.repeat(np.arange(1, j_max + 1), i_max)
    I_all = np.tile(I_1d, n_chunks)
    J_all = np.tile(J_1d, n_chunks)

    # Time-varying fields
    fields = {k: np.zeros(total_size) for k in
              ['XMIN', 'YMIN', 'V', 'VM', 'COLAT', 'ALOCT', 'BMIN']}

    rcm_to_field = {
        'XMIN': 'rcmxmin', 'YMIN': 'rcmymin', 'V': 'rcmv',
        'VM': 'rcmvm', 'COLAT': 'colat', 'ALOCT': 'aloct',
        'BMIN': 'rcmbmin',
    }

    for ci, step in enumerate(steps):
        s = f_in[f'Step#{step}']
        idx = ci * chunk_size
        for out_key, rcm_key in rcm_to_field.items():
            fields[out_key][idx:idx + chunk_size] = s[rcm_key][:].flatten()
        if ci % 50 == 0:
            print(f'  Step {step} ({ci}/{n_chunks})')

    f_in.close()

    # Write output
    print(f'Writing {args.output}...')
    f_out = h5py.File(args.output, 'w')
    f_out.create_dataset('I', data=I_all)
    f_out.create_dataset('J', data=J_all)
    for k, v in fields.items():
        f_out.create_dataset(k, data=v)
    f_out.create_dataset('PVG', data=np.zeros(total_size))

    f_out.attrs['source'] = args.rcm_h5
    f_out.attrs['step_range'] = args.steps
    f_out.attrs['i_max'] = i_max
    f_out.attrs['j_max'] = j_max
    f_out.attrs['n_chunks'] = n_chunks
    f_out.attrs['colat_unit'] = 'radians'
    f_out.close()

    print(f'Done in {time.time() - t0:.1f}s')
    print(f'  {args.output}: {n_chunks} chunks, {j_max}x{i_max} grid')


if __name__ == '__main__':
    main()
