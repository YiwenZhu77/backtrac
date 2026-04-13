#!/usr/bin/env python
"""Multi-threaded backward trace video with ΔBz colorbar and classified TPs"""

import sys, os
sys.path.insert(0, '/glade/work/yizhu/OpOF/kaipy-private')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import h5py
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.use('Agg')

RUN_DIR = '/glade/derecho/scratch/sbao/sp13_075'
TP_FILE = '/glade/derecho/scratch/yizhu/backtrac_pilot/sp13_10k.000001.h5part'
FIGDIR = '/glade/work/yizhu/backtrac/pilot/frames_10k_v4'
NCPU = 8
STRIDE = 4
MaxK = 500.0

os.makedirs(FIGDIR, exist_ok=True)

# ============ Load data in main process ============
import kaipy.gamera.magsphere as msph
import kaipy.kaiViz as kv

print('Loading GAMERA...')
os.chdir(RUN_DIR)
gam = msph.GamsphPipe(RUN_DIR, 'msphere', doFast=True)
gam_uts = gam.UT
base_ut = datetime(2013, 3, 17, 0, 0, 0)

# Load classification
data15 = np.load('/glade/derecho/scratch/yizhu/backtrac_pilot/bubble_map_r15.npz')
bmap15 = data15['bubble_map']; arc_mlt15 = data15['arc_mlt']; step_times15 = data15['step_times']

print('Loading TPs...')
f = h5py.File(TP_FILE, 'r')
steps = sorted([k for k in f.keys() if k.startswith('Step#')], key=lambda s: int(s.split('#')[1]))
valid = [s for s in steps if len(f[s].keys()) > 0]
Np = len(f[valid[0]]['x'][:])
Nt = len(valid)
x_tp = np.array([f[s]['x'][:] for s in valid])
y_tp = np.array([f[s]['y'][:] for s in valid])
K_tp = np.array([f[s]['K'][:] for s in valid])
isIn_tp = np.array([f[s]['isIn'][:] for s in valid])
f.close()

r_eq_tp = np.sqrt(x_tp**2 + y_tp**2)
mlt_tp = (np.degrees(np.arctan2(y_tp, x_tp)) / 15.0 + 12) % 24
t_tp = 36000 - np.arange(Nt) * 60

# Classify at 15 RE
pclass = np.zeros(Np, dtype=int)
for p in range(Np):
    blowup = np.where(K_tp[:, p] > MaxK)[0]
    max_ti = blowup[0] if len(blowup) > 0 else Nt
    crossed = np.where(r_eq_tp[:max_ti, p] > 15.0)[0]
    if len(crossed) == 0:
        pclass[p] = -1 if len(blowup) > 0 else 3
        continue
    ti = crossed[0]
    t_cross = t_tp[ti]; mlt_cross = mlt_tp[ti, p]
    if not ((mlt_cross >= 18) or (mlt_cross <= 6)):
        pclass[p] = 2; continue
    si_bmap = np.argmin(np.abs(step_times15 - t_cross))
    ai_bmap = np.argmin(np.abs(arc_mlt15 - mlt_cross))
    if si_bmap < bmap15.shape[0] and ai_bmap < bmap15.shape[1] and bmap15[si_bmap, ai_bmap]:
        pclass[p] = 1
    else:
        pclass[p] = 2

# Pre-read all needed GAMERA slices into memory
frame_indices = list(range(0, Nt, STRIDE))
Nf = len(frame_indices)

print(f'Pre-reading {Nf} GAMERA dBz slices...')
dBz_cache = {}
for fi, ti in enumerate(frame_indices):
    t_ut = int(t_tp[ti])
    target_dt = base_ut + timedelta(seconds=t_ut)
    gam_si = np.argmin([abs((ut - target_dt).total_seconds()) for ut in gam_uts])
    dBz_cache[fi] = (gam.DelBz(gam_si), gam.xxi.copy(), gam.yyi.copy(), t_ut, gam_si)
    if fi % 10 == 0:
        print(f'  {fi}/{Nf}')

print(f'All data loaded. Rendering {Nf} frames on {NCPU} processes...')

# ============ Render function (uses global data via fork) ============
def render_frame(fi):
    ti = frame_indices[fi]
    dBz, xxi, yyi, t_ut, gam_si = dBz_cache[fi]
    t_ut_h = t_ut / 3600.0

    fig, ax = plt.subplots(figsize=(10, 8))

    pcm = ax.pcolormesh(xxi, yyi, dBz, cmap='RdBu_r',
                        vmin=-30, vmax=30, shading='auto', rasterized=True, alpha=0.5)
    cbar = plt.colorbar(pcm, ax=ax, label='$\Delta B_z$ (nT)', shrink=0.7)

    active = (isIn_tp[ti] > 0.5) & (K_tp[ti] < MaxK)

    m = active & (pclass == 3)
    if m.sum() > 0:
        ax.scatter(x_tp[ti,m], y_tp[ti,m], c='gray', s=3, alpha=0.2, zorder=3)

    m = active & (pclass == 2)
    if m.sum() > 0:
        ax.scatter(x_tp[ti,m], y_tp[ti,m], c='#3498db', s=5, alpha=0.4, zorder=4)

    m = active & (pclass == 1)
    if m.sum() > 0:
        ax.scatter(x_tp[ti,m], y_tp[ti,m], c='red', s=15, alpha=0.9,
                  edgecolors='k', linewidths=0.3, zorder=5)

    # Earth
    circle_e = plt.Circle((0,0), 1, color='white', fill=True, zorder=10)
    ax.add_patch(circle_e)
    circle_e2 = plt.Circle((0,0), 1, color='k', fill=False, lw=1, zorder=10)
    ax.add_patch(circle_e2)

    ax.add_patch(plt.Circle((0,0), 6, color='cyan', fill=False, ls='--', alpha=0.5, lw=1))
    ax.add_patch(plt.Circle((0,0), 15, color='orange', fill=False, ls='--', alpha=0.5, lw=1))

    ax.set_xlim(-25, 10); ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_xlabel('X (R$_E$)', fontsize=13); ax.set_ylabel('Y (R$_E$)', fontsize=13)

    n_b = (active & (pclass==1)).sum()
    n_c = (active & (pclass==2)).sum()
    n_p = (active & (pclass==3)).sum()

    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='k', markeredgewidth=0.5, label=f'Bubble ({n_b})'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#3498db', markersize=8,
               label=f'Convection ({n_c})'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markersize=6,
               label=f'Pre-existing ({n_p})'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=12, framealpha=0.9,
             edgecolor='k', fancybox=False)

    ut_str = f'{int(t_ut_h):02d}:{int((t_ut_h%1)*60):02d}'
    bwd_h = (36000 - t_ut) / 3600
    ax.set_title(f'H$^+$ Backward Trace  |  {ut_str} UT ({bwd_h:.1f}h bwd)  |  '
                f'sp13-075 $\Delta B_z$', fontsize=13)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/frame_{fi:04d}.png', dpi=100)
    plt.close()
    return fi

# ============ Parallel render using fork ============
if __name__ == '__main__':
    mp.set_start_method('fork')  # fork inherits all loaded data
    with ProcessPoolExecutor(max_workers=NCPU) as executor:
        futures = {executor.submit(render_frame, fi): fi for fi in range(Nf)}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 10 == 0:
                print(f'  Rendered {done}/{Nf}')
    print(f'Done. {Nf} frames in {FIGDIR}/')
