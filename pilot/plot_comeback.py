#!/usr/bin/env python3
"""Video showing particles that leave RC then come back. Multithreaded."""

import sys, os
sys.path.insert(0, '/glade/work/yizhu/OpOF/kaipy-private')
os.chdir('/glade/derecho/scratch/sbao/sp13_075')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import kaipy.gamera.magsphere as msph

FIGDIR = '/glade/work/yizhu/backtrac/pilot/comeback_frames'
os.makedirs(FIGDIR, exist_ok=True)
RC_R = 6.6
NCPU = 16

# Load
print('Loading GAMERA...')
gam = msph.GamsphPipe('/glade/derecho/scratch/sbao/sp13_075', 'msphere', doFast=True)

print('Pre-reading dBz...')
dBz_cache = {}
for t in range(241):
    gs = 600 - t
    if gs < 360: break
    dBz_cache[t] = (gam.DelBz(gs), gam.xxi.copy(), gam.yyi.copy())

print('Loading trajectories...')
res = np.load('/glade/derecho/scratch/yizhu/backtrac_pilot/sina_1k_traj.npy')
Nt = min(241, res.shape[0])
Np = res.shape[1]

# Classify
r_traj = np.sqrt(res[:Nt,:,0]**2 + res[:Nt,:,1]**2)
alive = np.isfinite(res[0,:,0])

never_left = np.ones(Np, dtype=bool)
ever_left = np.zeros(Np, dtype=bool)
came_back = np.zeros(Np, dtype=bool)

for pi in range(Np):
    if not alive[pi]: continue
    r = r_traj[:, pi]
    v = np.isfinite(r)
    if v.sum() < 2: continue
    rv = r[v]
    left = False
    for t in range(len(rv)):
        if rv[t] > RC_R:
            left = True
            ever_left[pi] = True
            never_left[pi] = False
        elif left and rv[t] < RC_R:
            came_back[pi] = True

left_stayed = ever_left & ~came_back
print(f'Never left: {(never_left&alive).sum()}, Left+came back: {came_back.sum()}, Left+stayed out: {left_stayed.sum()}')

# Render
def render(frame):
    if frame not in dBz_cache: return
    dBz, xxi, yyi = dBz_cache[frame]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.pcolormesh(xxi, yyi, dBz, cmap='RdBu_r', vmin=-30, vmax=30, shading='auto', zorder=1)
    ax.add_patch(plt.Circle((0,0), 1, color='k', fill=True, zorder=10))
    ax.add_patch(plt.Circle((0,0), RC_R, color='orange', fill=False, ls='--', lw=2, alpha=0.7, zorder=8))

    v = np.isfinite(res[frame,:,0])

    # Never left (gray)
    m = v & never_left
    if m.sum() > 0:
        ax.scatter(res[frame,m,0], res[frame,m,1], c='gray', s=5, alpha=0.3, edgecolors='none', zorder=3)

    # Left and stayed out (blue)
    m = v & left_stayed
    if m.sum() > 0:
        ax.scatter(res[frame,m,0], res[frame,m,1], c='dodgerblue', s=10, alpha=0.6, edgecolors='navy', linewidths=0.2, zorder=4)

    # Came back (red)
    m = v & came_back
    if m.sum() > 0:
        ax.scatter(res[frame,m,0], res[frame,m,1], c='red', s=12, alpha=0.7, edgecolors='darkred', linewidths=0.3, zorder=5)

    ax.set_xlim(-20, 8); ax.set_ylim(-15, 15); ax.set_aspect('equal')
    ax.set_xlabel('X_eq (RE)'); ax.set_ylabel('Y_eq (RE)')

    n_nl = (v & never_left).sum()
    n_cb = (v & came_back).sum()
    n_ls = (v & left_stayed).sum()
    t_hr = 10.0 - frame/60
    ax.set_title(f'{t_hr:.2f} UT ({frame} min bwd)  |  '
                 f'gray=never left({n_nl})  blue=left({n_ls})  red=came back({n_cb})  '
                 f'orange={RC_R} RE', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/frame_{frame:04d}.png', dpi=90, bbox_inches='tight')
    plt.close(fig)

frames = list(range(len(dBz_cache)))
print(f'Rendering {len(frames)} frames on {NCPU} cores...')
with ProcessPoolExecutor(max_workers=NCPU) as pool:
    list(pool.map(render, frames))
print('Done.')
