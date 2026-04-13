#!/usr/bin/env python3
"""Video: only comeback particles, RC=8 RE. Multiprocessing."""

import sys, os
sys.path.insert(0, '/glade/work/yizhu/OpOF/kaipy-private')
os.chdir('/glade/derecho/scratch/sbao/sp13_075')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
import kaipy.gamera.magsphere as msph

FIGDIR = '/glade/work/yizhu/backtrac/pilot/comeback2_frames'
os.makedirs(FIGDIR, exist_ok=True)
RC_R = 8.0
NCPU = 16

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
came_back = np.zeros(Np, dtype=bool)

for pi in range(Np):
    if not alive[pi]: continue
    r = r_traj[:, pi]
    v = np.isfinite(r)
    if v.sum() < 2: continue
    rv = r[v]
    left = False
    for t in range(len(rv)):
        if rv[t] > RC_R: left = True
        elif left and rv[t] < RC_R: came_back[pi] = True; break

print(f'Came back to r<{RC_R}: {came_back.sum()}/{alive.sum()}')

def render(frame):
    if frame not in dBz_cache: return
    dBz, xxi, yyi = dBz_cache[frame]
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.pcolormesh(xxi, yyi, dBz, cmap='RdBu_r', vmin=-30, vmax=30, shading='auto', zorder=1)
    ax.add_patch(plt.Circle((0,0), 1, color='k', fill=True, zorder=10))
    ax.add_patch(plt.Circle((0,0), RC_R, color='orange', fill=False, ls='--', lw=2, alpha=0.7, zorder=8))

    v = np.isfinite(res[frame,:,0]) & came_back
    if v.sum() > 0:
        # Trail
        t0 = max(0, frame-8)
        for pi in np.where(v)[0]:
            ax.plot(res[t0:frame+1, pi, 0], res[t0:frame+1, pi, 1], 'r-', alpha=0.3, lw=0.5, zorder=4)
        ax.scatter(res[frame,v,0], res[frame,v,1], c='red', s=18, alpha=0.8,
                   edgecolors='darkred', linewidths=0.3, zorder=5)

    ax.set_xlim(-25, 10); ax.set_ylim(-18, 18); ax.set_aspect('equal')
    ax.set_xlabel('X_eq (RE)'); ax.set_ylabel('Y_eq (RE)')
    t_hr = 10.0 - frame/60
    ax.set_title(f'{t_hr:.2f} UT ({frame} min bwd)  |  '
                 f'Red = left r>{RC_R} then came back ({v.sum()} visible)  |  '
                 f'orange = {RC_R} RE', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/frame_{frame:04d}.png', dpi=90, bbox_inches='tight')
    plt.close(fig)

frames = list(range(len(dBz_cache)))
print(f'Rendering {len(frames)} frames on {NCPU} cores (multiprocessing.Pool)...')
with Pool(NCPU) as pool:
    pool.map(render, frames)
print('Done.')
