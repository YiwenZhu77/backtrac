#!/usr/bin/env python3
"""Video: FO vs Sina V_eff, bubble = nightside midnight sector, r>6, dBz>15."""

import sys, os
sys.path.insert(0, '/glade/work/yizhu/OpOF/kaipy-private')
os.chdir('/glade/derecho/scratch/sbao/sp13_075')

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import LinearNDInterpolator
import kaipy.gamera.magsphere as msph

FIGDIR = '/glade/work/yizhu/backtrac/pilot/repl_out/figures/fo_veff_dbz_v5_frames'
os.makedirs(FIGDIR, exist_ok=True)
NCPU = 16
DBZ_THRESH = 15.0

print('Loading GAMERA...')
gam = msph.GamsphPipe('/glade/derecho/scratch/sbao/sp13_075', 'msphere', doFast=True)

print('Pre-reading dBz...')
dBz_cache = {}
for t_bwd in range(241):
    gs = 600 - t_bwd
    if gs < 360: break
    dBz_cache[t_bwd] = (gam.DelBz(gs), gam.xxi.copy(), gam.yyi.copy())

print('Loading trajectories...')
f = h5py.File('/glade/derecho/scratch/yizhu/backtrac_pilot/sp13_eqproj.000001.h5part', 'r')
steps = sorted([k for k in f.keys() if k.startswith('Step#')], key=lambda s: int(s.split('#')[1]))
xeq = np.array([f[s]['xeq'][:] for s in steps])
yeq = np.array([f[s]['yeq'][:] for s in steps])
isIn = np.array([f[s]['isIn'][:] for s in steps])
f.close()
xeq[isIn < 0.5] = np.nan; yeq[isIn < 0.5] = np.nan

res_rw = np.load('/glade/derecho/scratch/yizhu/backtrac_pilot/sina_1k_traj.npy')
Nt = min(241, xeq.shape[0], res_rw.shape[0])

def build_dbz_interp(dBz, xxi, yyi):
    xc = 0.5*(xxi[:-1,:]+xxi[1:,:]); xc = 0.5*(xc[:,:-1]+xc[:,1:])
    yc = 0.5*(yyi[:-1,:]+yyi[1:,:]); yc = 0.5*(yc[:,:-1]+yc[:,1:])
    valid = np.isfinite(dBz) & (np.sqrt(xc**2+yc**2) > 2)
    return LinearNDInterpolator(
        np.column_stack([xc[valid].ravel(), yc[valid].ravel()]),
        dBz[valid].ravel()
    )

def is_bubble_region(x, y, dbz_val):
    """Bubble criteria: midnight sector, r>6, nightside, dBz > threshold"""
    r = np.sqrt(x**2 + y**2)
    phi = np.degrees(np.arctan2(y, x))
    mlt = (phi / 15.0 + 12) % 24
    # Midnight sector: MLT 21-03 (±3h around midnight)
    midnight = (mlt >= 21) | (mlt <= 3)
    return (dbz_val > DBZ_THRESH) & (x < 0) & (r > 6) & midnight

def classify_dbz(x_traj, y_traj, label):
    Np = x_traj.shape[1]
    ever_in_bubble = np.zeros(Np, dtype=bool)
    for t in range(Nt):
        if t not in dBz_cache: continue
        dBz, xxi_t, yyi_t = dBz_cache[t]
        interp = build_dbz_interp(dBz, xxi_t, yyi_t)
        xt = x_traj[t]; yt = y_traj[t]
        v = np.isfinite(xt)
        if v.sum() == 0: continue
        dbz_at = interp(xt[v], yt[v])
        in_bub = is_bubble_region(xt[v], yt[v], dbz_at)
        idx_v = np.where(v)[0]
        ever_in_bubble[idx_v[in_bub]] = True
    pre = ~ever_in_bubble & np.isfinite(x_traj[0])
    n = np.sum(np.isfinite(x_traj[0]))
    print(f'{label} (N={n}): bubble={ever_in_bubble.sum()} ({ever_in_bubble.sum()/n*100:.1f}%), non-bubble={pre.sum()} ({pre.sum()/n*100:.1f}%)')
    return ever_in_bubble, pre

print('Classifying...')
bub_fo, nobub_fo = classify_dbz(xeq[:Nt], yeq[:Nt], 'FO')
bub_rw, nobub_rw = classify_dbz(res_rw[:Nt,:,0], res_rw[:Nt,:,1], 'Sina V_eff')

def render_frame(frame):
    if frame not in dBz_cache: return
    dBz, xxi_f, yyi_f = dBz_cache[frame]
    xc = 0.5*(xxi_f[:-1,:]+xxi_f[1:,:]); xc = 0.5*(xc[:,:-1]+xc[:,1:])
    yc = 0.5*(yyi_f[:-1,:]+yyi_f[1:,:]); yc = 0.5*(yc[:,:-1]+yc[:,1:])
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for col, (label, xs, ys, bub, nobub) in enumerate([
        ('FO (eq-mapped)', xeq, yeq, bub_fo, nobub_fo),
        ('Sina V_eff', res_rw[:,:,0], res_rw[:,:,1], bub_rw, nobub_rw),
    ]):
        ax = axes[col]
        ax.pcolormesh(xxi_f, yyi_f, dBz, cmap='RdBu_r', vmin=-30, vmax=30, shading='auto', zorder=1)
        ax.contour(xc, yc, dBz, levels=[DBZ_THRESH], colors='lime', linewidths=1.5, zorder=2)
        ax.add_patch(plt.Circle((0,0), 1, color='k', fill=True, zorder=10))
        v = np.isfinite(xs[frame])
        m_nb = v & nobub; m_b = v & bub
        if m_nb.sum() > 0:
            ax.scatter(xs[frame,m_nb], ys[frame,m_nb], c='dodgerblue', s=8, alpha=0.5, edgecolors='none', zorder=3)
        if m_b.sum() > 0:
            ax.scatter(xs[frame,m_b], ys[frame,m_b], c='red', s=14, alpha=0.8, edgecolors='darkred', linewidths=0.3, zorder=5)
        ax.set_xlim(-20, 8); ax.set_ylim(-15, 15); ax.set_aspect('equal')
        ax.set_xlabel('X_eq (RE)'); ax.set_ylabel('Y_eq (RE)')
        ax.set_title(f'{label}  N={v.sum()}  [bub:{m_b.sum()} other:{m_nb.sum()}]', fontsize=11)
    t_hr = 10.0 - frame/60
    fig.suptitle(f'{t_hr:.2f} UT ({frame} min bwd)  |  bubble: midnight(21-03 MLT), x<0, r>6, ΔBz>{DBZ_THRESH:.0f}nT',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/frame_{frame:04d}.png', dpi=90, bbox_inches='tight')
    plt.close(fig)

frames = list(range(len(dBz_cache)))
print(f'Rendering {len(frames)} frames on {NCPU} cores...')
with ProcessPoolExecutor(max_workers=NCPU) as pool:
    futures = [pool.submit(render_frame, f) for f in frames]
    done = 0
    for fut in futures:
        fut.result(); done += 1
        if done % 30 == 0: print(f'  {done}/{len(frames)}')
print('Done.')
