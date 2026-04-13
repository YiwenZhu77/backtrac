#!/usr/bin/env python3
"""Generate 6 videos: FO and Sina, each with bubble/pre-existing/other shown individually."""

import sys, os
sys.path.insert(0, '/glade/work/yizhu/OpOF/kaipy-private')
os.chdir('/glade/derecho/scratch/sbao/sp13_075')

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import median_filter
import kaipy.gamera.magsphere as msph

OUTDIR = '/glade/work/yizhu/backtrac/pilot/class_videos'
os.makedirs(OUTDIR, exist_ok=True)
NCPU = 16
DBZ_THRESH = 15.0
RC_R = 8.0

# ============ Load GAMERA dBz ============
print('Loading GAMERA...')
gam = msph.GamsphPipe('/glade/derecho/scratch/sbao/sp13_075', 'msphere', doFast=True)
print('Pre-reading dBz...')
dBz_cache = {}
for t in range(241):
    gs = 600 - t
    if gs < 360: break
    dBz_cache[t] = (gam.DelBz(gs), gam.xxi.copy(), gam.yyi.copy())
print(f'  {len(dBz_cache)} slices')

# ============ Load trajectories ============
print('Loading FO...')
f = h5py.File('/glade/derecho/scratch/yizhu/backtrac_pilot/sp13_eqproj.000001.h5part', 'r')
steps = sorted([k for k in f.keys() if k.startswith('Step#')], key=lambda s: int(s.split('#')[1]))
Nt = len(steps)
xeq = np.array([f[s]['xeq'][:] for s in steps])
yeq = np.array([f[s]['yeq'][:] for s in steps])
isIn = np.array([f[s]['isIn'][:] for s in steps])
K0 = f[steps[0]]['K'][:]
f.close()
xeq[isIn < 0.5] = np.nan; yeq[isIn < 0.5] = np.nan

print('Loading Sina...')
res_rw = np.load('/glade/derecho/scratch/yizhu/backtrac_pilot/sina_1k_traj.npy')
Nt = min(241, xeq.shape[0], res_rw.shape[0])

# ============ Classify ============
print('Classifying...')

def build_interp(dBz, xxi, yyi):
    xc = 0.5*(xxi[:-1,:]+xxi[1:,:]); xc = 0.5*(xc[:,:-1]+xc[:,1:])
    yc = 0.5*(yyi[:-1,:]+yyi[1:,:]); yc = 0.5*(yc[:,:-1]+yc[:,1:])
    v = np.isfinite(dBz) & (np.sqrt(xc**2+yc**2) > 2)
    return LinearNDInterpolator(np.column_stack([xc[v].ravel(), yc[v].ravel()]), dBz[v].ravel())

def classify_bubble(x_t, y_t):
    Np = x_t.shape[1]
    ever = np.zeros(Np, dtype=bool)
    for t in range(Nt):
        if t not in dBz_cache: continue
        dBz, xi, yi = dBz_cache[t]
        interp = build_interp(dBz, xi, yi)
        xt = x_t[t]; yt = y_t[t]
        v = np.isfinite(xt) & ~ever
        if v.sum() == 0: continue
        dbz_at = interp(xt[v], yt[v])
        r = np.sqrt(xt[v]**2+yt[v]**2)
        phi = np.degrees(np.arctan2(yt[v], xt[v]))
        mlt = (phi/15+12) % 24
        mid = (mlt >= 21) | (mlt <= 3)
        ib = (dbz_at > DBZ_THRESH) & (xt[v] < 0) & (r > 6) & mid
        ever[np.where(v)[0][ib]] = True
    return ever

def smooth_r(r_2d):
    out = np.full_like(r_2d, np.nan)
    for pi in range(r_2d.shape[1]):
        r = r_2d[:, pi]
        v = np.isfinite(r)
        if v.sum() < 5: continue
        rf = r.copy(); rf[~v] = 0
        rm = median_filter(rf, size=5)
        rm[~v] = np.nan
        out[:, pi] = rm
    return out

def classify_pre(r_smooth):
    pre = np.ones(r_smooth.shape[1], dtype=bool)
    for pi in range(r_smooth.shape[1]):
        r = r_smooth[:, pi]
        if not np.isfinite(r[0] if len(r) > 0 else np.nan):
            pre[pi] = False
        elif np.any(np.isfinite(r) & (r > RC_R)):
            pre[pi] = False
    return pre

# FO
bub_fo = classify_bubble(xeq[:Nt], yeq[:Nt])
r_fo_eq = np.sqrt(xeq[:Nt]**2 + yeq[:Nt]**2)
r_fo_sm = smooth_r(r_fo_eq)
pre_fo = classify_pre(r_fo_sm)
other_fo = np.isfinite(xeq[0]) & ~bub_fo & ~pre_fo

# Sina
bub_si = classify_bubble(res_rw[:Nt,:,0], res_rw[:Nt,:,1])
r_si = np.sqrt(res_rw[:Nt,:,0]**2 + res_rw[:Nt,:,1]**2)
r_si[r_si < 1] = np.nan
r_si_sm = smooth_r(r_si)
pre_si = classify_pre(r_si_sm)
other_si = np.isfinite(res_rw[0,:,0]) & ~bub_si & ~pre_si

n_fo = np.isfinite(xeq[0]).sum(); n_si = np.isfinite(res_rw[0,:,0]).sum()
print(f'FO:   bub={bub_fo.sum()} pre={pre_fo.sum()} other={other_fo.sum()}')
print(f'Sina: bub={bub_si.sum()} pre={pre_si.sum()} other={other_si.sum()}')

# ============ Render function ============
def render_one(args):
    frame, method, cat, xs, ys, mask, color, label = args
    if frame not in dBz_cache: return
    dBz, xxi, yyi = dBz_cache[frame]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.pcolormesh(xxi, yyi, dBz, cmap='RdBu_r', vmin=-30, vmax=30, shading='auto', zorder=1)

    xc = 0.5*(xxi[:-1,:]+xxi[1:,:]); xc = 0.5*(xc[:,:-1]+xc[:,1:])
    yc = 0.5*(yyi[:-1,:]+yyi[1:,:]); yc = 0.5*(yc[:,:-1]+yc[:,1:])
    ax.contour(xc, yc, dBz, levels=[DBZ_THRESH], colors='lime', linewidths=1, zorder=2)

    ax.add_patch(plt.Circle((0,0), 1, color='k', fill=True, zorder=10))
    ax.add_patch(plt.Circle((0,0), RC_R, color='orange', fill=False, ls='--', lw=1.5, alpha=0.5, zorder=8))

    v = np.isfinite(xs[frame]) & mask
    # Trails
    t0 = max(0, frame-6)
    for pi in np.where(v)[0][::2]:
        ax.plot(xs[t0:frame+1, pi], ys[t0:frame+1, pi], color=color, alpha=0.2, lw=0.4, zorder=3)
    if v.sum() > 0:
        ax.scatter(xs[frame, v], ys[frame, v], c=color, s=12, alpha=0.7,
                   edgecolors='k', linewidths=0.2, zorder=5)

    ax.set_xlim(-20, 8); ax.set_ylim(-15, 15); ax.set_aspect('equal')
    ax.set_xlabel('X_eq (RE)'); ax.set_ylabel('Y_eq (RE)')
    t_hr = 10.0 - frame/60
    ax.set_title(f'{method} — {label} ({v.sum()})  |  {t_hr:.2f} UT ({frame} min bwd)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    outdir = os.path.join(OUTDIR, f'{method}_{cat}_frames')
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f'frame_{frame:04d}.png'), dpi=80, bbox_inches='tight')
    plt.close(fig)

# ============ Build task list ============
frames = list(range(len(dBz_cache)))
tasks = []

for method, xs, ys, bub, pre, oth in [
    ('FO', xeq, yeq, bub_fo, pre_fo, other_fo),
    ('Sina', res_rw[:,:,0], res_rw[:,:,1], bub_si, pre_si, other_si),
]:
    for cat, mask, color, label in [
        ('bubble', bub, 'red', 'Bubble'),
        ('preexist', pre, 'gray', 'Pre-existing'),
        ('other', oth, 'dodgerblue', 'Other'),
    ]:
        for f in frames:
            tasks.append((f, method, cat, xs, ys, mask, color, label))

print(f'Rendering {len(tasks)} frames ({len(tasks)//len(frames)} videos x {len(frames)} frames) on {NCPU} cores...')
with Pool(NCPU) as pool:
    pool.map(render_one, tasks)

# ============ Encode videos ============
import subprocess
print('Encoding videos...')
for method in ['FO', 'Sina']:
    for cat in ['bubble', 'preexist', 'other']:
        fdir = os.path.join(OUTDIR, f'{method}_{cat}_frames')
        out = os.path.join(OUTDIR, f'{method}_{cat}.mp4')
        subprocess.run([
            'ffmpeg', '-y', '-framerate', '12',
            '-i', os.path.join(fdir, 'frame_%04d.png'),
            '-vcodec', 'libx264', '-crf', '26', '-preset', 'fast',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            '-vf', 'scale=1280:-2', out,
        ], capture_output=True)
        print(f'  {out}')

print('Done.')
