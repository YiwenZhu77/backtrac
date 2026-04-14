#!/usr/bin/env python3
"""Video: FO vs Sina with bndloc boundary only (no buffer line)."""

import sys, os
sys.path.insert(0, '/glade/work/yizhu/OpOF/kaipy-private')
os.chdir('/glade/derecho/scratch/sbao/sp13_075')

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy.ndimage import median_filter
from scipy.interpolate import LinearNDInterpolator
import kaipy.gamera.magsphere as msph

FIGDIR = '/glade/work/yizhu/backtrac/pilot/rcmbound_frames'
os.makedirs(FIGDIR, exist_ok=True)
NCPU = 16
DBZ_THRESH = 15.0
RC_R = 8.0

print('Loading GAMERA...')
gam = msph.GamsphPipe('/glade/derecho/scratch/sbao/sp13_075', 'msphere', doFast=True)
print('Pre-reading dBz...')
dBz_cache = {}
for t in range(241):
    gs = 600 - t
    if gs < 360: break
    dBz_cache[t] = (gam.DelBz(gs), gam.xxi.copy(), gam.yyi.copy())

print('Loading bndloc boundaries...')
f_rcm = h5py.File('/glade/derecho/scratch/sbao/sp13_075/msphere.rcm.h5', 'r')
bnd_xy = {}
for step in range(360, 601):
    s = f_rcm[f'Step#{step}']
    bl = s['rcmbndloc'][:]
    xm = s['rcmxmin'][:]; ym = s['rcmymin'][:]
    bx=[]; by=[]
    for j in range(len(bl)):
        ib = int(bl[j])
        if 0 < ib < xm.shape[1]:
            bx.append(xm[j,ib]); by.append(ym[j,ib])
    bnd_xy[step-360] = (np.array(bx), np.array(by))
f_rcm.close()

print('Loading trajectories...')
f = h5py.File('/glade/derecho/scratch/yizhu/backtrac_pilot/sp13_eqproj.000001.h5part', 'r')
steps = sorted([k for k in f.keys() if k.startswith('Step#')], key=lambda s: int(s.split('#')[1]))
xeq = np.array([f[s]['xeq'][:] for s in steps])
yeq = np.array([f[s]['yeq'][:] for s in steps])
isIn = np.array([f[s]['isIn'][:] for s in steps])
K0 = f[steps[0]]['K'][:]
f.close()
xeq[isIn<0.5] = np.nan; yeq[isIn<0.5] = np.nan

res_rw = np.load('/glade/derecho/scratch/yizhu/backtrac_pilot/sina_1k_traj.npy')
Nt = min(241, xeq.shape[0], res_rw.shape[0])

# Classify (same as before)
print('Classifying...')

def build_interp(dBz, xxi, yyi):
    xc=0.5*(xxi[:-1,:]+xxi[1:,:]); xc=0.5*(xc[:,:-1]+xc[:,1:])
    yc=0.5*(yyi[:-1,:]+yyi[1:,:]); yc=0.5*(yc[:,:-1]+yc[:,1:])
    v=np.isfinite(dBz)&(np.sqrt(xc**2+yc**2)>2)
    return LinearNDInterpolator(np.column_stack([xc[v].ravel(),yc[v].ravel()]),dBz[v].ravel())

def classify_bubble(xt, yt):
    Np=xt.shape[1]; ever=np.zeros(Np,dtype=bool)
    for t in range(Nt):
        if t not in dBz_cache: continue
        dBz,xi,yi=dBz_cache[t]; interp=build_interp(dBz,xi,yi)
        x=xt[t];y=yt[t]; v=np.isfinite(x)&~ever
        if v.sum()==0: continue
        d=interp(x[v],y[v]); r=np.sqrt(x[v]**2+y[v]**2)
        phi=np.degrees(np.arctan2(y[v],x[v])); mlt=(phi/15+12)%24
        mid=(mlt>=21)|(mlt<=3)
        ib=(d>DBZ_THRESH)&(x[v]<0)&(r>6)&mid
        ever[np.where(v)[0][ib]]=True
    return ever

def smooth_r(r2d):
    out=np.full_like(r2d,np.nan)
    for pi in range(r2d.shape[1]):
        r=r2d[:,pi]; v=np.isfinite(r)
        if v.sum()<5: continue
        rf=r.copy(); rf[~v]=0; rm=median_filter(rf,size=5); rm[~v]=np.nan; out[:,pi]=rm
    return out

bub_fo=classify_bubble(xeq[:Nt],yeq[:Nt])
bub_si=classify_bubble(res_rw[:Nt,:,0],res_rw[:Nt,:,1])

r_fo_sm=smooth_r(np.sqrt(xeq[:Nt]**2+yeq[:Nt]**2))
r_si_raw=np.sqrt(res_rw[:Nt,:,0]**2+res_rw[:Nt,:,1]**2); r_si_raw[r_si_raw<1]=np.nan
r_si_sm=smooth_r(r_si_raw)

pre_fo=np.array([np.isfinite(xeq[0,pi]) and not np.any(np.isfinite(r_fo_sm[:,pi])&(r_fo_sm[:,pi]>RC_R)) for pi in range(1000)])
pre_si=np.array([np.isfinite(res_rw[0,pi,0]) and not np.any(np.isfinite(r_si_sm[:,pi])&(r_si_sm[:,pi]>RC_R)) for pi in range(1000)])
other_fo=np.isfinite(xeq[0])&~bub_fo&~pre_fo
other_si=np.isfinite(res_rw[0,:,0])&~bub_si&~pre_si

print(f'FO: bub={bub_fo.sum()} pre={pre_fo.sum()} other={other_fo.sum()}')
print(f'Si: bub={bub_si.sum()} pre={pre_si.sum()} other={other_si.sum()}')

def render(frame):
    if frame not in dBz_cache: return
    dBz,xxi,yyi=dBz_cache[frame]
    bx,by=bnd_xy[frame]
    xc=0.5*(xxi[:-1,:]+xxi[1:,:]); xc=0.5*(xc[:,:-1]+xc[:,1:])
    yc=0.5*(yyi[:-1,:]+yyi[1:,:]); yc=0.5*(yc[:,:-1]+yc[:,1:])

    fig,axes=plt.subplots(1,2,figsize=(18,8))
    for col,(label,xs,ys,bub,pre,oth) in enumerate([
        ('FO',xeq,yeq,bub_fo,pre_fo,other_fo),
        ("Sina's",res_rw[:,:,0],res_rw[:,:,1],bub_si,pre_si,other_si),
    ]):
        ax=axes[col]
        ax.pcolormesh(xxi,yyi,dBz,cmap='RdBu_r',vmin=-30,vmax=30,shading='auto',zorder=1)
        ax.contour(xc,yc,dBz,levels=[DBZ_THRESH],colors='lime',linewidths=1,zorder=2)
        ax.plot(bx,by,'m-',lw=2,alpha=0.8,zorder=8)
        ax.add_patch(plt.Circle((0,0),1,color='k',fill=True,zorder=10))
        ax.add_patch(plt.Circle((0,0),RC_R,color='orange',fill=False,ls='--',lw=1.5,alpha=0.4,zorder=7))

        v=np.isfinite(xs[frame])
        for mask,c,s,a in [(v&pre,'gray',5,0.3),(v&oth,'dodgerblue',8,0.5),(v&bub,'red',12,0.7)]:
            if mask.sum()>0:
                ax.scatter(xs[frame,mask],ys[frame,mask],c=c,s=s,alpha=a,edgecolors='none',zorder=5)

        ax.set_xlim(-20,8); ax.set_ylim(-15,15); ax.set_aspect('equal')
        ax.set_xlabel('X_eq (RE)'); ax.set_ylabel('Y_eq (RE)')
        nb=(v&bub).sum(); np_=(v&pre).sum(); no=(v&oth).sum()
        ax.set_title(f'{label} N={v.sum()} [bub:{nb} pre:{np_} oth:{no}]',fontsize=11)

    t_hr=10.0-frame/60
    fig.suptitle(f'{t_hr:.2f} UT ({frame} min bwd) | magenta=RCM bndloc | green=dBz {DBZ_THRESH:.0f}nT | orange={RC_R} RE',
                 fontsize=11,fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/frame_{frame:04d}.png',dpi=90,bbox_inches='tight')
    plt.close(fig)

frames=list(range(len(dBz_cache)))
print(f'Rendering {len(frames)} frames on {NCPU} cores...')
with Pool(NCPU) as pool:
    pool.map(render, frames)
print('Done.')
