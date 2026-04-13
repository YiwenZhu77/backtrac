#!/usr/bin/env python3
"""Sina rewrite with hybrid: V_eff inside RCM, ExB-only (frozen-in) outside RCM."""
import numpy as np, h5py, time
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count

DATA = '/glade/derecho/scratch/yizhu/backtrac_pilot/sp13_075_rcm_mage.h5'
RCM_FILE = '/glade/derecho/scratch/sbao/sp13_075/msphere.rcm.h5'
RE = 6.37e6; RI = RE + 1e5; B0 = 30570e-9; EPS = 1e-10
i_max = 180; j_max = 361; chunk_size = i_max * j_max; j_period = j_max - 3
I_vals = np.arange(1, i_max+1, dtype=np.float64)
J_vals = np.arange(1, j_max+1, dtype=np.float64)

# Load RCM boundary for each chunk
print('Loading RCM boundary...')
f_rcm = h5py.File(RCM_FILE, 'r')
bndloc_all = {}  # chunk -> bndloc array (361,)
for step in range(360, 601):
    bndloc_all[step - 360] = f_rcm[f'Step#{step}']['rcmbndloc'][:]
f_rcm.close()

print('Loading field chunks...')
t0 = time.time()
f_data = h5py.File(DATA, 'r')
all_chunks = {}
for ci in range(241):
    idx = ci * chunk_size; d = {}
    for k in ['XMIN','YMIN','V','VM','COLAT','ALOCT']:
        d[k] = f_data[k][idx:idx+chunk_size].reshape(j_max, i_max)
    d['ALOCT_sin'] = np.sin(d['ALOCT']); d['ALOCT_cos'] = np.cos(d['ALOCT'])
    all_chunks[ci] = d
f_data.close()
print(f'Loaded in {time.time()-t0:.1f}s')

def mk_i(d2d, fill=None):
    return RegularGridInterpolator((J_vals, I_vals), d2d, method='linear', bounds_error=False, fill_value=fill)

def integrate_one(args):
    I_p, J_p, lam, n_steps = args
    I_p, J_p = float(I_p), float(J_p)
    sub_dt = -60.0 / 10
    cd0 = all_chunks[240]
    x0 = float(mk_i(cd0['XMIN'], fill=np.nan)([J_p, I_p])[0])
    y0 = float(mk_i(cd0['YMIN'], fill=np.nan)([J_p, I_p])[0])
    traj = [(x0, y0)]

    for step in range(n_steps):
        chunk = 240 - step
        if chunk < 0: break
        cd = all_chunks[chunk]
        ic = mk_i(cd['COLAT'], fill=None)
        ias = mk_i(cd['ALOCT_sin'], fill=0.0); iac = mk_i(cd['ALOCT_cos'], fill=0.0)
        iv = mk_i(cd['V'], fill=0.0); ivm = mk_i(cd['VM'], fill=0.0)
        bndloc = bndloc_all[chunk]

        stopped = False
        for _ in range(10):
            theta = ic([J_p, I_p])[0]
            if np.isnan(theta): stopped = True; break

            # Check if inside RCM: I >= bndloc[J]
            j_idx = int(round(J_p)) - 1  # 0-based
            j_idx = max(0, min(j_idx, len(bndloc) - 1))
            inside_rcm = I_p >= bndloc[j_idx]

            # Use lambda for V_eff inside RCM, lambda=0 (ExB only) outside
            lam_eff = lam if inside_rcm else 0.0

            d = 0.5
            # dtheta/dI, dphi/dJ
            dth = (ic([J_p, I_p+d])[0] - ic([J_p, I_p-d])[0]) / (2*d)
            Jm=J_p-d; Jp=J_p+d
            if Jm<1: Jm+=j_period
            if Jp>j_max: Jp-=j_period
            sm=ias([Jm,I_p])[0]; cm=iac([Jm,I_p])[0]; sp=ias([Jp,I_p])[0]; cp=iac([Jp,I_p])[0]
            dphi = np.arctan2(sp,cp)-np.arctan2(sm,cm)
            if dphi>np.pi: dphi-=2*np.pi
            elif dphi<-np.pi: dphi+=2*np.pi
            dph = dphi/(2*d)
            dth = max(abs(dth),1e-8)*(1 if dth>=0 else -1)
            dph = max(abs(dph),1e-8)*(1 if dph>=0 else -1)

            # dVeff
            VmI=iv([J_p,I_p-d])[0]+lam_eff*ivm([J_p,I_p-d])[0]
            VpI=iv([J_p,I_p+d])[0]+lam_eff*ivm([J_p,I_p+d])[0]
            dVeI=(VpI-VmI)/(2*d)
            Jm2=J_p-d; Jp2=J_p+d
            if Jm2<1: Jm2+=j_period
            if Jp2>j_max: Jp2-=j_period
            VmJ=iv([Jm2,I_p])[0]+lam_eff*ivm([Jm2,I_p])[0]
            VpJ=iv([Jp2,I_p])[0]+lam_eff*ivm([Jp2,I_p])[0]
            dVeJ=(VpJ-VmJ)/(2*d)

            Br = -2*B0*(RE/RI)**3*np.cos(theta)
            fac = RI**2*abs(Br)*np.sin(theta)*dph*dth
            fac = max(abs(fac),EPS)*(np.sign(fac) if fac!=0 else 1)
            I_p += sub_dt*dVeJ/fac; J_p += sub_dt*(-dVeI/fac)
            if J_p<1: J_p+=j_period
            if J_p>j_max: J_p-=j_period
            if I_p<1.5 or I_p>i_max-0.5: stopped=True; break

        if stopped: traj.append((np.nan, np.nan)); break
        ix_s=mk_i(cd['XMIN'],fill=np.nan); iy_s=mk_i(cd['YMIN'],fill=np.nan)
        traj.append((float(ix_s([J_p,I_p])[0]), float(iy_s([J_p,I_p])[0])))
    return traj

# Prepare particles
idx_1k = np.load('/glade/derecho/scratch/yizhu/backtrac_pilot/sp13_1k_indices.npy')
f_fo = h5py.File('/glade/derecho/scratch/yizhu/backtrac_pilot/sp13_10k.000001.h5part', 'r')
K_all=f_fo['Step#0']['K'][:]; x_all=f_fo['Step#0']['x'][:]; y_all=f_fo['Step#0']['y'][:]
f_fo.close()
cd_init=all_chunks[240]; xm=cd_init['XMIN']; ym=cd_init['YMIN']; vm=cd_init['VM']
vld=(vm>-99999)&(np.sqrt(xm**2+ym**2)>2); jj,ii=np.where(vld)
tr=cKDTree(np.column_stack([xm[vld],ym[vld]])); ivm_init=mk_i(vm,fill=np.nan)
tasks=[]; pmap=[]
for pi in range(1000):
    pid=idx_1k[pi]; d,idx=tr.query([x_all[pid],y_all[pid]])
    if d>0.5: continue
    I_s=float(ii[idx]+1); J_s=float(jj[idx]+1)
    vm_val=float(ivm_init([J_s,I_s])[0])
    if np.isnan(vm_val) or vm_val<0.1: continue
    tasks.append((I_s,J_s,1000.0*K_all[pid]/vm_val,240)); pmap.append((pi,int(pid),float(K_all[pid])))
print(f'{len(tasks)} particles')

n_cores = min(32, cpu_count())
print(f'Running hybrid Sina on {n_cores} cores...')
t_start = time.time()
with Pool(n_cores) as pool:
    results = pool.map(integrate_one, tasks)
print(f'Done in {time.time()-t_start:.1f}s')

res = np.full((241, 1000, 2), np.nan)
for ri, traj in enumerate(results):
    pi = pmap[ri][0]
    for ti, (x, y) in enumerate(traj):
        if ti > 240: break
        res[ti, pi] = [x, y]
np.save('/glade/derecho/scratch/yizhu/backtrac_pilot/sina_hybrid_1k_traj.npy', res)

# Quick stats
res_rw = np.load('/glade/derecho/scratch/yizhu/backtrac_pilot/sina_1k_traj.npy')
print(f'\nSurvival comparison:')
print(f'{"t":>5} {"Hybrid":>7} {"V_eff":>7}')
for t in [30, 60, 120, 240]:
    nh = np.sum(np.isfinite(res[t,:,0]))
    nv = np.sum(np.isfinite(res_rw[t,:,0]))
    print(f'{t:5d} {nh:7d} {nv:7d}')
