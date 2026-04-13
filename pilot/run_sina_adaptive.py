#!/usr/bin/env python3
"""Sina rewrite with ADAPTIVE differencing (matching Sina's original compute_adaptive_steps)."""
import numpy as np, h5py, time
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count

DATA = '/glade/derecho/scratch/yizhu/backtrac_pilot/sp13_075_rcm_mage.h5'
RE = 6.37e6; RI = RE + 1e5; B0 = 30570e-9; EPS = 1e-10
i_max = 180; j_max = 361; chunk_size = i_max * j_max; j_period = j_max - 3
I_vals = np.arange(1, i_max+1, dtype=np.float64)
J_vals = np.arange(1, j_max+1, dtype=np.float64)

print('Loading chunks...')
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
        stopped = False
        for _ in range(10):
            theta = ic([J_p, I_p])[0]
            if np.isnan(theta): stopped = True; break
            # Adaptive delta (Sina's method)
            ini_d = 0.01
            cm = ic([J_p, I_p-ini_d])[0]; cp = ic([J_p, I_p+ini_d])[0]
            dth0 = (cp-cm)/(2*ini_d)
            Jm=J_p-ini_d; Jp=J_p+ini_d
            if Jm<1: Jm+=j_period
            if Jp>j_max: Jp-=j_period
            sm=ias([Jm,I_p])[0]; cmm=iac([Jm,I_p])[0]; sp=ias([Jp,I_p])[0]; cpp=iac([Jp,I_p])[0]
            dp = np.arctan2(sp,cpp)-np.arctan2(sm,cmm)
            if dp>np.pi: dp-=2*np.pi
            elif dp<-np.pi: dp+=2*np.pi
            dph0 = dp/(2*ini_d)
            L_th = 1.0/max(abs(dth0),1e-6); L_ph = 1.0/max(abs(dph0),1e-6)
            dI = max(0.01, min(0.001*L_th, 0.1)); dJ = max(0.01, min(0.001*L_ph, 0.1))
            # Second pass
            cm = ic([J_p, I_p-dI])[0]; cp = ic([J_p, I_p+dI])[0]
            dth = (cp-cm)/(2*dI)
            Jm=J_p-dJ; Jp=J_p+dJ
            if Jm<1: Jm+=j_period
            if Jp>j_max: Jp-=j_period
            sm=ias([Jm,I_p])[0]; cmm=iac([Jm,I_p])[0]; sp=ias([Jp,I_p])[0]; cpp=iac([Jp,I_p])[0]
            dp = np.arctan2(sp,cpp)-np.arctan2(sm,cmm)
            if dp>np.pi: dp-=2*np.pi
            elif dp<-np.pi: dp+=2*np.pi
            dph = dp/(2*dJ)
            dth = max(abs(dth),1e-8)*(1 if dth>=0 else -1)
            dph = max(abs(dph),1e-8)*(1 if dph>=0 else -1)
            # dVeff with adaptive delta
            VmI=iv([J_p,I_p-dI])[0]+lam*ivm([J_p,I_p-dI])[0]
            VpI=iv([J_p,I_p+dI])[0]+lam*ivm([J_p,I_p+dI])[0]
            dVeI=(VpI-VmI)/(2*dI)
            Jm2=J_p-dJ; Jp2=J_p+dJ
            if Jm2<1: Jm2+=j_period
            if Jp2>j_max: Jp2-=j_period
            VmJ=iv([Jm2,I_p])[0]+lam*ivm([Jm2,I_p])[0]
            VpJ=iv([Jp2,I_p])[0]+lam*ivm([Jp2,I_p])[0]
            dVeJ=(VpJ-VmJ)/(2*dJ)
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
print(f'Running on {n_cores} cores...')
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
np.save('/glade/derecho/scratch/yizhu/backtrac_pilot/sina_adaptive_1k_traj.npy', res)

# Compare
res_fixed = np.load('/glade/derecho/scratch/yizhu/backtrac_pilot/sina_orig_1k_traj.npy')
res_rw = np.load('/glade/derecho/scratch/yizhu/backtrac_pilot/sina_1k_traj.npy')
print(f'\nSurvival: adaptive={np.sum(np.isfinite(res[60,:,0]))}, fixed={np.sum(np.isfinite(res_fixed[60,:,0]))}, rewrite={np.sum(np.isfinite(res_rw[60,:,0]))}')
print(f'\nAdaptive vs Fixed original:')
for t in [1,5,10,20,60]:
    v=np.isfinite(res[t,:,0])&np.isfinite(res_fixed[t,:,0])
    if v.sum()<5: continue
    dx=np.sqrt((res[t,v,0]-res_fixed[t,v,0])**2+(res[t,v,1]-res_fixed[t,v,1])**2)
    print(f'  t={t}: median dx={np.median(dx):.4f} RE')
print(f'\nAdaptive vs Fixed-delta rewrite:')
for t in [1,5,10,20,60]:
    v=np.isfinite(res[t,:,0])&np.isfinite(res_rw[t,:,0])
    if v.sum()<5: continue
    dx=np.sqrt((res[t,v,0]-res_rw[t,v,0])**2+(res[t,v,1]-res_rw[t,v,1])**2)
    print(f'  t={t}: median dx={np.median(dx):.4f} RE')
