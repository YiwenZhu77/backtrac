"""
Analyze sp13 production backward trace results.
Sciola (2023) method:
  - Evaluate at 6 RE nightside arc
  - Bubble criteria: delta_B_tau > 10 nT AND Vr < 0, tau=2 min
  - Energy-weighted bubble fraction (enthalpy flux proxy)
  - Flag ±1 RE around bubble, extend 1 min forward
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 12

# ============ Config ============
H5FILE = '/glade/derecho/scratch/yizhu/backtrac_pilot/sp13_prod.000001.h5part'
oBScl = 0.2672  # code → nT
BOUNDARY_R = 6.0  # RE (Sciola)
MAX_K = 500.0  # keV cutoff
TAU_STEPS = 4  # 4 * 30s = 120s = 2 min
DTOUT = 30.0  # seconds per output step
FIGDIR = '/glade/work/yizhu/backtrac/pilot/figures'

import os
os.makedirs(FIGDIR, exist_ok=True)

# ============ Read data ============
print("Reading h5part...")
f = h5py.File(H5FILE, 'r')
steps = sorted([k for k in f.keys() if k.startswith('Step#')], key=lambda s: int(s.split('#')[1]))
valid = [s for s in steps if len(f[s].keys()) > 0]
Np = len(f[valid[0]]['x'][:])
Nt = len(valid)
print(f"Particles: {Np}, Valid steps: {Nt}")

x = np.array([f[s]['x'][:] for s in valid])
y = np.array([f[s]['y'][:] for s in valid])
z = np.array([f[s]['z'][:] for s in valid])
Bz = np.array([f[s]['Bz'][:] for s in valid]) * oBScl  # nT
K = np.array([f[s]['K'][:] for s in valid])
isIn = np.array([f[s]['isIn'][:] for s in valid])
f.close()

t_phys = 36000 - np.arange(Nt) * DTOUT  # backward
r_eq = np.sqrt(x**2 + y**2)

# ============ Classify at 6 RE ============
print(f"\nClassifying at {BOUNDARY_R} RE...")

crossings = []  # dicts
for p in range(Np):
    # Skip unphysical
    if np.any(K[:, p] > MAX_K):
        continue

    # Find first output step where r_eq > 6 RE (backward = leaving RC)
    crossed = np.where(r_eq[:, p] > BOUNDARY_R)[0]
    if len(crossed) == 0:
        continue

    ti = crossed[0]
    if ti + TAU_STEPS >= Nt or ti == 0:
        continue

    # Bz at crossing and delta_B_tau
    Bz_cross = Bz[ti, p]
    Bz_prev = np.mean(Bz[ti+1:ti+TAU_STEPS+1, p])
    dBz = Bz_cross - Bz_prev

    # Physical radial velocity
    dr_bwd = r_eq[ti, p] - r_eq[ti-1, p]
    Vr_phys = -dr_bwd / DTOUT * 6371  # km/s

    # MLT
    phi = np.degrees(np.arctan2(y[ti, p], x[ti, p]))
    mlt = (phi / 15.0 + 12) % 24

    # Nightside
    is_nightside = (mlt >= 18) or (mlt <= 6)

    # Sciola criteria
    is_bubble = (dBz > 10) and (Vr_phys < 0) and is_nightside

    crossings.append({
        'p': p, 'ti': ti, 't': t_phys[ti],
        'x': x[ti,p], 'y': y[ti,p], 'r': r_eq[ti,p],
        'Bz': Bz_cross, 'dBz': dBz, 'Vr': Vr_phys,
        'K': K[ti,p], 'K_start': K[0,p],
        'mlt': mlt, 'nightside': is_nightside, 'bubble': is_bubble
    })

# ============ Statistics ============
night = [c for c in crossings if c['nightside']]
bubble = [c for c in night if c['bubble']]
conv = [c for c in night if not c['bubble']]

print(f"\n{'='*50}")
print(f"Total crossing 6 RE: {len(crossings)}")
print(f"Nightside (18-06 MLT): {len(night)}")
print(f"  Bubble (dBz>10 & Vr<0): {len(bubble)} ({len(bubble)/max(len(night),1)*100:.1f}%)")
print(f"  Convection: {len(conv)} ({len(conv)/max(len(night),1)*100:.1f}%)")

# Energy-weighted (enthalpy flux proxy)
K_bubble = sum(c['K'] for c in bubble)
K_conv = sum(c['K'] for c in conv)
K_total = K_bubble + K_conv
print(f"\nEnergy-weighted (enthalpy flux proxy):")
print(f"  Bubble energy: {K_bubble:.0f} keV ({K_bubble/max(K_total,1)*100:.1f}%)")
print(f"  Convection energy: {K_conv:.0f} keV ({K_conv/max(K_total,1)*100:.1f}%)")
print(f"  Sciola (2023): >=50%")

# By energy band
print(f"\nBubble fraction by energy:")
for lo, hi in [(10,30), (30,50), (50,80), (80,120), (120,200)]:
    nb = sum(1 for c in bubble if lo <= c['K'] < hi)
    nc = sum(1 for c in conv if lo <= c['K'] < hi)
    nt = nb + nc
    frac = nb/max(nt,1)*100
    print(f"  {lo:3d}-{hi:3d} keV: {nb}/{nt} = {frac:.0f}%")

# ============ Plots ============
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Polar plot of crossings at 6 RE arc
ax = axes[0,0]
for c in conv:
    ax.scatter(c['x'], c['y'], c='steelblue', s=8, alpha=0.3)
for c in bubble:
    ax.scatter(c['x'], c['y'], c='red', s=20, alpha=0.7, zorder=5)
ax.add_patch(plt.Circle((0,0), 1, color='k', fill=True))
ax.add_patch(plt.Circle((0,0), BOUNDARY_R, color='gray', fill=False, ls='--', alpha=0.3))
ax.set_xlabel('X (RE)'); ax.set_ylabel('Y (RE)')
ax.set_title(f'6 RE Crossings: Bubble (red) vs Conv (blue)')
ax.set_aspect('equal'); ax.grid(alpha=0.2)
ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)

# 2. Energy histogram
ax = axes[0,1]
K_b = [c['K'] for c in bubble]
K_c = [c['K'] for c in conv]
bins = np.linspace(0, 200, 20)
ax.hist(K_c, bins=bins, color='steelblue', alpha=0.5, label=f'Convection ({len(conv)})')
ax.hist(K_b, bins=bins, color='red', alpha=0.7, label=f'Bubble ({len(bubble)})')
ax.set_xlabel('K at 6 RE crossing (keV)'); ax.set_ylabel('Count')
ax.set_title('Energy Distribution at 6 RE')
ax.legend(); ax.grid(alpha=0.2)

# 3. Bubble fraction by energy
ax = axes[1,0]
E_centers = []
fracs = []
for lo, hi in [(10,20),(20,30),(30,40),(40,50),(50,60),(60,80),(80,100),(100,130),(130,200)]:
    nb = sum(1 for c in bubble if lo <= c['K'] < hi)
    nc = sum(1 for c in conv if lo <= c['K'] < hi)
    nt = nb + nc
    if nt > 0:
        E_centers.append((lo+hi)/2)
        fracs.append(nb/nt*100)
ax.bar(E_centers, fracs, width=[e*0.3 for e in E_centers], color='#e74c3c', alpha=0.7, edgecolor='k')
ax.axhline(y=50, color='gray', ls='--', alpha=0.5, label='Sciola (2023): 50%')
ax.set_xlabel('Energy (keV)'); ax.set_ylabel('Bubble Fraction (%)')
ax.set_title('Bubble Contribution by Energy')
ax.legend(); ax.grid(alpha=0.2)
ax.set_xlim(0, 210)

# 4. Crossing time distribution
ax = axes[1,1]
t_b = [c['t']/3600 for c in bubble]
t_c = [c['t']/3600 for c in conv]
bins_t = np.linspace(6, 10, 20)
ax.hist(t_c, bins=bins_t, color='steelblue', alpha=0.5, label='Convection')
ax.hist(t_b, bins=bins_t, color='red', alpha=0.7, label='Bubble')
ax.set_xlabel('UT (hours)'); ax.set_ylabel('Count')
ax.set_title('6 RE Crossing Time')
ax.legend(); ax.grid(alpha=0.2)

plt.suptitle(f'BackTrac Production: 6 RE Sciola Analysis\n'
             f'{len(night)} nightside crossings, '
             f'Bubble: {len(bubble)} ({len(bubble)/max(len(night),1)*100:.0f}%), '
             f'Energy-weighted: {K_bubble/max(K_total,1)*100:.0f}%', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/sp13_production_sciola.png', dpi=150)
plt.savefig(f'{FIGDIR}/sp13_production_sciola.pdf')
print(f"\nSaved figures to {FIGDIR}/")
