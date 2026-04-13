# BackTrac: Backward Test Particle Tracing in MAGE

Determine what fraction of storm-time ring current H+ originates from plasma sheet bubbles (BBFs) vs quiet convection, using backward particle tracing through GAMERA-RCM coupled MHD fields.

## Methods

Two independent backward tracers are compared:

| Method | Code | Physics | Speed |
|--------|------|---------|-------|
| **Full Orbit (FO)** | CHIMP Boris pusher | Complete Lorentz force, E=-VxB | ~10h for 10k TPs (Derecho, 128 cores) |
| **Sina's V_eff** | `pilot/run_sina_veff.py` | Bounce-averaged drift, V_eff=V+lambda*V_M | ~3 min for 1k TPs (32 cores) |

### Full Orbit (FO)

Boris pusher in CHIMP with fixed dt=0.01s. Uses GAMERA E=-VxB (includes both electrostatic and inductive E field). Equatorial projection via field line tracing (`doEQProj=T` outputs `xeq`, `yeq`).

CHIMP source: `kaiju-private` repo (Bitbucket, private), branch `backtrac`.

### Sina's V_eff Backtracker

Bounce-averaged drift using RCM's effective potential:

```
V_eff = V + lambda * V_M
```

- `V`: electrostatic potential from REMIX (Vasyliunas equation)
- `V_M = (bVol * 1e-9)^(-2/3)`: encodes flux tube volume (magnetic geometry)
- `lambda`: adiabatic invariant, fixed per particle
- Energy: `K[keV] = |lambda| * V_M / 1000` (varies with position)
- `lambda * V_M` = particle energy in eV (= Volts for q=e)

Inductive E field is captured implicitly through the time-varying (I,J) -> (x,y) mapping via XMIN(I,J,t).

Adapted from Sina Sadeghzadeh's standalone RCM backtracker. Key fix for MAGE: COLAT stored in radians, not degrees.

## Bubble Identification

A particle is classified as *bubble origin* if it **ever** passes through a region satisfying all of:

- dBz > 15 nT (dipolarization signature)
- MLT 21-03 (midnight sector)
- x < 0 (nightside)
- r > 6 RE (not inner ring current)

## Pilot Results

1000 H+ ions, 10-200 keV, seeded uniformly in r=4.2-6.6 RE.  
4-hour backward trace: 10:00 -> 06:00 UT, St. Patrick's Day storm (17 March 2013).

|  | FO | Sina V_eff |
|--|-----|-----------|
| Bubble | 30% | 16% |
| Pre-existing | 15% | 70% |
| Convection | 56% | 15% |

FO sees more bubbles because full-orbit grad-curv drift carries particles through bubble channels faster than bounce-averaged drift. Direction agreement cos(theta) = 0.98 at 5 min; speed ratio ~0.83.

## Directory Structure

```
backtrac/
  pilot/
    run_sina_veff.py        # Sina V_eff backtracker (multiprocessing)
    make_comparison_video.py # FO vs Sina video with dBz + bubble contour
    make_video_mt.py        # FO-only video with classified TPs
    analyze_production.py   # Sciola bubble classification at fixed radius
    run_pilot.pbs           # PBS job script for CHIMP FO run
    sp13_backtrace.xml      # CHIMP XML config for backward trace
  slides/
    backtrac_pilot.tex      # Beamer slides (LaTeX)
    backtrac_pilot.pptx     # PowerPoint version with embedded video
  rcm_backtrac/             # Sina's original code (separate git repo)
```

## Prerequisites

### For Sina V_eff (`run_sina_veff.py`)

```bash
pip install numpy scipy h5py
```

Input data (from MAGE sp13_075 simulation):
- `sp13_075_rcm_mage.h5`: RCM fields (V, VM, COLAT, ALOCT, XMIN, YMIN) converted to Sina format
- `sp13_10k.000001.h5part`: CHIMP FO output for initial conditions
- `sp13_1k_indices.npy`: subset indices

### For CHIMP FO

Requires CHIMP build from `kaiju-private` (branch `backtrac`). Key XML settings:

```xml
<pusher doBackward="T" doFixedDt="T" imeth="FO"/>
<output doEQProj="T" dtOut="60.0"/>
<init file="sp13_1k.tpInit" format="phaseSpace"/>
```

### For comparison video

```bash
pip install matplotlib
# Also needs kaipy (private): sys.path.insert(0, '/path/to/kaipy-private')
```

## Quick Start

### Run Sina V_eff (1000 particles, ~3 min on 32 cores)

```bash
cd pilot
python3 run_sina_veff.py
# Output: sina_1k_traj.npy (241 timesteps x 1000 particles x 2 [x,y])
```

### Generate comparison video

```bash
cd pilot
python3 make_comparison_video.py
# Renders 241 frames, then use ffmpeg:
ffmpeg -framerate 12 -i frames/frame_%04d.png -vcodec libx264 -crf 24 \
  -pix_fmt yuv420p -movflags +faststart output.mp4
```

### Run CHIMP FO (requires Derecho)

```bash
ssh derecho
cd /glade/derecho/scratch/yizhu/backtrac_pilot
qsub run_pilot.pbs
```

## Data Locations (NCAR GLADE)

```
/glade/derecho/scratch/sbao/sp13_075/          # GAMERA-RCM simulation
/glade/derecho/scratch/yizhu/backtrac_pilot/   # Working directory
  sp13_eqproj.000001.h5part                    # FO output with xeq/yeq
  sp13_075_rcm_mage.h5                         # RCM data in Sina format
  sina_1k_traj.npy                             # Sina V_eff trajectories
```

## References

- Sciola et al. (2023): Bubble contribution to ring current via enthalpy flux
- Yang et al. (2015): RCM-E backward trace, ~60% bubble pressure contribution
- Sadeghzadeh (2024): RCM standalone backtracker (adapted here for MAGE)

## Authors

Y. Zhu, S. Bao, S. Sadeghzadeh, F. Toffoletto  
Department of Physics and Astronomy, Rice University
