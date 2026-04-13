# BackTrac

Backward test particle tracing through MAGE RCM fields to determine the origin of storm-time ring current H+ ions (bubble vs convection vs pre-existing).

Uses the RCM effective potential V_eff = V + lambda * V_M for bounce-averaged drift on the ionospheric (I,J) grid, with inductive E field captured implicitly through the time-varying XMIN(I,J,t) mapping.

## Install

```bash
pip install -e .
```

## Quick Start

### 1. Run backward trace

```bash
python scripts/run_backtrace.py configs/sp13_075.yaml \
    --h5part /path/to/sp13_10k.000001.h5part \
    --indices /path/to/sp13_1k_indices.npy \
    -o results/traj.npy
```

Or with uniform seeding (no h5part needed):

```bash
python scripts/run_backtrace.py configs/sp13_075.yaml \
    --uniform -o results/traj.npy
```

### 2. Classify bubble origins

```bash
python scripts/analyze.py configs/sp13_075.yaml results/traj.npy \
    --meta results/traj_meta.npy
```

### 3. Generate comparison video

```bash
python scripts/make_video.py configs/sp13_075.yaml \
    results/traj.npy results/traj_class.npz
```

## Configuration

Edit `configs/sp13_075.yaml`:

```yaml
rcm_data: /path/to/sp13_075_rcm_mage.h5   # MAGE RCM data (Sina format)
gamera_dir: /path/to/sp13_075/             # GAMERA simulation (for dBz)
n_particles: 1000
n_cores: 32

bubble:
  dbz_threshold: 15.0     # nT (dipolarization)
  mlt_min: 21.0            # Midnight sector start
  mlt_max: 3.0             # Midnight sector end
  r_min: 6.0               # Exclude inner RC
```

## Data Preparation

BackTrac needs MAGE RCM output converted to a flat HDF5 format. Use the conversion script:

```bash
python scripts/convert_mage.py /path/to/msphere.rcm.h5 \
    --steps 360-600 -o sp13_075_rcm_mage.h5
```

The output contains per-chunk arrays of V, VM, COLAT, ALOCT, XMIN, YMIN flattened as 1D datasets.

**Important**: MAGE stores COLAT in **radians**, not degrees. BackTrac handles this correctly.

## Package Structure

```
backtrac/
  backtrac/              # Python package
    config.py            # Configuration (dataclass + YAML loading)
    data.py              # MAGE RCM data loader and interpolation
    integrator.py        # V_eff drift integrator (multiprocessing)
    particles.py         # Particle initialization
    classify.py          # Bubble classification using GAMERA dBz
    viz.py               # Video and figure rendering
  scripts/               # CLI entry points
    run_backtrace.py     # Run backward trace
    analyze.py           # Classify and report statistics
  configs/               # YAML config files
    sp13_075.yaml        # St. Patrick's Day 2013 storm
```

## Physics

**Drift equation** on ionospheric (I,J) grid:

```
dI/dt = (1/fac) * dV_eff/dJ
dJ/dt = -(1/fac) * dV_eff/dI
```

where:
- `V_eff = V + lambda * V_M`
- `V`: electrostatic potential from REMIX (Volts)
- `V_M = (bVol * 1e-9)^(-2/3)`: magnetic geometry factor
- `lambda`: adiabatic invariant (fixed per particle)
- `lambda * V_M = K[eV]`: particle energy at given position
- `fac = r_i^2 * |Br| * sin(theta) * dphi/dJ * dtheta/dI`: Jacobian

**Bubble identification**: particle is *bubble origin* if it ever passes through a region with dBz > threshold in the midnight sector.

## References

- Sciola et al. (2023): Bubble contribution to ring current
- Yang et al. (2015): RCM-E backward trace
- Sadeghzadeh (2024): RCM standalone backtracker

## Authors

Y. Zhu, S. Bao, S. Sadeghzadeh, F. Toffoletto — Rice University
