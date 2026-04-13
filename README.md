# BackTrac

Backward test particle tracing through MAGE RCM fields to determine the origin of storm-time ring current H+ ions (bubble vs convection vs pre-existing).

Uses the RCM effective potential V_eff = V + lambda * V_M for bounce-averaged drift on the ionospheric (I,J) grid, with inductive E field captured implicitly through the time-varying XMIN(I,J,t) mapping.

> **Status**: Pilot study using the St. Patrick's Day 2013 GAMERA-RCM simulation (sp13_075, S. Bao).

## Install

```bash
pip install -e .
```

## Usage

### Step 1: Convert MAGE RCM output

Convert `msphere.rcm.h5` from your MAGE simulation to BackTrac format:

```bash
python scripts/convert_mage.py /path/to/msphere.rcm.h5 \
    --steps 360-600 -o data/rcm_data.h5
```

The step range should cover your backward trace interval (e.g., 360-600 = 6-10 UT for sp13_075). MAGE stores COLAT in radians — BackTrac handles this correctly.

### Step 2: Run backward trace

With uniform particle seeding (no external data needed):

```bash
python scripts/run_backtrace.py configs/sp13_075.yaml \
    --uniform -o results/traj.npy
```

Or initialize from a CHIMP h5part file:

```bash
python scripts/run_backtrace.py configs/sp13_075.yaml \
    --h5part /path/to/particles.h5part \
    -o results/traj.npy
```

### Step 3: Classify bubble origins

```bash
python scripts/analyze.py configs/sp13_075.yaml results/traj.npy \
    --meta results/traj_meta.npy
```

### Step 4 (optional): Generate video

```bash
python scripts/make_video.py configs/sp13_075.yaml \
    results/traj.npy results/traj_class.npz -o results/video.mp4
```

Requires GAMERA simulation directory for dBz background (set `gamera_dir` in config).

## Configuration

Edit `configs/sp13_075.yaml` for your simulation:

```yaml
rcm_data: data/rcm_data.h5          # Output from convert_mage.py
gamera_dir: /path/to/simulation/    # For dBz bubble identification

n_particles: 1000
n_cores: 32
K_min: 10.0         # keV
K_max: 200.0        # keV
r_min: 4.2          # RE, seed inner radius
r_max: 6.6          # RE, seed outer radius

bubble:
  dbz_threshold: 15.0   # nT
  mlt_min: 21.0          # Midnight sector
  mlt_max: 3.0
  r_min: 6.0             # Exclude inner RC
```

## Package Structure

```
backtrac/
  backtrac/              # Python package
    config.py            # Configuration (dataclass + YAML)
    data.py              # MAGE RCM data loader
    integrator.py        # V_eff drift integrator (multiprocessing)
    particles.py         # Particle initialization
    classify.py          # Bubble classification using GAMERA dBz
    viz.py               # Video rendering
  scripts/               # CLI entry points
    convert_mage.py      # Convert MAGE output to BackTrac format
    run_backtrace.py     # Run backward trace
    analyze.py           # Classify and report statistics
  configs/
    sp13_075.yaml        # Example config
```

## Physics

**Drift equation** on ionospheric (I,J) grid:

```
dI/dt = (1/fac) * dV_eff/dJ
dJ/dt = -(1/fac) * dV_eff/dI
```

- `V_eff = V + lambda * V_M`
- `V`: electrostatic potential from REMIX (Volts)
- `V_M = (bVol * 1e-9)^(-2/3)`: magnetic geometry factor from flux tube volume
- `lambda * V_M = K[eV]`: particle energy at given position
- `lambda`: adiabatic invariant, fixed per particle (energy changes as particle moves through varying V_M)
- `fac`: Jacobian for (I,J) grid (uses dipole Br, ~4% error vs actual)

**Bubble identification**: particle is *bubble origin* if it ever occupies a region with dBz > threshold in the midnight sector during the backward trace.

## References

- Sciola et al. (2023), JGR: Bubble contribution to ring current
- Yang et al. (2015), JGR: RCM-E backward trace
- Sadeghzadeh (2024): RCM standalone backtracker (adapted for MAGE in this work)

## Authors

Y. Zhu, S. Bao, S. Sadeghzadeh, F. Toffoletto — Rice University
