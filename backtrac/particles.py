"""Particle initialization for backward tracing."""

import numpy as np
import h5py
from scipy.spatial import cKDTree
from .config import RunConfig
from .data import MageRCMData


def init_from_h5part(h5part_path: str, data: MageRCMData, cfg: RunConfig,
                     indices: np.ndarray = None):
    """Initialize particles from a CHIMP h5part output file.

    Reads (x, y, K) from Step#0, maps to (I, J) on RCM grid,
    computes lambda = 1000 * K / VM.

    Args:
        h5part_path: Path to CHIMP h5part file.
        data: Loaded MageRCMData (must have called load_all()).
        cfg: RunConfig.
        indices: Optional subset indices into the h5part particle array.

    Returns:
        particles: list of (I, J, lambda) tuples
        metadata: list of (index, K_keV) tuples
    """
    with h5py.File(h5part_path, 'r') as f:
        K_all = f['Step#0']['K'][:]
        x_all = f['Step#0']['x'][:]
        y_all = f['Step#0']['y'][:]

    if indices is not None:
        K_all = K_all[indices]
        x_all = x_all[indices]
        y_all = y_all[indices]

    # Build KD-tree from RCM grid at start chunk
    cd = data.get_chunk(cfg.start_chunk)
    xm, ym, vm = cd['XMIN'], cd['YMIN'], cd['VM']
    valid = (vm > -99999) & (np.sqrt(xm ** 2 + ym ** 2) > 2)
    jj, ii = np.where(valid)
    tree = cKDTree(np.column_stack([xm[valid], ym[valid]]))
    ivm = data.interpolator(vm, fill=np.nan)

    particles = []
    metadata = []
    for pi in range(len(K_all)):
        d, idx = tree.query([x_all[pi], y_all[pi]])
        if d > 0.5:
            continue
        I_s = float(ii[idx] + 1)
        J_s = float(jj[idx] + 1)
        vm_val = float(ivm([J_s, I_s])[0])
        if np.isnan(vm_val) or vm_val < 0.1:
            continue
        lam = 1000.0 * K_all[pi] / vm_val
        particles.append((I_s, J_s, lam))
        metadata.append((pi, float(K_all[pi])))

    return particles, metadata


def init_uniform(data: MageRCMData, cfg: RunConfig,
                 n_particles: int = 1000, seed: int = 42):
    """Initialize particles uniformly in r_eq, random MLT and energy.

    Args:
        data: Loaded MageRCMData.
        cfg: RunConfig.
        n_particles: Number of particles to create.
        seed: Random seed.

    Returns:
        particles: list of (I, J, lambda) tuples
        metadata: list of (index, K_keV) tuples
    """
    rng = np.random.RandomState(seed)
    cd = data.get_chunk(cfg.start_chunk)
    xm, ym, vm = cd['XMIN'], cd['YMIN'], cd['VM']

    valid = (vm > 0) & (np.sqrt(xm ** 2 + ym ** 2) > cfg.r_min) & \
            (np.sqrt(xm ** 2 + ym ** 2) < cfg.r_max)
    jj, ii = np.where(valid)
    if len(jj) == 0:
        raise ValueError("No valid grid points in seed region")

    tree = cKDTree(np.column_stack([xm[valid], ym[valid]]))
    ivm = data.interpolator(vm, fill=np.nan)

    # Uniform in r, random MLT
    r_seed = rng.uniform(cfg.r_min, cfg.r_max, n_particles)
    phi_seed = rng.uniform(0, 2 * np.pi, n_particles)
    x_seed = r_seed * np.cos(phi_seed)
    y_seed = r_seed * np.sin(phi_seed)

    # Log-uniform energy
    K_seed = np.exp(rng.uniform(np.log(cfg.K_min), np.log(cfg.K_max), n_particles))

    particles = []
    metadata = []
    for pi in range(n_particles):
        d, idx = tree.query([x_seed[pi], y_seed[pi]])
        if d > 0.5:
            continue
        I_s = float(ii[idx] + 1)
        J_s = float(jj[idx] + 1)
        vm_val = float(ivm([J_s, I_s])[0])
        if np.isnan(vm_val) or vm_val < 0.1:
            continue
        lam = 1000.0 * K_seed[pi] / vm_val
        particles.append((I_s, J_s, lam))
        metadata.append((pi, float(K_seed[pi])))

    return particles, metadata
