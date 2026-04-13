"""V_eff drift integrator for backward particle tracing."""

import numpy as np
from multiprocessing import Pool, cpu_count
from .config import RunConfig, PhysicsConfig
from .data import MageRCMData


def _compute_drift(I: float, J: float, lam: float,
                   ic, ias, iac, iv, ivm,
                   phys: PhysicsConfig, j_max: int, j_period: int):
    """Compute dI/dt, dJ/dt at position (I, J) with given lambda.

    Returns (dI_dt, dJ_dt) or (nan, nan) if invalid.
    """
    theta = ic([J, I])[0]
    if np.isnan(theta):
        return np.nan, np.nan

    d = 0.5  # finite difference step

    # dtheta/dI
    dth = (ic([J, I + d])[0] - ic([J, I - d])[0]) / (2 * d)

    # dphi/dJ (with wrapping)
    Jm, Jp = J - d, J + d
    if Jm < 1: Jm += j_period
    if Jp > j_max: Jp -= j_period
    sm = ias([Jm, I])[0]; cm = iac([Jm, I])[0]
    sp = ias([Jp, I])[0]; cp = iac([Jp, I])[0]
    dphi = np.arctan2(sp, cp) - np.arctan2(sm, cm)
    if dphi > np.pi: dphi -= 2 * np.pi
    elif dphi < -np.pi: dphi += 2 * np.pi
    dph = dphi / (2 * d)

    dth = max(abs(dth), 1e-8) * (1 if dth >= 0 else -1)
    dph = max(abs(dph), 1e-8) * (1 if dph >= 0 else -1)

    # dV_eff/dI and dV_eff/dJ
    VmI = iv([J, I - d])[0] + lam * ivm([J, I - d])[0]
    VpI = iv([J, I + d])[0] + lam * ivm([J, I + d])[0]
    dVeI = (VpI - VmI) / (2 * d)

    Jm2, Jp2 = J - d, J + d
    if Jm2 < 1: Jm2 += j_period
    if Jp2 > j_max: Jp2 -= j_period
    VmJ = iv([Jm2, I])[0] + lam * ivm([Jm2, I])[0]
    VpJ = iv([Jp2, I])[0] + lam * ivm([Jp2, I])[0]
    dVeJ = (VpJ - VmJ) / (2 * d)

    # Jacobian (fac) with dipole Br
    Br = -2 * phys.B0 * (phys.RE / phys.RI) ** 3 * np.cos(theta)
    fac = phys.RI ** 2 * abs(Br) * np.sin(theta) * dph * dth
    fac = max(abs(fac), phys.epsilon) * (np.sign(fac) if fac != 0 else 1)

    return dVeJ / fac, -dVeI / fac


def integrate_particle(args):
    """Integrate one particle backward. Called by multiprocessing.Pool.

    Args: (I_start, J_start, lambda, n_steps, cfg_dict)
    Returns: list of (x, y) tuples, one per timestep.
    """
    I_start, J_start, lam, n_steps, cfg_dict = args

    # Reconstruct objects from serializable dict
    cfg = RunConfig(**{k: v for k, v in cfg_dict.items()
                       if k not in ('physics', 'bubble')})
    cfg.physics = PhysicsConfig(**cfg_dict.get('physics', {}))

    data = MageRCMData(cfg)
    data.load_all()

    phys = cfg.physics
    I_p, J_p = float(I_start), float(J_start)
    sub_dt = phys.dt_chunk / phys.n_substeps
    j_period = data.j_period

    # Initial position
    x0, y0 = data.get_xy_at(cfg.start_chunk, I_p, J_p)
    traj = [(x0, y0)]

    for step in range(n_steps):
        chunk = cfg.start_chunk - step
        if chunk < 0:
            break

        cd = data.get_chunk(chunk)
        ic = data.interpolator(cd['COLAT'], fill=None)
        ias = data.interpolator(cd['ALOCT_sin'], fill=0.0)
        iac = data.interpolator(cd['ALOCT_cos'], fill=0.0)
        iv = data.interpolator(cd['V'], fill=0.0)
        ivm = data.interpolator(cd['VM'], fill=0.0)

        stopped = False
        for _ in range(phys.n_substeps):
            dIdt, dJdt = _compute_drift(
                I_p, J_p, lam, ic, ias, iac, iv, ivm,
                phys, cfg.j_max, j_period,
            )
            if np.isnan(dIdt):
                stopped = True
                break

            I_p += sub_dt * dIdt
            J_p += sub_dt * dJdt

            if J_p < 1: J_p += j_period
            if J_p > cfg.j_max: J_p -= j_period
            if I_p < 1.5 or I_p > cfg.i_max - 0.5:
                stopped = True
                break

        if stopped:
            traj.append((np.nan, np.nan))
            break

        x, y = data.get_xy_at(chunk, I_p, J_p)
        traj.append((x, y))

    return traj


def run_backtrace(particles, cfg: RunConfig):
    """Run backward trace for a list of particles using multiprocessing.

    Args:
        particles: list of (I, J, lambda) tuples
        cfg: RunConfig

    Returns:
        np.ndarray of shape (n_steps+1, n_particles, 2)
    """
    from dataclasses import asdict
    cfg_dict = asdict(cfg)

    n_steps = cfg.start_chunk
    tasks = [(I, J, lam, n_steps, cfg_dict) for I, J, lam in particles]

    n_cores = min(cfg.n_cores, cpu_count())
    print(f'Running {len(tasks)} particles on {n_cores} cores...')

    with Pool(n_cores) as pool:
        results = pool.map(integrate_particle, tasks)

    # Pack into array
    out = np.full((n_steps + 1, len(particles), 2), np.nan)
    for pi, traj in enumerate(results):
        for ti, (x, y) in enumerate(traj):
            if ti > n_steps:
                break
            out[ti, pi] = [x, y]

    return out
