"""Configuration management for BackTrac."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class PhysicsConfig:
    """Physical constants and numerical parameters."""
    RE: float = 6.37e6          # Earth radius [m]
    RI: float = 6.47e6          # Ionosphere radius [m] (RE + 100 km)
    B0: float = 30570e-9        # Dipole equatorial field [T]
    epsilon: float = 1e-10      # Floor for Jacobian fac

    n_substeps: int = 10        # Euler substeps per minute
    dt_chunk: float = -60.0     # Seconds per chunk (negative = backward)

    j_period_offset: int = 3    # j_period = j_max - this


@dataclass
class BubbleConfig:
    """Bubble identification criteria."""
    dbz_threshold: float = 15.0     # nT
    mlt_min: float = 21.0           # MLT range start (hours)
    mlt_max: float = 3.0            # MLT range end (hours, wraps around midnight)
    r_min: float = 6.0              # Minimum r to count [RE]
    x_max: float = 0.0              # Must be nightside (x < x_max)


@dataclass
class RunConfig:
    """Runtime configuration for a backtrace run."""
    # Data paths
    rcm_data: str = ""              # MAGE RCM data HDF5 (Sina format)
    gamera_dir: str = ""            # GAMERA simulation directory (for dBz)
    output_dir: str = ""            # Output directory

    # Grid
    i_max: int = 180
    j_max: int = 361
    n_chunks: int = 241             # Number of time chunks (0..240)
    start_chunk: int = 240          # Backward start (latest time)

    # Particles
    n_particles: int = 1000
    K_min: float = 10.0             # keV
    K_max: float = 200.0            # keV
    r_min: float = 4.2              # RE, inner seed radius
    r_max: float = 6.6              # RE, outer seed radius

    # Parallel
    n_cores: int = 32

    # Physics
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    bubble: BubbleConfig = field(default_factory=BubbleConfig)


def load_config(path: str) -> RunConfig:
    """Load configuration from YAML file."""
    with open(path) as f:
        d = yaml.safe_load(f)

    cfg = RunConfig()

    # Top-level fields
    for key in ['rcm_data', 'gamera_dir', 'output_dir', 'i_max', 'j_max',
                'n_chunks', 'start_chunk', 'n_particles', 'K_min', 'K_max',
                'r_min', 'r_max', 'n_cores']:
        if key in d:
            setattr(cfg, key, d[key])

    # Nested physics
    if 'physics' in d:
        for key, val in d['physics'].items():
            setattr(cfg.physics, key, val)

    # Nested bubble
    if 'bubble' in d:
        for key, val in d['bubble'].items():
            setattr(cfg.bubble, key, val)

    return cfg
