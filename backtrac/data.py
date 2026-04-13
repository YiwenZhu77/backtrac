"""Load and cache MAGE RCM field data."""

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from .config import RunConfig


class MageRCMData:
    """Manages MAGE RCM field data: V, VM, COLAT, ALOCT, XMIN, YMIN.

    Data is organized in chunks (one per minute). Each chunk contains
    2D fields on the RCM ionospheric (J, I) grid.
    """

    FIELDS = ['XMIN', 'YMIN', 'V', 'VM', 'COLAT', 'ALOCT']

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.i_max = cfg.i_max
        self.j_max = cfg.j_max
        self.chunk_size = self.i_max * self.j_max
        self.j_period = self.j_max - cfg.physics.j_period_offset

        self.I_vals = np.arange(1, self.i_max + 1, dtype=np.float64)
        self.J_vals = np.arange(1, self.j_max + 1, dtype=np.float64)

        self._chunks: dict[int, dict[str, np.ndarray]] = {}

    def load_all(self):
        """Load all chunks into memory."""
        with h5py.File(self.cfg.rcm_data, 'r') as f:
            for ci in range(self.cfg.n_chunks):
                self._chunks[ci] = self._read_chunk(f, ci)

    def _read_chunk(self, f: h5py.File, chunk_idx: int) -> dict[str, np.ndarray]:
        idx = chunk_idx * self.chunk_size
        d = {}
        for k in self.FIELDS:
            d[k] = f[k][idx:idx + self.chunk_size].reshape(self.j_max, self.i_max)
        d['ALOCT_sin'] = np.sin(d['ALOCT'])
        d['ALOCT_cos'] = np.cos(d['ALOCT'])
        return d

    def get_chunk(self, chunk_idx: int) -> dict[str, np.ndarray]:
        """Get field data for a specific chunk."""
        return self._chunks[chunk_idx]

    def interpolator(self, data_2d: np.ndarray, fill=None) -> RegularGridInterpolator:
        """Build a RegularGridInterpolator on the (J, I) grid."""
        return RegularGridInterpolator(
            (self.J_vals, self.I_vals), data_2d,
            method='linear', bounds_error=False, fill_value=fill,
        )

    def get_xy_at(self, chunk_idx: int, I: float, J: float):
        """Map (I, J) to equatorial (x, y) for a given chunk."""
        cd = self.get_chunk(chunk_idx)
        ix = self.interpolator(cd['XMIN'], fill=np.nan)
        iy = self.interpolator(cd['YMIN'], fill=np.nan)
        return float(ix([J, I])[0]), float(iy([J, I])[0])

    def get_vm_at(self, chunk_idx: int, I: float, J: float) -> float:
        """Get V_M at (I, J) for a given chunk."""
        cd = self.get_chunk(chunk_idx)
        ivm = self.interpolator(cd['VM'], fill=np.nan)
        return float(ivm([J, I])[0])
