"""Bubble classification using GAMERA equatorial dBz."""

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from .config import BubbleConfig


class BubbleClassifier:
    """Classify particles as bubble or non-bubble based on dBz at their position.

    A particle is 'bubble origin' if it EVER passes through a region
    satisfying all bubble criteria simultaneously.
    """

    def __init__(self, cfg: BubbleConfig):
        self.cfg = cfg

    def is_bubble_region(self, x: np.ndarray, y: np.ndarray,
                         dbz: np.ndarray) -> np.ndarray:
        """Check which positions satisfy bubble criteria."""
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.degrees(np.arctan2(y, x))
        mlt = (phi / 15.0 + 12) % 24

        if self.cfg.mlt_min > self.cfg.mlt_max:
            # Wraps around midnight (e.g., 21-03)
            midnight = (mlt >= self.cfg.mlt_min) | (mlt <= self.cfg.mlt_max)
        else:
            midnight = (mlt >= self.cfg.mlt_min) & (mlt <= self.cfg.mlt_max)

        return (
            (dbz > self.cfg.dbz_threshold) &
            (x < self.cfg.x_max) &
            (r > self.cfg.r_min) &
            midnight
        )

    def classify(self, trajectories: np.ndarray,
                 dbz_interps: list) -> np.ndarray:
        """Classify particles based on their trajectories.

        Args:
            trajectories: (n_steps, n_particles, 2) array of (x, y) positions.
            dbz_interps: list of callables, one per timestep.
                Each takes (x, y) arrays and returns dBz values.

        Returns:
            Boolean array of shape (n_particles,): True = bubble origin.
        """
        n_steps, n_particles, _ = trajectories.shape
        ever_in_bubble = np.zeros(n_particles, dtype=bool)

        for t in range(min(n_steps, len(dbz_interps))):
            interp = dbz_interps[t]
            if interp is None:
                continue

            xt = trajectories[t, :, 0]
            yt = trajectories[t, :, 1]
            valid = np.isfinite(xt) & ~ever_in_bubble

            if valid.sum() == 0:
                continue

            dbz_at = interp(xt[valid], yt[valid])
            in_bub = self.is_bubble_region(xt[valid], yt[valid], dbz_at)
            idx_valid = np.where(valid)[0]
            ever_in_bubble[idx_valid[in_bub]] = True

        return ever_in_bubble


def load_gamera_dbz(gamera_dir: str, step: int):
    """Load dBz from GAMERA and build an interpolator.

    Requires kaipy. Returns a callable interp(x, y) -> dBz.
    """
    import sys
    # Try common kaipy paths
    for kp in ['/glade/work/yizhu/OpOF/kaipy-private', './kaipy-private']:
        if kp not in sys.path:
            sys.path.insert(0, kp)

    import os
    old_cwd = os.getcwd()
    os.chdir(gamera_dir)

    import kaipy.gamera.magsphere as msph
    gam = msph.GamsphPipe(gamera_dir, 'msphere', doFast=True)
    dbz = gam.DelBz(step)
    xxi = gam.xxi.copy()
    yyi = gam.yyi.copy()

    os.chdir(old_cwd)

    # Cell centers
    xc = 0.5 * (xxi[:-1, :] + xxi[1:, :])
    xc = 0.5 * (xc[:, :-1] + xc[:, 1:])
    yc = 0.5 * (yyi[:-1, :] + yyi[1:, :])
    yc = 0.5 * (yc[:, :-1] + yc[:, 1:])

    valid = np.isfinite(dbz) & (np.sqrt(xc ** 2 + yc ** 2) > 2)
    interp = LinearNDInterpolator(
        np.column_stack([xc[valid].ravel(), yc[valid].ravel()]),
        dbz[valid].ravel(),
    )
    return interp, dbz, xxi, yyi
