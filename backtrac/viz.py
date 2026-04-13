"""Visualization: trajectory videos and figures."""

import numpy as np
import os


def render_frame(frame: int, trajectories: np.ndarray, bubble_mask: np.ndarray,
                 dbz_data: tuple, output_dir: str, dbz_threshold: float = 15.0):
    """Render a single frame showing particles on dBz background.

    Args:
        frame: Timestep index.
        trajectories: (n_steps, n_particles, 2) array of (x, y).
        bubble_mask: boolean array (n_particles,) — True = bubble origin.
        dbz_data: (dBz_2d, xxi, yyi) for background.
        output_dir: Directory to save frame PNGs.
        dbz_threshold: Contour level for bubble boundary.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dBz, xxi, yyi = dbz_data
    xc = 0.5 * (xxi[:-1, :] + xxi[1:, :])
    xc = 0.5 * (xc[:, :-1] + xc[:, 1:])
    yc = 0.5 * (yyi[:-1, :] + yyi[1:, :])
    yc = 0.5 * (yc[:, :-1] + yc[:, 1:])

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    ax.pcolormesh(xxi, yyi, dBz, cmap='RdBu_r', vmin=-30, vmax=30,
                  shading='auto', zorder=1)
    ax.contour(xc, yc, dBz, levels=[dbz_threshold],
               colors='lime', linewidths=1.5, zorder=2)
    ax.add_patch(plt.Circle((0, 0), 1, color='k', fill=True, zorder=10))

    v = np.isfinite(trajectories[frame, :, 0])
    m_nb = v & ~bubble_mask
    m_b = v & bubble_mask

    if m_nb.sum() > 0:
        ax.scatter(trajectories[frame, m_nb, 0], trajectories[frame, m_nb, 1],
                   c='dodgerblue', s=8, alpha=0.5, edgecolors='none', zorder=3)
    if m_b.sum() > 0:
        ax.scatter(trajectories[frame, m_b, 0], trajectories[frame, m_b, 1],
                   c='red', s=14, alpha=0.8, edgecolors='darkred',
                   linewidths=0.3, zorder=5)

    ax.set_xlim(-20, 8)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_xlabel('X_eq (RE)')
    ax.set_ylabel('Y_eq (RE)')

    t_hr = 10.0 - frame / 60
    ax.set_title(
        f'{t_hr:.2f} UT ({frame} min bwd)  |  '
        f'N={v.sum()} [bubble:{m_b.sum()} other:{m_nb.sum()}]  |  '
        f'green={dbz_threshold:.0f}nT',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'frame_{frame:04d}.png'),
                dpi=90, bbox_inches='tight')
    plt.close(fig)


def make_video(frame_dir: str, output_path: str, fps: int = 12):
    """Combine frames into mp4 using ffmpeg."""
    import subprocess
    cmd = [
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', os.path.join(frame_dir, 'frame_%04d.png'),
        '-vcodec', 'libx264', '-crf', '24', '-preset', 'fast',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-vf', 'scale=1280:-2',
        output_path,
    ]
    subprocess.run(cmd, capture_output=True)
