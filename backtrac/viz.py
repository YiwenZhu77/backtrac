"""Visualization: comparison videos and figures."""

import numpy as np
import os


def render_frame(frame: int, trajectories: dict, bubble_masks: dict,
                 dbz_data: tuple, output_dir: str, dbz_threshold: float = 15.0):
    """Render a single comparison frame.

    Args:
        frame: Timestep index.
        trajectories: dict mapping label -> (n_steps, n_particles, 2) array.
        bubble_masks: dict mapping label -> boolean array (n_particles,).
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

    n_panels = len(trajectories)
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 8))
    if n_panels == 1:
        axes = [axes]

    for ax, (label, traj) in zip(axes, trajectories.items()):
        bub = bubble_masks.get(label, np.zeros(traj.shape[1], dtype=bool))

        ax.pcolormesh(xxi, yyi, dBz, cmap='RdBu_r', vmin=-30, vmax=30,
                      shading='auto', zorder=1)
        ax.contour(xc, yc, dBz, levels=[dbz_threshold],
                   colors='lime', linewidths=1.5, zorder=2)
        ax.add_patch(plt.Circle((0, 0), 1, color='k', fill=True, zorder=10))

        v = np.isfinite(traj[frame, :, 0])
        m_nb = v & ~bub
        m_b = v & bub

        if m_nb.sum() > 0:
            ax.scatter(traj[frame, m_nb, 0], traj[frame, m_nb, 1],
                       c='dodgerblue', s=8, alpha=0.5, edgecolors='none', zorder=3)
        if m_b.sum() > 0:
            ax.scatter(traj[frame, m_b, 0], traj[frame, m_b, 1],
                       c='red', s=14, alpha=0.8, edgecolors='darkred',
                       linewidths=0.3, zorder=5)

        ax.set_xlim(-20, 8)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        ax.set_xlabel('X_eq (RE)')
        ax.set_ylabel('Y_eq (RE)')
        ax.set_title(f'{label}  N={v.sum()}  [bub:{m_b.sum()} other:{m_nb.sum()}]')

    t_hr = 10.0 - frame / 60
    fig.suptitle(
        f'{t_hr:.2f} UT ({frame} min bwd)  |  '
        f'green={dbz_threshold:.0f}nT contour  |  red=bubble  blue=non-bubble',
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
        '-vf', 'scale=1600:-2',
        output_path,
    ]
    subprocess.run(cmd, capture_output=True)
