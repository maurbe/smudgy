import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from matplotlib.patches import Ellipse

from smudgy.utils import project_smoTens_to_2d

np.random.seed(0)

# -----------------------------
# 1) helper: map string to indices
# -----------------------------
def get_plane_indices(projection_plane):
    mapping = {
        'xy': ([0, 1], 'z', 0),
        'xz': ([0, 2], 'y', 1),
        'yz': ([1, 2], 'x', 0),
    }
    return mapping[projection_plane]

# -----------------------------
# 2) ellipsoid helper (3D)
# -----------------------------
def plot_ellipsoid(ax, center, H, color, n=20):
    eigvals, eigvecs = np.linalg.eigh(H)
    radii = np.sqrt(np.maximum(eigvals, 1e-12))

    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=-1)
    ellipsoid = (sphere @ np.diag(radii)) @ eigvecs.T

    X = ellipsoid[..., 0] + center[0]
    Y = ellipsoid[..., 1] + center[1]
    Z = ellipsoid[..., 2] + center[2]

    ax.plot_surface(X, Y, Z, color=color, alpha=0.8, linewidth=0)

# -----------------------------
# 3) plotting function
# -----------------------------
def plot_projection_demo(N, projection_plane='xy'):

    boxsize = 1
    pos = np.random.uniform(0.1, boxsize-0.1, size=(N, 3))

    def random_spd():
        A = np.random.uniform(-10, 10, size=(3, 3)) * 0.01
        return A @ A.T + 0.01 * np.eye(3)

    H = np.stack([random_spd() for _ in range(N)])

    plane_idx, zdir, z_wall = get_plane_indices(projection_plane)

    # projection
    proj = project_smoTens_to_2d(H, plane=plane_idx)[0]

    fig = plt.figure(figsize=(3, 3))
    ax3d = fig.add_subplot(111, projection="3d")

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pos)))

    for i in range(len(pos)):

        # 2D logic
        H2 = proj[i]
        eigvals, eigvecs = np.linalg.eigh(H2)
        radii = np.sqrt(np.maximum(eigvals, 1e-12))
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        # determine 2D center based on plane
        center_2d = (pos[i, plane_idx[0]], pos[i, plane_idx[1]])

        ellipse = Ellipse(xy=center_2d,
                          width=2*radii[0],
                          height=2*radii[1],
                          angle=angle,
                          facecolor=colors[i],
                          alpha=0.8,
                          edgecolor='none',
                          )

        ax3d.add_patch(ellipse)
        art3d.pathpatch_2d_to_3d(ellipse, z=z_wall, zdir=zdir)

        # 3D
        plot_ellipsoid(ax3d, pos[i], H[i], color=colors[i])

    ax3d.set_title("Projected smoothing ellipsoids", fontsize=9)

    ax3d.set_xlabel(r'$x$')
    ax3d.set_ylabel(r'$y$')
    ax3d.set_zlabel(r'$z$')

    ax3d.set_xlim(0, 1)
    ax3d.set_ylim(0, 1)
    ax3d.set_zlim(0, 1)

    return fig, ax3d
