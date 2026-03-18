# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Generate the silhouette library icon.

A white squircle containing a black silhouette formed by the axes and
the area under an omni-domain power-duration curve shape.

Run from the project root:

    uv run docs/generate_icon.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def omni_curve(t, cp, p_max, w_prime, a, tcp_max):
    base = w_prime / t * (1 - np.exp(-t * (p_max - cp) / w_prime)) + cp
    return np.where(t <= tcp_max, base, base - a * np.log(t / tcp_max))


def main():
    size = 1024
    dpi = 256
    fig_size = size / dpi

    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    fig.patch.set_alpha(0)
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # White squircle background (no border)
    pad = 0.02
    squircle = FancyBboxPatch(
        (pad, pad), 1 - 2 * pad, 1 - 2 * pad,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor="white",
        edgecolor="none",
    )
    ax.add_patch(squircle)

    # Margins inside the squircle for the plot area
    left = 0.20
    bottom = 0.18
    right = 0.78
    top = 0.68

    # Generate the omni curve in data space (log time axis)
    t = np.logspace(np.log10(1), np.log10(7200), 500)
    power = omni_curve(t, cp=300, p_max=1100, w_prime=10000, a=50, tcp_max=1800)

    # Normalize to [0, 1] for placement in icon
    t_norm = (np.log10(t) - np.log10(1)) / (np.log10(7200) - np.log10(1))
    p_min, p_max_val = power.min(), power.max()
    p_norm = (power - p_min) / (p_max_val - p_min)

    # Map to icon coordinates
    x_curve = left + t_norm * (right - left)
    y_curve = bottom + p_norm * (top - bottom)

    # Build the silhouette path: axes + filled area under curve
    # The shape is: bottom-left corner -> up the y-axis -> along the curve
    # -> down to the x-axis -> back along the x-axis
    ax_weight = 0.07  # axis thickness
    overshoot_x = 0.12  # x-axis overshoot past plot area
    overshoot_y = 0.24  # y-axis overshoot above plot area

    # Trace the outer boundary of the silhouette shape clockwise.
    # The shape is: x-axis bar + filled area under curve + y-axis bar,
    # with both axes extending slightly beyond the plot area.
    pts = []

    # Bottom-left corner (outer) -> right along x-axis bottom edge (past plot area)
    pts.append((left - ax_weight, bottom - ax_weight))
    pts.append((right + overshoot_x, bottom - ax_weight))
    # Up to x-axis top edge at the overshoot end
    pts.append((right + overshoot_x, bottom))
    # Left along x-axis top edge to where the curve ends
    pts.append((right, bottom))

    # Along the curve right-to-left (top of filled area)
    for xc, yc in zip(x_curve[::-1], y_curve[::-1]):
        pts.append((xc, yc))

    # From curve start up to y-axis top (with overshoot)
    pts.append((left, top + overshoot_y))
    # Across to outer y-axis edge
    pts.append((left - ax_weight, top + overshoot_y))
    # Down outer y-axis back to start
    pts.append((left - ax_weight, bottom - ax_weight))

    verts = pts
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 2) + [Path.CLOSEPOLY]

    path = Path(verts, codes)
    patch = PathPatch(path, facecolor="black", edgecolor="none")
    patch.set_clip_path(squircle)
    ax.add_patch(patch)

    fig.savefig("docs/icon.png", dpi=dpi, transparent=True)
    print("Saved docs/icon.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
