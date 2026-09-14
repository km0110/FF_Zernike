#!/usr/bin/env python3
"""
plot_cost_surface_comparison.py — Compare int_para2axi_old_fom/'s and int_para2axi_new_fom/'s cached
cost_surface_exact contour grids side by side, one column per FOM version,
one row per Zernike-coefficient pair (1,2 / 1,3 / 2,3), sharing one color
scale so the old-FOM and new-FOM cost landscapes can be compared directly.

Reads the cached per-pair CSVs written by each folder's
plot_cost_surface_pairs.py (cost_surface_<pair>_exact_grid.csv) -- does not
recompute anything. The target coefficient (cyan star) is recovered as the
grid's own center, since plot_cost_surface_pairs.py always sweeps
target +/- half_range.

Output: comparison_plots/cost_surface_exact_comparison.pdf/.png

Run from: FF_Zernike/cost_landscape_exploration/
    python plot_cost_surface_comparison.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

SOURCES = {
    "old_fom": {"dir": "int_para2axi_old_fom", "label": r"FOM/$I_T^2$"},
    "new_fom": {"dir": "int_para2axi_new_fom", "label": r"FOM/$I_T^2$ + $b(1-\mu)^2$"},
}

PAIRS = [(1, 2), (1, 3), (2, 3)]
N_LEVELS = 21

OUT_DIR = "../comparison_plots"
os.makedirs(OUT_DIR, exist_ok=True)

font = {"size": 20}
plt.rc("font", **font)
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


def order_of_magnitude(values):
    return int(np.floor(np.log10(np.max(np.abs(values)))))


data = {}
for key, src in SOURCES.items():
    for idx_x, idx_y in PAIRS:
        pair_label = f"{idx_x}{idx_y}"
        csv_path = os.path.join(src["dir"], f"cost_surface_{pair_label}_exact_grid.csv")
        saved = np.genfromtxt(csv_path, delimiter=",", names=True)
        xs = np.unique(saved[saved.dtype.names[0]])
        ys = np.unique(saved[saved.dtype.names[1]])
        Cost = saved[saved.dtype.names[2]].reshape(len(ys), len(xs))
        log_cost = np.log10(np.clip(Cost, 1e-12, None))
        data[(key, idx_x, idx_y)] = (xs, ys, log_cost)

# One shared exponent per Zernike coefficient index, derived from the first
# source only -- both FOM columns describe the same physical setup, so a
# given coefficient's magnitude is the same in either column, and using one
# shared exponent keeps both columns on the same axis scale.
first_source_key = next(iter(SOURCES))
axis_exponent = {}
for idx_x, idx_y in PAIRS:
    xs, ys, _ = data[(first_source_key, idx_x, idx_y)]
    axis_exponent[idx_x] = order_of_magnitude(xs)
    axis_exponent[idx_y] = order_of_magnitude(ys)


def zlabel(idx):
    return f"$a_{{{2 * idx}}}$"


def zlabel_scaled(idx):
    return rf"$a_{{{2 * idx}}}$ ($\times 10^{{{axis_exponent[idx]}}}$)"


scale_source_key = "new_fom"
vmin = min(log_cost.min() for (k, ix, iy), (xs, ys, log_cost) in data.items() if k == scale_source_key)
vmax = max(log_cost.max() for (k, ix, iy), (xs, ys, log_cost) in data.items() if k == scale_source_key)
levels = np.linspace(vmin, vmax, N_LEVELS)

fig, axes = plt.subplots(len(PAIRS), len(SOURCES), figsize=(11, 13))

for row, (idx_x, idx_y) in enumerate(PAIRS):
    for col, (key, src) in enumerate(SOURCES.items()):
        xs, ys, log_cost = data[(key, idx_x, idx_y)]
        xs = xs / 10 ** axis_exponent[idx_x]
        ys = ys / 10 ** axis_exponent[idx_y]
        ax = axes[row, col]

        cf = ax.contourf(xs, ys, log_cost, levels=levels, vmin=vmin, vmax=vmax, cmap="inferno", extend="both")
        ax.set_box_aspect(1)

        target_x, target_y = xs.mean(), ys.mean()
        target_handle = ax.scatter([target_x], [target_y], color="cyan", s=200, marker="*",
                                    label="target", zorder=10, edgecolors="black", linewidths=0.75)

        fixed_idx = next(i for i in (1, 2, 3) if i not in (idx_x, idx_y))
        ax.set_xlabel(zlabel_scaled(idx_x))
        ax.set_ylabel(zlabel_scaled(idx_y))
        ax.set_title(f"{zlabel(idx_x)}, {zlabel(idx_y)}  ({zlabel(fixed_idx)} fixed)")

fig.subplots_adjust(hspace=0.5, wspace=0.35, right=0.86)

cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
fig.colorbar(cf, cax=cbar_ax, label="log10(cost)")

for col, (key, src) in enumerate(SOURCES.items()):
    col_center = (axes[0, col].get_position().x0 + axes[0, col].get_position().x1) / 2
    fig.text(col_center, axes[0, col].get_position().y1 + 0.06, src["label"],
              ha="center", fontsize=24)

fig.legend([target_handle], ["Target Mirror"], loc="lower center", bbox_to_anchor=(0.5, 0),
           ncol=1, fontsize=20, frameon=False)

out_pdf = os.path.join(OUT_DIR, "cost_surface_exact_comparison.pdf")
out_png = os.path.join(OUT_DIR, "cost_surface_exact_comparison.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_pdf} and {out_png}")
