#!/usr/bin/env python3
"""
plot_cost_line_slices_comparison.py — Overlay int_para2axi_old_fom/'s and int_para2axi_new_fom/'s cached
cost_line_slices data on the same three panels (init->final, init->target,
final->target), one color per FOM version, so the old-FOM and new-FOM cost
landscapes along these lines can be compared directly.

Reads the cached per-pair CSVs written by each folder's
plot_cost_line_slices.py (cost_line_slices_<from>_<to>_line.csv) -- does not
recompute anything.

Output: comparison_plots/cost_line_slices_comparison.pdf/.png

Run from: FF_Zernike/
    python plot_cost_line_slices_comparison.py
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SOURCES = {
    "old_fom": {"dir": "int_para2axi_old_fom", "label": "Old FOM", "color": "tab:blue"},
    "new_fom": {"dir": "int_para2axi_new_fom", "label": "New FOM", "color": "tab:red"},
}

SLICE_DEFS = [("init", "final"), ("init", "target"), ("final", "target")]

OUT_DIR = "../comparison_plots"
os.makedirs(OUT_DIR, exist_ok=True)

font = {"size": 15}
plt.rc("font", **font)
plt.rcParams["lines.linewidth"] = 2
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

data = {}
for key, src in SOURCES.items():
    for from_name, to_name in SLICE_DEFS:
        csv_path = os.path.join(src["dir"], f"cost_line_slices_{from_name}_{to_name}_line.csv")
        saved = np.genfromtxt(csv_path, delimiter=",", names=True)
        data[(key, from_name, to_name)] = (saved["t"], saved["log10_cost"])

vmin = min(log_cost.min() for (t, log_cost) in data.values())
vmax = max(log_cost.max() for (t, log_cost) in data.values())

fig, axes = plt.subplots(1, len(SLICE_DEFS), figsize=(15, 5))

for ax, (from_name, to_name) in zip(axes, SLICE_DEFS):
    for key, src in SOURCES.items():
        t, log_cost = data[(key, from_name, to_name)]
        ax.plot(t, log_cost, color=src["color"], label=src["label"])
    ax.set_ylim(vmin, vmax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.axvline(0, color="black", lw=1.25, ls="--", alpha=0.8, zorder=10)
    ax.axvline(1, color="black", lw=1.25, ls="--", alpha=0.8, zorder=10)
    ax.set_xlabel(rf"$t$  (0 = {from_name}, 1 = {to_name})")
    ax.set_ylabel("log10(cost)")
    ax.set_title(f"Slice: {from_name} " + r"$\to$" + f" {to_name}")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0),
           ncol=len(SOURCES), fontsize=15, frameon=False)

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.subplots_adjust(bottom=0.22)

out_pdf = os.path.join(OUT_DIR, "cost_line_slices_comparison.pdf")
out_png = os.path.join(OUT_DIR, "cost_line_slices_comparison.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out_pdf} and {out_png}")
