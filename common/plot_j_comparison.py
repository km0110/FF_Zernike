#!/usr/bin/env python3
"""
plot_j_comparison.py — Plot final on-axis intensity for each j (number of
Zernike modes) value in a runs directory, overlaid on a single figure with
the target as a black dashed line.

Shared across the Zernike-coefficient-parameterized directories (originally
adam and adam_sgdr) -- lives in common/.

Usage: python plot_j_comparison.py <parent_dir> [--j-filter 4,6,8,10] [--xlim -2,7]
  e.g. python plot_j_comparison.py int_para2axi_n
       python plot_j_comparison.py int_para2axi_n_eps1e-06 --j-filter 4,6,8,10

--j-filter restricts the plot to specific j values (default: plot every j found).
--xlim sets explicit x-axis bounds in cm (default: auto).

Matches run directory names built by either the legacy "..._n_<j>_..." naming
or the current "..._j_<j>_..." naming, since existing run directories on disk
predate this rename and are left as-is.

Run from inside the experiment directory (e.g. adam/) so the parent_dir
argument and adjoint_intensity_zernike.py resolve correctly.
"""

import os
import re
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import adjoint_intensity_zernike as adj

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

parser = argparse.ArgumentParser()
parser.add_argument("parent_dir")
parser.add_argument("--j-filter", type=str, default=None,
                     help="Comma-separated j values to include (default: all j found)")
parser.add_argument("--xlim", type=str, default=None,
                     help="Comma-separated xmin,xmax in cm (default: auto)")
args = parser.parse_args()

PARENT   = args.parent_dir.rstrip("/")
RUNS_DIR = os.path.join(PARENT, "runs")
OUTPUT   = os.path.join(PARENT, "j_comparison.pdf")

# ── Collect runs sorted by j ───────────────────────────────────────────────────
runs = []
for d in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
    m = re.search(r"_[jn]_(\d+)_", os.path.basename(d))
    if m:
        runs.append((int(m.group(1)), d))

runs.sort(key=lambda x: x[0])

if args.j_filter is not None:
    j_keep = {int(x) for x in args.j_filter.split(",")}
    runs = [(j, d) for j, d in runs if j in j_keep]

if not runs:
    print(f"No runs found in {RUNS_DIR}")
    sys.exit(1)

# ── Load params from first run (physical params identical across all runs) ─────
first_run = runs[0][1]
param_file = [f for f in glob.glob(os.path.join(first_run, "*.txt")) if not f.endswith("time.txt")][0]
params = adj.initialize_variables_from_file(param_file)

f_targ = params["f_targ"]
f_init = params["f_init"]
wl     = params["wl"]
k0     = 2 * np.pi / wl
fn     = params["f_num"]
w0     = f_init / (2 * fn)
rmax   = params["r_maxf"] * w0
nr     = params["nr"]
dz     = params["dz"]
nzf    = params["nzf"]
L      = params["L"]

zF     = np.linspace(f_targ - dz, f_targ + L + dz, nzf)
z_cm   = (zF - f_targ) * 100

[r, k, T_R2K, T_K2R] = adj.DHTsetup(nr, rmax, 0)

phi_I      = np.exp(-((r / w0) ** 40))
zm_t       = adj.create_axiparabola(r, w0, L, f_targ)
ftf_t      = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)
It         = adj.calc_intensity(ftf_t)
maxIt      = np.max(It)
It_oa_norm = It[0, :] / maxIt

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

colors = cm.viridis(np.linspace(0.1, 0.9, len(runs)))

for color, (j_val, run_dir) in zip(colors, runs):
    int_oa = np.genfromtxt(os.path.join(run_dir, "int_oa.csv"), delimiter=",")
    final  = int_oa[-1] if int_oa.ndim > 1 else int_oa
    ax.plot(z_cm, final, color=color, label=f"j={j_val}")

ax.plot(z_cm, It_oa_norm, "k--", linewidth=2, label="Target")

ax.set_xlabel(r"$z - f$ (cm)", fontsize=14)
ax.set_ylabel("Normalized On-Axis Intensity", fontsize=14)
ax.set_title(f"Final On-Axis Intensity by Number of Zernike Modes ({PARENT})", fontsize=13)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)
if args.xlim is not None:
    xmin, xmax = (float(x) for x in args.xlim.split(","))
    ax.set_xlim(xmin, xmax)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
print(f"Saved: {OUTPUT}")
