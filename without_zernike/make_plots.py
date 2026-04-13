#!/usr/bin/env python3
"""
make_plots.py — Generate plots from a completed simulation run.

Usage: python make_plots.py <path to run directory>

The run directory must contain a parameter file (*.txt, not time.txt)
written there by main.py, plus all the CSV data files.
"""

import sys
import os
import glob
import time
import numpy as np
from datetime import timedelta
from adjoint import *
from plot import *

if len(sys.argv) != 2:
    print("Usage: python make_plots.py <path to run directory>")
    sys.exit(1)

path = sys.argv[1].rstrip("/")

if not os.path.isdir(path):
    print(f"Directory not found: {path}")
    sys.exit(1)

# Locate the parameter file saved by main.py
param_files = [
    f for f in glob.glob(os.path.join(path, "*.txt")) if not f.endswith("time.txt")
]
if not param_files:
    print(f"No parameter file found in {path}")
    sys.exit(1)
param_file = param_files[0]

# ── Reconstruct parameters and derived quantities ──────────────────────────────
params = initialize_variables_from_file(param_file)

f_targ = params["f_targ"]
f_init = params["f_init"]
ell = 0
wl = params["wl"]
k0 = 2 * np.pi / wl
fn = params["f_num"]
w0 = f_init / (2 * fn)
rmax = params["r_maxf"] * w0
nr = params["nr"]

dz = params["dz"]
nzf = params["nzf"]
L = params.get("L", 0)
zF = np.linspace(f_targ - dz, f_targ + L + dz, nzf)
fl = len(zF) // 2

lr = params["lr"]
skip = params["skip"]

# Hankel transform arrays
[r, k] = DHTarrays(nr, rmax, ell)
[T_R2K, T_K2R] = DHTtransformMatrix(nr, rmax, ell)

# Target field and intensity (fast recompute — no disk I/O needed)
phi_I = np.exp(-((r / w0) ** 2))
zm_t = create_starting_optic(r, 2 * f_targ)
ftf_t = forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t)
It = calc_intensity(ftf_t)

# ── Load CSV data ──────────────────────────────────────────────────────────────
data_names = [
    "phi_r_OA.csv",
    "phi_r_focus.csv",
    "gradient.csv",
    "mirror.csv",
    "source.csv",
    "int_oa.csv",
    "int_f.csv",
    "psi.csv",
]
filenames = [os.path.join(path, s) for s in data_names]
phr_oa, phr_f, gr, mr, src, ir_oa, ir_f, psis = [
    np.genfromtxt(f, delimiter=",") for f in filenames
]
pr2, rt, pt2 = np.genfromtxt(os.path.join(path, "data.csv"), delimiter=",", unpack=True)
n_steps = len(gr)  # number of completed iterations

# Ensure per-iteration subdirectories exist (safe to call on existing dirs)
for subdir in ["pr_f", "pr_oa", "I_OA", "I_f", "g", "mirror", "psi_S"]:
    os.makedirs(os.path.join(path, subdir), exist_ok=True)

# ── Generate plots ─────────────────────────────────────────────────────────────
print("Generating plots...")
pstart = time.time()

plot_total_change(path, r, w0, gr, lr, wl, n_steps)
# plot_max_grad(path, gr)
# plot_integrals(path, pr2, rt, pt2)
plot_mirror_comp(path, r, w0, zm_t, mr)
plot_mirror_comp(path, r, w0, zm_t, mr, do_shift=True)
# plot_mirror_comp_each(path, r, w0, zm_t, mr, skip)
# plot_phi_f(path, phr_f, r, w0, ftf_t, fl, skip)
plot_phi_f_over_iterations(path, phr_f, r, w0, ftf_t, fl)
plot_phi_OA(path, phr_oa, zF, f_targ, ftf_t, skip)
plot_phi_OA_over_iterations(path, phr_oa, zF, f_targ, ftf_t)
plot_I_OA(path, ir_oa, zF, f_targ, It[0, :], skip)
plot_I_OA_over_iterations(path, ir_oa, zF, f_targ, It[0, :])
plot_I_f(path, ir_f, r, w0, It[:, fl], skip)
plot_I_f_over_iterations(path, ir_f, r, w0, It[:, fl])
plot_psi_and_source(path, psis, r, w0, src, zF, f_targ, skip)

pend = time.time()
elapsed = str(timedelta(seconds=pend - pstart))
print(f"Done. Plotting time: {elapsed}")

with open(os.path.join(path, "time.txt"), "a") as f:
    f.write(f"\nPlotting time: {elapsed}")
