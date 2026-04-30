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
import adjoint as adj
import plot as plots

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
params = adj.initialize_variables_from_file(param_file)

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
print(L)
zF = np.linspace(f_targ - dz, f_targ + L + dz, nzf)
fl = len(zF) // 2

lr = params["lr"]
skip = params["skip"]

# Hankel transform arrays
[r, k] = adj.DHTarrays(nr, rmax, ell)
[T_R2K, T_K2R] = adj.DHTtransformMatrix(nr, rmax, ell)

power = 40 if 'axi' in path else 2
# Target field and intensity (fast recompute — no disk I/O needed)
phi_I = np.exp(-((r / w0) ** power))
zm_t = adj.create_axiparabola(r, w0, L, f_targ) if 'axi' in path else adj.create_starting_optic(r, 2 * f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t)
It = adj.calc_intensity(ftf_t)

# ── Load CSV data ──────────────────────────────────────────────────────────────
is_int = 'int' in os.path.basename(path)

if is_int:
    data_names = [
        "gradient.csv",
        "mirror.csv",
        "source.csv",
        "int_oa.csv",
        "int_f.csv",
        "psi.csv",
    ]
    filenames = [os.path.join(path, s) for s in data_names]
    gr, mr, src, ir_oa, ir_f, psis = [
        np.genfromtxt(f, delimiter=",") for f in filenames
    ]
    phr_oa, phr_f = None, None
else:
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

# Ensure per-iteration subdirectories exist (safe to call on existing dirs)
subdirs = ["I_OA", "I_f", "g", "mirror", "psi_S"]
if not is_int:
    subdirs += ["pr_f", "pr_oa"]
for subdir in subdirs:
    os.makedirs(os.path.join(path, subdir), exist_ok=True)

rr, rt, tt = np.genfromtxt(path + "/data.csv", delimiter=",", unpack=True)
figom = adj.FOM(rr, rt, tt)
mu_val = adj.calc_mu(rt, tt)
plots.plot_fom_mu(path, figom, mu_val)
plots.plot_percent_change_mu(path, mu_val)
# ── Generate plots ─────────────────────────────────────────────────────────────
print("Generating plots...")
pstart = time.time()
n_steps = len(gr)  # number of completed iterations
st=[100,500,4999]
# st=None
plots.plot_total_change(path, r, w0, gr, lr, wl, n_steps)
plots.plot_gradient(path, gr, r, w0)
# plot_max_grad(path, gr)
# plot_integrals(path, pr2, rt, pt2)
plots.plot_mirror_comp(path, r, w0, zm_t, mr, do_shift=True)
# plot_mirror_comp_each(path, r, w0, zm_t, mr, skip)
if not is_int:
    plots.plot_phi_f_over_iterations(path, phr_f, r, w0, np.real(ftf_t), fl, steps=st)
    plots.plot_phi_OA_over_iterations(path, phr_oa, zF, f_targ, np.real(ftf_t), steps=st)
    # plot_phi_f(path, phr_f, r, w0, ftf_t, fl, skip)
    # plot_phi_OA(path, phr_oa, zF, f_targ, ftf_t, skip)

# plot_I_OA(path, ir_oa, zF, f_targ, It[0, :], skip)
plots.plot_I_OA_over_iterations(path, ir_oa, zF, f_targ, It[0, :],steps=st)
# plot_I_f(path, ir_f, r, w0, It[:, fl], skip)
plots.plot_I_f_over_iterations(path, ir_f, r, w0, It[:, fl],steps=st)
plots.plot_step0_intensity(path, ir_oa, ir_f, zF, f_targ, r, w0, It, fl)
if not is_int:
    plots.plot_step0_phi(path, phr_oa, phr_f, zF, f_targ, r, w0, np.real(ftf_t), fl)
# plot_psi_and_source(path, psis, r, w0, src, zF, f_targ, skip)

pend = time.time()
elapsed = str(timedelta(seconds=pend - pstart))
print(f"Done. Plotting time: {elapsed}")

with open(os.path.join(path, "time.txt"), "a") as f:
    f.write(f"\nPlotting time: {elapsed}")
