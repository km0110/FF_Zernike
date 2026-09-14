#!/usr/bin/env python3
"""
plot_adam_vs_newfom_n4.py — Compare the final on-axis intensity (I_OA) from
the "adam" and "adam_newfom" int_para2axi optimizers for n=4 Zernike modes,
overlaid against the target profile.

Usage: python plot_adam_vs_newfom_n4.py

Can be run from anywhere; it chdirs into the repo root (needs
common/adjoint_intensity_zernike.py and common/BesselZeroes0.csv).
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
import adjoint_intensity_zernike as adj

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

RUNS = [
    (r"FOM", "adam/int_para2axi_n_f20/runs/int_para2axi_adam_d_lr_1e-08_ft_0.5_fi_0.51_n_4_nr_2001_rmx_1.3_nzf_1001_dz_0.05_L_0.05_st_12601_sk_10_fn_20_eps1e-6_tol_1e-11"),
    (r"FOM/$I_T^2$ + $b (1-\mu)^2$", "adam_newfom/int_para2axi_n_f20/runs/int_para2axi_adam_d_ft_0.5_fi_0.51_n_4_nr_2001_rmx_1.3_nzf_1001_dz_0.05_L_0.05_st_5000_sk_10_fn_20_al_1.0"),
]

OUTPUT = "comparison_plots/adam_vs_newfom_n4_int_oa.pdf"

# ── Load params (physical params are identical between the two runs) ──────────
first_run_dir = RUNS[0][1]
param_file = [f for f in glob.glob(os.path.join(first_run_dir, "*.txt")) if not f.endswith("time.txt")][0]
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

zF   = np.linspace(f_targ - dz, f_targ + L + dz, nzf)
z_cm = (zF - f_targ) * 100

[r, k, T_R2K, T_K2R] = adj.DHTsetup(nr, rmax, 0)

phi_I      = np.exp(-((r / w0) ** 40))
zm_t       = adj.create_axiparabola(r, w0, L, f_targ)
ftf_t      = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)
It         = adj.calc_intensity(ftf_t)
It_oa_norm = It[0, :] / np.max(It)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

colors      = ["tab:blue", "tab:red"]
linestyles  = ["-", ":"]
linewidths  = [3, 4]

for color, linestyle, linewidth, (label, run_dir) in zip(colors, linestyles, linewidths, RUNS):
    int_oa = np.genfromtxt(os.path.join(run_dir, "int_oa.csv"), delimiter=",")
    final  = int_oa[-1] if int_oa.ndim > 1 else int_oa
    ax.plot(z_cm, final, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

ax.plot(z_cm, It_oa_norm, "k--", linewidth=3, alpha=0.6, label="Target")

ax.set_xlabel(r"$z - f$ (cm)", fontsize=20)
ax.set_ylabel("Normalized On-Axis Intensity", fontsize=20)
ax.legend(fontsize=15)
ax.set_xlim(-2, 7)
ax.tick_params(labelsize=20)

fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150)
print(f"Saved: {OUTPUT}")
