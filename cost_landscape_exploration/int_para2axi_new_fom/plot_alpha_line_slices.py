#!/usr/bin/env python
"""
True full-dimensional cost slice from the initial guess to the exact
axiparabola target, evaluated with several different b penalty weights
overlaid on one plot. Irsq/IrIt/Itsq only depend on the mirror shape (not on
b), so they're computed once per point along the line and then combined
into FOM = T1 + b*(1-mu)**2 for each b -- no extra forward-model
evaluations needed per b.

Usage:
    python plot_alpha_line_slices.py <param_file> [out_prefix]

<param_file> describes the physical setup (f_targ, f_init, L, f_num, nr,
nzf, r_maxf, dz, wl, conic, delta, n). Its own "b" value is ignored --
see BS below. Output is <out_prefix>.pdf/.png (out_prefix defaults to
"alpha_line_slices"). Data is cached to <out_prefix>_init_target_line.csv
and reused on subsequent runs.
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import adjoint_intensity_newfom as adj

font = {"size": 15}
plt.rc("font", **font)
plt.rcParams["lines.linewidth"] = 2
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

SLICE_N = 200          # points evaluated along the line
T_RANGE = (-1, 2)      # 0 = init, 1 = target
BS = [0, 50, 100, 150, 200]
B_COLORS = ["tab:gray", "tab:blue", "tab:orange", "tab:red", "tab:purple"]


def create_axiparabola(r, R, L, f0):
    return R**2 / (4 * L) * np.log(1 + L / f0 * (r / R) ** 2)


def generate_optic(a_n, zernpol):
    zm = adj.generate_optic(a_n, zernpol)
    return zm - zm[0]


if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

param_file = sys.argv[1]
out_prefix = sys.argv[2] if len(sys.argv) >= 3 else "alpha_line_slices"

params = adj.initialize_variables_from_file(param_file)

f_targ = params["f_targ"]
ell = 0
f_init = params["f_init"]
wl = params["wl"]
k0 = 2 * np.pi / wl
fn = params["f_num"]
w0 = f_init / (2 * fn)
rmax = params["r_maxf"] * w0
nr = params["nr"]
wR = 2 / np.pi * fn * wl
L = params["L"]

dz = params["dz"]
nzf = params["nzf"]
zF = np.linspace(f_targ - dz, f_targ + L + dz, nzf)

num = params["n"]
delta = params["delta"]
power = 40

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])
phi_I = np.exp(-((r / w0) ** power))

rho = r / rmax
Z_fit = adj.calc_Z(rho, num)
Z_plus = adj.calc_Z_plus(Z_fit)
wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + L + dz, delta)
a_init = adj.create_initial_coeff(rmax, 2 * f_init, num)

zm_t = create_axiparabola(r, w0, L, f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)
It = adj.calc_intensity(ftf_t)
maxIt = np.max(It)
It = It / maxIt

a_target = Z_plus @ zm_t
print("a_target:", a_target)


def eval_parts(a_n):
    zm_r = generate_optic(a_n, Z_fit)
    ftf = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0, phase=prop_phase)
    Ir = adj.calc_intensity(ftf) / maxIt
    return adj.int_pairs(r, zF, wt, Ir, It)  # Irsq, IrIt, Itsq


csv_path = f"{out_prefix}_init_target_line.csv"
b_cols = [f"log10_cost_b{b}" for b in BS]

if os.path.exists(csv_path):
    print(f"Found existing line data at {csv_path} -- loading it instead of recomputing.")
    saved = np.genfromtxt(csv_path, delimiter=",", names=True)
    slice_t = saved["t"]
    log_costs = {b: saved[col] for b, col in zip(BS, b_cols)}
else:
    t_low, t_high = T_RANGE
    slice_t = np.linspace(t_low, t_high, SLICE_N)
    log_costs = {b: np.empty(SLICE_N) for b in BS}
    for i, t in enumerate(slice_t):
        a_n = a_init + t * (a_target - a_init)
        Irsq, IrIt, Itsq = eval_parts(a_n)
        mu = adj.calc_mu(IrIt, Itsq)
        for b in BS:
            cost = adj.FOM(Irsq, IrIt, Itsq, b)
            log_costs[b][i] = np.log10(max(cost, 1e-12))
        if i % 25 == 0:
            print(f"  {i + 1}/{SLICE_N} done")
    header = "t," + ",".join(b_cols)
    data = np.column_stack([slice_t] + [log_costs[b] for b in BS])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"Saved line data to {csv_path}")

fig, ax = plt.subplots(figsize=(8, 6))
for b, color in zip(BS, B_COLORS):
    ax.plot(slice_t, log_costs[b], color=color, label=rf"$b={b}$")

ax.xaxis.set_major_locator(MultipleLocator(1))
ax.axvline(0, color="black", lw=1.25, ls="--", alpha=0.8, zorder=10)
ax.axvline(1, color="black", lw=1.25, ls="--", alpha=0.8, zorder=10)
ax.set_xlabel(r"$t$  (0 = init, 1 = target)")
ax.set_ylabel("log10(cost)")
ax.set_title("Slice: init " + r"$\to$" + " target, by " + r"$b$")
ax.legend(loc="upper right", fontsize=12)

plt.tight_layout()
combined_out_path = f"{out_prefix}.pdf"
plt.savefig(combined_out_path)
combined_png_path = combined_out_path.rsplit(".", 1)[0] + ".png"
plt.savefig(combined_png_path, dpi=200)
plt.close()
print(f"Saved to {combined_out_path} and {combined_png_path}")
