#!/usr/bin/env python
"""
True full-dimensional cost slices between the three key points in Zernike
coefficient space -- initial guess, a completed run's final mirror, and the
exact axiparabola target. Unlike plot_cost_surface_pairs.py's contour panels
(which fix the third coefficient at the target's value), every point
plotted here is a real forward-propagation + cost evaluation with ALL
coefficients varying together along the true straight line between the two
endpoints -- an exact cost profile, not a 2D-slice approximation.

Usage:
    python plot_cost_line_slices.py <param_file> <final_run_dir> [out_prefix]

<param_file> describes the physical setup (f_targ, f_init, L, f_num, nr,
nzf, r_maxf, dz, wl, conic, delta, n, b). <final_run_dir> is a completed
Adam run's directory (its last zernike_coeff.csv row is used as the final
mirror). Output is <out_prefix>_line_slices.pdf/.png (out_prefix defaults
to "cost_line_slices"). Per-pair (t, cost) arrays are cached to
<out_prefix>_<from>_<to>_line.csv and reused on subsequent runs.
"""

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import adjoint_intensity_zernike as adj

font = {"size": 15}
plt.rc("font", **font)
plt.rcParams["lines.linewidth"] = 2
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

SLICE_N = 200  # points evaluated along each line

# Each entry: (from, to, line color, (t_low, t_high))
SLICE_DEFS = [
    ("init", "final", "tab:gray", (-1, 2)),
    ("init", "target", "tab:purple", (-0.5, 2)),
    ("final", "target", "tab:green", (-0.5, 2)),
]


def create_axiparabola(r, R, L, f0):
    return R**2 / (4 * L) * np.log(1 + L / f0 * (r / R) ** 2)


def generate_optic(a_n, zernpol):
    zm = adj.generate_optic(a_n, zernpol)
    return zm - zm[0]


if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

param_file = sys.argv[1]
final_run_dir = sys.argv[2]
out_prefix = sys.argv[3] if len(sys.argv) >= 4 else "cost_line_slices"

a_final = np.genfromtxt(os.path.join(final_run_dir, "zernike_coeff.csv"), delimiter=",")[-1]
print("a_final (last row of that run's zernike_coeff.csv):", a_final)

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


def evaluate(a_n):
    zm_r = generate_optic(a_n, Z_fit)
    ftf = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0, phase=prop_phase)
    Ir = adj.calc_intensity(ftf) / maxIt
    Irsq, IrIt, Itsq = adj.int_pairs(r, zF, wt, Ir, It)
    fom = adj.FOM(Irsq, IrIt, Itsq)
    cost = fom / Itsq
    return float(cost)


points = {"init": a_init, "final": a_final, "target": a_target}

entries = []
for from_name, to_name, color, (t_low, t_high) in SLICE_DEFS:
    csv_path = f"{out_prefix}_{from_name}_{to_name}_line.csv"
    print(f"--- slice: {from_name} -> {to_name} ---")

    if os.path.exists(csv_path):
        print(f"Found existing line data at {csv_path} -- loading it instead of recomputing.")
        saved = np.genfromtxt(csv_path, delimiter=",", names=True)
        slice_t = saved["t"]
        log_cost = saved["log10_cost"]
    else:
        p0, p1 = points[from_name], points[to_name]
        slice_t = np.linspace(t_low, t_high, SLICE_N)
        cost = np.array([evaluate(p0 + t * (p1 - p0)) for t in slice_t])
        log_cost = np.log10(np.clip(cost, 1e-12, None))
        np.savetxt(csv_path, np.column_stack([slice_t, log_cost]),
                   delimiter=",", header="t,log10_cost", comments="")
        print(f"Saved line data to {csv_path}")

    entries.append({
        "from_name": from_name, "to_name": to_name, "color": color,
        "slice_t": slice_t, "log_cost": log_cost,
    })

vmin = min(e["log_cost"].min() for e in entries)
vmax = max(e["log_cost"].max() for e in entries)

fig, axes = plt.subplots(1, len(SLICE_DEFS), figsize=(15, 5))

for ax, e in zip(axes, entries):
    from_name, to_name = e["from_name"], e["to_name"]
    ax.plot(e["slice_t"], e["log_cost"], color=e["color"])
    ax.set_ylim(vmin, vmax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.axvline(0, color="black", lw=1.25, ls="--", alpha=0.8, zorder=10)
    ax.axvline(1, color="black", lw=1.25, ls="--", alpha=0.8, zorder=10)
    ax.set_xlabel(rf"$t$  (0 = {from_name}, 1 = {to_name})")
    ax.set_ylabel("log10(cost)")
    ax.set_title(f"Slice: {from_name} " + r"$\to$" + f" {to_name}")

plt.tight_layout()
combined_out_path = f"{out_prefix}.pdf"
plt.savefig(combined_out_path)
combined_png_path = combined_out_path.rsplit(".", 1)[0] + ".png"
plt.savefig(combined_png_path, dpi=200)
plt.close()
print(f"Saved to {combined_out_path} and {combined_png_path}")
