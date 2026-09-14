#!/usr/bin/env python
"""
Cost-landscape slice: linearly interpolates between two Zernike-coefficient
solutions and evaluates the plain intensity FOM/Itsq along that line (target
= on-axis parabola), to check whether they sit in separate basins (a ridge
between them) or the same one. No b*(1-mu)^2 penalty -- that term was
built for int_para2axi, where mu has a known degenerate-minimum problem tied
to the axiparabola target; main_int_para2para_adam.py doesn't use it, so
this landscape sticks to what the optimizer actually minimizes.

Usage:
    python plot_cost_landscape.py <param_file> <coeff_csv_a> <coeff_csv_b> [out.pdf]

<param_file> just needs to describe the same physical setup the two
solutions were optimized under (f_targ, f_init, f_num, nr, nzf, r_maxf, dz,
wl, delta, n) -- any of the matching run's saved .txt files works, including
the copy left inside a completed run's output folder.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import adjoint_intensity_zernike as adj

font = {"size": 30}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 4
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

if len(sys.argv) not in (4, 5):
    print(__doc__)
    sys.exit(1)

param_file = sys.argv[1]
coeff_a_path = sys.argv[2]
coeff_b_path = sys.argv[3]
out_path = sys.argv[4] if len(sys.argv) == 5 else "cost_landscape.pdf"

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

dz = params["dz"]
nzf = params["nzf"]
zi = f_targ - dz
zf = f_targ + dz
zF = np.linspace(zi, zf, nzf)

num = params["n"]
delta = params["delta"]

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])

phi_I = np.exp(-((r / w0) ** 2))
zm_t = adj.create_starting_optic(r, 2 * f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)

It = adj.calc_intensity(ftf_t)
maxIt = np.max(It)
It = It / maxIt

rho = r / rmax
Z_fit = adj.calc_Z(rho, num)
wt = adj.weight(r, wR, zF, zi, zf, delta)


def evaluate(a_n):
    zm_r = adj.generate_optic(a_n, Z_fit)
    ftf = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0, phase=prop_phase)
    Ir = adj.calc_intensity(ftf) / maxIt
    Irsq, IrIt, Itsq = adj.int_pairs(r, zF, wt, Ir, It)
    fom = adj.FOM(Irsq, IrIt, Itsq)
    mu = adj.calc_mu(IrIt, Itsq)
    return float(fom / Itsq), float(mu)


a_a = np.genfromtxt(coeff_a_path, delimiter=",")
a_b = np.genfromtxt(coeff_b_path, delimiter=",")

# Overshoot slightly past both endpoints to see whether the landscape keeps
# improving beyond either known solution, not just between them.
ts = np.linspace(-0.15, 1.15, 131)
fom_over_itsq = []
mus = []
for t in ts:
    a_t = (1 - t) * a_a + t * a_b
    fom_it, mu = evaluate(a_t)
    fom_over_itsq.append(fom_it)
    mus.append(mu)

fom_over_itsq = np.array(fom_over_itsq)
mus = np.array(mus)

print(f"FOM/Itsq(A)={fom_over_itsq[np.argmin(np.abs(ts - 0))]:.4e}  "
      f"FOM/Itsq(B)={fom_over_itsq[np.argmin(np.abs(ts - 1))]:.4e}  "
      f"max={fom_over_itsq.max():.4e} at t={ts[np.argmax(fom_over_itsq)]:.3f}")

fig, ax1 = plt.subplots()
ax1.plot(ts, fom_over_itsq, color="tab:blue")
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$\mathrm{FOM}/I_{tsq}$")
ax1.axvline(0, color="k", lw=1.5, ls="--", alpha=0.4)
ax1.axvline(1, color="k", lw=1.5, ls="--", alpha=0.4)
ymin, ymax = ax1.get_ylim()
label_y = ymin + 0.06 * (ymax - ymin)
ax1.text(0, label_y, "Solution $A$", ha="center", va="bottom", fontsize=22)
ax1.text(1, label_y, "Solution $B$", ha="center", va="bottom", fontsize=22)
ax1.set_title(r"$\mathrm{FOM}/I_{tsq}$ along the path between two solutions (int para2para)")

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel(r"$\mu$", color=color2)
ax2.plot(ts, mus, color=color2, ls=":")
ax2.axhline(1, color=color2, lw=1.5, ls="--", alpha=0.5)
ax2.tick_params(axis="y", labelcolor=color2)

plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
png_path = out_path.rsplit(".", 1)[0] + ".png"
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {out_path} and {png_path}")
