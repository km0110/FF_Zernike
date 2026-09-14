#!/usr/bin/env python
"""
Cost-landscape slice: linearly interpolates between two Zernike-coefficient
solutions and evaluates the field FOM/ptsq = 0.5*(prsq - prpt^2/ptsq)/ptsq
along that line (target = axiparabola), to check whether they sit in
separate basins (a ridge between them) or the same one. Normalized by ptsq
for consistency with the intensity scripts' FOM/Itsq.

Usage:
    python plot_cost_landscape.py <param_file> <coeff_csv_a> <coeff_csv_b> [out.pdf]

<param_file> just needs to describe the same physical setup the two
solutions were optimized under (f_targ, f_init, L, f_num, nr, nzf, r_maxf,
dz, wl, delta, n) -- any of the matching run's saved .txt files works,
including the copy left inside a completed run's output folder.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import adjoint_field_zernike as adj

font = {"size": 30}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 4
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


def create_axiparabola(r, R, L, f0, taylor=False):
    if taylor:
        sag = r**2 / (4 * f0) * (1 - L / (2 * f0) * (r / R) ** 2)
    else:
        sag = R**2 / (4 * L) * np.log(1 + L / f0 * (r / R) ** 2)
    return sag


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
L = params["L"]

dz = params["dz"]
nzf = params["nzf"]
zi = f_targ - dz
zf = f_targ + L + dz
zF = np.linspace(zi, zf, nzf)

num = params["n"]
delta = params["delta"]
power = 40

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])

phi_I = np.exp(-((r / w0) ** power))
zm_t = create_axiparabola(r, w0, L, f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)

rho = r / rmax
Z_fit = adj.calc_Z(rho, num)


def evaluate(a_n):
    zm_r = adj.generate_optic(a_n, Z_fit)
    ftf_r = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_r, k0, phase=prop_phase)
    pR, pT = adj.weighted_phi(r, wR, zF, zi, zf, ftf_r, ftf_t, delta)
    prsq, prpt, ptsq = adj.int_pairs(r, zF, pR, pT, ftf_r, ftf_t)
    fom = adj.FOM(prsq, prpt, ptsq)
    mu = adj.calc_mu(prpt, ptsq)
    return float(fom / ptsq), float(mu)


a_a = np.genfromtxt(coeff_a_path, delimiter=",")
a_b = np.genfromtxt(coeff_b_path, delimiter=",")

# Overshoot slightly past both endpoints to see whether the landscape keeps
# improving beyond either known solution, not just between them.
ts = np.linspace(-0.15, 1.15, 131)
fom_over_ptsq = []
mus = []
for t in ts:
    a_t = (1 - t) * a_a + t * a_b
    fom, mu = evaluate(a_t)
    fom_over_ptsq.append(fom)
    mus.append(mu)

fom_over_ptsq = np.array(fom_over_ptsq)
mus = np.array(mus)

print(f"FOM/ptsq(A)={fom_over_ptsq[np.argmin(np.abs(ts - 0))]:.4e}  "
      f"FOM/ptsq(B)={fom_over_ptsq[np.argmin(np.abs(ts - 1))]:.4e}  "
      f"max={fom_over_ptsq.max():.4e} at t={ts[np.argmax(fom_over_ptsq)]:.3f}")

fig, ax1 = plt.subplots()
ax1.plot(ts, fom_over_ptsq, color="tab:blue")
ax1.set_xlabel(r"$t$")
ax1.set_ylabel(r"$\mathrm{FOM}/p_{tt}$")
ax1.axvline(0, color="k", lw=1.5, ls="--", alpha=0.4)
ax1.axvline(1, color="k", lw=1.5, ls="--", alpha=0.4)
ymin, ymax = ax1.get_ylim()
label_y = ymin + 0.06 * (ymax - ymin)
ax1.text(0, label_y, "Solution $A$", ha="center", va="bottom", fontsize=22)
ax1.text(1, label_y, "Solution $B$", ha="center", va="bottom", fontsize=22)
ax1.set_title(r"Field FOM$/p_{tt}$ along the path between two solutions (para2axi)")

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
