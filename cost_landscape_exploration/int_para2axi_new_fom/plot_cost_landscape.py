#!/usr/bin/env python
"""
Cost-landscape slice: linearly interpolates between two Zernike-coefficient
solutions and evaluates FOM/Itsq + b*(1-mu)^2 along that line, for a
sweep of b values, to check whether they sit in separate basins (a ridge
between them) or the same one, and how much b affects that ridge.

Usage:
    python plot_cost_landscape.py <param_file> <coeff_csv_a> <coeff_csv_b> [out.pdf]

<param_file> just needs to describe the same physical setup the two
solutions were optimized under (f_targ, f_init, L, f_num, nr, nzf, r_maxf,
dz, wl, conic, delta, n) -- any of the matching run's saved .txt files
works, including the copy left inside a completed run's output folder.
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

BS = [0, 0.5, 1, 2, 5, 10, 20]


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
zF = np.linspace(f_targ - dz, f_targ + L + dz, nzf)

num = params["n"]
delta = params["delta"]
power = 40

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])

phi_I = np.exp(-((r / w0) ** power))
zm_t = create_axiparabola(r, w0, L, f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)

It = adj.calc_intensity(ftf_t)
maxIt = np.max(It)
It = It / maxIt

rho = r / rmax
Z_fit = adj.calc_Z(rho, num)
wt = adj.weight(r, wR, zF, f_targ - dz, f_targ + L + dz, delta)


def evaluate(a_n):
    # Only the raw ingredients are computed here (one forward-model pass per
    # t) -- cost for each b is assembled afterward from these, since
    # fom_over_itsq and mu don't depend on b at all.
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

cmap = mpl.colormaps["viridis"]
BS_MAIN = [x for x in BS if x != 20]

fig, ax1 = plt.subplots(figsize=(16, 9))

for b in BS_MAIN:
    cost = fom_over_itsq + b * (1 - mus) ** 2
    color = cmap(BS.index(b) / max(len(BS) - 1, 1))
    # cost >= 0 always in this branch (FOM and the penalty are both
    # non-negative for b >= 0), so a log scale is safe here -- and it's
    # what actually reveals the sharp, many-orders-of-magnitude structure
    # near each solution that a linear scale flattens into "touches zero".
    ax1.plot(ts, np.clip(cost, 1e-12, None), color=color, label=rf"$b={b}$")
    print(f"[b={b:>4}] cost(A)={cost[np.argmin(np.abs(ts - 0))]:.4e}  "
          f"cost(B)={cost[np.argmin(np.abs(ts - 1))]:.4e}  "
          f"max={cost.max():.4e} at t={ts[np.argmax(cost)]:.3f}")

ax1.set_yscale("log")
ax1.set_xlabel(r"$t$")
ax1.set_ylabel("Cost")
ax1.axvline(0, color="k", lw=1.5, ls="--", alpha=0.4)
ax1.axvline(1, color="k", lw=1.5, ls="--", alpha=0.4)
ax1.set_title(r"$\mathrm{FOM}/I_{tsq} + b(1-\mu)^2$ along the path between two solutions")

ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel(r"$\mu$", color=color2)
ax2.plot(ts, mus, color=color2, ls=":")
ax2.axhline(1, color=color2, lw=1.5, ls="--", alpha=0.5)
ax2.tick_params(axis="y", labelcolor=color2)

ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=22, frameon=False)

plt.tight_layout()
plt.savefig(out_path, bbox_inches="tight")
png_path = out_path.rsplit(".", 1)[0] + ".png"
plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {out_path} and {png_path}")

# Isolated penalty term alone (no FOM/Itsq), same b sweep and color
# scheme as above, so it's directly comparable to the combined-cost plot.
fig2, ax3 = plt.subplots()
ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$b(1-\mu)^2$")

for i, b in enumerate(BS):
    penalty = b * (1 - mus) ** 2
    color = cmap(i / max(len(BS) - 1, 1))
    ax3.plot(ts, penalty, color=color, label=rf"$b={b}$")

ax3.axvline(0, color="k", lw=1.5, ls="--", alpha=0.4)
ax3.axvline(1, color="k", lw=1.5, ls="--", alpha=0.4)
ax3.set_title(r"$b(1-\mu)^2$ alone along the path between two solutions")
ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=22, frameon=False)

plt.tight_layout()
penalty_out_path = out_path.rsplit(".", 1)[0] + "_penalty_only.pdf"
plt.savefig(penalty_out_path, bbox_inches="tight")
penalty_png_path = penalty_out_path.rsplit(".", 1)[0] + ".png"
plt.savefig(penalty_png_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved to {penalty_out_path} and {penalty_png_path}")
