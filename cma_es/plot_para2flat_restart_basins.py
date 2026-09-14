#!/usr/bin/env python
"""
Compares the best on-axis intensity profile (best_int_oa.csv) across the
distinct basins found by one CMA-ES restart run, alongside the axiparabola
starting guess and the target flat-top profile. One representative restart
is plotted per distinct best_cost value (restarts converging to the same
cost land in the same basin and produce near-identical curves, so plotting
all of them would just overplot). Axis conventions match plot.py's
plot_I_OA_over_iterations: (z-f) in cm on x, "arb. units" on y.

Usage:
    python plot_para2flat_restart_basins.py <run_dir>

<run_dir> is a completed restart run's top-level directory (the one
containing restart_0, restart_1, ..., and restarts_summary.txt) -- e.g.
para2flat_restarts_test/runs/int_para2flat_cma_..._restarts.
"""

import os
import re
import sys
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
import adjoint_intensity_zernike as adj

font = {"size": 15}
plt.rc("font", **font)
plt.rcParams["lines.linewidth"] = 2
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


def create_flat(r, z, fl, sigma, order):
    R = np.exp(-(((r**2) / (2 * sigma**2)) ** order))[:, np.newaxis]
    Z = np.exp(-((((z - fl - sigma) ** 2) / (sigma**2)) ** order))[np.newaxis, :]
    return R * Z


def find_params_file(run_dir):
    excluded = {"time.txt", "restarts_summary.txt"}
    candidates = [f for f in glob.glob(os.path.join(run_dir, "*.txt"))
                  if os.path.basename(f) not in excluded]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one params .txt in {run_dir}, found {candidates}")
    return candidates[0]


def read_summary(run_dir):
    # restarts_summary.txt: "restart\tseed\tgenerations\tstop_reason\tbest_cost[\t<== winner]"
    best_cost = {}
    winner_idx = None
    with open(os.path.join(run_dir, "restarts_summary.txt")) as f:
        next(f)  # header
        for line in f:
            fields = line.rstrip("\n").split("\t")
            idx = int(fields[0])
            best_cost[idx] = float(fields[4])
            if len(fields) > 5:
                winner_idx = idx
    return best_cost, winner_idx


if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

run_dir = sys.argv[1].rstrip("/")
params = adj.initialize_variables_from_file(find_params_file(run_dir))

f_targ, dz, L, nzf = params["f_targ"], params["dz"], params["L"], params["nzf"]
zF = np.linspace(f_targ - dz, f_targ + L + dz, nzf)

wl = params["wl"]
k0 = 2 * np.pi / wl
fn = params["f_num"]
w0 = params["f_init"] / (2 * fn)
rmax = params["r_maxf"] * w0
nr = params["nr"]
ell = 0

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
phi_I = np.exp(-((r / w0) ** 40))

# Same reference calculation main_int_para2flat_cma.py uses: It is the
# abstract flat-top target shape, and maxIt is the least-squares scalar that
# best matches the axiparabola's real on-axis intensity to that shape --
# it's what best_int_oa.csv was already divided by, so the axiparabola and
# target curves below have to go through the same normalization to land on
# the same scale as the restarts' saved curves.
It = create_flat(r, zF, f_targ, L / 2, 20)
zm_axi = adj.create_axiparabola(r, w0, L, f_targ)
ftf_axi = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_axi, k0)
It_axi = adj.calc_intensity(ftf_axi)
maxIt = np.sum(It_axi[0, :] * It[0, :]) / np.sum(It[0, :] ** 2)

best_cost, winner_idx = read_summary(run_dir)

# Group restarts by their (already-rounded, as saved) best_cost so restarts
# that landed in the same basin collapse to one representative curve.
by_cost = {}
for idx, cost in best_cost.items():
    by_cost.setdefault(f"{cost:.6e}", []).append(idx)

representatives = []
for cost_key, idxs in sorted(by_cost.items()):
    idxs_sorted = sorted(idxs)
    rep_idx = winner_idx if winner_idx in idxs_sorted else idxs_sorted[0]
    representatives.append((rep_idx, best_cost[rep_idx], idxs_sorted))

fig, ax = plt.subplots(figsize=(8, 5))
basin_colors = ["tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
x = (zF - f_targ) * 100

for basin_num, ((rep_idx, cost, member_idxs), color) in enumerate(zip(representatives, basin_colors), start=1):
    int_oa = np.genfromtxt(os.path.join(run_dir, "restart_%d" % rep_idx, "best_int_oa.csv"),
                            delimiter=",")
    is_winner = rep_idx == winner_idx
    label = "CMA-ES Basin %d (n=%d)%s\nF'=%.2e" % (
        basin_num, len(member_idxs), " *" if is_winner else "", cost)
    ax.plot(x, int_oa, color=color, lw=2.5, alpha=1.0, label=label)

ax.plot(x, It_axi[0, :] / maxIt, c="tab:red", linestyle="-.", linewidth=2.5,
        zorder=3, label="Axiparabola")
ax.plot(x, It[0, :], c="k", linestyle="--", linewidth=3, label=r"$I_T$")

ax.set_xlabel(r"$z-f$ (cm)")
ax.set_ylabel("On-axis Intensity (arb. units)")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_xlim(-2, 7)
ax.legend(loc="best", fontsize=9)
ax.set_title("Best On-Axis Intensity by Restart Basin")

plt.tight_layout()
out_path = os.path.join(run_dir, "restarts_int_oa_comparison.pdf")
plt.savefig(out_path)
plt.close()
print(f"Saved to {out_path}")
