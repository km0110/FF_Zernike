#!/usr/bin/env python
"""
Final on-axis intensity profile (best_int_oa.csv) for one or more completed
CMA-ES runs, overlaid on shared axes alongside the para2axi target profile
(the axiparabola's own on-axis intensity). Companion to
plot_convergence_history.py: pass the same run_dirs to compare cost-vs-
generation and final-shape side by side for the same set of runs/basins.

No forward-model evaluations are run against each run's own optimized
mirror -- best_int_oa.csv is read as-is. The target curve is recomputed
once (physics setup is shared across runs in a batch, e.g. a reliability
study -- see tally_reliability_batch.py) since it isn't saved per-run.

Usage:
    python plot_basins_int_oa.py <out_prefix> <run_dir> [<run_dir> ...]

Each <run_dir> is a completed CMA-ES run's directory (contains
best_int_oa.csv and its own saved params .txt). Output is
<out_prefix>.pdf/.png.
"""

import os
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

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def find_run_params_file(run_dir):
    excluded = {"time.txt", "restarts_summary.txt"}
    candidates = [f for f in glob.glob(os.path.join(run_dir, "*.txt"))
                  if os.path.basename(f) not in excluded]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one params .txt in {run_dir}, found {candidates}")
    return candidates[0]


def final_true_cost(run_dir):
    _run_params = adj.initialize_variables_from_file(find_run_params_file(run_dir))
    b = _run_params.get("b", _run_params.get("lam", 1.0))
    data = np.genfromtxt(os.path.join(run_dir, "data.csv"), delimiter=",")
    Irsq, IrIt, Itsq = data[:, 0], data[:, 1], data[:, 2]
    fom = adj.FOM(Irsq, IrIt, Itsq)
    mu = adj.calc_mu(IrIt, Itsq)
    return float((fom / Itsq + b * np.abs(1 - mu))[-1])


def basin_labels(run_dirs, gap_threshold=1.0):
    # Basin membership/size is a property of the whole batch a run came
    # from (e.g. all 100 dirs in a reliability study's runs/), not just the
    # handful of representative run_dirs passed on the command line -- so
    # for each distinct parent "runs/" directory among run_dirs, pull in
    # every completed sibling run and auto-cluster by final cost (same
    # biggest-log10-gap approach as tally_reliability_batch.py) to get an
    # accurate n= count. Basins are numbered 1..k in ascending cost (lowest
    # cost = Basin 1), matching plot_para2flat_restart_basins.py's convention.
    basin_of = {}
    for runs_root in {os.path.dirname(os.path.abspath(rd.rstrip("/"))) for rd in run_dirs}:
        siblings = [d for d in sorted(glob.glob(os.path.join(runs_root, "*")))
                    if os.path.isfile(os.path.join(d, "data.csv"))]
        costs = np.array([final_true_cost(d) for d in siblings])
        log_costs = np.log10(np.clip(costs, 1e-15, None))
        order = np.argsort(log_costs)

        groups = [[order[0]]]
        for i in range(1, len(order)):
            if log_costs[order[i]] - log_costs[order[i - 1]] > gap_threshold:
                groups.append([])
            groups[-1].append(order[i])

        for basin_num, group in enumerate(groups, start=1):
            for idx in group:
                basin_of[os.path.abspath(siblings[idx])] = (basin_num, len(group))
    return basin_of


if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

out_prefix = sys.argv[1]
run_dirs = sys.argv[2:]

# Physics setup -- read from the first run's own saved param file. Matches
# tally_reliability_batch.py's assumption that within one batch these are
# all identical except seed_offset.
params = adj.initialize_variables_from_file(find_run_params_file(run_dirs[0]))

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

# Target = the axiparabola's own on-axis intensity (int_para2axi optimizes
# a Zernike-perturbed parabola to reproduce this), normalized the same way
# main_int_para2axi_cma.py normalizes it -- best_int_oa.csv was saved
# through that same maxIt, so both curves land on the same scale.
zm_t = adj.create_axiparabola(r, w0, L, f_targ)
ftf_t = adj.forward_propagation(k, phi_I, T_R2K, T_K2R, zF, zm_t, k0)
It = adj.calc_intensity(ftf_t)
maxIt = np.max(It)
target_int_oa = It[0, :] / maxIt

basin_of = basin_labels(run_dirs)

runs = []
for run_dir in run_dirs:
    true_cost = final_true_cost(run_dir)
    basin_num, n = basin_of[os.path.abspath(run_dir.rstrip("/"))]
    label = "Basin %d (n=%d)\n$F'$=%.2e" % (basin_num, n, true_cost)

    int_oa = np.genfromtxt(os.path.join(run_dir, "best_int_oa.csv"), delimiter=",")
    runs.append({"label": label, "int_oa": int_oa, "basin_num": basin_num})

# Plot in basin order (1, 2, 3, ...) rather than command-line argument order,
# so the legend always lists the largest/lowest-cost basin first regardless
# of which run_dir happened to be passed first.
runs.sort(key=lambda run: run["basin_num"])

fig, ax = plt.subplots(figsize=(8, 5))
x = (zF - f_targ) * 100

for run, color in zip(runs, COLORS):
    ax.plot(x, run["int_oa"], color=color, lw=2.5, label=run["label"])

ax.plot(x, target_int_oa, c="k", linestyle="--", linewidth=3, label=r"$I_T$ (axiparabola)")

ax.set_xlabel(r"$z-f$ (cm)")
ax.set_ylabel("On-axis Intensity (arb. units)")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_xlim(-2, 7)
ax.set_ylim(0, 1.5)
ax.legend(loc="best", fontsize=9)

plt.tight_layout()
out_path_pdf = f"{out_prefix}.pdf"
plt.savefig(out_path_pdf)
out_path_png = out_path_pdf.rsplit(".", 1)[0] + ".png"
plt.savefig(out_path_png, dpi=200)
plt.close()
print(f"Saved to {out_path_pdf} and {out_path_png}")
