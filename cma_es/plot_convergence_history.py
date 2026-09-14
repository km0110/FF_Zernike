#!/usr/bin/env python
"""
Convergence history for one or more completed CMA-ES runs: mu and true cost
(fom/Itsq + b*|1-mu|) plotted against generation, one line per run on
shared axes. Based on this codebase's existing plot_fom_mu (plot.py), but
overlaying multiple runs for direct comparison instead of one run per
figure, and plotting the actual penalized cost (not raw fom) since that's
what the optimizer is actually minimizing.

No forward-model evaluations are run by this script; it only reads what
each run already computed and saved (data.csv for Irsq/IrIt/Itsq, and that
run's own saved parameter file for b).

Usage:
    python plot_convergence_history.py <out_prefix> <run_dir> [<run_dir> ...]

Each <run_dir> is a completed CMA-ES run's directory. Output is
<out_prefix>.pdf/.png.
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

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def display_name(run_dir):
    # Runs named like "..._trial3" show up as "Trial 3"; anything else
    # falls back to the raw directory basename.
    basename = os.path.basename(run_dir.rstrip("/"))
    match = re.search(r"trial(\d+)$", basename)
    return f"Trial {match.group(1)}" if match else basename


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

basin_of = basin_labels(run_dirs)

runs = []
for run_dir in run_dirs:
    _run_params = adj.initialize_variables_from_file(find_run_params_file(run_dir))
    b = _run_params.get("b", _run_params.get("lam", 1.0))

    data = np.genfromtxt(os.path.join(run_dir, "data.csv"), delimiter=",")
    Irsq, IrIt, Itsq = data[:, 0], data[:, 1], data[:, 2]
    fom = adj.FOM(Irsq, IrIt, Itsq)
    mu = adj.calc_mu(IrIt, Itsq)
    true_cost = fom / Itsq + b * np.abs(1 - mu)

    basin_num, n = basin_of[os.path.abspath(run_dir.rstrip("/"))]
    label = "Basin %d (n=%d)\n$F'$=%.2e" % (basin_num, n, true_cost[-1])

    runs.append({"name": display_name(run_dir), "label": label, "mu": mu, "cost": true_cost,
                 "basin_num": basin_num})
    print(f"{display_name(run_dir)}: {len(mu)} generations, b={b}, "
          f"cost {true_cost[0]:.3e} -> {true_cost[-1]:.3e}")

# Plot in basin order (1, 2, 3, ...) rather than command-line argument order,
# so the legend always lists the largest/lowest-cost basin first regardless
# of which run_dir happened to be passed first.
runs.sort(key=lambda run: run["basin_num"])

fig, (ax_mu, ax_cost) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

for run, color in zip(runs, COLORS):
    generations = np.arange(len(run["mu"]))
    ax_mu.plot(generations, run["mu"], color=color, label=run["label"])
    ax_cost.plot(generations, run["cost"], color=color, label=run["label"])

ax_mu.axhline(1, color="black", lw=1, ls="--", alpha=0.6, zorder=0)
ax_mu.set_ylabel(r"$\mu$")
ax_mu.legend(loc="best", fontsize=10)

ax_cost.set_yscale("log")
ax_cost.set_xlabel("Generation")
ax_cost.set_ylabel(r"$F'$")

plt.tight_layout()
combined_out_path = f"{out_prefix}.pdf"
plt.savefig(combined_out_path)
combined_png_path = combined_out_path.rsplit(".", 1)[0] + ".png"
plt.savefig(combined_png_path, dpi=200)
plt.close()
print(f"Saved to {combined_out_path} and {combined_png_path}")
