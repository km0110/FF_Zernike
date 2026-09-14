#!/usr/bin/env python
"""
Learning rate scan: runs a main optimization script over a configurable range
of learning rates using a base parameter file, then produces a summary plot of
FOM and mu vs iterations for each lr so you can identify the best step size.

Shared across all experiment directories (adam, adam_newfom, adam_sgdr,
rop, sgdr, fixed_lr) -- this file lives outside all of them in common/.
Run it from inside the experiment directory (same as before) so the main
script resolves correctly.

Which FOM module to use is genuinely different per directory (Zernike-standard,
Zernike+b-penalty "newfom", or without_zernike's direct-pixel parameterization) and
must be given explicitly via --fom-module.

Usage:
    python lr_scan.py <main_script.py> <base_param_file> [scan_dir]
        --fom-module {zernike,zernike_newfom,local}
        [--lr-min X] [--lr-max Y] [--n-lr N] [--lr-list "a,b,c"]
        [--scan-steps S]

Example:
    python lr_scan.py main_int_para2axi.py int_para2axi/runs/.../params.txt int_para2axi/lr_scan

If scan_dir is omitted, defaults to "lr_scan".

By default the learning rates scanned are np.geomspace(1e-12, 1e-7, 6); pass
--lr-min/--lr-max/--n-lr to use a different geometric range, or --lr-list for
an explicit comma-separated list (overrides --lr-min/--lr-max/--n-lr).
--scan-steps overrides the base param file's steps value for each scan run
(default 200). Summary plot is saved to <scan_dir>/lr_scan_summary.pdf
"""

import sys
import os
import math
import shutil
import inspect
import argparse
import subprocess
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 3
plt.rc("font", size=26)


def write_param_file(fpath, params):
    with open(fpath, "w") as f:
        for name, value in params.items():
            f.write(f"{name} = {repr(value)}\n")


def load_run_data(run_dir, b=1.0):
    data_csv = os.path.join(run_dir, "data.csv")
    if not os.path.exists(data_csv):
        return None
    rr, rt, tt = np.genfromtxt(data_csv, delimiter=",", unpack=True)
    rr, rt, tt = np.atleast_1d(rr), np.atleast_1d(rt), np.atleast_1d(tt)
    # adam_newfom's adjoint_intensity.FOM takes an extra b weighting term;
    # other directories' FOM doesn't accept it.
    fom_kwargs = {"b": b} if "b" in inspect.signature(adj.FOM).parameters else {}
    fom = adj.FOM(rr, rt, tt, **fom_kwargs)
    mu = adj.calc_mu(rt, tt)
    return fom, mu


def plot_summary(run_results, scan_dir, runs_dir):
    valid = [(lr, d) for lr, d in run_results if d is not None]
    if not valid:
        print("No valid runs to plot.")
        return

    for lr, d in run_results:
        if d is None:
            print(f"Skipping lr={lr:.2e}: no data.csv found.")

    lrs = [lr for lr, _ in valid]
    log_lrs = np.log10(lrs)
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=log_lrs[0], vmax=log_lrs[-1])
    colors = [cmap(norm(ll)) for ll in log_lrs]

    ncols = min(len(valid), 2)
    nrows_top = math.ceil(len(valid) / ncols)
    fig_height = 4.5 * nrows_top + 5.5
    fig = plt.figure(figsize=(14, fig_height))
    gs = fig.add_gridspec(
        nrows_top + 1, ncols,
        height_ratios=[1] * nrows_top + [1.4],
        hspace=0.55, wspace=0.35,
    )
    gs_bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[nrows_top, :], wspace=0.35)
    ax_fin  = fig.add_subplot(gs_bottom[0])
    ax_stab = fig.add_subplot(gs_bottom[1])

    final_mus = []
    oscillations = []

    for i, ((lr, (_, mu)), color) in enumerate(zip(valid, colors)):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.plot(mu, color=color, lw=2)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.3)
        ax.axhline(1, color="k", lw=0.8, ls="--", alpha=0.3)
        ax.set_title(f"lr = {lr:.2e}", fontsize=20)
        ax.set_xlabel("Iteration", fontsize=16)
        ax.set_ylabel(r"$\mu$", fontsize=16)
        ax.tick_params(labelsize=14)
        tail_start = max(0, int(len(mu) * 0.8))
        final_mus.append(mu[-1])
        oscillations.append(np.std(mu[tail_start:]))

    ax_fin.semilogx(lrs, final_mus, "o-", color="steelblue", lw=2, ms=9)
    ax_fin.set_xlabel("Learning rate")
    ax_fin.set_ylabel(r"Final $\mu$")
    ax_fin.set_title(r"Final $\mu$ vs Learning Rate")
    ax_fin.set_xticks(lrs)
    ax_fin.set_xticklabels([f"{lr:.2e}" for lr in lrs], rotation=45, ha="right", fontsize=14)
    ax_fin.grid(True, which="both", alpha=0.3)

    ax_stab.loglog(lrs, oscillations, "s-", color="firebrick", lw=2, ms=9)
    ax_stab.set_xlabel("Learning rate")
    ax_stab.set_ylabel(r"$\sigma(\mu)$, last 20\%")
    ax_stab.set_title("Oscillation Amplitude vs Learning Rate")
    ax_stab.set_xticks(lrs)
    ax_stab.set_xticklabels([f"{lr:.2e}" for lr in lrs], rotation=45, ha="right", fontsize=14)
    ax_stab.grid(True, which="both", alpha=0.3)

    out = os.path.join(runs_dir, "lr_scan_summary.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved to {out}")


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("main_script")
parser.add_argument("base_param_file")
parser.add_argument("scan_dir", nargs="?", default="lr_scan")
parser.add_argument("--lr-min", type=float, default=1e-12)
parser.add_argument("--lr-max", type=float, default=1e-7)
parser.add_argument("--n-lr", type=int, default=6)
parser.add_argument("--lr-list", type=str, default=None,
                     help="Comma-separated explicit lr values; overrides --lr-min/--lr-max/--n-lr")
parser.add_argument("--scan-steps", type=int, default=200)
parser.add_argument("--fom-module", choices=["zernike", "zernike_newfom", "local"], default="zernike")

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)
args = parser.parse_args()

if args.fom_module == "local":
    # without_zernike's own real (non-shared) adjoint_intensity.py.
    sys.path.insert(0, os.getcwd())
    import adjoint_intensity as adj
elif args.fom_module == "zernike_newfom":
    import adjoint_intensity_newfom as adj
else:
    import adjoint_intensity_zernike as adj

main_script = args.main_script
base_param_file = args.base_param_file
scan_dir = args.scan_dir
LR_LIST = [float(x) for x in args.lr_list.split(",")] if args.lr_list else np.geomspace(args.lr_min, args.lr_max, args.n_lr).tolist()
SCAN_STEPS = args.scan_steps

if not os.path.exists(main_script):
    print(f"Error: '{main_script}' not found.")
    sys.exit(1)
if not os.path.exists(base_param_file):
    print(f"Error: '{base_param_file}' not found.")
    sys.exit(1)

script_stem = os.path.splitext(os.path.basename(main_script))[0]
run_prefix = script_stem[len("main_"):] if script_stem.startswith("main_") else script_stem

base_params = adj.initialize_variables_from_file(base_param_file)
if base_params is None:
    sys.exit(1)
b = base_params.get("b", base_params.get("alpha", 1.0))

os.makedirs(scan_dir, exist_ok=True)

run_results = []
run_dirs = []
for lr in LR_LIST:
    params = dict(base_params)
    params["lr"] = lr
    params["steps"] = SCAN_STEPS

    fname = f"lr_{lr:.2e}.txt"
    fpath = os.path.join(scan_dir, fname)
    write_param_file(fpath, params)

    # Main scripts use fpath.split("/")[0] as their output root, so the run
    # lands in <first_dir>/runs/<prefix>_<stem>/ regardless of scan_dir depth.
    run_root = fpath.split("/")[0]
    fname_stem = fname[:-4]
    run_dir = os.path.join(run_root, "runs", f"{run_prefix}_{fname_stem}")
    run_dirs.append(run_dir)

    print(f"\n{'='*60}\nRunning lr = {lr:.2e}  ->  {run_dir}\n{'='*60}")
    subprocess.run(["python", main_script, fpath])

    data = load_run_data(run_dir, b)
    run_results.append((lr, data))

runs_dir = os.path.join(fpath.split("/")[0], "runs")
os.makedirs(runs_dir, exist_ok=True)
plot_summary(run_results, scan_dir, runs_dir)

for run_dir in run_dirs:
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
        print(f"Deleted {run_dir}")

print("Done.")
