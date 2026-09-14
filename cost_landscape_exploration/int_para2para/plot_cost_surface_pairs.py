#!/usr/bin/env python
"""
2D contour plots over every pairwise combination of the non-piston Zernike
coefficients (piston, index 0, is held fixed at the target's value and
never swept, since it's just a global phase offset that doesn't affect the
intensity FOM), for the intensity-matching (int) para2para setup. For n=3
(para2para's usual setting) that's just the single pair (a_2, a_4); for
larger n every combination of the remaining indices is plotted, one
contour+slice row each in a single combined figure. Every grid point is a
real forward-propagation + FOM/Itsq evaluation, no shortcuts.

Usage:
    python plot_cost_surface_pairs.py <param_file> [out_prefix] [final_run_dir]

<param_file> describes the physical setup (f_targ, f_init, f_num, nr, nzf,
r_maxf, dz, wl, delta, n). Output is <out_prefix>.pdf/.png (out_prefix
defaults to "cost_surface"). Per-pair grid data is cached to
<out_prefix>_<pair>_grid.csv and reused on subsequent runs instead of
recomputing. <final_run_dir> is optional: a completed run's directory, used
to mark where its final mirror ended up (last row of its zernike_coeff.csv).
"""

import os
import sys
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import RegularGridInterpolator

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import adjoint_intensity_zernike as adj

font = {"size": 15}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (12, 9)
plt.rcParams["lines.linewidth"] = 2
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

GRID_N = 40             # points per axis (GRID_N^2 total forward-model evals, per plot)
HALF_RANGE_FRAC = 0.1   # sweep +/- this fraction of the target's own coefficient magnitude
N_LEVELS = 21            # shared filled-contour levels across every plot
SLICE_N = 200            # points sampled along the init->target slice


def zlabel(idx):
    # calc_Z's column idx corresponds to Zernike radial order m = 2*idx
    # (col 0 -> m=0 piston, col 1 -> m=2, col 2 -> m=4, col 3 -> m=6, ...).
    return f"$a_{{{2 * idx}}}$"


def generate_optic(a_n, zernpol):
    zm = adj.generate_optic(a_n, zernpol)
    return zm - zm[0]


def line_t_range(x0, y0, dx, dy, xmin, xmax, ymin, ymax):
    # Clips the infinite line point(t) = (x0,y0) + t*(dx,dy) to the
    # [xmin,xmax] x [ymin,ymax] box, returning the t range that stays inside
    # it on both axes -- i.e. how far the init->target line extends before
    # leaving the visible plot in either direction.
    t_low, t_high = -np.inf, np.inf
    if dx != 0:
        tx1, tx2 = (xmin - x0) / dx, (xmax - x0) / dx
        t_low, t_high = max(t_low, min(tx1, tx2)), min(t_high, max(tx1, tx2))
    if dy != 0:
        ty1, ty2 = (ymin - y0) / dy, (ymax - y0) / dy
        t_low, t_high = max(t_low, min(ty1, ty2)), min(t_high, max(ty1, ty2))
    return t_low, t_high


if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

param_file = sys.argv[1]
out_prefix = sys.argv[2] if len(sys.argv) >= 3 else "cost_surface"
# Optional: a completed run's directory, to mark where its final mirror
# ended up. Adam runs don't save a separate "best" coefficients file the
# way CMA-ES does, so this pulls the last row of zernike_coeff.csv instead.
final_run_dir = sys.argv[3] if len(sys.argv) >= 4 else None
a_final = None
if final_run_dir is not None:
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

dz = params["dz"]
nzf = params["nzf"]
zi = f_targ - dz
zf = f_targ + dz
zF = np.linspace(zi, zf, nzf)

num = params["n"]
delta = params["delta"]

PAIRS = list(itertools.combinations(range(1, num), 2))
if not PAIRS:
    print(f"n={num} leaves no non-piston pair to sweep (need n>=3).")
    sys.exit(1)

r, k, T_R2K, T_K2R = adj.DHTsetup(nr, rmax, ell)
prop_phase = np.exp(-1j * k**2 / (2 * k0) * zF[:, np.newaxis])
phi_I = np.exp(-((r / w0) ** 2))

rho = r / rmax
Z_fit = adj.calc_Z(rho, num)
Z_plus = adj.calc_Z_plus(Z_fit)
a_init = adj.create_initial_coeff(rmax, 2 * f_init, num)
wt = adj.weight(r, wR, zF, zi, zf, delta)

zm_t = adj.create_starting_optic(r, 2 * f_targ)
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


target_cost = evaluate(a_target)
init_cost = evaluate(a_init)
print(f"cost at target: {target_cost:.4e}   cost at init: {init_cost:.4e}")

# ── Phase 1: gather every pair's grid, computing or loading from cache ──
entries = []

for idx_x, idx_y in PAIRS:
    pair_label = f"{idx_x}{idx_y}"
    csv_path = f"{out_prefix}_{pair_label}_grid.csv"
    print(f"--- pair a[{idx_x}], a[{idx_y}] ---")

    if os.path.exists(csv_path):
        print(f"Found existing grid data at {csv_path} -- loading it instead of recomputing.")
        saved = np.genfromtxt(csv_path, delimiter=",", names=True)
        xs = np.unique(saved[saved.dtype.names[0]])
        ys = np.unique(saved[saved.dtype.names[1]])
        Cost = saved[saved.dtype.names[2]].reshape(len(ys), len(xs))
    else:
        # Floor for coefficients whose target AND initial-guess value are
        # both ~0 -- without this, half_range collapses to floating-point
        # noise instead of a physically meaningful sweep range. Scaled off
        # the largest non-null coefficient so it stays proportionate to
        # this particular target/config.
        min_abs_half = 0.05 * max(abs(a_target[1:]))

        def half_range(idx):
            dist_to_init = abs(a_init[idx] - a_target[idx])
            return max(HALF_RANGE_FRAC * abs(a_target[idx]), 1.2 * dist_to_init, min_abs_half)

        half_x = half_range(idx_x)
        half_y = half_range(idx_y)
        xs = np.linspace(a_target[idx_x] - half_x, a_target[idx_x] + half_x, GRID_N)
        ys = np.linspace(a_target[idx_y] - half_y, a_target[idx_y] + half_y, GRID_N)
        X, Y = np.meshgrid(xs, ys)
        Cost = np.zeros_like(X)

        total = GRID_N * GRID_N
        done = 0
        for i in range(GRID_N):
            for j in range(GRID_N):
                a = np.copy(a_target)
                a[idx_x] = X[i, j]
                a[idx_y] = Y[i, j]
                Cost[i, j] = evaluate(a)
                done += 1
            print(f"  row {i + 1}/{GRID_N} done ({done}/{total} evals)")

        np.savetxt(csv_path, np.column_stack([X.ravel(), Y.ravel(), Cost.ravel()]),
                   delimiter=",", header=f"a[{idx_x}],a[{idx_y}],cost", comments="")
        print(f"Saved raw grid values to {csv_path}")

    entries.append({"idx_x": idx_x, "idx_y": idx_y, "xs": xs, "ys": ys, "Cost": Cost})

# ── Phase 2: shared color scale across every plot ──
global_log_cost = np.concatenate([np.log10(np.clip(e["Cost"], 1e-12, None)).ravel() for e in entries])
vmin, vmax = global_log_cost.min(), global_log_cost.max()
levels = np.linspace(vmin, vmax, N_LEVELS)
print(f"\nShared color scale: log10(cost) in [{vmin:.3f}, {vmax:.3f}]")

# ── Phase 3: one combined figure -- one row per pair, 2 columns (contour, slice) ──
fig, axes = plt.subplots(len(PAIRS), 2, figsize=(10, 4.3 * len(PAIRS)),
                          gridspec_kw={"width_ratios": [1.5, 1]}, squeeze=False)

for row, e in enumerate(entries):
    idx_x, idx_y = e["idx_x"], e["idx_y"]
    xs, ys, Cost = e["xs"], e["ys"], e["Cost"]
    log_cost = np.log10(np.clip(Cost, 1e-12, None))
    ax, ax_slice = axes[row]

    cf = ax.contourf(xs, ys, log_cost, levels=levels, vmin=vmin, vmax=vmax, cmap="inferno")
    fig.colorbar(cf, ax=ax, label="log10(cost)")

    dx = a_target[idx_x] - a_init[idx_x]
    dy = a_target[idx_y] - a_init[idx_y]
    t_low, t_high = line_t_range(a_init[idx_x], a_init[idx_y], dx, dy,
                                  xs.min(), xs.max(), ys.min(), ys.max())

    ax.scatter([a_target[idx_x]], [a_target[idx_y]], color="cyan", s=200, marker="*",
               label="target", zorder=10, edgecolors="black", linewidths=0.75)
    ax.scatter([a_init[idx_x]], [a_init[idx_y]], color="blue", s=120, marker="o",
               label="initial guess", zorder=10, edgecolors="black", linewidths=0.75)
    ax.plot([a_init[idx_x] + t_low * dx, a_init[idx_x] + t_high * dx],
            [a_init[idx_y] + t_low * dy, a_init[idx_y] + t_high * dy],
            color="white", lw=2, ls="--", zorder=9)
    if a_final is not None:
        ax.scatter([a_final[idx_x]], [a_final[idx_y]], color="lime", s=200, marker="s",
                   label="final mirror", zorder=10, edgecolors="black", linewidths=0.75)

    fixed_idx = [i for i in range(num) if i not in (idx_x, idx_y)]
    fixed_label = ", ".join(zlabel(i) for i in fixed_idx) if fixed_idx else "none"
    ax.set_xlabel(zlabel(idx_x))
    ax.set_ylabel(zlabel(idx_y))
    ax.set_title(f"{zlabel(idx_x)}, {zlabel(idx_y)}  ({fixed_label} fixed)")
    ax.legend(loc="upper left", fontsize=10)

    # Slice straight through the contour along the line from the initial
    # guess to the target, extended in both directions to the edges of the
    # visible plot, interpolated off the already-computed grid (no extra
    # forward-model evaluations needed).
    interp = RegularGridInterpolator((ys, xs), log_cost, bounds_error=False, fill_value=None)
    slice_t = np.linspace(t_low, t_high, SLICE_N)
    slice_x = a_init[idx_x] + slice_t * dx
    slice_y = a_init[idx_y] + slice_t * dy
    slice_log_cost = interp(np.column_stack([slice_y, slice_x]))

    ax_slice.plot(slice_t, slice_log_cost, color="tab:purple")
    ax_slice.set_ylim(vmin, vmax)
    # Fixed tick spacing (rather than a fixed xlim) keeps each row's own
    # natural data range -- no clipped/wasted space -- while still
    # guaranteeing 0 and 1 (init/target) always land exactly on a tick.
    ax_slice.xaxis.set_major_locator(MultipleLocator(1))
    ax_slice.axvline(0, color="black", lw=1.25, ls="--", alpha=0.8, zorder=10)
    ax_slice.axvline(1, color="black", lw=1.25, ls="--", alpha=0.8, zorder=10)
    # The "0=initial guess, 1=target" explanation and the "Slice: init ->
    # target" title only need to be said once per figure, not repeated
    # identically on every row.
    ax_slice.set_xlabel(r"$t$  (0 = initial guess, 1 = target)" if row == len(PAIRS) - 1 else r"$t$")
    ax_slice.set_ylabel("log10(cost)")
    if row == 0:
        ax_slice.set_title("Slice: init " + r"$\to$" + " target")

plt.tight_layout()
out_path = f"{out_prefix}.pdf"
plt.savefig(out_path)
png_path = out_path.rsplit(".", 1)[0] + ".png"
plt.savefig(png_path, dpi=200)
plt.close()
print(f"Saved to {out_path} and {png_path}")
